"""M3 — Cross-attention reranker (Giai đoạn 4).

Mục tiêu: Sau khi bi-encoder (E5 / M2) đã lấy top-K candidates, dùng
cross-encoder ghép (query, doc) trong cùng input để re-rank chính xác hơn.
Cross-encoder mạnh hơn bi-encoder vì self-attention nhìn được mọi token giữa
query và doc, đổi lại đắt hơn (chỉ chạy được trên K nhỏ).

Kiến trúc:
- Backbone: `vinai/phobert-base-v2` (pretrained, fine-tune full).
- Input: "{query}" + [SEP] + "{doc_text}" (gộp 1 sequence, max_len=256).
- Head: AutoModelForSequenceClassification num_labels=1 → scalar score.
- Loss: pairwise hinge — max(0, m - s(q, pos) + s(q, neg)) với m=0.3.
  (BCE pointwise có thể dùng nhưng pairwise margin học rank tốt hơn.)
- Dataset: `train_triplets.jsonl` (60k triplets có hard negative).

Eval: hai-tầng:
1. Lấy top-K=20 candidates từ E5 fine-tuned (best retrieval Giai đoạn 3).
2. Cross-encoder rescore K candidates.
3. Tính R@1, R@5, R@10, MRR, NDCG@10 trên thứ tự mới.
4. So với baseline E5 alone để tính lợi ích reranker.

CLI:
    python -m src.models.cross_reranker --epochs 2 --batch 16
    python -m src.models.cross_reranker --eval-only --top-k 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "data" / "models" / "cross_reranker"

sys.path.insert(0, str(ROOT))
from src.embedding.evaluator import metrics_from_scores  # noqa: E402


def load_corpus_map() -> dict[str, dict]:
    """Trả {doc_id: {ten_mon, text, ...}}."""
    m: dict[str, dict] = {}
    with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            m[d["doc_id"]] = d
    return m


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


class TripletDataset(Dataset):
    """Dataset (query, pos_text, neg_text)."""

    def __init__(self, triplets: list[dict], id2doc: dict[str, dict]) -> None:
        self.items: list[tuple[str, str, str]] = []
        for t in triplets:
            pos = id2doc.get(t["positive_doc_id"])
            neg = id2doc.get(t["negative_doc_id"])
            if pos is None or neg is None:
                continue
            self.items.append((t["query"], pos["text"], neg["text"]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[str, str, str]:
        return self.items[i]


def collate_triplet(
    batch: list[tuple[str, str, str]], tokenizer, max_len: int
) -> dict:
    queries = [b[0] for b in batch]
    pos_texts = [b[1] for b in batch]
    neg_texts = [b[2] for b in batch]
    pos_enc = tokenizer(
        queries,
        pos_texts,
        padding=True,
        truncation="longest_first",
        max_length=max_len,
        return_tensors="pt",
    )
    neg_enc = tokenizer(
        queries,
        neg_texts,
        padding=True,
        truncation="longest_first",
        max_length=max_len,
        return_tensors="pt",
    )
    return {"pos": pos_enc, "neg": neg_enc}


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    margin: float,
    grad_accum: int = 1,
) -> dict:
    """Train pairwise hinge: max(0, margin - s(pos) + s(neg))."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(loader) * epochs // grad_accum
    warmup = max(int(0.05 * total_steps), 100)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    history: list[dict] = []
    step = 0
    optim.zero_grad()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        t0 = time.time()
        for i, batch in enumerate(loader):
            pos = {k: v.to(device, non_blocking=True) for k, v in batch["pos"].items()}
            neg = {k: v.to(device, non_blocking=True) for k, v in batch["neg"].items()}
            s_pos = model(**pos).logits.squeeze(-1)
            s_neg = model(**neg).logits.squeeze(-1)
            loss = F.relu(margin - s_pos + s_neg).mean()
            (loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
            total_loss += loss.item() * s_pos.size(0)
            total_correct += (s_pos > s_neg).sum().item()
            total_n += s_pos.size(0)
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        dt = time.time() - t0
        print(
            f"[train] epoch={ep}/{epochs} loss={train_loss:.4f} "
            f"pair_acc={train_acc:.4f} ({dt:.1f}s)"
        )
        history.append(
            {
                "epoch": ep,
                "loss": train_loss,
                "pair_acc": train_acc,
                "lr": sched.get_last_lr()[0],
                "secs": dt,
            }
        )
    return {"history": history}


@torch.no_grad()
def evaluate_two_stage(
    model: nn.Module,
    tokenizer,
    id2doc: dict[str, dict],
    test_set: list[dict],
    device: torch.device,
    top_k: int = 20,
    max_len: int = 256,
    batch: int = 32,
) -> tuple[dict, dict]:
    """Two-stage eval: E5 retrieve top-K → cross-encoder rerank.

    Returns:
        (metrics_baseline_E5, metrics_after_rerank)
    """
    # 1. E5 retrieval (cùng cách với src/embedding/evaluator.py).
    from sentence_transformers import SentenceTransformer

    e5_path = EMB_DIR / "e5_finetuned"
    e5 = SentenceTransformer(str(e5_path))
    if torch.cuda.is_available():
        e5.to("cuda")
    doc_ids = list(id2doc.keys())
    doc_texts = [id2doc[i]["text"] for i in doc_ids]
    queries = [ex["query"] for ex in test_set]
    q_emb = e5.encode(
        [f"query: {q}" for q in queries],
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    d_emb = e5.encode(
        [f"passage: {t}" for t in doc_texts],
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores_e5 = q_emb @ d_emb.T  # (N_query, N_doc)
    base_metrics = metrics_from_scores(scores_e5, doc_ids, test_set)
    print("[stage1 E5 baseline]")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Giải phóng E5 khỏi GPU để dành cho reranker.
    del e5
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. Cross-encoder rerank top-K.
    model.eval()
    # Lấy top-K idx cho mỗi query.
    topk_idx = np.argsort(-scores_e5, axis=1)[:, :top_k]  # (N, K)
    # Score sau rerank: cho query qi, candidate ci, score = cross-encoder(qi, ci_doc).
    # Khởi tạo bằng E5 score (cho doc ngoài top-K giữ nguyên rank cũ).
    new_scores = scores_e5.copy().astype(np.float32) * 0.01  # giảm trọng số gốc
    # Đảo các giá trị top-K cũ về 0 trước, sẽ set bằng score reranker — và doc
    # ngoài top-K vẫn giữ trật tự tương đối nhưng score thấp hơn top-K (× 0.01).

    print(f"[stage2 rerank top-{top_k}]")
    for qi in range(0, len(queries), batch):
        chunk_q = queries[qi : qi + batch]
        chunk_idx = topk_idx[qi : qi + batch]  # (b, K)
        # Build pairs (query × K docs).
        flat_q: list[str] = []
        flat_d: list[str] = []
        for q, ks in zip(chunk_q, chunk_idx):
            for k in ks:
                flat_q.append(q)
                flat_d.append(doc_texts[k])
        enc = tokenizer(
            flat_q,
            flat_d,
            padding=True,
            truncation="longest_first",
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        # Chia nhỏ thêm để tránh OOM (b × K pairs).
        bs_inner = 64
        s_list: list[np.ndarray] = []
        for j in range(0, enc["input_ids"].size(0), bs_inner):
            sub = {k: v[j : j + bs_inner] for k, v in enc.items()}
            out = model(**sub).logits.squeeze(-1).cpu().numpy()
            s_list.append(out)
        s = np.concatenate(s_list)  # (b × K,)
        s = s.reshape(len(chunk_q), top_k)
        # Ghi đè new_scores tại top-K idx.
        for bi, ks in enumerate(chunk_idx):
            new_scores[qi + bi, ks] = s[bi]

    rerank_metrics = metrics_from_scores(new_scores, doc_ids, test_set)
    print("[stage2 after rerank]")
    for k, v in rerank_metrics.items():
        print(f"  {k}: {v:.4f}")

    return base_metrics, rerank_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=30000, help="Cắt bớt để train nhanh.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument(
        "--backbone",
        default="vinai/phobert-base-v2",
        help="Pretrained model làm cross-encoder backbone.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} backbone={args.backbone}")

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    id2doc = load_corpus_map()
    print(f"[corpus] n_docs={len(id2doc)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT_DIR / "checkpoint"

    if not args.eval_only:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.backbone, num_labels=1, ignore_mismatched_sizes=True
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] params={n_params/1e6:.2f}M")

        triplets = load_jsonl(EMB_DIR / "train_triplets.jsonl")
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(triplets))
        triplets = [triplets[i] for i in idx[: args.n_train]]
        ds = TripletDataset(triplets, id2doc)
        print(f"[data] train_triplets={len(ds)}")

        def _coll(b: list[tuple[str, str, str]]) -> dict:
            return collate_triplet(b, tokenizer, args.max_len)

        loader = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            collate_fn=_coll,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        hist = train(
            model, loader, device, args.epochs, args.lr, args.margin, args.grad_accum
        )

        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"[save] -> {ckpt_dir}")
        history = hist["history"]
    else:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        print(f"[load] <- {ckpt_dir}")
        history = []

    # Eval two-stage.
    test_set = load_jsonl(EMB_DIR / "test.jsonl")
    base_metrics, rerank_metrics = evaluate_two_stage(
        model,
        tokenizer,
        id2doc,
        test_set,
        device,
        top_k=args.top_k,
        max_len=args.max_len,
        batch=args.batch,
    )

    result = {
        "model": f"cross_reranker_{args.backbone.split('/')[-1]}",
        "args": vars(args),
        "metrics_e5_baseline": base_metrics,
        "metrics_after_rerank": rerank_metrics,
        "history": history,
    }
    with open(OUT_DIR / "rerank_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[save] -> {OUT_DIR / 'rerank_results.json'}")

    # Delta comparison (tránh ký tự non-ASCII vì console Windows cp1252).
    print("\n=== Delta (rerank - E5 baseline) ===")
    for k in base_metrics:
        delta = rerank_metrics[k] - base_metrics[k]
        sign = "+" if delta >= 0 else ""
        print(f"  {k}: {sign}{delta:.4f}")


if __name__ == "__main__":
    main()
