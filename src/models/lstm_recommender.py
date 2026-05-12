"""M1 — LSTM/GRU baseline sequence recommender (Giai đoạn 4).

Bài toán: Cho query mô tả profile sinh viên (text), gợi ý top-K môn từ corpus
438 môn (5 ngành).

Kiến trúc baseline (cố tình đơn giản, không tận dụng pretrained weights):
- Tokenizer: PhoBERT (subword), nhưng embedding layer init random từ đầu.
- Encoder: BiLSTM/BiGRU 2 lớp + mean-pool (mask-aware) + linear projection.
- Course embedding table: nn.Embedding(438, d) học từ đầu.
- Score = q · C^T → softmax 438 lớp (full softmax classification).

Loss: CrossEntropyLoss (mỗi training pair là (query, positive_doc_id) → predict
positive_doc_id trong số 438 môn). Đây là biến thể của in-batch contrastive khi
mở rộng "negatives" lên toàn bộ corpus (nhỏ — 438 môn → khả thi).

Đánh giá: cùng test 500 pairs + cùng metrics_from_scores() ở
`src/embedding/evaluator.py` → kết quả so sánh apple-to-apple với PhoBERT/E5
fine-tuned và GNN của Giai đoạn 3.

CLI:
    # Train + eval (mặc định cả 2)
    python -m src.models.lstm_recommender --epochs 5
    # Chỉ eval (sau khi đã train)
    python -m src.models.lstm_recommender --eval-only
    # Đổi sang GRU
    python -m src.models.lstm_recommender --rnn gru --epochs 5
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
OUT_DIR = ROOT / "data" / "models" / "lstm_baseline"

# Cho phép import evaluator (cùng project, tính metrics).
sys.path.insert(0, str(ROOT))
from src.embedding.evaluator import metrics_from_scores  # noqa: E402


def load_corpus() -> tuple[list[str], list[str]]:
    """Đọc corpus.jsonl → (doc_ids, texts). Thứ tự cố định = thứ tự lớp."""
    ids: list[str] = []
    texts: list[str] = []
    with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["doc_id"])
            texts.append(d["text"])
    return ids, texts


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


class PairDataset(Dataset):
    """Dataset (query_text, positive_doc_idx) cho cross-entropy training."""

    def __init__(
        self,
        pairs: list[dict],
        doc_id_to_idx: dict[str, int],
        tokenizer,
        max_len: int = 128,
    ) -> None:
        # Lọc pair có doc_id tồn tại trong corpus (có thể có pair bị orphan).
        self.items: list[tuple[str, int]] = []
        for p in pairs:
            idx = doc_id_to_idx.get(p["positive_doc_id"])
            if idx is not None:
                self.items.append((p["query"], idx))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[str, int]:
        return self.items[i]


def collate_fn(batch: list[tuple[str, int]], tokenizer, max_len: int) -> dict:
    queries = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    enc = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


class LSTMRecommender(nn.Module):
    """BiLSTM/BiGRU encoder + course embedding lookup.

    Args:
        vocab_size: kích thước vocab của tokenizer.
        n_courses: số môn trong corpus (438).
        emb_dim: chiều token embedding.
        hidden: chiều ẩn của RNN (mỗi chiều, total = 2 × hidden vì bi-directional).
        n_layers: số lớp RNN.
        out_dim: chiều cuối cùng để score với course embedding.
        rnn_type: 'lstm' hoặc 'gru'.
        pad_token_id: token PAD để mask khi pool.
    """

    def __init__(
        self,
        vocab_size: int,
        n_courses: int,
        emb_dim: int = 128,
        hidden: int = 256,
        n_layers: int = 2,
        out_dim: int = 128,
        rnn_type: str = "lstm",
        pad_token_id: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden, out_dim),
            nn.Tanh(),
        )
        # Bảng course embedding học từ đầu.
        self.course_emb = nn.Embedding(n_courses, out_dim)
        nn.init.normal_(self.course_emb.weight, std=0.02)

    def encode_query(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode batch query → vector (B, out_dim), L2-normalized."""
        x = self.token_emb(input_ids)  # (B, T, E)
        out, _ = self.rnn(x)  # (B, T, 2H)
        # Mean-pool có mask.
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom  # (B, 2H)
        q = self.proj(pooled)  # (B, out_dim)
        return F.normalize(q, dim=-1)

    def all_course_emb(self) -> torch.Tensor:
        """Trả course embedding L2-normalized (n_courses, out_dim)."""
        return F.normalize(self.course_emb.weight, dim=-1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Score query × all courses → logits (B, n_courses)."""
        q = self.encode_query(input_ids, attention_mask)
        c = self.all_course_emb()
        # Temperature scaling giúp softmax nhọn hơn, learn faster.
        return (q @ c.t()) * 20.0


def train(
    model: LSTMRecommender,
    loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    """Train CE loss, log loss + val accuracy mỗi epoch."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    history: list[dict] = []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        t0 = time.time()
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(input_ids, mask)
            loss = F.cross_entropy(logits, labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_n += labels.size(0)
        sched.step()
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        # Validation accuracy.
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, mask)
                v_correct += (logits.argmax(dim=-1) == labels).sum().item()
                v_total += labels.size(0)
        val_acc = v_correct / max(v_total, 1)
        dt = time.time() - t0
        print(
            f"[train] epoch={ep}/{epochs} loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)"
        )
        history.append(
            {
                "epoch": ep,
                "loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": sched.get_last_lr()[0],
                "secs": dt,
            }
        )
    return {"history": history}


@torch.no_grad()
def evaluate_on_test(
    model: LSTMRecommender,
    tokenizer,
    doc_ids: list[str],
    test_set: list[dict],
    device: torch.device,
    max_len: int,
    batch: int = 64,
) -> dict:
    """Tính score [N_query × N_doc] và trả metrics."""
    model.eval()
    queries = [ex["query"] for ex in test_set]
    q_embs: list[np.ndarray] = []
    for i in range(0, len(queries), batch):
        chunk = queries[i : i + batch]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        q = model.encode_query(enc["input_ids"], enc["attention_mask"])
        q_embs.append(q.cpu().numpy())
    Q = np.concatenate(q_embs, axis=0)  # (N, D)
    C = model.all_course_emb().cpu().numpy()  # (n_courses, D)
    scores = Q @ C.T  # (N, n_courses) — đã L2-normalized → cosine
    return metrics_from_scores(scores, doc_ids, test_set)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnn", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument(
        "--tokenizer",
        default="vinai/phobert-base-v2",
        help="HF tokenizer (chỉ dùng tokenize, embedding init random).",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} rnn={args.rnn}")

    # 1. Tokenizer + corpus.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    doc_ids, _ = load_corpus()
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    print(f"[corpus] n_docs={len(doc_ids)} vocab_size={tokenizer.vocab_size}")

    # 2. Dataset.
    pairs = load_jsonl(EMB_DIR / "train_pairs.jsonl")
    # Shuffle + tách val.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    n_val = max(int(len(pairs) * args.val_frac), 200)
    val_idx = set(idx[:n_val].tolist())
    train_pairs = [pairs[i] for i in idx[n_val:]]
    val_pairs = [pairs[i] for i in idx[:n_val]]
    train_ds = PairDataset(train_pairs, doc_id_to_idx, tokenizer, args.max_len)
    val_ds = PairDataset(val_pairs, doc_id_to_idx, tokenizer, args.max_len)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    def _coll(b: list[tuple[str, int]]) -> dict:
        return collate_fn(b, tokenizer, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=_coll,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=_coll,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # 3. Model.
    model = LSTMRecommender(
        vocab_size=tokenizer.vocab_size,
        n_courses=len(doc_ids),
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        n_layers=args.n_layers,
        out_dim=args.out_dim,
        rnn_type=args.rnn,
        pad_token_id=tokenizer.pad_token_id or 1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / f"{args.rnn}_best.pt"

    # 4. Train (nếu không --eval-only).
    history: dict = {}
    if not args.eval_only:
        history = train(
            model, train_loader, val_loader, device, args.epochs, args.lr
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "doc_ids": doc_ids,
                "vocab_size": tokenizer.vocab_size,
            },
            ckpt_path,
        )
        print(f"[save] -> {ckpt_path}")
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[load] <- {ckpt_path}")

    # 5. Evaluate trên test 500.
    test_set = load_jsonl(EMB_DIR / "test.jsonl")
    metrics = evaluate_on_test(
        model, tokenizer, doc_ids, test_set, device, args.max_len
    )
    print(f"[eval] test_set={len(test_set)}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    result = {
        "model": f"lstm_baseline_{args.rnn}",
        "args": vars(args),
        "metrics": metrics,
        "history": history.get("history", []),
        "n_params": n_params,
    }
    with open(OUT_DIR / f"{args.rnn}_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[save] -> {OUT_DIR / f'{args.rnn}_results.json'}")


if __name__ == "__main__":
    main()
