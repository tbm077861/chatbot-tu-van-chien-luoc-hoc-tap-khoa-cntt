"""M2 — Transformer bi-encoder dense retriever (Giai đoạn 4).

Mục tiêu: So sánh kiến trúc Transformer (self-attention) với RNN (LSTM/GRU) của
M1 trong cùng điều kiện training (from-scratch, cùng tokenizer, cùng course
embedding table, cùng CE loss). E5 fine-tuned của Giai đoạn 3 đóng vai trò
reference "M2 with pretrained init".

Kiến trúc bi-encoder:
- Tokenizer PhoBERT (chỉ tokenize, embedding init random).
- TransformerEncoder 4 lớp, d_model=128, n_heads=4, ff=512, dropout=0.1.
- Học positional embedding (learnable, max_len=128).
- Mean-pool có mask → linear projection → 128-dim query embedding.
- Course embedding table (438 × 128) học từ đầu, dot-product score.
- Loss: CrossEntropyLoss với 438 lớp + temperature 20.

CLI:
    # Train + eval (mặc định)
    python -m src.models.transformer_retriever --epochs 5
    # Chỉ eval (sau khi đã train)
    python -m src.models.transformer_retriever --eval-only
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "data" / "models" / "transformer_bi"

sys.path.insert(0, str(ROOT))
from src.embedding.evaluator import metrics_from_scores  # noqa: E402
from src.models.lstm_recommender import (  # noqa: E402
    PairDataset,
    collate_fn,
    load_corpus,
    load_jsonl,
)


class TransformerRecommender(nn.Module):
    """Bi-encoder Transformer + course embedding table.

    Args:
        vocab_size: kích thước vocab tokenizer.
        n_courses: số môn (438).
        max_len: độ dài tối đa cho positional embedding.
        d_model: chiều mô hình Transformer.
        n_heads: số attention heads.
        n_layers: số encoder layers.
        ff_dim: chiều feed-forward.
        out_dim: chiều embedding cuối (để score với course).
        pad_token_id: token PAD để mask.
        dropout: dropout chung.
    """

    def __init__(
        self,
        vocab_size: int,
        n_courses: int,
        max_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 512,
        out_dim: int = 128,
        pad_token_id: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )
        self.course_emb = nn.Embedding(n_courses, out_dim)
        nn.init.normal_(self.course_emb.weight, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode_query(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode batch query → (B, out_dim), L2-normalized."""
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        # Trong nn.TransformerEncoder, padding_mask True = bị mask (ignored).
        pad_mask = attention_mask == 0
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, T, d)
        # Mean-pool theo mask thực (1 ở token, 0 ở pad).
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        q = self.proj(pooled)
        return F.normalize(q, dim=-1)

    def all_course_emb(self) -> torch.Tensor:
        return F.normalize(self.course_emb.weight, dim=-1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        q = self.encode_query(input_ids, attention_mask)
        c = self.all_course_emb()
        return (q @ c.t()) * 20.0  # temperature


def train(
    model: TransformerRecommender,
    loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    """Train CE loss. Transformer dùng lr nhỏ hơn LSTM và warmup linear."""
    optim = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98)
    )
    total_steps = len(loader) * epochs
    warmup_steps = max(int(0.05 * total_steps), 100)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # cosine decay sau warmup.
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    history: list[dict] = []
    step = 0
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
            sched.step()
            step += 1
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_n += labels.size(0)
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        # Val.
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
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"lr={sched.get_last_lr()[0]:.2e} ({dt:.1f}s)"
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
    model: TransformerRecommender,
    tokenizer,
    doc_ids: list[str],
    test_set: list[dict],
    device: torch.device,
    max_len: int,
    batch: int = 64,
) -> dict:
    """Eval cùng giao thức M1: encode query → score với course_emb → metrics."""
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
    Q = np.concatenate(q_embs, axis=0)
    C = model.all_course_emb().cpu().numpy()
    scores = Q @ C.T
    return metrics_from_scores(scores, doc_ids, test_set)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--ff-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument(
        "--tokenizer",
        default="vinai/phobert-base-v2",
        help="HF tokenizer (chỉ dùng tokenize).",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} transformer bi-encoder")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    doc_ids, _ = load_corpus()
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    print(f"[corpus] n_docs={len(doc_ids)} vocab_size={tokenizer.vocab_size}")

    pairs = load_jsonl(EMB_DIR / "train_pairs.jsonl")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pairs))
    n_val = max(int(len(pairs) * args.val_frac), 200)
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

    model = TransformerRecommender(
        vocab_size=tokenizer.vocab_size,
        n_courses=len(doc_ids),
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        out_dim=args.out_dim,
        pad_token_id=tokenizer.pad_token_id or 1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] params={n_params/1e6:.2f}M d_model={args.d_model} "
        f"n_heads={args.n_heads} n_layers={args.n_layers}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "transformer_best.pt"

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

    test_set = load_jsonl(EMB_DIR / "test.jsonl")
    metrics = evaluate_on_test(
        model, tokenizer, doc_ids, test_set, device, args.max_len
    )
    print(f"[eval] test_set={len(test_set)}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    result = {
        "model": "transformer_bi_scratch",
        "args": vars(args),
        "metrics": metrics,
        "history": history.get("history", []),
        "n_params": n_params,
    }
    with open(OUT_DIR / "transformer_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[save] -> {OUT_DIR / 'transformer_results.json'}")


if __name__ == "__main__":
    main()
