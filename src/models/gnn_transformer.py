"""M4 — GNN + Transformer fusion bi-encoder (Giai đoạn 4).

Mục tiêu: Học cách kết hợp signal **text (E5)** + **đồ thị tiên quyết (GCN)**
ở phía doc, thay vì late-fusion α-weighted cố định như Stage 3 hybrid.

Kiến trúc:
- Query: E5 fine-tuned encode (frozen) → 768-dim → MLP projector → 256-dim.
- Doc:
    - text_emb = E5 fine-tuned (frozen) → 768-dim.
    - gnn_emb = GCN node embedding (frozen, từ Stage 3) → 128-dim.
    - concat → 896-dim → MLP projector → 256-dim.
- Score: cos(q_proj, d_proj).
- Loss: full-softmax CE 438 lớp (cùng giao thức M1/M2).

Tại sao learned fusion thắng α cố định?
- α=0.85 fix có thể không tối ưu cho mọi query. MLP học weight non-linear giữa
  text và graph dimensions theo từng context.
- Linear projection còn căn chỉnh không gian text 768-dim và graph 128-dim về
  cùng space 256-dim → cosine có ý nghĩa.

Cache: precompute E5(query) cho train 60k pairs + test 500 để train cực nhanh
(forward pass chỉ là 2 MLP). Cache lưu ở `data/embeddings/cache_e5/`.

Ablation: cờ `--no-gnn` để mask GNN portion (zero-out) → baseline so sánh.

CLI:
    python -m src.models.gnn_transformer --epochs 10 --batch 256
    python -m src.models.gnn_transformer --epochs 10 --batch 256 --no-gnn
    python -m src.models.gnn_transformer --eval-only
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
CACHE_DIR = EMB_DIR / "cache_e5"
OUT_DIR = ROOT / "data" / "models" / "gnn_transformer"

sys.path.insert(0, str(ROOT))
from src.embedding.evaluator import metrics_from_scores  # noqa: E402
from src.models.lstm_recommender import load_corpus, load_jsonl  # noqa: E402


def encode_with_e5(texts: list[str], is_query: bool, batch: int = 64) -> np.ndarray:
    """Encode batch texts bằng E5 fine-tuned (Stage 3), L2-normalized."""
    from sentence_transformers import SentenceTransformer

    model_path = EMB_DIR / "e5_finetuned"
    model = SentenceTransformer(str(model_path))
    if torch.cuda.is_available():
        model.to("cuda")
    prefix = "query: " if is_query else "passage: "
    inputs = [prefix + t for t in texts]
    emb = model.encode(
        inputs,
        batch_size=batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb.astype(np.float32)


def build_or_load_cache(force: bool = False) -> dict:
    """Cache: doc_text_emb [438×768], gnn_emb [438×128], train_q_emb [Nq×768],
    test_q_emb [500×768] + doc_id list. Trả dict numpy arrays."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "doc_text": CACHE_DIR / "doc_text_e5.npy",
        "doc_gnn": CACHE_DIR / "doc_gnn_gcn.npy",
        "train_q": CACHE_DIR / "train_q_e5.npy",
        "test_q": CACHE_DIR / "test_q_e5.npy",
        "doc_ids": CACHE_DIR / "doc_ids.json",
        "train_labels": CACHE_DIR / "train_labels.npy",
    }
    if all(p.exists() for p in paths.values()) and not force:
        print(f"[cache] hit -> {CACHE_DIR}")
        return {
            "doc_text": np.load(paths["doc_text"]),
            "doc_gnn": np.load(paths["doc_gnn"]),
            "train_q": np.load(paths["train_q"]),
            "test_q": np.load(paths["test_q"]),
            "doc_ids": json.loads(paths["doc_ids"].read_text(encoding="utf-8")),
            "train_labels": np.load(paths["train_labels"]),
        }

    print(f"[cache] miss, build -> {CACHE_DIR}")
    # Corpus.
    doc_ids, doc_texts = load_corpus()
    print(f"[cache] encode docs n={len(doc_ids)}")
    doc_text_emb = encode_with_e5(doc_texts, is_query=False)
    np.save(paths["doc_text"], doc_text_emb)

    # GCN embedding align thứ tự với doc_ids.
    z = np.load(EMB_DIR / "gnn_gcn" / "node_embeddings.npy").astype(np.float32)
    with open(EMB_DIR / "gnn_gcn" / "node_ids.json", encoding="utf-8") as f:
        gnn_ids: list[str] = json.load(f)
    gnn_pos = {nid: i for i, nid in enumerate(gnn_ids)}
    z_aligned = np.zeros((len(doc_ids), z.shape[1]), dtype=np.float32)
    for i, d in enumerate(doc_ids):
        if d in gnn_pos:
            z_aligned[i] = z[gnn_pos[d]]
    # L2 normalize GCN (E5 đã normalized, để đồng nhất).
    z_aligned = z_aligned / (np.linalg.norm(z_aligned, axis=1, keepdims=True) + 1e-9)
    np.save(paths["doc_gnn"], z_aligned)

    # Train queries.
    pairs = load_jsonl(EMB_DIR / "train_pairs.jsonl")
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    train_queries: list[str] = []
    train_labels: list[int] = []
    for p in pairs:
        idx = doc_id_to_idx.get(p["positive_doc_id"])
        if idx is None:
            continue
        train_queries.append(p["query"])
        train_labels.append(idx)
    print(f"[cache] encode train queries n={len(train_queries)}")
    train_q_emb = encode_with_e5(train_queries, is_query=True)
    np.save(paths["train_q"], train_q_emb)
    np.save(paths["train_labels"], np.array(train_labels, dtype=np.int64))

    # Test queries.
    test_set = load_jsonl(EMB_DIR / "test.jsonl")
    print(f"[cache] encode test queries n={len(test_set)}")
    test_q_emb = encode_with_e5([ex["query"] for ex in test_set], is_query=True)
    np.save(paths["test_q"], test_q_emb)

    paths["doc_ids"].write_text(
        json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "doc_text": doc_text_emb,
        "doc_gnn": z_aligned,
        "train_q": train_q_emb,
        "test_q": test_q_emb,
        "doc_ids": doc_ids,
        "train_labels": np.array(train_labels, dtype=np.int64),
    }


class CachedQueryDataset(Dataset):
    def __init__(self, q_emb: np.ndarray, labels: np.ndarray) -> None:
        self.q = torch.from_numpy(q_emb)
        self.y = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q[i], self.y[i]


class FusionRetriever(nn.Module):
    """Bi-encoder với learned linear fusion text+GNN ở phía doc.

    Args:
        text_dim: 768 (E5).
        gnn_dim: 128 (GCN).
        proj_dim: 256, chiều cosine cuối.
        n_docs: 438.
        doc_text: tensor [n_docs, text_dim] (frozen).
        doc_gnn: tensor [n_docs, gnn_dim] (frozen).
        use_gnn: nếu False → mask GNN bằng zero (ablation).
    """

    def __init__(
        self,
        text_dim: int,
        gnn_dim: int,
        proj_dim: int,
        n_docs: int,
        doc_text: torch.Tensor,
        doc_gnn: torch.Tensor,
        use_gnn: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_gnn = use_gnn
        # Doc features là buffer (không train), gắn vào device cùng module.
        self.register_buffer("doc_text", doc_text)
        if use_gnn:
            self.register_buffer("doc_gnn", doc_gnn)
        else:
            # Vẫn giữ shape để code đơn giản, nhưng toàn 0 → không học signal GNN.
            self.register_buffer("doc_gnn", torch.zeros_like(doc_gnn))

        doc_in = text_dim + gnn_dim
        self.doc_proj = nn.Sequential(
            nn.Linear(doc_in, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.q_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(np.log(20.0)))

    def get_doc_embeddings(self) -> torch.Tensor:
        """Trả [n_docs, proj_dim] L2-normalized."""
        d = torch.cat([self.doc_text, self.doc_gnn], dim=-1)  # type: ignore[arg-type]
        d = self.doc_proj(d)
        return F.normalize(d, dim=-1)

    def encode_query(self, q_emb: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q_emb)
        return F.normalize(q, dim=-1)

    def forward(self, q_emb: torch.Tensor) -> torch.Tensor:
        q = self.encode_query(q_emb)
        d = self.get_doc_embeddings()
        return (q @ d.t()) * self.temperature.exp()


def train(
    model: FusionRetriever,
    loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    history: list[dict] = []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        t0 = time.time()
        for q, y in loader:
            q = q.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(q)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += y.size(0)
        sched.step()
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        # Val.
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for q, y in val_loader:
                q = q.to(device)
                y = y.to(device)
                logits = model(q)
                v_correct += (logits.argmax(dim=-1) == y).sum().item()
                v_total += y.size(0)
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
                "secs": dt,
            }
        )
    return {"history": history}


@torch.no_grad()
def evaluate(
    model: FusionRetriever,
    test_q: np.ndarray,
    doc_ids: list[str],
    test_set: list[dict],
    device: torch.device,
    batch: int = 256,
) -> dict:
    model.eval()
    Q = torch.from_numpy(test_q).to(device)
    q_embs: list[np.ndarray] = []
    for i in range(0, Q.size(0), batch):
        q = model.encode_query(Q[i : i + batch])
        q_embs.append(q.cpu().numpy())
    q_all = np.concatenate(q_embs, axis=0)
    d_all = model.get_doc_embeddings().cpu().numpy()
    scores = q_all @ d_all.T
    return metrics_from_scores(scores, doc_ids, test_set)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--no-gnn", action="store_true", help="Ablation: tắt GNN signal.")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "no_gnn" if args.no_gnn else "with_gnn"
    print(f"[init] device={device} variant={tag}")

    cache = build_or_load_cache(force=args.rebuild_cache)
    doc_text = torch.from_numpy(cache["doc_text"])
    doc_gnn = torch.from_numpy(cache["doc_gnn"])
    train_q = cache["train_q"]
    test_q = cache["test_q"]
    doc_ids = cache["doc_ids"]
    train_labels = cache["train_labels"]
    print(
        f"[data] doc_text={doc_text.shape} doc_gnn={doc_gnn.shape} "
        f"train_q={train_q.shape} test_q={test_q.shape}"
    )

    # Tách val.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_labels))
    n_val = max(int(len(train_labels) * args.val_frac), 200)
    tr_q = train_q[idx[n_val:]]
    tr_y = train_labels[idx[n_val:]]
    va_q = train_q[idx[:n_val]]
    va_y = train_labels[idx[:n_val]]
    train_ds = CachedQueryDataset(tr_q, tr_y)
    val_ds = CachedQueryDataset(va_q, va_y)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = FusionRetriever(
        text_dim=doc_text.shape[1],
        gnn_dim=doc_gnn.shape[1],
        proj_dim=args.proj_dim,
        n_docs=len(doc_ids),
        doc_text=doc_text,
        doc_gnn=doc_gnn,
        use_gnn=not args.no_gnn,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable_params={n_params/1e6:.3f}M")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / f"{tag}_best.pt"
    history: dict = {}
    if not args.eval_only:
        history = train(model, train_loader, val_loader, device, args.epochs, args.lr)
        torch.save(
            {"model_state": model.state_dict(), "args": vars(args), "doc_ids": doc_ids},
            ckpt_path,
        )
        print(f"[save] -> {ckpt_path}")
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[load] <- {ckpt_path}")

    test_set = load_jsonl(EMB_DIR / "test.jsonl")
    metrics = evaluate(model, test_q, doc_ids, test_set, device)
    print(f"[eval] test_set={len(test_set)}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    result = {
        "model": f"gnn_transformer_fusion_{tag}",
        "args": vars(args),
        "metrics": metrics,
        "history": history.get("history", []),
        "trainable_params": n_params,
    }
    with open(OUT_DIR / f"{tag}_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[save] -> {OUT_DIR / f'{tag}_results.json'}")


if __name__ == "__main__":
    main()
