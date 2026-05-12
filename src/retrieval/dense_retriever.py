"""Dense retriever — M4 (no_gnn) bi-encoder + E5 fine-tuned (Giai đoạn 5).

Vai trò trong pipeline RAG:
    User query → E5 encode (768-d) → M4 q_proj MLP → 256-d query
    Score = cosine(query, cached_doc_proj[438, 256])
    → top-K (doc_id, score).

Tại sao chọn M4 no_gnn?
- STATUS.md mục "Kết quả Giai đoạn 4": M4 no_gnn đạt R@1=0.706, MRR=0.811 —
  cao nhất trong toàn bộ M1–M5. GNN phần không đóng góp signal (graph thưa).
- Inference rất rẻ: E5 (đã load sẵn) + 2 MLP nhỏ ~256-d.

Cache doc embedding: tính 1 lần khi khởi tạo, lưu trong memory `self.doc_embs`.

Ví dụ dùng:
    retriever = DenseRetrieverM4.load_default()
    hits = retriever.search("Em ngành CS HK5, định hướng AI", top_k=10)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"
M4_DIR = ROOT / "data" / "models" / "gnn_transformer"
CACHE_DIR = EMB_DIR / "cache_e5"

sys.path.insert(0, str(ROOT))
from src.models.gnn_transformer import FusionRetriever  # noqa: E402


class DenseRetrieverM4:
    """Wrapper inference cho M4 (no_gnn) bi-encoder.

    Args:
        model: `FusionRetriever` đã load checkpoint, ở mode eval.
        e5_model: `SentenceTransformer` E5 fine-tuned, đã chuyển device.
        doc_ids: list mã doc khớp thứ tự cache.
        doc_embs: tensor [n_docs, proj_dim] đã chuẩn hoá L2, sẵn trên device.
        device: torch.device sử dụng.
    """

    def __init__(
        self,
        model: FusionRetriever,
        e5_model,
        doc_ids: list[str],
        doc_embs: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.model = model
        self.e5_model = e5_model
        self.doc_ids = list(doc_ids)
        self.doc_embs = doc_embs  # [n_docs, proj_dim], L2-normalized, on device
        self.device = device

    @classmethod
    def load_default(
        cls,
        ckpt_path: Path | None = None,
        e5_path: Path | None = None,
        device: torch.device | None = None,
    ) -> "DenseRetrieverM4":
        """Load M4 no_gnn checkpoint + E5 fine-tuned từ artifact mặc định."""
        ckpt_path = ckpt_path or (M4_DIR / "no_gnn_best.pt")
        e5_path = e5_path or (EMB_DIR / "e5_finetuned")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load cached E5 doc embeddings + GCN (GCN sẽ bị mask khi use_gnn=False).
        doc_text = torch.from_numpy(np.load(CACHE_DIR / "doc_text_e5.npy"))
        doc_gnn = torch.from_numpy(np.load(CACHE_DIR / "doc_gnn_gcn.npy"))
        with open(CACHE_DIR / "doc_ids.json", encoding="utf-8") as f:
            doc_ids: list[str] = json.load(f)

        # 2. Khởi tạo FusionRetriever đúng kiến trúc lúc train.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ckpt["args"]
        model = FusionRetriever(
            text_dim=doc_text.shape[1],
            gnn_dim=doc_gnn.shape[1],
            proj_dim=args.get("proj_dim", 256),
            n_docs=len(doc_ids),
            doc_text=doc_text,
            doc_gnn=doc_gnn,
            use_gnn=False,  # no_gnn variant
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()

        # 3. Precompute doc projections (438 × 256) — chỉ tính 1 lần.
        with torch.no_grad():
            doc_embs = model.get_doc_embeddings()  # đã L2-normalized

        # 4. Load E5.
        from sentence_transformers import SentenceTransformer

        e5 = SentenceTransformer(str(e5_path))
        e5.to(device)
        e5.eval()

        return cls(model, e5, doc_ids, doc_embs, device)

    @torch.no_grad()
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query → [1, proj_dim], L2-normalized."""
        # E5 cần prefix "query: " theo training scheme Stage 3.
        e5_emb = self.e5_model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q = torch.from_numpy(e5_emb).to(self.device)
        return self.model.encode_query(q)  # [1, proj_dim]

    @torch.no_grad()
    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Trả top-K (doc_id, cosine_score).

        Args:
            query: câu hỏi tiếng Việt.
            top_k: số kết quả trả.

        Returns:
            List (doc_id, score), score là cosine ∈ [-1, 1].
        """
        q = self.encode_query(query)  # [1, D]
        scores = (q @ self.doc_embs.t()).squeeze(0)  # [n_docs]
        n = min(top_k, scores.size(0))
        top_vals, top_idx = torch.topk(scores, n)
        return [
            (self.doc_ids[int(i)], float(s))
            for s, i in zip(top_vals.tolist(), top_idx.tolist())
        ]
