"""Retrieval module — Giai đoạn 5.

- `BM25Retriever`: sparse keyword retrieval qua rank_bm25.
- `DenseRetrieverM4`: dense retrieval dùng M4 (no_gnn) + E5 encoder.
- `HybridRetriever`: late fusion RRF (Reciprocal Rank Fusion) M4 + BM25.
- `CrossEncoderReranker`: rerank top-K bằng M3 PhoBERT cross-encoder (tuỳ chọn).
- `ConstraintChecker`: kiểm tra prerequisite + giới hạn tín chỉ.
"""

from .bm25_retriever import BM25Retriever
from .constraint_checker import ConstraintChecker, ConstraintReport
from .dense_retriever import DenseRetrieverM4
from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker

__all__ = [
    "BM25Retriever",
    "ConstraintChecker",
    "ConstraintReport",
    "DenseRetrieverM4",
    "HybridRetriever",
    "CrossEncoderReranker",
]
