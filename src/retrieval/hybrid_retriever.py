"""Hybrid retriever — RRF fusion dense (M4) + sparse (BM25).

Mục đích: kết hợp ưu điểm cả 2:
- Dense (M4 + E5): hiểu ngữ nghĩa, paraphrase, định hướng nghề nghiệp.
- Sparse (BM25): mạnh khi query chứa mã môn cụ thể, tên môn chính xác.

Thuật toán: **Reciprocal Rank Fusion (RRF)** (Cormack et al. 2009):
    score_rrf(d) = Σ_{r ∈ retrievers}  1 / (k + rank_r(d))
trong đó `k=60` (giá trị thực nghiệm chuẩn). RRF không cần normalize score
giữa các retriever vì chỉ dùng rank.

Có thể weight không đều bằng `weights=(w_dense, w_sparse)` — score nhân với
trọng số tương ứng. Mặc định (1.0, 1.0) → trung lập.
"""

from __future__ import annotations

from collections import defaultdict

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetrieverM4


class HybridRetriever:
    """Late fusion dense + sparse bằng RRF.

    Args:
        dense: instance `DenseRetrieverM4`.
        sparse: instance `BM25Retriever`.
        rrf_k: hằng số trong RRF (mặc định 60).
        weights: (w_dense, w_sparse) — trọng số cho từng retriever.
    """

    def __init__(
        self,
        dense: DenseRetrieverM4,
        sparse: BM25Retriever,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.w_dense, self.w_sparse = weights

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = 50,
    ) -> list[tuple[str, float]]:
        """Search hybrid.

        Args:
            query: câu hỏi.
            top_k: số kết quả trả về cuối cùng.
            candidate_pool: lấy bao nhiêu candidate từ mỗi retriever trước khi fusion.
                Lớn hơn top_k để tăng recall.

        Returns:
            List (doc_id, rrf_score) sort giảm dần.
        """
        dense_hits = self.dense.search(query, top_k=candidate_pool)
        sparse_hits = self.sparse.search(query, top_k=candidate_pool)

        rrf: dict[str, float] = defaultdict(float)
        # Dense: rank bắt đầu từ 0 → +1 trong công thức.
        for rank, (doc_id, _score) in enumerate(dense_hits):
            rrf[doc_id] += self.w_dense / (self.rrf_k + rank + 1)
        for rank, (doc_id, _score) in enumerate(sparse_hits):
            rrf[doc_id] += self.w_sparse / (self.rrf_k + rank + 1)

        fused = sorted(rrf.items(), key=lambda x: -x[1])
        return fused[:top_k]

    def search_with_components(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = 50,
    ) -> dict:
        """Search nhưng trả thêm thông tin debug của từng retriever.

        Returns:
            dict {
              "fused": [(doc_id, rrf_score), ...],
              "dense": [(doc_id, score), ...],
              "sparse": [(doc_id, score), ...],
            }
        """
        dense_hits = self.dense.search(query, top_k=candidate_pool)
        sparse_hits = self.sparse.search(query, top_k=candidate_pool)

        rrf: dict[str, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(dense_hits):
            rrf[doc_id] += self.w_dense / (self.rrf_k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse_hits):
            rrf[doc_id] += self.w_sparse / (self.rrf_k + rank + 1)

        return {
            "fused": sorted(rrf.items(), key=lambda x: -x[1])[:top_k],
            "dense": dense_hits[:top_k],
            "sparse": sparse_hits[:top_k],
        }
