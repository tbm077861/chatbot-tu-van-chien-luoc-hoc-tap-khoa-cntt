"""BM25 sparse retriever cho corpus 438 môn (Giai đoạn 5).

Bổ trợ cho dense retriever (M4) khi query chứa từ khoá rõ ràng (mã môn, tên môn,
ngành) — keyword matching mạnh hơn embedding ở các trường hợp này.

Tokenizer: đơn giản hoá tiếng Việt (lowercase + bỏ dấu câu + split whitespace).
Không dùng pyvi/underthesea để giảm dependency; corpus ngắn nên tokenizer thô
vẫn đủ.

Ví dụ dùng:
    bm25 = BM25Retriever.from_corpus_file(Path("data/embeddings/corpus.jsonl"))
    hits = bm25.search("máy học deep learning", top_k=20)
    # hits = [(doc_id, score), ...]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi


# Bỏ dấu câu và ký tự đặc biệt nhưng giữ chữ có dấu tiếng Việt.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản: lowercase → bỏ dấu câu → split whitespace.

    Args:
        text: câu input bất kỳ (tiếng Việt hoặc trộn).

    Returns:
        Danh sách token, đã lọc token rỗng.
    """
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return [tok for tok in t.split(" ") if tok]


class BM25Retriever:
    """BM25Okapi wrapper khớp interface với DenseRetrieverM4.

    Args:
        doc_ids: list mã doc (theo thứ tự cố định, dùng làm index).
        doc_texts: list text mô tả từng môn (đã tokenize ở init).
        k1: tham số BM25 saturation (mặc định 1.5).
        b: tham số BM25 length normalization (mặc định 0.75).
    """

    def __init__(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(doc_ids) != len(doc_texts):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) và doc_texts ({len(doc_texts)}) lệch nhau."
            )
        self.doc_ids = list(doc_ids)
        self.tokenized_corpus = [_tokenize(t) for t in doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    @classmethod
    def from_corpus_file(cls, path: Path, **kwargs) -> "BM25Retriever":
        """Khởi tạo từ `corpus.jsonl` (mỗi dòng có `doc_id` và `text`)."""
        ids: list[str] = []
        texts: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ids.append(d["doc_id"])
                texts.append(d["text"])
        return cls(ids, texts, **kwargs)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Trả top-K (doc_id, bm25_score) cho query.

        Args:
            query: câu hỏi tiếng Việt.
            top_k: số kết quả trả về.

        Returns:
            List (doc_id, score), score cao = relevant hơn. Có thể trả ít hơn
            `top_k` nếu corpus nhỏ hơn.
        """
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        n = min(top_k, len(scores))
        # argsort descending; argpartition nhanh hơn cho top-K nhỏ.
        idx_top = scores.argsort()[::-1][:n]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idx_top]

    def search_batch(
        self, queries: Iterable[str], top_k: int = 20
    ) -> list[list[tuple[str, float]]]:
        """Search nhiều query (đơn giản loop vì BM25 đã rất nhanh)."""
        return [self.search(q, top_k) for q in queries]
