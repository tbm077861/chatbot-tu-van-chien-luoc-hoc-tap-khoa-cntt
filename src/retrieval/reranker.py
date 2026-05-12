"""Cross-encoder reranker — wrap M3 PhoBERT đã train (Giai đoạn 5).

M3 (Stage 4) là PhoBERT-base-v2 fine-tune cross-encoder. Đặc điểm:
- Score chính xác hơn bi-encoder vì self-attention nhìn được toàn bộ
  (query, doc) trong cùng input.
- Đắt: forward 1 lần cho mỗi (query, doc) pair → chỉ rerank top-K nhỏ.

Trong pipeline RAG, reranker này là TUỲ CHỌN (mặc định OFF) vì STATUS.md ghi:
M3 cải thiện R@10 (+0.01) nhưng làm giảm R@1 (-0.014) và MRR (-0.0035).
Bật khi cần đa dạng top-10, không bật khi ưu tiên top-1 chính xác.

Ví dụ:
    reranker = CrossEncoderReranker.load_default()
    new_hits = reranker.rerank("ngành CS HK5", candidates)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "data" / "models" / "cross_reranker" / "checkpoint"
CORPUS_PATH = ROOT / "data" / "embeddings" / "corpus.jsonl"


class CrossEncoderReranker:
    """Wrap M3 PhoBERT cross-encoder cho inference rerank.

    Args:
        model: AutoModelForSequenceClassification (num_labels=1) đã load.
        tokenizer: AutoTokenizer tương ứng.
        id2text: dict {doc_id: text mô tả môn}.
        device: torch.device.
        max_len: max_length cho tokenizer (mặc định 256, khớp training).
    """

    def __init__(
        self,
        model,
        tokenizer,
        id2text: dict[str, str],
        device: torch.device,
        max_len: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.id2text = id2text
        self.device = device
        self.max_len = max_len

    @classmethod
    def load_default(
        cls,
        ckpt_dir: Path | None = None,
        corpus_path: Path | None = None,
        device: torch.device | None = None,
    ) -> "CrossEncoderReranker":
        """Load M3 checkpoint từ artifact Stage 4."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        ckpt_dir = ckpt_dir or CKPT_DIR
        corpus_path = corpus_path or CORPUS_PATH
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
        model.to(device).eval()

        id2text: dict[str, str] = {}
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                id2text[d["doc_id"]] = d["text"]

        return cls(model, tokenizer, id2text, device)

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_k: int | None = None,
        batch_size: int = 32,
    ) -> list[tuple[str, float]]:
        """Rescore candidates bằng cross-encoder.

        Args:
            query: câu hỏi.
            candidates: list (doc_id, score_cũ) từ retriever.
            top_k: chỉ trả top-K sau rerank (mặc định = len(candidates)).
            batch_size: batch size khi forward.

        Returns:
            List (doc_id, score_mới) sort giảm dần theo score cross-encoder.
            Doc không có text trong corpus → giữ nguyên ở cuối với score gốc.
        """
        if not candidates:
            return []

        # Tách doc có / không có text.
        with_text: list[tuple[str, float]] = []
        without_text: list[tuple[str, float]] = []
        for doc_id, sc in candidates:
            if doc_id in self.id2text:
                with_text.append((doc_id, sc))
            else:
                without_text.append((doc_id, sc))

        if not with_text:
            return candidates[:top_k] if top_k else candidates

        doc_ids = [d for d, _ in with_text]
        texts = [self.id2text[d] for d in doc_ids]
        queries = [query] * len(doc_ids)

        scores: list[float] = []
        for i in range(0, len(doc_ids), batch_size):
            q_batch = queries[i : i + batch_size]
            t_batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                q_batch,
                t_batch,
                padding=True,
                truncation="longest_first",
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc).logits.squeeze(-1).cpu().tolist()
            if isinstance(out, float):
                out = [out]
            scores.extend(out)

        reranked = list(zip(doc_ids, scores))
        reranked.sort(key=lambda x: -x[1])
        result = reranked + without_text  # các môn không có text giữ cuối
        if top_k is not None:
            result = result[:top_k]
        return result
