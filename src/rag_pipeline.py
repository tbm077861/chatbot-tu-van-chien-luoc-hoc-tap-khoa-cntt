"""RAG Pipeline orchestrator — Giai đoạn 5.

End-to-end flow:
    1. Hybrid retrieval: M4 dense + BM25 sparse → RRF fusion → top-K candidates.
    2. (Optional) Cross-encoder reranker → rescore top-K.
    3. Constraint check: lọc theo prereq + cảnh báo TC.
    4. Format context từ valid candidates.
    5. Generator (Qwen-7B+LoRA hoặc Stub) → response text.
    6. Parse mã môn từ response → final recommendations.

Sử dụng:
    pipeline = RagPipeline.load_default(use_llm=True, use_reranker=False)
    result = pipeline.answer(
        query="Em ngành CS HK5, định hướng AI",
        nganh="CS",
        completed=["004247", "001782", ...],
    )
    print(result.response)
    print(result.recommendations)

CLI:
    python -m src.rag_pipeline --query "..." --nganh CS --stub  # nhanh, không LLM
    python -m src.rag_pipeline --query "..." --nganh CS         # full Qwen
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "data" / "embeddings"

sys.path.insert(0, str(ROOT))
from src.generation.generator import (  # noqa: E402
    QwenGenerator,
    StubGenerator,
    parse_recommendations,
)
from src.generation.prompt_templates import build_user_message, format_context  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.constraint_checker import (  # noqa: E402
    ConstraintChecker,
    ConstraintReport,
)
from src.retrieval.dense_retriever import DenseRetrieverM4  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.reranker import CrossEncoderReranker  # noqa: E402


@dataclass
class RagResult:
    """Kết quả end-to-end của 1 query qua pipeline."""

    query: str
    nganh: str
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    """(doc_id, fused_score) trước constraint."""

    reranked: list[tuple[str, float]] | None = None
    """(doc_id, ce_score) sau reranker; None nếu không bật."""

    constraint: ConstraintReport | None = None
    """Báo cáo constraint sau filter."""

    context_docs: list[dict] = field(default_factory=list)
    """Doc metadata đưa vào prompt."""

    response: str = ""
    """Text generation từ Qwen/Stub."""

    recommendations: list[str] = field(default_factory=list)
    """doc_id parse từ response — final answer."""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "nganh": self.nganh,
            "retrieved": self.retrieved,
            "reranked": self.reranked,
            "constraint": self.constraint.to_dict() if self.constraint else None,
            "context_docs": self.context_docs,
            "response": self.response,
            "recommendations": self.recommendations,
        }


class RagPipeline:
    """Orchestrator gọi retriever → (rerank) → constraint → generator.

    Args:
        hybrid: HybridRetriever (dense + sparse + RRF).
        constraint: ConstraintChecker.
        generator: QwenGenerator hoặc StubGenerator.
        reranker: CrossEncoderReranker, None nếu tắt.
        id2doc: dict {doc_id: metadata} để format context.
        candidate_pool: số candidate lấy từ retriever trước khi rerank/filter.
        top_k_context: số doc cuối đưa vào prompt LLM.
    """

    def __init__(
        self,
        hybrid: HybridRetriever,
        constraint: ConstraintChecker,
        generator: QwenGenerator | StubGenerator,
        id2doc: dict[str, dict],
        reranker: CrossEncoderReranker | None = None,
        candidate_pool: int = 30,
        top_k_context: int = 10,
    ) -> None:
        self.hybrid = hybrid
        self.constraint = constraint
        self.generator = generator
        self.reranker = reranker
        self.id2doc = id2doc
        self.candidate_pool = candidate_pool
        self.top_k_context = top_k_context

    @classmethod
    def load_default(
        cls,
        use_llm: bool = True,
        use_reranker: bool = False,
        load_4bit: bool = True,
        candidate_pool: int = 30,
        top_k_context: int = 10,
    ) -> "RagPipeline":
        """Load đầy đủ pipeline từ artifact mặc định.

        Args:
            use_llm: True → QwenGenerator (cần GPU); False → StubGenerator.
            use_reranker: True → bật M3 cross-encoder rerank.
            load_4bit: chỉ relevant khi use_llm; 4-bit quantization để vừa VRAM.
        """
        print("[pipeline] loading dense retriever (M4 + E5)...")
        dense = DenseRetrieverM4.load_default()

        print("[pipeline] loading BM25 retriever...")
        sparse = BM25Retriever.from_corpus_file(EMB_DIR / "corpus.jsonl")

        print("[pipeline] loading constraint checker (5 ngành)...")
        constraint = ConstraintChecker.load_default()

        reranker: CrossEncoderReranker | None = None
        if use_reranker:
            print("[pipeline] loading M3 cross-encoder reranker...")
            reranker = CrossEncoderReranker.load_default()

        generator: QwenGenerator | StubGenerator
        if use_llm:
            print("[pipeline] loading M5 Qwen-7B + LoRA generator...")
            generator = QwenGenerator.load_default(load_4bit=load_4bit)
        else:
            print("[pipeline] using StubGenerator (template, no LLM).")
            generator = StubGenerator()

        # Load corpus metadata.
        id2doc: dict[str, dict] = {}
        with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                id2doc[d["doc_id"]] = d

        hybrid = HybridRetriever(dense, sparse)
        return cls(
            hybrid=hybrid,
            constraint=constraint,
            generator=generator,
            id2doc=id2doc,
            reranker=reranker,
            candidate_pool=candidate_pool,
            top_k_context=top_k_context,
        )

    def _docs_for_context(self, doc_ids: list[str]) -> list[dict]:
        """Convert list doc_id → list metadata dict cho prompt template."""
        docs: list[dict] = []
        for did in doc_ids:
            meta = self.id2doc.get(did)
            if meta is None:
                continue
            docs.append(
                {
                    "doc_id": did,
                    "ten_mon": meta.get("ten_mon", "?"),
                    "ma_mon": meta.get("ma_mon", "?"),
                    "so_tc": meta.get("so_tc", "?"),
                    "loai": meta.get("loai", "?"),
                    "hk_chuan": meta.get("hk_chuan"),
                }
            )
        return docs

    def answer(
        self,
        query: str,
        nganh: str,
        completed: list[str] | None = None,
        in_progress: list[str] | None = None,
        hk_hien_tai: int | None = None,
        gpa: float | None = None,
        dinh_huong: str | None = None,
        max_new_tokens: int = 512,
    ) -> RagResult:
        """Chạy end-to-end pipeline cho 1 query.

        Args:
            query: câu hỏi tiếng Việt.
            nganh: mã ngành sinh viên (CS/IS/DS/SE/IT).
            completed: danh sách mã môn (6 chữ số) đã hoàn thành.
            in_progress: danh sách mã môn đang học cùng kỳ (cho song hành).
            hk_hien_tai: học kỳ chuẩn bị đăng ký.
            gpa: GPA tích lũy (thang 4).
            dinh_huong: cluster định hướng (AI/ML, DB/BigData, v.v.).
            max_new_tokens: cap generation length.

        Returns:
            `RagResult` chứa trace mọi bước + recommendations cuối cùng.
        """
        result = RagResult(query=query, nganh=nganh)

        # 1. Hybrid retrieval.
        fused = self.hybrid.search(
            query, top_k=self.candidate_pool, candidate_pool=self.candidate_pool * 2
        )
        result.retrieved = fused

        # 2. Optional rerank.
        candidates: list[tuple[str, float]] = fused
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, fused)
            result.reranked = candidates

        # 3. Constraint filter — chỉ kiểm môn cùng ngành.
        candidate_ids = [d for d, _ in candidates]
        report = self.constraint.check_recommendations(
            candidate_ids,
            nganh=nganh,
            completed=completed,
            in_progress=in_progress,
        )
        result.constraint = report

        # 4. Lấy top-K valid → format context.
        valid_top = report.valid[: self.top_k_context]
        context_docs = self._docs_for_context(valid_top)
        result.context_docs = context_docs
        ctx_str = format_context(context_docs)

        # 5. Build user message + generate.
        user_msg = build_user_message(
            question=query,
            nganh=nganh,
            hk_hien_tai=hk_hien_tai,
            gpa=gpa,
            da_hoan_thanh=completed,
            dinh_huong=dinh_huong,
            retrieved_context=ctx_str,
        )
        gen_kwargs: dict = {}
        if isinstance(self.generator, QwenGenerator):
            gen_kwargs["max_new_tokens"] = max_new_tokens
        response = self.generator.generate(user_msg, context_docs, **gen_kwargs)
        result.response = response

        # 6. Parse recommendations từ response.
        valid_ids = set(self.id2doc.keys())
        result.recommendations = parse_recommendations(response, nganh, valid_ids)

        return result


def _cli() -> None:
    ap = argparse.ArgumentParser(description="RAG pipeline smoke test")
    ap.add_argument("--query", required=True)
    ap.add_argument("--nganh", choices=["CS", "IS", "DS", "SE", "IT"], required=True)
    ap.add_argument("--completed", nargs="*", default=[])
    ap.add_argument("--hk", type=int, default=None)
    ap.add_argument("--gpa", type=float, default=None)
    ap.add_argument("--dinh-huong", default=None)
    ap.add_argument("--stub", action="store_true", help="Dùng StubGenerator.")
    ap.add_argument("--rerank", action="store_true", help="Bật M3 reranker.")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    pipeline = RagPipeline.load_default(
        use_llm=not args.stub,
        use_reranker=args.rerank,
        load_4bit=not args.no_4bit,
    )

    result = pipeline.answer(
        query=args.query,
        nganh=args.nganh,
        completed=args.completed,
        hk_hien_tai=args.hk,
        gpa=args.gpa,
        dinh_huong=args.dinh_huong,
    )

    print("\n=== Retrieved (top 10 fused) ===")
    for did, sc in result.retrieved[:10]:
        ten = pipeline.id2doc.get(did, {}).get("ten_mon", "?")
        print(f"  {did}  {ten[:40]:40s}  {sc:.4f}")
    if result.reranked:
        print("\n=== Reranked (top 10) ===")
        for did, sc in result.reranked[:10]:
            ten = pipeline.id2doc.get(did, {}).get("ten_mon", "?")
            print(f"  {did}  {ten[:40]:40s}  {sc:.4f}")
    print(f"\n=== Constraint report ===")
    if result.constraint:
        print(f"  Valid: {len(result.constraint.valid)} môn, tổng TC={result.constraint.total_tc}")
        print(f"  Violations: {len(result.constraint.violations)}")
        for w in result.constraint.warnings:
            print(f"  ⚠ {w}")
    print("\n=== Generated response ===")
    print(result.response)
    print(f"\n=== Parsed recommendations: {result.recommendations} ===")


if __name__ == "__main__":
    _cli()
