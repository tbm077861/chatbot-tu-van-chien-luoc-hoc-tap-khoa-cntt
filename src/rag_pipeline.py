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
from src.generation.intent_classifier import Intent, classify_intent  # noqa: E402
from src.generation.prompt_templates import (  # noqa: E402
    SYSTEM_PROMPT,
    build_chat_user_message,
    build_user_message,
    enrich_short_response,
    format_context,
)
from src.generation.regulation_qa import RegulationQA  # noqa: E402
from src.retrieval.prereq_qa import PrereqQA  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.constraint_checker import (  # noqa: E402
    ConstraintChecker,
    ConstraintReport,
)
from src.retrieval.dense_retriever import DenseRetrieverM4  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.reranker import CrossEncoderReranker  # noqa: E402


@dataclass
class ChatResult:
    """Kết quả 1 lượt chat (turn). Khác `RagResult` ở chỗ giữ messages history."""

    response: str = ""
    """Assistant text trả về cho user."""

    intent: str = "recommend"
    """Intent đã phân loại: recommend / regulation / prereq."""

    recommendations: list[str] = field(default_factory=list)
    """doc_id parse từ response (intent recommend)."""

    retrieved: list[tuple[str, float]] = field(default_factory=list)
    """Top-K candidates trước constraint (debug, intent recommend)."""

    context_docs: list[dict] = field(default_factory=list)
    """Doc đưa vào prompt (intent recommend)."""

    target_courses: list[str] = field(default_factory=list)
    """Mã môn extract từ query (intent prereq)."""

    constraint: ConstraintReport | None = None
    """Báo cáo constraint (intent recommend)."""

    used_query: str = ""
    """Query thật sự đưa vào pipeline (= message user cuối)."""

    next_messages: list[dict] = field(default_factory=list)
    """messages sau khi append assistant turn — để UI lưu lại."""

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "intent": self.intent,
            "recommendations": self.recommendations,
            "retrieved": self.retrieved,
            "context_docs": self.context_docs,
            "target_courses": self.target_courses,
            "constraint": self.constraint.to_dict() if self.constraint else None,
            "used_query": self.used_query,
            "next_messages": self.next_messages,
        }


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
        # Khi bật LLM, ép retriever lên CPU để dành 8GB VRAM cho Qwen-7B 4-bit.
        # Inference E5 trên CPU cho 1 query ~50-100ms — chấp nhận được.
        retriever_device = torch.device("cpu") if use_llm else None
        print(
            f"[pipeline] loading dense retriever (M4 + E5) "
            f"on {retriever_device or 'auto'}..."
        )
        dense = DenseRetrieverM4.load_default(device=retriever_device)

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

    def chat(
        self,
        messages: list[dict],
        nganh: str,
        completed: list[str] | None = None,
        in_progress: list[str] | None = None,
        profile_summary_md: str = "",
        hk_target: int | None = None,
        only_tu_chon: bool = True,
        max_new_tokens: int = 768,
    ) -> "ChatResult":
        """Multi-turn chat — wrap retrieval + constraint + generator chat.

        Args:
            messages: lịch sử conversation, list[{role, content}]. Phần tử
                cuối phải là role=user (câu hỏi mới nhất).
            nganh: ngành của sinh viên (lấy từ profile).
            completed: list mã môn đã hoàn thành (từ profile).
            in_progress: mã môn cùng kỳ (optional).
            profile_summary_md: bảng điểm dạng markdown — chỉ inject ở turn 1
                để Qwen có context. Turn 2+ Qwen nhớ qua history.
            max_new_tokens: cap generation.

        Returns:
            ChatResult với response + recommendations + next_messages
            (= messages + assistant turn mới).

        Raises:
            ValueError: nếu messages rỗng hoặc message cuối không phải user.
        """
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("messages phải kết thúc bằng role='user'")

        query = messages[-1]["content"].strip()
        is_first_turn = (
            sum(1 for m in messages if m.get("role") == "user") == 1
        )

        # Phân loại intent → dispatch sang handler tương ứng.
        intent: Intent = classify_intent(query)
        if intent == "regulation":
            return self._chat_regulation(messages, query, max_new_tokens)
        if intent == "prereq":
            return self._chat_prereq(
                messages, query, nganh, completed, hk_target, max_new_tokens
            )
        # intent == "recommend" — luồng cũ.

        # 1. Retrieval theo query mới nhất.
        fused = self.hybrid.search(
            query,
            top_k=self.candidate_pool,
            candidate_pool=self.candidate_pool * 2,
        )

        # 2. Optional rerank.
        candidates: list[tuple[str, float]] = fused
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, fused)

        # 3. Constraint — chỉ giữ môn cùng ngành + hợp lệ prereq.
        candidate_ids = [d for d, _ in candidates]
        report = self.constraint.check_recommendations(
            candidate_ids,
            nganh=nganh,
            completed=completed,
            in_progress=in_progress,
        )

        # 4. Filter thêm: chỉ môn tự chọn + đúng HK chuẩn (yêu cầu Stage 6 v2).
        #    Curriculum mỗi HK có pool tu_chon riêng — bot phải gợi đúng pool đó.
        valid_filtered: list[str] = []
        seen: set[str] = set()
        for did in report.valid:
            meta = self.id2doc.get(did, {})
            if only_tu_chon and meta.get("loai") != "tu_chon":
                continue
            if hk_target is not None and meta.get("hk_chuan") != hk_target:
                continue
            if did not in seen:
                valid_filtered.append(did)
                seen.add(did)

        # Force-include ALL môn tu_chon của HK target trong NGÀNH user — đảm
        # bảo Qwen thấy toàn bộ pool, kể cả môn retriever không bắt được trong
        # top-K (vì retriever search globally). Ưu tiên thứ tự retriever, rồi
        # thêm môn còn lại theo thứ tự corpus.
        if hk_target is not None:
            completed_set = set(completed or [])
            for did, meta in self.id2doc.items():
                if not did.startswith(f"{nganh}_"):
                    continue
                if meta.get("loai") != "tu_chon":
                    continue
                if meta.get("hk_chuan") != hk_target:
                    continue
                if meta.get("ma_mon") in completed_set:
                    continue  # bỏ môn đã học
                if did in seen:
                    continue
                valid_filtered.append(did)
                seen.add(did)

        # Fallback cuối: nếu vẫn rỗng (HK target không có tu_chon), nới hk_chuan.
        if not valid_filtered and hk_target is not None:
            for did in report.valid:
                meta = self.id2doc.get(did, {})
                if only_tu_chon and meta.get("loai") != "tu_chon":
                    continue
                hk_c = meta.get("hk_chuan")
                if hk_c is not None and hk_c <= hk_target:
                    if did not in seen:
                        valid_filtered.append(did)
                        seen.add(did)

        valid_top = valid_filtered[: self.top_k_context]
        context_docs = self._docs_for_context(valid_top)
        ctx_str = format_context(context_docs)

        # 5. Build new user content cho turn cuối (thay vì raw query).
        enriched_user = build_chat_user_message(
            question=query,
            profile_summary_md=profile_summary_md if is_first_turn else "",
            retrieved_context=ctx_str,
            is_first_turn=is_first_turn,
            hk_target=hk_target,
        )

        # Rebuild messages: thay user cuối bằng enriched version.
        chat_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages[:-1]:
            chat_messages.append({"role": m["role"], "content": m["content"]})
        chat_messages.append({"role": "user", "content": enriched_user})

        # 6. Generate. LoRA Stage 4 train trên output ngắn nhưng vẫn cho giải
        #    thích được nếu prompt rõ + profile có hook (môn điểm cao liên quan
        #    định hướng). disable_lora=True trên 4-bit có overhead lớn → giữ
        #    LoRA. Fallback explanation ở dưới nếu response quá ngắn.
        if isinstance(self.generator, QwenGenerator):
            response = self.generator.chat(
                chat_messages, max_new_tokens=max_new_tokens
            )
        else:
            response = self.generator.chat(chat_messages, retrieved_docs=context_docs)

        recs = parse_recommendations(response, nganh, set(self.id2doc.keys()))

        # Fallback giải thích: nếu Qwen response quá ngắn (LoRA Stage 4 thường
        # output 1-2 môn không giải thích), append cluster-based template để
        # user luôn nhận được giải thích đầy đủ.
        if len(response) < 300 or len(recs) < 3:
            response = enrich_short_response(
                response=response,
                recommendations=recs,
                context_docs=context_docs,
                query=query,
                profile_summary_md=profile_summary_md,
            )

        # next_messages = history gốc + assistant response (để UI giữ tiếp).
        next_messages = list(messages) + [{"role": "assistant", "content": response}]

        return ChatResult(
            response=response,
            intent="recommend",
            recommendations=recs,
            retrieved=fused,
            context_docs=context_docs,
            constraint=report,
            used_query=query,
            next_messages=next_messages,
        )

    # ===== Multi-intent handlers (Stage 7) =====

    def _chat_regulation(
        self,
        messages: list[dict],
        query: str,
        max_new_tokens: int,
    ) -> "ChatResult":
        """Intent regulation: inject regulations.json → Qwen chat."""
        # Lazy-init handler (chỉ load 1 lần/instance).
        if not hasattr(self, "_reg_qa"):
            self._reg_qa = RegulationQA()
        response = self._reg_qa.answer(
            history=messages,
            generator=self.generator,
            max_new_tokens=max_new_tokens,
        )
        next_messages = list(messages) + [{"role": "assistant", "content": response}]
        return ChatResult(
            response=response,
            intent="regulation",
            used_query=query,
            next_messages=next_messages,
        )

    def _chat_prereq(
        self,
        messages: list[dict],
        query: str,
        nganh: str,
        completed: list[str] | None,
        hk_target: int | None,
        max_new_tokens: int,
    ) -> "ChatResult":
        """Intent prereq: extract môn → lookup graph → Qwen format response."""
        if not hasattr(self, "_prereq_qa"):
            self._prereq_qa = PrereqQA(self.constraint, self.id2doc)
        response, targets = self._prereq_qa.answer(
            history=messages,
            nganh=nganh,
            completed=completed,
            generator=self.generator,
            hk_target=hk_target,
            max_new_tokens=max_new_tokens,
        )
        next_messages = list(messages) + [{"role": "assistant", "content": response}]
        return ChatResult(
            response=response,
            intent="prereq",
            target_courses=targets,
            used_query=query,
            next_messages=next_messages,
        )


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
