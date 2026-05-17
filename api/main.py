"""FastAPI backend cho RAG chatbot tư vấn học phần (Giai đoạn 6).

Expose `RagPipeline.answer()` qua HTTP. Pipeline load 1 lần ở startup
(lifespan). Mode generator chọn bằng env var:

| ENV | Giá trị | Hành vi |
|---|---|---|
| `USE_LLM`       | "1" hoặc "true" | Dùng QwenGenerator (cần GPU + Qwen-7B đã cache HF). Mặc định StubGenerator. |
| `USE_RERANKER`  | "1" hoặc "true" | Bật M3 cross-encoder rerank. Mặc định OFF (theo kết luận Stage 4). |
| `LOAD_4BIT`     | "0" để tắt 4-bit | Mặc định 4-bit. Tắt cần >15GB VRAM. |
| `TOP_K`         | int | Số recommendations cuối trả về (mặc định 10). |
| `HF_TOKEN`      | string | Tải base Qwen từ HuggingFace. Đọc từ `.env`. |
| `API_HOST`      | "0.0.0.0" | Host bind. Mặc định 127.0.0.1. |
| `API_PORT`      | int | Mặc định 8000. |

Endpoints:
- `GET /health` — liveness + mode hiện tại.
- `GET /info` — corpus size, ngành hỗ trợ, top_k, ...
- `POST /answer` — body `AnswerRequest` → `AnswerResponse`.

Chạy:
    venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000
hoặc:
    venv/Scripts/python.exe api/main.py
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.data.grade_table_builder import build_grade_table  # noqa: E402
from src.data.profile_loader import (  # noqa: E402
    build_profile_from_grades,
    load_so_tc_map,
)
from src.rag_pipeline import RagPipeline  # noqa: E402


def _env_bool(name: str, default: bool = False) -> bool:
    """Đọc env var dạng boolean. '1', 'true', 'yes', 'on' → True."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# ===== Pydantic schemas =====


class AnswerRequest(BaseModel):
    """Request body cho POST /answer."""

    query: str = Field(..., description="Câu hỏi tiếng Việt của sinh viên.")
    nganh: str = Field(
        ...,
        description="Mã ngành: CS / IS / DS / SE / IT.",
        pattern="^(CS|IS|DS|SE|IT)$",
    )
    completed: list[str] = Field(
        default_factory=list,
        description="List mã môn (6 chữ số) đã hoàn thành. Có thể rỗng (cold start).",
    )
    in_progress: list[str] = Field(
        default_factory=list,
        description="Mã môn đang học cùng kỳ (để xét điều kiện song hành).",
    )
    hk_hien_tai: int | None = Field(
        default=None, description="Học kỳ chuẩn bị đăng ký (1..9)."
    )
    gpa: float | None = Field(default=None, description="GPA tích lũy thang 4 hoặc 10.")
    dinh_huong: str | None = Field(
        default=None,
        description="Cluster định hướng nghề: 'AI/ML', 'DB/BigData', 'Java/Web', v.v.",
    )
    max_new_tokens: int = Field(
        default=512, ge=64, le=1024, description="Cap generation length cho LLM mode."
    )


class RetrievedItem(BaseModel):
    doc_id: str
    ten_mon: str
    ma_mon: str
    so_tc: int | str = ""
    loai: str = ""
    score: float


class ConstraintInfo(BaseModel):
    valid_count: int
    violation_count: int
    total_tc: int | float
    warnings: list[str]


class AnswerResponse(BaseModel):
    query: str
    nganh: str
    response: str = Field(..., description="Text trả lời tự nhiên từ generator.")
    recommendations: list[RetrievedItem] = Field(
        ..., description="Danh sách môn gợi ý cuối cùng (đã parse + map metadata)."
    )
    retrieved_top: list[RetrievedItem] = Field(
        ..., description="Top 10 từ retriever (trước constraint), dùng để debug/inspect."
    )
    constraint: ConstraintInfo | None = None
    timing_ms: float


# ===== Schemas cho /chat multi-turn (Stage 6 rewrite) =====


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ProfilePayload(BaseModel):
    """Profile sinh viên gửi cùng request /chat (lấy từ StudentProfile FE đã parse)."""

    nganh: str = Field(..., pattern="^(CS|IS|DS|SE|IT)$")
    completed: list[str] = Field(
        default_factory=list, description="List mã môn 6 chữ số đã đạt."
    )
    in_progress: list[str] = Field(default_factory=list)
    summary_md: str = Field(
        "", description="Bảng điểm dạng markdown — inject vào prompt turn 1."
    )
    hk_target: int | None = Field(
        default=None,
        ge=1,
        le=9,
        description="HK chuẩn bị đăng ký môn tự chọn (1-9). Lọc curriculum theo HK này.",
    )
    only_tu_chon: bool = Field(
        default=True,
        description="Chỉ gợi ý môn tự chọn (mặc định). False = cho cả bắt buộc.",
    )


class ChatRequest(BaseModel):
    profile: ProfilePayload
    messages: list[ChatMessage] = Field(
        ..., min_length=1, description="Lịch sử chat, phần tử cuối phải là role='user'."
    )
    max_new_tokens: int = Field(default=768, ge=128, le=1536)


class ChatResponseModel(BaseModel):
    response: str
    intent: str = Field(
        default="recommend", description="recommend / regulation / prereq."
    )
    recommendations: list[RetrievedItem]
    context_docs: list[RetrievedItem] = Field(
        default_factory=list,
        description="Pool môn thật sự bot xét (intent recommend).",
    )
    retrieved_top: list[RetrievedItem]
    target_courses: list[str] = Field(
        default_factory=list, description="Mã môn extract từ query (intent prereq)."
    )
    constraint: ConstraintInfo | None = None
    used_query: str
    timing_ms: float


# ===== Schemas cho v2 chat (Stage 7 — grades dict thay vì xlsx summary) =====


class ChatV2Request(BaseModel):
    """Request /chat v2: profile build từ grades dict + nganh + hk_target."""

    nganh: str = Field(..., pattern="^(CS|IS|DS|SE|IT)$")
    hk_target: int = Field(..., ge=1, le=9)
    grades: dict[str, float] = Field(
        default_factory=dict,
        description="Map ma_mon (6 chữ số) → diem (0-10). Bỏ trống = chưa học.",
    )
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int = Field(default=768, ge=128, le=1536)


class GradeTableItem(BaseModel):
    ma_mon: str
    ten_mon: str
    hk_chuan: int
    loai: str
    so_tc: int
    prereq: list[str] = Field(default_factory=list)
    loai_dieu_kien: str | None = None
    khong_tinh_gpa: bool = False


# ===== Pipeline singleton + lifespan =====


class _State:
    pipeline: RagPipeline | None = None
    mode: str = "uninitialized"


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline 1 lần khi server start, giải phóng khi shutdown."""
    use_llm = _env_bool("USE_LLM", default=False)
    use_reranker = _env_bool("USE_RERANKER", default=False)
    load_4bit = _env_bool("LOAD_4BIT", default=True)
    top_k = int(os.getenv("TOP_K", "10"))

    print(f"[api] startup: USE_LLM={use_llm} USE_RERANKER={use_reranker} TOP_K={top_k}")
    t0 = time.time()
    state.pipeline = RagPipeline.load_default(
        use_llm=use_llm,
        use_reranker=use_reranker,
        load_4bit=load_4bit,
        top_k_context=top_k,
    )
    state.mode = f"{'qwen-7b-lora' if use_llm else 'stub'}+{'rerank' if use_reranker else 'norerank'}"
    print(f"[api] pipeline loaded in {time.time()-t0:.1f}s — mode={state.mode}")
    yield
    print("[api] shutdown")
    state.pipeline = None


app = FastAPI(
    title="CK_NLP RAG Chatbot",
    description="RAG tư vấn học phần — Giai đoạn 6.",
    version="0.6.0",
    lifespan=lifespan,
)

# Cho Streamlit local + dev tools gọi cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Endpoints =====


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check."""
    return {
        "status": "ok" if state.pipeline is not None else "loading",
        "mode": state.mode,
    }


@app.get("/info")
def info() -> dict[str, Any]:
    """Metadata pipeline để debug + UI hiển thị."""
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa load xong.")
    return {
        "mode": state.mode,
        "corpus_size": len(state.pipeline.id2doc),
        "supported_nganh": ["CS", "IS", "DS", "SE", "IT"],
        "candidate_pool": state.pipeline.candidate_pool,
        "top_k_context": state.pipeline.top_k_context,
    }


def _item_from_doc(doc_id: str, score: float, id2doc: dict) -> RetrievedItem:
    """Map doc_id + score → RetrievedItem có metadata."""
    meta = id2doc.get(doc_id, {})
    return RetrievedItem(
        doc_id=doc_id,
        ten_mon=meta.get("ten_mon", "?"),
        ma_mon=meta.get("ma_mon", doc_id.split("_")[-1] if "_" in doc_id else "?"),
        so_tc=meta.get("so_tc", ""),
        loai=meta.get("loai", ""),
        score=float(score),
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    """Chạy pipeline trên 1 câu hỏi, trả về response + recommendations."""
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa load xong.")

    t0 = time.time()
    try:
        result = state.pipeline.answer(
            query=req.query,
            nganh=req.nganh,
            completed=req.completed or None,
            in_progress=req.in_progress or None,
            hk_hien_tai=req.hk_hien_tai,
            gpa=req.gpa,
            dinh_huong=req.dinh_huong,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e
    elapsed_ms = (time.time() - t0) * 1000

    # Map recommendations doc_id → metadata + retrieval score (nếu có).
    score_map = dict(result.retrieved)
    recs = [
        _item_from_doc(did, score_map.get(did, 0.0), state.pipeline.id2doc)
        for did in result.recommendations
    ]
    retrieved_top = [
        _item_from_doc(did, sc, state.pipeline.id2doc)
        for did, sc in result.retrieved[:10]
    ]

    constraint_info: ConstraintInfo | None = None
    if result.constraint is not None:
        constraint_info = ConstraintInfo(
            valid_count=len(result.constraint.valid),
            violation_count=len(result.constraint.violations),
            total_tc=result.constraint.total_tc,
            warnings=list(result.constraint.warnings),
        )

    return AnswerResponse(
        query=req.query,
        nganh=req.nganh,
        response=result.response,
        recommendations=recs,
        retrieved_top=retrieved_top,
        constraint=constraint_info,
        timing_ms=elapsed_ms,
    )


@app.post("/chat", response_model=ChatResponseModel)
def chat(req: ChatRequest) -> ChatResponseModel:
    """Multi-turn chat — wrap `RagPipeline.chat()`.

    Streamlit gọi endpoint này mỗi lần user gửi message mới. Backend stateless,
    UI giữ messages history qua `st.session_state`.
    """
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa load xong.")
    if req.messages[-1].role != "user":
        raise HTTPException(
            status_code=400, detail="messages[-1] phải có role='user'."
        )

    t0 = time.time()
    try:
        result = state.pipeline.chat(
            messages=[m.model_dump() for m in req.messages],
            nganh=req.profile.nganh,
            completed=req.profile.completed or None,
            in_progress=req.profile.in_progress or None,
            profile_summary_md=req.profile.summary_md,
            hk_target=req.profile.hk_target,
            only_tu_chon=req.profile.only_tu_chon,
            max_new_tokens=req.max_new_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e
    elapsed_ms = (time.time() - t0) * 1000

    score_map = dict(result.retrieved)
    recs = [
        _item_from_doc(did, score_map.get(did, 0.0), state.pipeline.id2doc)
        for did in result.recommendations
    ]
    retrieved_top = [
        _item_from_doc(did, sc, state.pipeline.id2doc)
        for did, sc in result.retrieved[:10]
    ]

    constraint_info: ConstraintInfo | None = None
    if result.constraint is not None:
        constraint_info = ConstraintInfo(
            valid_count=len(result.constraint.valid),
            violation_count=len(result.constraint.violations),
            total_tc=result.constraint.total_tc,
            warnings=list(result.constraint.warnings),
        )

    # context_docs đã có trong result — bot thật sự chỉ xét các môn này.
    context_items = [
        RetrievedItem(
            doc_id=d.get("doc_id", "?"),
            ten_mon=d.get("ten_mon", "?"),
            ma_mon=d.get("ma_mon", "?"),
            so_tc=d.get("so_tc", ""),
            loai=d.get("loai", ""),
            score=float(score_map.get(d.get("doc_id", ""), 0.0)),
        )
        for d in result.context_docs
    ]

    return ChatResponseModel(
        response=result.response,
        recommendations=recs,
        context_docs=context_items,
        retrieved_top=retrieved_top,
        constraint=constraint_info,
        used_query=result.used_query,
        timing_ms=elapsed_ms,
    )


# ===== Endpoints Stage 7 (grade-table + chat v2) =====


@app.get("/curriculum/grade-table", response_model=list[GradeTableItem])
def curriculum_grade_table(nganh: str, hk_target: int) -> list[GradeTableItem]:
    """Trả list môn cần điền điểm (HK1..HK(hk_target-1) trong ngành).

    Frontend Streamlit gọi mỗi khi user đổi `nganh` hoặc `hk_target` để render
    bảng grade table.
    """
    if nganh not in ("CS", "IS", "DS", "SE", "IT"):
        raise HTTPException(status_code=400, detail=f"nganh không hợp lệ: {nganh}")
    try:
        rows = build_grade_table(nganh, hk_target)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return [GradeTableItem(**r) for r in rows]


@app.post("/chat/v2", response_model=ChatResponseModel)
def chat_v2(req: ChatV2Request) -> ChatResponseModel:
    """Multi-intent chat (Stage 7) — input là grades dict, không cần upload file.

    Khác `/chat` cũ:
    - Profile build trực tiếp từ `grades` + `nganh` + `hk_target`.
    - Tự động dispatch intent: recommend / regulation / prereq.
    """
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa load xong.")
    if req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="messages[-1] phải có role='user'.")

    t0 = time.time()

    # Build profile từ grades + curriculum.
    try:
        rows = build_grade_table(req.nganh, req.hk_target)
        so_tc = load_so_tc_map("data/embeddings/corpus.jsonl")
        # so_tc_map có thể không đủ nếu mã môn không có trong corpus; rows có
        # so_tc nên build_profile dùng meta_by_ma OK.
        _ = so_tc
        profile = build_profile_from_grades(
            req.nganh, req.grades, rows, hk_target=req.hk_target
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        result = state.pipeline.chat(
            messages=[m.model_dump() for m in req.messages],
            nganh=req.nganh,
            completed=profile.completed or None,
            in_progress=None,
            profile_summary_md=profile.summary_markdown(),
            hk_target=req.hk_target,
            only_tu_chon=True,
            max_new_tokens=req.max_new_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e
    elapsed_ms = (time.time() - t0) * 1000

    score_map = dict(result.retrieved)
    recs = [
        _item_from_doc(did, score_map.get(did, 0.0), state.pipeline.id2doc)
        for did in result.recommendations
    ]
    retrieved_top = [
        _item_from_doc(did, sc, state.pipeline.id2doc)
        for did, sc in result.retrieved[:10]
    ]
    context_items = [
        RetrievedItem(
            doc_id=d.get("doc_id", "?"),
            ten_mon=d.get("ten_mon", "?"),
            ma_mon=d.get("ma_mon", "?"),
            so_tc=d.get("so_tc", ""),
            loai=d.get("loai", ""),
            score=float(score_map.get(d.get("doc_id", ""), 0.0)),
        )
        for d in result.context_docs
    ]
    constraint_info: ConstraintInfo | None = None
    if result.constraint is not None:
        constraint_info = ConstraintInfo(
            valid_count=len(result.constraint.valid),
            violation_count=len(result.constraint.violations),
            total_tc=result.constraint.total_tc,
            warnings=list(result.constraint.warnings),
        )

    return ChatResponseModel(
        response=result.response,
        intent=result.intent,
        recommendations=recs,
        context_docs=context_items,
        retrieved_top=retrieved_top,
        target_courses=list(result.target_courses),
        constraint=constraint_info,
        used_query=result.used_query,
        timing_ms=elapsed_ms,
    )


def _main() -> None:
    """Entry point khi chạy `python api/main.py`."""
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    _main()
