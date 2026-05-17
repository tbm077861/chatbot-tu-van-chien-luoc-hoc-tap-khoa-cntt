"""Prompt templates tiếng Việt cho M5 Qwen generator (Giai đoạn 5).

SYSTEM_PROMPT bắt buộc phải khớp **chính xác** với prompt lúc fine-tune LoRA
(`src/models/prepare_sft_data.py`). Lệch SYSTEM → distribution shift → output
ngắn/sai format (lesson learned từ M5 eval v1 → v2, xem STATUS.md 2026-05-12).

USER message kết cấu 3 phần:
1. Profile sinh viên (ngành, HK, GPA, môn đã học, định hướng).
2. Retrieved context (top-K môn cùng metadata) — đây là phần RAG augmentation.
3. Câu hỏi cụ thể.

Khi profile/context rỗng (cold-start), vẫn dùng template nhưng để trống các
trường — model học cách handle qua training data.
"""

from __future__ import annotations


# Lấy đúng từ prepare_sft_data.py (Stage 4) để khớp distribution training.
SYSTEM_PROMPT = (
    "Bạn là chatbot tư vấn đăng ký học phần thông minh tại trường đại học. "
    "Nhiệm vụ của bạn là giúp sinh viên lập kế hoạch học tập tối ưu dựa trên "
    "chương trình khung ngành học, các ràng buộc tiên quyết, lịch sử điểm và "
    "định hướng nghề nghiệp của sinh viên, và quy định học vụ (giới hạn tín "
    "chỉ, điều kiện tốt nghiệp). "
    "Luôn giải thích ngắn gọn lý do gợi ý. Trả lời bằng tiếng Việt."
)


NGANH_TEN: dict[str, str] = {
    "CS": "Khoa học Máy tính",
    "IS": "Hệ thống Thông tin",
    "DS": "Khoa học Dữ liệu",
    "SE": "Kỹ thuật Phần mềm",
    "IT": "Công nghệ Thông tin",
}


def format_context(docs: list[dict]) -> str:
    """Format danh sách retrieved doc → block context dạng list đánh số.

    Args:
        docs: list dict với khoá: `doc_id`, `ten_mon`, `ma_mon`, `so_tc`, `loai`,
            (tuỳ chọn) `hk_chuan`, `prereq_info`.

    Returns:
        Chuỗi nhiều dòng, mỗi dòng 1 môn — để LLM tham chiếu khi sinh response.
        Trả chuỗi rỗng nếu list rỗng (giảm noise prompt).
    """
    if not docs:
        return ""
    lines = []
    for i, d in enumerate(docs, 1):
        ten = d.get("ten_mon", "?")
        ma = d.get("ma_mon", "?")
        tc = d.get("so_tc", "?")
        loai = d.get("loai", "?")
        extra: list[str] = []
        hk = d.get("hk_chuan")
        if hk:
            extra.append(f"HK chuẩn {hk}")
        prereq = d.get("prereq_info")
        if prereq:
            extra.append(f"tiên quyết: {prereq}")
        extra_str = f" ({'; '.join(extra)})" if extra else ""
        lines.append(f"{i}. {ten} (mã {ma}, {tc} TC, {loai}){extra_str}")
    return "\n".join(lines)


def build_user_message(
    question: str,
    nganh: str | None = None,
    hk_hien_tai: int | None = None,
    gpa: float | None = None,
    da_hoan_thanh: list[str] | None = None,
    diem_tu_chon: dict[str, float] | None = None,
    dinh_huong: str | None = None,
    retrieved_context: str = "",
) -> str:
    """Tạo USER message theo template project_instructions.md mục 9.

    Profile fields đều optional — chỉ chèn block "## Thông tin sinh viên" khi
    có ít nhất 1 thông tin. Tương tự với retrieved context.

    Args:
        question: câu hỏi gốc của sinh viên (luôn bắt buộc).
        nganh: mã ngành (CS/IS/DS/SE/IT).
        hk_hien_tai: học kỳ chuẩn bị đăng ký.
        gpa: GPA tích lũy thang 4.
        da_hoan_thanh: list mã môn đã hoàn thành.
        diem_tu_chon: dict {ma_mon: diem}.
        dinh_huong: cluster định hướng (AI/ML, DB/BigData, v.v.).
        retrieved_context: output của `format_context()`.

    Returns:
        Chuỗi USER content sẵn sàng đưa vào chat template.
    """
    parts: list[str] = []

    # Block profile (chỉ thêm khi có info).
    profile_lines: list[str] = []
    if nganh:
        ten_nganh = NGANH_TEN.get(nganh, nganh)
        profile_lines.append(f"- Ngành: {nganh} ({ten_nganh})")
    if hk_hien_tai is not None:
        profile_lines.append(f"- Học kỳ hiện tại: {hk_hien_tai}")
    if gpa is not None:
        profile_lines.append(f"- GPA tích lũy: {gpa:.2f}")
    if da_hoan_thanh:
        profile_lines.append(
            f"- Môn đã hoàn thành ({len(da_hoan_thanh)}): {', '.join(da_hoan_thanh[:20])}"
            + (" ..." if len(da_hoan_thanh) > 20 else "")
        )
    if diem_tu_chon:
        items = [f"{ma}={d:.1f}" for ma, d in list(diem_tu_chon.items())[:10]]
        profile_lines.append(
            "- Điểm học phần tự chọn: " + ", ".join(items)
            + (" ..." if len(diem_tu_chon) > 10 else "")
        )
    if dinh_huong:
        profile_lines.append(f"- Định hướng: {dinh_huong}")
    if profile_lines:
        parts.append("## Thông tin sinh viên\n" + "\n".join(profile_lines))

    # Block context.
    if retrieved_context.strip():
        parts.append("## Học phần liên quan (top-K từ hệ thống)\n" + retrieved_context)

    # Câu hỏi.
    parts.append("## Câu hỏi\n" + question.strip())

    return "\n\n".join(parts)


# ===== Cluster mapping cho fallback explanation (Stage 7) =====
#
# Map keyword định hướng (user gõ trong chat) → cluster + lý do template.
# Dùng khi Qwen LoRA trả response quá ngắn — Python tự generate explanation.

_CLUSTER_KEYWORDS: list[tuple[str, str, str]] = [
    # (keyword, cluster_name, reason_template)
    (
        "ai/ml",
        "AI/ML",
        "phục vụ định hướng AI/ML — kỹ năng phân tích dữ liệu, lập trình thuật toán, toán nền.",
    ),
    (
        "ai",
        "AI/ML",
        "phục vụ định hướng AI — kỹ năng phân tích dữ liệu, lập trình thuật toán.",
    ),
    (
        "machine learning",
        "AI/ML",
        "phục vụ định hướng Machine Learning — toán, lập trình, phân tích.",
    ),
    (
        "máy học",
        "AI/ML",
        "phục vụ định hướng Máy học — toán, lập trình, xử lý dữ liệu.",
    ),
    (
        "data",
        "Data Science",
        "phục vụ định hướng Khoa học Dữ liệu — phân tích, xử lý big data.",
    ),
    (
        "dữ liệu",
        "Data Science",
        "phục vụ định hướng Dữ liệu — phân tích, xử lý big data.",
    ),
    (
        "web",
        "Web/Java",
        "phục vụ định hướng Web — lập trình ứng dụng, framework, network.",
    ),
    (
        "mobile",
        "Mobile",
        "phục vụ định hướng Mobile — lập trình ứng dụng đa nền tảng.",
    ),
    (
        "nét",
        ".NET",
        "phục vụ định hướng .NET — lập trình hướng đối tượng, sự kiện.",
    ),
    (
        ".net",
        ".NET",
        "phục vụ định hướng .NET — lập trình hướng đối tượng, sự kiện.",
    ),
    (
        "java",
        "Java/Web",
        "phục vụ định hướng Java — lập trình hướng đối tượng, sự kiện, phân tán.",
    ),
]


def _detect_cluster(query: str) -> tuple[str, str]:
    """Trả (cluster_name, reason_template) từ keyword trong query.

    Default: ("Tổng quát", "bổ sung kiến thức theo chương trình khung.")
    """
    q = query.lower()
    for kw, cluster, reason in _CLUSTER_KEYWORDS:
        if kw in q:
            return cluster, reason
    return ("Tổng quát", "bổ sung kiến thức theo chương trình khung và mở rộng kỹ năng nghề.")


def enrich_short_response(
    response: str,
    recommendations: list[str],
    context_docs: list[dict],
    query: str,
    profile_summary_md: str = "",
) -> str:
    """Bổ sung block giải thích nếu Qwen response thiếu (Stage 7 fallback).

    Logic:
    - Detect cluster từ query (keyword AI/ML, Web, Data, ...).
    - Lấy top 3-5 môn từ recommendations (Qwen parsed) hoặc context_docs (fallback).
    - Cho mỗi môn append 2-3 dòng giải thích template (loại, HK, lý do cluster, link điểm).

    Args:
        response: text Qwen đã sinh (có thể ngắn).
        recommendations: list doc_id Qwen parse được.
        context_docs: pool môn bot đã xét.
        query: câu hỏi user.
        profile_summary_md: bảng điểm dạng markdown (dùng để link "điểm cao").

    Returns:
        Response gốc + block giải thích bổ sung.
    """
    cluster, reason = _detect_cluster(query)

    # Ưu tiên dùng recommendations Qwen đã parse; fallback context_docs nếu rỗng.
    rec_set = set(recommendations)
    docs_to_explain: list[dict] = []
    for d in context_docs:
        if d.get("doc_id") in rec_set:
            docs_to_explain.append(d)
    if len(docs_to_explain) < 3:
        # Bổ sung từ pool đầu (Qwen có thể đã miss).
        for d in context_docs[:5]:
            if d.get("doc_id") not in {x.get("doc_id") for x in docs_to_explain}:
                docs_to_explain.append(d)
            if len(docs_to_explain) >= 5:
                break

    if not docs_to_explain:
        return response

    # Trích các môn điểm cao trong profile để reference (nếu có).
    high_score_hint = ""
    if "Top 5 môn điểm cao nhất" in profile_summary_md:
        # Lấy block top điểm cao (sau dòng "Top 5...", trước dòng trống tiếp).
        lines = profile_summary_md.split("\n")
        try:
            start = next(
                i for i, l in enumerate(lines) if "Top 5 môn điểm cao" in l
            ) + 1
            top_lines = []
            for line in lines[start:]:
                line = line.strip()
                if not line or not line.startswith("-"):
                    break
                top_lines.append(line.lstrip("- ").strip())
            if top_lines:
                high_score_hint = (
                    f"\n_Nền tảng bạn đã có_: {'; '.join(top_lines[:3])}.\n"
                )
        except StopIteration:
            pass

    lines = [
        response.strip(),
        "",
        f"### Giải thích chi tiết (định hướng: {cluster})",
        f"_Các môn dưới đây {reason}_",
        high_score_hint,
    ]

    for i, d in enumerate(docs_to_explain[:5], 1):
        ten = d.get("ten_mon", "?")
        ma = d.get("ma_mon", "?")
        tc = d.get("so_tc", "?")
        hk_c = d.get("hk_chuan", "?")
        loai = d.get("loai", "?")
        lines.append(
            f"**{i}. {ten}** (mã {ma}, {tc} TC, {loai}, HK chuẩn {hk_c})"
        )
        lines.append(
            f"  → Phù hợp định hướng **{cluster}**. "
            f"Là môn {loai} của HK{hk_c} — nằm trong kế hoạch chuẩn ngành."
        )
        lines.append("")

    lines.append(
        "_(Phần giải thích này được sinh tự động vì model trả lời ngắn. "
        "Để có giải thích sâu hơn, vui lòng nhập điểm thêm các môn nền tảng "
        "liên quan định hướng — ví dụ Toán ứng dụng, Phương pháp tính, "
        "Lập trình — vào bảng điểm.)_"
    )
    return "\n".join(lines)


# ===== Chat multi-turn (Giai đoạn 6 rewrite) =====
#
# SYSTEM giữ nguyên `SYSTEM_PROMPT` ở trên vì M5 LoRA fine-tune trên prompt
# đó — đổi sẽ gây distribution shift (bài học từ M5 eval v1).
#
# Cải thiện "giải thích chi tiết" thực hiện qua **user message** turn 1:
# inject bảng điểm + định hướng + yêu cầu rõ ràng giải thích. Qwen2.5-Instruct
# multi-turn tự nhớ bảng điểm sang turn 2+.


def build_chat_user_message(
    question: str,
    profile_summary_md: str = "",
    retrieved_context: str = "",
    is_first_turn: bool = True,
    hk_target: int | None = None,
) -> str:
    """User message cho chat mode. Khác `build_user_message` ở 2 điểm:

    1. Profile dạng **markdown summary** đã render sẵn (từ `StudentProfile`),
       không phải list mã môn rời. Chứa GPA, top điểm cao — gợi ý định hướng.
    2. Turn 1 inject đầy đủ profile; turn 2+ chỉ inject retrieved_context +
       câu hỏi (Qwen tự nhớ profile qua conversation history).

    Args:
        question: câu hỏi mới nhất của user.
        profile_summary_md: output của `StudentProfile.summary_markdown()`.
            Bỏ qua nếu không có hoặc không phải turn đầu.
        retrieved_context: output của `format_context()`, đã retrieve cho
            query này. Có thể rỗng nếu retriever không tìm được.
        is_first_turn: True → chèn full profile; False → bỏ profile để tiết
            kiệm token.

    Returns:
        Chuỗi user content sẵn sàng đưa vào chat template.
    """
    parts: list[str] = []

    if is_first_turn and profile_summary_md.strip():
        parts.append("## Bảng điểm sinh viên\n" + profile_summary_md.strip())

    if hk_target is not None:
        parts.append(
            f"## Mục tiêu\nSinh viên chuẩn bị đăng ký môn **tự chọn** "
            f"cho **học kỳ {hk_target}**. Danh sách dưới đây CHỈ chứa "
            f"môn tự chọn của HK{hk_target} trong chương trình khung."
        )

    if retrieved_context.strip():
        parts.append(
            f"## Môn tự chọn HK{hk_target} có thể đăng ký\n" + retrieved_context
            if hk_target is not None
            else "## Học phần liên quan (top-K từ hệ thống)\n" + retrieved_context
        )

    if is_first_turn:
        instr = (
            "Hãy:\n"
            "1. Gợi ý 3-5 môn tự chọn TỪ DANH SÁCH TRÊN phù hợp với định "
            "hướng/yêu cầu của sinh viên (KHÔNG được tự thêm môn ngoài danh sách).\n"
            "2. **Giải thích cụ thể từng môn**: tại sao chọn, liên hệ với "
            "điểm cao trong bảng điểm (nếu có), môn này hỗ trợ kỹ năng gì cho "
            "định hướng nghề.\n"
            "3. Cảnh báo nếu cần (môn nào nên ưu tiên, môn nào nên để sau)."
        )
        parts.append("## Yêu cầu\n" + instr)

    parts.append("## Câu hỏi\n" + question.strip())

    return "\n\n".join(parts)
