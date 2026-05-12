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
