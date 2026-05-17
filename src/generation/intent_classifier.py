"""Intent classifier heuristic — phân loại câu hỏi của SV (Giai đoạn 7).

3 intent:
- `recommend`: tư vấn chọn môn tự chọn (default).
- `regulation`: hỏi về quy định học vụ (TC max, cảnh báo học vụ, tốt nghiệp...).
- `prereq`: hỏi về khả năng đăng ký 1 môn cụ thể (tiên quyết, song hành...).

Dùng keyword matching đơn giản — đủ cho phổ câu hỏi SV hỏi thực tế. Khi
unclear, default về `recommend` (intent phổ biến nhất).

Ví dụ:
    classify_intent("Em định hướng AI/ML, chọn môn nào HK3?") → "recommend"
    classify_intent("Số TC tối đa mỗi HK là bao nhiêu?") → "regulation"
    classify_intent("Tôi có thể học Máy học ở HK5 không?") → "prereq"
"""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["recommend", "regulation", "prereq"]

# Keywords thiên về tra cứu quy định học vụ (IUH).
_REGULATION_KEYWORDS = [
    "quy định",
    "quy che",
    "quy chế",
    "giới hạn",
    "tín chỉ tối đa",
    "tín chỉ tối thiểu",
    "tc tối",
    "tc max",
    "tc min",
    "cảnh báo học",
    "thôi học",
    "đình chỉ",
    "tốt nghiệp",
    "điều kiện tốt nghiệp",
    "gpa tối thiểu",
    "gpa min",
    "đăng ký vượt",
    "học lại",
    "cải thiện điểm",
    "xếp loại",
    "thang điểm",
    "quy đổi điểm",
    "điểm chữ",
    "diem chu",
    "điểm a+",
    "rớt môn",
    "bao nhiêu tín chỉ",
    "không tính gpa",
    "p/f",
]

# Keywords thiên về hỏi prereq / khả năng đăng ký 1 môn.
_PREREQ_KEYWORDS = [
    "tiên quyết",
    "tien quyet",
    "song hành",
    "song hanh",
    "học trước",
    "học sau",
    "có thể học",
    "co the hoc",
    "có học được",
    "đăng ký được",
    "dang ky duoc",
    "được không",
    "duoc khong",
    "cần học gì",
    "cần học trước",
    "phải học gì",
    "điều kiện học",
    "điều kiện môn",
    "loại điều kiện",
    "mối quan hệ giữa",
    "liên quan giữa",
    "(a)",
    "(b)",
    "(c)",
]

# Pattern: "môn X có thể học/đăng ký" hoặc "học môn X ở HK Y được không"
_PREREQ_PATTERN = re.compile(
    r"(môn|mon)\s+\w+.{0,40}\b(c[óo]\s+(thể|the)|đăng\s*k[ýy]|h[ọo]c)\b",
    re.IGNORECASE,
)


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Lower-cased substring check."""
    low = text.lower()
    return any(kw in low for kw in keywords)


def classify_intent(query: str) -> Intent:
    """Phân loại câu hỏi thành 1 trong 3 intent (heuristic keyword).

    Thứ tự ưu tiên: regulation > prereq > recommend (default).
    Lý do: regulation keywords ít ambiguous nhất; prereq có overlap với
    recommend (vd "nên học môn nào") nên check sau; default recommend.

    Args:
        query: câu hỏi của sinh viên (tiếng Việt).

    Returns:
        "recommend" | "regulation" | "prereq".
    """
    q = query.strip()
    if not q:
        return "recommend"

    if _contains_any(q, _REGULATION_KEYWORDS):
        return "regulation"

    if _contains_any(q, _PREREQ_KEYWORDS) or _PREREQ_PATTERN.search(q):
        return "prereq"

    return "recommend"


def _cli() -> None:
    """Smoke test với vài ví dụ."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    samples = [
        ("Em định hướng AI/ML, chọn môn nào HK3?", "recommend"),
        ("Số TC tối đa mỗi HK là bao nhiêu?", "regulation"),
        ("Tôi có thể học Máy học (mã 015028) ở HK5 không?", "prereq"),
        ("Cảnh báo học vụ khi nào?", "regulation"),
        ("Môn 015028 cần học trước môn gì?", "prereq"),
        ("Em GPA 6.5 muốn cải thiện chọn môn gì?", "recommend"),
        ("Điều kiện tốt nghiệp là gì?", "regulation"),
        ("Mối quan hệ giữa Máy học và Trí tuệ nhân tạo?", "prereq"),
    ]
    pass_n = 0
    for q, expected in samples:
        got = classify_intent(q)
        ok = "OK" if got == expected else "FAIL"
        pass_n += got == expected
        print(f"[{ok}] {got:10s} (expected {expected:10s}) — {q}")
    print(f"\n{pass_n}/{len(samples)} pass")


if __name__ == "__main__":
    _cli()
