"""Prereq Q&A handler — trả lời câu hỏi về tiên quyết / khả năng học môn (Stage 7).

Cách hoạt động:
1. Extract mã môn 6 chữ số hoặc tên môn (fuzzy match) từ query.
2. Lookup prereq từ `ConstraintChecker.get_prereqs()` (gọn lỗi graph).
3. Đối chiếu với `completed` của SV → đủ điều kiện hay thiếu môn gì.
4. Build structured info → inject vào prompt Qwen → response tự nhiên.

Ví dụ:
    handler = PrereqQA(checker, id2doc)
    response = handler.answer(history, nganh, completed, generator)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from src.retrieval.constraint_checker import ConstraintChecker

SYSTEM_PREREQ = (
    "Bạn là chatbot tư vấn đăng ký học phần tại Đại học Công nghiệp TP.HCM. "
    "Nhiệm vụ: trả lời câu hỏi về điều kiện tiên quyết / mối quan hệ giữa các "
    "môn học DỰA VÀO dữ liệu được cung cấp dưới đây. "
    "Nếu sinh viên thiếu điều kiện, hãy chỉ rõ môn nào cần học trước. "
    "Trả lời bằng tiếng Việt, ngắn gọn, có giải thích."
)


# Pattern bắt mã môn 6 chữ số trong query (có/không ngoặc, có/không "mã").
_MA_PATTERN = re.compile(r"(?:m[ãa]\s+)?\(?(\d{6})\)?", re.IGNORECASE)


def _normalize_vn(text: str) -> str:
    """Bỏ dấu tiếng Việt + lowercase để fuzzy match."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


def extract_target_courses(
    query: str,
    id2doc: dict[str, dict],
    nganh: str,
    max_results: int = 3,
) -> list[str]:
    """Detect mã môn trong query (regex 6 chữ số + fuzzy name).

    Args:
        query: câu hỏi user.
        id2doc: map doc_id → metadata (có ten_mon).
        nganh: ngành SV để ưu tiên doc_id `{nganh}_*`.
        max_results: cap số môn trả về (top match đầu tiên).

    Returns:
        List doc_id (vd `["CS_015028"]`) theo thứ tự xuất hiện trong query.
    """
    found: list[str] = []
    seen: set[str] = set()

    # 1. Regex 6-digit code (chính xác nhất).
    for m in _MA_PATTERN.finditer(query):
        ma = m.group(1).zfill(6)
        doc_id = f"{nganh}_{ma}"
        if doc_id in id2doc and doc_id not in seen:
            found.append(doc_id)
            seen.add(doc_id)
        elif doc_id not in seen:
            # Có thể mã thuộc ngành khác — thử các ngành khác.
            for d_id in id2doc:
                if d_id.endswith(f"_{ma}") and d_id not in seen:
                    found.append(d_id)
                    seen.add(d_id)
                    break

    # 2. Fuzzy match tên môn (tìm môn ngành user có tên xuất hiện trong query).
    if len(found) < max_results:
        q_norm = _normalize_vn(query)
        # Sort theo độ dài giảm dần để match câu dài trước ("trí tuệ nhân tạo"
        # match trước "trí tuệ"). Chỉ xét cùng ngành.
        same_nganh = [
            (did, _normalize_vn(meta.get("ten_mon", "")))
            for did, meta in id2doc.items()
            if did.startswith(f"{nganh}_") and meta.get("ten_mon")
        ]
        same_nganh.sort(key=lambda x: -len(x[1]))
        for did, ten_norm in same_nganh:
            if len(found) >= max_results:
                break
            if did in seen:
                continue
            if ten_norm and ten_norm in q_norm:
                found.append(did)
                seen.add(did)

    return found[:max_results]


class PrereqQA:
    """Handler intent `prereq` — extract môn + lookup graph + Qwen response."""

    def __init__(
        self,
        checker: ConstraintChecker,
        id2doc: dict[str, dict],
    ) -> None:
        self.checker = checker
        self.id2doc = id2doc

    def _format_course_info(
        self,
        doc_id: str,
        completed_set: set[str],
    ) -> str:
        """Format thông tin 1 môn + prereq + status đối với SV."""
        meta = self.id2doc.get(doc_id, {})
        ten = meta.get("ten_mon", "?")
        ma = meta.get("ma_mon", "?")
        tc = meta.get("so_tc", "?")
        hk_c = meta.get("hk_chuan", "?")
        loai = meta.get("loai", "?")

        prereqs = self.checker.get_prereqs(doc_id)
        ht = prereqs.get("hoc_truoc", [])
        tq = prereqs.get("tien_quyet", [])
        sh = prereqs.get("song_hanh", [])

        def _line(ma_list: Iterable[str], label: str) -> list[str]:
            lines = []
            for prereq_ma in ma_list:
                prereq_did = f"{doc_id.split('_')[0]}_{prereq_ma}"
                ten_p = self.id2doc.get(prereq_did, {}).get("ten_mon", "?")
                ok = "✓ đã đạt" if prereq_ma in completed_set else "✗ CHƯA"
                lines.append(f"    - **{ten_p}** (mã {prereq_ma}) — {ok}")
            return lines

        parts = [
            f"### {ten} (mã {ma}, {tc} TC, {loai}, HK chuẩn {hk_c})",
        ]

        if not (ht or tq or sh):
            parts.append("- Môn này KHÔNG có điều kiện tiên quyết → có thể học bất cứ HK nào ≥ HK chuẩn.")
        else:
            if ht:
                parts.append("- **Học trước (a)** — phải học (chưa cần đạt) trước:")
                parts.extend(_line(ht, "ht"))
            if tq:
                parts.append("- **Tiên quyết (b)** — phải đạt trước:")
                parts.extend(_line(tq, "tq"))
            if sh:
                parts.append("- **Song hành (c)** — học cùng kỳ được:")
                parts.extend(_line(sh, "sh"))

        # Tổng kết status.
        missing_required = [
            p
            for p in (ht + tq)
            if p not in completed_set
        ]
        if missing_required:
            parts.append(
                f"\n→ **Chưa đủ điều kiện** học môn này. Còn thiếu "
                f"{len(missing_required)} môn (xem dấu ✗ ở trên)."
            )
        else:
            parts.append("\n→ **Đủ điều kiện** đăng ký môn này.")

        if doc_id.split("_", 1)[1] in completed_set:
            parts.append("⚠ SV đã hoàn thành môn này — không cần học lại trừ khi muốn cải thiện điểm.")

        return "\n".join(parts)

    def build_messages(
        self,
        history: list[dict],
        nganh: str,
        completed: list[str] | None,
        hk_target: int | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Build messages list. Trả thêm list doc_id đã extract để debug/UI."""
        if not history or history[-1].get("role") != "user":
            raise ValueError("history[-1] phải có role='user'")
        query = history[-1]["content"]

        targets = extract_target_courses(query, self.id2doc, nganh)
        completed_set = set(completed or [])

        # Build info block.
        if targets:
            info_blocks = [
                self._format_course_info(did, completed_set) for did in targets
            ]
            info_text = "\n\n".join(info_blocks)
        else:
            info_text = (
                "Không nhận diện được môn cụ thể nào trong câu hỏi. "
                "Vui lòng nói rõ tên hoặc mã môn (vd: 'Tôi có thể học môn Máy học "
                "(015028) không?')."
            )

        hk_str = (
            f"Sinh viên đang chuẩn bị đăng ký HK{hk_target}.\n\n"
            if hk_target
            else ""
        )

        user_content = (
            f"{hk_str}"
            f"## Thông tin điều kiện môn học\n{info_text}\n\n"
            f"## Câu hỏi\n{query.strip()}"
        )

        msgs = [{"role": "system", "content": SYSTEM_PREREQ}]
        for m in history[:-1]:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_content})
        return msgs, targets

    def answer(
        self,
        history: list[dict],
        nganh: str,
        completed: list[str] | None,
        generator,
        hk_target: int | None = None,
        max_new_tokens: int = 512,
    ) -> tuple[str, list[str]]:
        """Trả response từ Qwen + list mã môn đã extract.

        Returns:
            (response_text, targets) — targets để UI hiển thị "Đã nhận diện môn".
        """
        msgs, targets = self.build_messages(history, nganh, completed, hk_target)

        # Detect Stub: trả thông tin tiên quyết dạng markdown (không cần LLM).
        if type(generator).__name__.startswith("Stub"):
            response = self._stub_fallback(
                history[-1]["content"], targets, set(completed or [])
            )
            return response, targets

        try:
            response = generator.chat(msgs, max_new_tokens=max_new_tokens)
        except (TypeError, AttributeError):
            response = self._stub_fallback(
                history[-1]["content"], targets, set(completed or [])
            )
        return response, targets

    def _stub_fallback(
        self,
        query: str,  # noqa: ARG002
        targets: list[str],
        completed_set: set[str],
    ) -> str:
        if not targets:
            return (
                "Mình không nhận diện được môn nào trong câu hỏi. Vui lòng "
                "nói rõ mã môn (6 chữ số) hoặc tên môn đầy đủ."
            )
        return "\n\n".join(
            self._format_course_info(did, completed_set) for did in targets
        )
