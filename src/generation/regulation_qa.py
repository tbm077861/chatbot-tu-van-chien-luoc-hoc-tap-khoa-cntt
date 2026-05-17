"""Regulation Q&A handler — trả lời câu hỏi về quy định học vụ IUH (Stage 7).

Cách hoạt động:
1. Load `data/processed/regulations.json` (9 nhóm rule, ~150 dòng JSON).
2. Format thành block markdown rõ ràng.
3. Inject TOÀN BỘ vào user prompt (rule ngắn, fit context) + câu hỏi.
4. Qwen-7B + LoRA generate response.

Không cần retriever vì regulations.json đủ nhỏ — đảm bảo coverage cao hơn
BM25/embedding retrieve top-K.

Ví dụ dùng:
    handler = RegulationQA()
    response = handler.answer(query, messages_history, generator)
"""

from __future__ import annotations

import json
from pathlib import Path

REGULATIONS_PATH = Path("data/processed/regulations.json")

SYSTEM_REG = (
    "Bạn là chatbot tư vấn quy định học vụ tại Đại học Công nghiệp TP.HCM (IUH). "
    "Trả lời ngắn gọn, chính xác, BÁM SÁT quy định được cung cấp. "
    "Nếu câu hỏi không liên quan đến quy định, hãy nói rõ là không có thông tin. "
    "Trả lời bằng tiếng Việt tự nhiên."
)


def _format_regulations(reg: dict) -> str:
    """Format regulations.json thành block markdown để Qwen dễ đọc."""
    parts: list[str] = []
    parts.append(f"**Nguồn**: {reg.get('nguon', '?')}")

    dk = reg.get("dang_ky_hoc_phan", {})
    parts.append(
        f"\n## Đăng ký học phần\n"
        f"- TC tối thiểu mỗi HK: **{dk.get('tc_min', '?')}**\n"
        f"- TC tối đa mỗi HK: **{dk.get('tc_max', '?')}**"
    )

    cb = reg.get("canh_bao_hoc_tap", {})
    if cb:
        parts.append(
            "\n## Cảnh báo học tập\n"
            f"- TC nợ tối đa: {cb.get('tc_no_max', '?')}\n"
            f"- Tỷ lệ không đạt tối đa: {cb.get('ti_le_khong_dat_max', '?'):.0%}\n"
            f"- ĐTBHL tối thiểu (HK1): {cb.get('dtbhl_min_hk1', '?')}\n"
            f"- ĐTBHL tối thiểu (HK tiếp): {cb.get('dtbhl_min_hk_tiep', '?')}\n"
            f"- ĐTBHLTL năm 1/2/3/4+: "
            f"{cb.get('dtbhltl_nam1','?')}/{cb.get('dtbhltl_nam2','?')}/"
            f"{cb.get('dtbhltl_nam3','?')}/{cb.get('dtbhltl_nam4_plus','?')}\n"
            f"- Bị thôi học sau {cb.get('so_lan_canh_bao_lien_tiep_bi_thoi_hoc', '?')} "
            f"lần cảnh báo liên tiếp"
        )

    tn = reg.get("dieu_kien_tot_nghiep", {})
    if tn:
        parts.append(
            "\n## Điều kiện tốt nghiệp\n"
            f"- GPA tối thiểu (thang 4): **{tn.get('gpa_min_thang4', '?')}**\n"
            + "\n".join(f"- {c}" for c in tn.get("dieu_kien", []))
        )

    bqd = reg.get("bang_quy_doi_diem", [])
    if bqd:
        parts.append("\n## Bảng quy đổi điểm (thang 10 → thang 4)")
        for row in bqd:
            parts.append(
                f"- {row.get('thang10_tu')}-{row.get('thang10_den')} → "
                f"{row.get('thang4')} ({row.get('diem_chu')})"
            )

    xl = reg.get("xep_loai_hoc_luc", [])
    if xl:
        parts.append("\n## Xếp loại học lực (theo GPA thang 4)")
        for row in xl:
            parts.append(
                f"- {row.get('diem_tu')}–{row.get('diem_den')}: {row.get('xep_loai')}"
            )

    dk_hp = reg.get("loai_dieu_kien_hoc_phan", {})
    if dk_hp:
        parts.append("\n## Loại điều kiện học phần")
        for key, val in dk_hp.items():
            parts.append(f"- **({val['ky_hieu']})** {key}: {val['mo_ta']}")

    hl = reg.get("hoc_lai_va_cai_thien", {})
    if hl:
        parts.append("\n## Học lại & cải thiện điểm")
        for key, val in hl.items():
            parts.append(f"- {val}")

    htgpa = reg.get("hoc_phan_dieu_kien", {})
    if htgpa:
        parts.append("\n## Học phần không tính GPA")
        for item in htgpa.get("khong_tinh_gpa", []):
            parts.append(f"- {item}")
        if htgpa.get("ghi_chu"):
            parts.append(f"- _Ghi chú_: {htgpa['ghi_chu']}")

    return "\n".join(parts)


class RegulationQA:
    """Handler cho intent `regulation` — trả lời quy định học vụ qua Qwen."""

    def __init__(self, regulations_path: Path | str = REGULATIONS_PATH) -> None:
        with open(regulations_path, encoding="utf-8") as f:
            self._reg = json.load(f)
        self._reg_md = _format_regulations(self._reg)

    @property
    def regulations_md(self) -> str:
        """Block markdown đầy đủ, expose để debug/inspect."""
        return self._reg_md

    def build_messages(
        self,
        history: list[dict],
        query: str,
    ) -> list[dict]:
        """Build messages list cho Qwen — inject toàn bộ regulations vào user turn.

        Khác `RagPipeline.chat()` ở chỗ KHÔNG inject profile/grade (vì intent
        regulation không cần điểm SV). Giữ history để hỗ trợ multi-turn.
        """
        # User content turn cuối: regulations + câu hỏi.
        user_content = (
            f"## Quy định học vụ\n{self._reg_md}\n\n## Câu hỏi\n{query.strip()}"
        )
        msgs = [{"role": "system", "content": SYSTEM_REG}]
        for m in history[:-1]:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def answer(
        self,
        history: list[dict],
        generator,
        max_new_tokens: int = 512,
    ) -> str:
        """Trả response từ Qwen.

        Args:
            history: list[{role, content}], message cuối phải là user.
            generator: QwenGenerator hoặc StubGenerator có method `chat()`.
            max_new_tokens: cap generation.

        Returns:
            Response text.
        """
        if not history or history[-1].get("role") != "user":
            raise ValueError("history phải kết thúc bằng role='user'")
        query = history[-1]["content"]

        # Detect Stub: không có khả năng hiểu prompt → fallback markdown.
        # Heuristic: class name chứa "Stub".
        if type(generator).__name__.startswith("Stub"):
            return self._stub_fallback(query)

        msgs = self.build_messages(history, query)
        try:
            return generator.chat(msgs, max_new_tokens=max_new_tokens)
        except (TypeError, AttributeError):
            return self._stub_fallback(query)

    def _stub_fallback(self, query: str) -> str:
        """Fallback khi không có Qwen — trả toàn bộ regulations markdown."""
        return (
            f"_(Stub mode — không có LLM để tự nhiên hoá. Đây là quy định "
            f"học vụ IUH có thể trả lời câu hỏi '{query}':)_\n\n"
            f"{self._reg_md}"
        )


def _cli() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    h = RegulationQA()
    print("=== regulations.md (preview) ===")
    print(h.regulations_md[:600])
    print("...\n")
    # Test stub fallback
    history = [{"role": "user", "content": "Số tín chỉ tối đa mỗi HK?"}]

    class _NoChat:
        pass

    print("=== Stub fallback ===")
    print(h.answer(history, _NoChat()))


if __name__ == "__main__":
    _cli()
