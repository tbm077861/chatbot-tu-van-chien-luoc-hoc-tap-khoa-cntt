"""Parse template `manual_satisfaction_template.md` sau khi user fill, tổng hợp điểm.

Đọc từng `## Case N — Ngành X` block, tìm dòng `- [x] **K — ...**` để
lấy điểm 1-5 + dòng `**Ghi chú**:` để lấy note.

Output:
- Bảng phân bố điểm theo ngành.
- Avg score tổng + theo ngành.
- User satisfaction rate (≥ 4).
- Ghi JSON tóm tắt để có thể chèn vào STATUS.

Ví dụ dùng:
    python -m src.evaluation.parse_manual_eval \\
        --input data/evaluation/manual_satisfaction_template.md \\
        --output data/evaluation/manual_satisfaction_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


_RE_CASE_HEADER = re.compile(r"^##\s+Case\s+(\d+)\s+—\s+Ngành\s+(\w+)", re.MULTILINE)
_RE_TICK = re.compile(r"^- \[x\] \*\*(\d) —", re.MULTILINE | re.IGNORECASE)
_RE_NOTE = re.compile(
    r"^\*\*Ghi chú\*\*:\s*(.+?)$", re.MULTILINE | re.DOTALL
)


def parse_template(md_path: Path) -> list[dict]:
    """Trả list dict {case_idx, nganh, rating, note} với rating None nếu chưa fill."""
    text = md_path.read_text(encoding="utf-8")

    # Split thành block theo header. Lưu offset để lấy content giữa headers.
    headers = list(_RE_CASE_HEADER.finditer(text))
    cases: list[dict] = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        case_idx = int(m.group(1))
        nganh = m.group(2)

        ticks = _RE_TICK.findall(block)
        rating = int(ticks[0]) if ticks else None

        note_match = _RE_NOTE.search(block)
        note = note_match.group(1).strip() if note_match else ""
        # Loại bỏ placeholder mặc định.
        if note.startswith("_(") or "viết ở đây" in note:
            note = ""

        cases.append(
            {"case": case_idx, "nganh": nganh, "rating": rating, "note": note}
        )
    return cases


def summarize(cases: list[dict]) -> dict:
    n_total = len(cases)
    n_filled = sum(1 for c in cases if c["rating"] is not None)
    n_unfilled = n_total - n_filled

    dist = {k: 0 for k in (1, 2, 3, 4, 5)}
    sum_score = 0
    by_nganh: dict[str, list[int]] = defaultdict(list)
    for c in cases:
        if c["rating"] is None:
            continue
        dist[c["rating"]] += 1
        sum_score += c["rating"]
        by_nganh[c["nganh"]].append(c["rating"])

    avg = sum_score / n_filled if n_filled else 0.0
    satisfied = dist[4] + dist[5]
    satisfaction_rate = satisfied / n_filled if n_filled else 0.0

    by_nganh_avg = {
        nganh: sum(scores) / len(scores) for nganh, scores in by_nganh.items()
    }

    return {
        "n_total": n_total,
        "n_filled": n_filled,
        "n_unfilled": n_unfilled,
        "rating_distribution": dist,
        "avg_score": avg,
        "satisfaction_rate_ge4": satisfaction_rate,
        "by_nganh_avg": by_nganh_avg,
    }


def format_table(summary: dict) -> str:
    dist = summary["rating_distribution"]
    n = summary["n_filled"] or 1
    lines = [
        "## Phân bố điểm",
        "",
        "| Mức | Mô tả | Số case | % |",
        "|---:|---|---:|---:|",
    ]
    labels = {5: "Xuất sắc", 4: "Tốt", 3: "Tạm", 2: "Yếu", 1: "Tệ"}
    for k in (5, 4, 3, 2, 1):
        cnt = dist[k]
        pct = cnt / n * 100
        lines.append(f"| {k} | {labels[k]} | {cnt} | {pct:.1f}% |")
    lines.append(f"| Tổng (đã fill) | | {summary['n_filled']} | 100% |")
    if summary["n_unfilled"]:
        lines.append(f"| Chưa fill | | {summary['n_unfilled']} | — |")

    lines += [
        "",
        f"**Avg score**: {summary['avg_score']:.3f}/5",
        f"**Satisfaction rate (≥ 4)**: {summary['satisfaction_rate_ge4']*100:.1f}%",
        "",
        "## Avg theo ngành",
        "",
        "| Ngành | Avg | N |",
        "|---|---:|---:|",
    ]
    for nganh in sorted(summary["by_nganh_avg"]):
        avg = summary["by_nganh_avg"][nganh]
        n_n = sum(1 for k, v in summary.get("_per_nganh_n", {}).items() if k == nganh)
        lines.append(f"| {nganh} | {avg:.3f} | — |")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/evaluation/manual_satisfaction_template.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/manual_satisfaction_summary.json"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Không tìm thấy {args.input}")

    cases = parse_template(args.input)
    summary = summarize(cases)

    print(format_table(summary))

    if summary["n_filled"] < summary["n_total"]:
        print(
            f"\n[warn] Còn {summary['n_unfilled']}/{summary['n_total']} case "
            "chưa tick — kết quả tạm thời."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "cases": cases}, f, ensure_ascii=False, indent=2
        )
    print(f"\nĐã ghi {args.output}.")


if __name__ == "__main__":
    main()
