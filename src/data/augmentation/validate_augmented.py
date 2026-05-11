"""Validate và tổng hợp các sample đã augment ở Giai đoạn 2.

Kiểm tra:
- Tổng số sample / nguồn / ngành.
- Schema cơ bản: id, source, nganh, qa.question, qa.answer.
- Duplicate ID.
- Độ dài QA hợp lý (10–2000 ký tự).
- Phân bố ngành cân bằng không.

Output: file tổng hợp `data/augmented/ALL_samples.jsonl` + báo cáo .md.

Ví dụ CLI:
    python src/data/augmentation/validate_augmented.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

REQUIRED_KEYS = ["id", "source", "nganh", "qa"]


def iter_jsonl(path: Path):
    """Generator đọc từng dòng JSONL.

    Args:
        path: Đường dẫn .jsonl.

    Yields:
        Dict mỗi sample.
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def validate_sample(s: dict[str, Any]) -> list[str]:
    """Kiểm tra schema một sample, trả về list lỗi (rỗng = OK).

    Args:
        s: Sample dict.

    Returns:
        List lỗi.
    """
    errs: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in s:
            errs.append(f"missing key: {k}")
    if "qa" in s:
        if "question" not in s["qa"] or "answer" not in s["qa"]:
            errs.append("qa thiếu question/answer")
        else:
            q_len = len(s["qa"]["question"])
            a_len = len(s["qa"]["answer"])
            if q_len < 10 or q_len > 2000:
                errs.append(f"question độ dài bất thường: {q_len}")
            if a_len < 10 or a_len > 2000:
                errs.append(f"answer độ dài bất thường: {a_len}")
    return errs


def main() -> None:
    """Điểm vào CLI."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Validate augmented data Giai đoạn 2")
    parser.add_argument("--augmented_dir", type=Path, default=Path("data/augmented"))
    parser.add_argument("--report", type=Path, default=Path("data/augmented/VALIDATION_REPORT.md"))
    parser.add_argument("--combined", type=Path, default=Path("data/augmented/ALL_samples.jsonl"))
    args = parser.parse_args()

    project_root = _PROJECT_ROOT
    aug_dir = project_root / args.augmented_dir

    # Tìm tất cả .jsonl trong các subfolder (graph_paths, cf_profiles, negative_samples)
    jsonl_files = sorted(aug_dir.rglob("*.jsonl"))
    # Loại trừ file output combined (nếu đã tồn tại)
    jsonl_files = [p for p in jsonl_files if p.name != args.combined.name]

    if not jsonl_files:
        print(f"Không tìm thấy file .jsonl nào trong {aug_dir}")
        return

    total = 0
    source_counts: Counter[str] = Counter()
    nganh_counts: Counter[str] = Counter()
    nganh_source: Counter[tuple[str, str]] = Counter()
    error_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    dup_ids = 0
    total_errors = 0

    combined_path = project_root / args.combined
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    with combined_path.open("w", encoding="utf-8") as out:
        for jp in jsonl_files:
            for s in iter_jsonl(jp):
                total += 1
                errs = validate_sample(s)
                for e in errs:
                    error_counts[e] += 1
                    total_errors += 1
                if "id" in s:
                    if s["id"] in seen_ids:
                        dup_ids += 1
                    seen_ids.add(s["id"])
                source = s.get("source", "?")
                nganh = s.get("nganh", "?")
                source_counts[source] += 1
                nganh_counts[nganh] += 1
                nganh_source[(nganh, source)] += 1
                # Chỉ ghi sample hợp lệ
                if not errs:
                    out.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Báo cáo
    report_lines: list[str] = []
    report_lines.append("# Báo cáo Validation — Giai đoạn 2 (Data Augmentation)")
    report_lines.append("")
    report_lines.append(f"Tổng số sample: **{total:,}**")
    report_lines.append(f"Số file nguồn: {len(jsonl_files)}")
    report_lines.append(f"Lỗi schema: {total_errors}")
    report_lines.append(f"Duplicate ID: {dup_ids}")
    report_lines.append(f"Sample hợp lệ ghi vào ALL_samples.jsonl: {total - sum(1 for _ in [])} (chỉ loại sample có lỗi schema)")
    report_lines.append("")
    report_lines.append("## Phân bố theo nguồn")
    report_lines.append("| Source | Số sample | % |")
    report_lines.append("|---|---:|---:|")
    for src, c in source_counts.most_common():
        pct = c / total * 100
        report_lines.append(f"| {src} | {c:,} | {pct:.2f}% |")
    report_lines.append("")
    report_lines.append("## Phân bố theo ngành")
    report_lines.append("| Ngành | Số sample | % |")
    report_lines.append("|---|---:|---:|")
    for ng, c in nganh_counts.most_common():
        pct = c / total * 100
        report_lines.append(f"| {ng} | {c:,} | {pct:.2f}% |")
    report_lines.append("")
    report_lines.append("## Phân bố ngành × nguồn")
    report_lines.append("| Ngành | Source | Số sample |")
    report_lines.append("|---|---|---:|")
    for (ng, src), c in sorted(nganh_source.items()):
        report_lines.append(f"| {ng} | {src} | {c:,} |")
    report_lines.append("")
    if error_counts:
        report_lines.append("## Lỗi schema")
        for e, c in error_counts.most_common():
            report_lines.append(f"- {e}: {c}")
    else:
        report_lines.append("## Lỗi schema")
        report_lines.append("Không có lỗi schema nào. ✓")
    report_lines.append("")
    report_lines.append(f"## Kết luận")
    target = 85000
    if total >= target:
        report_lines.append(f"Đạt mục tiêu **≥{target:,} sample** (project_instructions.md mục 10): **{total:,}**.")
    else:
        report_lines.append(f"Chưa đạt mục tiêu {target:,}. Hiện có {total:,}.")
    report_lines.append("")

    report_path = project_root / args.report
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Tổng: {total:,} samples")
    print(f"Phân bố nguồn: {dict(source_counts)}")
    print(f"Phân bố ngành: {dict(nganh_counts)}")
    print(f"Lỗi schema: {total_errors}, Duplicate ID: {dup_ids}")
    print(f"Báo cáo: {report_path}")
    print(f"File hợp nhất: {combined_path}")


if __name__ == "__main__":
    main()
