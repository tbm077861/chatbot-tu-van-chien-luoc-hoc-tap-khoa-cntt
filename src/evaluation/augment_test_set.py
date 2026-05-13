"""Augment `data/embeddings/test.jsonl` với 3 trường profile (Giai đoạn 5+).

Mục đích: test set Stage 3 chỉ có `(query, positive_doc_ids, source, nganh)`.
Để eval pipeline Stage 5 ở 2 mode (cold/warm), cần thêm:
- `hk_completed`: học kỳ sinh viên vừa hoàn thành (parse từ query text).
- `hk_target`: học kỳ chuẩn bị đăng ký.
- `completed_ma_mon`: list mã môn HK1..hk_completed (tất cả bắt buộc + tự chọn
  có trong curriculum chuẩn).

Output: `data/embeddings/test_with_profile.jsonl` (superset của test.jsonl).
Mỗi dòng giữ nguyên 4 trường cũ + 3 trường mới.

Pattern parsing query (đã test 500/500 query match):
- completed: "hoàn thành tới HK\\d" | "hoàn thành học kỳ \\d" | "vừa/đã xong HK\\d"
- target: "đăng ký .*? HK\\d" | "đăng ký .*? học kỳ \\d" | "HK\\d em nên đăng ký"

Lưu ý: completed_ma_mon là **ALL môn HK1..hk_completed của ngành** (best-case
warm), không phải subset thực mà SV đã học. Mục đích eval là kiểm tra pipeline
khi có đủ profile, không phải mô phỏng chính xác lịch sử cá nhân.

CLI:
    python -m src.evaluation.augment_test_set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"
CURR_DIR = ROOT / "data" / "processed" / "curriculum_graph"

sys.path.insert(0, str(ROOT))


# Regex bắt HK đã hoàn thành (đa pattern, đủ cho 100% test set).
_COMPLETED_PATTERNS = [
    re.compile(r"hoàn thành tới HK\s*(\d)", re.IGNORECASE),
    re.compile(r"hoàn thành HK\s*(\d)", re.IGNORECASE),
    re.compile(r"hoàn thành học kỳ\s*(\d)", re.IGNORECASE),
    re.compile(r"vừa xong HK\s*(\d)", re.IGNORECASE),
    re.compile(r"đã xong HK\s*(\d)", re.IGNORECASE),
]

# Regex bắt HK target (đăng ký).
_TARGET_PATTERNS = [
    re.compile(r"đăng ký.*?HK\s*(\d)", re.IGNORECASE),
    re.compile(r"đăng ký.*?học kỳ\s*(\d)", re.IGNORECASE),
    re.compile(r"(?:Học kỳ|HK)\s*(\d)\s+em nên", re.IGNORECASE),
]


def parse_hk(text: str, patterns: list[re.Pattern]) -> int | None:
    """Trả int HK đầu tiên match một trong các pattern."""
    for p in patterns:
        m = p.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def load_curriculum_by_nganh() -> dict[str, dict]:
    """Load 5 curriculum JSON → {nganh: dict}."""
    out: dict[str, dict] = {}
    for nganh in ("CS", "IS", "DS", "SE", "IT"):
        p = CURR_DIR / f"{nganh}_curriculum.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            out[nganh] = json.load(f)
    return out


def build_completed_ma_mon(
    curriculum: dict,
    hk_completed: int,
    include_tu_chon: bool = True,
) -> list[str]:
    """Trả list mã môn HK1..hk_completed (mặc định gồm cả tự chọn).

    Args:
        curriculum: dict đã load từ <NGANH>_curriculum.json.
        hk_completed: học kỳ vừa hoàn thành (1-based).
        include_tu_chon: nếu False, chỉ lấy môn `loai == "bat_buoc"`.

    Returns:
        List mã môn 6 chữ số.
    """
    ma_list: list[str] = []
    for hk in curriculum["hoc_ky"]:
        if hk["hk_so"] > hk_completed:
            break
        for hp in hk["hoc_phan"]:
            if include_tu_chon or hp.get("loai") == "bat_buoc":
                ma_list.append(hp["ma_mon"])
    return ma_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=EMB_DIR / "test.jsonl",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=EMB_DIR / "test_with_profile.jsonl",
    )
    ap.add_argument(
        "--only-bat-buoc",
        action="store_true",
        help="completed_ma_mon chỉ chứa môn bắt buộc.",
    )
    args = ap.parse_args()

    curr_by_nganh = load_curriculum_by_nganh()
    print(f"[curriculum] loaded {len(curr_by_nganh)} ngành")

    items: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"[input] {args.input}: {len(items)} records")

    stats = {
        "matched_completed": 0,
        "matched_target": 0,
        "missing_curriculum": 0,
    }
    out_items: list[dict] = []
    for it in items:
        new = dict(it)
        nganh = it.get("nganh", "")
        query = it["query"]

        hk_completed = parse_hk(query, _COMPLETED_PATTERNS)
        hk_target = parse_hk(query, _TARGET_PATTERNS)
        if hk_completed is not None:
            stats["matched_completed"] += 1
        if hk_target is not None:
            stats["matched_target"] += 1

        new["hk_completed"] = hk_completed
        new["hk_target"] = hk_target

        completed_ma_mon: list[str] = []
        if hk_completed is not None and nganh in curr_by_nganh:
            completed_ma_mon = build_completed_ma_mon(
                curr_by_nganh[nganh],
                hk_completed,
                include_tu_chon=not args.only_bat_buoc,
            )
        elif nganh not in curr_by_nganh:
            stats["missing_curriculum"] += 1
        new["completed_ma_mon"] = completed_ma_mon

        out_items.append(new)

    with open(args.output, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[stats] matched_completed: {stats['matched_completed']}/{len(items)}")
    print(f"[stats] matched_target:    {stats['matched_target']}/{len(items)}")
    print(f"[stats] missing_curriculum: {stats['missing_curriculum']}")
    # Phân bố hk_completed.
    from collections import Counter

    dist = Counter(it["hk_completed"] for it in out_items)
    print(f"[dist] hk_completed: {dict(sorted(dist.items(), key=lambda x: (x[0] is None, x[0])))}")
    avg_completed = (
        sum(len(it["completed_ma_mon"]) for it in out_items) / len(out_items)
    )
    print(f"[avg] completed_ma_mon size: {avg_completed:.1f} môn/query")
    print(f"[save] -> {args.output}")


if __name__ == "__main__":
    main()
