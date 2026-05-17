"""Sinh grade table cho UI từ curriculum JSON (Giai đoạn 7 — multi-intent chatbot).

Khi user chọn (nganh, hk_target) ở Streamlit sidebar, frontend gọi API để lấy
list môn HK1..HK(target-1) (cả bắt buộc + tự chọn) trong ngành đó. User chỉ
việc điền điểm vào table — KHÔNG cần upload file.

Schema item:
    {
        ma_mon: "001234",
        ten_mon: "Tên môn",
        hk_chuan: 2,
        loai: "bat_buoc" | "tu_chon",
        so_tc: 3,
        prereq: ["003456"],
        loai_dieu_kien: "a" | "b" | "c" | null,
        khong_tinh_gpa: false,
    }

Ví dụ dùng:
    rows = build_grade_table("CS", hk_target=3)
    # Trả ~19 môn của HK1+HK2 ngành CS.
"""

from __future__ import annotations

import json
from pathlib import Path

CURRICULUM_DIR = Path("data/processed/curriculum_graph")


def build_grade_table(
    nganh: str,
    hk_target: int,
    curriculum_dir: Path | str = CURRICULUM_DIR,
) -> list[dict]:
    """Trả list môn cần nhập điểm (HK1..HK(hk_target-1) trong ngành).

    Args:
        nganh: CS / IS / DS / SE / IT.
        hk_target: HK chuẩn bị đăng ký (1-9). Hàm trả môn của HK < hk_target.
        curriculum_dir: thư mục chứa `<NGANH>_curriculum.json`.

    Returns:
        List dict (xem schema module docstring). Sort theo
        `(hk_chuan ASC, loai bat_buoc trước, ma_mon ASC)` cho UX nhất quán.

    Raises:
        FileNotFoundError: ngành không có curriculum JSON.
        ValueError: hk_target ngoài [1, 9].
    """
    if not 1 <= hk_target <= 9:
        raise ValueError(f"hk_target phải trong [1, 9], nhận {hk_target}")

    path = Path(curriculum_dir) / f"{nganh}_curriculum.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict] = []
    for hk in data.get("hoc_ky", []):
        hk_so = hk.get("hk_so")
        if hk_so is None or hk_so >= hk_target:
            continue
        for hp in hk.get("hoc_phan", []):
            ma = str(hp.get("ma_mon", "")).zfill(6)
            if not ma or ma == "000000":
                continue
            rows.append(
                {
                    "ma_mon": ma,
                    "ten_mon": hp.get("ten_mon", "?"),
                    "hk_chuan": hk_so,
                    "loai": hp.get("loai", "?"),
                    "so_tc": hp.get("so_tc", 0),
                    "prereq": hp.get("dieu_kien", []),
                    "loai_dieu_kien": hp.get("loai_dieu_kien"),
                    "khong_tinh_gpa": hp.get("khong_tinh_gpa", False),
                }
            )

    # Sort: HK chuẩn tăng, bắt buộc trước tự chọn, theo ma_mon để ổn định.
    rows.sort(
        key=lambda r: (r["hk_chuan"], 0 if r["loai"] == "bat_buoc" else 1, r["ma_mon"])
    )
    return rows


def _cli() -> None:
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nganh", required=True, choices=["CS", "IS", "DS", "SE", "IT"])
    ap.add_argument("--hk", type=int, required=True, help="HK target (1-9)")
    args = ap.parse_args()

    rows = build_grade_table(args.nganh, args.hk)
    print(f"{args.nganh} HK target={args.hk} → {len(rows)} môn cần điền điểm:")
    print()
    by_hk: dict = {}
    for r in rows:
        by_hk.setdefault(r["hk_chuan"], []).append(r)
    for hk_so in sorted(by_hk):
        print(f"--- HK {hk_so} ---")
        for r in by_hk[hk_so]:
            tag = "[BB]" if r["loai"] == "bat_buoc" else "[TC]"
            print(f"  {tag} {r['ma_mon']}  {r['ten_mon']}  ({r['so_tc']} TC)")
        print()


if __name__ == "__main__":
    _cli()
