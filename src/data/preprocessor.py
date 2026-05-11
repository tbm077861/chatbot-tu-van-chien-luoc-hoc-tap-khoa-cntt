"""Tiền xử lý dữ liệu điểm học phần tự chọn từ CSV.

Xử lý 5 file: CS_TuChon.csv, IS_TuChon.csv, DS_TuChon.csv, SE_TuChon.csv, IT_TuChon.csv.
Output: data/processed/student_profiles/<NGANH>_profiles.parquet + .csv

Ví dụ sử dụng CLI:
    # Xử lý một ngành
    python src/data/preprocessor.py --nganh IS

    # Xử lý tất cả 5 ngành và gộp lại
    python src/data/preprocessor.py --all

    # Xử lý + EDA (in thống kê)
    python src/data/preprocessor.py --all --eda
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

GRADES_DIR = Path("data/raw/grades")
OUTPUT_DIR = Path("data/processed/student_profiles")

# Mapping tên ngành → file
NGANH_FILES: dict[str, str] = {
    "CS": "CS_TuChon.csv",
    "IS": "IS_TuChon.csv",
    "DS": "DS_TuChon.csv",
    "SE": "SE_TuChon.csv",
    "IT": "IT_TuChon.csv",
}

# Bảng quy đổi điểm 10 → 4 theo Điều 19 quy chế đào tạo IUH
DIEM_10_TO_4: list[tuple[float, float, float]] = [
    (9.0, 10.0, 4.0),   # A+
    (8.5, 8.9, 3.8),    # A
    (8.0, 8.4, 3.5),    # B+
    (7.0, 7.9, 3.0),    # B
    (6.0, 6.9, 2.5),    # C+
    (5.5, 5.9, 2.0),    # C
    (5.0, 5.4, 1.5),    # D+
    (4.0, 4.9, 1.0),    # D
    (0.0, 3.9, 0.0),    # F
]

DIEM_10_TO_CHU: list[tuple[float, float, str]] = [
    (9.0, 10.0, "A+"),
    (8.5, 8.9, "A"),
    (8.0, 8.4, "B+"),
    (7.0, 7.9, "B"),
    (6.0, 6.9, "C+"),
    (5.5, 5.9, "C"),
    (5.0, 5.4, "D+"),
    (4.0, 4.9, "D"),
    (0.0, 3.9, "F"),
]


def _detect_sep(path: Path) -> str:
    """Tự động phát hiện separator (comma hoặc pipe) từ dòng header.

    Args:
        path: Đường dẫn file CSV.

    Returns:
        Ký tự separator, mặc định "," nếu không phát hiện được "|".
    """
    with path.open(encoding="utf-8") as f:
        header = f.readline()
    return "|" if "|" in header else ","


def _parse_ten_dot(ten_dot: str) -> tuple[int, str]:
    """Parse cột TenDot thành (hk_so, nam_hoc).

    Args:
        ten_dot: Chuỗi dạng "HK1 (2019-2020)" hoặc "HK2 (2020-2021)".

    Returns:
        Tuple (hk_so: int, nam_hoc: str). Trả về (-1, "") nếu không parse được.
    """
    m = re.match(r"HK(\d+)\s*\((\d{4}-\d{4})\)", str(ten_dot).strip())
    if m:
        return int(m.group(1)), m.group(2)
    return -1, ""


def _diem10_to_diem4(diem: float) -> float:
    """Quy đổi điểm thang 10 sang thang 4 theo bảng quy đổi IUH.

    Args:
        diem: Điểm thang 10 (0.0 – 10.0).

    Returns:
        Điểm thang 4 tương ứng. Trả về 0.0 nếu không khớp.
    """
    for lo, hi, d4 in DIEM_10_TO_4:
        if lo <= round(diem, 1) <= hi:
            return d4
    return 0.0


def _diem10_to_chu(diem: float) -> str:
    """Quy đổi điểm thang 10 sang thang chữ theo bảng IUH.

    Args:
        diem: Điểm thang 10.

    Returns:
        Ký tự điểm chữ (A+, A, B+, ..., F).
    """
    for lo, hi, chu in DIEM_10_TO_CHU:
        if lo <= round(diem, 1) <= hi:
            return chu
    return "F"


def load_raw(path: Path, nganh: str) -> pd.DataFrame:
    """Load file CSV thô, tự detect separator.

    Args:
        path: Đường dẫn file CSV.
        nganh: Mã ngành — gắn vào cột 'nganh'.

    Returns:
        DataFrame thô với cột 'nganh' được thêm vào.

    Raises:
        FileNotFoundError: File không tồn tại.
    """
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    sep = _detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype={"MaMonHoc": str})
    df["nganh"] = nganh
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch và chuẩn hóa DataFrame điểm tự chọn.

    Các bước xử lý:
    1. Chuẩn hóa MaMonHoc → 6 chữ số (pad '0' bên trái)
    2. Parse TenDot → cột hk_so (int) và nam_hoc (str)
    3. Xử lý missing DiemTongKet (loại bỏ nếu null)
    4. Loại duplicate cùng (IDSinhVien, nganh, MaMonHoc) → giữ điểm cao nhất
    5. Thêm DiemTongKet4 (thang 4) và DiemChu (thang chữ)
    6. Thêm cột 'dat' (True nếu điểm >= 5.0)

    Args:
        df: DataFrame thô từ load_raw().

    Returns:
        DataFrame đã làm sạch.
    """
    df = df.copy()

    # 1. Pad MaMonHoc
    df["MaMonHoc"] = df["MaMonHoc"].astype(str).str.strip().str.zfill(6)

    # 2. Parse TenDot
    parsed = df["TenDot"].apply(_parse_ten_dot)
    df["hk_so"] = parsed.apply(lambda x: x[0])
    df["nam_hoc"] = parsed.apply(lambda x: x[1])

    # 3. Loại bỏ dòng thiếu điểm
    n_before = len(df)
    df = df.dropna(subset=["DiemTongKet"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  → Loại {n_dropped} dòng thiếu DiemTongKet")

    df["DiemTongKet"] = df["DiemTongKet"].astype(float)

    # 4. Loại duplicate: cùng SV + ngành + môn → giữ điểm cao nhất
    n_before = len(df)
    df = (
        df.sort_values("DiemTongKet", ascending=False)
        .drop_duplicates(subset=["IDSinhVien", "nganh", "MaMonHoc"], keep="first")
        .reset_index(drop=True)
    )
    n_dup = n_before - len(df)
    if n_dup > 0:
        print(f"  → Loại {n_dup} dòng duplicate (giữ điểm cao nhất)")

    # 5. Quy đổi điểm
    df["DiemTongKet4"] = df["DiemTongKet"].apply(_diem10_to_diem4)
    df["DiemChu"] = df["DiemTongKet"].apply(_diem10_to_chu)

    # 6. Cờ đạt/không đạt
    df["dat"] = df["DiemTongKet"] >= 5.0

    # Sắp xếp cột hợp lý
    cols_order = [
        "IDSinhVien", "nganh", "MaMonHoc", "TenMonHoc",
        "TenDot", "hk_so", "nam_hoc",
        "DiemTongKet", "DiemTongKet4", "DiemChu", "dat",
        "ThuocKCNTT",
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    return df


def build_student_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp profile từng sinh viên: GPA, số môn đã học, số môn đạt.

    GPA chỉ tính các môn ThuocKCNTT=1 và DiemChu không phải F
    (tương đương môn tính vào ĐTBCTL theo quy chế).

    Args:
        df: DataFrame đã clean từ hàm clean().

    Returns:
        DataFrame profile cấp sinh viên, mỗi dòng là một (IDSinhVien, nganh).
    """
    # Môn tính GPA: thuộc CNTT và có điểm đạt (DiemChu != F)
    df_gpa = df[df["ThuocKCNTT"] == 1].copy()

    gpa = (
        df_gpa.groupby(["IDSinhVien", "nganh"])
        .apply(
            lambda g: pd.Series({
                "gpa_thang4": (
                    g["DiemTongKet4"].sum() / len(g)
                    if len(g) > 0 else 0.0
                ),
                "so_mon_cntt": len(g),
                "so_mon_dat_cntt": int(g["dat"].sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    tong = (
        df.groupby(["IDSinhVien", "nganh"])
        .agg(
            so_mon_total=("MaMonHoc", "count"),
            so_mon_dat_total=("dat", "sum"),
            diem_tb_total=("DiemTongKet", "mean"),
        )
        .reset_index()
    )

    profiles = tong.merge(gpa, on=["IDSinhVien", "nganh"], how="left")
    profiles["gpa_thang4"] = profiles["gpa_thang4"].fillna(0.0).round(2)
    profiles["diem_tb_total"] = profiles["diem_tb_total"].round(2)

    return profiles


def save(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    """Lưu DataFrame ra .parquet (chính) và .csv (debug).

    Args:
        df: DataFrame cần lưu.
        name: Tên file không có extension.
        output_dir: Thư mục đích.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    csv_path = output_dir / f"{name}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {parquet_path} ({len(df)} rows)")
    print(f"  Saved: {csv_path}")


def process_nganh(
    nganh: str,
    grades_dir: Path,
    output_dir: Path,
    eda: bool = False,
) -> pd.DataFrame:
    """Pipeline đầy đủ cho một ngành: load → clean → build profile → save.

    Args:
        nganh: Mã ngành (CS/IS/DS/SE/IT).
        grades_dir: Thư mục chứa file CSV raw.
        output_dir: Thư mục output.
        eda: Nếu True, in thống kê phân phối điểm.

    Returns:
        DataFrame đã clean của ngành đó.

    Raises:
        FileNotFoundError: File CSV không tồn tại.
    """
    filename = NGANH_FILES[nganh]
    path = grades_dir / filename
    print(f"\n[{nganh}] Đang xử lý {filename}...")

    df_raw = load_raw(path, nganh)
    print(f"  Raw: {len(df_raw)} rows, {df_raw['IDSinhVien'].nunique()} sinh viên")

    df_clean = clean(df_raw)
    print(f"  Clean: {len(df_clean)} rows")

    profiles = build_student_profiles(df_clean)
    print(f"  Profiles: {len(profiles)} sinh viên")

    save(df_clean, f"{nganh}_grades_clean", output_dir)
    save(profiles, f"{nganh}_profiles", output_dir)

    if eda:
        _print_eda(df_clean, profiles, nganh)

    return df_clean


def _print_eda(df: pd.DataFrame, profiles: pd.DataFrame, nganh: str) -> None:
    """In thống kê EDA cơ bản cho một ngành.

    Args:
        df: DataFrame grades đã clean.
        profiles: DataFrame profiles sinh viên.
        nganh: Mã ngành.
    """
    print(f"\n  === EDA [{nganh}] ===")
    print(f"  Số sinh viên: {df['IDSinhVien'].nunique()}")
    print(f"  Số môn duy nhất: {df['MaMonHoc'].nunique()}")
    print(f"  Phân phối DiemChu:")
    dist = df["DiemChu"].value_counts().sort_index()
    for grade, cnt in dist.items():
        pct = cnt / len(df) * 100
        print(f"    {grade}: {cnt} ({pct:.1f}%)")
    print(f"  GPA trung bình (thang 4): {profiles['gpa_thang4'].mean():.2f}")
    print(f"  GPA min/max: {profiles['gpa_thang4'].min():.2f} / {profiles['gpa_thang4'].max():.2f}")

    print(f"  Top 5 môn phổ biến nhất:")
    top5 = df.groupby(["MaMonHoc", "TenMonHoc"]).size().sort_values(ascending=False).head(5)
    for (ma, ten), cnt in top5.items():
        print(f"    {ma} {ten}: {cnt} lượt")


def process_all(
    grades_dir: Path,
    output_dir: Path,
    eda: bool = False,
) -> pd.DataFrame:
    """Xử lý tất cả 5 ngành và gộp thành một DataFrame tổng.

    Args:
        grades_dir: Thư mục chứa các file CSV raw.
        output_dir: Thư mục output.
        eda: Nếu True, in EDA cho từng ngành và toàn bộ.

    Returns:
        DataFrame gộp của 5 ngành đã clean.
    """
    all_dfs: list[pd.DataFrame] = []
    errors: list[str] = []

    for nganh in NGANH_FILES:
        try:
            df = process_nganh(nganh, grades_dir, output_dir, eda=eda)
            all_dfs.append(df)
        except FileNotFoundError as e:
            print(f"[{nganh}] LỖI: {e}")
            errors.append(nganh)

    if not all_dfs:
        raise RuntimeError("Không xử lý được ngành nào.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    save(df_all, "ALL_grades_clean", output_dir)

    if errors:
        print(f"\nCác ngành bị lỗi: {errors}")
    else:
        print(f"\n✓ Tổng cộng {len(df_all)} records, {df_all['IDSinhVien'].nunique()} sinh viên (5 ngành)")

    return df_all


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiền xử lý dữ liệu điểm học phần tự chọn")
    p.add_argument("--nganh", choices=list(NGANH_FILES.keys()), help="Xử lý một ngành")
    p.add_argument("--all", action="store_true", help="Xử lý tất cả 5 ngành")
    p.add_argument("--eda", action="store_true", help="In thống kê EDA")
    p.add_argument(
        "--input_dir", type=Path, default=GRADES_DIR,
        help="Thư mục chứa file CSV (mặc định: data/raw/grades)",
    )
    p.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR,
        help="Thư mục output (mặc định: data/processed/student_profiles)",
    )
    return p


def main() -> None:
    """Điểm vào CLI."""
    args = _build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    grades_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir

    if args.all:
        process_all(grades_dir, output_dir, eda=args.eda)
    elif args.nganh:
        process_nganh(args.nganh, grades_dir, output_dir, eda=args.eda)
    else:
        _build_arg_parser().print_help()


if __name__ == "__main__":
    main()
