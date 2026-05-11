
"""
validate_generated_data.py — Kiểm tra chất lượng dữ liệu điểm SV sinh bởi AI
===============================================================================

VỊ TRÍ: rag-chatbot-hocphan/scripts/validate_generated_data.py

CÁCH DÙNG:

    # 1. Validate 1 file (kiểm tra cơ bản — không cần biết ngành)
    python scripts/validate_generated_data.py --file data/raw/grades/batches/CS_batch_01.csv

    # 2. Validate 1 file + kiểm tra mã môn hợp lệ theo ngành
    python scripts/validate_generated_data.py --file CS_batch_01.csv --nganh CS

    # 3. Validate cả thư mục batch
    python scripts/validate_generated_data.py --dir data/raw/grades/batches/ --nganh CS

    # 4. Strict mode: coi cảnh báo như lỗi
    python scripts/validate_generated_data.py --file CS_batch_01.csv --nganh CS --strict

INPUT: file CSV hoặc pipe-separated (|) do AI sinh ra, có header:
    IDSinhVien|MaMonHoc|TenMonHoc|TenDot|DiemTongKet|ThuocKCNTT

OUTPUT: báo cáo trên terminal — ĐẠT (✅) hoặc KHÔNG ĐẠT (❌) kèm chi tiết lỗi.

CÀI ĐẶT: chỉ cần pandas + numpy (đã có sẵn trong requirements giai đoạn 1).
    pip install pandas numpy
"""

import re
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════
# DANH SÁCH MÔN TỰ CHỌN HỢP LỆ THEO NGÀNH
# ═══════════════════════════════════════════════════════════════
#
# Format: "ma_mon": ("ten_mon", thuoc_kcntt)
#
# ĐỂ THÊM NGÀNH MỚI:
#   1. Parse HTML curriculum của ngành đó
#   2. Lấy tất cả môn có loai="tu_chon"
#   3. Thêm vào dict VALID_COURSES bên dưới theo format tương tự CS
#
# Nếu chưa có danh sách → script vẫn chạy được,
# chỉ bỏ qua bước kiểm tra mã môn hợp lệ.
# ═══════════════════════════════════════════════════════════════

VALID_COURSES: dict[str, dict[str, tuple[str, int]]] = {
    "CS": {
        # HK2 tự chọn (chọn 3TC)
        "003605": ("Phương pháp tính", 0),
        "003631": ("Vật lý đại cương", 0),
        "003697": ("Toán ứng dụng", 0),
        "003783": ("Hàm phức và phép biến đổi Laplace", 0),
        "003822": ("Logic học", 0),
        # HK3 tự chọn (chọn 5TC)
        "001479": ("Địa lí kinh tế", 0),
        "003582": ("Kỹ năng xây dựng kế hoạch", 0),
        "003877": ("Môi trường và con người", 0),
        "014233": ("Ngôn ngữ Python", 1),
        "014234": ("Tính toán số & Matlab", 1),
        "014235": ("Ngôn ngữ R", 1),
        "015395": ("Ứng dụng hóa học trong Công nghiệp", 0),
        "015396": ("Ứng dụng 5S và Kaizen trong sản xuất", 0),
        "015400": ("Công nghệ thông tin trong chuyển đổi số", 1),
        # HK4 tự chọn (chọn 3TC)
        "004022": ("Lập trình phân tích dữ liệu 1", 1),
        "004119": ("Lập trình hướng đối tượng", 1),
        "004351": ("Kỹ thuật điện tử", 1),
        # HK5 tự chọn (chọn 7TC)
        "001730": ("Automat & ngôn ngữ hình thức", 1),
        "002876": ("Lập Trình Hướng Sự Kiện với Công Nghệ Java", 1),
        "002990": ("Lập Trình Hướng Sự Kiện với Công Nghệ .NET", 1),
        "003953": ("Lập trình phân tích dữ liệu 2", 1),
        "004021": ("Lập trình GUI với Qt Framework", 1),
        "014181": ("Hệ quản trị CSDL NoSQL MongoDB", 1),
        # HK6 tự chọn (chọn 3TC)
        "001729": ("Tương tác người máy", 1),
        "003952": ("Tiếp thị điện tử", 0),
        "004024": ("Nhập môn dữ liệu lớn", 1),
        "004196": ("Quản lý dự án CNTT", 1),
        # HK7 tự chọn (chọn 6TC)
        "001758": ("Phân tích thiết kế giải thuật", 1),
        "002804": ("Lập trình phân tán với công nghệ Java", 1),
        "003406": ("Lập trình Phân Tán Công Nghệ .NET", 1),
        "003622": ("Tâm lý học đại cương", 0),
        "003633": ("Tiếng Việt thực hành", 0),
        "003634": ("Âm nhạc – Nhạc lý và Guitar căn bản", 0),
        "003664": ("Xã hội học", 0),
        "003733": ("Cơ sở văn hóa Việt Nam", 0),
        "003748": ("Hội Họa", 0),
        "014192": ("Kỹ năng sử dụng bàn phím và thiết bị văn phòng", 0),
        # HK8 tự chọn (chọn 6TC)
        "001899": ("Công nghệ phần mềm", 1),
        "004023": ("Đảm bảo chất lượng và Kiểm thử phần mềm", 1),
        "015033": ("Đồ họa máy tính", 1),
        "015036": ("Ngôn ngữ lập trình", 1),
        "015060": ("Phân tích và quản lý yêu cầu", 1),
        "015436": ("Phát triển giao diện ứng dụng", 1),
        "015458": ("Kiến trúc hướng dịch vụ và Điện toán đám mây", 1),
    },

    # ───────────────────────────────────────────────────────────
    # TODO: Thêm ngành IS, DS, SE, IT khi đã parse curriculum
    # Copy format từ CS ở trên, thay mã môn + tên + ThuocKCNTT
    #
    # "IS": {
    #     "003197": ("Kỹ năng xây dựng kế hoạch", 0),
    #     ...
    # },
    # "DS": { ... },
    # "SE": { ... },
    # "IT": { ... },
    # ───────────────────────────────────────────────────────────
}


# ═══════════════════════════════════════════════════════════════
# PHÂN PHỐI ĐIỂM KỲ VỌNG
# ═══════════════════════════════════════════════════════════════

EXPECTED_DIST = {
    #  label        lo    hi   expected%
    "F":           (0.0,  3.9, 0.03),   # 3% không đạt
    "D":           (4.0,  4.9, 0.05),   # 5% đạt yếu
    "D+":          (5.0,  5.4, 0.07),   # 7% đạt tối thiểu
    "C":           (5.5,  5.9, 0.10),   # 10% trung bình
    "C+":          (6.0,  6.9, 0.20),   # 20% trung bình khá
    "B":           (7.0,  7.9, 0.30),   # 30% khá (đông nhất)
    "B+":          (8.0,  8.4, 0.12),   # 12% giỏi
    "A":           (8.5,  8.9, 0.08),   # 8% giỏi xuất sắc
    "A+":          (9.0, 10.0, 0.05),   # 5% xuất sắc
}

DIST_TOLERANCE = 0.10  # ±10 percentage points


# ═══════════════════════════════════════════════════════════════
# ĐỌC FILE
# ═══════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """Đọc file CSV/pipe/tab. Tự detect separator."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {filepath}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    first_line = raw.strip().split("\n")[0]

    if "|" in first_line:
        sep = "|"
    elif "\t" in first_line:
        sep = "\t"
    else:
        sep = ","

    df = pd.read_csv(filepath, sep=sep, dtype=str, encoding="utf-8")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# ═══════════════════════════════════════════════════════════════
# 8 BƯỚC KIỂM TRA
# ═══════════════════════════════════════════════════════════════

def check_columns(df):
    errors, warnings = [], []
    expected = {"IDSinhVien", "MaMonHoc", "TenMonHoc", "TenDot", "DiemTongKet", "ThuocKCNTT"}
    missing = expected - set(df.columns)
    extra = set(df.columns) - expected
    if missing:
        errors.append(f"THIẾU CỘT: {missing}")
    if extra:
        warnings.append(f"Cột thừa (bỏ qua): {extra}")
    return errors, warnings


def check_id(df):
    errors, warnings = [], []
    ids = df["IDSinhVien"].astype(str)

    bad_format = ids[~ids.str.match(r"^\d{7}$")]
    if len(bad_format) > 0:
        errors.append(f"ID sai format ({len(bad_format)} dòng): {bad_format.head(5).tolist()}")

    good = ids[ids.str.match(r"^\d{7}$")]
    if len(good) > 0:
        years = good.str[:2].astype(int)
        bad_year = good[(years < 16) | (years > 20)]
        if len(bad_year) > 0:
            warnings.append(f"ID năm lạ ({len(bad_year)} dòng ngoài 2016–2019)")

    return errors, warnings


def check_ma_mon(df, nganh):
    errors, warnings = [], []
    codes = df["MaMonHoc"].astype(str).str.strip()

    bad = codes[~codes.str.match(r"^\d{6}$")]
    if len(bad) > 0:
        errors.append(f"Mã môn sai format ({len(bad)} dòng): {bad.unique()[:5].tolist()}")

    if nganh and nganh in VALID_COURSES:
        valid_set = set(VALID_COURSES[nganh].keys())
        actual_set = set(codes.str.zfill(6))
        invalid = actual_set - valid_set
        if invalid:
            errors.append(f"Mã môn không thuộc ngành {nganh} ({len(invalid)} mã): {sorted(invalid)[:10]}")
    elif nganh:
        warnings.append(f"Chưa có danh sách ngành {nganh} → bỏ qua kiểm tra mã môn.")

    return errors, warnings


def check_ten_dot(df):
    """
    TenDot phải đúng format HK{1|2|3} (YYYY-YYYY).
    CHÚ Ý: File điểm chỉ dùng HK1 (lẻ), HK2 (chẵn), HK3 (hè).
    HK4–HK9 là hệ thống chương trình khung — KHÔNG dùng ở đây.
    """
    errors, warnings = [], []
    pattern = re.compile(r"^HK([123])\s*\((\d{4})-(\d{4})\)$")
    dots = df["TenDot"].astype(str).str.strip()

    bad_format, bad_year = [], []
    for val in dots.unique():
        m = pattern.match(val)
        if not m:
            bad_format.append(val)
        else:
            y1, y2 = int(m.group(2)), int(m.group(3))
            if y2 != y1 + 1:
                bad_year.append(val)

    if bad_format:
        errors.append(f"TenDot sai format ({len(bad_format)} giá trị): {bad_format[:5]}. Phải là HK{{1|2|3}} (YYYY-YYYY)")

    if bad_year:
        errors.append(f"TenDot sai năm (year2 ≠ year1+1): {bad_year[:5]}")

    wrong_hk = dots[dots.str.match(r"^HK[456789]")]
    if len(wrong_hk) > 0:
        errors.append(
            f"TenDot dùng HK4–HK9 ({len(wrong_hk)} dòng) — SAI. "
            f"File điểm chỉ dùng HK1/HK2/HK3. HK4–9 là hệ thống chương trình khung."
        )

    return errors, warnings


def check_diem(df):
    errors, warnings, stats = [], [], {}
    diem = pd.to_numeric(df["DiemTongKet"], errors="coerce")

    # n_nan = diem.isna().sum()
    # if n_nan > 0:
    #     errors.append(f"Điểm không hợp lệ ({n_nan} giá trị không phải số)")

    out = diem[(diem < 0) | (diem > 10)].dropna()
    if len(out) > 0:
        errors.append(f"Điểm ngoài [0–10] ({len(out)} dòng): {out.head(5).tolist()}")

    valid = diem.dropna()
    if len(valid) > 0:
        stats["diem_mean"] = round(valid.mean(), 2)
        stats["diem_std"] = round(valid.std(), 2)
        stats["diem_min"] = round(valid.min(), 1)
        stats["diem_max"] = round(valid.max(), 1)

    return errors, warnings, stats


def check_thuoc_kcntt(df, nganh):
    errors, warnings = [], []
    vals = pd.to_numeric(df["ThuocKCNTT"], errors="coerce")
    bad = vals[~vals.isin([0, 1])]
    if len(bad) > 0:
        errors.append(f"ThuocKCNTT sai ({len(bad)} giá trị không phải 0/1)")

    if nganh and nganh in VALID_COURSES:
        mismatches = []
        for _, row in df.head(200).iterrows():  # check 200 dòng đầu để nhanh
            ma = str(row["MaMonHoc"]).strip().zfill(6)
            if ma in VALID_COURSES[nganh]:
                expected = VALID_COURSES[nganh][ma][1]
                actual = int(float(row["ThuocKCNTT"])) if pd.notna(row["ThuocKCNTT"]) else 0
                if actual != expected:
                    mismatches.append(f"{ma} (cần {expected}, có {actual})")
        if mismatches:
            warnings.append(f"ThuocKCNTT sai ({len(mismatches)} môn): {mismatches[:3]}")

    return errors, warnings


def check_distribution(df):
    errors, warnings = [], []
    diem = pd.to_numeric(df["DiemTongKet"], errors="coerce").dropna()

    if len(diem) < 50:
        warnings.append(f"Chỉ {len(diem)} điểm — quá ít để kiểm phân phối (cần ≥50)")
        return errors, warnings, ""

    total = len(diem)
    lines = []
    for label, (lo, hi, expected) in EXPECTED_DIST.items():
        count = ((diem >= lo) & (diem <= hi)).sum()
        actual = count / total
        diff = abs(actual - expected)
        ok = "✓" if diff <= DIST_TOLERANCE else "✗"
        lines.append(
            f"  {lo:4.1f}–{hi:4.1f} ({label:>8}): "
            f"kỳ vọng {expected*100:5.1f}% | thực tế {actual*100:5.1f}% ({count:>4}) | {ok}"
        )
        if diff > DIST_TOLERANCE:
            warnings.append(f"Phân phối lệch ({label}): kỳ vọng {expected*100:.0f}%, thực tế {actual*100:.1f}%")

    return errors, warnings, "\n".join(lines)


def check_student_patterns(df):
    errors, warnings, stats = [], [], {}
    groups = df.groupby("IDSinhVien")
    mon_per_sv = groups.size()

    stats["so_sv"] = len(mon_per_sv)
    stats["so_mon_unique"] = df["MaMonHoc"].nunique()
    stats["mon_per_sv_mean"] = round(mon_per_sv.mean(), 1)
    stats["mon_per_sv_min"] = int(mon_per_sv.min())
    stats["mon_per_sv_max"] = int(mon_per_sv.max())

    if len(mon_per_sv) > 10 and mon_per_sv.std() < 1.0:
        warnings.append(
            f"Số môn quá đồng đều: std={mon_per_sv.std():.2f}. "
            f"Tất cả SV học {mon_per_sv.min()}–{mon_per_sv.max()} môn. Thực tế nên 4–12."
        )

    too_few = (mon_per_sv < 3).sum()
    too_many = (mon_per_sv > 15).sum()
    if too_few:
        warnings.append(f"{too_few} SV học < 3 môn (quá ít)")
    if too_many:
        warnings.append(f"{too_many} SV học > 15 môn (quá nhiều)")

    diem_num = pd.to_numeric(df["DiemTongKet"], errors="coerce")
    df_tmp = df.assign(_diem=diem_num)
    sv_std = df_tmp.groupby("IDSinhVien")["_diem"].std().dropna()
    sv_std_ok = sv_std[mon_per_sv >= 3]

    if len(sv_std_ok) > 10:
        avg = sv_std_ok.mean()
        stats["sv_diem_std_tb"] = round(avg, 2)
        if avg < 0.3:
            warnings.append(f"Điểm quá đều trong SV (std TB={avg:.2f}). Thực tế ≈ 0.5–1.5.")
        elif avg > 2.5:
            warnings.append(f"Điểm quá lung tung (std TB={avg:.2f}). Thực tế ≈ 0.5–1.5.")

    dup = df.duplicated(subset=["IDSinhVien", "MaMonHoc"], keep=False)
    if dup.sum() > 0:
        examples = df[dup][["IDSinhVien", "MaMonHoc"]].drop_duplicates().head(3)
        errors.append(f"Trùng lặp: {dup.sum()} dòng cùng SV+môn. VD: {examples.values.tolist()}")

    return errors, warnings, stats


# ═══════════════════════════════════════════════════════════════
# CHẠY + IN BÁO CÁO
# ═══════════════════════════════════════════════════════════════

def validate_file(filepath: str, nganh: str | None = None, strict: bool = False) -> bool:
    print(f"\n{'═'*60}")
    print(f"📄 FILE: {filepath}")
    if nganh:
        print(f"📌 NGÀNH: {nganh}")
    print(f"{'═'*60}")

    df = load_data(filepath)
    print(f"   Đọc được: {len(df):,} dòng, {len(df.columns)} cột")

    all_errors, all_warnings = [], []

    e, w = check_columns(df)
    all_errors += e; all_warnings += w
    if e:
        return _print_result(all_errors, all_warnings, {}, "", strict)

    e, w = check_id(df)
    all_errors += e; all_warnings += w

    e, w = check_ma_mon(df, nganh)
    all_errors += e; all_warnings += w

    e, w = check_ten_dot(df)
    all_errors += e; all_warnings += w

    e, w, diem_stats = check_diem(df)
    all_errors += e; all_warnings += w

    e, w = check_thuoc_kcntt(df, nganh)
    all_errors += e; all_warnings += w

    e, w, dist_text = check_distribution(df)
    all_errors += e; all_warnings += w

    e, w, sv_stats = check_student_patterns(df)
    all_errors += e; all_warnings += w

    stats = {**diem_stats, **sv_stats}
    return _print_result(all_errors, all_warnings, stats, dist_text, strict)


def _print_result(errors, warnings, stats, dist_text, strict) -> bool:
    if stats:
        print(f"\n📊 THỐNG KÊ:")
        for k, v in stats.items():
            print(f"   {k}: {v}")

    if dist_text:
        print(f"\n📈 PHÂN PHỐI ĐIỂM:")
        print(dist_text)

    if errors:
        print(f"\n❌ LỖI ({len(errors)}) — PHẢI SỬA:")
        for i, e in enumerate(errors, 1):
            print(f"   {i}. {e}")
    else:
        print(f"\n✅ Không có lỗi.")

    if warnings:
        print(f"\n⚠️  CẢNH BÁO ({len(warnings)}):")
        for i, w in enumerate(warnings, 1):
            print(f"   {i}. {w}")

    print(f"\n{'─'*60}")
    passed = len(errors) == 0 and (not strict or len(warnings) == 0)

    if passed and not warnings:
        print("🎉 ĐẠT — sẵn sàng gộp vào dataset.")
    elif passed:
        print("✅ ĐẠT (có cảnh báo nhẹ).")
    else:
        print("❌ KHÔNG ĐẠT — sửa lỗi rồi chạy lại.")

    return passed


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Kiểm tra chất lượng dữ liệu điểm SV sinh bởi AI",
        epilog="""
VÍ DỤ:
  python scripts/validate_generated_data.py --file CS_batch_01.csv
  python scripts/validate_generated_data.py --file CS_batch_01.csv --nganh CS
  python scripts/validate_generated_data.py --dir batches/ --nganh CS --strict
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--file", help="1 file CSV cần kiểm tra")
    ap.add_argument("--dir", help="Thư mục chứa nhiều file batch")
    ap.add_argument("--nganh", help="Mã ngành (CS/IS/DS/SE/IT)")
    ap.add_argument("--strict", action="store_true", help="Coi cảnh báo như lỗi")
    args = ap.parse_args()

    nganh = args.nganh.upper() if args.nganh else None

    files = []
    if args.file:
        files = [args.file]
    elif args.dir:
        p = Path(args.dir)
        files = sorted(str(f) for f in p.glob("*.csv"))
        files += sorted(str(f) for f in p.glob("*.txt"))
        if not files:
            print(f"❌ Không tìm thấy file trong {args.dir}")
            sys.exit(1)
    else:
        ap.print_help()
        sys.exit(0)

    results = []
    for fp in files:
        try:
            ok = validate_file(fp, nganh=nganh, strict=args.strict)
            results.append((fp, ok))
        except Exception as exc:
            print(f"\n❌ LỖI ĐỌC FILE {fp}: {exc}")
            results.append((fp, False))

    if len(results) > 1:
        n_pass = sum(1 for _, p in results if p)
        print(f"\n{'═'*60}")
        print(f"TỔNG KẾT: {n_pass}/{len(results)} file đạt")
        for fp, ok in results:
            print(f"  {'✅' if ok else '❌'} {Path(fp).name}")

    sys.exit(0 if all(p for _, p in results) else 1)