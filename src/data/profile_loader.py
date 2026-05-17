"""Parse bảng điểm xlsx của 1 sinh viên → profile object để feed chatbot.

Schema xlsx user upload (theo `project_instructions.md` mục 2.2):
    IDSinhVien | MaMonHoc | TenMonHoc | TenDot | DiemTongKet | ThuocKCNTT

Nếu xlsx có nhiều sinh viên (vd full `IS_TuChon.xlsx`) → mặc định pick SV đầu;
gọi `load_student_profiles_from_xlsx` để lấy list tất cả ID + dropdown.

Ví dụ dùng:
    from src.data.profile_loader import load_profile_from_xlsx
    profile = load_profile_from_xlsx("data/raw/grades/IS_TuChon.xlsx",
                                     nganh="IS", id_sinhvien=19087654)
    print(profile.summary_markdown())
"""

from __future__ import annotations

import io
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

# Cluster mapping cho phân loại môn — dùng compute điểm trung bình theo nhóm.
# Mỗi cluster: list từ khoá tiếng Việt thường gặp trong tên môn. Match
# case-insensitive sau khi normalize (bỏ dấu). Thứ tự ưu tiên theo list —
# cluster đầu tiên match sẽ được gán.
COURSE_CLUSTERS: list[tuple[str, list[str]]] = [
    (
        "Lập trình nền tảng",
        [
            "nhap mon lap trinh",
            "ky thuat lap trinh",
            "cau truc du lieu",
            "huong doi tuong",
            "nhap mon tin hoc",
        ],
    ),
    (
        "Toán & Lý thuyết",
        [
            "toan cao cap",
            "toan ung dung",
            "vat ly",
            "phuong phap tinh",
            "logic",
            "cau truc roi rac",
            "xac suat",
            "thong ke",
            "automat",
            "do thi",
            "ham phuc",
            "giai tich",
        ],
    ),
    (
        "AI/ML",
        [
            "tri tue nhan tao",
            "may hoc",
            "hoc sau",
            "khai thac du lieu",
            "nhan dang",
            "machine learning",
        ],
    ),
    (
        "Đồ họa & CV",
        ["xu ly anh", "do hoa", "computer vision"],
    ),
    (
        "DB & Dữ liệu",
        [
            "co so du lieu",
            "csdl",
            "nosql",
            "phan tich du lieu",
            "lap trinh phan tich",
            "du lieu lon",
            "big data",
            "data",
        ],
    ),
    (
        "Web/Java/.NET",
        [
            "java",
            "web",
            "huong su kien",
            "phan tan",
            "mang",
            "gui",
            ".net",
            "thuong mai dien tu",
            "tiep thi dien tu",
        ],
    ),
    (
        "Đại cương / Kỹ năng",
        [
            "triet hoc",
            "kinh te",
            "mac",
            "lenin",
            "anh van",
            "tieng anh",
            "ky nang",
            "the chat",
            "quoc phong",
            "lich su",
            "phap luat",
            "moi truong",
        ],
    ),
]


def _normalize_vn(text: str) -> str:
    """Bỏ dấu tiếng Việt + lowercase. Dùng cho fuzzy keyword match."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


def classify_course(ten_mon: str) -> str:
    """Gán cluster cho 1 môn từ tên (substring match, không dấu, lowercase).

    Returns:
        Tên cluster (string), hoặc "Khác" nếu không match cluster nào.
    """
    n = _normalize_vn(ten_mon)
    for cluster, kws in COURSE_CLUSTERS:
        for kw in kws:
            if kw in n:
                return cluster
    return "Khác"


# Bảng quy đổi điểm 10 → 4 (Điều 19 quy chế đào tạo IUH), dùng chung với preprocessor.
DIEM_10_TO_4: list[tuple[float, float, float]] = [
    (9.0, 10.0, 4.0),  # A+
    (8.5, 8.9, 3.8),
    (8.0, 8.4, 3.5),
    (7.0, 7.9, 3.0),
    (6.0, 6.9, 2.5),
    (5.5, 5.9, 2.0),
    (5.0, 5.4, 1.5),
    (4.0, 4.9, 1.0),
    (0.0, 3.9, 0.0),
]


def _diem_10_to_4(diem10: float) -> float:
    """Quy đổi điểm thang 10 → thang 4 theo bảng IUH."""
    for lo, hi, ten in DIEM_10_TO_4:
        if lo <= diem10 <= hi:
            return ten
    return 0.0


def _parse_tendot(tendot: str) -> tuple[int | None, int | None]:
    """Parse 'HK1 (2019-2020)' → (hk_so=1, start_year=2019). Trả (None, None) nếu fail."""
    try:
        # "HK1 (2019-2020)"
        hk_part, year_part = tendot.split(" ", 1)
        hk_so = int(hk_part.replace("HK", "").strip())
        year_str = year_part.strip("() ").split("-")[0]
        return hk_so, int(year_str)
    except Exception:  # noqa: BLE001
        return None, None


def _infer_nhap_hoc(id_sinhvien: int | str) -> int | None:
    """Suy năm nhập học từ 2 chữ số đầu IDSinhVien.

    Quy ước IUH: ID 8 chữ số, 2 đầu = năm (19 → 2019, 22 → 2022).
    """
    s = str(id_sinhvien)
    if len(s) < 2 or not s[:2].isdigit():
        return None
    yy = int(s[:2])
    return 2000 + yy if yy < 50 else 1900 + yy  # tương thích nếu sau 2050 …


def _compute_curriculum_hk(start_year: int, hk_so: int, nhap_hoc: int) -> int | None:
    """`curriculum_hk = (start_year − nhập_học) × 2 + hk_so`, HK3 (hè) → None."""
    if hk_so == 3:
        return None  # hè
    return (start_year - nhap_hoc) * 2 + hk_so


@dataclass
class StudentProfile:
    """Profile sinh viên đã parse — đủ dùng cho prompt LLM + retrieval."""

    nganh: str
    id_sinhvien: str
    nhap_hoc: int | None
    n_mon: int  # tổng môn có trong bảng điểm (kể cả failed)
    n_pass: int  # đạt (DiemTongKet >= 5)
    n_fail: int  # không đạt
    gpa_10: float  # GPA thang 10 (chỉ tính môn đạt)
    gpa_4: float  # quy đổi thang 4
    completed: list[str] = field(default_factory=list)
    """Mã môn đã đạt (6 chữ số) — feed cho ConstraintChecker."""

    failed: list[str] = field(default_factory=list)
    """Mã môn không đạt — có thể đề xuất học lại."""

    top_diem_cao: list[dict] = field(default_factory=list)
    """Top 5 môn điểm cao nhất (đã pass): {ma, ten, diem}. Hint định hướng."""

    top_diem_thap: list[dict] = field(default_factory=list)
    """Top 5 môn điểm thấp nhất (đã pass) — môn user có thể muốn cải thiện."""

    current_hk: int | None = None
    """HK cao nhất từng học + 1 (HK kế tiếp). None nếu không suy được."""

    cluster_avg: dict[str, dict] = field(default_factory=dict)
    """Điểm TB theo nhóm môn (cluster). Vd:
    {"Toán & Lý thuyết": {"avg": 8.2, "n": 3, "courses": [...]},
     "Lập trình nền tảng": {"avg": 5.5, "n": 2, ...}}.
    Dùng để Qwen reasoning 'em mạnh nhóm X, yếu nhóm Y'."""

    raw_records: list[dict] = field(default_factory=list)
    """Toàn bộ records để debug/inspect."""

    def summary_markdown(self) -> str:
        """Format profile thành markdown để inject vào prompt LLM."""
        top_lines = "\n".join(
            f"  - {m['ten_mon']} (mã {m['ma_mon']}, điểm {m['diem']:.1f})"
            for m in self.top_diem_cao
        ) or "  (chưa có)"

        cluster_block = ""
        if self.cluster_avg:
            # Sort cluster theo avg desc, bỏ "Khác" + cluster có < 1 môn.
            ranked = sorted(
                (
                    (name, info)
                    for name, info in self.cluster_avg.items()
                    if name != "Khác" and info["n"] >= 1
                ),
                key=lambda x: -x[1]["avg"],
            )
            if ranked:
                cluster_lines = "\n".join(
                    f"  - **{name}** (n={info['n']} môn): TB **{info['avg']:.2f}/10**"
                    for name, info in ranked
                )
                cluster_block = (
                    f"\n- Điểm trung bình theo nhóm môn "
                    f"(để Bot suy luận em mạnh/yếu mảng nào):\n{cluster_lines}"
                )

        return (
            f"- Ngành: **{self.nganh}**\n"
            f"- ID sinh viên: {self.id_sinhvien}"
            f" (nhập học {self.nhap_hoc or '?'})\n"
            f"- Số môn đã học: {self.n_mon} (đạt {self.n_pass}, "
            f"không đạt {self.n_fail})\n"
            f"- GPA: **{self.gpa_4:.2f}/4** (≈ {self.gpa_10:.2f}/10)\n"
            f"- Học kỳ kế tiếp: HK{self.current_hk or '?'}\n"
            f"- Top 5 môn điểm cao nhất:\n{top_lines}"
            f"{cluster_block}"
        )


def _read_csv_smart(source, dtype: dict) -> pd.DataFrame:
    """Đọc CSV với auto-detect sep `|` hoặc `,`.

    Pandas's sniffer (sep=None) hay nhầm khi cột có Unicode tiếng Việt nên ta
    thử thủ công: đọc header line, đếm `|` vs `,`, chọn separator đa số.
    """
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        head = source.read(1024)
        if isinstance(head, bytes):
            head_str = head.decode("utf-8", errors="ignore")
        else:
            head_str = head
        source.seek(0)
    else:
        with open(source, encoding="utf-8") as f:
            head_str = f.read(1024)
    sep = "|" if head_str.count("|") > head_str.count(",") else ","
    return pd.read_csv(source, dtype=dtype, sep=sep)


def _read_xlsx_or_csv(source: str | bytes | Path | io.BytesIO) -> pd.DataFrame:
    """Đọc xlsx hoặc csv. CSV của khoa có 2 sep (CS/DS/SE/IT `|`, IS `,`)."""
    dtype = {"MaMonHoc": str, "IDSinhVien": str}
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix.lower() == ".csv":
            return _read_csv_smart(p, dtype)
        return pd.read_excel(p, dtype=dtype)

    if hasattr(source, "name") and isinstance(source.name, str):
        if source.name.lower().endswith(".csv"):
            return _read_csv_smart(source, dtype)
    try:
        return pd.read_excel(source, dtype=dtype)
    except Exception:  # noqa: BLE001
        if hasattr(source, "seek"):
            source.seek(0)
        return _read_csv_smart(source, dtype)


def list_student_ids(source) -> list[str]:
    """Liệt kê các IDSinhVien duy nhất trong file. Dùng cho dropdown UI."""
    df = _read_xlsx_or_csv(source)
    return sorted(df["IDSinhVien"].astype(str).unique().tolist())


def load_profile_from_xlsx(
    source,
    nganh: str,
    id_sinhvien: str | int | None = None,
    so_tc_map: dict[str, int] | None = None,
) -> StudentProfile:
    """Parse xlsx → StudentProfile của 1 sinh viên.

    Args:
        source: đường dẫn file hoặc bytes/file-like (từ Streamlit uploader).
        nganh: mã ngành (CS/IS/DS/SE/IT). Vì xlsx không có cột ngành.
        id_sinhvien: lọc theo 1 SV. None → lấy SV đầu tiên xuất hiện.
        so_tc_map: dict {ma_mon: so_tc} để tính tổng TC. Optional.

    Returns:
        StudentProfile đã compute đầy đủ stats.

    Raises:
        ValueError: thiếu cột, không tìm thấy SV, file rỗng.
    """
    df = _read_xlsx_or_csv(source)
    required = {"IDSinhVien", "MaMonHoc", "TenMonHoc", "TenDot", "DiemTongKet"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File thiếu cột: {missing}")

    df["IDSinhVien"] = df["IDSinhVien"].astype(str)
    df["MaMonHoc"] = df["MaMonHoc"].astype(str).str.zfill(6)

    if id_sinhvien is None:
        id_sinhvien = df["IDSinhVien"].iloc[0]
    id_sinhvien = str(id_sinhvien)

    sub = df[df["IDSinhVien"] == id_sinhvien].copy()
    if sub.empty:
        raise ValueError(f"Không tìm thấy IDSinhVien={id_sinhvien} trong file.")

    # Xử lý null DiemTongKet (môn chưa có điểm → bỏ).
    sub = sub.dropna(subset=["DiemTongKet"]).copy()
    sub["DiemTongKet"] = sub["DiemTongKet"].astype(float)
    sub["diem_4"] = sub["DiemTongKet"].apply(_diem_10_to_4)

    # Pass/fail theo Điều 19 — đạt khi >= 5.0 thang 10.
    sub["pass"] = sub["DiemTongKet"] >= 5.0

    # Dedup môn học lại: cùng MaMonHoc → giữ lần điểm CAO nhất.
    sub = sub.sort_values("DiemTongKet", ascending=False).drop_duplicates(
        subset=["MaMonHoc"], keep="first"
    )

    nhap_hoc = _infer_nhap_hoc(id_sinhvien)

    # Tính curriculum_hk cho mỗi record.
    def _hk_row(row):
        hk_so, year = _parse_tendot(row["TenDot"])
        if hk_so is None or year is None or nhap_hoc is None:
            return None
        return _compute_curriculum_hk(year, hk_so, nhap_hoc)

    sub["curriculum_hk"] = sub.apply(_hk_row, axis=1)

    # HK kế tiếp = max curriculum_hk + 1 (chỉ tính hk_so 1/2 — bỏ hè).
    valid_hk = sub["curriculum_hk"].dropna()
    current_hk = int(valid_hk.max()) + 1 if not valid_hk.empty else None
    if current_hk is not None and current_hk > 9:
        current_hk = 9  # cap ở HK9 (cuối khoá).

    passed = sub[sub["pass"]]
    failed = sub[~sub["pass"]]

    # GPA chỉ tính môn đạt + có thuộc KCNTT (theo Điều 19 IUH).
    if so_tc_map is not None and "ThuocKCNTT" in sub.columns:
        gpa_records = passed[passed["ThuocKCNTT"] == 1].copy()
        gpa_records["so_tc"] = gpa_records["MaMonHoc"].map(so_tc_map).fillna(0).astype(int)
        total_tc = gpa_records["so_tc"].sum()
        if total_tc > 0:
            gpa_10 = (gpa_records["DiemTongKet"] * gpa_records["so_tc"]).sum() / total_tc
            gpa_4 = (gpa_records["diem_4"] * gpa_records["so_tc"]).sum() / total_tc
        else:
            gpa_10 = passed["DiemTongKet"].mean() if not passed.empty else 0.0
            gpa_4 = passed["diem_4"].mean() if not passed.empty else 0.0
    else:
        # Fallback: trung bình không trọng số.
        gpa_10 = passed["DiemTongKet"].mean() if not passed.empty else 0.0
        gpa_4 = passed["diem_4"].mean() if not passed.empty else 0.0

    # Top điểm cao / thấp (chỉ trên môn pass).
    passed_sorted = passed.sort_values("DiemTongKet", ascending=False)
    top_cao = [
        {
            "ma_mon": r["MaMonHoc"],
            "ten_mon": r["TenMonHoc"],
            "diem": float(r["DiemTongKet"]),
        }
        for _, r in passed_sorted.head(5).iterrows()
    ]
    top_thap = [
        {
            "ma_mon": r["MaMonHoc"],
            "ten_mon": r["TenMonHoc"],
            "diem": float(r["DiemTongKet"]),
        }
        for _, r in passed_sorted.tail(5)[::-1].iterrows()
    ]

    raw_records = sub[
        ["MaMonHoc", "TenMonHoc", "TenDot", "DiemTongKet", "diem_4", "pass", "curriculum_hk"]
    ].to_dict(orient="records")

    return StudentProfile(
        nganh=nganh,
        id_sinhvien=id_sinhvien,
        nhap_hoc=nhap_hoc,
        n_mon=len(sub),
        n_pass=len(passed),
        n_fail=len(failed),
        gpa_10=float(gpa_10),
        gpa_4=float(gpa_4),
        completed=passed["MaMonHoc"].tolist(),
        failed=failed["MaMonHoc"].tolist(),
        top_diem_cao=top_cao,
        top_diem_thap=top_thap,
        current_hk=current_hk,
        raw_records=raw_records,
    )


def build_profile_from_grades(
    nganh: str,
    grades: dict[str, float],
    curriculum_rows: list[dict],
    hk_target: int | None = None,
    id_sinhvien: str = "user",
) -> StudentProfile:
    """Tạo StudentProfile từ dict điểm user nhập (Giai đoạn 7 — không cần xlsx).

    Args:
        nganh: CS / IS / DS / SE / IT.
        grades: {ma_mon (6 chữ số): diem (thang 10)}. Bỏ trống = chưa học.
        curriculum_rows: output của `build_grade_table(nganh, hk_target)` — chứa
            metadata (ten_mon, so_tc, hk_chuan, khong_tinh_gpa) cho từng môn.
            Tránh phải đọc lại curriculum trong hàm.
        hk_target: HK chuẩn bị đăng ký (set vào `current_hk`). None → suy
            từ max(hk_chuan) + 1.
        id_sinhvien: mặc định "user" (không có ID thực vì không upload).

    Returns:
        StudentProfile sẵn sàng feed vào pipeline.chat().
    """
    # Index curriculum theo ma_mon để lookup nhanh.
    meta_by_ma = {r["ma_mon"]: r for r in curriculum_rows}

    passed_records: list[dict] = []
    failed_ma: list[str] = []
    for ma, diem in grades.items():
        ma6 = str(ma).zfill(6)
        if diem is None:
            continue
        try:
            d = float(diem)
        except (TypeError, ValueError):
            continue
        if d < 0 or d > 10:
            continue
        meta = meta_by_ma.get(ma6, {})
        rec = {
            "ma_mon": ma6,
            "ten_mon": meta.get("ten_mon", "?"),
            "so_tc": int(meta.get("so_tc", 0) or 0),
            "hk_chuan": meta.get("hk_chuan"),
            "khong_tinh_gpa": meta.get("khong_tinh_gpa", False),
            "diem": d,
            "diem_4": _diem_10_to_4(d),
        }
        if d >= 5.0:
            passed_records.append(rec)
        else:
            failed_ma.append(ma6)

    # GPA weighted theo TC, chỉ tính môn có tính GPA (khong_tinh_gpa=False).
    gpa_pool = [r for r in passed_records if not r["khong_tinh_gpa"]]
    total_tc = sum(r["so_tc"] for r in gpa_pool)
    if total_tc > 0:
        gpa_10 = sum(r["diem"] * r["so_tc"] for r in gpa_pool) / total_tc
        gpa_4 = sum(r["diem_4"] * r["so_tc"] for r in gpa_pool) / total_tc
    elif gpa_pool:
        gpa_10 = sum(r["diem"] for r in gpa_pool) / len(gpa_pool)
        gpa_4 = sum(r["diem_4"] for r in gpa_pool) / len(gpa_pool)
    else:
        gpa_10 = 0.0
        gpa_4 = 0.0

    sorted_pass = sorted(passed_records, key=lambda r: -r["diem"])
    top_cao = [
        {"ma_mon": r["ma_mon"], "ten_mon": r["ten_mon"], "diem": r["diem"]}
        for r in sorted_pass[:5]
    ]
    top_thap = [
        {"ma_mon": r["ma_mon"], "ten_mon": r["ten_mon"], "diem": r["diem"]}
        for r in sorted_pass[-5:][::-1]
    ]

    # Compute cluster avg cho cả pass + fail (vì điểm thấp môn fail vẫn quan
    # trọng cho reasoning — vd Nhập môn LP 4.0 là red flag).
    by_cluster: dict[str, list[dict]] = {}
    for ma, diem in grades.items():
        ma6 = str(ma).zfill(6)
        if diem is None:
            continue
        try:
            d = float(diem)
        except (TypeError, ValueError):
            continue
        if d < 0 or d > 10:
            continue
        meta = meta_by_ma.get(ma6, {})
        cluster = classify_course(meta.get("ten_mon", ""))
        by_cluster.setdefault(cluster, []).append(
            {
                "ma_mon": ma6,
                "ten_mon": meta.get("ten_mon", "?"),
                "diem": d,
                "loai": meta.get("loai", "?"),
            }
        )
    cluster_avg: dict[str, dict] = {}
    for cluster, courses in by_cluster.items():
        if not courses:
            continue
        avg = sum(c["diem"] for c in courses) / len(courses)
        cluster_avg[cluster] = {
            "avg": float(avg),
            "n": len(courses),
            "courses": courses,
        }

    if hk_target is None:
        hks = [r["hk_chuan"] for r in passed_records if r["hk_chuan"] is not None]
        hk_target = (max(hks) + 1) if hks else None

    return StudentProfile(
        nganh=nganh,
        id_sinhvien=id_sinhvien,
        nhap_hoc=None,
        n_mon=len(grades),
        n_pass=len(passed_records),
        n_fail=len(failed_ma),
        gpa_10=float(gpa_10),
        gpa_4=float(gpa_4),
        completed=[r["ma_mon"] for r in passed_records],
        failed=failed_ma,
        top_diem_cao=top_cao,
        top_diem_thap=top_thap,
        current_hk=hk_target,
        cluster_avg=cluster_avg,
        raw_records=passed_records,
    )


def load_required_courses(
    nganh: str,
    hk_max: int,
    curriculum_dir: Path | str = "data/processed/curriculum_graph",
) -> list[str]:
    """Trả list mã môn **bắt buộc** từ HK1 đến HK(hk_max-1).

    File `<NGANH>_TuChon.xlsx` chỉ có môn tự chọn → cần infer môn bắt buộc đã
    pass từ curriculum (giả định SV ở HK X đã hoàn thành tất cả môn bắt buộc
    của HK 1..X-1, vì hệ KCNTT không cho đăng ký HK X nếu thiếu).

    Args:
        nganh: CS/IS/DS/SE/IT.
        hk_max: HK chuẩn bị đăng ký (exclusive). VD `hk_max=3` → trả môn bắt
            buộc HK1+HK2.
        curriculum_dir: thư mục chứa `<NGANH>_curriculum.json`.

    Returns:
        List mã môn 6 chữ số. Rỗng nếu không tìm thấy file hoặc hk_max ≤ 1.
    """
    if hk_max <= 1:
        return []
    import json as _json

    path = Path(curriculum_dir) / f"{nganh}_curriculum.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = _json.load(f)
    out: list[str] = []
    for hk in data.get("hoc_ky", []):
        if hk.get("hk_so") is None or hk["hk_so"] >= hk_max:
            continue
        for hp in hk.get("hoc_phan", []):
            if hp.get("loai") == "bat_buoc":
                ma = str(hp.get("ma_mon", "")).zfill(6)
                if ma and ma != "000000":
                    out.append(ma)
    return out


def load_so_tc_map(corpus_path: Path | str) -> dict[str, int]:
    """Map ma_mon (6 chữ số) → so_tc từ corpus.jsonl. Dùng để tính GPA weighted."""
    import json

    out: dict[str, int] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ma = str(d.get("ma_mon", "")).zfill(6)
            tc = d.get("so_tc")
            if ma and tc is not None:
                try:
                    out[ma] = int(tc)
                except (ValueError, TypeError):
                    pass
    return out


def _cli() -> None:
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Đường dẫn xlsx/csv.")
    ap.add_argument("--nganh", required=True, choices=["CS", "IS", "DS", "SE", "IT"])
    ap.add_argument("--id", dest="id_sv", default=None, help="IDSinhVien. None → SV đầu.")
    ap.add_argument(
        "--corpus",
        default="data/embeddings/corpus.jsonl",
        help="Để load so_tc cho tính GPA weighted.",
    )
    args = ap.parse_args()

    so_tc_map = load_so_tc_map(args.corpus) if Path(args.corpus).exists() else None
    profile = load_profile_from_xlsx(args.input, args.nganh, args.id_sv, so_tc_map)
    print(profile.summary_markdown())
    print(f"\n— Completed: {len(profile.completed)} môn, "
          f"vd: {profile.completed[:5]}{'...' if len(profile.completed) > 5 else ''}")


if __name__ == "__main__":
    _cli()
