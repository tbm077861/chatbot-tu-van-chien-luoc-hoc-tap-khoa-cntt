"""Parse HTML chương trình khung từ bảng #chuongtrinhkhungtbl.

Ví dụ sử dụng CLI:
    # Parse một ngành
    python src/data/parser.py --input data/raw/curriculum/CS_curriculum.html --nganh CS

    # Parse tất cả 5 ngành
    python src/data/parser.py --all

    # Output mặc định: data/processed/curriculum_graph/<NGANH>_curriculum.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

# Ánh xạ tên ngành → file HTML
NGANH_FILES: dict[str, str] = {
    "CS": "data/raw/curriculum/CS_curriculum.html",
    "IS": "data/raw/curriculum/IS_curriculum.html",
    "DS": "data/raw/curriculum/DS_curriculum.html",
    "SE": "data/raw/curriculum/SE_curriculum.html",
    "IT": "data/raw/curriculum/IT_curriculum.html",
}

# Regex tách mã môn + loại điều kiện: "004247(a)" hoặc "001782,001611(a)"
_DIEU_KIEN_RE = re.compile(r"^([\d,]+)\(([abc])\)$")


def _parse_dieu_kien(cell_text: str) -> tuple[list[str], str | None]:
    """Tách danh sách mã môn tiên quyết và loại điều kiện từ text cell.

    Args:
        cell_text: Nội dung text của cột "Học phần học trước/tiên quyết/song hành",
            ví dụ "004247(a)" hay "001782,001611(a)" hay "" (rỗng).

    Returns:
        Tuple (danh_sach_ma_mon, loai_dieu_kien). loai_dieu_kien là "a", "b", "c"
        hoặc None nếu không có điều kiện.
    """
    text = cell_text.strip()
    if not text:
        return [], None

    m = _DIEU_KIEN_RE.match(text)
    if not m:
        # Thử bỏ khoảng trắng thừa rồi match lại
        text_clean = re.sub(r"\s+", "", text)
        m = _DIEU_KIEN_RE.match(text_clean)
    if not m:
        return [], None

    ma_mons_raw, loai = m.group(1), m.group(2)
    # Pad 0 bên trái đến 6 chữ số
    ma_mons = [code.strip().zfill(6) for code in ma_mons_raw.split(",") if code.strip()]
    return ma_mons, loai


def _parse_tong_tc(cell_text: str) -> int:
    """Lấy số tín chỉ từ text dạng 'Tổng số TC: 11' hoặc chỉ '11'.

    Args:
        cell_text: Nội dung text cột tổng tín chỉ.

    Returns:
        Số tín chỉ nguyên, 0 nếu không parse được.
    """
    nums = re.findall(r"\d+", cell_text)
    return int(nums[-1]) if nums else 0


def _is_starred(cell_tag: Tag) -> bool:
    """Kiểm tra môn có dấu * (không tính GPA) dựa trên span màu đỏ trong tên môn.

    Args:
        cell_tag: Tag <td> chứa tên môn học.

    Returns:
        True nếu có dấu * đỏ, False nếu không.
    """
    span = cell_tag.find("span", style=lambda v: v and "color:red" in v)
    return span is not None and "*" in span.get_text()


def parse_html(html_path: Path, nganh: str) -> dict[str, Any]:
    """Parse file HTML chương trình khung thành dict JSON chuẩn.

    Xử lý cấu trúc bảng với 3 loại row:
    - `hockytr` : header học kỳ (VD: "HỌC KỲ 1")
    - `HocKyRowCls` : header nhóm học phần ("Học phần bắt buộc" / "Học phần tự chọn")
    - `HocPhanRowCls` : dữ liệu từng môn học

    Args:
        html_path: Đường dẫn đến file HTML chứa bảng #chuongtrinhkhungtbl.
        nganh: Mã ngành (CS, IS, DS, SE, IT).

    Returns:
        Dict theo schema chuẩn dự án với khoá "nganh" và "hoc_ky".

    Raises:
        ValueError: Không tìm thấy bảng #chuongtrinhkhungtbl trong HTML.
        FileNotFoundError: File HTML không tồn tại.
    """
    if not html_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {html_path}")

    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", id="chuongtrinhkhungtbl")
    if table is None:
        raise ValueError(f"Không tìm thấy bảng #chuongtrinhkhungtbl trong {html_path}")

    hoc_ky_list: list[dict[str, Any]] = []
    current_hk: dict[str, Any] | None = None
    current_loai: str = "bat_buoc"

    for row in table.find_all("tr"):
        classes = row.get("class", [])
        cells = row.find_all("td")
        if not cells:
            continue

        # --- Header học kỳ ---
        if "hockytr" in classes:
            hk_text = cells[0].get_text(strip=True)
            # Lấy số học kỳ từ text "HỌC KỲ 1"
            hk_nums = re.findall(r"\d+", hk_text)
            hk_so = int(hk_nums[0]) if hk_nums else len(hoc_ky_list) + 1

            # Lấy tổng TC từ cell cuối (colspan)
            tong_tc = 0
            for c in reversed(cells):
                txt = c.get_text(strip=True)
                if txt:
                    tong_tc = _parse_tong_tc(txt)
                    break

            current_hk = {
                "hk_so": hk_so,
                "tong_tc": tong_tc,
                "hoc_phan": [],
            }
            hoc_ky_list.append(current_hk)
            current_loai = "bat_buoc"  # reset khi sang học kỳ mới
            continue

        # --- Header nhóm học phần ---
        if "HocKyRowCls" in classes:
            nhom_text = cells[0].get_text(strip=True).lower()
            if "tự chọn" in nhom_text:
                current_loai = "tu_chon"
            else:
                current_loai = "bat_buoc"
            continue

        # --- Dòng summary cuối bảng ---
        if "SummaryRowCls" in classes:
            continue

        # --- Dòng môn học ---
        if "HocPhanRowCls" in classes:
            if current_hk is None or len(cells) < 8:
                continue

            # Cột: STT | Mã môn | Tên môn | Mã học phần | Điều kiện | TC | LT | TH | ...
            ma_mon_raw = cells[1].get_text(strip=True)
            ma_mon = ma_mon_raw.zfill(6)

            ten_mon_cell = cells[2]
            ten_mon = ten_mon_cell.get_text(strip=True)
            # Bỏ ký tự " *" ở cuối tên nếu có (lấy sạch)
            ten_mon = re.sub(r"\s*\*\s*$", "", ten_mon).strip()
            khong_tinh_gpa = _is_starred(ten_mon_cell)

            ma_hoc_phan = cells[3].get_text(strip=True) or None

            # Điều kiện: lấy text thuần (không có HTML span)
            dieu_kien_cell = cells[4]
            # Xóa span giữ lại text số và (a/b/c)
            dieu_kien_raw = dieu_kien_cell.get_text(strip=True)
            dieu_kien_list, loai_dieu_kien = _parse_dieu_kien(dieu_kien_raw)

            try:
                so_tc = int(cells[5].get_text(strip=True))
            except (ValueError, IndexError):
                so_tc = 0

            try:
                so_tiet_lt = int(cells[6].get_text(strip=True))
            except (ValueError, IndexError):
                so_tiet_lt = 0

            try:
                so_tiet_th = int(cells[7].get_text(strip=True))
            except (ValueError, IndexError):
                so_tiet_th = 0

            hoc_phan: dict[str, Any] = {
                "ma_mon": ma_mon,
                "ten_mon": ten_mon,
                "ma_hoc_phan": ma_hoc_phan,
                "loai": current_loai,
                "so_tc": so_tc,
                "so_tiet_lt": so_tiet_lt,
                "so_tiet_th": so_tiet_th,
                "khong_tinh_gpa": khong_tinh_gpa,
                "dieu_kien": dieu_kien_list,
                "loai_dieu_kien": loai_dieu_kien,
            }
            current_hk["hoc_phan"].append(hoc_phan)

    return {"nganh": nganh, "hoc_ky": hoc_ky_list}


def parse_and_save(
    html_path: Path,
    nganh: str,
    output_dir: Path,
) -> Path:
    """Parse HTML và lưu kết quả ra file JSON.

    Args:
        html_path: Đường dẫn file HTML đầu vào.
        nganh: Mã ngành.
        output_dir: Thư mục lưu file JSON output.

    Returns:
        Đường dẫn file JSON đã lưu.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data = parse_html(html_path, nganh)

    out_path = output_dir / f"{nganh}_curriculum.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_mon = sum(len(hk["hoc_phan"]) for hk in data["hoc_ky"])
    print(f"[{nganh}] Đã parse {len(data['hoc_ky'])} học kỳ, {total_mon} học phần → {out_path}")
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse HTML chương trình khung → JSON",
    )
    p.add_argument(
        "--input",
        type=Path,
        help="Đường dẫn file HTML đầu vào (dùng kèm --nganh).",
    )
    p.add_argument(
        "--nganh",
        choices=list(NGANH_FILES.keys()),
        help="Mã ngành (CS/IS/DS/SE/IT). Bắt buộc khi dùng --input.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Parse tất cả 5 ngành từ đường dẫn mặc định.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/curriculum_graph"),
        help="Thư mục lưu JSON output (mặc định: data/processed/curriculum_graph).",
    )
    return p


def main() -> None:
    """Điểm vào CLI."""
    args = _build_arg_parser().parse_args()

    # Xác định working directory là gốc dự án (nơi chứa src/)
    project_root = Path(__file__).resolve().parents[2]

    if args.all:
        errors: list[str] = []
        for nganh, rel_path in NGANH_FILES.items():
            html_path = project_root / rel_path
            try:
                parse_and_save(html_path, nganh, project_root / args.output_dir)
            except (FileNotFoundError, ValueError) as e:
                print(f"[{nganh}] LỖI: {e}")
                errors.append(nganh)
        if errors:
            print(f"\nCác ngành bị lỗi: {errors}")
        else:
            print("\nParse tất cả ngành thành công.")
    elif args.input and args.nganh:
        parse_and_save(args.input, args.nganh, project_root / args.output_dir)
    else:
        _build_arg_parser().print_help()


if __name__ == "__main__":
    main()
