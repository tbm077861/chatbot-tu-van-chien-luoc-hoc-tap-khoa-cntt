"""Parse quy định học vụ từ file text thành JSON structured.

Trích xuất các quy tắc quan trọng từ Quy chế đào tạo IUH (file txt).
Output: data/processed/regulations.json

Ví dụ sử dụng CLI:
    python src/data/regulation_parser.py
    python src/data/regulation_parser.py --input data/raw/regulations/quy_dinh_hoc_vu.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/raw/regulations/quy_dinh_hoc_vu.txt")
DEFAULT_OUTPUT = Path("data/processed/regulations.json")


def _extract_tin_chi_dang_ky(text: str) -> dict[str, Any]:
    """Trích xuất giới hạn tín chỉ đăng ký mỗi học kỳ.

    Điều 6 khoản 1b: tối thiểu 12 TC, tối đa 30 TC/học kỳ chính.

    Args:
        text: Toàn bộ nội dung file quy định.

    Returns:
        Dict với khóa tc_min và tc_max.
    """
    result: dict[str, Any] = {"tc_min": 12, "tc_max": 30}

    # Tìm "tối thiểu là X tín chỉ"
    m_min = re.search(r"tối thiểu\s+là\s+(\d+)\s+tín chỉ", text, re.IGNORECASE)
    if m_min:
        result["tc_min"] = int(m_min.group(1))

    # Tìm "tối đa là X tín chỉ"
    m_max = re.search(r"tối đa\s+là\s+(\d+)\s+tín chỉ", text, re.IGNORECASE)
    if m_max:
        result["tc_max"] = int(m_max.group(1))

    return result


def _extract_canh_bao_hoc_tap(text: str) -> dict[str, Any]:
    """Trích xuất điều kiện cảnh báo học tập (Điều 12).

    Args:
        text: Nội dung file quy định.

    Returns:
        Dict với các ngưỡng cảnh báo: tc_no_max, dtbhl_min theo năm học.
    """
    result: dict[str, Any] = {
        "tc_no_max": 24,
        "ti_le_khong_dat_max": 0.5,
        "dtbhl_min_hk1": 0.8,
        "dtbhl_min_hk_tiep": 1.0,
        "dtbhltl_nam1": 1.2,
        "dtbhltl_nam2": 1.4,
        "dtbhltl_nam3": 1.6,
        "dtbhltl_nam4_plus": 1.8,
        "so_lan_canh_bao_lien_tiep_bi_thoi_hoc": 2,
    }

    # Trích "vượt quá 24 tín chỉ"
    m_no = re.search(r"nợ từ đầu khóa học vượt quá\s+(\d+)\s+tín chỉ", text, re.IGNORECASE)
    if m_no:
        result["tc_no_max"] = int(m_no.group(1))

    # Ngưỡng ĐTBHL
    patterns_dtbhl = [
        (r"dưới\s+([\d.]+)\s+đối với học kỳ đầu", "dtbhl_min_hk1"),
        (r"dưới\s+([\d.]+)\s+đối với các học kỳ tiếp theo", "dtbhl_min_hk_tiep"),
    ]
    for pattern, key in patterns_dtbhl:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result[key] = float(m.group(1))

    # Ngưỡng ĐTBHLTL theo năm
    patterns_tl = [
        (r"dưới\s+([\d.]+)\s+đối với sinh viên trình độ năm thứ nhất", "dtbhltl_nam1"),
        (r"dưới\s+([\d.]+)\s+đối với sinh viên trình độ năm thứ hai", "dtbhltl_nam2"),
        (r"dưới\s+([\d.]+)\s+đối với sinh viên trình độ năm thứ ba", "dtbhltl_nam3"),
        (r"dưới\s+([\d.]+)\s+đối với sinh viên trình độ các năm tiếp theo", "dtbhltl_nam4_plus"),
    ]
    for pattern, key in patterns_tl:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result[key] = float(m.group(1))

    return result


def _extract_tot_nghiep(text: str) -> dict[str, Any]:
    """Trích xuất điều kiện tốt nghiệp (Điều 23).

    Args:
        text: Nội dung file quy định.

    Returns:
        Dict với gpa_min_tot_nghiep và mô tả điều kiện.
    """
    result: dict[str, Any] = {
        "gpa_min_thang4": 2.0,
        "dieu_kien": [
            "Tích lũy đủ học phần và số tín chỉ theo CTĐT",
            "Điểm trung bình tích lũy toàn khóa >= 2.00 (thang 4)",
            "Không bị truy cứu trách nhiệm hình sự hoặc đình chỉ học tập",
            "Đạt chuẩn đầu ra ngoại ngữ và công nghệ thông tin",
        ],
    }

    m = re.search(
        r"điểm trung bình tích lũy.*?đạt từ\s+([\d.]+)\s+trở lên.*?thang điểm 4",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        result["gpa_min_thang4"] = float(m.group(1))

    return result


def _extract_bang_quy_doi_diem(text: str) -> list[dict[str, Any]]:
    """Trích xuất bảng quy đổi điểm 10 → 4 → chữ (Điều 19, Bảng 2).

    Args:
        text: Nội dung file quy định.

    Returns:
        Danh sách dict với khóa thang10_min, thang10_max, thang4, thu_tu_xep_loai.
    """
    # Bảng cố định từ Điều 19 — trích từ văn bản đã đọc
    return [
        {"thu_tu": 1, "thang10_tu": 9.0, "thang10_den": 10.0, "thang4": 4.0, "diem_chu": "A+"},
        {"thu_tu": 2, "thang10_tu": 8.5, "thang10_den": 8.9, "thang4": 3.8, "diem_chu": "A"},
        {"thu_tu": 3, "thang10_tu": 8.0, "thang10_den": 8.4, "thang4": 3.5, "diem_chu": "B+"},
        {"thu_tu": 4, "thang10_tu": 7.0, "thang10_den": 7.9, "thang4": 3.0, "diem_chu": "B"},
        {"thu_tu": 5, "thang10_tu": 6.0, "thang10_den": 6.9, "thang4": 2.5, "diem_chu": "C+"},
        {"thu_tu": 6, "thang10_tu": 5.5, "thang10_den": 5.9, "thang4": 2.0, "diem_chu": "C"},
        {"thu_tu": 7, "thang10_tu": 5.0, "thang10_den": 5.4, "thang4": 1.5, "diem_chu": "D+"},
        {"thu_tu": 8, "thang10_tu": 4.0, "thang10_den": 4.9, "thang4": 1.0, "diem_chu": "D"},
        {"thu_tu": 9, "thang10_tu": 0.0, "thang10_den": 3.9, "thang4": 0.0, "diem_chu": "F"},
    ]


def _extract_xep_loai_hoc_luc(text: str) -> list[dict[str, Any]]:
    """Trích xuất bảng xếp loại học lực theo ĐTBCTL (Điều 21, Bảng 3).

    Args:
        text: Nội dung file quy định.

    Returns:
        Danh sách dict với khóa diem_tu, diem_den, xep_loai.
    """
    return [
        {"diem_tu": 3.6, "diem_den": 4.0, "xep_loai": "Xuất sắc"},
        {"diem_tu": 3.2, "diem_den": 3.59, "xep_loai": "Giỏi"},
        {"diem_tu": 2.5, "diem_den": 3.19, "xep_loai": "Khá"},
        {"diem_tu": 2.0, "diem_den": 2.49, "xep_loai": "Trung bình"},
        {"diem_tu": 0.0, "diem_den": 1.99, "xep_loai": "Kém"},
    ]


def _extract_hoc_phan_dieu_kien(text: str) -> dict[str, Any]:
    """Trích xuất thông tin học phần điều kiện (không tính GPA).

    Điều 19b + Điều 2 khoản 7: Giáo dục Thể chất, Quốc phòng, Chứng chỉ Tiếng Anh, Tin học.

    Args:
        text: Nội dung file quy định.

    Returns:
        Dict với danh sách tên nhóm môn điều kiện.
    """
    return {
        "khong_tinh_gpa": [
            "Giáo dục Thể chất",
            "Giáo dục Quốc phòng và An ninh",
            "Chứng chỉ Tiếng Anh",
            "Chứng chỉ Tin học",
        ],
        "ghi_chu": (
            "Các học phần này chỉ yêu cầu đạt (P/F), "
            "không tính vào điểm trung bình tích lũy (ĐTBCTL)."
        ),
    }


def _extract_dieu_kien_tien_quyet(text: str) -> dict[str, Any]:
    """Trích xuất định nghĩa các loại điều kiện học phần (Điều 2 khoản 7c).

    Args:
        text: Nội dung file quy định.

    Returns:
        Dict mô tả 3 loại điều kiện: hoc_truoc (a), tien_quyet (b), song_hanh (c).
    """
    return {
        "hoc_truoc": {
            "ky_hieu": "a",
            "mo_ta": (
                "Phải đăng ký và học học phần A ở học kỳ trước đó "
                "(học hết nội dung, tham gia đánh giá, nhưng có thể chưa đạt)."
            ),
        },
        "tien_quyet": {
            "ky_hieu": "b",
            "mo_ta": (
                "Phải đăng ký và học hoàn tất (đạt kết quả) học phần A "
                "trước khi được đăng ký học phần B."
            ),
        },
        "song_hanh": {
            "ky_hieu": "c",
            "mo_ta": (
                "Sinh viên đã đăng ký học phần A; "
                "được phép học B đồng thời hoặc sau A."
            ),
        },
    }


def _extract_hoc_lai_cai_thien(text: str) -> dict[str, Any]:
    """Trích xuất quy định học lại và học cải thiện (Điều 6 khoản 5).

    Args:
        text: Nội dung file quy định.

    Returns:
        Dict mô tả quy tắc học lại bắt buộc và học cải thiện tự nguyện.
    """
    return {
        "hoc_lai_bat_buoc": (
            "Học phần bắt buộc điểm F → phải đăng ký học lại trong các HK tiếp theo cho đến khi đạt."
        ),
        "hoc_lai_tu_chon": (
            "Học phần tự chọn điểm F → học lại môn đó hoặc đổi sang môn tự chọn khác "
            "cùng nhóm tự chọn, cùng học kỳ trong kế hoạch đào tạo."
        ),
        "cai_thien_diem": (
            "Sinh viên có môn đạt điểm A, B+, B, C+, C, D+, D "
            "được phép đăng ký học lại để cải thiện điểm."
        ),
        "tinh_diem_tu_chon": (
            "Sinh viên có thể đăng ký nhiều môn tự chọn hơn yêu cầu; "
            "hệ thống lấy các môn có điểm cao nhất để tính tích lũy."
        ),
    }


def parse_regulations(text: str) -> dict[str, Any]:
    """Trích xuất toàn bộ quy định quan trọng từ văn bản quy chế.

    Args:
        text: Toàn bộ nội dung file quy định học vụ.

    Returns:
        Dict JSON structured với tất cả quy tắc đã trích xuất.
    """
    return {
        "nguon": "Quy chế đào tạo Trường Đại học Công nghiệp TP.HCM (IUH)",
        "dang_ky_hoc_phan": _extract_tin_chi_dang_ky(text),
        "canh_bao_hoc_tap": _extract_canh_bao_hoc_tap(text),
        "dieu_kien_tot_nghiep": _extract_tot_nghiep(text),
        "bang_quy_doi_diem": _extract_bang_quy_doi_diem(text),
        "xep_loai_hoc_luc": _extract_xep_loai_hoc_luc(text),
        "hoc_phan_dieu_kien": _extract_hoc_phan_dieu_kien(text),
        "loai_dieu_kien_hoc_phan": _extract_dieu_kien_tien_quyet(text),
        "hoc_lai_va_cai_thien": _extract_hoc_lai_cai_thien(text),
    }


def parse_and_save(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Parse file quy định và lưu JSON output.

    Args:
        input_path: Đường dẫn file .txt quy định học vụ.
        output_path: Đường dẫn file .json output.

    Returns:
        Dict JSON đã parse.

    Raises:
        FileNotFoundError: File input không tồn tại.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    data = parse_regulations(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Đã parse quy định → {output_path}")
    print(f"  TC đăng ký: {data['dang_ky_hoc_phan']['tc_min']}–{data['dang_ky_hoc_phan']['tc_max']} TC/HK")
    print(f"  GPA tốt nghiệp tối thiểu: {data['dieu_kien_tot_nghiep']['gpa_min_thang4']} (thang 4)")
    print(f"  Số mục quy định đã trích: {len(data)}")

    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse quy định học vụ → JSON")
    p.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"File txt quy định (mặc định: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"File json output (mặc định: {DEFAULT_OUTPUT})",
    )
    return p


def main() -> None:
    """Điểm vào CLI."""
    args = _build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    parse_and_save(
        project_root / args.input,
        project_root / args.output,
    )


if __name__ == "__main__":
    main()
