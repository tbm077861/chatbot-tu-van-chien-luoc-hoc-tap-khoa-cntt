"""Collaborative Filtering augmentation — sinh virtual student profile bằng SVD.

Ý tưởng:
1. Pivot ma trận (sinh viên × môn) từ student_profiles thực; ô trống = chưa học.
2. Áp dụng Truncated SVD trên ma trận đã mask (mean-fill, biased SVD) để học
   latent factor U, V phản ánh sở thích/xu hướng học.
3. Sample latent vector mới (Gaussian quanh mean+std của population) → reconstruct
   → predicted grade cho mọi môn của ngành đó.
4. Threshold: chỉ giữ k môn predicted > 5.0 làm "đã học" của virtual student.
5. Chuyển thành training sample dạng QA (giống graph_sampler).

Ví dụ CLI:
    # Sinh 6000 virtual profile/ngành × 5 ngành = ~30k
    python src/data/augmentation/cf_augment.py --all --n_virtual 6000

    # Test 100 virtual profile cho CS
    python src/data/augmentation/cf_augment.py --nganh CS --n_virtual 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_grades_for_nganh(project_root: Path, nganh: str) -> pd.DataFrame:
    """Load file <NGANH>_grades_clean.parquet.

    Args:
        project_root: Thư mục gốc.
        nganh: Mã ngành.

    Returns:
        DataFrame cột [IDSinhVien, MaMonHoc, TenMonHoc, DiemTongKet, ThuocKCNTT].
    """
    path = project_root / "data/processed/student_profiles" / f"{nganh}_grades_clean.parquet"
    df = pd.read_parquet(path)
    # Lấy điểm tốt nhất cho mỗi (SV, môn) — tránh duplicate
    df = (
        df.groupby(["IDSinhVien", "MaMonHoc"], as_index=False)
        .agg({"TenMonHoc": "first", "DiemTongKet": "max", "ThuocKCNTT": "first"})
    )
    return df


def build_pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Pivot DataFrame thành ma trận (SV × Môn).

    Args:
        df: DataFrame grades.

    Returns:
        Tuple (pivot_df, mon_codes, sv_ids):
            - pivot_df: index=IDSinhVien, columns=MaMonHoc, value=DiemTongKet, NaN nếu chưa học.
            - mon_codes: list mã môn theo thứ tự cột.
            - sv_ids: list IDSinhVien theo thứ tự hàng.
    """
    pivot = df.pivot_table(
        index="IDSinhVien",
        columns="MaMonHoc",
        values="DiemTongKet",
        aggfunc="max",
    )
    return pivot, list(pivot.columns), list(pivot.index)


def fit_svd(pivot: pd.DataFrame, n_components: int, seed: int) -> tuple[TruncatedSVD, np.ndarray, np.ndarray, float]:
    """Fit Truncated SVD trên pivot matrix đã mean-fill.

    Args:
        pivot: DataFrame pivot (SV × Môn), có NaN.
        n_components: Số latent factor.
        seed: Random seed.

    Returns:
        Tuple (svd_model, U_matrix, V_matrix, global_mean):
            - U: latent của SV (n_sv × k).
            - V: latent của môn (k × n_mon).
            - global_mean: điểm trung bình toàn population (dùng để de-bias).
    """
    global_mean = float(pivot.stack().mean())
    # Fill NaN bằng global mean để chạy SVD; SVD học được pattern lệch khỏi mean.
    filled = pivot.fillna(global_mean).to_numpy(dtype=np.float32)
    # Trừ mean (centering)
    centered = filled - global_mean

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    U = svd.fit_transform(centered)            # (n_sv, k)
    V = svd.components_                        # (k, n_mon)
    return svd, U, V, global_mean


def generate_virtual_latent(U: np.ndarray, n_virtual: int, rng: np.random.Generator) -> np.ndarray:
    """Sinh latent vector mới bằng cách sample từ phân phối Gaussian quanh mean/std của U.

    Args:
        U: Ma trận latent của sinh viên thật (n_sv × k).
        n_virtual: Số virtual student cần sinh.
        rng: NumPy random generator.

    Returns:
        Ma trận latent virtual (n_virtual × k).
    """
    mean = U.mean(axis=0)
    std = U.std(axis=0)
    # Sample từ N(mean, std) per dimension (giả định độc lập latent dim)
    return rng.normal(loc=mean, scale=std, size=(n_virtual, U.shape[1])).astype(np.float32)


def predict_grades(U_virtual: np.ndarray, V: np.ndarray, global_mean: float) -> np.ndarray:
    """Tính điểm dự đoán: grades = U_virtual @ V + global_mean.

    Args:
        U_virtual: Latent virtual students (n × k).
        V: Latent môn (k × n_mon).
        global_mean: Mean để cộng lại sau centering.

    Returns:
        Ma trận điểm (n × n_mon), clip [0, 10].
    """
    pred = U_virtual @ V + global_mean
    return np.clip(pred, 0.0, 10.0)


def latent_to_profile(
    pred_grades: np.ndarray,
    mon_codes: list[str],
    course_info: dict[str, dict[str, Any]],
    rng: random.Random,
    n_courses_range: tuple[int, int] = (5, 25),
    grade_threshold: float = 5.0,
) -> list[dict[str, Any]]:
    """Chuyển ma trận điểm dự đoán thành list virtual student profile.

    Mỗi virtual student:
    - Số môn đã học: random trong n_courses_range (mô phỏng SV ở các giai đoạn khác nhau).
    - Chọn các môn có pred_grade cao nhất + thêm ít nhiễu để đa dạng.
    - Loại bỏ môn pred_grade < threshold.

    Args:
        pred_grades: (n_virtual × n_mon).
        mon_codes: list mã môn theo cột.
        course_info: {ma_mon: {ten_mon, so_tc, khong_tinh_gpa}}, dùng khi build profile.
        rng: random.Random instance.
        n_courses_range: tuple (min, max) số môn mỗi virtual SV "đã học".
        grade_threshold: chỉ giữ môn có pred >= threshold.

    Returns:
        List profile, mỗi profile có "courses": [{ma_mon, ten_mon, diem, so_tc, khong_tinh_gpa}, ...].
    """
    profiles: list[dict[str, Any]] = []
    n_virtual = pred_grades.shape[0]

    for i in range(n_virtual):
        n_take = rng.randint(*n_courses_range)
        # Thêm nhiễu nhỏ để hai virtual student không lấy cùng top-k
        scores = pred_grades[i] + np.array(
            [rng.gauss(0, 0.3) for _ in range(pred_grades.shape[1])], dtype=np.float32
        )
        # Top-n_take môn có score cao nhất, lọc theo threshold
        order = np.argsort(-scores)
        chosen: list[dict[str, Any]] = []
        for idx in order:
            if len(chosen) >= n_take:
                break
            grade = float(pred_grades[i, idx])
            if grade < grade_threshold:
                continue
            ma = mon_codes[idx]
            info = course_info.get(ma, {})
            chosen.append({
                "ma_mon": ma,
                "ten_mon": info.get("ten_mon", "?"),
                "so_tc": info.get("so_tc", 3),
                "diem": round(grade, 1),
                "khong_tinh_gpa": info.get("khong_tinh_gpa", False),
                "hk_so": info.get("hk_so", 0),
            })
        if len(chosen) >= 3:  # virtual SV phải có ≥3 môn để có ý nghĩa
            profiles.append({"virtual_id": i, "courses": chosen})

    return profiles


def load_course_info(project_root: Path, nganh: str) -> dict[str, dict[str, Any]]:
    """Đọc curriculum JSON → ánh xạ ma_mon → thông tin môn.

    Args:
        project_root: Thư mục gốc.
        nganh: Mã ngành.

    Returns:
        Dict ma_mon → {ten_mon, so_tc, khong_tinh_gpa, loai, hk_so}.
    """
    path = project_root / "data/processed/curriculum_graph" / f"{nganh}_curriculum.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    info: dict[str, dict[str, Any]] = {}
    for hk in data["hoc_ky"]:
        for hp in hk["hoc_phan"]:
            info[hp["ma_mon"]] = {
                "ten_mon": hp["ten_mon"],
                "so_tc": hp["so_tc"],
                "khong_tinh_gpa": hp["khong_tinh_gpa"],
                "loai": hp["loai"],
                "hk_so": hk["hk_so"],
            }
    return info


def compute_gpa(courses: list[dict[str, Any]]) -> float:
    """GPA thang 10 chỉ tính môn không có khong_tinh_gpa và đã đạt (diem >= 4.0).

    Args:
        courses: List môn đã học.

    Returns:
        GPA tích luỹ thang 10, làm tròn 2 chữ số.
    """
    items = [c for c in courses if not c.get("khong_tinh_gpa", False) and c["diem"] >= 4.0]
    if not items:
        return 0.0
    total_tc = sum(c["so_tc"] for c in items)
    if total_tc == 0:
        return 0.0
    return round(sum(c["diem"] * c["so_tc"] for c in items) / total_tc, 2)


def profile_to_samples(
    profile: dict[str, Any],
    nganh: str,
    course_info: dict[str, dict[str, Any]],
    sample_id_prefix: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Tạo training sample từ một virtual profile.

    Mỗi profile cắt thành 1-2 sample bằng cách:
    - Sort courses theo hk_so, chọn split point ngẫu nhiên.
    - history = các môn trước split, target = môn sau split (cùng HK gần nhất).

    Args:
        profile: Virtual student profile.
        nganh: Mã ngành.
        course_info: Map ma_mon → info.
        sample_id_prefix: Prefix cho ID sample.
        rng: random instance.

    Returns:
        List samples.
    """
    courses = sorted(profile["courses"], key=lambda c: (c["hk_so"], c["ma_mon"]))
    if len(courses) < 4:
        return []

    samples: list[dict[str, Any]] = []
    n_cuts = min(2, len(courses) - 2)
    cuts = rng.sample(range(2, len(courses) - 1), k=n_cuts)

    for ci, cut in enumerate(sorted(cuts)):
        history_courses = courses[:cut]
        target_courses = courses[cut:cut + min(5, len(courses) - cut)]

        completed_codes = [c["ma_mon"] for c in history_courses]
        gpa = compute_gpa(history_courses)
        last_hk = max(c["hk_so"] for c in history_courses) if history_courses else 1
        target_hk = max(c["hk_so"] for c in target_courses) if target_courses else last_hk + 1

        question = (
            f"Em là sinh viên ngành {nganh}, đã học {len(history_courses)} môn với "
            f"GPA tích luỹ {gpa:.2f}. Em vừa hoàn thành tới HK{last_hk}. "
            f"Dựa trên lịch sử học tập, em nên đăng ký môn gì ở HK{target_hk}?"
        )
        answer = (
            f"Dựa trên xu hướng học tập tương tự các sinh viên khác, "
            f"HK{target_hk} bạn nên cân nhắc {len(target_courses)} môn: "
            + ", ".join(f"{c['ma_mon']} ({c['ten_mon']})" for c in target_courses)
            + "."
        )

        samples.append({
            "id": f"{sample_id_prefix}_cf{ci + 1}",
            "source": "cf_svd_augmentation",
            "nganh": nganh,
            "context": {
                "completed_codes": completed_codes,
                "current_hk": last_hk,
                "gpa_thang10": gpa,
            },
            "history": [
                {
                    "ma_mon": c["ma_mon"],
                    "ten_mon": c["ten_mon"],
                    "so_tc": c["so_tc"],
                    "diem": c["diem"],
                }
                for c in history_courses
            ],
            "target": {
                "hk_so": target_hk,
                "course_codes": [c["ma_mon"] for c in target_courses],
                "course_names": [c["ten_mon"] for c in target_courses],
                "total_tc": sum(c["so_tc"] for c in target_courses),
            },
            "qa": {"question": question, "answer": answer},
        })

    return samples


def generate_for_nganh(
    nganh: str,
    n_virtual: int,
    project_root: Path,
    n_components: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Pipeline đầy đủ cho một ngành.

    Args:
        nganh: Mã ngành.
        n_virtual: Số virtual student.
        project_root: Thư mục gốc.
        n_components: Số latent factor SVD.
        seed: Random seed.

    Returns:
        List training samples.
    """
    df = load_grades_for_nganh(project_root, nganh)
    pivot, mon_codes, _ = build_pivot(df)

    n_real_sv = pivot.shape[0]
    n_mon = pivot.shape[1]
    # Bảo đảm n_components < min(n_sv, n_mon)
    k = min(n_components, min(n_real_sv, n_mon) - 1)
    _, U, V, global_mean = fit_svd(pivot, k, seed)

    np_rng = np.random.default_rng(seed)
    rng = random.Random(seed)

    U_virtual = generate_virtual_latent(U, n_virtual, np_rng)
    pred = predict_grades(U_virtual, V, global_mean)

    course_info = load_course_info(project_root, nganh)
    profiles = latent_to_profile(pred, mon_codes, course_info, rng)

    all_samples: list[dict[str, Any]] = []
    for i, profile in enumerate(profiles):
        prefix = f"{nganh}_cf_{i:05d}"
        all_samples.extend(profile_to_samples(profile, nganh, course_info, prefix, rng))

    return all_samples


def save_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Lưu samples ra JSONL.

    Args:
        samples: List samples.
        output_path: Đường dẫn file .jsonl.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CF (SVD) augmentation cho student profiles")
    p.add_argument("--nganh", choices=["CS", "IS", "DS", "SE", "IT"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--n_virtual", type=int, default=6000, help="Số virtual student/ngành")
    p.add_argument("--n_components", type=int, default=20, help="Số latent factor SVD")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=Path, default=Path("data/augmented/cf_profiles"))
    return p


def main() -> None:
    """Điểm vào CLI."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _build_arg_parser().parse_args()
    project_root = _PROJECT_ROOT

    nganh_list = ["CS", "IS", "DS", "SE", "IT"] if args.all else (
        [args.nganh] if args.nganh else []
    )
    if not nganh_list:
        _build_arg_parser().print_help()
        return

    output_dir = project_root / args.output_dir
    grand_total = 0
    for idx, nganh in enumerate(nganh_list):
        samples = generate_for_nganh(
            nganh, args.n_virtual, project_root, args.n_components,
            seed=args.seed + idx * 1000,
        )
        out_path = output_dir / f"{nganh}_cf_samples.jsonl"
        save_samples(samples, out_path)
        print(f"[{nganh}] {len(samples)} samples → {out_path.name}")
        grand_total += len(samples)

    print(f"\nTổng cộng: {grand_total} CF samples từ {len(nganh_list)} ngành.")


if __name__ == "__main__":
    main()
