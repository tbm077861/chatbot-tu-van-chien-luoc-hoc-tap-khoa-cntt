"""Graph-based path sampling — sinh kịch bản học tập từ chương trình khung.

Mỗi sample mô phỏng một sinh viên đi qua HK1→HK_max theo curriculum, có:
- Tôn trọng prerequisite graph (môn cha phải đạt trước).
- Tôn trọng giới hạn tín chỉ học vụ (12–30 TC/HK theo regulations.json).
- Biến thể student-style: học sinh giỏi/trung bình/yếu, có thể trượt một số môn.
- Career bias (tuỳ chọn): ưu tiên môn tự chọn theo định hướng nghề.

Mỗi path sinh ra nhiều training sample (cắt ở mỗi mốc HK):
  (history HK1..HK_k, target = các môn nên học HK_{k+1})

Ví dụ CLI:
    # Sinh 800 path × 5 ngành = ~20k samples
    python src/data/augmentation/graph_sampler.py --all --n_paths 800

    # Sinh riêng cho CS với 100 paths để test nhanh
    python src/data/augmentation/graph_sampler.py --nganh CS --n_paths 100
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

# Cho phép chạy CLI từ thư mục dự án và import src.*
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Định nghĩa career cluster ngành CS (theo project_instructions.md mục 5).
# Các ngành khác sẽ sample không bias.
CAREER_CLUSTERS: dict[str, dict[str, list[str]]] = {
    "CS": {
        "AI/ML": ["001954", "015028", "015030", "015031", "015032"],
        "CV": ["015029", "015031", "015032"],
        "DB/BigData": ["001922", "014181", "004099", "004024", "003953"],
        "Java/Web": ["002876", "002804", "002399", "004227"],
        "NET": ["002990", "003406"],
        "Theory": ["001814", "001730", "001758"],
    },
}

# Profile sinh viên: mean điểm + std, dùng cho Gaussian sampling.
STUDENT_TYPES: dict[str, tuple[float, float]] = {
    "gioi":      (8.3, 0.8),
    "kha":       (7.2, 1.0),
    "trung_binh": (6.0, 1.2),
    "yeu":       (4.8, 1.4),
}
STUDENT_WEIGHTS = [0.20, 0.45, 0.25, 0.10]  # tỉ lệ kiểu sinh viên trong sample


@dataclass
class Course:
    """Một học phần trong chương trình khung."""
    ma_mon: str
    ten_mon: str
    so_tc: int
    loai: str  # "bat_buoc" | "tu_chon"
    hk_so: int
    khong_tinh_gpa: bool
    prereqs: list[str] = field(default_factory=list)


@dataclass
class StudentSnapshot:
    """Trạng thái sinh viên tại một thời điểm (kết thúc HK k)."""
    student_type: str
    career: str | None
    completed: list[dict[str, Any]] = field(default_factory=list)  # [{ma_mon, ten_mon, diem, so_tc, hk}, ...]
    failed_pending: list[str] = field(default_factory=list)        # mã môn bị F, chờ học lại
    current_hk: int = 0

    def passed_codes(self) -> set[str]:
        """Trả về set mã môn đã đạt (điểm >= 4.0 và không nằm trong failed_pending)."""
        return {c["ma_mon"] for c in self.completed if c["diem"] >= 4.0}

    def gpa(self) -> float:
        """Tính GPA tích luỹ (thang 10) chỉ trên môn tính GPA và đã đạt."""
        items = [c for c in self.completed if not c["khong_tinh_gpa"] and c["diem"] >= 4.0]
        if not items:
            return 0.0
        total_tc = sum(c["so_tc"] for c in items)
        if total_tc == 0:
            return 0.0
        return round(sum(c["diem"] * c["so_tc"] for c in items) / total_tc, 2)


def load_curriculum(curriculum_path: Path) -> tuple[dict[int, list[Course]], dict[str, Course]]:
    """Load curriculum JSON → ánh xạ (HK → courses) và (ma_mon → Course).

    Args:
        curriculum_path: Đường dẫn file <NGANH>_curriculum.json.

    Returns:
        Tuple (hk_map, course_map):
            - hk_map: {hk_so: [Course, ...]}
            - course_map: {ma_mon: Course}
    """
    with curriculum_path.open(encoding="utf-8") as f:
        data = json.load(f)

    hk_map: dict[int, list[Course]] = {}
    course_map: dict[str, Course] = {}

    for hk in data["hoc_ky"]:
        hk_so = hk["hk_so"]
        courses: list[Course] = []
        for hp in hk["hoc_phan"]:
            c = Course(
                ma_mon=hp["ma_mon"],
                ten_mon=hp["ten_mon"],
                so_tc=hp["so_tc"],
                loai=hp["loai"],
                hk_so=hk_so,
                khong_tinh_gpa=hp["khong_tinh_gpa"],
                prereqs=list(hp.get("dieu_kien", [])),
            )
            courses.append(c)
            course_map[c.ma_mon] = c
        hk_map[hk_so] = courses

    return hk_map, course_map


def sample_grade(student_type: str, rng: random.Random) -> float:
    """Sinh điểm cho một môn theo Gaussian quanh mean của student_type.

    Args:
        student_type: Một trong "gioi"/"kha"/"trung_binh"/"yeu".
        rng: Random instance để reproducible.

    Returns:
        Điểm thang 10, clip về [0.0, 10.0].
    """
    mean, std = STUDENT_TYPES[student_type]
    grade = rng.gauss(mean, std)
    return round(max(0.0, min(10.0, grade)), 1)


def pick_career(nganh: str, rng: random.Random) -> str | None:
    """Chọn ngẫu nhiên một định hướng nghề (chỉ áp dụng cho ngành có cluster).

    Args:
        nganh: Mã ngành.
        rng: Random instance.

    Returns:
        Tên cluster hoặc None nếu ngành không có định nghĩa cluster, hoặc 30% trả None
        (sinh viên chưa rõ định hướng).
    """
    if nganh not in CAREER_CLUSTERS:
        return None
    if rng.random() < 0.3:
        return None
    return rng.choice(list(CAREER_CLUSTERS[nganh].keys()))


def prioritize_electives(
    electives: list[Course],
    career: str | None,
    nganh: str,
    rng: random.Random,
) -> list[Course]:
    """Sắp xếp môn tự chọn theo career bias: môn thuộc cluster đứng trước.

    Args:
        electives: Danh sách môn tự chọn của HK đó.
        career: Tên cluster định hướng, hoặc None.
        nganh: Mã ngành.
        rng: Random instance.

    Returns:
        List electives đã sort, môn thuộc career cluster lên đầu.
    """
    if career is None or nganh not in CAREER_CLUSTERS:
        # Không có bias: shuffle ngẫu nhiên
        result = list(electives)
        rng.shuffle(result)
        return result

    cluster_codes = set(CAREER_CLUSTERS[nganh].get(career, []))
    in_cluster = [c for c in electives if c.ma_mon in cluster_codes]
    out_cluster = [c for c in electives if c.ma_mon not in cluster_codes]
    rng.shuffle(in_cluster)
    rng.shuffle(out_cluster)
    return in_cluster + out_cluster


def select_courses_for_hk(
    hk_courses: list[Course],
    student: StudentSnapshot,
    nganh: str,
    tc_min: int,
    tc_max: int,
    rng: random.Random,
) -> list[Course]:
    """Chọn các môn sẽ học trong một HK: bắt buộc thoả prereq + tự chọn theo career.

    Logic:
    1. Bắt đầu với các môn cần học lại (failed_pending nếu prereq đã thoả).
    2. Thêm tất cả môn bắt buộc của HK có prereq đã hoàn thành.
    3. Thêm môn tự chọn (ưu tiên theo career) cho đến khi vượt tc_max hoặc hết môn.
    4. Đảm bảo tổng TC >= tc_min nếu có thể; vượt nhẹ tc_max được phép tới +5 TC.

    Args:
        hk_courses: Tất cả môn của HK theo curriculum.
        student: Trạng thái sinh viên hiện tại.
        nganh: Mã ngành.
        tc_min: TC tối thiểu HK (12).
        tc_max: TC tối đa HK (30, được phép soft +5).
        rng: Random instance.

    Returns:
        Danh sách Course đã chọn cho HK này.
    """
    passed = student.passed_codes()
    selected: list[Course] = []
    total_tc = 0

    # 1. Học lại môn failed nếu prereq đã thoả
    for ma in list(student.failed_pending):
        # Truy ngược Course từ hk_courses (hoặc bỏ qua nếu không có trong HK này)
        # Để đơn giản: ta cho phép học lại bất kỳ HK nào, không cần đợi HK gốc.
        # Tìm trong toàn course_map (truyền qua closure không tiện) → bỏ qua phần này
        # trong bản đầu, failed_pending sẽ được xử lý ở build_simulation_path.
        pass

    # 2. Môn bắt buộc HK này (prereq đã thoả)
    must_learn = [c for c in hk_courses if c.loai == "bat_buoc"]
    for c in must_learn:
        if all(p in passed for p in c.prereqs):
            selected.append(c)
            total_tc += c.so_tc

    # 3. Môn tự chọn HK này
    electives = [c for c in hk_courses if c.loai == "tu_chon"]
    electives = prioritize_electives(electives, student.career, nganh, rng)
    for c in electives:
        if all(p in passed for p in c.prereqs):
            if total_tc + c.so_tc <= tc_max + 5:  # soft cap
                selected.append(c)
                total_tc += c.so_tc
            if total_tc >= tc_max:
                break

    # 4. Nếu vẫn dưới tc_min và còn elective khả thi → thêm
    if total_tc < tc_min:
        for c in electives:
            if c in selected:
                continue
            if all(p in passed for p in c.prereqs):
                selected.append(c)
                total_tc += c.so_tc
                if total_tc >= tc_min:
                    break

    return selected


def simulate_student_path(
    nganh: str,
    hk_map: dict[int, list[Course]],
    course_map: dict[str, Course],
    tc_min: int,
    tc_max: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Mô phỏng một sinh viên giả lập đi qua toàn bộ curriculum.

    Args:
        nganh: Mã ngành.
        hk_map: HK → [Course, ...] (từ load_curriculum).
        course_map: ma_mon → Course (từ load_curriculum).
        tc_min: TC tối thiểu/HK theo regulations.
        tc_max: TC tối đa/HK theo regulations.
        rng: Random instance.

    Returns:
        Danh sách "trạng thái cuối mỗi HK" — mỗi entry gồm hk, courses, snapshot.
    """
    student_type = rng.choices(list(STUDENT_TYPES.keys()), weights=STUDENT_WEIGHTS, k=1)[0]
    career = pick_career(nganh, rng)
    snap = StudentSnapshot(student_type=student_type, career=career)

    hk_history: list[dict[str, Any]] = []

    for hk_so in sorted(hk_map.keys()):
        snap.current_hk = hk_so

        # Học lại các môn failed_pending trước (nếu prereq thoả)
        retake_records: list[dict[str, Any]] = []
        passed_now = snap.passed_codes()
        for ma in list(snap.failed_pending):
            c = course_map.get(ma)
            if c is None:
                continue
            if all(p in passed_now for p in c.prereqs):
                grade = sample_grade(student_type, rng)
                # Học lại thường điểm khá hơn (+0.5 bonus)
                grade = round(min(10.0, grade + 0.5), 1)
                record = {
                    "ma_mon": ma,
                    "ten_mon": c.ten_mon,
                    "so_tc": c.so_tc,
                    "diem": grade,
                    "hk": hk_so,
                    "khong_tinh_gpa": c.khong_tinh_gpa,
                    "is_retake": True,
                }
                snap.completed.append(record)
                retake_records.append(record)
                if grade >= 4.0:
                    snap.failed_pending.remove(ma)

        # Chọn môn HK này
        chosen = select_courses_for_hk(
            hk_map[hk_so], snap, nganh, tc_min, tc_max, rng
        )

        records: list[dict[str, Any]] = []
        for c in chosen:
            grade = sample_grade(student_type, rng)
            record = {
                "ma_mon": c.ma_mon,
                "ten_mon": c.ten_mon,
                "so_tc": c.so_tc,
                "diem": grade,
                "hk": hk_so,
                "khong_tinh_gpa": c.khong_tinh_gpa,
                "is_retake": False,
            }
            snap.completed.append(record)
            records.append(record)
            if grade < 4.0 and c.loai == "bat_buoc":
                snap.failed_pending.append(c.ma_mon)

        hk_history.append({
            "hk_so": hk_so,
            "courses": retake_records + records,
            "total_tc": sum(r["so_tc"] for r in retake_records + records),
            "gpa_sau_hk": snap.gpa(),
        })

    return [{
        "student_type": snap.student_type,
        "career": snap.career,
        "nganh": nganh,
        "history": hk_history,
    }]


def path_to_training_samples(
    path: dict[str, Any],
    sample_id_prefix: str,
) -> list[dict[str, Any]]:
    """Cắt một path thành nhiều training sample (mỗi sample = history + next-target).

    Tại mỗi mốc HK k (1 <= k < hk_max), tạo một sample:
        - history: tất cả HK đã học từ HK1..HKk
        - target: danh sách môn HK_{k+1}
        - context: snapshot trạng thái sau HKk

    Args:
        path: Output của simulate_student_path()[0].
        sample_id_prefix: Tiền tố cho ID, ví dụ "CS_path_0001".

    Returns:
        List samples, mỗi sample là dict có keys: id, nganh, context, history, target, qa.
    """
    history_list = path["history"]
    samples: list[dict[str, Any]] = []

    for k in range(len(history_list) - 1):
        history = history_list[: k + 1]
        next_hk = history_list[k + 1]

        # Tính các môn đã đạt
        completed_codes = [
            c["ma_mon"] for hk in history for c in hk["courses"] if c["diem"] >= 4.0
        ]
        completed_codes = list(dict.fromkeys(completed_codes))  # dedup, giữ thứ tự

        last_gpa = history[-1]["gpa_sau_hk"]
        next_codes = [c["ma_mon"] for c in next_hk["courses"]]
        next_names = [c["ten_mon"] for c in next_hk["courses"]]

        # Tạo QA pair tự nhiên tiếng Việt
        career_str = path["career"] if path["career"] else "chưa xác định"
        question = (
            f"Em là sinh viên ngành {path['nganh']}, đã hoàn thành học kỳ {history[-1]['hk_so']} "
            f"với GPA tích luỹ {last_gpa:.2f} (thang 10). Định hướng nghề: {career_str}. "
            f"Em đã học {len(completed_codes)} môn. Học kỳ {next_hk['hk_so']} em nên đăng ký môn nào?"
        )
        answer_courses = ", ".join(f"{c['ma_mon']} ({c['ten_mon']})" for c in next_hk["courses"])
        answer = (
            f"Học kỳ {next_hk['hk_so']} bạn nên đăng ký {len(next_hk['courses'])} môn "
            f"({next_hk['total_tc']} TC): {answer_courses}. "
            f"Các môn này đã thoả điều kiện tiên quyết và phù hợp định hướng {career_str}."
        )

        sample = {
            "id": f"{sample_id_prefix}_ck{k + 1}",
            "source": "graph_path_sampling",
            "nganh": path["nganh"],
            "context": {
                "student_type": path["student_type"],
                "career": path["career"],
                "current_hk": history[-1]["hk_so"],
                "completed_codes": completed_codes,
                "gpa_thang10": last_gpa,
            },
            "history": history,
            "target": {
                "hk_so": next_hk["hk_so"],
                "course_codes": next_codes,
                "course_names": next_names,
                "total_tc": next_hk["total_tc"],
            },
            "qa": {"question": question, "answer": answer},
        }
        samples.append(sample)

    return samples


def generate_samples_for_nganh(
    nganh: str,
    n_paths: int,
    project_root: Path,
    regulations: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Sinh tất cả samples cho một ngành.

    Args:
        nganh: Mã ngành.
        n_paths: Số path mô phỏng (mỗi path → ~hk_max-1 samples).
        project_root: Thư mục gốc dự án.
        regulations: Dict regulations.json.
        seed: Seed cho RNG (reproducible).

    Returns:
        List tất cả training samples.
    """
    rng = random.Random(seed)
    curriculum_path = project_root / "data/processed/curriculum_graph" / f"{nganh}_curriculum.json"
    hk_map, course_map = load_curriculum(curriculum_path)

    tc_min = regulations["dang_ky_hoc_phan"]["tc_min"]
    tc_max = regulations["dang_ky_hoc_phan"]["tc_max"]

    all_samples: list[dict[str, Any]] = []
    for i in range(n_paths):
        paths = simulate_student_path(nganh, hk_map, course_map, tc_min, tc_max, rng)
        for j, p in enumerate(paths):
            prefix = f"{nganh}_path_{i:04d}_{j}"
            all_samples.extend(path_to_training_samples(p, prefix))

    return all_samples


def save_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Lưu samples ra file JSON Lines (.jsonl) — một sample/dòng.

    Args:
        samples: List training samples.
        output_path: Đường dẫn file .jsonl.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def validate_samples(samples: list[dict[str, Any]], graph: nx.DiGraph) -> dict[str, int]:
    """Validate ràng buộc của samples: prereq được thoả khi target courses xuất hiện.

    Args:
        samples: List training samples.
        graph: Prerequisite graph của ngành tương ứng (dùng chung 1 graph).

    Returns:
        Dict thống kê: total, violation_prereq, no_target.
    """
    stats = {"total": len(samples), "violation_prereq": 0, "no_target": 0}
    for s in samples:
        completed = set(s["context"]["completed_codes"])
        targets = s["target"]["course_codes"]
        target_set = set(targets)
        if not targets:
            stats["no_target"] += 1
            continue
        for code in targets:
            if code not in graph:
                continue
            for prereq in graph.predecessors(code):
                # Cho phép song_hanh: prereq nằm trong cùng HK target
                edge_type = graph.edges[prereq, code].get("loai_dieu_kien", "hoc_truoc")
                if edge_type == "song_hanh" and prereq in target_set:
                    continue
                if prereq not in completed:
                    stats["violation_prereq"] += 1
                    break
    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Graph-based path sampling cho RAG chatbot")
    p.add_argument("--nganh", choices=["CS", "IS", "DS", "SE", "IT"])
    p.add_argument("--all", action="store_true", help="Sinh cho cả 5 ngành")
    p.add_argument("--n_paths", type=int, default=800, help="Số path/ngành (mặc định 800)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/augmented/graph_paths"),
        help="Thư mục lưu .jsonl",
    )
    return p


def main() -> None:
    """Điểm vào CLI."""
    # Ép stdout dùng utf-8 để in tiếng Việt trên Windows console (cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[3]

    regulations_path = project_root / "data/processed/regulations.json"
    with regulations_path.open(encoding="utf-8") as f:
        regulations = json.load(f)

    nganh_list = ["CS", "IS", "DS", "SE", "IT"] if args.all else (
        [args.nganh] if args.nganh else []
    )
    if not nganh_list:
        _build_arg_parser().print_help()
        return

    output_dir = project_root / args.output_dir
    grand_total = 0
    for idx, nganh in enumerate(nganh_list):
        samples = generate_samples_for_nganh(
            nganh, args.n_paths, project_root, regulations,
            seed=args.seed + idx * 1000,
        )
        out_path = output_dir / f"{nganh}_graph_paths.jsonl"
        save_samples(samples, out_path)

        # Validate
        graph_path = project_root / "data/processed/curriculum_graph" / f"{nganh}_prereq_graph.gpickle"
        with graph_path.open("rb") as f:
            graph: nx.DiGraph = pickle.load(f)
        stats = validate_samples(samples, graph)
        viol_pct = (stats["violation_prereq"] / stats["total"] * 100) if stats["total"] else 0
        print(
            f"[{nganh}] {len(samples)} samples → {out_path.name} "
            f"(violation_prereq: {stats['violation_prereq']} = {viol_pct:.2f}%, "
            f"no_target: {stats['no_target']})"
        )
        grand_total += len(samples)

    print(f"\nTổng cộng: {grand_total} samples từ {len(nganh_list)} ngành.")


if __name__ == "__main__":
    main()
