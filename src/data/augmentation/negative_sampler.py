"""Constraint-based negative sampling — sinh kịch bản vi phạm ràng buộc.

Dùng cho:
- Hard negative trong contrastive learning của embedding model.
- Training data dạy LLM nhận diện và cảnh báo lỗi đăng ký.

5 loại vi phạm được sinh:
1. vi_pham_prereq — đăng ký môn có prereq chưa hoàn thành.
2. vuot_tc_max — tổng TC > 30 (max theo regulations).
3. duoi_tc_min — tổng TC < 12 (min theo regulations, không phải HK cuối).
4. sai_thu_tu_hk — đăng ký môn HK xa trước môn HK gần.
5. lap_lai_diem_cao — học lại môn đã đạt với điểm cao (A/A+/B+).

Mỗi sample là một QA pair: câu hỏi chứa kế hoạch sai → câu trả lời giải thích vi phạm.

Ví dụ CLI:
    python src/data/augmentation/negative_sampler.py --all --n_per_type 600
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import networkx as nx

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_curriculum_and_graph(
    project_root: Path, nganh: str
) -> tuple[dict[str, Any], nx.DiGraph, dict[str, dict[str, Any]]]:
    """Load curriculum JSON + prereq graph + course map.

    Args:
        project_root: Thư mục gốc.
        nganh: Mã ngành.

    Returns:
        Tuple (curriculum_dict, graph, course_map).
    """
    json_path = project_root / "data/processed/curriculum_graph" / f"{nganh}_curriculum.json"
    graph_path = project_root / "data/processed/curriculum_graph" / f"{nganh}_prereq_graph.gpickle"

    with json_path.open(encoding="utf-8") as f:
        curriculum = json.load(f)
    with graph_path.open("rb") as f:
        graph: nx.DiGraph = pickle.load(f)

    course_map: dict[str, dict[str, Any]] = {}
    for hk in curriculum["hoc_ky"]:
        for hp in hk["hoc_phan"]:
            course_map[hp["ma_mon"]] = {
                **hp,
                "hk_so": hk["hk_so"],
            }
    return curriculum, graph, course_map


def courses_at_hk(course_map: dict[str, dict[str, Any]], hk: int) -> list[dict[str, Any]]:
    """Lấy tất cả môn ở một HK cụ thể."""
    return [c for c in course_map.values() if c["hk_so"] == hk]


def courses_up_to_hk(course_map: dict[str, dict[str, Any]], hk: int) -> list[dict[str, Any]]:
    """Lấy tất cả môn từ HK1 đến HK k."""
    return [c for c in course_map.values() if 1 <= c["hk_so"] <= hk]


def gen_vi_pham_prereq(
    nganh: str,
    graph: nx.DiGraph,
    course_map: dict[str, dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sinh kịch bản vi phạm prereq: đăng ký môn nhưng thiếu prereq.

    Args:
        nganh: Mã ngành.
        graph: Prereq graph.
        course_map: Map ma_mon → info.
        n: Số sample cần sinh.
        rng: random.Random.

    Returns:
        List samples.
    """
    # Chỉ lấy môn có ít nhất 1 prereq
    courses_with_prereq = [c for c in graph.nodes() if list(graph.predecessors(c))]
    samples: list[dict[str, Any]] = []
    attempts = 0
    while len(samples) < n and attempts < n * 5:
        attempts += 1
        target_code = rng.choice(courses_with_prereq)
        if target_code not in course_map:
            continue
        target = course_map[target_code]
        prereqs = list(graph.predecessors(target_code))

        # Chọn ngẫu nhiên prereq bị thiếu (1-2 môn)
        n_missing = rng.randint(1, min(2, len(prereqs)))
        missing = rng.sample(prereqs, k=n_missing)
        present = [p for p in prereqs if p not in missing]

        # Tạo completed = các môn HK trước hk_target, trừ missing
        target_hk = target["hk_so"]
        all_prev = [c["ma_mon"] for c in courses_up_to_hk(course_map, target_hk - 1)]
        completed = [c for c in all_prev if c not in missing] + present
        completed = list(dict.fromkeys(completed))

        missing_names = [course_map.get(m, {}).get("ten_mon", m) for m in missing]
        question = (
            f"Em ngành {nganh}, đang ở HK{target_hk}, đã hoàn thành {len(completed)} môn. "
            f"Em muốn đăng ký môn {target_code} ({target['ten_mon']}) trong HK{target_hk} này. "
            f"Có hợp lệ không?"
        )
        answer = (
            f"KHÔNG hợp lệ. Môn {target_code} ({target['ten_mon']}) yêu cầu hoàn thành các môn "
            f"tiên quyết: {', '.join(missing)} ({', '.join(missing_names)}). "
            f"Bạn chưa học/đạt các môn này, do đó không thể đăng ký. "
            f"Hãy đăng ký các môn còn thiếu trước."
        )

        samples.append({
            "id": f"{nganh}_neg_prereq_{len(samples):05d}",
            "source": "negative_sampling",
            "violation_type": "vi_pham_prereq",
            "nganh": nganh,
            "context": {
                "completed_codes": completed,
                "current_hk": target_hk,
            },
            "invalid_target": {
                "course_code": target_code,
                "course_name": target["ten_mon"],
                "hk_so": target_hk,
                "missing_prereqs": missing,
            },
            "qa": {"question": question, "answer": answer},
        })

    return samples


def gen_vuot_tc_max(
    nganh: str,
    course_map: dict[str, dict[str, Any]],
    tc_max: int,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sinh kịch bản vượt giới hạn TC tối đa.

    Args:
        nganh: Mã ngành.
        course_map: Map ma_mon → info.
        tc_max: TC tối đa cho phép.
        n: Số sample.
        rng: random.Random.

    Returns:
        List samples.
    """
    samples: list[dict[str, Any]] = []
    all_codes = list(course_map.keys())
    for i in range(n):
        # Chọn ngẫu nhiên HK
        hk = rng.randint(2, 8)
        # Lấy đủ môn để tổng TC > tc_max
        selected: list[dict[str, Any]] = []
        total_tc = 0
        candidates = rng.sample(all_codes, k=min(20, len(all_codes)))
        for code in candidates:
            c = course_map[code]
            selected.append(c)
            total_tc += c["so_tc"]
            if total_tc > tc_max + 3:
                break
        if total_tc <= tc_max:
            continue

        codes_list = [c["ma_mon"] for c in selected]
        names_list = [c["ten_mon"] for c in selected]
        question = (
            f"Em ngành {nganh}, dự kiến đăng ký HK{hk} các môn sau: "
            + ", ".join(f"{c['ma_mon']} ({c['ten_mon']}, {c['so_tc']} TC)" for c in selected)
            + f". Tổng cộng {total_tc} TC. Có hợp lệ không?"
        )
        answer = (
            f"KHÔNG hợp lệ. Quy định học vụ IUH giới hạn tối đa {tc_max} TC/HK, "
            f"nhưng kế hoạch này có {total_tc} TC (vượt {total_tc - tc_max} TC). "
            f"Hãy giảm bớt số môn để tổng TC ≤ {tc_max}."
        )
        samples.append({
            "id": f"{nganh}_neg_tcmax_{i:05d}",
            "source": "negative_sampling",
            "violation_type": "vuot_tc_max",
            "nganh": nganh,
            "context": {"current_hk": hk},
            "invalid_target": {
                "course_codes": codes_list,
                "course_names": names_list,
                "total_tc": total_tc,
                "tc_max": tc_max,
            },
            "qa": {"question": question, "answer": answer},
        })

    return samples


def gen_duoi_tc_min(
    nganh: str,
    course_map: dict[str, dict[str, Any]],
    tc_min: int,
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sinh kịch bản dưới giới hạn TC tối thiểu.

    Args:
        nganh: Mã ngành.
        course_map: Map ma_mon → info.
        tc_min: TC tối thiểu.
        n: Số sample.
        rng: random.Random.

    Returns:
        List samples.
    """
    samples: list[dict[str, Any]] = []
    all_codes = list(course_map.keys())
    for i in range(n):
        hk = rng.randint(2, 7)  # Không phải HK cuối (HK8/9 có thể ít TC hợp lệ)
        n_courses = rng.randint(1, 3)
        selected = [course_map[c] for c in rng.sample(all_codes, k=n_courses)]
        total_tc = sum(c["so_tc"] for c in selected)
        if total_tc >= tc_min:
            continue

        question = (
            f"Em ngành {nganh}, dự kiến đăng ký HK{hk} chỉ {len(selected)} môn: "
            + ", ".join(f"{c['ma_mon']} ({c['ten_mon']}, {c['so_tc']} TC)" for c in selected)
            + f". Tổng {total_tc} TC. Em không phải năm cuối. Có hợp lệ không?"
        )
        answer = (
            f"KHÔNG hợp lệ. Quy định học vụ IUH yêu cầu tối thiểu {tc_min} TC/HK đối với "
            f"sinh viên không phải HK cuối, nhưng kế hoạch này chỉ có {total_tc} TC "
            f"(thiếu {tc_min - total_tc} TC). Hãy đăng ký thêm môn để đạt ≥{tc_min} TC."
        )
        samples.append({
            "id": f"{nganh}_neg_tcmin_{i:05d}",
            "source": "negative_sampling",
            "violation_type": "duoi_tc_min",
            "nganh": nganh,
            "context": {"current_hk": hk},
            "invalid_target": {
                "course_codes": [c["ma_mon"] for c in selected],
                "course_names": [c["ten_mon"] for c in selected],
                "total_tc": total_tc,
                "tc_min": tc_min,
            },
            "qa": {"question": question, "answer": answer},
        })

    return samples


def gen_sai_thu_tu_hk(
    nganh: str,
    course_map: dict[str, dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sinh kịch bản đăng ký môn HK xa trước môn HK gần (thiếu nền tảng).

    Args:
        nganh: Mã ngành.
        course_map: Map ma_mon → info.
        n: Số sample.
        rng: random.Random.

    Returns:
        List samples.
    """
    samples: list[dict[str, Any]] = []
    late_courses = [c for c in course_map.values() if c["hk_so"] >= 7]
    early_courses = [c for c in course_map.values() if c["hk_so"] <= 3]
    if not late_courses or not early_courses:
        return samples

    for i in range(n):
        target = rng.choice(late_courses)
        # SV ở HK2 hoặc HK3 mà muốn học môn HK7+
        current_hk = rng.randint(2, 3)
        completed_n = rng.randint(5, 10)
        completed = rng.sample([c["ma_mon"] for c in early_courses], k=min(completed_n, len(early_courses)))

        question = (
            f"Em ngành {nganh} đang ở HK{current_hk}, mới học {len(completed)} môn. "
            f"Em muốn đăng ký môn {target['ma_mon']} ({target['ten_mon']}) — "
            f"thuộc HK{target['hk_so']} trong chương trình khung. Có nên không?"
        )
        answer = (
            f"KHÔNG nên. Môn {target['ma_mon']} ({target['ten_mon']}) thuộc HK{target['hk_so']} "
            f"theo lộ trình chuẩn, đòi hỏi nhiều môn nền tảng các HK trước. "
            f"Bạn mới ở HK{current_hk}, học vượt sẽ thiếu kiến thức tiền đề và rủi ro thi trượt. "
            f"Hãy theo đúng lộ trình HK của chương trình khung."
        )
        samples.append({
            "id": f"{nganh}_neg_thutu_{i:05d}",
            "source": "negative_sampling",
            "violation_type": "sai_thu_tu_hk",
            "nganh": nganh,
            "context": {
                "current_hk": current_hk,
                "completed_codes": completed,
            },
            "invalid_target": {
                "course_code": target["ma_mon"],
                "course_name": target["ten_mon"],
                "target_hk": target["hk_so"],
            },
            "qa": {"question": question, "answer": answer},
        })

    return samples


def gen_lap_lai_diem_cao(
    nganh: str,
    course_map: dict[str, dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sinh kịch bản học lại môn đã đạt điểm cao (vô lý theo quy định cải thiện).

    Args:
        nganh: Mã ngành.
        course_map: Map ma_mon → info.
        n: Số sample.
        rng: random.Random.

    Returns:
        List samples.
    """
    samples: list[dict[str, Any]] = []
    all_codes = list(course_map.keys())
    for i in range(n):
        target = course_map[rng.choice(all_codes)]
        old_grade = round(rng.uniform(8.5, 10.0), 1)  # A hoặc A+
        diem_chu = "A+" if old_grade >= 9.0 else "A"

        question = (
            f"Em ngành {nganh} đã học môn {target['ma_mon']} ({target['ten_mon']}) "
            f"và đạt điểm {old_grade} ({diem_chu}). Em muốn học lại môn này để cải thiện. "
            f"Có hợp lý không?"
        )
        answer = (
            f"KHÔNG hợp lý. Theo quy chế IUH, sinh viên chỉ nên học lại để cải thiện khi "
            f"điểm thấp (D, D+, C, C+, B). Điểm {diem_chu} ({old_grade}) của bạn đã ở mức tối "
            f"ưu — học lại sẽ tốn thời gian, học phí mà không có lợi ích đáng kể. "
            f"Hãy dành thời gian cho môn khó hoặc môn còn thiếu."
        )
        samples.append({
            "id": f"{nganh}_neg_laplai_{i:05d}",
            "source": "negative_sampling",
            "violation_type": "lap_lai_diem_cao",
            "nganh": nganh,
            "context": {
                "course_code": target["ma_mon"],
                "old_grade": old_grade,
                "diem_chu": diem_chu,
            },
            "invalid_target": {
                "action": "hoc_lai_cai_thien",
                "course_code": target["ma_mon"],
                "course_name": target["ten_mon"],
            },
            "qa": {"question": question, "answer": answer},
        })

    return samples


def generate_for_nganh(
    nganh: str,
    n_per_type: int,
    project_root: Path,
    regulations: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Sinh toàn bộ 5 loại negative samples cho một ngành.

    Args:
        nganh: Mã ngành.
        n_per_type: Số sample/loại.
        project_root: Thư mục gốc.
        regulations: Dict regulations.
        seed: Random seed.

    Returns:
        List samples kết hợp 5 loại.
    """
    rng = random.Random(seed)
    _, graph, course_map = load_curriculum_and_graph(project_root, nganh)

    tc_min = regulations["dang_ky_hoc_phan"]["tc_min"]
    tc_max = regulations["dang_ky_hoc_phan"]["tc_max"]

    samples: list[dict[str, Any]] = []
    samples.extend(gen_vi_pham_prereq(nganh, graph, course_map, n_per_type, rng))
    samples.extend(gen_vuot_tc_max(nganh, course_map, tc_max, n_per_type, rng))
    samples.extend(gen_duoi_tc_min(nganh, course_map, tc_min, n_per_type, rng))
    samples.extend(gen_sai_thu_tu_hk(nganh, course_map, n_per_type, rng))
    samples.extend(gen_lap_lai_diem_cao(nganh, course_map, n_per_type, rng))

    return samples


def save_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Lưu samples ra JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Constraint-based negative sampling")
    p.add_argument("--nganh", choices=["CS", "IS", "DS", "SE", "IT"])
    p.add_argument("--all", action="store_true")
    p.add_argument(
        "--n_per_type", type=int, default=600,
        help="Số sample mỗi loại vi phạm/ngành (5 loại × 5 ngành = tổng 15k với 600)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=Path, default=Path("data/augmented/negative_samples"))
    return p


def main() -> None:
    """Điểm vào CLI."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _build_arg_parser().parse_args()
    project_root = _PROJECT_ROOT

    with (project_root / "data/processed/regulations.json").open(encoding="utf-8") as f:
        regulations = json.load(f)

    nganh_list = ["CS", "IS", "DS", "SE", "IT"] if args.all else (
        [args.nganh] if args.nganh else []
    )
    if not nganh_list:
        _build_arg_parser().print_help()
        return

    output_dir = project_root / args.output_dir
    grand_total = 0
    type_counts: dict[str, int] = {}
    for idx, nganh in enumerate(nganh_list):
        samples = generate_for_nganh(
            nganh, args.n_per_type, project_root, regulations,
            seed=args.seed + idx * 1000,
        )
        out_path = output_dir / f"{nganh}_negative.jsonl"
        save_samples(samples, out_path)
        for s in samples:
            type_counts[s["violation_type"]] = type_counts.get(s["violation_type"], 0) + 1
        print(f"[{nganh}] {len(samples)} samples → {out_path.name}")
        grand_total += len(samples)

    print(f"\nTổng cộng: {grand_total} negative samples từ {len(nganh_list)} ngành.")
    print("Phân bố theo loại vi phạm:")
    for t, c in sorted(type_counts.items()):
        print(f"  - {t}: {c}")


if __name__ == "__main__":
    main()
