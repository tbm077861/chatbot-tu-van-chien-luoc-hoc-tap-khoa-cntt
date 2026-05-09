"""Xây dựng và validate prerequisite graph (NetworkX) từ JSON curriculum.

Ví dụ sử dụng CLI:
    # Build graph cho một ngành
    python src/data/graph_builder.py --nganh CS

    # Build và validate tất cả 5 ngành
    python src/data/graph_builder.py --all

    # Output: data/processed/curriculum_graph/<NGANH>_prereq_graph.gpickle
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx

# Thư mục chứa JSON curriculum (output của parser.py)
CURRICULUM_DIR = Path("data/processed/curriculum_graph")

# Ánh xạ loại điều kiện → nhãn edge
LOAI_DIEU_KIEN_LABEL: dict[str, str] = {
    "a": "hoc_truoc",    # phải hoàn thành trước
    "b": "tien_quyet",   # tiên quyết bắt buộc
    "c": "song_hanh",    # học cùng lúc được
}


def build_graph(curriculum: dict[str, Any]) -> nx.DiGraph:
    """Xây dựng directed prerequisite graph từ dict curriculum đã parse.

    Node: mã môn (str, 6 chữ số). Attributes: ten_mon, nganh, hk_so, loai,
    so_tc, khong_tinh_gpa.
    Edge: (prereq → mon). Attributes: loai_dieu_kien ("hoc_truoc"/"tien_quyet"/"song_hanh").

    Args:
        curriculum: Dict JSON từ parser.py, khoá "nganh" và "hoc_ky".

    Returns:
        DiGraph NetworkX với node/edge attributes đầy đủ.
    """
    G = nx.DiGraph()
    nganh = curriculum["nganh"]

    for hk in curriculum["hoc_ky"]:
        hk_so = hk["hk_so"]
        for hp in hk["hoc_phan"]:
            ma_mon: str = hp["ma_mon"]

            # Thêm node với attributes
            G.add_node(
                ma_mon,
                ten_mon=hp["ten_mon"],
                nganh=nganh,
                hk_so=hk_so,
                loai=hp["loai"],
                so_tc=hp["so_tc"],
                khong_tinh_gpa=hp["khong_tinh_gpa"],
                ma_hoc_phan=hp.get("ma_hoc_phan"),
            )

            # Thêm edge từ prereq → môn hiện tại
            loai_nhan = LOAI_DIEU_KIEN_LABEL.get(
                hp.get("loai_dieu_kien") or "", "hoc_truoc"
            )
            for prereq_ma in hp.get("dieu_kien", []):
                G.add_edge(prereq_ma, ma_mon, loai_dieu_kien=loai_nhan)

    return G


def validate_graph(G: nx.DiGraph, nganh: str) -> list[str]:
    """Validate prerequisite graph: không có cycle, prereq node tồn tại.

    Args:
        G: DiGraph cần validate.
        nganh: Mã ngành (dùng trong thông báo lỗi).

    Returns:
        Danh sách chuỗi cảnh báo. List rỗng = không có lỗi.
    """
    warnings: list[str] = []

    # 1. Kiểm tra cycle
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            for cycle in cycles:
                warnings.append(f"[{nganh}] CYCLE phát hiện: {' → '.join(cycle)}")
    except nx.NetworkXError as e:
        warnings.append(f"[{nganh}] Lỗi khi kiểm tra cycle: {e}")

    # 2. Kiểm tra prereq node chưa được khai báo trong curriculum
    # (có edge vào nhưng node không có attribute 'ten_mon')
    for node in G.nodes():
        if "ten_mon" not in G.nodes[node]:
            in_edges = list(G.in_edges(node))
            out_edges = list(G.out_edges(node))
            warnings.append(
                f"[{nganh}] Prereq node '{node}' không có trong curriculum "
                f"(được tham chiếu bởi: {[e[1] for e in in_edges or out_edges]})"
            )

    return warnings


def save_graph(G: nx.DiGraph, output_path: Path) -> None:
    """Lưu graph ra file gpickle (nhị phân, nhanh load).

    Args:
        G: DiGraph cần lưu.
        output_path: Đường dẫn file .gpickle đích.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load graph từ file gpickle.

    Args:
        graph_path: Đường dẫn file .gpickle.

    Returns:
        DiGraph đã load.

    Raises:
        FileNotFoundError: File không tồn tại.
    """
    if not graph_path.exists():
        raise FileNotFoundError(f"Không tìm thấy graph file: {graph_path}")
    with graph_path.open("rb") as f:
        return pickle.load(f)


def build_and_save(
    nganh: str,
    curriculum_dir: Path,
    output_dir: Path,
) -> tuple[nx.DiGraph, list[str]]:
    """Load JSON curriculum, build graph, validate, và lưu file.

    Args:
        nganh: Mã ngành (CS/IS/DS/SE/IT).
        curriculum_dir: Thư mục chứa file JSON từ parser.py.
        output_dir: Thư mục lưu file .gpickle output.

    Returns:
        Tuple (graph, danh_sach_canh_bao).

    Raises:
        FileNotFoundError: Không tìm thấy file JSON curriculum.
    """
    json_path = curriculum_dir / f"{nganh}_curriculum.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {json_path}. Hãy chạy parser.py --nganh {nganh} trước."
        )

    with json_path.open(encoding="utf-8") as f:
        curriculum = json.load(f)

    G = build_graph(curriculum)
    warnings = validate_graph(G, nganh)

    out_path = output_dir / f"{nganh}_prereq_graph.gpickle"
    save_graph(G, out_path)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    status = "OK" if not warnings else f"{len(warnings)} CẢNH BÁO"
    print(
        f"[{nganh}] Graph: {n_nodes} nodes, {n_edges} edges → {out_path} [{status}]"
    )
    for w in warnings:
        print(f"  ⚠ {w}")

    return G, warnings


def print_graph_summary(G: nx.DiGraph, nganh: str) -> None:
    """In thống kê tóm tắt về graph: môn có nhiều prereq nhất, chuỗi dài nhất.

    Args:
        G: DiGraph prerequisite.
        nganh: Mã ngành (hiển thị trong header).
    """
    print(f"\n=== Tóm tắt graph [{nganh}] ===")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Môn có in-degree cao nhất (nhiều prereq nhất)
    in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)
    top5 = in_degrees[:5]
    print("  Môn nhiều prereq nhất:")
    for node, deg in top5:
        if deg > 0:
            ten = G.nodes[node].get("ten_mon", node)
            print(f"    {node} {ten}: {deg} prereq")

    # Môn nằm ở cuối chuỗi dài nhất (longest path)
    if nx.is_directed_acyclic_graph(G):
        try:
            longest = nx.dag_longest_path(G)
            print(f"  Chuỗi prereq dài nhất ({len(longest)} bước):")
            chain_names = [
                f"{m}({G.nodes[m].get('ten_mon', '?')[:20]})" for m in longest
            ]
            print(f"    {' → '.join(chain_names)}")
        except Exception:
            pass


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build prerequisite graph từ JSON curriculum")
    p.add_argument("--nganh", choices=["CS", "IS", "DS", "SE", "IT"])
    p.add_argument("--all", action="store_true", help="Build graph cho tất cả 5 ngành")
    p.add_argument(
        "--curriculum_dir",
        type=Path,
        default=CURRICULUM_DIR,
        help="Thư mục chứa JSON curriculum (output của parser.py)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=CURRICULUM_DIR,
        help="Thư mục lưu .gpickle graph (mặc định cùng curriculum_dir)",
    )
    p.add_argument("--summary", action="store_true", help="In thống kê tóm tắt sau khi build")
    return p


def main() -> None:
    """Điểm vào CLI."""
    args = _build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]

    curriculum_dir = project_root / args.curriculum_dir
    output_dir = project_root / args.output_dir

    nganh_list = ["CS", "IS", "DS", "SE", "IT"] if args.all else ([args.nganh] if args.nganh else [])
    if not nganh_list:
        _build_arg_parser().print_help()
        return

    all_warnings: list[str] = []
    for nganh in nganh_list:
        try:
            G, warnings = build_and_save(nganh, curriculum_dir, output_dir)
            all_warnings.extend(warnings)
            if args.summary:
                print_graph_summary(G, nganh)
        except FileNotFoundError as e:
            print(f"[{nganh}] LỖI: {e}")

    if all_warnings:
        print(f"\nTổng cộng {len(all_warnings)} cảnh báo cần xem lại.")
    else:
        print("\nTất cả graph hợp lệ (không có cycle, không thiếu node).")


if __name__ == "__main__":
    main()
