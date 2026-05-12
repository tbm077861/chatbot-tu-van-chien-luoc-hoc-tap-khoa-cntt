"""Constraint checker — kiểm tra prerequisite + giới hạn tín chỉ (Giai đoạn 5).

Sau khi retriever trả top-K candidate, layer này lọc/cảnh báo:
1. **Prerequisite (a/b)**: môn đề xuất có đủ điều kiện học trước/tiên quyết
   trong `da_hoan_thanh` chưa? (Loại "c" song hành → chỉ cần đang học cùng kỳ.)
2. **Giới hạn TC/HK**: tổng tín chỉ của tập đề xuất + môn bắt buộc của kỳ phải
   nằm trong [12, 30] (theo `regulations.json` IUH).
3. **Đã học rồi**: filter các môn sinh viên đã hoàn thành (trừ khi cải thiện).

Trả `ConstraintReport` chứa danh sách hợp lệ + danh sách vi phạm + cảnh báo TC.
Module này KHÔNG remove môn — pipeline RAG quyết định có hiển thị hay không
(constraint-aware reranking).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from src.data.graph_builder import load_graph

ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = ROOT / "data" / "processed" / "curriculum_graph"
REG_PATH = ROOT / "data" / "processed" / "regulations.json"

NGANH_LIST = ("CS", "IS", "DS", "SE", "IT")

# Loại điều kiện học phần (xem regulations.json).
# Theo project_instructions.md mục 2.1:
# a (hoc_truoc): phải học (chưa cần đạt) trước.
# b (tien_quyet): phải đạt trước.
# c (song_hanh): học cùng kỳ được.
PREREQ_KINDS_REQUIRED = ("hoc_truoc", "tien_quyet")


@dataclass
class ConstraintReport:
    """Kết quả kiểm tra ràng buộc cho 1 tập đề xuất."""

    valid: list[str] = field(default_factory=list)
    """Doc IDs vượt qua tất cả ràng buộc."""

    violations: list[dict] = field(default_factory=list)
    """Mỗi vi phạm: {doc_id, ma_mon, reason, missing_prereqs?}."""

    warnings: list[str] = field(default_factory=list)
    """Cảnh báo không chặn (ví dụ tổng TC ngoài khoảng)."""

    total_tc: int = 0
    """Tổng tín chỉ của tập `valid`."""

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "violations": self.violations,
            "warnings": self.warnings,
            "total_tc": self.total_tc,
        }


class ConstraintChecker:
    """Kiểm tra ràng buộc trên prerequisite graph + quy định IUH.

    Args:
        graphs: dict {nganh: nx.DiGraph}, load sẵn lúc khởi tạo.
        tc_min: TC tối thiểu/HK (12 theo quy định IUH).
        tc_max: TC tối đa/HK (30 theo quy định IUH).
    """

    def __init__(
        self,
        graphs: dict[str, nx.DiGraph],
        tc_min: int = 12,
        tc_max: int = 30,
    ) -> None:
        self.graphs = graphs
        self.tc_min = tc_min
        self.tc_max = tc_max

    @classmethod
    def load_default(cls) -> "ConstraintChecker":
        """Load 5 graph + quy định TC từ artifact mặc định."""
        graphs: dict[str, nx.DiGraph] = {}
        for nganh in NGANH_LIST:
            path = GRAPH_DIR / f"{nganh}_prereq_graph.gpickle"
            if path.exists():
                graphs[nganh] = load_graph(path)
        # TC limits theo regulations.
        tc_min, tc_max = 12, 30
        if REG_PATH.exists():
            with open(REG_PATH, encoding="utf-8") as f:
                reg = json.load(f)
            tc_min = int(reg.get("dang_ky_hoc_phan", {}).get("tc_min", tc_min))
            tc_max = int(reg.get("dang_ky_hoc_phan", {}).get("tc_max", tc_max))
        return cls(graphs, tc_min=tc_min, tc_max=tc_max)

    def _doc_to_ma(self, doc_id: str) -> tuple[str, str]:
        """Tách doc_id `CS_001782` → (`CS`, `001782`)."""
        if "_" not in doc_id:
            raise ValueError(f"doc_id sai format (cần `NGANH_MAMON`): {doc_id}")
        nganh, ma = doc_id.split("_", 1)
        return nganh, ma

    def get_prereqs(self, doc_id: str) -> dict[str, list[str]]:
        """Trả {kind: [prereq_ma, ...]} cho 1 môn.

        kind ∈ {"hoc_truoc", "tien_quyet", "song_hanh"}.
        """
        nganh, ma = self._doc_to_ma(doc_id)
        G = self.graphs.get(nganh)
        out: dict[str, list[str]] = {
            "hoc_truoc": [],
            "tien_quyet": [],
            "song_hanh": [],
        }
        if G is None or ma not in G:
            return out
        for prereq_ma, _, data in G.in_edges(ma, data=True):
            kind = data.get("loai_dieu_kien", "hoc_truoc")
            out.setdefault(kind, []).append(prereq_ma)
        return out

    def get_so_tc(self, doc_id: str) -> int:
        """Trả số tín chỉ. 0 nếu không tìm thấy môn trong graph."""
        nganh, ma = self._doc_to_ma(doc_id)
        G = self.graphs.get(nganh)
        if G is None or ma not in G:
            return 0
        return int(G.nodes[ma].get("so_tc", 0))

    def get_ten_mon(self, doc_id: str) -> str:
        """Trả tên môn cho tiện log/UI."""
        nganh, ma = self._doc_to_ma(doc_id)
        G = self.graphs.get(nganh)
        if G is None or ma not in G:
            return doc_id
        return str(G.nodes[ma].get("ten_mon", doc_id))

    def check_recommendations(
        self,
        candidates: list[str],
        nganh: str,
        completed: list[str] | None = None,
        in_progress: list[str] | None = None,
    ) -> ConstraintReport:
        """Lọc + cảnh báo danh sách đề xuất.

        Args:
            candidates: list doc_id (theo thứ tự retriever).
            nganh: ngành sinh viên (CS/IS/DS/SE/IT) — để xử lý mã môn-only input.
            completed: list mã môn đã hoàn thành (6 chữ số), dùng kiểm prereq.
            in_progress: list mã môn đang học cùng kỳ (cho điều kiện song hành).

        Returns:
            `ConstraintReport`. `valid` giữ thứ tự gốc.
        """
        completed_set = set(completed or [])
        in_progress_set = set(in_progress or [])
        report = ConstraintReport()
        seen: set[str] = set()  # tránh trùng

        for doc_id in candidates:
            if doc_id in seen:
                continue
            seen.add(doc_id)

            try:
                d_nganh, ma = self._doc_to_ma(doc_id)
            except ValueError as e:
                report.violations.append(
                    {"doc_id": doc_id, "reason": "doc_id_format", "detail": str(e)}
                )
                continue

            if d_nganh != nganh:
                # Cross-major suggestion — chưa hỗ trợ.
                report.violations.append(
                    {"doc_id": doc_id, "reason": "khac_nganh", "detail": d_nganh}
                )
                continue

            G = self.graphs.get(nganh)
            if G is None or ma not in G:
                report.violations.append(
                    {
                        "doc_id": doc_id,
                        "ma_mon": ma,
                        "reason": "khong_co_trong_curriculum",
                    }
                )
                continue

            # Đã hoàn thành → bỏ (trừ khi user muốn cải thiện, để pipeline quyết định).
            if ma in completed_set:
                report.violations.append(
                    {
                        "doc_id": doc_id,
                        "ma_mon": ma,
                        "ten_mon": self.get_ten_mon(doc_id),
                        "reason": "da_hoan_thanh",
                    }
                )
                continue

            # Kiểm prereq.
            prereqs = self.get_prereqs(doc_id)
            missing: list[str] = []
            for kind in PREREQ_KINDS_REQUIRED:
                for p in prereqs.get(kind, []):
                    if p not in completed_set:
                        missing.append(p)
            # song_hanh: cần đang học hoặc đã học.
            for p in prereqs.get("song_hanh", []):
                if p not in completed_set and p not in in_progress_set:
                    missing.append(p)

            if missing:
                report.violations.append(
                    {
                        "doc_id": doc_id,
                        "ma_mon": ma,
                        "ten_mon": self.get_ten_mon(doc_id),
                        "reason": "thieu_prereq",
                        "missing_prereqs": sorted(set(missing)),
                    }
                )
                continue

            report.valid.append(doc_id)
            report.total_tc += self.get_so_tc(doc_id)

        # Cảnh báo TC.
        if report.total_tc < self.tc_min:
            report.warnings.append(
                f"Tổng TC = {report.total_tc} < {self.tc_min} (TC tối thiểu/HK). "
                "Cân nhắc thêm môn để tránh cảnh báo học tập."
            )
        elif report.total_tc > self.tc_max:
            report.warnings.append(
                f"Tổng TC = {report.total_tc} > {self.tc_max} (TC tối đa/HK). "
                "Cần bớt môn để tuân thủ quy chế."
            )

        return report
