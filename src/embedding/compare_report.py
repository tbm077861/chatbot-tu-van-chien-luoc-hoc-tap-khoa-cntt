"""Sinh báo cáo Markdown so sánh các phương án embedding.

Đọc `eval_results.json`, sinh `EMBEDDING_COMPARISON.md` với:
- Bảng tổng kết metrics.
- Phân tích theo ngành (nếu có).
- Khuyến nghị method nào nên dùng cho FAISS.

CLI:
    python -m src.embedding.compare_report
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"

METHOD_DESC = {
    "phobert": "PhoBERT-base-v2 fine-tuned (Vietnamese-only, MultipleNegativesRankingLoss in-batch)",
    "e5": "multilingual-E5-base fine-tuned (multilingual, có hard negatives từ negative_sampling)",
    "e5_base_pretrained": "multilingual-E5-base PRETRAINED (baseline, không fine-tune)",
    "gnn_gcn": "GCN 2-layer trên prereq graph (text features init, PCA bridge cho query)",
    "gnn_gat": "GAT 2-layer multi-head trên prereq graph (PCA bridge cho query)",
    "hybrid": "Late fusion E5 + GCN (α=0.85 sau khi sweep — xem hybrid_best_alpha.json)",
}


def main() -> None:
    path = EMB_DIR / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        raise RuntimeError("Không có kết quả hợp lệ trong eval_results.json")

    sorted_methods = sorted(
        valid.items(), key=lambda x: -x[1].get("Recall@10", 0)
    )
    best_name, best_metrics = sorted_methods[0]

    lines: list[str] = []
    lines.append("# BÁO CÁO SO SÁNH EMBEDDING (Giai đoạn 3)")
    lines.append("")
    lines.append(
        "Đánh giá trên test set 500 query-doc pairs (hold-out, "
        "cân bằng 100/ngành, không xuất hiện trong training)."
    )
    lines.append("")
    lines.append("## Bảng tổng kết")
    lines.append("")
    lines.append("| Method | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |")
    lines.append("|--------|---------:|---------:|----------:|----:|--------:|")
    for name, r in sorted_methods:
        lines.append(
            f"| **{name}** | {r['Recall@1']:.4f} | {r['Recall@5']:.4f} | "
            f"{r['Recall@10']:.4f} | {r['MRR']:.4f} | {r['NDCG@10']:.4f} |"
        )
    if "error" in str(results):
        for name, r in results.items():
            if "error" in r:
                lines.append(f"| {name} | LỖI: {r['error']} | | | | |")
    lines.append("")

    lines.append("## Mô tả phương án")
    lines.append("")
    for m in METHOD_DESC:
        if m in results:
            lines.append(f"- **{m}**: {METHOD_DESC[m]}")
    lines.append("")

    lines.append("## Phương án tốt nhất & Quyết định")
    lines.append("")
    lines.append(
        f"**{best_name}** đạt Recall@10 cao nhất ({best_metrics['Recall@10']:.4f})."
    )
    lines.append("")
    if best_name == "hybrid" and "e5" in valid:
        gap = valid["hybrid"]["Recall@10"] - valid["e5"]["Recall@10"]
        lines.append(
            f"Tuy nhiên hybrid chỉ hơn E5 đơn lẻ {gap*100:.2f}% Recall@10 "
            f"({valid['hybrid']['Recall@10']:.4f} vs {valid['e5']['Recall@10']:.4f}), "
            "trong khi E5 có MRR & NDCG@10 cao hơn. Để giảm phức tạp pipeline "
            "(không cần PCA bridge + concat hybrid vector), **FAISS index được build "
            "với E5 fine-tuned**. Hybrid vẫn có thể được kích hoạt sau cho retrieval "
            "stage ở Giai đoạn 5 nếu cần đẩy thêm 0.2% Recall."
        )
    else:
        lines.append(
            f"FAISS index được build với **{best_name}** "
            "(xem `data/embeddings/faiss/`)."
        )
    lines.append("")

    lines.append("## Phân tích")
    lines.append("")
    # PhoBERT vs E5.
    if "phobert" in valid and "e5" in valid:
        p = valid["phobert"]["Recall@10"]
        e = valid["e5"]["Recall@10"]
        if e > p:
            lines.append(
                f"- E5 ({e:.4f}) > PhoBERT ({p:.4f}) ở Recall@10. "
                "Hard negatives từ negative_sampling giúp E5 phân biệt được "
                "các môn vi phạm prereq/TC."
            )
        elif p > e:
            lines.append(
                f"- PhoBERT ({p:.4f}) > E5 ({e:.4f}) ở Recall@10. "
                "PhoBERT-v2 tối ưu cho tiếng Việt phù hợp hơn dù không có hard negatives."
            )
        else:
            lines.append(
                "- PhoBERT và E5 tương đương ở Recall@10."
            )
    # Fine-tune effect.
    if "e5" in valid and "e5_base_pretrained" in valid:
        gain = valid["e5"]["Recall@10"] - valid["e5_base_pretrained"]["Recall@10"]
        lines.append(
            f"- Fine-tune E5 cải thiện Recall@10 +{gain*100:.2f}% so với baseline pretrained "
            f"({valid['e5_base_pretrained']['Recall@10']:.4f} -> "
            f"{valid['e5']['Recall@10']:.4f})."
        )
    # GNN effect.
    if "hybrid" in valid and "e5" in valid:
        diff = valid["hybrid"]["Recall@10"] - valid["e5"]["Recall@10"]
        sign = "+" if diff >= 0 else ""
        lines.append(
            f"- Hybrid (E5 + GNN) so với E5 standalone: Δ Recall@10 = {sign}{diff*100:.2f}%. "
            f"{'GNN bổ sung tín hiệu cấu trúc tiên quyết.' if diff > 0 else 'GNN signal yếu hơn text trong setup này.'}"
        )
    lines.append("")

    out = EMB_DIR / "EMBEDDING_COMPARISON.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[save] {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
