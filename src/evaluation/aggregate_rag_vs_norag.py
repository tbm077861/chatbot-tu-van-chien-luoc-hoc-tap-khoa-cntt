"""Tổng hợp metrics RAG vs noRAG từ predictions Kaggle (Giai đoạn 6).

Đọc 2 file JSONL (`predictions_rag.jsonl`, `predictions_norag.jsonl`) sinh
bởi `notebooks/kaggle_m5_rag_eval.py`, tính Hit@K / MRR / NDCG@10 cho cả 2
variant và in bảng so sánh.

Cũng kiểm tra alignment idx + gold giữa 2 file để chắc đang so cùng 1 tập câu hỏi.

Ví dụ dùng CLI:
    python -m src.evaluation.aggregate_rag_vs_norag \\
        --rag data/kaggle_export/predictions_rag.jsonl \\
        --norag data/kaggle_export/predictions_norag.jsonl \\
        --output data/evaluation/rag_vs_norag_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import compute_retrieval_metrics


def load_predictions(path: Path) -> list[dict]:
    """Đọc file JSONL predictions sinh bởi Kaggle notebook."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def aggregate(
    rag_path: Path,
    norag_path: Path,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """So sánh metrics RAG vs noRAG trên cùng tập query.

    Returns:
        Dict gồm metrics 2 variant + delta + thống kê thô.
    """
    rag = load_predictions(rag_path)
    norag = load_predictions(norag_path)

    if len(rag) != len(norag):
        raise ValueError(
            f"Số dòng RAG={len(rag)} ≠ noRAG={len(norag)} — "
            "2 file phải sinh từ cùng input set."
        )
    for r, n in zip(rag, norag):
        if r["idx"] != n["idx"]:
            raise ValueError(f"idx mismatch: rag={r['idx']} noRAG={n['idx']}")
        if r["gold"] != n["gold"]:
            raise ValueError(f"gold mismatch tại idx={r['idx']}")

    preds_rag = [r["predicted_doc_ids"] for r in rag]
    preds_norag = [r["predicted_doc_ids"] for r in norag]
    gold = [r["gold"] for r in rag]  # cùng gold ở 2 variant

    metrics_rag = compute_retrieval_metrics(preds_rag, gold, ks=ks)
    metrics_norag = compute_retrieval_metrics(preds_norag, gold, ks=ks)

    delta = {
        k: metrics_rag[k] - metrics_norag[k]
        for k in metrics_rag
        if k in metrics_norag
    }

    pred_stats_rag = _pred_length_stats(preds_rag)
    pred_stats_norag = _pred_length_stats(preds_norag)

    return {
        "n_queries": len(rag),
        "metrics": {"rag": metrics_rag, "norag": metrics_norag},
        "delta_rag_minus_norag": delta,
        "pred_length_stats": {"rag": pred_stats_rag, "norag": pred_stats_norag},
    }


def _pred_length_stats(preds: list[list[str]]) -> dict[str, float]:
    """Thống kê độ dài predictions (để debug parse regex / generation behavior)."""
    lens = [len(p) for p in preds]
    n = max(1, len(lens))
    return {
        "avg": sum(lens) / n,
        "empty": sum(1 for x in lens if x == 0),
        "ge_1": sum(1 for x in lens if x >= 1),
        "ge_5": sum(1 for x in lens if x >= 5),
    }


def format_table(result: dict) -> str:
    """Format bảng markdown so sánh RAG vs noRAG."""
    m_rag = result["metrics"]["rag"]
    m_norag = result["metrics"]["norag"]
    delta = result["delta_rag_minus_norag"]
    keys = [k for k in m_rag if k in m_norag]

    lines = [
        f"| Metric | RAG | noRAG | Δ (RAG − noRAG) |",
        f"|---|---:|---:|---:|",
    ]
    for k in keys:
        lines.append(f"| {k} | {m_rag[k]:.4f} | {m_norag[k]:.4f} | {delta[k]:+.4f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rag",
        type=Path,
        required=True,
        help="Đường dẫn predictions_rag.jsonl từ Kaggle.",
    )
    parser.add_argument(
        "--norag",
        type=Path,
        required=True,
        help="Đường dẫn predictions_norag.jsonl từ Kaggle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/rag_vs_norag_metrics.json"),
        help="File JSON để ghi kết quả tổng hợp.",
    )
    args = parser.parse_args()

    result = aggregate(args.rag, args.norag)

    print(f"n_queries: {result['n_queries']}\n")
    print(format_table(result))
    print("\nPred length stats:")
    for variant in ("rag", "norag"):
        s = result["pred_length_stats"][variant]
        print(
            f"  {variant}: avg={s['avg']:.2f}  empty={s['empty']}  "
            f"≥1={s['ge_1']}  ≥5={s['ge_5']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {args.output}")


if __name__ == "__main__":
    main()
