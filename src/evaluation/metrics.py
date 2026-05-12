"""Metrics dùng chung cho evaluation Giai đoạn 5+ (RAG pipeline).

- `compute_retrieval_metrics`: Recall@K, MRR, NDCG@10 từ list predictions.
- `compute_constraint_metrics`: Constraint Satisfaction Rate + Credit Load Validity.

Khác với `src/embedding/evaluator.py` (làm việc với score matrix), file này
nhận list dự đoán đã sắp xếp — phù hợp với output của LLM generator (string parse).
"""

from __future__ import annotations

import math


def compute_retrieval_metrics(
    predictions: list[list[str]],
    gold: list[list[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Tính Recall@K, MRR, NDCG@10 từ predictions + gold.

    Args:
        predictions: list[list[doc_id]], mỗi sample list các doc đề xuất sắp
            xếp theo độ ưu tiên.
        gold: list[list[doc_id]], mỗi sample list các doc đúng (positive).
        ks: tuple K cho Recall@K.

    Returns:
        Dict {Recall@1, Recall@5, ..., MRR, NDCG@10}.
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"len(predictions)={len(predictions)} ≠ len(gold)={len(gold)}"
        )
    n = len(predictions)
    if n == 0:
        return {f"Recall@{k}": 0.0 for k in ks} | {"MRR": 0.0, "NDCG@10": 0.0}

    recall = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    for pred, g in zip(predictions, gold):
        gold_set = set(g)
        if not gold_set:
            continue
        for k in ks:
            if set(pred[:k]) & gold_set:
                recall[k] += 1
        rank = next((i + 1 for i, d in enumerate(pred) if d in gold_set), None)
        if rank is not None:
            mrr_sum += 1.0 / rank
        dcg = sum(
            1.0 / math.log2(i + 2) for i, d in enumerate(pred[:10]) if d in gold_set
        )
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), 10)))
        if idcg > 0:
            ndcg_sum += dcg / idcg

    out: dict[str, float] = {f"Recall@{k}": recall[k] / n for k in ks}
    out["MRR"] = mrr_sum / n
    out["NDCG@10"] = ndcg_sum / n
    return out


def compute_constraint_metrics(reports: list) -> dict[str, float]:
    """Tính tỷ lệ vi phạm + tín chỉ hợp lệ trên list ConstraintReport.

    Args:
        reports: list `ConstraintReport`.

    Returns:
        Dict {
          constraint_satisfaction_rate,  # % sample không có violation
          credit_load_validity,          # % sample có total_tc ∈ [tc_min, tc_max]
          avg_valid_count,
          avg_violation_count,
          avg_total_tc,
        }
    """
    n = len(reports)
    if n == 0:
        return {
            "constraint_satisfaction_rate": 0.0,
            "credit_load_validity": 0.0,
            "avg_valid_count": 0.0,
            "avg_violation_count": 0.0,
            "avg_total_tc": 0.0,
        }
    no_violation = sum(1 for r in reports if not r.violations)
    no_warning = sum(1 for r in reports if not r.warnings)
    avg_valid = sum(len(r.valid) for r in reports) / n
    avg_viol = sum(len(r.violations) for r in reports) / n
    avg_tc = sum(r.total_tc for r in reports) / n
    return {
        "constraint_satisfaction_rate": no_violation / n,
        "credit_load_validity": no_warning / n,
        "avg_valid_count": avg_valid,
        "avg_violation_count": avg_viol,
        "avg_total_tc": avg_tc,
    }
