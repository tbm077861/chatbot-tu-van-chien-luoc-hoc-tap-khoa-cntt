"""RAGAS-proxy eval (Giai đoạn 6) — không dùng API LLM.

Vì project yêu cầu không gọi API LLM (chi tiết trong `STATUS.md`), RAGAS gốc
(cần LLM judge cho faithfulness/answer relevancy) không khả thi trực tiếp.
File này tính 4 metric **xấp xỉ** dựa trên doc_id matching + cosine similarity
với E5 fine-tuned (đã có sẵn từ Giai đoạn 3).

| Metric | Định nghĩa proxy |
|---|---|
| `context_recall`     | |gold ∩ retrieved_valid| / |gold| — đúng nghĩa gốc khi gold là doc_id, không cần LLM |
| `context_precision`  | Trung bình precision tại các rank chứa doc gold (AP@K, K=len(context_doc_ids)) |
| `faithfulness`       | |predicted ∩ context| / |predicted| — % môn được khuyên có trong context cung cấp |
| `answer_relevancy`   | cosine(E5(query), E5(response)) — proxy cho việc câu trả lời có bám sát câu hỏi |

Optional: nếu user sau này có Qwen judge (Kaggle), có thể swap metric
faithfulness/relevancy bằng phiên bản LLM-based. Hiện tại proxy đủ để so
sánh tương đối RAG vs noRAG.

Ví dụ dùng:
    python -m src.evaluation.ragas_eval \\
        --rag data/kaggle_export/predictions_rag.jsonl \\
        --norag data/kaggle_export/predictions_norag.jsonl \\
        --e5 data/embeddings/e5_finetuned \\
        --output data/evaluation/ragas_proxy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def context_recall(gold: Sequence[str], retrieved: Sequence[str]) -> float:
    """|gold ∩ retrieved| / |gold|."""
    g = set(gold)
    if not g:
        return 0.0
    return len(g & set(retrieved)) / len(g)


def context_precision_ap(gold: Sequence[str], retrieved: Sequence[str]) -> float:
    """Average Precision tại các rank chứa doc gold (analog cho RAGAS context_precision).

    Tương đương MAP cho 1 query. K = len(retrieved). Nếu retrieved rỗng → 0.
    """
    g = set(gold)
    if not g or not retrieved:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in g:
            hits += 1
            precision_sum += hits / i
    return precision_sum / min(len(g), len(retrieved)) if hits else 0.0


def faithfulness_proxy(predicted: Sequence[str], context: Sequence[str]) -> float:
    """% predicted doc_id có mặt trong context cung cấp.

    Trả 1.0 nếu predicted rỗng (không có claim nào → không có gì để contradict).
    Trả 0.0 nếu context rỗng nhưng predicted có (mọi claim đều unsupported).
    """
    if not predicted:
        return 1.0
    if not context:
        return 0.0
    return len(set(predicted) & set(context)) / len(set(predicted))


def answer_relevancy_proxy(
    query: str,
    response: str,
    embedder,  # SentenceTransformer (lazy import)
) -> float:
    """cosine(E5(query), E5(response)).

    Dùng E5 cần prefix 'query: ' (theo convention E5 family).
    """
    import torch  # local import để file dùng được ngay cả khi chưa cài torch

    if not response.strip():
        return 0.0
    embs = embedder.encode(
        [f"query: {query}", f"query: {response}"],
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    sim = (embs[0] * embs[1]).sum().item()
    return float(max(0.0, sim))


def evaluate_records(
    records: list[dict],
    embedder=None,
    response_field: str = "response",
) -> dict:
    """Tính 4 metric proxy trên list records.

    Args:
        records: list dict đọc từ predictions_{rag,norag}.jsonl.
        embedder: SentenceTransformer instance (E5). Nếu None → bỏ qua
            answer_relevancy.
        response_field: tên field chứa text trả lời (mặc định 'response').

    Returns:
        Dict {context_recall, context_precision, faithfulness,
               answer_relevancy?, n}.
    """
    n = len(records)
    if n == 0:
        return {
            "context_recall": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "n": 0,
        }

    cr = cp = fa = ar = 0.0
    ar_count = 0
    for r in records:
        gold = r.get("gold", [])
        retrieved = r.get("retrieved_valid", [])
        context = r.get("context_doc_ids", []) or retrieved
        predicted = r.get("predicted_doc_ids", [])

        cr += context_recall(gold, retrieved)
        cp += context_precision_ap(gold, retrieved)
        fa += faithfulness_proxy(predicted, context)

        if embedder is not None:
            ar += answer_relevancy_proxy(
                r.get("query", ""), r.get(response_field, ""), embedder
            )
            ar_count += 1

    out: dict = {
        "context_recall": cr / n,
        "context_precision": cp / n,
        "faithfulness": fa / n,
        "n": n,
    }
    if ar_count > 0:
        out["answer_relevancy"] = ar / ar_count
    return out


def load_e5(model_dir: Path):
    """Load E5 fine-tuned (lazy import sentence-transformers)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_dir))


def format_table(rag_m: dict, norag_m: dict) -> str:
    keys = ("context_recall", "context_precision", "faithfulness", "answer_relevancy")
    lines = [
        "| Metric | RAG | noRAG | Delta (RAG - noRAG) |",
        "|---|---:|---:|---:|",
    ]
    for k in keys:
        if k not in rag_m or k not in norag_m:
            continue
        delta = rag_m[k] - norag_m[k]
        lines.append(f"| {k} | {rag_m[k]:.4f} | {norag_m[k]:.4f} | {delta:+.4f} |")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag", type=Path, required=True)
    parser.add_argument("--norag", type=Path, required=True)
    parser.add_argument(
        "--e5",
        type=str,
        default="data/embeddings/e5_finetuned",
        help="Folder E5 fine-tuned. Truyền '' (chuỗi rỗng) để bỏ qua answer_relevancy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/ragas_proxy.json"),
    )
    args = parser.parse_args()

    rag = load_jsonl(args.rag)
    norag = load_jsonl(args.norag)

    embedder = None
    e5_path = Path(args.e5) if args.e5 else None
    if e5_path and e5_path.exists() and (e5_path / "config.json").exists():
        print(f"Loading E5 từ {e5_path} ...")
        embedder = load_e5(e5_path)
        print("E5 loaded.")
    else:
        print("Bỏ qua answer_relevancy (không có E5 model hợp lệ).")

    rag_m = evaluate_records(rag, embedder)
    norag_m = evaluate_records(norag, embedder)

    print(f"\nn_queries: {rag_m['n']}\n")
    print(format_table(rag_m, norag_m))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "note": (
                    "RAGAS-proxy (no LLM judge). context_recall/precision dựa "
                    "doc_id; faithfulness = |pred ∩ context|/|pred|; "
                    "answer_relevancy = cos(E5(query), E5(response))."
                ),
                "rag": rag_m,
                "norag": norag_m,
                "delta_rag_minus_norag": {
                    k: rag_m[k] - norag_m[k]
                    for k in rag_m
                    if k in norag_m and isinstance(rag_m[k], (int, float)) and k != "n"
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nĐã ghi {args.output}")


if __name__ == "__main__":
    main()
