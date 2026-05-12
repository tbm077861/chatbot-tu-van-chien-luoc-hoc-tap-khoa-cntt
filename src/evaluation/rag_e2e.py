"""End-to-end eval RAG pipeline trên 100 câu hỏi mẫu (Giai đoạn 5).

Quy trình:
1. Lấy N câu (mặc định 100) từ `data/embeddings/test.jsonl`, cân bằng 20/ngành.
2. Chạy `RagPipeline.answer()` cho từng câu.
3. Đo lường:
   - Retrieval (sau hybrid + constraint): Recall@K, MRR, NDCG@10 so với gold.
   - Generation (parse mã từ response): Recall@K, MRR, NDCG@10.
   - Constraint: satisfaction rate, credit load validity.
4. Lưu `data/evaluation/rag_e2e_<mode>.json` + sample generations.

Mặc định dùng StubGenerator để chạy nhanh trên CPU (~1 phút/100 query).
Dùng `--use-llm` để bật Qwen-7B 4-bit (~10 phút trên RTX 5070).

Ví dụ CLI:
    python -m src.evaluation.rag_e2e --n 100 --stub
    python -m src.evaluation.rag_e2e --n 100 --use-llm
    python -m src.evaluation.rag_e2e --n 100 --use-llm --rerank
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "data" / "evaluation"

sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import (  # noqa: E402
    compute_constraint_metrics,
    compute_retrieval_metrics,
)
from src.rag_pipeline import RagPipeline  # noqa: E402


_HK_RE = re.compile(r"\bHK\s*(\d)\b")


def detect_hk(query: str) -> int | None:
    """Suy ra HK mục tiêu từ query (thô)."""
    m = _HK_RE.search(query)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def load_balanced_test(n_total: int, seed: int = 42) -> list[dict]:
    """Lấy n_total câu cân bằng theo ngành.

    Args:
        n_total: tổng số câu mong muốn (chia đều 5 ngành).
        seed: random seed cho permutation.

    Returns:
        List record. Mỗi record có khoá `query`, `positive_doc_ids`, `nganh`.
    """
    import numpy as np

    items: list[dict] = []
    with open(EMB_DIR / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    by_nganh: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_nganh[it.get("nganh", "?")].append(it)

    per_nganh = max(1, n_total // 5)
    rng = np.random.default_rng(seed)
    sampled: list[dict] = []
    for nganh in ("CS", "IS", "DS", "SE", "IT"):
        pool = by_nganh.get(nganh, [])
        if not pool:
            continue
        idx = rng.permutation(len(pool))[:per_nganh]
        sampled.extend(pool[i] for i in idx)
    # Truncate tới n_total chính xác.
    sampled = sampled[:n_total]
    return sampled


def main() -> None:
    ap = argparse.ArgumentParser(description="E2E eval RAG pipeline.")
    ap.add_argument("--n", type=int, default=100, help="Số câu hỏi mẫu.")
    ap.add_argument("--stub", action="store_true", help="Dùng StubGenerator.")
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="Bật Qwen LoRA generator (yêu cầu GPU + bitsandbytes).",
    )
    ap.add_argument("--rerank", action="store_true", help="Bật M3 cross-encoder.")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k-context", type=int, default=10)
    ap.add_argument("--candidate-pool", type=int, default=30)
    ap.add_argument(
        "--out-suffix",
        default=None,
        help="Suffix tên file kết quả. Mặc định dựa trên flags.",
    )
    args = ap.parse_args()

    if not args.stub and not args.use_llm:
        # Default behavior: stub nếu user không chỉ định.
        args.stub = True
        print("[note] Mặc định dùng StubGenerator. Thêm --use-llm để dùng Qwen.")

    use_llm = args.use_llm and not args.stub

    test_records = load_balanced_test(args.n, seed=args.seed)
    print(f"[data] loaded {len(test_records)} queries (target={args.n})")

    pipeline = RagPipeline.load_default(
        use_llm=use_llm,
        use_reranker=args.rerank,
        load_4bit=not args.no_4bit,
        top_k_context=args.top_k_context,
        candidate_pool=args.candidate_pool,
    )

    # Run.
    retrieval_preds: list[list[str]] = []
    generation_preds: list[list[str]] = []
    gold: list[list[str]] = []
    reports = []
    samples: list[dict] = []

    t0 = time.time()
    for i, rec in enumerate(test_records):
        query = rec["query"]
        nganh = rec.get("nganh", "")
        hk = detect_hk(query)
        result = pipeline.answer(
            query=query,
            nganh=nganh,
            hk_hien_tai=hk,
        )
        # Retrieval prediction = doc_ids valid sau constraint, theo thứ tự retriever.
        retrieval_preds.append(
            result.constraint.valid if result.constraint else [d for d, _ in result.retrieved]
        )
        generation_preds.append(result.recommendations)
        gold.append(rec["positive_doc_ids"])
        if result.constraint is not None:
            reports.append(result.constraint)

        # Lưu sample đầu mỗi ngành để inspect.
        if len(samples) < 10 and (i % max(1, len(test_records) // 10) == 0):
            samples.append(
                {
                    "idx": i,
                    "query": query,
                    "nganh": nganh,
                    "gold": rec["positive_doc_ids"],
                    "retrieval_valid": (
                        result.constraint.valid if result.constraint else []
                    )[:10],
                    "violations_count": (
                        len(result.constraint.violations) if result.constraint else 0
                    ),
                    "response": result.response,
                    "recommendations": result.recommendations,
                }
            )

        if (i + 1) % 10 == 0 or i + 1 == len(test_records):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(test_records) - i - 1)
            print(
                f"[run] {i+1}/{len(test_records)} elapsed={elapsed:.1f}s "
                f"eta={eta:.1f}s"
            )

    dt = time.time() - t0
    print(f"[done] {len(test_records)} queries in {dt:.1f}s "
          f"({dt/len(test_records):.2f}s/query)")

    # Metrics.
    retr_metrics = compute_retrieval_metrics(retrieval_preds, gold)
    gen_metrics = compute_retrieval_metrics(generation_preds, gold)
    constraint_metrics = compute_constraint_metrics(reports)

    print("\n=== Retrieval metrics (after constraint filter) ===")
    for k, v in retr_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n=== Generation metrics (parsed from LLM response) ===")
    for k, v in gen_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n=== Constraint metrics ===")
    for k, v in constraint_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.out_suffix
    if suffix is None:
        mode = "qwen" if use_llm else "stub"
        rr = "_rerank" if args.rerank else ""
        suffix = f"{mode}{rr}"
    out_path = OUT_DIR / f"rag_e2e_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "n_queries": len(test_records),
                    "use_llm": use_llm,
                    "use_reranker": args.rerank,
                    "candidate_pool": args.candidate_pool,
                    "top_k_context": args.top_k_context,
                    "seed": args.seed,
                },
                "retrieval_metrics": retr_metrics,
                "generation_metrics": gen_metrics,
                "constraint_metrics": constraint_metrics,
                "samples": samples,
                "duration_sec": dt,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()
