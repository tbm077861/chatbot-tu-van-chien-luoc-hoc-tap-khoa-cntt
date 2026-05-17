"""End-to-end eval RAG pipeline trên 100 câu hỏi mẫu (Giai đoạn 5).

Quy trình:
1. Lấy N câu (mặc định 100) từ test set, cân bằng 20/ngành.
   - Mode `cold` (mặc định): từ `data/embeddings/test.jsonl`, KHÔNG truyền completed.
   - Mode `warm`: từ `data/embeddings/test_with_profile.jsonl`, truyền
     `completed_ma_mon` (mã môn HK1..hk_completed) vào pipeline.
   - Mode `both`: chạy lần lượt cả 2, lưu 2 file kết quả riêng.
2. Chạy `RagPipeline.answer()` cho từng câu.
3. Đo:
   - Retrieval (sau hybrid + constraint): Recall@K, MRR, NDCG@10 so với gold.
   - Generation (parse mã từ response): Recall@K, MRR, NDCG@10.
   - Constraint: satisfaction rate, credit load validity.
4. Lưu `data/evaluation/rag_e2e_<mode>_<generator>.json`.

Cold vs Warm:
- Cold mô phỏng người dùng mới (chưa khai báo profile) — pipeline chỉ recommend
  được môn HK1-2.
- Warm mô phỏng người dùng có profile (đã hoàn thành HK1..N) — constraint
  filter chỉ loại các môn thật sự thiếu prereq → top valid kéo về gold.

Mặc định dùng StubGenerator để chạy nhanh CPU (~1s/100 query).
Dùng `--use-llm` để bật Qwen-7B 4-bit (~10 phút trên RTX 5070).

Ví dụ CLI:
    python -m src.evaluation.rag_e2e --n 100 --mode cold
    python -m src.evaluation.rag_e2e --n 100 --mode warm
    python -m src.evaluation.rag_e2e --n 100 --mode both
    python -m src.evaluation.rag_e2e --n 100 --mode both --use-llm --rerank
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


def load_balanced_test(
    n_total: int,
    seed: int = 42,
    test_path: Path | None = None,
) -> list[dict]:
    """Lấy n_total câu cân bằng theo ngành.

    Args:
        n_total: tổng số câu mong muốn (chia đều 5 ngành).
        seed: random seed cho permutation (ổn định giữa các lần chạy).
        test_path: file test.jsonl (mặc định `data/embeddings/test.jsonl`).
            Truyền `test_with_profile.jsonl` để có completed_ma_mon.

    Returns:
        List record. Mỗi record có khoá `query`, `positive_doc_ids`, `nganh`,
        và (nếu là augmented file) `hk_completed`, `hk_target`, `completed_ma_mon`.
    """
    import numpy as np

    if test_path is None:
        test_path = EMB_DIR / "test.jsonl"
    items: list[dict] = []
    with open(test_path, encoding="utf-8") as f:
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


def _format_metrics_block(name: str, m: dict) -> str:
    """Format dict metrics → block log nhiều dòng."""
    lines = [f"=== {name} ==="]
    for k, v in m.items():
        lines.append(f"  {k}: {v:.4f}")
    return "\n".join(lines)


def _run_one_mode(
    mode: str,
    pipeline: RagPipeline,
    test_records: list[dict],
) -> dict:
    """Chạy eval 1 mode (cold hoặc warm) trên cùng pipeline đã load.

    Args:
        mode: "cold" → không truyền completed; "warm" → truyền completed_ma_mon.
        pipeline: RagPipeline đã khởi tạo.
        test_records: list query (đã sample cân bằng).

    Returns:
        Dict {retrieval_metrics, generation_metrics, constraint_metrics, samples,
        duration_sec, predictions, gold}.
    """
    retrieval_preds: list[list[str]] = []
    generation_preds: list[list[str]] = []
    gold: list[list[str]] = []
    reports = []
    samples: list[dict] = []

    t0 = time.time()
    for i, rec in enumerate(test_records):
        query = rec["query"]
        nganh = rec.get("nganh", "")
        # Cold: hk từ query; warm: dùng hk_target ưu tiên trong file.
        hk = rec.get("hk_target") or detect_hk(query)
        completed = rec.get("completed_ma_mon") if mode == "warm" else None

        result = pipeline.answer(
            query=query,
            nganh=nganh,
            hk_hien_tai=hk,
            completed=completed,
        )

        retrieval_preds.append(
            result.constraint.valid
            if result.constraint
            else [d for d, _ in result.retrieved]
        )
        generation_preds.append(result.recommendations)
        gold.append(rec["positive_doc_ids"])
        if result.constraint is not None:
            reports.append(result.constraint)

        if len(samples) < 10 and (i % max(1, len(test_records) // 10) == 0):
            samples.append(
                {
                    "idx": i,
                    "query": query,
                    "nganh": nganh,
                    "hk_completed": rec.get("hk_completed"),
                    "hk_target": rec.get("hk_target"),
                    "n_completed_passed": len(completed) if completed else 0,
                    "gold": rec["positive_doc_ids"],
                    "retrieval_valid": (
                        result.constraint.valid if result.constraint else []
                    )[:10],
                    "violations_count": (
                        len(result.constraint.violations)
                        if result.constraint
                        else 0
                    ),
                    "response": result.response,
                    "recommendations": result.recommendations,
                }
            )

        if (i + 1) % 10 == 0 or i + 1 == len(test_records):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(test_records) - i - 1)
            print(
                f"[{mode}] {i+1}/{len(test_records)} elapsed={elapsed:.1f}s "
                f"eta={eta:.1f}s"
            )

    dt = time.time() - t0

    retr_metrics = compute_retrieval_metrics(retrieval_preds, gold)
    gen_metrics = compute_retrieval_metrics(generation_preds, gold)
    constraint_metrics = compute_constraint_metrics(reports)

    print()
    print(_format_metrics_block(f"[{mode}] Retrieval (after constraint)", retr_metrics))
    print()
    print(_format_metrics_block(f"[{mode}] Generation (parsed)", gen_metrics))
    print()
    print(_format_metrics_block(f"[{mode}] Constraint", constraint_metrics))

    return {
        "mode": mode,
        "retrieval_metrics": retr_metrics,
        "generation_metrics": gen_metrics,
        "constraint_metrics": constraint_metrics,
        "samples": samples,
        "duration_sec": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="E2E eval RAG pipeline.")
    ap.add_argument("--n", type=int, default=100, help="Số câu hỏi mẫu.")
    ap.add_argument(
        "--mode",
        choices=["cold", "warm", "both"],
        default="cold",
        help="cold = không completed; warm = pass completed từ file augmented; "
        "both = chạy cả 2 và xuất 2 file kết quả.",
    )
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
        "--profile-test",
        type=Path,
        default=EMB_DIR / "test_with_profile.jsonl",
        help="File test có completed_ma_mon (cho mode warm/both).",
    )
    args = ap.parse_args()

    if not args.stub and not args.use_llm:
        args.stub = True
        print("[note] Mặc định dùng StubGenerator. Thêm --use-llm để dùng Qwen.")
    use_llm = args.use_llm and not args.stub

    modes = ["cold", "warm"] if args.mode == "both" else [args.mode]
    if "warm" in modes and not args.profile_test.exists():
        raise FileNotFoundError(
            f"Cần file {args.profile_test} cho mode warm. "
            f"Chạy: python -m src.evaluation.augment_test_set"
        )

    # Mode warm/both luôn cần test_with_profile.jsonl; cold cũng dùng được
    # (bỏ qua các trường thừa).
    test_path = args.profile_test if args.profile_test.exists() else None
    test_records = load_balanced_test(args.n, seed=args.seed, test_path=test_path)
    print(f"[data] loaded {len(test_records)} queries from {test_path or 'default'}")

    # Load pipeline 1 lần dùng cho cả 2 mode (tiết kiệm bộ nhớ với LLM).
    pipeline = RagPipeline.load_default(
        use_llm=use_llm,
        use_reranker=args.rerank,
        load_4bit=not args.no_4bit,
        top_k_context=args.top_k_context,
        candidate_pool=args.candidate_pool,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generator_tag = "qwen" if use_llm else "stub"
    rr_tag = "_rerank" if args.rerank else ""

    all_results: dict[str, dict] = {}
    for mode in modes:
        print(f"\n========== MODE: {mode.upper()} ==========")
        result = _run_one_mode(mode, pipeline, test_records)
        out_path = OUT_DIR / f"rag_e2e_{mode}_{generator_tag}{rr_tag}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "n_queries": len(test_records),
                        "mode": mode,
                        "use_llm": use_llm,
                        "use_reranker": args.rerank,
                        "candidate_pool": args.candidate_pool,
                        "top_k_context": args.top_k_context,
                        "seed": args.seed,
                    },
                    **result,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n[save] -> {out_path}")
        all_results[mode] = result

    # Nếu chạy `both`, in bảng so sánh ngắn cuối.
    if len(all_results) == 2:
        print("\n========== COLD vs WARM ==========")
        cold = all_results["cold"]
        warm = all_results["warm"]
        rows = [
            ("Retrieval", cold["retrieval_metrics"], warm["retrieval_metrics"]),
            ("Generation", cold["generation_metrics"], warm["generation_metrics"]),
            ("Constraint", cold["constraint_metrics"], warm["constraint_metrics"]),
        ]
        for name, c, w in rows:
            print(f"\n[{name}]")
            for k in c:
                delta = w[k] - c[k]
                sign = "+" if delta >= 0 else ""
                print(
                    f"  {k:30s}  cold={c[k]:.4f}  warm={w[k]:.4f}  Δ={sign}{delta:.4f}"
                )


if __name__ == "__main__":
    main()
