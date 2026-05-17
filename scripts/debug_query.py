"""Debug 1 query — dump full pipeline trace ra JSON để inspect.

Khi bot trả "tệ", dùng script này để biết tệ ở khâu nào:
- Retrieval kém (gold không có trong top 30)?
- Constraint lọc quá tay (valid count thấp)?
- Generator hallucinate (response có mã không có trong context)?
- Parser miss (response đúng nhưng regex không bắt)?

Output JSON gồm:
- query, profile (nganh/hk/completed/...)
- retrieved (top 30, (doc_id, score) + tên môn)
- reranked (nếu bật)
- constraint: valid, violations chi tiết, warnings, total_tc
- context_docs (top-K đưa vào LLM prompt)
- response (raw text từ generator)
- recommendations (parse từ response)
- timing

Ví dụ:
    python -m scripts.debug_query \\
        --query "Em ngành CS HK5, định hướng AI/ML" \\
        --nganh CS --hk 5 --dinh-huong AI/ML \\
        --completed 004247 001782 003575 \\
        --out data/evaluation/debug_query_1.json

Mặc định dùng stub (nhanh). Thêm `--use-llm` để chạy Qwen-7B thật.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RagPipeline  # noqa: E402


def _enrich(items: list[tuple[str, float]], id2doc: dict) -> list[dict]:
    """Map (doc_id, score) → dict thêm ten_mon, ma_mon, so_tc, loai."""
    out = []
    for did, sc in items:
        meta = id2doc.get(did, {})
        out.append(
            {
                "doc_id": did,
                "ten_mon": meta.get("ten_mon", "?"),
                "ma_mon": meta.get("ma_mon", "?"),
                "so_tc": meta.get("so_tc", "?"),
                "loai": meta.get("loai", "?"),
                "hk_chuan": meta.get("hk_chuan"),
                "score": float(sc),
            }
        )
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", required=True)
    ap.add_argument("--nganh", choices=["CS", "IS", "DS", "SE", "IT"], required=True)
    ap.add_argument("--completed", nargs="*", default=[])
    ap.add_argument("--in-progress", nargs="*", default=[])
    ap.add_argument("--hk", type=int, default=None)
    ap.add_argument("--gpa", type=float, default=None)
    ap.add_argument("--dinh-huong", default=None)
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="Dùng Qwen-7B+LoRA (mặc định Stub).",
    )
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="Bật M3 cross-encoder reranker.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/evaluation/debug_query.json"),
        help="File JSON để ghi full trace.",
    )
    args = ap.parse_args()

    print(f"[debug] loading pipeline (use_llm={args.use_llm}, rerank={args.rerank})...")
    t0 = time.time()
    pipeline = RagPipeline.load_default(
        use_llm=args.use_llm, use_reranker=args.rerank
    )
    print(f"[debug] pipeline ready in {time.time()-t0:.1f}s")

    # Pad mã môn 6 chữ số.
    completed = [m.zfill(6) for m in args.completed]
    in_progress = [m.zfill(6) for m in args.in_progress]

    print(f"[debug] running pipeline.answer()...")
    t1 = time.time()
    result = pipeline.answer(
        query=args.query,
        nganh=args.nganh,
        completed=completed or None,
        in_progress=in_progress or None,
        hk_hien_tai=args.hk,
        gpa=args.gpa,
        dinh_huong=args.dinh_huong,
    )
    answer_ms = (time.time() - t1) * 1000

    # Print human-readable summary.
    print("\n" + "=" * 60)
    print(f"Query: {args.query}")
    print(f"Profile: {args.nganh}, HK={args.hk}, GPA={args.gpa}, định hướng={args.dinh_huong}")
    print(f"Completed: {len(completed)} môn")
    print("=" * 60)

    print(f"\n--- Retrieved (top 10 / {len(result.retrieved)}) ---")
    for did, sc in result.retrieved[:10]:
        ten = pipeline.id2doc.get(did, {}).get("ten_mon", "?")
        print(f"  {did:14s}  {ten[:50]:50s}  {sc:.4f}")

    cons = result.constraint
    print(f"\n--- Constraint ---")
    if cons:
        print(f"  Valid:      {len(cons.valid)} môn, tổng TC={cons.total_tc}")
        print(f"  Violations: {len(cons.violations)} môn")
        for v in cons.violations[:5]:
            print(f"    × {v.get('doc_id')} — {v.get('reason')}")
        if cons.warnings:
            print(f"  Warnings:")
            for w in cons.warnings:
                print(f"    ⚠ {w}")

    print(f"\n--- Context truyền cho LLM ({len(result.context_docs)} doc) ---")
    for d in result.context_docs[:5]:
        print(f"  - {d.get('doc_id')}: {d.get('ten_mon')} ({d.get('so_tc')} TC)")

    print(f"\n--- Response ---")
    print(result.response[:500])

    print(f"\n--- Parsed recommendations ({len(result.recommendations)}) ---")
    for rec in result.recommendations:
        ten = pipeline.id2doc.get(rec, {}).get("ten_mon", "?")
        print(f"  {rec}: {ten}")

    print(f"\nTiming: {answer_ms:.0f} ms")

    # Save full JSON trace.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump = {
        "query": args.query,
        "profile": {
            "nganh": args.nganh,
            "hk_hien_tai": args.hk,
            "gpa": args.gpa,
            "dinh_huong": args.dinh_huong,
            "completed": completed,
            "in_progress": in_progress,
        },
        "config": {
            "use_llm": args.use_llm,
            "use_reranker": args.rerank,
            "top_k_context": pipeline.top_k_context,
            "candidate_pool": pipeline.candidate_pool,
        },
        "retrieved_top30": _enrich(result.retrieved[:30], pipeline.id2doc),
        "reranked": _enrich(result.reranked or [], pipeline.id2doc) if result.reranked else None,
        "constraint": {
            "valid": result.constraint.valid if result.constraint else [],
            "violations": result.constraint.violations if result.constraint else [],
            "warnings": result.constraint.warnings if result.constraint else [],
            "total_tc": result.constraint.total_tc if result.constraint else 0,
        } if result.constraint else None,
        "context_docs": result.context_docs,
        "response": result.response,
        "recommendations": result.recommendations,
        "timing_ms": answer_ms,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[debug] full trace saved to {args.out}")


if __name__ == "__main__":
    main()
