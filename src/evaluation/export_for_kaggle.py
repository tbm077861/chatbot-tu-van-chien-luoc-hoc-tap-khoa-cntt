"""Export pipeline retrieval output → JSONL cho Kaggle generate (Giai đoạn 6).

Lý do tách 2 môi trường:
- **Local (RTX 5070 8GB)**: chạy retrieval + constraint nhanh, không cần LLM.
- **Kaggle T4×2 (16GB)**: chạy M5 Qwen-7B + LoRA generation (đã verify Stage 4).

Workflow:
    Local                                      Kaggle
    -----                                      ------
    1. Load pipeline (use_llm=False)
    2. For each query:
         - hybrid retrieve top-K
         - constraint check
         - build user_message (rag + no_rag variants)
    3. Save rag_inputs.jsonl + norag_inputs.jsonl
    4. Upload to Kaggle dataset           →   5. Load M5 LoRA
                                              6. Apply chat template + generate
                                              7. Save predictions.jsonl
    8. Download predictions.jsonl         ←
    9. Compute Hit@K/MRR/RAGAS local

Output schema mỗi dòng JSONL:
    {
      "idx": int,
      "query": str,           # gốc
      "nganh": str,
      "hk_completed": int | null,
      "hk_target": int | null,
      "gold": list[str],      # positive_doc_ids
      "retrieved_valid": list[str],  # sau constraint, top-K
      "context_doc_ids": list[str],  # docs đưa vào prompt
      "system_prompt": str,   # khớp training M5
      "user_message": str,    # đã format đủ profile + context + question
    }

CLI:
    python -m src.evaluation.export_for_kaggle --n 100 --mode warm
    python -m src.evaluation.export_for_kaggle --n 100 --mode both
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR = ROOT / "data" / "kaggle_export"

sys.path.insert(0, str(ROOT))
from src.evaluation.rag_e2e import load_balanced_test  # noqa: E402
from src.generation.prompt_templates import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_message,
    format_context,
)
from src.rag_pipeline import RagPipeline  # noqa: E402


_HK_RE = re.compile(r"\bHK\s*(\d)\b")


def detect_hk(query: str) -> int | None:
    """Suy ra HK target nếu record không có sẵn."""
    m = _HK_RE.search(query)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _build_record_pair(
    idx: int,
    rec: dict,
    pipeline: RagPipeline,
    mode: str,
) -> tuple[dict, dict]:
    """Chạy retrieval+constraint cho 1 query và trả về 2 record (rag, norag).

    Returns:
        (rag_record, norag_record). Cả 2 cùng query/gold; khác nhau ở
        user_message — RAG có context block, no-RAG thì không.
    """
    query = rec["query"]
    nganh = rec.get("nganh", "")
    hk_target = rec.get("hk_target") or detect_hk(query)
    completed = rec.get("completed_ma_mon") if mode == "warm" else None

    result = pipeline.answer(
        query=query,
        nganh=nganh,
        hk_hien_tai=hk_target,
        completed=completed,
    )

    context_docs = result.context_docs
    ctx_str = format_context(context_docs)

    # RAG variant — có context block.
    rag_user = build_user_message(
        question=query,
        nganh=nganh,
        hk_hien_tai=hk_target,
        da_hoan_thanh=completed,
        retrieved_context=ctx_str,
    )
    # No-RAG variant — không context, chỉ profile + question.
    norag_user = build_user_message(
        question=query,
        nganh=nganh,
        hk_hien_tai=hk_target,
        da_hoan_thanh=completed,
        retrieved_context="",
    )

    base = {
        "idx": idx,
        "query": query,
        "nganh": nganh,
        "hk_completed": rec.get("hk_completed"),
        "hk_target": hk_target,
        "gold": rec.get("positive_doc_ids", []),
        "retrieved_valid": (
            result.constraint.valid[:20] if result.constraint else []
        ),
        "context_doc_ids": [d["doc_id"] for d in context_docs],
        "system_prompt": SYSTEM_PROMPT,
    }
    rag_record = {**base, "user_message": rag_user, "variant": "rag"}
    norag_record = {**base, "user_message": norag_user, "variant": "norag"}
    return rag_record, norag_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument(
        "--mode",
        choices=["cold", "warm"],
        default="warm",
        help="cold = không truyền completed; warm = truyền từ test_with_profile.jsonl.",
    )
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--top-k-context", type=int, default=10)
    ap.add_argument("--candidate-pool", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    profile_path = EMB_DIR / "test_with_profile.jsonl"
    if args.mode == "warm" and not profile_path.exists():
        raise FileNotFoundError(
            f"Mode warm cần {profile_path}. Chạy `python -m src.evaluation.augment_test_set`."
        )
    test_path = profile_path if profile_path.exists() else None
    test_records = load_balanced_test(args.n, seed=args.seed, test_path=test_path)
    print(f"[data] loaded {len(test_records)} queries (mode={args.mode})")

    # Pipeline KHÔNG dùng LLM — chỉ retrieve + constraint.
    pipeline = RagPipeline.load_default(
        use_llm=False,
        use_reranker=args.rerank,
        candidate_pool=args.candidate_pool,
        top_k_context=args.top_k_context,
    )

    rag_records: list[dict] = []
    norag_records: list[dict] = []
    for i, rec in enumerate(test_records):
        rag_r, norag_r = _build_record_pair(i, rec, pipeline, args.mode)
        rag_records.append(rag_r)
        norag_records.append(norag_r)
        if (i + 1) % 20 == 0:
            print(f"[export] {i+1}/{len(test_records)}")

    rag_path = args.out_dir / f"rag_inputs_{args.mode}.jsonl"
    norag_path = args.out_dir / f"norag_inputs_{args.mode}.jsonl"
    for path, items in ((rag_path, rag_records), (norag_path, norag_records)):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[save] -> {path}  ({path.stat().st_size / 1024:.1f} KB)")

    # Lưu README để dễ dùng trên Kaggle.
    readme = args.out_dir / "README.md"
    readme.write_text(
        f"""# Kaggle export — RAG eval Stage 6

Sinh từ `src/evaluation/export_for_kaggle.py` (mode={args.mode}, n={len(test_records)},
rerank={args.rerank}, top_k_context={args.top_k_context}).

## Files

- `rag_inputs_{args.mode}.jsonl`: prompt có context (RAG variant).
- `norag_inputs_{args.mode}.jsonl`: prompt không context (baseline).

## Schema mỗi dòng

```
idx, query, nganh, hk_completed, hk_target, gold,
retrieved_valid, context_doc_ids,
system_prompt, user_message, variant
```

## Cách dùng trên Kaggle

1. Upload 2 file này + LoRA adapter (`m5_lora_for_kaggle/`) làm Kaggle dataset.
2. Trong notebook: load Qwen-7B 4-bit + adapter, đọc từng dòng JSONL,
   `apply_chat_template([system, user_message])` → generate.
3. Output: `predictions_{{rag,norag}}.jsonl` với thêm trường `response` và
   `predicted_doc_ids` (parse bằng regex `\\((?:m[ãa]\\s+)?(\\d{{6}})\\)`).
4. Download về local để compute metrics + RAGAS.
""",
        encoding="utf-8",
    )
    print(f"[save] -> {readme}")


if __name__ == "__main__":
    main()
