"""Sinh template manual user-satisfaction eval — 50 case (Giai đoạn 6).

Đầu vào: `predictions_rag.jsonl` từ Kaggle (Stage 6), `corpus.jsonl` để lấy tên môn.
Đầu ra: file markdown chứa 50 case stratified 10/ngành. Mỗi case có:
- Query gốc + profile rút gọn (HK, số môn đã hoàn thành).
- Gold answers (mã + tên môn).
- Bot response (text Qwen-7B+LoRA thật từ Kaggle).
- Bot recommendations đã parse + tên môn.
- Top 5 retrieved (từ retrieved_valid).
- Ô đánh giá thang 5 (checkbox markdown) + comment cho user fill in.

Sau khi user fill xong, có thể chạy `parse_manual_eval.py` để tổng hợp điểm trung bình.

Ví dụ dùng:
    python -m src.evaluation.build_manual_eval_template \\
        --predictions data/kaggle_export/stage6_rag_eval/predictions_rag.jsonl \\
        --corpus data/embeddings/corpus.jsonl \\
        --output data/evaluation/manual_satisfaction_template.md \\
        --per-nganh 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_id2doc(corpus_path: Path) -> dict[str, dict]:
    """Map doc_id → metadata (ten_mon, ma_mon, so_tc, loai)."""
    out: dict[str, dict] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def format_doc(doc_id: str, id2doc: dict[str, dict]) -> str:
    """`CS_001234` → `Tên môn (mã 001234, 3 TC, tu_chon)`."""
    meta = id2doc.get(doc_id)
    if not meta:
        return f"`{doc_id}` (không có metadata)"
    ten = meta.get("ten_mon", "?")
    ma = meta.get("ma_mon", "?")
    tc = meta.get("so_tc", "?")
    loai = meta.get("loai", "?")
    return f"**{ten}** (mã {ma}, {tc} TC, {loai})"


def sample_stratified(
    records: list[dict],
    per_nganh: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Lấy `per_nganh` case mỗi ngành, ổn định theo seed."""
    by_nganh: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_nganh[r["nganh"]].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for nganh in sorted(by_nganh):
        pool = by_nganh[nganh]
        if len(pool) <= per_nganh:
            out.extend(pool)
        else:
            out.extend(rng.sample(pool, per_nganh))
    return out


def render_case(idx: int, ex: dict, id2doc: dict[str, dict]) -> str:
    """Render 1 case thành markdown block."""
    nganh = ex["nganh"]
    hk_completed = ex.get("hk_completed", "?")
    hk_target = ex.get("hk_target", "?")
    query = ex["query"]
    response = ex.get("response", "(không có)")

    gold_lines = "\n".join(
        f"  - {format_doc(d, id2doc)}" for d in ex.get("gold", [])
    ) or "  - (không có)"

    pred = ex.get("predicted_doc_ids", [])
    pred_lines = "\n".join(
        f"  {i+1}. {format_doc(d, id2doc)}" for i, d in enumerate(pred)
    ) or "  - (model không output mã môn nào parse được)"

    retrieved = ex.get("retrieved_valid", [])[:5]
    ret_lines = "\n".join(
        f"  - {format_doc(d, id2doc)}" for d in retrieved
    ) or "  - (không có)"

    # Đánh dấu predicted có nằm trong gold không (cho user reference nhanh).
    gold_set = set(ex.get("gold", []))
    hits = [d for d in pred if d in gold_set]
    hit_note = (
        f"  - Predicted ∩ Gold: **{len(hits)}/{len(pred)}** môn trùng"
        if pred
        else "  - (không có pred để so)"
    )

    return f"""## Case {idx} — Ngành {nganh}

**Query**:
> {query}

**Profile**:
- Ngành: {nganh}
- Học kỳ đã hoàn thành: {hk_completed}, mục tiêu HK{hk_target}
- Số môn retriever lấy được (sau constraint filter): {len(ex.get('retrieved_valid', []))} môn

**Gold answers** (đáp án thực tế từ test set):
{gold_lines}

**Bot response** (Qwen-7B+LoRA, có context RAG):
> {response.strip().replace(chr(10), chr(10) + '> ')}

**Bot recommendations** (parse từ response):
{pred_lines}

{hit_note}

**Top 5 retrieved** (debug — môn retriever tìm được):
{ret_lines}

**Đánh giá của bạn** (tick 1 ô, xoá `[ ]` thành `[x]`):
- [ ] **5 — Xuất sắc**: đáp ứng đúng + giải thích rõ + ưu tiên đúng cluster định hướng
- [ ] **4 — Tốt**: đúng môn nhưng thiếu giải thích / thứ tự chưa tối ưu
- [ ] **3 — Tạm được**: 1–2 môn đúng, có lỗi nhẹ (lặp môn, vi phạm tiên quyết nhỏ)
- [ ] **2 — Yếu**: sai môn / vi phạm tiên quyết rõ / response lan man
- [ ] **1 — Tệ**: rỗng, hallucinate môn không tồn tại, sai hoàn toàn

**Ghi chú**: _(viết ở đây — vd: "Thiếu môn Học sâu mà gold có", "Đề xuất Tiếng Anh trong khi user là HK7", v.v.)_

---
"""


def render_summary_section(n_cases: int) -> str:
    """Section cuối file để user điền điểm trung bình."""
    return f"""# Tổng kết user satisfaction ({n_cases} case)

Sau khi tick xong từng case, đếm số lượng mỗi mức + tính điểm trung bình:

| Mức | Mô tả | Số case |
|---:|---|---:|
| 5 | Xuất sắc | __ |
| 4 | Tốt | __ |
| 3 | Tạm được | __ |
| 2 | Yếu | __ |
| 1 | Tệ | __ |
| **Tổng** | | **{n_cases}** |

**Điểm trung bình**: `(5×n5 + 4×n4 + 3×n3 + 2×n2 + 1×n1) / {n_cases}` = __

**User satisfaction rate** (≥ 4 điểm): `(n5 + n4) / {n_cases}` × 100% = __ %

## Quan sát chính (điền sau khi đánh giá)

1. **Loại lỗi phổ biến nhất**: …
2. **Cluster nào pipeline làm tốt**: …
3. **Đề xuất cải thiện**: …
"""


def render_header(n_cases: int, predictions_path: Path) -> str:
    return f"""# Manual User-Satisfaction Eval — {n_cases} case

> Sinh tự động từ `{predictions_path}` (Qwen-7B+LoRA, mode warm, top-K=10, sinh trên Kaggle T4×2).
> Stratified random 10 case/ngành × 5 ngành = {n_cases} case.
> Đánh giá thủ công bằng tay — không có ground truth tự động cho user satisfaction.
>
> **File liên quan** (nếu muốn xem đầy đủ profile của 1 case):
> - Predictions Qwen: `data/kaggle_export/stage6_rag_eval/predictions_rag.jsonl`
> - Input gốc với profile chi tiết (môn đã học, GPA): `data/kaggle_export/rag_inputs_warm.jsonl`
> - Test set (gold + completed_ma_mon list 50 môn/query): `data/embeddings/test_with_profile.jsonl`

## Hướng dẫn

1. Đọc từng case theo thứ tự.
2. So sánh **Bot response** với **Gold answers** + **Top 5 retrieved** (để hiểu retriever đã tìm được gì).
3. Tick 1 ô đánh giá 1–5 (xoá `[ ]` thay bằng `[x]`).
4. Ghi note ngắn (lỗi cụ thể, gợi ý cải thiện).
5. Sau khi xong tất cả: điền section "Tổng kết user satisfaction" ở cuối file.

## Thang điểm (tham khảo)

- **5 — Xuất sắc**: Bot trả đúng môn theo định hướng nghề, không vi phạm prereq, giải thích rõ.
- **4 — Tốt**: Đúng môn nhưng giải thích sơ sài hoặc thứ tự ưu tiên chưa tối ưu.
- **3 — Tạm được**: 1–2 môn đúng, có lỗi nhẹ.
- **2 — Yếu**: Sai môn hoặc vi phạm tiên quyết rõ ràng.
- **1 — Tệ**: Rỗng, hallucinate (môn không có trong corpus), sai hoàn toàn.

---

"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/kaggle_export/stage6_rag_eval/predictions_rag.jsonl"),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/embeddings/corpus.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/manual_satisfaction_template.md"),
    )
    parser.add_argument(
        "--per-nganh",
        type=int,
        default=10,
        help="Số case mỗi ngành (mặc định 10 → 50 case tổng).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"Không tìm thấy {args.predictions}")
    if not args.corpus.exists():
        raise SystemExit(f"Không tìm thấy {args.corpus}")

    records = load_jsonl(args.predictions)
    id2doc = build_id2doc(args.corpus)
    print(f"Đọc {len(records)} predictions + {len(id2doc)} corpus docs.")

    sampled = sample_stratified(records, per_nganh=args.per_nganh, seed=args.seed)
    print(f"Stratified sample: {len(sampled)} case ({args.per_nganh}/ngành).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(render_header(len(sampled), args.predictions))
        for i, ex in enumerate(sampled, start=1):
            f.write(render_case(i, ex, id2doc))
        f.write(render_summary_section(len(sampled)))

    print(f"Đã ghi {args.output} ({args.output.stat().st_size//1024} KB).")


if __name__ == "__main__":
    main()
