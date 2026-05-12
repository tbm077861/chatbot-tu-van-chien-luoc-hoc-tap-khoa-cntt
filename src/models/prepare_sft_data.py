"""Chuẩn bị dữ liệu SFT cho M5 — LoRA fine-tune Qwen2.5-7B-Instruct.

Chuyển `train_pairs.jsonl` (60k cặp query → 1 positive) + `corpus.jsonl`
(438 môn) thành các bản ghi chat ở định dạng Qwen chat template:

{
  "messages": [
    {"role": "system", "content": "<SYSTEM_PROMPT>"},
    {"role": "user", "content": "<query>"},
    {"role": "assistant", "content": "<gợi ý môn>"}
  ],
  "positive_doc_ids": ["..."]   # giữ lại để eval đối chiếu (không train)
}

Vì không dùng API LLM (yêu cầu giáo viên), response tự sinh bằng template
xác định: lấy ten_mon, so_tc, hk_chuan của các positive_doc_ids, gộp thành
danh sách có format ổn định. Format này được dùng nhất quán cả ở training
và evaluation parsing.

Để tăng quality, mỗi query gốc có 1 positive_doc_id. Ta gộp lại theo query
text → list các positive_doc_id (vì cùng query lặp lại với nhiều positives
khác nhau trong train_pairs), tránh học "1 query = 1 đáp án" cố định.

CLI:
    python -m src.models.prepare_sft_data
    python -m src.models.prepare_sft_data --max-samples 10000 --output kaggle_data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"
OUT_DIR_DEFAULT = ROOT / "data" / "sft"

SYSTEM_PROMPT = (
    "Bạn là chatbot tư vấn đăng ký học phần thông minh tại trường đại học. "
    "Nhiệm vụ của bạn là giúp sinh viên lập kế hoạch học tập tối ưu dựa trên "
    "chương trình khung ngành học, các ràng buộc tiên quyết, lịch sử điểm và "
    "định hướng nghề nghiệp của sinh viên, và quy định học vụ (giới hạn tín "
    "chỉ, điều kiện tốt nghiệp). "
    "Luôn giải thích ngắn gọn lý do gợi ý. Trả lời bằng tiếng Việt."
)

NGANH_NAMES = {
    "CS": "Khoa học Máy tính",
    "IS": "Hệ thống Thông tin",
    "DS": "Khoa học Dữ liệu",
    "SE": "Kỹ thuật Phần mềm",
    "IT": "Công nghệ Thông tin",
}


def load_jsonl(p: Path) -> list[dict]:
    out: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_corpus_map() -> dict[str, dict]:
    return {d["doc_id"]: d for d in load_jsonl(EMB_DIR / "corpus.jsonl")}


def format_response(doc_ids: list[str], id2doc: dict[str, dict]) -> str:
    """Render danh sách gợi ý theo template ổn định."""
    lines: list[str] = ["Dựa trên profile của bạn, mình gợi ý các môn sau:"]
    seen: set[str] = set()
    n = 0
    for did in doc_ids:
        if did in seen:
            continue
        doc = id2doc.get(did)
        if doc is None:
            continue
        seen.add(did)
        n += 1
        name = doc["ten_mon"]
        ma = doc["ma_mon"]
        tc = doc.get("so_tc", 0)
        loai = doc.get("loai", "bat_buoc")
        loai_vn = "bắt buộc" if loai == "bat_buoc" else "tự chọn"
        lines.append(f"{n}. **{name}** (mã {ma}, {tc} TC, {loai_vn})")
    if n == 0:
        lines.append("(Hiện tại chưa có gợi ý phù hợp, bạn nên trao đổi cố vấn.)")
    return "\n".join(lines)


def build(
    train_pairs_path: Path,
    test_path: Path,
    id2doc: dict[str, dict],
    max_samples: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Build SFT train + eval data.

    Train:
        Gộp các pair có cùng query text → 1 example với positive_doc_ids list.
        Sau đó shuffle + giới hạn max_samples.
    Eval:
        Mỗi item của test.jsonl giữ nguyên positive_doc_ids cho metrics.
    """
    pairs = load_jsonl(train_pairs_path)
    # Gộp theo query (cùng query có thể xuất hiện ở nhiều pair khác doc).
    bucket: dict[str, list[str]] = defaultdict(list)
    for p in pairs:
        if p["positive_doc_id"] in id2doc:
            bucket[p["query"]].append(p["positive_doc_id"])
    items: list[tuple[str, list[str]]] = []
    for q, dids in bucket.items():
        # Dedup theo doc_id, giữ tối đa 5 môn (giống ngữ cảnh recommendation).
        uniq: list[str] = []
        seen: set[str] = set()
        for d in dids:
            if d not in seen:
                uniq.append(d)
                seen.add(d)
            if len(uniq) >= 5:
                break
        items.append((q, uniq))

    import random

    rng = random.Random(seed)
    rng.shuffle(items)
    if max_samples > 0:
        items = items[:max_samples]

    train_records: list[dict] = []
    for q, dids in items:
        train_records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": format_response(dids, id2doc)},
                ],
                "positive_doc_ids": dids,
            }
        )

    test_raw = load_jsonl(test_path)
    test_records: list[dict] = []
    for ex in test_raw:
        test_records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["query"]},
                ],
                "positive_doc_ids": ex["positive_doc_ids"],
                "nganh": ex.get("nganh", ""),
            }
        )
    return train_records, test_records


def main() -> None:
    # Console Windows mặc định cp1252, ép UTF-8 để in ký tự VN không crash.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Số example train tối đa (0 = không giới hạn). Mặc định 20k đủ "
        "cho LoRA 1-2 epoch trên T4×2 trong thời gian Kaggle (~9h).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        default=str(OUT_DIR_DEFAULT),
        help="Thư mục output (mặc định data/sft).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2doc = load_corpus_map()
    print(f"[corpus] n_docs={len(id2doc)}")

    train_recs, test_recs = build(
        EMB_DIR / "train_pairs.jsonl",
        EMB_DIR / "test.jsonl",
        id2doc,
        args.max_samples,
        args.seed,
    )
    print(f"[train] examples={len(train_recs)}")
    print(f"[test] examples={len(test_recs)}")

    train_out = out_dir / "qwen_sft_train.jsonl"
    test_out = out_dir / "qwen_sft_test.jsonl"
    corpus_out = out_dir / "corpus.jsonl"

    with open(train_out, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(test_out, "w", encoding="utf-8") as f:
        for r in test_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Copy corpus (Kaggle notebook cần để map doc_id ↔ ten_mon khi parse).
    with open(corpus_out, "w", encoding="utf-8") as f:
        for did, d in id2doc.items():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # In ví dụ kiểm tra format.
    print("\n=== Ví dụ training record ===")
    sample = train_recs[0]
    print(f"system: {sample['messages'][0]['content'][:120]}...")
    print(f"user: {sample['messages'][1]['content']}")
    print(f"assistant:\n{sample['messages'][2]['content']}")
    print(f"positive_doc_ids: {sample['positive_doc_ids']}")

    print(f"\n[save] {train_out} ({train_out.stat().st_size//1024} KB)")
    print(f"[save] {test_out} ({test_out.stat().st_size//1024} KB)")
    print(f"[save] {corpus_out} ({corpus_out.stat().st_size//1024} KB)")
    print(
        "\nĐể upload Kaggle dataset, dùng 3 file trên trong:\n  "
        f"{out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
