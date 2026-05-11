"""Chuẩn bị training data cho fine-tune embedding (Giai đoạn 3).

Sinh hai file:
- `train_pairs.jsonl`: (query, positive_doc_id) — dùng cho PhoBERT (in-batch
  negatives via MultipleNegativesRankingLoss).
- `train_triplets.jsonl`: (query, positive_doc_id, hard_negative_doc_id) — dùng
  cho E5 (đa dạng hơn nhờ hard negatives từ `negative_sampling`).

Logic:
- Mở rộng mỗi query (có K positive docs) thành K cặp.
- Subsample tối đa `--max_pairs` để hạn chế thời gian train.
- Với triplets: mỗi cặp pair với 1 hard negative cùng ngành; nếu hết hard
  negative thì sample ngẫu nhiên doc khác ngành làm "easy negative" thay thế.

CLI:
    python -m src.embedding.prepare_training --seed 42 --max_pairs 60000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"


def load_corpus() -> dict[str, dict]:
    docs: dict[str, dict] = {}
    with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["doc_id"]] = d
    return docs


def load_positives() -> list[dict]:
    items: list[dict] = []
    with open(EMB_DIR / "train_pool.jsonl", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_hard_negatives_by_nganh() -> dict[str, list[str]]:
    pool: dict[str, list[str]] = defaultdict(list)
    with open(EMB_DIR / "hard_negatives.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pool[obj["nganh"]].extend(obj["negative_doc_ids"])
    # Dedup nhưng giữ tần suất bằng cách dùng list (cho phép sample có lặp).
    return dict(pool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=60000,
        help="Số tối đa training pair (sau khi expand & subsample).",
    )
    parser.add_argument(
        "--neg_per_query",
        type=int,
        default=1,
        help="Số hard negative ghép vào mỗi triplet.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    corpus = load_corpus()
    positives = load_positives()
    hard_neg_pool = load_hard_negatives_by_nganh()

    # 1. Expand thành các cặp (query, positive_doc_id).
    pairs: list[tuple[str, str, str]] = []  # (query, pos, nganh)
    for ex in positives:
        for pos_doc_id in ex["positive_doc_ids"]:
            pairs.append((ex["query"], pos_doc_id, ex["nganh"]))

    rng.shuffle(pairs)
    if len(pairs) > args.max_pairs:
        pairs = pairs[: args.max_pairs]

    # 2. Lưu pairs (cho PhoBERT).
    out_pairs = EMB_DIR / "train_pairs.jsonl"
    with open(out_pairs, "w", encoding="utf-8") as f:
        for q, pos, _ in pairs:
            f.write(
                json.dumps(
                    {"query": q, "positive_doc_id": pos},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[train_pairs] {len(pairs)} pairs -> {out_pairs}")

    # 3. Sinh triplets (cho E5).
    all_doc_ids = list(corpus.keys())
    by_nganh_docs: dict[str, list[str]] = defaultdict(list)
    for d in corpus.values():
        by_nganh_docs[d["nganh"]].append(d["doc_id"])

    triplets: list[dict] = []
    for q, pos, nganh in pairs:
        pool = hard_neg_pool.get(nganh, [])
        if pool:
            # Sample hard negative, đảm bảo khác positive.
            for _ in range(20):  # max retry
                neg = rng.choice(pool)
                if neg != pos:
                    break
            else:
                neg = rng.choice(all_doc_ids)
        else:
            neg = rng.choice(all_doc_ids)
        triplets.append(
            {
                "query": q,
                "positive_doc_id": pos,
                "negative_doc_id": neg,
            }
        )

    out_triplets = EMB_DIR / "train_triplets.jsonl"
    with open(out_triplets, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[train_triplets] {len(triplets)} triplets -> {out_triplets}")

    # 4. Báo cáo phân bố.
    by_nganh_pairs: dict[str, int] = defaultdict(int)
    for _, _, nganh in pairs:
        by_nganh_pairs[nganh] += 1
    print(f"[pairs/nganh] {dict(by_nganh_pairs)}")


if __name__ == "__main__":
    main()
