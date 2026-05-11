"""Fine-tune text embedding models (PhoBERT / multilingual-E5).

Hỗ trợ hai backbone:
- `vinai/phobert-base-v2`: dùng MultipleNegativesRankingLoss với in-batch
  negatives (pairs).
- `intfloat/multilingual-e5-base`: dùng MultipleNegativesRankingLoss với
  triplet (query, positive, hard_negative) — hard_negative đưa thêm tín hiệu
  từ negative_sampling. Tự động thêm prefix "query:"/"passage:" theo convention E5.

CLI:
    python -m src.embedding.text_embedder --model phobert --epochs 1 --batch 32
    python -m src.embedding.text_embedder --model e5 --epochs 1 --batch 16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"

MODELS = {
    "phobert": {
        "hf_name": "vinai/phobert-base-v2",
        "max_seq_len": 128,
        "out_dir": EMB_DIR / "phobert_finetuned",
        "data_file": "train_pairs.jsonl",
        "use_triplet": False,
        "use_prefix": False,
    },
    "e5": {
        "hf_name": "intfloat/multilingual-e5-base",
        "max_seq_len": 192,
        "out_dir": EMB_DIR / "e5_finetuned",
        "data_file": "train_triplets.jsonl",
        "use_triplet": True,
        "use_prefix": True,
    },
}


def load_corpus_text() -> dict[str, str]:
    """Đọc corpus.jsonl, trả về dict doc_id -> text."""
    docs: dict[str, str] = {}
    with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["doc_id"]] = d["text"]
    return docs


def build_examples(
    data_path: Path,
    docs: dict[str, str],
    use_triplet: bool,
    use_prefix: bool,
) -> list[InputExample]:
    """Đọc training data, tạo InputExample list.

    Args:
        data_path: Đường dẫn train_pairs.jsonl hoặc train_triplets.jsonl.
        docs: Map doc_id -> text.
        use_triplet: True nếu dataset có hard negative (E5).
        use_prefix: True nếu cần thêm prefix "query:"/"passage:" (E5).
    """
    examples: list[InputExample] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["query"]
            pos = docs.get(obj["positive_doc_id"])
            if not pos:
                continue
            if use_prefix:
                q = f"query: {q}"
                pos = f"passage: {pos}"
            if use_triplet:
                neg = docs.get(obj["negative_doc_id"])
                if not neg:
                    continue
                if use_prefix:
                    neg = f"passage: {neg}"
                examples.append(InputExample(texts=[q, pos, neg]))
            else:
                examples.append(InputExample(texts=[q, pos]))
    return examples


def build_model(hf_name: str, max_seq_len: int) -> SentenceTransformer:
    """Build SentenceTransformer từ HuggingFace backbone + mean pooling."""
    word_emb = models.Transformer(hf_name, max_seq_length=max_seq_len)
    # sentence-transformers 5+ đổi tên; fallback nếu cần.
    dim = (
        word_emb.get_embedding_dimension()
        if hasattr(word_emb, "get_embedding_dimension")
        else word_emb.get_word_embedding_dimension()
    )
    pooling = models.Pooling(dim, pooling_mode="mean")
    return SentenceTransformer(modules=[word_emb, pooling])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số sample (debug); None = full.",
    )
    args = parser.parse_args()

    cfg = MODELS[args.model]
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[setup] device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[setup] model={cfg['hf_name']} max_seq={cfg['max_seq_len']}")

    docs = load_corpus_text()
    print(f"[data] {len(docs)} corpus docs")

    examples = build_examples(
        EMB_DIR / cfg["data_file"],
        docs,
        use_triplet=cfg["use_triplet"],
        use_prefix=cfg["use_prefix"],
    )
    if args.limit:
        examples = examples[: args.limit]
    print(f"[data] {len(examples)} training examples")

    model = build_model(cfg["hf_name"], cfg["max_seq_len"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(
        examples,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
    )

    # MultipleNegativesRankingLoss = InfoNCE in-batch.
    # Khi input là triplet, các negative trong batch + negative thứ 3 đều dùng làm âm.
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    n_steps = len(dataloader) * args.epochs
    warmup = math.ceil(n_steps * 0.1)
    print(f"[train] steps={n_steps} warmup={warmup} batch={args.batch} lr={args.lr}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model.fit(
        train_objectives=[(dataloader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=str(out_dir),
        show_progress_bar=True,
        use_amp=True,
    )
    print(f"[train] done in {time.time() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
