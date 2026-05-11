"""Build FAISS index từ embedding của phương án tốt nhất (Giai đoạn 3).

Quy ước:
- Đọc tên model tốt nhất từ tham số CLI (`--method`) hoặc auto-pick từ
  `eval_results.json` (theo Recall@10).
- Encode toàn bộ corpus -> FAISS IndexFlatIP (inner product trên vector đã L2).
- Lưu index + metadata để Giai đoạn 5 (retrieval) load.

CLI:
    python -m src.embedding.build_faiss_index --method e5
    python -m src.embedding.build_faiss_index --auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"


def load_corpus() -> tuple[list[str], list[str], list[dict]]:
    """Đọc corpus đầy đủ. Trả (doc_ids, texts, full_metadata)."""
    ids: list[str] = []
    texts: list[str] = []
    meta: list[dict] = []
    with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["doc_id"])
            texts.append(d["text"])
            meta.append(d)
    return ids, texts, meta


def auto_pick_method() -> str:
    """Đọc eval_results.json, chọn method có Recall@10 cao nhất."""
    path = EMB_DIR / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(
            "Chưa có eval_results.json. Chạy evaluator trước hoặc dùng --method."
        )
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    valid = {k: v for k, v in results.items() if "error" not in v}
    best = max(valid.items(), key=lambda kv: kv[1].get("Recall@10", 0))
    print(f"[auto] best method = {best[0]} (R@10={best[1]['Recall@10']:.4f})")
    return best[0]


def encode_corpus(method: str, texts: list[str]) -> np.ndarray:
    """Encode corpus theo method. Trả [N × dim] đã L2-normalize."""
    from sentence_transformers import SentenceTransformer

    if method == "phobert":
        model = SentenceTransformer(str(EMB_DIR / "phobert_finetuned"))
        prefix_docs = texts
    elif method == "e5":
        model = SentenceTransformer(str(EMB_DIR / "e5_finetuned"))
        prefix_docs = [f"passage: {t}" for t in texts]
    elif method == "e5_base_pretrained":
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        prefix_docs = [f"passage: {t}" for t in texts]
    elif method == "hybrid":
        # Hybrid: concat L2-norm(E5_doc) + L2-norm(GNN_doc_aligned).
        # Note: query lúc retrieve phải làm tương tự (E5 + PCA(E5)→gnn_dim).
        return _encode_hybrid_corpus(texts)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if torch.cuda.is_available():
        model.to("cuda")
    emb = model.encode(
        prefix_docs,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.astype(np.float32)


def _encode_hybrid_corpus(texts: list[str]) -> np.ndarray:
    """Concat E5_text + GNN_aligned (cả 2 đã L2-normalize)."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    # E5.
    model = SentenceTransformer(str(EMB_DIR / "e5_finetuned"))
    if torch.cuda.is_available():
        model.to("cuda")
    prefix_docs = [f"passage: {t}" for t in texts]
    e5_emb = model.encode(
        prefix_docs,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    # GNN.
    z = np.load(EMB_DIR / "gnn_gcn" / "node_embeddings.npy")
    with open(EMB_DIR / "gnn_gcn" / "node_ids.json", encoding="utf-8") as f:
        gnn_ids = json.load(f)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    # Align thứ tự.
    corpus_ids, _, _ = load_corpus()
    gnn_pos = {nid: i for i, nid in enumerate(gnn_ids)}
    z_aligned = np.zeros((len(corpus_ids), z.shape[1]), dtype=np.float32)
    for i, d in enumerate(corpus_ids):
        if d in gnn_pos:
            z_aligned[i] = z[gnn_pos[d]]
    # Vì query không có GNN trực tiếp, ta lưu thêm projection PCA(e5)->gnn_dim
    # vào file metadata để retrieval xài.
    pca = PCA(n_components=z.shape[1])
    pca.fit(e5_emb)
    np.save(EMB_DIR / "hybrid_pca_components.npy", pca.components_)
    np.save(EMB_DIR / "hybrid_pca_mean.npy", pca.mean_)

    concat = np.concatenate([e5_emb, z_aligned], axis=1).astype(np.float32)
    # L2 normalize lại sau khi concat.
    norm = np.linalg.norm(concat, axis=1, keepdims=True) + 1e-9
    concat = concat / norm
    return concat


def build_index(emb: np.ndarray) -> faiss.Index:
    """Build FAISS IndexFlatIP. Vector phải L2-normalize trước -> inner product = cosine."""
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method để index (phobert/e5/hybrid). Bỏ trống nếu dùng --auto.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-pick từ eval_results.json (Recall@10).",
    )
    args = parser.parse_args()

    method = args.method or (auto_pick_method() if args.auto else "e5")
    print(f"[build] method={method}")

    doc_ids, texts, meta = load_corpus()
    print(f"[corpus] {len(doc_ids)} docs")

    emb = encode_corpus(method, texts)
    print(f"[encode] {emb.shape}")

    index = build_index(emb)

    out_dir = EMB_DIR / "faiss"
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "doc_ids.json", "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {"method": method, "dim": int(emb.shape[1]), "n_docs": len(doc_ids)},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[save] -> {out_dir}")


if __name__ == "__main__":
    main()
