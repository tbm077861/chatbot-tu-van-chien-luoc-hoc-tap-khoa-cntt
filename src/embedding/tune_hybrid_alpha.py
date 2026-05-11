"""Quét α cho hybrid fusion (E5 + GNN) để tìm điểm tối ưu.

Chạy nhanh: chỉ encode 1 lần text + GNN, sau đó test nhiều α với cùng score
matrices. Không re-encode model.

CLI:
    python -m src.embedding.tune_hybrid_alpha --backbone gcn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["gcn", "gat"], default="gcn")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    from src.embedding.evaluator import (
        load_corpus,
        load_test,
        metrics_from_scores,
    )

    test_set = load_test()
    doc_ids, doc_texts = load_corpus()

    # Encode 1 lần E5.
    model = SentenceTransformer(str(EMB_DIR / "e5_finetuned"))
    if torch.cuda.is_available():
        model.to("cuda")
    queries = [f"query: {ex['query']}" for ex in test_set]
    docs_prefixed = [f"passage: {t}" for t in doc_texts]
    q_emb = model.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    d_emb = model.encode(
        docs_prefixed,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores_text = q_emb @ d_emb.T

    # GNN.
    gnn_dir = EMB_DIR / f"gnn_{args.backbone}"
    z = np.load(gnn_dir / "node_embeddings.npy")
    with open(gnn_dir / "node_ids.json", encoding="utf-8") as f:
        gnn_ids = json.load(f)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    gnn_pos = {nid: i for i, nid in enumerate(gnn_ids)}
    z_aligned = np.zeros((len(doc_ids), z.shape[1]), dtype=z.dtype)
    for i, d in enumerate(doc_ids):
        if d in gnn_pos:
            z_aligned[i] = z[gnn_pos[d]]
    pca = PCA(n_components=z.shape[1])
    pca.fit(d_emb)
    q_proj = pca.transform(q_emb)
    q_proj = q_proj / (np.linalg.norm(q_proj, axis=1, keepdims=True) + 1e-9)
    scores_gnn = q_proj @ z_aligned.T

    # Quét α.
    print(f"--- alpha sweep (backbone={args.backbone}) ---")
    print(f"{'alpha':>6} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'NDCG@10':>8}")
    best = None
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        scores = alpha * scores_text + (1 - alpha) * scores_gnn
        m = metrics_from_scores(scores, doc_ids, test_set)
        print(
            f"{alpha:>6.2f} {m['Recall@1']:>7.4f} {m['Recall@5']:>7.4f} "
            f"{m['Recall@10']:>7.4f} {m['MRR']:>7.4f} {m['NDCG@10']:>8.4f}"
        )
        if best is None or m["Recall@10"] > best[1]["Recall@10"]:
            best = (alpha, m)
    print(f"\n[best α] {best[0]} -> R@10={best[1]['Recall@10']:.4f}")

    # Lưu best alpha.
    with open(EMB_DIR / "hybrid_best_alpha.json", "w", encoding="utf-8") as f:
        json.dump({"alpha": best[0], "backbone": args.backbone, **best[1]}, f, indent=2)


if __name__ == "__main__":
    main()
