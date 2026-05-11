"""Đánh giá phương án embedding trên test set 500 pairs (Giai đoạn 3).

Metrics:
- Recall@1, Recall@5, Recall@10: tỉ lệ có ít nhất 1 gold doc trong top-K.
- MRR (Mean Reciprocal Rank): trung bình 1/rank của gold đầu tiên.
- NDCG@10: binary relevance.

Hỗ trợ encode bằng:
- SentenceTransformer (PhoBERT/E5 fine-tuned).
- File embedding pre-computed (.npy + node_ids.json) cho GNN/Hybrid.
- E5 cần thêm prefix `query:`/`passage:`.

CLI:
    python -m src.embedding.evaluator --method phobert
    python -m src.embedding.evaluator --method e5
    python -m src.embedding.evaluator --method gnn_gcn
    python -m src.embedding.evaluator --method hybrid
    python -m src.embedding.evaluator --method all  # chạy tất cả
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "data" / "embeddings"


def load_test() -> list[dict]:
    items: list[dict] = []
    with open(EMB_DIR / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_corpus() -> tuple[list[str], list[str]]:
    """Đọc corpus. Trả (doc_ids, texts) — thứ tự ổn định."""
    ids: list[str] = []
    texts: list[str] = []
    with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["doc_id"])
            texts.append(d["text"])
    return ids, texts


def encode_with_st(
    model_path: str,
    queries: list[str],
    docs: list[str],
    use_e5_prefix: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode queries & docs bằng SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
    if use_e5_prefix:
        queries = [f"query: {q}" for q in queries]
        docs = [f"passage: {d}" for d in docs]
    q_emb = model.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    d_emb = model.encode(
        docs,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return q_emb, d_emb


def metrics_from_scores(
    scores: np.ndarray,
    doc_ids: list[str],
    test_set: list[dict],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Tính Recall@K, MRR, NDCG@10 từ ma trận score [N_query × N_doc]."""
    doc_idx = {d: i for i, d in enumerate(doc_ids)}
    n = len(test_set)
    recall = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for qi, ex in enumerate(test_set):
        gold_ids = set(ex["positive_doc_ids"])
        gold_idx = {doc_idx[g] for g in gold_ids if g in doc_idx}
        if not gold_idx:
            continue
        # Sắp xếp giảm dần theo score.
        order = np.argsort(-scores[qi])
        # Recall@K.
        for k in ks:
            top_k = set(order[:k].tolist())
            if top_k & gold_idx:
                recall[k] += 1
        # MRR.
        for rank, doc_i in enumerate(order, start=1):
            if int(doc_i) in gold_idx:
                mrr_sum += 1.0 / rank
                break
        # NDCG@10.
        dcg = 0.0
        for rank, doc_i in enumerate(order[:10], start=1):
            if int(doc_i) in gold_idx:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(gold_idx), 10)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        if idcg > 0:
            ndcg_sum += dcg / idcg

    out = {f"Recall@{k}": recall[k] / n for k in ks}
    out["MRR"] = mrr_sum / n
    out["NDCG@10"] = ndcg_sum / n
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Trả ma trận cosine [A × B]. Yêu cầu a, b đã chuẩn hóa L2."""
    return a @ b.T


def evaluate_text_method(
    method: str, doc_ids: list[str], doc_texts: list[str], test_set: list[dict]
) -> dict:
    """Eval PhoBERT hoặc E5 fine-tuned."""
    if method == "phobert":
        path = EMB_DIR / "phobert_finetuned"
        use_prefix = False
    elif method == "e5":
        path = EMB_DIR / "e5_finetuned"
        use_prefix = True
    elif method == "e5_base_pretrained":
        path = "intfloat/multilingual-e5-base"
        use_prefix = True
    else:
        raise ValueError(method)
    queries = [ex["query"] for ex in test_set]
    q_emb, d_emb = encode_with_st(str(path), queries, doc_texts, use_prefix)
    scores = cosine_sim(q_emb, d_emb)
    return metrics_from_scores(scores, doc_ids, test_set)


def evaluate_gnn_method(
    backbone: str, doc_ids: list[str], test_set: list[dict], text_method: str
) -> dict:
    """Eval GNN: query encode bằng text model (text_method), doc encode bằng GNN node emb.

    Vì GNN chỉ encode node (= doc), query phải dùng text projector. Để đơn giản:
    project query -> doc space bằng text embedder, rồi nearest neighbor trên hybrid.
    Tuy nhiên GNN dim (128) khác text dim (768) -> không match trực tiếp.

    Workaround đơn giản: dùng PhoBERT encode query VÀ text features của node là
    PhoBERT (đã init làm node feature). Vì GNN output đã pha trộn text + topology,
    ta đo similarity giữa query_emb_phobert và doc_emb_gnn — nhưng dim khác nhau.
    -> Padding/projection sẽ làm méo. Thay vào đó, ta đánh giá GNN-standalone
    bằng cách dùng node embedding làm doc embedding và dùng nearest neighbor:
    encode query bằng MEAN của gold docs trong batch không khả thi.

    Cách thực hiện: **GNN-augmented PhoBERT** — dùng concat của PhoBERT embedding
    (cho query và doc) với GNN node embedding (cho doc, query mặc định = mean của
    GNN embedding các môn đã học trong context — nhưng test set không có context).

    Vì lý do thực tế, GNN được dùng dưới dạng **doc embedding bổ sung** trong
    hybrid (xem hybrid_embedder). Để có metric riêng, ta encode query bằng
    PhoBERT (cùng đầu vào với GNN init feature) rồi map qua một linear projection
    trained-from-zero — nhưng không có training data cho mapping này.

    -> Giải pháp đơn giản: eval GNN bằng cách dùng PhoBERT mean pool encode query,
    rồi tính cosine với node embedding GNN sau khi project query qua PCA cho khớp
    dim. Đây là baseline yếu — kết quả thực sự sẽ thể hiện trong hybrid.
    """
    from sentence_transformers import SentenceTransformer

    gnn_dir = EMB_DIR / f"gnn_{backbone}"
    z = np.load(gnn_dir / "node_embeddings.npy")
    with open(gnn_dir / "node_ids.json", encoding="utf-8") as f:
        gnn_ids: list[str] = json.load(f)
    # L2 normalize.
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    text_path = (
        EMB_DIR / "phobert_finetuned"
        if text_method == "phobert"
        else EMB_DIR / "e5_finetuned"
    )
    text_use_prefix = text_method == "e5"
    model = SentenceTransformer(str(text_path))
    if torch.cuda.is_available():
        model.to("cuda")
    queries = [ex["query"] for ex in test_set]
    if text_use_prefix:
        queries = [f"query: {q}" for q in queries]
    q_emb = model.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Reduce text query dim -> gnn dim bằng PCA fit trên text encoding của doc.
    docs_text: list[str] = []
    with open(EMB_DIR / "corpus.jsonl", encoding="utf-8") as f:
        id2text = {json.loads(l)["doc_id"]: json.loads(l)["text"] for l in f}
    docs_text = [id2text[i] for i in gnn_ids]
    if text_use_prefix:
        docs_text = [f"passage: {t}" for t in docs_text]
    d_text_emb = model.encode(
        docs_text,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # PCA project (q, d_text) -> gnn dim. Vì query và doc nằm trong cùng không
    # gian text, dùng PCA fit trên d_text rồi transform cả 2.
    from sklearn.decomposition import PCA

    pca = PCA(n_components=z.shape[1])
    pca.fit(d_text_emb)
    q_proj = pca.transform(q_emb)
    q_proj = q_proj / (np.linalg.norm(q_proj, axis=1, keepdims=True) + 1e-9)

    # Score = cosine(query_proj, gnn_node_emb) — đo "query gần đến môn nào trong
    # không gian GNN-augmented".
    scores = q_proj @ z.T
    return metrics_from_scores(scores, gnn_ids, test_set)


def evaluate_hybrid_method(
    doc_ids: list[str],
    doc_texts: list[str],
    test_set: list[dict],
    alpha: float = 0.7,
    gnn_backbone: str = "gcn",
) -> dict:
    """Hybrid late fusion: score = α · text_cosine + (1-α) · gnn_cosine.

    Args:
        alpha: trọng số text (1 = chỉ text, 0 = chỉ GNN).
    """
    # Text similarity (E5 fine-tuned, có ưu thế hơn PhoBERT ở mặt đa ngôn ngữ).
    queries = [ex["query"] for ex in test_set]
    q_emb_text, d_emb_text = encode_with_st(
        str(EMB_DIR / "e5_finetuned"), queries, doc_texts, use_e5_prefix=True
    )
    scores_text = q_emb_text @ d_emb_text.T

    # GNN similarity (qua PCA projection như trong evaluate_gnn_method).
    gnn_dir = EMB_DIR / f"gnn_{gnn_backbone}"
    z = np.load(gnn_dir / "node_embeddings.npy")
    with open(gnn_dir / "node_ids.json", encoding="utf-8") as f:
        gnn_ids: list[str] = json.load(f)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    # Align thứ tự GNN node với corpus doc_ids.
    gnn_pos = {nid: i for i, nid in enumerate(gnn_ids)}
    z_aligned = np.zeros((len(doc_ids), z.shape[1]), dtype=z.dtype)
    for i, d in enumerate(doc_ids):
        if d in gnn_pos:
            z_aligned[i] = z[gnn_pos[d]]

    from sklearn.decomposition import PCA

    pca = PCA(n_components=z.shape[1])
    pca.fit(d_emb_text)
    q_proj = pca.transform(q_emb_text)
    q_proj = q_proj / (np.linalg.norm(q_proj, axis=1, keepdims=True) + 1e-9)
    scores_gnn = q_proj @ z_aligned.T

    scores = alpha * scores_text + (1 - alpha) * scores_gnn
    return metrics_from_scores(scores, doc_ids, test_set)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "phobert",
            "e5",
            "e5_base_pretrained",
            "gnn_gcn",
            "gnn_gat",
            "hybrid",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    test_set = load_test()
    doc_ids, doc_texts = load_corpus()
    print(f"[eval] |test|={len(test_set)} |corpus|={len(doc_ids)}")

    results: dict[str, dict] = {}
    methods = (
        [args.method]
        if args.method != "all"
        else ["e5_base_pretrained", "phobert", "e5", "gnn_gcn", "gnn_gat", "hybrid"]
    )

    for m in methods:
        print(f"\n--- Đánh giá: {m} ---")
        try:
            if m in ("phobert", "e5", "e5_base_pretrained"):
                r = evaluate_text_method(m, doc_ids, doc_texts, test_set)
            elif m == "gnn_gcn":
                r = evaluate_gnn_method("gcn", doc_ids, test_set, "phobert")
            elif m == "gnn_gat":
                r = evaluate_gnn_method("gat", doc_ids, test_set, "phobert")
            elif m == "hybrid":
                r = evaluate_hybrid_method(
                    doc_ids, doc_texts, test_set, args.alpha
                )
            else:
                continue
            results[m] = r
            for k, v in r.items():
                print(f"  {k}: {v:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[m] = {"error": str(e)}

    # Lưu kết quả.
    out = EMB_DIR / "eval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] -> {out}")

    # Bảng tổng kết.
    print("\n=== TỔNG KẾT (sort by Recall@10 desc) ===")
    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        sorted_methods = sorted(
            valid.items(), key=lambda x: -x[1].get("Recall@10", 0)
        )
        header = ["Method", "R@1", "R@5", "R@10", "MRR", "NDCG@10"]
        print(" | ".join(f"{h:>20}" for h in header))
        for name, r in sorted_methods:
            row = [
                name,
                f"{r['Recall@1']:.4f}",
                f"{r['Recall@5']:.4f}",
                f"{r['Recall@10']:.4f}",
                f"{r['MRR']:.4f}",
                f"{r['NDCG@10']:.4f}",
            ]
            print(" | ".join(f"{c:>20}" for c in row))


if __name__ == "__main__":
    main()
