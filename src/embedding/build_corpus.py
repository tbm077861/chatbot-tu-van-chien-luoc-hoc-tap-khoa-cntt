"""Xây dựng corpus học phần + split test set (Giai đoạn 3).

Module này có hai nhiệm vụ:

1. **Corpus** (`data/embeddings/corpus.jsonl`):
   Mỗi dòng là một document mô tả một học phần dưới dạng văn bản tiếng Việt
   để embedding. Key định danh là `(nganh, ma_mon)`.

2. **Test set** (`data/embeddings/test.jsonl`) và **train pool**
   (`data/embeddings/train_pool.jsonl`): Sample 500 query-positive pairs từ
   `data/augmented/ALL_samples.jsonl` (hold-out cho đánh giá), phần còn lại
   để training. Chỉ lấy positive samples (graph_path_sampling, cf_svd_augmentation),
   negative_sampling để dành riêng làm hard negatives cho training.

CLI:
    python -m src.embedding.build_corpus --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
CURRICULUM_DIR = ROOT / "data" / "processed" / "curriculum_graph"
AUG_FILE = ROOT / "data" / "augmented" / "ALL_samples.jsonl"
OUT_DIR = ROOT / "data" / "embeddings"

NGANH_VN = {
    "CS": "Khoa học Máy tính",
    "IS": "Hệ thống Thông tin",
    "DS": "Khoa học Dữ liệu",
    "SE": "Kỹ thuật Phần mềm",
    "IT": "Công nghệ Thông tin",
}

LOAI_DK_VN = {
    "a": "học trước",
    "b": "tiên quyết",
    "c": "song hành",
}


def _format_course_doc(
    nganh: str,
    hp: dict,
    hk_so: int,
    code2name: dict[tuple[str, str], str],
) -> str:
    """Sinh văn bản tiếng Việt mô tả 1 học phần để embedding.

    Args:
        nganh: Mã ngành ("CS", "IS", "DS", "SE", "IT").
        hp: Dict học phần từ JSON curriculum.
        hk_so: Học kỳ chuẩn (1–9) của môn này trong chương trình khung.
        code2name: Map (nganh, ma_mon) -> ten_mon để dịch mã prereq sang tên.

    Returns:
        Chuỗi mô tả tự nhiên dùng cho embedding.
    """
    loai = "bắt buộc" if hp["loai"] == "bat_buoc" else "tự chọn"
    parts = [
        f"Môn {hp['ten_mon']} (mã {hp['ma_mon']}) thuộc ngành {NGANH_VN[nganh]}.",
        f"Số tín chỉ: {hp['so_tc']}.",
        f"Học kỳ chuẩn: {hk_so}.",
        f"Loại: {loai}.",
    ]
    if hp.get("dieu_kien"):
        # Dịch mã môn prereq sang tên môn để embedding có ngữ cảnh.
        names = []
        for prereq_code in hp["dieu_kien"]:
            name = code2name.get((nganh, prereq_code), prereq_code)
            names.append(f"{name} ({prereq_code})")
        dk_label = LOAI_DK_VN.get(hp.get("loai_dieu_kien") or "a", "học trước")
        parts.append(f"Điều kiện {dk_label}: " + ", ".join(names) + ".")
    if hp.get("khong_tinh_gpa"):
        parts.append("Không tính GPA.")
    return " ".join(parts)


def build_corpus() -> list[dict]:
    """Quét 5 file curriculum JSON, sinh corpus document/môn."""
    # Bước 1: build map (nganh, ma_mon) -> ten_mon để dịch prereq.
    code2name: dict[tuple[str, str], str] = {}
    for path in sorted(CURRICULUM_DIR.glob("*_curriculum.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        nganh = data["nganh"]
        for hk in data["hoc_ky"]:
            for hp in hk["hoc_phan"]:
                code2name[(nganh, hp["ma_mon"])] = hp["ten_mon"]

    # Bước 2: sinh corpus.
    corpus: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(CURRICULUM_DIR.glob("*_curriculum.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        nganh = data["nganh"]
        for hk in data["hoc_ky"]:
            for hp in hk["hoc_phan"]:
                key = (nganh, hp["ma_mon"])
                if key in seen:
                    continue
                seen.add(key)
                corpus.append(
                    {
                        "doc_id": f"{nganh}_{hp['ma_mon']}",
                        "nganh": nganh,
                        "ma_mon": hp["ma_mon"],
                        "ten_mon": hp["ten_mon"],
                        "so_tc": hp["so_tc"],
                        "hk_chuan": hk["hk_so"],
                        "loai": hp["loai"],
                        "khong_tinh_gpa": hp.get("khong_tinh_gpa", False),
                        "dieu_kien": hp.get("dieu_kien", []),
                        "loai_dieu_kien": hp.get("loai_dieu_kien"),
                        "text": _format_course_doc(nganh, hp, hk["hk_so"], code2name),
                    }
                )
    return corpus


def _stream_aug(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def split_train_test(
    valid_doc_ids: set[str],
    seed: int = 42,
    n_test: int = 500,
) -> tuple[list[dict], list[dict]]:
    """Split ALL_samples.jsonl thành test (500) + train pool (còn lại).

    Args:
        valid_doc_ids: Set doc_id hợp lệ (có trong corpus).
        seed: Seed random.
        n_test: Số test pair (default 500).

    Returns:
        (test_samples, train_pool_samples). Mỗi item gồm:
        - query: câu hỏi
        - positive_doc_ids: list[str] doc_id ("CS_001954")
        - negative_doc_ids: list[str] (chỉ với negative_sampling)
        - source: nguồn augmentation
        - nganh: mã ngành
    """
    rng = random.Random(seed)
    positives: list[dict] = []
    negatives: list[dict] = []

    for obj in _stream_aug(AUG_FILE):
        nganh = obj["nganh"]
        question = obj.get("qa", {}).get("question")
        if not question:
            continue
        if obj["source"] == "negative_sampling":
            inv = obj.get("invalid_target", {})
            codes = inv.get("course_codes", [])
            doc_ids = [
                f"{nganh}_{c}" for c in codes if f"{nganh}_{c}" in valid_doc_ids
            ]
            if not doc_ids:
                continue
            negatives.append(
                {
                    "query": question,
                    "negative_doc_ids": doc_ids,
                    "source": obj["source"],
                    "violation_type": obj.get("violation_type"),
                    "nganh": nganh,
                }
            )
        else:
            tgt = obj.get("target", {})
            codes = tgt.get("course_codes", [])
            doc_ids = [
                f"{nganh}_{c}" for c in codes if f"{nganh}_{c}" in valid_doc_ids
            ]
            if not doc_ids:
                continue
            positives.append(
                {
                    "query": question,
                    "positive_doc_ids": doc_ids,
                    "source": obj["source"],
                    "nganh": nganh,
                }
            )

    # Cân bằng test set theo ngành: 100 sample/ngành (5 ngành × 100 = 500).
    # Ưu tiên graph_path_sampling vì target từ chương trình khung thật.
    by_nganh: dict[str, list[dict]] = defaultdict(list)
    for p in positives:
        by_nganh[p["nganh"]].append(p)

    test: list[dict] = []
    test_keys: set[int] = set()
    n_per_nganh = n_test // 5
    for nganh, items in by_nganh.items():
        # Ưu tiên graph_path_sampling, sau đó cf_svd_augmentation.
        items_sorted = sorted(
            items, key=lambda x: 0 if x["source"] == "graph_path_sampling" else 1
        )
        rng.shuffle(items_sorted)
        # Lấy n_per_nganh đầu tiên (đã ưu tiên graph trước).
        picked = items_sorted[:n_per_nganh]
        for p in picked:
            test.append(p)
            test_keys.add(id(p))

    train_pool = [p for p in positives if id(p) not in test_keys]
    # Hard negatives để dành cho training; lưu kèm để prepare_training.py dùng.
    train_pool_obj = {"positives": train_pool, "negatives": negatives}
    return test, train_pool_obj  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=500)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Corpus.
    corpus = build_corpus()
    corpus_path = OUT_DIR / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[corpus] {len(corpus)} docs -> {corpus_path}")

    # 2. Test / Train split.
    valid_ids = {d["doc_id"] for d in corpus}
    test, train_pool_obj = split_train_test(valid_ids, args.seed, args.n_test)

    test_path = OUT_DIR / "test.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for ex in test:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[test] {len(test)} samples -> {test_path}")

    train_pool_path = OUT_DIR / "train_pool.jsonl"
    with open(train_pool_path, "w", encoding="utf-8") as f:
        # Positives.
        for ex in train_pool_obj["positives"]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(
        f"[train_pool positives] {len(train_pool_obj['positives'])} samples "
        f"-> {train_pool_path}"
    )

    neg_path = OUT_DIR / "hard_negatives.jsonl"
    with open(neg_path, "w", encoding="utf-8") as f:
        for ex in train_pool_obj["negatives"]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(
        f"[hard_negatives] {len(train_pool_obj['negatives'])} samples -> {neg_path}"
    )

    # Quick check phân bố test.
    by_nganh: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    for ex in test:
        by_nganh[ex["nganh"]] += 1
        by_source[ex["source"]] += 1
    print(f"[test/nganh] {dict(by_nganh)}")
    print(f"[test/source] {dict(by_source)}")


if __name__ == "__main__":
    main()
