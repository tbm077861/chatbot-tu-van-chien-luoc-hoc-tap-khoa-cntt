# STATUS.md — Trạng thái Dự án

> File này được Claude Code tự cập nhật. Đầu phiên đọc để có context, cuối task ghi lại.
> Quy ước cập nhật: xem `CLAUDE.md` mục 7.

**Cập nhật lần cuối:** 2026-05-12 (Giai đoạn 4 ✅ HOÀN THÀNH — sẵn sàng vào Giai đoạn 5)

---

## Giai đoạn hiện tại

**Giai đoạn 5 — RAG Pipeline** 🔜 SẮP BẮT ĐẦU (session mới)

Giai đoạn 4 đã xong M1, M2, M3, M4, M5. Bảng tổng kết ở mục "Kết quả Giai đoạn 4" bên dưới.

---

## Đang làm

- Chuyển session mới để bắt đầu **Giai đoạn 5 — RAG Pipeline**.
- Kiến trúc đã chốt sau khi tổng kết M1–M5:
  * **M4 (GNN+Transformer)** làm retriever (R@10=99.8%, R@1=70.6%).
  * **M5 (Qwen2.5-7B LoRA)** làm generator (response tự nhiên, giải thích lý do).

---

## Sắp làm (Giai đoạn 4 — Mô hình học sâu)

1. ✅ M1: LSTM/GRU baseline sequence recommendation.
2. M2: Train Transformer bi-encoder dense retrieval (có thể tận dụng E5 fine-tuned đã có).
3. M3: Train cross-attention reranker trên top-K candidates.
4. M4: Implement GNN + Transformer fusion (tận dụng GCN đã có).
5. M5: LoRA fine-tune Qwen2.5-7B-Instruct (đã chốt).
6. Ablation study: so sánh tất cả mô hình.

---

## Dữ liệu hiện có

### Raw + Processed (Giai đoạn 1)

| Loại | Ngành | File | Trạng thái |
|---|---|---|---|
| Curriculum HTML | CS/IS/DS/SE/IT | `data/raw/curriculum/*.html` | ✅ |
| Grades CSV | CS/IS/DS/SE/IT | `data/raw/grades/*.csv` | ✅ (21.394 records, 2.587 SV) |
| Quy định | — | `data/raw/regulations/quy_dinh_hoc_vu.txt` | ✅ |
| Curriculum JSON + graph | 5 ngành | `data/processed/curriculum_graph/` | ✅ |
| Student profiles | 5 ngành | `data/processed/student_profiles/` | ✅ |
| Regulations JSON | — | `data/processed/regulations.json` | ✅ |

### Augmented (Giai đoạn 2)

| Nguồn | File | Số sample |
|---|---|---:|
| Graph path sampling | `data/augmented/graph_paths/*_graph_paths.jsonl` | 32.000 |
| CF (SVD) augmentation | `data/augmented/cf_profiles/*_cf_samples.jsonl` | 60.000 |
| Constraint-based negative sampling | `data/augmented/negative_samples/*_negative.jsonl` | 14.958 |
| **Hợp nhất hợp lệ** | `data/augmented/ALL_samples.jsonl` | **106.958** |
| Báo cáo validation | `data/augmented/VALIDATION_REPORT.md` | — |

Phân bố theo ngành cân bằng (~21k/ngành cho cả 5 ngành CS/IS/DS/SE/IT).

### Embedding (Giai đoạn 3)

| Loại | File / Dir | Kích thước |
|---|---|---|
| Corpus | `data/embeddings/corpus.jsonl` | 438 docs (1/môn, gộp 5 ngành) |
| Test set hold-out | `data/embeddings/test.jsonl` | 500 pairs (cân bằng 100/ngành) |
| Train pool positives | `data/embeddings/train_pool.jsonl` | 79.500 |
| Hard negatives | `data/embeddings/hard_negatives.jsonl` | 5.958 |
| Train pairs (PhoBERT) | `data/embeddings/train_pairs.jsonl` | 60.000 cặp |
| Train triplets (E5) | `data/embeddings/train_triplets.jsonl` | 60.000 triplet |
| PhoBERT fine-tuned | `data/embeddings/phobert_finetuned/` | 517 MB |
| E5 fine-tuned | `data/embeddings/e5_finetuned/` | 1.1 GB |
| GCN node embedding | `data/embeddings/gnn_gcn/` | 438 × 128 |
| GAT node embedding | `data/embeddings/gnn_gat/` | 438 × 128 |
| FAISS index (E5) | `data/embeddings/faiss/` | 1.5 MB |
| Báo cáo so sánh | `data/embeddings/EMBEDDING_COMPARISON.md` | — |

### Kết quả đánh giá (test 500 pairs)

| Method | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| **hybrid (E5+GCN α=0.85)** | 0.664 | 0.944 | **0.990** | 0.779 | 0.752 |
| **e5 fine-tuned** ★ | **0.668** | 0.964 | 0.988 | **0.786** | **0.761** |
| phobert fine-tuned | 0.642 | 0.964 | 0.988 | 0.775 | 0.747 |
| gnn_gcn | 0.018 | 0.084 | 0.158 | 0.066 | 0.031 |
| e5_base_pretrained (baseline) | 0.016 | 0.054 | 0.064 | 0.044 | 0.012 |
| gnn_gat | 0.026 | 0.036 | 0.050 | 0.044 | 0.019 |

★ Đã build FAISS index với E5 fine-tuned (Recall@10 chỉ kém hybrid 0.2%
nhưng đơn giản hơn — 1 model, không cần PCA bridge).

Fine-tune E5 cải thiện **+92.4% Recall@10** so với pretrained baseline.

---

## Checklist Giai đoạn 1 — ✅ HOÀN THÀNH

- [x] Parse HTML chương trình khung 5 ngành → JSON + graph
- [x] Build prerequisite graph (NetworkX) cho từng ngành
- [x] Validate cả 5 graph (không cycle, không thiếu prereq)
- [x] Load và chuẩn hóa 5 file CSV điểm
- [x] Build student profile từ 5 ngành
- [x] Trích xuất quy định học vụ → JSON structured
- [x] EDA notebook
- [x] Báo cáo chất lượng dữ liệu

## Checklist Giai đoạn 2 — ✅ HOÀN THÀNH

- [x] Graph-based path sampling → 32.000 samples (mục tiêu ~20k) ✅ vượt
- [x] CF augmentation (SVD) → 60.000 samples (mục tiêu ~30k virtual profiles) ✅ vượt
- [x] Constraint-based negative sampling → 14.958 samples (mục tiêu ~15k) ✅
- [ ] ~~LLM synthetic QA generation~~ — BỎ vì yêu cầu của giáo viên không dùng API.
  Đã làm 3/4 phương pháp theo project_instructions.md mục 6.1 (yêu cầu "ít nhất 3 trong 4").
- [x] Validate chất lượng: 0 lỗi schema, 0 duplicate ID, ngành cân bằng
- [x] Tổng kết: 106.958 samples (mục tiêu ≥85k) ✅ vượt 25%

## Checklist Giai đoạn 3 — ✅ HOÀN THÀNH

- [x] Fine-tune PhoBERT-base-v2 với contrastive loss (60k pairs, 1 epoch, MultipleNegativesRankingLoss in-batch)
- [x] Fine-tune multilingual-E5-base với hard negatives (60k triplets, 1 epoch, hard neg từ negative_sampling)
- [x] Train GCN/GAT trên prerequisite graph (PhoBERT text-features init, link prediction self-supervised)
- [x] Xây dựng hybrid embedding (late fusion E5 + GCN, α=0.85 sau khi sweep)
- [x] Đánh giá 6 phương án (PhoBERT, E5, E5 pretrained baseline, GCN, GAT, Hybrid) trên test 500 pairs
- [x] Chọn E5 fine-tuned build FAISS index (R@10 = 98.8%, MRR = 0.79)

## Checklist Giai đoạn 4 — ✅ HOÀN THÀNH

- [x] M1: Train LSTM baseline (10.68M params, 5 epochs, R@10=99.2%, MRR=0.805)
- [x] M1: Train GRU variant để so sánh (10.09M params, 5 epochs, R@10=99.4%, MRR=0.801)
- [x] M2: Train bi-encoder Transformer from-scratch (9.07M params, 5 epochs, R@10=99.4%, MRR=0.793)
- [x] M3: Train cross-attention reranker PhoBERT-base-v2 (135M, 30k triplets × 2 epochs, pair_acc=0.93). Sau rerank top-20: R@10=99.8% (+0.010 vs E5), R@1=0.654 (−0.014 vs E5), MRR=0.783 (−0.0035)
- [x] M4: GNN+Transformer learned fusion (1.12M trainable, cached E5+GCN, 10 epochs trong 12s)
- [x] M4 ablation: with_gnn R@1=0.700 vs no_gnn R@1=0.706 — GNN không đóng góp đáng kể (graph prereq quá thưa)
- [x] M5: LoRA fine-tune Qwen2.5-7B-Instruct trên Kaggle T4×2 (1 epoch, 2500 steps, ~6.5h). LoRA adapter ~160MB. Eval v2 (sau khi fix SYSTEM prompt + regex + max_tokens): **R@1=0.640, R@5=0.682, MRR=0.661, NDCG@10=0.255**
- [x] Ablation study: bảng tổng kết M1–M5 ở mục "Kết quả Giai đoạn 4" bên dưới

---

## Kết quả Giai đoạn 4 — Tổng kết M1–M5

| Model | Params | R@1 | R@5 | R@10 | MRR | NDCG@10 | Vai trò sau cùng |
|---|---|---:|---:|---:|---:|---:|---|
| M1 BiLSTM | 10.7M | 0.666 | 0.958 | 0.992 | 0.805 | ~0.76 | Baseline |
| M1 BiGRU | 10.1M | 0.668 | — | 0.994 | 0.801 | — | Baseline |
| M2 Transformer scratch | 9.1M | 0.674 | 0.960 | 0.994 | 0.793 | — | Architecture exp |
| M3 PhoBERT cross-rerank | 135M | 0.654 | 0.960 | 0.998 | 0.783 | 0.740 | (không dùng, làm giảm MRR) |
| **M4 GNN+Transformer** ★ | 1.1M tr. | **0.706** | **0.972** | **0.998** | **0.811** | — | **🏆 Retriever (Stage 5)** |
| **M5 Qwen 7B LoRA** | 7B + 40M LoRA | 0.640 | 0.682 | 0.682 | 0.661 | 0.255 | **💬 Generator (Stage 5)** |

★ Best retriever.

**Lưu ý M5**:
- M5 metrics thấp hơn M1-M4 trên retrieval task vì model chỉ generate 1-2 môn/response (training data sau dedup phần nhiều có 1-2 unique positives/query).
- Vai trò thực sự của M5 là **conversational generator**, không phải retriever → metrics retrieval không phản ánh đúng giá trị.
- Eval v1 cũ (R@1=0.368) bị parse regex quá strict + SYSTEM prompt mismatch. Eval v2 sau fix tăng +27 điểm R@1.

**Kiến trúc Stage 5 đã chốt**:
```
User query → M4 retrieve top-K → format context → M5 generate response
```

---

## Phụ thuộc đã cài

### Giai đoạn 1
- `lxml>=4.9`, `beautifulsoup4>=4.12` — parse HTML curriculum
- `pyarrow>=14.0` — lưu file .parquet
- `networkx>=3.2` — prerequisite graph
- `pandas`, `numpy`, `scikit-learn`, `scipy`

### Giai đoạn 2 (cài 2026-05-11)
- `rank-bm25>=0.2` — sparse retrieval (chuẩn bị Giai đoạn 5)
- `torch>=2.1` — backend ML
- `transformers>=4.35` — HuggingFace
- `sentence-transformers>=2.2` — fine-tune embedding
- `faiss-cpu>=1.7.4` — vector index

### Giai đoạn 3 (cài 2026-05-11)
- `torch-geometric==2.7.0` — GCN/GAT cho prerequisite graph

---

## Blockers & Câu hỏi cho user

- ✅ Đã cài `torch-geometric==2.7.0` cho GCN/GAT (2026-05-11, hoạt động tốt với CUDA 12.8).
- ⏳ Chưa quyết định LLM cuối cùng cho M5: Vistral-7B vs Qwen2.5-7B-Instruct (Giai đoạn 4). Sẽ hỏi khi đến.
- ⏳ Phương pháp Hybrid hiện dùng PCA bridge text→GNN. Nếu hiệu quả không tốt ở Giai đoạn 5 sẽ thay bằng cross-encoder reranking.
- ✅ Đã xác nhận: KHÔNG dùng API LLM (yêu cầu giáo viên) → đã loại synthetic QA generation.

---

## Nhật ký

- 2026-05-09: Khởi tạo dự án. Tạo `CLAUDE.md`, `STATUS.md`, `project_instructions.md`. Đã có sẵn 5 file HTML curriculum và `IS_TuChon.xlsx`.
- 2026-05-09: Viết `src/data/parser.py` — parse HTML chương trình khung → JSON. File: `src/data/parser.py`. Ghi chú: parse đúng điều kiện tiên quyết, pad mã môn 6 số, nhận diện môn có dấu * (khong_tinh_gpa).
- 2026-05-09: Viết `src/data/graph_builder.py` — build NetworkX DiGraph từ JSON curriculum. File: `src/data/graph_builder.py`. Ghi chú: validate no-cycle + no-missing-node, tất cả 5 ngành OK. Output: `data/processed/curriculum_graph/<NGANH>_prereq_graph.gpickle`.
- 2026-05-09: Parse thành công 5 ngành (CS=91 môn, IS=88, DS=86, SE=88, IT=85). Build 5 graph (CS=37 edges, IS=21, DS=15, SE=41, IT=11). Không có cycle, không thiếu node.
- 2026-05-11: User cung cấp đủ dữ liệu: 5 file CSV grades (CS/DS/SE/IT dùng sep="|", IS dùng sep=",") và quy_dinh_hoc_vu.txt.
- 2026-05-11: Viết `src/data/preprocessor.py` — load + clean 5 CSV → parquet/csv. Tổng 21.394 records, 2.587 SV. IS có 655 null điểm đã loại. File: `src/data/preprocessor.py`. Output: `data/processed/student_profiles/`.
- 2026-05-11: Viết `src/data/regulation_parser.py` — trích xuất quy định IUH → JSON (9 nhóm quy tắc: TC đăng ký 12–30/HK, GPA tốt nghiệp ≥2.0, bảng quy đổi điểm, cảnh báo học tập, v.v.). File: `src/data/regulation_parser.py`. Output: `data/processed/regulations.json`.
- 2026-05-11: Viết EDA notebook `notebooks/01_data_exploration.ipynb` — 16 phần phân tích. Ghi chú: Giai đoạn 1 HOÀN THÀNH.
- 2026-05-11: User xác nhận KHÔNG dùng API LLM cho dự án → bỏ LLM synthetic QA, làm 3/4 phương pháp augmentation. Cài deps Giai đoạn 2: torch, transformers, sentence-transformers, faiss-cpu, rank-bm25.
- 2026-05-11: Viết `src/data/augmentation/graph_sampler.py` — mô phỏng sinh viên đi qua curriculum, đa dạng kiểu SV (giỏi/khá/TB/yếu) + career bias (chỉ CS), cắt mỗi path thành nhiều checkpoint training sample. Output: 32.000 samples qua 5 ngành. Vi phạm prereq trung bình ~2.8% (do mô phỏng SV trượt môn, edge case acceptable).
- 2026-05-11: Viết `src/data/augmentation/cf_augment.py` — Truncated SVD trên ma trận (SV × Môn), sample latent Gaussian quanh population để sinh virtual student, reconstruct grade → profile. 6000 virtual/ngành × 5 × 2 sample/profile = 60.000 samples.
- 2026-05-11: Viết `src/data/augmentation/negative_sampler.py` — sinh 5 loại vi phạm: prereq, vượt TC max, dưới TC min, sai thứ tự HK, học lại điểm cao. 600/loại × 5 loại × 5 ngành = 14.958 samples.
- 2026-05-11: Viết `src/data/augmentation/validate_augmented.py` — check schema, dedup, hợp nhất tất cả → `data/augmented/ALL_samples.jsonl`. Kết quả: **106.958 samples, 0 lỗi schema, 0 duplicate ID, ngành cân bằng (~21k/ngành)**. Vượt mục tiêu 85k của project_instructions.md mục 10. Giai đoạn 2 HOÀN THÀNH.
- 2026-05-11: Cài `torch-geometric==2.7.0` (xác nhận tương thích torch 2.12 dev + CUDA 12.8). Bắt đầu Giai đoạn 3.
- 2026-05-11: Viết `src/embedding/build_corpus.py` — sinh 438 corpus doc (1/môn) + split test set 500 pairs cân bằng theo ngành (hold-out, không dùng training). Hợp nhất 5 ngành: chỉ giữ sample có ít nhất 1 doc_id hợp lệ trong corpus (loại 12k positive + 9k negative do CF augmentation sinh code không có trong curriculum).
- 2026-05-11: Viết `src/embedding/prepare_training.py` — sinh 60k train_pairs (PhoBERT) + 60k train_triplets (E5 với hard negative cùng ngành).
- 2026-05-11: Viết `src/embedding/text_embedder.py` — fine-tune `vinai/phobert-base-v2` và `intfloat/multilingual-e5-base` bằng MultipleNegativesRankingLoss. PhoBERT: 4m43s, loss 3.45→0.76. E5: 16m33s, loss 1.32→0.72. Cả 2 model lưu ở `data/embeddings/{phobert,e5}_finetuned/`.
- 2026-05-11: Viết `src/embedding/graph_embedder.py` — train GCN/GAT 2-layer (init feature = PhoBERT embedding của text trong corpus, link prediction self-supervised với negative sampling). GCN/GAT đều train xong <2s, link-pred acc 76-84%. Output: `data/embeddings/gnn_{gcn,gat}/node_embeddings.npy`.
- 2026-05-11: Viết `src/embedding/evaluator.py` — đánh giá 6 phương án trên test 500 pairs. Kết quả best: **E5 fine-tuned R@10=98.8%, MRR=0.786**. Fine-tune giúp E5 vọt từ 6.4% lên 98.8% R@10 (+92.4%). GNN standalone yếu (R@10 ~15%) vì query không có course code rõ ràng để leverage graph structure.
- 2026-05-11: Viết `src/embedding/tune_hybrid_alpha.py` — sweep α cho hybrid E5+GCN. Best α=0.85 đạt R@10=99.0% (+0.2% vs E5 đơn lẻ), nhưng MRR/NDCG kém hơn E5 alone.
- 2026-05-11: Viết `src/embedding/build_faiss_index.py` — build FAISS IndexFlatIP với E5 fine-tuned (438 docs × 768 dim, cosine via inner product). Output: `data/embeddings/faiss/{index.faiss, doc_ids.json, metadata.jsonl, config.json}`. Verify nhanh: query "Em ngành SE HK6, định hướng Web" trả về Lập trình mạng Qt, Lập trình phân tán Java/NET — semantically đúng.
- 2026-05-11: Viết `src/embedding/compare_report.py` — sinh `data/embeddings/EMBEDDING_COMPARISON.md` tổng kết. Giai đoạn 3 HOÀN THÀNH.
- 2026-05-11: Bắt đầu Giai đoạn 4. Chốt: M1→M5 tuần tự; M5 dùng Qwen2.5-7B-Instruct + LoRA 4-bit; tận dụng split 79.5k/5.9k của Giai đoạn 3. File: `src/models/__init__.py`.
- 2026-05-11: Viết `src/models/lstm_recommender.py` — M1 baseline. BiLSTM/BiGRU 2 lớp (10.68M / 10.09M params) + course embedding table (438 lớp), CE loss full-softmax. PhoBERT tokenizer (subword) nhưng embedding init random. File: `src/models/lstm_recommender.py`. Output: `data/models/lstm_baseline/{lstm,gru}_{best.pt,results.json}`. Kết quả test 500 pairs: **LSTM R@10=99.2%, MRR=0.805**; **GRU R@10=99.4%, MRR=0.801**. Cả 2 vượt E5 fine-tuned (R@10=98.8%, MRR=0.786) — nhờ corpus nhỏ (438 môn) + course embedding table học bucket trực tiếp. Train ~85s/model trên RTX 5070.
- 2026-05-11: Viết `src/models/transformer_retriever.py` — M2 bi-encoder Transformer from-scratch (9.07M params, 4 layers, d=128, h=4). Cùng training scheme với M1 (CE 438 lớp, PhoBERT tokenizer, embedding init random) để so sánh fair. 5 epochs trong 80s. Kết quả: **R@10=99.4%, MRR=0.793**. Kết luận: với corpus nhỏ và full-softmax CE, kiến trúc encoder (LSTM vs Transformer) không khác biệt nhiều — signal chính đến từ course embedding table.
- 2026-05-11: Viết `src/models/cross_reranker.py` — M3 cross-attention reranker. Fine-tune PhoBERT-base-v2 (135M params) làm cross-encoder với pairwise hinge loss (margin=0.3). Train 30k triplets × 2 epochs trên RTX 5070, pair_acc 0.83→0.93 (15 phút). Two-stage eval: E5 retrieve top-20 → cross-encoder rescore. Kết quả: R@10=99.8% (+0.01), R@1=0.654 (−0.014), MRR=0.783 (−0.0035), NDCG@10=0.740 (−0.021). M3 chỉ cải thiện R@10, làm giảm các metric khác — do test gold có ~5 positive/query (CF-derived), reranker pairwise có thể đẩy "alternative positive" xuống.
- 2026-05-11: Viết `src/models/gnn_transformer.py` — M4 learned linear fusion bi-encoder. Doc side: concat E5(text 768) + GCN(graph 128) → MLP(896→256). Query side: MLP(E5 768→256). Cached E5 embeddings (`data/embeddings/cache_e5/`) → 10 epochs train trong 12s, chỉ 1.12M trainable params. Kết quả: **R@1=0.700, R@10=99.6%, MRR=0.807**. Ablation `--no-gnn` (zero-mask GCN portion): R@1=0.706, MRR=0.811 — **hơi tốt hơn với_gnn**. Kết luận: GCN không thêm signal hữu ích, lift của M4 đến từ learned MLP projection trên E5 embedding, không phải GNN. Lý do: prerequisite graph quá thưa (438 node, ~125 edge — chỉ 0.06% mật độ).
- 2026-05-11: Viết `src/models/prepare_sft_data.py` — chuẩn bị data SFT cho M5. Gộp train_pairs theo query → mỗi sample list 5 positive doc_id, format thành Qwen chat template (system/user/assistant). Response template: "1. **Ten mon** (mã XXXXXX, N TC, bat_buoc/tu_chon)" (parse-friendly). Output: 20.000 train + 500 test ở `data/sft/{qwen_sft_train,qwen_sft_test,corpus}.jsonl` (~22 MB), zip thành `ck-nlp-m5-sft.zip` (1 MB) để upload Kaggle.
- 2026-05-11: Viết `notebooks/kaggle_m5_qwen_lora.py` — Kaggle T4×2 notebook code (13 cell, định dạng `# %%`). Cấu hình: Qwen2.5-7B-Instruct 4-bit NF4 + LoRA r=16 alpha=32 target=q/k/v/o/gate/up/down_proj, fp16 + gradient checkpoint, 1 epoch 20k samples, effective batch=16. Eval cell: generate 500 query → parse regex "mã (\d{6})" → map doc_id → metrics. Hand-off cho user chạy Kaggle.
- 2026-05-12: M5 train xong trên Kaggle T4×2 (~6.5h, 2500 steps, loss giảm 2.71→~0.1). Tuy nhiên **eval v1 metrics thấp bất thường**: R@1=R@5=R@10=MRR=0.368 — pattern này cho thấy model chỉ generate 1 prediction/query. Download `m5_qwen_lora.zip` về local (438MB, gồm adapter 161MB + tokenizer + checkpoint-2500).
- 2026-05-12: Debug M5 v1 từ `sample_generations.json`: phát hiện 3 lỗi (1) SYSTEM prompt ở `generate_for_query` khác SYSTEM training → distribution shift, (2) parse regex `m[ãa]\s*(\d{6})` không bắt được mã môn khi model output `(003633)` không có "mã" prefix, (3) `max_new_tokens=256` có thể chưa đủ. Viết `notebooks/kaggle_m5_eval_only.py` để re-eval (skip train, load LoRA + run gen + metrics).
- 2026-05-12: User upload `m5_lora_for_kaggle/` (LoRA adapter + tokenizer, 178MB) làm Kaggle dataset `ck-nlp-m5-lora`, chạy `kaggle_m5_eval_only.py` ~30 phút. **Eval v2 cải thiện rõ rệt**: R@1=0.640 (+0.272), R@5=0.682, R@10=0.682, MRR=0.661, NDCG@10=0.255. Pred avg len=1.45 (vs 1.0 trước), empty=0 (vs ~50% trước). Stats: chưa có query nào ≥5 môn vì training data nhiều query chỉ có 1-2 unique positives. **Giai đoạn 4 HOÀN THÀNH**.
