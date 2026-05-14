# STATUS.md — Trạng thái Dự án

> File này được Claude Code tự cập nhật. Đầu phiên đọc để có context, cuối task ghi lại.
> Quy ước cập nhật: xem `CLAUDE.md` mục 7.

**Cập nhật lần cuối:** 2026-05-14 (Stage 6 rewrite chat UX — upload bảng điểm + multi-turn chat + giải thích)

---

## Giai đoạn hiện tại

**Giai đoạn 6 — Đánh giá & Triển khai** 🔄 ĐANG LÀM

Đã hoàn thành 3/7 task con: notebook Kaggle RAG-vs-noRAG, aggregator local, RAGAS-proxy local. Còn lại: chạy Kaggle (user) → aggregate → RAGAS → FastAPI → Streamlit → 50 case manual.

---

## Đang làm

- **Rewrite UX (2026-05-14)** sau khi user clarify scope: chatbot phải là **chat multi-turn với upload bảng điểm + giải thích chi tiết**, không phải REST single-turn nhập tay mã môn. Đã viết lại 6 file: `src/data/profile_loader.py` (mới), `src/generation/prompt_templates.py` (thêm `build_chat_user_message`), `src/generation/generator.py` (thêm method `chat()`), `src/rag_pipeline.py` (thêm `chat() + ChatResult`), `api/main.py` (endpoint POST /chat), `frontend/app.py` (rewrite hoàn toàn: file uploader + st.chat_message). Backend `/answer` cũ vẫn giữ cho backward compat.
- Test E2E stub mode 2 turn với SV thật (IS_TuChon.csv, IDSinhVien=1677250): OK, turn 1 197ms, turn 2 72ms.
- Pending: test E2E với `USE_LLM=1` (Qwen) + manual eval 50 case nếu user còn cần.

---

## Sắp làm (Giai đoạn 6 — Đánh giá & Triển khai)

**Chiến lược 2 môi trường (plan B):**
- **Local**: retrieval + constraint + RAGAS + Hit@K (đã sẵn sàng).
- **Kaggle T4×2 (16GB)**: M5 Qwen-7B generation only (tránh OOM 8GB local).

**Thứ tự task con + trạng thái**:
1. ✅ `python -m src.evaluation.export_for_kaggle --n 100 --mode warm` → `data/kaggle_export/{rag,norag}_inputs_warm.jsonl` (đã xong từ Stage 5).
2. ✅ `notebooks/kaggle_m5_rag_eval.py` — load Qwen-7B+LoRA, đọc 2 JSONL, generate cả 2 variant, lưu `predictions_{rag,norag}.jsonl` + `eval_summary.json`.
3. ✅ `src/evaluation/aggregate_rag_vs_norag.py` — Hit@1/5/10, MRR, NDCG@10 cho 2 variant + delta, output `data/evaluation/rag_vs_norag_metrics.json`.
4. ✅ `src/evaluation/ragas_eval.py` — RAGAS-proxy (không API LLM): context_recall, context_precision (AP), faithfulness (proxy = |pred ∩ context|/|pred|), answer_relevancy (cos E5).
5. ✅ User chạy Kaggle T4×2 (~25 phút) → 100 predictions/variant + bảng so sánh ở mục "Kết quả Giai đoạn 6".
6. ✅ FastAPI backend `api/main.py` — expose `pipeline.answer()`; lifespan load 1 lần; env toggle `USE_LLM`, `USE_RERANKER`, `TOP_K`. Routes: `/health`, `/info`, `/answer`, `/docs`. Verify stub: 26ms warm, response tiếng Việt OK.
7. ✅ Streamlit UI `frontend/app.py` — single page: sidebar (ngành/HK/GPA/định hướng/môn đã hoàn thành) + main (query + bảng recommendations + expander top retrieved + constraint warnings). Backend URL qua env `STREAMLIT_API_URL` (mặc định http://127.0.0.1:8000).
8. 🟡 **Code đã xong**: sinh template + parser. Chờ user fill rating 1-5 cho 50 case ở `data/evaluation/manual_satisfaction_template.md`.

**Optional**: nếu user fix torch CUDA + có VRAM ≥10GB, có thể dùng `--use-llm` local thay Kaggle.

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

## Checklist Giai đoạn 5 — ✅ HOÀN THÀNH

- [x] Build FAISS + BM25 hybrid retrieval — `src/retrieval/{dense_retriever.py, bm25_retriever.py, hybrid_retriever.py}` (RRF fusion k=60, mặc định weights 1.0/1.0)
- [x] Implement constraint checker (prereq + TC) — `src/retrieval/constraint_checker.py` (load 5 graph + regulations.json, kiểm a/b prereq + song hành c + tổng TC 12-30/HK)
- [x] Integrate reranker vào pipeline — `src/retrieval/reranker.py` (wrap M3 PhoBERT cross-encoder, mặc định OFF vì M3 làm giảm MRR theo Stage 4)
- [x] Kết nối LLM generator — `src/generation/{generator.py, prompt_templates.py}` (Qwen2.5-7B 4-bit + LoRA local; có StubGenerator fallback không cần GPU)
- [x] End-to-end test với 100 câu hỏi mẫu — `src/evaluation/rag_e2e.py` + metrics ở `src/evaluation/metrics.py` (cân bằng 20/ngành từ test set Stage 3)
- [x] Orchestrator: `src/rag_pipeline.py` (RagPipeline.answer() trả RagResult với trace mọi bước)

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

## Kết quả Giai đoạn 5 — E2E pipeline

**File output**: `data/evaluation/rag_e2e_stub.json` (mode=stub, không cần GPU)

**Pipeline**:
```
query → HybridRetriever (M4 dense + BM25 sparse, RRF k=60)
      → ConstraintChecker (lọc thiếu prereq, cảnh báo TC ngoài [12,30])
      → format_context (top-K valid → text block)
      → Generator (Qwen-7B+LoRA *hoặc* Stub) → response
      → parse_recommendations → [(doc_id, ...)]
```

**Eval 2 mode (cold/warm), 100 query cân bằng 20/ngành, stub generator, ~3s/mode**:

| Layer | Mode | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| Retrieval | cold | 0.340 | 0.820 | **0.970** | 0.525 | 0.437 |
| Retrieval | **warm** | **0.550** | **0.850** | 0.960 | **0.677** | **0.589** |
| Generation | cold | 0.340 | 0.820 | 0.820 | 0.503 | 0.294 |
| Generation | warm | 0.550 | 0.850 | 0.850 | 0.660 | 0.405 |

**Δ warm − cold**: R@1 **+0.21**, MRR **+0.15**, NDCG@10 **+0.15** — chứng minh constraint-aware pipeline hoạt động đúng khi có profile.

**So sánh với baseline Stage 4** (cùng test 500 pairs, nhưng không phải subset 100):
- M4 raw R@10=0.998 → warm pipeline R@10=0.960 (giảm 0.038 do filter `da_hoan_thanh`).
- M4 raw R@1=0.706 → warm pipeline R@1=0.550 (giảm 0.156 — cost của constraint filter; ưu tiên môn hợp lệ hơn cosine cao nhất).

**Constraint metrics**:
- `constraint_satisfaction_rate=0.00` cả 2 mode — vì test set có violations dạng `da_hoan_thanh` (warm) hoặc `thieu_prereq` (cold). Đây không phải tệ — pipeline báo rõ môn nào không hợp lệ.
- `credit_load_validity≈0.23` — vì pipeline KHÔNG cắt top-K theo TC, chỉ cảnh báo. Giai đoạn 6 có thể thêm post-processing select-K-min-TC.

**File output**:
- `data/embeddings/test_with_profile.jsonl` — test set augmented với `hk_completed`, `hk_target`, `completed_ma_mon` (avg 50.2 môn/query).
- `data/evaluation/rag_e2e_cold_stub.json` + `..._warm_stub.json`.

**Khi `--use-llm`**: chưa chạy. Dự kiến generation NDCG@10 cao hơn stub (vì Qwen có thể recommend môn ngoài top-K context dựa trên training memory).

---

## Kết quả Giai đoạn 6 — RAG vs noRAG (100 query, M5 Qwen-7B+LoRA trên Kaggle T4×2)

**File output**:
- Predictions: `data/kaggle_export/stage6_rag_eval/predictions_{rag,norag}.jsonl`
- Metrics retrieval: `data/evaluation/rag_vs_norag_metrics.json`
- Metrics RAGAS-proxy: `data/evaluation/ragas_proxy.json`

**Retrieval metrics (Hit@K / MRR / NDCG@10)**:

| Metric | RAG | noRAG | Δ (RAG − noRAG) |
|---|---:|---:|---:|
| Hit@1 | **0.660** | 0.530 | **+0.130** |
| Hit@5 | **0.690** | 0.530 | **+0.160** |
| Hit@10 | **0.690** | 0.530 | **+0.160** |
| MRR | **0.675** | 0.530 | **+0.145** |
| NDCG@10 | **0.263** | 0.138 | **+0.125** |

**RAGAS-proxy** (không API LLM, theo yêu cầu giáo viên):

| Metric | RAG | noRAG | Δ |
|---|---:|---:|---:|
| context_recall (|gold∩retrieved|/|gold|) | 0.773 | 0.773 | 0.000 |
| context_precision (AP@K) | 0.537 | 0.537 | 0.000 |
| faithfulness (|pred∩context|/|pred|) | **0.955** | 0.690 | **+0.265** |
| answer_relevancy (cos E5(q, response)) | **0.444** | 0.169 | **+0.275** |

**Đọc**:
- RAG cải thiện rõ rệt tất cả metric (Hit@1 +13đ, MRR +14.5đ, NDCG@10 +12.5đ).
- `faithfulness` +26.5đ: RAG giúp model gợi ý môn nằm trong context cung cấp (less hallucination).
- `answer_relevancy` +27.5đ: response RAG bám sát query hơn (do prompt có context guide).
- `context_recall/precision` bằng nhau ở 2 variant vì cùng dùng 1 retriever — chỉ khác có/không nhét context vào prompt.
- Pred length: rag avg=1.67, norag avg=0.97 (3 query noRAG output rỗng) → noRAG dễ "lười" hơn khi không có context.

**Kết luận E2E**: RAG pipeline (M4 retriever + constraint + M5 generator) thắng baseline noRAG (M5 generator alone) ở mọi metric, chứng minh giá trị của context augmentation cho task tư vấn học phần.

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

### Giai đoạn 5 (cài 2026-05-13)
- Tận dụng deps đã cài (rank-bm25, faiss-cpu, transformers, sentence-transformers, peft).
- `bitsandbytes==0.49.2` + `accelerate==1.13.0` — đã cài để dùng 4-bit Qwen.
- ⚠️ Trước đây phát hiện torch 2.11.0+cpu (CPU-only) — Stage 6 đã reinstall xong (xem dưới).

### Giai đoạn 6 (cài 2026-05-13)
- `torch==2.12.0.dev20260408+cu128` (reinstall từ CPU build, pytorch.org nightly cu128).
- `peft==0.19.1` (cho LoRA inference local).
- `fastapi==0.136.1`, `uvicorn[standard]==0.46.0`, `python-dotenv==1.2.2`, `streamlit==1.57.0`.
- HF_TOKEN trong `.env` (gitignored — repo này không phải git nên thực tế chưa có .gitignore).

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
- 2026-05-12: Bắt đầu Giai đoạn 5. Chốt kiến trúc với user: (1) Retriever = M4 no_gnn + BM25 hybrid (RRF), (2) Reranker M3 = tuỳ chọn (mặc định OFF), (3) Generator = M5 Qwen-7B + LoRA local 4-bit.
- 2026-05-12: Viết module `src/retrieval/`: `bm25_retriever.py` (BM25Okapi + tokenizer tiếng Việt thô), `dense_retriever.py` (FusionRetriever no_gnn + E5, doc embeddings precompute), `hybrid_retriever.py` (RRF fusion k=60), `constraint_checker.py` (load 5 graph + regulations.json, kiểm a/b/c prereq + TC 12-30), `reranker.py` (M3 PhoBERT cross-encoder optional).
- 2026-05-12: Viết module `src/generation/`: `prompt_templates.py` (SYSTEM khớp Stage 4 training + USER template profile/context/question), `generator.py` (QwenGenerator 4-bit NF4 + LoRA, StubGenerator fallback, parse_recommendations regex).
- 2026-05-12: Viết `src/rag_pipeline.py` orchestrator (RagPipeline.answer() → RagResult với trace retrieve/rerank/constraint/generate).
- 2026-05-12: Viết `src/evaluation/{metrics.py, rag_e2e.py}` — eval 100 query cân bằng 20/ngành từ test set Stage 3. Chạy stub mode 3.1s. Retrieval R@10=0.97, R@1=0.34, MRR=0.525.
- 2026-05-12: Phát hiện R@1 thấp do eval cold-start (không truyền `completed`). Triển khai phương án B+C: (B) viết `src/evaluation/augment_test_set.py` parse `hk_completed`/`hk_target` từ query bằng 5+3 regex pattern (match 500/500), tạo `data/embeddings/test_with_profile.jsonl` có thêm trường `completed_ma_mon` (avg 50.2 môn/query). (C) update `rag_e2e.py` thêm flag `--mode cold|warm|both`; warm mode truyền `completed` vào pipeline. Kết quả warm vs cold: R@1 +0.21 (0.34→0.55), MRR +0.15, NDCG@10 +0.15 — chứng minh constraint-aware pipeline đúng nguyên lý.
- 2026-05-13: Cài bitsandbytes 0.49.2 + accelerate 1.13.0 để validate `--use-llm`. Phát hiện venv hiện tại có torch 2.11.0+**cpu** (không CUDA), RTX 5070 8GB. Plan A (reinstall torch CUDA) rủi ro break torch-geometric + 8GB VRAM chật cho Qwen-7B. Chốt **plan B**: local làm retrieval+constraint, Kaggle T4×2 chạy M5 generation (môi trường Stage 4 đã verify).
- 2026-05-13: Viết `src/evaluation/export_for_kaggle.py` — chạy pipeline (use_llm=False) → export 2 JSONL `data/kaggle_export/{rag,norag}_inputs_warm.jsonl` với schema 11 trường (idx, query, nganh, hk_completed, hk_target, gold, retrieved_valid, context_doc_ids, system_prompt, user_message, variant). Test 20 queries OK: rag user_message=1176 chars (có context block), norag=420 chars (chỉ profile+question). Sẵn sàng upload Kaggle. **Giai đoạn 5 HOÀN THÀNH**.
- 2026-05-13: Bắt đầu Giai đoạn 6. Chốt với user: làm Kaggle notebook + RAGAS-proxy local trước (block các task khác).
- 2026-05-13: Viết `notebooks/kaggle_m5_rag_eval.py` — load Qwen2.5-7B 4-bit + LoRA, đọc cả 2 JSONL (`rag_inputs_warm.jsonl`, `norag_inputs_warm.jsonl`), apply Qwen chat template với `system_prompt`+`user_message` đã render sẵn, generate greedy (max_new_tokens=512), parse `predicted_doc_ids` bằng regex 6-digit, lưu `predictions_{rag,norag}.jsonl` + `eval_summary.json` ở `/kaggle/working/stage6_rag_eval/`. Sanity check assert 2 file align theo idx + gold.
- 2026-05-13: Viết `src/evaluation/aggregate_rag_vs_norag.py` — đọc 2 file predictions, tái dùng `metrics.compute_retrieval_metrics`, in bảng markdown so sánh + ghi `data/evaluation/rag_vs_norag_metrics.json`. Smoke test 10 query stub OK.
- 2026-05-13: Viết `src/evaluation/ragas_eval.py` — RAGAS-proxy (không API LLM, theo yêu cầu giáo viên): (1) `context_recall` = |gold ∩ retrieved|/|gold|; (2) `context_precision` = AP@K dựa retrieved; (3) `faithfulness` proxy = |predicted ∩ context|/|predicted|; (4) `answer_relevancy` = cos(E5(query), E5(response)) với E5 fine-tuned Stage 3. Smoke test với E5 (199 weight load) chạy OK; flag `--e5 ''` để skip embedding metric.
- 2026-05-13: Re-export 100 query (`--n 100`), user chạy `kaggle_m5_rag_eval.py` trên Kaggle T4×2 ~25 phút → `predictions_{rag,norag}.jsonl` (100 dòng/file) + `eval_summary.json`. Kết quả: **RAG Hit@1=0.66 vs noRAG 0.53 (+0.13)**, MRR +0.145, NDCG@10 +0.125; RAGAS-proxy faithfulness +0.265, answer_relevancy +0.275 — pipeline RAG thắng baseline ở mọi metric. Chi tiết ở mục "Kết quả Giai đoạn 6".
- 2026-05-13: Reinstall `torch` từ CPU build (2.11.0+cpu) sang **CUDA build 2.12.0.dev20260408+cu128** từ pytorch.org nightly để dùng GPU RTX 5070 Laptop (Blackwell sm_120, 8.55GB). Cài thêm `peft 0.19.1`. Backup pip freeze cũ ở `requirements_before_torch_cuda_reinstall.txt`. Verify: `torch.cuda.is_available()=True`, full stack import OK (torch, transformers, sentence_transformers, peft, bitsandbytes, faiss).
- 2026-05-13: Cài thêm `fastapi 0.136.1`, `uvicorn[standard] 0.46.0`, `python-dotenv 1.2.2`, `streamlit 1.57.0` (đã có sẵn trong nhóm `# API & UI` của requirements.txt).
- 2026-05-13: Viết `api/main.py` FastAPI backend — wrap `RagPipeline.answer()` qua HTTP. Pipeline load 1 lần ở lifespan startup. Env toggle: `USE_LLM`, `USE_RERANKER`, `LOAD_4BIT`, `TOP_K`, `HF_TOKEN` (đọc từ `.env`), `API_HOST`, `API_PORT`. Endpoints: `GET /health`, `GET /info`, `POST /answer`, `/docs` (Swagger). Schema `AnswerRequest`/`AnswerResponse`/`RetrievedItem`/`ConstraintInfo` qua Pydantic. CORS allow-all để Streamlit local connect. Verify stub mode end-to-end: pipeline load <10s, `/answer` cold 144ms / warm 26ms, response tiếng Việt UTF-8 đúng, constraint warnings hiển thị.
- 2026-05-13: Viết `frontend/app.py` Streamlit UI — single page chat. Sidebar: ngành/HK/GPA/định hướng/môn đã hoàn thành (text area parse 6-digit). Main: query input → POST `/answer` → markdown response + bảng pandas recommendations + expander Top retrieved (debug) + expander Constraint check (warnings). Backend URL `STREAMLIT_API_URL` (mặc định http://127.0.0.1:8000). Cache `/info` 30s. Verify AST + imports OK.
- 2026-05-13: Bắt đầu smoke test QwenGenerator local (`python -m src.generation.generator`) để verify VRAM 8GB đủ cho Qwen-7B 4-bit. Lần đầu phải tải base Qwen2.5-7B-Instruct (~15GB) từ HuggingFace. Tốc độ unauthenticated ~75 KB/s → set HF_TOKEN vào `.env` tăng lên ~1 MB/s, nhưng bị disconnect ở 12GB/15GB (file 3,4 dở 50%).
- 2026-05-13: Restart smoke test với HF_TOKEN → resume 4 file tải xong trong 2:07 phút (tốc độ cuối ~25 MB/s — HF rate-limit unauthenticated nặng hơn nhiều). Load 339 weights vào VRAM trong 8s. Sửa bug encoding cp1252 ở `_smoke_test()` (thêm `sys.stdout.reconfigure(encoding='utf-8')`). **Smoke test SUCCESS trên RTX 5070 8GB**: generate response tiếng Việt thật ("Lập trình phân tích dữ liệu 2", "Lập trình GUI với Qt Framework"). VRAM 8GB đủ cho Qwen-7B 4-bit + LoRA + KV cache → user có thể `USE_LLM=1` cho FastAPI local.
- 2026-05-13: Viết `src/evaluation/build_manual_eval_template.py` — sinh markdown 50 case stratified 10/ngành từ `predictions_rag.jsonl` (Qwen-7B+LoRA thật từ Kaggle). Mỗi case gồm: query, profile, gold answers + tên môn, bot response, bot recommendations parse + tên, top 5 retrieved, ô đánh giá 1-5 (checkbox), ghi chú. Output: `data/evaluation/manual_satisfaction_template.md` (126 KB).
- 2026-05-13: Viết `src/evaluation/parse_manual_eval.py` — đọc lại file sau khi user fill `[x]`, regex lấy rating + note từng case, output bảng phân bố điểm + avg/ngành + satisfaction rate (≥4 / total) + JSON tóm tắt. Smoke test (chưa fill): n_filled=0, warn 50/50 chưa tick — OK.
- 2026-05-13: **Giai đoạn 6 code HOÀN THÀNH**. Còn lại user action: (a) fill 50 case ở `manual_satisfaction_template.md`, (b) chạy `python -m src.evaluation.parse_manual_eval` để tổng hợp.
- 2026-05-13: Fix bug khi user chạy API với `USE_LLM=1` lần đầu — accelerate báo lỗi `ValueError: Some modules are dispatched on the CPU or the disk` vì E5+M4 retriever đã chiếm VRAM trước khi Qwen-7B 4-bit dispatch. Sửa `rag_pipeline.load_default()` — khi `use_llm=True` thì ép retriever device=cpu (E5 inference 1 query trên CPU ~50-100ms, chấp nhận được). Verify lại API USE_LLM=1: pipeline load 27s (E5 CPU <1s + Qwen 4-bit 10s + LoRA), `/answer` end-to-end 8.7s/query, Qwen trả "Lập trình phân tích dữ liệu 2" + "NoSQL MongoDB" — phù hợp CS HK5 AI/ML.
- 2026-05-14: User clarify scope: chatbot phải là **chat multi-turn với upload bảng điểm + giải thích chi tiết**, không phải REST single-turn lấy profile từ form. Trước đó mình đoán sai UX (textarea nhập tay list mã môn, single-turn) — user thử thì kết quả tệ vì cold-start + query OOD. Rewrite 6 file: `src/data/profile_loader.py` (parse `<NGANH>_TuChon.xlsx` → `StudentProfile` với GPA, top điểm cao, môn đã hoàn thành); update `prompt_templates.py` (thêm `build_chat_user_message` inject summary bảng điểm + yêu cầu giải thích cụ thể); update `generator.py` (`QwenGenerator.chat(messages)` + `StubGenerator.chat()`); update `rag_pipeline.py` (`chat(profile, messages) → ChatResult` + retrieve theo query mới nhất, inject profile summary ở turn 1, Qwen tự nhớ qua history); thêm endpoint `POST /chat` ở `api/main.py`; rewrite `frontend/app.py` Streamlit với `st.file_uploader` + `st.chat_message` + `st.session_state.messages`. Test E2E stub 2 turn với SV thực `IDSinhVien=1677250` (IS, 10 môn, GPA 3.33/4): turn 1 197ms, turn 2 72ms — OK.
