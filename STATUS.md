# STATUS.md — Trạng thái Dự án

> File này được Claude Code tự cập nhật. Đầu phiên đọc để có context, cuối task ghi lại.
> Quy ước cập nhật: xem `CLAUDE.md` mục 7.

**Cập nhật lần cuối:** 2026-05-11 (Hoàn tất Giai đoạn 2)

---

## Giai đoạn hiện tại

**Giai đoạn 2 — Tăng cường Dữ liệu** ✅ HOÀN THÀNH

---

## Đang làm

- (Giai đoạn 2 hoàn tất. Sẵn sàng cho Giai đoạn 3 — Embedding.)

---

## Sắp làm (Giai đoạn 3 — Embedding)

1. Fine-tune PhoBERT-base-v2 với contrastive loss trên QA pairs.
2. Fine-tune multilingual-E5-base với hard negatives (từ `negative_samples`).
3. Train GCN/GAT trên prerequisite graph (cần cài torch-geometric — sẽ hỏi user).
4. Xây dựng hybrid embedding (late fusion text + graph).
5. Đánh giá tất cả phương án trên test set 500 query-doc pairs.
6. Chọn embedding tốt nhất build FAISS index.

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

---

## Blockers & Câu hỏi cho user

- ⏳ Sẽ cần cài `torch-geometric` / `dgl` cho GNN ở Giai đoạn 3 (M4). Sẽ hỏi khi đến.
- ⏳ Chưa quyết định LLM cuối cùng cho generation: Vistral-7B vs Qwen2.5-7B-Instruct (Giai đoạn 4).
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
