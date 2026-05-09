# PROJECT INSTRUCTIONS — RAG Chatbot Tư Vấn Đăng Ký Học Phần

## 1. MÔ TẢ DỰ ÁN

Xây dựng hệ thống chatbot RAG (Retrieval-Augmented Generation) tư vấn chiến lược đăng ký học phần cho sinh viên đại học, hỗ trợ 5 ngành: **CS (Computer Science), IS (Information Systems), DS (Data Science), SE (Software Engineering), IT (Information Technology)**.

Chatbot phải:
- Gợi ý học phần tự chọn phù hợp với profile, điểm số, mục tiêu của từng sinh viên
- Đảm bảo đúng ràng buộc tiên quyết (học trước/tiên quyết/song hành)
- Không vượt quá giới hạn tín chỉ mỗi học kỳ theo quy định học vụ
- Trả lời bằng tiếng Việt tự nhiên, giải thích rõ lý do gợi ý

---

## 2. CẤU TRÚC DỮ LIỆU

### 2.1 Chương trình khung (Curriculum)

**Nguồn:** HTML từ trang web trường, lấy bằng F12 → copy innerHTML bảng `#chuongtrinhkhungtbl`

**Cấu trúc sau khi parse (JSON):**
```json
{
  "nganh": "CS",
  "hoc_ky": [
    {
      "hk_so": 1,
      "tong_tc": 11,
      "hoc_phan": [
        {
          "ma_mon": "004247",
          "ten_mon": "Nhập môn Lập trình",
          "ma_hoc_phan": "4220004247",
          "loai": "bat_buoc",
          "so_tc": 2,
          "so_tiet_lt": 0,
          "so_tiet_th": 60,
          "dieu_kien": [],
          "loai_dieu_kien": null
        },
        {
          "ma_mon": "001782",
          "ten_mon": "Kỹ thuật lập trình",
          "loai": "bat_buoc",
          "so_tc": 3,
          "dieu_kien": ["004247"],
          "loai_dieu_kien": "a"
        }
      ]
    }
  ]
}
```

**Ký hiệu điều kiện học phần:**
- `(a)` = học trước — phải hoàn thành (đạt) môn đó trước
- `(b)` = tiên quyết — điều kiện bắt buộc
- `(c)` = song hành — học cùng lúc được

**Ngành đã có dữ liệu chương trình khung:** CS, IS, DS, SE, IT (đầy đủ HTML, chưa parse).
**Ngành cần bổ sung xlsx điểm:** CS, DS, SE, IT (hiện chỉ có IS)

### 2.2 Dữ liệu điểm sinh viên học phần tự chọn

**File:** `{NGANH}_TuChon.xlsx` (ví dụ: `IS_TuChon.xlsx`, `CS_TuChon.xlsx`)

**Schema cột:**
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `IDSinhVien` | int | Mã số sinh viên (ẩn danh) |
| `MaMonHoc` | str | Mã môn học (6 ký số, ví dụ `003197`) |
| `TenMonHoc` | str | Tên môn học |
| `TenDot` | str | Học kỳ học, format `HK{n} ({year}-{year})` ví dụ `HK1 (2019-2020)` |
| `DiemTongKet` | float | Điểm tổng kết (thang 10) |
| `ThuocKCNTT` | int | 1 = thuộc khối kiến thức CNTT, 0 = không |

**Ngành đã có file:** IS
**Ngành cần bổ sung:** CS, DS, SE, IT — mục tiêu ~100.000 bản ghi tổng cộng

### 2.3 Quy định học vụ

**Cần thu thập:**
- Số tín chỉ tối đa được đăng ký mỗi học kỳ (thường 25 TC)
- Số tín chỉ tối thiểu để không bị cảnh báo học vụ (thường 14 TC)
- Điều kiện xét tốt nghiệp (tổng TC, GPA tối thiểu)
- Quy tắc đăng ký học phần cải thiện, học lại
- Các môn có dấu `*` không tính vào GPA (Giáo dục thể chất, Quốc phòng)

---

## 3. KIẾN TRÚC HỆ THỐNG

```
[Thu thập dữ liệu]
    HTML chương trình khung × 5 ngành
    xlsx điểm tự chọn × 5 ngành  
    Quy định học vụ (text/PDF)
        ↓
[Tiền xử lý]
    Parser HTML → Prerequisite Graph (NetworkX)
    Chuẩn hóa xlsx → Pandas DataFrame
    Trích xuất entity từ quy định học vụ
        ↓
[Tăng cường dữ liệu → ~100k mẫu]
    (1) Graph-based path sampling
    (2) Collaborative Filtering augmentation (SVD/ALS)
    (3) LLM synthetic QA generation
    (4) Constraint-based negative sampling
        ↓
[Embedding]
    Text: PhoBERT-base-v2 hoặc multilingual-E5 (fine-tuned)
    Graph: GCN/GAT trên prerequisite graph
    Hybrid: fusion text + graph embedding
    Evaluation: Recall@K, MRR, NDCG@10
        ↓
[Vector Store]
    FAISS (dense) + BM25 (sparse) → Hybrid retrieval
        ↓
[Deep Learning Models — nhiều hướng]
    Baseline: LSTM/GRU sequence model
    Mid:      Transformer cross-attention bi-encoder
    Advanced: GNN + Transformer fusion
    SOTA:     LLM fine-tuned (Vistral-7B / Qwen2.5-7B) với LoRA
        ↓
[RAG Pipeline]
    Query understanding → Intent classification
    Hybrid retrieval (dense + sparse)
    Constraint-aware reranking (kiểm tra tiên quyết, TC)
    Augmented generation với LLM + prompt template
        ↓
[Chatbot Interface]
    Web UI + đánh giá RAGAS / Hit@K / MRR
    Feedback loop → continual learning
```

---

## 4. DEPENDENCY GRAPH NGÀNH CS (ĐÃ PARSE)

Các chuỗi quan hệ tiên quyết quan trọng ngành CS:

```
004247 (Nhập môn LP) 
  → 001782 (Kỹ thuật LP)    [HK2]
      → 015029 (Xử lý ảnh)  [HK5, cùng với 001611]
  → 001611 (CTDL & GT)      [HK3]
      → 015029 (Xử lý ảnh)

001508 (Cấu trúc rời rạc)   [HK3]
  → 001954 (Trí tuệ nhân tạo) [HK4]
      → 015028 (Máy học)      [HK5]
          → 015030 (Học sâu)  [HK6]
  → 015029 + 015028 → 015032 (Nhận dạng mẫu) [HK6]
  → 001814 (Lý thuyết đồ thị) [HK5]

001922 (Hệ CSDL)            [HK3, prereq: 002793]
  → 003791 (Phân tích TK hệ thống) [HK4]
  → 014181 (NoSQL MongoDB)  [HK5 tự chọn]
      → 002804 (LP phân tán Java) [HK7]
  → 004099 (Khai thác DL)   [HK6, cùng 001954]

004119 (LP hướng đối tượng) [HK4 tự chọn]
  → 002876 (LP hướng sự kiện Java) [HK5]
      → 002804 (LP phân tán Java)  [HK7]
  → 002990 (LP hướng sự kiện .NET) [HK5]
  → 004021 (LP GUI Qt)       [HK5]
```

---

## 5. PHÂN LOẠI HỌC PHẦN TỰ CHỌN NGÀNH CS

Theo cluster định hướng nghề nghiệp:

| Cluster | Tên | Môn chính |
|---------|-----|-----------|
| AI/ML | Trí tuệ nhân tạo & Học máy | 001954, 015028, 015030, 015032, 015031 |
| CV | Computer Vision | 015029, 015031, 015032 |
| DB/BigData | Cơ sở dữ liệu & Dữ liệu lớn | 001922, 014181, 004099, 004024, 003953 |
| Java/Web | Lập trình Java & Web | 002876, 002804, 002399, 004227 |
| NET | Lập trình .NET | 002990, 003406 |
| Theory | Lý thuyết CS | 001814, 001730, 001758 |
| General | Kỹ năng mềm / Đại cương | 003622, 003633, 003664, v.v. |

---

## 6. YÊU CẦU KỸ THUẬT CHI TIẾT

### 6.1 Yêu cầu Dữ liệu

- **Quy mô:** ~100.000 mẫu training (QA pairs + recommendation samples)
- **Tiền xử lý bắt buộc:**
  - Chuẩn hóa tên môn học (lowercase, bỏ dấu thừa)
  - Parse `TenDot` → (`hk_so`: int, `nam_hoc`: str)
  - Xử lý missing values trong cột điểm
  - Loại bỏ duplicate (cùng SV, cùng môn, khác đợt → giữ điểm cao nhất)
- **Tăng cường — phải có ít nhất 3 trong 4 phương pháp sau:**
  1. Graph-based path sampling từ prerequisite graph
  2. Collaborative Filtering (SVD/ALS) → virtual student profiles
  3. LLM synthetic QA generation (dùng GPT-4o/Gemini API)
  4. Constraint-based negative sampling (kịch bản vi phạm)

### 6.2 Yêu cầu Embedding

- Phải fine-tune, không dùng off-the-shelf hoàn toàn
- Phải đánh giá bằng bộ test tự xây dựng (≥500 query-doc pairs)
- Metrics: Recall@1, Recall@5, Recall@10, MRR, NDCG@10
- Phải thử ít nhất 3 phương án và so sánh kết quả

### 6.3 Yêu cầu Mô hình học sâu

Phải triển khai theo thứ tự độ phức tạp tăng dần:

| Giai đoạn | Mô hình | Mục tiêu |
|-----------|---------|----------|
| M1 | LSTM/GRU | Baseline sequence recommendation |
| M2 | Transformer (bi-encoder) | Dense retrieval |
| M3 | Cross-attention reranker | Reranking top-K candidates |
| M4 | GNN + Transformer | Graph-aware recommendation |
| M5 | LLM fine-tuned (LoRA) | End-to-end generation |

### 6.4 Yêu cầu RAG Pipeline

- Hybrid retrieval: dense (embedding) + sparse (BM25) với RRF fusion
- Constraint checker: kiểm tra tiên quyết, tổng TC, đã học chưa
- Reranker: cross-encoder score + constraint penalty
- Generator: LLM với structured prompt chứa student profile + retrieved context

---

## 7. STACK CÔNG NGHỆ

```
Ngôn ngữ:        Python 3.10+
Data:            pandas, numpy, networkx, scikit-learn
NLP/Embedding:   transformers (HuggingFace), sentence-transformers, 
                 peft (LoRA), torch
Graph ML:        torch-geometric (PyG), dgl
Vector Store:    faiss-cpu / faiss-gpu, rank_bm25
LLM:             Vistral-7B hoặc Qwen2.5-7B-Instruct (local),
                 OpenAI API / Gemini API (cho synthetic data generation)
Experiment:      mlflow, wandb
Evaluation:      ragas, beir
API:             FastAPI
Frontend:        React hoặc Streamlit (prototype)
```

---

## 8. CẤU TRÚC THƯ MỤC DỰ ÁN

```
rag-chatbot-hocphan/
├── data/
│   ├── raw/
│   │   ├── curriculum/          # HTML chương trình khung × 5 ngành
│   │   │   ├── CS_curriculum.html
│   │   │   ├── IS_curriculum.html
│   │   │   ├── DS_curriculum.html
│   │   │   ├── SE_curriculum.html
│   │   │   └── IT_curriculum.html
│   │   ├── grades/              # xlsx điểm tự chọn × 5 ngành
│   │   │   ├── CS_TuChon.xlsx
│   │   │   ├── IS_TuChon.xlsx
│   │   │   ├── DS_TuChon.xlsx
│   │   │   ├── SE_TuChon.xlsx
│   │   │   └── IT_TuChon.xlsx
│   │   └── regulations/         # Quy định học vụ
│   │       └── quy_dinh_hoc_vu.txt
│   ├── processed/
│   │   ├── curriculum_graph/    # NetworkX graphs sau khi parse
│   │   ├── student_profiles/    # DataFrame profiles sinh viên
│   │   └── qa_pairs/            # QA pairs sau augmentation
│   └── augmented/
│       ├── synthetic_qa/        # LLM-generated QA
│       ├── cf_profiles/         # CF-augmented student profiles
│       └── negative_samples/    # Constraint violation samples
├── src/
│   ├── data/
│   │   ├── parser.py            # HTML → JSON curriculum
│   │   ├── graph_builder.py     # JSON → NetworkX prerequisite graph
│   │   ├── preprocessor.py      # Tiền xử lý xlsx điểm
│   │   └── augmentation/
│   │       ├── graph_sampler.py
│   │       ├── cf_augment.py
│   │       ├── llm_synthetic.py
│   │       └── negative_sampler.py
│   ├── embedding/
│   │   ├── text_embedder.py     # PhoBERT / E5 fine-tuning
│   │   ├── graph_embedder.py    # GCN/GAT node embedding
│   │   ├── hybrid_embedder.py   # Fusion
│   │   └── evaluator.py         # Recall@K, MRR, NDCG
│   ├── models/
│   │   ├── lstm_recommender.py
│   │   ├── transformer_retriever.py
│   │   ├── gnn_transformer.py
│   │   └── llm_finetuner.py     # LoRA fine-tuning
│   ├── retrieval/
│   │   ├── vector_store.py      # FAISS index
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py  # RRF fusion
│   │   └── constraint_checker.py
│   ├── generation/
│   │   ├── prompt_templates.py
│   │   └── generator.py
│   └── evaluation/
│       ├── ragas_eval.py
│       └── metrics.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_analysis.ipynb
│   ├── 03_augmentation_experiments.ipynb
│   ├── 04_embedding_evaluation.ipynb
│   └── 05_model_comparison.ipynb
├── api/
│   └── main.py                  # FastAPI endpoints
├── frontend/
│   └── app.py                   # Streamlit UI prototype
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── rag_config.yaml
├── tests/
└── requirements.txt
```

---

## 9. PROMPT TEMPLATE MẪU CHO RAG GENERATION

```python
SYSTEM_PROMPT = """Bạn là chatbot tư vấn đăng ký học phần thông minh tại trường đại học.
Nhiệm vụ của bạn là giúp sinh viên lập kế hoạch học tập tối ưu dựa trên:
- Chương trình khung ngành học và các ràng buộc tiên quyết
- Lịch sử điểm và định hướng nghề nghiệp của sinh viên
- Quy định học vụ (giới hạn tín chỉ, điều kiện tốt nghiệp)
Luôn giải thích rõ lý do gợi ý. Cảnh báo nếu sinh viên thiếu điều kiện tiên quyết."""

USER_TEMPLATE = """
## Thông tin sinh viên
- Ngành: {nganh}
- Học kỳ hiện tại: {hk_hien_tai}
- GPA tích lũy: {gpa}
- Môn đã hoàn thành: {da_hoan_thanh}
- Điểm học phần tự chọn đã học: {diem_tu_chon}
- Định hướng: {dinh_huong}

## Thông tin truy xuất (Retrieved Context)
{retrieved_context}

## Câu hỏi
{question}
"""
```

---

## 10. GIAI ĐOẠN THỰC HIỆN & CHECKLIST

### Giai đoạn 1 — Thu thập & Tiền xử lý Dữ liệu
- [ ] Parse HTML chương trình khung 5 ngành → JSON
- [ ] Build prerequisite graph (NetworkX) cho từng ngành
- [ ] Load và chuẩn hóa 5 file xlsx điểm
- [ ] Trích xuất quy định học vụ → structured format
- [ ] EDA: thống kê phân phối điểm, môn phổ biến, cluster sinh viên

### Giai đoạn 2 — Tăng cường Dữ liệu
- [ ] Graph-based path sampling → ~20k samples
- [ ] CF augmentation (SVD/ALS) → ~30k virtual profiles
- [ ] LLM synthetic QA generation → ~20k QA pairs
- [ ] Negative sampling → ~15k negative examples
- [ ] Validate chất lượng: kiểm tra ràng buộc hợp lệ
- [ ] Tổng kết: đạt ≥ 85k samples (target 100k)

### Giai đoạn 3 — Embedding
- [ ] Fine-tune PhoBERT-base-v2 với contrastive loss
- [ ] Fine-tune multilingual-E5-base với hard negatives
- [ ] Train GCN/GAT trên prerequisite graph
- [ ] Xây dựng hybrid embedding (late fusion)
- [ ] Đánh giá tất cả phương án trên test set 500 pairs
- [ ] Chọn embedding tốt nhất để build FAISS index

### Giai đoạn 4 — Mô hình học sâu
- [ ] M1: Train LSTM baseline, ghi nhận metrics
- [ ] M2: Train bi-encoder Transformer, so sánh với M1
- [ ] M3: Train cross-attention reranker
- [ ] M4: Implement GNN + Transformer fusion
- [ ] M5: LoRA fine-tune Vistral-7B / Qwen2.5-7B
- [ ] Ablation study: so sánh tất cả mô hình

### Giai đoạn 5 — RAG Pipeline
- [ ] Build FAISS + BM25 hybrid retrieval
- [ ] Implement constraint checker (tiên quyết + TC)
- [ ] Integrate reranker vào pipeline
- [ ] Kết nối LLM generator
- [ ] End-to-end test với 100 câu hỏi mẫu

### Giai đoạn 6 — Đánh giá & Triển khai
- [ ] Đánh giá RAGAS (faithfulness, answer relevancy, context recall)
- [ ] Đánh giá Hit@1, Hit@5, MRR, NDCG@10
- [ ] So sánh RAG vs non-RAG baseline
- [ ] Build FastAPI backend
- [ ] Build Streamlit UI prototype
- [ ] Thu thập phản hồi → chuẩn bị continual learning

---

## 11. METRICS ĐÁNH GIÁ

### Retrieval Metrics
- **Recall@K (K=1,5,10):** Tỷ lệ relevant document xuất hiện trong top-K kết quả
- **MRR (Mean Reciprocal Rank):** Vị trí trung bình của relevant document đầu tiên
- **NDCG@10:** Đánh giá có trọng số vị trí

### Generation Metrics (RAGAS)
- **Faithfulness:** Câu trả lời có trung thực với context không?
- **Answer Relevancy:** Câu trả lời có trả lời đúng câu hỏi không?
- **Context Recall:** Context retrieved có đủ thông tin không?
- **Context Precision:** Context retrieved có nhiễu không?

### Recommendation Metrics
- **Constraint Satisfaction Rate:** % gợi ý không vi phạm ràng buộc tiên quyết
- **Credit Load Validity:** % kế hoạch có tổng TC hợp lệ
- **User Satisfaction (manual):** Đánh giá thủ công trên 50 case

---

## 12. CÁC VẤN ĐỀ ĐẶC THÙ CẦN LƯU Ý

1. **Đa ngành:** Mã môn có thể giống nhau ở các ngành khác nhau → luôn kết hợp `(nganh, ma_mon)` làm key

2. **Môn không tính GPA:** Các môn có dấu `*` (Giáo dục thể chất, Quốc phòng, Chứng chỉ Tiếng Anh) → loại khỏi tính toán GPA

3. **Học phần tự chọn nhóm:** Mỗi học kỳ chỉ cần chọn đủ N tín chỉ từ danh sách tự chọn, không phải học tất cả → constraint là "chọn ≥ min_tc từ pool tự chọn kỳ đó"

4. **Tiến độ học tập cá nhân:** Sinh viên có thể học trễ hơn hoặc sớm hơn kế hoạch chuẩn → không cứng nhắc theo học kỳ, chỉ theo prerequisite graph

5. **Ngôn ngữ:** Tất cả output chatbot phải bằng tiếng Việt tự nhiên, có thể dùng thuật ngữ chuyên ngành CNTT nhưng cần giải thích nếu cần

6. **Cold start:** Sinh viên năm 1 chưa có lịch sử điểm → fallback sang content-based recommendation dựa trên mục tiêu nghề nghiệp

---

## 13. VÍ DỤ CASE STUDY (Dùng để test)

**Input:** Sinh viên ngành CS, đang chuẩn bị đăng ký HK5, đã hoàn thành tất cả môn HK1–HK4 theo chuẩn + đã học tự chọn `004119 (LP hướng đối tượng)` điểm 8.5, `001954 (Trí tuệ nhân tạo)` điểm 9.0. Định hướng: AI/ML.

**Output mong đợi:**
- Gợi ý bắt buộc: 001814, 001923, 015028, 015029, 015258
- Gợi ý tự chọn ưu tiên: 002876 (LP Java, prereq 004119 ✓), 014181 (NoSQL, prereq 001922 ✓)
- Giải thích: "Vì bạn định hướng AI/ML và điểm Trí tuệ nhân tạo cao (9.0), môn Máy học (015028) là nền tảng quan trọng để tiếp tục Học sâu ở HK6..."
- Cảnh báo nếu thêm tự chọn vượt 25 TC/kỳ

---

*Cập nhật lần cuối: Tháng 5/2026. Bổ sung dữ liệu ngành IS, DS, SE, IT khi có.*