# STATUS.md — Trạng thái Dự án

> File này được Claude Code tự cập nhật. Đầu phiên đọc để có context, cuối task ghi lại.
> Quy ước cập nhật: xem `CLAUDE.md` mục 7.

**Cập nhật lần cuối:** 2026-05-11 (Hoàn tất Giai đoạn 1)

---

## Giai đoạn hiện tại

**Giai đoạn 1 — Thu thập & Tiền xử lý Dữ liệu** ✅ HOÀN THÀNH

---

## Đang làm

- (Giai đoạn 1 hoàn tất. Sẵn sàng cho Giai đoạn 2.)

---

## Sắp làm (Giai đoạn 2 — Embedding & RAG)

1. Xác định embedding model (mặc định: `sentence-transformers/distiluse-base-multilingual-cased-v2`).
2. Xây dựng course embedding từ curriculum (tên, mô tả, tiên quyết).
3. Xây dựng student profile embedding từ điểm số và hồ sơ.
4. Viết module retrieval (BM25 + vector similarity).
5. Tạo training data cho LLM generation (QA pairs từ regulations + student data).

---

## Dữ liệu hiện có

| Loại | Ngành | File | Trạng thái |
|---|---|---|---|
| Curriculum HTML | CS | `data/raw/curriculum/CS_curriculum.html` | ✅ Có |
| Curriculum HTML | IS | `data/raw/curriculum/IS_curriculum.html` | ✅ Có |
| Curriculum HTML | DS | `data/raw/curriculum/DS_curriculum.html` | ✅ Có |
| Curriculum HTML | SE | `data/raw/curriculum/SE_curriculum.html` | ✅ Có |
| Curriculum HTML | IT | `data/raw/curriculum/IT_curriculum.html` | ✅ Có |
| Grades CSV | CS | `data/raw/grades/CS_TuChon.csv` | ✅ Có (4799 rows, 500 SV) |
| Grades CSV | IS | `data/raw/grades/IS_TuChon.csv` | ✅ Có (4670 rows, 591 SV, 655 null điểm) |
| Grades CSV | DS | `data/raw/grades/DS_TuChon.csv` | ✅ Có (4620 rows, 500 SV) |
| Grades CSV | SE | `data/raw/grades/SE_TuChon.csv` | ✅ Có (4257 rows, 500 SV) |
| Grades CSV | IT | `data/raw/grades/IT_TuChon.csv` | ✅ Có (3703 rows, 500 SV) |
| Quy định học vụ | — | `data/raw/regulations/quy_dinh_hoc_vu.txt` | ✅ Có |

---

## Checklist Giai đoạn 1 — ✅ HOÀN THÀNH

- [x] Parse HTML chương trình khung CS → JSON + graph
- [x] Parse HTML chương trình khung IS → JSON + graph
- [x] Parse HTML chương trình khung DS → JSON + graph
- [x] Parse HTML chương trình khung SE → JSON + graph
- [x] Parse HTML chương trình khung IT → JSON + graph
- [x] Build prerequisite graph (NetworkX) cho từng ngành
- [x] Validate cả 5 graph (không cycle, không thiếu prereq)
- [x] Load và chuẩn hóa 5 file CSV điểm (tổng 21.394 records, 2.587 SV)
- [x] Build student profile từ 5 ngành
- [x] Trích xuất quy định học vụ → JSON structured
- [x] EDA notebook với dữ liệu hiện có (`notebooks/01_data_exploration.ipynb`)
- [x] Báo cáo chất lượng dữ liệu (16 phần, không lỗi quan trọng)

---

## Phụ thuộc đã cài

- `lxml>=4.9` — dùng cho BeautifulSoup parser
- `beautifulsoup4>=4.12` — parse HTML curriculum
- `pyarrow>=14.0` — lưu file .parquet

---

## Blockers & Câu hỏi cho user

- ⏳ Chưa quyết định LLM cuối cùng cho generation: Vistral-7B vs Qwen2.5-7B-Instruct (sẽ quyết khi đến giai đoạn 4).

---

## Nhật ký

- 2026-05-09: Khởi tạo dự án. Tạo `CLAUDE.md`, `STATUS.md`, `project_instructions.md`. Đã có sẵn 5 file HTML curriculum và `IS_TuChon.xlsx`.
- 2026-05-09: Viết `src/data/parser.py` — parse HTML chương trình khung → JSON. File: `src/data/parser.py`. Ghi chú: parse đúng điều kiện tiên quyết, pad mã môn 6 số, nhận diện môn có dấu * (khong_tinh_gpa).
- 2026-05-09: Viết `src/data/graph_builder.py` — build NetworkX DiGraph từ JSON curriculum. File: `src/data/graph_builder.py`. Ghi chú: validate no-cycle + no-missing-node, tất cả 5 ngành OK. Output: `data/processed/curriculum_graph/<NGANH>_prereq_graph.gpickle`.
- 2026-05-09: Parse thành công 5 ngành (CS=91 môn, IS=88, DS=86, SE=88, IT=85). Build 5 graph (CS=37 edges, IS=21, DS=15, SE=41, IT=11). Không có cycle, không thiếu node.
- 2026-05-11: User cung cấp đủ dữ liệu: 5 file CSV grades (CS/DS/SE/IT dùng sep="|", IS dùng sep=",") và quy_dinh_hoc_vu.txt.
- 2026-05-11: Viết `src/data/preprocessor.py` — load + clean 5 CSV → parquet/csv. Tổng 21.394 records, 2.587 SV. IS có 655 null điểm đã loại. File: `src/data/preprocessor.py`. Output: `data/processed/student_profiles/`.
- 2026-05-11: Viết `src/data/regulation_parser.py` — trích xuất quy định IUH → JSON (9 nhóm quy tắc: TC đăng ký 12–30/HK, GPA tốt nghiệp ≥2.0, bảng quy đổi điểm, cảnh báo học tập, v.v.). File: `src/data/regulation_parser.py`. Output: `data/processed/regulations.json`.
- 2026-05-11: Viết EDA notebook `notebooks/01_data_exploration.ipynb` — 16 phần phân tích: thống kê sinh viên, phân bố GPA, chất lượng dữ liệu, tần suất môn học, prerequisite graph, quy định học vụ, xếp loại, graduation threshold, và báo cáo chất lượng. Ghi chú: Giai đoạn 1 HOÀN THÀNH, không lỗi quan trọng, sẵn sàng cho Giai đoạn 2 (Embedding & RAG).
