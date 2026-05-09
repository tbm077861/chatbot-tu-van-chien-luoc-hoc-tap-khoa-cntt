# STATUS.md — Trạng thái Dự án

> File này được Claude Code tự cập nhật. Đầu phiên đọc để có context, cuối task ghi lại.
> Quy ước cập nhật: xem `CLAUDE.md` mục 7.

**Cập nhật lần cuối:** 2026-05-09

---

## Giai đoạn hiện tại

**Giai đoạn 1 — Thu thập & Tiền xử lý Dữ liệu**

---

## Đang làm

- Viết `src/data/preprocessor.py` — load + clean `IS_TuChon.xlsx`.

---

## Sắp làm (theo thứ tự ưu tiên)

1. ~~Viết `src/data/parser.py`~~ ✅ Hoàn thành.
2. ~~Viết `src/data/graph_builder.py`~~ ✅ Hoàn thành.
3. Viết `src/data/preprocessor.py` — load + clean `IS_TuChon.xlsx`.
4. Viết `src/data/regulation_parser.py` — parse quy định học vụ.
5. EDA notebook `notebooks/01_data_exploration.ipynb`.
6. Test toàn bộ pipeline giai đoạn 1 với dữ liệu hiện có.

---

## Dữ liệu hiện có

| Loại | Ngành | File | Trạng thái |
|---|---|---|---|
| Curriculum HTML | CS | `data/raw/curriculum/CS_curriculum.html` | ✅ Có |
| Curriculum HTML | IS | `data/raw/curriculum/IS_curriculum.html` | ✅ Có |
| Curriculum HTML | DS | `data/raw/curriculum/DS_curriculum.html` | ✅ Có |
| Curriculum HTML | SE | `data/raw/curriculum/SE_curriculum.html` | ✅ Có |
| Curriculum HTML | IT | `data/raw/curriculum/IT_curriculum.html` | ✅ Có |
| Grades xlsx | IS | `data/raw/grades/IS_TuChon.xlsx` | ✅ Có |
| Grades xlsx | CS | `data/raw/grades/CS_TuChon.xlsx` | ❌ Chưa có |
| Grades xlsx | DS | `data/raw/grades/DS_TuChon.xlsx` | ❌ Chưa có |
| Grades xlsx | SE | `data/raw/grades/SE_TuChon.xlsx` | ❌ Chưa có |
| Grades xlsx | IT | `data/raw/grades/IT_TuChon.xlsx` | ❌ Chưa có |
| Quy định học vụ | — | `data/raw/regulations/quy_dinh_hoc_vu.txt` | ❌ Chưa có |

---

## Checklist Giai đoạn 1

- [x] Parse HTML chương trình khung CS → JSON + graph
- [x] Parse HTML chương trình khung IS → JSON + graph
- [x] Parse HTML chương trình khung DS → JSON + graph
- [x] Parse HTML chương trình khung SE → JSON + graph
- [x] Parse HTML chương trình khung IT → JSON + graph
- [x] Build prerequisite graph (NetworkX) cho từng ngành
- [x] Validate cả 5 graph (không cycle, không thiếu prereq)
- [ ] Load và chuẩn hóa `IS_TuChon.xlsx`
- [ ] Build student profile từ IS_TuChon
- [ ] Trích xuất quy định học vụ → JSON structured (default values, chờ file thật)
- [ ] EDA notebook với dữ liệu hiện có
- [ ] Báo cáo chất lượng dữ liệu

---

## Phụ thuộc đã cài

- `lxml>=4.9` — cài vào venv (đã có trong requirements.txt), dùng cho BeautifulSoup parser
- `beautifulsoup4>=4.12` — cài vào venv, parse HTML curriculum

---

## Blockers & Câu hỏi cho user

- ⏳ Chưa có file xlsx của 4 ngành CS, DS, SE, IT — cần user bổ sung khi có.
- ⏳ Chưa có file quy định học vụ thực tế — đang dùng giá trị mặc định trong code.
- ⏳ Chưa quyết định LLM cuối cùng cho generation: Vistral-7B vs Qwen2.5-7B-Instruct (sẽ quyết khi đến giai đoạn 4).

---

## Nhật ký

- 2026-05-09: Khởi tạo dự án. Tạo `CLAUDE.md`, `STATUS.md`, `project_instructions.md`. Đã có sẵn 5 file HTML curriculum và `IS_TuChon.xlsx`.
- 2026-05-09: Viết `src/data/parser.py` — parse HTML chương trình khung → JSON. File: `src/data/parser.py`. Ghi chú: parse đúng điều kiện tiên quyết, pad mã môn 6 số, nhận diện môn có dấu * (khong_tinh_gpa).
- 2026-05-09: Viết `src/data/graph_builder.py` — build NetworkX DiGraph từ JSON curriculum. File: `src/data/graph_builder.py`. Ghi chú: validate no-cycle + no-missing-node, tất cả 5 ngành OK. Output: `data/processed/curriculum_graph/<NGANH>_prereq_graph.gpickle`.
- 2026-05-09: Parse thành công 5 ngành (CS=91 môn, IS=88, DS=86, SE=88, IT=85). Build 5 graph (CS=37 edges, IS=21, DS=15, SE=41, IT=11). Không có cycle, không thiếu node.