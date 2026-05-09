# CLAUDE.md

> File này được Claude Code đọc tự động ở mỗi phiên làm việc.
> Giữ ngắn gọn — context chi tiết nằm trong `project_instructions.md` và `STATUS.md`.

---

## 1. Dự án

**RAG Chatbot tư vấn đăng ký học phần** cho sinh viên 5 ngành CNTT: CS, IS, DS, SE, IT.

Mục tiêu: gợi ý học phần tự chọn dựa trên profile, điểm số, định hướng nghề; đảm bảo đúng tiên quyết và giới hạn tín chỉ; trả lời tiếng Việt tự nhiên, có giải thích.

Chi tiết đầy đủ: đọc `project_instructions.md`.

---

## 2. Quy tắc giao tiếp

- **Trả lời và comment code bằng tiếng Việt.**
- **Đầu phiên làm việc:** đọc `STATUS.md` để biết đã làm tới đâu, đang làm gì.
- **Sau khi hoàn thành task:** tự cập nhật `STATUS.md` theo format ở mục 7 — không hỏi xin phép.
- Yêu cầu mơ hồ → hỏi lại 1–2 câu trước khi code; đừng đoán mò.
- Sửa file >200 dòng → đọc cả file rồi mới chỉnh từng phần.

---

## 3. Quy tắc kỹ thuật

### Code style
- Python 3.10+, **type hints bắt buộc** ở mọi function public.
- **Docstring tiếng Việt** kiểu Google: mô tả ngắn, `Args`, `Returns`, `Raises` (nếu có).
- Module-level docstring đầu file: mô tả mục đích + ví dụ dùng CLI.
- Tên biến/hàm tiếng Anh; chú thích logic phức tạp bằng tiếng Việt.
- Format bằng `ruff format` trước khi báo "xong".

### Cấu trúc dự án
- Cấu trúc thư mục đã định nghĩa trong `project_instructions.md` mục 8 — **không tự ý đổi**.
- File mới chỉ tạo trong đúng thư mục được chỉ định (parser → `src/data/`, embedding → `src/embedding/`, v.v.).
- Một file một trách nhiệm. Vượt 500 dòng → tách ra.

### Dữ liệu
- Mã môn học **luôn 6 chữ số**, pad 0 bên trái: `"003197"` không phải `"3197"` hay `3197`.
- Key định danh môn = `(nganh, ma_mon)` vì mã môn có thể trùng giữa các ngành.
- Môn có dấu `*` (Giáo dục thể chất, Quốc phòng, Chứng chỉ Tiếng Anh) → loại khỏi tính GPA.
- Output dữ liệu: `.parquet` (chính, nhanh) + `.csv` (debug, dễ xem).
- **Không commit** file dữ liệu thô lên git (đã có `.gitignore`).
- **Hai hệ thống học kỳ:** Chương trình khung dùng HK1–HK9 (liên tục),
  file điểm TenDot dùng HK1/HK2/HK3 + năm học (lẻ/chẵn/hè).
  Luôn quy đổi bằng cột `curriculum_hk` trước khi map với prerequisite graph.
  Công thức: `curriculum_hk = (start_year - năm_nhập_học) × 2 + hk_so`.
  HK3 hè → `curriculum_hk = None` (xử lý riêng).
### Phụ thuộc
- **Không tự ý `pip install`** thư viện nặng (`torch`, `transformers`, `faiss`, `sentence-transformers`, `peft`, `ragas`). Hỏi user trước.
- Nếu user đồng ý, thêm vào `requirements.txt` đúng nhóm (Core / NLP / RAG / API).
- Sau cài đặt, ghi log vào `STATUS.md` mục "Phụ thuộc đã cài".

### Test
- Mỗi module trong `src/` nên có file test tương ứng trong `tests/`.
- Test **luôn chạy với dữ liệu nhỏ trước** (file `IS_TuChon.xlsx`) trước khi áp dụng dữ liệu lớn.
- Không bao giờ xóa hoặc ghi đè file trong `data/raw/`.

### Bảo mật
- API keys luôn đọc từ `.env` qua `python-dotenv`. Tuyệt đối không hardcode.
- File `.env` không được commit (đã có trong `.gitignore`).
- Trước khi commit: `git diff --staged | grep -iE "sk-|api[_-]?key"` — không được có match.

---

## 4. Trạng thái hiện tại — TỰ CẬP NHẬT

Trạng thái chi tiết (đã làm/đang làm/sắp làm) nằm trong **`STATUS.md`**.
Claude phải:
1. **Đầu phiên:** đọc `STATUS.md`.
2. **Cuối mỗi task:** cập nhật `STATUS.md` theo format ở mục 7.

---

## 5. Stack công nghệ

| Lớp | Thư viện chính |
|---|---|
| Data | pandas, numpy, openpyxl, pyarrow |
| Graph | networkx |
| Parser | beautifulsoup4, lxml |
| NLP/Embedding | transformers, sentence-transformers, peft, torch |
| Vector | faiss-cpu, rank-bm25 |
| LLM | Vistral-7B / Qwen2.5-7B (local), OpenAI/Gemini (synthetic) |
| API | fastapi, uvicorn |
| UI | streamlit |
| Eval | ragas, beir |

Version chi tiết: `requirements.txt`.

---

## 6. Lệnh thường dùng

```bash
# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate

# Parse một ngành curriculum
python src/data/parser.py --input data/raw/curriculum/CS_curriculum.html

# Parse tất cả curriculum
python src/data/parser.py --all

# Preprocess grades + EDA
python src/data/preprocessor.py --input_dir data/raw/grades --eda

# Lint & format
ruff check src/
ruff format src/

# Test
pytest tests/ -v
```

---

## 7. Quy ước cập nhật `STATUS.md`

### 7.1 Khi nào cập nhật
- Hoàn thành một mục trong checklist `project_instructions.md` mục 10.
- Tạo mới hoặc thay đổi đáng kể một file trong `src/`.
- Cài thêm thư viện vào `requirements.txt`.
- Phát sinh blocker hoặc câu hỏi cần user trả lời.
- User cung cấp dữ liệu mới (HTML curriculum, xlsx grades, file quy định).

### 7.2 Cập nhật cái gì
1. **Đánh dấu `[x]`** mục đã xong trong checklist của giai đoạn hiện tại.
2. **Cập nhật `## Đang làm`** với task mới.
3. **Thêm vào `## Nhật ký`** dòng có format:
   `- YYYY-MM-DD: <hành động>. File: <path>. Ghi chú: <tùy chọn>`
4. Nếu có blocker → thêm vào `## Blockers & Câu hỏi cho user`.
5. Cập nhật `## Dữ liệu hiện có` nếu có data mới.

### 7.3 Format thông báo cuối phản hồi

Sau mỗi task có cập nhật, kết thúc bằng block:

📝 Đã cập nhật STATUS.md:

[x] <mục đã tick>

Nhật ký: <dòng đã thêm>

Blocker mới (nếu có): <mô tả>

### 7.4 Không cần cập nhật khi
- Sửa lỗi nhỏ (typo, format, comment).
- Trả lời câu hỏi không tạo file mới.
- User đang exploration (chỉ chạy thử, chưa commit).

---

## 8. Khi không chắc

Quyết định lớn (architecture, thuật toán, schema dữ liệu) mà chưa rõ:
1. **Dừng code lại.**
2. Trình bày 2–3 phương án với ưu/nhược ngắn gọn.
3. Đề xuất phương án mặc định kèm lý do.
4. Hỏi user xác nhận trước khi tiến hành.

Đừng đoán mò rồi viết 500 dòng code sai hướng.