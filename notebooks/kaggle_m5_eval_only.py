# %% [markdown]
# # M5 — Re-evaluation only (KHÔNG train lại)
#
# Notebook này load LoRA adapter đã train (từ zip download lần trước) và
# **chạy lại evaluation** với 3 fix:
#
# 1. **SYSTEM prompt** khớp với training (lần trước bị mismatch → model output ngắn).
# 2. **Parse regex** bắt mã môn có/không có chữ "mã" prefix.
# 3. **max_new_tokens** tăng 256 → 512 để model output đủ 5 môn.
#
# Setup yêu cầu:
# - Accelerator: **GPU T4×2** (Settings → Accelerator)
# - Internet: **ON** (tải base model Qwen2.5-7B)
# - 2 datasets cần add:
#   * `ck-nlp-m5-sft` — chứa qwen_sft_test.jsonl + corpus.jsonl
#   * `ck-nlp-m5-lora` — upload mới: giải nén `m5_qwen_lora.zip`, upload folder
#     `qwen25_lora/` (gồm adapter_model.safetensors, adapter_config.json,
#     tokenizer files) làm Kaggle dataset.
#
# Thời gian ước tính: tải base model ~10 phút + generation 500 query ~15 phút = ~30 phút.

# %% [markdown]
# ## Cell 1 — Cài thư viện

# %%
# !pip install -q -U "transformers>=4.45" "peft>=0.13" "accelerate>=1.0" "bitsandbytes>=0.43"

# %% [markdown]
# ## Cell 2 — Import + check GPU

# %%
import gc
import json
import math
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("n_gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} {p.total_memory/1e9:.1f}GB")

# %% [markdown]
# ## Cell 3 — Paths
#
# Sửa `LORA_DIR` nếu tên dataset Kaggle khác. Cấu trúc folder LoRA dataset
# phải chứa trực tiếp `adapter_model.safetensors` + `adapter_config.json` +
# `tokenizer.json` ở cấp gốc (không lồng `checkpoint-2500/`).

# %%
DATA_DIR = Path("/kaggle/input/ck-nlp-m5-sft")
LORA_DIR = Path("/kaggle/input/ck-nlp-m5-lora")  # upload từ zip
OUT_DIR = Path("/kaggle/working/qwen25_lora_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("Test file:", (DATA_DIR / "qwen_sft_test.jsonl").exists())
print("Corpus file:", (DATA_DIR / "corpus.jsonl").exists())
print("LoRA adapter:", (LORA_DIR / "adapter_model.safetensors").exists())
print("LoRA config:", (LORA_DIR / "adapter_config.json").exists())

# %% [markdown]
# ## Cell 4 — Load test data + corpus

# %%
def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


test_records = load_jsonl(DATA_DIR / "qwen_sft_test.jsonl")
corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
id2doc = {d["doc_id"]: d for d in corpus}
all_doc_ids = set(id2doc.keys())
print(f"test={len(test_records)} corpus={len(corpus)}")

# %% [markdown]
# ## Cell 5 — Load base model 4-bit + LoRA adapter

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(str(LORA_DIR), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # left-pad cho generation

t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,
)
print(f"Loaded base in {time.time()-t0:.1f}s")

model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
model.eval()
print("Loaded LoRA adapter. Device map:")
for k, v in list(model.hf_device_map.items())[:3]:
    print(f"  {k}: {v}")

# %% [markdown]
# ## Cell 6 — Helpers (FIXED)
#
# **FIX 1 — SYSTEM prompt**: dùng đúng SYSTEM từ `prepare_sft_data.py`
# (training). Phiên bản trước dùng SYSTEM ngắn khác → distribution shift.
#
# **FIX 2 — Regex**: chấp nhận mã môn có hoặc không có chữ "mã" prefix,
# miễn là nằm trong cặp ngoặc đơn.

# %%
SYSTEM_PROMPT = (
    "Bạn là chatbot tư vấn đăng ký học phần thông minh tại trường đại học. "
    "Nhiệm vụ của bạn là giúp sinh viên lập kế hoạch học tập tối ưu dựa trên "
    "chương trình khung ngành học, các ràng buộc tiên quyết, lịch sử điểm và "
    "định hướng nghề nghiệp của sinh viên, và quy định học vụ (giới hạn tín "
    "chỉ, điều kiện tốt nghiệp). "
    "Luôn giải thích ngắn gọn lý do gợi ý. Trả lời bằng tiếng Việt."
)

NGANH_KEYWORDS = {
    "CS": ["CS", "Khoa học Máy tính", "khoa hoc may tinh"],
    "IS": ["IS", "Hệ thống Thông tin", "he thong thong tin"],
    "DS": ["DS", "Khoa học Dữ liệu", "khoa hoc du lieu"],
    "SE": ["SE", "Kỹ thuật Phần mềm", "ky thuat phan mem"],
    "IT": ["IT", "Công nghệ Thông tin", "cong nghe thong tin"],
}


def detect_nganh(query: str) -> str:
    """Suy ngành từ text query (xuất hiện ở user message)."""
    q_low = query.lower()
    for nganh, kws in NGANH_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in q_low:
                return nganh
    return ""


# Regex linh hoạt: bắt 6-digit code trong dấu ngoặc, có/không "mã" prefix.
# Match được cả: "(003633, 3 TC)", "(001758)", "(mã 001724, 3 credits)".
MA_PATTERN = re.compile(r"\((?:m[ãa]\s+)?(\d{6})", re.IGNORECASE)


def parse_recommendations(text: str, nganh: str, all_ids: set[str]) -> list[str]:
    """Trích doc_ids từ text generation, giữ thứ tự xuất hiện."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in MA_PATTERN.finditer(text):
        ma = m.group(1).zfill(6)
        doc_id = f"{nganh}_{ma}"
        if doc_id in all_ids and doc_id not in seen_set:
            seen.append(doc_id)
            seen_set.add(doc_id)
    return seen


def generate_for_query(query: str, max_new_tokens: int = 512) -> str:
    """Apply Qwen chat template + generate."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# Smoke test: chạy 1 query để xem format output.
print("=== Smoke test ===")
sample_q = test_records[0]["messages"][1]["content"]
sample_nganh = test_records[0].get("nganh", "") or detect_nganh(sample_q)
sample_text = generate_for_query(sample_q, max_new_tokens=512)
sample_pred = parse_recommendations(sample_text, sample_nganh, all_doc_ids)
print(f"Query: {sample_q[:100]}...")
print(f"Nganh: {sample_nganh}")
print(f"Generated:\n{sample_text}")
print(f"Parsed pred: {sample_pred}")
print(f"Gold: {test_records[0]['positive_doc_ids']}")

# %% [markdown]
# ## Cell 7 — Generate trên toàn bộ test set + tính metrics

# %%
def metrics_from_predictions(
    predictions: list[list[str]],
    test_set: list[dict],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    n = len(test_set)
    recall = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    for pred, ex in zip(predictions, test_set):
        gold = set(ex["positive_doc_ids"])
        if not gold:
            continue
        for k in ks:
            if set(pred[:k]) & gold:
                recall[k] += 1
        rank = next((i + 1 for i, d in enumerate(pred) if d in gold), None)
        if rank is not None:
            mrr_sum += 1.0 / rank
        dcg = sum(
            1.0 / math.log2(i + 2) for i, d in enumerate(pred[:10]) if d in gold
        )
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
        if idcg > 0:
            ndcg_sum += dcg / idcg
    out = {f"Recall@{k}": recall[k] / n for k in ks}
    out["MRR"] = mrr_sum / n
    out["NDCG@10"] = ndcg_sum / n
    return out


predictions: list[list[str]] = []
generations: list[str] = []

t0 = time.time()
for i, ex in enumerate(test_records):
    query = ex["messages"][1]["content"]
    nganh = ex.get("nganh", "") or detect_nganh(query)
    text = generate_for_query(query, max_new_tokens=512)
    pred = parse_recommendations(text, nganh, all_doc_ids)
    predictions.append(pred)
    generations.append(text)
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(test_records) - i - 1)
        print(f"[gen] {i+1}/{len(test_records)} elapsed={elapsed:.1f}s eta={eta:.1f}s")

print(f"Generation done in {(time.time()-t0)/60:.1f}min")

# Đếm thống kê để dễ debug.
n_pred_lengths = [len(p) for p in predictions]
print(f"\nPredictions stats:")
print(f"  avg len: {sum(n_pred_lengths)/len(n_pred_lengths):.2f}")
print(f"  empty: {sum(1 for x in n_pred_lengths if x == 0)}")
print(f"  >= 1: {sum(1 for x in n_pred_lengths if x >= 1)}")
print(f"  >= 5: {sum(1 for x in n_pred_lengths if x >= 5)}")

metrics = metrics_from_predictions(predictions, test_records)
print("\n=== M5 (Qwen2.5-7B-Instruct + LoRA) — RE-EVAL ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# %% [markdown]
# ## Cell 8 — Lưu kết quả + sample generations để inspect

# %%
with open(OUT_DIR / "eval_results.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": "qwen2.5-7b-instruct-lora",
            "eval_version": "v2_fixed_parse",
            "fixes": [
                "SYSTEM prompt matches training",
                "Parse regex accepts code with/without 'mã' prefix",
                "max_new_tokens 256 -> 512",
            ],
            "metrics": metrics,
            "pred_length_stats": {
                "avg": sum(n_pred_lengths) / len(n_pred_lengths),
                "empty": sum(1 for x in n_pred_lengths if x == 0),
                "ge_1": sum(1 for x in n_pred_lengths if x >= 1),
                "ge_5": sum(1 for x in n_pred_lengths if x >= 5),
            },
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

# Lưu 50 samples (thay vì 6) để inspect kỹ hơn.
samples = []
sample_indices = list(range(0, len(test_records), max(1, len(test_records) // 50)))[:50]
for i in sample_indices:
    samples.append(
        {
            "idx": i,
            "query": test_records[i]["messages"][1]["content"],
            "nganh": test_records[i].get("nganh", ""),
            "gold": test_records[i]["positive_doc_ids"],
            "pred": predictions[i],
            "text": generations[i],
        }
    )
with open(OUT_DIR / "sample_generations.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

# Lưu full predictions để có thể recompute metrics local nếu cần.
with open(OUT_DIR / "all_predictions.jsonl", "w", encoding="utf-8") as f:
    for i, (pred, text) in enumerate(zip(predictions, generations)):
        f.write(
            json.dumps(
                {
                    "idx": i,
                    "pred": pred,
                    "gold": test_records[i]["positive_doc_ids"],
                    "text": text,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

print(f"\nFiles in {OUT_DIR}:")
for p in OUT_DIR.iterdir():
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")

# %% [markdown]
# ## Cell 9 — (Tuỳ chọn) Đóng gói để download
#
# Output ở `/kaggle/working/qwen25_lora_eval/` đã có trong tab Output. Hoặc:
#
# ```python
# import shutil
# shutil.make_archive("/kaggle/working/m5_eval_v2", "zip", str(OUT_DIR))
# ```
