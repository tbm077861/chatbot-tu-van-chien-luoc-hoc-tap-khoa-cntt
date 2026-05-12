# %% [markdown]
# # M5 — LoRA fine-tune Qwen2.5-7B-Instruct trên Kaggle T4×2
#
# Mục tiêu: fine-tune Qwen2.5-7B-Instruct bằng LoRA trên 20k example SFT, sau
# đó eval generation trên 500 test query → tính Recall@K, MRR.
#
# Setup yêu cầu:
# - Kaggle Notebook, **Accelerator: GPU T4×2** (Settings → Accelerator)
# - **Internet: ON** (để tải model từ HF Hub)
# - Persistence: **Variables and Files** (giữ output giữa các session nếu chạy lâu)
# - Dataset đã upload: `ck-nlp-m5-sft` (gồm `qwen_sft_train.jsonl`,
#   `qwen_sft_test.jsonl`, `corpus.jsonl`)

# %% [markdown]
# ## Cell 1 — Cài thư viện
# Kaggle đã có sẵn torch + transformers nhưng version cũ. Nâng + cài peft/trl.

# %%
# !pip install -q -U "transformers>=4.45" "peft>=0.13" "trl>=0.11" \
#     "accelerate>=1.0" "bitsandbytes>=0.43" "datasets>=3.0"

# %% [markdown]
# ## Cell 2 — Import + kiểm tra GPU

# %%
import gc
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("n_gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} {p.total_memory/1e9:.1f}GB")

# %% [markdown]
# ## Cell 3 — Đường dẫn input/output
# Sửa `DATA_DIR` thành đường dẫn Kaggle dataset thật của bạn (xem mục
# "Add data" trong notebook).

# %%
DATA_DIR = Path("/kaggle/input/ck-nlp-m5-sft")
OUT_DIR = Path("/kaggle/working/qwen25_lora")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 1024  # đủ cho query + 5 môn gợi ý
BATCH_PER_DEVICE = 2  # T4 16GB × 2 → tổng 4 sample/step
GRAD_ACCUM = 4  # effective batch = 16
EPOCHS = 1  # 1 epoch ~ 20k/16 = 1250 steps
LR = 1e-4
LORA_R = 16
LORA_ALPHA = 32

print("Train file:", (DATA_DIR / "qwen_sft_train.jsonl").exists())
print("Test file:", (DATA_DIR / "qwen_sft_test.jsonl").exists())
print("Corpus file:", (DATA_DIR / "corpus.jsonl").exists())

# %% [markdown]
# ## Cell 4 — Load dataset SFT
# File có sẵn ở định dạng chuẩn `{"messages": [...]}` mà TRL SFTTrainer hiểu.

# %%
def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


train_records = load_jsonl(DATA_DIR / "qwen_sft_train.jsonl")
test_records = load_jsonl(DATA_DIR / "qwen_sft_test.jsonl")
corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
id2doc = {d["doc_id"]: d for d in corpus}
print(f"train={len(train_records)} test={len(test_records)} corpus={len(corpus)}")

# Ví dụ messages
print("\n--- Ví dụ training record ---")
ex = train_records[0]
for m in ex["messages"]:
    print(f"[{m['role']}] {m['content'][:160]}")

# Chỉ giữ field 'messages' để fit SFTTrainer; bỏ 'positive_doc_ids' (dùng sau).
train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_records])
print(train_ds)

# %% [markdown]
# ## Cell 5 — Load tokenizer + model 4-bit
# Với T4×2 (32GB tổng), có thể chạy fp16 không quantize nhưng 4-bit + LoRA an
# toàn hơn về memory + cho phép batch lớn hơn. NF4 + double-quant là cấu hình
# QLoRA tiêu chuẩn.

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # right-pad cho SFT (causal LM)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",  # tự chia sharded qua 2 T4
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
print(f"Loaded model in {time.time()-t0:.1f}s")
model.config.use_cache = False  # cần tắt khi train
model.config.pretraining_tp = 1

# In ra cách model được phân bổ qua các GPU
print("\nDevice map:")
for k, v in model.hf_device_map.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## Cell 6 — Prepare model cho k-bit training + add LoRA adapter
# Target modules cho Qwen: tất cả linear projections trong attention + MLP.

# %%
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## Cell 7 — TrainingArguments + SFTTrainer
# TRL ≥ 0.11 dùng `SFTConfig` thay vì `TrainingArguments`.

# %%
sft_config = SFTConfig(
    output_dir=str(OUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_PER_DEVICE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=False,
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    report_to="none",
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field=None,  # dùng messages → trainer tự apply chat template
    packing=False,
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    args=sft_config,
)
print("Train steps:", len(train_ds) // (BATCH_PER_DEVICE * GRAD_ACCUM * max(1, torch.cuda.device_count())) * EPOCHS)

# %% [markdown]
# ## Cell 8 — Train
# Trên T4×2 với 20k example, fp16, gradient checkpoint, max_len 1024, ước lượng
# **~2.5–3.5 giờ / epoch**. Theo dõi log: nên thấy loss giảm từ ~1.5 → ~0.4.

# %%
trainer.train()
trainer.save_model(str(OUT_DIR))  # lưu adapter (~50-100MB)
tokenizer.save_pretrained(str(OUT_DIR))
print("Saved LoRA adapter to", OUT_DIR)

# %% [markdown]
# ## Cell 9 — Giải phóng training state, load lại model + adapter cho inference
# Sau khi train, model trong RAM có optimizer state + gradient → giải phóng để
# có chỗ load cho inference.

# %%
del trainer
del model
gc.collect()
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, str(OUT_DIR))
model.eval()
print("Loaded base + LoRA adapter for inference.")

# %% [markdown]
# ## Cell 10 — Helper: generate + parse course list
# Generation sinh ra text dạng:
# ```
# Dựa trên profile của bạn, mình gợi ý các môn sau:
# 1. **Tên môn 1** (mã 003197, 3 TC, tự chọn)
# 2. ...
# ```
# Parse regex `mã (\d{6})` → trích doc_ids, map ngành từ query.

# %%
NGANH_KEYWORDS = {
    "CS": ["CS", "Khoa học Máy tính", "khoa hoc may tinh"],
    "IS": ["IS", "Hệ thống Thông tin", "he thong thong tin"],
    "DS": ["DS", "Khoa học Dữ liệu", "khoa hoc du lieu"],
    "SE": ["SE", "Kỹ thuật Phần mềm", "ky thuat phan mem"],
    "IT": ["IT", "Công nghệ Thông tin", "cong nghe thong tin"],
}


def detect_nganh(query: str) -> str:
    """Suy ra ngành từ text query (xuất hiện ở user message)."""
    q_low = query.lower()
    for nganh, kws in NGANH_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in q_low:
                return nganh
    return ""


MA_PATTERN = re.compile(r"m[ãa]\s*(\d{6})", re.IGNORECASE)


def parse_recommendations(text: str, nganh: str, all_doc_ids: set[str]) -> list[str]:
    """Trích doc_ids từ text generation. Có 3 chiến lược:
    1. Tìm "mã 003197" → ghép với nganh thành doc_id.
    2. Nếu không có mã, match theo ten_mon (chậm hơn, dùng khi cần).
    3. Bỏ qua trùng lặp, giữ thứ tự xuất hiện.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in MA_PATTERN.finditer(text):
        ma = m.group(1).zfill(6)
        doc_id = f"{nganh}_{ma}"
        if doc_id in all_doc_ids and doc_id not in seen_set:
            seen.append(doc_id)
            seen_set.add(doc_id)
    return seen


def generate_for_query(
    query: str, max_new_tokens: int = 256
) -> str:
    """Apply Qwen chat template + generate."""
    SYSTEM = (
        "Bạn là chatbot tư vấn đăng ký học phần thông minh tại trường đại học. "
        "Trả lời bằng tiếng Việt, format danh sách 1. **Tên môn** (mã XXXXXX, ...)."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
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
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# Smoke test
print(generate_for_query(test_records[0]["messages"][1]["content"], max_new_tokens=200))

# %% [markdown]
# ## Cell 11 — Eval trên test 500
# Generate cho mỗi query, parse, tính metrics Recall@K + MRR + NDCG@10.
# Trên T4×2 với 4-bit + batch_size=1 generate: ~3-4 query/s → 500 query ~ 2-3
# phút.

# %%
def metrics_from_predictions(
    predictions: list[list[str]],  # list of ranked doc_ids per query
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


all_doc_ids = set(id2doc.keys())
predictions: list[list[str]] = []
generations: list[str] = []

t0 = time.time()
for i, ex in enumerate(test_records):
    query = ex["messages"][1]["content"]
    nganh = ex.get("nganh", "") or detect_nganh(query)
    text = generate_for_query(query, max_new_tokens=256)
    pred = parse_recommendations(text, nganh, all_doc_ids)
    predictions.append(pred)
    generations.append(text)
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(test_records) - i - 1)
        print(f"[gen] {i+1}/{len(test_records)} elapsed={elapsed:.1f}s eta={eta:.1f}s")

print(f"Generation done in {(time.time()-t0)/60:.1f}min")

metrics = metrics_from_predictions(predictions, test_records)
print("\n=== M5 (Qwen2.5-7B-Instruct + LoRA) ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# %% [markdown]
# ## Cell 12 — Lưu kết quả
# Save:
# - adapter weights (đã save ở Cell 8 ở /kaggle/working/qwen25_lora/)
# - eval results JSON
# - sample generations (50 example để xem qualitative)

# %%
with open(OUT_DIR / "eval_results.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": "qwen2.5-7b-instruct-lora",
            "config": {
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "epochs": EPOCHS,
                "lr": LR,
                "batch_per_device": BATCH_PER_DEVICE,
                "grad_accum": GRAD_ACCUM,
                "max_seq_len": MAX_SEQ_LEN,
                "n_train": len(train_records),
                "n_test": len(test_records),
            },
            "metrics": metrics,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

# Sample generations
samples = []
for i in [0, 1, 50, 100, 250, 499]:
    samples.append(
        {
            "query": test_records[i]["messages"][1]["content"],
            "nganh": test_records[i].get("nganh", ""),
            "gold": test_records[i]["positive_doc_ids"],
            "pred": predictions[i],
            "text": generations[i],
        }
    )
with open(OUT_DIR / "sample_generations.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print("Files in", OUT_DIR, ":")
for p in OUT_DIR.iterdir():
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")

# %% [markdown]
# ## Cell 13 — (Tuỳ chọn) Đóng gói để download
# Toàn bộ thư mục `/kaggle/working/qwen25_lora` là **output của notebook** —
# sau khi Run All xong, mở tab "Output" của notebook bên phải để tải về máy.
#
# Hoặc dùng zip để gọn 1 file:
# ```python
# import shutil
# shutil.make_archive("/kaggle/working/m5_qwen_lora", "zip", str(OUT_DIR))
# ```
