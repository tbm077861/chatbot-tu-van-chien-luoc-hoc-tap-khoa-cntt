# %% [markdown]
# # Giai đoạn 6 — Eval RAG vs non-RAG với M5 (Qwen2.5-7B-Instruct + LoRA)
#
# Notebook này sinh predictions cho **cả 2 variant** (RAG / noRAG) trên
# Kaggle T4×2, dùng cùng base model + LoRA adapter của Stage 4.
#
# Input (đã chuẩn bị bởi `src/evaluation/export_for_kaggle.py`):
#   * `rag_inputs_warm.jsonl`   — user_message có context block top-K.
#   * `norag_inputs_warm.jsonl` — user_message chỉ có profile + question.
#
# Output về `/kaggle/working/`:
#   * `predictions_rag.jsonl`   — mỗi dòng: idx, query, nganh, gold,
#                                 retrieved_valid, context_doc_ids,
#                                 response (text), predicted_doc_ids.
#   * `predictions_norag.jsonl` — tương tự (retrieved_valid = []).
#   * `eval_summary.json`       — Hit@1/5/10, MRR, NDCG@10 cho cả 2 variant.
#
# Setup Kaggle:
#   * Accelerator: **GPU T4×2**
#   * Internet: **ON** (tải base Qwen2.5-7B)
#   * 2 datasets:
#     - `ck-nlp-m5-lora` — folder `m5_lora_for_kaggle/` (adapter + tokenizer).
#     - `ck-nlp-stage6-inputs` — upload 2 file JSONL từ `data/kaggle_export/`.
#
# Thời gian ước tính: tải base ~10 phút + generate 200 query × ~6s = ~30 phút.

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
# Sửa các path bên dưới nếu tên dataset Kaggle khác.

# %%
INPUTS_DIR = Path("/kaggle/input/ck-nlp-stage6-inputs")
LORA_DIR = Path("/kaggle/input/ck-nlp-m5-lora")
OUT_DIR = Path("/kaggle/working/stage6_rag_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

RAG_FILE = INPUTS_DIR / "rag_inputs_warm.jsonl"
NORAG_FILE = INPUTS_DIR / "norag_inputs_warm.jsonl"

print("RAG inputs:", RAG_FILE.exists(), RAG_FILE)
print("noRAG inputs:", NORAG_FILE.exists(), NORAG_FILE)
print("LoRA adapter:", (LORA_DIR / "adapter_model.safetensors").exists())
print("LoRA config:", (LORA_DIR / "adapter_config.json").exists())

# %% [markdown]
# ## Cell 4 — Load 2 input JSONL

# %%
def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


rag_records = load_jsonl(RAG_FILE)
norag_records = load_jsonl(NORAG_FILE)
print(f"rag={len(rag_records)} norag={len(norag_records)}")

# Sanity check: 2 file phải cùng số dòng + cùng idx + cùng gold.
assert len(rag_records) == len(norag_records), "Số dòng RAG/noRAG khác nhau!"
for r, n in zip(rag_records, norag_records):
    assert r["idx"] == n["idx"], f"idx mismatch: {r['idx']} vs {n['idx']}"
    assert r["gold"] == n["gold"], f"gold mismatch tại idx={r['idx']}"
print("Sanity check OK: 2 file align theo idx + gold.")

# Build tập tất cả doc_id để filter parse (gộp gold + retrieved_valid).
all_doc_ids: set[str] = set()
for r in rag_records:
    all_doc_ids.update(r["gold"])
    all_doc_ids.update(r["retrieved_valid"])
print(f"Tổng unique doc_ids xuất hiện: {len(all_doc_ids)}")

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
tokenizer.padding_side = "left"

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
print("Loaded LoRA adapter.")

# %% [markdown]
# ## Cell 6 — Helpers (parse + generate)
#
# `system_prompt` và `user_message` đã được render sẵn ở
# `src/generation/prompt_templates.py` rồi ghi vào JSONL — Kaggle chỉ cần
# `apply_chat_template` rồi `generate`. Không tự xây lại prompt ở đây để
# tránh distribution shift với training (đã bị burn ở v1 eval).

# %%
# Regex linh hoạt: bắt 6-digit code trong dấu ngoặc, có/không "mã" prefix.
MA_PATTERN = re.compile(r"\((?:m[ãa]\s+)?(\d{6})", re.IGNORECASE)


def parse_recommendations(text: str, nganh: str, all_ids: set[str]) -> list[str]:
    """Trích doc_id (`{nganh}_{ma}`) từ text generation, giữ thứ tự xuất hiện."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in MA_PATTERN.finditer(text):
        ma = m.group(1).zfill(6)
        doc_id = f"{nganh}_{ma}"
        if doc_id in all_ids and doc_id not in seen_set:
            seen.append(doc_id)
            seen_set.add(doc_id)
    return seen


@torch.no_grad()
def generate_response(system_prompt: str, user_message: str, max_new_tokens: int = 512) -> str:
    """Apply Qwen chat template + generate (greedy)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(
        model.device
    )
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


# Smoke test trên record đầu tiên của cả 2 variant.
print("=== Smoke test (RAG variant) ===")
ex = rag_records[0]
text = generate_response(ex["system_prompt"], ex["user_message"])
pred = parse_recommendations(text, ex["nganh"], all_doc_ids)
print(f"Query: {ex['query'][:100]}...")
print(f"Nganh: {ex['nganh']}")
print(f"Gold:  {ex['gold']}")
print(f"Pred:  {pred}")
print(f"Text (first 400 chars):\n{text[:400]}")

print("\n=== Smoke test (noRAG variant) ===")
ex = norag_records[0]
text = generate_response(ex["system_prompt"], ex["user_message"])
pred = parse_recommendations(text, ex["nganh"], all_doc_ids)
print(f"Pred:  {pred}")
print(f"Text (first 400 chars):\n{text[:400]}")

# %% [markdown]
# ## Cell 7 — Hàm generate batch cho 1 variant + lưu

# %%
def run_variant(
    records: list[dict],
    variant: str,
    out_path: Path,
    max_new_tokens: int = 512,
) -> tuple[list[list[str]], list[str]]:
    """Generate toàn bộ records, parse predictions, lưu JSONL."""
    predictions: list[list[str]] = []
    responses: list[str] = []
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(records):
            text = generate_response(
                ex["system_prompt"], ex["user_message"], max_new_tokens
            )
            pred = parse_recommendations(text, ex["nganh"], all_doc_ids)
            predictions.append(pred)
            responses.append(text)
            fout.write(
                json.dumps(
                    {
                        "idx": ex["idx"],
                        "query": ex["query"],
                        "nganh": ex["nganh"],
                        "hk_completed": ex.get("hk_completed"),
                        "hk_target": ex.get("hk_target"),
                        "gold": ex["gold"],
                        "retrieved_valid": ex.get("retrieved_valid", []),
                        "context_doc_ids": ex.get("context_doc_ids", []),
                        "response": text,
                        "predicted_doc_ids": pred,
                        "variant": variant,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(records) - i - 1)
                print(
                    f"[{variant}] {i+1}/{len(records)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                )
    print(f"[{variant}] done in {(time.time()-t0)/60:.2f}min → {out_path.name}")
    return predictions, responses


# %% [markdown]
# ## Cell 8 — Chạy variant RAG

# %%
preds_rag, resp_rag = run_variant(
    rag_records, "rag", OUT_DIR / "predictions_rag.jsonl"
)

# %% [markdown]
# ## Cell 9 — Chạy variant noRAG

# %%
preds_norag, resp_norag = run_variant(
    norag_records, "norag", OUT_DIR / "predictions_norag.jsonl"
)

# %% [markdown]
# ## Cell 10 — Compute metrics (Hit@K, MRR, NDCG@10) cho 2 variant

# %%
def compute_metrics(
    predictions: list[list[str]],
    records: list[dict],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    n = len(records)
    recall = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    pred_lens = []
    for pred, ex in zip(predictions, records):
        pred_lens.append(len(pred))
        gold = set(ex["gold"])
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
    out = {f"Hit@{k}": recall[k] / n for k in ks}
    out["MRR"] = mrr_sum / n
    out["NDCG@10"] = ndcg_sum / n
    out["pred_len_avg"] = sum(pred_lens) / max(1, n)
    out["pred_empty"] = sum(1 for x in pred_lens if x == 0)
    return out


metrics_rag = compute_metrics(preds_rag, rag_records)
metrics_norag = compute_metrics(preds_norag, norag_records)

print("=== M5 + RAG (top-K context) ===")
for k, v in metrics_rag.items():
    print(f"  {k}: {v:.4f}")

print("\n=== M5 + noRAG (baseline) ===")
for k, v in metrics_norag.items():
    print(f"  {k}: {v:.4f}")

print("\n=== Δ (RAG − noRAG) ===")
for k in ("Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG@10"):
    delta = metrics_rag[k] - metrics_norag[k]
    print(f"  {k}: {delta:+.4f}")

# %% [markdown]
# ## Cell 11 — Lưu summary + đóng gói

# %%
summary = {
    "model": "qwen2.5-7b-instruct-lora",
    "n_queries": len(rag_records),
    "mode": "warm",
    "max_new_tokens": 512,
    "decoding": {
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    },
    "metrics": {"rag": metrics_rag, "norag": metrics_norag},
    "delta_rag_minus_norag": {
        k: metrics_rag[k] - metrics_norag[k]
        for k in ("Hit@1", "Hit@5", "Hit@10", "MRR", "NDCG@10")
    },
}
with open(OUT_DIR / "eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# In ra danh sách file để user biết download cái gì.
print(f"\nFiles in {OUT_DIR}:")
for p in OUT_DIR.iterdir():
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")

# (Tuỳ chọn) đóng gói thành zip để download nhanh:
# import shutil
# shutil.make_archive("/kaggle/working/stage6_rag_eval", "zip", str(OUT_DIR))
