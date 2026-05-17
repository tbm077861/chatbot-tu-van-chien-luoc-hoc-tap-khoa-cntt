"""Generator — M5 Qwen2.5-7B-Instruct + LoRA adapter (Giai đoạn 5).

Inference pipeline:
1. Load Qwen2.5-7B-Instruct 4-bit (NF4 double quant, fp16 compute).
2. Attach LoRA adapter `src/models/kaggle_output/m5_lora_for_kaggle/`.
3. Apply chat template với SYSTEM + USER message → generate.
4. Parse output: trích mã môn từ pattern `(mã 003633)` hoặc `(003633)`.

VRAM cần: ~6GB cho 4-bit Qwen-7B + LoRA. RTX 5070 12GB đủ.

Có 2 lớp generator:
- `QwenGenerator`: real LLM (cần GPU + bitsandbytes).
- `StubGenerator`: template-based fallback (không LLM, để test pipeline).

CLI smoke test:
    python -m src.generation.generator --query "Em ngành CS HK5 định hướng AI"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LORA_DIR = ROOT / "src" / "models" / "kaggle_output" / "m5_lora_for_kaggle"

sys.path.insert(0, str(ROOT))
from src.generation.prompt_templates import SYSTEM_PROMPT  # noqa: E402


# Regex bắt mã môn (6 chữ số) trong dấu ngoặc đơn, có/không "mã" prefix.
# Khớp với regex Stage 4 kaggle_m5_eval_only.py để parse output M5 chuẩn.
_MA_PATTERN = re.compile(r"\((?:m[ãa]\s+)?(\d{6})", re.IGNORECASE)


def parse_recommendations(
    text: str, nganh: str, valid_doc_ids: set[str] | None = None
) -> list[str]:
    """Trích doc_id từ response text, giữ thứ tự xuất hiện.

    Args:
        text: response text từ LLM.
        nganh: mã ngành để build doc_id = f"{nganh}_{ma_mon}".
        valid_doc_ids: nếu cung cấp, chỉ giữ doc_id có trong set này.

    Returns:
        List doc_id duy nhất, theo thứ tự xuất hiện trong text.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _MA_PATTERN.finditer(text):
        ma = m.group(1).zfill(6)
        doc_id = f"{nganh}_{ma}"
        if valid_doc_ids is not None and doc_id not in valid_doc_ids:
            continue
        if doc_id not in seen_set:
            seen.append(doc_id)
            seen_set.add(doc_id)
    return seen


class StubGenerator:
    """Generator giả lập — format response từ retrieved docs, không gọi LLM.

    Dùng để test pipeline retrieval/constraint mà không cần GPU.
    Output format khớp pattern parse_recommendations để metrics vẫn tính được.
    """

    def __init__(self, max_items: int = 5) -> None:
        self.max_items = max_items

    def generate(
        self,
        user_message: str,  # noqa: ARG002  (giữ signature đồng nhất)
        retrieved_docs: list[dict],
        **kwargs,  # noqa: ARG002
    ) -> str:
        """Trả response template-based.

        Args:
            user_message: bỏ qua trong stub (giữ signature).
            retrieved_docs: list dict {doc_id, ten_mon, ma_mon, so_tc, loai}.

        Returns:
            Chuỗi response tiếng Việt liệt kê top-K môn.
        """
        if not retrieved_docs:
            return "Hiện chưa tìm được học phần phù hợp với câu hỏi của bạn."
        lines = ["Gợi ý các học phần phù hợp:"]
        for i, d in enumerate(retrieved_docs[: self.max_items], 1):
            ten = d.get("ten_mon", "?")
            ma = d.get("ma_mon", "?")
            tc = d.get("so_tc", "?")
            loai = d.get("loai", "?")
            lines.append(f"{i}. **{ten}** (mã {ma}, {tc} TC, {loai})")
        return "\n".join(lines)

    def chat(
        self,
        messages: list[dict],  # noqa: ARG002 (signature compat với Qwen)
        retrieved_docs: list[dict] | None = None,
        **kwargs,  # noqa: ARG002
    ) -> str:
        """Multi-turn signature — stub không hiểu history, format y `generate()`."""
        return self.generate("", retrieved_docs or [])


class QwenGenerator:
    """M5 Qwen2.5-7B + LoRA adapter, inference local 4-bit.

    Args:
        model: HuggingFace causal LM đã attach LoRA.
        tokenizer: AutoTokenizer của Qwen.
        device: thiết bị (CUDA khuyến nghị).
    """

    def __init__(self, model, tokenizer, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load_default(
        cls,
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_dir: Path | None = None,
        load_4bit: bool = True,
    ) -> "QwenGenerator":
        """Load Qwen-7B 4-bit + LoRA adapter M5.

        Args:
            base_model_name: HuggingFace model ID.
            lora_dir: thư mục chứa adapter_model.safetensors. Mặc định
                `src/models/kaggle_output/m5_lora_for_kaggle/`.
            load_4bit: nếu False, load fp16 (yêu cầu ~15GB VRAM).

        Returns:
            Instance QwenGenerator sẵn sàng generate.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        lora_dir = lora_dir or LORA_DIR
        if not (lora_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Không tìm thấy LoRA adapter ở {lora_dir}. "
                f"Đã tải về từ Kaggle chưa? (xem STATUS.md 2026-05-12)"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda" and load_4bit:
            print(
                "[warn] Không có CUDA — load fp32 (chậm, RAM lớn). "
                "Cân nhắc dùng StubGenerator."
            )
            load_4bit = False

        tokenizer = AutoTokenizer.from_pretrained(str(lora_dir), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        kwargs: dict = {"trust_remote_code": True}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = "auto"
            kwargs["dtype"] = torch.float16
        else:
            kwargs["dtype"] = torch.float32 if device.type == "cpu" else torch.float16

        print(f"[load] base model {base_model_name} (4bit={load_4bit})...")
        base = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        if not load_4bit:
            base.to(device)

        from peft import PeftModel

        print(f"[load] LoRA adapter {lora_dir}...")
        model = PeftModel.from_pretrained(base, str(lora_dir))
        model.eval()

        return cls(model, tokenizer, device)

    @torch.no_grad()
    def generate(
        self,
        user_message: str,
        retrieved_docs: list[dict] | None = None,  # noqa: ARG002 (kept for signature)
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.05,
        do_sample: bool = False,
    ) -> str:
        """Generate response cho 1 user message (single-turn, dùng cho /answer cũ).

        Multi-turn chat dùng `chat()`. Giữ method này cho backward compat.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return self._generate_from_messages(
            messages, max_new_tokens, repetition_penalty, do_sample
        )

    @torch.no_grad()
    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 768,
        repetition_penalty: float = 1.05,
        do_sample: bool = False,
        disable_lora: bool = False,
    ) -> str:
        """Multi-turn chat — apply Qwen chat template cho list message.

        Args:
            messages: list[{role, content}]. Phải bắt đầu bằng role=system
                (nếu thiếu, sẽ tự prepend SYSTEM_PROMPT để khớp training).
            max_new_tokens: 768 mặc định cho chat (cần verbose hơn để giải thích).
            repetition_penalty / do_sample: same as `generate()`.
            disable_lora: True → bypass LoRA adapter, dùng base Qwen-Instruct.
                LoRA Stage 4 train trên output ngắn (1-2 môn, không giải thích).
                Khi cần response dài có giải thích (intent recommend), nên bật
                để base model follow prompt tốt hơn.

        Returns:
            Assistant response text (đã decode, skip special tokens).
        """
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        if disable_lora and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                return self._generate_from_messages(
                    messages, max_new_tokens, repetition_penalty, do_sample
                )
        return self._generate_from_messages(
            messages, max_new_tokens, repetition_penalty, do_sample
        )

    @torch.no_grad()
    def _generate_from_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        repetition_penalty: float,
        do_sample: bool,
    ) -> str:
        """Helper chung — apply chat template + generate. Dùng nội bộ."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=8192
        ).to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].size(1) :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _smoke_test() -> None:
    """Smoke test CLI: load model và generate 1 query test."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query",
        default="Em là sinh viên CS HK5, định hướng AI/ML, nên đăng ký môn nào?",
    )
    ap.add_argument("--stub", action="store_true", help="Dùng StubGenerator (không LLM).")
    ap.add_argument("--no-4bit", action="store_true", help="Tắt 4-bit (cần >15GB VRAM).")
    args = ap.parse_args()

    if args.stub:
        gen: StubGenerator | QwenGenerator = StubGenerator()
        # Fake retrieved docs để stub generate.
        fake_docs = [
            {"ten_mon": "Máy học", "ma_mon": "015028", "so_tc": 3, "loai": "tu_chon"},
        ]
        out = gen.generate(args.query, fake_docs)
    else:
        gen = QwenGenerator.load_default(load_4bit=not args.no_4bit)
        out = gen.generate(args.query, max_new_tokens=256)
    print("=== Generated ===")
    print(out)


if __name__ == "__main__":
    _smoke_test()
