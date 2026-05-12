"""Generation module — Giai đoạn 5.

- `prompt_templates`: SYSTEM + USER template tiếng Việt (khớp training M5).
- `QwenGenerator`: load Qwen2.5-7B-Instruct 4-bit + LoRA adapter M5 từ Kaggle,
  generate response cho 1 query.
- `StubGenerator`: fallback template-based generator (không cần GPU).
"""

from .generator import QwenGenerator, StubGenerator, parse_recommendations
from .prompt_templates import SYSTEM_PROMPT, build_user_message, format_context

__all__ = [
    "QwenGenerator",
    "StubGenerator",
    "SYSTEM_PROMPT",
    "build_user_message",
    "format_context",
    "parse_recommendations",
]
