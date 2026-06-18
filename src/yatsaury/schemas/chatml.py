"""ChatML schema adapter."""
from __future__ import annotations

from yatsaury.models import Sample
from yatsaury.schemas.base import register_schema


class ChatmlSchema:
    """Render samples as ChatML message arrays (system/user/assistant)."""

    name = "chatml"
    supported_types = {"qa", "instruction", "summary"}

    def __init__(self, system_prompt: str = "You are a helpful assistant.") -> None:
        self._system_prompt = system_prompt

    def supports(self, dataset_type: str) -> bool:
        return dataset_type in self.supported_types

    def render(self, sample: Sample) -> dict:
        """Render a Sample as a ChatML messages dict."""
        question = sample.payload.get("question", sample.payload.get("instruction", ""))
        answer = sample.payload.get("answer", sample.payload.get("output", ""))
        return {
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }


# Register at import time
register_schema(ChatmlSchema())
