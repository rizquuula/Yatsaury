"""ShareGPT schema adapter."""
from __future__ import annotations

from yatsaury.models import Sample
from yatsaury.schemas.base import register_schema


class ShareGptSchema:
    """Render samples in ShareGPT conversation format."""

    name = "sharegpt"
    supported_types = {"qa", "instruction", "summary"}

    def supports(self, dataset_type: str) -> bool:
        return dataset_type in self.supported_types

    def render(self, sample: Sample) -> dict:
        """Render a Sample as a ShareGPT conversations dict."""
        payload = sample.payload
        dtype = str(sample.dataset_type)

        if dtype == "qa":
            human_value = payload["question"]
            gpt_value = payload["answer"]
        elif dtype == "instruction":
            instruction = payload["instruction"]
            input_text = payload.get("input", "")
            human_value = f"{instruction}\n{input_text}" if input_text else instruction
            gpt_value = payload["output"]
        else:
            # summary
            human_value = f"Summarize:\n{payload['passage']}"
            gpt_value = payload["summary"]

        return {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ]
        }


# Register at import time
register_schema(ShareGptSchema())
