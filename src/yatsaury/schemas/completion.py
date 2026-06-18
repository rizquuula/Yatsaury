"""Completion schema adapter."""
from __future__ import annotations

from yatsaury.models import Sample
from yatsaury.schemas.base import register_schema


class CompletionSchema:
    """Render samples in OpenAI-style completion format."""

    name = "completion"
    supported_types = {"qa", "instruction", "summary"}

    def supports(self, dataset_type: str) -> bool:
        return dataset_type in self.supported_types

    def render(self, sample: Sample) -> dict:
        """Render a Sample as a completion prompt/completion dict."""
        payload = sample.payload
        dtype = str(sample.dataset_type)

        if dtype == "qa":
            prompt = f"{payload['question']}\n\n"
            completion = f" {payload['answer']}"
        elif dtype == "instruction":
            prompt = f"{payload['instruction']}\n\n"
            completion = f" {payload['output']}"
        else:
            # summary
            prompt = f"Summarize:\n{payload['passage']}\n\n"
            completion = f" {payload['summary']}"

        return {"prompt": prompt, "completion": completion}


# Register at import time
register_schema(CompletionSchema())
