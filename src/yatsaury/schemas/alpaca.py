"""Alpaca schema adapter."""
from __future__ import annotations

from yatsaury.models import Sample
from yatsaury.schemas.base import register_schema


class AlpacaSchema:
    """Render samples in Alpaca instruction-tuning format."""

    name = "alpaca"
    supported_types = {"qa", "instruction", "summary"}

    def supports(self, dataset_type: str) -> bool:
        return dataset_type in self.supported_types

    def render(self, sample: Sample) -> dict:
        """Render a Sample as an Alpaca dict."""
        payload = sample.payload
        dtype = str(sample.dataset_type)

        if dtype == "qa":
            return {
                "instruction": payload["question"],
                "input": "",
                "output": payload["answer"],
            }
        if dtype == "instruction":
            return {
                "instruction": payload["instruction"],
                "input": payload.get("input", ""),
                "output": payload["output"],
            }
        # summary
        return {
            "instruction": "Summarize the following passage.",
            "input": payload["passage"],
            "output": payload["summary"],
        }


# Register at import time
register_schema(AlpacaSchema())
