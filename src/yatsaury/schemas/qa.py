"""QA (flat question/answer) schema adapter."""
from __future__ import annotations

from yatsaury.models import Sample
from yatsaury.schemas.base import register_schema


class QaSchema:
    """Render samples as flat {"question": ..., "answer": ...} dicts."""

    name = "qa"
    supported_types = {"qa", "instruction"}

    def supports(self, dataset_type: str) -> bool:
        return dataset_type in self.supported_types

    def render(self, sample: Sample) -> dict:
        """Render a Sample as a flat Q&A dict."""
        if sample.dataset_type.value == "instruction":
            return {
                "question": sample.payload.get("instruction", ""),
                "answer": sample.payload.get("output", ""),
            }
        return {
            "question": sample.payload.get("question", ""),
            "answer": sample.payload.get("answer", ""),
        }


# Register at import time
register_schema(QaSchema())
