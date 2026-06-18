"""Raw schema adapter."""
from __future__ import annotations

from yatsaury.models import Sample
from yatsaury.schemas.base import register_schema


class RawSchema:
    """Render samples as their full serialized model dump."""

    name = "raw"
    supported_types = {"qa", "instruction", "summary", "rag"}

    def supports(self, dataset_type: str) -> bool:
        return dataset_type in self.supported_types

    def render(self, sample: Sample) -> dict:
        """Render a Sample as its full model_dump (JSON-safe)."""
        return sample.model_dump(mode="json")


# Register at import time
register_schema(RawSchema())
