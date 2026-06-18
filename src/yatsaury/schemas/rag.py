"""RAG schema adapter."""
from __future__ import annotations

from yatsaury.models import Sample
from yatsaury.schemas.base import register_schema


class RagSchema:
    """Render samples as RAG retrieval documents."""

    name = "rag"
    supported_types = {"rag"}

    def supports(self, dataset_type: str) -> bool:
        return dataset_type in self.supported_types

    def render(self, sample: Sample) -> dict:
        """Render a Sample as a RAG document dict."""
        payload = sample.payload
        return {
            "id": sample.chunk_id,
            "text": payload["text"],
            "title": payload.get("title", ""),
            "page": payload.get("page"),
            "source": sample.source_citation.source_uri,
            "char_span": payload.get("char_span"),
        }


# Register at import time
register_schema(RagSchema())
