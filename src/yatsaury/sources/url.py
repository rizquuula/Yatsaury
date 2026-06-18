"""UrlLoader — fetches and extracts content from HTTP/HTTPS URLs."""
from __future__ import annotations

import re
from uuid import uuid4

import httpx
import trafilatura

from yatsaury.models import Document, SourceType


class UrlLoader:
    """Load web pages via HTTP, extracting main content with trafilatura."""

    def supports(self, uri: str) -> bool:
        """Return True for http:// and https:// URIs only."""
        return uri.startswith("http://") or uri.startswith("https://")

    def load(self, uri: str) -> Document:
        """Fetch URL and extract text content.

        Uses httpx for HTTP, trafilatura for content extraction.
        Falls back to stripped HTML text if trafilatura returns None.
        Raises httpx.HTTPStatusError on non-2xx responses.
        """
        response = httpx.get(uri, follow_redirects=True, timeout=30)
        response.raise_for_status()

        html = response.text
        extracted = trafilatura.extract(html)
        meta = trafilatura.extract_metadata(html)

        if extracted:
            raw_text = extracted
        else:
            # Minimal HTML tag strip as fallback
            raw_text = re.sub(r"<[^>]+>", " ", html)
            raw_text = re.sub(r"\s+", " ", raw_text).strip()

        title = ""
        if meta and getattr(meta, "title", None):
            title = meta.title

        return Document(
            id=uuid4().hex,
            source_uri=uri,
            source_type=SourceType.url,
            raw_text=raw_text,
            title=title,
            metadata={"url": uri, "loader": "trafilatura"},
        )
