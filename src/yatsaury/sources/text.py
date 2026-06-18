"""TextLoader — loads plain text strings or .txt/.md files."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from yatsaury.models import Document, SourceType


class TextLoader:
    """Load plain text content from a raw string or a .txt/.md file."""

    _FILE_SUFFIXES = {".txt", ".md"}

    def supports(self, uri: str) -> bool:
        """Return True for plain strings, .txt files, and .md files.

        Returns False for URLs and .pdf files.
        """
        if uri.startswith("http://") or uri.startswith("https://"):
            return False
        p = Path(uri)
        # Check if it looks like a file path with a known extension
        if p.suffix:
            return p.suffix.lower() in self._FILE_SUFFIXES
        # No extension → treat as raw text
        return True

    def load(self, uri: str) -> Document:
        """Load a Document from a raw string or a .txt/.md file path."""
        p = Path(uri)
        if p.suffix.lower() in self._FILE_SUFFIXES and p.exists():
            raw_text = p.read_text(encoding="utf-8")
            # Title = first non-empty line, up to 80 chars
            title = ""
            for line in raw_text.splitlines():
                stripped = line.strip()
                if stripped:
                    title = stripped[:80]
                    break
            return Document(
                id=uuid4().hex,
                source_uri=str(p),
                source_type=SourceType.text,
                raw_text=raw_text,
                title=title,
            )
        else:
            # Treat uri as raw text content
            return Document(
                id=uuid4().hex,
                source_uri="<text>",
                source_type=SourceType.text,
                raw_text=uri,
                title="",
            )
