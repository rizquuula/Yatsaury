"""PdfLoader — loads PDF files using PyMuPDF (fitz) with pypdf fallback."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import fitz  # PyMuPDF — module-level import enables patch("yatsaury.sources.pdf.fitz")

from yatsaury.models import Document, SourceType


class PdfLoader:
    """Load PDF files. Tries PyMuPDF first, falls back to pypdf."""

    def supports(self, uri: str) -> bool:
        """Return True only for local .pdf file paths (case-insensitive).

        Returns False for URLs (http/https) and non-pdf extensions.
        """
        if uri.startswith("http://") or uri.startswith("https://"):
            return False
        p = Path(uri)
        return p.suffix.lower() == ".pdf"

    def load(self, uri: str) -> Document:
        """Load a PDF Document, trying PyMuPDF first then pypdf as fallback."""
        path = Path(uri)
        try:
            return self._load_pymupdf(path)
        except Exception:
            return self._load_pypdf(path)

    def _load_pymupdf(self, path: Path) -> Document:
        pdf = fitz.open(str(path))
        pages_text = []
        for page in pdf:
            pages_text.append(page.get_text())
        raw_text = "\n\n".join(pages_text)
        page_count = len(pdf)
        title = pdf.metadata.get("title", "") or path.stem
        pdf.close()
        return Document(
            id=uuid4().hex,
            source_uri=str(path),
            source_type=SourceType.pdf,
            raw_text=raw_text,
            title=title,
            metadata={"page_count": page_count, "loader": "pymupdf"},
        )

    def _load_pypdf(self, path: Path) -> Document:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
        raw_text = "\n\n".join(pages_text)
        page_count = len(reader.pages)
        info = reader.metadata or {}
        title = getattr(info, "title", None) or path.stem
        return Document(
            id=uuid4().hex,
            source_uri=str(path),
            source_type=SourceType.pdf,
            raw_text=raw_text,
            title=title,
            metadata={"page_count": page_count, "loader": "pypdf"},
        )
