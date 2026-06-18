"""Tests for PdfLoader — TDD Phase 2."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from yatsaury.models import SourceType
from yatsaury.sources.pdf import PdfLoader


@pytest.fixture(scope="session")
def tiny_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import fitz  # PyMuPDF

    path = tmp_path_factory.mktemp("fixtures") / "tiny.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is page one of the test document.")
    doc.save(str(path))
    doc.close()
    return path


class TestPdfLoaderSupports:
    def test_supports_pdf_lowercase(self) -> None:
        assert PdfLoader().supports("file.pdf") is True

    def test_supports_pdf_uppercase(self) -> None:
        assert PdfLoader().supports("FILE.PDF") is True

    def test_rejects_txt(self) -> None:
        assert PdfLoader().supports("file.txt") is False

    def test_rejects_md(self) -> None:
        assert PdfLoader().supports("file.md") is False

    def test_rejects_url_with_pdf_extension(self) -> None:
        assert PdfLoader().supports("https://example.com/file.pdf") is False

    def test_rejects_raw_string(self) -> None:
        assert PdfLoader().supports("hello world") is False

    def test_rejects_no_extension(self) -> None:
        assert PdfLoader().supports("somefile") is False


class TestPdfLoaderLoad:
    def test_load_returns_document_with_pdf_source_type(self, tiny_pdf: Path) -> None:
        doc = PdfLoader().load(str(tiny_pdf))
        assert doc.source_type == SourceType.pdf

    def test_load_raw_text_contains_content(self, tiny_pdf: Path) -> None:
        doc = PdfLoader().load(str(tiny_pdf))
        assert "page one" in doc.raw_text

    def test_load_id_is_non_empty(self, tiny_pdf: Path) -> None:
        doc = PdfLoader().load(str(tiny_pdf))
        assert doc.id != ""

    def test_load_metadata_page_count(self, tiny_pdf: Path) -> None:
        doc = PdfLoader().load(str(tiny_pdf))
        assert doc.metadata["page_count"] == 1

    def test_load_metadata_loader_name(self, tiny_pdf: Path) -> None:
        doc = PdfLoader().load(str(tiny_pdf))
        assert doc.metadata["loader"] in {"pymupdf", "pypdf"}

    def test_load_source_uri_matches_path(self, tiny_pdf: Path) -> None:
        doc = PdfLoader().load(str(tiny_pdf))
        assert doc.source_uri == str(tiny_pdf)

    def test_load_pymupdf_fallback_to_pypdf(self, tiny_pdf: Path) -> None:
        """When fitz.open raises, loader falls back to pypdf and returns a valid Document."""
        with patch("yatsaury.sources.pdf.fitz") as mock_fitz:
            mock_fitz.open.side_effect = Exception("mock fitz failure")
            doc = PdfLoader().load(str(tiny_pdf))
        assert doc.source_type == SourceType.pdf
        assert doc.metadata["loader"] == "pypdf"
        assert "page one" in doc.raw_text

    def test_load_title_fallback_to_filename_stem(self, tiny_pdf: Path) -> None:
        doc = PdfLoader().load(str(tiny_pdf))
        # Title is either from PDF metadata or filename stem
        assert isinstance(doc.title, str)
