"""Tests for TextLoader source."""
from __future__ import annotations

from pathlib import Path

import pytest

from yatsaury.models import Document, SourceType
from yatsaury.sources.base import SourceLoader, resolve_loader
from yatsaury.sources.text import TextLoader


class TestTextLoaderSupports:
    def test_plain_string(self):
        assert TextLoader().supports("hello world") is True

    def test_txt_file(self):
        assert TextLoader().supports("file.txt") is True

    def test_md_file(self):
        assert TextLoader().supports("file.md") is True

    def test_pdf_file(self):
        assert TextLoader().supports("file.pdf") is False

    def test_url(self):
        assert TextLoader().supports("https://example.com") is False

    def test_http_url(self):
        assert TextLoader().supports("http://example.com/page") is False


class TestTextLoaderLoad:
    def test_load_raw_string(self):
        doc = TextLoader().load("hello world")
        assert isinstance(doc, Document)
        assert doc.source_type == SourceType.text
        assert doc.raw_text == "hello world"
        assert doc.id != ""

    def test_load_txt_file(self, tmp_path: Path):
        f = tmp_path / "sample.txt"
        f.write_text("First line\nSecond line\n", encoding="utf-8")
        doc = TextLoader().load(str(f))
        assert doc.raw_text == "First line\nSecond line\n"
        assert doc.source_type == SourceType.text
        assert doc.id != ""

    def test_load_md_file(self, tmp_path: Path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\nContent here.", encoding="utf-8")
        doc = TextLoader().load(str(f))
        assert doc.raw_text == "# Title\nContent here."
        assert doc.source_type == SourceType.text

    def test_title_from_file_first_line(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("My Document Title\nSome content.", encoding="utf-8")
        doc = TextLoader().load(str(f))
        assert doc.title == "My Document Title"

    def test_title_empty_for_raw_string(self):
        doc = TextLoader().load("some text content")
        assert doc.title == ""

    def test_source_uri_for_raw_string(self):
        doc = TextLoader().load("some text content")
        assert doc.source_uri == "<text>"

    def test_source_uri_for_file(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")
        doc = TextLoader().load(str(f))
        assert doc.source_uri == str(f)


class TestResolveLoader:
    def test_resolves_text_loader(self):
        loader = resolve_loader("hello world", [TextLoader()])
        assert isinstance(loader, SourceLoader)

    def test_raises_for_unsupported(self):
        with pytest.raises(ValueError, match="No loader"):
            resolve_loader("file.pdf", [TextLoader()])
