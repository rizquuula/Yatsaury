"""Tests for resolve_loader auto-registration — TDD Phase 2."""
from __future__ import annotations

import pytest

from yatsaury.sources.base import resolve_loader
from yatsaury.sources.pdf import PdfLoader
from yatsaury.sources.text import TextLoader
from yatsaury.sources.url import UrlLoader


class TestResolveLoaderDefaults:
    def test_resolves_pdf_uri_to_pdf_loader(self) -> None:
        loader = resolve_loader("file.pdf")
        assert isinstance(loader, PdfLoader)

    def test_resolves_https_uri_to_url_loader(self) -> None:
        loader = resolve_loader("https://example.com")
        assert isinstance(loader, UrlLoader)

    def test_resolves_http_uri_to_url_loader(self) -> None:
        loader = resolve_loader("http://example.com/page")
        assert isinstance(loader, UrlLoader)

    def test_resolves_txt_file_to_text_loader(self) -> None:
        loader = resolve_loader("file.txt")
        assert isinstance(loader, TextLoader)

    def test_resolves_md_file_to_text_loader(self) -> None:
        loader = resolve_loader("file.md")
        assert isinstance(loader, TextLoader)

    def test_resolves_raw_text_to_text_loader(self) -> None:
        loader = resolve_loader("hello world")
        assert isinstance(loader, TextLoader)

    def test_resolves_unknown_extension_to_text_loader(self) -> None:
        """TextLoader is the catch-all — unknown extensions with no suffix route to text."""
        loader = resolve_loader("somefile")
        assert isinstance(loader, TextLoader)


class TestResolveLoaderCustomLoaders:
    def test_custom_loaders_override_defaults(self) -> None:
        """Passing explicit loaders list skips auto-registration."""
        loader = resolve_loader("file.txt", loaders=[TextLoader()])
        assert isinstance(loader, TextLoader)

    def test_empty_custom_loaders_raises_value_error(self) -> None:
        """An empty explicit list raises ValueError with no fallback."""
        with pytest.raises(ValueError, match="No loader found"):
            resolve_loader("file.pdf", loaders=[])

    def test_empty_custom_loaders_with_any_uri_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_loader("https://example.com", loaders=[])

    def test_custom_loaders_order_matters(self) -> None:
        """First matching loader wins."""
        # PdfLoader before TextLoader — pdf URI hits PdfLoader
        loader = resolve_loader("document.pdf", loaders=[PdfLoader(), TextLoader()])
        assert isinstance(loader, PdfLoader)
