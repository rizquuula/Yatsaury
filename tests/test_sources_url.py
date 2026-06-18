"""Tests for UrlLoader — TDD Phase 2."""
from __future__ import annotations

import httpx
import pytest
import respx

from yatsaury.models import SourceType
from yatsaury.sources.url import UrlLoader


class TestUrlLoaderSupports:
    def test_supports_https(self) -> None:
        assert UrlLoader().supports("https://example.com") is True

    def test_supports_http(self) -> None:
        assert UrlLoader().supports("http://example.com/page") is True

    def test_rejects_txt(self) -> None:
        assert UrlLoader().supports("file.txt") is False

    def test_rejects_pdf(self) -> None:
        assert UrlLoader().supports("file.pdf") is False

    def test_rejects_raw_text(self) -> None:
        assert UrlLoader().supports("some random text") is False


class TestUrlLoaderLoad:
    _html = (
        "<html><body><article>"
        "<p>The Prophet Muhammad was born in Mecca.</p>"
        "</article></body></html>"
    )

    @respx.mock
    def test_load_returns_url_source_type(self) -> None:
        respx.get("https://example.com/sirah").mock(
            return_value=httpx.Response(200, text=self._html)
        )
        doc = UrlLoader().load("https://example.com/sirah")
        assert doc.source_type == SourceType.url

    @respx.mock
    def test_load_raw_text_contains_content(self) -> None:
        respx.get("https://example.com/sirah").mock(
            return_value=httpx.Response(200, text=self._html)
        )
        doc = UrlLoader().load("https://example.com/sirah")
        assert "Mecca" in doc.raw_text or "Muhammad" in doc.raw_text

    @respx.mock
    def test_load_source_uri(self) -> None:
        respx.get("https://example.com/sirah").mock(
            return_value=httpx.Response(200, text=self._html)
        )
        doc = UrlLoader().load("https://example.com/sirah")
        assert doc.source_uri == "https://example.com/sirah"

    @respx.mock
    def test_load_metadata_url(self) -> None:
        respx.get("https://example.com/sirah").mock(
            return_value=httpx.Response(200, text=self._html)
        )
        doc = UrlLoader().load("https://example.com/sirah")
        assert doc.metadata["url"] == "https://example.com/sirah"

    @respx.mock
    def test_load_metadata_loader(self) -> None:
        respx.get("https://example.com/sirah").mock(
            return_value=httpx.Response(200, text=self._html)
        )
        doc = UrlLoader().load("https://example.com/sirah")
        assert doc.metadata["loader"] == "trafilatura"

    @respx.mock
    def test_load_raises_on_http_error(self) -> None:
        respx.get("https://example.com/notfound").mock(
            return_value=httpx.Response(404)
        )
        with pytest.raises(httpx.HTTPStatusError):
            UrlLoader().load("https://example.com/notfound")

    @respx.mock
    def test_load_fallback_when_trafilatura_returns_none(self) -> None:
        """When trafilatura.extract returns None, raw HTML text is used as fallback."""
        from unittest.mock import patch

        html = "<html><body><p>Fallback content here.</p></body></html>"
        respx.get("https://example.com/page").mock(
            return_value=httpx.Response(200, text=html)
        )
        with patch("yatsaury.sources.url.trafilatura") as mock_traf:
            mock_traf.extract.return_value = None
            mock_traf.extract_metadata.return_value = None
            doc = UrlLoader().load("https://example.com/page")
        assert doc.raw_text.strip() != ""

    @respx.mock
    def test_load_id_is_non_empty(self) -> None:
        respx.get("https://example.com/sirah").mock(
            return_value=httpx.Response(200, text=self._html)
        )
        doc = UrlLoader().load("https://example.com/sirah")
        assert doc.id != ""
