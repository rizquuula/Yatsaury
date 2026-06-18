"""Tests for Yatsaury Settings config."""

import os
from pathlib import Path

import pytest

from yatsaury.config import Settings


class TestSettingsDefaults:
    def test_default_base_url(self) -> None:
        s = Settings()
        assert s.base_url == "https://api.openai.com/v1"

    def test_default_model(self) -> None:
        s = Settings()
        assert s.model == "gpt-4o-mini"

    def test_default_chunk_size(self) -> None:
        s = Settings()
        assert s.chunk_size == 512

    def test_default_chunk_overlap(self) -> None:
        s = Settings()
        assert s.chunk_overlap == 64

    def test_default_per_chunk(self) -> None:
        s = Settings()
        assert s.per_chunk == 3

    def test_default_min_score(self) -> None:
        s = Settings()
        assert s.min_score == 0.7

    def test_default_lang(self) -> None:
        s = Settings()
        assert s.lang == "auto"

    def test_default_workspace(self) -> None:
        s = Settings()
        assert s.workspace == Path("./.yatsaury")

    def test_default_web_host(self) -> None:
        s = Settings()
        assert s.web_host == "127.0.0.1"

    def test_default_web_port(self) -> None:
        s = Settings()
        assert s.web_port == 8080

    def test_api_key_is_secret_str(self) -> None:
        from pydantic import SecretStr
        s = Settings()
        assert isinstance(s.api_key, SecretStr)

    def test_api_key_masked_in_repr(self) -> None:
        from pydantic import SecretStr
        s = Settings(api_key=SecretStr("super-secret-key"))
        repr_str = repr(s.api_key)
        assert "super-secret-key" not in repr_str
        assert "**" in repr_str


class TestSettingsEnvOverride:
    def test_model_override_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YATSAURY_MODEL", "gpt-4o")
        s = Settings()
        assert s.model == "gpt-4o"

    def test_chunk_size_override_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YATSAURY_CHUNK_SIZE", "1024")
        s = Settings()
        assert s.chunk_size == 1024

    def test_web_port_override_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YATSAURY_WEB_PORT", "9090")
        s = Settings()
        assert s.web_port == 9090


class TestSettingsOpenAIFallback:
    def test_openai_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fallback")
        monkeypatch.delenv("YATSAURY_API_KEY", raising=False)
        s = Settings()
        assert s.api_key.get_secret_value() == "sk-test-fallback"

    def test_openai_base_url_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.openai.com/v1")
        monkeypatch.delenv("YATSAURY_BASE_URL", raising=False)
        s = Settings()
        assert s.base_url == "https://custom.openai.com/v1"
