"""Yatsaury settings — layered configuration via pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = SecretStr("")
    model: str = "gpt-4o-mini"
    judge_model: str = ""
    judge_batch_size: int = 1

    # Generation
    chunk_size: int = 512
    chunk_overlap: int = 64
    per_chunk: int = 3
    min_score: float = 70.0
    lang: str = "auto"

    # Web / workspace
    workspace: Path = Path("./.yatsaury")
    web_host: str = "127.0.0.1"
    web_port: int = 8080

    model_config = SettingsConfigDict(
        env_prefix="YATSAURY_",
        env_file=".env",
        toml_file="config.toml",
        secrets_dir=None,
        extra="ignore",
    )

    @model_validator(mode="after")
    def apply_openai_fallbacks(self) -> Settings:
        """Fall back to OPENAI_API_KEY / OPENAI_BASE_URL if YATSAURY_ vars not set."""
        if not self.api_key.get_secret_value():
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if openai_key:
                object.__setattr__(self, "api_key", SecretStr(openai_key))
        if self.base_url == "https://api.openai.com/v1":
            openai_url = os.environ.get("OPENAI_BASE_URL", "")
            if openai_url:
                object.__setattr__(self, "base_url", openai_url)
        return self
