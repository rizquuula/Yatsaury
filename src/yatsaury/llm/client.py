"""LLM client wrapper with retry and JSON-mode support."""
from __future__ import annotations

import json

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    """Thin wrapper around OpenAI-compatible API with JSON-mode and retry."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        _retry_wait=None,  # injectable for testing
    ) -> None:
        self._model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        wait = _retry_wait if _retry_wait is not None else wait_exponential(min=1, max=10)
        self.complete_json = retry(
            stop=stop_after_attempt(3),
            wait=wait,
            reraise=True,
        )(self._complete_json_impl)

    def _complete_json_impl(self, messages: list[dict], schema_hint: str = "") -> dict:
        """Call the LLM in JSON mode and return parsed dict.

        Raises ValueError if response is not valid JSON after all retries.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {content!r}") from exc
