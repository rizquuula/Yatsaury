"""Tests for LLMClient — mocked, no real HTTP."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from tenacity import wait_none

from yatsaury.llm.client import LLMClient


def make_mock_response(content: str) -> MagicMock:
    """Build a mock that looks like an OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMClientInit:
    def test_base_url_passed_to_openai(self):
        with patch("yatsaury.llm.client.OpenAI") as mock_openai:
            LLMClient(base_url="http://localhost:11434/v1", api_key="test", model="llama3")
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:11434/v1"


class TestLLMClientCompleteJson:
    def test_happy_path_returns_parsed_dict(self):
        with patch("yatsaury.llm.client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            payload = {"question": "Q?", "answer": "A."}
            mock_client.chat.completions.create.return_value = make_mock_response(
                json.dumps(payload)
            )
            client = LLMClient(base_url="http://x", api_key="k", model="m")
            result = client.complete_json([{"role": "user", "content": "hi"}])
            assert result == payload

    def test_retry_on_api_error(self):
        import openai
        with patch("yatsaury.llm.client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            good_payload = {"ok": True}
            # First call raises, second succeeds
            mock_client.chat.completions.create.side_effect = [
                openai.APIError("server error", request=MagicMock(), body=None),
                make_mock_response(json.dumps(good_payload)),
            ]
            client = LLMClient(base_url="http://x", api_key="k", model="m", _retry_wait=wait_none())
            result = client.complete_json([{"role": "user", "content": "hi"}])
            assert result == good_payload
            assert mock_client.chat.completions.create.call_count == 2

    def test_invalid_json_raises_value_error(self):
        with patch("yatsaury.llm.client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.return_value = make_mock_response(
                "not valid json {{{"
            )
            client = LLMClient(base_url="http://x", api_key="k", model="m", _retry_wait=wait_none())
            # With retries exhausted (all return bad JSON), raises ValueError
            with pytest.raises(ValueError, match="Invalid JSON"):
                client.complete_json([{"role": "user", "content": "hi"}])
