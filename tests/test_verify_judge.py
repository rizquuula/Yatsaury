"""Tests for judge_sample, judge_prompt, verify_samples in yatsaury.quality.verify."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yatsaury.models import Citation, DatasetType, Sample
from yatsaury.quality.verify import JudgeResult, judge_sample, verify_samples


def make_sample(source_text: str, supporting_quote: str) -> Sample:
    return Sample(
        id="s1",
        chunk_id="c1",
        dataset_type=DatasetType.qa,
        payload={"question": "Q", "answer": "A"},
        source_text=source_text,
        supporting_quote=supporting_quote,
        source_citation=Citation(title="T", source_uri="uri://x"),
    )


def test_judge_happy_path() -> None:
    """Mock LLM returning valid grounding result."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    llm = MagicMock()
    llm.complete_json.return_value = {
        "grounding_score": 0.9,
        "is_supported": True,
        "rationale": "Supported.",
    }
    result = judge_sample(sample, llm)
    assert result == JudgeResult(grounding_score=0.9, is_supported=True, rationale="Supported.")


def test_judge_not_supported() -> None:
    """Mock LLM returning low-score unsupported result."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    llm = MagicMock()
    llm.complete_json.return_value = {
        "grounding_score": 0.2,
        "is_supported": False,
        "rationale": "Not found.",
    }
    result = judge_sample(sample, llm)
    assert result == JudgeResult(grounding_score=0.2, is_supported=False, rationale="Not found.")


def test_judge_missing_key_raises() -> None:
    """LLM response missing required key raises ValueError."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    llm = MagicMock()
    llm.complete_json.return_value = {"is_supported": True}
    with pytest.raises(ValueError):
        judge_sample(sample, llm)


def test_judge_score_clamped_high() -> None:
    """grounding_score above 1.0 is clamped to 1.0."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    llm = MagicMock()
    llm.complete_json.return_value = {
        "grounding_score": 1.5,
        "is_supported": True,
        "rationale": "...",
    }
    result = judge_sample(sample, llm)
    assert result.grounding_score == 1.0


def test_judge_score_clamped_low() -> None:
    """grounding_score below 0.0 is clamped to 0.0."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    llm = MagicMock()
    llm.complete_json.return_value = {
        "grounding_score": -0.5,
        "is_supported": False,
        "rationale": "...",
    }
    result = judge_sample(sample, llm)
    assert result.grounding_score == 0.0


def test_judge_model_override() -> None:
    """When judge_model is provided, a new LLMClient is created with that model."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    real_llm_mock = MagicMock()
    real_llm_mock._model = "base-model"
    real_llm_mock._client = MagicMock()
    real_llm_mock._client.base_url = "http://localhost:1234/v1"
    real_llm_mock._client.api_key = "test-key"

    with patch("yatsaury.quality.verify.LLMClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.complete_json.return_value = {
            "grounding_score": 0.9,
            "is_supported": True,
            "rationale": "x",
        }
        result = judge_sample(sample, real_llm_mock, judge_model="other-model")
        MockClient.assert_called_once()
        call_kwargs = MockClient.call_args
        assert (
            call_kwargs.kwargs.get("model") == "other-model"
            or (len(call_kwargs.args) > 2 and call_kwargs.args[2] == "other-model")
        )
    assert result.grounding_score == 0.9


def test_verify_samples_pass() -> None:
    """Sample passing quote_check and judge with score 0.9 is kept."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    llm = MagicMock()
    llm.complete_json.return_value = {
        "grounding_score": 0.9,
        "is_supported": True,
        "rationale": "Good.",
    }
    results = verify_samples([sample], llm, min_score=0.7)
    assert len(results) == 1
    assert results[0].grounding_score == 0.9
    assert results[0].verified is True


def test_verify_samples_fail_quote_check() -> None:
    """Sample failing quote_check is dropped without calling LLM."""
    sample = make_sample("The sky is blue.", "The grass is greener.")
    llm = MagicMock()
    results = verify_samples([sample], llm)
    assert len(results) == 0
    assert llm.complete_json.call_count == 0


def test_verify_samples_fail_min_score() -> None:
    """Sample passing quote_check but with judge score 0.4 is dropped when min_score=0.7."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    llm = MagicMock()
    llm.complete_json.return_value = {
        "grounding_score": 0.4,
        "is_supported": True,
        "rationale": "Low.",
    }
    results = verify_samples([sample], llm, min_score=0.7)
    assert len(results) == 0


def test_verify_samples_no_judge() -> None:
    """With use_judge=False, judge is not called; all quote-passing samples are kept."""
    sample = make_sample("The sky is blue.", "The sky is blue.")
    llm = MagicMock()
    results = verify_samples([sample], llm, use_judge=False)
    assert len(results) == 1
    assert llm.complete_json.call_count == 0
    # grounding_score unchanged (default 0.0)
    assert results[0].grounding_score == 0.0


def test_verify_samples_mixed() -> None:
    """3 samples: one fails quote_check, one fails min_score, one passes."""
    s_fail_quote = make_sample("Sky is blue.", "Not in source at all xyz.")
    s_fail_score = make_sample("Sky is blue.", "Sky is blue.")
    s_pass = make_sample(
        "The ocean is deep and vast.", "The ocean is deep and vast."
    )
    # We need s_fail_score and s_pass to get different judge scores.
    # Use side_effect to return different values per call.
    llm = MagicMock()
    llm.complete_json.side_effect = [
        {"grounding_score": 0.3, "is_supported": False, "rationale": "Low."},
        {"grounding_score": 0.95, "is_supported": True, "rationale": "High."},
    ]
    results = verify_samples(
        [s_fail_quote, s_fail_score, s_pass], llm, min_score=0.7
    )
    assert len(results) == 1
    assert results[0].grounding_score == 0.95
    assert results[0].verified is True
