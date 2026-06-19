"""Tests for judge_sample, judge_sample_batch, verify_samples in yatsaury.quality.verify."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yatsaury.models import Citation, DatasetType, Sample
from yatsaury.quality.verify import (
    JudgeResult,
    judge_sample,
    judge_sample_batch,
    verify_samples,
)


def make_sample(
    source_text: str,
    supporting_quote: str,
    dataset_type: DatasetType = DatasetType.qa,
    payload: dict | None = None,
) -> Sample:
    if payload is None:
        payload = {"question": "Q?", "answer": "A."}
    return Sample(
        id="s1",
        chunk_id="c1",
        dataset_type=dataset_type,
        payload=payload,
        source_text=source_text,
        supporting_quote=supporting_quote,
        source_citation=Citation(title="T", source_uri="uri://x"),
    )


# ── Single-item judge ────────────────────────────────────────────────────────

class TestJudgeSample:
    def test_happy_path(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": 85,
            "is_accepted": True,
            "rationale": "Grounding: strong. Clarity: clear. Completeness: full. Relevance: on-topic.",
        }
        result = judge_sample(sample, llm)
        assert result == JudgeResult(
            quality_score=85,
            is_accepted=True,
            rationale="Grounding: strong. Clarity: clear. Completeness: full. Relevance: on-topic.",
        )

    def test_low_score_not_accepted(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": 25,
            "is_accepted": False,
            "rationale": "Grounding: weak. Clarity: poor. Completeness: partial. Relevance: low.",
        }
        result = judge_sample(sample, llm)
        assert result.quality_score == 25
        assert result.is_accepted is False

    def test_missing_key_raises(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {"is_accepted": True}
        with pytest.raises(ValueError):
            judge_sample(sample, llm)

    def test_score_clamped_high(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": 150,
            "is_accepted": True,
            "rationale": "...",
        }
        result = judge_sample(sample, llm)
        assert result.quality_score == 100

    def test_score_clamped_low(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": -10,
            "is_accepted": False,
            "rationale": "...",
        }
        result = judge_sample(sample, llm)
        assert result.quality_score == 0

    def test_score_float_rounded(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": 72.7,
            "is_accepted": True,
            "rationale": "...",
        }
        result = judge_sample(sample, llm)
        assert result.quality_score == 73

    def test_judge_model_override(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        real_llm_mock = MagicMock()
        real_llm_mock._client.base_url = "http://localhost:1234/v1"
        real_llm_mock._client.api_key = "test-key"

        with patch("yatsaury.quality.verify.LLMClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.complete_json.return_value = {
                "quality_score": 90,
                "is_accepted": True,
                "rationale": "x",
            }
            result = judge_sample(sample, real_llm_mock, judge_model="other-model")
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args
            assert (
                call_kwargs.kwargs.get("model") == "other-model"
                or (len(call_kwargs.args) > 2 and call_kwargs.args[2] == "other-model")
            )
        assert result.quality_score == 90


# ── Per-type prompt dispatch ─────────────────────────────────────────────────

class TestPerTypePrompts:
    def _get_messages(self, sample: Sample) -> list[dict]:
        from yatsaury.quality.verify import _judge_messages
        return _judge_messages(sample)

    def test_qa_prompt_contains_question_answer(self) -> None:
        sample = make_sample(
            "Source.", "Source.",
            dataset_type=DatasetType.qa,
            payload={"question": "What?", "answer": "This."},
        )
        messages = self._get_messages(sample)
        user_content = messages[1]["content"]
        assert "Question: What?" in user_content
        assert "Answer: This." in user_content

    def test_qa_system_mentions_quality_dimensions(self) -> None:
        sample = make_sample("S.", "S.", dataset_type=DatasetType.qa)
        messages = self._get_messages(sample)
        system = messages[0]["content"]
        assert "Grounding" in system
        assert "Clarity" in system
        assert "Completeness" in system
        assert "Relevance" in system

    def test_instruction_prompt_contains_instruction_output(self) -> None:
        sample = make_sample(
            "Source.", "Source.",
            dataset_type=DatasetType.instruction,
            payload={"instruction": "Do this.", "input": "", "output": "Done."},
        )
        messages = self._get_messages(sample)
        user_content = messages[1]["content"]
        assert "Instruction: Do this." in user_content
        assert "Output: Done." in user_content

    def test_instruction_system_mentions_triples(self) -> None:
        sample = make_sample(
            "S.", "S.", dataset_type=DatasetType.instruction,
            payload={"instruction": "I", "input": "", "output": "O"},
        )
        messages = self._get_messages(sample)
        assert "instruction" in messages[0]["content"].lower()

    def test_summary_prompt_contains_summary_key_points(self) -> None:
        sample = make_sample(
            "Long passage.", "Long passage.",
            dataset_type=DatasetType.summary,
            payload={"passage": "Long passage.", "summary": "Short.", "key_points": ["p1", "p2"]},
        )
        messages = self._get_messages(sample)
        user_content = messages[1]["content"]
        assert "Summary: Short." in user_content
        assert "p1" in user_content

    def test_summary_system_mentions_summarization(self) -> None:
        sample = make_sample(
            "S.", "S.", dataset_type=DatasetType.summary,
            payload={"passage": "S.", "summary": "S.", "key_points": []},
        )
        messages = self._get_messages(sample)
        assert "summarization" in messages[0]["content"].lower()

    def test_score_rubric_in_all_types(self) -> None:
        for dtype, payload in [
            (DatasetType.qa, {"question": "Q", "answer": "A"}),
            (DatasetType.instruction, {"instruction": "I", "input": "", "output": "O"}),
            (DatasetType.summary, {"passage": "P", "summary": "S", "key_points": []}),
        ]:
            sample = make_sample("Src.", "Src.", dataset_type=dtype, payload=payload)
            messages = self._get_messages(sample)
            system = messages[0]["content"]
            assert "0–20" in system or "0-20" in system
            assert "81–100" in system or "81-100" in system


# ── Batch judgment ───────────────────────────────────────────────────────────

class TestJudgeSampleBatch:
    def test_empty_returns_empty(self) -> None:
        llm = MagicMock()
        results = judge_sample_batch([], llm)
        assert results == []
        llm.complete_json.assert_not_called()

    def test_single_item_uses_single_prompt(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": 80,
            "is_accepted": True,
            "rationale": "Good.",
        }
        results = judge_sample_batch([sample], llm)
        assert len(results) == 1
        assert results[0].quality_score == 80
        llm.complete_json.assert_called_once()

    def test_batch_sends_one_call_for_multiple_items(self) -> None:
        source = "Shared source text about Islam."
        samples = [
            make_sample(source, source, payload={"question": f"Q{i}?", "answer": f"A{i}."})
            for i in range(3)
        ]
        llm = MagicMock()
        llm.complete_json.return_value = {
            "results": [
                {"quality_score": 80, "is_accepted": True, "rationale": "Good."},
                {"quality_score": 55, "is_accepted": False, "rationale": "Mediocre."},
                {"quality_score": 90, "is_accepted": True, "rationale": "Excellent."},
            ]
        }
        results = judge_sample_batch(samples, llm)
        assert llm.complete_json.call_count == 1
        assert len(results) == 3
        assert results[0].quality_score == 80
        assert results[1].quality_score == 55
        assert results[1].is_accepted is False
        assert results[2].quality_score == 90

    def test_batch_mismatch_returns_empty(self) -> None:
        source = "Source text."
        samples = [
            make_sample(source, source, payload={"question": f"Q{i}?", "answer": f"A{i}."})
            for i in range(3)
        ]
        llm = MagicMock()
        llm.complete_json.return_value = {
            "results": [
                {"quality_score": 80, "is_accepted": True, "rationale": "Good."},
            ]
        }
        results = judge_sample_batch(samples, llm)
        assert results == []

    def test_batch_prompt_contains_shared_source(self) -> None:
        source = "Unique source content XYZ."
        samples = [
            make_sample(source, source, payload={"question": f"Q{i}?", "answer": f"A{i}."})
            for i in range(2)
        ]
        llm = MagicMock()
        llm.complete_json.return_value = {
            "results": [
                {"quality_score": 75, "is_accepted": True, "rationale": "OK."},
                {"quality_score": 80, "is_accepted": True, "rationale": "Good."},
            ]
        }
        judge_sample_batch(samples, llm)
        call_messages = llm.complete_json.call_args[0][0]
        user_content = call_messages[1]["content"]
        assert "Unique source content XYZ." in user_content
        assert "Item 1:" in user_content
        assert "Item 2:" in user_content


# ── verify_samples ───────────────────────────────────────────────────────────

class TestVerifySamples:
    def test_pass(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": 85,
            "is_accepted": True,
            "rationale": "Good.",
        }
        results = verify_samples([sample], llm, min_score=70.0)
        assert len(results) == 1
        assert results[0].quality_score == 85.0
        assert results[0].verified is True

    def test_fail_quote_check(self) -> None:
        sample = make_sample("The sky is blue.", "The grass is greener.")
        llm = MagicMock()
        results = verify_samples([sample], llm)
        assert len(results) == 0
        assert llm.complete_json.call_count == 0

    def test_fail_min_score(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": 40,
            "is_accepted": False,
            "rationale": "Poor.",
        }
        results = verify_samples([sample], llm, min_score=70.0)
        assert len(results) == 0

    def test_no_judge_skips_llm(self) -> None:
        sample = make_sample("The sky is blue.", "The sky is blue.")
        llm = MagicMock()
        results = verify_samples([sample], llm, use_judge=False)
        assert len(results) == 1
        assert llm.complete_json.call_count == 0
        assert results[0].quality_score == 0.0

    def test_mixed_samples(self) -> None:
        s_fail_quote = make_sample("Sky is blue.", "Not in source at all xyz.")
        s_fail_score = make_sample("Sky is blue.", "Sky is blue.")
        s_pass = make_sample("The ocean is deep and vast.", "The ocean is deep and vast.")
        llm = MagicMock()
        llm.complete_json.side_effect = [
            {"quality_score": 35, "is_accepted": False, "rationale": "Poor."},
            {"quality_score": 88, "is_accepted": True, "rationale": "Great."},
        ]
        results = verify_samples(
            [s_fail_quote, s_fail_score, s_pass], llm, min_score=70.0
        )
        assert len(results) == 1
        assert results[0].quality_score == 88.0
        assert results[0].verified is True

    def test_batch_size_groups_same_source(self) -> None:
        source = "Shared source text."
        samples = [
            make_sample(source, source, payload={"question": f"Q{i}?", "answer": f"A{i}."})
            for i in range(3)
        ]
        llm = MagicMock()
        llm.complete_json.return_value = {
            "results": [
                {"quality_score": 80, "is_accepted": True, "rationale": "Good."},
                {"quality_score": 75, "is_accepted": True, "rationale": "Good."},
                {"quality_score": 85, "is_accepted": True, "rationale": "Great."},
            ]
        }
        results = verify_samples(samples, llm, min_score=70.0, batch_size=3)
        assert llm.complete_json.call_count == 1
        assert len(results) == 3

    def test_batch_size_1_calls_per_sample(self) -> None:
        source = "Shared source text."
        samples = [
            make_sample(source, source, payload={"question": f"Q{i}?", "answer": f"A{i}."})
            for i in range(3)
        ]
        llm = MagicMock()
        llm.complete_json.return_value = {
            "quality_score": 80,
            "is_accepted": True,
            "rationale": "Good.",
        }
        results = verify_samples(samples, llm, min_score=70.0, batch_size=1)
        assert llm.complete_json.call_count == 3
        assert len(results) == 3
