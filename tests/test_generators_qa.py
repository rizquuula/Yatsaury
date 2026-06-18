"""Tests for QaGenerator."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yatsaury.generators.base import get_generator
from yatsaury.generators.qa import QaGenerator
from yatsaury.models import Chunk, DatasetType


def make_chunk() -> Chunk:
    return Chunk(
        id="chk_abc12345_0000",
        doc_id="abc12345",
        text="The Prophet Muhammad (peace be upon him) was born in Makkah in 570 CE.",
        token_count=20,
        char_span=(0, 70),
        ordinal=0,
    )


class TestQaGenerator:
    def test_returns_sample_on_valid_response(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "pairs": [
                {
                    "question": "Where was the Prophet born?",
                    "answer": "The Prophet was born in Makkah.",
                    "supporting_quote": "born in Makkah in 570 CE",
                }
            ]
        }
        gen = QaGenerator()
        samples = gen.generate(chunk, n=1, llm=mock_llm)
        assert len(samples) == 1
        s = samples[0]
        assert s.dataset_type == DatasetType.qa
        assert s.chunk_id == chunk.id
        assert s.payload["question"] == "Where was the Prophet born?"
        assert s.payload["answer"] == "The Prophet was born in Makkah."
        assert s.supporting_quote == "born in Makkah in 570 CE"
        assert s.source_text == chunk.text

    def test_insufficient_returns_empty(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {"insufficient": True}
        gen = QaGenerator()
        samples = gen.generate(chunk, n=3, llm=mock_llm)
        assert samples == []

    def test_empty_supporting_quote_dropped(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "pairs": [
                {
                    "question": "Q?",
                    "answer": "A.",
                    "supporting_quote": "",  # empty → drop
                },
                {
                    "question": "Q2?",
                    "answer": "A2.",
                    "supporting_quote": "born in Makkah",  # valid
                },
            ]
        }
        gen = QaGenerator()
        samples = gen.generate(chunk, n=2, llm=mock_llm)
        assert len(samples) == 1
        assert samples[0].payload["question"] == "Q2?"

    def test_sample_dataset_type_is_qa(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "pairs": [{"question": "Q?", "answer": "A.", "supporting_quote": "Makkah"}]
        }
        gen = QaGenerator()
        samples = gen.generate(chunk, n=1, llm=mock_llm)
        assert samples[0].dataset_type == DatasetType.qa

    def test_sample_chunk_id_matches(self):
        chunk = make_chunk()
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "pairs": [{"question": "Q?", "answer": "A.", "supporting_quote": "Makkah"}]
        }
        samples = QaGenerator().generate(chunk, n=1, llm=mock_llm)
        assert samples[0].chunk_id == chunk.id

    def test_dataset_type_attribute(self):
        assert QaGenerator().dataset_type == "qa"


class TestGeneratorRegistry:
    def test_get_qa_generator(self):
        gen = get_generator("qa")
        assert gen.dataset_type == "qa"

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            get_generator("nonexistent_type_xyz")
