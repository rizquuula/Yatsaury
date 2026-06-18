"""Tests for Yatsaury data models."""

import pytest
from pydantic import ValidationError

from yatsaury.models import (
    Chunk,
    Citation,
    DatasetType,
    Document,
    Sample,
    SourceType,
)


class TestSourceType:
    def test_valid_values(self) -> None:
        assert SourceType.pdf == "pdf"
        assert SourceType.url == "url"
        assert SourceType.text == "text"


class TestDocument:
    def test_valid_construction(self) -> None:
        doc = Document(
            id="01J1XXXXXXXXXXX",
            source_uri="https://example.com/doc.pdf",
            source_type=SourceType.pdf,
            raw_text="Hello world",
        )
        assert doc.id == "01J1XXXXXXXXXXX"
        assert doc.source_type == SourceType.pdf
        assert doc.title == ""
        assert doc.metadata == {}

    def test_json_round_trip(self) -> None:
        doc = Document(
            id="abc123",
            source_uri="s3://bucket/file.txt",
            source_type=SourceType.text,
            raw_text="content",
            title="My Doc",
            metadata={"author": "Alice"},
        )
        json_str = doc.model_dump_json()
        restored = Document.model_validate_json(json_str)
        assert restored == doc

    def test_invalid_source_type(self) -> None:
        with pytest.raises(ValidationError):
            Document(
                id="x",
                source_uri="x",
                source_type="invalid_type",  # type: ignore[arg-type]
                raw_text="x",
            )


class TestChunk:
    def test_valid_construction(self) -> None:
        chunk = Chunk(
            id="c1",
            doc_id="d1",
            text="some text",
            token_count=10,
            char_span=(0, 9),
            ordinal=0,
        )
        assert chunk.page is None
        assert chunk.ordinal == 0

    def test_json_round_trip(self) -> None:
        chunk = Chunk(
            id="c2",
            doc_id="d2",
            text="text",
            token_count=5,
            char_span=(0, 4),
            page=3,
            ordinal=1,
        )
        restored = Chunk.model_validate_json(chunk.model_dump_json())
        assert restored == chunk

    def test_negative_token_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                id="c3",
                doc_id="d3",
                text="x",
                token_count=-1,
                char_span=(0, 1),
                ordinal=0,
            )


class TestCitation:
    def test_valid_construction(self) -> None:
        cit = Citation(title="Source", source_uri="https://example.com")
        assert cit.page is None
        assert cit.char_span is None

    def test_json_round_trip(self) -> None:
        cit = Citation(
            title="Book",
            page=42,
            char_span=(100, 200),
            source_uri="file:///book.pdf",
        )
        restored = Citation.model_validate_json(cit.model_dump_json())
        assert restored == cit


class TestDatasetType:
    def test_valid_values(self) -> None:
        assert DatasetType.qa == "qa"
        assert DatasetType.instruction == "instruction"
        assert DatasetType.rag == "rag"
        assert DatasetType.summary == "summary"


class TestSample:
    def _make_citation(self) -> Citation:
        return Citation(title="Ref", source_uri="https://example.com")

    def test_valid_construction(self) -> None:
        sample = Sample(
            id="s1",
            chunk_id="c1",
            dataset_type=DatasetType.qa,
            payload={"question": "What?", "answer": "This."},
            source_text="original text",
            supporting_quote="This.",
            source_citation=self._make_citation(),
        )
        assert sample.grounding_score == 0.0
        assert sample.verified is False
        assert sample.dedup_hash == ""
        assert sample.fact_id is None
        assert sample.lang == "auto"

    def test_json_round_trip(self) -> None:
        sample = Sample(
            id="s2",
            chunk_id="c2",
            dataset_type=DatasetType.rag,
            payload={"context": "ctx", "query": "q"},
            source_text="src",
            supporting_quote="ctx",
            source_citation=self._make_citation(),
            grounding_score=0.85,
            verified=True,
            dedup_hash="abc123",
            fact_id="f1",
            lang="en",
        )
        restored = Sample.model_validate_json(sample.model_dump_json())
        assert restored == sample

    def test_invalid_dataset_type(self) -> None:
        with pytest.raises(ValidationError):
            Sample(
                id="s3",
                chunk_id="c3",
                dataset_type="bad_type",  # type: ignore[arg-type]
                payload={},
                source_text="x",
                supporting_quote="x",
                source_citation=self._make_citation(),
            )
