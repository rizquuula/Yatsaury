"""Tests for quote_check and normalize_ws in yatsaury.quality.verify."""
from __future__ import annotations

from yatsaury.models import Citation, DatasetType, Sample
from yatsaury.quality.verify import normalize_ws, quote_check


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


def test_exact_substring() -> None:
    """supporting_quote is an exact substring of source_text."""
    sample = make_sample("The quick brown fox jumps.", "quick brown fox")
    assert quote_check(sample) is True


def test_whitespace_fuzzy_double_space() -> None:
    """Double spaces in quote are normalized; still found in source."""
    sample = make_sample("some text here", "some  text")
    assert quote_check(sample) is True


def test_whitespace_fuzzy_newline() -> None:
    """Newline in quote normalized to space; still found in source."""
    sample = make_sample("line1 line2 more", "line1\nline2")
    assert quote_check(sample) is True


def test_not_in_source() -> None:
    """Quote not present in source text at all."""
    sample = make_sample("The quick brown fox.", "lazy dog")
    assert quote_check(sample) is False


def test_invented_sentence() -> None:
    """Invented sentence not in source."""
    sample = make_sample("The sky is blue.", "The grass is always greener.")
    assert quote_check(sample) is False


def test_empty_quote() -> None:
    """Empty supporting_quote returns False."""
    sample = make_sample("Some source text.", "")
    assert quote_check(sample) is False


def test_full_source_as_quote() -> None:
    """supporting_quote equals source_text exactly."""
    text = "The entire source text is the quote."
    sample = make_sample(text, text)
    assert quote_check(sample) is True


def test_single_word_exists() -> None:
    """Single-word quote that exists in source."""
    sample = make_sample("Python is a great language.", "Python")
    assert quote_check(sample) is True


def test_normalize_ws_collapses_tabs() -> None:
    """normalize_ws collapses tabs and mixed whitespace to single spaces."""
    result = normalize_ws("hello\t\tworld\n  foo")
    assert result == "hello world foo"
