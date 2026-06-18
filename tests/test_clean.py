"""Tests for clean_text — TDD Phase 2."""
from __future__ import annotations

from yatsaury.processing.clean import clean_text


class TestCleanText:
    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_dehyphenation_basic(self) -> None:
        assert clean_text("some-\nword") == "someword"

    def test_dehyphenation_preserves_non_hyphen_breaks(self) -> None:
        result = clean_text("line one\nline two")
        assert "line one" in result
        assert "line two" in result

    def test_whitespace_collapse_within_line(self) -> None:
        assert clean_text("hello   world") == "hello world"

    def test_tab_collapse(self) -> None:
        assert clean_text("hello\t\tworld") == "hello world"

    def test_crlf_normalized(self) -> None:
        assert clean_text("a\r\nb") == "a\nb"

    def test_multiple_blank_lines_collapsed(self) -> None:
        # More than 2 consecutive blank lines → max 1 blank line between paragraphs
        result = clean_text("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_three_blank_lines_collapsed(self) -> None:
        result = clean_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_two_blank_lines_preserved(self) -> None:
        # Exactly 2 consecutive newlines (1 blank line) should be preserved
        result = clean_text("a\n\nb")
        assert result == "a\n\nb"

    def test_leading_trailing_whitespace_per_line(self) -> None:
        result = clean_text("  hello  \n  world  ")
        assert result == "hello\nworld"

    def test_whole_text_strip(self) -> None:
        assert clean_text("\n\nhello\n\n") == "hello"

    def test_mixed_normalization(self) -> None:
        text = "  First para-\ngraph.  \r\n\n\n\n  Second   paragraph.  \n"
        result = clean_text(text)
        # De-hyphenated
        assert "paragraph." in result
        # No leading/trailing whitespace on result
        assert result == result.strip()
        # No CRLF
        assert "\r" not in result
        # At most one blank line between paragraphs
        assert "\n\n\n" not in result

    def test_whitespace_only_string(self) -> None:
        assert clean_text("   \n\n   \n") == ""

    def test_single_line_no_change(self) -> None:
        assert clean_text("hello world") == "hello world"
