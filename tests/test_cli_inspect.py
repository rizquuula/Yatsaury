"""Tests for the inspect CLI command — TDD Phase 2."""
from __future__ import annotations

from typer.testing import CliRunner

from yatsaury.cli import app

runner = CliRunner()


class TestInspectCommand:
    def test_inspect_exits_zero(self) -> None:
        result = runner.invoke(app, ["inspect", "-i", "examples/sirah_sample.txt"])
        assert result.exit_code == 0, result.output

    def test_inspect_output_contains_chunks(self) -> None:
        result = runner.invoke(app, ["inspect", "-i", "examples/sirah_sample.txt"])
        assert "Chunks" in result.output

    def test_inspect_output_contains_characters(self) -> None:
        result = runner.invoke(app, ["inspect", "-i", "examples/sirah_sample.txt"])
        assert "Characters" in result.output

    def test_inspect_output_contains_total_tokens(self) -> None:
        result = runner.invoke(app, ["inspect", "-i", "examples/sirah_sample.txt"])
        assert "Total tokens" in result.output

    def test_inspect_with_custom_chunk_size(self) -> None:
        result = runner.invoke(
            app,
            ["inspect", "-i", "examples/sirah_sample.txt", "--chunk-size", "100"],
        )
        assert result.exit_code == 0, result.output

    def test_inspect_missing_input_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["inspect"])
        assert result.exit_code != 0

    def test_inspect_shows_source_uri(self) -> None:
        result = runner.invoke(app, ["inspect", "-i", "examples/sirah_sample.txt"])
        assert "examples/sirah_sample.txt" in result.output or "Source" in result.output
