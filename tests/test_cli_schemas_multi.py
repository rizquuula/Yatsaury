"""Tests for `yatsaury schemas` command and multi-flag generate."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from yatsaury.cli import app

runner = CliRunner()


class TestSchemasCommand:
    def test_schemas_exits_zero(self):
        """schemas command exits 0."""
        result = runner.invoke(app, ["schemas"])
        assert result.exit_code == 0, f"Unexpected error: {result.output}"

    def test_schemas_lists_chatml(self):
        result = runner.invoke(app, ["schemas"])
        assert "chatml" in result.output

    def test_schemas_lists_alpaca(self):
        result = runner.invoke(app, ["schemas"])
        assert "alpaca" in result.output

    def test_schemas_lists_sharegpt(self):
        result = runner.invoke(app, ["schemas"])
        assert "sharegpt" in result.output

    def test_schemas_lists_completion(self):
        result = runner.invoke(app, ["schemas"])
        assert "completion" in result.output

    def test_schemas_lists_rag(self):
        result = runner.invoke(app, ["schemas"])
        assert "rag" in result.output

    def test_schemas_lists_raw(self):
        result = runner.invoke(app, ["schemas"])
        assert "raw" in result.output

    def test_schemas_lists_qa(self):
        result = runner.invoke(app, ["schemas"])
        assert "qa" in result.output

    def test_schemas_output_contains_supports(self):
        """Each line should contain 'supports:'."""
        result = runner.invoke(app, ["schemas"])
        assert "supports:" in result.output


class TestGenerateMultiFlags:
    def test_generate_multi_type_and_schema_exits_zero(self, tmp_path: Path):
        """generate with multiple -t and -s flags exits 0."""
        with patch("yatsaury.pipeline.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]
            mock_orch_cls.return_value = mock_orch

            result = runner.invoke(
                app,
                [
                    "generate",
                    "-i", "examples/sirah_sample.txt",
                    "-t", "qa",
                    "-t", "instruction",
                    "-s", "chatml",
                    "-s", "alpaca",
                    "-f", "jsonl",
                    "-o", str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, f"Unexpected error: {result.output}"
        assert "4 records" in result.output

    def test_generate_incompatible_pair_exits_zero(self, tmp_path: Path):
        """rag type + chatml schema: pipeline skips → 0 records, no crash."""
        with patch("yatsaury.pipeline.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = []
            mock_orch_cls.return_value = mock_orch

            result = runner.invoke(
                app,
                [
                    "generate",
                    "-i", "examples/sirah_sample.txt",
                    "-t", "rag",
                    "-s", "chatml",
                    "-f", "jsonl",
                    "-o", str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, f"Unexpected error: {result.output}"
        assert "0 records" in result.output
