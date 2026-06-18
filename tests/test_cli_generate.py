"""Tests for the generate CLI command."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from yatsaury.cli import app

runner = CliRunner()


class TestGenerateCommand:
    def test_generate_basic_exits_zero(self, tmp_path: Path):
        """generate with mocked Orchestrator.run exits 0."""
        with patch("yatsaury.pipeline.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = [{"messages": []}, {"messages": []}]
            mock_orch_cls.return_value = mock_orch

            result = runner.invoke(
                app,
                [
                    "generate",
                    "-i", "examples/sirah_sample.txt",
                    "-t", "qa",
                    "-s", "chatml",
                    "-f", "jsonl",
                    "-o", str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, f"Unexpected error: {result.output}"
        assert "2 records" in result.output

    def test_generate_outputs_record_count(self, tmp_path: Path):
        """stdout must contain the record count."""
        with patch("yatsaury.pipeline.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = [{"q": "Q", "a": "A"}]
            mock_orch_cls.return_value = mock_orch

            result = runner.invoke(
                app,
                [
                    "generate",
                    "-i", "examples/sirah_sample.txt",
                    "-s", "qa",
                    "-o", str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0, f"Unexpected error: {result.output}"
        assert "1 record" in result.output

    def test_generate_with_schema_qa(self, tmp_path: Path):
        """generate -s qa exits 0."""
        with patch("yatsaury.pipeline.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = []
            mock_orch_cls.return_value = mock_orch

            result = runner.invoke(
                app,
                [
                    "generate",
                    "-i", "examples/sirah_sample.txt",
                    "-s", "qa",
                    "-o", str(tmp_path / "out"),
                ],
            )
        assert result.exit_code == 0
