"""Tests for --dry-run mode."""
from __future__ import annotations


def test_dry_run_makes_no_llm_calls(tmp_path):
    from unittest.mock import patch

    from yatsaury.pipeline import Orchestrator, OrchestratorConfig

    config = OrchestratorConfig(dry_run=True, output_dir=tmp_path, llm_api_key="test")

    with patch("yatsaury.pipeline.LLMClient") as mock_llm_cls:
        orch = Orchestrator(config)
        records = orch.run(["hello world from a test"])

    mock_llm_cls.return_value.complete_json.assert_not_called()
    assert records == []


def test_dry_run_returns_empty_records(tmp_path):
    from yatsaury.pipeline import Orchestrator, OrchestratorConfig
    config = OrchestratorConfig(dry_run=True, output_dir=tmp_path, llm_api_key="test")
    orch = Orchestrator(config)
    records = orch.run(["hello world"])
    assert records == []


def test_dry_run_cli_flag(tmp_path):
    from unittest.mock import patch

    from typer.testing import CliRunner

    from yatsaury.cli import app

    runner = CliRunner()
    with patch("yatsaury.pipeline.Orchestrator") as MockOrch:
        MockOrch.return_value.run.return_value = []
        result = runner.invoke(app, [
            "generate", "-i", "examples/sirah_sample.txt",
            "--dry-run", "-o", str(tmp_path)
        ])
    assert result.exit_code == 0
