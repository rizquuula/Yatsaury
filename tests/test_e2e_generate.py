"""End-to-end test against a local Ollama instance.

Run with: pytest -m e2e
Excluded from default run by addopts = '-m "not e2e"' in pyproject.toml.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from yatsaury.cli import app


@pytest.mark.e2e
def test_e2e_generate_ollama(tmp_path):
    """Real run against local Ollama. Requires: ollama serve + llama3.1 pulled."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "generate",
            "-i", "examples/sirah_sample.txt",
            "-t", "qa",
            "-s", "chatml",
            "-f", "jsonl",
            "-o", str(tmp_path / "out"),
            "--base-url", "http://localhost:11434/v1",
            "--model", "llama3.1",
            "--limit-chunks", "2",
        ],
    )
    assert result.exit_code == 0
    out_file = tmp_path / "out" / "chatml.jsonl"
    assert out_file.exists()
