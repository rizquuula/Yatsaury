"""Tests for `yatsaury config` CLI command."""
from __future__ import annotations

from typer.testing import CliRunner

from yatsaury.cli import app


def test_config_show_exits_0():
    runner = CliRunner()
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0


def test_config_show_masks_api_key(monkeypatch):
    monkeypatch.setenv("YATSAURY_API_KEY", "sk-supersecret")
    runner = CliRunner()
    result = runner.invoke(app, ["config"])
    assert "sk-supersecret" not in result.output
    assert "***" in result.output


def test_config_show_no_key_shows_not_set():
    runner = CliRunner()
    result = runner.invoke(app, ["config"], env={"YATSAURY_API_KEY": "", "OPENAI_API_KEY": ""})
    assert "not set" in result.output


def test_config_show_displays_model():
    runner = CliRunner()
    result = runner.invoke(app, ["config"])
    assert "model" in result.output.lower()


def test_config_show_displays_workspace():
    runner = CliRunner()
    result = runner.invoke(app, ["config"])
    assert "workspace" in result.output.lower()
