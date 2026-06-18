"""Tests for the `web` CLI command."""
from __future__ import annotations

from unittest.mock import patch


def test_web_command_sets_up_store_and_app(tmp_path):
    from typer.testing import CliRunner

    from yatsaury.cli import app

    runner = CliRunner()
    with (
        patch("nicegui.ui.run") as mock_run,
        patch("yatsaury.web.app.create_app") as mock_create,
    ):
        result = runner.invoke(
            app,
            [
                "web",
                "--no-open",
                "--workspace",
                str(tmp_path),
                "--host",
                "127.0.0.1",
                "--port",
                "9999",
            ],
        )
    assert result.exit_code == 0, result.output
    assert mock_run.called
    assert mock_create.called


def test_web_command_prints_url(tmp_path):
    from typer.testing import CliRunner

    from yatsaury.cli import app

    runner = CliRunner()
    with patch("nicegui.ui.run"), patch("yatsaury.web.app.create_app"):
        result = runner.invoke(
            app,
            [
                "web",
                "--no-open",
                "--workspace",
                str(tmp_path),
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
            ],
        )
    assert "127.0.0.1" in result.output
    assert "8080" in result.output
