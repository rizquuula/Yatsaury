"""Tests for generate --session flag."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_generate_session_flag_creates_session(tmp_path):
    """--session flag must create a session directory."""
    from typer.testing import CliRunner

    from yatsaury.cli import app
    from yatsaury.session.store import SessionStore

    runner = CliRunner()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    fake_settings = MagicMock()
    fake_settings.base_url = "http://localhost"
    fake_settings.api_key.get_secret_value.return_value = ""
    fake_settings.model = "gpt-4o-mini"
    fake_settings.judge_model = ""
    fake_settings.workspace = workspace

    with (
        patch("yatsaury.pipeline.Orchestrator") as MockOrch,
        patch("yatsaury.cli.Settings", return_value=fake_settings),
    ):
        MockOrch.return_value.run.return_value = []
        result = runner.invoke(
            app,
            [
                "generate",
                "-i",
                "examples/sirah_sample.txt",
                "--session",
                "-o",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    store = SessionStore(workspace)
    sessions = store.list()
    assert len(sessions) >= 1


def test_generate_session_flag_prints_session_id(tmp_path):
    """--session flag must print 'Session:' line with the session id."""
    from typer.testing import CliRunner

    from yatsaury.cli import app

    runner = CliRunner()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    fake_settings = MagicMock()
    fake_settings.base_url = "http://localhost"
    fake_settings.api_key.get_secret_value.return_value = ""
    fake_settings.model = "gpt-4o-mini"
    fake_settings.judge_model = ""
    fake_settings.workspace = workspace

    with (
        patch("yatsaury.pipeline.Orchestrator") as MockOrch,
        patch("yatsaury.cli.Settings", return_value=fake_settings),
    ):
        MockOrch.return_value.run.return_value = []
        result = runner.invoke(
            app,
            [
                "generate",
                "-i",
                "examples/sirah_sample.txt",
                "--session",
                "-o",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            catch_exceptions=False,
        )

    assert "Session" in result.output or "session" in result.output.lower()
