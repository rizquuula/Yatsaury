"""CLI smoke tests for Yatsaury."""

from typer.testing import CliRunner

from yatsaury.cli import app

runner = CliRunner()

EXPECTED_SUBCOMMANDS = [
    "generate",
    "inspect",
    "verify",
    "export",
    "schemas",
    "web",
    "config",
]


class TestHelp:
    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_all_subcommands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in EXPECTED_SUBCOMMANDS:
            assert cmd in result.output, f"Subcommand '{cmd}' not found in --help output"

    def test_generate_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0

    def test_inspect_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0

    def test_verify_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["verify", "--help"])
        assert result.exit_code == 0

    def test_export_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0

    def test_schemas_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["schemas", "--help"])
        assert result.exit_code == 0

    def test_web_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["web", "--help"])
        assert result.exit_code == 0

    def test_config_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
