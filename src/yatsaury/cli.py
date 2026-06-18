"""Yatsaury CLI — Turn source material into LLM training datasets."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="yatsaury",
    help="Turn source material into LLM training datasets.",
    no_args_is_help=True,
)


@app.command()
def generate(
    ctx: typer.Context,
) -> None:
    """Generate training samples from source documents."""
    typer.echo("generate: not yet implemented")


@app.command()
def inspect(
    ctx: typer.Context,
) -> None:
    """Inspect a dataset or generated samples."""
    typer.echo("inspect: not yet implemented")


@app.command()
def verify(
    ctx: typer.Context,
) -> None:
    """Verify grounding scores for generated samples."""
    typer.echo("verify: not yet implemented")


@app.command()
def export(
    ctx: typer.Context,
) -> None:
    """Export a dataset to a target format."""
    typer.echo("export: not yet implemented")


@app.command()
def schemas(
    ctx: typer.Context,
) -> None:
    """Show JSON schemas for all dataset types."""
    typer.echo("schemas: not yet implemented")


@app.command()
def web(
    ctx: typer.Context,
) -> None:
    """Launch the Yatsaury web UI."""
    typer.echo("web: not yet implemented")


@app.command(name="config")
def config_cmd(
    ctx: typer.Context,
) -> None:
    """Show or validate current configuration."""
    typer.echo("config: not yet implemented")
