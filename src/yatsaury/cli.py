"""Yatsaury CLI — Turn source material into LLM training datasets."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="yatsaury",
    help="Turn source material into LLM training datasets.",
    no_args_is_help=True,
)


@app.command()
def generate(
    input: list[str] = typer.Option(
        ..., "-i", "--input", help="Input file paths or URLs (repeatable)"
    ),
    type: list[str] = typer.Option(
        ["qa"], "-t", "--type", help="Dataset type(s): qa, instruction, rag, summary"
    ),
    schema: list[str] = typer.Option(
        ["chatml"],
        "-s",
        "--schema",
        help="Record schema(s): chatml, qa, sharegpt, alpaca, completion, rag, raw",
    ),
    format: list[str] = typer.Option(
        ["jsonl"], "-f", "--format", help="Output format(s): jsonl, hf, csv"
    ),
    output: Path = typer.Option(Path("./output"), "-o", "--output"),
    chunk_size: int = typer.Option(512, "--chunk-size"),
    chunk_overlap: int = typer.Option(64, "--chunk-overlap"),
    per_chunk: int = typer.Option(3, "-n", "--per-chunk"),
    model: str = typer.Option("", "--model"),
    base_url: str = typer.Option("", "--base-url"),
    api_key: str = typer.Option("", "--api-key"),
    limit_chunks: int = typer.Option(0, "--limit-chunks", help="0 = no limit"),
) -> None:
    """Generate training samples from source documents."""
    from yatsaury.config import Settings
    from yatsaury.pipeline import Orchestrator, OrchestratorConfig

    settings = Settings()
    config = OrchestratorConfig(
        dataset_types=type,
        schema_names=schema,
        output_formats=format,
        output_dir=output,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        per_chunk=per_chunk,
        llm_base_url=base_url or settings.base_url,
        llm_api_key=api_key or settings.api_key.get_secret_value(),
        llm_model=model or settings.model,
    )
    orch = Orchestrator(config)
    records = orch.run(input)
    count = len(records)
    noun = "record" if count == 1 else "records"
    typer.echo(f"Generated {count} {noun}.")


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
