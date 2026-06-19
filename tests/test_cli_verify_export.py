"""Tests for verify and export CLI commands."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from yatsaury.cli import app
from yatsaury.models import Citation, DatasetType, Sample

runner = CliRunner()


def make_sample(
    sid: str = "s1",
    supporting_quote: str = "The quick brown fox.",
    source_text: str = "The quick brown fox.",
) -> Sample:
    return Sample(
        id=sid,
        chunk_id="c1",
        dataset_type=DatasetType.qa,
        payload={"question": "Q?", "answer": "A."},
        source_text=source_text,
        supporting_quote=supporting_quote,
        source_citation=Citation(title="Book", source_uri="uri://x"),
        quality_score=0.9,
        verified=True,
    )


class TestVerifyCommand:
    def test_verify_no_judge_keeps_passing(self, tmp_path: Path) -> None:
        """verify --no-judge keeps only samples whose supporting_quote is in source_text."""
        s1 = make_sample(
            "s1",
            supporting_quote="The quick brown fox.",
            source_text="The quick brown fox.",
        )
        s2 = make_sample(
            "s2",
            supporting_quote="lazy dog",
            source_text="The lazy dog sleeps soundly.",
        )
        s3 = make_sample(
            "s3",
            supporting_quote="THIS QUOTE DOES NOT EXIST",
            source_text="Something entirely different.",
        )

        jsonl_path = tmp_path / "samples.jsonl"
        jsonl_path.write_text(
            "\n".join([s1.model_dump_json(), s2.model_dump_json(), s3.model_dump_json()])
            + "\n"
        )

        with patch("yatsaury.llm.client.LLMClient") as mock_llm_cls:
            mock_llm_cls.return_value = MagicMock()
            result = runner.invoke(
                app,
                [
                    "verify",
                    "-i", str(jsonl_path),
                    "--no-judge",
                ],
            )

        assert result.exit_code == 0, f"Unexpected error: {result.output}"
        assert "Kept 2/3" in result.output

    def test_verify_missing_input_exits_nonzero(self) -> None:
        """verify without -i should exit with non-zero code."""
        result = runner.invoke(app, ["verify"])
        assert result.exit_code != 0


class TestExportCommand:
    def test_export_jsonl(self, tmp_path: Path) -> None:
        """export -s chatml -f jsonl writes chatml.jsonl and reports count."""
        import yatsaury.exporters.jsonl  # noqa: F401 — triggers registration
        import yatsaury.schemas.chatml  # noqa: F401 — triggers registration

        s1 = make_sample("s1")
        jsonl_path = tmp_path / "samples.jsonl"
        jsonl_path.write_text(s1.model_dump_json() + "\n")

        out_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "export",
                "-i", str(jsonl_path),
                "-s", "chatml",
                "-f", "jsonl",
                "-o", str(out_dir),
            ],
        )

        assert result.exit_code == 0, f"Unexpected error: {result.output}"
        assert "Exported 1 records" in result.output
        assert (out_dir / "chatml.jsonl").exists()

    def test_export_missing_input_exits_nonzero(self) -> None:
        """export without -i should exit with non-zero code."""
        result = runner.invoke(app, ["export"])
        assert result.exit_code != 0
