"""Tests for CsvReviewExporter."""
from __future__ import annotations

import csv
from pathlib import Path

from yatsaury.exporters.review_csv import CsvReviewExporter
from yatsaury.models import Citation, DatasetType, Sample


def make_sample(
    sid: str = "s1",
    question: str = "Q?",
    answer: str = "A.",
) -> Sample:
    return Sample(
        id=sid,
        chunk_id="c1",
        dataset_type=DatasetType.qa,
        payload={"question": question, "answer": answer},
        source_text="source",
        supporting_quote="source",
        source_citation=Citation(title="MyBook", page=42, source_uri="uri://x"),
        quality_score=0.85,
        verified=True,
    )


class TestCsvReviewExporterExport:
    def test_export_creates_file(self, tmp_path: Path):
        exporter = CsvReviewExporter()
        out = tmp_path / "review.csv"
        exporter.export_samples([make_sample("s1"), make_sample("s2")], out)
        assert out.exists()

    def test_export_header(self, tmp_path: Path):
        exporter = CsvReviewExporter()
        out = tmp_path / "review.csv"
        exporter.export_samples([make_sample("s1")], out)
        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == CsvReviewExporter.COLUMNS

    def test_export_approved_empty(self, tmp_path: Path):
        exporter = CsvReviewExporter()
        out = tmp_path / "review.csv"
        exporter.export_samples([make_sample("s1"), make_sample("s2")], out)
        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["approved"] == ""

    def test_export_row_count(self, tmp_path: Path):
        exporter = CsvReviewExporter()
        out = tmp_path / "review.csv"
        exporter.export_samples([make_sample("s1"), make_sample("s2")], out)
        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_parent_dir_created(self, tmp_path: Path):
        exporter = CsvReviewExporter()
        out = tmp_path / "nested" / "deep" / "review.csv"
        exporter.export_samples([make_sample("s1")], out)
        assert out.exists()

    def test_unicode_preserved(self, tmp_path: Path):
        arabic_question = "ما هو الإسلام؟"
        exporter = CsvReviewExporter()
        out = tmp_path / "unicode.csv"
        exporter.export_samples([make_sample("s1", question=arabic_question)], out)
        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["question"] == arabic_question


class TestCsvReviewExporterLoadApproved:
    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        fieldnames = CsvReviewExporter.COLUMNS
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                full_row = {col: row.get(col, "") for col in fieldnames}
                writer.writerow(full_row)

    def test_load_approved_yes(self, tmp_path: Path):
        path = tmp_path / "review.csv"
        self._write_csv(path, [{"id": "s1", "approved": "yes"}])
        results = CsvReviewExporter().load_approved(path)
        assert len(results) == 1
        assert results[0]["id"] == "s1"

    def test_load_approved_true(self, tmp_path: Path):
        path = tmp_path / "review.csv"
        self._write_csv(path, [{"id": "s1", "approved": "true"}])
        results = CsvReviewExporter().load_approved(path)
        assert len(results) == 1

    def test_load_approved_1(self, tmp_path: Path):
        path = tmp_path / "review.csv"
        self._write_csv(path, [{"id": "s1", "approved": "1"}])
        results = CsvReviewExporter().load_approved(path)
        assert len(results) == 1

    def test_load_approved_x(self, tmp_path: Path):
        path = tmp_path / "review.csv"
        self._write_csv(path, [{"id": "s1", "approved": "x"}])
        results = CsvReviewExporter().load_approved(path)
        assert len(results) == 1

    def test_load_approved_empty_ignored(self, tmp_path: Path):
        path = tmp_path / "review.csv"
        self._write_csv(
            path,
            [
                {"id": "s1", "approved": ""},
                {"id": "s2", "approved": "   "},
            ],
        )
        results = CsvReviewExporter().load_approved(path)
        assert len(results) == 0

    def test_load_approved_case_insensitive(self, tmp_path: Path):
        path = tmp_path / "review.csv"
        self._write_csv(
            path,
            [
                {"id": "s1", "approved": "YES"},
                {"id": "s2", "approved": "True"},
            ],
        )
        results = CsvReviewExporter().load_approved(path)
        assert len(results) == 2

    def test_round_trip(self, tmp_path: Path):
        exporter = CsvReviewExporter()
        out = tmp_path / "review.csv"
        samples = [make_sample(f"s{i}") for i in range(1, 4)]
        exporter.export_samples(samples, out)

        # Read the exported CSV, mark two rows as approved, rewrite
        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        rows[0]["approved"] = "yes"
        rows[1]["approved"] = "yes"
        rows[2]["approved"] = ""

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        approved = exporter.load_approved(out)
        assert len(approved) == 2


class TestCsvReviewExporterProtocol:
    def test_export_protocol_writes_dicts(self, tmp_path: Path):
        exporter = CsvReviewExporter()
        out = tmp_path / "out.csv"
        records = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
        exporter.export(records, out)
        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_export_protocol_empty_writes_empty_file(self, tmp_path: Path):
        exporter = CsvReviewExporter()
        out = tmp_path / "empty.csv"
        exporter.export([], out)
        assert out.exists()
        assert out.read_text() == ""


class TestCsvExporterRegistry:
    def test_csv_registered(self):
        from yatsaury.exporters.base import get_exporter

        exporter_cls = get_exporter("csv")
        assert exporter_cls is CsvReviewExporter
