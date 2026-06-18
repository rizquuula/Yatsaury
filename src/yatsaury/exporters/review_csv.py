"""CSV exporter for human review of grounding quality."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from yatsaury.exporters.base import register_exporter
from yatsaury.models import Sample

_TRUTHY_VALUES = {"yes", "1", "true", "x", "✓", "y"}


class CsvReviewExporter:
    """Export samples to a CSV for human review with an 'approved' column."""

    COLUMNS = [
        "id",
        "dataset_type",
        "question",
        "answer",
        "supporting_quote",
        "source_title",
        "source_page",
        "grounding_score",
        "verified",
        "approved",
    ]

    def export_samples(self, samples: list[Sample], out_path: Path) -> None:
        """Write samples to CSV. 'approved' column is empty for reviewer to fill."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            for sample in samples:
                writer.writerow(
                    {
                        "id": sample.id,
                        "dataset_type": sample.dataset_type.value,
                        "question": sample.payload.get("question", ""),
                        "answer": sample.payload.get("answer", ""),
                        "supporting_quote": sample.supporting_quote,
                        "source_title": sample.source_citation.title,
                        "source_page": sample.source_citation.page or "",
                        "grounding_score": sample.grounding_score,
                        "verified": sample.verified,
                        "approved": "",
                    }
                )

    def load_approved(self, csv_path: Path) -> list[dict]:
        """Read the reviewed CSV and return rows where 'approved' column is truthy.

        Truthy values (case-insensitive, stripped): yes, 1, true, x, ✓, y
        Returns list of raw row dicts.
        """
        results = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                approved_val = row.get("approved", "").strip().lower()
                if approved_val in _TRUTHY_VALUES:
                    results.append(dict(row))
        return results

    def export(self, records: Iterable[dict], out_path: Path) -> None:
        """Implement Exporter protocol — write raw dicts to CSV."""
        records = list(records)
        if not records:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("")
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


# Register at import time
register_exporter("csv", CsvReviewExporter)
