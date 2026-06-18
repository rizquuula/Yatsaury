"""JSONL exporter — one JSON object per line."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from yatsaury.exporters.base import register_exporter


class JsonlExporter:
    """Export records as newline-delimited JSON (JSONL)."""

    def export(self, records: Iterable[dict], out_path: Path) -> None:
        """Write each record as a JSON line to out_path."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Register at import time
register_exporter("jsonl", JsonlExporter)
