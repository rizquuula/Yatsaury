"""Tests for JsonlExporter."""
from __future__ import annotations

import json
from pathlib import Path

from yatsaury.exporters.base import get_exporter, register_exporter
from yatsaury.exporters.jsonl import JsonlExporter


class TestJsonlExporter:
    def test_exports_three_records(self, tmp_path: Path):
        records = [
            {"q": "What is Islam?", "a": "A monotheistic religion."},
            {"q": "Who was the Prophet?", "a": "Muhammad (peace be upon him)."},
            {"q": "What is the Quran?", "a": "The holy book of Islam."},
        ]
        out = tmp_path / "out.jsonl"
        JsonlExporter().export(records, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

    def test_each_line_valid_json(self, tmp_path: Path):
        records = [{"key": "value"}, {"num": 42}]
        out = tmp_path / "out.jsonl"
        JsonlExporter().export(records, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_round_trips_original_dicts(self, tmp_path: Path):
        records = [{"a": 1, "b": "two"}, {"c": [1, 2, 3]}]
        out = tmp_path / "out.jsonl"
        JsonlExporter().export(records, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        for original, line in zip(records, lines):
            assert json.loads(line) == original

    def test_creates_parent_directories(self, tmp_path: Path):
        out = tmp_path / "nested" / "deep" / "out.jsonl"
        JsonlExporter().export([{"x": 1}], out)
        assert out.exists()

    def test_unicode_preserved(self, tmp_path: Path):
        records = [
            {"arabic": "بسم الله الرحمن الرحيم"},
            {"malay": "Dalam nama Allah Yang Maha Pemurah lagi Maha Penyayang"},
        ]
        out = tmp_path / "unicode.jsonl"
        JsonlExporter().export(records, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["arabic"] == "بسم الله الرحمن الرحيم"
        expected_malay = "Dalam nama Allah Yang Maha Pemurah lagi Maha Penyayang"
        assert json.loads(lines[1])["malay"] == expected_malay
        # ensure_ascii=False: no escaped unicode
        assert "\\u" not in lines[0]

    def test_empty_iterable(self, tmp_path: Path):
        out = tmp_path / "empty.jsonl"
        JsonlExporter().export([], out)
        assert out.exists()
        assert out.read_text() == ""


class TestExporterRegistry:
    def test_get_jsonl_exporter(self):
        exporter_cls = get_exporter("jsonl")
        assert exporter_cls is JsonlExporter

    def test_register_custom_exporter(self):
        class FakeExporter:
            pass
        register_exporter("fake", FakeExporter)
        assert get_exporter("fake") is FakeExporter
