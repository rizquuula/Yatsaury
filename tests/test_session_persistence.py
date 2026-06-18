"""Tests for session persistence (re-export without LLM)."""
from __future__ import annotations

import json

from yatsaury.session.store import SessionStore


def test_session_keeps_samples_jsonl(tmp_path):
    """After a session is 'done', samples.jsonl must exist."""
    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    samples_path = store.path_for(s.id, "samples.jsonl")
    from yatsaury.models import Citation, DatasetType, Sample

    sample = Sample(
        id="s1",
        chunk_id="c1",
        dataset_type=DatasetType.qa,
        payload={"question": "Q?", "answer": "A."},
        source_text="A.",
        supporting_quote="A.",
        source_citation=Citation(title="Test", source_uri="test.pdf"),
    )
    samples_path.write_text(sample.model_dump_json() + "\n")
    assert samples_path.exists()


def test_reexport_uses_samples_jsonl_without_llm(tmp_path):
    """Re-exporting a finished session uses saved samples, calls no LLM."""
    from yatsaury.exporters.jsonl import JsonlExporter
    from yatsaury.models import Citation, DatasetType, Sample
    from yatsaury.schemas.qa import QaSchema

    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    samples_path = store.path_for(s.id, "samples.jsonl")

    sample = Sample(
        id="s1",
        chunk_id="c1",
        dataset_type=DatasetType.qa,
        payload={"question": "Q?", "answer": "A."},
        source_text="A.",
        supporting_quote="A.",
        source_citation=Citation(title="Test", source_uri="test.pdf"),
    )
    samples_path.write_text(sample.model_dump_json() + "\n")

    samples = [
        Sample.model_validate_json(line)
        for line in samples_path.read_text().splitlines()
        if line.strip()
    ]
    adapter = QaSchema()
    records = [adapter.render(s) for s in samples if adapter.supports(s.dataset_type.value)]
    out = store.path_for(s.id, "outputs", "qa.jsonl")
    JsonlExporter().export(records, out)

    assert out.exists()
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["question"] == "Q?"
