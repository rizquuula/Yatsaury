"""Tests for background job runner."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from yatsaury.session.models import SessionInput, SessionStatus
from yatsaury.session.store import SessionStore


def test_run_generation_job_updates_status_to_running(tmp_path):
    from yatsaury.pipeline import OrchestratorConfig
    from yatsaury.web.jobs import run_generation_job

    store = SessionStore(tmp_path)
    s = store.create("Test", [SessionInput(uri="hello")], {})
    config = OrchestratorConfig(output_dir=tmp_path / "out")

    statuses = []
    original_update = store.update

    def tracking_update(sid, **fields):
        if "status" in fields:
            statuses.append(fields["status"])
        return original_update(sid, **fields)

    store.update = tracking_update

    with patch("yatsaury.web.jobs.Orchestrator") as MockOrch:
        MockOrch.return_value.run.return_value = []
        asyncio.get_event_loop().run_until_complete(
            run_generation_job(s.id, store, config, ["hello"])
        )

    assert SessionStatus.running in statuses
    assert SessionStatus.done in statuses


def test_run_generation_job_sets_done_on_success(tmp_path):
    from yatsaury.pipeline import OrchestratorConfig
    from yatsaury.web.jobs import run_generation_job

    store = SessionStore(tmp_path)
    s = store.create("Test", [SessionInput(uri="hello")], {})
    config = OrchestratorConfig(output_dir=tmp_path / "out")

    with patch("yatsaury.web.jobs.Orchestrator") as MockOrch:
        MockOrch.return_value.run.return_value = [{"q": "Q"}]
        asyncio.get_event_loop().run_until_complete(
            run_generation_job(s.id, store, config, ["hello"])
        )

    final = store.get(s.id)
    assert final.status == SessionStatus.done
    assert final.counts.get("kept", 0) == 1


def test_run_generation_job_sets_error_on_failure(tmp_path):
    from yatsaury.pipeline import OrchestratorConfig
    from yatsaury.web.jobs import run_generation_job

    store = SessionStore(tmp_path)
    s = store.create("Test", [SessionInput(uri="hello")], {})
    config = OrchestratorConfig(output_dir=tmp_path / "out")

    with patch("yatsaury.web.jobs.Orchestrator") as MockOrch:
        MockOrch.return_value.run.side_effect = RuntimeError("boom")
        asyncio.get_event_loop().run_until_complete(
            run_generation_job(s.id, store, config, ["hello"])
        )

    final = store.get(s.id)
    assert final.status == SessionStatus.error
    assert "boom" in (final.error or "")
