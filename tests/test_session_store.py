"""Tests for SessionStore."""
from __future__ import annotations

import time

import pytest

from yatsaury.session.models import SessionInput, SessionStatus
from yatsaury.session.store import SessionStore


def test_create_returns_session(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create("Sirah v1", [SessionInput(uri="file.pdf")], config={})
    assert session.id
    assert session.title == "Sirah v1"
    assert session.status == SessionStatus.queued


def test_create_writes_session_json(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    assert (tmp_path / "sessions" / s.id / "session.json").exists()


def test_create_makes_sources_and_outputs_dirs(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    assert (tmp_path / "sessions" / s.id / "sources").is_dir()
    assert (tmp_path / "sessions" / s.id / "outputs").is_dir()


def test_list_sorted_newest_first(tmp_path):
    store = SessionStore(tmp_path)
    store.create("Alpha", [], {})
    time.sleep(0.02)
    store.create("Beta", [], {})
    sessions = store.list()
    assert sessions[0].id >= sessions[1].id


def test_get_returns_session(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    fetched = store.get(s.id)
    assert fetched.id == s.id


def test_get_missing_raises_key_error(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(KeyError):
        store.get("nonexistent")


def test_update_status(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    updated = store.update(s.id, status=SessionStatus.running)
    assert updated.status == SessionStatus.running
    assert store.get(s.id).status == SessionStatus.running


def test_update_progress(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    store.update(s.id, progress=0.5)
    assert store.get(s.id).progress == 0.5


def test_status_transitions(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    store.update(s.id, status=SessionStatus.running)
    store.update(s.id, status=SessionStatus.done, progress=1.0)
    final = store.get(s.id)
    assert final.status == SessionStatus.done


def test_path_for(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("Test", [], {})
    p = store.path_for(s.id, "outputs", "chatml.jsonl")
    assert str(p).endswith("chatml.jsonl")
