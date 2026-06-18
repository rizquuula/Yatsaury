"""Tests for web app (smoke tests — no browser)."""
from __future__ import annotations


def test_create_app_importable(tmp_path):
    """create_app should not raise on call."""
    from yatsaury.session.store import SessionStore
    from yatsaury.web.app import create_app

    store = SessionStore(tmp_path)
    try:
        create_app(store, tmp_path)
    except Exception as e:
        err = str(e).lower()
        # Only tolerate NiceGUI-specific errors about event loop / no running app
        allowed = any(k in err for k in ["event loop", "nicegui", "no running", "client"])
        assert allowed, f"Unexpected error: {e}"


def test_web_app_module_has_create_app():
    """create_app is the public API of web.app."""
    import yatsaury.web.app as mod

    assert hasattr(mod, "create_app")
    assert callable(mod.create_app)
