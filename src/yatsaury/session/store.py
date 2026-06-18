from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from yatsaury.session.models import Session, SessionInput


class SessionStore:
    """Manages session directories under workspace/sessions/."""

    def __init__(self, workspace: Path) -> None:
        self._root = workspace / "sessions"
        self._root.mkdir(parents=True, exist_ok=True)

    def create(self, title: str, inputs: list[SessionInput], config: dict) -> Session:
        """Create a new session directory and session.json."""
        now = datetime.now(UTC)
        slug = title.lower().replace(" ", "-")[:20]
        session_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{slug}"
        session = Session(
            id=session_id,
            title=title,
            created_at=now.isoformat(),
            inputs=inputs,
            config=config,
        )
        session_dir = self._root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "sources").mkdir(exist_ok=True)
        (session_dir / "outputs").mkdir(exist_ok=True)
        self._write(session)
        return session

    def list(self) -> list[Session]:
        """Return all sessions sorted newest-first."""
        sessions = []
        for d in self._root.iterdir():
            if d.is_dir():
                json_path = d / "session.json"
                if json_path.exists():
                    sessions.append(Session.model_validate_json(json_path.read_text()))
        return sorted(sessions, key=lambda s: s.id, reverse=True)

    def get(self, session_id: str) -> Session:
        path = self._root / session_id / "session.json"
        if not path.exists():
            raise KeyError(f"Session not found: {session_id}")
        return Session.model_validate_json(path.read_text())

    def update(self, session_id: str, **fields) -> Session:
        """Update fields on an existing session and persist."""
        session = self.get(session_id)
        updated = session.model_copy(update=fields)
        self._write(updated)
        return updated

    def path_for(self, session_id: str, *parts: str) -> Path:
        """Return path under the session directory."""
        return self._root / session_id / Path(*parts)

    def _write(self, session: Session) -> None:
        path = self._root / session.id / "session.json"
        path.write_text(session.model_dump_json(indent=2))
