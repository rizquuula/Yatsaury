"""Background job runner for the web UI."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from yatsaury.pipeline import Orchestrator, OrchestratorConfig
from yatsaury.session.models import SessionStatus
from yatsaury.session.store import SessionStore

logger = logging.getLogger(__name__)


async def run_generation_job(
    session_id: str,
    store: SessionStore,
    config: OrchestratorConfig,
    source_uris: list[str],
) -> None:
    """Run the Orchestrator in a thread pool (non-blocking) and update session status."""
    store.update(session_id, status=SessionStatus.running)

    cb = _progress_cb(session_id, store)
    loop = asyncio.get_event_loop()

    try:
        orch = Orchestrator(config)
        records = await loop.run_in_executor(
            None, lambda: orch.run(source_uris, progress_cb=cb)
        )
        store.update(
            session_id,
            status=SessionStatus.done,
            progress=1.0,
            counts={"kept": len(records)},
        )
    except Exception as exc:
        logger.exception("Job %s failed", session_id)
        store.update(session_id, status=SessionStatus.error, error=str(exc))


def _progress_cb(session_id: str, store: SessionStore) -> Callable[[str, float], None]:
    def cb(message: str, progress: float) -> None:
        try:
            store.update(session_id, progress=progress)
        except Exception:
            pass

    return cb
