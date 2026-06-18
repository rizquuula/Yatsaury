"""NiceGUI web application for Yatsaury."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from nicegui import ui

from yatsaury.session.models import SessionStatus
from yatsaury.session.store import SessionStore

logger = logging.getLogger(__name__)


def create_app(store: SessionStore, workspace: Path) -> None:
    """Register the NiceGUI page routes."""

    @ui.page("/")
    async def index() -> None:
        with ui.column().classes("w-full max-w-2xl mx-auto p-4 gap-4"):
            ui.label("Yatsaury").classes("text-2xl font-bold")

            # --- Process form ---
            with ui.card().classes("w-full"):
                ui.label("New Dataset").classes("text-lg font-semibold")
                title_input = ui.input("Title", placeholder="e.g. Sirah v1").classes("w-full")
                source_input = ui.textarea(
                    "Source (URL or text)", placeholder="Paste URL or raw text"
                ).classes("w-full")

                type_select = ui.select(
                    ["qa", "instruction", "rag", "summary"],
                    multiple=True,
                    value=["qa"],
                    label="Dataset type",
                ).classes("w-full")
                schema_select = ui.select(
                    ["chatml", "sharegpt", "alpaca", "qa", "completion", "rag", "raw"],
                    multiple=True,
                    value=["chatml"],
                    label="Schema",
                ).classes("w-full")
                format_select = ui.select(
                    ["jsonl", "hf", "csv"],
                    multiple=True,
                    value=["jsonl"],
                    label="Format",
                ).classes("w-full")

                status_label = ui.label("").classes("text-sm text-gray-500")

                async def on_process() -> None:
                    from yatsaury.pipeline import OrchestratorConfig
                    from yatsaury.session.models import SessionInput
                    from yatsaury.web.jobs import run_generation_job

                    title = title_input.value or "Untitled"
                    source = source_input.value or ""
                    if not source.strip():
                        status_label.set_text("Error: no source provided")
                        return

                    session_input = SessionInput(uri=source)
                    config_dict = {
                        "dataset_types": type_select.value,
                        "schema_names": schema_select.value,
                        "output_formats": format_select.value,
                    }
                    session = store.create(title, [session_input], config_dict)
                    out_dir = workspace / "sessions" / session.id / "outputs"
                    config = OrchestratorConfig(
                        dataset_types=type_select.value,
                        schema_names=schema_select.value,
                        output_formats=format_select.value,
                        output_dir=out_dir,
                    )
                    status_label.set_text(f"Job started: {session.id}")
                    asyncio.create_task(
                        run_generation_job(session.id, store, config, [source])
                    )
                    refresh_history()

                ui.button("Process", on_click=on_process).classes("mt-2")

            # --- History ---
            ui.separator()
            ui.label("History").classes("text-lg font-semibold")
            history_container = ui.column().classes("w-full gap-2")

            def refresh_history() -> None:
                history_container.clear()
                with history_container:
                    sessions = store.list()
                    if not sessions:
                        ui.label("No sessions yet.").classes("text-gray-400")
                    for s in sessions:
                        icon = (
                            "done"
                            if s.status == SessionStatus.done
                            else ("error" if s.status == SessionStatus.error else "pending")
                        )
                        kept = s.counts.get("kept", "?")
                        with ui.card().classes("w-full"):
                            ui.label(
                                f"[{icon}] {s.title}  —  {kept} records  [{s.status}]"
                            )
                            ui.label(s.id).classes("text-xs text-gray-400")

            refresh_history()
            ui.timer(3.0, refresh_history)
