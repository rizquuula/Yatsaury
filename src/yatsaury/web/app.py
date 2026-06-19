"""NiceGUI web application for Yatsaury."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from nicegui import app, ui

from yatsaury.session.store import SessionStore

from . import components, theme

logger = logging.getLogger(__name__)

_DARK_MODE_KEY = "dark_mode"


def create_app(store: SessionStore, workspace: Path) -> None:
    """Register the NiceGUI page routes."""

    @ui.page("/")
    async def index() -> None:
        theme.apply_theme()

        # Light-first dark-mode handle, persisted across reloads when possible.
        dark = ui.dark_mode()
        try:
            dark.value = bool(app.storage.user.get(_DARK_MODE_KEY, False))

            def _persist_dark(event: object) -> None:
                value = getattr(event, "value", False)
                try:
                    app.storage.user[_DARK_MODE_KEY] = bool(value)
                except Exception:
                    pass

            dark.on_value_change(_persist_dark)
        except Exception:
            # Storage may be unavailable (no storage_secret); fall back to light.
            dark.disable()

        with (
            ui.column()
            .classes(f"{theme.PAGE} {theme.SECTION_GAP}")
            .style("background: var(--surface-page)")
        ):
            components.brand_header(dark)

            # --- Process form ---
            with ui.card().classes(theme.CARD).style(theme.card_style()):
                theme.strong(ui.label("New Dataset").classes("text-lg font-semibold"))

                theme.text(ui.label("Title").classes("text-sm font-medium"))
                title_input = ui.input(placeholder="e.g. Sirah v1").classes("w-full")

                theme.text(ui.label("Source").classes("text-sm font-medium"))
                source_input = (
                    ui.textarea(placeholder="Paste a URL or raw text")
                    .props("rows=5")
                    .classes("w-full")
                )
                theme.muted(
                    ui.label("A URL to fetch, or raw text to use directly.").classes("text-xs")
                )

                # --- Output options (responsive, stacks on mobile) ---
                theme.strong(ui.label("Output options").classes("text-sm font-semibold mt-2"))
                with ui.element("div").classes("w-full flex flex-col md:flex-row gap-4"):
                    type_select = ui.select(
                        ["qa", "instruction", "rag", "summary"],
                        multiple=True,
                        value=["qa"],
                        label="Dataset type",
                    ).classes("w-full md:flex-1")
                    schema_select = ui.select(
                        ["chatml", "sharegpt", "alpaca", "qa", "completion", "rag", "raw"],
                        multiple=True,
                        value=["chatml"],
                        label="Schema",
                    ).classes("w-full md:flex-1")
                    format_select = ui.select(
                        ["jsonl", "hf", "csv"],
                        multiple=True,
                        value=["jsonl"],
                        label="Format",
                    ).classes("w-full md:flex-1")

                # --- Inline feedback area (replaces gray status label) ---
                feedback = ui.column().classes("w-full gap-0")

                # --- Validation helper text + footer button ---
                helper = theme.muted(ui.label("Add a source to enable Process.").classes("text-xs"))

                async def on_process() -> None:
                    from yatsaury.pipeline import OrchestratorConfig
                    from yatsaury.session.models import SessionInput
                    from yatsaury.web.jobs import run_generation_job

                    title = title_input.value or "Untitled"
                    source = source_input.value or ""
                    feedback.clear()
                    if not source.strip():
                        with feedback:
                            with ui.row().classes("items-center gap-1"):
                                ui.icon("error", size="1rem").style("color: var(--q-negative)")
                                ui.label("Error: please provide a source first.").classes(
                                    "text-sm"
                                ).style("color: var(--q-negative)")
                        return

                    process_btn.props("loading")
                    try:
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
                        asyncio.create_task(run_generation_job(session.id, store, config, [source]))
                        with feedback:
                            with (
                                ui.row()
                                .classes("items-center gap-1 rounded-lg px-3 py-2")
                                .style(theme.subtle_style())
                            ):
                                ui.icon("check_circle", size="1rem").style(
                                    "color: var(--q-positive)"
                                )
                                theme.text(
                                    ui.label(f"Job started: {session.id}").classes("text-sm")
                                )
                        refresh_history()
                    finally:
                        process_btn.props(remove="loading")

                with ui.row().classes("w-full items-center justify-end gap-3 mt-2"):
                    process_btn = ui.button("Process", on_click=on_process).props(
                        "color=primary unelevated"
                    )

                def _sync_process_enabled() -> None:
                    has_source = bool((source_input.value or "").strip())
                    process_btn.set_enabled(has_source)
                    helper.set_visibility(not has_source)

                source_input.on_value_change(lambda _e: _sync_process_enabled())
                _sync_process_enabled()

            # --- History ---
            with ui.column().classes(f"w-full {theme.CARD_GAP}"):
                with ui.row().classes("w-full items-center justify-between"):
                    theme.strong(ui.label("History").classes("text-lg font-semibold"))
                    count_label = theme.muted(ui.label("").classes("text-sm"))

                history_container = ui.column().classes("w-full gap-3")

            def refresh_history() -> None:
                history_container.clear()
                sessions = store.list()
                n = len(sessions)
                count_label.set_text(f"{n} session" + ("" if n == 1 else "s"))
                with history_container:
                    for s in sessions:
                        components.session_row(s)

            refresh_history()
            ui.timer(3.0, refresh_history)
