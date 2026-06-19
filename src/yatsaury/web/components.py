"""Dumb render helpers for the Yatsaury web UI (presentation only).

These functions contain no business logic — they take already-resolved data and
render NiceGUI elements styled exclusively from theme tokens.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui
from nicegui.elements.dark_mode import DarkMode

from yatsaury.session.models import Session, SessionStatus

from . import theme

# Status -> (Quasar color name, icon, human label)
_STATUS_META: dict[SessionStatus, tuple[str, str, str]] = {
    SessionStatus.done: ("positive", "check_circle", "Done"),
    SessionStatus.running: ("info", "autorenew", "Running"),
    SessionStatus.queued: ("warning", "schedule", "Queued"),
    SessionStatus.error: ("negative", "error", "Error"),
}


def brand_header(dark_mode_handle: DarkMode) -> None:
    """Header row: brand mark + title + tagline (left), theme toggle (right)."""
    with ui.row().classes("w-full items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("dataset", size="2rem").style("color: var(--text-strong)")
            with ui.column().classes("gap-0"):
                theme.strong(ui.label("Yatsaury").classes("text-2xl font-bold leading-tight"))
                theme.muted(ui.label("Generate training datasets").classes("text-sm leading-tight"))

        toggle = (
            ui.button(icon="dark_mode", on_click=dark_mode_handle.toggle)
            .props("flat round dense")
            .classes("min-w-[44px] min-h-[44px]")
            .style("color: var(--text-muted)")
        )
        toggle.props('aria-label="Toggle light/dark theme"')
        with toggle:
            ui.tooltip("Toggle light/dark theme")

        def _sync_icon() -> None:
            # Show the icon for the action the button performs.
            toggle.props(f"icon={'light_mode' if dark_mode_handle.value else 'dark_mode'}")

        dark_mode_handle.on_value_change(lambda _e: _sync_icon())
        _sync_icon()


def status_chip(status: SessionStatus) -> None:
    """Small rounded chip with a colored dot + text label (color never the only signal)."""
    color, _icon, label = _STATUS_META.get(status, ("info", "help", str(status).title()))
    pulse = " yat-pulse" if status == SessionStatus.running else ""
    with ui.row().classes("items-center gap-2 rounded-full px-3 py-1").style(theme.subtle_style()):
        ui.element("div").classes(f"w-2 h-2 rounded-full{pulse}").style(
            f"background: var(--q-{color})"
        )
        theme.text(ui.label(label).classes("text-xs font-medium"))


def session_row(session: Session) -> None:
    """A structured hairline card representing one session (one semantic unit)."""
    kept = session.counts.get("kept", "?")
    with (
        ui.card()
        .classes("w-full rounded-xl border p-4 gap-2 min-h-[48px]")
        .style(theme.card_style())
    ):
        with ui.row().classes("w-full items-center justify-between gap-4 no-wrap"):
            with ui.column().classes("gap-1 min-w-0"):
                theme.strong(ui.label(session.title).classes("text-base font-medium truncate"))
                theme.muted(ui.label(session.id).classes("text-xs"))
            with ui.row().classes("items-center gap-3 shrink-0"):
                status_chip(session.status)
                theme.text(ui.label(f"{kept} records").classes("text-sm font-medium"))

        if session.status == SessionStatus.running:
            ui.linear_progress(value=session.progress, show_value=False).props(
                "rounded color=info"
            ).classes("w-full")

        if session.status == SessionStatus.error and session.error:
            with ui.row().classes("items-center gap-1"):
                ui.icon("error", size="1rem").style("color: var(--q-negative)")
                ui.label(session.error).classes("text-xs").style("color: var(--q-negative)")


def empty_state() -> None:
    """Centered placeholder shown when there are no sessions yet."""
    with ui.column().classes("w-full items-center justify-center gap-2 py-12"):
        ui.icon("inbox", size="2.5rem").style("color: var(--text-muted)")
        theme.strong(ui.label("No datasets yet").classes("text-lg font-medium"))
        theme.muted(
            ui.label("Fill in the form above and press Process to get started.").classes(
                "text-sm text-center"
            )
        )


def skeleton_rows(n: int = 3) -> None:
    """Render n shimmer placeholder rows matching the session-row layout."""
    for _ in range(n):
        with (
            ui.card()
            .classes("w-full rounded-xl border p-4 gap-2 min-h-[48px]")
            .style(theme.card_style())
        ):
            with ui.row().classes("w-full items-center justify-between gap-4 no-wrap"):
                with ui.column().classes("gap-2 min-w-0"):
                    ui.element("div").classes("yat-shimmer rounded h-4 w-40")
                    ui.element("div").classes("yat-shimmer rounded h-3 w-24")
                ui.element("div").classes("yat-shimmer rounded-full h-6 w-20")


def error_state(retry: Callable[[], None]) -> None:
    """Error placeholder with a Retry button wired to the retry callback."""
    with ui.column().classes("w-full items-center justify-center gap-3 py-12"):
        ui.icon("cloud_off", size="2.5rem").style("color: var(--q-negative)")
        theme.strong(ui.label("Couldn't load history").classes("text-lg font-medium"))
        ui.button("Retry", icon="refresh", on_click=retry).props("flat color=primary")
