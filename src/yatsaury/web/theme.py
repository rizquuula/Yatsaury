"""Single source of truth for Yatsaury web theming tokens.

Light-first, emerald/teal accent, minimal & airy aesthetic (hairline borders,
no drop shadows, generous whitespace). All colors and reused spacing live here
so no component carries raw hex or magic spacing.
"""

from __future__ import annotations

from typing import TypeVar

from nicegui import ui
from nicegui.element import Element

E = TypeVar("E", bound=Element)

# --- Quasar/NiceGUI brand colors (emerald/teal accent) ---
_PRIMARY = "#059669"
_SECONDARY = "#0d9488"
_ACCENT = "#10b981"
_POSITIVE = "#059669"
_NEGATIVE = "#dc2626"
_WARNING = "#d97706"
_INFO = "#2563eb"
_DARK = "#0f172a"

# --- Semantic CSS custom properties injected once on :root / body.body--dark ---
_CSS = """
:root {
  --surface: #ffffff;
  --surface-page: #f8fafc;
  --surface-subtle: #f1f5f9;
  --border: #e2e8f0;
  --text-strong: #0f172a;
  --text: #334155;
  --text-muted: #64748b;
}

body.body--dark {
  --surface: #1e293b;
  --surface-page: #0f172a;
  --surface-subtle: #334155;
  --border: #334155;
  --text-strong: #f1f5f9;
  --text: #cbd5e1;
  --text-muted: #94a3b8;
}

/* Page background driven by the semantic token. */
body, .q-page, .nicegui-content {
  background: var(--surface-page);
  color: var(--text);
}

/* Visible emerald focus ring on every interactive element. */
button:focus-visible,
a:focus-visible,
[tabindex]:focus-visible,
input:focus-visible,
textarea:focus-visible,
.q-field--focused .q-field__control,
.q-btn:focus-visible {
  outline: 2px solid #10b981;
  outline-offset: 2px;
}

/* Motion: gentle, <=400ms, ease-out. Respect reduced-motion. */
@keyframes yat-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

@keyframes yat-shimmer {
  0% { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}

.yat-pulse {
  animation: yat-pulse 1.4s ease-in-out infinite;
}

.yat-shimmer {
  background: linear-gradient(
    90deg,
    var(--surface-subtle) 0%,
    var(--border) 50%,
    var(--surface-subtle) 100%
  );
  background-size: 800px 100%;
  animation: yat-shimmer 1.6s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
  .yat-pulse { animation: none; }
  .yat-shimmer { animation: none; }
}
"""


def apply_theme() -> None:
    """Register brand colors and inject the semantic CSS token block.

    Call once per page render before building the UI.
    """
    ui.colors(
        primary=_PRIMARY,
        secondary=_SECONDARY,
        accent=_ACCENT,
        positive=_POSITIVE,
        negative=_NEGATIVE,
        warning=_WARNING,
        info=_INFO,
        dark=_DARK,
    )
    ui.add_css(_CSS)


# --- Reusable Tailwind class strings (8px grid; no magic spacing elsewhere) ---
PAGE = "w-full max-w-3xl mx-auto px-6 py-10"
SECTION_GAP = "gap-8"  # 32px between major sections
CARD = "w-full rounded-xl border p-6 gap-4"  # 24px padding, 16px in-card gap
CARD_GAP = "gap-4"  # 16px
ROW_GAP = "gap-4"

# --- Style strings binding Tailwind-unreachable tokens ---
_SURFACE_STYLE = "background: var(--surface); border-color: var(--border)"
_SUBTLE_STYLE = "background: var(--surface-subtle); border-color: var(--border)"


def card_style() -> str:
    """Inline style applying surface + border tokens to a card element."""
    return _SURFACE_STYLE


def subtle_style() -> str:
    """Inline style for a subtle (recessed) surface."""
    return _SUBTLE_STYLE


def strong(element: E) -> E:
    """Apply the strong text color token to a text element. Returns element."""
    element.style("color: var(--text-strong)")
    return element


def text(element: E) -> E:
    """Apply the default body text color token. Returns element."""
    element.style("color: var(--text)")
    return element


def muted(element: E) -> E:
    """Apply the muted (WCAG-safe slate-500) text color token. Returns element."""
    element.style("color: var(--text-muted)")
    return element
