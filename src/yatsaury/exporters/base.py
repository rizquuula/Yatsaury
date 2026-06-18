"""Exporter protocol and registry."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol


class Exporter(Protocol):
    def export(self, records: Iterable[dict], out_path: Path) -> None: ...


_REGISTRY: dict[str, type] = {}


def register_exporter(name: str, exporter_cls: type) -> None:
    """Register an exporter class under the given name."""
    _REGISTRY[name] = exporter_cls


def get_exporter(name: str) -> type:
    """Return the exporter class for the given name."""
    if name not in _REGISTRY:
        raise KeyError(f"No exporter registered for: {name!r}")
    return _REGISTRY[name]
