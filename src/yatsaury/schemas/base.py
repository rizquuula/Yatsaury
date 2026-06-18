"""SchemaAdapter protocol and registry."""
from __future__ import annotations

from typing import Protocol

from yatsaury.models import Sample


class SchemaAdapter(Protocol):
    name: str

    def supports(self, dataset_type: str) -> bool: ...
    def render(self, sample: Sample) -> dict: ...


_REGISTRY: dict[str, SchemaAdapter] = {}


def register_schema(adapter: SchemaAdapter) -> None:
    """Register a schema adapter instance."""
    _REGISTRY[adapter.name] = adapter


def get_schema(name: str) -> SchemaAdapter:
    """Return the schema adapter for the given name."""
    if name not in _REGISTRY:
        raise KeyError(f"No schema registered for: {name!r}")
    return _REGISTRY[name]


def list_schemas() -> list[str]:
    """Return list of registered schema names."""
    return list(_REGISTRY.keys())
