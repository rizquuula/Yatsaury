"""SourceLoader protocol and loader registry."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from yatsaury.models import Document


@runtime_checkable
class SourceLoader(Protocol):
    def supports(self, uri: str) -> bool: ...
    def load(self, uri: str) -> Document: ...


def resolve_loader(uri: str, loaders: list[SourceLoader] | None = None) -> SourceLoader:
    """Return the first loader whose supports() returns True."""
    if loaders is None:
        loaders = []
    for loader in loaders:
        if loader.supports(uri):
            return loader
    raise ValueError(f"No loader found for uri: {uri!r}")
