"""SourceLoader protocol and loader registry."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from yatsaury.models import Document


@runtime_checkable
class SourceLoader(Protocol):
    def supports(self, uri: str) -> bool: ...
    def load(self, uri: str) -> Document: ...


def _default_loaders() -> list[SourceLoader]:
    from yatsaury.sources.pdf import PdfLoader
    from yatsaury.sources.text import TextLoader
    from yatsaury.sources.url import UrlLoader

    return [PdfLoader(), UrlLoader(), TextLoader()]  # order matters: text last (catch-all)


def resolve_loader(uri: str, loaders: list[SourceLoader] | None = None) -> SourceLoader:
    """Return the first loader whose supports() returns True.

    When loaders is None, uses the default set: PdfLoader, UrlLoader, TextLoader.
    """
    if loaders is None:
        loaders = _default_loaders()
    for loader in loaders:
        if loader.supports(uri):
            return loader
    raise ValueError(f"No loader found for uri: {uri!r}")
