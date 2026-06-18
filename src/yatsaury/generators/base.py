"""Generator protocol and registry."""
from __future__ import annotations

from typing import Protocol

from yatsaury.llm.client import LLMClient
from yatsaury.models import Chunk, Sample


class Generator(Protocol):
    dataset_type: str

    def generate(self, chunk: Chunk, n: int, llm: LLMClient) -> list[Sample]: ...


_REGISTRY: dict[str, Generator] = {}


def register_generator(gen: Generator) -> None:
    """Register a generator instance by its dataset_type."""
    _REGISTRY[gen.dataset_type] = gen


def get_generator(dataset_type: str) -> Generator:
    """Return the generator for the given dataset_type."""
    if dataset_type not in _REGISTRY:
        raise KeyError(f"No generator registered for: {dataset_type!r}")
    return _REGISTRY[dataset_type]
