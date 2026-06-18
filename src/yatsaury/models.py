"""Yatsaury data models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    pdf = "pdf"
    url = "url"
    text = "text"


class Document(BaseModel):
    id: str
    source_uri: str
    source_type: SourceType
    raw_text: str
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    token_count: int = Field(ge=0)
    char_span: tuple[int, int]
    page: int | None = None
    ordinal: int


class Citation(BaseModel):
    title: str
    page: int | None = None
    char_span: tuple[int, int] | None = None
    source_uri: str


class DatasetType(StrEnum):
    qa = "qa"
    instruction = "instruction"
    rag = "rag"
    summary = "summary"


class Sample(BaseModel):
    id: str
    chunk_id: str
    dataset_type: DatasetType
    payload: dict[str, Any] = Field(default_factory=dict)
    source_text: str
    supporting_quote: str
    source_citation: Citation
    grounding_score: float = 0.0
    verified: bool = False
    dedup_hash: str = ""
    fact_id: str | None = None
    lang: str = "auto"
