from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


class SessionInput(BaseModel):
    uri: str
    kind: str = "file"  # "file" | "url" | "text"


class Session(BaseModel):
    id: str
    title: str
    created_at: str
    status: SessionStatus = SessionStatus.queued
    progress: float = 0.0
    inputs: list[SessionInput] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
