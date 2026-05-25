from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from .common import StrictModel, utc_now_iso


ArtifactKindLiteral = Literal[
    "adapted",
    "embeddings",
    "predictions",
    "coverage",
    "analysis",
    "manifests",
    "logs",
    "report",
    "input",
    "planner",
]

LogLevelLiteral = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class ArtifactRecord(StrictModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    stage: str
    kind: ArtifactKindLiteral
    path: str
    model_id: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str = Field(default_factory=utc_now_iso)


class ArtifactRegistry(StrictModel):
    run_id: str
    root_dir: str
    generated_at_utc: str = Field(default_factory=utc_now_iso)
    records: list[ArtifactRecord] = Field(default_factory=list)


class LogEvent(StrictModel):
    run_id: str
    ts_utc: str = Field(default_factory=utc_now_iso)
    level: LogLevelLiteral = "INFO"
    agent: str
    message: str
    model_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class LogManifest(StrictModel):
    run_id: str
    log_jsonl_path: str
    created_at_utc: str = Field(default_factory=utc_now_iso)
    updated_at_utc: str = Field(default_factory=utc_now_iso)
    event_count: int = 0

