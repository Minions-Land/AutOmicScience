from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mas_v2.contracts import ArtifactRecord, ArtifactRegistry, LogEvent, LogManifest
from mas_v2.contracts.common import utc_now_iso


class ArtifactRegistryStore:
    """Small JSON-backed artifact registry helper for v2 scaffolding."""

    def __init__(self, *, run_id: str, root_dir: str | Path, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path).resolve()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry = ArtifactRegistry(run_id=run_id, root_dir=str(Path(root_dir).resolve()))
        if self.registry_path.exists():
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            self._registry = ArtifactRegistry.model_validate(payload)

    @property
    def value(self) -> ArtifactRegistry:
        return self._registry

    def register(
        self,
        *,
        stage: str,
        kind: str,
        path: str | Path,
        model_id: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            run_id=self._registry.run_id,
            stage=stage,
            kind=kind,  # validated by pydantic literal in ArtifactRecord
            path=str(Path(path).resolve()),
            model_id=model_id,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._registry.records.append(record)
        self._registry.generated_at_utc = utc_now_iso()
        self.save()
        return record

    def save(self) -> None:
        self.registry_path.write_text(
            self._registry.model_dump_json(indent=2),
            encoding="utf-8",
        )


class LogManifestStore:
    """Write JSONL run logs while maintaining a compact manifest."""

    def __init__(self, *, run_id: str, log_jsonl_path: str | Path, manifest_path: str | Path) -> None:
        self.log_jsonl_path = Path(log_jsonl_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.log_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest = LogManifest(run_id=run_id, log_jsonl_path=str(self.log_jsonl_path))
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._manifest = LogManifest.model_validate(payload)

    @property
    def value(self) -> LogManifest:
        return self._manifest

    def append(self, event: LogEvent) -> None:
        line = event.model_dump_json()
        with self.log_jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")
        self._manifest.event_count += 1
        self._manifest.updated_at_utc = utc_now_iso()
        self.save()

    def append_message(
        self,
        *,
        agent: str,
        message: str,
        level: str = "INFO",
        model_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        event = LogEvent(
            run_id=self._manifest.run_id,
            level=level,
            agent=agent,
            message=message,
            model_id=model_id,
            context=context or {},
        )
        self.append(event)

    def save(self) -> None:
        self.manifest_path.write_text(
            self._manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

