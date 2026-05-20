from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class StructuredRunLogger:
    def __init__(self, *, root: str | Path, component: str):
        self.component = component
        self.log_dir = Path(root).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.log_dir / "run.log"
        self.events_path = self.log_dir / "events.jsonl"
        self.errors_path = self.log_dir / "errors.jsonl"
        self._start = time.perf_counter()
        for path in (self.run_log_path, self.events_path, self.errors_path):
            path.touch(exist_ok=True)

    def manifest(self) -> dict[str, str]:
        return {
            "run_log": str(self.run_log_path),
            "events_jsonl": str(self.events_path),
            "errors_jsonl": str(self.errors_path),
        }

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _append_log(self, text: str) -> None:
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")

    def event(self, name: str, *, status: str = "info", payload: dict[str, Any] | None = None) -> None:
        record = {
            "ts": self._ts(),
            "component": self.component,
            "name": name,
            "status": status,
            "payload": payload or {},
        }
        self._append_jsonl(self.events_path, record)
        self._append_log(f"[{record['ts']}] [{self.component}] [{status}] {name} | {json.dumps(record['payload'], ensure_ascii=False, default=str)}")

    def error(self, name: str, exc: BaseException, *, payload: dict[str, Any] | None = None) -> None:
        record = {
            "ts": self._ts(),
            "component": self.component,
            "name": name,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "payload": payload or {},
            "traceback": traceback.format_exc(),
        }
        self._append_jsonl(self.errors_path, record)
        self._append_log(f"[{record['ts']}] [{self.component}] [error] {name} | {record['error_message']}")

    @contextmanager
    def span(self, name: str, *, payload: dict[str, Any] | None = None) -> Iterator[None]:
        start = time.perf_counter()
        self.event(name, status="start", payload=payload)
        try:
            yield
        except Exception as exc:
            self.error(name, exc, payload={"duration_sec": round(time.perf_counter() - start, 4), **(payload or {})})
            raise
        else:
            self.event(name, status="end", payload={"duration_sec": round(time.perf_counter() - start, 4), **(payload or {})})

    def finalize(self, *, status: str, payload: dict[str, Any] | None = None) -> None:
        self.event("run_finished", status=status, payload={"total_duration_sec": round(time.perf_counter() - self._start, 4), **(payload or {})})
