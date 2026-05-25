from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


class StructuredSkillLogger:
    def __init__(self, *, artifact_root: Path, component: str):
        self.component = component
        self.log_dir = artifact_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.log_dir / "run.log"
        self.events_path = self.log_dir / "events.jsonl"
        self.tool_calls_path = self.log_dir / "tool_calls.jsonl"
        self.errors_path = self.log_dir / "errors.jsonl"
        self.manifest_path = self.log_dir / "manifest.json"
        self._start_ts = time.perf_counter()
        for path in (self.run_log_path, self.events_path, self.tool_calls_path, self.errors_path):
            path.touch(exist_ok=True)
        self._write_manifest()

    def manifest(self) -> dict[str, str]:
        return {
            "log_dir": str(self.log_dir),
            "run_log": str(self.run_log_path),
            "events_jsonl": str(self.events_path),
            "tool_calls_jsonl": str(self.tool_calls_path),
            "errors_jsonl": str(self.errors_path),
            "manifest_json": str(self.manifest_path),
        }

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "component": self.component,
                    **self.manifest(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        safe_payload = _json_safe(payload)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")

    def _append_run_log(self, message: str) -> None:
        with self.run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def _format_payload(self, payload: dict[str, Any] | None) -> str:
        if not payload:
            return ""
        parts: list[str] = []
        for key, value in payload.items():
            rendered = _json_safe(value)
            if isinstance(rendered, (dict, list)):
                text = json.dumps(rendered, ensure_ascii=False, separators=(",", ":"))
            else:
                text = str(rendered)
            parts.append(f"{key}={text}")
        return " | " + ", ".join(parts) if parts else ""

    def event(
        self,
        name: str,
        *,
        status: str = "info",
        payload: dict[str, Any] | None = None,
        event_type: str = "event",
    ) -> None:
        record = {
            "ts": self._timestamp(),
            "component": self.component,
            "event_type": event_type,
            "name": name,
            "status": status,
            "payload": payload or {},
        }
        self._append_jsonl(self.events_path, record)
        self._append_run_log(
            f"[{record['ts']}] [{self.component}] [{event_type}/{status}] {name}{self._format_payload(payload)}"
        )

    def tool_call(
        self,
        tool_name: str,
        *,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "ts": self._timestamp(),
            "component": self.component,
            "tool_name": tool_name,
            "status": status,
            "payload": payload or {},
        }
        self._append_jsonl(self.tool_calls_path, record)
        self._append_run_log(
            f"[{record['ts']}] [{self.component}] [tool/{status}] {tool_name}{self._format_payload(payload)}"
        )

    def error(
        self,
        name: str,
        exc: BaseException,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "ts": self._timestamp(),
            "component": self.component,
            "name": name,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "payload": payload or {},
            "traceback": traceback.format_exc(),
        }
        self._append_jsonl(self.errors_path, record)
        self._append_run_log(
            f"[{record['ts']}] [{self.component}] [error] {name}{self._format_payload({'error': record['error_message'], **(payload or {})})}"
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        payload: dict[str, Any] | None = None,
        event_type: str = "step",
    ) -> Iterator[None]:
        start = time.perf_counter()
        self.event(name, status="start", payload=payload, event_type=event_type)
        try:
            yield
        except Exception as exc:
            duration_sec = round(time.perf_counter() - start, 4)
            self.error(name, exc, payload={"duration_sec": duration_sec, **(payload or {})})
            raise
        else:
            duration_sec = round(time.perf_counter() - start, 4)
            self.event(
                name,
                status="end",
                payload={"duration_sec": duration_sec, **(payload or {})},
                event_type=event_type,
            )

    def finalize(self, *, status: str, payload: dict[str, Any] | None = None) -> None:
        total_duration = round(time.perf_counter() - self._start_ts, 4)
        merged = {"total_duration_sec": total_duration, **(payload or {})}
        self.event("run_finished", status=status, payload=merged, event_type="lifecycle")

