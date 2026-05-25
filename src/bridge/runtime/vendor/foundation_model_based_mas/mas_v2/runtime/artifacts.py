from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunWorkspace:
    run_id: str
    root: Path
    input_dir: Path
    planner_dir: Path
    executor_dir: Path
    report_dir: Path
    logs_dir: Path
    registry_path: Path


def _timestamped_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def build_run_workspace(output_root: str | Path, *, run_id: str = "") -> RunWorkspace:
    root_dir = Path(output_root).resolve()
    resolved_run_id = run_id or _timestamped_run_id("mas_v2")
    run_root = root_dir / resolved_run_id
    input_dir = run_root / "input"
    planner_dir = run_root / "planner"
    executor_dir = run_root / "executor"
    report_dir = run_root / "report"
    logs_dir = run_root / "logs"
    for path in (run_root, input_dir, planner_dir, executor_dir, report_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    registry_path = run_root / "artifact_registry.json"
    if not registry_path.exists():
        registry_path.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
    return RunWorkspace(
        run_id=resolved_run_id,
        root=run_root,
        input_dir=input_dir,
        planner_dir=planner_dir,
        executor_dir=executor_dir,
        report_dir=report_dir,
        logs_dir=logs_dir,
        registry_path=registry_path,
    )


def register_artifacts(registry_path: str | Path, component: str, payload: dict[str, Any]) -> str:
    path = Path(registry_path).resolve()
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {}
    registry[str(component)] = payload
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path)
