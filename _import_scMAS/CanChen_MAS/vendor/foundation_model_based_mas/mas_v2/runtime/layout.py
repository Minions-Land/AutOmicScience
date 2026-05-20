from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifactLayout:
    model_id: str
    root: Path
    adapted_dir: Path
    embeddings_dir: Path
    predictions_dir: Path
    coverage_dir: Path
    analysis_dir: Path
    manifests_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class RunLayout:
    run_id: str
    root: Path
    input_dir: Path
    planner_dir: Path
    executor_dir: Path
    report_dir: Path
    logs_dir: Path

    def model_layout(self, model_id: str) -> ModelArtifactLayout:
        model_root = self.executor_dir / model_id / "artifacts"
        layout = ModelArtifactLayout(
            model_id=model_id,
            root=model_root,
            adapted_dir=model_root / "adapted",
            embeddings_dir=model_root / "embeddings",
            predictions_dir=model_root / "predictions",
            coverage_dir=model_root / "coverage",
            analysis_dir=model_root / "analysis",
            manifests_dir=model_root / "manifests",
            logs_dir=model_root / "logs",
        )
        for path in [
            layout.root,
            layout.adapted_dir,
            layout.embeddings_dir,
            layout.predictions_dir,
            layout.coverage_dir,
            layout.analysis_dir,
            layout.manifests_dir,
            layout.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return layout


def default_run_id(prefix: str = "mas_v2") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def ensure_run_layout(output_root: str | Path, run_id: str) -> RunLayout:
    output_root = Path(output_root).resolve()
    root = output_root / run_id
    layout = RunLayout(
        run_id=run_id,
        root=root,
        input_dir=root / "input",
        planner_dir=root / "planner",
        executor_dir=root / "executor",
        report_dir=root / "report",
        logs_dir=root / "logs",
    )
    for path in [
        layout.root,
        layout.input_dir,
        layout.planner_dir,
        layout.executor_dir,
        layout.report_dir,
        layout.logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return layout

