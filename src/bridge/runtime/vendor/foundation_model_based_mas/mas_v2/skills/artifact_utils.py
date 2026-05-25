from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ArtifactRegistryEntry
from .io_utils import read_json, write_json


@dataclass
class SkillArtifactDirs:
    output_dir: Path
    artifact_dir: Path
    adapted_dir: Path
    embedding_dir: Path
    prediction_dir: Path
    coverage_dir: Path
    analysis_dir: Path
    manifest_dir: Path
    log_dir: Path
    summary_path: Path


def prepare_skill_artifact_dirs(output_dir: str | Path) -> SkillArtifactDirs:
    root = Path(output_dir).resolve()
    artifact_dir = root / "artifacts"
    dirs = SkillArtifactDirs(
        output_dir=root,
        artifact_dir=artifact_dir,
        adapted_dir=artifact_dir / "adapted",
        embedding_dir=artifact_dir / "embeddings",
        prediction_dir=artifact_dir / "predictions",
        coverage_dir=artifact_dir / "coverage",
        analysis_dir=artifact_dir / "analysis",
        manifest_dir=artifact_dir / "manifests",
        log_dir=artifact_dir / "logs",
        summary_path=root / "summary.json",
    )
    for path in (
        dirs.output_dir,
        dirs.artifact_dir,
        dirs.adapted_dir,
        dirs.embedding_dir,
        dirs.prediction_dir,
        dirs.coverage_dir,
        dirs.analysis_dir,
        dirs.manifest_dir,
        dirs.log_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def update_artifact_registry(
    *,
    registry_path: str | Path,
    stage: str,
    artifacts: dict[str, str],
    metadata: dict[str, Any] | None = None,
) -> str:
    resolved = Path(registry_path).resolve()
    if resolved.exists():
        payload = read_json(resolved)
    else:
        payload = {"entries": []}
    entries = payload.get("entries", [])
    for key, value in artifacts.items():
        entry = ArtifactRegistryEntry(
            stage=stage,
            key=key,
            path=str(Path(value).resolve()),
            metadata=metadata or {},
        )
        entries.append(entry.model_dump())
    payload["entries"] = entries
    return write_json(resolved, payload)

