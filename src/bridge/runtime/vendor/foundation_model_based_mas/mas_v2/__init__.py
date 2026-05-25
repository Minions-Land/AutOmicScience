"""MAS v2 public package exports."""

from .contracts import (
    AdaptationResult,
    AnalysisResult,
    ArtifactRecord,
    ArtifactRegistry,
    DatasetIntakeBundle,
    DatasetSource,
    EmbeddingPackage,
    KNNTransferResult,
    LogEvent,
    LogManifest,
    ModelSelectionPlan,
    ReferenceAssetPackage,
    RunProfile,
)
from .runtime import load_run_profile

try:
    from .graph.pipeline import MASV2Pipeline
except Exception:  # pragma: no cover - graph entrypoint may not exist during scaffolding.
    MASV2Pipeline = None  # type: ignore[assignment]

__all__ = [
    "AdaptationResult",
    "AnalysisResult",
    "ArtifactRecord",
    "ArtifactRegistry",
    "DatasetIntakeBundle",
    "DatasetSource",
    "EmbeddingPackage",
    "KNNTransferResult",
    "LogEvent",
    "LogManifest",
    "MASV2Pipeline",
    "ModelSelectionPlan",
    "ReferenceAssetPackage",
    "RunProfile",
    "load_run_profile",
]
