from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


UNKNOWN_LABEL = "__unknown__"


class SkillResponse(BaseModel):
    skill_name: str
    status: Literal["success", "failed"]
    model_id: str = ""
    output_dir: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class CoverageRecord(BaseModel):
    reference_path: str = ""
    n_reference_cells: int
    gene_name_source: str
    gene_overlap_count: int
    coverage_ratio: float
    candidate_scores: list[dict[str, Any]] = Field(default_factory=list)


class QueryCoverageRecord(BaseModel):
    gene_name_source: str
    gene_overlap_count: int
    coverage_ratio: float
    candidate_scores: list[dict[str, Any]] = Field(default_factory=list)


class EmbeddingPackage(SkillResponse):
    reference_species: str = ""
    query_species: str = ""
    reference_n_cells: int = 0
    query_n_cells: int = 0
    embedding_dim: int = 0
    reference_label_key: str = ""
    query_label_key: str = ""
    coverage: dict[str, Any] = Field(default_factory=dict)
    run_id: str = ""
    artifact_registry_path: str = ""


class KNNTransferResult(SkillResponse):
    reference_label_key: str = ""
    query_label_key: str = ""
    n_reference_cells: int = 0
    n_query_cells: int = 0
    n_neighbors: int = 0
    run_id: str = ""
    artifact_registry_path: str = ""


class AnalysisResult(SkillResponse):
    run_name: str = ""
    n_total: int = 0
    n_unknown: int = 0
    n_known: int = 0
    unknown_rate: float = 0.0
    coarse_accuracy: float | None = None
    coarse_macro_f1: float | None = None
    run_id: str = ""
    artifact_registry_path: str = ""


class ReferenceAssetPackage(BaseModel):
    model_id: str
    reference_embeddings_path: str
    reference_obs_path: str
    coverage_json: str = ""
    source_manifest: str = ""
    reference_label_key: str = ""
    dataset_fingerprint: str = ""

    @model_validator(mode="after")
    def _validate_paths(self) -> "ReferenceAssetPackage":
        required = [
            ("reference_embeddings_path", self.reference_embeddings_path),
            ("reference_obs_path", self.reference_obs_path),
        ]
        missing = [name for name, path in required if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"ReferenceAssetPackage missing files for: {', '.join(missing)}")
        return self


class BaseEmbeddingSkillInput(BaseModel):
    output_dir: str
    model_id: str
    run_id: str = ""

    reference_path: str = ""
    reference_paths: list[str] = Field(default_factory=list)
    reference_obs_names_paths: list[str] = Field(default_factory=list)
    query_path: str

    reference_label_key: str = "cell_type"
    query_label_key: str = ""

    reference_species: str = "mouse"
    query_species: str = "mouse"
    reference_gene_name_key: str = ""
    query_gene_name_key: str = ""

    x_source: Literal["X", "layers", "raw"] = "X"
    layer_name: str = ""

    device: str = ""
    batch_size: int = 64
    random_seed: int = 3028
    max_reference_cells: int = 50000
    max_reference_cells_per_label: int = 5000
    max_query_cells: int = 20000
    persist_embeddings: bool = True

    artifact_registry_path: str = ""

    @model_validator(mode="after")
    def _validate_inputs(self) -> "BaseEmbeddingSkillInput":
        refs = self.reference_paths or ([self.reference_path] if self.reference_path else [])
        if not refs:
            raise ValueError("Provide reference_path or reference_paths.")
        for ref in refs:
            if not Path(ref).exists():
                raise FileNotFoundError(f"Reference path not found: {ref}")
        if not Path(self.query_path).exists():
            raise FileNotFoundError(f"Query path not found: {self.query_path}")
        if self.x_source == "layers" and not self.layer_name:
            raise ValueError("layer_name must be provided when x_source == 'layers'.")
        return self

    def resolved_reference_paths(self) -> list[str]:
        return self.reference_paths or [self.reference_path]


class KNNTransferSkillInput(BaseModel):
    output_dir: str
    model_id: str
    run_id: str = ""

    reference_embeddings_path: str = ""
    query_embeddings_path: str
    reference_obs_path: str = ""
    query_obs_path: str

    reference_asset_package_path: str = ""

    reference_label_key: str
    query_label_key: str = ""
    k: int = 25
    metric: str = "cosine"
    min_vote_share: float = 0.5
    max_mean_distance: float | None = None
    artifact_registry_path: str = ""

    @model_validator(mode="after")
    def _validate_embeddings(self) -> "KNNTransferSkillInput":
        if not self.reference_asset_package_path:
            if not self.reference_embeddings_path or not self.reference_obs_path:
                raise ValueError(
                    "Provide reference_embeddings_path/reference_obs_path or reference_asset_package_path."
                )
        if self.k <= 0:
            raise ValueError("k must be > 0.")
        return self


class PredictionAnalysisSkillInput(BaseModel):
    output_dir: str
    model_id: str
    run_id: str = ""
    run_name: str = ""

    prediction_csv_path: str
    query_obs_path: str = ""
    reference_label_key: str
    query_label_key: str = ""
    reference_eval_mapping: dict[str, str] = Field(default_factory=dict)
    query_eval_mapping: dict[str, str] = Field(default_factory=dict)
    artifact_registry_path: str = ""

    @model_validator(mode="after")
    def _validate_inputs(self) -> "PredictionAnalysisSkillInput":
        if not Path(self.prediction_csv_path).exists():
            raise FileNotFoundError(f"prediction_csv_path not found: {self.prediction_csv_path}")
        if self.query_obs_path and not Path(self.query_obs_path).exists():
            raise FileNotFoundError(f"query_obs_path not found: {self.query_obs_path}")
        return self


class ArtifactRegistryEntry(BaseModel):
    stage: str
    key: str
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)

