from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class ArtifactPaths(BaseModel):
    paths: dict[str, str] = Field(default_factory=dict)


class DatasetFingerprint(BaseModel):
    algorithm: str = "sha256"
    value: str
    source_paths: list[str] = Field(default_factory=list)


class RawDatasetSource(BaseModel):
    source_type: Literal["raw_dataset"]
    path: str = ""
    h5ad_path: str = ""
    npz_path: str = ""
    mtx_path: str = ""
    obs_path: str = ""
    var_path: str = ""
    member_path: str = ""
    species: str = ""
    preferred_gene_name_key: str = ""
    preferred_x_source: Literal["X", "layers", "raw"] = "X"
    preferred_layer_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> "RawDatasetSource":
        if not any([self.path, self.h5ad_path, self.npz_path, self.mtx_path]):
            raise ValueError("RawDatasetSource requires at least one primary path field.")
        return self


class PreparedDatasetSource(BaseModel):
    source_type: Literal["prepared_dataset"]
    path: str = ""
    h5ad_path: str = ""
    obs_path: str = ""
    member_path: str = ""
    species: str = ""
    preferred_gene_name_key: str = ""
    preferred_x_source: Literal["X", "layers", "raw"] = "X"
    preferred_layer_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> "PreparedDatasetSource":
        if not any([self.path, self.h5ad_path, self.obs_path]):
            raise ValueError("PreparedDatasetSource requires at least one path field.")
        return self


class ReferenceAssetPackage(BaseModel):
    source_type: Literal["reference_asset_package"] = "reference_asset_package"
    model_id: str
    reference_embeddings_path: str
    reference_obs_path: str
    coverage_json: str
    source_manifest: str
    reference_label_key: str
    dataset_fingerprint: str
    summary_json: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


DatasetSource = Annotated[
    RawDatasetSource | PreparedDatasetSource | ReferenceAssetPackage,
    Field(discriminator="source_type"),
]


class SamplingPlan(BaseModel):
    random_seed: int = 3028
    max_reference_cells: int = 0
    max_reference_cells_per_label: int = 0
    max_query_cells: int = 0


class AnalysisRunConfig(BaseModel):
    run_name: str
    query_label_key: str
    reference_eval_mapping: dict[str, str] = Field(default_factory=dict)
    query_eval_mapping: dict[str, str] = Field(default_factory=dict)


class InputProfile(BaseModel):
    dataset_id: str
    task_request: str
    dataset_description: str = ""
    reference_source: DatasetSource
    query_source: DatasetSource
    reference_asset_packages: dict[str, ReferenceAssetPackage] = Field(default_factory=dict)


class PlannerProfile(BaseModel):
    candidate_models: list[str] = Field(
        default_factory=lambda: [
            "geneformer",
            "nicheformer",
            "scgpt_generic",
            "scgpt_human",
            "scgpt_generic_brain",
            "uce",
            "uce_33l",
        ]
    )
    max_selected_models: int = 3
    excluded_models: list[str] = Field(default_factory=list)
    require_llm: bool = False
    llm_prefix: str = "OPENAI"


class ExecutorProfile(BaseModel):
    reference_label_key: str
    batch_size: int = 64
    device: str = ""
    k: int = 25
    metric: str = "cosine"
    min_vote_share: float = 0.5
    max_mean_distance: float | None = None
    persist_embeddings: bool = True
    max_adapter_retries: int = 2
    sampling: SamplingPlan = Field(default_factory=SamplingPlan)
    model_species_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)


class AnalysisProfile(BaseModel):
    runs: list[AnalysisRunConfig] = Field(default_factory=list)


class ReporterProfile(BaseModel):
    llm_prefix: str = "OPENAI"
    use_llm: bool = True
    deterministic_fallback: bool = True


class LoggingProfile(BaseModel):
    output_root: str = "outputs/mas_v2"
    env_path: str = ".env"
    enable_tracing: bool = True


class RunProfile(BaseModel):
    input: InputProfile
    planner: PlannerProfile = Field(default_factory=PlannerProfile)
    executor: ExecutorProfile
    analysis: AnalysisProfile
    reporter: ReporterProfile = Field(default_factory=ReporterProfile)
    logging: LoggingProfile = Field(default_factory=LoggingProfile)


class DatasetDescriptor(BaseModel):
    role: Literal["reference", "query"]
    source_type: str
    resolved_paths: list[str] = Field(default_factory=list)
    file_types: list[str] = Field(default_factory=list)
    n_obs: int | None = None
    n_vars: int | None = None
    obs_keys: list[str] = Field(default_factory=list)
    var_keys: list[str] = Field(default_factory=list)
    layers: list[str] = Field(default_factory=list)
    obsm_keys: list[str] = Field(default_factory=list)
    candidate_label_keys: list[str] = Field(default_factory=list)
    candidate_gene_name_keys: list[str] = Field(default_factory=list)
    candidate_x_sources: list[str] = Field(default_factory=list)
    candidate_layers: list[str] = Field(default_factory=list)
    spatial_keys: list[str] = Field(default_factory=list)
    species_hint: str = ""
    panel_size: int | None = None
    matrix_sparsity: float | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetIntakeBundle(BaseModel):
    dataset_id: str
    task_request: str
    dataset_description: str
    reference: DatasetDescriptor
    query: DatasetDescriptor
    fingerprint: DatasetFingerprint
    sampling_plan: SamplingPlan
    scdesign3_context: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)


class ModelSelectionItem(BaseModel):
    model_id: str
    priority_rank: int
    selection_rationale: str
    required_reference_mode: str
    required_query_view: str
    availability_status: str
    judge_status: str = "pending"
    judge_notes: list[str] = Field(default_factory=list)
    score: float = 0.0


class ModelSelectionPlan(BaseModel):
    selected_models: list[ModelSelectionItem] = Field(default_factory=list)
    rejected_models: list[dict[str, Any]] = Field(default_factory=list)
    candidate_models: list[str] = Field(default_factory=list)
    judge_reviews: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)


class DatasetView(BaseModel):
    dataset_path: str = ""
    obs_csv: str = ""
    n_cells: int | None = None
    n_genes: int | None = None
    gene_name_source: str = ""
    x_source: str = ""
    layer_name: str = ""
    species: str = ""


class CellOrderArtifacts(BaseModel):
    reference_obs_names_path: str = ""
    query_obs_names_path: str = ""
    reference_obs_csv: str = ""
    query_obs_csv: str = ""


class RepairRecord(BaseModel):
    attempt: int
    action: str
    reason: str
    status: Literal["retrying", "validated", "failed"]


class AdaptationResult(BaseModel):
    status: Literal["success", "failed"]
    model_id: str
    output_dir: str
    reference_view: DatasetView = Field(default_factory=DatasetView)
    query_view: DatasetView = Field(default_factory=DatasetView)
    sampling_manifest: dict[str, Any] = Field(default_factory=dict)
    cell_order_artifacts: CellOrderArtifacts = Field(default_factory=CellOrderArtifacts)
    coverage_metrics: dict[str, Any] = Field(default_factory=dict)
    repair_history: list[RepairRecord] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str = ""


class EmbeddingPackage(BaseModel):
    skill_name: str = "foundation_embedding_skill"
    status: Literal["success", "failed"]
    model_id: str
    output_dir: str
    run_id: str
    artifact_registry_path: str = ""
    reference_species: str = ""
    query_species: str = ""
    reference_n_cells: int = 0
    query_n_cells: int = 0
    embedding_dim: int = 0
    reference_label_key: str = ""
    query_label_key: str = ""
    coverage: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str = ""


class KNNTransferResult(BaseModel):
    skill_name: str = "shared_knn_transfer_skill"
    status: Literal["success", "failed"]
    model_id: str
    output_dir: str
    run_id: str
    artifact_registry_path: str = ""
    reference_label_key: str
    query_label_key: str = ""
    n_reference_cells: int = 0
    n_query_cells: int = 0
    n_neighbors: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str = ""


class AnalysisResult(BaseModel):
    skill_name: str = "shared_prediction_analysis_skill"
    status: Literal["success", "failed"]
    model_id: str
    output_dir: str
    run_id: str
    artifact_registry_path: str = ""
    run_name: str
    n_total: int = 0
    n_unknown: int = 0
    n_known: int = 0
    unknown_rate: float = 0.0
    coarse_accuracy: float | None = None
    coarse_macro_f1: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str = ""
