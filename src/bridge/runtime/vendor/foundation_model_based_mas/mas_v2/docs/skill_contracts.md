# MAS v2 Shared Skill Contracts

This document is the compact grounding spec for MAS v2 skill-to-skill handoff.

## Pipeline Contract
`input -> planner -> planner_judge -> adapter -> adapter_judge -> foundation_embedding_skill -> shared_knn_transfer_skill -> shared_prediction_analysis_skill -> reporter`

## Core Pydantic Types
- `RunProfile`: top-level config; fixed sections `input/planner/executor/analysis/reporter/logging`.
- `DatasetSource`: discriminated union with `raw_dataset`, `prepared_dataset`, `reference_asset_package`.
- `DatasetIntakeBundle`: first-seen dataset profile + sampling plan + `scdesign3_context` placeholder + warnings + fingerprint.
- `ModelSelectionPlan`: model picks with `model_id`, `priority_rank`, `selection_rationale`, `required_reference_mode`, `required_query_view`, `availability_status`, `judge_status`.
- `AdaptationResult`: adapter output with reference/query view paths, selected matrix source/layer, selected gene keys, sampling manifest, cell-order artifacts, coverage, repair history.
- `ReferenceAssetPackage`: packaged reference embeddings + obs + coverage + provenance manifest.
- `EmbeddingPackage`: model embedding output package with artifact paths and shape metadata.
- `KNNTransferResult`: shared KNN prediction output + neighbors package.
- `AnalysisResult`: shared analysis output (unknown rate + confusion artifacts).
- `ArtifactRegistry` and `LogManifest`: run-level artifact and log bookkeeping.

## Artifact Directory Convention
`outputs/mas_v2/<run_id>/`
- `input/`
- `planner/`
- `executor/<model_id>/artifacts/adapted`
- `executor/<model_id>/artifacts/embeddings`
- `executor/<model_id>/artifacts/predictions`
- `executor/<model_id>/artifacts/coverage`
- `executor/<model_id>/artifacts/analysis`
- `executor/<model_id>/artifacts/manifests`
- `executor/<model_id>/artifacts/logs`
- `report/`
- `logs/`

## Invariants
- Large arrays must be written to files and passed as artifact paths.
- `reference_embeddings_npy` row order must match `reference_obs_csv`.
- `query_embeddings_npy` row order must match `query_obs_csv`.
- `neighbors.npz` query row index must match `query_obs_csv` row order.
- Adapter judge may run up to 2 repair rounds before returning failed status.

