# Adapter Guidance: geneformer

## Goal
Produce `AdaptationResult` for `model_id=geneformer` from first-seen input data.

## Input Preference
- Prefer `h5ad` source.
- If `raw_dataset` has `npz+h5ad` sidecar, use h5ad obs/var and load matrix from npz if needed.
- For `mtx_triplet`, build temporary h5ad-like view and persist it under `artifacts/adapted/`.

## Required Adapter Decisions
- Select `selected_x_source` from `X/layers/raw`.
- If `layers` is selected, write `selected_layer_name`.
- Pick reference/query gene-name key (`var` column) or fallback to `var_names`.
- Emit deterministic cell-order files for both reference and query.
- Emit sampling manifest and coverage metrics.

## Model-Specific Hints
- Keep human-readable gene symbols when possible.
- Prefer overlap-stable gene naming between reference/query to maximize coverage.
- Record any forced gene filtering in repair history.

## Hard Output Checklist
- `reference_view.dataset_path` and `query_view.dataset_path`
- `cell_order_artifacts.reference_obs_names_path`
- `cell_order_artifacts.query_obs_names_path`
- `sampling_manifest`
- `coverage_metrics`
- `repair_history`
- `status` and `error` (error required when failed)

