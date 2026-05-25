# Adapter Guidance: scgpt_generic

## Goal
Produce `AdaptationResult` for `model_id=scgpt_generic` without using the legacy `scgpt` path.

## Input Preference
- `prepared_dataset` is preferred.
- `raw_dataset` is allowed when gene-name keys and expression source can be resolved.
- `reference_asset_package` is allowed for benchmark mode.

## Required Adapter Decisions
- Resolve a valid gene-name key for both sides; fallback to `var_names` if needed.
- Choose matrix source (`X/layers/raw`) and record chosen layer.
- Create adapted reference/query views with deterministic row order.
- Emit sampling manifest and overlap coverage.
- Include repair loop outputs when adapter_judge requests fixes.

## Model-Specific Hints
- Keep adapter outputs compact and avoid model-irrelevant columns.
- If no valid overlap genes are found, return failed status with explicit reason.
- Preserve label-related fields (`Level0/Level1/Level1_5/Level2/Level3`) in query obs when available.

## Hard Output Checklist
- `status`
- `reference_view.dataset_path`
- `query_view.dataset_path`
- `cell_order_artifacts.reference_obs_names_path`
- `cell_order_artifacts.query_obs_names_path`
- `coverage_metrics`
- `error` when failed

