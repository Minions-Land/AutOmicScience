# Adapter Guidance: nicheformer

## Goal
Produce `AdaptationResult` for `model_id=nicheformer` with explicit spatial metadata exposure.

## Input Preference
- Prefer `h5ad` with spatial coordinates in `obsm` or coordinate columns in `obs`.
- Accept `prepared_dataset` directly when reference has already been panel-aligned.
- Accept `reference_asset_package` mode for benchmark fallback.

## Required Adapter Decisions
- Select expression source (`X/layers/raw`) and persist selected layer name.
- Extract spatial keys (for example `spatial`, `X_spatial`, or coordinate columns) into manifest metadata.
- Pick gene name keys and record fallback behavior.
- Produce stable cell-order artifacts and sampling manifest.
- Compute gene overlap / coverage stats and warnings.

## Model-Specific Hints
- Favor preserving neighborhood-relevant metadata in adapted `obs_csv`.
- If spatial keys are missing, mark warning and keep pipeline runnable.
- Keep adaptation mode explicit (`subset`, `fixed_panel`, `vocab_overlap`, or `passthrough`).

## Hard Output Checklist
- `reference_view` + `query_view`
- `selected_x_source` + `selected_layer_name`
- `selected_reference_gene_name_key` + `selected_query_gene_name_key`
- `sampling_manifest`
- `cell_order_artifacts`
- `coverage_metrics`
- `repair_history`

