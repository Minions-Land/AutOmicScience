from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from mas_v2.contracts.schemas import AdaptationResult, ReferenceAssetPackage
from mas_v2.runtime.artifacts import register_artifacts
from tools_layer.skill_tools.multi_foundation_knn_annotation_skill import (
    _adata_matrix_slice_fixed_panel,
    _build_encoder,
    _select_fixed_panel_gene_names,
    _select_genes_for_model,
)
from tools_layer.skill_tools.uce_knn_annotation_skill import _adata_matrix_slice, _resolve_device, _write_json


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix / norms).astype(np.float32)


def load_reference_asset_package(source: Any) -> ReferenceAssetPackage | None:
    if getattr(source, "source_type", "") != "reference_asset_package":
        return None
    if isinstance(source, ReferenceAssetPackage):
        return source
    return ReferenceAssetPackage.model_validate(source)


def compute_query_embeddings(
    *,
    model_id: str,
    query_view_path: str,
    query_species: str,
    gene_name_key: str,
    x_source: str,
    layer_name: str,
    batch_size: int,
    device: str,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    query_adata = ad.read_h5ad(query_view_path, backed="r")
    try:
        normalized_model_id = model_id.strip().lower()
        if normalized_model_id == "scgpt_generic":
            query_gene_selection = _select_genes_for_model(
                query_adata,
                model_id=model_id,
                species=query_species,
                preferred_key=gene_name_key,
                x_source=x_source,
                layer_name=layer_name,
            )
            panel_gene_names = list(query_gene_selection.gene_names)
            fixed_panel = _select_fixed_panel_gene_names(
                query_adata,
                preferred_key=gene_name_key,
                x_source=x_source,
                layer_name=layer_name,
                panel_gene_names=panel_gene_names,
            )
            query_matrix = _adata_matrix_slice_fixed_panel(
                query_adata,
                np.arange(query_adata.n_obs, dtype=np.int64),
                x_source=x_source,
                layer_name=layer_name,
                source_column_indices=fixed_panel.source_column_indices,
                panel_output_positions=fixed_panel.panel_output_positions,
                panel_size=len(panel_gene_names),
            )
            encoder = _build_encoder(
                model_id,
                gene_names=panel_gene_names,
                species=query_species,
                device=device,
                random_seed=random_seed,
            )
            coverage_payload = {
                "gene_name_source": fixed_panel.source_key,
                "gene_overlap_count": int(fixed_panel.overlap_count),
                "candidate_scores": fixed_panel.candidate_scores,
                "coverage": {
                    "n_mapped": int(fixed_panel.overlap_count),
                    "n_total": int(len(panel_gene_names)),
                    "coverage_ratio": float(fixed_panel.overlap_count / max(1, len(panel_gene_names))),
                },
                "panel_gene_names": panel_gene_names,
            }
        else:
            selection = _select_genes_for_model(
                query_adata,
                model_id=model_id,
                species=query_species,
                preferred_key=gene_name_key,
                x_source=x_source,
                layer_name=layer_name,
            )
            query_matrix = _adata_matrix_slice(
                query_adata,
                np.arange(query_adata.n_obs, dtype=np.int64),
                x_source=x_source,
                layer_name=layer_name,
                gene_keep_mask=selection.gene_keep_mask,
            )
            encoder = _build_encoder(
                model_id,
                gene_names=selection.gene_names,
                species=query_species,
                device=device,
                random_seed=random_seed,
            )
            coverage_payload = {
                "gene_name_source": selection.source_key,
                "gene_overlap_count": int(selection.overlap_count),
                "candidate_scores": selection.candidate_scores,
                "coverage": {
                    "n_mapped": int(getattr(encoder.coverage_info, "n_mapped")),
                    "n_total": int(getattr(encoder.coverage_info, "n_total")),
                    "coverage_ratio": float(getattr(encoder.coverage_info, "coverage_ratio")),
                },
            }
        embeddings = row_normalize(encoder.encode(query_matrix, batch_size))
        return embeddings, coverage_payload
    finally:
        if getattr(query_adata, "file", None) is not None:
            query_adata.file.close()


def compute_reference_embeddings(
    *,
    model_id: str,
    reference_view_path: str,
    reference_species: str,
    gene_name_key: str,
    x_source: str,
    layer_name: str,
    batch_size: int,
    device: str,
    random_seed: int,
    scgpt_generic_panel_gene_names: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    reference_adata = ad.read_h5ad(reference_view_path, backed="r")
    try:
        normalized_model_id = model_id.strip().lower()
        if normalized_model_id == "scgpt_generic":
            panel_gene_names = list(scgpt_generic_panel_gene_names or [])
            fixed_panel = _select_fixed_panel_gene_names(
                reference_adata,
                preferred_key=gene_name_key,
                x_source=x_source,
                layer_name=layer_name,
                panel_gene_names=panel_gene_names,
            )
            reference_matrix = _adata_matrix_slice_fixed_panel(
                reference_adata,
                np.arange(reference_adata.n_obs, dtype=np.int64),
                x_source=x_source,
                layer_name=layer_name,
                source_column_indices=fixed_panel.source_column_indices,
                panel_output_positions=fixed_panel.panel_output_positions,
                panel_size=len(panel_gene_names),
            )
            encoder = _build_encoder(
                model_id,
                gene_names=panel_gene_names,
                species=reference_species,
                device=device,
                random_seed=random_seed,
            )
            coverage_payload = {
                "reference_path": str(Path(reference_view_path).resolve()),
                "n_reference_cells": int(reference_adata.n_obs),
                "gene_name_source": fixed_panel.source_key,
                "gene_overlap_count": int(fixed_panel.overlap_count),
                "candidate_scores": fixed_panel.candidate_scores,
                "coverage": {
                    "n_mapped": int(fixed_panel.overlap_count),
                    "n_total": int(len(panel_gene_names)),
                    "coverage_ratio": float(fixed_panel.overlap_count / max(1, len(panel_gene_names))),
                },
            }
        else:
            selection = _select_genes_for_model(
                reference_adata,
                model_id=model_id,
                species=reference_species,
                preferred_key=gene_name_key,
                x_source=x_source,
                layer_name=layer_name,
            )
            reference_matrix = _adata_matrix_slice(
                reference_adata,
                np.arange(reference_adata.n_obs, dtype=np.int64),
                x_source=x_source,
                layer_name=layer_name,
                gene_keep_mask=selection.gene_keep_mask,
            )
            encoder = _build_encoder(
                model_id,
                gene_names=selection.gene_names,
                species=reference_species,
                device=device,
                random_seed=random_seed,
            )
            coverage_payload = {
                "reference_path": str(Path(reference_view_path).resolve()),
                "n_reference_cells": int(reference_adata.n_obs),
                "gene_name_source": selection.source_key,
                "gene_overlap_count": int(selection.overlap_count),
                "candidate_scores": selection.candidate_scores,
                "coverage": {
                    "n_mapped": int(getattr(encoder.coverage_info, "n_mapped")),
                    "n_total": int(getattr(encoder.coverage_info, "n_total")),
                    "coverage_ratio": float(getattr(encoder.coverage_info, "coverage_ratio")),
                },
            }
        embeddings = row_normalize(encoder.encode(reference_matrix, batch_size))
        return embeddings, coverage_payload
    finally:
        if getattr(reference_adata, "file", None) is not None:
            reference_adata.file.close()


def finalize_embedding_package(
    *,
    output_dir: Path,
    model_id: str,
    run_id: str,
    reference_species: str,
    query_species: str,
    reference_obs_path: str,
    query_obs_path: str,
    reference_label_key: str,
    query_label_key: str,
    reference_embeddings_path: str,
    query_embeddings_path: str,
    coverage_payload: dict[str, Any],
    registry_path: str | Path,
) -> dict[str, Any]:
    manifest_dir = output_dir / "artifacts" / "manifests"
    coverage_dir = output_dir / "artifacts" / "coverage"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    coverage_json = coverage_dir / f"{model_id}__coverage.json"
    _write_json(coverage_json, coverage_payload)
    input_config = manifest_dir / "input_config.json"
    _write_json(
        input_config,
        {
            "model_id": model_id,
            "run_id": run_id,
            "reference_label_key": reference_label_key,
            "query_label_key": query_label_key,
            "reference_species": reference_species,
            "query_species": query_species,
        },
    )
    artifacts = {
        "reference_embeddings_npy": str(reference_embeddings_path),
        "query_embeddings_npy": str(query_embeddings_path),
        "reference_obs_csv": str(reference_obs_path),
        "query_obs_csv": str(query_obs_path),
        "coverage_json": str(coverage_json),
        "input_config_json": str(input_config),
    }
    registry = register_artifacts(registry_path, f"embedding.{model_id}", artifacts)
    return {"artifacts": artifacts, "artifact_registry_path": registry}
