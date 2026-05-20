from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from mas_v2.contracts.schemas import (
    AdaptationResult,
    CellOrderArtifacts,
    DatasetDescriptor,
    DatasetView,
    ModelSelectionItem,
    ReferenceAssetPackage,
    RepairRecord,
    RunProfile,
)
from mas_v2.runtime.artifacts import RunWorkspace, register_artifacts
from mas_v2.runtime.logging import StructuredRunLogger
from tools_layer.skill_tools.multi_foundation_knn_annotation_skill import _sample_reference_indices
from tools_layer.skill_tools.uce_knn_annotation_skill import _candidate_gene_name_columns


def _primary_h5ad_path(source: Any) -> str:
    for field in ("h5ad_path", "path"):
        value = str(getattr(source, field, "") or "").strip()
        if value.endswith(".h5ad"):
            return value
    raise FileNotFoundError(f"No .h5ad path available for source: {source}")


def _pick_x_source(descriptor: DatasetDescriptor, preferred: str, preferred_layer: str) -> tuple[str, str]:
    if preferred == "layers" and preferred_layer and preferred_layer in descriptor.candidate_layers:
        return "layers", preferred_layer
    if preferred in descriptor.candidate_x_sources and preferred != "layers":
        return preferred, preferred_layer
    if "X" in descriptor.candidate_x_sources:
        return "X", ""
    if descriptor.candidate_layers:
        return "layers", descriptor.candidate_layers[0]
    if "raw" in descriptor.candidate_x_sources:
        return "raw", ""
    raise ValueError(f"No usable x_source for descriptor={descriptor.role}")


def _pick_gene_key(descriptor: DatasetDescriptor, preferred: str) -> str:
    if preferred and preferred in descriptor.candidate_gene_name_keys:
        return preferred
    if descriptor.candidate_gene_name_keys:
        return descriptor.candidate_gene_name_keys[0]
    return "var_names"


def _subset_query_indices(n_obs: int, cap: int, seed: int) -> np.ndarray:
    indices = np.arange(n_obs, dtype=np.int64)
    if cap <= 0 or n_obs <= cap:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=cap, replace=False).astype(np.int64))


def _write_obs_names(obs: pd.DataFrame, path: Path) -> str:
    export = pd.DataFrame({"obs_name": obs.index.astype(str)})
    export.to_csv(path, index=False)
    return str(path)


def _candidate_gene_overlap(adata: ad.AnnData, candidate_keys: list[str], preferred_key: str) -> tuple[str, int, list[dict[str, Any]]]:
    var_frame = adata.raw.var.copy() if adata.raw is not None else adata.var.copy()
    candidate_scores: list[dict[str, Any]] = []
    best_key = "var_names"
    best_overlap = 0
    for key in ["var_names", *candidate_keys]:
        if key == "var_names":
            values = [str(item) for item in var_frame.index.tolist()]
        elif key in var_frame.columns:
            values = [str(item) for item in var_frame[key].tolist()]
        else:
            continue
        non_blank = [value for value in values if value and not value.startswith("Blank")]
        overlap = len(non_blank)
        candidate_scores.append({"source_key": key, "non_blank_genes": overlap, "n_genes_total": len(values)})
        if overlap > best_overlap or (overlap == best_overlap and key == preferred_key):
            best_key = key
            best_overlap = overlap
    return best_key, best_overlap, candidate_scores


def _write_subset(source_path: str, row_indices: np.ndarray, output_path: Path) -> tuple[str, pd.DataFrame, int]:
    adata = ad.read_h5ad(source_path, backed="r")
    try:
        subset = adata[row_indices].to_memory()
        subset.write_h5ad(output_path)
        obs = subset.obs.copy()
        return str(output_path), obs, int(subset.n_vars if subset.raw is None else subset.raw.n_vars)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


class AdapterAgent:
    def adapt(
        self,
        *,
        model_item: ModelSelectionItem,
        profile: RunProfile,
        intake_bundle: Any,
        workspace: RunWorkspace,
        logger: StructuredRunLogger,
        feedback: list[str] | None = None,
        attempt: int = 1,
        reference_source_override: Any | None = None,
    ) -> AdaptationResult:
        model_id = model_item.model_id
        model_dir = workspace.executor_dir / model_id / "artifacts"
        adapted_dir = model_dir / "adapted"
        manifest_dir = model_dir / "manifests"
        for path in (adapted_dir, manifest_dir):
            path.mkdir(parents=True, exist_ok=True)

        reference_source = reference_source_override or profile.input.reference_source
        query_source = profile.input.query_source
        repair_history: list[RepairRecord] = []

        try:
            query_pref_gene_key = getattr(query_source, "preferred_gene_name_key", "")
            query_pref_x = getattr(query_source, "preferred_x_source", "X")
            query_pref_layer = getattr(query_source, "preferred_layer_name", "")
            query_x_source, query_layer_name = _pick_x_source(intake_bundle.query, query_pref_x, query_pref_layer)
            query_gene_key = _pick_gene_key(intake_bundle.query, query_pref_gene_key)

            query_h5ad = _primary_h5ad_path(query_source)
            query_adata = ad.read_h5ad(query_h5ad, backed="r")
            try:
                query_indices = _subset_query_indices(
                    query_adata.n_obs,
                    profile.executor.sampling.max_query_cells,
                    profile.executor.sampling.random_seed + attempt,
                )
            finally:
                if getattr(query_adata, "file", None) is not None:
                    query_adata.file.close()

            query_subset_path = adapted_dir / f"{model_id}__query_subset.h5ad"
            query_dataset_path, query_obs, query_n_genes = _write_subset(query_h5ad, query_indices, query_subset_path)
            query_obs_path = adapted_dir / f"{model_id}__query_obs.csv"
            query_obs.to_csv(query_obs_path)
            query_names_path = adapted_dir / f"{model_id}__query_obs_names.csv"
            _write_obs_names(query_obs, query_names_path)
            query_view = DatasetView(
                dataset_path=query_dataset_path,
                obs_csv=str(query_obs_path),
                n_cells=int(query_obs.shape[0]),
                n_genes=query_n_genes,
                gene_name_source=query_gene_key,
                x_source=query_x_source,
                layer_name=query_layer_name,
                species=intake_bundle.query.species_hint or getattr(query_source, "species", ""),
            )

            if isinstance(reference_source, ReferenceAssetPackage):
                ref_obs = pd.read_csv(reference_source.reference_obs_path, index_col=0)
                reference_view = DatasetView(
                    dataset_path=str(Path(reference_source.source_manifest).resolve()),
                    obs_csv=str(Path(reference_source.reference_obs_path).resolve()),
                    n_cells=int(ref_obs.shape[0]),
                    n_genes=intake_bundle.reference.n_vars,
                    gene_name_source="asset_package",
                    x_source="asset_package",
                    species=intake_bundle.reference.species_hint or reference_source.metadata.get("reference_species", ""),
                )
                reference_names_path = adapted_dir / f"{model_id}__reference_obs_names.csv"
                _write_obs_names(ref_obs, reference_names_path)
                reference_obs_path = Path(reference_source.reference_obs_path).resolve()
                reference_dataset_path = str(Path(reference_source.source_manifest).resolve())
                reference_gene_overlap = int(json.loads(Path(reference_source.coverage_json).read_text(encoding="utf-8")).get("reference_records", [{}])[0].get("coverage", {}).get("n_mapped", 0))
                reference_candidate_scores = [{"source_key": "asset_package", "non_blank_genes": reference_gene_overlap, "n_genes_total": intake_bundle.reference.n_vars or 0}]
            else:
                ref_pref_gene_key = getattr(reference_source, "preferred_gene_name_key", "")
                ref_pref_x = getattr(reference_source, "preferred_x_source", "X")
                ref_pref_layer = getattr(reference_source, "preferred_layer_name", "")
                reference_x_source, reference_layer_name = _pick_x_source(intake_bundle.reference, ref_pref_x, ref_pref_layer)
                reference_gene_key = _pick_gene_key(intake_bundle.reference, ref_pref_gene_key)
                reference_h5ad = _primary_h5ad_path(reference_source)
                ref_adata = ad.read_h5ad(reference_h5ad, backed="r")
                try:
                    ref_obs_all = ref_adata.obs.copy()
                    if profile.executor.reference_label_key not in ref_obs_all.columns:
                        raise KeyError(f"reference_label_key {profile.executor.reference_label_key!r} not found in reference obs")
                    reference_indices = _sample_reference_indices(
                        ref_obs_all,
                        profile.executor.reference_label_key,
                        max_total=profile.executor.sampling.max_reference_cells,
                        max_per_label=profile.executor.sampling.max_reference_cells_per_label,
                        seed=profile.executor.sampling.random_seed + attempt,
                    )
                    candidate_keys = _candidate_gene_name_columns(ref_adata.raw.var.copy() if ref_adata.raw is not None else ref_adata.var.copy(), reference_gene_key)
                    reference_gene_key, reference_gene_overlap, reference_candidate_scores = _candidate_gene_overlap(ref_adata, candidate_keys, reference_gene_key)
                finally:
                    if getattr(ref_adata, "file", None) is not None:
                        ref_adata.file.close()

                reference_subset_path = adapted_dir / f"{model_id}__reference_subset.h5ad"
                reference_dataset_path, ref_obs, reference_n_genes = _write_subset(reference_h5ad, reference_indices, reference_subset_path)
                ref_obs_path = adapted_dir / f"{model_id}__reference_obs.csv"
                ref_obs.to_csv(ref_obs_path)
                reference_names_path = adapted_dir / f"{model_id}__reference_obs_names.csv"
                _write_obs_names(ref_obs, reference_names_path)
                reference_obs_path = ref_obs_path
                reference_view = DatasetView(
                    dataset_path=reference_dataset_path,
                    obs_csv=str(ref_obs_path),
                    n_cells=int(ref_obs.shape[0]),
                    n_genes=reference_n_genes,
                    gene_name_source=reference_gene_key,
                    x_source=reference_x_source,
                    layer_name=reference_layer_name,
                    species=intake_bundle.reference.species_hint or getattr(reference_source, "species", ""),
                )

            query_candidate_scores = [{"source_key": query_gene_key, "non_blank_genes": intake_bundle.query.n_vars or 0, "n_genes_total": intake_bundle.query.n_vars or 0}]
            query_gene_overlap = intake_bundle.query.n_vars or 0
            coverage_metrics = {
                "reference_gene_overlap_count": int(reference_gene_overlap),
                "query_gene_overlap_count": int(query_gene_overlap),
                "reference_candidate_scores": reference_candidate_scores,
                "query_candidate_scores": query_candidate_scores,
                "feedback": list(feedback or []),
            }
            sampling_manifest = {
                "attempt": attempt,
                "random_seed": profile.executor.sampling.random_seed,
                "max_reference_cells": profile.executor.sampling.max_reference_cells,
                "max_reference_cells_per_label": profile.executor.sampling.max_reference_cells_per_label,
                "max_query_cells": profile.executor.sampling.max_query_cells,
                "feedback": list(feedback or []),
            }
            manifest_path = manifest_dir / f"{model_id}__adaptation_manifest.json"
            manifest_path.write_text(json.dumps(sampling_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

            result = AdaptationResult(
                status="success",
                model_id=model_id,
                output_dir=str((workspace.executor_dir / model_id).resolve()),
                reference_view=reference_view,
                query_view=query_view,
                sampling_manifest=sampling_manifest,
                cell_order_artifacts=CellOrderArtifacts(
                    reference_obs_names_path=str(reference_names_path),
                    query_obs_names_path=str(query_names_path),
                    reference_obs_csv=str(reference_obs_path),
                    query_obs_csv=str(query_obs_path),
                ),
                coverage_metrics=coverage_metrics,
                repair_history=repair_history,
                artifacts={
                    "adaptation_manifest_json": str(manifest_path),
                    "reference_view_path": str(reference_view.dataset_path),
                    "query_view_path": str(query_view.dataset_path),
                },
            )
            output_path = manifest_dir / f"{model_id}__adaptation_result.json"
            output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            result.artifacts["adaptation_result_json"] = str(output_path)
            register_artifacts(workspace.registry_path, f"adapter.{model_id}", result.artifacts)
            logger.event("adapter_agent.completed", payload={"model_id": model_id, "attempt": attempt})
            return result
        except Exception as exc:  # noqa: BLE001
            if feedback:
                repair_history.append(RepairRecord(attempt=attempt, action="repair_feedback", reason="; ".join(feedback), status="failed"))
            logger.error("adapter_agent.failed", exc, payload={"model_id": model_id, "attempt": attempt})
            return AdaptationResult(
                status="failed",
                model_id=model_id,
                output_dir=str((workspace.executor_dir / model_id).resolve()),
                repair_history=repair_history,
                error=f"{type(exc).__name__}: {exc}",
            )


class AdapterJudge:
    def validate(self, result: AdaptationResult, profile: RunProfile) -> list[str]:
        issues: list[str] = []
        if result.status != "success":
            issues.append(result.error or "adapter returned failure")
            return issues
        if not result.reference_view.obs_csv:
            issues.append("missing reference obs csv")
        if not result.query_view.obs_csv:
            issues.append("missing query obs csv")
        if result.reference_view.n_cells is not None and result.reference_view.n_cells <= 0:
            issues.append("empty reference view")
        if result.query_view.n_cells is not None and result.query_view.n_cells <= 0:
            issues.append("empty query view")
        if int(result.coverage_metrics.get("reference_gene_overlap_count", 0)) <= 0:
            issues.append("zero reference gene overlap")
        if int(result.coverage_metrics.get("query_gene_overlap_count", 0)) <= 0:
            issues.append("zero query gene overlap")
        if profile.input.reference_source.source_type != "reference_asset_package" and not Path(result.reference_view.dataset_path).exists():
            issues.append("missing reference view dataset file")
        if not Path(result.query_view.dataset_path).exists():
            issues.append("missing query view dataset file")
        if result.reference_view.obs_csv and not Path(result.reference_view.obs_csv).exists():
            issues.append("missing reference obs csv file")
        if result.query_view.obs_csv and not Path(result.query_view.obs_csv).exists():
            issues.append("missing query obs csv file")
        return issues
