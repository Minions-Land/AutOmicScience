from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
from pandas.api.types import is_numeric_dtype

from mas_v2.contracts.schemas import (
    DatasetDescriptor,
    DatasetFingerprint,
    DatasetIntakeBundle,
    PreparedDatasetSource,
    RawDatasetSource,
    ReferenceAssetPackage,
    RunProfile,
)
from mas_v2.runtime.artifacts import RunWorkspace, register_artifacts
from mas_v2.runtime.logging import StructuredRunLogger
from tools_layer.skill_tools.uce_knn_annotation_skill import _candidate_gene_name_columns


LABEL_HINTS = ("label", "type", "cell", "class", "subclass", "supertype", "level", "cluster", "batch", "sample")
SPATIAL_HINTS = ("spatial", "x_centroid", "y_centroid", "xy", "coord")


def _resolved_paths_for_source(source: RawDatasetSource | PreparedDatasetSource) -> list[str]:
    paths = [
        getattr(source, "path", ""),
        getattr(source, "h5ad_path", ""),
        getattr(source, "npz_path", ""),
        getattr(source, "mtx_path", ""),
        getattr(source, "obs_path", ""),
        getattr(source, "var_path", ""),
    ]
    return [str(Path(path).resolve()) for path in paths if path]


def _infer_file_types(paths: list[str]) -> list[str]:
    types: list[str] = []
    for item in paths:
        suffix = "".join(Path(item).suffixes) or Path(item).suffix
        if suffix:
            types.append(suffix.lstrip("."))
    return sorted(set(types))


def _fingerprint(paths: list[str]) -> DatasetFingerprint:
    hasher = hashlib.sha256()
    resolved: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        resolved.append(str(path))
        if path.exists():
            stat = path.stat()
            payload = f"{path}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8")
        else:
            payload = f"{path}|missing".encode("utf-8")
        hasher.update(payload)
    return DatasetFingerprint(value=hasher.hexdigest(), source_paths=resolved)


def _guess_species(source_metadata: dict[str, Any], obs: pd.DataFrame, fallback: str) -> str:
    for key in ("species", "organism", "reference_species", "query_species"):
        value = str(source_metadata.get(key, "")).strip()
        if value:
            return value
    for key in ("species", "organism"):
        if key in obs.columns:
            series = obs[key].astype(str)
            if not series.empty:
                return str(series.iloc[0])
    return fallback


def _candidate_label_keys(obs: pd.DataFrame) -> list[str]:
    ordered: list[str] = []
    for column in obs.columns:
        lower = str(column).strip().lower()
        if any(hint in lower for hint in LABEL_HINTS):
            ordered.append(str(column))
            continue
        if not is_numeric_dtype(obs[column]):
            ordered.append(str(column))
    return list(dict.fromkeys(ordered))


def _candidate_spatial_keys(obs: pd.DataFrame, obsm_keys: list[str]) -> list[str]:
    found = [key for key in obsm_keys if any(hint in key.lower() for hint in SPATIAL_HINTS)]
    for column in obs.columns:
        if any(hint in str(column).lower() for hint in SPATIAL_HINTS):
            found.append(str(column))
    return list(dict.fromkeys(found))


def _estimate_sparsity(adata: ad.AnnData) -> float | None:
    if adata.n_obs == 0 or adata.n_vars == 0:
        return None
    sample_rows = min(int(adata.n_obs), 256)
    matrix = adata.X[:sample_rows]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return None
    return float(1.0 - (np.count_nonzero(matrix) / matrix.size))


def _inspect_h5ad(role: str, path: Path, *, species_hint: str, metadata: dict[str, Any]) -> DatasetDescriptor:
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs
        var = adata.raw.var.copy() if adata.raw is not None else adata.var.copy()
        var_keys = list(map(str, var.columns))
        gene_keys = ["var_names", *_candidate_gene_name_columns(var, "")]
        return DatasetDescriptor(
            role=role,
            source_type="h5ad",
            resolved_paths=[str(path)],
            file_types=_infer_file_types([str(path)]),
            n_obs=int(adata.n_obs),
            n_vars=int(adata.n_vars if adata.raw is None else adata.raw.n_vars),
            obs_keys=list(map(str, obs.columns)),
            var_keys=var_keys,
            layers=list(map(str, adata.layers.keys())),
            obsm_keys=list(map(str, adata.obsm.keys())),
            candidate_label_keys=_candidate_label_keys(obs),
            candidate_gene_name_keys=list(dict.fromkeys(gene_keys)),
            candidate_x_sources=["X", *(["layers"] if len(adata.layers.keys()) else []), *(["raw"] if adata.raw is not None else [])],
            candidate_layers=list(map(str, adata.layers.keys())),
            spatial_keys=_candidate_spatial_keys(obs, list(map(str, adata.obsm.keys()))),
            species_hint=_guess_species(metadata, obs, species_hint),
            panel_size=int(adata.n_vars),
            matrix_sparsity=_estimate_sparsity(adata),
            warnings=[],
            metadata=dict(metadata),
        )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def _inspect_sidecar(role: str, source: RawDatasetSource | PreparedDatasetSource) -> DatasetDescriptor:
    paths = _resolved_paths_for_source(source)
    h5ad_path = source.h5ad_path or source.path
    if h5ad_path and str(h5ad_path).endswith(".h5ad"):
        return _inspect_h5ad(role, Path(h5ad_path).resolve(), species_hint=source.species, metadata=source.metadata)
    if getattr(source, "mtx_path", "") and getattr(source, "obs_path", "") and getattr(source, "var_path", ""):
        obs = pd.read_csv(str(getattr(source, "obs_path")))
        var = pd.read_csv(str(getattr(source, "var_path")))
        shape = scipy.io.mmread(str(getattr(source, "mtx_path"))).shape
        return DatasetDescriptor(
            role=role,
            source_type="mtx_triplet",
            resolved_paths=paths,
            file_types=_infer_file_types(paths),
            n_obs=int(shape[0]),
            n_vars=int(shape[1]),
            obs_keys=list(map(str, obs.columns)),
            var_keys=list(map(str, var.columns)),
            candidate_label_keys=_candidate_label_keys(obs),
            candidate_gene_name_keys=["var_names", *_candidate_gene_name_columns(var, "")],
            candidate_x_sources=["X"],
            species_hint=source.species,
            panel_size=int(shape[1]),
            warnings=[],
            metadata=dict(source.metadata),
        )
    if getattr(source, "npz_path", "") and h5ad_path:
        return _inspect_h5ad(role, Path(h5ad_path).resolve(), species_hint=source.species, metadata=source.metadata)
    raise FileNotFoundError(f"Unable to inspect dataset source for role={role}: {source.model_dump()}")


def _inspect_asset_package(role: str, source: ReferenceAssetPackage) -> DatasetDescriptor:
    obs = pd.read_csv(source.reference_obs_path, index_col=0)
    coverage = json.loads(Path(source.coverage_json).read_text(encoding="utf-8"))
    panel_size = None
    query_record = coverage.get("query_record", {})
    reference_records = coverage.get("reference_records", [])
    if reference_records:
        panel_size = int(reference_records[0].get("coverage", {}).get("n_total", 0)) or None
    if panel_size is None:
        panel_size = int(query_record.get("coverage", {}).get("n_total", 0)) or None
    return DatasetDescriptor(
        role=role,
        source_type="reference_asset_package",
        resolved_paths=[
            str(Path(source.reference_embeddings_path).resolve()),
            str(Path(source.reference_obs_path).resolve()),
            str(Path(source.coverage_json).resolve()),
            str(Path(source.source_manifest).resolve()),
        ],
        file_types=_infer_file_types(
            [
                source.reference_embeddings_path,
                source.reference_obs_path,
                source.coverage_json,
                source.source_manifest,
            ]
        ),
        n_obs=int(obs.shape[0]),
        n_vars=panel_size,
        obs_keys=list(map(str, obs.columns)),
        candidate_label_keys=_candidate_label_keys(obs),
        candidate_gene_name_keys=[],
        candidate_x_sources=[],
        candidate_layers=[],
        spatial_keys=[],
        species_hint=str(source.metadata.get("reference_species", "")),
        panel_size=panel_size,
        warnings=[],
        metadata=dict(source.metadata),
    )


class InputAgent:
    def run(
        self,
        profile: RunProfile,
        workspace: RunWorkspace,
        logger: StructuredRunLogger,
    ) -> DatasetIntakeBundle:
        with logger.span("input_agent.run", payload={"dataset_id": profile.input.dataset_id}):
            reference_source = profile.input.reference_source
            query_source = profile.input.query_source

            if isinstance(reference_source, ReferenceAssetPackage):
                reference = _inspect_asset_package("reference", reference_source)
                reference_paths = reference.resolved_paths
            else:
                reference = _inspect_sidecar("reference", reference_source)
                reference_paths = reference.resolved_paths

            if isinstance(query_source, ReferenceAssetPackage):
                query = _inspect_asset_package("query", query_source)
                query_paths = query.resolved_paths
            else:
                query = _inspect_sidecar("query", query_source)
                query_paths = query.resolved_paths

            fingerprint = _fingerprint([*reference_paths, *query_paths])
            warnings = [*reference.warnings, *query.warnings]
            scdesign3_context = {
                "synthetic_origin": profile.input.query_source.metadata.get("synthetic_origin", ""),
                "conditioning_keys": list(profile.input.query_source.metadata.get("conditioning_keys", [])),
                "batch_keys": list(profile.input.query_source.metadata.get("batch_keys", [])),
                "anchor_summary_path": str(profile.input.query_source.metadata.get("anchor_summary_path", "")),
                "generator_name": str(profile.input.query_source.metadata.get("generator_name", "")),
            }
            bundle = DatasetIntakeBundle(
                dataset_id=profile.input.dataset_id,
                task_request=profile.input.task_request,
                dataset_description=profile.input.dataset_description,
                reference=reference,
                query=query,
                fingerprint=fingerprint,
                sampling_plan=profile.executor.sampling,
                scdesign3_context=scdesign3_context,
                warnings=warnings,
            )
            output_path = workspace.input_dir / "intake_bundle.json"
            output_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
            bundle.artifacts["intake_bundle_json"] = str(output_path)
            register_artifacts(workspace.registry_path, "input", {"intake_bundle_json": str(output_path)})
            logger.event("input_agent.completed", payload={"output_path": str(output_path), "query_n_obs": query.n_obs})
            return bundle
