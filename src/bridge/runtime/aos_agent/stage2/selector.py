from __future__ import annotations

import csv
import json
import os
import pickle
import re
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

from aos_agent import paths
from aos_agent.eval.label_transfer import (
    EmbeddingAdapter,
    MatrixBundle,
    _align_reference_and_query,
    _load_reference_from_standard_bundle,
    _prototype_predict,
    _split_method,
    _vote_knn,
)
from aos_agent.io import ensure_dir, normalize_gene_name, read_json, read_standard_bundle, stratified_indices, write_json
from aos_agent.llm_config import build_openai_client, default_llm_model


DEFAULT_ARTIFACT_BUNDLE = paths.AOS_ROOT / "artifacts" / "stage1_full"
DEFAULT_PREPARED_SOURCE_ROOT = paths.DATA_DIR / "prepared_sources"
QUERY_PANEL_PREPARED_SOURCE_ROOT = paths.DATA_DIR / "prepared_sources_query_panel"
DEFAULT_CAPABILITY_DIR = paths.AOS_ROOT / "configs" / "capability"
DEFAULT_STAGE2_ROOT = paths.RUNS_DIR / "stage2_selection"
DEFAULT_LLM_MODEL = default_llm_model()
DEFAULT_EXCLUDED_MODEL_IDS = {"expression_log1p_knn", "expression_log1p_prototype"}
DEFAULT_ENV_PATH = paths.AOS_ROOT / ".env"
DEFAULT_SELECTION_STRATEGY = "one_by_one"
DEFAULT_SELECTION_OBJECTIVE = "unified_rank"
LEGACY_SELECTION_OBJECTIVES = {"consensus", "best_single_ablation"}
DEFAULT_ITERATIVE_EXCLUDE_SCOPE = "family"
DEFAULT_NEAR_TIE_MARGIN = 0.01
RANK_AGGREGATION_METHOD = "evidence_group_rank_v1"
RANK_AGGREGATION_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group": "query_source_gene_fit",
        "description": (
            "query-visible gene-name overlap between the test panel and candidate reference/source; "
            "uses no query labels, expression values, path, or dataset id"
        ),
        "axes": (
            ("source_similarity", "gene-only source similarity score"),
            ("query_gene_coverage", "fraction of query genes represented in the reference/source panel"),
            ("min_panel_coverage", "coverage of the smaller gene panel"),
            ("gene_set_jaccard", "gene-set specificity between query and reference/source"),
        ),
    },
    {
        "group": "stage1_annotation_ability",
        "description": (
            "source-disaggregated Stage-1 no-training annotation evidence; includes a one-standard-error "
            "lower bound so unstable synthetic benchmarks do not dominate on raw mean alone"
        ),
        "axes": (
            ("source_model_macro_f1_lcb", "source benchmark macro-F1 lower confidence proxy: mean minus one standard error"),
            ("source_model_macro_f1", "mean source benchmark macro-F1"),
            ("baseline_macro_f1_mean", "unperturbed Stage-1 source benchmark anchor"),
        ),
    },
    {
        "group": "synthetic_variant_robustness",
        "description": "synthetic-variant stability evidence from Stage-1 scDesign3 perturbations",
        "axes": (
            ("robustness", "mean clipped variant/baseline score ratio"),
            ("robustness_median", "median clipped variant/baseline score ratio"),
            ("robustness_q25", "lower-quartile clipped variant/baseline score ratio"),
        ),
    },
    {
        "group": "benchmark_provenance",
        "description": "benchmark evidence provenance: exact source benchmark > self-holdout > model fallback",
        "axes": (
            ("benchmark_evidence_reliability", "ordinal provenance rank"),
        ),
    },
)
RANK_AGGREGATION_AXES: tuple[tuple[str, str], ...] = tuple(
    axis for group in RANK_AGGREGATION_GROUPS for axis in group["axes"]
)
ANNOTATION_ANCHOR_TIE_AXES: tuple[tuple[str, str], ...] = (
    ("source_model_macro_f1_lcb", "risk-adjusted source-specific annotation macro-F1"),
    ("source_model_macro_f1", "mean source-specific annotation macro-F1"),
    ("baseline_macro_f1_mean", "unperturbed source benchmark macro-F1"),
    ("robustness_median", "median synthetic-variant robustness"),
)

KNOWN_QUERY_PATHS = {
    "kukanja_ms": paths.KUKANJA_MS_NPZ,
    "kukanja_eae": paths.KUKANJA_EAE_NPZ,
    "seaad_merfish_140gene_test": paths.SEAAD_TEST_H5AD,
}

NO_TRAINING_EVALUATION = "raw_label_transfer_no_training"
TRAINED_CONTRACT = "seaad_140_npz"
DEFAULT_TOP_K = 3


def _prepared_source_roots(prepared_source_root: Path = DEFAULT_PREPARED_SOURCE_ROOT) -> list[Path]:
    roots: list[Path] = []
    for root in [Path(prepared_source_root), QUERY_PANEL_PREPARED_SOURCE_ROOT]:
        if root not in roots:
            roots.append(root)
    return roots


def _resolve_prepared_source_dir(source_id: str, prepared_source_root: Path = DEFAULT_PREPARED_SOURCE_ROOT) -> Path:
    fallback = Path(prepared_source_root) / source_id
    for root in _prepared_source_roots(prepared_source_root):
        candidate = root / source_id
        if (candidate / "source_manifest.json").exists() or (
            (candidate / "counts.mtx").exists() and (candidate / "obs.csv").exists() and (candidate / "var.csv").exists()
        ):
            return candidate
    return fallback


@dataclass
class QueryLoadResult:
    bundle: MatrixBundle
    adapter: str
    native_label_column: str | None
    coarse_label_column: str | None
    sample_column: str | None


def _write_yaml(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False, allow_unicode=False, width=120)
    return path


def _read_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_query_path(dataset_id: str, input_path: str | Path | None = None) -> Path:
    if input_path:
        return Path(input_path)
    if dataset_id in KNOWN_QUERY_PATHS:
        return KNOWN_QUERY_PATHS[dataset_id]
    raise ValueError(f"Input path is required for unknown dataset_id={dataset_id!r}")


def _inverse_map(mapping: dict[str, int]) -> dict[int, str]:
    return {int(value): str(key) for key, value in mapping.items()}


def _decode_label(values: np.ndarray, mapping: dict[str, int]) -> list[str]:
    inverse = _inverse_map(mapping)
    return [inverse.get(int(value), str(value)) for value in values]


def _sample_index_random(n_cells: int, *, max_cells: int, seed: int) -> np.ndarray:
    if max_cells <= 0 or max_cells >= n_cells:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(n_cells), size=max_cells, replace=False).astype(np.int64)
    idx.sort()
    return idx


def _sample_index_from_obs(obs: pd.DataFrame, *, max_cells: int, seed: int) -> np.ndarray:
    if max_cells <= 0 or max_cells >= len(obs):
        return np.arange(len(obs), dtype=np.int64)
    for column in ("native_label", "coarse_label", "sample_id", "donor_id"):
        if column in obs.columns:
            labels = obs[column].astype(str).fillna("unknown")
            usable = ~labels.isin(["", "nan", "None", "unknown"])
            if usable.any():
                idx = stratified_indices(labels, max_cells=max_cells, min_per_group=3, seed=seed)
                if len(idx):
                    return idx
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(obs)), size=max_cells, replace=False).astype(np.int64)
    idx.sort()
    return idx


def _query_adapter_from_path(path: Path) -> str:
    if path.suffix == ".npz":
        return "npz_kukanja"
    if path.suffix == ".h5ad":
        return "h5ad"
    return path.suffix.lstrip(".") or "unknown"


def _load_query_gene_profile(path: str | Path) -> dict[str, Any]:
    """Load only query-side gene names for deployable Stage-2 selection.

    Stage 2 must not observe query labels, expression summaries, donor/sample
    composition, or path-derived biological hints.  Execution still needs the
    real path in the final plan, but the selector evidence is restricted to the
    gene panel.
    """
    path = Path(path)
    adapter = _query_adapter_from_path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        meta = data["meta"].item()
        genes = [normalize_gene_name(x) for x in meta["gene_names"]]
        return {"query_adapter": adapter, "genes": genes, "n_vars": len(genes), "n_obs_profiled": 0}
    if path.suffix == ".h5ad":
        backed = ad.read_h5ad(path, backed="r")
        try:
            genes = _feature_symbols_from_adata(backed)
            return {
                "query_adapter": adapter,
                "genes": genes,
                "n_vars": int(backed.shape[1]),
                "n_obs_profiled": 0,
            }
        finally:
            if getattr(backed, "file", None) is not None:
                backed.file.close()
    raise ValueError(f"Unsupported query format for gene-only Stage-2 profile: {path}")


def _load_kukanja_npz(path: str | Path, *, max_cells: int = 0, seed: int = 3028) -> QueryLoadResult:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    meta = data["meta"].item()
    genes = [str(item) for item in meta["gene_names"]]
    label_names = [str(item) for item in meta["label_names"]]
    label_maps = meta["label_maps"]

    coarse_key = label_names[0]
    native_key = label_names[-1]
    sample_inverse = _inverse_map(meta.get("sample_map", {}))
    sample_ids = np.asarray([sample_inverse.get(int(value), str(value)) for value in data["sample_ids"]], dtype=str)

    if "cell_ids" in data.files:
        cell_ids = [str(item) for item in data["cell_ids"]]
    else:
        cell_ids = [f"{path.stem}_cell_{idx}" for idx in range(data["X"].shape[0])]

    obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "sample_id": sample_ids,
            "native_label": _decode_label(data[f"y_{native_key}"], label_maps[native_key]),
            "coarse_label": _decode_label(data[f"y_{coarse_key}"], label_maps[coarse_key]),
        }
    )
    if "disease_score" in data.files:
        obs["disease_score"] = data["disease_score"]
    if "spatial" in data.files:
        obs["spatial_x"] = data["spatial"][:, 0]
        obs["spatial_y"] = data["spatial"][:, 1]

    idx = _sample_index_from_obs(obs, max_cells=max_cells, seed=seed)
    X = sparse.csr_matrix(data["X"][idx, :])
    obs = obs.iloc[idx].reset_index(drop=True)
    var = pd.DataFrame({"feature_id": genes}, index=pd.Index(genes, name="feature_id"))
    return QueryLoadResult(
        bundle=MatrixBundle(X=X, obs=obs, var=var, genes=[normalize_gene_name(g) for g in genes]),
        adapter="npz_kukanja",
        native_label_column=native_key,
        coarse_label_column=coarse_key,
        sample_column="sample_ids",
    )


def _feature_symbols_from_adata(adata: ad.AnnData) -> list[str]:
    for column in ("feature_id", "feature_name", "gene_symbol", "gene_name"):
        if column in adata.var.columns:
            return [normalize_gene_name(x) for x in adata.var[column].astype(str)]
    return [normalize_gene_name(x) for x in adata.var_names.astype(str)]


def _load_h5ad_query(path: str | Path, *, max_cells: int = 0, seed: int = 3028) -> QueryLoadResult:
    path = Path(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        obs = backed.obs.copy()
        native_col = None
        for candidate in ("Supertype", "native_label", "cell_type", "subtype_annotation", "Class"):
            if candidate in obs.columns:
                native_col = candidate
                break
        coarse_col = None
        for candidate in ("Subclass", "coarse_label", "Class", "cell_type_annot"):
            if candidate in obs.columns:
                coarse_col = candidate
                break
        sample_col = None
        for candidate in ("Donor ID", "donor_id", "sample_id", "sample", "Section"):
            if candidate in obs.columns:
                sample_col = candidate
                break

        out_obs = pd.DataFrame({"cell_id": obs.index.astype(str)})
        if native_col:
            out_obs["native_label"] = obs[native_col].astype(str).to_numpy()
        if coarse_col:
            out_obs["coarse_label"] = obs[coarse_col].astype(str).to_numpy()
        if sample_col:
            out_obs["sample_id"] = obs[sample_col].astype(str).to_numpy()

        idx = _sample_index_from_obs(out_obs, max_cells=max_cells, seed=seed)
        sub = backed[idx, :].to_memory()
        genes = _feature_symbols_from_adata(sub)
        var = sub.var.copy()
        if "feature_id" not in var.columns:
            var["feature_id"] = genes
        return QueryLoadResult(
            bundle=MatrixBundle(
                X=sparse.csr_matrix(sub.X),
                obs=out_obs.iloc[idx].reset_index(drop=True),
                var=var,
                genes=genes,
            ),
            adapter="h5ad",
            native_label_column=native_col,
            coarse_label_column=coarse_col,
            sample_column=sample_col,
        )
    finally:
        if getattr(backed, "file", None) is not None:
            backed.file.close()


def load_query_bundle(
    input_path: str | Path,
    *,
    dataset_id: str,
    max_cells: int = 0,
    seed: int = 3028,
) -> QueryLoadResult:
    path = Path(input_path)
    if path.suffix == ".npz":
        return _load_kukanja_npz(path, max_cells=max_cells, seed=seed)
    if path.suffix == ".h5ad":
        return _load_h5ad_query(path, max_cells=max_cells, seed=seed)
    raise ValueError(f"Unsupported query format for {dataset_id}: {path}")


def _label_counts(obs: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in obs.columns:
        return {}
    values = obs[column].astype(str)
    values = values[~values.isin(["", "nan", "None", "unknown"])]
    return {str(k): int(v) for k, v in values.value_counts().items()}


def _build_gene_only_profile(
    *,
    dataset_id: str,
    input_path: str | Path,
    profile_cells: int,
) -> dict[str, Any]:
    gene_profile = _load_query_gene_profile(input_path)
    genes = [normalize_gene_name(g) for g in gene_profile["genes"]]
    return {
        "schema_version": "aos.stage2.query_profile.gene_only.v1",
        "query_visibility": "gene_names_only",
        "dataset_id": dataset_id,
        "input_path": str(input_path),
        "query_adapter": gene_profile["query_adapter"],
        "profile_cells": int(profile_cells),
        "n_obs_profiled": int(gene_profile.get("n_obs_profiled", 0) or 0),
        "n_vars": int(gene_profile["n_vars"]),
        "genes": genes,
    }


def _selector_visible_query_profile(query_profile: dict[str, Any]) -> dict[str, Any]:
    genes = [normalize_gene_name(g) for g in query_profile.get("genes", [])]
    return {
        "schema_version": "aos.stage2.selector_visible_query_profile.v1",
        "query_visibility": "gene_names_only",
        "n_vars": len(genes),
        "genes": genes,
    }


def profile_query(
    *,
    dataset_id: str,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_cells: int = 20_000,
    seed: int = 3028,
) -> dict[str, Any]:
    resolved_input = _resolve_query_path(dataset_id, input_path)
    out_dir = ensure_dir(output_dir or (DEFAULT_STAGE2_ROOT / dataset_id))
    profile = _build_gene_only_profile(dataset_id=dataset_id, input_path=resolved_input, profile_cells=max_cells)
    profile_path = out_dir / "query_profile.json"
    write_json(profile, profile_path)
    return {
        "dataset_id": dataset_id,
        "input_path": str(resolved_input),
        "query_adapter": profile["query_adapter"],
        "query_profile_path": str(profile_path),
        "output_dir": str(out_dir),
    }


def _load_source_bundle(source_id: str, *, max_cells: int, seed: int, prepared_source_root: Path = DEFAULT_PREPARED_SOURCE_ROOT) -> MatrixBundle:
    errors: list[str] = []
    source_dir = _resolve_prepared_source_dir(source_id, prepared_source_root)
    for candidate_root in _prepared_source_roots(prepared_source_root):
        candidate_dir = candidate_root / source_id
        if candidate_dir != source_dir and not candidate_dir.exists():
            continue
        try:
            X, obs, var = read_standard_bundle(candidate_dir)
            source_dir = candidate_dir
            break
        except Exception as exc:
            errors.append(f"{candidate_dir}: {type(exc).__name__}: {exc}")
    else:
        raise FileNotFoundError(f"Could not load prepared source bundle for {source_id}. Tried: {'; '.join(errors)}")
    idx = _sample_index_random(len(obs), max_cells=max_cells, seed=seed)
    X = X[idx, :].tocsr()
    obs = obs.iloc[idx].reset_index(drop=True)
    genes = []
    for value in var["feature_id"].astype(str).tolist() if "feature_id" in var.columns else var.index.astype(str).tolist():
        genes.append(normalize_gene_name(value))
    return MatrixBundle(X=X, obs=obs, var=var, genes=genes)


def _source_gene_profile(source_id: str, *, prepared_source_root: Path = DEFAULT_PREPARED_SOURCE_ROOT) -> dict[str, Any]:
    source_dir = _resolve_prepared_source_dir(source_id, prepared_source_root)
    var_path = source_dir / "var.csv"
    if var_path.exists():
        var = pd.read_csv(var_path)
        if "feature_id" in var.columns:
            genes = [normalize_gene_name(x) for x in var["feature_id"].astype(str)]
        else:
            genes = [normalize_gene_name(x) for x in var.iloc[:, 0].astype(str)]
        return {
            "dataset_id": source_id,
            "input_path": str(source_dir),
            "query_adapter": "standard_bundle",
            "n_vars": len(genes),
            "genes": genes,
        }
    bundle = _load_source_bundle(source_id, max_cells=0, seed=3028, prepared_source_root=prepared_source_root)
    return {
        "dataset_id": source_id,
        "input_path": str(source_dir),
        "query_adapter": "standard_bundle",
        "n_vars": len(bundle.genes),
        "genes": [normalize_gene_name(g) for g in bundle.genes],
    }


def _source_profiles(
    source_ids: list[str],
    *,
    max_cells: int,
    seed: int,
    prepared_source_root: Path = DEFAULT_PREPARED_SOURCE_ROOT,
) -> dict[str, dict[str, Any]]:
    profiles = {}
    for source_id in source_ids:
        profiles[source_id] = _source_gene_profile(source_id, prepared_source_root=prepared_source_root)
    return profiles


def _discover_source_ids(artifact_bundle: Path, *, prepared_source_root: Path = DEFAULT_PREPARED_SOURCE_ROOT) -> list[str]:
    """Return source ids that can be used as label-transfer references."""
    source_ids: list[str] = []
    source_manifest_path = artifact_bundle / "manifests" / "synthetic_sources.json"
    if source_manifest_path.exists():
        source_manifest = read_json(source_manifest_path)
        source_ids.extend(str(item["source_id"]) for item in source_manifest.get("sources", []) if item.get("source_id"))

    for root in _prepared_source_roots(prepared_source_root):
        if root.exists():
            for manifest in sorted(root.glob("*/source_manifest.json")):
                try:
                    payload = read_json(manifest)
                except Exception:
                    continue
                source_id = str(payload.get("source_id") or manifest.parent.name)
                if source_id:
                    source_ids.append(source_id)

    return sorted(dict.fromkeys(source_ids))


def _similarity(query_profile: dict[str, Any], source_profile: dict[str, Any]) -> dict[str, Any]:
    qgenes = set(query_profile["genes"])
    sgenes = set(source_profile["genes"])
    shared = sorted(qgenes & sgenes)
    shared_count = len(shared)
    query_coverage = shared_count / max(1, len(qgenes))
    source_coverage = shared_count / max(1, len(sgenes))
    min_panel_coverage = shared_count / max(1, min(len(qgenes), len(sgenes)))
    jaccard = shared_count / max(1, len(qgenes | sgenes))
    gene_score = 0.0
    if shared_count:
        # Query-side Stage 2 evidence is restricted to the gene names.  No
        # expression moments, variable-gene ranks, labels, sample/donor counts,
        # path-derived modality hints, or query results enter this score.
        gene_score = float(np.mean([query_coverage, min_panel_coverage, jaccard]))
    fused = float(gene_score)
    return {
        "source_id": source_profile["dataset_id"],
        "shared_genes": shared_count,
        "shared_gene_names": shared,
        "query_gene_coverage": query_coverage,
        "source_gene_coverage": source_coverage,
        "min_panel_coverage": min_panel_coverage,
        "gene_set_jaccard": jaccard,
        "gene_score": float(gene_score),
        "expression_score": "",
        "celltype_score": "",
        "similarity_score": float(fused),
    }


def _load_score_rows(artifact_bundle: Path) -> list[dict[str, str]]:
    score_path = artifact_bundle / "scores" / "full_model_variant_scores.csv"
    if not score_path.exists():
        score_path = paths.AOS_ROOT / "reports" / "stage1_benchmark" / "full_model_variant_scores.csv"
    if not score_path.exists():
        score_path = paths.AOS_ROOT / "full_model_variant_scores.csv"
    if not score_path.exists():
        return []
    with score_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_capabilities(capability_dir: Path = DEFAULT_CAPABILITY_DIR) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(capability_dir.glob("*.yaml")):
        if path.name == "index.yaml":
            continue
        data = _read_yaml(path) or {}
        model_id = str(data.get("model_id", path.stem))
        data["_capability_yaml"] = str(path)
        out[model_id] = data
    return out


def _source_descriptions(prepared_source_root: Path = DEFAULT_PREPARED_SOURCE_ROOT) -> dict[str, dict[str, Any]]:
    descriptions: dict[str, dict[str, Any]] = {}
    for root in _prepared_source_roots(prepared_source_root):
        for manifest in sorted(root.glob("*/source_manifest.json")):
            try:
                payload = read_json(manifest)
            except Exception:
                continue
            source_id = str(payload.get("source_id") or manifest.parent.name)
            descriptions[source_id] = {
                "source_group": source_id,
                "description": (
                    f"{payload.get('species', 'unknown')} prepared source `{source_id}`; "
                    f"{payload.get('n_obs', '')} cells, {payload.get('n_vars', '')} genes."
                ),
                "species": payload.get("species", ""),
                "source_path": str(manifest.parent),
                "n_obs": payload.get("n_obs"),
                "n_vars": payload.get("n_vars"),
                "label_column": payload.get("label_column"),
                "sample_column": payload.get("sample_column"),
            }
    return descriptions


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std_or_zero(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _se_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(_std_or_zero(values) / np.sqrt(len(values)))


def _quantile_or_default(values: list[float], q: float, default: float) -> float:
    return float(np.quantile(values, q)) if values else float(default)


def _source_composite_score(
    *,
    baseline_macro_f1_mean: float | None,
    variant_macro_f1_mean: float | None,
    robustness_ratio_mean: float | None,
) -> float:
    parts: list[tuple[float, float]] = []
    if baseline_macro_f1_mean is not None:
        parts.append((0.45, baseline_macro_f1_mean))
    if variant_macro_f1_mean is not None:
        parts.append((0.40, variant_macro_f1_mean))
    if robustness_ratio_mean is not None:
        parts.append((0.15, robustness_ratio_mean))
    if not parts:
        return 0.0
    total = sum(weight for weight, _ in parts)
    return float(sum(weight * value for weight, value in parts) / max(total, 1e-9))


def _aggregate_scores(
    score_rows: list[dict[str, str]],
    *,
    prepared_source_root: Path = DEFAULT_PREPARED_SOURCE_ROOT,
) -> dict[tuple[str, str], dict[str, Any]]:
    descriptions = _source_descriptions(prepared_source_root)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        if row.get("evaluation_type") != NO_TRAINING_EVALUATION:
            continue
        grouped[(row["model"], row["source_group"])].append(row)

    out = {}
    for key, rows in grouped.items():
        macro = [float(row["macro_f1"]) for row in rows if row.get("macro_f1")]
        baseline_macro = [
            float(row["macro_f1"])
            for row in rows
            if row.get("macro_f1") and str(row.get("variant", "")).startswith("baseline")
        ]
        variant_macro = [
            float(row["macro_f1"])
            for row in rows
            if row.get("macro_f1") and not str(row.get("variant", "")).startswith("baseline")
        ]
        ratios = []
        for row in rows:
            if row.get("variant", "").startswith("baseline"):
                continue
            try:
                ratios.append(min(1.0, max(0.0, float(row.get("ratio_vs_baseline", "")))))
            except ValueError:
                pass
        baseline_mean = _mean_or_zero(baseline_macro) if baseline_macro else None
        variant_mean = _mean_or_zero(variant_macro) if variant_macro else None
        robustness_mean = _mean_or_zero(ratios) if ratios else None
        macro_mean = _mean_or_zero(macro)
        macro_std = _std_or_zero(macro)
        macro_se = _se_or_zero(macro)
        source_desc = descriptions.get(key[1], {"description": f"Benchmark source group `{key[1]}`."})
        out[key] = {
            "model_id": key[0],
            "source_id": key[1],
            "source_model_macro_f1": macro_mean,
            "source_model_macro_f1_std": macro_std,
            "source_model_macro_f1_se": macro_se,
            "source_model_macro_f1_lcb": max(0.0, macro_mean - macro_se),
            "source_model_macro_f1_best": float(max(macro)) if macro else 0.0,
            "source_dataset_composite_score": _source_composite_score(
                baseline_macro_f1_mean=baseline_mean,
                variant_macro_f1_mean=variant_mean,
                robustness_ratio_mean=robustness_mean,
            ),
            "baseline_macro_f1_mean": baseline_mean or 0.0,
            "variant_macro_f1_mean": variant_mean or 0.0,
            "robustness": robustness_mean if robustness_mean is not None else 0.5,
            "robustness_median": _quantile_or_default(ratios, 0.50, robustness_mean if robustness_mean is not None else 0.5),
            "robustness_q25": _quantile_or_default(ratios, 0.25, robustness_mean if robustness_mean is not None else 0.5),
            "score_rows": len(rows),
            "source_dataset_description": source_desc.get("description", ""),
            "source_dataset_species": source_desc.get("species", ""),
            "source_dataset_path": source_desc.get("source_path", ""),
        }
    return out


def _model_score_fallbacks(score_aggs: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model_id, _source_id), row in score_aggs.items():
        grouped[model_id].append(row)

    out: dict[str, dict[str, Any]] = {}
    for model_id, rows in grouped.items():
        macro = [float(row.get("source_model_macro_f1", 0.0)) for row in rows]
        best = [float(row.get("source_model_macro_f1_best", 0.0)) for row in rows]
        composite = [float(row.get("source_dataset_composite_score", 0.0)) for row in rows]
        robustness = [float(row.get("robustness", 0.5)) for row in rows]
        lcb = [float(row.get("source_model_macro_f1_lcb", row.get("source_model_macro_f1", 0.0))) for row in rows]
        robustness_median = [float(row.get("robustness_median", row.get("robustness", 0.5))) for row in rows]
        robustness_q25 = [float(row.get("robustness_q25", row.get("robustness", 0.5))) for row in rows]
        out[model_id] = {
            "model_id": model_id,
            "source_model_macro_f1": float(np.mean(macro)) if macro else 0.0,
            "source_model_macro_f1_std": 0.0,
            "source_model_macro_f1_se": 0.0,
            "source_model_macro_f1_lcb": float(np.mean(lcb)) if lcb else 0.0,
            "source_model_macro_f1_best": float(np.mean(best)) if best else 0.0,
            "source_dataset_composite_score": float(np.mean(composite)) if composite else 0.0,
            "baseline_macro_f1_mean": 0.0,
            "variant_macro_f1_mean": 0.0,
            "robustness": float(np.mean(robustness)) if robustness else 0.5,
            "robustness_median": float(np.mean(robustness_median)) if robustness_median else 0.5,
            "robustness_q25": float(np.mean(robustness_q25)) if robustness_q25 else 0.5,
            "score_rows": int(sum(int(row.get("score_rows", 0)) for row in rows)),
            "score_source": "model_mean_stage1_sources",
            "source_dataset_description": "Fallback average across source-specific benchmark rows for this model.",
            "source_dataset_species": "",
            "source_dataset_path": "",
        }
    return out


def _fallback_score_for_capability(model_id: str, capability: dict[str, Any]) -> dict[str, Any]:
    source_scores = capability.get("stage1_evaluation", {}).get("source_dataset_scores", [])
    composite_values: list[float] = []
    for row in source_scores if isinstance(source_scores, list) else []:
        if not bool(row.get("score_available", False)):
            continue
        if str(row.get("status", "")) not in {"scored", "scored_with_skips"}:
            continue
        try:
            value = row.get("composite_score")
            if value is not None:
                composite_values.append(float(value))
        except Exception:
            pass
    if composite_values:
        score = float(np.mean(composite_values))
        score_source = "capability_source_dataset_scores_mean"
    else:
        score = 0.0
        score_source = "no_source_dataset_score"
    return {
        "model_id": model_id,
        "source_model_macro_f1": score,
        "source_model_macro_f1_std": 0.0,
        "source_model_macro_f1_se": 0.0,
        "source_model_macro_f1_lcb": score,
        "source_model_macro_f1_best": score,
        "source_dataset_composite_score": score,
        "baseline_macro_f1_mean": 0.0,
        "variant_macro_f1_mean": 0.0,
        "robustness": 0.5,
        "robustness_median": 0.5,
        "robustness_q25": 0.5,
        "score_rows": 0,
        "score_source": score_source,
        "source_dataset_description": "Fallback score; no exact source-specific benchmark row was available.",
        "source_dataset_species": "",
        "source_dataset_path": "",
    }


def _reference_self_holdout_score_entry(capability: dict[str, Any], source_id: str) -> dict[str, Any]:
    rows = capability.get("stage1_evaluation", {}).get("reference_self_holdout_scores", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("source_group", "")) == str(source_id):
            return row
    return {}


def _score_info_from_reference_self_holdout(model_id: str, row: dict[str, Any]) -> dict[str, Any]:
    composite = float(row.get("composite_score", row.get("mean_macro_f1", 0.0)) or 0.0)
    mean_macro = float(row.get("mean_macro_f1", composite) or composite)
    return {
        "model_id": model_id,
        "source_model_macro_f1": mean_macro,
        "source_model_macro_f1_std": 0.0,
        "source_model_macro_f1_se": 0.0,
        "source_model_macro_f1_lcb": mean_macro,
        "source_model_macro_f1_best": composite,
        "source_dataset_composite_score": composite,
        "baseline_macro_f1_mean": 0.0,
        "variant_macro_f1_mean": 0.0,
        "robustness": float(row.get("label_overlap_fraction", 0.5) or 0.5),
        "robustness_median": float(row.get("label_overlap_fraction", 0.5) or 0.5),
        "robustness_q25": float(row.get("label_overlap_fraction", 0.5) or 0.5),
        "score_rows": int(row.get("rows", 0) or 0),
        "score_source": "reference_self_holdout_no_query",
        "source_dataset_description": (
            "Reference-source self-holdout benchmark; no query test set, query labels, "
            "or Stage4 result was used."
        ),
        "source_dataset_species": str(row.get("species", "")),
        "source_dataset_path": str(row.get("source_path", "")),
        "reference_self_holdout_task_scores": row.get("task_scores", {}),
    }


def _capability_source_score_entry(capability: dict[str, Any], source_id: str) -> dict[str, Any]:
    rows = capability.get("stage1_evaluation", {}).get("source_dataset_scores", [])
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("source_group", "")) == str(source_id):
            return row
    return {}


def _is_scored_capability_source_entry(row: dict[str, Any]) -> bool:
    if not row:
        return False
    if not bool(row.get("score_available", False)):
        return False
    if str(row.get("status", "")) not in {"scored", "scored_with_skips"}:
        return False
    return row.get("composite_score") is not None


def _float_or_default(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _score_info_from_capability_source(model_id: str, row: dict[str, Any]) -> dict[str, Any]:
    composite = _float_or_default(row.get("composite_score"), 0.0)
    mean_macro = _float_or_default(row.get("mean_macro_f1"), composite)
    best_macro = _float_or_default(row.get("best_macro_f1"), mean_macro)
    baseline = _float_or_default(row.get("baseline_macro_f1_mean"), 0.0)
    variant = _float_or_default(row.get("variant_macro_f1_mean"), mean_macro)
    robustness = _float_or_default(row.get("robustness_ratio_mean"), 0.5)
    source_desc = row.get("source_description", {}) if isinstance(row.get("source_description", {}), dict) else {}
    return {
        "model_id": model_id,
        "source_model_macro_f1": mean_macro,
        "source_model_macro_f1_std": 0.0,
        "source_model_macro_f1_se": 0.0,
        "source_model_macro_f1_lcb": mean_macro,
        "source_model_macro_f1_best": best_macro,
        "source_dataset_composite_score": composite,
        "baseline_macro_f1_mean": baseline,
        "variant_macro_f1_mean": variant,
        "robustness": robustness,
        "robustness_median": robustness,
        "robustness_q25": robustness,
        "score_rows": int(row.get("rows", 0) or 0),
        "score_source": "capability_source_dataset_scores_source",
        "source_dataset_description": str(source_desc.get("description", "")),
        "source_dataset_species": str(source_desc.get("species", "")),
        "source_dataset_path": str(source_desc.get("source_path", "")),
    }


def _method_parts(model_id: str, capability: dict[str, Any]) -> tuple[str, str, str]:
    defaults = capability.get("executor_defaults", {}) if isinstance(capability, dict) else {}
    embedding = defaults.get("embedding_method")
    transfer = defaults.get("transfer_method")
    if embedding and transfer:
        return str(model_id), str(embedding), str(transfer)
    embedding, transfer = _split_method(model_id)
    return model_id, embedding, transfer


def _normalize_selection_objective(selection_objective: str) -> tuple[str, str]:
    requested = str(selection_objective or DEFAULT_SELECTION_OBJECTIVE)
    if requested == DEFAULT_SELECTION_OBJECTIVE:
        return requested, requested
    if requested in LEGACY_SELECTION_OBJECTIVES:
        return DEFAULT_SELECTION_OBJECTIVE, requested
    raise ValueError(
        "selection_objective must be unified_rank; legacy aliases consensus and "
        "best_single_ablation are accepted but normalized to unified_rank"
    )


def _benchmark_evidence_reliability(score_source: str) -> tuple[float, str]:
    """Ordinal provenance tier used as one equal rank-aggregation axis."""
    source = str(score_source)
    if source == "exact_stage1_source":
        return 3.0, "exact source-specific Stage-1 synthetic benchmark"
    if source == "capability_source_dataset_scores_source":
        return 3.0, "source-specific Stage-1 capability-card benchmark"
    if source == "reference_self_holdout_no_query":
        return 2.0, "reference self-holdout benchmark without query labels/results"
    if source in {"model_mean_stage1_sources", "capability_source_dataset_scores_mean"}:
        return 1.0, "model-level Stage-1 fallback across sources"
    return 0.0, "no benchmark evidence"


@lru_cache(maxsize=16)
def _load_model_vocab_genes(base_method: str, species: str = "") -> frozenset[str]:
    method = str(base_method)
    genes: set[str] = set()
    try:
        if method == "geneformer_raw":
            path = paths.GENEFORMER_CHECKPOINT_DIR / "Geneformer_code_only" / "geneformer" / "gene_name_id_dict_gc104M.pkl"
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            genes = {normalize_gene_name(gene) for gene in payload.keys()}
        elif method in {"scgpt_brain_raw", "scgpt_human_raw"}:
            model_name = "brain" if method == "scgpt_brain_raw" else "human"
            path = paths.SCGPT_CHECKPOINT_ROOT / model_name / "vocab.json"
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            genes = {normalize_gene_name(gene) for gene in payload.keys() if not str(gene).startswith("<")}
        elif method == "nicheformer_raw":
            path = paths.NICHEFORMER_CHECKPOINT_DIR / "gene_name_id_dict_gc104M.pkl"
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            genes = {normalize_gene_name(gene) for gene in payload.keys()}
        elif method in {"uce_4l_raw", "uce_33l_raw"}:
            root = (
                paths.UCE_4L_MODEL_DIR
                if method == "uce_4l_raw"
                else paths.UCE_33L_MODEL_DIR
            )
            frame = pd.read_csv(root / "species_chrom.csv")
            species_key = str(species or "").strip().lower()
            if species_key in {"human", "homo sapiens"}:
                frame = frame[frame["species"].astype(str).str.lower() == "human"]
            elif species_key in {"mouse", "mus musculus"}:
                frame = frame[frame["species"].astype(str).str.lower() == "mouse"]
            genes = {normalize_gene_name(gene) for gene in frame["gene_symbol"].astype(str)}
    except Exception:
        genes = set()
    return frozenset(genes)


def _model_gene_coverage(
    *,
    embedding_method: str,
    shared_genes: list[str],
    source_species: str = "",
) -> dict[str, Any]:
    normalized_shared = [normalize_gene_name(gene) for gene in shared_genes]
    if not normalized_shared:
        return {"model_gene_coverage": 0.0, "model_usable_shared_genes": 0, "model_vocab_size": 0}
    vocab = _load_model_vocab_genes(str(embedding_method), str(source_species or ""))
    if not vocab:
        return {
            "model_gene_coverage": 1.0,
            "model_usable_shared_genes": len(normalized_shared),
            "model_vocab_size": 0,
            "model_vocab_status": "not_available_assumed_pass_through",
        }
    usable = sum(1 for gene in normalized_shared if gene in vocab)
    return {
        "model_gene_coverage": float(usable / max(1, len(normalized_shared))),
        "model_usable_shared_genes": int(usable),
        "model_vocab_size": int(len(vocab)),
        "model_vocab_status": "loaded",
    }


def _apply_rank_aggregation(candidates: list[dict[str, Any]]) -> None:
    """Assign a score by equal-vote evidence groups over pre-query evidence.

    Each evidence group receives one vote.  Axes inside a group are summarized
    with percentile ranks first, then the group scores are averaged.  This keeps
    the ranking interpretable without manually tuning numeric weights.
    """
    if not candidates:
        return
    frame = pd.DataFrame(candidates)
    axis_names = [axis for axis, _description in RANK_AGGREGATION_AXES]
    tie_axis_names = [axis for axis, _description in ANNOTATION_ANCHOR_TIE_AXES]
    for axis in sorted(set(axis_names + tie_axis_names)):
        values = pd.to_numeric(frame.get(axis, pd.Series([0.0] * len(frame))), errors="coerce").fillna(0.0)
        ranks = values.rank(method="average", pct=True, ascending=True)
        for idx, value in enumerate(ranks.astype(float).tolist()):
            candidates[idx][f"{axis}_rank_percentile"] = float(value)
    for idx, row in enumerate(candidates):
        group_values: list[float] = []
        group_payloads: list[dict[str, Any]] = []
        for group in RANK_AGGREGATION_GROUPS:
            group_name = str(group["group"])
            axes = list(group["axes"])
            axis_scores = [float(row.get(f"{axis}_rank_percentile", 0.0)) for axis, _description in axes]
            group_score = float(np.mean(axis_scores)) if axis_scores else 0.0
            row[f"{group_name}_evidence_score"] = group_score
            group_values.append(group_score)
            group_payloads.append(
                {
                    "group": group_name,
                    "description": group["description"],
                    "score": group_score,
                    "axes": [
                        {
                            "axis": axis,
                            "description": description,
                            "raw_value": row.get(axis, 0.0),
                            "rank_percentile": row.get(f"{axis}_rank_percentile", 0.0),
                        }
                        for axis, description in axes
                    ],
                }
            )
        tie_values = [float(row.get(f"{axis}_rank_percentile", 0.0)) for axis in tie_axis_names]
        row["score"] = float(np.mean(group_values)) if group_values else 0.0
        row["score_band"] = round(float(row["score"]), 2)
        row["annotation_anchor_tie_score"] = float(np.mean(tie_values)) if tie_values else 0.0
        row["rank_aggregation_method"] = RANK_AGGREGATION_METHOD
        row["rank_aggregation_groups"] = group_payloads
        row["rank_aggregation_axes"] = [
            {"axis": axis, "description": description, "rank_percentile": row.get(f"{axis}_rank_percentile", 0.0)}
            for axis, description in RANK_AGGREGATION_AXES
        ]
        row["annotation_anchor_tie_axes"] = [
            {"axis": axis, "description": description, "rank_percentile": row.get(f"{axis}_rank_percentile", 0.0)}
            for axis, description in ANNOTATION_ANCHOR_TIE_AXES
        ]
        row["base_selection_score"] = row["score"]
        row["selection_adjustment"] = 0.0
        row["selection_adjustment_reasons"] = [
            "score is an equal-vote aggregation over pre-query evidence groups; no manual numeric weights are used"
        ]


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("score_band", round(float(row.get("score", 0.0)), 2))),
        float(row.get("annotation_anchor_tie_score", 0.0)),
        float(row.get("score", 0.0)),
    )


def _select_with_family_diversity(candidates: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    ready = [row for row in sorted(candidates, key=_candidate_sort_key, reverse=True) if row["execution_ready"]]
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    seen_models: set[str] = set()
    for row in ready:
        family = row.get("family", "")
        if row["model_id"] in seen_models:
            continue
        if family in seen_families:
            continue
        selected.append(row)
        seen_families.add(family)
        seen_models.add(row["model_id"])
        if len(selected) >= top_k:
            return selected
    for row in ready:
        if row["model_id"] in seen_models:
            continue
        if row in selected:
            continue
        selected.append(row)
        seen_models.add(row["model_id"])
        if len(selected) >= top_k:
            break
    return selected


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _call_openai_json(*, model: str, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    _load_env_file()
    client = build_openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    meta = {
        "provider": "openai",
        "model": model,
        "response_id": getattr(response, "id", ""),
        "usage": getattr(response, "usage", None).model_dump() if getattr(response, "usage", None) is not None else {},
    }
    return _json_from_text(content), content, meta


def _top_counts(counts: dict[str, int], limit: int = 20) -> dict[str, int]:
    return dict(Counter({str(k): int(v) for k, v in counts.items()}).most_common(limit))


def _candidate_for_llm(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model_id",
        "source_id",
        "family",
        "shared_genes",
        "query_gene_coverage",
        "source_gene_coverage",
        "min_panel_coverage",
        "gene_set_jaccard",
        "model_gene_coverage",
        "model_usable_shared_genes",
        "source_similarity",
        "source_model_macro_f1",
        "source_model_macro_f1_std",
        "source_model_macro_f1_se",
        "source_model_macro_f1_lcb",
        "source_model_macro_f1_best",
        "source_dataset_composite_score",
        "baseline_macro_f1_mean",
        "variant_macro_f1_mean",
        "robustness",
        "robustness_median",
        "robustness_q25",
        "score_source",
        "benchmark_evidence_reliability",
        "benchmark_evidence_label",
        "rank_aggregation_method",
        "rank_aggregation_groups",
        "base_selection_score",
        "selection_adjustment",
        "selection_adjustment_reasons",
        "query_source_gene_fit_evidence_score",
        "stage1_annotation_ability_evidence_score",
        "synthetic_variant_robustness_evidence_score",
        "benchmark_provenance_evidence_score",
        "query_gene_coverage_rank_percentile",
        "min_panel_coverage_rank_percentile",
        "gene_set_jaccard_rank_percentile",
        "source_model_macro_f1_lcb_rank_percentile",
        "source_model_macro_f1_rank_percentile",
        "baseline_macro_f1_mean_rank_percentile",
        "robustness_rank_percentile",
        "robustness_median_rank_percentile",
        "robustness_q25_rank_percentile",
        "benchmark_evidence_reliability_rank_percentile",
        "score",
        "execution_ready",
        "not_ready_reasons",
        "source_benchmark_status",
        "source_benchmark_score_available",
        "reference_self_holdout_task_scores",
        "method",
        "embedding_method",
        "transfer_method",
    ]
    out = {key: row.get(key) for key in keys}
    description = str(row.get("source_dataset_description", ""))
    if description:
        out["source_dataset_description"] = description[:220]
    return out


def _build_llm_observe(
    *,
    query_profile: dict[str, Any],
    similarities: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_k: int,
    min_shared_genes: int,
    max_candidates: int,
    excluded_model_ids: set[str],
    selection_objective: str = DEFAULT_SELECTION_OBJECTIVE,
    selection_round: int = 1,
    previous_selections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ready = [row for row in candidates if row.get("execution_ready")]
    candidate_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    effective_max_candidates = min(int(max_candidates), 40)
    # Always expose the best available source for every model.  Otherwise a
    # strong backbone can disappear from the LLM context when one family has
    # many near-duplicate high-scoring references.
    best_by_model: dict[str, dict[str, Any]] = {}
    for row in ready:
        model_id = str(row["model_id"])
        if model_id not in best_by_model or float(row["score"]) > float(best_by_model[model_id]["score"]):
            best_by_model[model_id] = row
    for row in sorted(best_by_model.values(), key=lambda item: float(item["score"]), reverse=True):
        key = (row["model_id"], row["source_id"])
        candidate_rows.append(_candidate_for_llm(row))
        seen.add(key)
        if len(candidate_rows) >= effective_max_candidates:
            break
    if len(candidate_rows) < min(effective_max_candidates, len(candidates)):
        for row in ready:
            key = (row["model_id"], row["source_id"])
            if key in seen:
                continue
            candidate_rows.append(_candidate_for_llm(row))
            seen.add(key)
            if len(candidate_rows) >= effective_max_candidates:
                break

    return {
        "schema_version": "aos.stage2.llm_observe.v1",
        "query_visibility": "gene_names_only",
        "query_identifier_redacted": True,
        "query_path_redacted": True,
        "query_adapter_redacted": True,
        "top_k": int(top_k),
        "selection_objective": selection_objective,
        "selection_round": int(selection_round),
        "previous_selections": previous_selections or [],
        "min_shared_genes": int(min_shared_genes),
        "selection_policy": {
            "llm_is_final_selector": True,
            "planner_agent": "llm_stage2_planner",
            "reviewer_agent": "llm_stage2_reviewer",
            "deterministic_scores_are_evidence_card_only": True,
            "rank_aggregation_method": RANK_AGGREGATION_METHOD,
            "rank_aggregation_groups": [
                {
                    "group": group["group"],
                    "description": group["description"],
                    "axes": [{"axis": axis, "description": description} for axis, description in group["axes"]],
                }
                for group in RANK_AGGREGATION_GROUPS
            ],
            "rank_aggregation_axes": [
                {"axis": axis, "description": description} for axis, description in RANK_AGGREGATION_AXES
            ],
            "rank_aggregation_axis_weighting": "none; axes are grouped before voting",
            "rank_aggregation_group_weighting": "equal; one evidence-group vote per group",
            "single_rule_for_top1_and_topn": True,
            "rank1_is_top1_ablation": True,
            "species_is_hard_filter": False,
            "training_or_finetuning_allowed": False,
            "trained_seaad_140_heads_excluded": True,
            "direct_head_adapters_disabled": True,
            "execution_contract": "raw_label_transfer_knn_or_prototype_only",
            "require_family_diversity_when_possible": True,
            "excluded_model_ids": sorted(excluded_model_ids),
        },
        "selection_notes": (
            "Planner sees only query gene names plus reference/source and capability-card evidence. Reviewer audits "
            "the planner decision. Query labels, label counts, expression values, top-variable genes, donor/sample "
            "composition, query path, query adapter, query dataset id, query performance, and Stage4 results are hidden."
        ),
        "query_summary": {
            "n_vars": int(query_profile.get("n_vars", 0)),
            "query_genes": list(query_profile.get("genes", []))[:1000],
            "query_genes_truncated": len(list(query_profile.get("genes", []) or [])) > 1000,
        },
        "source_similarity_top": sorted(similarities, key=lambda item: item["similarity_score"], reverse=True)[:12],
        "candidate_pairs": candidate_rows,
    }


def _render_llm_prompt(observe: dict[str, Any], *, previous: dict[str, Any] | None = None) -> tuple[str, str]:
    selection_objective = str(observe.get("selection_objective", DEFAULT_SELECTION_OBJECTIVE))
    system_prompt = (
        "You are the AutOmicScience stage-2 planner agent. Select source+model execution pairs for "
        "no-training cross-species annotation. You may use only query gene names plus provided reference/source "
        "and benchmark evidence. Do not infer from hidden query labels, query expression values, query path, "
        "query dataset id, query performance, or query-inference results. Species mismatch is not a hard filter. "
        "Return only valid JSON."
    )
    response_contract = {
        "thought_summary": "brief audit-friendly summary, no hidden chain-of-thought",
        "selected_pairs": [
            {
                "rank": 1,
                "model_id": "candidate model_id",
                "source_id": "candidate source_id",
                "rationale": "why this pair should execute next",
            }
        ],
        "rejected_pair_notes": [
            {"model_id": "candidate model_id", "source_id": "candidate source_id", "reason": "short reason"}
        ],
        "review_flags": ["optional concerns"],
    }
    instructions = [
        "Choose exactly top_k pairs when enough execution-ready candidates exist.",
        "Choose only pairs that appear in observe.candidate_pairs and have execution_ready=true.",
        "observe.candidate_pairs is the authoritative compact candidate table; do not claim candidate identifiers are unavailable when it is present.",
        "Do not choose excluded_model_ids.",
        "Do not choose trained SEA-AD 140-gene heads; all selected pairs must be raw label-transfer adapters.",
        "Use at most one model per family when at least top_k execution-ready families are available.",
        "A selected model_id must be unique because the executor binds one source/reference per model.",
        "Every selected pair must include rank, model_id, source_id, and rationale.",
        "There is only one objective: unified_rank. Rank-1 from this same ordered selection is the top-1 ablation; ranks 1..top_k are the downstream execution set.",
        "Do not switch criteria between the first pick and later picks. Later one-by-one rounds apply the same rule after excluding prior selected models/families.",
        "Use only pre-execution evidence: query gene names, gene overlap, source similarity, and source-disaggregated benchmark rows.",
        "Do not infer from any query label, label count, expression statistic, sample/donor composition, query path, query dataset id, previously measured query performance, or prior query outcome.",
        "Treat benchmark fields as benchmark-on-that-source evidence, not as a direct estimate of the query score.",
        "source_dataset_composite_score may appear for audit compatibility, but the rank aggregation uses evidence groups instead of that preweighted composite.",
        "Use model_gene_coverage as an input-contract audit field; do not over-reward tiny vocabulary differences once the candidate is executable.",
        "The score column is equal evidence-group rank aggregation: query-source gene fit, Stage-1 annotation ability, synthetic robustness, and benchmark provenance each get one group vote.",
        "source_model_macro_f1_lcb is a pre-query risk-adjusted benchmark field: source_model_macro_f1 minus one standard error across Stage-1 rows.",
        "If overriding the top evidence-card row, explain which evidence group or capability-card fact justifies the override.",
        "When score_source is model_mean_stage1_sources, treat benchmark fields as weak fallback model evidence and prefer strong gene/reference fit.",
        "When a candidate has high source benchmark but weak gene/source fit, prefer the pair with clearer input compatibility and reference coverage.",
        "Mention uncertainty in the rationale when benchmark evidence is source-specific or the query gene panel is small.",
    ]
    if int(observe.get("top_k", 0)) == 1:
        instructions.extend(
            [
                "This is one round of a one-by-one planner: choose exactly one next source+model pair.",
                "Previous selections are shown in observe.previous_selections and their model_ids are already excluded.",
                "Do not re-rank or revise previous selections; apply unified_rank to the remaining candidates only.",
            ]
        )
    if selection_objective != DEFAULT_SELECTION_OBJECTIVE:
        instructions.append(
            f"Requested legacy objective `{selection_objective}` has been normalized by the executor to unified_rank; follow unified_rank only."
        )
    payload: dict[str, Any] = {
        "instructions": instructions,
        "response_contract": response_contract,
        "observe": observe,
    }
    if previous:
        payload["previous_attempt"] = previous
        payload["repair_instruction"] = (
            "Fix the previous JSON so the deterministic reviewer passes. "
            "Return exactly one selected pair for one-by-one rounds, using exact model_id/source_id values from observe.candidate_pairs."
        )
    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return system_prompt, user_prompt


def _review_llm_payload(
    *,
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    top_k: int,
    excluded_model_ids: set[str],
) -> dict[str, Any]:
    candidate_by_key = {(row["model_id"], row["source_id"]): row for row in candidates}
    ready_count = sum(1 for row in candidates if row.get("execution_ready"))
    selected = payload.get("selected_pairs", [])
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(selected, list):
        errors.append("selected_pairs must be a list")
        selected = []
    expected = min(int(top_k), int(ready_count))
    if len(selected) != expected:
        errors.append(f"selected_pairs length {len(selected)} != expected {expected}")

    seen_models: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    seen_families: set[str] = set()
    ready_families = {
        str(row.get("family", ""))
        for row in candidates
        if row.get("execution_ready") and row.get("family")
    }
    require_family_diversity = len(ready_families) >= expected
    for idx, item in enumerate(selected, start=1):
        if not isinstance(item, dict):
            errors.append(f"selected_pairs[{idx}] is not an object")
            continue
        model_id = str(item.get("model_id", ""))
        source_id = str(item.get("source_id", ""))
        if not model_id or not source_id:
            errors.append(f"selected_pairs[{idx}] missing model_id/source_id")
            continue
        if model_id in excluded_model_ids:
            errors.append(f"{model_id} is excluded")
        if model_id in seen_models:
            errors.append(f"{model_id} selected more than once")
        seen_models.add(model_id)
        key = (model_id, source_id)
        if key in seen_pairs:
            errors.append(f"{model_id}:{source_id} selected more than once")
        seen_pairs.add(key)
        candidate = candidate_by_key.get(key)
        if candidate is None:
            errors.append(f"{model_id}:{source_id} is not in candidate_pairs")
            continue
        if not candidate.get("execution_ready"):
            errors.append(f"{model_id}:{source_id} is not execution_ready: {candidate.get('not_ready_reasons', '')}")
        family = str(candidate.get("family", ""))
        if require_family_diversity and family in seen_families:
            errors.append(
                f"family {family} selected more than once while {len(ready_families)} execution-ready families exist"
            )
        if family:
            seen_families.add(family)
        if not item.get("rationale"):
            warnings.append(f"{model_id}:{source_id} has empty rationale")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "n_selected": len(selected),
        "expected_selected": expected,
    }


def _render_llm_reviewer_prompt(
    *,
    observe: dict[str, Any],
    planner_payload: dict[str, Any],
    deterministic_review: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You are the AutOmicScience stage-2 reviewer agent. Audit a planner-selected model/source pair list. "
        "You must use only the supplied observe payload, planner JSON, and deterministic review. "
        "Reject decisions that use hidden query labels, query expression, query identity/path, Stage3/Stage4 outcomes, "
        "manual objective switching, direct trained heads, or candidates outside the table. Return only JSON."
    )
    response_contract = {
        "status": "passed or failed",
        "accepted": True,
        "audit_summary": "short audit-friendly explanation",
        "errors": ["blocking issues, empty if passed"],
        "warnings": ["non-blocking concerns"],
        "planner_feedback": "repair advice if failed",
    }
    instructions = [
        "Pass only if the planner selected exact candidates from observe.candidate_pairs.",
        "Pass only if the same unified_rank logic is used for rank-1 and later ranks.",
        "Pass only if no query labels, query expression statistics, query path/id/adapter, donor/sample composition, or prior query outcomes are referenced.",
        "Pass only if trained SEA-AD 140-gene heads and direct classifier heads are not selected.",
        "Pass only if species mismatch is not treated as a hard rejection reason.",
        "The score is equal evidence-group rank aggregation, not a manually weighted objective; require rationale for overrides of the top score row.",
        "The score must be based on query gene overlap, source-disaggregated benchmark ability, synthetic robustness, and benchmark provenance groups, not a second-layer weighted composite benchmark score.",
        "If deterministic_review failed, the reviewer status must be failed.",
    ]
    payload = {
        "instructions": instructions,
        "response_contract": response_contract,
        "observe": observe,
        "planner_payload": planner_payload,
        "deterministic_review": deterministic_review,
    }
    return system_prompt, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _review_planner_with_llm(
    *,
    observe: dict[str, Any],
    planner_payload: dict[str, Any],
    deterministic_review: dict[str, Any],
    trace_dir: Path,
    llm_model: str,
    attempt: int,
) -> dict[str, Any]:
    system_prompt, user_prompt = _render_llm_reviewer_prompt(
        observe=observe,
        planner_payload=planner_payload,
        deterministic_review=deterministic_review,
    )
    (trace_dir / f"reviewer_prompt_attempt_{attempt}.md").write_text(
        "## System\n\n" + system_prompt + "\n\n## User\n\n```json\n" + user_prompt + "\n```\n",
        encoding="utf-8",
    )
    parsed, raw_text, meta = _call_openai_json(model=llm_model, system_prompt=system_prompt, user_prompt=user_prompt)
    (trace_dir / f"reviewer_response_attempt_{attempt}.txt").write_text(raw_text, encoding="utf-8")
    accepted = bool(parsed.get("accepted", False))
    errors = parsed.get("errors", [])
    if not isinstance(errors, list):
        errors = [str(errors)]
    status = str(parsed.get("status", "")).lower()
    if deterministic_review.get("status") != "passed":
        status = "failed"
        accepted = False
        errors.append("deterministic_review_failed")
    if status not in {"passed", "failed"}:
        status = "passed" if accepted and not errors else "failed"
    out = {
        "status": status,
        "accepted": bool(accepted and status == "passed" and not errors),
        "audit_summary": str(parsed.get("audit_summary", "")),
        "errors": errors,
        "warnings": parsed.get("warnings", []) if isinstance(parsed.get("warnings", []), list) else [],
        "planner_feedback": str(parsed.get("planner_feedback", "")),
        "meta": meta,
    }
    write_json(out, trace_dir / f"reviewer_review_attempt_{attempt}.json")
    return out


def _select_with_llm(
    *,
    query_profile: dict[str, Any],
    similarities: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_k: int,
    min_shared_genes: int,
    out_dir: Path,
    llm_model: str,
    llm_max_candidates: int,
    llm_retry_limit: int,
    excluded_model_ids: set[str],
    selection_objective: str = "consensus",
    selection_round: int = 1,
    previous_selections: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_dir = ensure_dir(out_dir / "llm_selection")
    observe = _build_llm_observe(
        query_profile=query_profile,
        similarities=similarities,
        candidates=candidates,
        top_k=top_k,
        min_shared_genes=min_shared_genes,
        max_candidates=llm_max_candidates,
        excluded_model_ids=excluded_model_ids,
        selection_objective=selection_objective,
        selection_round=selection_round,
        previous_selections=previous_selections,
    )
    write_json(observe, trace_dir / "observe.json")

    previous: dict[str, Any] | None = None
    last_review: dict[str, Any] = {}
    last_reviewer_review: dict[str, Any] = {}
    last_payload: dict[str, Any] = {}
    for attempt in range(1, max(0, llm_retry_limit) + 2):
        system_prompt, user_prompt = _render_llm_prompt(observe, previous=previous)
        (trace_dir / f"prompt_attempt_{attempt}.md").write_text(
            "## System\n\n" + system_prompt + "\n\n## User\n\n```json\n" + user_prompt + "\n```\n",
            encoding="utf-8",
        )
        try:
            parsed, raw_text, meta = _call_openai_json(model=llm_model, system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:  # noqa: BLE001
            error_payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "attempt": attempt}
            write_json(error_payload, trace_dir / f"review_attempt_{attempt}.json")
            raise
        (trace_dir / f"response_attempt_{attempt}.txt").write_text(raw_text, encoding="utf-8")
        write_json({"parsed": parsed, "meta": meta}, trace_dir / f"parsed_attempt_{attempt}.json")
        review = _review_llm_payload(
            payload=parsed,
            candidates=candidates,
            top_k=top_k,
            excluded_model_ids=excluded_model_ids,
        )
        write_json(review, trace_dir / f"review_attempt_{attempt}.json")
        last_review = review
        last_payload = parsed
        if review["status"] == "passed":
            reviewer_review = _review_planner_with_llm(
                observe=observe,
                planner_payload=parsed,
                deterministic_review=review,
                trace_dir=trace_dir,
                llm_model=llm_model,
                attempt=attempt,
            )
            last_reviewer_review = reviewer_review
            if reviewer_review.get("accepted"):
                break
            previous = {"response": parsed, "review": review, "llm_reviewer_review": reviewer_review}
            continue
        previous = {"response": parsed, "review": review}

    if last_review.get("status") != "passed" or not last_reviewer_review.get("accepted", False):
        review_errors = list(last_review.get("errors", []))
        review_errors.extend(str(x) for x in last_reviewer_review.get("errors", []))
        raise RuntimeError(
            "LLM stage2 selection failed deterministic review: "
            + "; ".join(review_errors or ["unknown review failure"])
        )

    candidate_by_key = {(row["model_id"], row["source_id"]): row for row in candidates}
    selected: list[dict[str, Any]] = []
    for item in last_payload.get("selected_pairs", []):
        row = dict(candidate_by_key[(str(item["model_id"]), str(item["source_id"]))])
        row["rank"] = int(item.get("rank") or (len(selected) + 1))
        row["rationale"] = str(item.get("rationale", "")).strip()
        row["selection_method"] = "llm_react_batch"
        selected.append(row)
    selected.sort(key=lambda row: int(row.get("rank", 999999)))
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank

    final_payload = {
        "status": "passed",
        "llm_model": llm_model,
        "trace_dir": str(trace_dir),
        "thought_summary": last_payload.get("thought_summary", ""),
        "review": last_review,
        "llm_reviewer_review": last_reviewer_review,
        "selected_pairs": [
            {"rank": row["rank"], "model_id": row["model_id"], "source_id": row["source_id"], "score": row["score"]}
            for row in selected
        ],
        "review_flags": last_payload.get("review_flags", []),
        "rejected_pair_notes": last_payload.get("rejected_pair_notes", []),
    }
    write_json(final_payload, trace_dir / "final_selection.json")
    return selected, final_payload


def _select_with_llm_iterative(
    *,
    query_profile: dict[str, Any],
    similarities: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_k: int,
    min_shared_genes: int,
    out_dir: Path,
    llm_model: str,
    llm_max_candidates: int,
    llm_retry_limit: int,
    excluded_model_ids: set[str],
    selection_objective: str,
    iterative_exclude_scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_dir = ensure_dir(out_dir / "llm_selection")
    selected: list[dict[str, Any]] = []
    dynamic_excluded = set(excluded_model_ids)
    excluded_families: set[str] = set()
    round_payloads: list[dict[str, Any]] = []
    previous_selection_summaries: list[dict[str, Any]] = []

    for round_idx in range(1, int(top_k) + 1):
        remaining = [
            row
            for row in candidates
            if row.get("execution_ready")
            and row["model_id"] not in dynamic_excluded
            and (iterative_exclude_scope != "family" or str(row.get("family", "")) not in excluded_families)
        ]
        if not remaining:
            break
        round_dir = ensure_dir(trace_dir / f"round_{round_idx}")
        round_selected, round_info = _select_with_llm(
            query_profile=query_profile,
            similarities=similarities,
            candidates=remaining,
            top_k=1,
            min_shared_genes=min_shared_genes,
            out_dir=round_dir,
            llm_model=llm_model,
            llm_max_candidates=llm_max_candidates,
            llm_retry_limit=llm_retry_limit,
            excluded_model_ids=dynamic_excluded,
            selection_objective=selection_objective,
            selection_round=round_idx,
            previous_selections=previous_selection_summaries,
        )
        if not round_selected:
            break
        row = dict(round_selected[0])
        row["rank"] = round_idx
        row["selection_method"] = "llm_react_one_by_one"
        selected.append(row)
        dynamic_excluded.add(str(row["model_id"]))
        family = str(row.get("family", ""))
        if family:
            excluded_families.add(family)
        previous_selection_summaries.append(
            {
                "round": round_idx,
                "model_id": row["model_id"],
                "source_id": row["source_id"],
                "family": family,
                "score": row["score"],
                "shared_genes": row.get("shared_genes", 0),
                "source_similarity": row.get("source_similarity", 0.0),
                "rationale": row.get("rationale", ""),
            }
        )

        round_payloads.append(
            {
                "round": round_idx,
                "selected_model_id": row["model_id"],
                "selected_source_id": row["source_id"],
                "selected_family": family,
                "round_trace_dir": round_info.get("trace_dir", ""),
                "llm_selection": round_info,
                "excluded_after_round": sorted(dynamic_excluded),
                "excluded_families_after_round": sorted(excluded_families),
            }
        )

    ready_rows = [row for row in candidates if row.get("execution_ready") and row["model_id"] not in excluded_model_ids]
    if iterative_exclude_scope == "family":
        expected = min(int(top_k), len({str(row.get("family", "")) for row in ready_rows if row.get("family")}))
    else:
        expected = min(int(top_k), len({str(row["model_id"]) for row in ready_rows}))
    final_payload = {
        "status": "passed" if selected else "failed",
        "llm_model": llm_model,
        "trace_dir": str(trace_dir),
        "selection_strategy": "one_by_one",
        "selection_objective": selection_objective,
        "iterative_exclude_scope": iterative_exclude_scope,
        "review": {
            "status": "passed" if len(selected) == expected else "needs_review",
            "n_selected": len(selected),
            "expected_selected": expected,
        },
        "selected_pairs": [
            {"rank": row["rank"], "model_id": row["model_id"], "source_id": row["source_id"], "score": row["score"]}
            for row in selected
        ],
        "rounds": round_payloads,
    }
    write_json(final_payload, trace_dir / "final_selection.json")
    return selected, final_payload


def select_models(
    *,
    query_profile_path: str | Path,
    output_dir: str | Path | None = None,
    artifact_bundle: str | Path = DEFAULT_ARTIFACT_BUNDLE,
    prepared_source_root: str | Path = DEFAULT_PREPARED_SOURCE_ROOT,
    capability_dir: str | Path = DEFAULT_CAPABILITY_DIR,
    top_k: int = DEFAULT_TOP_K,
    min_shared_genes: int = 30,
    max_source_profile_cells: int = 20_000,
    max_query_cells: int = 5_000,
    max_reference_cells: int = 1_000,
    k: int = 15,
    seed: int = 3028,
    device: str = "",
    batch_size: int = 16,
    llm_mode: str = "required",
    llm_model: str | None = None,
    llm_max_candidates: int = 80,
    llm_retry_limit: int = 2,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    selection_objective: str = DEFAULT_SELECTION_OBJECTIVE,
    iterative_exclude_scope: str = DEFAULT_ITERATIVE_EXCLUDE_SCOPE,
    excluded_model_ids: list[str] | None = None,
    include_default_excluded: bool = False,
) -> dict[str, Any]:
    query_profile_path = Path(query_profile_path)
    query_profile = read_json(query_profile_path)
    dataset_id = query_profile["dataset_id"]
    selector_query_profile = _selector_visible_query_profile(query_profile)
    out_dir = ensure_dir(output_dir or query_profile_path.parent)
    artifact_bundle = Path(artifact_bundle)
    prepared_source_root = Path(prepared_source_root)
    if llm_mode not in {"required", "optional", "off"}:
        raise ValueError("llm_mode must be one of: required, optional, off")
    if selection_strategy not in {"batch", "iterative", "one_by_one"}:
        raise ValueError("selection_strategy must be one of: batch, iterative, one_by_one")
    normalized_selection_strategy = "one_by_one" if selection_strategy in {"iterative", "one_by_one"} else "batch"
    selection_objective, requested_selection_objective = _normalize_selection_objective(selection_objective)
    if iterative_exclude_scope not in {"model", "family"}:
        raise ValueError("iterative_exclude_scope must be one of: model, family")
    _load_env_file()
    llm_model = llm_model or default_llm_model(DEFAULT_LLM_MODEL)
    excluded = set() if include_default_excluded else set(DEFAULT_EXCLUDED_MODEL_IDS)
    excluded.update(str(x) for x in (excluded_model_ids or []) if str(x))

    source_ids = _discover_source_ids(artifact_bundle, prepared_source_root=prepared_source_root)
    source_profiles = _source_profiles(
        source_ids,
        max_cells=max_source_profile_cells,
        seed=seed,
        prepared_source_root=prepared_source_root,
    )

    similarities = [_similarity(selector_query_profile, source_profiles[source_id]) for source_id in source_ids]
    sim_by_source = {row["source_id"]: row for row in similarities}
    pd.DataFrame(similarities).sort_values("similarity_score", ascending=False).to_csv(
        out_dir / "source_similarity.csv",
        index=False,
    )

    capabilities = _load_capabilities(Path(capability_dir))
    source_descriptions = _source_descriptions(prepared_source_root)
    score_aggs = _aggregate_scores(_load_score_rows(artifact_bundle), prepared_source_root=prepared_source_root)
    model_fallbacks = _model_score_fallbacks(score_aggs)
    candidates: list[dict[str, Any]] = []
    raw_model_ids = sorted(
        model_id
        for model_id, capability in capabilities.items()
        if capability.get("evaluator", "") == "raw_label_transfer"
        and capability.get("data_constraints", {}).get("compatible_contract") != TRAINED_CONTRACT
        and model_id not in excluded
        and not bool(capability.get("selection_disabled", False))
        and not bool(capability.get("execution_disabled", False))
    )
    for model_id in raw_model_ids:
        capability = capabilities.get(model_id, {})
        contract = capability.get("data_constraints", {}).get("compatible_contract")
        evaluator = capability.get("evaluator", "")
        if contract == TRAINED_CONTRACT or evaluator != "raw_label_transfer":
            continue
        for source_id in source_ids:
            score_info = score_aggs.get((model_id, source_id))
            capability_source_entry = _capability_source_score_entry(capability, source_id)
            if score_info is not None:
                score_info = {**score_info, "score_source": "exact_stage1_source"}
            elif _is_scored_capability_source_entry(capability_source_entry):
                score_info = _score_info_from_capability_source(model_id, capability_source_entry)
            else:
                self_holdout_entry = _reference_self_holdout_score_entry(capability, source_id)
                if self_holdout_entry:
                    score_info = _score_info_from_reference_self_holdout(model_id, self_holdout_entry)
                else:
                    score_info = model_fallbacks.get(model_id) or _fallback_score_for_capability(model_id, capability)
            similarity = sim_by_source.get(source_id)
            if not similarity:
                continue
            reference_path = _resolve_prepared_source_dir(source_id, prepared_source_root)
            method, embedding, transfer = _method_parts(model_id, capability)
            shared_genes = int(similarity["shared_genes"])
            source_species = str(
                score_info.get("source_dataset_species")
                or source_descriptions.get(source_id, {}).get("species", "")
                or ""
            )
            model_gene = _model_gene_coverage(
                embedding_method=embedding,
                shared_genes=list(similarity.get("shared_gene_names", [])),
                source_species=source_species,
            )
            execution_ready = bool(reference_path.exists() and shared_genes >= min_shared_genes)
            not_ready_reasons = []
            if not reference_path.exists():
                not_ready_reasons.append(f"reference_path not found: {reference_path}")
            if shared_genes < min_shared_genes:
                not_ready_reasons.append(f"shared_genes {shared_genes} < min_shared_genes {min_shared_genes}")
            benchmark_component = float(
                score_info.get("source_dataset_composite_score", score_info.get("source_model_macro_f1", 0.0))
            )
            reliability, reliability_label = _benchmark_evidence_reliability(str(score_info.get("score_source", "")))
            candidates.append(
                {
                    "model_id": model_id,
                    "source_id": source_id,
                    "family": capability.get("family", "unknown"),
                    "capability_yaml": capability.get("_capability_yaml", str(DEFAULT_CAPABILITY_DIR / f"{model_id}.yaml")),
                    "reference_path": str(reference_path),
                    "method": method,
                    "embedding_method": embedding,
                    "transfer_method": transfer,
                    "shared_genes": shared_genes,
                    "query_gene_coverage": float(similarity.get("query_gene_coverage", 0.0)),
                    "source_gene_coverage": float(similarity.get("source_gene_coverage", 0.0)),
                    "min_panel_coverage": float(similarity.get("min_panel_coverage", 0.0)),
                    "gene_set_jaccard": float(similarity.get("gene_set_jaccard", 0.0)),
                    "model_gene_coverage": float(model_gene.get("model_gene_coverage", 0.0)),
                    "model_usable_shared_genes": int(model_gene.get("model_usable_shared_genes", 0)),
                    "model_vocab_size": int(model_gene.get("model_vocab_size", 0)),
                    "model_vocab_status": str(model_gene.get("model_vocab_status", "")),
                    "source_similarity": float(similarity["similarity_score"]),
                    "source_model_macro_f1": float(score_info["source_model_macro_f1"]),
                    "source_model_macro_f1_std": float(score_info.get("source_model_macro_f1_std", 0.0)),
                    "source_model_macro_f1_se": float(score_info.get("source_model_macro_f1_se", 0.0)),
                    "source_model_macro_f1_lcb": float(
                        score_info.get(
                            "source_model_macro_f1_lcb",
                            max(
                                0.0,
                                float(score_info.get("source_model_macro_f1", 0.0))
                                - float(score_info.get("source_model_macro_f1_se", 0.0)),
                            ),
                        )
                    ),
                    "source_model_macro_f1_best": float(score_info.get("source_model_macro_f1_best", 0.0)),
                    "source_dataset_composite_score": benchmark_component,
                    "baseline_macro_f1_mean": float(score_info.get("baseline_macro_f1_mean", 0.0)),
                    "variant_macro_f1_mean": float(score_info.get("variant_macro_f1_mean", 0.0)),
                    "robustness": float(score_info["robustness"]),
                    "robustness_median": float(score_info.get("robustness_median", score_info.get("robustness", 0.5))),
                    "robustness_q25": float(score_info.get("robustness_q25", score_info.get("robustness", 0.5))),
                    "score_source": str(score_info.get("score_source", "")),
                    "benchmark_evidence_reliability": float(reliability),
                    "benchmark_evidence_label": reliability_label,
                    "base_selection_score": 0.0,
                    "selection_adjustment": 0.0,
                    "selection_adjustment_reasons": [],
                    "source_dataset_description": str(score_info.get("source_dataset_description", "")),
                    "source_dataset_species": source_species,
                    "source_dataset_path": str(score_info.get("source_dataset_path", "")),
                    "source_benchmark_status": str(capability_source_entry.get("status", "")),
                    "source_benchmark_score_available": bool(capability_source_entry.get("score_available", False)),
                    "source_benchmark_rows": list(capability_source_entry.get("benchmark_rows", []))
                    if isinstance(capability_source_entry.get("benchmark_rows", []), list)
                    else [],
                    "reference_self_holdout_task_scores": score_info.get("reference_self_holdout_task_scores", {}),
                    "source_skip_reasons": list(capability_source_entry.get("skip_reasons", []))
                    if isinstance(capability_source_entry.get("skip_reasons", []), list)
                    else [],
                    "score": 0.0,
                    "execution_ready": execution_ready,
                    "not_ready_reasons": "; ".join(not_ready_reasons),
                }
            )

    _apply_rank_aggregation(candidates)
    candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)
    pd.DataFrame(candidates).to_csv(out_dir / "candidate_pairs.csv", index=False)

    llm_selection: dict[str, Any] = {"status": "disabled", "mode": llm_mode}
    if llm_mode == "off":
        selected = _select_with_family_diversity(candidates, top_k=top_k)
        for rank, row in enumerate(selected, start=1):
            row["rank"] = rank
            row["rationale"] = (
                f"Deterministic fallback selected equal-rank score={row['score']:.4f} using "
                f"{row.get('rank_aggregation_method', RANK_AGGREGATION_METHOD)}; species was not used as a filter."
            )
            row["selection_method"] = "deterministic_fallback"
    else:
        try:
            if normalized_selection_strategy == "one_by_one":
                selected, llm_selection = _select_with_llm_iterative(
                    query_profile=selector_query_profile,
                    similarities=similarities,
                    candidates=candidates,
                    top_k=top_k,
                    min_shared_genes=min_shared_genes,
                    out_dir=out_dir,
                    llm_model=llm_model,
                    llm_max_candidates=llm_max_candidates,
                    llm_retry_limit=llm_retry_limit,
                    excluded_model_ids=excluded,
                    selection_objective=selection_objective,
                    iterative_exclude_scope=iterative_exclude_scope,
                )
            else:
                selected, llm_selection = _select_with_llm(
                    query_profile=selector_query_profile,
                    similarities=similarities,
                    candidates=candidates,
                    top_k=top_k,
                    min_shared_genes=min_shared_genes,
                    out_dir=out_dir,
                    llm_model=llm_model,
                    llm_max_candidates=llm_max_candidates,
                    llm_retry_limit=llm_retry_limit,
                    excluded_model_ids=excluded,
                    selection_objective=selection_objective,
                )
        except Exception as exc:  # noqa: BLE001
            llm_selection = {
                "status": "failed",
                "mode": llm_mode,
                "llm_model": llm_model,
                "selection_strategy": normalized_selection_strategy,
                "selection_strategy_requested": selection_strategy,
                "selection_objective": selection_objective,
                "selection_objective_requested": requested_selection_objective,
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(llm_selection, out_dir / "llm_selection_failed.json")
            if llm_mode == "required":
                raise
            selected = _select_with_family_diversity(candidates, top_k=top_k)
            for rank, row in enumerate(selected, start=1):
                row["rank"] = rank
                row["rationale"] = (
                    f"Deterministic fallback after LLM failure: equal-rank score={row['score']:.4f} using "
                    f"{row.get('rank_aggregation_method', RANK_AGGREGATION_METHOD)}."
                )
                row["selection_method"] = "deterministic_fallback_after_llm_failure"
    selected_keys = {(row["model_id"], row["source_id"]) for row in selected}
    rejected = []
    for row in candidates:
        if (row["model_id"], row["source_id"]) in selected_keys:
            continue
        reason = row["not_ready_reasons"] or "lower ranked than selected source+model pairs"
        rejected.append(
            {
                "model_id": row["model_id"],
                "source_id": row["source_id"],
                "score": row["score"],
                "execution_ready": row["execution_ready"],
                "reason": reason,
            }
        )

    review_reasons = []
    if len(selected) < top_k:
        review_reasons.append(f"Only {len(selected)}/{top_k} execution-ready source+model pairs were available.")
    for row in selected:
        cap = capabilities.get(row["model_id"], {})
        if cap.get("data_constraints", {}).get("compatible_contract") == TRAINED_CONTRACT:
            review_reasons.append(f"{row['model_id']} uses a trained SEA-AD 140-gene head.")
        if not row["execution_ready"]:
            review_reasons.append(f"{row['model_id']}:{row['source_id']} is not execution-ready.")
    review = {
        "status": "passed" if not review_reasons and llm_selection.get("status") in {"passed", "disabled"} else "needs_review",
        "reasons": review_reasons,
        "rules": [
            "species mismatch is not a rejection reason",
            "trained SEA-AD 140-gene heads are excluded by default",
            "selected items must include model_id/source_id/reference_path/capability_yaml/shared_genes",
            "stage2 selection must be made by LLM unless llm_mode=off is explicit",
            "LLM planner selections must pass an independent LLM reviewer audit",
            "candidate score is equal evidence-group rank aggregation, not a manually weighted linear objective",
        ],
        "llm_selection": llm_selection,
    }
    write_json(review, out_dir / "review.json")

    selected_model_ids = [row["model_id"] for row in selected]
    model_id_payload = {
        "dataset_id": dataset_id,
        "selected_model_ids": selected_model_ids,
        "selected_pairs": [
            {"rank": row["rank"], "model_id": row["model_id"], "source_id": row["source_id"], "score": row["score"]}
            for row in selected
        ],
    }
    _write_yaml(model_id_payload, out_dir / "selected_model_ids.yaml")

    plan = {
        "dataset_id": dataset_id,
        "query_path": query_profile["input_path"],
        "query_adapter": query_profile["query_adapter"],
        "query_profile_path": str(query_profile_path),
        "artifact_bundle": str(artifact_bundle),
        "prepared_source_root": str(prepared_source_root),
        "capability_dir": str(capability_dir),
        "selected_model_ids": selected_model_ids,
        "selected_pairs": selected,
        "rejected_pairs": rejected,
        "execution_defaults": {
            "max_query_cells": int(max_query_cells),
            "max_reference_cells": int(max_reference_cells),
            "min_shared_genes": int(min_shared_genes),
            "k": int(k),
            "seed": int(seed),
            "device": device,
            "batch_size": int(batch_size),
        },
        "review_status": review,
        "selection_policy": {
            "selector": (
                "llm_react_one_by_one"
                if llm_mode != "off" and normalized_selection_strategy == "one_by_one"
                else "llm_react_batch"
                if llm_mode != "off"
                else "deterministic_fallback"
            ),
            "num_models": int(top_k),
            "selection_strategy": normalized_selection_strategy,
            "selection_strategy_requested": selection_strategy,
            "selection_objective": selection_objective,
            "selection_objective_requested": requested_selection_objective,
            "single_rule_for_top1_and_topn": True,
            "rank1_is_top1_ablation": True,
            "planner_agent": "llm_stage2_planner" if llm_mode != "off" else "deterministic_rank_fallback",
            "reviewer_agent": "llm_stage2_reviewer" if llm_mode != "off" else "deterministic_reviewer",
            "rank_aggregation_method": RANK_AGGREGATION_METHOD,
            "rank_aggregation_axis_weighting": "none; axes are grouped before voting",
            "rank_aggregation_group_weighting": "equal; one evidence-group vote per group",
            "rank_aggregation_groups": [
                {
                    "group": group["group"],
                    "description": group["description"],
                    "axes": [{"axis": axis, "description": description} for axis, description in group["axes"]],
                }
                for group in RANK_AGGREGATION_GROUPS
            ],
            "rank_aggregation_axes": [
                {"axis": axis, "description": description} for axis, description in RANK_AGGREGATION_AXES
            ],
            "iterative_exclude_scope": iterative_exclude_scope,
            "llm_mode": llm_mode,
            "llm_model": llm_model if llm_mode != "off" else "",
            "deterministic_similarity_is_evidence_only": llm_mode != "off",
            "query_visibility": "gene_names_only",
            "query_identifier_visible_to_selector": False,
            "query_path_visible_to_selector": False,
            "query_adapter_visible_to_selector": False,
            "query_labels_visible_to_selector": False,
            "query_expression_visible_to_selector": False,
            "excluded_model_ids": sorted(excluded),
        },
    }
    plan_path = _write_yaml(plan, out_dir / "selected_execution_plan.yaml")

    report_lines = [
        f"# Stage-2 Selection Report: {dataset_id}",
        "",
        "Species was not used as a filter or penalty.",
        "Selection policy: pre-execution LLM planner with query visibility restricted to gene names only.",
        "Hidden from selector: query dataset id, query path, query adapter, labels/counts, expression summaries, sample/donor composition, query-result evidence, and Stage4 results.",
        f"Selection strategy: `{normalized_selection_strategy}`.",
        f"Selection objective: `{selection_objective}`.",
        f"Requested objective: `{requested_selection_objective}`.",
        f"Planner/reviewer agents: `{'llm_stage2_planner' if llm_mode != 'off' else 'deterministic_rank_fallback'}` / `{'llm_stage2_reviewer' if llm_mode != 'off' else 'deterministic_reviewer'}`.",
        f"Rank aggregation: `{RANK_AGGREGATION_METHOD}` with equal evidence-group votes.",
        f"Iterative exclude scope: `{iterative_exclude_scope}`.",
        f"Prepared source root: `{prepared_source_root}`.",
        f"Selection mode: `{llm_mode}`.",
        f"LLM selector status: `{llm_selection.get('status', 'unknown')}`.",
        f"LLM trace dir: `{llm_selection.get('trace_dir', '')}`.",
        "",
        "## Selected Pairs",
        "",
        "| rank | model_id | source_id | score | shared_genes | query_gene_coverage | score_source | rationale |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in selected:
        report_lines.append(
            f"| {row['rank']} | {row['model_id']} | {row['source_id']} | {row['score']:.4f} | "
            f"{row['shared_genes']} | {row.get('query_gene_coverage', 0.0):.4f} | "
            f"{row.get('score_source', '')} | {row['rationale']} |"
        )
    report_lines.extend(
        [
            "",
            "## Selected Source Benchmark Evidence",
            "",
            "| model_id | source_id | composite | mean macro-F1 | LCB macro-F1 | SE | baseline mean | variant mean | robustness median | robustness q25 | source description |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in selected:
        description = str(row.get("source_dataset_description", "")).replace("|", "/")
        report_lines.append(
            f"| {row['model_id']} | {row['source_id']} | {row.get('source_dataset_composite_score', 0.0):.4f} | "
            f"{row.get('source_model_macro_f1', 0.0):.4f} | {row.get('source_model_macro_f1_lcb', 0.0):.4f} | "
            f"{row.get('source_model_macro_f1_se', 0.0):.4f} | "
            f"{row.get('baseline_macro_f1_mean', 0.0):.4f} | {row.get('variant_macro_f1_mean', 0.0):.4f} | "
            f"{row.get('robustness_median', 0.0):.4f} | {row.get('robustness_q25', 0.0):.4f} | {description} |"
        )
    report_lines.extend(
        [
            "",
            "## Top Source Similarities",
            "",
            "| source_id | similarity | shared_genes | query_coverage | min_panel_coverage | jaccard | expression_score | celltype_score |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(similarities, key=lambda item: item["similarity_score"], reverse=True):
        report_lines.append(
            f"| {row['source_id']} | {row['similarity_score']:.4f} | {row['shared_genes']} | "
            f"{row.get('query_gene_coverage', 0.0):.4f} | {row.get('min_panel_coverage', 0.0):.4f} | "
            f"{row.get('gene_set_jaccard', 0.0):.4f} | {row['expression_score']} | {row['celltype_score']} |"
        )
    report_path = out_dir / "selection_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "dataset_id": dataset_id,
        "output_dir": str(out_dir),
        "selected_execution_plan": str(plan_path),
        "prepared_source_root": str(prepared_source_root),
        "num_models": int(top_k),
        "selection_strategy": normalized_selection_strategy,
        "selection_strategy_requested": selection_strategy,
        "selection_objective": selection_objective,
        "selection_objective_requested": requested_selection_objective,
        "iterative_exclude_scope": iterative_exclude_scope,
        "selected_model_ids": selected_model_ids,
        "source_similarity_csv": str(out_dir / "source_similarity.csv"),
        "candidate_pairs_csv": str(out_dir / "candidate_pairs.csv"),
        "review_json": str(out_dir / "review.json"),
        "selection_report": str(report_path),
        "llm_selection": llm_selection,
        "llm_trace_dir": llm_selection.get("trace_dir", ""),
    }


def _prediction_writer(path: Path) -> tuple[Any, csv.DictWriter]:
    handle = path.open("w", newline="", encoding="utf-8")
    columns = [
        "dataset_id",
        "model_id",
        "source_id",
        "method",
        "task",
        "cell_id",
        "sample_id",
        "true_label",
        "pred_label",
        "confidence",
    ]
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    return handle, writer


def _source_species(reference_path: str | Path) -> str:
    manifest = Path(reference_path) / "source_manifest.json"
    if not manifest.exists():
        return ""
    return str(read_json(manifest).get("species", ""))


def _metric_for_task(
    *,
    dataset_id: str,
    model_id: str,
    source_id: str,
    method: str,
    embedding_method: str,
    transfer_method: str,
    task: str,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    n_shared_genes: int,
    n_reference_cells: int,
    n_query_cells: int,
) -> dict[str, Any]:
    true_labels = true_labels.astype(str)
    pred_labels = pred_labels.astype(str)
    ref_label_set = set(ref_labels.astype(str))
    true_in_ref = np.asarray([x in ref_label_set for x in true_labels], dtype=bool)
    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "source_id": source_id,
        "method": method,
        "embedding_method": embedding_method,
        "transfer_method": transfer_method,
        "task": task,
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
        "label_overlap_fraction": float(true_in_ref.mean()) if len(true_in_ref) else 0.0,
        "n_reference_cells": int(n_reference_cells),
        "n_query_cells": int(n_query_cells),
        "n_shared_genes": int(n_shared_genes),
        "n_ref_labels": int(len(ref_label_set)),
        "n_true_labels": int(len(set(true_labels))),
    }


def run_cross_species_plan(
    *,
    plan_path: str | Path,
    output_dir: str | Path | None = None,
    max_query_cells: int = 0,
    max_reference_cells: int = 0,
    min_shared_genes: int = 0,
    k: int = 0,
    device: str | None = None,
    batch_size: int = 0,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    plan = _read_yaml(plan_path)
    defaults = plan.get("execution_defaults", {})
    dataset_id = plan["dataset_id"]
    out_dir = ensure_dir(output_dir or (plan_path.parent / "subset_smoke"))

    def _resolve_limit(value: int, default: Any, fallback: int) -> int:
        value = int(value)
        if value < 0:
            return 0
        if value == 0:
            default_value = int(default or fallback)
            return 0 if default_value < 0 else default_value
        return value

    max_query_cells = _resolve_limit(max_query_cells, defaults.get("max_query_cells", 5_000), 5_000)
    max_reference_cells = _resolve_limit(max_reference_cells, defaults.get("max_reference_cells", 1_000), 1_000)
    min_shared_genes = int(min_shared_genes or defaults.get("min_shared_genes", 30))
    k = int(k or defaults.get("k", 15))
    seed = int(defaults.get("seed", 3028))
    device = defaults.get("device", "") if device is None else device
    batch_size = int(batch_size or defaults.get("batch_size", 16))

    query = load_query_bundle(plan["query_path"], dataset_id=dataset_id, max_cells=max_query_cells, seed=seed).bundle
    metrics_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    predictions_path = out_dir / "predictions.csv"
    pred_handle, pred_writer = _prediction_writer(predictions_path)
    prediction_count = 0

    try:
        for pair in plan.get("selected_pairs", []):
            model_id = pair["model_id"]
            source_id = pair["source_id"]
            method = pair["method"]
            embedding_method = pair["embedding_method"]
            transfer_method = pair["transfer_method"]
            try:
                ref, _ = _load_reference_from_standard_bundle(
                    pair["reference_path"],
                    genes=query.genes,
                    max_cells=max_reference_cells,
                    seed=seed,
                )
                ref_aligned, query_aligned, shared_genes = _align_reference_and_query(
                    ref,
                    query,
                    min_shared_genes=min_shared_genes,
                )
                adapter = EmbeddingAdapter(
                    base_method=embedding_method,
                    genes=shared_genes,
                    species=_source_species(pair["reference_path"]),
                    device=device or "",
                    batch_size=batch_size,
                )
                adapter.fit()
                ref_emb = adapter.transform(ref_aligned.X)
                query_emb = adapter.transform(query_aligned.X)
            except Exception as exc:
                skip_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "source_id": source_id,
                        "method": method,
                        "task": "",
                        "stage": "embedding_or_alignment",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
                continue

            for task in ("native_label", "coarse_label"):
                if task not in ref_aligned.obs.columns or task not in query_aligned.obs.columns:
                    skip_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "model_id": model_id,
                            "source_id": source_id,
                            "method": method,
                            "task": task,
                            "stage": "labels",
                            "reason": f"Missing labels for task={task}",
                            "traceback": "",
                        }
                    )
                    continue

                ref_labels = ref_aligned.obs[task].astype(str).to_numpy()
                true_labels = query_aligned.obs[task].astype(str).to_numpy()
                usable = ~pd.Series(true_labels).isin(["", "nan", "None", "unknown"]).to_numpy()
                if int(usable.sum()) == 0:
                    skip_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "model_id": model_id,
                            "source_id": source_id,
                            "method": method,
                            "task": task,
                            "stage": "labels",
                            "reason": "No usable query truth labels.",
                            "traceback": "",
                        }
                    )
                    continue

                if transfer_method == "knn":
                    pred, conf = _vote_knn(ref_emb, query_emb, ref_labels, k=k)
                elif transfer_method == "prototype":
                    pred, conf = _prototype_predict(ref_emb, query_emb, ref_labels)
                else:
                    raise ValueError(f"Unsupported transfer method: {transfer_method}")

                metrics_rows.append(
                    _metric_for_task(
                        dataset_id=dataset_id,
                        model_id=model_id,
                        source_id=source_id,
                        method=method,
                        embedding_method=embedding_method,
                        transfer_method=transfer_method,
                        task=task,
                        true_labels=true_labels[usable],
                        pred_labels=pred[usable],
                        ref_labels=ref_labels,
                        n_shared_genes=len(shared_genes),
                        n_reference_cells=ref_aligned.X.shape[0],
                        n_query_cells=int(usable.sum()),
                    )
                )

                obs = query_aligned.obs.iloc[np.flatnonzero(usable)].reset_index(drop=True)
                cell_ids = obs["cell_id"].astype(str).to_numpy() if "cell_id" in obs.columns else np.asarray(
                    [f"cell_{idx}" for idx in range(len(obs))],
                    dtype=str,
                )
                sample_ids = obs["sample_id"].astype(str).to_numpy() if "sample_id" in obs.columns else np.asarray(
                    [""] * len(obs),
                    dtype=str,
                )
                # Full Kukanja runs write >1M rows per model. Vectorized CSV
                # appends are much faster than Python-row csv.DictWriter loops.
                pd.DataFrame(
                    {
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "source_id": source_id,
                        "method": method,
                        "task": task,
                        "cell_id": cell_ids,
                        "sample_id": sample_ids,
                        "true_label": true_labels[usable].astype(str),
                        "pred_label": pred[usable].astype(str),
                        "confidence": conf[usable].astype(float),
                    }
                ).to_csv(pred_handle, header=False, index=False)
                prediction_count += len(obs)
    finally:
        pred_handle.close()

    metric_columns = [
        "dataset_id",
        "model_id",
        "source_id",
        "method",
        "embedding_method",
        "transfer_method",
        "task",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "label_overlap_fraction",
        "n_reference_cells",
        "n_query_cells",
        "n_shared_genes",
        "n_ref_labels",
        "n_true_labels",
    ]
    skip_columns = ["dataset_id", "model_id", "source_id", "method", "task", "stage", "reason", "traceback"]
    metrics = pd.DataFrame(metrics_rows, columns=metric_columns)
    skips = pd.DataFrame(skip_rows, columns=skip_columns)
    metrics_path = out_dir / "metrics.csv"
    skips_path = out_dir / "skips_and_failures.csv"
    metrics.to_csv(metrics_path, index=False)
    skips.to_csv(skips_path, index=False)
    summary = {
        "dataset_id": dataset_id,
        "plan_path": str(plan_path),
        "output_dir": str(out_dir),
        "max_query_cells": max_query_cells,
        "max_reference_cells": max_reference_cells,
        "min_shared_genes": min_shared_genes,
        "k": k,
        "device": device,
        "batch_size": batch_size,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "skips_path": str(skips_path),
        "n_metric_rows": int(len(metrics)),
        "n_prediction_rows": int(prediction_count),
        "n_skips": int(len(skips)),
    }
    write_json(summary, out_dir / "run_summary.json")
    return summary
