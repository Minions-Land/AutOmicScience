from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def normalize_gene_symbol(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text.upper()


def normalize_label(value: Any) -> str:
    text = str(value).strip()
    return text if text else "__empty__"


def resolve_device(device: str) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def upper_set(values: list[str]) -> set[str]:
    return {normalize_gene_symbol(value) for value in values if normalize_gene_symbol(value)}


def row_normalize(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (embeddings / norms).astype(np.float32)


def cap_indices(indices: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    if max_cells <= 0 or indices.size <= max_cells:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=int(max_cells), replace=False)
    return np.sort(chosen.astype(np.int64))


def sample_reference_indices(
    obs: pd.DataFrame,
    label_key: str,
    max_total: int,
    max_per_label: int,
    seed: int,
) -> np.ndarray:
    if label_key not in obs.columns:
        raise KeyError(f"reference_label_key {label_key!r} not found in reference obs columns")
    labels = obs[label_key].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    chosen_chunks: list[np.ndarray] = []
    for offset, label_value in enumerate(sorted(set(labels))):
        label_indices = np.where(labels == label_value)[0].astype(np.int64)
        if max_per_label > 0 and label_indices.size > max_per_label:
            local_rng = np.random.default_rng(seed + offset + 17)
            label_indices = np.sort(local_rng.choice(label_indices, size=max_per_label, replace=False).astype(np.int64))
        chosen_chunks.append(label_indices)
    if not chosen_chunks:
        return np.zeros((0,), dtype=np.int64)
    combined = np.sort(np.concatenate(chosen_chunks).astype(np.int64))
    if max_total > 0 and combined.size > max_total:
        combined = np.sort(rng.choice(combined, size=max_total, replace=False).astype(np.int64))
    return combined


def load_cached_reference_indices(ref_obs_all: pd.DataFrame, cache_path: str | Path) -> np.ndarray:
    cache_df = pd.read_csv(cache_path)
    if "obs_name" in cache_df.columns:
        obs_names = cache_df["obs_name"].astype(str).tolist()
    elif len(cache_df.columns) >= 1:
        obs_names = cache_df.iloc[:, 0].astype(str).tolist()
    else:
        raise RuntimeError(f"Cached reference obs-name file has no columns: {cache_path}")

    if not obs_names:
        raise RuntimeError(f"Cached reference obs-name file is empty: {cache_path}")

    indexer = ref_obs_all.index.get_indexer(pd.Index(obs_names, dtype=object))
    missing_mask = indexer < 0
    if bool(np.any(missing_mask)):
        missing = [obs_names[i] for i, flag in enumerate(missing_mask.tolist()) if flag][:10]
        raise RuntimeError(
            f"Cached reference subset contains {int(missing_mask.sum())} obs names missing from reference; "
            f"examples: {missing}"
        )
    return np.asarray(indexer, dtype=np.int64)


def get_var_frame(adata: ad.AnnData, *, x_source: str, layer_name: str) -> pd.DataFrame:
    if x_source == "layers" and not layer_name:
        raise ValueError('layer_name must be provided when x_source == "layers".')
    if x_source == "raw":
        if adata.raw is None:
            raise ValueError('Requested x_source="raw" but adata.raw is None.')
        return adata.raw.var.copy()
    return adata.var.copy()


def candidate_gene_name_columns(var_frame: pd.DataFrame, preferred_key: str) -> list[str]:
    priority = [preferred_key] if preferred_key else []
    priority.extend(
        [
            "gene_name",
            "gene_symbol",
            "symbol",
            "gene",
            "feature_name",
            "gene_short_name",
            "features",
            "feature",
            "gene_ids",
        ]
    )
    ordered: list[str] = []
    seen: set[str] = set()

    def add(column_name: str) -> None:
        if column_name and column_name not in seen and column_name in var_frame.columns:
            seen.add(column_name)
            ordered.append(column_name)

    for column_name in priority:
        add(column_name)
    for column_name in var_frame.columns:
        series = var_frame[column_name]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if str(column_name) not in seen:
            seen.add(str(column_name))
            ordered.append(str(column_name))
    return ordered


@dataclass
class GeneSelection:
    gene_names: list[str]
    gene_keep_mask: np.ndarray
    source_key: str
    overlap_count: int
    candidate_scores: list[dict[str, Any]]


@dataclass
class FixedPanelSelection:
    panel_gene_names: list[str]
    source_key: str
    overlap_count: int
    candidate_scores: list[dict[str, Any]]
    source_column_indices: np.ndarray
    panel_output_positions: np.ndarray


def select_gene_names_by_vocab(
    adata: ad.AnnData,
    *,
    preferred_key: str,
    x_source: str,
    layer_name: str,
    vocab_upper: set[str],
) -> GeneSelection:
    var_frame = get_var_frame(adata, x_source=x_source, layer_name=layer_name)
    candidate_payloads: list[tuple[str, list[str], np.ndarray, int]] = []

    var_names = [str(item) for item in var_frame.index.tolist()]
    var_norm = [normalize_gene_symbol(item) for item in var_names]
    var_keep_mask = np.asarray([bool(item) and not str(raw).startswith("Blank") for item, raw in zip(var_norm, var_names)], dtype=bool)
    var_overlap = sum(item in vocab_upper for item, keep in zip(var_norm, var_keep_mask) if keep)
    candidate_payloads.append(("var_names", var_norm, var_keep_mask, int(var_overlap)))

    for column_name in candidate_gene_name_columns(var_frame, preferred_key):
        raw_values = [str(item) for item in var_frame[column_name].tolist()]
        norm_values = [normalize_gene_symbol(item) for item in raw_values]
        keep_mask = np.asarray([bool(item) and not str(raw).startswith("Blank") for item, raw in zip(norm_values, raw_values)], dtype=bool)
        overlap_count = sum(item in vocab_upper for item, keep in zip(norm_values, keep_mask) if keep)
        candidate_payloads.append((str(column_name), norm_values, keep_mask, int(overlap_count)))

    candidate_scores = [
        {
            "source_key": source_key,
            "overlap_count": int(overlap_count),
            "n_genes_total": int(len(names)),
            "n_genes_after_blank_filter": int(keep_mask.sum()),
        }
        for source_key, names, keep_mask, overlap_count in candidate_payloads
    ]
    best_source_key, best_names, best_keep_mask, best_overlap = max(
        candidate_payloads,
        key=lambda item: (item[3], int(item[0] == preferred_key), int(item[0] == "var_names")),
    )
    filtered_gene_names = [name for name, keep in zip(best_names, best_keep_mask) if keep]
    return GeneSelection(
        gene_names=filtered_gene_names,
        gene_keep_mask=best_keep_mask,
        source_key=best_source_key,
        overlap_count=int(best_overlap),
        candidate_scores=candidate_scores,
    )


def select_fixed_panel_gene_names(
    adata: ad.AnnData,
    *,
    preferred_key: str,
    x_source: str,
    layer_name: str,
    panel_gene_names: list[str],
) -> FixedPanelSelection:
    var_frame = get_var_frame(adata, x_source=x_source, layer_name=layer_name)
    panel_norm = [normalize_gene_symbol(item) for item in panel_gene_names]
    candidate_payloads: list[tuple[str, list[str], int]] = []
    var_names = [str(item) for item in var_frame.index.tolist()]
    candidate_payloads.append(("var_names", var_names, 0))
    for column_name in candidate_gene_name_columns(var_frame, preferred_key):
        candidate_payloads.append((str(column_name), [str(item) for item in var_frame[column_name].tolist()], 0))

    resolved_payloads: list[tuple[str, list[str], dict[str, int], int]] = []
    for source_key, raw_names, _ in candidate_payloads:
        norm_names = [normalize_gene_symbol(item) for item in raw_names]
        index_by_name: dict[str, int] = {}
        for idx, (raw_name, norm_name) in enumerate(zip(raw_names, norm_names)):
            if not norm_name or str(raw_name).startswith("Blank"):
                continue
            index_by_name.setdefault(norm_name, idx)
        overlap_count = sum(1 for gene_name in panel_norm if gene_name in index_by_name)
        resolved_payloads.append((source_key, raw_names, index_by_name, int(overlap_count)))

    candidate_scores = [
        {
            "source_key": source_key,
            "overlap_count": int(overlap_count),
            "n_genes_total": int(len(raw_names)),
            "n_panel_total": int(len(panel_gene_names)),
        }
        for source_key, raw_names, _, overlap_count in resolved_payloads
    ]
    best_source_key, _, best_index_by_name, best_overlap = max(
        resolved_payloads,
        key=lambda item: (item[3], int(item[0] == preferred_key), int(item[0] == "var_names")),
    )

    source_column_indices: list[int] = []
    panel_output_positions: list[int] = []
    for panel_idx, norm_name in enumerate(panel_norm):
        source_idx = best_index_by_name.get(norm_name)
        if source_idx is None:
            continue
        source_column_indices.append(int(source_idx))
        panel_output_positions.append(int(panel_idx))

    return FixedPanelSelection(
        panel_gene_names=list(panel_gene_names),
        source_key=best_source_key,
        overlap_count=int(best_overlap),
        candidate_scores=candidate_scores,
        source_column_indices=np.asarray(source_column_indices, dtype=np.int64),
        panel_output_positions=np.asarray(panel_output_positions, dtype=np.int64),
    )


def adata_matrix_slice(
    adata: ad.AnnData,
    row_indices: np.ndarray,
    *,
    x_source: str,
    layer_name: str,
    gene_keep_mask: np.ndarray,
) -> np.ndarray:
    if x_source == "raw":
        matrix = adata.raw[row_indices, :].X
    elif x_source == "layers":
        matrix = adata.layers[layer_name][row_indices, :]
    else:
        matrix = adata.X[row_indices, :]
    if sp.issparse(matrix):
        matrix = matrix[:, gene_keep_mask].toarray()
    else:
        matrix = np.asarray(matrix[:, gene_keep_mask])
    return np.asarray(matrix, dtype=np.float32)


def adata_matrix_slice_fixed_panel(
    adata: ad.AnnData,
    row_indices: np.ndarray,
    *,
    x_source: str,
    layer_name: str,
    source_column_indices: np.ndarray,
    panel_output_positions: np.ndarray,
    panel_size: int,
) -> np.ndarray:
    if source_column_indices.size == 0:
        return np.zeros((int(len(row_indices)), int(panel_size)), dtype=np.float32)
    if x_source == "raw":
        matrix = adata.raw[row_indices, source_column_indices].X
    elif x_source == "layers":
        matrix = adata.layers[layer_name][row_indices, :][:, source_column_indices]
    else:
        matrix = adata.X[row_indices, :][:, source_column_indices]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    aligned = np.zeros((int(len(row_indices)), int(panel_size)), dtype=np.float32)
    aligned[:, panel_output_positions] = matrix
    return aligned


def coverage_payload(n_mapped: int, n_total: int) -> dict[str, Any]:
    return {
        "n_mapped": int(n_mapped),
        "n_total": int(n_total),
        "coverage_ratio": float(int(n_mapped) / max(1, int(n_total))),
    }

