from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import io as spio
from scipy import sparse


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    return path


def read_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False, allow_unicode=True)
    return path


def normalize_gene_name(value: Any) -> str:
    return str(value).strip().upper()


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    present = set(columns)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def to_csr(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def write_standard_bundle(
    *,
    counts_cell_by_gene: sparse.spmatrix,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    output_dir: str | Path,
    source_metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    counts_path = output_dir / "counts.mtx"
    obs_path = output_dir / "obs.csv"
    var_path = output_dir / "var.csv"

    obs_out = obs.copy()
    var_out = var.copy()
    if "cell_id" not in obs_out.columns:
        obs_out.insert(0, "cell_id", obs_out.index.astype(str))
    if "feature_id" not in var_out.columns:
        var_out.insert(0, "feature_id", var_out.index.astype(str))

    spio.mmwrite(str(counts_path), to_csr(counts_cell_by_gene).tocoo())
    obs_out.to_csv(obs_path, index=False)
    var_out.to_csv(var_path, index=False)

    manifest = {
        "counts_path": str(counts_path),
        "obs_path": str(obs_path),
        "var_path": str(var_path),
        "count_orientation": "cell_by_gene",
        "cell_id_col": "cell_id",
        "feature_id_col": "feature_id",
        "n_obs": int(obs.shape[0]),
        "n_vars": int(var.shape[0]),
        **(source_metadata or {}),
    }
    write_json(manifest, output_dir / "source_manifest.json")
    return output_dir


def read_standard_bundle(path: str | Path, orientation: str = "cell_by_gene"):
    path = Path(path)
    counts = spio.mmread(str(path / "counts.mtx"))
    counts = to_csr(counts)
    if orientation == "gene_by_cell":
        counts = counts.transpose().tocsr()
    elif orientation != "cell_by_gene":
        raise ValueError(f"Unsupported orientation: {orientation}")
    obs = pd.read_csv(path / "obs.csv")
    var = pd.read_csv(path / "var.csv")
    if "cell_id" in obs.columns:
        obs = obs.set_index("cell_id", drop=False)
    if "feature_id" in var.columns:
        var = var.set_index("feature_id", drop=False)
    return counts, obs, var


def stratified_indices(
    labels: pd.Series,
    *,
    max_cells: int,
    min_per_group: int = 0,
    seed: int = 3028,
) -> np.ndarray:
    if max_cells <= 0 or max_cells >= len(labels):
        return np.arange(len(labels), dtype=np.int64)

    rng = np.random.default_rng(seed)
    labels = labels.astype(str).fillna("__missing__")
    group_to_indices = {
        group: np.flatnonzero(labels.to_numpy() == group)
        for group in labels.value_counts().index.tolist()
    }
    chosen: list[np.ndarray] = []
    remaining_budget = int(max_cells)
    for group, idx in group_to_indices.items():
        if remaining_budget <= 0:
            break
        take = min(len(idx), min_per_group, remaining_budget)
        if take > 0:
            chosen.append(rng.choice(idx, size=take, replace=False))
            remaining_budget -= take
            group_to_indices[group] = np.setdiff1d(idx, chosen[-1], assume_unique=False)

    if remaining_budget > 0:
        pool = np.concatenate([idx for idx in group_to_indices.values() if len(idx)])
        if len(pool):
            take = min(remaining_budget, len(pool))
            chosen.append(rng.choice(pool, size=take, replace=False))

    if not chosen:
        return np.array([], dtype=np.int64)
    out = np.concatenate(chosen).astype(np.int64)
    out.sort()
    return out
