#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

from novaeve_bio import paths
from novaeve_bio.eval.label_transfer import EmbeddingAdapter, MatrixBundle, _prototype_predict, _split_method, _vote_knn
from novaeve_bio.io import ensure_dir, normalize_gene_name, read_json, read_standard_bundle, stratified_indices


DEFAULT_METHODS = [
    "geneformer_raw_knn",
    "nicheformer_raw_knn",
    "scgpt_brain_raw_knn",
    "scgpt_human_raw_knn",
    "uce_33l_raw_knn",
    "uce_4l_raw_knn",
]


def _feature_genes(var: pd.DataFrame) -> list[str]:
    if "feature_id" in var.columns:
        values = var["feature_id"].astype(str).tolist()
    else:
        values = var.index.astype(str).tolist()
    return [normalize_gene_name(x) for x in values]


def _load_bundle(source_dir: Path, *, max_cells: int, seed: int) -> tuple[MatrixBundle, dict[str, Any]]:
    X, obs, var = read_standard_bundle(source_dir)
    manifest_path = source_dir / "source_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    fine_col = "native_label" if "native_label" in obs.columns else None
    coarse_col = "coarse_label" if "coarse_label" in obs.columns else None
    if fine_col is None:
        for candidate in ("cell_type", "Supertype", "subclass", "cluster"):
            if candidate in obs.columns:
                fine_col = candidate
                break
    if fine_col is None:
        raise ValueError(f"No usable label column in {source_dir}")
    labels = obs[fine_col].astype(str).fillna("unknown")
    usable = ~labels.isin(["", "nan", "None", "unknown"]).to_numpy()
    if int(usable.sum()) == 0:
        raise ValueError(f"No usable labels in {source_dir}")
    X = X[usable, :].tocsr()
    obs = obs.iloc[np.flatnonzero(usable)].reset_index(drop=True)
    labels = obs[fine_col].astype(str).fillna("unknown")
    idx = stratified_indices(labels, max_cells=max_cells, min_per_group=5, seed=seed)
    X = X[idx, :].tocsr()
    obs = obs.iloc[idx].reset_index(drop=True)
    obs["native_label"] = obs[fine_col].astype(str)
    if coarse_col and coarse_col in obs.columns:
        obs["coarse_label"] = obs[coarse_col].astype(str)
    else:
        obs["coarse_label"] = obs["native_label"]
    if "cell_id" not in obs.columns:
        obs.insert(0, "cell_id", [f"{source_dir.name}_cell_{i}" for i in range(len(obs))])
    genes = _feature_genes(var)
    return MatrixBundle(X=X, obs=obs, var=var, genes=genes), manifest


def _split_indices(labels: pd.Series, *, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    labels = labels.astype(str).reset_index(drop=True)
    for _, group_idx in labels.groupby(labels).groups.items():
        idx = np.asarray(list(group_idx), dtype=np.int64)
        rng.shuffle(idx)
        if len(idx) < 2:
            train_parts.append(idx)
            continue
        n_train = int(round(len(idx) * train_fraction))
        n_train = max(1, min(len(idx) - 1, n_train))
        train_parts.append(idx[:n_train])
        test_parts.append(idx[n_train:])
    train = np.sort(np.concatenate(train_parts)) if train_parts else np.asarray([], dtype=np.int64)
    test = np.sort(np.concatenate(test_parts)) if test_parts else np.asarray([], dtype=np.int64)
    if len(test) == 0:
        raise ValueError("No heldout cells after stratified split")
    return train, test


def _metric_row(
    *,
    model_id: str,
    source_id: str,
    source_dir: Path,
    species: str,
    task: str,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    ref_labels: np.ndarray,
    confidence: np.ndarray,
    n_reference_cells: int,
    n_query_cells: int,
    n_genes: int,
) -> dict[str, Any]:
    ref_set = set(ref_labels.astype(str))
    true_labels = true_labels.astype(str)
    pred_labels = pred_labels.astype(str)
    in_ref = np.asarray([x in ref_set for x in true_labels], dtype=bool)
    return {
        "evaluation_type": "reference_self_holdout_no_query",
        "source_group": source_id,
        "source_path": str(source_dir),
        "model": model_id,
        "family": model_id.split("_", 1)[0],
        "task": task,
        "macro_f1": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "weighted_f1": float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
        "confidence_mean": float(np.mean(confidence)) if len(confidence) else 0.0,
        "label_overlap_fraction": float(in_ref.mean()) if len(in_ref) else 0.0,
        "n_reference_cells": int(n_reference_cells),
        "n_query_cells": int(n_query_cells),
        "n_genes": int(n_genes),
        "n_ref_labels": int(len(ref_set)),
        "n_true_labels": int(len(set(true_labels))),
        "species": species,
    }


def run_source(
    *,
    source_dir: Path,
    methods: list[str],
    max_cells: int,
    train_fraction: float,
    k: int,
    device: str,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_id = source_dir.name
    bundle, manifest = _load_bundle(source_dir, max_cells=max_cells, seed=seed)
    species = str(manifest.get("species") or "")
    train_idx, test_idx = _split_indices(bundle.obs["native_label"], train_fraction=train_fraction, seed=seed)
    ref = MatrixBundle(
        X=bundle.X[train_idx, :].tocsr(),
        obs=bundle.obs.iloc[train_idx].reset_index(drop=True),
        var=bundle.var,
        genes=bundle.genes,
    )
    query = MatrixBundle(
        X=bundle.X[test_idx, :].tocsr(),
        obs=bundle.obs.iloc[test_idx].reset_index(drop=True),
        var=bundle.var,
        genes=bundle.genes,
    )
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for method in methods:
        try:
            embedding_method, transfer_method = _split_method(method)
            adapter = EmbeddingAdapter(
                base_method=embedding_method,
                genes=bundle.genes,
                species=species,
                device=device,
                batch_size=batch_size,
            )
            adapter.fit()
            ref_emb = adapter.transform(ref.X)
            query_emb = adapter.transform(query.X)
            for task in ("native_label", "coarse_label"):
                ref_labels = ref.obs[task].astype(str).to_numpy()
                true_labels = query.obs[task].astype(str).to_numpy()
                if transfer_method == "knn":
                    pred, conf = _vote_knn(ref_emb, query_emb, ref_labels, k=k)
                elif transfer_method == "prototype":
                    pred, conf = _prototype_predict(ref_emb, query_emb, ref_labels)
                else:
                    raise ValueError(f"Unsupported transfer method: {transfer_method}")
                rows.append(
                    _metric_row(
                        model_id=method,
                        source_id=source_id,
                        source_dir=source_dir,
                        species=species,
                        task=task,
                        true_labels=true_labels,
                        pred_labels=pred,
                        ref_labels=ref_labels,
                        confidence=conf,
                        n_reference_cells=ref.X.shape[0],
                        n_query_cells=query.X.shape[0],
                        n_genes=len(bundle.genes),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            skips.append(
                {
                    "evaluation_type": "reference_self_holdout_no_query",
                    "source_group": source_id,
                    "model": method,
                    "stage": "self_holdout",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                }
            )
    return rows, skips


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-query self-holdout benchmark on prepared reference sources.")
    parser.add_argument("--prepared-source-root", default=str(paths.DATA_DIR / "prepared_sources"))
    parser.add_argument("--output-dir", default=str(paths.RUNS_DIR / "reference_self_holdout"))
    parser.add_argument("--source", action="append", dest="sources", default=[])
    parser.add_argument("--method", action="append", dest="methods", default=[])
    parser.add_argument("--max-cells", type=int, default=12000)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3028)
    args = parser.parse_args()

    root = Path(args.prepared_source_root)
    out_dir = ensure_dir(args.output_dir)
    methods = args.methods or DEFAULT_METHODS
    source_ids = args.sources or sorted(p.name for p in root.glob("*/source_manifest.json"))
    all_rows: list[dict[str, Any]] = []
    all_skips: list[dict[str, Any]] = []
    for source_id in source_ids:
        source_dir = root / source_id
        rows, skips = run_source(
            source_dir=source_dir,
            methods=methods,
            max_cells=args.max_cells,
            train_fraction=args.train_fraction,
            k=args.k,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_rows.extend(rows)
        all_skips.extend(skips)
        pd.DataFrame(all_rows).to_csv(out_dir / "reference_self_holdout_scores.csv", index=False)
        pd.DataFrame(all_skips).to_csv(out_dir / "reference_self_holdout_skips.csv", index=False)
    summary = {
        "prepared_source_root": str(root),
        "output_dir": str(out_dir),
        "sources": source_ids,
        "methods": methods,
        "max_cells": args.max_cells,
        "train_fraction": args.train_fraction,
        "k": args.k,
        "seed": args.seed,
        "score_rows": len(all_rows),
        "skip_rows": len(all_skips),
        "scores_csv": str(out_dir / "reference_self_holdout_scores.csv"),
        "skips_csv": str(out_dir / "reference_self_holdout_skips.csv"),
    }
    (out_dir / "reference_self_holdout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
