#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(
    os.environ.get("AOS_MAS_FOUNDATION_MAS_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.foundation_fusion.main import (  # noqa: E402
    SEAAD_TO_COARSE,
    UNKNOWN_LABEL,
    build_target_augmented,
    compute_metrics,
    resolve_shared_prediction_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train lightweight reference-supervised logistic regression heads on existing "
            "foundation-model embedding artifacts, then apply them to the query set."
        )
    )
    parser.add_argument(
        "--model-root",
        action="append",
        required=True,
        help="Repeatable model spec: model_id=/path/to/output_root or /path/to/_shared_predictions/model_id",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["level0", "level1", "level1_5", "level2", "level3"],
    )
    parser.add_argument("--seed", type=int, default=3028)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-query-cells", type=int, default=0)
    parser.add_argument(
        "--write-per-cell",
        action="store_true",
        help="Write per-cell prediction CSVs. Disabled by default to keep large runs lightweight.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_model_roots(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --model-root {spec!r}; expected model_id=/path")
        model_id, raw_path = spec.split("=", 1)
        model_id = model_id.strip().lower()
        if not model_id:
            raise ValueError(f"Empty model id in spec {spec!r}")
        parsed[model_id] = Path(raw_path.strip()).resolve()
    return parsed


def load_shared_artifacts(model_id: str, root: Path) -> dict[str, Any]:
    shared_dir = resolve_shared_prediction_dir(root, model_id)
    ref_emb_path = shared_dir / "artifacts" / "embeddings" / f"{model_id}__reference_embeddings.npy"
    query_emb_path = shared_dir / "artifacts" / "embeddings" / f"{model_id}__query_embeddings.npy"
    ref_obs_path = shared_dir / "artifacts" / "predictions" / model_id / "reference_sampled_obs.csv"
    query_obs_path = shared_dir / "artifacts" / "predictions" / model_id / "query_sampled_obs.csv"
    query_pred_path = shared_dir / "artifacts" / "predictions" / model_id / "query_predictions.csv"

    ref_embeddings = np.load(ref_emb_path, mmap_mode="r")
    query_embeddings = np.load(query_emb_path, mmap_mode="r")
    ref_obs = pd.read_csv(ref_obs_path, index_col=0)
    query_obs = pd.read_csv(query_obs_path, index_col=0)
    query_pred = pd.read_csv(query_pred_path, index_col=0)
    ref_obs.index = ref_obs.index.astype(str)
    query_obs.index = query_obs.index.astype(str)
    query_pred.index = query_pred.index.astype(str)
    return {
        "shared_dir": shared_dir,
        "reference_embeddings": ref_embeddings,
        "query_embeddings": query_embeddings,
        "reference_obs": ref_obs,
        "query_obs": query_obs,
        "query_predictions": query_pred,
    }


def maybe_subsample_query(
    query_obs: pd.DataFrame,
    query_embeddings: np.ndarray,
    query_pred: pd.DataFrame,
    *,
    max_query_cells: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    if max_query_cells <= 0 or len(query_obs) <= max_query_cells:
        return (
            query_obs,
            np.asarray(query_embeddings),
            query_pred,
            {
                "applied": False,
                "n_cells_before": int(len(query_obs)),
                "n_cells_after": int(len(query_obs)),
                "seed": int(seed),
            },
        )

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(query_obs), size=max_query_cells, replace=False))
    selected_index = query_obs.index[selected]
    return (
        query_obs.iloc[selected].copy(),
        np.asarray(query_embeddings[selected]),
        query_pred.loc[selected_index].copy(),
        {
            "applied": True,
            "n_cells_before": int(len(query_obs)),
            "n_cells_after": int(max_query_cells),
            "seed": int(seed),
        },
    )


def stratified_split_indices(labels: np.ndarray, seed: int, calibration_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    calib_idx: list[np.ndarray] = []
    labels = np.asarray(labels, dtype=object)
    for offset, label in enumerate(sorted(set(labels.tolist()))):
        idx = np.where(labels == label)[0].astype(np.int64)
        local_rng = np.random.default_rng(seed + 97 + offset)
        perm = local_rng.permutation(idx)
        n_calib = max(1, int(round(idx.size * calibration_fraction)))
        n_calib = min(n_calib, max(1, idx.size - 1))
        calib_idx.append(np.sort(perm[:n_calib].astype(np.int64)))
        train_idx.append(np.sort(perm[n_calib:].astype(np.int64)))
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(calib_idx))


def fit_lr_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_iter: int,
) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",
        random_state=3028,
    )
    clf.fit(x_train, y_train)
    return clf


def calibrate_threshold(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int,
    seed: int,
) -> dict[str, float]:
    train_idx, calib_idx = stratified_split_indices(y, seed=seed, calibration_fraction=0.1)
    clf = fit_lr_classifier(x[train_idx], y[train_idx], max_iter=max_iter)
    calib_proba = clf.predict_proba(x[calib_idx])
    calib_pred = clf.classes_[np.argmax(calib_proba, axis=1)]
    max_prob = np.max(calib_proba, axis=1)
    correct_mask = calib_pred == y[calib_idx]
    if not bool(np.any(correct_mask)):
        return {"q10": 0.5, "q25": 0.6}
    correct_probs = max_prob[correct_mask]
    return {
        "q10": float(np.quantile(correct_probs, 0.10)),
        "q25": float(np.quantile(correct_probs, 0.25)),
    }


def predict_with_proba(
    clf: LogisticRegression,
    x: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_labels: list[np.ndarray] = []
    max_prob_chunks: list[np.ndarray] = []
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        proba = clf.predict_proba(x[start:end])
        max_prob = np.max(proba, axis=1)
        pred = clf.classes_[np.argmax(proba, axis=1)].astype(object)
        pred_labels.append(np.asarray(pred, dtype=object))
        max_prob_chunks.append(max_prob.astype(np.float32))
    return np.concatenate(pred_labels), np.concatenate(max_prob_chunks)


def apply_rejection_threshold(
    pred_labels: np.ndarray,
    max_prob: np.ndarray,
    *,
    threshold: float | None,
) -> np.ndarray:
    pred = np.asarray(pred_labels, dtype=object).copy()
    if threshold is None:
        return pred
    return np.where(np.asarray(max_prob) >= float(threshold), pred, UNKNOWN_LABEL).astype(object)


def map_reference_labels_to_coarse(labels: np.ndarray) -> np.ndarray:
    mapped = []
    for label in np.asarray(labels, dtype=object):
        if str(label) == UNKNOWN_LABEL:
            mapped.append(UNKNOWN_LABEL)
        else:
            mapped.append(SEAAD_TO_COARSE.get(str(label), "__unmapped_reference__"))
    return np.asarray(mapped, dtype=object)


def evaluate_methods(
    *,
    model_id: str,
    query_obs: pd.DataFrame,
    query_pred_knn: pd.DataFrame,
    methods: dict[str, np.ndarray],
    levels: list[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "model_id": model_id,
        "levels": {},
    }
    knn_coarse = map_reference_labels_to_coarse(query_pred_knn["Subclass__pred"].astype(str).to_numpy())
    for level in levels:
        target_df = build_target_augmented(query_obs, level)
        y_true = target_df["target_augmented"].astype(str).to_numpy(dtype=object)
        shared_mask = target_df["is_shared_truth"].to_numpy(dtype=bool)
        level_metrics = {
            "knn_baseline": compute_metrics(y_true, knn_coarse, shared_mask),
        }
        for method_name, pred in methods.items():
            level_metrics[method_name] = compute_metrics(y_true, pred, shared_mask)
        results["levels"][level] = level_metrics
    return results


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Reference-Supervised Logistic Transfer",
        "",
        "## Policy",
        "",
        "- Logistic regression is trained only on reference embeddings and reference labels.",
        "- Query labels are used only once for final evaluation.",
        "- This is a separate reference-supervised transfer track, not the label-free KNN track.",
        "",
    ]
    sampling = summary.get("query_sampling", {})
    lines.extend(
        [
            "## Query Sampling",
            "",
            f"- Applied: `{sampling.get('applied', False)}`",
            f"- Cells before: `{sampling.get('n_cells_before', 0)}`",
            f"- Cells after: `{sampling.get('n_cells_after', 0)}`",
            "",
        ]
    )
    for model_id, payload in summary["models"].items():
        lines.extend(
            [
                f"## {model_id}",
                "",
                f"- Reference cells: `{payload['reference_n_cells']}`",
                f"- Query cells: `{payload['query_n_cells']}`",
                f"- Calibration thresholds: `q10={payload['thresholds']['q10']:.4f}`, `q25={payload['thresholds']['q25']:.4f}`",
                "",
            ]
        )
        for level, metrics in payload["evaluation"]["levels"].items():
            lines.extend(
                [
                    f"### {level}",
                    "",
                    "| Method | Acc | Macro-F1 | Shared Acc | Shared Macro-F1 | OOD Reject | Accepted Frac |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for method_name, method_metrics in metrics.items():
                lines.append(
                    "| "
                    f"{method_name} | "
                    f"{method_metrics['overall_accuracy']:.4f} | "
                    f"{method_metrics['overall_macro_f1']:.4f} | "
                    f"{method_metrics['shared_accuracy']:.4f} | "
                    f"{method_metrics['shared_macro_f1']:.4f} | "
                    f"{method_metrics['ood_rejection_rate']:.4f} | "
                    f"{method_metrics['accepted_fraction']:.4f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_roots = parse_model_roots(args.model_root)
    summary: dict[str, Any] = {
        "seed": int(args.seed),
        "levels": list(args.levels),
        "query_sampling": {},
        "models": {},
        "evaluation_policy": (
            "Reference labels are used for LR training only on reference embeddings. "
            "Query labels are used only once for final evaluation."
        ),
    }

    for model_id, root in model_roots.items():
        artifacts = load_shared_artifacts(model_id, root)
        query_obs, query_embeddings, query_pred, query_sampling = maybe_subsample_query(
            artifacts["query_obs"],
            artifacts["query_embeddings"],
            artifacts["query_predictions"],
            max_query_cells=args.max_query_cells,
            seed=args.seed,
        )
        if not summary["query_sampling"]:
            summary["query_sampling"] = query_sampling

        ref_embeddings = np.asarray(artifacts["reference_embeddings"])
        ref_obs = artifacts["reference_obs"].copy()
        subclass_labels = ref_obs["Subclass"].astype(str).to_numpy(dtype=object)
        coarse_labels = ref_obs["Subclass"].astype(str).map(lambda x: SEAAD_TO_COARSE.get(str(x), "__unmapped_reference__")).to_numpy(dtype=object)

        coarse_thresholds = calibrate_threshold(ref_embeddings, coarse_labels, max_iter=args.max_iter, seed=args.seed)
        subclass_thresholds = calibrate_threshold(ref_embeddings, subclass_labels, max_iter=args.max_iter, seed=args.seed)

        coarse_clf = fit_lr_classifier(ref_embeddings, coarse_labels, max_iter=args.max_iter)
        subclass_clf = fit_lr_classifier(ref_embeddings, subclass_labels, max_iter=args.max_iter)

        coarse_pred_plain, coarse_prob_plain = predict_with_proba(
            coarse_clf,
            query_embeddings,
            chunk_size=args.chunk_size,
        )
        coarse_pred_q10 = apply_rejection_threshold(
            coarse_pred_plain,
            coarse_prob_plain,
            threshold=coarse_thresholds["q10"],
        )
        subclass_pred_plain, subclass_prob_plain = predict_with_proba(
            subclass_clf,
            query_embeddings,
            chunk_size=args.chunk_size,
        )
        subclass_pred_q10 = apply_rejection_threshold(
            subclass_pred_plain,
            subclass_prob_plain,
            threshold=subclass_thresholds["q10"],
        )

        methods = {
            "coarse_lr_balanced": np.asarray(coarse_pred_plain, dtype=object),
            "coarse_lr_balanced_q10_reject": np.asarray(coarse_pred_q10, dtype=object),
            "subclass_lr_balanced_mapped": map_reference_labels_to_coarse(subclass_pred_plain),
            "subclass_lr_balanced_q10_reject_mapped": map_reference_labels_to_coarse(subclass_pred_q10),
        }

        evaluation = evaluate_methods(
            model_id=model_id,
            query_obs=query_obs,
            query_pred_knn=query_pred,
            methods=methods,
            levels=args.levels,
        )

        model_dir = output_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        per_cell_path: str | None = None
        if args.write_per_cell:
            per_cell = pd.DataFrame(index=query_obs.index.copy())
            per_cell["knn_pred_raw"] = query_pred["Subclass__pred"].astype(str).values
            per_cell["knn_pred_coarse"] = map_reference_labels_to_coarse(query_pred["Subclass__pred"].astype(str).to_numpy())
            per_cell["coarse_lr_balanced__pred"] = methods["coarse_lr_balanced"]
            per_cell["coarse_lr_balanced__max_prob"] = coarse_prob_plain
            per_cell["coarse_lr_balanced_q10_reject__pred"] = methods["coarse_lr_balanced_q10_reject"]
            per_cell["subclass_lr_balanced__pred_raw"] = subclass_pred_plain
            per_cell["subclass_lr_balanced__pred"] = methods["subclass_lr_balanced_mapped"]
            per_cell["subclass_lr_balanced__max_prob"] = subclass_prob_plain
            per_cell["subclass_lr_balanced_q10_reject__pred"] = methods["subclass_lr_balanced_q10_reject_mapped"]

            per_cell_csv = model_dir / "per_cell_predictions.csv"
            per_cell.to_csv(per_cell_csv)
            per_cell_path = str(per_cell_csv)

        model_summary = {
            "model_id": model_id,
            "shared_dir": str(artifacts["shared_dir"]),
            "reference_n_cells": int(ref_embeddings.shape[0]),
            "query_n_cells": int(query_embeddings.shape[0]),
            "thresholds": {
                "coarse_q10": float(coarse_thresholds["q10"]),
                "coarse_q25": float(coarse_thresholds["q25"]),
                "subclass_q10": float(subclass_thresholds["q10"]),
                "subclass_q25": float(subclass_thresholds["q25"]),
                "q10": float(coarse_thresholds["q10"]),
                "q25": float(coarse_thresholds["q25"]),
            },
            "evaluation": evaluation,
            "artifacts": {
                "per_cell_predictions_csv": per_cell_path,
            },
        }
        (model_dir / "summary.json").write_text(json.dumps(model_summary, indent=2, default=str), encoding="utf-8")
        summary["models"][model_id] = model_summary

    summary_path = output_dir / "reference_supervised_transfer_summary.json"
    report_path = output_dir / "reference_supervised_transfer_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown_report(report_path, summary)
    print(json.dumps({"summary_json": str(summary_path), "summary_markdown": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
