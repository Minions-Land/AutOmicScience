from __future__ import annotations

import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .artifact_utils import prepare_skill_artifact_dirs, update_artifact_registry
from .contracts import AnalysisResult, PredictionAnalysisSkillInput
from .io_utils import read_obs_csv, write_json
from .logging_utils import StructuredSkillLogger


SKILL_NAME = "shared_prediction_analysis_skill"
SKILL_DESCRIPTION = (
    "Shared model-agnostic analysis over KNN predictions, including unknown breakdown and mapped evaluation."
)


def _mapped_label_metrics(
    *,
    pred_raw: np.ndarray,
    truth_raw: np.ndarray,
    reference_eval_mapping: dict[str, str],
    query_eval_mapping: dict[str, str],
) -> tuple[float | None, float | None, pd.DataFrame, list[dict[str, Any]]]:
    pred_coarse = np.asarray(
        [
            reference_eval_mapping.get(str(value), "__unmapped_ref__") if value != "__unknown__" else "__unknown__"
            for value in pred_raw
        ],
        dtype=object,
    )
    truth_coarse = np.asarray(
        [query_eval_mapping.get(str(value), "__unmapped_query__") for value in truth_raw],
        dtype=object,
    )
    known_mask = pred_coarse != "__unknown__"
    if int(known_mask.sum()) == 0:
        return None, None, pd.DataFrame(), []

    pred_known = pred_coarse[known_mask]
    truth_known = truth_coarse[known_mask]
    labels = sorted(set(pred_known.tolist()) | set(truth_known.tolist()))
    conf_matrix = pd.DataFrame(0, index=labels, columns=labels)
    for truth_value, pred_value in zip(truth_known, pred_known):
        conf_matrix.loc[truth_value, pred_value] += 1
    conf_matrix.index.name = "truth_coarse"
    conf_matrix.columns.name = "pred_coarse"

    per_class: list[dict[str, Any]] = []
    for label in labels:
        tp = int(conf_matrix.loc[label, label]) if label in conf_matrix.index and label in conf_matrix.columns else 0
        fp = int(conf_matrix[label].sum()) - tp if label in conf_matrix.columns else 0
        fn = int(conf_matrix.loc[label].sum()) - tp if label in conf_matrix.index else 0
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class.append(
            {
                "class": label,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": tp + fn,
            }
        )
    coarse_accuracy = float(accuracy_score(truth_known, pred_known))
    coarse_macro_f1 = float(f1_score(truth_known, pred_known, average="macro"))
    return coarse_accuracy, coarse_macro_f1, conf_matrix, per_class


def shared_prediction_analysis_skill(**kwargs) -> dict[str, Any]:
    args = PredictionAnalysisSkillInput(**kwargs)
    dirs = prepare_skill_artifact_dirs(args.output_dir)
    logger = StructuredSkillLogger(artifact_root=dirs.artifact_dir, component=SKILL_NAME)
    write_json(dirs.manifest_dir / "analysis_input_config.json", args.model_dump())

    try:
        pred_df = pd.read_csv(Path(args.prediction_csv_path).resolve(), index_col=0)
        pred_col = f"{args.reference_label_key}__pred"
        if pred_col not in pred_df.columns:
            raise KeyError(f"column {pred_col!r} not found in prediction CSV")

        if args.query_obs_path:
            query_obs = read_obs_csv(args.query_obs_path)
            if args.query_label_key and args.query_label_key in query_obs.columns:
                pred_df[args.query_label_key] = query_obs.loc[pred_df.index, args.query_label_key].astype(str).values

        pred_raw = pred_df[pred_col].astype(str).to_numpy()
        has_truth = bool(args.query_label_key and args.query_label_key in pred_df.columns)
        truth_raw = pred_df[args.query_label_key].astype(str).to_numpy() if has_truth else np.asarray([], dtype=object)

        n_total = int(pred_raw.size)
        unknown_mask = pred_raw == "__unknown__"
        n_unknown = int(unknown_mask.sum())
        n_known = int(n_total - n_unknown)
        unknown_rate = float(n_unknown / max(n_total, 1))

        unknown_breakdown_csv = dirs.analysis_dir / "unknown_breakdown.csv"
        raw_crosstab_csv = dirs.analysis_dir / "raw_crosstab.csv"
        raw_crosstab_norm_csv = dirs.analysis_dir / "raw_crosstab_normalized.csv"
        confusion_matrix_csv = dirs.analysis_dir / "confusion_matrix.csv"
        per_class_metrics_csv = dirs.analysis_dir / "per_class_metrics.csv"

        coarse_accuracy = None
        coarse_macro_f1 = None
        unknown_breakdown: list[dict[str, Any]] = []

        if has_truth:
            unknown_by_truth = Counter(truth_raw[unknown_mask])
            total_by_truth = Counter(truth_raw)
            for truth_label in sorted(total_by_truth):
                n_cells = int(total_by_truth[truth_label])
                n_unknown_for_label = int(unknown_by_truth.get(truth_label, 0))
                unknown_breakdown.append(
                    {
                        "query_label": str(truth_label),
                        "n_cells": n_cells,
                        "n_unknown": n_unknown_for_label,
                        "unknown_rate": round(n_unknown_for_label / max(n_cells, 1), 4),
                    }
                )
            pd.DataFrame(unknown_breakdown).to_csv(unknown_breakdown_csv, index=False)

            known_pred = pred_raw[~unknown_mask]
            known_truth = truth_raw[~unknown_mask]
            if known_pred.size > 0:
                cross_tab = pd.crosstab(
                    pd.Series(known_truth, name=f"query_{args.query_label_key}"),
                    pd.Series(known_pred, name=f"reference_{args.reference_label_key}_pred"),
                )
                cross_tab.to_csv(raw_crosstab_csv)
                cross_tab.div(cross_tab.sum(axis=1), axis=0).round(3).to_csv(raw_crosstab_norm_csv)

            if args.reference_eval_mapping and args.query_eval_mapping:
                coarse_accuracy, coarse_macro_f1, conf_matrix, per_class = _mapped_label_metrics(
                    pred_raw=pred_raw,
                    truth_raw=truth_raw,
                    reference_eval_mapping=args.reference_eval_mapping,
                    query_eval_mapping=args.query_eval_mapping,
                )
                if not conf_matrix.empty:
                    conf_matrix.to_csv(confusion_matrix_csv)
                if per_class:
                    pd.DataFrame(per_class).to_csv(per_class_metrics_csv, index=False)

        analysis_payload = {
            "skill_name": SKILL_NAME,
            "status": "success",
            "model_id": args.model_id,
            "run_name": args.run_name,
            "n_total": n_total,
            "n_unknown": n_unknown,
            "n_known": n_known,
            "unknown_rate": round(unknown_rate, 6),
            "coarse_accuracy": coarse_accuracy,
            "coarse_macro_f1": coarse_macro_f1,
            "unknown_breakdown": unknown_breakdown,
        }
        analysis_json = write_json(dirs.analysis_dir / "analysis.json", analysis_payload)
        artifacts = {
            "analysis_json": str(analysis_json),
            "unknown_breakdown_csv": str(unknown_breakdown_csv) if unknown_breakdown_csv.exists() else "",
            "raw_crosstab_csv": str(raw_crosstab_csv) if raw_crosstab_csv.exists() else "",
            "raw_crosstab_normalized_csv": str(raw_crosstab_norm_csv) if raw_crosstab_norm_csv.exists() else "",
            "confusion_matrix_csv": str(confusion_matrix_csv) if confusion_matrix_csv.exists() else "",
            "per_class_metrics_csv": str(per_class_metrics_csv) if per_class_metrics_csv.exists() else "",
            "logs_manifest_json": str(Path(logger.manifest()["manifest_json"]).resolve()),
        }
        registry_path = args.artifact_registry_path or str(dirs.manifest_dir / "artifact_registry.json")
        artifact_registry_path = update_artifact_registry(
            registry_path=registry_path,
            stage="prediction_analysis",
            artifacts={key: value for key, value in artifacts.items() if value},
            metadata={"model_id": args.model_id, "run_id": args.run_id, "run_name": args.run_name},
        )

        result = AnalysisResult(
            skill_name=SKILL_NAME,
            status="success",
            model_id=args.model_id,
            output_dir=str(dirs.output_dir),
            run_name=args.run_name,
            n_total=n_total,
            n_unknown=n_unknown,
            n_known=n_known,
            unknown_rate=unknown_rate,
            coarse_accuracy=coarse_accuracy,
            coarse_macro_f1=coarse_macro_f1,
            artifacts=artifacts,
            metrics={},
            error="",
            run_id=args.run_id,
            artifact_registry_path=str(Path(artifact_registry_path).resolve()),
        )
        write_json(dirs.summary_path, result.model_dump())
        logger.finalize(status="success", payload={"summary_json": str(dirs.summary_path)})
        return result.model_dump()
    except Exception as exc:  # noqa: BLE001
        logger.error(SKILL_NAME, exc, payload={"output_dir": str(dirs.output_dir)})
        result = AnalysisResult(
            skill_name=SKILL_NAME,
            status="failed",
            model_id=args.model_id,
            output_dir=str(dirs.output_dir),
            artifacts={"logs_manifest_json": str(Path(logger.manifest()["manifest_json"]).resolve())},
            error=f"{type(exc).__name__}: {exc}",
            run_id=args.run_id,
            artifact_registry_path=args.artifact_registry_path,
        )
        payload = result.model_dump()
        payload["traceback"] = traceback.format_exc()
        write_json(dirs.summary_path, payload)
        logger.finalize(status="failed", payload={"summary_json": str(dirs.summary_path), "error": result.error})
        return payload


SKILL_TOOLS = [
    {
        "name": SKILL_NAME,
        "description": SKILL_DESCRIPTION,
        "tool": shared_prediction_analysis_skill,
        "args_schema": PredictionAnalysisSkillInput,
        "return_direct": False,
    }
]

SKILL_TOOL_CATALOG = [
    {
        "name": SKILL_NAME,
        "description": SKILL_DESCRIPTION,
    }
]

