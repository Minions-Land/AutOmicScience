from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, prefix: str = "") -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}macro_precision": float(macro_p),
        f"{prefix}macro_recall": float(macro_r),
        f"{prefix}macro_f1": float(macro_f1),
        f"{prefix}micro_precision": float(micro_p),
        f"{prefix}micro_recall": float(micro_r),
        f"{prefix}micro_f1": float(micro_f1),
        f"{prefix}weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}n": int(len(y_true)),
    }


def metrics_from_prediction_npz(path: str, model_id: str, dataset_id: str) -> list[dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    rows: list[dict[str, Any]] = []
    for task in ("class", "subclass", "supertype"):
        true_key = f"{task}_true_ids"
        pred_key = f"{task}_pred_ids"
        if true_key not in data.files or pred_key not in data.files:
            continue
        metrics = classification_metrics(data[true_key], data[pred_key])
        rows.append({"model_id": model_id, "dataset_id": dataset_id, "task": task, **metrics})
    return rows
