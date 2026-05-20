from __future__ import annotations

import importlib
import importlib.util
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from novaeve_bio import paths
from novaeve_bio.eval.datasets import prepare_all_eval_datasets
from novaeve_bio.eval.metrics import classification_metrics, metrics_from_prediction_npz
from novaeve_bio.eval.registry import ModelSpec, artifact_exists, load_model_registry
from novaeve_bio.io import ensure_dir, read_json, write_json


MCP_TOOLS_DIR = paths.LEGACY_ROOT / "tools_layer" / "mcp_tools"


def _skip(model: ModelSpec, dataset: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "dataset_id": dataset.get("dataset_id", ""),
        "family": model.family,
        "evaluator": model.evaluator,
        "reason": reason,
    }


def _load_npz(npz_path: str) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _write_prediction_rows(
    *,
    rows: list[dict[str, Any]],
    model_id: str,
    dataset_id: str,
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray | None = None,
    cell_ids: np.ndarray | None = None,
) -> None:
    if cell_ids is None:
        cell_ids = np.asarray([f"cell_{idx}" for idx in range(len(y_true))], dtype=str)
    if confidence is None:
        confidence = np.full(len(y_true), np.nan)
    for cell_id, truth, pred, conf in zip(cell_ids, y_true, y_pred, confidence):
        rows.append(
            {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "task": task,
                "cell_id": str(cell_id),
                "true_id": int(truth),
                "pred_id": int(pred),
                "confidence": float(conf) if np.isfinite(conf) else "",
            }
        )


def run_sklearn_model(
    model: ModelSpec,
    dataset: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = _load_npz(dataset["npz_path"])
    x = np.log1p(np.asarray(data["X"], dtype=np.float32))
    eval_mask = ~np.asarray(data.get("is_scmas_dummy", np.zeros(x.shape[0], dtype=bool)), dtype=bool)
    model_dir = Path(model.raw["checkpoint_dir"])
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for task in ("class", "subclass", "supertype"):
        clf = joblib.load(model_dir / f"{task}_model.pkl")
        y_true_all = np.asarray(data[f"y_{task}"], dtype=np.int64)
        y_pred_all = np.asarray(clf.predict(x), dtype=np.int64)
        y_true = y_true_all[eval_mask]
        y_pred = y_pred_all[eval_mask]
        confidence = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(x)
            confidence = np.max(proba, axis=1)[eval_mask]
        _write_prediction_rows(
            rows=prediction_rows,
            model_id=model.model_id,
            dataset_id=dataset["dataset_id"],
            task=task,
            y_true=y_true,
            y_pred=y_pred,
            confidence=confidence,
        )
        metric_rows.append(
            {
                "model_id": model.model_id,
                "dataset_id": dataset["dataset_id"],
                "task": task,
                **classification_metrics(y_true, y_pred),
            }
        )
    return prediction_rows, metric_rows


def run_sklearn_model_external(
    model: ModelSpec,
    dataset: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conda_prefix = model.raw.get("runtime_conda_prefix")
    if not conda_prefix:
        return run_sklearn_model(model, dataset, output_dir)
    worker_dir = ensure_dir(output_dir / model.model_id / dataset["dataset_id"] / "_external_worker")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{paths.SCMAS_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    cmd = [
        "conda",
        "run",
        "-p",
        str(conda_prefix),
        "env",
        f"PYTHONPATH={env['PYTHONPATH']}",
        "python",
        "-m",
        "scmas.eval.sklearn_worker",
        "--model-id",
        model.model_id,
        "--checkpoint-dir",
        model.raw["checkpoint_dir"],
        "--dataset-id",
        dataset["dataset_id"],
        "--npz-path",
        dataset["npz_path"],
        "--output-dir",
        str(worker_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(paths.SCMAS_ROOT), env=env)
    pred_path = worker_dir / "predictions.csv"
    metrics_path = worker_dir / "metrics.csv"
    predictions = pd.read_csv(pred_path).to_dict("records") if pred_path.exists() else []
    metrics = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    return predictions, metrics


def run_spatial_gnn_model(
    model: ModelSpec,
    dataset: dict[str, Any],
    output_dir: Path,
    *,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F
    from scipy.spatial import cKDTree

    mjm_root = paths.SEA_AD_MJM_ROOT
    if str(mjm_root) not in sys.path:
        sys.path.insert(0, str(mjm_root))
    from src.models.spatial_gnn import SpatialGNN

    data = _load_npz(dataset["npz_path"])
    x_all = np.asarray(data["X"], dtype=np.float32)
    eval_mask = ~np.asarray(data.get("is_scmas_dummy", np.zeros(x_all.shape[0], dtype=bool)), dtype=bool)
    x = x_all[eval_mask]
    spatial = np.asarray(data.get("spatial", np.zeros((x_all.shape[0], 2), dtype=np.float32)), dtype=np.float32)[eval_mask]
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        spatial = np.zeros((x.shape[0], 2), dtype=np.float32)
    spatial = spatial[:, :2]
    if not np.isfinite(spatial).all():
        spatial = np.nan_to_num(spatial, nan=0.0, posinf=0.0, neginf=0.0)

    n = x.shape[0]
    k = min(int(model.raw.get("k_neighbors", 15)), max(n - 1, 0))
    if k > 0:
        tree = cKDTree(spatial)
        _, indices = tree.query(spatial, k=k + 1)
        if indices.ndim == 1:
            indices = indices[:, None]
        src = np.repeat(np.arange(n), k)
        dst = indices[:, 1 : k + 1].reshape(-1)
        edge_index_np = np.stack([src, dst], axis=0)
    else:
        edge_index_np = np.zeros((2, 0), dtype=np.int64)

    output_num = model.raw.get("output_num", [3, 24, 137])
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = SpatialGNN(
        input_dim=x.shape[1],
        hidden_dim=int(model.raw.get("hidden_dim", 256)),
        latent_dim=int(model.raw.get("latent_dim", 128)),
        n_gcn_layers=int(model.raw.get("n_gcn_layers", 2)),
        dropout=float(model.raw.get("dropout", 0.3)),
        output_num=output_num,
    ).to(torch_device)
    state = torch.load(model.raw["checkpoint"], map_location=torch_device)
    net.load_state_dict(state)
    net.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(np.log1p(x)).to(torch_device)
        edge_index = torch.from_numpy(edge_index_np).long().to(torch_device)
        _, logits, z = net(x_tensor, edge_index)
        probs = [F.softmax(logit, dim=-1).detach().cpu().numpy() for logit in logits]
        preds = [prob.argmax(axis=1).astype(np.int64) for prob in probs]

    model_out = ensure_dir(output_dir / model.model_id / dataset["dataset_id"])
    np.savez_compressed(
        model_out / "prediction_records.npz",
        class_true_ids=np.asarray(data["y_class"], dtype=np.int64)[eval_mask],
        class_pred_ids=preds[0],
        class_confidence=probs[0].max(axis=1),
        subclass_true_ids=np.asarray(data["y_subclass"], dtype=np.int64)[eval_mask],
        subclass_pred_ids=preds[1],
        subclass_confidence=probs[1].max(axis=1),
        supertype_true_ids=np.asarray(data["y_supertype"], dtype=np.int64)[eval_mask],
        supertype_pred_ids=preds[2],
        supertype_confidence=probs[2].max(axis=1),
        latent=z.detach().cpu().numpy(),
    )

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for task, pred, prob in zip(("class", "subclass", "supertype"), preds, probs):
        y_true = np.asarray(data[f"y_{task}"], dtype=np.int64)[eval_mask]
        _write_prediction_rows(
            rows=prediction_rows,
            model_id=model.model_id,
            dataset_id=dataset["dataset_id"],
            task=task,
            y_true=y_true,
            y_pred=pred,
            confidence=prob.max(axis=1),
        )
        metric_rows.append(
            {
                "model_id": model.model_id,
                "dataset_id": dataset["dataset_id"],
                "task": task,
                **classification_metrics(y_true, pred),
            }
        )
    return prediction_rows, metric_rows


def _call_mcp_model(
    model: ModelSpec,
    dataset: dict[str, Any],
    output_dir: Path,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(MCP_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(MCP_TOOLS_DIR))
    module = importlib.import_module(model.raw["tool_module"])
    func = getattr(module, model.raw["tool_function"])

    model_out = ensure_dir(output_dir / model.model_id / dataset["dataset_id"])
    prediction_npz = model_out / "prediction_records.npz"
    result_json = model_out / "result.json"
    log_path = model_out / "run.log"

    kwargs: dict[str, Any] = {
        "npz_path": dataset["npz_path"],
        "h5ad_path": dataset["h5ad_path"],
        "best_clf_path": model.raw.get("checkpoint", ""),
        "log_path": str(log_path),
        "result_json_path": str(result_json),
        "prediction_npz_path": str(prediction_npz),
        "cell_metadata_h5ad_path": dataset["h5ad_path"],
        "dataset": "SEA_AD_MTP_ST",
        "output_num": model.raw.get("output_num", [3, 24, 137]),
        "freeze_backbone": "freeze" in model.model_id,
        "device": device,
        "seed": 3028,
        "mode": "test",
        "bs": int(model.raw.get("batch_size", batch_size)),
        "num_workers": num_workers,
    }
    if model.family == "scgpt":
        model_name = model.raw.get("model_name", "scGPT_brain")
        kwargs["variant"] = "human" if "human" in model_name.lower() else "brain"
        kwargs["checkpoint_dir"] = model.raw.get("pretrained_dir", "")
    if model.family == "nicheformer":
        kwargs["species"] = dataset.get("species", "human") or "human"
    try:
        func(**kwargs)
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    if prediction_npz.exists():
        data = np.load(prediction_npz, allow_pickle=True)
        cell_ids = data["cell_ids"] if "cell_ids" in data.files else None
        for task in ("class", "subclass", "supertype"):
            true_key = f"{task}_true_ids"
            pred_key = f"{task}_pred_ids"
            if true_key not in data.files or pred_key not in data.files:
                continue
            confidence = data[f"{task}_confidence"] if f"{task}_confidence" in data.files else None
            _write_prediction_rows(
                rows=prediction_rows,
                model_id=model.model_id,
                dataset_id=dataset["dataset_id"],
                task=task,
                y_true=data[true_key],
                y_pred=data[pred_key],
                confidence=confidence,
                cell_ids=cell_ids,
            )
        metric_rows = metrics_from_prediction_npz(str(prediction_npz), model.model_id, dataset["dataset_id"])
    elif result_json.exists():
        metric_rows.append(
            {
                "model_id": model.model_id,
                "dataset_id": dataset["dataset_id"],
                "task": "unknown",
                "result_json": str(result_json),
            }
        )
    return prediction_rows, metric_rows


def _variant_score_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or "dataset_id" not in metrics_df.columns or "macro_f1" not in metrics_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (model_id, task), sub in metrics_df.groupby(["model_id", "task"], dropna=False):
        baseline = sub[sub["dataset_id"].astype(str).str.contains("baseline", regex=False)]
        if baseline.empty:
            continue
        baseline_score = float(baseline["macro_f1"].mean())
        for _, row in sub.iterrows():
            score = float(row["macro_f1"])
            rows.append(
                {
                    "model_id": model_id,
                    "task": task,
                    "dataset_id": row["dataset_id"],
                    "baseline_score": baseline_score,
                    "variant_score": score,
                    "variant_minus_baseline": score - baseline_score,
                }
            )
    return pd.DataFrame(rows)


def run_evaluation(
    *,
    output_dir: str | Path,
    registry_path: str | Path = paths.SCMAS_ROOT / "configs" / "model_registry.yaml",
    dataset_manifest_path: str | Path | None = None,
    include_new_synthetic: bool = True,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    device: str = "",
    batch_size: int = 512,
    num_workers: int = 0,
    prepare_only: bool = False,
    max_cells: int = 0,
    seed: int = 3028,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    prepared_dir = ensure_dir(output_dir / "prepared_npz")
    if dataset_manifest_path:
        dataset_manifest = read_json(dataset_manifest_path)
    else:
        dataset_manifest = prepare_all_eval_datasets(
            prepared_dir,
            include_new_synthetic=include_new_synthetic,
            max_cells=max_cells,
            seed=seed,
        )

    prepared = [d for d in dataset_manifest.get("datasets", []) if d.get("status") == "prepared"]
    skipped_datasets = [d for d in dataset_manifest.get("datasets", []) if d.get("status") != "prepared"]
    if datasets:
        wanted = set(datasets)
        prepared = [d for d in prepared if d["dataset_id"] in wanted]

    registry = load_model_registry(registry_path)
    if models:
        wanted_models = set(models)
        registry = [m for m in registry if m.model_id in wanted_models]

    skip_rows: list[dict[str, Any]] = []
    for row in skipped_datasets:
        skip_rows.append({"model_id": "__dataset__", "dataset_id": row.get("dataset_id"), "reason": row.get("reason", row.get("status"))})

    if prepare_only:
        pd.DataFrame(skip_rows).to_csv(output_dir / "skips_and_failures.csv", index=False)
        return {"output_dir": str(output_dir), "prepared": prepared, "skips": skip_rows}

    all_predictions: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []

    for model in registry:
        ok, reason = artifact_exists(model)
        for dataset in prepared:
            if model.evaluator == "raw_backbone":
                skip_rows.append(_skip(model, dataset, model.raw.get("note", "raw backbone has no trained annotation head")))
                continue
            if model.raw.get("compatible_contract") != "seaad_140_npz":
                skip_rows.append(_skip(model, dataset, "first-stage runner only supports direct seaad_140_npz model evaluation; postprocessors need shared predictions"))
                continue
            if not ok:
                skip_rows.append(_skip(model, dataset, reason))
                continue
            try:
                if model.evaluator == "sklearn_pkl":
                    if (
                        model.raw.get("runtime_conda_prefix")
                        and model.model_id == "sklearn_xgboost"
                        and importlib.util.find_spec("xgboost") is None
                    ):
                        pred_rows, metric_rows = run_sklearn_model_external(model, dataset, output_dir)
                    else:
                        pred_rows, metric_rows = run_sklearn_model(model, dataset, output_dir)
                elif model.evaluator == "mcp_tool":
                    pred_rows, metric_rows = _call_mcp_model(
                        model,
                        dataset,
                        output_dir,
                        device=device,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                elif model.evaluator == "spatial_gnn":
                    pred_rows, metric_rows = run_spatial_gnn_model(
                        model,
                        dataset,
                        output_dir,
                        device=device,
                    )
                elif model.evaluator in {"scanvi_saved", "legacy_experiment"}:
                    skip_rows.append(_skip(model, dataset, f"{model.evaluator} artifact is registered but not safely reusable for arbitrary stage-1 inputs yet"))
                    continue
                else:
                    skip_rows.append(_skip(model, dataset, f"unsupported evaluator: {model.evaluator}"))
                    continue
                all_predictions.extend(pred_rows)
                all_metrics.extend(metric_rows)
                if not metric_rows:
                    skip_rows.append(_skip(model, dataset, "model finished but did not produce parseable metrics"))
            except Exception as exc:  # noqa: BLE001
                skip_rows.append(
                    {
                        **_skip(model, dataset, f"{type(exc).__name__}: {exc}"),
                        "traceback": traceback.format_exc(),
                    }
                )

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    skips_path = output_dir / "skips_and_failures.csv"
    pd.DataFrame(all_predictions).to_csv(predictions_path, index=False)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(skip_rows).to_csv(skips_path, index=False)
    variant_scores = _variant_score_rows(metrics_df)
    variant_scores.to_csv(output_dir / "variant_scores.csv", index=False)
    summary = {
        "output_dir": str(output_dir),
        "predictions_csv": str(predictions_path),
        "metrics_csv": str(metrics_path),
        "skips_and_failures_csv": str(skips_path),
        "variant_scores_csv": str(output_dir / "variant_scores.csv"),
        "n_predictions": len(all_predictions),
        "n_metric_rows": len(all_metrics),
        "n_skips": len(skip_rows),
    }
    write_json(summary, output_dir / "evaluation_summary.json")
    return summary
