#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

from novaeve_bio.io import ensure_dir, write_json
from novaeve_bio.stage2.selector import run_cross_species_plan


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False, width=120)
    return path


def _model_pair(plan: dict[str, Any], model_id: str) -> dict[str, Any]:
    for pair in plan.get("selected_pairs", []):
        if pair.get("model_id") == model_id:
            return dict(pair)
    raise ValueError(f"model_id={model_id!r} is not selected in plan")


def prepare_kukanja_chunks(
    *,
    query_path: str | Path,
    dataset_id: str,
    chunk_dir: str | Path,
    chunk_size: int,
) -> list[dict[str, Any]]:
    query_path = Path(query_path)
    chunk_dir = ensure_dir(chunk_dir)
    data = np.load(query_path, allow_pickle=True)
    n_obs = int(data["X"].shape[0])
    chunks: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, n_obs, chunk_size)):
        end = min(start + chunk_size, n_obs)
        chunk_path = chunk_dir / f"{dataset_id}_chunk_{chunk_index:04d}_{start:07d}_{end:07d}.npz"
        if not chunk_path.exists():
            payload: dict[str, Any] = {}
            for key in data.files:
                value = data[key]
                if hasattr(value, "shape") and value.shape and int(value.shape[0]) == n_obs:
                    payload[key] = value[start:end]
                else:
                    payload[key] = value
            payload["cell_ids"] = np.asarray(
                [f"{query_path.stem}_cell_{idx}" for idx in range(start, end)],
                dtype=object,
            )
            np.savez(chunk_path, **payload)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
                "n_cells": end - start,
                "query_path": str(chunk_path),
            }
        )
    return chunks


def _run_chunk(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("SCMAS_EMBEDDING_CACHE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    run_dir = ensure_dir(task["run_dir"])
    plan = copy.deepcopy(task["plan"])
    pair = copy.deepcopy(task["pair"])
    plan["query_path"] = task["query_path"]
    plan["query_adapter"] = "npz_kukanja_chunk"
    plan["selected_model_ids"] = [task["model_id"]]
    plan["selected_pairs"] = [pair]
    defaults = dict(plan.get("execution_defaults", {}) or {})
    defaults.update(
        {
            "max_query_cells": -1,
            "max_reference_cells": int(task["max_reference_cells"]),
            "min_shared_genes": int(task["min_shared_genes"]),
            "k": int(task["k"]),
            "device": task["device"],
            "batch_size": int(task["batch_size"]),
        }
    )
    plan["execution_defaults"] = defaults
    chunk_plan_path = _write_yaml(plan, run_dir / "selected_pair_plan.yaml")

    try:
        result = run_cross_species_plan(
            plan_path=chunk_plan_path,
            output_dir=run_dir,
            max_query_cells=-1,
            max_reference_cells=int(task["max_reference_cells"]),
            min_shared_genes=int(task["min_shared_genes"]),
            k=int(task["k"]),
            device=task["device"],
            batch_size=int(task["batch_size"]),
        )
        result.update(
            {
                "chunk_index": int(task["chunk_index"]),
                "start": int(task["start"]),
                "end": int(task["end"]),
                "device": task["device"],
                "status": "completed" if int(result.get("n_prediction_rows", 0)) > 0 else "skipped",
            }
        )
        write_json(result, run_dir / "run_summary.json")
        return result
    except Exception as exc:  # noqa: BLE001
        failure = {
            "chunk_index": int(task["chunk_index"]),
            "start": int(task["start"]),
            "end": int(task["end"]),
            "device": task["device"],
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "output_dir": str(run_dir),
            "predictions_path": str(run_dir / "predictions.csv"),
            "metrics_path": str(run_dir / "metrics.csv"),
            "skips_path": str(run_dir / "skips_and_failures.csv"),
            "n_prediction_rows": 0,
            "n_metric_rows": 0,
            "n_skips": 1,
        }
        write_json(failure, run_dir / "run_summary.json")
        return failure


def _append_csv(src: Path, dst: Path, *, write_header: bool) -> bool:
    if not src.exists() or src.stat().st_size == 0:
        return write_header
    with src.open("r", encoding="utf-8", newline="") as in_handle, dst.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as out_handle:
        reader = csv.reader(in_handle)
        writer = csv.writer(out_handle)
        header = next(reader, None)
        if header is None:
            return write_header
        if write_header:
            writer.writerow(header)
            write_header = False
        for row in reader:
            writer.writerow(row)
    return write_header


def combine_chunk_outputs(
    *,
    dataset_id: str,
    model_id: str,
    pair: dict[str, Any],
    chunk_results: list[dict[str, Any]],
    model_run_dir: str | Path,
    batch_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    model_run_dir = ensure_dir(model_run_dir)
    predictions_path = model_run_dir / "predictions.csv"
    metrics_path = model_run_dir / "metrics.csv"
    skips_path = model_run_dir / "skips_and_failures.csv"
    predictions_path.unlink(missing_ok=True)
    metrics_path.unlink(missing_ok=True)
    skips_path.unlink(missing_ok=True)

    true_by_task: dict[str, list[str]] = {}
    pred_by_task: dict[str, list[str]] = {}
    metric_parts: list[pd.DataFrame] = []
    skip_header = True
    pred_header = True
    prediction_rows = 0

    for result in sorted(chunk_results, key=lambda item: int(item["chunk_index"])):
        run_dir = Path(result["output_dir"])
        pred_csv = run_dir / "predictions.csv"
        if pred_csv.exists() and pred_csv.stat().st_size:
            df = pd.read_csv(pred_csv)
            prediction_rows += int(len(df))
            for task, group in df.groupby("task", sort=False):
                true_by_task.setdefault(str(task), []).extend(group["true_label"].astype(str).tolist())
                pred_by_task.setdefault(str(task), []).extend(group["pred_label"].astype(str).tolist())
            pred_header = _append_csv(pred_csv, predictions_path, write_header=pred_header)
        metrics_csv = run_dir / "metrics.csv"
        if metrics_csv.exists() and metrics_csv.stat().st_size:
            metric_parts.append(pd.read_csv(metrics_csv))
        skip_header = _append_csv(run_dir / "skips_and_failures.csv", skips_path, write_header=skip_header)

    if not predictions_path.exists():
        pd.DataFrame(
            columns=[
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
        ).to_csv(predictions_path, index=False)
    if not skips_path.exists():
        pd.DataFrame(columns=["dataset_id", "model_id", "source_id", "method", "task", "stage", "reason", "traceback"]).to_csv(
            skips_path,
            index=False,
        )

    metric_source = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    metric_rows: list[dict[str, Any]] = []
    for task, true_labels in true_by_task.items():
        pred_labels = pred_by_task.get(task, [])
        task_metric_parts = metric_source[metric_source["task"].astype(str) == str(task)] if not metric_source.empty else pd.DataFrame()
        n_query_cells = len(true_labels)
        if not task_metric_parts.empty and "n_query_cells" in task_metric_parts:
            denom = task_metric_parts["n_query_cells"].astype(float).clip(lower=0).sum()
            if denom:
                overlap = float(
                    (
                        task_metric_parts["label_overlap_fraction"].astype(float)
                        * task_metric_parts["n_query_cells"].astype(float)
                    ).sum()
                    / denom
                )
            else:
                overlap = 0.0
            n_ref = int(task_metric_parts["n_reference_cells"].dropna().astype(int).max())
            n_shared = int(task_metric_parts["n_shared_genes"].dropna().astype(int).max())
            n_ref_labels = int(task_metric_parts["n_ref_labels"].dropna().astype(int).max())
        else:
            overlap = 0.0
            n_ref = int(pair.get("n_reference_cells", 0) or 0)
            n_shared = int(pair.get("shared_genes", 0) or 0)
            n_ref_labels = 0
        metric_rows.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "source_id": pair["source_id"],
                "method": pair["method"],
                "embedding_method": pair["embedding_method"],
                "transfer_method": pair["transfer_method"],
                "task": task,
                "accuracy": float(accuracy_score(true_labels, pred_labels)),
                "macro_f1": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
                "label_overlap_fraction": overlap,
                "n_reference_cells": n_ref,
                "n_query_cells": n_query_cells,
                "n_shared_genes": n_shared,
                "n_ref_labels": n_ref_labels,
                "n_true_labels": int(len(set(true_labels))),
            }
        )
    pd.DataFrame(
        metric_rows,
        columns=[
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
        ],
    ).to_csv(metrics_path, index=False)

    failed_chunks = [row for row in chunk_results if row.get("status") == "failed"]
    status = "failed" if failed_chunks else ("completed" if prediction_rows > 0 and metric_rows else "skipped")
    summary = {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "source_id": pair["source_id"],
        "status": status,
        "chunked_execution": True,
        "chunk_size": int(chunk_size),
        "batch_size": int(batch_size),
        "n_chunks": int(len(chunk_results)),
        "n_failed_chunks": int(len(failed_chunks)),
        "output_dir": str(model_run_dir),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "skips_path": str(skips_path),
        "n_metric_rows": int(len(metric_rows)),
        "n_prediction_rows": int(prediction_rows),
        "chunk_results": chunk_results,
    }
    write_json(summary, model_run_dir / "run_summary.json")
    write_json({"chunks": chunk_results}, model_run_dir / "chunk_manifest.json")
    return summary


def rebuild_stage3_summary(*, plan_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    plan_path = Path(plan_path)
    output_dir = ensure_dir(output_dir)
    plan = _read_yaml(plan_path)
    model_root = output_dir / "model_runs"
    prediction_artifacts: dict[str, str] = {}
    metric_artifacts: dict[str, str] = {}
    adapter_specs: dict[str, str] = {}
    model_status: list[dict[str, Any]] = []
    completed_models: list[str] = []
    skipped_models: list[str] = []
    failed_models: list[str] = []

    for spec_path in sorted((output_dir / "adapter_specs").glob("*.yaml")):
        adapter_specs[spec_path.stem] = str(spec_path)

    for run_dir in sorted(model_root.iterdir() if model_root.exists() else []):
        if not run_dir.is_dir():
            continue
        model_id = run_dir.name
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {}
        pred = Path(summary.get("predictions_path") or run_dir / "predictions.csv")
        metrics = Path(summary.get("metrics_path") or run_dir / "metrics.csv")
        status = str(summary.get("status") or "")
        has_rows = int(summary.get("n_prediction_rows", 0) or 0) > 0
        has_metrics = int(summary.get("n_metric_rows", 0) or 0) > 0
        if not status:
            status = "completed" if has_rows and has_metrics else "skipped"
        row = {
            "model_id": model_id,
            "source_id": summary.get("source_id", ""),
            "status": status,
            "reason": summary.get("reason", ""),
            "adapter_spec": adapter_specs.get(model_id, ""),
            "predictions_path": str(pred),
            "metrics_path": str(metrics),
        }
        model_status.append(row)
        if status == "completed":
            completed_models.append(model_id)
            prediction_artifacts[model_id] = str(pred)
            metric_artifacts[model_id] = str(metrics)
        elif status == "failed":
            failed_models.append(model_id)
        else:
            skipped_models.append(model_id)

    aggregate_predictions = output_dir / "predictions.csv"
    aggregate_metrics = output_dir / "metrics.csv"
    aggregate_skips = output_dir / "skips_and_failures.csv"
    aggregate_predictions.unlink(missing_ok=True)
    aggregate_metrics.unlink(missing_ok=True)
    aggregate_skips.unlink(missing_ok=True)
    pred_header = True
    metrics_header = True
    skip_rows: list[dict[str, Any]] = []

    for row in model_status:
        if row["status"] == "completed":
            pred_header = _append_csv(Path(row["predictions_path"]), aggregate_predictions, write_header=pred_header)
            metrics_header = _append_csv(Path(row["metrics_path"]), aggregate_metrics, write_header=metrics_header)
        else:
            skip_rows.append(
                {
                    "dataset_id": plan["dataset_id"],
                    "model_id": row["model_id"],
                    "source_id": row["source_id"],
                    "stage": "adapter_execution",
                    "status": row["status"],
                    "reason": row["reason"],
                    "traceback": "",
                }
            )

    if not aggregate_predictions.exists():
        pd.DataFrame().to_csv(aggregate_predictions, index=False)
    if not aggregate_metrics.exists():
        pd.DataFrame().to_csv(aggregate_metrics, index=False)
    pd.DataFrame(skip_rows).to_csv(aggregate_skips, index=False)

    summary = {
        "dataset_id": plan["dataset_id"],
        "mode": "full",
        "plan_path": str(plan_path),
        "output_dir": str(output_dir),
        "selected_model_ids": list(plan.get("selected_model_ids", [])),
        "completed_models": completed_models,
        "skipped_models": skipped_models,
        "failed_models": failed_models,
        "prediction_artifacts": prediction_artifacts,
        "metric_artifacts": metric_artifacts,
        "adapter_specs": adapter_specs,
        "model_status": model_status,
        "predictions_csv": str(aggregate_predictions),
        "metrics_csv": str(aggregate_metrics),
        "skips_and_failures_csv": str(aggregate_skips),
        "adapter_review_json": str(output_dir / "adapter_review.json"),
        "llm_mode": "required",
        "llm_model": "gpt-5.5",
        "ready_for_consensus": len(completed_models) >= 2,
        "rebuilt_from_chunked_outputs": True,
    }
    write_json(summary, output_dir / "execution_summary.json")
    return summary


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    plan = _read_yaml(args.plan)
    pair = _model_pair(plan, args.model_id)
    dataset_id = str(plan["dataset_id"])
    output_dir = Path(args.output_dir)
    chunk_dir = output_dir / "prepared_inputs" / "query_chunks" / f"chunk_size_{args.chunk_size}"
    chunks = prepare_kukanja_chunks(
        query_path=plan["query_path"],
        dataset_id=dataset_id,
        chunk_dir=chunk_dir,
        chunk_size=args.chunk_size,
    )
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device, e.g. cuda:0,cuda:1")

    chunk_run_root = ensure_dir(output_dir / "prepared_inputs" / args.model_id / "chunk_runs")
    tasks = []
    results: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        run_dir = chunk_run_root / f"chunk_{chunk['chunk_index']:04d}"
        summary_path = run_dir / "run_summary.json"
        if args.resume and summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            pred_path = Path(existing.get("predictions_path") or run_dir / "predictions.csv")
            if int(existing.get("n_prediction_rows", 0) or 0) > 0 and pred_path.exists() and pred_path.stat().st_size > 0:
                existing.update(
                    {
                        "chunk_index": int(chunk["chunk_index"]),
                        "start": int(chunk["start"]),
                        "end": int(chunk["end"]),
                        "status": "completed",
                        "output_dir": str(run_dir),
                    }
                )
                results.append(existing)
                continue
        tasks.append(
            {
                **chunk,
                "plan": plan,
                "pair": pair,
                "model_id": args.model_id,
                "run_dir": str(run_dir),
                "device": "",
                "batch_size": args.batch_size,
                "max_reference_cells": args.max_reference_cells,
                "min_shared_genes": args.min_shared_genes,
                "k": args.k,
            }
        )

    active_devices = devices[: max(1, min(args.workers, len(devices)))]
    task_iter = iter(tasks)
    with ProcessPoolExecutor(max_workers=len(active_devices)) as pool:
        active = {}
        for device in active_devices:
            try:
                task = next(task_iter)
            except StopIteration:
                break
            task["device"] = device
            active[pool.submit(_run_chunk, task)] = device
        while active:
            for future in as_completed(list(active.keys())):
                device = active.pop(future)
                result = future.result()
                results.append(result)
                print(
                    f"[{args.model_id}] chunk={result.get('chunk_index')} status={result.get('status')} "
                    f"rows={result.get('n_prediction_rows')} device={result.get('device')}",
                    flush=True,
                )
                try:
                    task = next(task_iter)
                except StopIteration:
                    pass
                else:
                    task["device"] = device
                    active[pool.submit(_run_chunk, task)] = device
                break

    summary = combine_chunk_outputs(
        dataset_id=dataset_id,
        model_id=args.model_id,
        pair=pair,
        chunk_results=results,
        model_run_dir=output_dir / "model_runs" / args.model_id,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    if args.rebuild_summary:
        rebuild_stage3_summary(plan_path=args.plan, output_dir=output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected Stage3 full models over Kukanja in parallel chunks.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-reference-cells", type=int, default=5000)
    parser.add_argument("--min-shared-genes", type=int, default=30)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-summary", action="store_true")
    args = parser.parse_args()
    summary = run_model(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
