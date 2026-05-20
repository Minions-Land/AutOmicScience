from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

from novaeve_agent.embedding_cache import embedding_cache_key, load_embedding_cache, load_embedding_cache_metadata, save_embedding_cache
from novaeve_agent import paths
from novaeve_agent.io import ensure_dir, write_json, write_yaml
from novaeve_agent.stage2.selector import load_query_bundle
from novaeve_agent.stage4.consensus import UNKNOWN_LABEL, label_to_shared_coarse


LEGACY_ROOT = paths.LEGACY_ROOT
LEGACY_SCRIPT_DIR = paths.UCE_IMA_LEGACY_SCRIPT_DIR
IMA_REFERENCE_PATH = paths.IMA_REFERENCE_H5AD
IMA_CACHE_ROOT = paths.SCMAS_ROOT / "artifacts" / "reference_cache"
UCE_MODEL_DIR = paths.UCE_33L_MODEL_DIR
UCE_MODEL_FILE = "33l_8ep_1024t_1280.torch"
MODEL_ID = "uce_33l_ima_knn"
SOURCE_ID = "ima_sample_uce_embedding"
SHARED_LABELS = ["Astrocyte", "Endothelial", "Microglia", "Neuron", "OPC", "Oligodendrocyte", "Vascular"]


def _load_legacy_uce_helpers() -> Any:
    if str(LEGACY_ROOT) not in sys.path:
        sys.path.insert(0, str(LEGACY_ROOT))
    if str(LEGACY_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_SCRIPT_DIR))
    import run_uce_ima_to_kukanja as legacy

    return legacy


def _reference_cache(
    *,
    reference_path: str | Path,
    cache_root: str | Path,
    max_reference_cells_per_label: int,
    seed: int,
    reuse_existing: bool,
) -> dict[str, Any]:
    legacy = _load_legacy_uce_helpers()
    args = legacy.argparse.Namespace(
        reference_path=str(reference_path),
        reference_cache_root=str(cache_root),
        max_reference_cells_per_label=int(max_reference_cells_per_label),
        random_seed=int(seed),
        reuse_existing=bool(reuse_existing),
    )
    return legacy.prepare_reference_cache(args)


def _encode_query(
    *,
    X: sparse.csr_matrix,
    genes: list[str],
    device: str,
    batch_size: int,
    sample_size: int,
    pad_length: int,
    seed: int,
    model_dir: str | Path,
    model_file: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    legacy = _load_legacy_uce_helpers()
    encoder = legacy.SpeciesAwareUCEEncoder(
        species="mouse",
        gene_names=genes,
        device=legacy._resolve_device(device),
        chrom_order_mode="sorted",
        random_seed=int(seed),
        sample_size=int(sample_size),
        pad_length=int(pad_length),
        model_path=Path(model_dir).resolve() / model_file,
        token_file=Path(model_dir).resolve() / "all_tokens.torch",
        spec_chrom_csv_path=Path(model_dir).resolve() / "species_chrom.csv",
        offset_pkl_path=Path(model_dir).resolve() / "species_offsets.pkl",
        nlayers=33,
    )
    matrix = X.toarray().astype(np.float32, copy=False)
    embeddings = encoder.encode(matrix, batch_size=int(batch_size))
    coverage = encoder.coverage_info.gene_table.copy()
    payload = {
        "n_mapped": int(encoder.coverage_info.n_mapped),
        "n_total": int(encoder.coverage_info.n_total),
        "coverage_ratio": float(encoder.coverage_info.n_mapped / max(1, encoder.coverage_info.n_total)),
        "device": str(encoder.device),
    }
    return embeddings.astype(np.float32, copy=False), coverage, payload


def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32, copy=False)


def _torch_knn_vote(
    *,
    reference_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
    device: str,
    query_chunk_size: int,
    min_vote_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    k = max(1, min(int(k), int(reference_embeddings.shape[0])))
    ref = torch.from_numpy(_normalize_embeddings(reference_embeddings)).to(device)
    query_embeddings = _normalize_embeddings(query_embeddings)
    label_by_idx = np.asarray(reference_labels, dtype=object)
    preds: list[str] = []
    confidences: list[float] = []
    mean_distances: list[float] = []
    ref_t = ref.T.contiguous()
    with torch.inference_mode():
        for start in range(0, query_embeddings.shape[0], int(query_chunk_size)):
            query = torch.from_numpy(query_embeddings[start : start + int(query_chunk_size)]).to(device)
            sims = query @ ref_t
            values, indices = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
            idx_np = indices.cpu().numpy()
            dist_np = (1.0 - values.cpu().numpy()).astype(np.float32, copy=False)
            for row_idx, row in enumerate(idx_np):
                labels = [str(label_by_idx[int(idx)]) for idx in row]
                counts = Counter(label for label in labels if label != UNKNOWN_LABEL)
                if not counts:
                    preds.append(UNKNOWN_LABEL)
                    confidences.append(0.0)
                else:
                    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
                    share = float(count) / float(k)
                    preds.append(label if share >= min_vote_share else UNKNOWN_LABEL)
                    confidences.append(share)
                mean_distances.append(float(dist_np[row_idx].mean()))
            del query, sims, values, indices
    return (
        np.asarray(preds, dtype=object),
        np.asarray(confidences, dtype=np.float32),
        np.asarray(mean_distances, dtype=np.float32),
    )


def _metric_row(
    *,
    dataset_id: str,
    true_shared: np.ndarray,
    pred_shared: np.ndarray,
    n_query_cells: int,
    n_reference_cells: int,
    coverage: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "model_id": MODEL_ID,
        "source_id": SOURCE_ID,
        "method": MODEL_ID,
        "embedding_method": "uce_33l_ima",
        "transfer_method": "knn",
        "task": "coarse_label",
        "accuracy": float(accuracy_score(true_shared, pred_shared)),
        "macro_f1": float(f1_score(true_shared, pred_shared, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_shared, pred_shared, average="weighted", zero_division=0)),
        "label_overlap_fraction": float(np.mean(np.asarray(true_shared, dtype=object) != UNKNOWN_LABEL)),
        "n_reference_cells": int(n_reference_cells),
        "n_query_cells": int(n_query_cells),
        "n_shared_genes": int(coverage.get("n_mapped", 0)),
        "n_ref_labels": int(len(SHARED_LABELS)),
        "n_true_labels": int(len(set(np.asarray(true_shared, dtype=str)))),
        "k": int(k),
    }


def _write_adapter_spec(
    *,
    output_dir: Path,
    dataset_id: str,
    query_path: str | Path,
    reference_manifest: dict[str, Any],
    runtime: dict[str, Any],
) -> Path:
    spec = {
        "schema_version": "scmas.adapter_spec.v1",
        "react_policy": "deterministic_uce_ima_v1",
        "mode": "full" if int(runtime.get("max_query_cells", 0)) == 0 else "subset",
        "dataset_id": dataset_id,
        "query_path": str(query_path),
        "query_adapter": "npz_kukanja",
        "model_id": MODEL_ID,
        "source_id": SOURCE_ID,
        "capability_yaml": str(paths.SCMAS_ROOT / "configs" / "capability" / f"{MODEL_ID}.yaml"),
        "input_artifacts": {
            "query_path": str(query_path),
            "reference_path": str(IMA_REFERENCE_PATH),
            "reference_cache_manifest": reference_manifest.get("artifacts", {}).get("manifest_json", ""),
            "reference_embeddings_npy": reference_manifest.get("artifacts", {}).get("reference_embeddings_npy", ""),
            "reference_sampled_obs_csv": reference_manifest.get("artifacts", {}).get("reference_sampled_obs_csv", ""),
        },
        "gene_strategy": {
            "strategy": "uce_species_aware_gene_tokens",
            "query_species": "mouse",
            "species_is_filter": False,
        },
        "label_strategy": {
            "tasks": ["coarse_label"],
            "reference_label_source": "IMA shared CNS coarse labels",
            "query_truth_source": "query coarse_label, used only for scoring",
        },
        "runtime_payload": runtime,
        "actions": [
            {"action": "load_query_npz_kukanja", "path": str(query_path)},
            {"action": "invoke_raw_embedding_transfer", "executor": "scmas.eval.uce_ima.run_uce_ima_label_transfer"},
        ],
        "expected_outputs": {
            "model_run_dir": str(output_dir / "model_runs" / MODEL_ID),
            "predictions_csv": str(output_dir / "model_runs" / MODEL_ID / "predictions.csv"),
            "metrics_csv": str(output_dir / "model_runs" / MODEL_ID / "metrics.csv"),
            "run_summary_json": str(output_dir / "model_runs" / MODEL_ID / "run_summary.json"),
        },
        "thought": "Use UCE 33L embeddings for the query and the precomputed IMA embedding reference; no classifier head is used.",
    }
    path = ensure_dir(output_dir / "adapter_specs") / f"{MODEL_ID}.yaml"
    write_yaml(spec, path)
    return path


def run_uce_ima_label_transfer(
    *,
    dataset_id: str,
    query_path: str | Path,
    output_dir: str | Path,
    stage3_summary_path: str | Path | None = None,
    max_query_cells: int = 0,
    max_reference_cells_per_label: int = 5000,
    k: int = 25,
    min_vote_share: float = 0.5,
    device: str = "",
    batch_size: int = 64,
    query_chunk_size: int = 8192,
    sample_size: int = 1024,
    pad_length: int = 1536,
    seed: int = 3028,
    reuse_reference_cache: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    model_run_dir = ensure_dir(output_dir / "model_runs" / MODEL_ID)
    artifact_dir = ensure_dir(model_run_dir / "artifacts")
    started = time.time()
    reference_manifest = _reference_cache(
        reference_path=IMA_REFERENCE_PATH,
        cache_root=IMA_CACHE_ROOT,
        max_reference_cells_per_label=max_reference_cells_per_label,
        seed=seed,
        reuse_existing=reuse_reference_cache,
    )

    query = load_query_bundle(query_path, dataset_id=dataset_id, max_cells=max_query_cells, seed=seed).bundle
    query_X = query.X.tocsr()
    query_cache_key, query_cache_metadata = embedding_cache_key(
        base_method="uce_33l_ima",
        genes=query.genes,
        species="mouse",
        matrix=query_X,
        extra={
            "encoder": "SpeciesAwareUCEEncoder",
            "sample_size": int(sample_size),
            "pad_length": int(pad_length),
            "seed": int(seed),
            "model_file": str(UCE_MODEL_FILE),
        },
    )
    cached_query_embeddings = load_embedding_cache(query_cache_key)
    cached_metadata = load_embedding_cache_metadata(query_cache_key) if cached_query_embeddings is not None else {}
    if cached_query_embeddings is not None and cached_metadata.get("uce_query_coverage"):
        query_embeddings = cached_query_embeddings.astype(np.float32, copy=False)
        coverage = dict(cached_metadata.get("uce_query_coverage", {}))
        coverage_table = pd.DataFrame(cached_metadata.get("uce_query_coverage_table", []))
    else:
        query_embeddings, coverage_table, coverage = _encode_query(
            X=query_X,
            genes=query.genes,
            device=device,
            batch_size=batch_size,
            sample_size=sample_size,
            pad_length=pad_length,
            seed=seed,
            model_dir=UCE_MODEL_DIR,
            model_file=UCE_MODEL_FILE,
        )
        save_embedding_cache(
            cache_key=query_cache_key,
            embedding=query_embeddings,
            metadata={
                **query_cache_metadata,
                "uce_query_coverage": coverage,
                "uce_query_coverage_table": json.loads(coverage_table.to_json(orient="records")),
            },
        )
    coverage_table.to_csv(artifact_dir / "query_gene_coverage.csv", index=False)
    np.save(artifact_dir / "query_embeddings.npy", query_embeddings)

    reference_obs = pd.read_csv(reference_manifest["artifacts"]["reference_sampled_obs_csv"], index_col=0)
    reference_labels = reference_obs["Subclass"].astype(str).to_numpy(dtype=object)
    reference_embeddings = np.load(reference_manifest["artifacts"]["reference_embeddings_npy"])
    pred_shared, confidence, mean_distance = _torch_knn_vote(
        reference_embeddings=reference_embeddings,
        query_embeddings=query_embeddings,
        reference_labels=reference_labels,
        k=k,
        device=device,
        query_chunk_size=query_chunk_size,
        min_vote_share=min_vote_share,
    )

    obs = query.obs.reset_index(drop=True)
    cell_ids = obs["cell_id"].astype(str).to_numpy() if "cell_id" in obs.columns else np.asarray(
        [f"cell_{idx}" for idx in range(obs.shape[0])],
        dtype=str,
    )
    sample_ids = obs["sample_id"].astype(str).to_numpy() if "sample_id" in obs.columns else np.asarray([""] * obs.shape[0])
    true_raw = obs["coarse_label"].astype(str).to_numpy() if "coarse_label" in obs.columns else np.asarray([""] * obs.shape[0])
    true_shared = np.asarray([label_to_shared_coarse(value) for value in true_raw], dtype=object)

    predictions = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "model_id": MODEL_ID,
            "source_id": SOURCE_ID,
            "method": MODEL_ID,
            "task": "coarse_label",
            "cell_id": cell_ids,
            "sample_id": sample_ids,
            "true_label": true_raw,
            "pred_label": pred_shared,
            "confidence": confidence,
            "knn_mean_distance": mean_distance,
        }
    )
    metrics = pd.DataFrame(
        [
            _metric_row(
                dataset_id=dataset_id,
                true_shared=true_shared,
                pred_shared=pred_shared,
                n_query_cells=obs.shape[0],
                n_reference_cells=reference_embeddings.shape[0],
                coverage=coverage,
                k=k,
            )
        ]
    )
    predictions_path = model_run_dir / "predictions.csv"
    metrics_path = model_run_dir / "metrics.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    runtime = {
        "embedding_method": "uce_33l_ima",
        "transfer_method": "knn",
        "reference_path": str(IMA_REFERENCE_PATH),
        "reference_cache_manifest": reference_manifest.get("artifacts", {}).get("manifest_json", ""),
        "max_query_cells": int(max_query_cells),
        "max_reference_cells_per_label": int(max_reference_cells_per_label),
        "k": int(k),
        "min_vote_share": float(min_vote_share),
        "seed": int(seed),
        "device": str(device),
        "batch_size": int(batch_size),
        "query_chunk_size": int(query_chunk_size),
        "sample_size": int(sample_size),
        "pad_length": int(pad_length),
    }
    spec_path = _write_adapter_spec(
        output_dir=output_dir,
        dataset_id=dataset_id,
        query_path=query_path,
        reference_manifest=reference_manifest,
        runtime=runtime,
    )
    summary = {
        "dataset_id": dataset_id,
        "model_id": MODEL_ID,
        "source_id": SOURCE_ID,
        "status": "completed",
        "reason": "",
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "adapter_spec": str(spec_path),
        "n_prediction_rows": int(predictions.shape[0]),
        "n_metric_rows": int(metrics.shape[0]),
        "query_gene_coverage": coverage,
        "reference_manifest": reference_manifest,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary, model_run_dir / "run_summary.json")
    if stage3_summary_path:
        inject_uce_ima_into_stage3_summary(
            stage3_summary_path=stage3_summary_path,
            output_dir=output_dir,
            run_summary=summary,
        )
    return summary


def inject_uce_ima_into_stage3_summary(
    *,
    stage3_summary_path: str | Path,
    output_dir: str | Path,
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    stage3_summary_path = Path(stage3_summary_path)
    summary = json.loads(stage3_summary_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    model_id = MODEL_ID
    if model_id not in summary.get("completed_models", []):
        summary.setdefault("completed_models", []).append(model_id)
    for key in ("skipped_models", "failed_models"):
        summary[key] = [item for item in summary.get(key, []) if item != model_id]
    summary.setdefault("prediction_artifacts", {})[model_id] = str(run_summary["predictions_path"])
    summary.setdefault("metric_artifacts", {})[model_id] = str(run_summary["metrics_path"])
    summary.setdefault("adapter_specs", {})[model_id] = str(run_summary["adapter_spec"])
    status_rows = [row for row in summary.get("model_status", []) if row.get("model_id") != model_id]
    status_rows.append(
        {
            "model_id": model_id,
            "source_id": SOURCE_ID,
            "status": "completed",
            "reason": "",
            "adapter_spec": str(run_summary["adapter_spec"]),
            "predictions_path": str(run_summary["predictions_path"]),
            "metrics_path": str(run_summary["metrics_path"]),
        }
    )
    summary["model_status"] = status_rows

    pred_frames = []
    metric_frames = []
    for path in summary.get("prediction_artifacts", {}).values():
        pred_frames.append(pd.read_csv(path))
    for path in summary.get("metric_artifacts", {}).values():
        metric_frames.append(pd.read_csv(path))
    predictions_csv = output_dir / "predictions.csv"
    metrics_csv = output_dir / "metrics.csv"
    pd.concat(pred_frames, ignore_index=True).to_csv(predictions_csv, index=False)
    pd.concat(metric_frames, ignore_index=True).to_csv(metrics_csv, index=False)
    summary["predictions_csv"] = str(predictions_csv)
    summary["metrics_csv"] = str(metrics_csv)
    summary["ready_for_consensus"] = len(summary.get("completed_models", [])) >= 2
    write_json(summary, stage3_summary_path)
    return summary
