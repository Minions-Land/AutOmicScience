from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .artifact_utils import prepare_skill_artifact_dirs, update_artifact_registry
from .contracts import KNNTransferResult, KNNTransferSkillInput
from .io_utils import read_obs_csv, write_json
from .logging_utils import StructuredSkillLogger
from .reference_asset_package import load_reference_asset_package, reference_asset_payload


SKILL_NAME = "shared_knn_transfer_skill"
SKILL_DESCRIPTION = (
    "Shared model-agnostic KNN label transfer over embedding artifacts with unknown handling."
)


def _prediction_frame(
    *,
    query_obs: pd.DataFrame,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    reference_obs: pd.DataFrame,
    reference_label_key: str,
    min_vote_share: float,
    max_mean_distance: float | None,
) -> pd.DataFrame:
    result = pd.DataFrame(index=query_obs.index.copy())
    result["knn_mean_distance"] = neighbor_distances.mean(axis=1)
    result["knn_min_distance"] = neighbor_distances.min(axis=1)

    ref_labels = reference_obs[reference_label_key].astype(str).to_numpy()
    pred_labels: list[str] = []
    vote_shares: list[float] = []
    candidate_counts: list[int] = []
    unknown_flags: list[bool] = []

    for row_idx in range(neighbor_indices.shape[0]):
        labels = ref_labels[neighbor_indices[row_idx]]
        value_counts = pd.Series(labels).value_counts()
        best_label = str(value_counts.index[0])
        vote_share = float(value_counts.iloc[0] / labels.shape[0])
        is_unknown = vote_share < min_vote_share
        if max_mean_distance is not None:
            is_unknown = is_unknown or (float(neighbor_distances[row_idx].mean()) > float(max_mean_distance))
        pred_labels.append("__unknown__" if is_unknown else best_label)
        vote_shares.append(vote_share)
        candidate_counts.append(int(value_counts.shape[0]))
        unknown_flags.append(bool(is_unknown))

    result[f"{reference_label_key}__pred"] = pred_labels
    result[f"{reference_label_key}__vote_share"] = vote_shares
    result[f"{reference_label_key}__neighbor_label_cardinality"] = candidate_counts
    result[f"{reference_label_key}__is_unknown"] = unknown_flags
    return result


def shared_knn_transfer_skill(**kwargs) -> dict[str, Any]:
    args = KNNTransferSkillInput(**kwargs)
    dirs = prepare_skill_artifact_dirs(args.output_dir)
    logger = StructuredSkillLogger(artifact_root=dirs.artifact_dir, component=SKILL_NAME)
    write_json(dirs.manifest_dir / "knn_input_config.json", args.model_dump())

    try:
        if args.reference_asset_package_path:
            package = load_reference_asset_package(args.reference_asset_package_path)
            package_payload = reference_asset_payload(package)
            reference_embeddings_path = package.reference_embeddings_path
            reference_obs_path = package.reference_obs_path
            if not args.reference_label_key and package.reference_label_key:
                args.reference_label_key = package.reference_label_key
            write_json(dirs.manifest_dir / "resolved_reference_asset_package.json", package_payload)
        else:
            reference_embeddings_path = args.reference_embeddings_path
            reference_obs_path = args.reference_obs_path

        reference_embeddings = np.load(Path(reference_embeddings_path).resolve())
        query_embeddings = np.load(Path(args.query_embeddings_path).resolve())
        reference_obs = read_obs_csv(reference_obs_path)
        query_obs = read_obs_csv(args.query_obs_path)

        if reference_embeddings.shape[0] != reference_obs.shape[0]:
            raise ValueError(
                "reference_embeddings row count does not match reference_obs rows: "
                f"{reference_embeddings.shape[0]} vs {reference_obs.shape[0]}"
            )
        if query_embeddings.shape[0] != query_obs.shape[0]:
            raise ValueError(
                "query_embeddings row count does not match query_obs rows: "
                f"{query_embeddings.shape[0]} vs {query_obs.shape[0]}"
            )
        if args.reference_label_key not in reference_obs.columns:
            raise KeyError(f"reference_label_key {args.reference_label_key!r} missing in reference_obs")

        n_neighbors = int(min(args.k, max(1, reference_embeddings.shape[0])))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=args.metric)
        knn.fit(reference_embeddings)
        neighbor_distances, neighbor_indices = knn.kneighbors(query_embeddings)

        prediction_df = _prediction_frame(
            query_obs=query_obs,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            reference_obs=reference_obs,
            reference_label_key=args.reference_label_key,
            min_vote_share=args.min_vote_share,
            max_mean_distance=args.max_mean_distance,
        )
        if args.query_label_key and args.query_label_key in query_obs.columns:
            prediction_df[args.query_label_key] = query_obs[args.query_label_key].astype(str).values

        prediction_csv = dirs.prediction_dir / "query_predictions.csv"
        neighbors_npz = dirs.prediction_dir / "neighbors.npz"
        prediction_df.to_csv(prediction_csv)
        np.savez_compressed(
            neighbors_npz,
            distances=neighbor_distances.astype(np.float32),
            indices=neighbor_indices.astype(np.int64),
        )

        summary = {
            "skill_name": SKILL_NAME,
            "status": "success",
            "model_id": args.model_id,
            "output_dir": str(dirs.output_dir),
            "reference_label_key": args.reference_label_key,
            "query_label_key": args.query_label_key,
            "n_reference_cells": int(reference_obs.shape[0]),
            "n_query_cells": int(query_obs.shape[0]),
            "n_neighbors": n_neighbors,
            "k_requested": int(args.k),
            "metric": args.metric,
            "min_vote_share": float(args.min_vote_share),
            "max_mean_distance": args.max_mean_distance,
        }
        summary_json = write_json(dirs.prediction_dir / "summary.json", summary)
        artifacts = {
            "prediction_csv": str(prediction_csv),
            "neighbors_npz": str(neighbors_npz),
            "summary_json": str(summary_json),
            "logs_manifest_json": str(Path(logger.manifest()["manifest_json"]).resolve()),
        }
        registry_path = args.artifact_registry_path or str(dirs.manifest_dir / "artifact_registry.json")
        artifact_registry_path = update_artifact_registry(
            registry_path=registry_path,
            stage="knn_transfer",
            artifacts=artifacts,
            metadata={"model_id": args.model_id, "run_id": args.run_id},
        )
        result = KNNTransferResult(
            skill_name=SKILL_NAME,
            status="success",
            model_id=args.model_id,
            output_dir=str(dirs.output_dir),
            reference_label_key=args.reference_label_key,
            query_label_key=args.query_label_key,
            n_reference_cells=int(reference_obs.shape[0]),
            n_query_cells=int(query_obs.shape[0]),
            n_neighbors=n_neighbors,
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
        result = KNNTransferResult(
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
        "tool": shared_knn_transfer_skill,
        "args_schema": KNNTransferSkillInput,
        "return_direct": False,
    }
]

SKILL_TOOL_CATALOG = [
    {
        "name": SKILL_NAME,
        "description": SKILL_DESCRIPTION,
    }
]

