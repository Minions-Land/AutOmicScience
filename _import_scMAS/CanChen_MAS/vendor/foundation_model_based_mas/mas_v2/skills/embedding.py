from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mas_v2.contracts.schemas import AdaptationResult, EmbeddingPackage, ReferenceAssetPackage
from mas_v2.runtime.artifacts import register_artifacts
from mas_v2.skills.common import (
    compute_query_embeddings,
    compute_reference_embeddings,
    finalize_embedding_package,
    load_reference_asset_package,
)
from tools_layer.skill_tools.uce_knn_annotation_skill import _resolve_device


def _run_embedding_skill(
    *,
    model_id: str,
    output_dir: str,
    run_id: str,
    adaptation_result: AdaptationResult,
    reference_source: Any,
    reference_label_key: str,
    query_label_key: str,
    batch_size: int,
    device: str,
    random_seed: int,
    registry_path: str,
) -> EmbeddingPackage:
    model_output_dir = Path(output_dir).resolve()
    artifacts_dir = model_output_dir / "artifacts"
    embedding_dir = artifacts_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved_device = _resolve_device(device)
        asset_package = load_reference_asset_package(reference_source)
        query_embeddings, query_coverage = compute_query_embeddings(
            model_id=model_id,
            query_view_path=adaptation_result.query_view.dataset_path,
            query_species=adaptation_result.query_view.species,
            gene_name_key=adaptation_result.query_view.gene_name_source,
            x_source=adaptation_result.query_view.x_source,
            layer_name=adaptation_result.query_view.layer_name,
            batch_size=batch_size,
            device=resolved_device,
            random_seed=random_seed,
        )
        query_embeddings_path = embedding_dir / f"{model_id}__query_embeddings.npy"
        np.save(query_embeddings_path, query_embeddings.astype(np.float32))

        if asset_package is not None:
            reference_embeddings_path = str(Path(asset_package.reference_embeddings_path).resolve())
            reference_obs_path = str(Path(asset_package.reference_obs_path).resolve())
            reference_obs = pd.read_csv(reference_obs_path, index_col=0)
            reference_embeddings = np.load(reference_embeddings_path, mmap_mode="r")
            coverage_payload = json.loads(Path(asset_package.coverage_json).read_text(encoding="utf-8"))
            coverage_payload["query_record"] = query_coverage
            reference_n_cells = int(reference_obs.shape[0])
        else:
            panel_gene_names = list(query_coverage.get("panel_gene_names", []))
            reference_embeddings, reference_coverage = compute_reference_embeddings(
                model_id=model_id,
                reference_view_path=adaptation_result.reference_view.dataset_path,
                reference_species=adaptation_result.reference_view.species,
                gene_name_key=adaptation_result.reference_view.gene_name_source,
                x_source=adaptation_result.reference_view.x_source,
                layer_name=adaptation_result.reference_view.layer_name,
                batch_size=batch_size,
                device=resolved_device,
                random_seed=random_seed,
                scgpt_generic_panel_gene_names=panel_gene_names,
            )
            reference_embeddings_file = embedding_dir / f"{model_id}__reference_embeddings.npy"
            np.save(reference_embeddings_file, reference_embeddings.astype(np.float32))
            reference_embeddings_path = str(reference_embeddings_file)
            reference_obs_path = adaptation_result.reference_view.obs_csv
            coverage_payload = {
                "model_id": model_id,
                "reference_species": adaptation_result.reference_view.species,
                "query_species": adaptation_result.query_view.species,
                "reference_records": [reference_coverage],
                "query_record": query_coverage,
            }
            reference_n_cells = int(reference_embeddings.shape[0])

        package_bits = finalize_embedding_package(
            output_dir=model_output_dir,
            model_id=model_id,
            run_id=run_id,
            reference_species=adaptation_result.reference_view.species,
            query_species=adaptation_result.query_view.species,
            reference_obs_path=reference_obs_path,
            query_obs_path=adaptation_result.query_view.obs_csv,
            reference_label_key=reference_label_key,
            query_label_key=query_label_key,
            reference_embeddings_path=reference_embeddings_path,
            query_embeddings_path=str(query_embeddings_path),
            coverage_payload=coverage_payload,
            registry_path=registry_path,
        )
        embedding_dim = int(query_embeddings.shape[1]) if query_embeddings.ndim == 2 else 0
        package = EmbeddingPackage(
            status="success",
            model_id=model_id,
            output_dir=str(model_output_dir),
            run_id=run_id,
            artifact_registry_path=package_bits["artifact_registry_path"],
            reference_species=adaptation_result.reference_view.species,
            query_species=adaptation_result.query_view.species,
            reference_n_cells=reference_n_cells,
            query_n_cells=int(query_embeddings.shape[0]),
            embedding_dim=embedding_dim,
            reference_label_key=reference_label_key,
            query_label_key=query_label_key,
            coverage=coverage_payload,
            artifacts=package_bits["artifacts"],
        )
        summary_path = model_output_dir / "embedding_summary.json"
        summary_path.write_text(package.model_dump_json(indent=2), encoding="utf-8")
        register_artifacts(registry_path, f"embedding.{model_id}.summary", {"embedding_summary_json": str(summary_path)})
        return package
    except Exception as exc:  # noqa: BLE001
        failure = EmbeddingPackage(
            status="failed",
            model_id=model_id,
            output_dir=str(model_output_dir),
            run_id=run_id,
            reference_label_key=reference_label_key,
            query_label_key=query_label_key,
            error=f"{type(exc).__name__}: {exc}",
        )
        summary_path = model_output_dir / "embedding_summary.json"
        summary_path.write_text(failure.model_dump_json(indent=2), encoding="utf-8")
        return failure


def geneformer_embedding_skill(**kwargs) -> dict[str, Any]:
    package = _run_embedding_skill(model_id="geneformer", **kwargs)
    return package.model_dump()


def nicheformer_embedding_skill(**kwargs) -> dict[str, Any]:
    package = _run_embedding_skill(model_id="nicheformer", **kwargs)
    return package.model_dump()


def scgpt_generic_embedding_skill(**kwargs) -> dict[str, Any]:
    package = _run_embedding_skill(model_id="scgpt_generic", **kwargs)
    return package.model_dump()


def scgpt_human_embedding_skill(**kwargs) -> dict[str, Any]:
    package = _run_embedding_skill(model_id="scgpt_human", **kwargs)
    return package.model_dump()


def scgpt_generic_brain_embedding_skill(**kwargs) -> dict[str, Any]:
    package = _run_embedding_skill(model_id="scgpt_generic_brain", **kwargs)
    return package.model_dump()


def uce_embedding_skill(**kwargs) -> dict[str, Any]:
    package = _run_embedding_skill(model_id="uce", **kwargs)
    return package.model_dump()


def uce_33l_embedding_skill(**kwargs) -> dict[str, Any]:
    package = _run_embedding_skill(model_id="uce_33l", **kwargs)
    return package.model_dump()
