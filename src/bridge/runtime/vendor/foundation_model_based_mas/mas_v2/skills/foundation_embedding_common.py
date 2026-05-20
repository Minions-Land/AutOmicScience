from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

import anndata as ad
import numpy as np
import pandas as pd

from .artifact_utils import prepare_skill_artifact_dirs, update_artifact_registry
from .contracts import BaseEmbeddingSkillInput, EmbeddingPackage
from .data_utils import (
    adata_matrix_slice,
    adata_matrix_slice_fixed_panel,
    cap_indices,
    coverage_payload,
    load_cached_reference_indices,
    row_normalize,
    sample_reference_indices,
    select_fixed_panel_gene_names,
    select_gene_names_by_vocab,
)
from .io_utils import write_json
from .logging_utils import StructuredSkillLogger


BuildEncoderFn = Callable[[list[str], str], Any]
VocabFn = Callable[[], set[str]]


def _write_cell_order(path: Path, index: pd.Index) -> str:
    payload = pd.DataFrame({"obs_name": index.astype(str).tolist()})
    payload.to_csv(path, index=False)
    return str(path)


def run_embedding_skill(
    *,
    skill_name: str,
    args: BaseEmbeddingSkillInput,
    build_encoder: BuildEncoderFn,
    vocab_fn: VocabFn,
    use_query_panel_for_both_sides: bool = False,
) -> dict[str, Any]:
    dirs = prepare_skill_artifact_dirs(args.output_dir)
    logger = StructuredSkillLogger(artifact_root=dirs.artifact_dir, component=skill_name)
    input_config_json = write_json(dirs.manifest_dir / "input_config.json", args.model_dump())

    logger.event(
        "run_started",
        status="start",
        payload={
            "model_id": args.model_id,
            "reference_paths": args.resolved_reference_paths(),
            "query_path": args.query_path,
            "reference_label_key": args.reference_label_key,
            "query_label_key": args.query_label_key,
        },
        event_type="lifecycle",
    )

    try:
        query_adata = ad.read_h5ad(args.query_path, backed="r")
        try:
            query_obs_all = query_adata.obs.copy()
            query_indices = cap_indices(
                np.arange(query_adata.n_obs, dtype=np.int64),
                args.max_query_cells,
                args.random_seed,
            )
            query_obs = query_obs_all.iloc[query_indices].copy()

            query_panel_selection = None
            panel_gene_names: list[str] = []
            panel_size = 0
            if use_query_panel_for_both_sides:
                query_panel_selection = select_gene_names_by_vocab(
                    query_adata,
                    preferred_key=args.query_gene_name_key,
                    x_source=args.x_source,
                    layer_name=args.layer_name,
                    vocab_upper=vocab_fn(),
                )
                panel_gene_names = list(query_panel_selection.gene_names)
                panel_size = len(panel_gene_names)
                if panel_size <= 0:
                    raise RuntimeError("scgpt_generic panel_gene_names resolved to empty list.")

            reference_embeddings_parts: list[np.ndarray] = []
            reference_obs_parts: list[pd.DataFrame] = []
            reference_records: list[dict[str, Any]] = []
            shared_encoder = None
            for ref_idx, reference_path in enumerate(args.resolved_reference_paths()):
                ref_adata = ad.read_h5ad(reference_path, backed="r")
                try:
                    ref_obs_all = ref_adata.obs.copy()
                    cache_path = ""
                    if ref_idx < len(args.reference_obs_names_paths):
                        cache_path = str(args.reference_obs_names_paths[ref_idx] or "").strip()
                    if cache_path:
                        ref_indices = load_cached_reference_indices(ref_obs_all, cache_path)
                    else:
                        ref_indices = sample_reference_indices(
                            ref_obs_all,
                            args.reference_label_key,
                            max_total=args.max_reference_cells,
                            max_per_label=args.max_reference_cells_per_label,
                            seed=args.random_seed + ref_idx * 37,
                        )
                    ref_obs = ref_obs_all.iloc[ref_indices].copy()
                    ref_obs["__reference_source__"] = str(Path(reference_path).resolve())

                    if use_query_panel_for_both_sides:
                        fixed = select_fixed_panel_gene_names(
                            ref_adata,
                            preferred_key=args.reference_gene_name_key,
                            x_source=args.x_source,
                            layer_name=args.layer_name,
                            panel_gene_names=panel_gene_names,
                        )
                        ref_matrix = adata_matrix_slice_fixed_panel(
                            ref_adata,
                            ref_indices,
                            x_source=args.x_source,
                            layer_name=args.layer_name,
                            source_column_indices=fixed.source_column_indices,
                            panel_output_positions=fixed.panel_output_positions,
                            panel_size=panel_size,
                        )
                        if shared_encoder is None:
                            shared_encoder = build_encoder(panel_gene_names, args.reference_species)
                        ref_encoder = shared_encoder
                        payload = coverage_payload(fixed.overlap_count, panel_size)
                        gene_name_source = fixed.source_key
                        gene_overlap_count = int(fixed.overlap_count)
                        candidate_scores = fixed.candidate_scores
                    else:
                        selection = select_gene_names_by_vocab(
                            ref_adata,
                            preferred_key=args.reference_gene_name_key,
                            x_source=args.x_source,
                            layer_name=args.layer_name,
                            vocab_upper=vocab_fn(),
                        )
                        ref_matrix = adata_matrix_slice(
                            ref_adata,
                            ref_indices,
                            x_source=args.x_source,
                            layer_name=args.layer_name,
                            gene_keep_mask=selection.gene_keep_mask,
                        )
                        ref_encoder = build_encoder(selection.gene_names, args.reference_species)
                        payload = coverage_payload(ref_encoder.coverage_info.n_mapped, ref_encoder.coverage_info.n_total)
                        gene_name_source = selection.source_key
                        gene_overlap_count = int(selection.overlap_count)
                        candidate_scores = selection.candidate_scores

                    ref_embeddings = row_normalize(ref_encoder.encode(ref_matrix, args.batch_size))
                    reference_embeddings_parts.append(ref_embeddings)
                    reference_obs_parts.append(ref_obs)
                    reference_records.append(
                        {
                            "reference_path": str(Path(reference_path).resolve()),
                            "n_reference_cells": int(ref_obs.shape[0]),
                            "gene_name_source": gene_name_source,
                            "gene_overlap_count": gene_overlap_count,
                            "coverage_ratio": float(payload["coverage_ratio"]),
                            "candidate_scores": candidate_scores,
                        }
                    )
                finally:
                    if getattr(ref_adata, "file", None) is not None:
                        ref_adata.file.close()

            if not reference_embeddings_parts:
                raise RuntimeError(f"No reference embeddings produced for model {args.model_id}")

            reference_embeddings = np.vstack(reference_embeddings_parts)
            reference_obs = pd.concat(reference_obs_parts, axis=0)

            if use_query_panel_for_both_sides:
                fixed = select_fixed_panel_gene_names(
                    query_adata,
                    preferred_key=args.query_gene_name_key,
                    x_source=args.x_source,
                    layer_name=args.layer_name,
                    panel_gene_names=panel_gene_names,
                )
                query_matrix = adata_matrix_slice_fixed_panel(
                    query_adata,
                    query_indices,
                    x_source=args.x_source,
                    layer_name=args.layer_name,
                    source_column_indices=fixed.source_column_indices,
                    panel_output_positions=fixed.panel_output_positions,
                    panel_size=panel_size,
                )
                if shared_encoder is None:
                    shared_encoder = build_encoder(panel_gene_names, args.query_species)
                query_encoder = shared_encoder
                query_coverage = coverage_payload(fixed.overlap_count, panel_size)
                query_source_key = fixed.source_key
                query_overlap_count = int(fixed.overlap_count)
                query_candidate_scores = fixed.candidate_scores
            else:
                query_selection = select_gene_names_by_vocab(
                    query_adata,
                    preferred_key=args.query_gene_name_key,
                    x_source=args.x_source,
                    layer_name=args.layer_name,
                    vocab_upper=vocab_fn(),
                )
                query_matrix = adata_matrix_slice(
                    query_adata,
                    query_indices,
                    x_source=args.x_source,
                    layer_name=args.layer_name,
                    gene_keep_mask=query_selection.gene_keep_mask,
                )
                query_encoder = build_encoder(query_selection.gene_names, args.query_species)
                query_coverage = coverage_payload(query_encoder.coverage_info.n_mapped, query_encoder.coverage_info.n_total)
                query_source_key = query_selection.source_key
                query_overlap_count = int(query_selection.overlap_count)
                query_candidate_scores = query_selection.candidate_scores

            query_embeddings = row_normalize(query_encoder.encode(query_matrix, args.batch_size))

            reference_obs_csv = dirs.adapted_dir / "reference_obs.csv"
            query_obs_csv = dirs.adapted_dir / "query_obs.csv"
            reference_obs.to_csv(reference_obs_csv)
            query_obs.to_csv(query_obs_csv)
            reference_cell_order_csv = dirs.adapted_dir / "reference_cell_order.csv"
            query_cell_order_csv = dirs.adapted_dir / "query_cell_order.csv"
            _write_cell_order(reference_cell_order_csv, reference_obs.index)
            _write_cell_order(query_cell_order_csv, query_obs.index)

            reference_embeddings_npy = dirs.embedding_dir / "reference_embeddings.npy"
            query_embeddings_npy = dirs.embedding_dir / "query_embeddings.npy"
            if args.persist_embeddings:
                np.save(reference_embeddings_npy, reference_embeddings.astype(np.float32))
                np.save(query_embeddings_npy, query_embeddings.astype(np.float32))

            coverage_data = {
                "model_id": args.model_id,
                "reference_species": args.reference_species,
                "query_species": args.query_species,
                "reference_records": reference_records,
                "query_record": {
                    "gene_name_source": query_source_key,
                    "gene_overlap_count": query_overlap_count,
                    "coverage_ratio": float(query_coverage["coverage_ratio"]),
                    "candidate_scores": query_candidate_scores,
                },
            }
            coverage_json = write_json(dirs.coverage_dir / f"{args.model_id}__coverage.json", coverage_data)
            sampling_manifest_json = write_json(
                dirs.manifest_dir / "sampling_manifest.json",
                {
                    "reference_n_cells": int(reference_obs.shape[0]),
                    "query_n_cells": int(query_obs.shape[0]),
                    "max_reference_cells": args.max_reference_cells,
                    "max_reference_cells_per_label": args.max_reference_cells_per_label,
                    "max_query_cells": args.max_query_cells,
                    "random_seed": args.random_seed,
                    "reference_obs_names_paths": args.reference_obs_names_paths,
                },
            )

            artifacts = {
                "reference_embeddings_npy": str(reference_embeddings_npy) if args.persist_embeddings else "",
                "query_embeddings_npy": str(query_embeddings_npy) if args.persist_embeddings else "",
                "reference_obs_csv": str(reference_obs_csv),
                "query_obs_csv": str(query_obs_csv),
                "reference_cell_order_csv": str(reference_cell_order_csv),
                "query_cell_order_csv": str(query_cell_order_csv),
                "coverage_json": str(coverage_json),
                "input_config_json": str(input_config_json),
                "sampling_manifest_json": str(sampling_manifest_json),
                "logs_manifest_json": str(Path(logger.manifest()["manifest_json"]).resolve()),
            }

            registry_path = args.artifact_registry_path or str(dirs.manifest_dir / "artifact_registry.json")
            artifact_registry_path = update_artifact_registry(
                registry_path=registry_path,
                stage="foundation_embedding",
                artifacts={k: v for k, v in artifacts.items() if v},
                metadata={
                    "skill_name": skill_name,
                    "model_id": args.model_id,
                    "run_id": args.run_id,
                },
            )

            result = EmbeddingPackage(
                skill_name=skill_name,
                status="success",
                model_id=args.model_id,
                output_dir=str(dirs.output_dir),
                reference_species=args.reference_species,
                query_species=args.query_species,
                reference_n_cells=int(reference_obs.shape[0]),
                query_n_cells=int(query_obs.shape[0]),
                embedding_dim=int(query_embeddings.shape[1]) if query_embeddings.ndim == 2 and query_embeddings.size else 0,
                reference_label_key=args.reference_label_key,
                query_label_key=args.query_label_key,
                coverage=coverage_data,
                artifacts=artifacts,
                metrics={},
                error="",
                run_id=args.run_id,
                artifact_registry_path=str(Path(artifact_registry_path).resolve()),
            )
            write_json(dirs.summary_path, result.model_dump())
            logger.finalize(status="success", payload={"summary_json": str(dirs.summary_path)})
            return result.model_dump()
        finally:
            if getattr(query_adata, "file", None) is not None:
                query_adata.file.close()
    except Exception as exc:  # noqa: BLE001
        logger.error(skill_name, exc, payload={"output_dir": str(dirs.output_dir)})
        result = EmbeddingPackage(
            skill_name=skill_name,
            status="failed",
            model_id=args.model_id,
            output_dir=str(dirs.output_dir),
            artifacts={"logs_manifest_json": str(Path(logger.manifest()["manifest_json"]).resolve())},
            metrics={},
            error=f"{type(exc).__name__}: {exc}",
            run_id=args.run_id,
            artifact_registry_path=args.artifact_registry_path,
        )
        payload = result.model_dump()
        payload["traceback"] = traceback.format_exc()
        write_json(dirs.summary_path, payload)
        logger.finalize(
            status="failed",
            payload={"error": result.error, "summary_json": str(dirs.summary_path)},
        )
        return payload

