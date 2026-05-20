from __future__ import annotations

import argparse
from pathlib import Path

from novaeve_bio import paths
from novaeve_bio.data.catalog import build_dataset_catalog
from novaeve_bio.data.labels import build_seaad_label_maps
from novaeve_bio.data.prepare_sources import prepare_generation_sources
from novaeve_bio.data.reference import build_reference, build_seaad_test_h5ad
from novaeve_bio.eval.datasets import prepare_all_eval_datasets
from novaeve_bio.eval.label_transfer import evaluate_raw_label_transfer
from novaeve_bio.eval.run import run_evaluation
from novaeve_bio.eval.uce_ima import run_uce_ima_label_transfer
from novaeve_bio.io import write_json
from novaeve_bio.scdesign3.generate import preflight_scdesign3, run_generation_configs, write_generation_configs
from novaeve_bio.stage2.selector import profile_query, run_cross_species_plan, select_models
from novaeve_bio.stage3.adapter_executor import adapt_and_execute, inspect_model_contracts
from novaeve_bio.stage4.consensus import run_consensus


def _add_common_size_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-cells-per-source", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3028)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="novaeve_bio", description="scMAS multi-stage annotation workflow")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-label-maps", help="Build SEA-AD label/gene maps.")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("build-reference", help="Build merged human/mouse reference h5ad and SEA-AD real test h5ad.")
    _add_common_size_args(p)
    p.add_argument("--max-cells-seaad-reference", type=int, default=100_000)
    p.add_argument("--max-cells-seaad-test", type=int, default=50_000)
    p.add_argument("--include-smartseq", action="store_true")
    p.add_argument("--output", default=str(paths.REFERENCE_H5AD))

    p = sub.add_parser("build-seaad-test", help="Build SEA-AD MERFISH held-out donor test h5ad only.")
    p.add_argument("--max-cells", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=3028)
    p.add_argument("--output", default=str(paths.SEAAD_TEST_H5AD))

    p = sub.add_parser("prepare-sources", help="Prepare standard bundles used as scDesign3 source data.")
    _add_common_size_args(p)
    p.add_argument("--max-genes-per-source", type=int, default=0, help="Optional feature cap for smoke runs; 0 keeps all genes.")
    p.add_argument("--include-smartseq", action="store_true")
    p.add_argument("--include-seaad-reference", action="store_true")
    p.add_argument("--feature-panel", action="append", dest="feature_panels", default=[])
    p.add_argument("--source", action="append", dest="sources", default=[], help="Optional source_id filter; repeatable.")
    p.add_argument("--output-root", default=str(paths.PREPARED_SOURCE_DIR))

    p = sub.add_parser("write-scdesign3-configs", help="Write scDesign3 config JSON files for prepared sources.")
    p.add_argument("--prepared-source-root", default=str(paths.PREPARED_SOURCE_DIR))
    p.add_argument("--config-root", default=str(paths.SYNTHETIC_DIR / "_configs"))
    p.add_argument("--target-total", type=int, default=20_000)
    p.add_argument("--n-cores", type=int, default=8)
    p.add_argument("--seed", type=int, default=3028)

    p = sub.add_parser("preflight-scdesign3", help="Check scDesign3 runner and R environment.")
    p.add_argument("--rscript-path", default="Rscript")

    p = sub.add_parser("run-scdesign3", help="Run generated scDesign3 configs.")
    p.add_argument("--config-manifest", default=str(paths.SYNTHETIC_DIR / "_configs" / "generation_config_manifest.json"))
    p.add_argument("--rscript-path", default="Rscript")
    p.add_argument("--force-refit", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("prepare-eval-datasets", help="Convert benchmark inputs to model-specific SEA-AD 140-gene NPZs.")
    p.add_argument("--output-dir", default=str(paths.RUNS_DIR / "stage1_eval" / "prepared_npz"))
    p.add_argument("--no-new-synthetic", action="store_true")
    p.add_argument("--max-cells", type=int, default=0, help="Optional max real cells per dataset before dummy split padding.")
    p.add_argument("--seed", type=int, default=3028)

    p = sub.add_parser("build-dataset-catalog", help="Write a dataset role/source table for stage-1 planning and smoke runs.")
    p.add_argument("--output-dir", default=str(paths.SCMAS_ROOT / "reports"))
    p.add_argument("--no-shape-probe", action="store_true")

    p = sub.add_parser("raw-label-transfer-smoke", help="No-training label transfer over new scDesign3 variants.")
    p.add_argument("--output-dir", default=str(paths.RUNS_DIR / "raw_label_transfer_smoke"))
    p.add_argument("--synthetic-root", default=str(paths.SYNTHETIC_DIR))
    p.add_argument("--source", action="append", dest="sources", default=[], help="Optional source_id filter; repeatable.")
    p.add_argument("--variant", action="append", dest="variants", default=[], help="Optional variant_id filter; repeatable.")
    p.add_argument("--method", action="append", dest="methods", default=[], help="Repeatable method, e.g. expression_log1p_knn.")
    p.add_argument("--max-reference-cells", type=int, default=500)
    p.add_argument("--max-query-cells", type=int, default=200)
    p.add_argument("--min-shared-genes", type=int, default=50)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--device", default="")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=3028)
    p.add_argument("--include-existing-seaad", action="store_true")

    p = sub.add_parser("evaluate", help="Run model registry over prepared real/synthetic datasets.")
    p.add_argument("--output-dir", default=str(paths.RUNS_DIR / "stage1_eval"))
    p.add_argument("--registry", default=str(paths.SCMAS_ROOT / "configs" / "model_registry.yaml"))
    p.add_argument("--dataset-manifest", default="")
    p.add_argument("--no-new-synthetic", action="store_true")
    p.add_argument("--model", action="append", dest="models", default=[])
    p.add_argument("--dataset", action="append", dest="datasets", default=[])
    p.add_argument("--device", default="")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--max-cells", type=int, default=0, help="Optional max real cells per dataset when preparing inputs.")
    p.add_argument("--seed", type=int, default=3028)

    p = sub.add_parser("profile-query", help="Build a stage-2 query profile for model/source selection.")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--input", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-cells", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=3028)

    p = sub.add_parser("select-models", help="Select no-training source+model pairs for a profiled query dataset.")
    p.add_argument("--query-profile", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--artifact-bundle", default=str(paths.SCMAS_ROOT / "artifacts" / "stage1_full"))
    p.add_argument("--prepared-source-root", default=str(paths.DATA_DIR / "prepared_sources"))
    p.add_argument("--capability-dir", default=str(paths.SCMAS_ROOT / "configs" / "capability"))
    p.add_argument("--top-k", type=int, default=3, help="Backward-compatible alias for --num-models.")
    p.add_argument("--num-models", type=int, default=None, help="Number of model/source pairs Stage 2 should select.")
    p.add_argument("--min-shared-genes", type=int, default=30)
    p.add_argument("--max-source-profile-cells", type=int, default=20_000)
    p.add_argument("--max-query-cells", type=int, default=5_000)
    p.add_argument("--max-reference-cells", type=int, default=1_000)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--device", default="")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--llm-mode", choices=["required", "optional", "off"], default="required")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-max-candidates", type=int, default=80)
    p.add_argument("--llm-retry-limit", type=int, default=2)
    p.add_argument("--selection-strategy", choices=["batch", "iterative", "one_by_one"], default="one_by_one")
    p.add_argument(
        "--selection-objective",
        choices=["unified_rank", "consensus", "best_single_ablation"],
        default="unified_rank",
        help="Selection objective. Legacy aliases are accepted but normalized to unified_rank.",
    )
    p.add_argument("--iterative-exclude-scope", choices=["model", "family"], default="family")
    p.add_argument("--exclude-model", action="append", dest="excluded_models", default=[])
    p.add_argument("--seed", type=int, default=3028)

    p = sub.add_parser("run-cross-species-plan", help="Run subset no-training execution from selected_execution_plan.yaml.")
    p.add_argument("--plan", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-query-cells", type=int, default=0)
    p.add_argument("--max-reference-cells", type=int, default=0)
    p.add_argument("--min-shared-genes", type=int, default=0)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=0)

    p = sub.add_parser("inspect-model-contracts", help="Inspect capability YAML, registry artifacts, and wrapper signatures.")
    p.add_argument("--capability-dir", default=str(paths.SCMAS_ROOT / "configs" / "capability"))
    p.add_argument("--registry", default=str(paths.SCMAS_ROOT / "configs" / "model_registry.yaml"))
    p.add_argument("--output-dir", default="")

    p = sub.add_parser("adapt-and-execute", help="Run stage-3 adapter spec generation and whitelist execution from selected_execution_plan.yaml.")
    p.add_argument("--plan", required=True)
    p.add_argument("--mode", choices=["subset", "full"], default="subset")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default="")
    p.add_argument("--capability-dir", default=str(paths.SCMAS_ROOT / "configs" / "capability"))
    p.add_argument("--registry", default=str(paths.SCMAS_ROOT / "configs" / "model_registry.yaml"))
    p.add_argument("--retry-limit", type=int, default=2)
    p.add_argument("--llm-mode", choices=["required", "optional", "off"], default="required")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-retry-limit", type=int, default=2)

    p = sub.add_parser("run-consensus", help="Run stage-4 reference-enhanced consensus fusion from a stage-3 execution summary.")
    p.add_argument("--stage3-summary", required=True)
    p.add_argument("--mode", choices=["subset", "full"], default="subset")
    p.add_argument("--output-dir", default="")
    p.add_argument("--seed", type=int, default=3028)
    p.add_argument("--llm-policy-mode", choices=["required", "optional", "off"], default="off")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-cell-adjudication-mode", choices=["required", "optional", "off"], default="off")
    p.add_argument("--llm-cell-max-groups", type=int, default=120)
    p.add_argument("--llm-cell-batch-size", type=int, default=12)
    p.add_argument(
        "--model-scope",
        choices=["selected", "completed"],
        default="selected",
        help="Use only stage3 selected_model_ids by default; choose completed for explicit all-completed-model analysis.",
    )
    p.add_argument(
        "--skip-reference-geometry",
        action="store_true",
        help="Skip stage-4 reference geometry methods and run vote/confidence/capability consensus only.",
    )
    p.add_argument(
        "--execution-strategy",
        choices=["selected_only", "benchmark_all"],
        default="selected_only",
        help="`selected_only` first selects one fusion function on an unlabeled probe subset, then executes only that function on full data. `benchmark_all` runs every registered fusion function.",
    )
    p.add_argument(
        "--max-probe-cells",
        type=int,
        default=5000,
        help="Maximum unlabeled query cells used for Stage-4 pre-execution function selection when execution-strategy=selected_only.",
    )

    p = sub.add_parser("run-uce-ima-transfer", help="Run UCE 33L query embeddings against the IMA embedding reference.")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--query-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage3-summary", default="")
    p.add_argument("--max-query-cells", type=int, default=0)
    p.add_argument("--max-reference-cells-per-label", type=int, default=5000)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--min-vote-share", type=float, default=0.5)
    p.add_argument("--device", default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--query-chunk-size", type=int, default=8192)
    p.add_argument("--sample-size", type=int, default=1024)
    p.add_argument("--pad-length", type=int, default=1536)
    p.add_argument("--seed", type=int, default=3028)
    p.add_argument("--rebuild-reference-cache", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "build-label-maps":
        result = build_seaad_label_maps(force=args.force)
    elif args.cmd == "build-reference":
        result = build_reference(
            output_path=args.output,
            max_cells_per_source=args.max_cells_per_source,
            max_cells_seaad_reference=args.max_cells_seaad_reference,
            max_cells_seaad_test=args.max_cells_seaad_test,
            include_smartseq=args.include_smartseq,
            seed=args.seed,
        )
    elif args.cmd == "build-seaad-test":
        result = build_seaad_test_h5ad(
            output_path=args.output,
            max_cells=args.max_cells,
            seed=args.seed,
        )
    elif args.cmd == "prepare-sources":
        result = prepare_generation_sources(
            output_root=args.output_root,
            max_cells_per_source=args.max_cells_per_source,
            max_genes_per_source=args.max_genes_per_source,
            include_smartseq=args.include_smartseq,
            include_seaad_reference=args.include_seaad_reference,
            feature_panel_paths=args.feature_panels or None,
            sources=args.sources or None,
            seed=args.seed,
        )
    elif args.cmd == "write-scdesign3-configs":
        result = write_generation_configs(
            prepared_source_root=args.prepared_source_root,
            config_root=args.config_root,
            target_total=args.target_total,
            n_cores=args.n_cores,
            seed=args.seed,
        )
    elif args.cmd == "preflight-scdesign3":
        ok, reason = preflight_scdesign3(args.rscript_path)
        result = {"ok": ok, "reason": reason, "rscript_path": args.rscript_path}
    elif args.cmd == "run-scdesign3":
        result = run_generation_configs(
            args.config_manifest,
            rscript_path=args.rscript_path,
            force_refit=args.force_refit,
            dry_run=args.dry_run,
        )
    elif args.cmd == "prepare-eval-datasets":
        result = prepare_all_eval_datasets(
            args.output_dir,
            include_new_synthetic=not args.no_new_synthetic,
            max_cells=args.max_cells,
            seed=args.seed,
        )
    elif args.cmd == "evaluate":
        result = run_evaluation(
            output_dir=args.output_dir,
            registry_path=args.registry,
            dataset_manifest_path=args.dataset_manifest or None,
            include_new_synthetic=not args.no_new_synthetic,
            models=args.models or None,
            datasets=args.datasets or None,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prepare_only=args.prepare_only,
            max_cells=args.max_cells,
            seed=args.seed,
        )
    elif args.cmd == "build-dataset-catalog":
        result = build_dataset_catalog(
            output_dir=args.output_dir,
            include_shape_probe=not args.no_shape_probe,
        )
    elif args.cmd == "raw-label-transfer-smoke":
        result = evaluate_raw_label_transfer(
            output_dir=args.output_dir,
            synthetic_root=args.synthetic_root,
            source_ids=args.sources or None,
            variant_ids=args.variants or None,
            methods=args.methods or None,
            max_reference_cells=args.max_reference_cells,
            max_query_cells=args.max_query_cells,
            min_shared_genes=args.min_shared_genes,
            k=args.k,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            include_existing_seaad=args.include_existing_seaad,
        )
    elif args.cmd == "profile-query":
        result = profile_query(
            dataset_id=args.dataset_id,
            input_path=args.input or None,
            output_dir=args.output_dir or None,
            max_cells=args.max_cells,
            seed=args.seed,
        )
    elif args.cmd == "select-models":
        result = select_models(
            query_profile_path=args.query_profile,
            output_dir=args.output_dir or None,
            artifact_bundle=args.artifact_bundle,
            prepared_source_root=args.prepared_source_root,
            capability_dir=args.capability_dir,
            top_k=args.num_models if args.num_models is not None else args.top_k,
            min_shared_genes=args.min_shared_genes,
            max_source_profile_cells=args.max_source_profile_cells,
            max_query_cells=args.max_query_cells,
            max_reference_cells=args.max_reference_cells,
            k=args.k,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            llm_mode=args.llm_mode,
            llm_model=args.llm_model or None,
            llm_max_candidates=args.llm_max_candidates,
            llm_retry_limit=args.llm_retry_limit,
            selection_strategy=args.selection_strategy,
            selection_objective=args.selection_objective,
            iterative_exclude_scope=args.iterative_exclude_scope,
            excluded_model_ids=args.excluded_models or None,
        )
    elif args.cmd == "run-cross-species-plan":
        result = run_cross_species_plan(
            plan_path=args.plan,
            output_dir=args.output_dir or None,
            max_query_cells=args.max_query_cells,
            max_reference_cells=args.max_reference_cells,
            min_shared_genes=args.min_shared_genes,
            k=args.k,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.cmd == "inspect-model-contracts":
        result = inspect_model_contracts(
            capability_dir=args.capability_dir,
            registry_path=args.registry,
            output_dir=args.output_dir or None,
        )
    elif args.cmd == "adapt-and-execute":
        result = adapt_and_execute(
            plan_path=args.plan,
            mode=args.mode,
            output_dir=args.output_dir or None,
            capability_dir=args.capability_dir,
            registry_path=args.registry,
            resume=args.resume,
            retry_limit=args.retry_limit,
            llm_mode=args.llm_mode,
            llm_model=args.llm_model or None,
            llm_retry_limit=args.llm_retry_limit,
        )
    elif args.cmd == "run-consensus":
        result = run_consensus(
            stage3_summary_path=args.stage3_summary,
            mode=args.mode,
            output_dir=args.output_dir or None,
            seed=args.seed,
            llm_policy_mode=args.llm_policy_mode,
            llm_model=args.llm_model or None,
            llm_cell_adjudication_mode=args.llm_cell_adjudication_mode,
            llm_cell_max_groups=args.llm_cell_max_groups,
            llm_cell_batch_size=args.llm_cell_batch_size,
            model_scope=args.model_scope,
            skip_reference_geometry=args.skip_reference_geometry,
            execution_strategy=args.execution_strategy,
            max_probe_cells=args.max_probe_cells,
        )
    elif args.cmd == "run-uce-ima-transfer":
        result = run_uce_ima_label_transfer(
            dataset_id=args.dataset_id,
            query_path=args.query_path,
            output_dir=args.output_dir,
            stage3_summary_path=args.stage3_summary or None,
            max_query_cells=args.max_query_cells,
            max_reference_cells_per_label=args.max_reference_cells_per_label,
            k=args.k,
            min_vote_share=args.min_vote_share,
            device=args.device,
            batch_size=args.batch_size,
            query_chunk_size=args.query_chunk_size,
            sample_size=args.sample_size,
            pad_length=args.pad_length,
            seed=args.seed,
            reuse_reference_cache=not args.rebuild_reference_cache,
        )
    else:
        parser.error(f"Unhandled command: {args.cmd}")
        return

    print(write_json(result, Path(paths.RUNS_DIR) / f"last_{args.cmd}.json"))


if __name__ == "__main__":
    main()
