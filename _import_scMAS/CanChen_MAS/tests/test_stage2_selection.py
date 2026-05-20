from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from scmas.io import read_json, write_json, write_standard_bundle, write_yaml
from scmas.stage2.selector import (
    _capability_source_score_entry,
    _score_info_from_capability_source,
    load_query_bundle,
    profile_query,
    select_models,
)


def test_kukanja_npz_adapter_reads_gene_names_and_label_maps(tmp_path):
    path = tmp_path / "tiny_kukanja.npz"
    meta = {
        "label_names": ["broad", "fine"],
        "gene_names": ["GeneA", "GeneB", "GeneC"],
        "label_maps": {
            "broad": {"Astro": 0, "Neuron": 1},
            "fine": {"Astro_A": 0, "Neuron_A": 1},
        },
        "sample_map": {"S1": 0, "S2": 1},
    }
    np.savez(
        path,
        X=np.asarray([[1, 0, 2], [0, 3, 1], [2, 2, 0]], dtype=np.float32),
        sample_ids=np.asarray([0, 1, 1], dtype=np.int64),
        meta=np.asarray(meta, dtype=object),
        y_broad=np.asarray([0, 1, 1], dtype=np.int64),
        y_fine=np.asarray([0, 1, 1], dtype=np.int64),
    )

    loaded = load_query_bundle(path, dataset_id="tiny_kukanja", max_cells=2, seed=1)
    assert loaded.adapter == "npz_kukanja"
    assert loaded.bundle.X.shape == (2, 3)
    assert loaded.bundle.genes == ["GENEA", "GENEB", "GENEC"]
    assert {"native_label", "coarse_label", "sample_id"}.issubset(loaded.bundle.obs.columns)


def test_h5ad_adapter_detects_gene_label_and_donor_columns(tmp_path):
    path = tmp_path / "tiny.h5ad"
    adata = ad.AnnData(
        X=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "Supertype": ["A", "B", "A"],
                "Subclass": ["AA", "BB", "AA"],
                "Donor ID": ["D1", "D2", "D1"],
            },
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame({"feature_id": ["GeneA", "GeneB"]}, index=["GeneA", "GeneB"]),
    )
    adata.write_h5ad(path)

    loaded = load_query_bundle(path, dataset_id="tiny_h5ad", max_cells=2, seed=1)
    assert loaded.adapter == "h5ad"
    assert loaded.native_label_column == "Supertype"
    assert loaded.coarse_label_column == "Subclass"
    assert loaded.sample_column == "Donor ID"
    assert loaded.bundle.genes == ["GENEA", "GENEB"]


def test_profile_query_is_gene_only_even_when_labels_exist(tmp_path):
    path = tmp_path / "tiny_labeled.h5ad"
    adata = ad.AnnData(
        X=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
        obs=pd.DataFrame(
            {"Supertype": ["A", "B", "A"], "Subclass": ["AA", "BB", "AA"], "Donor ID": ["D1", "D2", "D1"]},
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame({"feature_id": ["GeneA", "GeneB"]}, index=["GeneA", "GeneB"]),
    )
    adata.write_h5ad(path)

    result = profile_query(dataset_id="tiny_labeled", input_path=path, output_dir=tmp_path, max_cells=2, seed=1)
    profile = read_json(result["query_profile_path"])
    assert profile["query_visibility"] == "gene_names_only"
    assert profile["genes"] == ["GENEA", "GENEB"]
    forbidden = {
        "native_label_column",
        "coarse_label_column",
        "sample_column",
        "native_label_counts",
        "coarse_label_counts",
        "sample_counts",
        "mean_by_gene",
        "detection_by_gene",
        "top_variable_genes",
    }
    assert forbidden.isdisjoint(profile)


def test_capability_source_scores_feed_stage2_evidence():
    capability = {
        "stage1_evaluation": {
            "source_dataset_scores": [
                {
                    "source_group": "source_a",
                    "status": "scored",
                    "score_available": True,
                    "composite_score": 0.42,
                    "mean_macro_f1": 0.4,
                    "best_macro_f1": 0.5,
                    "baseline_macro_f1_mean": 0.45,
                    "variant_macro_f1_mean": 0.35,
                    "robustness_ratio_mean": 0.75,
                    "rows": 10,
                    "source_description": {
                        "description": "Prepared source A.",
                        "species": "mouse",
                        "source_path": "data/prepared_sources/source_a",
                    },
                }
            ]
        }
    }

    row = _capability_source_score_entry(capability, "source_a")
    score_info = _score_info_from_capability_source("model_a", row)

    assert score_info["score_source"] == "capability_source_dataset_scores_source"
    assert score_info["source_dataset_composite_score"] == 0.42
    assert score_info["source_model_macro_f1"] == 0.4
    assert score_info["robustness"] == 0.75
    assert score_info["source_dataset_path"] == "data/prepared_sources/source_a"


def test_selector_outputs_execution_ready_model_ids_without_species_filter(tmp_path):
    genes = [f"GENE{idx:03d}" for idx in range(40)]
    prepared_root = tmp_path / "prepared_sources"
    for source_id, offset in [("mouse_reference", 0), ("human_reference", 5)]:
        source_genes = genes[offset:] + [f"EXTRA{idx:03d}" for idx in range(offset)]
        write_standard_bundle(
            counts_cell_by_gene=sparse.csr_matrix(np.ones((6, len(source_genes)), dtype=np.float32)),
            obs=pd.DataFrame(
                {
                    "cell_id": [f"{source_id}_cell_{idx}" for idx in range(6)],
                    "native_label": ["Astro", "Neuron", "Astro", "Neuron", "Astro", "Neuron"],
                    "coarse_label": ["Astro", "Neuron", "Astro", "Neuron", "Astro", "Neuron"],
                    "sample_id": ["S1", "S1", "S2", "S2", "S3", "S3"],
                }
            ),
            var=pd.DataFrame({"feature_id": source_genes}),
            output_dir=prepared_root / source_id,
            source_metadata={"source_id": source_id, "species": "mouse" if source_id.startswith("mouse") else "human"},
        )

    cap_dir = tmp_path / "capability"
    cap_dir.mkdir()
    for model_id, family, embedding in [
        ("geneformer_raw_knn", "geneformer", "geneformer_raw"),
        ("scgpt_human_raw_knn", "scgpt", "scgpt_human_raw"),
        ("uce_4l_raw_knn", "uce", "uce_4l_raw"),
    ]:
        write_yaml(
            {
                "model_id": model_id,
                "family": family,
                "evaluator": "raw_label_transfer",
                "data_constraints": {"compatible_contract": "source_gene_panel_label_transfer"},
                "input_requirements": {"required_formats": ["standard_bundle"], "required_fields": ["X"]},
                "executor_defaults": {"embedding_method": embedding, "transfer_method": "knn"},
                "artifacts": {},
            },
            cap_dir / f"{model_id}.yaml",
        )

    profile = {
        "dataset_id": "fake_cross_species_query",
        "input_path": "/tmp/fake_cross_species_query.h5ad",
        "query_adapter": "h5ad",
        "profile_cells": 50,
        "n_obs_profiled": 50,
        "n_vars": len(genes),
        "native_label_column": "native_label",
        "coarse_label_column": "coarse_label",
        "sample_column": "sample_id",
        "native_label_counts": {"Astrocytes": 25, "Neurons": 25},
        "coarse_label_counts": {"Astrocytes": 25, "Neurons": 25},
        "sample_counts": {"S1": 50},
        "genes": genes,
        "mean_by_gene": {gene: float(idx + 1) for idx, gene in enumerate(genes)},
        "detection_by_gene": {gene: 0.5 for gene in genes},
        "top_variable_genes": genes[:20],
    }
    profile_path = tmp_path / "query_profile.json"
    write_json(profile, profile_path)

    result = select_models(
        query_profile_path=profile_path,
        output_dir=tmp_path,
        artifact_bundle=tmp_path / "artifacts",
        prepared_source_root=prepared_root,
        capability_dir=cap_dir,
        top_k=3,
        min_shared_genes=10,
        max_source_profile_cells=50,
        max_query_cells=100,
        max_reference_cells=50,
        llm_mode="off",
    )
    plan = yaml.safe_load(Path(result["selected_execution_plan"]).read_text())
    source_similarity = pd.read_csv(tmp_path / "source_similarity.csv")
    assert source_similarity["expression_score"].isna().all()
    assert source_similarity["celltype_score"].isna().all()
    assert plan["selection_policy"]["query_visibility"] == "gene_names_only"
    assert plan["selection_policy"]["selection_objective"] == "unified_rank"
    assert plan["selection_policy"]["single_rule_for_top1_and_topn"] is True
    assert plan["selection_policy"]["rank_aggregation_method"] == "evidence_group_rank_v1"
    assert plan["selection_policy"]["rank_aggregation_group_weighting"].startswith("equal")
    assert plan["selection_policy"]["query_labels_visible_to_selector"] is False
    assert plan["selection_policy"]["query_expression_visible_to_selector"] is False
    assert len(plan["selected_model_ids"]) == 3
    assert plan["review_status"]["status"] == "passed"
    for pair in plan["selected_pairs"]:
        assert pair["model_id"]
        assert pair["source_id"]
        assert pair["reference_path"]
        assert pair["capability_yaml"]
        assert pair["shared_genes"] >= 10
        assert pair["execution_ready"] is True
        assert pair["rank_aggregation_method"] == "evidence_group_rank_v1"
        assert "model_gene_coverage" in pair
        assert "source_model_macro_f1_lcb" in pair
        assert "query_source_gene_fit_evidence_score" in pair
        assert pair["selection_adjustment"] == 0.0
        capability = yaml.safe_load(Path(pair["capability_yaml"]).read_text())
        assert capability["data_constraints"]["compatible_contract"] != "seaad_140_npz"
