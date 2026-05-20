from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from scmas.io import write_standard_bundle, write_yaml
from scmas.stage3.adapter_executor import (
    ModelContract,
    _build_adapter_specs,
    adapt_and_execute,
    inspect_model_contracts,
    validate_adapter_spec,
)


def _minimal_spec() -> dict:
    return {
        "model_id": "m1",
        "source_id": "s1",
        "input_artifacts": {"query_path": "/tmp/query.h5ad"},
        "gene_strategy": {"strategy": "none"},
        "label_strategy": {"strategy": "none"},
        "runtime_payload": {"skip_reason": "unit_test"},
        "actions": [{"action": "skip_with_reason", "reason": "unit_test"}],
        "expected_outputs": {"predictions_csv": "/tmp/pred.csv", "metrics_csv": "/tmp/metrics.csv"},
    }


def _write_stage3_contract_inputs(tmp_path: Path) -> tuple[dict, Path, Path]:
    cap_dir = tmp_path / "capability"
    cap_dir.mkdir()
    for payload in [
        {
            "model_id": "expression_log1p_prototype",
            "family": "expression_baseline",
            "evaluator": "raw_label_transfer",
            "input_requirements": {"required_formats": ["standard_bundle"], "required_fields": ["X"]},
            "data_constraints": {"compatible_contract": "source_gene_panel_label_transfer"},
            "artifacts": {},
        },
        {
            "model_id": "geneformer_raw_knn",
            "family": "geneformer",
            "evaluator": "raw_label_transfer",
            "input_requirements": {"required_formats": ["standard_bundle"], "required_fields": ["X"]},
            "data_constraints": {"compatible_contract": "source_gene_panel_label_transfer"},
            "artifacts": {},
        },
        {
            "model_id": "sklearn_lr",
            "family": "sklearn",
            "evaluator": "sklearn_pkl",
            "input_requirements": {"required_formats": ["npz"], "required_fields": ["X"]},
            "data_constraints": {"compatible_contract": "seaad_140_npz"},
            "artifacts": {},
        },
    ]:
        write_yaml(payload, cap_dir / f"{payload['model_id']}.yaml")
    registry = tmp_path / "registry.yaml"
    write_yaml({"models": []}, registry)
    plan = {
        "dataset_id": "tiny_query",
        "query_path": str(tmp_path / "tiny_query.npz"),
        "query_adapter": "npz_kukanja",
        "selected_model_ids": ["expression_log1p_prototype"],
        "selected_pairs": [
            {
                "model_id": "expression_log1p_prototype",
                "source_id": "allen_mouse_reference_bf8cb800",
                "capability_yaml": str(cap_dir / "expression_log1p_prototype.yaml"),
                "reference_path": str(tmp_path / "prepared_sources" / "allen_mouse_reference_bf8cb800"),
                "method": "expression_log1p_prototype",
                "embedding_method": "expression_log1p",
                "transfer_method": "prototype",
                "shared_genes": 40,
                "execution_ready": True,
            }
        ],
        "execution_defaults": {"min_shared_genes": 10, "max_query_cells": 100, "max_reference_cells": 50},
    }
    return plan, cap_dir, registry


def test_inspect_model_contracts_extracts_wrapper_signature(tmp_path):
    wrapper = tmp_path / "tool_wrapper.py"
    wrapper.write_text(
        "def demo_classifier_tool(npz_path, h5ad_path, result_json_path, *, bs=16, device='cpu'):\n"
        "    return None\n",
        encoding="utf-8",
    )
    cap_dir = tmp_path / "capability"
    cap_dir.mkdir()
    write_yaml(
        {
            "model_id": "demo_model",
            "family": "demo",
            "evaluator": "mcp_tool",
            "input_requirements": {
                "required_formats": ["npz", "h5ad"],
                "required_fields": ["X"],
                "gene_contract": "SEA-AD MERFISH 140-gene panel",
            },
            "data_constraints": {"compatible_contract": "seaad_140_npz"},
            "artifacts": {"wrapper": {"path": str(wrapper)}},
        },
        cap_dir / "demo_model.yaml",
    )
    registry = tmp_path / "registry.yaml"
    write_yaml({"models": []}, registry)

    result = inspect_model_contracts(capability_dir=cap_dir, registry_path=registry, output_dir=tmp_path / "contracts")
    contract = result["contracts"][0]
    assert contract["model_id"] == "demo_model"
    assert contract["wrapper_signature"]["status"] == "ok"
    assert [p["name"] for p in contract["wrapper_signature"]["parameters"]] == [
        "npz_path",
        "h5ad_path",
        "result_json_path",
        "bs",
        "device",
    ]
    assert Path(result["contracts_json"]).exists()


def test_adapter_spec_schema_rejects_unknown_actions_and_executable_keys():
    spec = _minimal_spec()
    validate_adapter_spec(spec)

    bad_action = _minimal_spec()
    bad_action["actions"] = [{"action": "run_arbitrary_shell"}]
    with pytest.raises(ValueError, match="Unknown adapter action"):
        validate_adapter_spec(bad_action)

    bad_key = _minimal_spec()
    bad_key["runtime_payload"]["cmd"] = "rm -rf /"
    with pytest.raises(ValueError, match="forbidden executable key"):
        validate_adapter_spec(bad_key)


def test_stage3_specs_bind_selected_pairs_and_skip_unselected_raw(tmp_path):
    plan, cap_dir, registry = _write_stage3_contract_inputs(tmp_path)
    contract_rows = inspect_model_contracts(
        capability_dir=cap_dir,
        registry_path=registry,
    )["contracts"]
    contracts = {row["model_id"]: ModelContract(**row) for row in contract_rows}

    specs = _build_adapter_specs(plan=plan, contracts=contracts, output_dir=tmp_path, mode="subset")
    by_model = {spec["model_id"]: spec for spec in specs}

    selected = by_model["expression_log1p_prototype"]
    assert selected["source_id"] == "allen_mouse_reference_bf8cb800"
    assert selected["input_artifacts"]["reference_path"].endswith("allen_mouse_reference_bf8cb800")
    assert any(action["action"] == "invoke_raw_embedding_transfer" for action in selected["actions"])
    assert selected["gene_strategy"]["species_is_filter"] is False

    skipped = by_model["geneformer_raw_knn"]
    assert skipped["actions"] == [
        {"action": "skip_with_reason", "reason": "not_selected_by_stage2_plan_no_bound_source_reference"}
    ]


def test_stage3_specs_disable_seaad_140_heads_and_use_raw_knn_policy(tmp_path):
    plan, cap_dir, registry = _write_stage3_contract_inputs(tmp_path)
    contract_rows = inspect_model_contracts(
        capability_dir=cap_dir,
        registry_path=registry,
    )["contracts"]
    contracts = {row["model_id"]: ModelContract(**row) for row in contract_rows}

    specs = _build_adapter_specs(plan=plan, contracts=contracts, output_dir=tmp_path, mode="subset")
    sklearn_lr = {spec["model_id"]: spec for spec in specs}["sklearn_lr"]
    assert sklearn_lr["runtime_payload"]["skip_reason"] == "direct_seaad_140_head_disabled_use_raw_label_transfer_knn"
    assert sklearn_lr["source_id"] == "disabled_direct_head"
    assert all(action["action"] != "write_seaad_140_npz" for action in sklearn_lr["actions"])


def test_adapt_and_execute_runs_tiny_raw_label_transfer_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("SCMAS_EMBEDDING_CACHE", "0")

    query_path = tmp_path / "tiny_kukanja.npz"
    meta = {
        "label_names": ["broad", "fine"],
        "gene_names": ["GeneA", "GeneB", "GeneC"],
        "label_maps": {
            "broad": {"Astro": 0, "Neuron": 1},
            "fine": {"Astro_A": 0, "Neuron_A": 1},
        },
        "sample_map": {"S1": 0},
    }
    np.savez(
        query_path,
        X=np.asarray([[3, 0, 1], [0, 2, 2], [4, 0, 0], [0, 3, 1]], dtype=np.float32),
        sample_ids=np.asarray([0, 0, 0, 0], dtype=np.int64),
        meta=np.asarray(meta, dtype=object),
        y_broad=np.asarray([0, 1, 0, 1], dtype=np.int64),
        y_fine=np.asarray([0, 1, 0, 1], dtype=np.int64),
    )

    reference_dir = tmp_path / "reference_bundle"
    write_standard_bundle(
        counts_cell_by_gene=sparse.csr_matrix(
            np.asarray([[4, 0, 1], [0, 3, 1], [5, 0, 0], [0, 4, 2]], dtype=np.float32)
        ),
        obs=pd.DataFrame(
            {
                "cell_id": ["r1", "r2", "r3", "r4"],
                "native_label": ["Astro_A", "Neuron_A", "Astro_A", "Neuron_A"],
                "coarse_label": ["Astro", "Neuron", "Astro", "Neuron"],
                "sample_id": ["RS1", "RS1", "RS2", "RS2"],
            }
        ),
        var=pd.DataFrame({"feature_id": ["GENEA", "GENEB", "GENEC"]}),
        output_dir=reference_dir,
        source_metadata={"species": "mouse"},
    )

    cap_dir = tmp_path / "capability"
    cap_dir.mkdir()
    write_yaml(
        {
            "model_id": "expression_log1p_prototype",
            "family": "expression_baseline",
            "evaluator": "raw_label_transfer",
            "input_requirements": {
                "required_formats": ["standard_bundle", "npz"],
                "required_fields": ["X", "gene symbols", "native_label"],
                "gene_contract": "source-specific gene panel",
            },
            "data_constraints": {"compatible_contract": "source_gene_panel_label_transfer"},
            "artifacts": {},
        },
        cap_dir / "expression_log1p_prototype.yaml",
    )
    registry = tmp_path / "registry.yaml"
    write_yaml({"models": []}, registry)
    plan_path = tmp_path / "selected_execution_plan.yaml"
    write_yaml(
        {
            "dataset_id": "tiny_query",
            "query_path": str(query_path),
            "query_adapter": "npz_kukanja",
            "selected_model_ids": ["expression_log1p_prototype"],
            "selected_pairs": [
                {
                    "model_id": "expression_log1p_prototype",
                    "source_id": "tiny_reference",
                    "capability_yaml": str(cap_dir / "expression_log1p_prototype.yaml"),
                    "reference_path": str(reference_dir),
                    "method": "expression_log1p_prototype",
                    "embedding_method": "expression_log1p",
                    "transfer_method": "prototype",
                    "shared_genes": 3,
                    "execution_ready": True,
                }
            ],
            "execution_defaults": {
                "max_query_cells": 4,
                "max_reference_cells": 4,
                "min_shared_genes": 2,
                "k": 1,
                "seed": 1,
                "device": "",
                "batch_size": 4,
            },
        },
        plan_path,
    )

    result = adapt_and_execute(
        plan_path=plan_path,
        mode="subset",
        output_dir=tmp_path / "stage3",
        capability_dir=cap_dir,
        registry_path=registry,
        retry_limit=0,
        llm_mode="off",
    )
    assert result["completed_models"] == ["expression_log1p_prototype"]
    assert result["ready_for_consensus"] is False
    assert Path(result["prediction_artifacts"]["expression_log1p_prototype"]).exists()
