from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scmas.io import write_json, write_yaml
from scmas.stage4.llm_adjudicator import LLM_ADJUDICATION_METHOD, adjudicate_low_consistency_cells
from scmas.stage4.policy import deterministic_policy_method, label_free_method_diagnostics
from scmas.stage4.consensus import (
    UNKNOWN_LABEL,
    _load_model_weight,
    agreement_then_confidence_predictions,
    build_consensus_frame,
    capability_weighted_vote_predictions,
    label_to_shared_coarse,
    majority_vote_predictions,
    normalize_model_predictions,
    run_consensus,
    seaad_id_to_label,
    seaad_label_to_shared_coarse,
)


def _label_maps() -> dict:
    return {
        "class_labels": ["Neuronal: GABAergic", "Non-neuronal and Non-neural"],
        "subclass_labels": ["Astrocyte", "Endothelial", "L2/3 IT"],
        "supertype_labels": ["Astro_1", "Endo_1", "L2/3 IT_1"],
        "supertype_to_subclass": {
            "Astro_1": "Astrocyte",
            "Endo_1": "Endothelial",
            "L2/3 IT_1": "L2/3 IT",
        },
        "supertype_to_class": {
            "Astro_1": "Non-neuronal and Non-neural",
            "Endo_1": "Non-neuronal and Non-neural",
            "L2/3 IT_1": "Neuronal: Glutamatergic",
        },
    }


def test_seaad_id_and_shared_label_mapping():
    maps = _label_maps()
    assert seaad_id_to_label("subclass", 0, maps) == "Astrocyte"
    assert seaad_label_to_shared_coarse("subclass", "L2/3 IT", maps) == "Neuron"
    assert seaad_label_to_shared_coarse("supertype", "Astro_1", maps) == "Astrocyte"
    assert seaad_label_to_shared_coarse("class", "Neuronal: GABAergic", maps) == "Neuron"


def test_general_label_mapping_to_shared_coarse_or_unknown():
    maps = _label_maps()
    assert label_to_shared_coarse("astrocyte", label_maps=maps) == "Astrocyte"
    assert label_to_shared_coarse("endothelial cell", label_maps=maps) == "Endothelial"
    assert label_to_shared_coarse("oligodendrocyte precursor cell", label_maps=maps) == "OPC"
    assert label_to_shared_coarse("ExN", label_maps=maps) == "Neuron"
    assert label_to_shared_coarse("InN", label_maps=maps) == "Neuron"
    assert label_to_shared_coarse("OL_0", label_maps=maps) == "Oligodendrocyte"
    assert label_to_shared_coarse("Macro", label_maps=maps) == "Microglia"
    assert label_to_shared_coarse("Peri", label_maps=maps) == "Vascular"
    assert label_to_shared_coarse("Schw", label_maps=maps) == UNKNOWN_LABEL


def test_model_weight_uses_capability_score_not_current_query_metrics(tmp_path):
    capability_path = tmp_path / "capability.yaml"
    write_yaml(
        {
            "stage1_evaluation": {
                "source_dataset_scores": [
                    {"source_group": "source_a", "score_available": True, "composite_score": 0.42}
                ]
            }
        },
        capability_path,
    )
    metrics = pd.DataFrame({"model_id": ["model_a"], "macro_f1": [0.99]})
    spec = {"source_id": "source_a", "capability_yaml": str(capability_path)}

    assert _load_model_weight(spec, metrics, "model_a") == 0.42

    spec["source_id"] = "missing_source"
    assert _load_model_weight(spec, metrics, "model_a") == 1.0


def test_normalize_stage3_prediction_schema_for_seaad_ids(tmp_path):
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "dataset_id": ["d1", "d1"],
            "task": ["subclass", "subclass"],
            "cell_id": ["c1", "c2"],
            "true_id": [0, 2],
            "pred_id": [0, 1],
            "confidence": [0.9, 0.8],
        }
    ).to_csv(pred_path, index=False)
    spec = {"model_id": "m1", "source_id": "direct", "capability_yaml": ""}
    out = normalize_model_predictions(
        dataset_id="d1",
        model_id="m1",
        prediction_path=pred_path,
        spec=spec,
        metrics=pd.DataFrame(),
        label_maps=_label_maps(),
    )
    assert out["task"].unique().tolist() == ["subclass"]
    assert out["true_shared"].tolist() == ["Astrocyte", "Neuron"]
    assert out["pred_shared"].tolist() == ["Astrocyte", "Endothelial"]
    assert np.allclose(out["support_base"].to_numpy(), [0.81, 0.64])


def test_position_cell_ids_remap_to_prepared_source_ids(tmp_path):
    import anndata as ad

    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    adata = ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "source_cell_id": ["real_1", "real_2", "dummy_1"],
                "is_scmas_dummy": [False, False, True],
            },
            index=["real_1", "real_2", "dummy_1"],
        ),
    )
    adata.write_h5ad(prepared_dir / "d1.h5ad")

    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "dataset_id": ["d1", "d1"],
            "task": ["subclass", "subclass"],
            "cell_id": ["cell_0", "cell_1"],
            "true_id": [0, 2],
            "pred_id": [0, 1],
            "confidence": [0.9, 0.8],
        }
    ).to_csv(pred_path, index=False)
    out = normalize_model_predictions(
        dataset_id="d1",
        model_id="m1",
        prediction_path=pred_path,
        spec={"model_id": "m1", "runtime_payload": {"prepared_input_dir": str(prepared_dir)}},
        metrics=pd.DataFrame(),
        label_maps=_label_maps(),
    )
    assert out["cell_id"].tolist() == ["real_1", "real_2"]


def test_vote_methods_are_deterministic():
    normalized = pd.DataFrame(
        {
            "dataset_id": ["d"] * 6,
            "model_id": ["m1", "m1", "m2", "m2", "m3", "m3"],
            "source_id": ["s"] * 6,
            "task": ["coarse_label"] * 6,
            "cell_id": ["c1", "c2", "c1", "c2", "c1", "c2"],
            "sample_id": [""] * 6,
            "true_raw": ["Astrocyte", "Neuron"] * 3,
            "pred_raw": ["Astrocyte", "Neuron", "Astrocyte", "Astrocyte", "Neuron", "Neuron"],
            "true_shared": ["Astrocyte", "Neuron"] * 3,
            "pred_shared": ["Astrocyte", "Neuron", "Astrocyte", "Astrocyte", "Neuron", "Neuron"],
            "confidence": [0.9, 0.7, 0.8, 0.95, 0.4, 0.8],
            "model_weight": [1.0, 1.0, 1.0, 1.0, 3.0, 3.0],
            "support_base": [0.81, 0.49, 0.64, 0.9025, 0.16, 0.64],
            "prediction_path": [""] * 6,
            "adapter_spec": [""] * 6,
        }
    )
    frame, model_ids = build_consensus_frame(normalized)
    assert model_ids == ["m1", "m2", "m3"]
    assert majority_vote_predictions(frame, model_ids).tolist() == ["Astrocyte", "Neuron"]
    assert capability_weighted_vote_predictions(frame, model_ids).tolist() == ["Neuron", "Neuron"]
    assert agreement_then_confidence_predictions(frame, model_ids).tolist() == ["Astrocyte", "Neuron"]


def test_stage4_policy_uses_equal_group_rank_not_fixed_thresholds():
    predictions = pd.DataFrame(
        {
            "m1__pred_shared": ["Astrocyte", "Neuron", "Neuron", "Astrocyte", "Microglia"],
            "m1__confidence": [0.9, 0.9, 0.9, 0.9, 0.9],
            "m2__pred_shared": ["Astrocyte", "Neuron", "Neuron", "Astrocyte", "Microglia"],
            "m2__confidence": [0.8, 0.8, 0.8, 0.8, 0.8],
            "m3__pred_shared": ["Astrocyte", "Neuron", "Astrocyte", "Astrocyte", "Microglia"],
            "m3__confidence": [0.7, 0.7, 0.7, 0.7, 0.7],
            "query_graph_refined_consensus__pred_shared": ["Astrocyte", "Neuron", "Neuron", "Astrocyte", "Microglia"],
            "stage2_primary_guarded_consensus__pred_shared": ["Astrocyte", "Neuron", "Astrocyte", "Astrocyte", "Microglia"],
            "single__m1__pred_shared": ["Astrocyte", "Neuron", "Neuron", "Astrocyte", "Microglia"],
        }
    )
    diagnostics = label_free_method_diagnostics(
        consensus_predictions=predictions,
        model_ids=["m1", "m2", "m3"],
        method_names=["query_graph_refined_consensus", "stage2_primary_guarded_consensus", "single__m1"],
        unknown_label=UNKNOWN_LABEL,
        primary_model="m1",
    )
    method, meta = deterministic_policy_method(
        diagnostics,
        {"query_graph_refined_consensus", "stage2_primary_guarded_consensus", "single__m1"},
    )
    assert method == "single__m1"
    assert meta["reason"] == "label_free_evidence_group_rank_v1"
    assert meta["selected_row"]["family"] == "single_model"
    assert meta["family_recommendations"][0]["champion_method"] == "single__m1"
    assert meta["family_champion_methods"] == ["single__m1"]
    assert meta["operating_regime_selection"]["selected_regime"] == "primary_floor_fallback"
    assert meta["operating_regime_candidate_methods"] == ["query_graph_refined_consensus", "single__m1"]
    assert meta["all_family_champion_methods"] == ["single__m1", "query_graph_refined_consensus", "stage2_primary_guarded_consensus"]
    assert meta["rank_aggregation_group_weighting"] == "equal; one group vote per label-free evidence group"


def test_llm_cell_adjudication_uses_label_free_group_evidence(tmp_path, monkeypatch):
    def fake_call_openai_json(*, model, system_prompt, user_prompt):
        return (
            {
                "groups": [
                    {
                        "group_id": "g00000",
                        "selected_label": "Neuron",
                        "confidence": 0.72,
                        "rationale": "two methods and one model support neuron",
                    }
                ]
            },
            '{"groups":[{"group_id":"g00000","selected_label":"Neuron","confidence":0.72,"rationale":"two methods and one model support neuron"}]}',
            {"model": model},
        )

    monkeypatch.setattr("scmas.stage4.llm_adjudicator._call_openai_json", fake_call_openai_json)
    predictions = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "true_shared": ["Astrocyte", "Neuron"],
            "m1__pred_shared": ["Astrocyte", "Astrocyte"],
            "m1__confidence": [0.6, 0.6],
            "m2__pred_shared": ["Neuron", "Neuron"],
            "m2__confidence": [0.8, 0.8],
            "stage2_primary_guarded_consensus__pred_shared": ["Astrocyte", "Astrocyte"],
            "confidence_weighted_vote__pred_shared": ["Neuron", "Neuron"],
        }
    )
    pred, meta = adjudicate_low_consistency_cells(
        consensus_predictions=predictions,
        model_ids=["m1", "m2"],
        candidate_method_names=["stage2_primary_guarded_consensus", "confidence_weighted_vote"],
        base_method="stage2_primary_guarded_consensus",
        allowed_labels=["Astrocyte", "Neuron", UNKNOWN_LABEL],
        unknown_label=UNKNOWN_LABEL,
        output_dir=tmp_path,
        llm_mode="required",
        llm_model="fake-model",
    )
    assert meta["status"] == "completed"
    assert pred is not None
    assert pred.tolist() == ["Neuron", "Neuron"]
    observe_text = (tmp_path / "llm_cell_adjudication" / "observe.json").read_text()
    assert "true_shared" not in observe_text
    assert LLM_ADJUDICATION_METHOD in observe_text


def test_run_consensus_tiny_stage3_summary(tmp_path):
    pred1 = tmp_path / "m1_predictions.csv"
    pred2 = tmp_path / "m2_predictions.csv"
    for path, preds in [(pred1, ["astrocyte", "neuron"]), (pred2, ["astrocyte", "astrocyte"])]:
        pd.DataFrame(
            {
                "dataset_id": ["tiny", "tiny"],
                "model_id": [path.stem.split("_")[0]] * 2,
                "source_id": ["s", "s"],
                "method": ["raw", "raw"],
                "task": ["coarse_label", "coarse_label"],
                "cell_id": ["c1", "c2"],
                "sample_id": ["s1", "s1"],
                "true_label": ["Astro_GM", "Neurons"],
                "pred_label": preds,
                "confidence": [0.9, 0.8],
            }
        ).to_csv(path, index=False)
    spec1 = tmp_path / "m1.yaml"
    spec2 = tmp_path / "m2.yaml"
    write_yaml({"model_id": "m1", "source_id": "s", "actions": [{"action": "skip_with_reason"}]}, spec1)
    write_yaml({"model_id": "m2", "source_id": "s", "actions": [{"action": "skip_with_reason"}]}, spec2)
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame({"model_id": ["m1", "m2"], "macro_f1": [0.5, 0.4]}).to_csv(metrics_path, index=False)
    summary_path = tmp_path / "stage3_summary.json"
    write_json(
        {
            "dataset_id": "tiny",
            "mode": "subset",
            "completed_models": ["m1", "m2"],
            "prediction_artifacts": {"m1": str(pred1), "m2": str(pred2)},
            "adapter_specs": {"m1": str(spec1), "m2": str(spec2)},
            "metrics_csv": str(metrics_path),
        },
        summary_path,
    )
    result = run_consensus(stage3_summary_path=summary_path, output_dir=tmp_path / "stage4")
    assert result["ready_for_report"] is True
    assert Path(result["prediction_artifacts"]["consensus_predictions"]).exists()
    assert Path(result["prediction_artifacts"]["fusion_function_calls"]).exists()
    assert Path(result["prediction_artifacts"]["fusion_function_calls_csv"]).exists()
    assert Path(result["prediction_artifacts"]["probe_function_calls"]).exists()
    assert result["execution_strategy"] == "selected_only"
    assert len([row for row in result["fusion_function_calls"] if row["status"] == "executed"]) == 1
    assert result["primary_method"] in result["fusion_methods"]


def test_run_consensus_random_mixed_model_subset(tmp_path):
    import anndata as ad

    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "source_cell_id": ["real_1", "real_2", "dummy_1"],
                "is_scmas_dummy": [False, False, True],
            },
            index=["real_1", "real_2", "dummy_1"],
        ),
    ).write_h5ad(prepared_dir / "mixed.h5ad")

    model_specs = {
        "label_explicit": (
            {
                "cell_id": ["real_1", "real_2"],
                "true_label": ["Astro_GM", "Neurons"],
                "pred_label": ["Astro", "ExN"],
            },
            {},
        ),
        "id_explicit": (
            {
                "cell_id": ["real_1", "real_2"],
                "true_id": [0, 2],
                "pred_id": [0, 2],
            },
            {},
        ),
        "position_sklearn": (
            {
                "cell_id": ["cell_0", "cell_1"],
                "true_id": [0, 2],
                "pred_id": [0, 2],
            },
            {"runtime_payload": {"prepared_input_dir": str(prepared_dir)}},
        ),
        "coarse_aliases": (
            {
                "cell_id": ["real_1", "real_2"],
                "true_label": ["astrocyte", "neuron"],
                "pred_label": ["Astro", "InN"],
            },
            {},
        ),
        "missing_confidence": (
            {
                "cell_id": ["real_1", "real_2"],
                "true_label": ["Astrocyte", "Neuron"],
                "pred_label": ["Astrocyte", "MSN"],
            },
            {},
        ),
        "macro_alias": (
            {
                "cell_id": ["real_1", "real_2"],
                "true_label": ["Astrocyte", "Neuron"],
                "pred_label": ["Macro", "OL_0"],
            },
            {},
        ),
    }
    rng = np.random.default_rng(3028)
    sampled_models = sorted(rng.choice(list(model_specs), size=5, replace=False).tolist())
    prediction_artifacts = {}
    adapter_specs = {}
    for idx, model_id in enumerate(sampled_models):
        columns, spec_extra = model_specs[model_id]
        frame = pd.DataFrame(
            {
                "dataset_id": ["mixed", "mixed"],
                "model_id": [model_id, model_id],
                "source_id": ["source", "source"],
                "method": ["compat", "compat"],
                "task": ["subclass", "subclass"],
                "confidence": [0.9 - (idx * 0.02), 0.8 - (idx * 0.02)],
                **columns,
            }
        )
        if model_id == "missing_confidence":
            frame = frame.drop(columns=["confidence"])
        pred_path = tmp_path / f"{model_id}_predictions.csv"
        frame.to_csv(pred_path, index=False)
        spec_path = tmp_path / f"{model_id}.yaml"
        spec = {"model_id": model_id, "source_id": "source", "actions": [{"action": "skip_with_reason"}], **spec_extra}
        write_yaml(spec, spec_path)
        prediction_artifacts[model_id] = str(pred_path)
        adapter_specs[model_id] = str(spec_path)

    summary_path = tmp_path / "stage3_summary.json"
    write_json(
        {
            "dataset_id": "mixed",
            "mode": "subset",
            "completed_models": sampled_models,
            "prediction_artifacts": prediction_artifacts,
            "adapter_specs": adapter_specs,
            "metrics_csv": "",
        },
        summary_path,
    )
    result = run_consensus(
        stage3_summary_path=summary_path,
        output_dir=tmp_path / "stage4_mixed",
        execution_strategy="benchmark_all",
    )
    assert result["ready_for_report"] is True
    assert len(result["normalized_models"]) == 5
    assert result["n_cells_aligned"] == 2
    assert "majority_vote" in result["fusion_methods"]
