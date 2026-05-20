from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scmas import paths
from scmas.io import write_json
from scmas.llm_config import build_openai_client, default_llm_model


DEFAULT_ENV_PATH = paths.SCMAS_ROOT / ".env"
# Legacy summary field kept only for audit continuity; stage-4 selection no longer
# uses fixed agreement thresholds or hand-tuned method families.
HIGH_CONSENSUS_MODEL_AGREEMENT = 0.85


def _load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _default_llm_model() -> str:
    _load_env_file()
    return default_llm_model()


DEFAULT_LLM_MODEL = _default_llm_model()


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _call_openai_json(*, model: str, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    _load_env_file()
    client = build_openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    meta = {
        "provider": "openai",
        "model": model,
        "response_id": getattr(response, "id", ""),
        "usage": getattr(response, "usage", None).model_dump() if getattr(response, "usage", None) is not None else {},
    }
    return _json_from_text(content), content, meta


def label_free_method_diagnostics(
    *,
    consensus_predictions: pd.DataFrame,
    model_ids: list[str],
    method_names: list[str],
    unknown_label: str,
    primary_model: str | None = None,
    method_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_agreements: list[float] = []
    for idx in consensus_predictions.index:
        known: list[str] = []
        for model_id in model_ids:
            model_col = f"{model_id}__pred_shared"
            if model_col not in consensus_predictions.columns:
                continue
            pred = str(consensus_predictions.at[idx, model_col])
            if pred != unknown_label:
                known.append(pred)
        if not known:
            global_agreements.append(0.0)
            continue
        counts = pd.Series(known).value_counts()
        global_agreements.append(float(counts.iloc[0] / len(known)))
    mean_global_agreement = float(np.mean(global_agreements)) if global_agreements else 0.0
    primary_col = f"{primary_model}__pred_shared" if primary_model else ""
    metadata_map = method_metadata or {}

    def _graph_signal(payload: dict[str, Any]) -> float:
        score_maps = []
        for key in ["graph_selection_scores", "query_graph_selection_scores", "reference_graph_selection_scores"]:
            score_map = payload.get(key)
            if isinstance(score_map, dict):
                score_maps.append(score_map)
        selected_values: list[float] = []
        for score_map in score_maps:
            values: list[float] = []
            for item in score_map.values():
                if isinstance(item, dict):
                    try:
                        values.append(float(item.get("score", 0.0) or 0.0))
                    except Exception:
                        pass
            if values:
                # The graph fusion code selects one geometry model per score map.
                # Summarize the usable graph signal by that selected model's score,
                # not by averaging in the rejected geometries.
                selected_values.append(max(values))
        return float(np.mean(selected_values)) if selected_values else 0.0

    for method in method_names:
        column = f"{method}__pred_shared"
        if column not in consensus_predictions.columns:
            continue
        preds = consensus_predictions[column].astype(str)
        non_unknown = preds != unknown_label
        vote_support: list[float] = []
        confidence_support: list[float] = []
        primary_agreement: list[float] = []
        for idx, pred in preds.items():
            if pred == unknown_label:
                vote_support.append(0.0)
                confidence_support.append(0.0)
                primary_agreement.append(0.0)
                continue
            matches = 0
            known = 0
            confs: list[float] = []
            for model_id in model_ids:
                model_col = f"{model_id}__pred_shared"
                conf_col = f"{model_id}__confidence"
                if model_col not in consensus_predictions.columns:
                    continue
                model_pred = str(consensus_predictions.at[idx, model_col])
                if model_pred == unknown_label:
                    continue
                known += 1
                if model_pred == pred:
                    matches += 1
                    if conf_col in consensus_predictions.columns:
                        try:
                            confs.append(float(consensus_predictions.at[idx, conf_col]))
                        except Exception:
                            pass
            vote_support.append(float(matches / max(1, known)))
            confidence_support.append(float(np.mean(confs)) if confs else 0.0)
            if primary_col and primary_col in consensus_predictions.columns:
                primary_pred = str(consensus_predictions.at[idx, primary_col])
                primary_agreement.append(float(primary_pred == pred and primary_pred != unknown_label))
            else:
                primary_agreement.append(0.0)
        payload = metadata_map.get(method, {}) if isinstance(metadata_map.get(method, {}), dict) else {}
        geometry_model_ids = payload.get("geometry_model_ids", [])
        logistic_model_ids = payload.get("reference_logistic_models", [])
        reference_member_count = max(
            len(geometry_model_ids) if isinstance(geometry_model_ids, list) else 0,
            len(logistic_model_ids) if isinstance(logistic_model_ids, list) else 0,
        )
        rows.append(
            {
                "method": method,
                "is_single_model": method.startswith("single__"),
                "accepted_fraction": float(non_unknown.mean()) if len(preds) else 0.0,
                "unknown_fraction": float((~non_unknown).mean()) if len(preds) else 0.0,
                "mean_vote_support": float(np.mean(vote_support)) if vote_support else 0.0,
                "mean_confidence_support": float(np.mean(confidence_support)) if confidence_support else 0.0,
                "primary_model_agreement": float(np.mean(primary_agreement)) if primary_agreement else 0.0,
                "reference_member_count": float(reference_member_count),
                "reference_model_coverage": float(reference_member_count / max(1, len(model_ids))),
                "graph_selection_signal": _graph_signal(payload),
                "reference_switch_fraction": _float_value(payload, "switched_fraction", 0.0),
                "n_predicted_labels": int(preds[non_unknown].nunique()),
                "n_non_unknown_predictions": int(non_unknown.sum()),
                "n_cells_evaluated": int(len(preds)),
                "mean_global_model_agreement": mean_global_agreement,
            }
        )
    return rows


def _float_value(row: dict[str, Any] | None, key: str, default: float) -> float:
    if not row:
        return default
    value = row.get(key, default)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(parsed):
        return default
    return parsed


def _bool_value(row: dict[str, Any] | None, key: str, default: bool = False) -> bool:
    if not row:
        return default
    value = row.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _row_by_method(diagnostics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("method", "")): row for row in diagnostics}


def _rank_percentile(values: pd.Series, *, ascending: bool) -> pd.Series:
    if values.empty:
        return values.astype(float)
    ranks = values.rank(method="average", ascending=ascending)
    if len(values) == 1:
        return pd.Series([1.0], index=values.index, dtype=float)
    return 1.0 - ((ranks - 1.0) / float(len(values) - 1))


def _method_family(method: str) -> str:
    if method.startswith("single__"):
        return "single_model"
    if method in {
        "majority_vote",
        "capability_weighted_vote",
        "confidence_weighted_vote",
        "best_confident_model",
        "agreement_then_confidence",
        "high_conf_majority_override_primary",
        "majority_override_lowest_accept_primary",
        "perfect_supported_majority_override_lowest_accept_primary",
        "weighted_consensus_fixed",
        "stage2_primary_guarded_consensus",
    }:
        return "vote_fusion"
    if method.startswith("reference_logistic"):
        return "reference_logistic"
    if method.startswith(("reference_enhanced", "query_graph", "neighbor_distribution", "density_calibrated")):
        return "reference_graph"
    return "other"


def _family_score_columns(family: str) -> list[str]:
    base = [
        "coverage_retention_evidence_score",
        "support_strength_evidence_score",
        "label_space_retention_evidence_score",
        "primary_model_agreement_rank_percentile",
    ]
    if family in {"reference_graph", "reference_logistic"}:
        return [*base, "reference_structure_evidence_score"]
    return base


def _family_recommendations(table: pd.DataFrame, *, primary_model_id: str | None = None) -> list[dict[str, Any]]:
    if table.empty:
        return []
    recommendations: list[dict[str, Any]] = []
    family_order = ["single_model", "vote_fusion", "reference_graph", "reference_logistic"]
    for family in family_order:
        family_table = table[table["method"].map(_method_family) == family].copy()
        if family_table.empty:
            continue
        score_columns = [column for column in _family_score_columns(family) if column in family_table.columns]
        if not score_columns:
            continue
        family_table["family_score"] = family_table[score_columns].mean(axis=1)
        family_table = family_table.sort_values(
            ["family_score", "composite_score", "method"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        selected_by = "family_score"
        if family == "single_model" and primary_model_id:
            primary_method = f"single__{primary_model_id}"
            primary_row = family_table[family_table["method"].astype(str) == primary_method]
            if not primary_row.empty:
                top = primary_row.iloc[0]
                selected_by = "stage2_rank1_primary_model"
            else:
                top = family_table.iloc[0]
        else:
            top = family_table.iloc[0]
        if family in {"reference_graph", "reference_logistic"}:
            summary = (
                f"reference_structure={float(top.get('reference_structure_evidence_score', 0.0)):.3f}, "
                f"support={float(top.get('support_strength_evidence_score', 0.0)):.3f}, "
                f"coverage={float(top.get('coverage_retention_evidence_score', 0.0)):.3f}, "
                f"labels={int(top.get('n_predicted_labels', 0))}, "
                f"primary_agreement={float(top.get('primary_model_agreement_rank_percentile', 0.0)):.3f}"
            )
        else:
            summary = (
                f"support={float(top.get('support_strength_evidence_score', 0.0)):.3f}, "
                f"coverage={float(top.get('coverage_retention_evidence_score', 0.0)):.3f}, "
                f"labels={int(top.get('n_predicted_labels', 0))}, "
                f"primary_agreement={float(top.get('primary_model_agreement_rank_percentile', 0.0)):.3f}"
            )
        recommendations.append(
            {
                "family": family,
                "champion_method": str(top["method"]),
                "family_score": float(top["family_score"]),
                "champion_composite_score": float(top.get("composite_score", 0.0)),
                "method_count": int(len(family_table)),
                "selected_by": selected_by,
                "summary": summary,
                "top_methods": [
                    {
                        "method": str(row["method"]),
                        "family_score": float(row["family_score"]),
                        "composite_score": float(row.get("composite_score", 0.0)),
                    }
                    for _, row in family_table.head(3).iterrows()
                ],
                "score_columns": score_columns,
                "description": {
                    "single_model": "single-model floor using the best raw embedding executor",
                    "vote_fusion": "non-reference vote / guard / confidence fusion family",
                    "reference_graph": "reference-neighbor and query-graph family",
                    "reference_logistic": "reference-logistic head family",
                }.get(family, "other"),
            }
        )
    recommendations.sort(key=lambda item: (-float(item["family_score"]), item["family"], item["champion_method"]))
    for rank, item in enumerate(recommendations, start=1):
        item["family_rank"] = rank
    return recommendations


def _deployable_family_recommendations(family_recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep vote methods as audit evidence unless no structural family exists.

    Non-reference vote fusion adds no new reference geometry or source prior; it is useful
    diagnostics, but the deployment choice should be the Stage-2 primary single floor or
    a reference-enhanced family when available.
    """
    structural = [item for item in family_recommendations if str(item.get("family", "")) != "vote_fusion"]
    return structural or family_recommendations


def _method_evidence_table(
    diagnostics: list[dict[str, Any]],
    available_methods: set[str],
) -> pd.DataFrame:
    rows = [dict(row) for row in diagnostics if str(row.get("method", "")) in available_methods]
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).drop_duplicates(subset=["method"], keep="first").copy()
    table["method"] = table["method"].astype(str)

    axis_specs = [
        ("accepted_fraction", False),
        ("unknown_fraction", True),
        ("mean_vote_support", False),
        ("mean_confidence_support", False),
        ("n_predicted_labels", False),
        ("reference_model_coverage", False),
        ("graph_selection_signal", False),
        ("reference_switch_fraction", False),
        ("primary_model_agreement", False),
    ]
    for axis, ascending in axis_specs:
        if axis not in table.columns:
            table[axis] = 0.0
        table[f"{axis}_rank_percentile"] = _rank_percentile(
            pd.to_numeric(table[axis], errors="coerce").fillna(0.0),
            ascending=ascending,
        )

    groups: dict[str, list[str]] = {
        "coverage_retention": [
            "accepted_fraction_rank_percentile",
            "unknown_fraction_rank_percentile",
        ],
        "support_strength": [
            "mean_vote_support_rank_percentile",
            "mean_confidence_support_rank_percentile",
        ],
        "label_space_retention": [
            "n_predicted_labels_rank_percentile",
        ],
        "reference_structure": [
            "reference_model_coverage_rank_percentile",
            "graph_selection_signal_rank_percentile",
            "reference_switch_fraction_rank_percentile",
        ],
    }
    for group_name, columns in groups.items():
        present = [column for column in columns if column in table.columns]
        table[f"{group_name}_evidence_score"] = table[present].mean(axis=1) if present else 0.0

    group_score_columns = [f"{group_name}_evidence_score" for group_name in groups]
    table["composite_score"] = table[group_score_columns].mean(axis=1)
    return table


def _pareto_front_mask(table: pd.DataFrame, score_columns: list[str]) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=bool)
    matrix = table[score_columns].fillna(0.0).to_numpy(dtype=float)
    dominated = np.zeros(matrix.shape[0], dtype=bool)
    for i in range(matrix.shape[0]):
        if dominated[i]:
            continue
        for j in range(matrix.shape[0]):
            if i == j:
                continue
            better_or_equal = np.all(matrix[j] >= matrix[i])
            strictly_better = np.any(matrix[j] > matrix[i])
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    return pd.Series(~dominated, index=table.index, dtype=bool)


def _sorted_method_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    order = [
        "is_pareto_front",
        "composite_score",
        "coverage_retention_evidence_score",
        "support_strength_evidence_score",
        "label_space_retention_evidence_score",
        "reference_structure_evidence_score",
        "method",
    ]
    ascending = [False, False, False, False, False, False, True]
    present_order = [column for column in order if column in table.columns]
    present_ascending = [ascending[order.index(column)] for column in present_order]
    return table.sort_values(present_order, ascending=present_ascending).reset_index(drop=True)


def _method_evidence_records(
    diagnostics: list[dict[str, Any]],
    available_methods: set[str],
) -> list[dict[str, Any]]:
    table = _method_evidence_table(diagnostics, available_methods)
    if table.empty:
        return []
    group_score_columns = [
        "coverage_retention_evidence_score",
        "support_strength_evidence_score",
        "label_space_retention_evidence_score",
        "reference_structure_evidence_score",
    ]
    table["is_pareto_front"] = _pareto_front_mask(table, group_score_columns)
    return _sorted_method_table(table).to_dict("records")


REFERENCE_GRAPH_METHOD_PRIORITY = [
    "reference_enhanced_confidence_switch_graph_consensus",
    "query_graph_refined_consensus",
    "reference_enhanced_graph_consensus",
    "density_calibrated_neighbor_distribution_consensus",
    "neighbor_distribution_consensus",
    "reference_enhanced_primary_blend_consensus",
]

REFERENCE_LOGISTIC_METHOD_PRIORITY = [
    "reference_logistic_consensus",
    "reference_logistic_primary_blend_consensus",
    "reference_logistic_graph_consensus",
]


def _method_record(records: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    return next((row for row in records if str(row.get("method", "")) == method), None)


def _first_available_record(records_by_method: dict[str, dict[str, Any]], methods: list[str]) -> dict[str, Any] | None:
    for method in methods:
        row = records_by_method.get(method)
        if row is not None:
            return row
    return None


def _best_single_record(
    records: list[dict[str, Any]],
    *,
    primary_model_id: str | None,
) -> dict[str, Any] | None:
    records_by_method = {str(row.get("method", "")): row for row in records}
    if primary_model_id:
        primary_row = records_by_method.get(f"single__{primary_model_id}")
        if primary_row is not None:
            return primary_row
    singles = [row for row in records if str(row.get("method", "")).startswith("single__")]
    if not singles:
        return None
    return sorted(
        singles,
        key=lambda row: (
            _float_value(row, "support_strength_evidence_score", 0.0),
            _float_value(row, "coverage_retention_evidence_score", 0.0),
            _float_value(row, "label_space_retention_evidence_score", 0.0),
            str(row.get("method", "")),
        ),
        reverse=True,
    )[0]


def _primary_floor_record(
    records: list[dict[str, Any]],
    *,
    primary_model_id: str | None,
) -> dict[str, Any] | None:
    records_by_method = {str(row.get("method", "")): row for row in records}
    if primary_model_id:
        guarded = records_by_method.get("stage2_primary_guarded_consensus")
        if guarded is not None:
            return guarded
    return _best_single_record(records, primary_model_id=primary_model_id)


def _same_or_broader_label_space(candidate: dict[str, Any], references: list[dict[str, Any] | None]) -> bool:
    candidate_labels = int(_float_value(candidate, "n_predicted_labels", 0.0))
    reference_labels = [
        int(_float_value(row, "n_predicted_labels", 0.0))
        for row in references
        if row is not None and int(_float_value(row, "n_predicted_labels", 0.0)) > 0
    ]
    if not reference_labels:
        return candidate_labels > 0
    return candidate_labels >= min(reference_labels)


def _operating_regime_selection(
    records: list[dict[str, Any]],
    *,
    primary_model_id: str | None = None,
) -> dict[str, Any]:
    records_by_method = {str(row.get("method", "")): row for row in records}
    primary_floor = _primary_floor_record(records, primary_model_id=primary_model_id)
    graph_champion = _first_available_record(records_by_method, REFERENCE_GRAPH_METHOD_PRIORITY)
    logistic_champion = _first_available_record(records_by_method, REFERENCE_LOGISTIC_METHOD_PRIORITY)
    single_floor = _best_single_record(records, primary_model_id=primary_model_id)

    decision_trace: list[dict[str, Any]] = []
    deployable_candidates = [
        row
        for row in [logistic_champion, graph_champion, primary_floor, single_floor]
        if row is not None
    ]
    deployable_methods = []
    for row in deployable_candidates:
        method = str(row.get("method", ""))
        if method and method not in deployable_methods:
            deployable_methods.append(method)

    if logistic_champion is not None and _float_value(logistic_champion, "reference_model_coverage", 0.0) > 0.0:
        logistic_accept = _float_value(logistic_champion, "accepted_fraction", 0.0)
        primary_accept = _float_value(primary_floor, "accepted_fraction", 1.0) if primary_floor is not None else 1.0
        graph_accept = _float_value(graph_champion, "accepted_fraction", 1.0) if graph_champion is not None else 1.0
        logistic_is_calibrated_boundary = (
            logistic_accept < primary_accept
            and logistic_accept < graph_accept
            and _same_or_broader_label_space(logistic_champion, [primary_floor, graph_champion])
        )
        decision_trace.append(
            {
                "regime": "reference_logistic_calibrated_boundary",
                "candidate_method": str(logistic_champion.get("method", "")),
                "passed": bool(logistic_is_calibrated_boundary),
                "rule": "reference-logistic must be more selective than the primary floor and reference-graph candidate while retaining the same coarse label space",
                "logistic_accepted_fraction": logistic_accept,
                "primary_floor_accepted_fraction": primary_accept,
                "graph_accepted_fraction": graph_accept,
                "logistic_n_predicted_labels": int(_float_value(logistic_champion, "n_predicted_labels", 0.0)),
            }
        )
        if logistic_is_calibrated_boundary:
            return {
                "selected_method": str(logistic_champion["method"]),
                "selected_row": logistic_champion,
                "selected_regime": "reference_logistic_calibrated_boundary",
                "deployable_methods": deployable_methods,
                "decision_trace": decision_trace,
            }

    if graph_champion is not None and _float_value(graph_champion, "reference_model_coverage", 0.0) > 0.0:
        graph_signal = _float_value(graph_champion, "graph_selection_signal", 0.0)
        graph_has_positive_lift = graph_signal > 0.0
        decision_trace.append(
            {
                "regime": "reference_graph_positive_lift",
                "candidate_method": str(graph_champion.get("method", "")),
                "passed": bool(graph_has_positive_lift),
                "rule": "reference-graph methods deploy only when the selected query/reference graph score is positive on the unlabeled run",
                "graph_selection_signal": graph_signal,
                "reference_switch_fraction": _float_value(graph_champion, "reference_switch_fraction", 0.0),
            }
        )
        if graph_has_positive_lift:
            return {
                "selected_method": str(graph_champion["method"]),
                "selected_row": graph_champion,
                "selected_regime": "reference_graph_positive_lift",
                "deployable_methods": deployable_methods,
                "decision_trace": decision_trace,
            }

    fallback = primary_floor or single_floor or (records[0] if records else {})
    decision_trace.append(
        {
            "regime": "primary_floor_fallback",
            "candidate_method": str(fallback.get("method", "")),
            "passed": True,
            "rule": "use the stage-2 rank-1 guarded primary floor when reference-logistic and reference-graph regimes do not pass label-free checks",
        }
    )
    return {
        "selected_method": str(fallback.get("method", "")),
        "selected_row": fallback,
        "selected_regime": "primary_floor_fallback",
        "deployable_methods": deployable_methods or [str(fallback.get("method", ""))],
        "decision_trace": decision_trace,
    }


def deterministic_policy_method(
    diagnostics: list[dict[str, Any]],
    available_methods: set[str],
    *,
    primary_model_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    evidence = _method_evidence_records(diagnostics, available_methods)
    if not evidence:
        fallback = sorted(available_methods)[0] if available_methods else ""
        return fallback, {
            "mode": "deterministic",
            "reason": "fallback_first_available_method",
            "rank_aggregation_method": "label_free_evidence_group_rank_v1",
            "evidence_table": [],
            "family_recommendations": [],
            "family_champion_methods": [],
            "all_family_champion_methods": [],
        }
    table = pd.DataFrame(evidence)
    family_recommendations = _family_recommendations(table, primary_model_id=primary_model_id)
    operating_regime = _operating_regime_selection(evidence, primary_model_id=primary_model_id)
    top = dict(operating_regime.get("selected_row", {}) or evidence[0])
    top["family"] = _method_family(str(top.get("method", "")))
    return str(top["method"]), {
        "mode": "deterministic",
        "reason": "label_free_evidence_group_rank_v1",
        "rank_aggregation_method": "label_free_evidence_group_rank_v1",
        "rank_aggregation_axis_weighting": "none; axes are grouped before averaging",
        "rank_aggregation_group_weighting": "equal; one group vote per label-free evidence group",
        "rank_aggregation_groups": [
            {
                "group": "coverage_retention",
                "axes": ["accepted_fraction", "unknown_fraction"],
                "description": "retain non-unknown predictions without collapsing to abstention",
            },
            {
                "group": "support_strength",
                "axes": ["mean_vote_support", "mean_confidence_support"],
                "description": "predictions align with model vote support and confidence support on the unlabeled probe",
            },
            {
                "group": "label_space_retention",
                "axes": ["n_predicted_labels"],
                "description": "avoid collapsing the query into too few coarse labels",
            },
            {
                "group": "reference_structure",
                "axes": ["reference_model_coverage", "graph_selection_signal", "reference_switch_fraction"],
                "description": "reward methods whose current-query probe shows usable reference-backed structure, graph lift, or selective switching support",
            },
        ],
        "selected_row": top,
        "operating_regime_selection": operating_regime,
        "pareto_front": [row["method"] for row in evidence if bool(row.get("is_pareto_front", False))],
        "evidence_table": evidence,
        "family_recommendations": family_recommendations,
        "family_champion_methods": [str(operating_regime.get("selected_method", ""))],
        "operating_regime_candidate_methods": [str(item) for item in operating_regime.get("deployable_methods", []) if str(item)],
        "all_family_champion_methods": [str(item["champion_method"]) for item in family_recommendations],
        "deployment_family_rule": (
            "label-free operating-regime gate: reference-logistic calibrated boundary, "
            "reference-graph positive lift, otherwise stage2 primary floor"
        ),
        "family_rank_aggregation_method": "equal-group family champion shortlist over operating regimes",
    }


def _llm_selected_method_allowed(
    *,
    method: str,
    diagnostics: list[dict[str, Any]],
    available_methods: set[str],
    family_champion_methods: list[str] | None = None,
    candidate_methods: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    if method not in available_methods:
        return False, {"reason": "method_not_available"}
    if candidate_methods:
        if method in candidate_methods:
            return True, {
                "reason": "label_free_candidate_shortlist",
                "candidate_methods": candidate_methods,
            }
        return False, {
            "reason": "method_not_in_label_free_candidate_shortlist",
            "candidate_methods": candidate_methods,
        }
    if family_champion_methods:
        if method in family_champion_methods:
            return True, {
                "reason": "family_champion_shortlist",
                "family_champion_methods": family_champion_methods,
            }
        return False, {
            "reason": "method_not_in_family_champion_shortlist",
            "family_champion_methods": family_champion_methods,
        }
    evidence = _method_evidence_records(diagnostics, available_methods)
    if not evidence:
        return True, {"reason": "no_evidence_table_available"}
    row = next((item for item in evidence if str(item.get("method", "")) == method), None)
    if row is None:
        return False, {"reason": "method_missing_evidence_row"}
    allowed = bool(row.get("is_pareto_front", False))
    return allowed, {
        "reason": "pareto_front_guard",
        "allowed": allowed,
        "selected_row": row,
        "pareto_front": [item["method"] for item in evidence if bool(item.get("is_pareto_front", False))],
        "deterministic_default": evidence[0]["method"],
    }


def select_policy_method(
    *,
    diagnostics: list[dict[str, Any]],
    available_methods: list[str],
    output_dir: str | Path,
    llm_mode: str = "off",
    llm_model: str | None = None,
    function_registry: list[dict[str, Any]] | None = None,
    selection_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    available = set(available_methods)
    _load_env_file()
    effective_llm_model = llm_model or default_llm_model(DEFAULT_LLM_MODEL)
    selection_context_payload = dict(selection_context or {})
    stage2_selection = selection_context_payload.get("stage2_selection", {})
    primary_model_id = ""
    if isinstance(stage2_selection, dict):
        selected_ids = stage2_selection.get("selected_model_ids", [])
        if isinstance(selected_ids, list) and selected_ids:
            primary_model_id = str(selected_ids[0])
    deterministic, deterministic_meta = deterministic_policy_method(
        diagnostics,
        available,
        primary_model_id=primary_model_id or None,
    )
    family_recommendations = list(deterministic_meta.get("family_recommendations", []))
    family_champion_methods = [str(item.get("champion_method", "")) for item in family_recommendations if str(item.get("champion_method", ""))]
    deployable_family_champion_methods = [str(item) for item in deterministic_meta.get("family_champion_methods", []) if str(item)]
    operating_regime_candidate_methods = [
        str(item) for item in deterministic_meta.get("operating_regime_candidate_methods", []) if str(item)
    ]
    deployable_function_calls = operating_regime_candidate_methods or deployable_family_champion_methods
    if llm_mode == "off":
        return deterministic, deterministic_meta
    trace_dir = Path(output_dir) / "llm_policy"
    trace_dir.mkdir(parents=True, exist_ok=True)
    evidence_table = deterministic_meta.get("evidence_table", [])
    if family_recommendations:
        probe_context = dict(selection_context_payload.get("probe", {}))
        probe_context["family_recommendations"] = family_recommendations
        probe_context["family_champion_methods"] = deployable_family_champion_methods
        probe_context["all_family_champion_methods"] = family_champion_methods
        probe_context["operating_regime_selection"] = deterministic_meta.get("operating_regime_selection", {})
        probe_context["operating_regime_candidate_methods"] = operating_regime_candidate_methods
        probe_context["deployable_function_calls"] = deployable_function_calls
        selection_context_payload["probe"] = probe_context
    observe = {
        "schema_version": "scmas.stage4.policy_observe.v1",
        "policy": {
            "query_labels_available_to_planner": False,
            "metrics_with_query_truth_available_to_planner": False,
            "allowed_methods": sorted(available),
            "allowed_function_calls": function_registry or [],
            "deterministic_default": deterministic,
            "planner_agent": "llm_stage4_planner",
            "reviewer_agent": "deterministic_stage4_reviewer",
            "rank_aggregation_method": "label_free_evidence_group_rank_v1",
            "llm_output_is_guarded_by_family_champion_reviewer": True,
            "llm_output_is_guarded_by_pareto_front_reviewer": False,
            "family_champion_methods": deployable_family_champion_methods,
            "deployable_function_calls": deployable_function_calls,
            "operating_regime_candidate_methods": operating_regime_candidate_methods,
            "operating_regime_selection": deterministic_meta.get("operating_regime_selection", {}),
            "all_family_champion_methods": family_champion_methods,
            "deployment_family_rule": deterministic_meta.get("deployment_family_rule", ""),
        },
        "label_free_method_diagnostics": diagnostics,
        "ranked_method_evidence": evidence_table,
        "family_recommendations": family_recommendations,
        "selection_context": selection_context_payload,
    }
    write_json(observe, trace_dir / "observe.json")
    system_prompt = (
        "You are the scMAS stage-4 planner agent. Choose one deployment function call for full execution "
        "using only label-free probe evidence, stage-2 selection evidence, stage-3 model summaries, and "
        "reference-geometry availability. Do not use or request query truth labels or query-truth metrics. "
        "Pick a family champion, not an arbitrary method from the full registry. Return only JSON."
    )
    user_prompt = json.dumps(
        {
            "instructions": [
                "Choose exactly one method from observe.policy.deployable_function_calls; this shortlist contains the label-free candidate representatives for reference-logistic, reference-graph, primary-floor, and single-model regimes.",
                "Use observe.policy.operating_regime_selection as the primary decision scaffold, but you may choose another deployable candidate when label-free diagnostics provide a better biological and statistical rationale.",
                "Use family_recommendations and operating_regime_candidate_methods as audit context for the candidate methods considered by the reviewer.",
                "Do not deploy a generic vote_fusion method unless it is the guarded primary-floor candidate in observe.policy.deployable_function_calls.",
                "Use ranked_method_evidence to break ties or explain why the deterministic default is not chosen.",
                "Use selection_context to understand which models/sources Stage 2 selected, how strong their Stage-1/source evidence is, and whether reference geometry or reference-logistic families are actually available.",
                "Treat single models and guarded-primary methods as conservative floors. Treat reference-backed graph/logistic methods as stronger only when the current unlabeled probe evidence supports their operating-regime checks.",
                "Do not infer from hidden query labels, query-truth metrics, or post-hoc benchmark results.",
                "Return deployment_function_call, deployment_family, and rationale. deployment_function_call must exactly match one allowed method/function name.",
            ],
            "response_contract": {
                "deployment_family": "single_model | vote_fusion | reference_graph | reference_logistic",
                "deployment_function_call": "method/function name",
                "rationale": "short audit-friendly reason",
            },
            "observe": observe,
        },
        ensure_ascii=False,
        indent=2,
    )
    (trace_dir / "prompt.md").write_text("## System\n\n" + system_prompt + "\n\n## User\n\n```json\n" + user_prompt + "\n```\n")
    try:
        parsed, raw_text, meta = _call_openai_json(model=effective_llm_model, system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception as exc:  # noqa: BLE001
        error = {"mode": llm_mode, "status": "failed", "error": f"{type(exc).__name__}: {exc}", **deterministic_meta}
        write_json(error, trace_dir / "review.json")
        if llm_mode == "required":
            raise
        return deterministic, error
    (trace_dir / "response.txt").write_text(raw_text)
    method = str(
        parsed.get("deployment_function_call")
        or parsed.get("deployment_method")
        or parsed.get("method")
        or ""
    )
    selected_family = str(parsed.get("deployment_family") or "").strip()
    review = {
        "mode": llm_mode,
        "status": "passed" if method in available else "failed",
        "llm_model": effective_llm_model,
        "meta": meta,
        "parsed": parsed,
        "deterministic_default": deterministic,
        "family_recommendations": family_recommendations,
        "family_champion_methods": deployable_family_champion_methods,
        "deployable_function_calls": deployable_function_calls,
        "operating_regime_candidate_methods": operating_regime_candidate_methods,
        "operating_regime_selection": deterministic_meta.get("operating_regime_selection", {}),
        "all_family_champion_methods": family_champion_methods,
        "selected_family": selected_family,
    }
    write_json(review, trace_dir / "review.json")
    if method in available:
        allowed, guard = _llm_selected_method_allowed(
            method=method,
            diagnostics=diagnostics,
            available_methods=available,
            family_champion_methods=deployable_family_champion_methods,
            candidate_methods=deployable_function_calls,
        )
        review = {**review, "label_free_guard": guard}
        write_json(review, trace_dir / "review.json")
        if allowed:
            return method, review
        fallback_review = {
            **review,
            "status": "rejected_by_label_free_guard",
            "fallback_method": deterministic,
            "fallback_meta": deterministic_meta,
        }
        write_json(fallback_review, trace_dir / "review.json")
        return deterministic, fallback_review
    fallback_review = {
        **review,
        "status": "rejected_unavailable_method",
        "fallback_method": deterministic,
        "fallback_meta": deterministic_meta,
    }
    write_json(fallback_review, trace_dir / "review.json")
    return deterministic, fallback_review
