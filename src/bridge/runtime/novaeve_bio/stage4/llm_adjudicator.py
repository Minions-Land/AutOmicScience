from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from novaeve_bio.io import ensure_dir, write_json
from novaeve_bio.llm_config import default_llm_model
from novaeve_bio.stage4.policy import DEFAULT_LLM_MODEL, HIGH_CONSENSUS_MODEL_AGREEMENT, _call_openai_json, _load_env_file


LLM_ADJUDICATION_METHOD = "llm_low_consistency_adjudication"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(parsed):
        return default
    return parsed


def _confidence_bin(value: Any, *, width: float = 0.10) -> float:
    parsed = min(max(_safe_float(value, 0.0), 0.0), 1.0)
    return round(round(parsed / width) * width, 2)


def _cell_vote_summary(row: pd.Series, model_ids: list[str], unknown_label: str) -> dict[str, Any]:
    votes: list[str] = []
    model_votes: list[dict[str, Any]] = []
    for model_id in model_ids:
        pred = str(row.get(f"{model_id}__pred_shared", unknown_label))
        conf = _safe_float(row.get(f"{model_id}__confidence", 0.0))
        model_votes.append({"model_id": model_id, "label": pred, "confidence_bin": _confidence_bin(conf)})
        if pred != unknown_label:
            votes.append(pred)
    counts = Counter(votes)
    if counts:
        top_label, top_count = counts.most_common(1)[0]
        agreement = top_count / len(votes)
    else:
        top_label, agreement = unknown_label, 0.0
    return {
        "model_votes": model_votes,
        "vote_counts": dict(sorted(counts.items())),
        "top_label": top_label,
        "agreement": float(agreement),
    }


def _method_predictions(row: pd.Series, method_names: list[str], unknown_label: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for method in method_names:
        col = f"{method}__pred_shared"
        if col in row.index:
            out[method] = str(row.get(col, unknown_label))
    return out


def _group_key(
    *,
    vote_summary: dict[str, Any],
    method_predictions: dict[str, str],
) -> str:
    payload = {
        "model_votes": vote_summary["model_votes"],
        "methods": method_predictions,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _build_evidence_groups(
    *,
    consensus_predictions: pd.DataFrame,
    model_ids: list[str],
    candidate_method_names: list[str],
    unknown_label: str,
    max_groups: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    groups: dict[str, dict[str, Any]] = {}
    cell_to_group: dict[str, str] = {}
    for _, row in consensus_predictions.iterrows():
        vote_summary = _cell_vote_summary(row, model_ids, unknown_label)
        method_predictions = _method_predictions(row, candidate_method_names, unknown_label)
        has_model_disagreement = vote_summary["agreement"] < 1.0
        method_labels = {label for label in method_predictions.values() if label != unknown_label}
        has_method_disagreement = len(method_labels) > 1
        if not has_model_disagreement and not has_method_disagreement:
            continue
        key = _group_key(vote_summary=vote_summary, method_predictions=method_predictions)
        cell_id = str(row.get("cell_id", ""))
        if key not in groups:
            group_id = f"g{len(groups):05d}"
            groups[key] = {
                "group_id": group_id,
                "cell_count": 0,
                "example_cell_ids": [],
                "model_votes": vote_summary["model_votes"],
                "vote_counts": vote_summary["vote_counts"],
                "top_vote_label": vote_summary["top_label"],
                "model_agreement": vote_summary["agreement"],
                "method_predictions": method_predictions,
            }
        group = groups[key]
        group["cell_count"] += 1
        if len(group["example_cell_ids"]) < 5 and cell_id:
            group["example_cell_ids"].append(cell_id)
        if cell_id:
            cell_to_group[cell_id] = group["group_id"]

    ordered = sorted(groups.values(), key=lambda item: (int(item["cell_count"]), item["group_id"]), reverse=True)
    selected = ordered[: int(max_groups)]
    selected_ids = {str(group["group_id"]) for group in selected}
    cell_to_group = {cell_id: group_id for cell_id, group_id in cell_to_group.items() if group_id in selected_ids}
    return selected, cell_to_group


def _chunks(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(chunk_size))
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _parse_group_decisions(
    parsed: dict[str, Any],
    *,
    allowed_labels: set[str],
) -> dict[str, dict[str, Any]]:
    raw = parsed.get("groups", [])
    if not isinstance(raw, list):
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id", "")).strip()
        label = str(item.get("selected_label", "")).strip()
        if not group_id or label not in allowed_labels:
            continue
        decisions[group_id] = {
            "group_id": group_id,
            "selected_label": label,
            "llm_confidence": _safe_float(item.get("confidence", 0.0)),
            "rationale": str(item.get("rationale", "")).strip()[:500],
        }
    return decisions


def adjudicate_low_consistency_cells(
    *,
    consensus_predictions: pd.DataFrame,
    model_ids: list[str],
    candidate_method_names: list[str],
    base_method: str,
    allowed_labels: list[str],
    unknown_label: str,
    output_dir: str | Path,
    llm_mode: str = "optional",
    llm_model: str | None = None,
    max_groups: int = 120,
    batch_size: int = 12,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if llm_mode == "off":
        return None, {"status": "skipped", "reason": "llm_cell_adjudication_off"}
    if f"{base_method}__pred_shared" not in consensus_predictions.columns:
        return None, {"status": "skipped", "reason": f"base_method_missing:{base_method}"}

    groups, cell_to_group = _build_evidence_groups(
        consensus_predictions=consensus_predictions,
        model_ids=model_ids,
        candidate_method_names=candidate_method_names,
        unknown_label=unknown_label,
        max_groups=max_groups,
    )
    if not groups:
        return None, {"status": "skipped", "reason": "no_disagreement_groups"}

    _load_env_file()
    effective_llm_model = llm_model or default_llm_model(DEFAULT_LLM_MODEL)
    trace_dir = ensure_dir(Path(output_dir) / "llm_cell_adjudication")
    observe = {
        "schema_version": "scmas.stage4.llm_cell_adjudication.v1",
        "method": LLM_ADJUDICATION_METHOD,
        "query_labels_available_to_llm": False,
        "metrics_with_query_truth_available_to_llm": False,
        "allowed_labels": allowed_labels,
        "unknown_label": unknown_label,
        "base_method": base_method,
        "model_ids": model_ids,
        "candidate_method_names": candidate_method_names,
        "max_groups": int(max_groups),
        "n_groups": len(groups),
        "n_cells_in_reviewed_groups": len(cell_to_group),
        "global_model_agreement_threshold": HIGH_CONSENSUS_MODEL_AGREEMENT,
        "evidence_groups": groups,
    }
    write_json(observe, trace_dir / "observe.json")
    with (trace_dir / "observe_groups.jsonl").open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False, sort_keys=True) + "\n")

    allowed = set(allowed_labels)
    decisions: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    system_prompt = (
        "You are the scMAS low-consistency cell-type adjudicator. Choose shared coarse cell labels using only "
        "model predictions, confidence bins, vote counts, and label-free consensus/reference outputs. "
        "Never request or infer hidden query truth labels. Return only JSON."
    )
    for batch_idx, batch in enumerate(_chunks(groups, batch_size), start=1):
        user_payload = {
            "instructions": [
                "For each evidence group, choose exactly one selected_label from allowed_labels.",
                "Use the model vote counts, model confidence bins, and candidate consensus method predictions as evidence.",
                "Prefer a specific biological label when evidence is coherent; use the unknown_label when evidence is too contradictory.",
                "Return one decision for every evidence group. If evidence is insufficient, set selected_label to the unknown_label.",
                "Do not use example_cell_ids as biological evidence; they are only audit ids.",
                "Do not return an error object. Return groups with group_id, selected_label, confidence, and a short rationale.",
            ],
            "response_contract": {
                "groups": [
                    {
                        "group_id": "string",
                        "selected_label": "one of allowed_labels",
                        "confidence": "0.0-1.0",
                        "rationale": "short label-free reason",
                    }
                ]
            },
            "allowed_labels": allowed_labels,
            "unknown_label": unknown_label,
            "base_method": base_method,
            "evidence_groups": batch,
        }
        user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
        (trace_dir / f"prompt_{batch_idx:03d}.md").write_text(
            "## System\n\n" + system_prompt + "\n\n## User\n\n```json\n" + user_prompt + "\n```\n",
            encoding="utf-8",
        )
        try:
            parsed, raw_text, meta = _call_openai_json(
                model=effective_llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"batch_{batch_idx}: {type(exc).__name__}: {exc}")
            if llm_mode == "required":
                raise
            continue
        (trace_dir / f"response_{batch_idx:03d}.txt").write_text(raw_text, encoding="utf-8")
        write_json({"parsed": parsed, "meta": meta}, trace_dir / f"response_{batch_idx:03d}.json")
        decisions.update(_parse_group_decisions(parsed, allowed_labels=allowed))

    base = consensus_predictions[f"{base_method}__pred_shared"].astype(str).to_numpy(dtype=object).copy()
    cell_ids = consensus_predictions["cell_id"].astype(str).tolist()
    reviewed_cells = 0
    changed_cells = 0
    decision_counts: dict[str, int] = defaultdict(int)
    for idx, cell_id in enumerate(cell_ids):
        group_id = cell_to_group.get(cell_id)
        if not group_id:
            continue
        decision = decisions.get(group_id)
        if not decision:
            continue
        reviewed_cells += 1
        label = str(decision["selected_label"])
        decision_counts[label] += 1
        if base[idx] != label:
            changed_cells += 1
        base[idx] = label

    review = {
        "status": "completed" if decisions else "failed_no_valid_decisions",
        "llm_mode": llm_mode,
        "llm_model": effective_llm_model,
        "base_method": base_method,
        "n_groups": len(groups),
        "n_decision_groups": len(decisions),
        "n_cells_in_reviewed_groups": len(cell_to_group),
        "n_cells_with_valid_llm_decision": reviewed_cells,
        "changed_fraction": float(changed_cells / max(1, len(base))),
        "decision_label_counts": dict(sorted(decision_counts.items())),
        "errors": errors,
    }
    write_json(review, trace_dir / "review.json")
    decisions_frame = pd.DataFrame(list(decisions.values()))
    decisions_frame.to_csv(trace_dir / "group_decisions.csv", index=False)
    if not decisions:
        if llm_mode == "required":
            raise RuntimeError("LLM low-consistency adjudication produced no valid decisions")
        return None, review
    return base, review
