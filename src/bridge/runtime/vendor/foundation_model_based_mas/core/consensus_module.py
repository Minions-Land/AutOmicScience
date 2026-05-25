from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import f1_score


ANNOTATION_LEVELS = ("class", "subclass", "supertype")
GROUND_TRUTH_LEVEL_ALIASES = {
    "class": "class",
    "cell_class": "class",
    "subclass": "subclass",
    "cell_subclass": "subclass",
    "supertype": "supertype",
    "cell_supertype": "supertype",
}
DEFAULT_MIN_SUPPORT_MARGIN = 0.10
DEFAULT_CONFIDENCE_POWER = 2.0
DEFAULT_REVIEW_MIN_SHARE = 0.55
DEFAULT_REVIEW_SINGLETON_MIN_SHARE = 0.65
DEFAULT_REVIEW_MIN_TOP_MODEL_CONFIDENCE = 0.50


def _normalize_level_name(raw_level: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(raw_level or "").strip().lower()).strip("_")
    return GROUND_TRUTH_LEVEL_ALIASES.get(normalized, "subclass")


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


class ConsensusModule:
    def __init__(self, *, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.capability_dir = self.project_root / "config" / "capability"
        self.outputs_dir = self.project_root / "outputs" / "consensus_module"
        self.logs_dir = self.project_root / "logs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("mas_consensus_module")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            file_handler = logging.FileHandler(self.logs_dir / "mas_consensus_module.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def _load_capability_index(self) -> dict[str, dict[str, Any]]:
        model_index: dict[str, dict[str, Any]] = {}
        for path in sorted(self.capability_dir.glob("**/*.yaml")):
            if not path.is_file():
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(payload, dict):
                continue
            model_id = str(payload.get("model_id", path.stem)).strip()
            if model_id:
                model_index[model_id] = payload
        return model_index

    def _resolve_model_weight(
        self,
        *,
        model_id: str,
        task_type: str,
        capability_index: dict[str, dict[str, Any]],
    ) -> dict[str, float | str]:
        capability = capability_index.get(model_id, {})
        task_scores = capability.get("task_scores", {}) if isinstance(capability, dict) else {}
        if not isinstance(task_scores, dict):
            task_scores = {}
        raw_weight = _safe_float(task_scores.get(task_type), 1.0)
        if raw_weight <= 0:
            raw_weight = 1.0
        return {
            "task_score": raw_weight,
            "source": f"capability.task_scores.{task_type}" if capability else "fallback.default_weight",
        }

    def _resolve_level_priors(
        self,
        *,
        model_id: str,
        task_type: str,
        capability_index: dict[str, dict[str, Any]],
        default_weight: float,
    ) -> dict[str, float]:
        capability = capability_index.get(model_id, {})
        annotation_profile = capability.get("annotation_profile", {}) if isinstance(capability, dict) else {}
        raw_level_priors = annotation_profile.get("level_priors", {}) if isinstance(annotation_profile, dict) else {}
        if not isinstance(raw_level_priors, dict):
            raw_level_priors = {}

        resolved: dict[str, float] = {}
        fallback = default_weight if default_weight > 0 else 1.0
        for level in ANNOTATION_LEVELS:
            prior = _safe_float(raw_level_priors.get(level), fallback)
            if prior <= 0:
                prior = fallback
            resolved[level] = float(prior)
        return resolved

    def _align_prediction_records(
        self,
        bundles: list[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        base_model_id = str(bundles[0]["model_id"])
        base_cell_ids = np.asarray(bundles[0]["npz"]["cell_ids"])
        aligned_indices: dict[str, np.ndarray] = {base_model_id: np.arange(base_cell_ids.shape[0], dtype=np.int64)}

        fast_path = True
        for bundle in bundles[1:]:
            cell_ids = np.asarray(bundle["npz"]["cell_ids"])
            if cell_ids.shape != base_cell_ids.shape or not np.array_equal(cell_ids, base_cell_ids):
                fast_path = False
                break

        if fast_path:
            for bundle in bundles[1:]:
                aligned_indices[str(bundle["model_id"])] = np.arange(base_cell_ids.shape[0], dtype=np.int64)
            return base_cell_ids, aligned_indices

        base_keys = [str(item) for item in base_cell_ids.tolist()]
        common_ids = set(base_keys)
        for bundle in bundles[1:]:
            common_ids &= {str(item) for item in np.asarray(bundle["npz"]["cell_ids"]).tolist()}

        kept_base_indices = np.array(
            [idx for idx, cell_id in enumerate(base_keys) if cell_id in common_ids],
            dtype=np.int64,
        )
        aligned_cell_ids = base_cell_ids[kept_base_indices]
        aligned_indices[base_model_id] = kept_base_indices

        aligned_key_order = [str(item) for item in aligned_cell_ids.tolist()]
        for bundle in bundles[1:]:
            cell_ids = np.asarray(bundle["npz"]["cell_ids"])
            lookup = {str(cell_id): idx for idx, cell_id in enumerate(cell_ids.tolist()) if str(cell_id) in common_ids}
            aligned_indices[str(bundle["model_id"])] = np.array([lookup[key] for key in aligned_key_order], dtype=np.int64)
        return aligned_cell_ids, aligned_indices

    def _get_npz_subset(
        self,
        npz: Any,
        key: str,
        indices: np.ndarray,
        *,
        default: Any = None,
    ) -> np.ndarray | Any:
        files = set(getattr(npz, "files", []))
        if key not in files:
            return default
        return np.asarray(npz[key])[indices]

    def _compute_level_payloads(
        self,
        *,
        bundles: list[dict[str, Any]],
        aligned_indices: dict[str, np.ndarray],
        level: str,
    ) -> dict[str, dict[str, np.ndarray | float]]:
        payloads: dict[str, dict[str, np.ndarray | float]] = {}
        for bundle in bundles:
            model_id = str(bundle["model_id"])
            npz = bundle["npz"]
            idx = aligned_indices[model_id]
            raw_conf = np.asarray(npz[f"{level}_confidence"])[idx].astype(np.float32)
            pred_ids = np.asarray(npz[f"{level}_pred_ids"])[idx].astype(np.int64)
            pred_labels = self._get_npz_subset(
                npz,
                f"{level}_pred_labels",
                idx,
                default=np.asarray([str(pred_id) for pred_id in pred_ids], dtype=object),
            )
            level_prior = float(bundle["level_priors"][level])
            confidence_signal = np.power(np.clip(raw_conf.astype(float), 1e-6, 1.0), DEFAULT_CONFIDENCE_POWER).astype(
                np.float32
            )
            support_score = (level_prior * confidence_signal.astype(float)).astype(np.float32)
            payloads[model_id] = {
                "pred_ids": pred_ids,
                "pred_labels": pred_labels,
                "raw_confidence": raw_conf,
                "confidence_signal": confidence_signal,
                "support_score": support_score,
                "level_prior": level_prior,
            }
        return payloads

    def _write_review_csv(
        self,
        *,
        csv_path: Path,
        rows: list[dict[str, Any]],
        model_ids: list[str],
        include_truth: bool,
    ) -> None:
        fieldnames = [
            "cell_id",
            "test_row_index",
            "final_pred_label",
            "selected_model_id",
            "consensus_status",
            "consensus_confidence",
            "support_margin",
            "best_support_share",
            "top_vote_count",
            "top_vote_fraction",
            "unique_pred_label_count",
            "max_supporting_model_confidence",
            "label_support_json",
        ]
        if include_truth:
            fieldnames.insert(2, "true_label")

        for model_id in model_ids:
            fieldnames.extend(
                [
                    f"{model_id}_pred_label",
                    f"{model_id}_raw_confidence",
                    f"{model_id}_confidence_signal",
                    f"{model_id}_support_score",
                    f"{model_id}_level_prior",
                ]
            )

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _arbitrate_annotation_level(
        self,
        *,
        bundles: list[dict[str, Any]],
        aligned_cell_ids: np.ndarray,
        aligned_indices: dict[str, np.ndarray],
        level: str,
        min_support_margin: float,
        level_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_ids = [str(bundle["model_id"]) for bundle in bundles]
        level_payloads = self._compute_level_payloads(bundles=bundles, aligned_indices=aligned_indices, level=level)
        base_model_id = model_ids[0]
        base_npz = bundles[0]["npz"]
        base_idx = aligned_indices[base_model_id]
        n_cells = aligned_cell_ids.shape[0]

        test_row_indices = self._get_npz_subset(
            base_npz,
            "test_row_indices",
            base_idx,
            default=np.arange(n_cells, dtype=np.int64),
        )
        true_ids = self._get_npz_subset(base_npz, f"{level}_true_ids", base_idx, default=None)
        true_labels = self._get_npz_subset(base_npz, f"{level}_true_labels", base_idx, default=None)
        has_ground_truth = true_ids is not None

        final_pred_ids = np.empty(n_cells, dtype=np.int64)
        final_pred_labels = np.empty(n_cells, dtype=object)
        consensus_status = np.empty(n_cells, dtype=object)
        consensus_confidence = np.zeros(n_cells, dtype=np.float32)
        support_margin = np.zeros(n_cells, dtype=np.float32)
        best_support_share = np.ones(n_cells, dtype=np.float32)
        unique_pred_label_count = np.zeros(n_cells, dtype=np.int16)
        top_vote_count = np.zeros(n_cells, dtype=np.int16)
        top_vote_fraction = np.zeros(n_cells, dtype=np.float32)
        max_supporting_model_confidence = np.zeros(n_cells, dtype=np.float32)
        correct = np.zeros(n_cells, dtype=bool) if has_ground_truth else None

        review_rows: list[dict[str, Any]] = []
        unanimous_count = 0
        resolved_count = 0
        review_count = 0

        for idx in range(n_cells):
            label_support: dict[str, float] = {}
            label_vote_counts: dict[str, int] = {}
            label_id_lookup: dict[str, int] = {}
            per_model_row: dict[str, Any] = {}
            ranked_models: list[tuple[str, str, float, float]] = []
            total_support = 0.0

            for model_id in model_ids:
                payload = level_payloads[model_id]
                pred_label = str(np.asarray(payload["pred_labels"])[idx])
                pred_id = int(np.asarray(payload["pred_ids"])[idx])
                raw_conf = _safe_float(np.asarray(payload["raw_confidence"])[idx], 0.0)
                confidence_signal = _safe_float(np.asarray(payload["confidence_signal"])[idx], 0.0)
                support = _safe_float(np.asarray(payload["support_score"])[idx], 0.0)
                level_prior = _safe_float(payload["level_prior"], 0.0)
                total_support += support
                label_support[pred_label] = label_support.get(pred_label, 0.0) + support
                label_vote_counts[pred_label] = label_vote_counts.get(pred_label, 0) + 1
                label_id_lookup.setdefault(pred_label, pred_id)
                ranked_models.append((model_id, pred_label, support, raw_conf))
                per_model_row[f"{model_id}_pred_label"] = pred_label
                per_model_row[f"{model_id}_raw_confidence"] = f"{raw_conf:.6f}"
                per_model_row[f"{model_id}_confidence_signal"] = f"{confidence_signal:.6f}"
                per_model_row[f"{model_id}_support_score"] = f"{support:.6f}"
                per_model_row[f"{model_id}_level_prior"] = f"{level_prior:.6f}"

            ranked_labels = sorted(
                label_support.items(),
                key=lambda item: (-item[1], -label_vote_counts.get(item[0], 0), item[0]),
            )
            top_label, top_label_support = ranked_labels[0]
            runnerup_label_support = ranked_labels[1][1] if len(ranked_labels) > 1 else 0.0
            unique_pred_label_count[idx] = int(len(ranked_labels))
            consensus_confidence[idx] = np.float32(top_label_support / max(total_support, 1e-12))
            support_margin[idx] = np.float32((top_label_support - runnerup_label_support) / max(total_support, 1e-12))
            best_support_share[idx] = np.float32(top_label_support / max(total_support, 1e-12))
            top_vote_count[idx] = int(label_vote_counts.get(top_label, 0))
            top_vote_fraction[idx] = np.float32(top_vote_count[idx] / max(len(model_ids), 1))

            top_support_models = [item for item in ranked_models if item[1] == top_label]
            selected_model_id, _, _, top_model_conf = max(top_support_models, key=lambda item: (item[2], item[3], item[0]))
            max_supporting_model_confidence[idx] = np.float32(top_model_conf)

            if len(ranked_labels) == 1:
                consensus_status[idx] = "unanimous"
                unanimous_count += 1
            else:
                is_review = (
                    consensus_confidence[idx] < DEFAULT_REVIEW_MIN_SHARE
                    or support_margin[idx] < min_support_margin
                    or max_supporting_model_confidence[idx] < DEFAULT_REVIEW_MIN_TOP_MODEL_CONFIDENCE
                    or (
                        top_vote_count[idx] == 1
                        and consensus_confidence[idx] < DEFAULT_REVIEW_SINGLETON_MIN_SHARE
                    )
                )
                if is_review:
                    consensus_status[idx] = "needs_review"
                    review_count += 1
                else:
                    consensus_status[idx] = "support_resolved"
                    resolved_count += 1

            final_pred_labels[idx] = top_label
            final_pred_ids[idx] = int(label_id_lookup[top_label])
            if consensus_status[idx] == "unanimous":
                resolved_count += 1

            if has_ground_truth and correct is not None:
                correct[idx] = final_pred_ids[idx] == int(true_ids[idx])

            if consensus_status[idx] == "needs_review":
                row = {
                    "cell_id": str(aligned_cell_ids[idx]),
                    "test_row_index": int(test_row_indices[idx]),
                    "final_pred_label": str(final_pred_labels[idx]),
                    "selected_model_id": str(selected_model_id),
                    "consensus_status": str(consensus_status[idx]),
                    "consensus_confidence": f"{float(consensus_confidence[idx]):.6f}",
                    "support_margin": f"{float(support_margin[idx]):.6f}",
                    "best_support_share": f"{float(best_support_share[idx]):.6f}",
                    "top_vote_count": int(top_vote_count[idx]),
                    "top_vote_fraction": f"{float(top_vote_fraction[idx]):.6f}",
                    "unique_pred_label_count": int(unique_pred_label_count[idx]),
                    "max_supporting_model_confidence": f"{float(max_supporting_model_confidence[idx]):.6f}",
                    "label_support_json": json.dumps(
                        {label: round(weight, 6) for label, weight in ranked_labels},
                        ensure_ascii=False,
                    ),
                    **per_model_row,
                }
                if has_ground_truth and true_labels is not None:
                    row["true_label"] = str(true_labels[idx])
                review_rows.append(row)

        consensus_npz_path = level_dir / f"{level}_consensus_records.npz"
        controversial_csv_path = level_dir / f"{level}_controversial_cells.csv"

        npz_payload: dict[str, Any] = {
            "cell_ids": aligned_cell_ids,
            "test_row_indices": test_row_indices,
            "final_pred_ids": final_pred_ids,
            "final_pred_labels": final_pred_labels,
            "consensus_status": consensus_status,
            "consensus_confidence": consensus_confidence,
            "support_margin": support_margin,
            "best_support_share": best_support_share,
            "unique_pred_label_count": unique_pred_label_count,
            "top_vote_count": top_vote_count,
            "top_vote_fraction": top_vote_fraction,
            "max_supporting_model_confidence": max_supporting_model_confidence,
            "has_ground_truth": np.asarray([has_ground_truth], dtype=bool),
        }
        if has_ground_truth and true_ids is not None and true_labels is not None and correct is not None:
            npz_payload["true_ids"] = true_ids
            npz_payload["true_labels"] = true_labels
            npz_payload["correct"] = correct
        np.savez_compressed(consensus_npz_path, **npz_payload)

        self._write_review_csv(
            csv_path=controversial_csv_path,
            rows=review_rows,
            model_ids=model_ids,
            include_truth=bool(has_ground_truth and true_labels is not None),
        )

        resolved_mask = consensus_status != "needs_review"
        summary: dict[str, Any] = {
            "n_cells": int(n_cells),
            "unanimous_count": int(unanimous_count),
            "support_resolved_count": int(np.sum(consensus_status == "support_resolved")),
            "review_count": int(review_count),
            "controversial_count": int(review_count),
            "resolved_count": int(resolved_count),
            "resolved_fraction": float(resolved_mask.mean()),
            "review_fraction": float(np.mean(consensus_status == "needs_review")),
            "controversial_fraction": float(np.mean(consensus_status == "needs_review")),
            "consensus_npz_path": str(consensus_npz_path),
            "controversial_csv_path": str(controversial_csv_path),
            "review_csv_path": str(controversial_csv_path),
            "has_ground_truth": bool(has_ground_truth),
        }
        if has_ground_truth and true_ids is not None and correct is not None:
            summary.update(
                {
                    "overall_accuracy": float(np.mean(correct.astype(np.float32))),
                    "weighted_all_accuracy": float(np.mean(correct.astype(np.float32))),
                    "overall_macro_f1": float(f1_score(true_ids, final_pred_ids, average="macro")),
                    "resolved_accuracy": float(np.mean(correct[resolved_mask].astype(np.float32)))
                    if resolved_mask.any()
                    else None,
                    "review_accuracy_if_forced": float(np.mean(correct[~resolved_mask].astype(np.float32)))
                    if (~resolved_mask).any()
                    else None,
                }
            )
        else:
            summary.update(
                {
                    "overall_accuracy": None,
                    "weighted_all_accuracy": None,
                    "overall_macro_f1": None,
                    "resolved_accuracy": None,
                    "review_accuracy_if_forced": None,
                }
            )

        debug = {
            "review_thresholds": {
                "min_support_margin": float(min_support_margin),
                "min_consensus_share": float(DEFAULT_REVIEW_MIN_SHARE),
                "singleton_min_consensus_share": float(DEFAULT_REVIEW_SINGLETON_MIN_SHARE),
                "min_top_model_confidence": float(DEFAULT_REVIEW_MIN_TOP_MODEL_CONFIDENCE),
            },
            "support_formula": f"level_prior * (raw_confidence ** {DEFAULT_CONFIDENCE_POWER:.1f})",
            "decision_formula": "aggregate support by predicted label across models; choose top-support label; send low-share or low-margin cells to needs_review",
            "label_aggregation": "sum support scores of all models that predict the same label",
            "uses_current_run_ground_truth_for_decision": False,
        }
        return summary, debug

    def _write_summary_markdown(
        self,
        *,
        summary_path: Path,
        result: dict[str, Any],
    ) -> None:
        lines = [
            "# Consensus Arbitration Summary",
            "",
            f"- Task type: `{result['task_type']}`",
            f"- Focus level: `{result['focus_level']}`",
            f"- Completed models used: `{', '.join(result['model_order'])}`",
            f"- Common cells aligned: `{result['n_common_cells']}`",
            f"- Min support margin: `{result['min_support_margin']}`",
            f"- Ground-truth used for decision: `False`",
            "",
            "## Model Weights",
            "",
            "| Model | Task Score | Normalized Weight | Source |",
            "|---|---:|---:|---|",
        ]
        for model_id, payload in result["model_weights"].items():
            lines.append(
                f"| {model_id} | {payload['task_score']:.6f} | {payload['normalized_weight']:.6f} | {payload['source']} |"
            )

        lines.extend(
            [
                "",
                "## Level Summary",
                "",
                "| Level | Resolved | Review | Accuracy | Macro-F1 | Resolved Accuracy | Review Accuracy If Forced |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for level, payload in result["level_summaries"].items():
            accuracy_text = f"{payload['overall_accuracy']:.4f}" if payload["overall_accuracy"] is not None else "NA"
            macro_f1_text = f"{payload['overall_macro_f1']:.4f}" if payload["overall_macro_f1"] is not None else "NA"
            resolved_accuracy_text = f"{payload['resolved_accuracy']:.4f}" if payload["resolved_accuracy"] is not None else "NA"
            review_accuracy_text = (
                f"{payload['review_accuracy_if_forced']:.4f}"
                if payload["review_accuracy_if_forced"] is not None
                else "NA"
            )
            lines.append(
                f"| {level} | {payload['resolved_count']} | {payload['review_count']} | "
                f"{accuracy_text} | {macro_f1_text} | {resolved_accuracy_text} | {review_accuracy_text} |"
            )

        lines.extend(
            [
                "",
                "## Formula",
                "",
                f"- Base support: `{result['formula_debug']['support_formula']}`",
                f"- Decision rule: `{result['formula_debug']['decision_formula']}`",
                "- Current-run labels are used only for post-hoc evaluation when available, never for arbitration.",
            ]
        )
        summary_path.write_text("\n".join(lines), encoding="utf-8")

    def run(
        self,
        *,
        task_type: str,
        execution_result: dict[str, Any],
        ground_truth_label_key: str = "",
        min_support_margin: float = DEFAULT_MIN_SUPPORT_MARGIN,
    ) -> dict[str, Any]:
        if task_type != "annotation":
            return {
                "consensus_status": "skipped",
                "reason": f"task_type '{task_type}' is not supported by ConsensusModule yet.",
                "task_type": task_type,
                "level_summaries": {},
            }

        path_results = execution_result.get("path_results", []) if isinstance(execution_result, dict) else []
        completed_paths = [
            path
            for path in path_results
            if path.get("status") == "completed" and path.get("prediction_artifact_path")
        ]
        if not completed_paths:
            return {
                "consensus_status": "skipped",
                "reason": "No completed path results with prediction artifacts were available.",
                "task_type": task_type,
                "level_summaries": {},
            }

        capability_index = self._load_capability_index()
        run_dir = self.outputs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        bundles: list[dict[str, Any]] = []
        model_weights: dict[str, dict[str, float | str]] = {}
        for path_result in completed_paths:
            model_id = str(path_result.get("selected_model_id", "")).strip()
            if not model_id:
                continue
            prediction_path = Path(str(path_result["prediction_artifact_path"])).resolve()
            if not prediction_path.exists():
                continue
            weight_payload = self._resolve_model_weight(
                model_id=model_id,
                task_type=task_type,
                capability_index=capability_index,
            )
            bundles.append(
                {
                    "path_id": str(path_result.get("path_id", "")),
                    "model_id": model_id,
                    "prediction_artifact_path": str(prediction_path),
                    "model_weight": float(weight_payload["task_score"]),
                    "level_priors": self._resolve_level_priors(
                        model_id=model_id,
                        task_type=task_type,
                        capability_index=capability_index,
                        default_weight=float(weight_payload["task_score"]),
                    ),
                    "npz": np.load(prediction_path, allow_pickle=True),
                }
            )
            model_weights[model_id] = weight_payload

        if not bundles:
            return {
                "consensus_status": "skipped",
                "reason": "Completed path results existed but their prediction artifacts were unreadable.",
                "task_type": task_type,
                "level_summaries": {},
            }

        aligned_cell_ids, aligned_indices = self._align_prediction_records(bundles)
        model_ids = [str(bundle["model_id"]) for bundle in bundles]
        total_global_weight = float(sum(float(bundle["model_weight"]) for bundle in bundles))
        normalized_model_weights = {
            model_id: {
                "task_score": float(model_weights[model_id]["task_score"]),
                "normalized_weight": float(model_weights[model_id]["task_score"]) / max(total_global_weight, 1e-12),
                "source": str(model_weights[model_id]["source"]),
            }
            for model_id in model_ids
        }

        focus_level = _normalize_level_name(ground_truth_label_key)
        level_summaries: dict[str, Any] = {}
        level_artifacts: dict[str, dict[str, str]] = {}
        formula_debug: dict[str, Any] = {"support_formula": "", "decision_formula": "", "levels": {}}
        for level in ANNOTATION_LEVELS:
            level_dir = run_dir / level
            level_dir.mkdir(parents=True, exist_ok=True)
            level_summary, level_debug = self._arbitrate_annotation_level(
                bundles=bundles,
                aligned_cell_ids=aligned_cell_ids,
                aligned_indices=aligned_indices,
                level=level,
                min_support_margin=min_support_margin,
                level_dir=level_dir,
            )
            level_summaries[level] = level_summary
            level_artifacts[level] = {
                "consensus_npz_path": level_summary["consensus_npz_path"],
                "controversial_csv_path": level_summary["controversial_csv_path"],
                "review_csv_path": level_summary["review_csv_path"],
            }
            formula_debug["support_formula"] = level_debug["support_formula"]
            formula_debug["decision_formula"] = level_debug["decision_formula"]
            formula_debug["label_aggregation"] = level_debug["label_aggregation"]
            formula_debug["uses_current_run_ground_truth_for_decision"] = level_debug[
                "uses_current_run_ground_truth_for_decision"
            ]
            formula_debug["levels"][level] = {
                "review_thresholds": level_debug["review_thresholds"],
            }

        result = {
            "consensus_status": "completed",
            "task_type": task_type,
            "focus_level": focus_level,
            "min_support_margin": float(min_support_margin),
            "run_dir": str(run_dir),
            "n_models": len(model_ids),
            "model_order": model_ids,
            "n_common_cells": int(aligned_cell_ids.shape[0]),
            "model_weights": normalized_model_weights,
            "level_summaries": level_summaries,
            "level_artifacts": level_artifacts,
            "formula_debug": formula_debug,
        }

        summary_json_path = run_dir / "consensus_summary.json"
        summary_md_path = run_dir / "consensus_summary.md"
        result["summary_json_path"] = str(summary_json_path)
        result["summary_markdown_path"] = str(summary_md_path)
        summary_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_summary_markdown(summary_path=summary_md_path, result=result)

        for bundle in bundles:
            bundle["npz"].close()

        self.logger.info(
            "[consensus.run.summary] %s",
            json.dumps(
                {
                    "consensus_status": result["consensus_status"],
                    "run_dir": result["run_dir"],
                    "n_models": result["n_models"],
                    "n_common_cells": result["n_common_cells"],
                    "focus_level": result["focus_level"],
                    "focus_level_summary": result["level_summaries"].get(focus_level, {}),
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        return result
