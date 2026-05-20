from __future__ import annotations

import json
import math
import re
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from novaeve_bio import paths
from novaeve_bio.eval.label_transfer import (
    EmbeddingAdapter,
    MatrixBundle,
    _align_reference_and_query,
    _load_reference_from_standard_bundle,
)
from novaeve_bio.io import ensure_dir, read_json, read_yaml, write_json
from novaeve_bio.stage2.selector import load_query_bundle
from novaeve_bio.stage4.llm_adjudicator import LLM_ADJUDICATION_METHOD, adjudicate_low_consistency_cells
from novaeve_bio.stage4.policy import HIGH_CONSENSUS_MODEL_AGREEMENT, label_free_method_diagnostics, select_policy_method
from novaeve_bio.stage4.reference_heads import train_reference_logistic_distribution


DEFAULT_STAGE4_ROOT = paths.RUNS_DIR / "stage4_consensus"
DEFAULT_PREPARED_SOURCE_ROOT = paths.DATA_DIR / "prepared_sources_full_100k_g500"
UNKNOWN_LABEL = "__unknown__"

SEAAD_TO_COARSE = {
    "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligodendrocyte",
    "OPC": "OPC",
    "Microglia-PVM": "Microglia",
    "Endothelial": "Endothelial",
    "VLMC": "Vascular",
    "L2/3 IT": "Neuron",
    "L4 IT": "Neuron",
    "L5 IT": "Neuron",
    "L5 ET": "Neuron",
    "L5/6 NP": "Neuron",
    "L6 CT": "Neuron",
    "L6 IT": "Neuron",
    "L6 IT Car3": "Neuron",
    "L6b": "Neuron",
    "Lamp5": "Neuron",
    "Lamp5 Lhx6": "Neuron",
    "Pvalb": "Neuron",
    "Sst": "Neuron",
    "Sst Chodl": "Neuron",
    "Vip": "Neuron",
    "Sncg": "Neuron",
    "Pax6": "Neuron",
    "Chandelier": "Neuron",
}
SHARED_COARSE_LABELS = sorted(set(SEAAD_TO_COARSE.values()))
CONSENSUS_LABELS = [*SHARED_COARSE_LABELS, UNKNOWN_LABEL]
CONSENSUS_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CONSENSUS_LABELS)}

TASK_PRIORITY = ["subclass", "coarse_label", "native_label", "supertype", "class"]


@dataclass
class FusionFunctionContext:
    """Runtime context passed to every stage-4 fusion function call."""

    frame: pd.DataFrame
    model_ids: list[str]
    geometry: dict[str, dict[str, np.ndarray]]
    seed: int
    primary_model: str
    method_preds: dict[str, np.ndarray] = field(default_factory=dict)
    method_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionFunction:
    """Auditable whitelist entry for one fusion method."""

    name: str
    description: str
    method_kind: str
    required_inputs: tuple[str, ...]
    default_parameters: dict[str, Any]
    call: Callable[[FusionFunctionContext], tuple[np.ndarray | None, dict[str, Any]]]


def _method_available(ctx: FusionFunctionContext, method: str) -> bool:
    return method in ctx.method_preds


def _single_model_fusion_function(model_id: str) -> FusionFunction:
    def _call(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
        pred = ctx.frame[f"{model_id}__pred_shared"].astype(str).to_numpy(dtype=object)
        return pred, {"model_id": model_id}

    return FusionFunction(
        name=f"single__{model_id}",
        description=f"Use only the stage-3 prediction from {model_id}.",
        method_kind="single_model",
        required_inputs=(f"{model_id}__pred_shared",),
        default_parameters={},
        call=_call,
    )


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    if not str(path):
        return pd.DataFrame()
    path = Path(path)
    if not path.exists() or path.is_dir() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    data = read_yaml(path)
    return data if isinstance(data, dict) else {}


def _clean_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text in {"", "nan", "None", "unknown", "Unknown", UNKNOWN_LABEL}:
        return ""
    return text


def load_seaad_label_maps(path: str | Path = paths.SEAAD_LABEL_MAPS_JSON) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    return read_json(path)


def seaad_id_to_label(task: str, value: Any, label_maps: dict[str, Any]) -> str:
    if value is None or not label_maps:
        return ""
    try:
        idx = int(value)
    except Exception:
        return _clean_label(value)
    labels = label_maps.get(f"{task}_labels", [])
    if 0 <= idx < len(labels):
        return str(labels[idx])
    return str(idx)


def seaad_label_to_shared_coarse(task: str, label: Any, label_maps: dict[str, Any]) -> str:
    text = _clean_label(label)
    if not text:
        return UNKNOWN_LABEL
    if task == "class":
        if "Neuronal" in text:
            return "Neuron"
        return UNKNOWN_LABEL
    if task == "supertype":
        subclass = label_maps.get("supertype_to_subclass", {}).get(text, text)
        return SEAAD_TO_COARSE.get(subclass, label_to_shared_coarse(subclass, label_maps=label_maps))
    if task == "subclass":
        return SEAAD_TO_COARSE.get(text, label_to_shared_coarse(text, label_maps=label_maps))
    return label_to_shared_coarse(text, label_maps=label_maps)


def label_to_shared_coarse(label: Any, *, label_maps: dict[str, Any] | None = None) -> str:
    text = _clean_label(label)
    if not text:
        return UNKNOWN_LABEL

    label_maps = label_maps or {}
    if text in SHARED_COARSE_LABELS:
        return text
    if text in SEAAD_TO_COARSE:
        return SEAAD_TO_COARSE[text]
    if text in label_maps.get("supertype_to_subclass", {}):
        return SEAAD_TO_COARSE.get(label_maps["supertype_to_subclass"][text], UNKNOWN_LABEL)
    if text in label_maps.get("supertype_to_class", {}):
        class_label = label_maps["supertype_to_class"][text]
        return "Neuron" if "Neuronal" in str(class_label) else UNKNOWN_LABEL

    normalized = text.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    normalized = " ".join(normalized.split())
    if normalized in {"exn", "inn", "msn", "excitatory neurons", "inhibitory neurons"}:
        return "Neuron"
    if normalized in {"ol", "ol 0", "ol 1"}:
        return "Oligodendrocyte"
    if normalized in {"micro", "macro"}:
        return "Microglia"
    if normalized in {"peri"}:
        return "Vascular"
    compact = f" {normalized} "
    if any(token in compact for token in [" astro", "astrocyte"]):
        return "Astrocyte"
    if any(token in compact for token in [" opc", " cop", "nfol", "mfol", "immature ol", "immature olg", "imolg"]):
        return "OPC"
    if "precursor" in compact and "oligodendrocyte" in compact:
        return "OPC"
    if any(token in compact for token in [" olig", "oligo", "olg", " mol", "mature oligodendrocyte"]):
        return "Oligodendrocyte"
    if any(token in compact for token in ["microglia", "migl", "myeloid", "macrophage", "monocyte", " pvm", "mp migl"]):
        return "Microglia"
    if any(token in compact for token in ["endothelial", " endo", "vec"]):
        return "Endothelial"
    if any(token in compact for token in ["vascular", "vlmc", "leptomeningeal", "pericyte", "smooth muscle"]):
        return "Vascular"
    if any(
        token in compact
        for token in [
            " neuron",
            "neurons",
            "gaba",
            "glut",
            "interneuron",
            "chandelier",
            "pvalb",
            "sst",
            "vip",
            "lamp5",
            "sncg",
            "pax6",
            " l2",
            " l3",
            " l4",
            " l5",
            " l6",
        ]
    ):
        return "Neuron"
    return UNKNOWN_LABEL


def _preferred_task(df: pd.DataFrame) -> str:
    present = {str(x) for x in df.get("task", pd.Series(dtype=str)).dropna().astype(str).unique()}
    for task in TASK_PRIORITY:
        if task in present:
            return task
    if present:
        return sorted(present)[0]
    return ""


def _load_model_weight(spec: dict[str, Any], metrics: pd.DataFrame, model_id: str) -> float:
    capability_path = spec.get("capability_yaml") or spec.get("model_contract", {}).get("capability_yaml")
    if capability_path and Path(capability_path).exists():
        payload = read_yaml(capability_path) or {}
        source_id = str(spec.get("source_id") or "")
        source_scores = payload.get("stage1_evaluation", {}).get("source_dataset_scores", [])
        if source_id and isinstance(source_scores, list):
            for row in source_scores:
                if str(row.get("source_group", "")) != source_id:
                    continue
                try:
                    score = float(row.get("composite_score") or 0.0)
                    if score > 0:
                        return score
                except Exception:
                    pass
    return 1.0


def _prepared_h5ad_from_spec(dataset_id: str, spec: dict[str, Any]) -> Path | None:
    candidate_dirs: list[Any] = []
    runtime = spec.get("runtime_payload", {}) if isinstance(spec.get("runtime_payload", {}), dict) else {}
    input_artifacts = spec.get("input_artifacts", {}) if isinstance(spec.get("input_artifacts", {}), dict) else {}
    candidate_dirs.extend([runtime.get("prepared_input_dir"), input_artifacts.get("prepared_input_dir")])
    for value in candidate_dirs:
        if not value:
            continue
        root = Path(value)
        preferred = root / f"{dataset_id}.h5ad"
        if preferred.exists():
            return preferred
        matches = sorted(root.glob("*.h5ad"))
        if matches:
            return matches[0]
    return None


def _position_cell_indices(cell_ids: pd.Series) -> np.ndarray | None:
    indices: list[int] = []
    for value in cell_ids.astype(str):
        match = re.fullmatch(r"cell_(\d+)", value)
        if match is None:
            return None
        indices.append(int(match.group(1)))
    return np.asarray(indices, dtype=np.int64)


def _prepared_source_cell_ids(h5ad_path: Path) -> list[str]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    if "is_scmas_dummy" in obs.columns:
        dummy = obs["is_scmas_dummy"].astype(str).str.lower().isin({"true", "1", "yes"})
        obs = obs.loc[~dummy].copy()
    if "source_cell_id" in obs.columns:
        return obs["source_cell_id"].astype(str).tolist()
    return obs.index.astype(str).tolist()


def _remap_position_cell_ids_from_prepared_input(dataset_id: str, pred: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    positions = _position_cell_indices(pred["cell_id"])
    if positions is None or len(positions) == 0:
        return pred
    h5ad_path = _prepared_h5ad_from_spec(dataset_id, spec)
    if h5ad_path is None:
        return pred
    try:
        source_cell_ids = _prepared_source_cell_ids(h5ad_path)
    except Exception:
        return pred
    if len(source_cell_ids) == 0 or int(positions.max()) >= len(source_cell_ids) or int(positions.min()) < 0:
        return pred
    pred = pred.copy()
    pred["cell_id"] = [source_cell_ids[int(idx)] for idx in positions]
    pred["cell_id_remap_source"] = str(h5ad_path)
    return pred


def normalize_model_predictions(
    *,
    dataset_id: str,
    model_id: str,
    prediction_path: str | Path,
    spec: dict[str, Any],
    metrics: pd.DataFrame,
    label_maps: dict[str, Any],
) -> pd.DataFrame:
    pred = _safe_read_csv(prediction_path)
    if pred.empty or "task" not in pred.columns:
        return pd.DataFrame()
    task = _preferred_task(pred)
    pred = pred[pred["task"].astype(str) == task].copy()
    if pred.empty:
        return pd.DataFrame()

    if "cell_id" not in pred.columns:
        pred["cell_id"] = [f"cell_{idx}" for idx in range(len(pred))]
    pred["cell_id"] = pred["cell_id"].astype(str)
    pred = _remap_position_cell_ids_from_prepared_input(dataset_id, pred, spec)
    pred = pred.drop_duplicates(subset=["cell_id"], keep="first").copy()

    if {"true_label", "pred_label"}.issubset(pred.columns):
        pred["true_raw"] = pred["true_label"].map(_clean_label)
        pred["pred_raw"] = pred["pred_label"].map(_clean_label)
        pred["true_shared"] = pred["true_raw"].map(lambda x: seaad_label_to_shared_coarse(task, x, label_maps))
        pred["pred_shared"] = pred["pred_raw"].map(lambda x: seaad_label_to_shared_coarse(task, x, label_maps))
    elif {"true_id", "pred_id"}.issubset(pred.columns):
        pred["true_raw"] = pred["true_id"].map(lambda x: seaad_id_to_label(task, x, label_maps))
        pred["pred_raw"] = pred["pred_id"].map(lambda x: seaad_id_to_label(task, x, label_maps))
        pred["true_shared"] = pred["true_raw"].map(lambda x: seaad_label_to_shared_coarse(task, x, label_maps))
        pred["pred_shared"] = pred["pred_raw"].map(lambda x: seaad_label_to_shared_coarse(task, x, label_maps))
    else:
        return pd.DataFrame()

    if "confidence" not in pred.columns:
        pred["confidence"] = 1.0
    pred["confidence"] = pd.to_numeric(pred["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    model_weight = _load_model_weight(spec, metrics, model_id)
    out = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "source_id": pred["source_id"].astype(str) if "source_id" in pred.columns else spec.get("source_id", ""),
            "task": task,
            "cell_id": pred["cell_id"].astype(str),
            "sample_id": pred["sample_id"].astype(str) if "sample_id" in pred.columns else "",
            "true_raw": pred["true_raw"].astype(str),
            "pred_raw": pred["pred_raw"].astype(str),
            "true_shared": pred["true_shared"].astype(str),
            "pred_shared": pred["pred_shared"].astype(str),
            "confidence": pred["confidence"].astype(float),
            "model_weight": float(model_weight),
            "support_base": np.square(pred["confidence"].astype(float).clip(1e-6, 1.0)),
            "prediction_path": str(prediction_path),
            "adapter_spec": spec.get("_spec_path", ""),
        }
    )
    return out


def _unique_preserve_order(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _stage3_model_scope(stage3_summary: dict[str, Any], *, model_scope: str = "selected") -> list[str]:
    completed = _unique_preserve_order(list(stage3_summary.get("completed_models", []) or []))
    if model_scope == "completed":
        return completed
    if model_scope != "selected":
        raise ValueError("model_scope must be one of: selected, completed")
    selected = _unique_preserve_order(list(stage3_summary.get("selected_model_ids", []) or []))
    if not selected:
        return completed
    completed_set = set(completed)
    return [model_id for model_id in selected if model_id in completed_set]


def normalize_stage3_predictions(
    stage3_summary: dict[str, Any],
    *,
    model_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_id = str(stage3_summary["dataset_id"])
    label_maps = load_seaad_label_maps()
    metrics = _safe_read_csv(stage3_summary.get("metrics_csv", ""))
    rows: list[pd.DataFrame] = []
    model_weights: dict[str, float] = {}
    model_tasks: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    target_model_ids = model_ids or _stage3_model_scope(stage3_summary, model_scope="selected")
    for model_id in target_model_ids:
        prediction_path = stage3_summary.get("prediction_artifacts", {}).get(model_id)
        spec_path = stage3_summary.get("adapter_specs", {}).get(model_id)
        if not prediction_path or not spec_path:
            skipped.append({"model_id": model_id, "reason": "missing_prediction_or_adapter_spec"})
            continue
        spec = _safe_read_yaml(spec_path)
        spec["_spec_path"] = str(spec_path)
        normalized = normalize_model_predictions(
            dataset_id=dataset_id,
            model_id=model_id,
            prediction_path=prediction_path,
            spec=spec,
            metrics=metrics,
            label_maps=label_maps,
        )
        if normalized.empty:
            skipped.append({"model_id": model_id, "reason": "no_normalizable_predictions"})
            continue
        rows.append(normalized)
        model_weights[model_id] = float(normalized["model_weight"].iloc[0])
        model_tasks[model_id] = str(normalized["task"].iloc[0])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metadata = {
        "model_weights": model_weights,
        "model_tasks": model_tasks,
        "normalization_skips": skipped,
        "target_model_ids": target_model_ids,
    }
    return out, metadata


def _mode_non_unknown(values: pd.Series) -> str:
    labels = [str(x) for x in values if str(x) and str(x) != UNKNOWN_LABEL]
    if not labels:
        return UNKNOWN_LABEL
    return Counter(labels).most_common(1)[0][0]


def build_consensus_frame(normalized: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if normalized.empty:
        return pd.DataFrame(), []
    model_to_cells = {
        model_id: set(group["cell_id"].astype(str))
        for model_id, group in normalized.groupby("model_id", sort=True)
    }
    model_ids = sorted(model_to_cells)
    common_cells = set.intersection(*(cells for cells in model_to_cells.values())) if model_to_cells else set()
    if len(common_cells) == 0:
        cell_counts = normalized.groupby("cell_id")["model_id"].nunique()
        common_cells = set(cell_counts[cell_counts >= 2].index.astype(str))
    if len(common_cells) == 0:
        return pd.DataFrame(), model_ids

    common_index = pd.Index(sorted(common_cells), name="cell_id")
    frame = pd.DataFrame(index=common_index)
    truth = normalized[normalized["cell_id"].isin(common_cells)].groupby("cell_id")["true_shared"].agg(_mode_non_unknown)
    raw_truth = normalized[normalized["cell_id"].isin(common_cells)].groupby("cell_id")["true_raw"].agg(lambda x: Counter(x.astype(str)).most_common(1)[0][0])
    frame["true_shared"] = truth.reindex(common_index).fillna(UNKNOWN_LABEL)
    frame["true_raw_mode"] = raw_truth.reindex(common_index).fillna("")
    for model_id in model_ids:
        sub = normalized[(normalized["model_id"] == model_id) & normalized["cell_id"].isin(common_cells)].drop_duplicates("cell_id")
        sub = sub.set_index("cell_id").reindex(common_index)
        frame[f"{model_id}__pred_shared"] = sub["pred_shared"].fillna(UNKNOWN_LABEL).astype(str)
        frame[f"{model_id}__pred_raw"] = sub["pred_raw"].fillna("").astype(str)
        frame[f"{model_id}__confidence"] = pd.to_numeric(sub["confidence"], errors="coerce").fillna(0.0).astype(float)
        frame[f"{model_id}__support_base"] = pd.to_numeric(sub["support_base"], errors="coerce").fillna(0.0).astype(float)
        frame[f"{model_id}__model_weight"] = pd.to_numeric(sub["model_weight"], errors="coerce").fillna(1.0).astype(float)
    return frame, model_ids


def _top_vote_count(values: list[str]) -> int:
    known = [value for value in values if value != UNKNOWN_LABEL]
    if not known:
        return 0
    return int(Counter(known).most_common(1)[0][1])


def _sample_probe_frame(
    frame: pd.DataFrame,
    model_ids: list[str],
    *,
    primary_model: str,
    max_probe_cells: int,
    seed: int,
) -> pd.DataFrame:
    if max_probe_cells <= 0 or len(frame) <= max_probe_cells:
        return frame.copy()
    probe = frame.copy()
    pred_columns = [f"{model_id}__pred_shared" for model_id in model_ids if f"{model_id}__pred_shared" in probe.columns]
    pred_matrix = probe[pred_columns].astype(str)
    n_non_unknown = (pred_matrix != UNKNOWN_LABEL).sum(axis=1).astype(int)
    top_vote = pred_matrix.apply(lambda row: _top_vote_count(row.tolist()), axis=1).astype(int)
    primary_pred = probe[f"{primary_model}__pred_shared"].astype(str) if f"{primary_model}__pred_shared" in probe.columns else pd.Series("", index=probe.index)
    strata = primary_pred + "|known=" + n_non_unknown.astype(str) + "|top=" + top_vote.astype(str)
    probe["_probe_stratum"] = strata.astype(str)

    rng = np.random.default_rng(seed)
    groups = []
    for _, idx in probe.groupby("_probe_stratum", sort=True).groups.items():
        group_index = pd.Index(idx)
        groups.append(group_index.to_numpy())
    if not groups:
        return frame.sample(n=max_probe_cells, random_state=seed).copy()

    total = len(probe)
    allocations = [max(1, int(round(len(group) / total * max_probe_cells))) for group in groups]
    selected_idx: list[str] = []
    for group, target in zip(groups, allocations, strict=False):
        if len(group) <= target:
            selected_idx.extend(group.tolist())
            continue
        chosen = rng.choice(group, size=target, replace=False)
        selected_idx.extend(chosen.tolist())
    if len(selected_idx) < max_probe_cells:
        remaining = probe.index.difference(pd.Index(selected_idx)).to_numpy()
        if len(remaining):
            extra = rng.choice(remaining, size=min(max_probe_cells - len(selected_idx), len(remaining)), replace=False)
            selected_idx.extend(extra.tolist())
    if len(selected_idx) > max_probe_cells:
        selected_idx = rng.choice(np.asarray(selected_idx, dtype=object), size=max_probe_cells, replace=False).tolist()
    return probe.loc[pd.Index(selected_idx)].drop(columns=["_probe_stratum"], errors="ignore").copy()


def _stage3_model_evidence(
    frame: pd.DataFrame,
    model_ids: list[str],
    *,
    primary_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary_pred = frame[f"{primary_model}__pred_shared"].astype(str) if f"{primary_model}__pred_shared" in frame.columns else pd.Series("", index=frame.index)
    for model_id in model_ids:
        pred_col = f"{model_id}__pred_shared"
        conf_col = f"{model_id}__confidence"
        if pred_col not in frame.columns:
            continue
        preds = frame[pred_col].astype(str)
        non_unknown = preds != UNKNOWN_LABEL
        agreement = (preds == primary_pred) & non_unknown & (primary_pred != UNKNOWN_LABEL)
        rows.append(
            {
                "model_id": model_id,
                "accepted_fraction": float(non_unknown.mean()) if len(preds) else 0.0,
                "unknown_fraction": float((~non_unknown).mean()) if len(preds) else 0.0,
                "mean_confidence": float(pd.to_numeric(frame.get(conf_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()),
                "n_predicted_labels": int(preds[non_unknown].nunique()),
                "primary_model_agreement": float(agreement.mean()) if len(agreement) else 0.0,
            }
        )
    return rows


def _stage2_selection_context(stage3_summary: dict[str, Any]) -> dict[str, Any]:
    plan_path = str(stage3_summary.get("plan_path", "") or "")
    if not plan_path:
        return {}
    plan = _safe_read_yaml(plan_path)
    selected_pairs = []
    for row in plan.get("selected_pairs", []) if isinstance(plan.get("selected_pairs", []), list) else []:
        if not isinstance(row, dict):
            continue
        selected_pairs.append(
            {
                "rank": int(row.get("rank", len(selected_pairs) + 1)),
                "model_id": str(row.get("model_id", "")),
                "source_id": str(row.get("source_id", "")),
                "reference_path": str(row.get("reference_path", "")),
                "score": float(row.get("score", 0.0) or 0.0),
                "source_similarity": float(row.get("source_similarity", 0.0) or 0.0),
                "shared_genes": int(row.get("shared_genes", 0) or 0),
                "source_model_macro_f1_lcb": float(row.get("source_model_macro_f1_lcb", 0.0) or 0.0),
                "source_model_macro_f1": float(row.get("source_model_macro_f1", 0.0) or 0.0),
                "robustness": float(row.get("robustness", 0.0) or 0.0),
                "benchmark_evidence_label": str(row.get("benchmark_evidence_label", "")),
            }
        )
    return {
        "dataset_id": str(plan.get("dataset_id", "")),
        "query_adapter": str(plan.get("query_adapter", "")),
        "query_profile_path": str(plan.get("query_profile_path", "")),
        "selected_model_ids": [str(x) for x in plan.get("selected_model_ids", []) or []],
        "selected_pairs": selected_pairs,
        "selection_policy": plan.get("selection_policy", {}),
    }


def _prediction_support(row: pd.Series, model_id: str, *, use_confidence: bool, use_weight: bool) -> float:
    support = 1.0
    if use_confidence:
        support *= float(row.get(f"{model_id}__support_base", 0.0))
    if use_weight:
        support *= float(row.get(f"{model_id}__model_weight", 1.0))
    return support


def _weighted_label_vote(
    frame: pd.DataFrame,
    model_ids: list[str],
    *,
    use_confidence: bool,
    use_weight: bool,
    require_non_tie: bool = True,
) -> np.ndarray:
    preds: list[str] = []
    for _, row in frame.iterrows():
        support_by_label: dict[str, float] = defaultdict(float)
        for model_id in model_ids:
            label = str(row.get(f"{model_id}__pred_shared", UNKNOWN_LABEL))
            if label == UNKNOWN_LABEL:
                continue
            support_by_label[label] += _prediction_support(row, model_id, use_confidence=use_confidence, use_weight=use_weight)
        if not support_by_label:
            preds.append(UNKNOWN_LABEL)
            continue
        ranked = sorted(support_by_label.items(), key=lambda item: (-item[1], item[0]))
        if require_non_tie and len(ranked) > 1 and np.isclose(ranked[0][1], ranked[1][1]):
            preds.append(UNKNOWN_LABEL)
        else:
            preds.append(ranked[0][0])
    return np.asarray(preds, dtype=object)


def majority_vote_predictions(frame: pd.DataFrame, model_ids: list[str]) -> np.ndarray:
    return _weighted_label_vote(frame, model_ids, use_confidence=False, use_weight=False)


def capability_weighted_vote_predictions(frame: pd.DataFrame, model_ids: list[str]) -> np.ndarray:
    return _weighted_label_vote(frame, model_ids, use_confidence=False, use_weight=True)


def confidence_weighted_vote_predictions(frame: pd.DataFrame, model_ids: list[str]) -> np.ndarray:
    return _weighted_label_vote(frame, model_ids, use_confidence=True, use_weight=True)


def agreement_then_confidence_predictions(frame: pd.DataFrame, model_ids: list[str]) -> np.ndarray:
    preds: list[str] = []
    for _, row in frame.iterrows():
        votes = [str(row[f"{model_id}__pred_shared"]) for model_id in model_ids if str(row[f"{model_id}__pred_shared"]) != UNKNOWN_LABEL]
        if not votes:
            preds.append(UNKNOWN_LABEL)
            continue
        ranked_counts = Counter(votes).most_common()
        if ranked_counts[0][1] >= 2:
            top_count = ranked_counts[0][1]
            top_labels = sorted(label for label, count in ranked_counts if count == top_count)
            if len(top_labels) == 1:
                preds.append(top_labels[0])
                continue
        best_label = UNKNOWN_LABEL
        best_score = -1.0
        for model_id in model_ids:
            label = str(row[f"{model_id}__pred_shared"])
            if label == UNKNOWN_LABEL:
                continue
            score = _prediction_support(row, model_id, use_confidence=True, use_weight=True)
            if score > best_score:
                best_score = score
                best_label = label
        preds.append(best_label)
    return np.asarray(preds, dtype=object)


def _lowest_accept_primary_model(frame: pd.DataFrame, model_ids: list[str]) -> str:
    accepted = {
        model_id: float((frame[f"{model_id}__pred_shared"].astype(str) != UNKNOWN_LABEL).mean())
        for model_id in model_ids
    }
    return min(model_ids, key=lambda model_id: (accepted[model_id], model_id))


def high_conf_majority_override_primary_predictions(frame: pd.DataFrame, model_ids: list[str]) -> tuple[np.ndarray, str]:
    primary_model = _lowest_accept_primary_model(frame, model_ids)
    preds: list[str] = []
    for _, row in frame.iterrows():
        primary = str(row[f"{primary_model}__pred_shared"])
        primary_conf = float(row[f"{primary_model}__confidence"])
        if primary != UNKNOWN_LABEL and primary_conf >= 0.99:
            preds.append(primary)
            continue
        votes = [str(row[f"{model_id}__pred_shared"]) for model_id in model_ids if str(row[f"{model_id}__pred_shared"]) != UNKNOWN_LABEL]
        if not votes:
            preds.append(primary)
            continue
        counts = Counter(votes)
        ranked = counts.most_common()
        top_count = ranked[0][1]
        top_labels = sorted(label for label, count in ranked if count == top_count)
        if len(top_labels) == 1 and top_count > (len(votes) / 2.0):
            avg_conf = float(
                np.mean(
                    [
                        float(row[f"{model_id}__confidence"])
                        for model_id in model_ids
                        if str(row[f"{model_id}__pred_shared"]) == top_labels[0]
                    ]
                )
            )
            preds.append(top_labels[0] if avg_conf >= 0.95 else primary)
        else:
            preds.append(primary)
    return np.asarray(preds, dtype=object), primary_model


def best_confident_model_predictions(frame: pd.DataFrame, model_ids: list[str]) -> np.ndarray:
    preds: list[str] = []
    for _, row in frame.iterrows():
        best_label = UNKNOWN_LABEL
        best_score = -1.0
        for model_id in model_ids:
            label = str(row[f"{model_id}__pred_shared"])
            if label == UNKNOWN_LABEL:
                continue
            score = _prediction_support(row, model_id, use_confidence=True, use_weight=True)
            if score > best_score:
                best_score = score
                best_label = label
        preds.append(best_label)
    return np.asarray(preds, dtype=object)


def majority_override_primary_predictions(
    frame: pd.DataFrame,
    model_ids: list[str],
    *,
    primary_model: str,
    require_perfect_support: bool,
    min_majority_avg_confidence: float | None = None,
) -> np.ndarray:
    preds: list[str] = []
    for _, row in frame.iterrows():
        primary = str(row[f"{primary_model}__pred_shared"])
        votes = [str(row[f"{model_id}__pred_shared"]) for model_id in model_ids if str(row[f"{model_id}__pred_shared"]) != UNKNOWN_LABEL]
        if not votes:
            preds.append(primary)
            continue
        counts = Counter(votes)
        ranked = counts.most_common()
        top_count = ranked[0][1]
        top_labels = sorted(label for label, count in ranked if count == top_count)
        if len(top_labels) != 1:
            preds.append(primary)
            continue
        top_label = top_labels[0]
        support_ok = top_count == len(votes) if require_perfect_support else top_count > (len(votes) / 2.0)
        if not support_ok:
            preds.append(primary)
            continue
        if min_majority_avg_confidence is not None:
            confs = [
                float(row[f"{model_id}__confidence"])
                for model_id in model_ids
                if str(row[f"{model_id}__pred_shared"]) == top_label
            ]
            if not confs or float(np.mean(confs)) < float(min_majority_avg_confidence):
                preds.append(primary)
                continue
        preds.append(top_label)
    return np.asarray(preds, dtype=object)


def weighted_consensus_fixed_predictions(
    frame: pd.DataFrame,
    model_ids: list[str],
    *,
    min_share: float = 0.50,
    min_margin: float = 0.10,
    unknown_vote_threshold: int | None = None,
) -> np.ndarray:
    threshold = int(unknown_vote_threshold if unknown_vote_threshold is not None else max(2, len(model_ids) - 1))
    preds: list[str] = []
    for _, row in frame.iterrows():
        unknown_votes = sum(1 for model_id in model_ids if str(row[f"{model_id}__pred_shared"]) == UNKNOWN_LABEL)
        if unknown_votes >= threshold:
            preds.append(UNKNOWN_LABEL)
            continue
        support_by_label: dict[str, float] = defaultdict(float)
        for model_id in model_ids:
            label = str(row[f"{model_id}__pred_shared"])
            if label == UNKNOWN_LABEL:
                continue
            support_by_label[label] += _prediction_support(row, model_id, use_confidence=True, use_weight=True)
        if not support_by_label:
            preds.append(UNKNOWN_LABEL)
            continue
        ranked = sorted(support_by_label.items(), key=lambda item: (-item[1], item[0]))
        total = sum(support_by_label.values())
        top_share = ranked[0][1] / max(total, 1e-12)
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = (ranked[0][1] - second) / max(total, 1e-12)
        preds.append(ranked[0][0] if top_share >= min_share and margin >= min_margin else UNKNOWN_LABEL)
    return np.asarray(preds, dtype=object)


def guarded_primary_fill_predictions(
    frame: pd.DataFrame,
    model_ids: list[str],
    *,
    primary_model: str,
    min_global_agreement: float = 0.85,
    min_support: int = 2,
    min_avg_confidence: float = 0.60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservative label-free fusion with a primary-model floor.

    The method starts from the Stage-2 primary model and only fills primary
    `__unknown__` predictions when the model set is globally coherent and the
    non-primary models agree with sufficient confidence. It never changes a
    non-unknown primary prediction, so it is a safe deployment fallback when
    weaker auxiliary models are not complementary.
    """
    if primary_model not in model_ids:
        primary_model = model_ids[0]
    primary = frame[f"{primary_model}__pred_shared"].astype(str).to_numpy(dtype=object).copy()
    agreement_values: list[float] = []
    for _, row in frame.iterrows():
        known = [
            str(row[f"{model_id}__pred_shared"])
            for model_id in model_ids
            if str(row[f"{model_id}__pred_shared"]) != UNKNOWN_LABEL
        ]
        if not known:
            agreement_values.append(0.0)
            continue
        agreement_values.append(Counter(known).most_common(1)[0][1] / len(known))
    mean_agreement = float(np.mean(agreement_values)) if agreement_values else 0.0
    if mean_agreement < float(min_global_agreement):
        return primary, {
            "adaptive_primary_model": primary_model,
            "guard_mode": "primary_only_low_global_agreement",
            "mean_global_agreement": mean_agreement,
            "min_global_agreement": float(min_global_agreement),
            "filled_fraction": 0.0,
        }

    filled = 0
    preds: list[str] = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        current = str(primary[idx])
        if current != UNKNOWN_LABEL:
            preds.append(current)
            continue
        votes: list[tuple[str, float]] = []
        for model_id in model_ids:
            if model_id == primary_model:
                continue
            label = str(row[f"{model_id}__pred_shared"])
            if label == UNKNOWN_LABEL:
                continue
            votes.append((label, float(row.get(f"{model_id}__confidence", 0.0))))
        if not votes:
            preds.append(current)
            continue
        counts = Counter(label for label, _ in votes)
        ranked = counts.most_common()
        top_label, top_count = ranked[0]
        tied = len(ranked) > 1 and ranked[1][1] == top_count
        confs = [conf for label, conf in votes if label == top_label]
        if not tied and top_count >= int(min_support) and float(np.mean(confs)) >= float(min_avg_confidence):
            preds.append(top_label)
            filled += 1
        else:
            preds.append(current)
    return np.asarray(preds, dtype=object), {
        "adaptive_primary_model": primary_model,
        "guard_mode": "primary_unknown_fill_only",
        "mean_global_agreement": mean_agreement,
        "min_global_agreement": float(min_global_agreement),
        "min_support": int(min_support),
        "min_avg_confidence": float(min_avg_confidence),
        "filled_fraction": float(filled / max(1, len(primary))),
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    shared_mask = y_true != UNKNOWN_LABEL
    accepted_mask = y_pred != UNKNOWN_LABEL
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "shared_accuracy": 0.0,
        "shared_macro_f1": 0.0,
        "ood_rejection_rate": 0.0,
        "accepted_fraction": float(np.mean(accepted_mask)) if len(accepted_mask) else 0.0,
        "accepted_accuracy": 0.0,
        "n_cells": int(len(y_true)),
    }
    if int(shared_mask.sum()) > 0:
        out["shared_accuracy"] = float(accuracy_score(y_true[shared_mask], y_pred[shared_mask]))
        out["shared_macro_f1"] = float(f1_score(y_true[shared_mask], y_pred[shared_mask], average="macro", zero_division=0))
    ood_mask = ~shared_mask
    if int(ood_mask.sum()) > 0:
        out["ood_rejection_rate"] = float(np.mean(y_pred[ood_mask] == UNKNOWN_LABEL))
    if int(accepted_mask.sum()) > 0:
        out["accepted_accuracy"] = float(accuracy_score(y_true[accepted_mask], y_pred[accepted_mask]))
    return out


def _source_species(reference_path: str | Path) -> str:
    manifest = Path(reference_path) / "source_manifest.json"
    if manifest.exists():
        return str(read_json(manifest).get("species", ""))
    return ""


def _topk_cosine(ref_emb: np.ndarray, query_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, min(int(k), ref_emb.shape[0]))
    ref_norm = normalize(ref_emb, norm="l2").astype(np.float32, copy=False)
    query_norm = normalize(query_emb, norm="l2").astype(np.float32, copy=False)
    all_idx: list[np.ndarray] = []
    all_dist: list[np.ndarray] = []
    ref_t = ref_norm.T
    for start in range(0, query_norm.shape[0], 2048):
        sims = query_norm[start : start + 2048] @ ref_t
        if k == sims.shape[1]:
            top = np.argsort(-sims, axis=1)[:, :k]
        else:
            part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            row = np.arange(part.shape[0])[:, None]
            order = np.argsort(-sims[row, part], axis=1)
            top = part[row, order]
        row = np.arange(top.shape[0])[:, None]
        dist = 1.0 - np.clip(sims[row, top], -1.0, 1.0)
        all_idx.append(top.astype(np.int64, copy=False))
        all_dist.append(dist.astype(np.float32, copy=False))
    return np.vstack(all_idx), np.vstack(all_dist)


def _shared_labels_from_ref(ref: MatrixBundle, label_maps: dict[str, Any]) -> np.ndarray:
    if "coarse_label" in ref.obs.columns:
        values = ref.obs["coarse_label"].astype(str)
    elif "native_label" in ref.obs.columns:
        values = ref.obs["native_label"].astype(str)
    else:
        values = pd.Series([UNKNOWN_LABEL] * ref.obs.shape[0])
    return values.map(lambda x: label_to_shared_coarse(x, label_maps=label_maps)).to_numpy(dtype=object)


def _distribution_from_neighbors(
    *,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    reference_shared_labels: np.ndarray,
    confidence: np.ndarray,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    distances = np.asarray(neighbor_distances, dtype=np.float32)
    if radii is None:
        weights = np.exp(-np.clip(distances, 0.0, None))
    else:
        local_radii = np.clip(radii[neighbor_indices], 1e-4, None)
        weights = np.exp(-np.clip(distances, 0.0, None) / local_radii)
    neighbor_labels = reference_shared_labels[neighbor_indices]
    masses = np.zeros((neighbor_indices.shape[0], len(CONSENSUS_LABELS)), dtype=np.float32)
    for label in SHARED_COARSE_LABELS:
        label_idx = CONSENSUS_LABEL_TO_INDEX[label]
        masses[:, label_idx] = (weights * (neighbor_labels == label)).sum(axis=1)
    known_mass = masses[:, :-1].sum(axis=1, keepdims=True)
    known_mass[known_mass == 0.0] = 1.0
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1, 1)
    masses[:, :-1] = masses[:, :-1] / known_mass * np.clip(confidence, 0.0, 1.0)
    masses[:, -1:] = 1.0 - np.clip(confidence, 0.0, 1.0)
    return masses


def _prototype_payload(ref_emb: np.ndarray, reference_shared_labels: np.ndarray) -> dict[str, Any] | None:
    ref_norm = normalize(ref_emb, norm="l2").astype(np.float32, copy=False)
    prototypes: list[np.ndarray] = []
    prototype_labels: list[str] = []
    radii_by_ref = np.ones(ref_norm.shape[0], dtype=np.float32)
    for label in SHARED_COARSE_LABELS:
        mask = reference_shared_labels == label
        if int(mask.sum()) == 0:
            continue
        members = ref_norm[mask]
        proto = members.mean(axis=0).astype(np.float32)
        proto = proto / max(float(np.linalg.norm(proto)), 1e-12)
        dist = 1.0 - np.clip(members @ proto, -1.0, 1.0)
        radius = max(float(np.median(dist)) if len(dist) else 1.0, 1e-4)
        radii_by_ref[mask] = radius
        prototypes.append(proto)
        prototype_labels.append(label)
    if not prototypes:
        return None
    return {
        "prototypes": np.stack(prototypes, axis=0).astype(np.float32),
        "prototype_labels": np.asarray(prototype_labels, dtype=object),
        "radii_by_ref": radii_by_ref,
    }


def _prototype_distribution(query_emb: np.ndarray, prototype_payload: dict[str, Any]) -> np.ndarray:
    q = normalize(query_emb, norm="l2").astype(np.float32, copy=False)
    protos = np.asarray(prototype_payload["prototypes"], dtype=np.float32)
    labels = np.asarray(prototype_payload["prototype_labels"], dtype=object)
    sims = q @ protos.T
    logits = np.full((q.shape[0], len(SHARED_COARSE_LABELS)), -20.0, dtype=np.float32)
    for idx, label in enumerate(SHARED_COARSE_LABELS):
        positions = np.flatnonzero(labels == label)
        if len(positions):
            logits[:, idx] = sims[:, positions].max(axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs.astype(np.float32)


def _load_chunked_query_embeddings(
    *,
    stage3_summary: dict[str, Any],
    model_id: str,
    dataset_id: str,
    adapter: EmbeddingAdapter,
    ref: MatrixBundle,
    min_shared_genes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reuse Stage3 chunked query embedding cache when available."""
    output_dir = Path(stage3_summary.get("output_dir", ""))
    run_summary_path = output_dir / "model_runs" / model_id / "run_summary.json"
    if not run_summary_path.exists():
        return None
    try:
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not payload.get("chunked_execution"):
        return None
    chunks = payload.get("chunk_results", [])
    if not isinstance(chunks, list) or not chunks:
        return None

    embeddings: list[np.ndarray] = []
    cell_ids: list[np.ndarray] = []
    expected_genes = [str(g) for g in adapter.genes]
    for chunk in sorted(chunks, key=lambda item: int(item.get("start", item.get("chunk_index", 0)))):
        if str(chunk.get("status", "completed")) != "completed":
            raise ValueError(f"chunked stage3 run has non-completed chunk for {model_id}: {chunk}")
        plan_path = chunk.get("plan_path")
        if not plan_path:
            raise ValueError(f"chunked stage3 run missing plan_path for {model_id}: {chunk}")
        chunk_plan = read_yaml(plan_path)
        query_path = chunk_plan.get("query_path")
        if not query_path:
            raise ValueError(f"chunked stage3 plan missing query_path for {model_id}: {plan_path}")
        query = load_query_bundle(query_path, dataset_id=dataset_id, max_cells=0, seed=seed).bundle
        _, query_aligned, shared_genes = _align_reference_and_query(ref, query, min_shared_genes=min_shared_genes)
        if [str(g) for g in shared_genes] != expected_genes:
            raise ValueError(f"chunked geometry shared genes differ for {model_id}: {len(shared_genes)} vs {len(expected_genes)}")
        embeddings.append(adapter.transform(query_aligned.X).astype(np.float32, copy=False))
        if "cell_id" in query_aligned.obs.columns:
            cell_ids.append(query_aligned.obs["cell_id"].astype(str).to_numpy())
        else:
            cell_ids.append(np.asarray([f"cell_{idx}" for idx in range(query_aligned.X.shape[0])], dtype=str))
    return np.vstack(embeddings).astype(np.float32, copy=False), np.concatenate(cell_ids).astype(str)


def build_reference_geometry(
    *,
    stage3_summary: dict[str, Any],
    frame: pd.DataFrame,
    normalized: pd.DataFrame,
    label_maps: dict[str, Any],
    model_ids: list[str] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    geometry: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []
    dataset_id = str(stage3_summary["dataset_id"])
    common_cells = list(frame.index.astype(str))
    target_model_ids = model_ids or _stage3_model_scope(stage3_summary, model_scope="selected")
    for model_id in target_model_ids:
        spec_path = stage3_summary.get("adapter_specs", {}).get(model_id)
        spec = _safe_read_yaml(spec_path) if spec_path else {}
        actions = {str(item.get("action", "")) for item in spec.get("actions", []) if isinstance(item, dict)}
        runtime = spec.get("runtime_payload", {}) if isinstance(spec, dict) else {}
        reference_path = runtime.get("reference_path") or spec.get("input_artifacts", {}).get("reference_path")
        embedding_method = runtime.get("embedding_method")
        if "invoke_raw_embedding_transfer" not in actions or not reference_path or not embedding_method:
            summary_rows.append({"model_id": model_id, "status": "reference_geometry_unavailable", "reason": "not_raw_label_transfer_selected_model"})
            continue
        try:
            max_query_cells = int(runtime.get("max_query_cells", 5000))
            max_reference_cells = int(runtime.get("max_reference_cells", 1000))
            min_shared_genes = int(runtime.get("min_shared_genes", 30))
            k = int(runtime.get("k", 15))
            seed = int(runtime.get("seed", 3028))
            batch_size = int(runtime.get("batch_size", 16))
            query = load_query_bundle(spec["query_path"], dataset_id=dataset_id, max_cells=max_query_cells, seed=seed).bundle
            ref, _ = _load_reference_from_standard_bundle(reference_path, genes=query.genes, max_cells=max_reference_cells, seed=seed)
            ref_aligned, query_aligned, shared_genes = _align_reference_and_query(ref, query, min_shared_genes=min_shared_genes)
            adapter = EmbeddingAdapter(
                base_method=str(embedding_method),
                genes=shared_genes,
                species=_source_species(reference_path),
                device=str(runtime.get("device", "") or ""),
                batch_size=batch_size,
            )
            adapter.fit()
            ref_emb = adapter.transform(ref_aligned.X)
            chunked_query = _load_chunked_query_embeddings(
                stage3_summary=stage3_summary,
                model_id=model_id,
                dataset_id=dataset_id,
                adapter=adapter,
                ref=ref,
                min_shared_genes=min_shared_genes,
                seed=seed,
            )
            if chunked_query is None:
                query_emb = adapter.transform(query_aligned.X)
                query_cell_ids = (
                    query_aligned.obs["cell_id"].astype(str).to_numpy()
                    if "cell_id" in query_aligned.obs.columns
                    else np.asarray([f"cell_{idx}" for idx in range(query_aligned.X.shape[0])], dtype=str)
                )
                geometry_mode = "full_query_transform"
            else:
                query_emb, query_cell_ids = chunked_query
                geometry_mode = "stage3_chunk_cache"
            positions = pd.Index(query_cell_ids).get_indexer(common_cells)
            if np.any(positions < 0):
                missing = int(np.sum(positions < 0))
                raise ValueError(f"geometry query bundle missing {missing}/{len(common_cells)} consensus cells")
            query_emb = query_emb[positions]
            neighbor_idx, neighbor_dist = _topk_cosine(ref_emb, query_emb, k=k)
            ref_labels = _shared_labels_from_ref(ref_aligned, label_maps)
            confidence = (
                normalized[(normalized["model_id"] == model_id)]
                .drop_duplicates("cell_id")
                .set_index("cell_id")
                .reindex(common_cells)["confidence"]
                .fillna(0.0)
                .to_numpy(np.float32)
            )
            neighbor_distribution = _distribution_from_neighbors(
                neighbor_indices=neighbor_idx,
                neighbor_distances=neighbor_dist,
                reference_shared_labels=ref_labels,
                confidence=confidence,
            )
            logistic = train_reference_logistic_distribution(
                reference_embeddings=ref_emb,
                reference_labels=ref_labels,
                query_embeddings=query_emb,
                consensus_labels=CONSENSUS_LABELS,
                unknown_label=UNKNOWN_LABEL,
                seed=seed,
            )
            proto = _prototype_payload(ref_emb, ref_labels)
            density_distribution = None
            reference_enhanced_distribution = None
            if proto is not None:
                density_distribution = _distribution_from_neighbors(
                    neighbor_indices=neighbor_idx,
                    neighbor_distances=neighbor_dist,
                    reference_shared_labels=ref_labels,
                    confidence=confidence,
                    radii=np.asarray(proto["radii_by_ref"], dtype=np.float32),
                )
                unknown_mass = density_distribution[:, -1:]
                known_mass = np.clip(1.0 - unknown_mass, 1e-12, None)
                density_known = density_distribution[:, :-1] / known_mass
                prototype_known = _prototype_distribution(query_emb, proto)
                hybrid_known = np.sqrt(np.clip(density_known, 1e-12, None) * np.clip(prototype_known, 1e-12, None))
                hybrid_known /= np.clip(hybrid_known.sum(axis=1, keepdims=True), 1e-12, None)
                reference_enhanced_distribution = np.zeros_like(density_distribution)
                reference_enhanced_distribution[:, :-1] = hybrid_known * known_mass
                reference_enhanced_distribution[:, -1:] = unknown_mass
            geometry[model_id] = {
                "query_embeddings": normalize(query_emb, norm="l2").astype(np.float32, copy=False),
                "neighbor_distribution": neighbor_distribution,
                "density_distribution": density_distribution if density_distribution is not None else neighbor_distribution,
                "reference_enhanced_distribution": reference_enhanced_distribution
                if reference_enhanced_distribution is not None
                else neighbor_distribution,
            }
            logistic_status = "unavailable"
            if logistic is not None:
                logistic_distribution, logistic_metadata = logistic
                geometry[model_id]["reference_logistic_distribution"] = logistic_distribution
                geometry[model_id]["reference_logistic_metadata"] = logistic_metadata
                logistic_status = "available"
            summary_rows.append(
                {
                    "model_id": model_id,
                    "status": "available",
                    "reference_path": str(reference_path),
                    "embedding_method": str(embedding_method),
                    "n_query_cells": int(query_emb.shape[0]),
                    "n_reference_cells": int(ref_emb.shape[0]),
                    "n_shared_genes": int(len(shared_genes)),
                    "geometry_mode": geometry_mode,
                    "reference_logistic_head": logistic_status,
                }
            )
        except Exception as exc:  # noqa: BLE001
            summary_rows.append(
                {
                    "model_id": model_id,
                    "status": "reference_geometry_unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                }
            )
    return geometry, pd.DataFrame(summary_rows)


def _subset_geometry_for_frame(
    geometry: dict[str, dict[str, np.ndarray]],
    *,
    full_frame: pd.DataFrame,
    subset_frame: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    if subset_frame.index.equals(full_frame.index):
        return geometry
    positions = pd.Index(full_frame.index.astype(str)).get_indexer(subset_frame.index.astype(str))
    if np.any(positions < 0):
        missing = int(np.sum(positions < 0))
        raise ValueError(f"probe frame is missing {missing} cells from full-frame geometry alignment")
    subset_geometry: dict[str, dict[str, np.ndarray]] = {}
    full_n = int(full_frame.shape[0])
    for model_id, payload in geometry.items():
        subset_payload: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and int(value.shape[0]) == full_n:
                subset_payload[key] = value[positions]
            else:
                subset_payload[key] = value
        subset_geometry[model_id] = subset_payload
    return subset_geometry


def _distribution_mean(distributions: dict[str, dict[str, np.ndarray]], model_ids: list[str], key: str) -> np.ndarray:
    stack = np.stack([distributions[model_id][key] for model_id in model_ids], axis=0).astype(np.float32)
    out = stack.mean(axis=0)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _distribution_to_pred(distribution: np.ndarray, fallback: np.ndarray, min_prob: float) -> np.ndarray:
    labels = np.asarray(CONSENSUS_LABELS, dtype=object)
    pred = labels[distribution.argmax(axis=1)]
    max_prob = distribution.max(axis=1)
    pred[max_prob < min_prob] = np.asarray(fallback, dtype=object)[max_prob < min_prob]
    return pred


def _unknown_gated_primary_blend(distributions: dict[str, dict[str, np.ndarray]], model_ids: list[str], key: str, primary_model: str) -> np.ndarray:
    combined = _distribution_mean(distributions, model_ids, key)
    primary = np.asarray(distributions[primary_model][key], dtype=np.float32)
    gate = combined[:, CONSENSUS_LABEL_TO_INDEX[UNKNOWN_LABEL]].reshape(-1, 1)
    out = ((1.0 - gate) * combined) + (gate * primary)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _diffuse_distribution(base_distribution: np.ndarray, query_embeddings: np.ndarray, *, seed: int) -> np.ndarray:
    n = query_embeddings.shape[0]
    if n <= 2:
        return base_distribution
    k = min(64 if n <= 200_000 else 32, n - 1)
    if n > 200_000:
        from pynndescent import NNDescent

        index = NNDescent(
            query_embeddings,
            n_neighbors=k + 1,
            metric="cosine",
            random_state=seed,
            n_jobs=-1,
            low_memory=True,
            verbose=False,
        )
        indices, distances = index.neighbor_graph
        if indices.shape[1] > k and np.mean(indices[:, 0] == np.arange(n)) > 0.90:
            indices = indices[:, 1 : k + 1]
            distances = distances[:, 1 : k + 1]
        else:
            indices = indices[:, :k]
            distances = distances[:, :k]
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(query_embeddings)
        distances, indices = nn.kneighbors(query_embeddings, return_distance=True)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    positive = distances[distances > 0]
    sigma = max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
    weights = np.exp(-np.clip(distances, 0.0, None) / sigma)
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    propagated = np.asarray(base_distribution, dtype=np.float32)
    base = np.asarray(base_distribution, dtype=np.float32)
    for _ in range(3):
        neighbor_avg = np.einsum("ij,ijk->ik", weights, propagated[indices])
        propagated = (0.25 * base) + (0.75 * neighbor_avg)
        propagated /= np.clip(propagated.sum(axis=1, keepdims=True), 1e-12, None)
    return propagated.astype(np.float32)


def _select_graph_distribution(
    base_distribution: np.ndarray,
    geometry: dict[str, dict[str, np.ndarray]],
    *,
    seed: int,
) -> tuple[str | None, np.ndarray | None, dict[str, dict[str, float]]]:
    n_cells = int(base_distribution.shape[0])
    base_unknown = float(base_distribution[:, CONSENSUS_LABEL_TO_INDEX[UNKNOWN_LABEL]].mean())
    base_conf = float(base_distribution.max(axis=1).mean())
    scored: dict[str, dict[str, float]] = {}
    distributions: dict[str, np.ndarray] = {}
    for model_id, payload in geometry.items():
        refined = _diffuse_distribution(base_distribution, payload["query_embeddings"], seed=seed)
        score = (base_unknown - float(refined[:, CONSENSUS_LABEL_TO_INDEX[UNKNOWN_LABEL]].mean())) + (
            float(refined.max(axis=1).mean()) - base_conf
        )
        scored[model_id] = {
            "score": float(score),
            "unknown_reduction": float(base_unknown - float(refined[:, CONSENSUS_LABEL_TO_INDEX[UNKNOWN_LABEL]].mean())),
            "confidence_change": float(float(refined.max(axis=1).mean()) - base_conf),
        }
        distributions[model_id] = refined
    if not scored:
        return None, None, {}
    selected = max(scored.items(), key=lambda item: (item[1]["score"], item[0]))[0]
    return selected, distributions[selected], scored


def _confidence_switch_distribution(
    base_distribution: np.ndarray,
    alternate_distribution: np.ndarray,
    *,
    base_max_threshold: float = 0.60,
    alternate_margin: float = 0.02,
) -> tuple[np.ndarray, dict[str, float]]:
    base = np.asarray(base_distribution, dtype=np.float32)
    alternate = np.asarray(alternate_distribution, dtype=np.float32)
    base_max = base.max(axis=1)
    alternate_max = alternate.max(axis=1)
    use_alternate = (base_max < float(base_max_threshold)) & (alternate_max >= (base_max + float(alternate_margin)))
    out = base.copy()
    out[use_alternate] = alternate[use_alternate]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out.astype(np.float32), {
        "base_max_threshold": float(base_max_threshold),
        "alternate_margin": float(alternate_margin),
        "switched_fraction": float(np.mean(use_alternate)) if len(use_alternate) else 0.0,
    }


def _call_majority_vote(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    return majority_vote_predictions(ctx.frame, ctx.model_ids), {}


def _call_capability_weighted_vote(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    return capability_weighted_vote_predictions(ctx.frame, ctx.model_ids), {}


def _call_confidence_weighted_vote(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    return confidence_weighted_vote_predictions(ctx.frame, ctx.model_ids), {}


def _call_best_confident_model(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    return best_confident_model_predictions(ctx.frame, ctx.model_ids), {}


def _call_agreement_then_confidence(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    return agreement_then_confidence_predictions(ctx.frame, ctx.model_ids), {}


def _call_high_conf_majority_override_primary(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    pred, primary = high_conf_majority_override_primary_predictions(ctx.frame, ctx.model_ids)
    return pred, {"adaptive_primary_model": primary}


def _call_majority_override_lowest_accept_primary(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    primary = ctx.method_metadata.get("high_conf_majority_override_primary", {}).get("adaptive_primary_model")
    if not primary:
        primary = _lowest_accept_primary_model(ctx.frame, ctx.model_ids)
    return (
        majority_override_primary_predictions(
            ctx.frame,
            ctx.model_ids,
            primary_model=str(primary),
            require_perfect_support=False,
        ),
        {"adaptive_primary_model": primary},
    )


def _call_perfect_supported_majority_override_lowest_accept_primary(
    ctx: FusionFunctionContext,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    primary = ctx.method_metadata.get("high_conf_majority_override_primary", {}).get("adaptive_primary_model")
    if not primary:
        primary = _lowest_accept_primary_model(ctx.frame, ctx.model_ids)
    return (
        majority_override_primary_predictions(
            ctx.frame,
            ctx.model_ids,
            primary_model=str(primary),
            require_perfect_support=True,
        ),
        {"adaptive_primary_model": primary},
    )


def _call_weighted_consensus_fixed(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    parameters = {"min_share": 0.50, "min_margin": 0.10, "unknown_vote_threshold": max(2, len(ctx.model_ids) - 1)}
    return weighted_consensus_fixed_predictions(ctx.frame, ctx.model_ids, **parameters), parameters


def _call_stage2_primary_guarded_consensus(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    return guarded_primary_fill_predictions(ctx.frame, ctx.model_ids, primary_model=str(ctx.primary_model))


def _geometry_model_ids(ctx: FusionFunctionContext) -> list[str]:
    return sorted(model_id for model_id in ctx.model_ids if model_id in ctx.geometry)


def _geometry_fallback(ctx: FusionFunctionContext) -> np.ndarray:
    if _method_available(ctx, "high_conf_majority_override_primary"):
        return ctx.method_preds["high_conf_majority_override_primary"]
    return majority_vote_predictions(ctx.frame, ctx.model_ids)


def _call_neighbor_distribution_consensus(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    geometry_model_ids = _geometry_model_ids(ctx)
    if len(geometry_model_ids) < 2:
        return None, {"skip_reason": "requires_at_least_two_reference_geometry_models"}
    dist = _distribution_mean(ctx.geometry, geometry_model_ids, "neighbor_distribution")
    return _distribution_to_pred(dist, _geometry_fallback(ctx), min_prob=2.0 / 3.0), {
        "geometry_model_ids": geometry_model_ids,
        "min_prob": 2.0 / 3.0,
    }


def _call_density_calibrated_neighbor_distribution_consensus(
    ctx: FusionFunctionContext,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    geometry_model_ids = _geometry_model_ids(ctx)
    if len(geometry_model_ids) < 2:
        return None, {"skip_reason": "requires_at_least_two_reference_geometry_models"}
    dist = _distribution_mean(ctx.geometry, geometry_model_ids, "density_distribution")
    return _distribution_to_pred(dist, _geometry_fallback(ctx), min_prob=2.0 / 3.0), {
        "geometry_model_ids": geometry_model_ids,
        "min_prob": 2.0 / 3.0,
    }


def _reference_enhanced_primary_distribution(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    geometry_model_ids = _geometry_model_ids(ctx)
    if len(geometry_model_ids) < 2:
        return None, {"skip_reason": "requires_at_least_two_reference_geometry_models"}
    primary = _lowest_accept_primary_model(ctx.frame, geometry_model_ids)
    dist = _unknown_gated_primary_blend(ctx.geometry, geometry_model_ids, "reference_enhanced_distribution", primary)
    return dist, {"geometry_model_ids": geometry_model_ids, "adaptive_primary_model": primary}


def _call_reference_enhanced_primary_blend_consensus(
    ctx: FusionFunctionContext,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    dist, meta = _reference_enhanced_primary_distribution(ctx)
    if dist is None:
        return None, meta
    return _distribution_to_pred(dist, _geometry_fallback(ctx), min_prob=0.55), {**meta, "min_prob": 0.55}


def _call_query_graph_refined_consensus(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    geometry_model_ids = _geometry_model_ids(ctx)
    if len(geometry_model_ids) < 2:
        return None, {"skip_reason": "requires_at_least_two_reference_geometry_models"}
    primary = _lowest_accept_primary_model(ctx.frame, geometry_model_ids)
    base_dist = _unknown_gated_primary_blend(ctx.geometry, geometry_model_ids, "neighbor_distribution", primary)
    graph_model, graph_dist, graph_scores = _select_graph_distribution(base_dist, ctx.geometry, seed=ctx.seed)
    if graph_dist is None:
        return None, {"skip_reason": "query_graph_distribution_unavailable", "geometry_model_ids": geometry_model_ids}
    return _distribution_to_pred(graph_dist, _geometry_fallback(ctx), min_prob=0.55), {
        "geometry_model_ids": geometry_model_ids,
        "graph_model": graph_model,
        "graph_selection_scores": graph_scores,
        "min_prob": 0.55,
    }


def _call_reference_enhanced_graph_consensus(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    ref_dist, meta = _reference_enhanced_primary_distribution(ctx)
    if ref_dist is None:
        return None, meta
    geometry_model_ids = list(meta.get("geometry_model_ids", []))
    graph_model, graph_dist, graph_scores = _select_graph_distribution(ref_dist, ctx.geometry, seed=ctx.seed)
    if graph_dist is None:
        return None, {"skip_reason": "reference_enhanced_graph_distribution_unavailable", **meta}
    return _distribution_to_pred(graph_dist, _geometry_fallback(ctx), min_prob=0.55), {
        **meta,
        "geometry_model_ids": geometry_model_ids,
        "graph_model": graph_model,
        "graph_selection_scores": graph_scores,
        "min_prob": 0.55,
    }


def _call_reference_enhanced_confidence_switch_graph_consensus(
    ctx: FusionFunctionContext,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    geometry_model_ids = _geometry_model_ids(ctx)
    if len(geometry_model_ids) < 2:
        return None, {"skip_reason": "requires_at_least_two_reference_geometry_models"}
    primary = _lowest_accept_primary_model(ctx.frame, geometry_model_ids)
    base_dist = _unknown_gated_primary_blend(ctx.geometry, geometry_model_ids, "neighbor_distribution", primary)
    _, query_graph_dist, query_graph_scores = _select_graph_distribution(base_dist, ctx.geometry, seed=ctx.seed)
    ref_dist = _unknown_gated_primary_blend(ctx.geometry, geometry_model_ids, "reference_enhanced_distribution", primary)
    _, ref_graph_dist, ref_graph_scores = _select_graph_distribution(ref_dist, ctx.geometry, seed=ctx.seed)
    if query_graph_dist is None or ref_graph_dist is None:
        return None, {"skip_reason": "requires_query_and_reference_graph_distributions", "geometry_model_ids": geometry_model_ids}
    switched, switch_meta = _confidence_switch_distribution(
        query_graph_dist,
        ref_graph_dist,
        base_max_threshold=0.60,
        alternate_margin=0.02,
    )
    return _distribution_to_pred(switched, _geometry_fallback(ctx), min_prob=0.55), {
        "geometry_model_ids": geometry_model_ids,
        "adaptive_primary_model": primary,
        "query_graph_selection_scores": query_graph_scores,
        "reference_graph_selection_scores": ref_graph_scores,
        "min_prob": 0.55,
        **switch_meta,
    }


def _logistic_model_ids(ctx: FusionFunctionContext) -> list[str]:
    return sorted(
        model_id
        for model_id in ctx.model_ids
        if model_id in ctx.geometry and "reference_logistic_distribution" in ctx.geometry[model_id]
    )


def _logistic_fallback(ctx: FusionFunctionContext) -> np.ndarray:
    if _method_available(ctx, "majority_vote"):
        return ctx.method_preds["majority_vote"]
    return majority_vote_predictions(ctx.frame, ctx.model_ids)


def _call_reference_logistic_consensus(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    logistic_model_ids = _logistic_model_ids(ctx)
    if not logistic_model_ids:
        return None, {"skip_reason": "reference_logistic_distribution_unavailable"}
    dist = _distribution_mean(ctx.geometry, logistic_model_ids, "reference_logistic_distribution")
    return _distribution_to_pred(dist, _logistic_fallback(ctx), min_prob=0.50), {
        "reference_logistic_models": logistic_model_ids,
        "training": "balanced LogisticRegression on reference embeddings only",
        "min_prob": 0.50,
    }


def _reference_logistic_blend_distribution(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    logistic_model_ids = _logistic_model_ids(ctx)
    if not logistic_model_ids:
        return None, {"skip_reason": "reference_logistic_distribution_unavailable"}
    primary = _lowest_accept_primary_model(ctx.frame, logistic_model_ids)
    dist = _unknown_gated_primary_blend(ctx.geometry, logistic_model_ids, "reference_logistic_distribution", primary)
    return dist, {"reference_logistic_models": logistic_model_ids, "adaptive_primary_model": primary}


def _call_reference_logistic_primary_blend_consensus(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    dist, meta = _reference_logistic_blend_distribution(ctx)
    if dist is None:
        return None, meta
    return _distribution_to_pred(dist, _logistic_fallback(ctx), min_prob=0.50), {**meta, "min_prob": 0.50}


def _call_reference_logistic_graph_consensus(ctx: FusionFunctionContext) -> tuple[np.ndarray | None, dict[str, Any]]:
    dist, meta = _reference_logistic_blend_distribution(ctx)
    if dist is None:
        return None, meta
    logistic_model_ids = list(meta.get("reference_logistic_models", []))
    logistic_geometry = {model_id: ctx.geometry[model_id] for model_id in logistic_model_ids}
    graph_model, graph_dist, graph_scores = _select_graph_distribution(dist, logistic_geometry, seed=ctx.seed)
    if graph_dist is None:
        return None, {"skip_reason": "reference_logistic_graph_distribution_unavailable", **meta}
    return _distribution_to_pred(graph_dist, _logistic_fallback(ctx), min_prob=0.50), {
        **meta,
        "graph_model": graph_model,
        "graph_selection_scores": graph_scores,
        "min_prob": 0.50,
    }


def _fusion_function_registry(model_ids: list[str]) -> list[FusionFunction]:
    functions: list[FusionFunction] = [_single_model_fusion_function(model_id) for model_id in model_ids]
    functions.extend(
        [
            FusionFunction("majority_vote", "Unweighted non-unknown majority vote.", "fusion", ("model_predictions",), {}, _call_majority_vote),
            FusionFunction(
                "capability_weighted_vote",
                "Vote weighted by pre-query capability-card model weights.",
                "fusion",
                ("model_predictions", "model_weight"),
                {},
                _call_capability_weighted_vote,
            ),
            FusionFunction(
                "confidence_weighted_vote",
                "Vote weighted by model confidence and capability-card weights.",
                "fusion",
                ("model_predictions", "confidence", "model_weight"),
                {},
                _call_confidence_weighted_vote,
            ),
            FusionFunction(
                "best_confident_model",
                "Per-cell choose the highest confidence-weighted non-unknown model prediction.",
                "fusion",
                ("model_predictions", "confidence", "model_weight"),
                {},
                _call_best_confident_model,
            ),
            FusionFunction(
                "agreement_then_confidence",
                "Use agreement when available, otherwise confidence-weighted fallback.",
                "fusion",
                ("model_predictions", "confidence", "model_weight"),
                {},
                _call_agreement_then_confidence,
            ),
            FusionFunction(
                "high_conf_majority_override_primary",
                "Conservative primary model with high-confidence majority override.",
                "fusion",
                ("model_predictions", "confidence"),
                {"primary_confidence": 0.99, "majority_avg_confidence": 0.95},
                _call_high_conf_majority_override_primary,
            ),
            FusionFunction(
                "majority_override_lowest_accept_primary",
                "Use the lowest-acceptance model as primary and override only by majority.",
                "fusion",
                ("model_predictions",),
                {"require_perfect_support": False},
                _call_majority_override_lowest_accept_primary,
            ),
            FusionFunction(
                "perfect_supported_majority_override_lowest_accept_primary",
                "Use the lowest-acceptance model as primary and override only by unanimous known support.",
                "fusion",
                ("model_predictions",),
                {"require_perfect_support": True},
                _call_perfect_supported_majority_override_lowest_accept_primary,
            ),
            FusionFunction(
                "weighted_consensus_fixed",
                "Fixed confidence/capability support consensus with explicit uncertainty rejection.",
                "fusion",
                ("model_predictions", "confidence", "model_weight"),
                {"min_share": 0.50, "min_margin": 0.10, "unknown_vote_threshold": "max(2,n_models-1)"},
                _call_weighted_consensus_fixed,
            ),
            FusionFunction(
                "stage2_primary_guarded_consensus",
                "Stage-2 rank1 primary-model guard; only fills unknowns under strong label-free agreement.",
                "fusion",
                ("model_predictions", "confidence", "stage2_rank1_model"),
                {"min_global_agreement": 0.85, "min_support": 2, "min_avg_confidence": 0.60},
                _call_stage2_primary_guarded_consensus,
            ),
            FusionFunction(
                "neighbor_distribution_consensus",
                "Average reference-neighbor label distributions across geometry-enabled models.",
                "fusion",
                ("reference_geometry",),
                {"min_prob": 2.0 / 3.0},
                _call_neighbor_distribution_consensus,
            ),
            FusionFunction(
                "density_calibrated_neighbor_distribution_consensus",
                "Average local-density calibrated reference-neighbor label distributions.",
                "fusion",
                ("reference_geometry",),
                {"min_prob": 2.0 / 3.0},
                _call_density_calibrated_neighbor_distribution_consensus,
            ),
            FusionFunction(
                "reference_enhanced_primary_blend_consensus",
                "Blend reference-neighbor and prototype geometry with unknown-gated primary distribution.",
                "fusion",
                ("reference_geometry", "reference_prototypes"),
                {"min_prob": 0.55},
                _call_reference_enhanced_primary_blend_consensus,
            ),
            FusionFunction(
                "query_graph_refined_consensus",
                "Diffuse neighbor-distribution consensus over query graph built from frozen embeddings.",
                "fusion",
                ("reference_geometry", "query_graph"),
                {"min_prob": 0.55},
                _call_query_graph_refined_consensus,
            ),
            FusionFunction(
                "reference_enhanced_graph_consensus",
                "Diffuse reference-enhanced distribution over query graph built from frozen embeddings.",
                "fusion",
                ("reference_geometry", "query_graph"),
                {"min_prob": 0.55},
                _call_reference_enhanced_graph_consensus,
            ),
            FusionFunction(
                "reference_enhanced_confidence_switch_graph_consensus",
                "Switch low-confidence query-graph cells to reference-enhanced graph distribution when stronger.",
                "fusion",
                ("reference_geometry", "query_graph"),
                {"base_max_threshold": 0.60, "alternate_margin": 0.02, "min_prob": 0.55},
                _call_reference_enhanced_confidence_switch_graph_consensus,
            ),
            FusionFunction(
                "reference_logistic_consensus",
                "Train balanced logistic heads on reference embeddings only and average query distributions.",
                "fusion",
                ("reference_logistic_distribution",),
                {"min_prob": 0.50},
                _call_reference_logistic_consensus,
            ),
            FusionFunction(
                "reference_logistic_primary_blend_consensus",
                "Unknown-gated primary blend of reference-only logistic head distributions.",
                "fusion",
                ("reference_logistic_distribution",),
                {"min_prob": 0.50},
                _call_reference_logistic_primary_blend_consensus,
            ),
            FusionFunction(
                "reference_logistic_graph_consensus",
                "Diffuse reference-only logistic consensus over query graph.",
                "fusion",
                ("reference_logistic_distribution", "query_graph"),
                {"min_prob": 0.50},
                _call_reference_logistic_graph_consensus,
            ),
        ]
    )
    return functions


def _execute_fusion_function(ctx: FusionFunctionContext, function: FusionFunction) -> dict[str, Any]:
    call_row: dict[str, Any] = {
        "function_name": function.name,
        "method_kind": function.method_kind,
        "required_inputs": list(function.required_inputs),
        "default_parameters": function.default_parameters,
        "description": function.description,
        "status": "pending",
    }
    try:
        pred, metadata = function.call(ctx)
        if pred is None:
            call_row.update({"status": "skipped", "skip_reason": str(metadata.get("skip_reason", "function_returned_no_prediction"))})
            ctx.method_metadata[function.name] = {**metadata, "function_call": call_row}
            return call_row
        pred = np.asarray(pred, dtype=object)
        if len(pred) != len(ctx.frame):
            raise ValueError(f"{function.name} returned {len(pred)} predictions for {len(ctx.frame)} cells")
        ctx.method_preds[function.name] = pred
        ctx.method_metadata[function.name] = {**metadata, "function_call": call_row}
        call_row.update({"status": "executed", "n_predictions": int(len(pred))})
        return call_row
    except Exception as exc:  # noqa: BLE001
        call_row.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
            }
        )
        ctx.method_metadata[function.name] = {"function_call": call_row}
        return call_row


def run_fusion_methods(
    *,
    frame: pd.DataFrame,
    model_ids: list[str],
    geometry: dict[str, dict[str, np.ndarray]],
    seed: int,
    primary_model: str | None = None,
    compute_truth_metrics: bool = True,
    selected_methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y_true = frame["true_shared"].to_numpy(dtype=object) if compute_truth_metrics else None
    if not primary_model or primary_model not in model_ids:
        primary_model = model_ids[0]
    registry = _fusion_function_registry(model_ids)
    if selected_methods:
        selected = {str(method) for method in selected_methods}
        registry = [function for function in registry if function.name in selected]
    ctx = FusionFunctionContext(
        frame=frame,
        model_ids=model_ids,
        geometry=geometry,
        seed=seed,
        primary_model=str(primary_model),
    )
    function_calls = [_execute_fusion_function(ctx, function) for function in registry]
    method_preds = ctx.method_preds
    method_metadata = ctx.method_metadata

    metrics_rows = []
    out = pd.DataFrame(index=frame.index.copy())
    out["cell_id"] = out.index.astype(str)
    out["true_shared"] = y_true
    out["true_raw_mode"] = frame["true_raw_mode"].astype(str).to_numpy()
    for model_id in model_ids:
        out[f"{model_id}__pred_shared"] = frame[f"{model_id}__pred_shared"].astype(str).to_numpy()
        out[f"{model_id}__confidence"] = frame[f"{model_id}__confidence"].astype(float).to_numpy()
    for method, pred in method_preds.items():
        out[f"{method}__pred_shared"] = pred
        row = {
            "method": method,
            "method_kind": "single_model" if method.startswith("single__") else "fusion",
            **method_metadata.get(method, {}),
        }
        if compute_truth_metrics and y_true is not None:
            row.update(_compute_metrics(y_true, pred))
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    if not metrics.empty:
        sort_columns = [column for column in ["macro_f1", "accuracy", "method"] if column in metrics.columns]
        sort_map = {"macro_f1": False, "accuracy": False, "method": True}
        sort_ascending = [sort_map[column] for column in sort_columns]
        metrics = metrics.sort_values(sort_columns, ascending=sort_ascending).reset_index(drop=True)
    method_metadata["_function_calls"] = function_calls
    method_metadata["_function_registry"] = [
        {
            "function_name": function.name,
            "description": function.description,
            "method_kind": function.method_kind,
            "required_inputs": list(function.required_inputs),
            "default_parameters": function.default_parameters,
        }
        for function in registry
    ]
    return out.reset_index(drop=True), metrics, method_metadata


def _model_agreement(frame: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, row in frame.iterrows():
        preds = [str(row[f"{model_id}__pred_shared"]) for model_id in model_ids]
        known = [pred for pred in preds if pred != UNKNOWN_LABEL]
        counts = Counter(known)
        top_count = counts.most_common(1)[0][1] if counts else 0
        rows.append(
            {
                "cell_id": str(cell_id),
                "true_shared": str(row["true_shared"]),
                "n_models": len(model_ids),
                "n_non_unknown": len(known),
                "top_vote_count": int(top_count),
                "agreement_fraction": float(top_count / max(1, len(known))),
                "unique_known_predictions": int(len(counts)),
                "predictions": json.dumps({model_id: str(row[f"{model_id}__pred_shared"]) for model_id in model_ids}, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _single_model_metrics(frame: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    y_true = frame["true_shared"].to_numpy(dtype=object)
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        pred_col = f"{model_id}__pred_shared"
        if pred_col not in frame.columns:
            continue
        rows.append(
            {
                "method": f"single__{model_id}",
                "method_kind": "single_model",
                **_compute_metrics(y_true, frame[pred_col].astype(str).to_numpy(dtype=object)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["macro_f1", "accuracy", "method"], ascending=[False, False, True]).reset_index(drop=True)


def _write_markdown_summary(summary: dict[str, Any], metrics: pd.DataFrame, path: Path) -> None:
    lines = [
        f"# Stage-4 Consensus: {summary['dataset_id']}",
        "",
        f"- mode: `{summary['mode']}`",
        f"- primary output method: `{summary['primary_method']}`",
        f"- best evaluated method: `{summary['best_method']}`",
        f"- best fusion method: `{summary.get('best_fusion_method', '')}`",
        f"- best single model method: `{summary.get('best_single_method', '')}`",
        f"- best fusion delta vs best single macro-F1: `{summary.get('best_fusion_vs_best_single_macro_f1_delta', 0.0):.4f}`",
        f"- model scope: `{summary.get('model_scope', 'selected')}`",
        f"- target models: `{len(summary.get('target_model_ids', summary.get('normalized_models', [])))}`",
        f"- completed models: `{len(summary['completed_models'])}`",
        f"- fusion methods: `{len(summary['fusion_methods'])}`",
        f"- ready_for_report: `{summary['ready_for_report']}`",
        "",
        "## Metrics",
        "",
        "| method | kind | accuracy | macro_f1 | weighted_f1 | shared_accuracy | accepted_fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['method']} | {row.get('method_kind', '')} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['weighted_f1']:.4f} | {row['shared_accuracy']:.4f} | {row['accepted_fraction']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_consensus_report(summary: dict[str, Any], metrics: pd.DataFrame, geometry_summary: pd.DataFrame, path: Path) -> None:
    best = metrics.iloc[0].to_dict() if not metrics.empty else {}
    geometry_available = (
        geometry_summary[geometry_summary["status"] == "available"]["model_id"].astype(str).tolist()
        if not geometry_summary.empty and "status" in geometry_summary.columns
        else []
    )
    skipped = (
        geometry_summary[geometry_summary["status"] != "available"][["model_id", "reason"]].to_dict("records")
        if not geometry_summary.empty and {"status", "reason"}.issubset(geometry_summary.columns)
        else []
    )
    lines = [
        f"# Consensus Report: {summary['dataset_id']}",
        "",
        "This report summarizes label-free consensus fusion over completed stage-3 model outputs. Query labels were used only for final evaluation.",
        "",
        "## Result",
        "",
        f"- Primary policy method: `{summary['primary_method']}`",
        f"- Best evaluated method: `{summary['best_method']}`",
        f"- Best fusion method: `{summary.get('best_fusion_method', '')}`",
        f"- Best single model method: `{summary.get('best_single_method', '')}`",
        f"- Best fusion delta vs best single macro-F1: `{summary.get('best_fusion_vs_best_single_macro_f1_delta', 0.0):.4f}`",
        f"- Best macro-F1: `{best.get('macro_f1', 0.0):.4f}`",
        f"- Best accuracy: `{best.get('accuracy', 0.0):.4f}`",
        "",
        "## Reference Geometry",
        "",
        f"- Geometry models used: `{', '.join(geometry_available) if geometry_available else 'none'}`",
    ]
    if skipped:
        lines.append("- Geometry skips are recorded in `reference_geometry_summary.csv`.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Consensus predictions: `{summary['prediction_artifacts']['consensus_predictions']}`",
            f"- Metrics: `{summary['prediction_artifacts']['consensus_metrics']}`",
            f"- Model agreement: `{summary['prediction_artifacts']['model_agreement']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_consensus(
    *,
    stage3_summary_path: str | Path,
    mode: str = "subset",
    output_dir: str | Path | None = None,
    seed: int = 3028,
    llm_policy_mode: str = "off",
    llm_model: str | None = None,
    llm_cell_adjudication_mode: str = "off",
    llm_cell_max_groups: int = 120,
    llm_cell_batch_size: int = 40,
    model_scope: str = "selected",
    skip_reference_geometry: bool = False,
    execution_strategy: str = "selected_only",
    max_probe_cells: int = 5000,
) -> dict[str, Any]:
    stage3_summary_path = Path(stage3_summary_path)
    stage3_summary = read_json(stage3_summary_path)
    dataset_id = str(stage3_summary["dataset_id"])
    out_dir = ensure_dir(output_dir or (DEFAULT_STAGE4_ROOT / dataset_id / mode))
    label_maps = load_seaad_label_maps()
    selected_model_ids = _unique_preserve_order(list(stage3_summary.get("selected_model_ids", []) or []))
    completed_model_ids = _unique_preserve_order(list(stage3_summary.get("completed_models", []) or []))
    target_model_ids = _stage3_model_scope(stage3_summary, model_scope=model_scope)

    normalized, normalization_meta = normalize_stage3_predictions(stage3_summary, model_ids=target_model_ids)
    normalized_path = out_dir / "normalized_predictions.csv"
    normalized.to_csv(normalized_path, index=False)
    frame, model_ids = build_consensus_frame(normalized)
    if frame.empty or len(model_ids) < 2:
        summary = {
            "dataset_id": dataset_id,
            "mode": mode,
            "stage3_summary": str(stage3_summary_path),
            "model_scope": model_scope,
            "selected_model_ids": selected_model_ids,
            "completed_models": completed_model_ids,
            "target_model_ids": target_model_ids,
            "fusion_methods": [],
            "best_method": "",
            "primary_method": "",
            "ready_for_report": False,
            "reference_sources_used": [],
            "skipped_geometry_models": normalization_meta.get("normalization_skips", []),
            "prediction_artifacts": {"normalized_predictions": str(normalized_path)},
            "reason": "fewer_than_two_normalized_models_or_no_aligned_cells",
        }
        write_json(summary, out_dir / "fusion_summary.json")
        return summary

    if skip_reference_geometry:
        geometry: dict[str, dict[str, np.ndarray]] = {}
        geometry_summary = pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "status": "reference_geometry_skipped",
                    "reason": "skip_reference_geometry_requested",
                    "n_query_cells": int(frame.shape[0]),
                }
                for model_id in model_ids
            ]
        )
    else:
        geometry, geometry_summary = build_reference_geometry(
            stage3_summary=stage3_summary,
            frame=frame,
            normalized=normalized,
            label_maps=label_maps,
            model_ids=model_ids,
        )
    primary_model = next((model_id for model_id in selected_model_ids if model_id in model_ids), model_ids[0])
    selection_context = {
        "stage2_selection": _stage2_selection_context(stage3_summary),
        "stage3_model_evidence": _stage3_model_evidence(frame, model_ids, primary_model=primary_model),
        "reference_geometry_summary": geometry_summary.to_dict("records"),
        "execution_strategy": execution_strategy,
    }

    if execution_strategy == "benchmark_all":
        consensus_predictions, consensus_metrics, method_metadata = run_fusion_methods(
            frame=frame,
            model_ids=model_ids,
            geometry=geometry,
            seed=seed,
            primary_model=primary_model,
        )
        preliminary_diagnostics = label_free_method_diagnostics(
            consensus_predictions=consensus_predictions,
            model_ids=model_ids,
            method_names=consensus_metrics["method"].astype(str).tolist(),
            unknown_label=UNKNOWN_LABEL,
            primary_model=primary_model,
            method_metadata=method_metadata,
        )
        mean_global_agreement = (
            float(preliminary_diagnostics[0].get("mean_global_model_agreement", 0.0))
            if preliminary_diagnostics
            else 0.0
        )
        llm_cell_adjudication_meta: dict[str, Any] = {
            "status": "skipped",
            "reason": "llm_cell_adjudication_off_or_benchmark_mode",
            "mean_global_model_agreement": mean_global_agreement,
            "high_global_model_agreement_threshold": HIGH_CONSENSUS_MODEL_AGREEMENT,
        }
        if llm_cell_adjudication_mode != "off":
            base_method = "stage2_primary_guarded_consensus"
            if f"{base_method}__pred_shared" not in consensus_predictions.columns:
                base_method = str(consensus_metrics.iloc[0]["method"]) if not consensus_metrics.empty else "majority_vote"
            candidate_method_priority = [
                "stage2_primary_guarded_consensus",
                "majority_vote",
                "confidence_weighted_vote",
                "agreement_then_confidence",
                "weighted_consensus_fixed",
                "query_graph_refined_consensus",
                "neighbor_distribution_consensus",
                "density_calibrated_neighbor_distribution_consensus",
                "reference_enhanced_graph_consensus",
                "reference_logistic_consensus",
            ]
            available_method_names = set(consensus_metrics["method"].astype(str).tolist())
            candidate_methods = [method for method in candidate_method_priority if method in available_method_names]
            llm_pred, llm_cell_adjudication_meta = adjudicate_low_consistency_cells(
                consensus_predictions=consensus_predictions,
                model_ids=model_ids,
                candidate_method_names=candidate_methods,
                base_method=base_method,
                allowed_labels=CONSENSUS_LABELS,
                unknown_label=UNKNOWN_LABEL,
                output_dir=out_dir,
                llm_mode=llm_cell_adjudication_mode,
                llm_model=llm_model,
                max_groups=llm_cell_max_groups,
                batch_size=llm_cell_batch_size,
            )
            llm_cell_adjudication_meta = {
                **llm_cell_adjudication_meta,
                "mean_global_model_agreement": mean_global_agreement,
                "high_global_model_agreement_threshold": HIGH_CONSENSUS_MODEL_AGREEMENT,
            }
            if llm_pred is not None:
                consensus_predictions[f"{LLM_ADJUDICATION_METHOD}__pred_shared"] = llm_pred
                llm_row = {
                    "method": LLM_ADJUDICATION_METHOD,
                    "method_kind": "fusion",
                    **_compute_metrics(frame["true_shared"].to_numpy(dtype=object), llm_pred),
                    **llm_cell_adjudication_meta,
                }
                consensus_metrics = (
                    pd.concat([consensus_metrics, pd.DataFrame([llm_row])], ignore_index=True)
                    .sort_values(["macro_f1", "accuracy", "method"], ascending=[False, False, True])
                    .reset_index(drop=True)
                )
                method_metadata[LLM_ADJUDICATION_METHOD] = llm_cell_adjudication_meta
        diagnostics = label_free_method_diagnostics(
            consensus_predictions=consensus_predictions,
            model_ids=model_ids,
            method_names=consensus_metrics["method"].astype(str).tolist(),
            unknown_label=UNKNOWN_LABEL,
            primary_model=primary_model,
            method_metadata=method_metadata,
        )
        primary_method, policy_review = select_policy_method(
            diagnostics=diagnostics,
            available_methods=consensus_metrics["method"].astype(str).tolist(),
            output_dir=out_dir,
            llm_mode=llm_policy_mode,
            llm_model=llm_model,
            function_registry=method_metadata.get("_function_registry", []),
            selection_context=selection_context,
        )
        best_method = str(consensus_metrics.iloc[0]["method"]) if not consensus_metrics.empty else ""
        single_metrics = consensus_metrics[consensus_metrics.get("method_kind", pd.Series(dtype=str)).astype(str) == "single_model"]
        fusion_metrics = consensus_metrics[consensus_metrics.get("method_kind", pd.Series(dtype=str)).astype(str) != "single_model"]
        best_single_method = str(single_metrics.iloc[0]["method"]) if not single_metrics.empty else ""
        best_fusion_method = str(fusion_metrics.iloc[0]["method"]) if not fusion_metrics.empty else ""
        best_single_macro = float(single_metrics.iloc[0]["macro_f1"]) if not single_metrics.empty else 0.0
        best_fusion_macro = float(fusion_metrics.iloc[0]["macro_f1"]) if not fusion_metrics.empty else 0.0
        if primary_method not in set(consensus_metrics["method"].astype(str)):
            primary_method = best_fusion_method or best_method
        consensus_predictions["primary_method"] = primary_method
        consensus_predictions["final_pred_shared"] = consensus_predictions[f"{primary_method}__pred_shared"]
        probe_predictions = pd.DataFrame()
        probe_diagnostics = diagnostics
        probe_method_metadata: dict[str, Any] = {}
        probe_function_calls: list[dict[str, Any]] = []
        probe_function_registry: list[dict[str, Any]] = []
    else:
        probe_frame = _sample_probe_frame(
            frame,
            model_ids,
            primary_model=primary_model,
            max_probe_cells=max_probe_cells,
            seed=seed,
        )
        probe_geometry = _subset_geometry_for_frame(
            geometry,
            full_frame=frame,
            subset_frame=probe_frame,
        )
        probe_predictions, probe_metrics, probe_method_metadata = run_fusion_methods(
            frame=probe_frame,
            model_ids=model_ids,
            geometry=probe_geometry,
            seed=seed,
            primary_model=primary_model,
            compute_truth_metrics=False,
        )
        probe_method_names = probe_metrics["method"].astype(str).tolist() if not probe_metrics.empty else []
        probe_diagnostics = label_free_method_diagnostics(
            consensus_predictions=probe_predictions,
            model_ids=model_ids,
            method_names=probe_method_names,
            unknown_label=UNKNOWN_LABEL,
            primary_model=primary_model,
            method_metadata=probe_method_metadata,
        )
        probe_function_calls = list(probe_method_metadata.get("_function_calls", []))
        probe_function_registry = list(probe_method_metadata.get("_function_registry", []))
        selection_context["probe"] = {
            "n_probe_cells": int(probe_frame.shape[0]),
            "sampling_strategy": "stage2_primary_prediction x known_vote_count x top_vote_count stratified subset",
            "available_probe_methods": probe_method_names,
        }
        primary_method, policy_review = select_policy_method(
            diagnostics=probe_diagnostics,
            available_methods=probe_method_names,
            output_dir=out_dir,
            llm_mode=llm_policy_mode,
            llm_model=llm_model,
            function_registry=probe_function_registry,
            selection_context=selection_context,
        )
        consensus_predictions, fusion_only_metrics, method_metadata = run_fusion_methods(
            frame=frame,
            model_ids=model_ids,
            geometry=geometry,
            seed=seed,
            primary_model=primary_model,
            compute_truth_metrics=True,
            selected_methods=[primary_method],
        )
        if f"{primary_method}__pred_shared" not in consensus_predictions.columns:
            raise RuntimeError(f"selected stage4 method did not execute on full frame: {primary_method}")
        consensus_predictions["primary_method"] = primary_method
        consensus_predictions["final_pred_shared"] = consensus_predictions[f"{primary_method}__pred_shared"]
        single_metrics = _single_model_metrics(frame, model_ids)
        consensus_metrics = pd.concat([fusion_only_metrics, single_metrics], ignore_index=True)
        if not consensus_metrics.empty:
            consensus_metrics = consensus_metrics.sort_values(["macro_f1", "accuracy", "method"], ascending=[False, False, True]).reset_index(drop=True)
        best_method = primary_method
        best_single_method = str(single_metrics.iloc[0]["method"]) if not single_metrics.empty else ""
        best_fusion_method = primary_method
        best_single_macro = float(single_metrics.iloc[0]["macro_f1"]) if not single_metrics.empty else 0.0
        fusion_row = consensus_metrics[consensus_metrics["method"].astype(str) == primary_method]
        best_fusion_macro = float(fusion_row.iloc[0]["macro_f1"]) if not fusion_row.empty else 0.0
        llm_cell_adjudication_meta = {
            "status": "skipped",
            "reason": "selected_only_execution_does_not_run_cell_level_post_hoc_adjudication",
        }

    agreement = _model_agreement(frame, model_ids)
    disagreement = consensus_predictions[
        (consensus_predictions["final_pred_shared"].astype(str) != consensus_predictions["true_shared"].astype(str))
        | (agreement["unique_known_predictions"].to_numpy() > 1)
    ].copy()
    disagreement["agreement_fraction"] = agreement.loc[disagreement.index, "agreement_fraction"].to_numpy()

    paths_out = {
        "normalized_predictions": str(normalized_path),
        "consensus_predictions": str(out_dir / "consensus_predictions.csv"),
        "consensus_metrics": str(out_dir / "consensus_metrics.csv"),
        "model_agreement": str(out_dir / "model_agreement.csv"),
        "label_disagreement_cases": str(out_dir / "label_disagreement_cases.csv"),
        "reference_geometry_summary": str(out_dir / "reference_geometry_summary.csv"),
        "method_diagnostics": str(out_dir / "method_diagnostics.csv"),
        "fusion_function_calls": str(out_dir / "fusion_function_calls.json"),
        "fusion_function_calls_csv": str(out_dir / "fusion_function_calls.csv"),
        "probe_method_diagnostics": str(out_dir / "probe_method_diagnostics.csv"),
        "probe_function_calls": str(out_dir / "probe_function_calls.json"),
        "probe_function_calls_csv": str(out_dir / "probe_function_calls.csv"),
        "fusion_summary": str(out_dir / "fusion_summary.json"),
        "fusion_summary_md": str(out_dir / "fusion_summary.md"),
        "consensus_report": str(out_dir / "consensus_report.md"),
    }
    consensus_predictions.to_csv(paths_out["consensus_predictions"], index=False)
    consensus_metrics.to_csv(paths_out["consensus_metrics"], index=False)
    agreement.to_csv(paths_out["model_agreement"], index=False)
    disagreement.to_csv(paths_out["label_disagreement_cases"], index=False)
    geometry_summary.to_csv(paths_out["reference_geometry_summary"], index=False)
    pd.DataFrame(probe_diagnostics).to_csv(paths_out["method_diagnostics"], index=False)
    pd.DataFrame(probe_diagnostics).to_csv(paths_out["probe_method_diagnostics"], index=False)
    function_call_payload = {
        "schema_version": "scmas.stage4.fusion_function_calls.v1",
        "dataset_id": dataset_id,
        "mode": mode,
        "execution_strategy": execution_strategy,
        "policy": {
            "query_labels_available_to_function_selector": False,
            "metrics_with_query_truth_available_to_function_selector": False,
            "function_parameters_are_fixed_by_registry": True,
        },
        "function_registry": method_metadata.get("_function_registry", []),
        "function_calls": method_metadata.get("_function_calls", []),
    }
    write_json(function_call_payload, paths_out["fusion_function_calls"])
    pd.DataFrame(method_metadata.get("_function_calls", [])).to_csv(paths_out["fusion_function_calls_csv"], index=False)
    write_json(
        {
            "schema_version": "scmas.stage4.probe_function_calls.v1",
            "dataset_id": dataset_id,
            "mode": mode,
            "execution_strategy": execution_strategy,
            "function_registry": probe_function_registry,
            "function_calls": probe_function_calls,
        },
        paths_out["probe_function_calls"],
    )
    pd.DataFrame(probe_function_calls).to_csv(paths_out["probe_function_calls_csv"], index=False)

    reference_sources = sorted(
        geometry_summary.loc[geometry_summary.get("status", pd.Series(dtype=str)) == "available", "reference_path"].dropna().astype(str).unique()
        if not geometry_summary.empty and "reference_path" in geometry_summary.columns
        else []
    )
    skipped_geometry = (
        geometry_summary[geometry_summary["status"] != "available"].to_dict("records")
        if not geometry_summary.empty and "status" in geometry_summary.columns
        else []
    )
    summary = {
        "dataset_id": dataset_id,
        "mode": mode,
        "stage3_summary": str(stage3_summary_path),
        "model_scope": model_scope,
        "execution_strategy": execution_strategy,
        "selected_model_ids": selected_model_ids,
        "completed_models": completed_model_ids,
        "target_model_ids": target_model_ids,
        "normalized_models": model_ids,
        "fusion_methods": consensus_metrics["method"].astype(str).tolist(),
        "best_method": best_method,
        "best_single_method": best_single_method,
        "best_fusion_method": best_fusion_method,
        "best_single_macro_f1": best_single_macro,
        "best_fusion_macro_f1": best_fusion_macro,
        "best_fusion_vs_best_single_macro_f1_delta": float(best_fusion_macro - best_single_macro),
        "primary_method": primary_method,
        "policy_review": policy_review,
        "llm_cell_adjudication": llm_cell_adjudication_meta,
        "skip_reference_geometry": bool(skip_reference_geometry),
        "ready_for_report": True,
        "n_cells_aligned": int(frame.shape[0]),
        "model_weights": normalization_meta.get("model_weights", {}),
        "model_tasks": normalization_meta.get("model_tasks", {}),
        "selection_context": selection_context,
        "method_metadata": method_metadata,
        "fusion_function_registry": method_metadata.get("_function_registry", []),
        "fusion_function_calls": method_metadata.get("_function_calls", []),
        "probe_function_registry": probe_function_registry,
        "probe_function_calls": probe_function_calls,
        "reference_sources_used": reference_sources,
        "skipped_geometry_models": skipped_geometry,
        "prediction_artifacts": paths_out,
    }
    write_json(summary, paths_out["fusion_summary"])
    _write_markdown_summary(summary, consensus_metrics, Path(paths_out["fusion_summary_md"]))
    _write_consensus_report(summary, consensus_metrics, geometry_summary, Path(paths_out["consensus_report"]))
    return summary
