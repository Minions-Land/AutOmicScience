from __future__ import annotations

import ast
import json
import os
import re
import shutil
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from novaeve_agent import paths
from novaeve_agent.eval.registry import ModelSpec, artifact_exists, load_model_registry, resolve_portable_path
from novaeve_agent.io import ensure_dir, read_yaml, write_json, write_yaml
from novaeve_agent.llm_config import build_openai_client, default_llm_model
from novaeve_agent.stage2.selector import run_cross_species_plan


DEFAULT_CAPABILITY_DIR = paths.SCMAS_ROOT / "configs" / "capability"
DEFAULT_REGISTRY_PATH = paths.SCMAS_ROOT / "configs" / "model_registry.yaml"
DEFAULT_STAGE3_ROOT = paths.RUNS_DIR / "stage3_adapter_execution"

SEAAD_140_CONTRACT = "seaad_140_npz"
RAW_LABEL_TRANSFER_CONTRACT = "source_gene_panel_label_transfer"
RAW_LABEL_TRANSFER_EVALUATOR = "raw_label_transfer"

ALLOWED_ACTIONS = {
    "load_query_npz_kukanja",
    "load_query_h5ad",
    "load_reference_standard_bundle",
    "align_shared_genes",
    "write_raw_label_transfer_input",
    "invoke_raw_embedding_transfer",
    "invoke_postprocessor",
    "skip_with_reason",
}
DEFAULT_LLM_MODEL = default_llm_model()
DEFAULT_ENV_PATH = paths.SCMAS_ROOT / ".env"


@dataclass
class ModelContract:
    model_id: str
    family: str
    evaluator: str
    capability_yaml: str
    registry_present: bool
    required_formats: list[str]
    required_fields: list[str]
    gene_contract: str
    compatible_contract: str
    runtime_args: dict[str, Any]
    artifacts: dict[str, Any]
    wrapper_signature: dict[str, Any]
    artifact_status: dict[str, Any]


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


def _load_capabilities(capability_dir: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(capability_dir).glob("*.yaml")):
        if path.name == "index.yaml":
            continue
        data = read_yaml(path) or {}
        model_id = str(data.get("model_id", path.stem))
        data["_capability_yaml"] = str(path)
        out[model_id] = data
    return out


def _registry_map(registry_path: str | Path) -> dict[str, ModelSpec]:
    if not Path(registry_path).exists():
        return {}
    return {model.model_id: model for model in load_model_registry(registry_path)}


def _wrapper_path_from_registry(model: ModelSpec | None) -> Path | None:
    if model is None:
        return None
    module = model.raw.get("tool_module")
    if not module:
        return None
    return paths.LEGACY_ROOT / "tools_layer" / "mcp_tools" / f"{module}.py"


def _wrapper_path_from_capability(capability: dict[str, Any]) -> Path | None:
    artifacts = capability.get("artifacts", {}) if isinstance(capability, dict) else {}
    wrapper = artifacts.get("wrapper", {}) if isinstance(artifacts, dict) else {}
    wrapper_path = wrapper.get("path") if isinstance(wrapper, dict) else None
    return resolve_portable_path(wrapper_path) if wrapper_path else None


def _literal_default(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else "<expr>"


def _extract_wrapper_signature(wrapper_path: Path | None, function_name: str | None = None) -> dict[str, Any]:
    if wrapper_path is None:
        return {}
    if not wrapper_path.exists():
        return {"path": str(wrapper_path), "status": "missing"}
    try:
        tree = ast.parse(wrapper_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(wrapper_path), "status": "parse_failed", "reason": f"{type(exc).__name__}: {exc}"}

    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    selected = None
    if function_name:
        selected = next((node for node in functions if node.name == function_name), None)
    if selected is None and functions:
        selected = next((node for node in functions if node.name.endswith("_tool")), functions[0])
    if selected is None:
        return {"path": str(wrapper_path), "status": "no_function_found"}

    positional = list(selected.args.args)
    defaults = [None] * (len(positional) - len(selected.args.defaults)) + list(selected.args.defaults)
    params = []
    for arg, default in zip(positional, defaults):
        params.append(
            {
                "name": arg.arg,
                "kind": "positional_or_keyword",
                "default": _literal_default(default),
            }
        )
    for arg, default in zip(selected.args.kwonlyargs, selected.args.kw_defaults):
        params.append({"name": arg.arg, "kind": "keyword_only", "default": _literal_default(default)})
    return {
        "path": str(wrapper_path),
        "status": "ok",
        "function": selected.name,
        "parameters": params,
    }


def _artifact_status(model: ModelSpec | None, artifacts: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if model is not None:
        ok, reason = artifact_exists(model)
        checks.append({"source": "registry", "ok": bool(ok), "reason": reason})
    for name, value in artifacts.items():
        if not isinstance(value, dict) or not value.get("path"):
            continue
        path = resolve_portable_path(value["path"])
        checks.append({"source": f"capability.{name}", "path": str(path), "ok": path.exists()})
    return {"ok": all(item.get("ok", False) for item in checks) if checks else True, "checks": checks}


def _contract_for_model(
    model_id: str,
    capability: dict[str, Any] | None,
    registry_model: ModelSpec | None,
) -> ModelContract:
    capability = capability or {}
    input_req = capability.get("input_requirements", {}) if isinstance(capability, dict) else {}
    constraints = capability.get("data_constraints", {}) if isinstance(capability, dict) else {}
    artifacts = capability.get("artifacts", {}) if isinstance(capability, dict) else {}
    evaluator = str(capability.get("evaluator") or (registry_model.evaluator if registry_model else "unknown"))
    family = str(capability.get("family") or (registry_model.family if registry_model else "unknown"))
    wrapper_path = _wrapper_path_from_capability(capability) or _wrapper_path_from_registry(registry_model)
    function_name = registry_model.raw.get("tool_function") if registry_model is not None else None
    return ModelContract(
        model_id=model_id,
        family=family,
        evaluator=evaluator,
        capability_yaml=str(capability.get("_capability_yaml", "")),
        registry_present=registry_model is not None,
        required_formats=[str(x) for x in input_req.get("required_formats", [])],
        required_fields=[str(x) for x in input_req.get("required_fields", [])],
        gene_contract=str(input_req.get("gene_contract", "")),
        compatible_contract=str(constraints.get("compatible_contract") or (registry_model.raw.get("compatible_contract") if registry_model else "")),
        runtime_args=dict(registry_model.raw) if registry_model is not None else dict(capability.get("executor_defaults", {}) or {}),
        artifacts=artifacts if isinstance(artifacts, dict) else {},
        wrapper_signature=_extract_wrapper_signature(wrapper_path, function_name),
        artifact_status=_artifact_status(registry_model, artifacts if isinstance(artifacts, dict) else {}),
    )


def inspect_model_contracts(
    *,
    capability_dir: str | Path = DEFAULT_CAPABILITY_DIR,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect capability YAML, registry entries, artifacts, and wrapper signatures."""
    capabilities = _load_capabilities(capability_dir)
    registry = _registry_map(registry_path)
    model_ids = sorted(set(capabilities) | set(registry))
    contracts = [
        asdict(_contract_for_model(model_id, capabilities.get(model_id), registry.get(model_id)))
        for model_id in model_ids
    ]
    result = {
        "capability_dir": str(capability_dir),
        "registry_path": str(registry_path),
        "n_models": len(contracts),
        "contracts": contracts,
    }
    if output_dir:
        out_dir = ensure_dir(output_dir)
        write_json(result, out_dir / "model_contracts.json")
        flat_rows = []
        for row in contracts:
            flat_rows.append(
                {
                    "model_id": row["model_id"],
                    "family": row["family"],
                    "evaluator": row["evaluator"],
                    "compatible_contract": row["compatible_contract"],
                    "required_formats": json.dumps(row["required_formats"]),
                    "required_fields": json.dumps(row["required_fields"]),
                    "artifact_ok": row["artifact_status"].get("ok", True),
                    "wrapper_status": row["wrapper_signature"].get("status", ""),
                    "capability_yaml": row["capability_yaml"],
                }
            )
        pd.DataFrame(flat_rows).to_csv(out_dir / "model_contracts.csv", index=False)
        lines = [
            "# Model Contracts",
            "",
            "| model_id | evaluator | compatible_contract | artifact_ok | wrapper_status |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for row in flat_rows:
            lines.append(
                f"| {row['model_id']} | {row['evaluator']} | {row['compatible_contract']} | "
                f"{row['artifact_ok']} | {row['wrapper_status']} |"
            )
        (out_dir / "model_contracts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["output_dir"] = str(out_dir)
        result["contracts_json"] = str(out_dir / "model_contracts.json")
        result["contracts_csv"] = str(out_dir / "model_contracts.csv")
        result["contracts_md"] = str(out_dir / "model_contracts.md")
    return result


def _query_load_action(plan: dict[str, Any]) -> str:
    adapter = str(plan.get("query_adapter", ""))
    query_path = Path(str(plan.get("query_path", "")))
    if adapter == "npz_kukanja" or query_path.suffix == ".npz":
        return "load_query_npz_kukanja"
    return "load_query_h5ad"


def _execution_defaults(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    defaults = dict(plan.get("execution_defaults", {}) or {})
    defaults.setdefault("max_query_cells", 5000)
    defaults.setdefault("max_reference_cells", 1000)
    defaults.setdefault("min_shared_genes", 30)
    defaults.setdefault("k", 15)
    defaults.setdefault("seed", 3028)
    defaults.setdefault("device", "")
    defaults.setdefault("batch_size", 16)
    defaults.setdefault("num_workers", 0)
    if mode == "full":
        # `run_cross_species_plan` treats 0 as "use plan default" for CLI compatibility.
        # Use -1 as an explicit full-query sentinel when stage3 is driving execution.
        defaults["max_query_cells"] = -1
        if int(defaults.get("batch_size", 0) or 0) <= 16:
            defaults["batch_size"] = 64
    return defaults


def _base_spec(
    *,
    model_id: str,
    source_id: str,
    contract: ModelContract,
    plan: dict[str, Any],
    output_dir: Path,
    mode: str,
) -> dict[str, Any]:
    model_run_dir = output_dir / "model_runs" / model_id
    return {
        "schema_version": "scmas.adapter_spec.v1",
        "react_policy": "deterministic_react_v1",
        "mode": mode,
        "dataset_id": plan["dataset_id"],
        "query_path": str(plan["query_path"]),
        "query_adapter": str(plan.get("query_adapter", "")),
        "model_id": model_id,
        "source_id": source_id,
        "capability_yaml": contract.capability_yaml,
        "model_contract": asdict(contract),
        "input_artifacts": {
            "query_path": str(plan["query_path"]),
            "capability_yaml": contract.capability_yaml,
        },
        "gene_strategy": {},
        "label_strategy": {},
        "runtime_payload": dict(_execution_defaults(plan, mode)),
        "actions": [],
        "expected_outputs": {
            "model_run_dir": str(model_run_dir),
            "predictions_csv": str(model_run_dir / "predictions.csv"),
            "metrics_csv": str(model_run_dir / "metrics.csv"),
            "run_summary_json": str(model_run_dir / "run_summary.json"),
        },
        "observe": {
            "dataset_id": plan["dataset_id"],
            "selected_model_ids": list(plan.get("selected_model_ids", [])),
            "query_profile_path": str(plan.get("query_profile_path", "")),
            "species_is_hard_filter": False,
        },
        "thought": "",
        "review": {},
    }


def _raw_label_transfer_spec(
    *,
    pair: dict[str, Any],
    contract: ModelContract,
    plan: dict[str, Any],
    output_dir: Path,
    mode: str,
) -> dict[str, Any]:
    spec = _base_spec(
        model_id=pair["model_id"],
        source_id=pair["source_id"],
        contract=contract,
        plan=plan,
        output_dir=output_dir,
        mode=mode,
    )
    input_yaml = output_dir / "prepared_inputs" / pair["model_id"] / "raw_label_transfer_input.yaml"
    spec["input_artifacts"].update(
        {
            "reference_path": str(pair["reference_path"]),
            "stage2_selected_pair": {
                "rank": pair.get("rank"),
                "score": pair.get("score"),
                "shared_genes": pair.get("shared_genes"),
            },
        }
    )
    spec["gene_strategy"] = {
        "strategy": "align_shared_genes",
        "selected_shared_genes": int(pair.get("shared_genes", 0)),
        "min_shared_genes": int(spec["runtime_payload"]["min_shared_genes"]),
        "species_is_filter": False,
    }
    spec["label_strategy"] = {
        "tasks": ["native_label", "coarse_label"],
        "reference_label_source": "selected reference standard bundle",
        "query_truth_source": "query obs labels when available",
        "on_label_mismatch": "score overlap explicitly; do not skip for species",
    }
    spec["runtime_payload"].update(
        {
            "method": pair["method"],
            "embedding_method": pair["embedding_method"],
            "transfer_method": pair["transfer_method"],
            "reference_path": str(pair["reference_path"]),
            "one_pair_plan_path": str(input_yaml.with_name("selected_pair_plan.yaml")),
        }
    )
    spec["actions"] = [
        {"action": _query_load_action(plan), "path": str(plan["query_path"])},
        {"action": "load_reference_standard_bundle", "path": str(pair["reference_path"])},
        {
            "action": "align_shared_genes",
            "min_shared_genes": int(spec["runtime_payload"]["min_shared_genes"]),
            "selected_shared_genes": int(pair.get("shared_genes", 0)),
        },
        {"action": "write_raw_label_transfer_input", "path": str(input_yaml)},
        {"action": "invoke_raw_embedding_transfer", "executor": "scmas.stage2.selector.run_cross_species_plan"},
    ]
    spec["thought"] = (
        "Use the stage-2 selected source+model pair and adapt the query by shared genes. "
        "Species is only metadata; the execution contract is shared genes plus labels."
    )
    return spec


def _skip_spec(
    *,
    model_id: str,
    source_id: str,
    contract: ModelContract,
    plan: dict[str, Any],
    output_dir: Path,
    mode: str,
    reason: str,
) -> dict[str, Any]:
    spec = _base_spec(
        model_id=model_id,
        source_id=source_id,
        contract=contract,
        plan=plan,
        output_dir=output_dir,
        mode=mode,
    )
    spec["runtime_payload"].update({"skip_reason": reason})
    spec["gene_strategy"] = {"strategy": "not_applicable_or_not_bound", "species_is_filter": False}
    spec["label_strategy"] = {"strategy": "not_applicable_or_not_bound"}
    spec["actions"] = [{"action": "skip_with_reason", "reason": reason}]
    spec["thought"] = reason
    return spec


def _build_adapter_specs(
    *,
    plan: dict[str, Any],
    contracts: dict[str, ModelContract],
    output_dir: Path,
    mode: str,
) -> list[dict[str, Any]]:
    selected_by_model = {pair["model_id"]: pair for pair in plan.get("selected_pairs", [])}
    specs: list[dict[str, Any]] = []
    for model_id in sorted(contracts):
        contract = contracts[model_id]
        if model_id in selected_by_model and contract.evaluator == RAW_LABEL_TRANSFER_EVALUATOR:
            specs.append(
                _raw_label_transfer_spec(
                    pair=selected_by_model[model_id],
                    contract=contract,
                    plan=plan,
                    output_dir=output_dir,
                    mode=mode,
                )
            )
            continue

        if contract.evaluator == RAW_LABEL_TRANSFER_EVALUATOR:
            specs.append(
                _skip_spec(
                    model_id=model_id,
                    source_id="unbound",
                    contract=contract,
                    plan=plan,
                    output_dir=output_dir,
                    mode=mode,
                    reason="not_selected_by_stage2_plan_no_bound_source_reference",
                )
            )
            continue

        if contract.compatible_contract == SEAAD_140_CONTRACT or contract.evaluator in {"mcp_tool", "sklearn_pkl", "spatial_gnn"}:
            specs.append(
                _skip_spec(
                    model_id=model_id,
                    source_id="disabled_direct_head",
                    contract=contract,
                    plan=plan,
                    output_dir=output_dir,
                    mode=mode,
                    reason="direct_seaad_140_head_disabled_use_raw_label_transfer_knn",
                )
            )
            continue

        if contract.evaluator == "scanvi_saved":
            reason = "scanvi_saved_requires_safe_h5ad_schema_and_saved_model_registry_match"
        elif contract.evaluator == "postprocessor":
            reason = "requires_shared_predictions"
        elif contract.evaluator == "raw_backbone":
            reason = "embedding_only_no_direct_annotation_head_use_raw_knn_adapter_instead"
        else:
            reason = f"unsupported_or_unbound_evaluator:{contract.evaluator}"
        specs.append(
            _skip_spec(
                model_id=model_id,
                source_id="unbound",
                contract=contract,
                plan=plan,
                output_dir=output_dir,
                mode=mode,
                reason=reason,
            )
        )
    return specs


def validate_adapter_spec(spec: dict[str, Any]) -> None:
    required = [
        "model_id",
        "source_id",
        "input_artifacts",
        "gene_strategy",
        "label_strategy",
        "runtime_payload",
        "actions",
        "expected_outputs",
    ]
    missing = [key for key in required if key not in spec or spec[key] in (None, "")]
    if missing:
        raise ValueError(f"AdapterSpec missing required fields: {', '.join(missing)}")
    if not isinstance(spec["actions"], list) or not spec["actions"]:
        raise ValueError("AdapterSpec actions must be a non-empty list")
    for action in spec["actions"]:
        if not isinstance(action, dict) or not action.get("action"):
            raise ValueError("AdapterSpec action entries must be dicts with an action field")
        action_name = str(action["action"])
        if action_name not in ALLOWED_ACTIONS:
            raise ValueError(f"Unknown adapter action: {action_name}")
        if action_name == "skip_with_reason" and not (action.get("reason") or spec["runtime_payload"].get("skip_reason")):
            raise ValueError("skip_with_reason requires a reason")
    dangerous_keys = {"cmd", "command", "code", "python", "shell", "subprocess"}
    serialized = json.dumps(spec, sort_keys=True, default=str).lower()
    for key in dangerous_keys:
        if f'"{key}"' in serialized:
            raise ValueError(f"AdapterSpec contains forbidden executable key: {key}")


def _llm_observe_for_adapter_spec(spec: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    contract = spec.get("model_contract", {}) if isinstance(spec.get("model_contract", {}), dict) else {}
    artifact_status = contract.get("artifact_status", {}) if isinstance(contract, dict) else {}
    wrapper_signature = contract.get("wrapper_signature", {}) if isinstance(contract, dict) else {}
    return {
        "schema_version": "scmas.stage3.llm_observe.v1",
        "dataset_id": spec.get("dataset_id"),
        "query_path": spec.get("query_path"),
        "query_adapter": spec.get("query_adapter"),
        "selected_model_ids": list(plan.get("selected_model_ids", [])),
        "model_id": spec.get("model_id"),
        "source_id": spec.get("source_id"),
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "immutable_fields": {
            "schema_version": spec.get("schema_version"),
            "mode": spec.get("mode"),
            "dataset_id": spec.get("dataset_id"),
            "query_path": spec.get("query_path"),
            "query_adapter": spec.get("query_adapter"),
            "model_id": spec.get("model_id"),
            "source_id": spec.get("source_id"),
            "capability_yaml": spec.get("capability_yaml"),
            "expected_outputs": spec.get("expected_outputs", {}),
        },
        "model_contract_summary": {
            "family": contract.get("family"),
            "evaluator": contract.get("evaluator"),
            "compatible_contract": contract.get("compatible_contract"),
            "required_formats": contract.get("required_formats", []),
            "required_fields": contract.get("required_fields", []),
            "gene_contract": contract.get("gene_contract", ""),
            "artifact_status": artifact_status,
            "wrapper_signature": wrapper_signature,
        },
        "stage2_pair": spec.get("input_artifacts", {}).get("stage2_selected_pair", {}),
        "deterministic_draft": spec,
    }


def _render_adapter_prompt(observe: dict[str, Any], previous: dict[str, Any] | None = None) -> tuple[str, str]:
    system_prompt = (
        "You are the scMAS stage-3 ReAct adapter agent. Produce one executable AdapterSpec JSON object. "
        "You may only choose from allowed_actions. You must not write code, shell commands, or arbitrary executable payloads. "
        "The deterministic reviewer will reject unknown actions, changed immutable fields, missing paths, or unsafe keys."
    )
    instructions = [
        "Return only JSON.",
        "Return the full AdapterSpec object, not a patch.",
        "Keep all immutable_fields exactly unchanged.",
        "Keep input_artifacts, runtime_payload defaults, and the action sequence from deterministic_draft unchanged unless the draft is internally invalid.",
        "Do not convert a skip_with_reason draft into an invocation; unselected models must remain skipped.",
        "Prefer the deterministic_draft unless a field is inconsistent with the observed contract.",
        "For unsupported or unsafe model contracts, use skip_with_reason with a clear reason.",
        "Do not construct SEA-AD 140-gene NPZ inputs or invoke trained direct heads; annotation execution must use raw label-transfer/kNN-style adapters only.",
        "Do not invent files or checkpoints; use only paths provided in observe.",
        "Species mismatch is not a hard filter; gene/input contract is the execution gate.",
    ]
    payload: dict[str, Any] = {
        "instructions": instructions,
        "response_contract": "A complete scMAS AdapterSpec JSON object.",
        "observe": observe,
    }
    if previous:
        payload["previous_attempt"] = previous
        payload["repair_instruction"] = "Repair the AdapterSpec so deterministic review passes."
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return system_prompt, user_prompt


def _review_llm_adapter_spec(candidate: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        validate_adapter_spec(candidate)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"schema_or_safety:{type(exc).__name__}: {exc}")

    immutable_keys = [
        "schema_version",
        "mode",
        "dataset_id",
        "query_path",
        "query_adapter",
        "model_id",
        "source_id",
        "capability_yaml",
    ]
    for key in immutable_keys:
        if candidate.get(key) != draft.get(key):
            errors.append(f"immutable_field_changed:{key}")
    if candidate.get("expected_outputs") != draft.get("expected_outputs"):
        errors.append("immutable_field_changed:expected_outputs")

    draft_artifacts = draft.get("input_artifacts", {}) if isinstance(draft.get("input_artifacts", {}), dict) else {}
    cand_artifacts = candidate.get("input_artifacts", {}) if isinstance(candidate.get("input_artifacts", {}), dict) else {}
    if cand_artifacts != draft_artifacts:
        errors.append("immutable_field_changed:input_artifacts")

    draft_runtime = draft.get("runtime_payload", {}) if isinstance(draft.get("runtime_payload", {}), dict) else {}
    cand_runtime = candidate.get("runtime_payload", {}) if isinstance(candidate.get("runtime_payload", {}), dict) else {}
    for key, value in draft_runtime.items():
        if cand_runtime.get(key) != value:
            errors.append(f"runtime_payload_changed:{key}")

    action_names = [
        str(item.get("action", ""))
        for item in candidate.get("actions", [])
        if isinstance(item, dict)
    ]
    draft_action_names = [
        str(item.get("action", ""))
        for item in draft.get("actions", [])
        if isinstance(item, dict)
    ]
    if not action_names:
        errors.append("missing_actions")
    if action_names != draft_action_names:
        errors.append("action_sequence_changed")
    if any(action not in ALLOWED_ACTIONS for action in action_names):
        errors.append("unknown_action_in_actions")
    if "invoke_raw_embedding_transfer" in action_names:
        if not candidate.get("runtime_payload", {}).get("reference_path"):
            errors.append("raw_transfer_missing_reference_path")
        if not candidate.get("runtime_payload", {}).get("embedding_method"):
            errors.append("raw_transfer_missing_embedding_method")
    if "skip_with_reason" in action_names and not (
        candidate.get("runtime_payload", {}).get("skip_reason")
        or any(item.get("reason") for item in candidate.get("actions", []) if isinstance(item, dict))
    ):
        errors.append("skip_missing_reason")
    return {"status": "passed" if not errors else "failed", "errors": errors, "warnings": warnings}


def _llm_react_adapter_spec(
    *,
    draft: dict[str, Any],
    plan: dict[str, Any],
    out_dir: Path,
    llm_model: str,
    retry_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_dir = ensure_dir(out_dir / "llm_react" / str(draft["model_id"]))
    observe = _llm_observe_for_adapter_spec(draft, plan)
    write_json(observe, trace_dir / "observe.json")
    previous: dict[str, Any] | None = None
    last_review: dict[str, Any] = {}
    last_candidate: dict[str, Any] = {}
    for attempt in range(1, max(0, retry_limit) + 2):
        system_prompt, user_prompt = _render_adapter_prompt(observe, previous=previous)
        (trace_dir / f"prompt_attempt_{attempt}.md").write_text(
            "## System\n\n" + system_prompt + "\n\n## User\n\n```json\n" + user_prompt + "\n```\n",
            encoding="utf-8",
        )
        parsed, raw_text, meta = _call_openai_json(model=llm_model, system_prompt=system_prompt, user_prompt=user_prompt)
        (trace_dir / f"response_attempt_{attempt}.txt").write_text(raw_text, encoding="utf-8")
        write_json({"parsed": parsed, "meta": meta}, trace_dir / f"parsed_attempt_{attempt}.json")
        review = _review_llm_adapter_spec(parsed, draft)
        write_json(review, trace_dir / f"review_attempt_{attempt}.json")
        last_review = review
        last_candidate = parsed
        if review["status"] == "passed":
            parsed["react_policy"] = "llm_react_v1"
            parsed["llm_trace_dir"] = str(trace_dir)
            parsed.setdefault("review", {})
            parsed["review"]["llm_adapter_review"] = review
            return parsed, {"status": "passed", "trace_dir": str(trace_dir), "review": review}
        previous = {"response": parsed, "review": review}
    raise RuntimeError(
        "LLM adapter spec failed deterministic review for "
        f"{draft.get('model_id')}: {'; '.join(last_review.get('errors', ['unknown']))}"
    )


def _spec_path(output_dir: Path, model_id: str) -> Path:
    return output_dir / "adapter_specs" / f"{model_id}.yaml"


def _empty_predictions(path: Path) -> None:
    columns = ["dataset_id", "model_id", "source_id", "method", "task", "cell_id", "sample_id", "true_label", "pred_label", "confidence"]
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _empty_metrics(path: Path) -> None:
    columns = ["dataset_id", "model_id", "source_id", "method", "task", "accuracy", "macro_f1", "weighted_f1"]
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _write_skip_run(spec: dict[str, Any], reason: str) -> dict[str, Any]:
    run_dir = ensure_dir(spec["expected_outputs"]["model_run_dir"])
    _empty_predictions(run_dir / "predictions.csv")
    _empty_metrics(run_dir / "metrics.csv")
    summary = {
        "dataset_id": spec["dataset_id"],
        "model_id": spec["model_id"],
        "source_id": spec["source_id"],
        "status": "skipped",
        "reason": reason,
        "predictions_path": str(run_dir / "predictions.csv"),
        "metrics_path": str(run_dir / "metrics.csv"),
    }
    write_json(summary, run_dir / "run_summary.json")
    return summary


def _execute_raw_label_transfer(spec: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    input_dir = ensure_dir(Path(spec["runtime_payload"]["one_pair_plan_path"]).parent)
    raw_input_path = input_dir / "raw_label_transfer_input.yaml"
    write_yaml(
        {
            "dataset_id": spec["dataset_id"],
            "query_path": spec["query_path"],
            "reference_path": spec["runtime_payload"]["reference_path"],
            "method": spec["runtime_payload"]["method"],
            "embedding_method": spec["runtime_payload"]["embedding_method"],
            "transfer_method": spec["runtime_payload"]["transfer_method"],
            "species_is_filter": False,
        },
        raw_input_path,
    )
    one_pair_plan = dict(plan)
    selected_pair = {
        "model_id": spec["model_id"],
        "source_id": spec["source_id"],
        "capability_yaml": spec["capability_yaml"],
        "reference_path": spec["runtime_payload"]["reference_path"],
        "method": spec["runtime_payload"]["method"],
        "embedding_method": spec["runtime_payload"]["embedding_method"],
        "transfer_method": spec["runtime_payload"]["transfer_method"],
        "shared_genes": spec["gene_strategy"].get("selected_shared_genes", 0),
        "execution_ready": True,
    }
    one_pair_plan["selected_model_ids"] = [spec["model_id"]]
    one_pair_plan["selected_pairs"] = [selected_pair]
    one_pair_plan_path = Path(spec["runtime_payload"]["one_pair_plan_path"])
    write_yaml(one_pair_plan, one_pair_plan_path)

    run_dir = ensure_dir(spec["expected_outputs"]["model_run_dir"])
    result = run_cross_species_plan(
        plan_path=one_pair_plan_path,
        output_dir=run_dir,
        max_query_cells=int(spec["runtime_payload"].get("max_query_cells", 0)),
        max_reference_cells=int(spec["runtime_payload"].get("max_reference_cells", 0)),
        min_shared_genes=int(spec["runtime_payload"].get("min_shared_genes", 0)),
        k=int(spec["runtime_payload"].get("k", 0)),
        device=spec["runtime_payload"].get("device", ""),
        batch_size=int(spec["runtime_payload"].get("batch_size", 16)),
    )
    status = "completed" if int(result.get("n_prediction_rows", 0)) > 0 and int(result.get("n_metric_rows", 0)) > 0 else "skipped"
    result["status"] = status
    if status != "completed":
        result["reason"] = "raw_embedding_transfer_produced_no_predictions_or_metrics"
    return result


def _execute_spec(spec: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    validate_adapter_spec(spec)
    action_names = [item["action"] for item in spec["actions"]]
    if "skip_with_reason" in action_names:
        reason = str(spec["runtime_payload"].get("skip_reason") or spec["actions"][-1].get("reason", "skipped"))
        return _write_skip_run(spec, reason)
    if "invoke_raw_embedding_transfer" in action_names:
        return _execute_raw_label_transfer(spec, plan)
    if "invoke_postprocessor" in action_names:
        return _write_skip_run(spec, "requires_shared_predictions")
    return _write_skip_run(spec, "no_invocation_action_in_adapter_spec")


def _review_spec(spec: dict[str, Any], result: dict[str, Any] | None = None, error: str = "") -> dict[str, Any]:
    checks = []
    try:
        validate_adapter_spec(spec)
        checks.append({"name": "schema", "status": "passed"})
    except Exception as exc:
        checks.append({"name": "schema", "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    for name, value in spec.get("input_artifacts", {}).items():
        if name.endswith("_path") or name.endswith("_yaml") or name.endswith("_dir"):
            if value and isinstance(value, str):
                checks.append({"name": f"artifact:{name}", "status": "passed" if Path(value).exists() else "missing", "path": value})
    artifact_status = spec.get("model_contract", {}).get("artifact_status", {})
    if artifact_status and artifact_status.get("ok") is False:
        checks.append({"name": "registered_artifacts", "status": "failed", "checks": artifact_status.get("checks", [])})
    if spec.get("model_contract", {}).get("compatible_contract") == SEAAD_140_CONTRACT:
        skip_reason = spec.get("runtime_payload", {}).get("skip_reason", "")
        ok = skip_reason == "direct_seaad_140_head_disabled_use_raw_label_transfer_knn"
        checks.append({"name": "direct_head_disabled_guard", "status": "passed" if ok else "failed"})
    if result is not None:
        for key in ("predictions_path", "metrics_path"):
            path = result.get(key) or spec.get("expected_outputs", {}).get(key.replace("_path", "_csv"))
            if path:
                checks.append({"name": f"output:{key}", "status": "passed" if Path(path).exists() else "missing", "path": str(path)})
    if error:
        checks.append({"name": "execution_error", "status": "failed", "reason": error})
    status = "passed" if all(item["status"] == "passed" for item in checks) and not error else "needs_review"
    return {"model_id": spec.get("model_id", ""), "status": status, "checks": checks}


def _read_run_rows(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_execution_summary_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Stage-3 Adapter Execution: {summary['dataset_id']}",
        "",
        f"- mode: {summary['mode']}",
        f"- ready_for_consensus: {summary['ready_for_consensus']}",
        f"- completed_models: {len(summary['completed_models'])}",
        f"- skipped_models: {len(summary['skipped_models'])}",
        f"- failed_models: {len(summary['failed_models'])}",
        "",
        "| model_id | status | reason | predictions | metrics |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_model = {row["model_id"]: row for row in summary.get("model_status", [])}
    for model_id in sorted(by_model):
        row = by_model[model_id]
        lines.append(
            f"| {model_id} | {row['status']} | {row.get('reason', '')} | "
            f"{row.get('predictions_path', '')} | {row.get('metrics_path', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def adapt_and_execute(
    *,
    plan_path: str | Path,
    mode: str = "subset",
    output_dir: str | Path | None = None,
    capability_dir: str | Path = DEFAULT_CAPABILITY_DIR,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    resume: bool = False,
    retry_limit: int = 2,
    llm_mode: str = "required",
    llm_model: str | None = None,
    llm_retry_limit: int = 2,
) -> dict[str, Any]:
    """Generate audited AdapterSpecs from a stage-2 plan and execute whitelist actions."""
    if mode not in {"subset", "full"}:
        raise ValueError("mode must be 'subset' or 'full'")
    if mode == "full" and not resume:
        raise ValueError("full mode requires --resume so it reuses the reviewed subset adapter contract")
    if llm_mode not in {"required", "optional", "off"}:
        raise ValueError("llm_mode must be one of: required, optional, off")
    _load_env_file()
    llm_model = llm_model or default_llm_model(DEFAULT_LLM_MODEL)

    plan_path = Path(plan_path)
    plan = read_yaml(plan_path)
    dataset_id = str(plan["dataset_id"])
    out_dir = ensure_dir(output_dir or (DEFAULT_STAGE3_ROOT / dataset_id / mode))
    ensure_dir(out_dir / "adapter_specs")
    ensure_dir(out_dir / "prepared_inputs")
    ensure_dir(out_dir / "model_runs")

    contract_result = inspect_model_contracts(capability_dir=capability_dir, registry_path=registry_path)
    contracts = {row["model_id"]: ModelContract(**row) for row in contract_result["contracts"]}
    specs = _build_adapter_specs(plan=plan, contracts=contracts, output_dir=out_dir, mode=mode)

    adapter_specs: dict[str, str] = {}
    model_status: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    llm_react_reviews: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    prediction_artifacts: dict[str, str] = {}
    metric_artifacts: dict[str, str] = {}
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    for draft_spec in specs:
        spec = draft_spec
        if llm_mode != "off":
            try:
                spec, llm_review = _llm_react_adapter_spec(
                    draft=draft_spec,
                    plan=plan,
                    out_dir=out_dir,
                    llm_model=str(llm_model),
                    retry_limit=llm_retry_limit,
                )
                llm_react_reviews.append({"model_id": spec["model_id"], **llm_review})
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "model_id": draft_spec.get("model_id", ""),
                    "status": "failed",
                    "mode": llm_mode,
                    "llm_model": llm_model,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                llm_react_reviews.append(failure)
                if llm_mode == "required":
                    write_json({"llm_react_reviews": llm_react_reviews}, out_dir / "llm_react_review.json")
                    raise
                spec = draft_spec
                spec.setdefault("review", {})
                spec["review"]["llm_adapter_review"] = failure

        spec_file = _spec_path(out_dir, spec["model_id"])
        write_yaml(spec, spec_file)
        adapter_specs[spec["model_id"]] = str(spec_file)

        result: dict[str, Any] | None = None
        last_error = ""
        for attempt in range(max(0, retry_limit) + 1):
            try:
                result = _execute_spec(spec, plan)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                spec.setdefault("react_trace", []).append(
                    {
                        "attempt": attempt + 1,
                        "review": "execution_failed",
                        "error": last_error,
                    }
                )
                if attempt >= retry_limit:
                    result = _write_skip_run(spec, f"adapter_execution_failed_after_retries:{last_error}")
                    result["status"] = "failed"
                    result["traceback"] = traceback.format_exc(limit=6)
                    write_json(result, Path(spec["expected_outputs"]["model_run_dir"]) / "run_summary.json")
        write_yaml(spec, spec_file)
        review = _review_spec(spec, result=result, error=last_error if result and result.get("status") == "failed" else "")
        reviews.append(review)

        assert result is not None
        status = str(result.get("status", "skipped"))
        pred_path = result.get("predictions_path") or spec["expected_outputs"]["predictions_csv"]
        metrics_path = result.get("metrics_path") or spec["expected_outputs"]["metrics_csv"]
        if status == "completed":
            prediction_artifacts[spec["model_id"]] = str(pred_path)
            metric_artifacts[spec["model_id"]] = str(metrics_path)
            pred_df = _read_run_rows(pred_path)
            metrics_df = _read_run_rows(metrics_path)
            if not pred_df.empty:
                all_predictions.append(pred_df)
            if not metrics_df.empty:
                all_metrics.append(metrics_df)
        else:
            skip_rows.append(
                {
                    "dataset_id": dataset_id,
                    "model_id": spec["model_id"],
                    "source_id": spec["source_id"],
                    "stage": "adapter_execution",
                    "status": status,
                    "reason": result.get("reason", spec.get("runtime_payload", {}).get("skip_reason", "")),
                    "traceback": result.get("traceback", ""),
                }
            )
        model_status.append(
            {
                "model_id": spec["model_id"],
                "source_id": spec["source_id"],
                "status": status,
                "reason": result.get("reason", ""),
                "adapter_spec": str(spec_file),
                "predictions_path": str(pred_path),
                "metrics_path": str(metrics_path),
            }
        )

    predictions_csv = out_dir / "predictions.csv"
    metrics_csv = out_dir / "metrics.csv"
    skips_csv = out_dir / "skips_and_failures.csv"
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(predictions_csv, index=False)
    else:
        _empty_predictions(predictions_csv)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(metrics_csv, index=False)
    else:
        _empty_metrics(metrics_csv)
    pd.DataFrame(skip_rows).to_csv(skips_csv, index=False)
    write_json({"reviews": reviews, "llm_react_reviews": llm_react_reviews}, out_dir / "adapter_review.json")
    write_json({"llm_react_reviews": llm_react_reviews}, out_dir / "llm_react_review.json")

    completed_models = [row["model_id"] for row in model_status if row["status"] == "completed"]
    skipped_models = [row["model_id"] for row in model_status if row["status"] == "skipped"]
    failed_models = [row["model_id"] for row in model_status if row["status"] == "failed"]
    summary = {
        "dataset_id": dataset_id,
        "mode": mode,
        "plan_path": str(plan_path),
        "output_dir": str(out_dir),
        "selected_model_ids": list(plan.get("selected_model_ids", [])),
        "completed_models": completed_models,
        "skipped_models": skipped_models,
        "failed_models": failed_models,
        "prediction_artifacts": prediction_artifacts,
        "metric_artifacts": metric_artifacts,
        "adapter_specs": adapter_specs,
        "model_status": model_status,
        "predictions_csv": str(predictions_csv),
        "metrics_csv": str(metrics_csv),
        "skips_and_failures_csv": str(skips_csv),
        "adapter_review_json": str(out_dir / "adapter_review.json"),
        "llm_mode": llm_mode,
        "llm_model": llm_model if llm_mode != "off" else "",
        "llm_react_review_json": str(out_dir / "llm_react_review.json"),
        "ready_for_consensus": len(completed_models) >= 2,
    }
    write_json(summary, out_dir / "execution_summary.json")
    _write_execution_summary_md(summary, out_dir / "execution_summary.md")

    # Keep a plan-local pointer for the next consensus stage.
    latest_pointer = plan_path.parent / f"stage3_{mode}_execution_summary.json"
    try:
        shutil.copyfile(out_dir / "execution_summary.json", latest_pointer)
    except OSError:
        pass
    return summary
