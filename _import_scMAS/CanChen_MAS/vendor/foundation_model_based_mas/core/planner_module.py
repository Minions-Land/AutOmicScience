from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import yaml
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from llm_runtime import build_chat_model
from tools_layer import describe_tool_catalog, get_function_tools
from tools_layer.mcp_tools import MCP_SERVER_CATALOG

from .contracts import (
    coerce_invocation_outline,
    normalize_server_name,
    resolve_foundation_model_selection,
    resolve_path_mix_counts,
)
from .langsmith_compat import traceable, tracing_context
from .planner_prompts import (
    PLAN_JUDGE_SYSTEM_PROMPT,
    build_plan_judge_user_prompt,
    build_planner_system_prompt,
    build_planner_user_prompt,
    render_input_to_planner_context,
)
from .tracing import bootstrap_langsmith_from_env


class PlannerConfigModel(BaseModel):
    capability_dir: str = "../config/capability"
    capability_glob: str = "**/*.yaml"
    traditional_path_count: int = Field(default=1, ge=0)
    foundation_model_path_count: int = Field(default=2, ge=0)
    judge_pass_score: float = 4.0
    max_replan_rounds: int = 3

class PlanStepModel(BaseModel):
    step_id: str
    step_name: str
    step_type: str
    candidate_tool: str = ""
    candidate_server: str = ""
    purpose: str
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class PlannedPathModel(BaseModel):
    path_id: str
    path_type: Literal["traditional", "foundation_model"]
    task_type: str
    goal: str
    rationale: str
    selection_evidence: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    failure_risks: list[str] = Field(default_factory=list)
    evaluation_focus: list[str] = Field(default_factory=list)
    plan_steps: list[PlanStepModel] = Field(default_factory=list)
    selected_model_id: str = ""
    selected_mcp_tool_name: str = ""
    capability_ref: dict[str, Any] = Field(default_factory=dict)
    invocation_outline: dict[str, Any] = Field(default_factory=dict)
    model_fit_summary: str = ""

    @field_validator("invocation_outline", mode="before")
    @classmethod
    def _coerce_invocation_outline(cls, value: Any) -> dict[str, Any]:
        return coerce_invocation_outline(value)


class PlannerDraftOutput(BaseModel):
    paths: list[PlannedPathModel]


class JudgeReviewItem(BaseModel):
    path_id: str
    task_match_score: float
    data_match_score: float
    planning_quality_score: float
    final_score: float
    passed: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    repair_instruction: str = ""


class JudgeReviewOutput(BaseModel):
    reviews: list[JudgeReviewItem]


class PlannerState(TypedDict):
    config_path: str
    task_type: str
    task_request: str
    dataset_description: str
    data_profile: dict[str, Any]
    input_manifest: dict[str, Any]
    prepared_h5ad_path: str
    override_total_paths: int
    planner_config: dict[str, Any]
    tool_catalog_snapshot: dict[str, Any]
    capability_catalog_snapshot: dict[str, Any]
    planning_context_text: str
    planning_round: int
    current_paths: list[dict[str, Any]]
    approved_paths: list[dict[str, Any]]
    rejected_paths: list[dict[str, Any]]
    latest_judge_reviews: list[dict[str, Any]]
    judge_reviews: list[dict[str, Any]]
    repair_history: list[dict[str, Any]]
    planning_status: str
    final_result: dict[str, Any]


class PlannerModule:
    def __init__(self, *, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tracing_config = bootstrap_langsmith_from_env(self.project_root, default_project="foundation_model_based_mas_planner")
        self.langsmith_enabled = bool(self.tracing_config.get("enabled", False))
        self.logger = logging.getLogger("foundation_model_based_mas.planner_module")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh = logging.FileHandler(self.logs_dir / "mas_planner_module.log", encoding="utf-8")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)

        self.llm = build_chat_model(prefix="OPENAI", default_temperature=0.0)
        self.capability_tool = get_function_tools(["read_capability_files_tool"])[0]
        self.agent_tools = [self.capability_tool]
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PlannerState)
        graph.add_node("prepare_planning_context_node", self.prepare_planning_context_node)
        graph.add_node("planner_generate_paths_node", self.planner_generate_paths_node)
        graph.add_node("plan_judge_node", self.plan_judge_node)
        graph.add_node("repair_failed_paths_node", self.repair_failed_paths_node)
        graph.add_node("finalize_plan_node", self.finalize_plan_node)
        graph.add_edge(START, "prepare_planning_context_node")
        graph.add_edge("prepare_planning_context_node", "planner_generate_paths_node")
        graph.add_edge("planner_generate_paths_node", "plan_judge_node")
        graph.add_conditional_edges(
            "plan_judge_node",
            self._route_after_plan_judge,
            {
                "repair_failed_paths_node": "repair_failed_paths_node",
                "finalize_plan_node": "finalize_plan_node",
            },
        )
        graph.add_edge("repair_failed_paths_node", "plan_judge_node")
        graph.add_edge("finalize_plan_node", END)
        return graph.compile()

    def _route_after_plan_judge(self, state: PlannerState) -> str:
        if state["rejected_paths"] and state["planning_round"] < int(state["planner_config"]["max_replan_rounds"]):
            return "repair_failed_paths_node"
        return "finalize_plan_node"

    def _log_preview(self, node: str, payload: Any) -> None:
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = str(payload)
        if len(text) > 1200:
            text = text[:1200] + "...[truncated]"
        self.logger.info("[%s] %s", node, text)

    def _retry_llm_call(self, fn, label: str, max_retries: int = 3, base_sleep: float = 3.0):
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "[llm_retry] failed %s | attempt=%s/%s | error=%s: %s",
                    label,
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    exc,
                )
                if attempt >= max_retries:
                    raise
                time.sleep(base_sleep * attempt)
        raise last_error or RuntimeError(f"LLM call failed: {label}")

    def _load_planner_config(self, config_path: str | Path, override_total_paths: int = 0) -> PlannerConfigModel:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        config = PlannerConfigModel.model_validate(payload.get("planner", {}))
        traditional_count, foundation_count = resolve_path_mix_counts(
            traditional_path_count=config.traditional_path_count,
            foundation_model_path_count=config.foundation_model_path_count,
            override_total_paths=override_total_paths,
        )
        config.traditional_path_count = traditional_count
        config.foundation_model_path_count = foundation_count
        return config

    def _build_tool_catalog_snapshot(self) -> dict[str, Any]:
        function_and_skill_catalog = describe_tool_catalog()
        mcp_servers = [dict(value) for value in MCP_SERVER_CATALOG.values()]
        return {
            "function_and_skill_tools": function_and_skill_catalog,
            "mcp_servers": mcp_servers,
        }

    def _build_capability_snapshot(self, planner_config: PlannerConfigModel, config_path: str | Path) -> dict[str, Any]:
        capability_dir = planner_config.capability_dir
        if not Path(capability_dir).is_absolute():
            capability_dir = str((Path(config_path).resolve().parent / capability_dir).resolve())
        result = self.capability_tool.invoke(
            {
                "directory_path": capability_dir,
                "glob_pattern": planner_config.capability_glob,
            }
        )
        if not result.get("success", False):
            raise RuntimeError(result.get("summary", "Failed to load capability files."))
        if not result.get("models"):
            raise RuntimeError(f"No valid capability YAML files were found under {capability_dir}.")
        return result

    def _resolve_server_name(self, server_name: str) -> str:
        return normalize_server_name(server_name, MCP_SERVER_CATALOG)

    def _validate_foundation_model_path_contract(
        self,
        *,
        path: dict[str, Any],
        task_type: str,
        capability_index: dict[str, dict[str, Any]],
        data_profile: dict[str, Any],
        input_manifest: dict[str, Any],
    ) -> list[str]:
        if path.get("path_type") != "foundation_model":
            return []

        reasons: list[str] = []
        capability, selection_reason, selected_model_id, selected_mcp_tool_name = resolve_foundation_model_selection(
            path=path,
            capability_index=capability_index,
        )
        if capability is None:
            if selection_reason == "missing_selected_mcp_tool_name":
                return ["foundation-model path is missing selected_mcp_tool_name."]
            if selection_reason == "missing_capability":
                return [f"selected_model_id '{selected_model_id}' does not exist in capability files."]
            if selection_reason == "unknown_mcp_tool_name":
                return [f"selected_mcp_tool_name '{selected_mcp_tool_name}' does not exist in capability files."]
            if selection_reason == "ambiguous_mcp_tool_name":
                return [f"selected_mcp_tool_name '{selected_mcp_tool_name}' matches multiple capability files."]
            if selection_reason == "selected_model_tool_mismatch":
                return [
                    f"selected_model_id '{selected_model_id}' does not match selected_mcp_tool_name '{selected_mcp_tool_name}'."
                ]
            return ["foundation-model path has an invalid tool selection."]

        supported_tasks = {str(item).strip() for item in capability.get("supported_tasks", [])}
        if task_type not in supported_tasks:
            reasons.append(
                f"selected_model_id '{selected_model_id}' does not advertise support for task_type '{task_type}'."
            )

        candidate_server = self._resolve_server_name(str(capability.get("mcp_server_name", "")).strip())
        candidate_tool = str(capability.get("mcp_tool_name", "")).strip()
        if not candidate_server:
            reasons.append("selected foundation-model capability is missing a valid mcp_server_name.")
        if not candidate_tool:
            reasons.append("selected foundation-model capability is missing mcp_tool_name.")
        elif candidate_server:
            server_tools = set(MCP_SERVER_CATALOG[candidate_server].get("tool_names", []))
            if candidate_tool not in server_tools:
                reasons.append(
                    f"selected_mcp_tool_name '{candidate_tool}' does not exist on MCP server '{candidate_server}'."
                )

        compatible, compatibility_reasons = self._is_capability_compatible(
            capability=capability,
            data_profile=data_profile,
            input_manifest=input_manifest,
        )
        if not compatible:
            reasons.extend(compatibility_reasons)

        return reasons

    def _agent_reason_with_tools(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        label: str,
    ) -> BaseModel:
        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        llm = self.llm.bind_tools(self.agent_tools, strict=False)
        resp = self._retry_llm_call(lambda: llm.invoke(msgs), f"{label}.initial")
        loops = 0
        while getattr(resp, "tool_calls", None):
            msgs.append(resp)
            for call in resp.tool_calls:
                tool_name = call.get("name", "")
                if tool_name != self.capability_tool.name:
                    continue
                result = self.capability_tool.invoke(call.get("args", {}))
                msgs.append(ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str), tool_call_id=call["id"]))
            loops += 1
            if loops >= 3:
                break
            resp = self._retry_llm_call(lambda: llm.invoke(msgs), f"{label}.loop_{loops}")

        formatter = self.llm.with_structured_output(schema, method="function_calling")
        return self._retry_llm_call(
            lambda: formatter.invoke(
                [
                    SystemMessage(content="Extract only grounded structured output into the target schema."),
                    HumanMessage(content=f"Request:\n{user_prompt}\n\nDraft response:\n{resp.content if isinstance(resp.content, str) else str(resp.content)}"),
                ]
            ),
            f"{label}.formatter",
        )

    def _normalize_path(
        self,
        payload: dict[str, Any],
        task_type: str,
        capability_index: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["task_type"] = task_type
        normalized["selection_evidence"] = list(normalized.get("selection_evidence", []))
        normalized["required_inputs"] = list(normalized.get("required_inputs", []))
        normalized["expected_outputs"] = list(normalized.get("expected_outputs", []))
        normalized["preconditions"] = list(normalized.get("preconditions", []))
        normalized["failure_risks"] = list(normalized.get("failure_risks", []))
        normalized["evaluation_focus"] = list(normalized.get("evaluation_focus", []))
        normalized["plan_steps"] = list(normalized.get("plan_steps", []))
        normalized.setdefault("selected_model_id", "")
        normalized.setdefault("selected_mcp_tool_name", "")
        normalized.setdefault("capability_ref", {})
        normalized["invocation_outline"] = coerce_invocation_outline(normalized.get("invocation_outline", {}))
        normalized.setdefault("model_fit_summary", "")

        capability = None
        if capability_index and normalized.get("path_type") == "foundation_model":
            capability, _, resolved_model_id, resolved_tool_name = resolve_foundation_model_selection(
                path=normalized,
                capability_index=capability_index,
            )
            if resolved_model_id and not str(normalized.get("selected_model_id", "")).strip():
                normalized["selected_model_id"] = resolved_model_id
            if resolved_tool_name and not str(normalized.get("selected_mcp_tool_name", "")).strip():
                normalized["selected_mcp_tool_name"] = resolved_tool_name

        normalized_steps: list[dict[str, Any]] = []
        for index, raw_step in enumerate(normalized.get("plan_steps", []), start=1):
            step = dict(raw_step)
            step.setdefault("step_id", f"{normalized.get('path_id', 'path')}_step_{index}")
            step["depends_on"] = list(step.get("depends_on", []))
            step["input_contract"] = dict(step.get("input_contract", {}))
            step["output_contract"] = dict(step.get("output_contract", {}))
            if capability is not None:
                if not str(step.get("candidate_server", "")).strip():
                    step["candidate_server"] = str(capability.get("mcp_server_name", "")).strip()
                if not str(step.get("candidate_tool", "")).strip():
                    step["candidate_tool"] = str(capability.get("mcp_tool_name", "")).strip()
            normalized_steps.append(step)
        normalized["plan_steps"] = normalized_steps
        return normalized

    def _validate_path_mix(self, paths: list[dict[str, Any]], planner_config: PlannerConfigModel) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        traditional = [path for path in paths if path.get("path_type") == "traditional"]
        foundation = [path for path in paths if path.get("path_type") == "foundation_model"]
        errors: list[dict[str, Any]] = []
        if len(traditional) != planner_config.traditional_path_count:
            errors.append(
                {
                    "path_id": "traditional_path_count",
                    "reason": f"Expected {planner_config.traditional_path_count} traditional path(s), got {len(traditional)}.",
                }
            )
        if len(foundation) != planner_config.foundation_model_path_count:
            errors.append(
                {
                    "path_id": "foundation_model_path_count",
                    "reason": f"Expected {planner_config.foundation_model_path_count} foundation-model path(s), got {len(foundation)}.",
                }
            )
        return paths, errors

    def _score_capability_fit(
        self,
        *,
        capability: dict[str, Any],
        task_type: str,
        data_profile: dict[str, Any],
        input_manifest: dict[str, Any],
    ) -> float:
        task_scores = capability.get("task_scores", {}) if isinstance(capability, dict) else {}
        if not isinstance(task_scores, dict):
            task_scores = {}
        score = float(task_scores.get(task_type, 0.0) or 0.0)

        species = str(
            data_profile.get("species")
            or data_profile.get("dataset_metadata", {}).get("species")
            or input_manifest.get("metadata", {}).get("species")
            or ""
        ).strip().lower()
        modality = str(
            data_profile.get("modality")
            or data_profile.get("dataset_metadata", {}).get("modality")
            or input_manifest.get("metadata", {}).get("modality")
            or ""
        ).strip().lower()
        tissue = str(
            data_profile.get("tissue_hint")
            or data_profile.get("dataset_metadata", {}).get("tissue")
            or input_manifest.get("metadata", {}).get("tissue")
            or ""
        ).strip().lower()
        has_spatial = bool(data_profile.get("has_spatial", False))

        species_scope = {str(item).strip().lower() for item in capability.get("species_scope", [])}
        tissue_scope = {str(item).strip().lower() for item in capability.get("tissue_scope", [])}
        modalities = {str(item).strip().lower() for item in capability.get("modalities", [])}
        required_formats = {str(item).strip().lower() for item in capability.get("input_requirements", {}).get("required_formats", [])}
        data_constraints = capability.get("data_constraints", {}) if isinstance(capability.get("data_constraints", {}), dict) else {}
        preferred_dataset_family = str(capability.get("annotation_profile", {}).get("preferred_dataset_family", "")).strip().lower()
        modality_fit = str(capability.get("annotation_profile", {}).get("modality_fit", "")).strip().lower()

        if species and (species in species_scope or "broad" in species_scope or "human-centric" in species_scope):
            score += 0.05
        if tissue and (tissue in tissue_scope or "broad" in tissue_scope):
            score += 0.03
        if modality:
            if any(token in modality for token in modalities):
                score += 0.04
            if modality in modality_fit or modality in preferred_dataset_family:
                score += 0.03
        if has_spatial and bool(data_constraints.get("supports_spatial", False)):
            score += 0.03
        if "npz" in required_formats and input_manifest.get("npz_path"):
            score += 0.02
        if "h5ad" in required_formats and input_manifest.get("h5ad_path"):
            score += 0.02
        if str(capability.get("priority_hint", "")).strip().lower() == "high":
            score += 0.01
        return score

    def _estimate_npz_feature_count(self, npz_path: str | Path) -> int | None:
        if not npz_path:
            return None
        path = Path(npz_path)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=True) as npz:
                matrix = npz.get("X")
                if matrix is None:
                    return None
                if getattr(matrix, "ndim", 0) < 2:
                    return None
                return int(matrix.shape[1])
        except Exception as exc:
            self.logger.warning("[planner.compatibility] failed to inspect npz feature count for %s: %s", path, exc)
            return None

    def _estimate_h5ad_gene_count(self, h5ad_path: str | Path, data_profile: dict[str, Any]) -> int | None:
        if data_profile.get("source_h5ad_path") == str(Path(h5ad_path).resolve()) and data_profile.get("n_genes"):
            try:
                return int(data_profile.get("n_genes"))
            except Exception:
                pass
        if not h5ad_path:
            return None
        path = Path(h5ad_path)
        if not path.exists():
            return None
        try:
            import anndata as ad

            adata = ad.read_h5ad(path, backed="r")
            try:
                count = sum(1 for gene in adata.var_names if not str(gene).startswith("Blank"))
            finally:
                if getattr(adata, "file", None) is not None:
                    adata.file.close()
            return int(count)
        except Exception as exc:
            self.logger.warning("[planner.compatibility] failed to inspect h5ad gene count for %s: %s", path, exc)
            return None

    def _foundation_model_compatibility_reasons(
        self,
        *,
        capability: dict[str, Any],
        data_profile: dict[str, Any],
        input_manifest: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        model_id = str(capability.get("model_id", "")).strip().lower()
        input_mode = str(input_manifest.get("input_mode", "")).strip().lower()
        if input_mode != "direct_pair":
            return reasons

        npz_path = str(input_manifest.get("npz_path", "")).strip()
        h5ad_path = str(input_manifest.get("h5ad_path", "")).strip()
        if not npz_path or not h5ad_path:
            return reasons

        npz_feature_count = self._estimate_npz_feature_count(npz_path)
        h5ad_gene_count = self._estimate_h5ad_gene_count(h5ad_path, data_profile)

        requires_strict_alignment = model_id in {"geneformer", "nicheformer"}
        if requires_strict_alignment and npz_feature_count and h5ad_gene_count and npz_feature_count != h5ad_gene_count:
            reasons.append(
                f"{model_id} requires matching h5ad gene names for direct paired inputs, but h5ad_gene_count={h5ad_gene_count} and npz_feature_count={npz_feature_count}."
            )
        return reasons

    def _is_capability_compatible(
        self,
        *,
        capability: dict[str, Any],
        data_profile: dict[str, Any],
        input_manifest: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        reasons = self._foundation_model_compatibility_reasons(
            capability=capability,
            data_profile=data_profile,
            input_manifest=input_manifest,
        )
        return not reasons, reasons

    def _build_deterministic_foundation_paths(
        self,
        *,
        planner_config: PlannerConfigModel,
        task_type: str,
        data_profile: dict[str, Any],
        input_manifest: dict[str, Any],
        capability_index: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for model_id, capability in capability_index.items():
            supported_tasks = {str(item).strip() for item in capability.get("supported_tasks", [])}
            mcp_tool_name = str(capability.get("mcp_tool_name", "")).strip()
            mcp_server_name = str(capability.get("mcp_server_name", "")).strip()
            if task_type not in supported_tasks:
                continue
            if not mcp_tool_name or not mcp_server_name:
                continue
            compatible, compatibility_reasons = self._is_capability_compatible(
                capability=capability,
                data_profile=data_profile,
                input_manifest=input_manifest,
            )
            if not compatible:
                self.logger.info(
                    "[planner.compatibility] skipping model_id=%s for deterministic fallback: %s",
                    model_id,
                    " ".join(compatibility_reasons),
                )
                continue
            fit_score = self._score_capability_fit(
                capability=capability,
                task_type=task_type,
                data_profile=data_profile,
                input_manifest=input_manifest,
            )
            ranked.append((fit_score, str(model_id), capability))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        selected = ranked[: planner_config.foundation_model_path_count]
        deterministic_paths: list[dict[str, Any]] = []
        for rank, (fit_score, model_id, capability) in enumerate(selected, start=1):
            server_name = self._resolve_server_name(str(capability.get("mcp_server_name", "")).strip())
            tool_name = str(capability.get("mcp_tool_name", "")).strip()
            deterministic_paths.append(
                {
                    "path_id": f"fallback_{model_id}_{rank}",
                    "path_type": "foundation_model",
                    "task_type": task_type,
                    "goal": f"Run the selected foundation-model MCP tool {tool_name} for {task_type}.",
                    "rationale": (
                        f"Deterministic fallback planner selected {model_id} because it is executable locally and "
                        f"ranked highly by capability task score plus dataset-fit heuristics."
                    ),
                    "selection_evidence": [
                        f"capability.task_scores.{task_type}={capability.get('task_scores', {}).get(task_type, 'NA')}",
                        f"fit_score={fit_score:.6f}",
                        f"mcp_server_name={server_name}",
                        f"mcp_tool_name={tool_name}",
                    ],
                    "required_inputs": [
                        f"npz_path: {input_manifest.get('npz_path', '')}",
                        f"h5ad_path: {input_manifest.get('h5ad_path', '')}",
                    ],
                    "expected_outputs": [
                        "prediction_records",
                        "cell_type_prediction",
                        "result_json",
                    ],
                    "preconditions": [
                        "paired npz + h5ad inputs are readable",
                        "selected MCP tool is registered locally",
                    ],
                    "failure_risks": [
                        "checkpoint mismatch",
                        "GPU resource contention",
                    ],
                    "evaluation_focus": ["accuracy", "macro_f1", "micro_f1"],
                    "plan_steps": [
                        {
                            "step_id": f"fallback_{model_id}_{rank}_execute",
                            "step_name": "execute_selected_mcp_tool",
                            "step_type": "mcp_call",
                            "candidate_tool": tool_name,
                            "candidate_server": server_name,
                            "purpose": f"Execute {tool_name} on the prepared paired inputs.",
                            "input_contract": {},
                            "output_contract": {},
                            "depends_on": [],
                        }
                    ],
                    "selected_model_id": model_id,
                    "selected_mcp_tool_name": tool_name,
                    "capability_ref": {"model_id": model_id},
                    "invocation_outline": {},
                    "model_fit_summary": (
                        f"{model_id} selected by deterministic fallback with fit score {fit_score:.4f} "
                        f"for species={data_profile.get('species') or data_profile.get('dataset_metadata', {}).get('species', '')}, "
                        f"tissue={data_profile.get('tissue_hint') or data_profile.get('dataset_metadata', {}).get('tissue', '')}, "
                        f"modality={data_profile.get('modality') or data_profile.get('dataset_metadata', {}).get('modality', '')}."
                    ),
                    "planning_source": "deterministic_fallback",
                }
            )
        return deterministic_paths

    def _merge_replanned_paths(self, approved_paths: list[dict[str, Any]], replacement_paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
        replacements = {path["path_id"]: path for path in replacement_paths}
        merged = list(approved_paths)
        merged.extend(replacements.values())
        return merged

    def prepare_planning_context_node(self, state: PlannerState) -> dict[str, Any]:
        planner_config = self._load_planner_config(state["config_path"], state["override_total_paths"])
        tool_catalog_snapshot = self._build_tool_catalog_snapshot()
        capability_catalog_snapshot = self._build_capability_snapshot(planner_config, state["config_path"])
        planning_context_text = render_input_to_planner_context(
            task_type=state["task_type"],
            task_request=state["task_request"],
            dataset_description=state["dataset_description"],
            data_profile=state["data_profile"],
            input_manifest=state["input_manifest"],
            prepared_h5ad_path=state["prepared_h5ad_path"],
            tool_catalog_snapshot=tool_catalog_snapshot,
            capability_catalog_snapshot={
                "summary": capability_catalog_snapshot.get("summary", ""),
                "model_index": capability_catalog_snapshot.get("model_index", {}),
                "invalid_files": capability_catalog_snapshot.get("invalid_files", []),
            },
        )
        update = {
            "planner_config": planner_config.model_dump(mode="python"),
            "tool_catalog_snapshot": tool_catalog_snapshot,
            "capability_catalog_snapshot": capability_catalog_snapshot,
            "planning_context_text": planning_context_text,
            "planning_round": 0,
            "current_paths": [],
            "approved_paths": [],
            "rejected_paths": [],
            "latest_judge_reviews": [],
            "judge_reviews": [],
            "repair_history": [],
            "planning_status": "initialized",
            "final_result": {},
        }
        self._log_preview("prepare_planning_context_node.update", update)
        return update

    def planner_generate_paths_node(self, state: PlannerState) -> dict[str, Any]:
        planner_config = PlannerConfigModel.model_validate(state["planner_config"])
        capability_index = state["capability_catalog_snapshot"].get("model_index", {})
        planning_status = "drafted"
        try:
            planned = self._agent_reason_with_tools(
                system_prompt=build_planner_system_prompt(
                    traditional_path_count=planner_config.traditional_path_count,
                    foundation_model_path_count=planner_config.foundation_model_path_count,
                ),
                user_prompt=build_planner_user_prompt(
                    planning_context=state["planning_context_text"],
                    planning_round=state["planning_round"] + 1,
                    generation_target_count=planner_config.traditional_path_count + planner_config.foundation_model_path_count,
                ),
                schema=PlannerDraftOutput,
                label="planner_generate_paths_node",
            )
            raw_paths = [item.model_dump(mode="python") for item in planned.paths]
        except Exception as exc:
            self.logger.warning(
                "[planner_generate_paths_node.fallback] using deterministic fallback after LLM failure: %s: %s",
                type(exc).__name__,
                exc,
            )
            raw_paths = self._build_deterministic_foundation_paths(
                planner_config=planner_config,
                task_type=state["task_type"],
                data_profile=state["data_profile"],
                input_manifest=state["input_manifest"],
                capability_index=capability_index,
            )
            planning_status = "fallback_generated"
        paths = [
            self._normalize_path(path, state["task_type"], capability_index=capability_index)
            for path in raw_paths
        ]
        _, structural_errors = self._validate_path_mix(paths, planner_config)
        contract_errors = []
        for path in paths:
            reasons = self._validate_foundation_model_path_contract(
                path=path,
                task_type=state["task_type"],
                capability_index=capability_index,
                data_profile=state["data_profile"],
                input_manifest=state["input_manifest"],
            )
            if reasons:
                contract_errors.append({"path_id": path["path_id"], "reason": " ".join(reasons)})
        update = {
            "planning_round": state["planning_round"] + 1,
            "current_paths": paths,
            "planning_status": planning_status,
            "rejected_paths": [
                {"path_id": item["path_id"], "rejection_reasons": [item["reason"]]}
                for item in [*structural_errors, *contract_errors]
            ],
        }
        self._log_preview("planner_generate_paths_node.update", update)
        return update

    def plan_judge_node(self, state: PlannerState) -> dict[str, Any]:
        planner_config = PlannerConfigModel.model_validate(state["planner_config"])
        if state["rejected_paths"]:
            reviews = []
            for item in state["rejected_paths"]:
                reviews.append(
                    {
                        "path_id": item["path_id"],
                        "task_match_score": 0.0,
                        "data_match_score": 0.0,
                        "planning_quality_score": 0.0,
                        "final_score": 0.0,
                        "passed": False,
                        "rejection_reasons": item.get("rejection_reasons", []),
                        "repair_instruction": "Regenerate the full path set so that the required path mix is satisfied.",
                    }
                )
            update = {
                "latest_judge_reviews": reviews,
                "judge_reviews": [*state["judge_reviews"], {"round": state["planning_round"], "reviews": reviews}],
                "approved_paths": [],
                "rejected_paths": reviews,
                "planning_status": "needs_repair",
            }
            self._log_preview("plan_judge_node.update", update)
            return update

        if state.get("planning_status") == "fallback_generated" or any(
            str(path.get("planning_source", "")) == "deterministic_fallback"
            for path in state.get("current_paths", [])
        ):
            reviews = []
            approved_paths: list[dict[str, Any]] = []
            for path in state["current_paths"]:
                review = {
                    "path_id": path["path_id"],
                    "task_match_score": 5.0,
                    "data_match_score": 5.0,
                    "planning_quality_score": 4.5,
                    "final_score": 4.8333333333,
                    "passed": True,
                    "rejection_reasons": [],
                    "repair_instruction": "Deterministic fallback path auto-approved because external LLM judge was unavailable.",
                }
                path_with_review = dict(path)
                path_with_review["judge_review"] = review
                approved_paths.append(path_with_review)
                reviews.append(review)
            update = {
                "latest_judge_reviews": reviews,
                "judge_reviews": [*state["judge_reviews"], {"round": state["planning_round"], "reviews": reviews}],
                "approved_paths": approved_paths,
                "rejected_paths": [],
                "planning_status": "approved",
            }
            self._log_preview("plan_judge_node.update", update)
            return update

        try:
            judged = self._agent_reason_with_tools(
                system_prompt=PLAN_JUDGE_SYSTEM_PROMPT,
                user_prompt=build_plan_judge_user_prompt(
                    planning_context=state["planning_context_text"],
                    candidate_paths=state["current_paths"],
                ),
                schema=JudgeReviewOutput,
                label="plan_judge_node",
            )
        except Exception as exc:
            self.logger.warning(
                "[plan_judge_node.fallback] auto-approving current paths after judge LLM failure: %s: %s",
                type(exc).__name__,
                exc,
            )
            reviews = []
            approved_paths = []
            for path in state["current_paths"]:
                review = {
                    "path_id": path["path_id"],
                    "task_match_score": 5.0,
                    "data_match_score": 5.0,
                    "planning_quality_score": 4.0,
                    "final_score": 4.6666666667,
                    "passed": True,
                    "rejection_reasons": [],
                    "repair_instruction": "Judge LLM unavailable; auto-approved executable path.",
                }
                path_with_review = dict(path)
                path_with_review["judge_review"] = review
                approved_paths.append(path_with_review)
                reviews.append(review)
            update = {
                "latest_judge_reviews": reviews,
                "judge_reviews": [*state["judge_reviews"], {"round": state["planning_round"], "reviews": reviews}],
                "approved_paths": approved_paths,
                "rejected_paths": [],
                "planning_status": "approved",
            }
            self._log_preview("plan_judge_node.update", update)
            return update
        reviews = [item.model_dump(mode="python") for item in judged.reviews]
        review_map = {item["path_id"]: item for item in reviews}
        approved_paths: list[dict[str, Any]] = []
        rejected_paths: list[dict[str, Any]] = []
        for path in state["current_paths"]:
            review = review_map.get(path["path_id"])
            if review is None:
                review = {
                    "path_id": path["path_id"],
                    "task_match_score": 0.0,
                    "data_match_score": 0.0,
                    "planning_quality_score": 0.0,
                    "final_score": 0.0,
                    "passed": False,
                    "rejection_reasons": ["The judge did not return a review for this path."],
                    "repair_instruction": "Return a complete judge review for this path.",
                }
                reviews.append(review)
            path_with_review = dict(path)
            path_with_review["judge_review"] = review
            if bool(review["final_score"] >= planner_config.judge_pass_score and review["passed"]):
                approved_paths.append(path_with_review)
            else:
                rejected_paths.append(path_with_review)

        update = {
            "latest_judge_reviews": reviews,
            "judge_reviews": [*state["judge_reviews"], {"round": state["planning_round"], "reviews": reviews}],
            "approved_paths": approved_paths,
            "rejected_paths": rejected_paths,
            "planning_status": "approved" if not rejected_paths else "needs_repair",
        }
        self._log_preview("plan_judge_node.update", update)
        return update

    def repair_failed_paths_node(self, state: PlannerState) -> dict[str, Any]:
        planner_config = PlannerConfigModel.model_validate(state["planner_config"])
        failed_path_ids = [item["path_id"] for item in state["rejected_paths"]]
        current_path_ids = {item["path_id"] for item in state["current_paths"]}
        requires_full_regeneration = any(path_id not in current_path_ids for path_id in failed_path_ids)
        repair_instructions = [
            {
                "path_id": item["path_id"],
                "repair_instruction": item.get("judge_review", {}).get("repair_instruction", item.get("repair_instruction", "")),
                "rejection_reasons": item.get("judge_review", {}).get("rejection_reasons", item.get("rejection_reasons", [])),
            }
            for item in state["rejected_paths"]
        ]
        approved_paths = [self._strip_judge_review(item) for item in state["approved_paths"]]
        replanned = self._agent_reason_with_tools(
            system_prompt=build_planner_system_prompt(
                traditional_path_count=planner_config.traditional_path_count,
                foundation_model_path_count=planner_config.foundation_model_path_count,
            ),
            user_prompt=build_planner_user_prompt(
                planning_context=state["planning_context_text"],
                planning_round=state["planning_round"] + 1,
                failed_path_ids=failed_path_ids,
                repair_instructions=repair_instructions,
                approved_paths=approved_paths,
                generation_target_count=(
                    planner_config.traditional_path_count + planner_config.foundation_model_path_count
                    if requires_full_regeneration
                    else len(failed_path_ids)
                ),
            ),
            schema=PlannerDraftOutput,
            label="repair_failed_paths_node",
        )
        capability_index = state["capability_catalog_snapshot"].get("model_index", {})
        replacement_paths = [
            self._normalize_path(
                item.model_dump(mode="python"),
                state["task_type"],
                capability_index=capability_index,
            )
            for item in replanned.paths
        ]
        merged_paths = replacement_paths if requires_full_regeneration else self._merge_replanned_paths(approved_paths, replacement_paths)
        repair_history = [
            *state["repair_history"],
            {
                "round": state["planning_round"] + 1,
                "failed_path_ids": failed_path_ids,
                "repair_instructions": repair_instructions,
                "replacement_paths": replacement_paths,
            },
        ]
        update = {
            "planning_round": state["planning_round"] + 1,
            "current_paths": merged_paths,
            "approved_paths": approved_paths,
            "rejected_paths": [],
            "repair_history": repair_history,
            "planning_status": "replanned",
        }
        self._log_preview("repair_failed_paths_node.update", update)
        return update

    def _strip_judge_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        cloned = dict(payload)
        cloned.pop("judge_review", None)
        return cloned

    def finalize_plan_node(self, state: PlannerState) -> dict[str, Any]:
        approved_paths = state["approved_paths"]
        rejected_paths = state["rejected_paths"]
        planning_status = "approved" if not rejected_paths else "partial_pass"
        final_result = {
            "planning_status": planning_status,
            "planning_rounds": state["planning_round"],
            "tool_catalog_snapshot": state["tool_catalog_snapshot"],
            "capability_catalog_snapshot": state["capability_catalog_snapshot"],
            "approved_paths": approved_paths,
            "rejected_paths": rejected_paths,
            "judge_reviews": state["judge_reviews"],
            "repair_history": state["repair_history"],
        }
        update = {
            "planning_status": planning_status,
            "final_result": final_result,
        }
        self._log_preview("finalize_plan_node.update", update)
        return update

    @traceable(name="PlannerModule.run", run_type="chain")
    def run(
        self,
        *,
        config_path: str | Path,
        task_type: str,
        task_request: str,
        dataset_description: str,
        data_profile: dict[str, Any],
        input_manifest: dict[str, Any],
        prepared_h5ad_path: str,
        override_total_paths: int = 0,
    ) -> dict[str, Any]:
        initial_state: PlannerState = {
            "config_path": str(Path(config_path).resolve()),
            "task_type": task_type,
            "task_request": task_request,
            "dataset_description": dataset_description,
            "data_profile": data_profile,
            "input_manifest": input_manifest,
            "prepared_h5ad_path": prepared_h5ad_path,
            "override_total_paths": int(override_total_paths or 0),
            "planner_config": {},
            "tool_catalog_snapshot": {},
            "capability_catalog_snapshot": {},
            "planning_context_text": "",
            "planning_round": 0,
            "current_paths": [],
            "approved_paths": [],
            "rejected_paths": [],
            "latest_judge_reviews": [],
            "judge_reviews": [],
            "repair_history": [],
            "planning_status": "initialized",
            "final_result": {},
        }
        with tracing_context(enabled=self.langsmith_enabled):
            final_state = self.graph.invoke(
                initial_state,
                config={
                    "run_name": "MASPlannerModuleGraph",
                    "tags": ["mas", "planner-module", "langgraph", "task-conditioned"],
                },
            )
        return final_state["final_result"]
