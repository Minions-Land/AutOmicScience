from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - local compatibility fallback.
    from .langgraph_compat import END, START, StateGraph

from core.langsmith_compat import tracing_context
from core.tracing import bootstrap_langsmith_from_env
from mas_v2.agents import AdapterAgent, AdapterJudge, InputAgent, PlannerAgent, PlannerJudge, ReporterAgent
from mas_v2.contracts.schemas import AdaptationResult, ModelSelectionItem, ModelSelectionPlan, RunProfile
from mas_v2.runtime import (
    StructuredRunLogger,
    build_asset_availability_registry,
    build_run_workspace,
    load_capability_registry,
    load_run_profile,
    register_artifacts,
)
from mas_v2.skills import (
    geneformer_embedding_skill,
    nicheformer_embedding_skill,
    scgpt_generic_brain_embedding_skill,
    scgpt_generic_embedding_skill,
    scgpt_human_embedding_skill,
    shared_knn_transfer_skill,
    shared_prediction_analysis_skill,
    uce_33l_embedding_skill,
    uce_embedding_skill,
)


EMBEDDING_SKILLS = {
    "geneformer": geneformer_embedding_skill,
    "nicheformer": nicheformer_embedding_skill,
    "scgpt_generic": scgpt_generic_embedding_skill,
    "scgpt_human": scgpt_human_embedding_skill,
    "scgpt_generic_brain": scgpt_generic_brain_embedding_skill,
    "uce": uce_embedding_skill,
    "uce_33l": uce_33l_embedding_skill,
}


class PipelineState(TypedDict):
    profile: RunProfile
    workspace: Any
    capability_registry: dict[str, dict[str, Any]]
    asset_registry: dict[str, dict[str, Any]]
    intake_bundle: Any
    model_selection_plan: dict[str, Any]
    model_results: list[dict[str, Any]]
    report_markdown: str


class ExecutorState(TypedDict):
    profile: RunProfile
    workspace: Any
    model_item: dict[str, Any]
    intake_bundle: Any
    adapter_attempt: int
    adapter_feedback: list[str]
    adaptation_result: dict[str, Any]
    adaptation_issues: list[str]
    embedding_package: dict[str, Any]
    knn_result: dict[str, Any]
    analysis_results: list[dict[str, Any]]
    final_result: dict[str, Any]


class MASV2Pipeline:
    def __init__(self, *, project_root: str | Path | None = None) -> None:
        self.project_root = Path(
            project_root
            or os.environ.get(
                "CANCHEN_MAS_FOUNDATION_MAS_ROOT",
                Path(__file__).resolve().parents[2],
            )
        ).expanduser().resolve()
        self.tracing_config = bootstrap_langsmith_from_env(
            self.project_root,
            env_path=self.project_root / ".env",
            default_project="foundation_model_based_mas_mas_v2",
        )
        self.langsmith_enabled = bool(self.tracing_config.get("enabled", False))
        self.input_agent = InputAgent()
        self.planner_agent = PlannerAgent()
        self.planner_judge = PlannerJudge()
        self.adapter_agent = AdapterAgent()
        self.adapter_judge = AdapterJudge()
        self.reporter_agent = ReporterAgent()
        self.executor_graph = self._build_executor_graph()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("input_node", self.input_node)
        graph.add_node("planner_node", self.planner_node)
        graph.add_node("planner_judge_node", self.planner_judge_node)
        graph.add_node("execute_models_node", self.execute_models_node)
        graph.add_node("reporter_node", self.reporter_node)
        graph.add_edge(START, "input_node")
        graph.add_edge("input_node", "planner_node")
        graph.add_edge("planner_node", "planner_judge_node")
        graph.add_edge("planner_judge_node", "execute_models_node")
        graph.add_edge("execute_models_node", "reporter_node")
        graph.add_edge("reporter_node", END)
        return graph.compile()

    def _build_executor_graph(self):
        graph = StateGraph(ExecutorState)
        graph.add_node("adapter_node", self.executor_adapter_node)
        graph.add_node("adapter_judge_node", self.executor_adapter_judge_node)
        graph.add_node("embedding_node", self.executor_embedding_node)
        graph.add_node("knn_node", self.executor_knn_node)
        graph.add_node("analysis_node", self.executor_analysis_node)
        graph.add_node("finalize_node", self.executor_finalize_node)
        graph.add_edge(START, "adapter_node")
        graph.add_edge("adapter_node", "adapter_judge_node")
        graph.add_conditional_edges(
            "adapter_judge_node",
            self._route_after_adapter_judge,
            {
                "adapter_node": "adapter_node",
                "embedding_node": "embedding_node",
                "finalize_node": "finalize_node",
            },
        )
        graph.add_edge("embedding_node", "knn_node")
        graph.add_edge("knn_node", "analysis_node")
        graph.add_edge("analysis_node", "finalize_node")
        graph.add_edge("finalize_node", END)
        return graph.compile()

    def _model_logger(self, workspace: Any, model_id: str) -> StructuredRunLogger:
        log_root = workspace.executor_dir / model_id / "artifacts" / "logs"
        return StructuredRunLogger(root=log_root, component=f"executor.{model_id}")

    def _reference_source_for_model(self, profile: RunProfile, model_id: str) -> Any:
        return profile.input.reference_asset_packages.get(model_id, profile.input.reference_source)

    def input_node(self, state: PipelineState) -> dict[str, Any]:
        logger = StructuredRunLogger(root=state["workspace"].logs_dir / "input", component="input")
        bundle = self.input_agent.run(state["profile"], state["workspace"], logger)
        logger.finalize(status="success", payload={"artifacts": bundle.artifacts})
        return {"intake_bundle": bundle}

    def planner_node(self, state: PipelineState) -> dict[str, Any]:
        logger = StructuredRunLogger(root=state["workspace"].logs_dir / "planner", component="planner")
        plan = self.planner_agent.run(
            profile=state["profile"],
            intake_bundle=state["intake_bundle"],
            capability_registry=state["capability_registry"],
            asset_registry=state["asset_registry"],
            workspace=state["workspace"],
            logger=logger,
        )
        logger.finalize(status="success", payload={"selected_models": [item.model_id for item in plan.selected_models]})
        return {"model_selection_plan": plan.model_dump()}

    def planner_judge_node(self, state: PipelineState) -> dict[str, Any]:
        logger = StructuredRunLogger(root=state["workspace"].logs_dir / "planner_judge", component="planner_judge")
        plan = ModelSelectionPlan.model_validate(state["model_selection_plan"])
        judged = self.planner_judge.run(
            plan=plan,
            profile=state["profile"],
            capability_registry=state["capability_registry"],
            asset_registry=state["asset_registry"],
            workspace=state["workspace"],
            logger=logger,
        )
        logger.finalize(status="success", payload={"approved": [item.model_id for item in judged.selected_models]})
        return {"model_selection_plan": judged.model_dump()}

    def execute_models_node(self, state: PipelineState) -> dict[str, Any]:
        plan = ModelSelectionPlan.model_validate(state["model_selection_plan"])
        model_results: list[dict[str, Any]] = []
        for item in plan.selected_models:
            executor_state: ExecutorState = {
                "profile": state["profile"],
                "workspace": state["workspace"],
                "model_item": item.model_dump(),
                "intake_bundle": state["intake_bundle"],
                "adapter_attempt": 1,
                "adapter_feedback": [],
                "adaptation_result": {},
                "adaptation_issues": [],
                "embedding_package": {},
                "knn_result": {},
                "analysis_results": [],
                "final_result": {},
            }
            result = self.executor_graph.invoke(executor_state)
            model_results.append(result["final_result"])
        register_artifacts(state["workspace"].registry_path, "executor", {"model_results": model_results})
        return {"model_results": model_results}

    def reporter_node(self, state: PipelineState) -> dict[str, Any]:
        logger = StructuredRunLogger(root=state["workspace"].logs_dir / "reporter", component="reporter")
        plan = ModelSelectionPlan.model_validate(state["model_selection_plan"])
        markdown = self.reporter_agent.run(
            profile=state["profile"],
            intake_bundle=state["intake_bundle"],
            plan=plan,
            model_results=state.get("model_results", []),
            workspace=state["workspace"],
            logger=logger,
        )
        logger.finalize(status="success", payload={"report_length": len(markdown)})
        return {"report_markdown": markdown}

    def executor_adapter_node(self, state: ExecutorState) -> dict[str, Any]:
        item = ModelSelectionItem.model_validate(state["model_item"])
        logger = self._model_logger(state["workspace"], item.model_id)
        result = self.adapter_agent.adapt(
            model_item=item,
            profile=state["profile"],
            intake_bundle=state["intake_bundle"],
            workspace=state["workspace"],
            logger=logger,
            feedback=state.get("adapter_feedback", []),
            attempt=state["adapter_attempt"],
            reference_source_override=self._reference_source_for_model(state["profile"], item.model_id),
        )
        return {"adaptation_result": result.model_dump()}

    def executor_adapter_judge_node(self, state: ExecutorState) -> dict[str, Any]:
        result = AdaptationResult.model_validate(state["adaptation_result"])
        issues = self.adapter_judge.validate(result=result, profile=state["profile"])
        update: dict[str, Any] = {"adaptation_issues": issues}
        if issues:
            update["adapter_feedback"] = issues
            update["adapter_attempt"] = state["adapter_attempt"] + 1
        return update

    def _route_after_adapter_judge(self, state: ExecutorState) -> str:
        issues = state.get("adaptation_issues", [])
        if not issues:
            return "embedding_node"
        if state["adapter_attempt"] <= max(1, int(state["profile"].executor.max_adapter_retries)):
            return "adapter_node"
        return "finalize_node"

    def executor_embedding_node(self, state: ExecutorState) -> dict[str, Any]:
        item = ModelSelectionItem.model_validate(state["model_item"])
        adaptation_result = AdaptationResult.model_validate(state["adaptation_result"])
        model_root = state["workspace"].executor_dir / item.model_id
        skill = EMBEDDING_SKILLS[item.model_id]
        embedding_package = skill(
            output_dir=str(model_root),
            run_id=state["workspace"].run_id,
            adaptation_result=adaptation_result,
            reference_source=self._reference_source_for_model(state["profile"], item.model_id),
            reference_label_key=state["profile"].executor.reference_label_key,
            query_label_key="",
            batch_size=state["profile"].executor.batch_size,
            device=state["profile"].executor.device,
            random_seed=state["profile"].executor.sampling.random_seed,
            registry_path=str(state["workspace"].registry_path),
        )
        return {"embedding_package": embedding_package}

    def executor_knn_node(self, state: ExecutorState) -> dict[str, Any]:
        item = ModelSelectionItem.model_validate(state["model_item"])
        embedding = state["embedding_package"]
        model_root = state["workspace"].executor_dir / item.model_id
        result = shared_knn_transfer_skill(
            output_dir=str(model_root),
            run_id=state["workspace"].run_id,
            model_id=item.model_id,
            reference_embeddings_path=embedding["artifacts"]["reference_embeddings_npy"],
            query_embeddings_path=embedding["artifacts"]["query_embeddings_npy"],
            reference_obs_path=embedding["artifacts"]["reference_obs_csv"],
            query_obs_path=embedding["artifacts"]["query_obs_csv"],
            reference_label_key=state["profile"].executor.reference_label_key,
            query_label_key="",
            k=state["profile"].executor.k,
            metric=state["profile"].executor.metric,
            min_vote_share=state["profile"].executor.min_vote_share,
            max_mean_distance=state["profile"].executor.max_mean_distance,
            registry_path=str(state["workspace"].registry_path),
        )
        return {"knn_result": result}

    def executor_analysis_node(self, state: ExecutorState) -> dict[str, Any]:
        item = ModelSelectionItem.model_validate(state["model_item"])
        analysis_results: list[dict[str, Any]] = []
        for run_cfg in state["profile"].analysis.runs:
            result = shared_prediction_analysis_skill(
                output_dir=str(state["workspace"].executor_dir / item.model_id),
                run_id=state["workspace"].run_id,
                model_id=item.model_id,
                run_name=run_cfg.run_name,
                prediction_csv_path=state["knn_result"]["artifacts"]["prediction_csv"],
                query_obs_path=state["embedding_package"]["artifacts"]["query_obs_csv"],
                reference_label_key=state["profile"].executor.reference_label_key,
                query_label_key=run_cfg.query_label_key,
                reference_eval_mapping=run_cfg.reference_eval_mapping,
                query_eval_mapping=run_cfg.query_eval_mapping,
                registry_path=str(state["workspace"].registry_path),
            )
            analysis_results.append(result)
        return {"analysis_results": analysis_results}

    def executor_finalize_node(self, state: ExecutorState) -> dict[str, Any]:
        item = ModelSelectionItem.model_validate(state["model_item"])
        embedding_status = state.get("embedding_package", {}).get("status", "")
        knn_status = state.get("knn_result", {}).get("status", "")
        analysis_results = state.get("analysis_results", [])
        analysis_failed = any(result.get("status") == "failed" for result in analysis_results)
        status = "success"
        if state.get("adaptation_issues") or embedding_status == "failed" or knn_status == "failed" or analysis_failed:
            status = "failed"
        final_result = {
            "model_id": item.model_id,
            "status": status,
            "adaptation_issues": list(state.get("adaptation_issues", [])),
            "adaptation_result": state.get("adaptation_result", {}),
            "embedding_package": state.get("embedding_package", {}),
            "knn_result": state.get("knn_result", {}),
            "analysis_results": analysis_results,
            "artifacts": {
                "model_root": str((state["workspace"].executor_dir / item.model_id).resolve()),
            },
        }
        return {"final_result": final_result}

    def run(self, profile_path: str | Path, *, run_id: str = "") -> dict[str, Any]:
        profile = load_run_profile(profile_path)
        return self.run_profile(profile, run_id=run_id)

    def run_profile(self, profile: RunProfile, *, run_id: str = "") -> dict[str, Any]:
        output_root = Path(profile.logging.output_root)
        if not output_root.is_absolute():
            output_root = (self.project_root / output_root).resolve()
        workspace = build_run_workspace(output_root, run_id=run_id)
        capability_registry = load_capability_registry()
        asset_registry = build_asset_availability_registry(profile, capability_registry)
        state: PipelineState = {
            "profile": profile,
            "workspace": workspace,
            "capability_registry": capability_registry,
            "asset_registry": asset_registry,
            "intake_bundle": {},
            "model_selection_plan": {},
            "model_results": [],
            "report_markdown": "",
        }
        with tracing_context(enabled=self.langsmith_enabled):
            final_state = self.graph.invoke(state, config={"run_name": "MASV2Pipeline", "tags": ["mas-v2", "langgraph"]})

        final_state_path = workspace.root / "final_state.json"
        persisted_state = {
            "profile": profile.model_dump(mode="json"),
            "workspace": {
                "run_id": workspace.run_id,
                "root": str(workspace.root),
                "input_dir": str(workspace.input_dir),
                "planner_dir": str(workspace.planner_dir),
                "executor_dir": str(workspace.executor_dir),
                "report_dir": str(workspace.report_dir),
                "logs_dir": str(workspace.logs_dir),
                "registry_path": str(workspace.registry_path),
            },
            "capability_registry": capability_registry,
            "asset_registry": asset_registry,
            "intake_bundle": final_state.get("intake_bundle", {}).model_dump(mode="json") if hasattr(final_state.get("intake_bundle"), "model_dump") else final_state.get("intake_bundle", {}),
            "model_selection_plan": final_state.get("model_selection_plan", {}),
            "model_results": final_state.get("model_results", []),
            "report_markdown": final_state.get("report_markdown", ""),
        }
        final_state_path.write_text(json.dumps(persisted_state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if final_state.get("report_markdown"):
            (workspace.report_dir / "final_report.md").write_text(final_state["report_markdown"], encoding="utf-8")
        register_artifacts(
            workspace.registry_path,
            "pipeline",
            {
                "final_state_json": str(final_state_path),
                "report_md": str(workspace.report_dir / "final_report.md"),
            },
        )
        return final_state
