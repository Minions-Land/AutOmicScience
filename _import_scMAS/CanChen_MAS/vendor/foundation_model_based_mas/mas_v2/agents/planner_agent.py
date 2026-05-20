from __future__ import annotations

from pathlib import Path
from typing import Any

from mas_v2.contracts.schemas import ModelSelectionItem, ModelSelectionPlan, RunProfile
from mas_v2.runtime.artifacts import RunWorkspace, register_artifacts
from mas_v2.runtime.logging import StructuredRunLogger


def _score_model(
    model_id: str,
    *,
    profile: RunProfile,
    intake_query: dict[str, Any],
    capability: dict[str, Any],
    availability: dict[str, Any],
) -> tuple[float, str]:
    score = float(capability.get("task_scores", {}).get("annotation", 0.0))
    reasons: list[str] = [f"base annotation score={score:.3f}"]
    if availability.get("status") == "available":
        score += 1.0
        reasons.append("assets available")
    else:
        score -= 5.0
        reasons.append(f"assets not ready: {availability.get('status')}")

    panel_size = int(intake_query.get("panel_size") or 0)
    spatial = bool(intake_query.get("spatial_keys"))
    species = str(intake_query.get("species_hint", "")).lower()

    if spatial and panel_size and panel_size <= 512:
        if model_id in {"geneformer", "nicheformer"}:
            score += 0.75
            reasons.append("targeted spatial panel fit")
        if model_id == "scgpt_generic":
            score += 0.45
            reasons.append("generic cross-dataset adaptation fit")
    if species and "mouse" in species and model_id == "scgpt_generic":
        score += 0.3
        reasons.append("mouse query favored by generic scGPT adapter")
    if availability.get("reference_mode") == "reference_asset_package":
        score += 1.5
        reasons.append("benchmark asset package available")
    if model_id in profile.planner.excluded_models:
        score -= 10.0
        reasons.append("explicitly excluded by profile")

    return score, "; ".join(reasons)


class PlannerAgent:
    def run(
        self,
        *,
        profile: RunProfile,
        intake_bundle: Any,
        capability_registry: dict[str, dict[str, Any]],
        asset_registry: dict[str, dict[str, Any]],
        workspace: RunWorkspace,
        logger: StructuredRunLogger,
    ) -> ModelSelectionPlan:
        with logger.span("planner_agent.run"):
            candidates = [
                model_id
                for model_id in profile.planner.candidate_models
                if model_id not in profile.planner.excluded_models and model_id in capability_registry
            ]
            ranked: list[tuple[float, str, str]] = []
            for model_id in candidates:
                score, rationale = _score_model(
                    model_id,
                    profile=profile,
                    intake_query=intake_bundle.query.model_dump(),
                    capability=capability_registry[model_id],
                    availability=asset_registry.get(model_id, {}),
                )
                ranked.append((score, model_id, rationale))
            ranked.sort(key=lambda item: (-item[0], item[1]))

            selected_models: list[ModelSelectionItem] = []
            rejected_models: list[dict[str, Any]] = []
            for _, model_id, rationale in ranked:
                availability = asset_registry.get(model_id, {})
                if len(selected_models) < profile.planner.max_selected_models:
                    selected_models.append(
                        ModelSelectionItem(
                            model_id=model_id,
                            priority_rank=len(selected_models) + 1,
                            selection_rationale=rationale,
                            required_reference_mode=str(availability.get("reference_mode", profile.input.reference_source.source_type)),
                            required_query_view=f"h5ad:{profile.input.query_source.preferred_x_source or 'X'}",
                            availability_status=str(availability.get("status", "unknown")),
                            score=float(next(item[0] for item in ranked if item[1] == model_id)),
                        )
                    )
                else:
                    rejected_models.append({"model_id": model_id, "reason": "not_in_top_k", "rationale": rationale})

            plan = ModelSelectionPlan(
                selected_models=selected_models,
                rejected_models=rejected_models,
                candidate_models=candidates,
            )
            output_path = workspace.planner_dir / "model_selection_plan.json"
            output_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            plan.artifacts["model_selection_plan_json"] = str(output_path)
            register_artifacts(workspace.registry_path, "planner", {"model_selection_plan_json": str(output_path)})
            logger.event("planner_agent.completed", payload={"selected_models": [item.model_id for item in selected_models]})
            return plan


class PlannerJudge:
    def run(
        self,
        *,
        plan: ModelSelectionPlan,
        profile: RunProfile,
        capability_registry: dict[str, dict[str, Any]],
        asset_registry: dict[str, dict[str, Any]],
        workspace: RunWorkspace,
        logger: StructuredRunLogger,
    ) -> ModelSelectionPlan:
        with logger.span("planner_judge.run"):
            approved: list[ModelSelectionItem] = []
            rejected = list(plan.rejected_models)
            reviews: list[dict[str, Any]] = []
            for item in plan.selected_models:
                notes: list[str] = []
                capability = capability_registry.get(item.model_id)
                availability = asset_registry.get(item.model_id, {})
                passed = True
                if capability is None:
                    passed = False
                    notes.append("missing capability entry")
                if availability.get("status") != "available":
                    passed = False
                    notes.append(f"asset status={availability.get('status', 'unknown')}")
                if capability and "annotation" not in capability.get("supported_tasks", []):
                    passed = False
                    notes.append("annotation unsupported")
                item.judge_status = "approved" if passed else "rejected"
                item.judge_notes = notes
                reviews.append({"model_id": item.model_id, "passed": passed, "notes": notes})
                if passed:
                    approved.append(item)
                else:
                    rejected.append({"model_id": item.model_id, "reason": "judge_rejected", "notes": notes})

            approved_ids = {item.model_id for item in approved}
            for model_id in plan.candidate_models:
                if len(approved) >= profile.planner.max_selected_models:
                    break
                if model_id in approved_ids:
                    continue
                availability = asset_registry.get(model_id, {})
                capability = capability_registry.get(model_id)
                if capability is None or availability.get("status") != "available":
                    continue
                approved.append(
                    ModelSelectionItem(
                        model_id=model_id,
                        priority_rank=len(approved) + 1,
                        selection_rationale="judge repair inserted first available alternate candidate",
                        required_reference_mode=str(availability.get("reference_mode", profile.input.reference_source.source_type)),
                        required_query_view=f"h5ad:{profile.input.query_source.preferred_x_source or 'X'}",
                        availability_status=str(availability.get("status", "unknown")),
                        judge_status="approved",
                        judge_notes=["repaired_from_candidate_pool"],
                    )
                )
                approved_ids.add(model_id)

            final_plan = ModelSelectionPlan(
                selected_models=approved[: profile.planner.max_selected_models],
                rejected_models=rejected,
                candidate_models=plan.candidate_models,
                judge_reviews=reviews,
                artifacts=dict(plan.artifacts),
            )
            output_path = workspace.planner_dir / "planner_judge_reviews.json"
            output_path.write_text(final_plan.model_dump_json(indent=2), encoding="utf-8")
            final_plan.artifacts["planner_judge_reviews_json"] = str(output_path)
            register_artifacts(workspace.registry_path, "planner_judge", {"planner_judge_reviews_json": str(output_path)})
            logger.event("planner_judge.completed", payload={"approved": [item.model_id for item in final_plan.selected_models]})
            return final_plan
