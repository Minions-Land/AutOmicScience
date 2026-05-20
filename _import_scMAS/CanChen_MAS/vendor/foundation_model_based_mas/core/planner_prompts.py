from __future__ import annotations

import json
from typing import Any


def render_input_to_planner_context(
    *,
    task_type: str,
    task_request: str,
    dataset_description: str,
    data_profile: dict[str, Any],
    input_manifest: dict[str, Any],
    prepared_h5ad_path: str,
    tool_catalog_snapshot: dict[str, Any],
    capability_catalog_snapshot: dict[str, Any],
) -> str:
    active_split = input_manifest.get("active_split", "default")
    split_file_counts = input_manifest.get("split_file_counts", {})
    available_splits = input_manifest.get("available_splits", [])
    return (
        f"Task type:\n{task_type}\n\n"
        f"Task request:\n{task_request}\n\n"
        f"Dataset description:\n{dataset_description}\n\n"
        f"Prepared input path:\n{prepared_h5ad_path}\n\n"
        f"Active split:\n{active_split}\n\n"
        f"Available splits:\n{json.dumps(available_splits, ensure_ascii=False, indent=2)}\n\n"
        f"Split file counts:\n{json.dumps(split_file_counts, ensure_ascii=False, indent=2)}\n\n"
        f"Structured data portrait:\n{json.dumps(data_profile, ensure_ascii=False, indent=2)}\n\n"
        f"Input manifest:\n{json.dumps(input_manifest, ensure_ascii=False, indent=2)}\n\n"
        f"Tool catalog snapshot:\n{json.dumps(tool_catalog_snapshot, ensure_ascii=False, indent=2)}\n\n"
        f"Capability catalog snapshot:\n{json.dumps(capability_catalog_snapshot, ensure_ascii=False, indent=2)}"
    )


def build_planner_system_prompt(*, traditional_path_count: int, foundation_model_path_count: int) -> str:
    total_paths = traditional_path_count + foundation_model_path_count
    return (
        "You are the Planner Agent in a task-conditioned bioinformatics MAS. "
        "Your job is pure planning only. Do not execute anything. "
        "You must produce exactly "
        f"{total_paths} planning paths: {traditional_path_count} traditional path and "
        f"{foundation_model_path_count} foundation-model paths. "
        "A traditional path must be task-conditioned and must not hardcode annotation-only logic. "
        "A foundation-model path must select exactly one MCP tool that exists in the capability files. "
        "Every foundation-model path must include selected_mcp_tool_name. "
        "Do not force execution details into the plan: the executor will resolve the server, parameters, and runtime defaults. "
        "selected_model_id is optional metadata only and may be omitted. "
        "plan_steps and invocation_outline are optional for foundation-model paths. "
        "You may use tools to inspect capability files. "
        "Do not invent tools, servers, models, or model capabilities. "
        "Every output path must be structured and self-consistent. "
        "If you receive repair instructions, only regenerate the failed paths and preserve approved paths."
    )


def build_planner_user_prompt(
    *,
    planning_context: str,
    planning_round: int,
    failed_path_ids: list[str] | None = None,
    repair_instructions: list[dict[str, Any]] | None = None,
    approved_paths: list[dict[str, Any]] | None = None,
    generation_target_count: int,
) -> str:
    if not failed_path_ids:
        return (
            f"Planning round: {planning_round}\n\n"
            f"Generate {generation_target_count} paths.\n\n"
            "Output contract for every foundation-model path:\n"
            "- selected_mcp_tool_name is required.\n"
            "- selected_model_id is optional.\n"
            "- plan_steps are optional and executor will synthesize execution from selected_mcp_tool_name.\n"
            "- invocation_outline is optional and may be empty.\n\n"
            f"{planning_context}"
        )

    return (
        f"Planning round: {planning_round}\n\n"
        "Regenerate only the failed paths listed below. "
        f"Return exactly {generation_target_count} replacement paths.\n\n"
        "Keep the same output contract:\n"
        "- selected_mcp_tool_name is required.\n"
        "- selected_model_id is optional.\n"
        "- plan_steps are optional and executor will synthesize execution from selected_mcp_tool_name.\n"
        "- invocation_outline is optional and may be empty.\n\n"
        f"Approved paths that must remain unchanged:\n{json.dumps(approved_paths or [], ensure_ascii=False, indent=2)}\n\n"
        f"Failed path ids:\n{json.dumps(failed_path_ids, ensure_ascii=False, indent=2)}\n\n"
        f"Repair instructions:\n{json.dumps(repair_instructions or [], ensure_ascii=False, indent=2)}\n\n"
        f"{planning_context}"
    )


PLAN_JUDGE_SYSTEM_PROMPT = (
    "You are the Plan Judge Agent in a task-conditioned bioinformatics MAS. "
    "You have the same tools and permissions as the planner. "
    "You must review each planning path against the input context, tool catalog, and capability files. "
    "For a foundation-model path to pass, it must be executable: selected_mcp_tool_name must exist in the capability files, "
    "the selected capability must support the current task_type, and the selected MCP tool must exist in the MCP catalog. "
    "For every path, output exactly three scores, each on a 0-6 scale: "
    "task_match_score, data_match_score, planning_quality_score. "
    "Then compute final_score as the arithmetic mean of the three scores. "
    "A path passes if final_score >= 4.0; otherwise it fails. "
    "For failed paths, provide concrete rejection reasons and a repair instruction that the planner can act on. "
    "Do not change approved paths; only explain whether each path passes."
)


def build_plan_judge_user_prompt(*, planning_context: str, candidate_paths: list[dict[str, Any]]) -> str:
    return (
        "Review the following candidate planning paths.\n\n"
        f"{planning_context}\n\n"
        f"Candidate paths:\n{json.dumps(candidate_paths, ensure_ascii=False, indent=2)}"
    )
