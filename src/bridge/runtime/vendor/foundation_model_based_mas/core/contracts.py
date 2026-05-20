from __future__ import annotations

import json
import re
from typing import Any


def resolve_path_mix_counts(
    *,
    traditional_path_count: int,
    foundation_model_path_count: int,
    override_total_paths: int = 0,
) -> tuple[int, int]:
    traditional_count = max(0, int(traditional_path_count))
    foundation_count = max(0, int(foundation_model_path_count))
    if not override_total_paths:
        return traditional_count, foundation_count

    total_paths = max(1, int(override_total_paths))
    traditional_count = min(traditional_count, total_paths)
    foundation_count = max(0, total_paths - traditional_count)
    return traditional_count, foundation_count


def coerce_invocation_outline(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    if value is None:
        return {}

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}

        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        candidates.extend(item.strip() for item in fenced if item.strip())
        brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(1).strip())

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return dict(parsed)

        return {"notes": text}

    if isinstance(value, list):
        return {"items": list(value)}

    return {"raw_value": value}


def normalize_server_name(server_name: str, server_catalog: dict[str, dict[str, Any]]) -> str:
    if not server_name:
        return ""
    if server_name in server_catalog:
        return server_name
    normalized = server_name.strip().lower()
    for known_name in server_catalog:
        if known_name.lower() == normalized:
            return known_name
    return ""


def resolve_foundation_model_selection(
    *,
    path: dict[str, Any],
    capability_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str, str, str]:
    selected_model_id = str(path.get("selected_model_id", "")).strip()
    selected_mcp_tool_name = str(path.get("selected_mcp_tool_name", "")).strip()

    if not selected_mcp_tool_name:
        for step in list(path.get("plan_steps", [])):
            candidate_tool = str(step.get("candidate_tool", "")).strip()
            if candidate_tool:
                selected_mcp_tool_name = candidate_tool
                break

    capability = capability_index.get(selected_model_id) if selected_model_id else None
    if capability is not None:
        capability_tool_name = str(capability.get("mcp_tool_name", "")).strip()
        if selected_mcp_tool_name and capability_tool_name and capability_tool_name != selected_mcp_tool_name:
            return None, "selected_model_tool_mismatch", selected_model_id, selected_mcp_tool_name
        if not selected_mcp_tool_name:
            selected_mcp_tool_name = capability_tool_name
        return capability, "resolved", selected_model_id, selected_mcp_tool_name

    if selected_model_id and not selected_mcp_tool_name:
        return None, "missing_capability", selected_model_id, selected_mcp_tool_name
    if not selected_mcp_tool_name:
        return None, "missing_selected_mcp_tool_name", selected_model_id, selected_mcp_tool_name

    matches = [
        capability
        for capability in capability_index.values()
        if str(capability.get("mcp_tool_name", "")).strip() == selected_mcp_tool_name
    ]
    if not matches:
        return None, "unknown_mcp_tool_name", selected_model_id, selected_mcp_tool_name
    if len(matches) > 1:
        return None, "ambiguous_mcp_tool_name", selected_model_id, selected_mcp_tool_name

    capability = matches[0]
    resolved_model_id = str(capability.get("model_id", "")).strip()
    return capability, "resolved", resolved_model_id, selected_mcp_tool_name


def classify_path_execution(
    *,
    path: dict[str, Any],
    task_type: str,
    capability_index: dict[str, dict[str, Any]],
    server_catalog: dict[str, dict[str, Any]],
) -> tuple[bool, str, dict[str, Any] | None]:
    if path.get("path_type") != "foundation_model":
        return False, "non_foundation_model_path", None
    if str(path.get("task_type", "")).strip() != str(task_type).strip():
        return False, "task_type_mismatch", None

    capability, selection_reason, _, _ = resolve_foundation_model_selection(
        path=path,
        capability_index=capability_index,
    )
    if capability is None:
        return False, selection_reason, None

    supported_tasks = {str(item).strip() for item in capability.get("supported_tasks", [])}
    if task_type not in supported_tasks:
        return False, "capability_does_not_support_task", capability

    if task_type != "annotation":
        return False, "unsupported_task", capability

    mcp_server = normalize_server_name(str(capability.get("mcp_server_name", "")).strip(), server_catalog)
    mcp_tool = str(capability.get("mcp_tool_name", "")).strip()
    if not mcp_server or not mcp_tool:
        return False, "missing_mcp_endpoint", capability

    server_tools = set(server_catalog.get(mcp_server, {}).get("tool_names", []))
    if mcp_tool not in server_tools:
        return False, "invalid_mcp_endpoint", capability

    return True, "runnable", capability
