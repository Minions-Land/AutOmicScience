from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class Input(BaseModel):
    directory_path: str
    glob_pattern: str = Field(default="**/*.yaml")


def _normalize_model_payload(raw: dict[str, Any], file_path: Path) -> dict[str, Any]:
    payload = dict(raw)
    payload["source_path"] = str(file_path.resolve())
    payload.setdefault("model_id", file_path.stem)
    payload.setdefault("display_name", payload["model_id"])
    payload.setdefault("supported_tasks", [])
    payload.setdefault("modalities", [])
    payload.setdefault("assay_technologies", [])
    payload.setdefault("species_scope", [])
    payload.setdefault("tissue_scope", [])
    payload.setdefault("mcp_server_name", "")
    payload.setdefault("mcp_tool_name", "")
    payload.setdefault("annotation_profile", {})
    payload.setdefault("executor_defaults", {})
    return payload


@tool(args_schema=Input)
def read_capability_files_tool(directory_path: str, glob_pattern: str = "**/*.yaml") -> dict[str, Any]:
    """Read capability YAML files and build a model index for planner and executor use."""
    directory = Path(directory_path).expanduser().resolve()
    if not directory.exists():
        return {
            "success": False,
            "summary": f"Capability directory does not exist: {directory}",
            "models": [],
            "model_index": {},
            "invalid_files": [],
        }
    if not directory.is_dir():
        return {
            "success": False,
            "summary": f"Capability path is not a directory: {directory}",
            "models": [],
            "model_index": {},
            "invalid_files": [],
        }

    models: list[dict[str, Any]] = []
    model_index: dict[str, dict[str, Any]] = {}
    invalid_files: list[dict[str, str]] = []

    for file_path in sorted(directory.glob(glob_pattern)):
        if not file_path.is_file():
            continue
        try:
            raw_payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw_payload, dict):
                raise TypeError("Capability YAML must deserialize to a mapping.")
            normalized = _normalize_model_payload(raw_payload, file_path)
            model_id = str(normalized["model_id"]).strip()
            if not model_id:
                raise ValueError("Capability YAML is missing a non-empty model_id.")
            models.append(normalized)
            model_index[model_id] = normalized
        except Exception as exc:
            invalid_files.append(
                {
                    "source_path": str(file_path.resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = (
        f"Loaded {len(models)} capability file(s) from {directory}."
        if models
        else f"No valid capability files were loaded from {directory}."
    )
    if invalid_files:
        summary += f" {len(invalid_files)} file(s) were invalid."

    return {
        "success": bool(models),
        "summary": summary,
        "directory_path": str(directory),
        "glob_pattern": glob_pattern,
        "models": models,
        "model_index": model_index,
        "invalid_files": invalid_files,
    }


TOOLS = [read_capability_files_tool]
TOOL_CATALOG = [
    {
        "name": "read_capability_files_tool",
        "kind": "function",
        "description": "Read capability YAML files and return a normalized capability catalog for planner/executor use.",
        "source_repo": "foundation_model_based_mas",
        "source_file": "tools_layer/function_tools/read_capability_files.py",
    }
]
