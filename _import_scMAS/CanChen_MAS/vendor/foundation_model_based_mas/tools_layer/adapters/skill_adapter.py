from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from ..skill_tools import SKILL_TOOL_CATALOG, SKILL_TOOLS


def _normalize_skill_tool(skill_def: dict) -> BaseTool:
    candidate = skill_def['tool']
    if isinstance(candidate, BaseTool):
        return candidate
    if callable(candidate):
        return StructuredTool.from_function(
            func=candidate,
            name=skill_def['name'],
            description=skill_def['description'],
            args_schema=skill_def.get('args_schema'),
            return_direct=skill_def.get('return_direct', False),
        )
    raise TypeError(f"Skill '{skill_def['name']}' must provide a BaseTool or callable.")


def get_skill_tools(names: list[str] | None = None) -> list[BaseTool]:
    selected = SKILL_TOOLS if names is None else [skill for skill in SKILL_TOOLS if skill['name'] in names]
    if names is not None:
        found_names = {skill['name'] for skill in selected}
        missing = [name for name in names if name not in found_names]
        if missing:
            raise KeyError(f'Unknown skill tools: {missing}')
    return [_normalize_skill_tool(skill_def) for skill_def in selected]


def describe_skill_tools() -> list[dict]:
    return list(SKILL_TOOL_CATALOG.values())
