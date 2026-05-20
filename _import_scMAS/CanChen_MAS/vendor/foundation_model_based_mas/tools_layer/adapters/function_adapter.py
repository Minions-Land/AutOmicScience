from __future__ import annotations

from langchain_core.tools import BaseTool

from ..function_tools import FUNCTION_TOOL_CATALOG, FUNCTION_TOOLS


def get_function_tools(names: list[str] | None = None) -> list[BaseTool]:
    if names is None:
        return list(FUNCTION_TOOLS.values())
    missing = [name for name in names if name not in FUNCTION_TOOLS]
    if missing:
        raise KeyError(f'Unknown function tools: {missing}')
    return [FUNCTION_TOOLS[name] for name in names]


def describe_function_tools() -> list[dict]:
    return list(FUNCTION_TOOL_CATALOG.values())
