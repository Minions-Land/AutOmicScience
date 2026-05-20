def _missing_dependency(*args, **kwargs):
    raise RuntimeError("Required tool-layer dependency is not available in this environment.")


def _missing_optional_tools(kind: str):
    def _fn(names=None):
        if names:
            raise RuntimeError(f"Requested {kind} tools but the required dependency is not available in this environment.")
        return []

    return _fn


try:
    from .adapters.function_adapter import describe_function_tools, get_function_tools
except Exception:  # pragma: no cover - environment-specific optional deps
    describe_function_tools = lambda: []
    get_function_tools = _missing_optional_tools("function")

try:
    from .adapters.mcp_adapter import describe_mcp_servers, get_mcp_tools, get_mcp_tools_sync
except Exception:  # pragma: no cover - environment-specific optional deps
    describe_mcp_servers = lambda: []
    get_mcp_tools = _missing_dependency
    get_mcp_tools_sync = _missing_dependency

try:
    from .adapters.skill_adapter import describe_skill_tools, get_skill_tools
except Exception:  # pragma: no cover - environment-specific optional deps
    try:
        from .skill_tools import SKILL_TOOL_CATALOG, SKILL_TOOLS

        def describe_skill_tools():
            return list(SKILL_TOOL_CATALOG.values())

        def get_skill_tools(names=None):
            selected = SKILL_TOOLS if names is None else [skill for skill in SKILL_TOOLS if skill["name"] in names]
            if names is not None:
                found_names = {skill["name"] for skill in selected}
                missing = [name for name in names if name not in found_names]
                if missing:
                    raise KeyError(f"Unknown skill tools: {missing}")
            return [skill["tool"] for skill in selected]
    except Exception:
        describe_skill_tools = lambda: []
        get_skill_tools = _missing_optional_tools("skill")


def build_langgraph_tools(
    *,
    function_names: list[str] | None = None,
    skill_names: list[str] | None = None,
    mcp_names: list[str] | None = None,
) -> list:
    tools = []
    tools.extend(get_function_tools(function_names))
    tools.extend(get_skill_tools(skill_names))
    if mcp_names:
        tools.extend(get_mcp_tools_sync(mcp_names))
    return tools

def describe_tool_catalog() -> list[dict]:
    return [
        *describe_function_tools(),
        *describe_skill_tools(),
        *describe_mcp_servers(),
    ]


__all__ = [
    'build_langgraph_tools',
    'describe_tool_catalog',
    'get_function_tools',
    'describe_function_tools',
    'get_skill_tools',
    'describe_skill_tools',
    'get_mcp_tools',
    'get_mcp_tools_sync',
    'describe_mcp_servers',
]
