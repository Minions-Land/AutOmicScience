from __future__ import annotations

import importlib
import pkgutil

FUNCTION_TOOLS: dict[str, object] = {}
FUNCTION_TOOL_CATALOG: dict[str, dict] = {}

for module_info in pkgutil.iter_modules(__path__, prefix=f'{__name__}.'):
    try:
        module = importlib.import_module(module_info.name)
    except Exception:
        continue
    for tool in getattr(module, 'TOOLS', []):
        FUNCTION_TOOLS[tool.name] = tool
    for item in getattr(module, 'TOOL_CATALOG', []):
        FUNCTION_TOOL_CATALOG[item['name']] = item

__all__ = ['FUNCTION_TOOLS', 'FUNCTION_TOOL_CATALOG']
