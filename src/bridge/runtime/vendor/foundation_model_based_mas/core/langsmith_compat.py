from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

try:
    from langsmith import traceable, tracing_context
except ImportError:
    def traceable(*args: Any, **kwargs: Any) -> Callable:
        def _decorator(fn: Callable) -> Callable:
            return fn
        return _decorator

    @contextmanager
    def tracing_context(*args: Any, **kwargs: Any):
        yield


__all__ = ["traceable", "tracing_context"]
