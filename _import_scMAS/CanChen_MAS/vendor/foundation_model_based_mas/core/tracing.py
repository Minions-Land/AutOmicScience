from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    _load_dotenv = None


LANGSMITH_API_KEY_KEYS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
LANGSMITH_ENDPOINT_KEYS = ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
LANGSMITH_PROJECT_KEYS = ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")


def _first_non_empty(keys: tuple[str, ...]) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def _parse_optional_bool(raw: str) -> bool | None:
    value = str(raw or "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip().lstrip("\ufeff")
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def _load_env_fallback(path: Path, *, override: bool = True) -> None:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_env_assignment(line)
        if not parsed:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value


def load_project_env(project_root: str | Path, env_path: str | Path | None = None) -> Path:
    project_root = Path(project_root).resolve()
    target = Path(env_path).resolve() if env_path else project_root / ".env"
    if target.exists():
        if _load_dotenv is not None:
            _load_dotenv(target, override=True)
        else:
            _load_env_fallback(target, override=True)
    return target


def bootstrap_langsmith_from_env(
    project_root: str | Path,
    *,
    env_path: str | Path | None = None,
    default_project: str | None = None,
    enable_tracing: bool = True,
) -> dict[str, Any]:
    env_file = load_project_env(project_root, env_path=env_path)
    api_key = _first_non_empty(LANGSMITH_API_KEY_KEYS)
    endpoint = _first_non_empty(LANGSMITH_ENDPOINT_KEYS)
    project = _first_non_empty(LANGSMITH_PROJECT_KEYS) or default_project or Path(project_root).resolve().name
    tracing_override = _parse_optional_bool(os.getenv("LANGSMITH_TRACING", "")) 
    if tracing_override is None:
        tracing_override = _parse_optional_bool(os.getenv("LANGCHAIN_TRACING_V2", ""))

    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
    if project:
        os.environ["LANGSMITH_PROJECT"] = project

    enabled = bool(enable_tracing and api_key)
    if tracing_override is not None:
        enabled = bool(enabled and tracing_override)
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else os.getenv("LANGCHAIN_TRACING_V2", "false")

    return {
        "enabled": enabled,
        "env_file": str(env_file),
        "project": project,
        "endpoint_configured": bool(endpoint),
    }
