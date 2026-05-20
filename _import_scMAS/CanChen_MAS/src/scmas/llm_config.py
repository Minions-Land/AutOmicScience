from __future__ import annotations

import os
from typing import Any


def first_env(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def default_llm_model(fallback: str = "gpt-5.5") -> str:
    return first_env(("SCMAS_LLM_MODEL", "OPENAI_MODEL", "LLM_MODEL"), fallback)


def build_openai_client() -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"openai package is not available: {type(exc).__name__}: {exc}") from exc

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    timeout = env_float("OPENAI_TIMEOUT", 60.0)
    max_retries = env_int("OPENAI_MAX_RETRIES", 2)
    trust_env = env_bool("OPENAI_TRUST_ENV", True)
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": (os.environ.get("OPENAI_BASE_URL") or "").strip() or None,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    try:
        import httpx

        kwargs["http_client"] = httpx.Client(trust_env=trust_env, timeout=timeout)
    except Exception:
        pass
    return OpenAI(**kwargs)
