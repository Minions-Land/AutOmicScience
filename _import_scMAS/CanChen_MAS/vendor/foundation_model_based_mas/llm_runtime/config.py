from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(slots=True)
class OpenAICompatibleConfig:
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.0
    request_timeout: float | None = None
    trust_env: bool = True

    def to_chat_kwargs(self) -> dict:
        kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.request_timeout is not None:
            kwargs["timeout"] = self.request_timeout
        return kwargs


@dataclass(slots=True)
class OpenAICompatibleEmbeddingConfig:
    api_key: str
    model: str
    base_url: str | None = None
    chunk_size: int = 64
    skip_empty: bool = True
    request_timeout: float | None = None
    trust_env: bool = True

    def to_embedding_kwargs(self) -> dict:
        kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "chunk_size": self.chunk_size,
            "skip_empty": self.skip_empty,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.request_timeout is not None:
            kwargs["timeout"] = self.request_timeout
        return kwargs


def _load_project_env(env_path: str | Path | None = None) -> None:
    target = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if target.exists():
        load_dotenv(target, override=False)


def _get_first_env(keys: list[str]) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def _get_temperature(keys: list[str], default: float) -> float:
    raw = _get_first_env(keys)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_optional_float(keys: list[str]) -> float | None:
    raw = _get_first_env(keys)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _get_bool(keys: list[str], default: bool) -> bool:
    raw = _get_first_env(keys)
    if not raw:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _load_openai_like_config(
    *,
    prefix: str,
    fallback_prefixes: tuple[str, ...],
    env_path: str | Path | None,
    require: bool,
    default_temperature: float,
    include_temperature: bool,
):
    _load_project_env(env_path)

    prefixes = (prefix, *fallback_prefixes)
    api_key = _get_first_env([f"{item}_API_KEY" for item in prefixes])
    model = _get_first_env([f"{item}_MODEL" for item in prefixes])
    base_url = _get_first_env([f"{item}_BASE_URL" for item in prefixes]) or None
    temperature = (
        _get_temperature([f"{item}_TEMPERATURE" for item in prefixes], default_temperature)
        if include_temperature
        else None
    )
    request_timeout = _get_optional_float([f"{item}_TIMEOUT" for item in prefixes])
    trust_env = _get_bool([f"{item}_TRUST_ENV" for item in prefixes], True)

    if require:
        missing = []
        if not api_key:
            missing.append(f"{prefix}_API_KEY")
        if not model:
            missing.append(f"{prefix}_MODEL")
        if missing:
            raise RuntimeError(
                "Missing environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them first."
            )

    return api_key, model, base_url, temperature, request_timeout, trust_env


def load_chat_config(
    *,
    prefix: str = "OPENAI",
    fallback_prefixes: tuple[str, ...] = (),
    env_path: str | Path | None = None,
    require: bool = True,
    default_temperature: float = 0.0,
) -> OpenAICompatibleConfig:
    api_key, model, base_url, temperature, request_timeout, trust_env = _load_openai_like_config(
        prefix=prefix,
        fallback_prefixes=fallback_prefixes,
        env_path=env_path,
        require=require,
        default_temperature=default_temperature,
        include_temperature=True,
    )
    return OpenAICompatibleConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature if temperature is not None else default_temperature,
        request_timeout=request_timeout,
        trust_env=trust_env,
    )


def load_embedding_config(
    *,
    prefix: str = "OPENAI_EMBEDDING",
    fallback_prefixes: tuple[str, ...] = (),
    env_path: str | Path | None = None,
    require: bool = True,
) -> OpenAICompatibleEmbeddingConfig:
    api_key, model, base_url, _, request_timeout, trust_env = _load_openai_like_config(
        prefix=prefix,
        fallback_prefixes=fallback_prefixes,
        env_path=env_path,
        require=require,
        default_temperature=0.0,
        include_temperature=False,
    )
    return OpenAICompatibleEmbeddingConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        chunk_size=64,
        skip_empty=True,
        request_timeout=request_timeout,
        trust_env=trust_env,
    )
