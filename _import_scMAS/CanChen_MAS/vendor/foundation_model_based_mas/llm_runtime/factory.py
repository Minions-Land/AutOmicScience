from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import (
    OpenAICompatibleConfig,
    OpenAICompatibleEmbeddingConfig,
    load_chat_config,
    load_embedding_config,
)


def _transport_kwargs(*, request_timeout: float | None, trust_env: bool) -> dict:
    if trust_env:
        return {}

    import httpx

    timeout = request_timeout if request_timeout is not None else None
    return {
        "http_client": httpx.Client(timeout=timeout, trust_env=False),
        "http_async_client": httpx.AsyncClient(timeout=timeout, trust_env=False),
    }


@lru_cache(maxsize=8)
def _cached_model(
    api_key: str,
    model: str,
    base_url: str | None,
    temperature: float,
    request_timeout: float | None,
    trust_env: bool,
) -> ChatOpenAI:
    config = OpenAICompatibleConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        request_timeout=request_timeout,
        trust_env=trust_env,
    )
    return ChatOpenAI(
        **config.to_chat_kwargs(),
        **_transport_kwargs(
            request_timeout=config.request_timeout,
            trust_env=config.trust_env,
        ),
    )


@lru_cache(maxsize=8)
def _cached_embeddings(
    api_key: str,
    model: str,
    base_url: str | None,
    chunk_size: int,
    skip_empty: bool,
    request_timeout: float | None,
    trust_env: bool,
) -> OpenAIEmbeddings:
    config = OpenAICompatibleEmbeddingConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        chunk_size=chunk_size,
        skip_empty=skip_empty,
        request_timeout=request_timeout,
        trust_env=trust_env,
    )
    return OpenAIEmbeddings(
        **config.to_embedding_kwargs(),
        **_transport_kwargs(
            request_timeout=config.request_timeout,
            trust_env=config.trust_env,
        ),
    )


def build_chat_model(
    *,
    prefix: str = "OPENAI",
    fallback_prefixes: tuple[str, ...] = (),
    default_temperature: float = 0.0,
) -> ChatOpenAI:
    config = load_chat_config(
        prefix=prefix,
        fallback_prefixes=fallback_prefixes,
        default_temperature=default_temperature,
    )
    return _cached_model(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        request_timeout=config.request_timeout,
        trust_env=config.trust_env,
    )


def build_embedding_model(
    *,
    prefix: str = "OPENAI_EMBEDDING",
    fallback_prefixes: tuple[str, ...] = (),
) -> OpenAIEmbeddings:
    config = load_embedding_config(
        prefix=prefix,
        fallback_prefixes=fallback_prefixes,
    )
    return _cached_embeddings(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        chunk_size=config.chunk_size,
        skip_empty=config.skip_empty,
        request_timeout=config.request_timeout,
        trust_env=config.trust_env,
    )


def build_rag_chat_model() -> ChatOpenAI:
    return build_chat_model(
        prefix="RAG_LLM",
        fallback_prefixes=("OPENAI",),
        default_temperature=0.0,
    )


def build_brick_chat_model() -> ChatOpenAI:
    return build_chat_model(
        prefix="BRICK_LLM",
        fallback_prefixes=("RAG_LLM", "OPENAI"),
        default_temperature=0.0,
    )


def build_brick_embedding_model() -> OpenAIEmbeddings:
    return build_embedding_model(
        prefix="BRICK_EMBEDDING",
        fallback_prefixes=("RAG_EMBEDDING", "OPENAI_EMBEDDING"),
    )
