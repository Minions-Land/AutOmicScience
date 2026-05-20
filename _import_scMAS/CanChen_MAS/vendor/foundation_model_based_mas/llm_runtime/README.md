# llm_runtime

This module centralizes all OpenAI-compatible runtime construction in `foundation_model_based_mas`.

## Responsibilities
- load `.env`
- build chat models
- build embedding models
- provide a consistent env naming convention for BRICK and future tools

## Main helpers
- `build_chat_model(...)`
- `build_embedding_model(...)`
- `build_rag_chat_model()`
- `build_brick_chat_model()`
- `build_brick_embedding_model()`

## Recommended env prefixes

### Generic chat
- `OPENAI_*`
- Optional transport controls:
  - `OPENAI_TRUST_ENV=true|false`
  - `OPENAI_TIMEOUT=30`

### Generic embeddings
- `OPENAI_EMBEDDING_*`
- Optional transport controls:
  - `OPENAI_EMBEDDING_TRUST_ENV=true|false`
  - `OPENAI_EMBEDDING_TIMEOUT=30`

### RAG-specific chat / embeddings
- `RAG_LLM_*`
- `RAG_EMBEDDING_*`

### BRICK-specific chat / embeddings
- `BRICK_LLM_*`
- `BRICK_EMBEDDING_*`

## Rule
If a module in this repository needs chat or embedding access, prefer calling this runtime instead of creating its own client directly.
