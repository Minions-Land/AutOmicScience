from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS

from BRICK.embedcode import notebook_creator_cell, pycode_creator
from llm_runtime import build_brick_embedding_model

from .registry import DEFAULT_VECTORSTORE_SPECS, VectorstoreSpec, get_vectorstore_spec


def _get_embeddings():
    return build_brick_embedding_model()


def _manifest_path(spec: VectorstoreSpec) -> Path:
    return spec.store_path / "manifest.json"


def _build_code_vectorstore(spec: VectorstoreSpec):
    return pycode_creator(
        data_type=spec.builder_options.get("data_type", "folder"),
        data_path=str(spec.source_path),
        db_name=str(spec.store_path),
        embedding_model=_get_embeddings(),
    )


def _build_notebook_vectorstore(spec: VectorstoreSpec):
    options = dict(spec.builder_options)
    return notebook_creator_cell(
        file_path=str(spec.source_path),
        include_outputs=options.get("include_outputs", False),
        max_output_length=options.get("max_output_length", 20),
        remove_newline=options.get("remove_newline", True),
        db_name=str(spec.store_path),
        embedding_model=_get_embeddings(),
    )


def build_vectorstore(spec: VectorstoreSpec, *, force_rebuild: bool = False) -> Path:
    index_file = spec.store_path / "index.faiss"
    if index_file.exists() and not force_rebuild:
        return spec.store_path

    spec.store_path.mkdir(parents=True, exist_ok=True)
    if spec.kind == "code":
        db = _build_code_vectorstore(spec)
    elif spec.kind == "notebook":
        db = _build_notebook_vectorstore(spec)
    else:
        raise ValueError(f"Unsupported vectorstore kind: {spec.kind}")

    manifest = {
        "name": spec.name,
        "kind": spec.kind,
        "source_path": str(spec.source_path),
        "store_path": str(spec.store_path),
        "description": spec.description,
        "document_count": len(getattr(db, "index_to_docstore_id", {})),
        "embedding_backend": type(_get_embeddings()).__name__,
        "build_mode": "original_brick_embedcode_pipeline",
    }
    _manifest_path(spec).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec.store_path


def load_vectorstore(name: str) -> FAISS:
    spec = get_vectorstore_spec(name)
    build_vectorstore(spec, force_rebuild=False)
    return FAISS.load_local(
        str(spec.store_path),
        embeddings=_get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def ensure_default_vectorstores(*, force_rebuild: bool = False) -> dict[str, Path]:
    built_paths: dict[str, Path] = {}
    for name, spec in DEFAULT_VECTORSTORE_SPECS.items():
        built_paths[name] = build_vectorstore(spec, force_rebuild=force_rebuild)
    return built_paths


def build_default_vectorstores(*, force_rebuild: bool = False) -> dict[str, Path]:
    return ensure_default_vectorstores(force_rebuild=force_rebuild)


def load_default_vectorstores(names: list[str] | None = None) -> dict[str, FAISS]:
    selected_names = names or list(DEFAULT_VECTORSTORE_SPECS.keys())
    return {name: load_vectorstore(name) for name in selected_names}
