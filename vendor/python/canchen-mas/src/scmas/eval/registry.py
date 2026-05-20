from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scmas.io import read_yaml
from scmas import paths


DEFAULT_REGISTRY = paths.SCMAS_ROOT / "configs" / "model_registry.yaml"

_PATH_KEYS = {"checkpoint", "checkpoint_dir", "entrypoint", "runtime_conda_prefix"}


def _default_env_paths() -> dict[str, Path]:
    return {
        "CANCHEN_MAS_ROOT": paths.SCMAS_ROOT,
        "CANCHEN_MAS_FOUNDATION_MAS_ROOT": paths.LEGACY_ROOT,
        "CANCHEN_MAS_FOUNDATION_CHECKPOINT_ROOT": paths.FOUNDATION_CHECKPOINT_ROOT,
        "CANCHEN_MAS_GENEFORMER_DIR": paths.GENEFORMER_CHECKPOINT_DIR,
        "CANCHEN_MAS_SCGPT_DIR": paths.SCGPT_CHECKPOINT_ROOT,
        "CANCHEN_MAS_NICHEFORMER_DIR": paths.NICHEFORMER_CHECKPOINT_DIR,
        "CANCHEN_MAS_UCE_4L_DIR": paths.UCE_4L_MODEL_DIR,
        "CANCHEN_MAS_UCE_33L_DIR": paths.UCE_33L_MODEL_DIR,
    }


def expand_path_text(value: str | Path) -> str:
    text = str(value)
    for name, default in _default_env_paths().items():
        replacement = os.environ.get(name, str(default))
        text = text.replace(f"${{{name}}}", replacement)
        text = text.replace(f"${name}", replacement)
    return os.path.expandvars(os.path.expanduser(text))


def resolve_portable_path(value: str | Path, *, base_dir: str | Path = paths.SCMAS_ROOT) -> Path:
    path = Path(expand_path_text(value))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


@dataclass
class ModelSpec:
    model_id: str
    family: str
    evaluator: str
    raw: dict[str, Any]


def load_model_registry(path: str | Path = DEFAULT_REGISTRY) -> list[ModelSpec]:
    data = read_yaml(path)
    out: list[ModelSpec] = []
    for item in data.get("models", []):
        raw = dict(item)
        for key in _PATH_KEYS:
            if raw.get(key):
                raw[key] = str(resolve_portable_path(raw[key], base_dir=paths.SCMAS_ROOT))
        out.append(
            ModelSpec(
                model_id=raw["model_id"],
                family=raw.get("family", "unknown"),
                evaluator=raw.get("evaluator", "unknown"),
                raw=raw,
            )
        )
    return out


def artifact_exists(spec: ModelSpec) -> tuple[bool, str]:
    raw = spec.raw
    for key in ("checkpoint", "checkpoint_dir", "entrypoint"):
        if raw.get(key):
            path = resolve_portable_path(raw[key])
            if not path.exists():
                return False, f"{key} not found: {path}"
    if spec.evaluator == "sklearn_pkl":
        root = resolve_portable_path(raw["checkpoint_dir"])
        missing = [f"{task}_model.pkl" for task in ("class", "subclass", "supertype") if not (root / f"{task}_model.pkl").exists()]
        if missing:
            return False, f"missing sklearn model files: {', '.join(missing)}"
    return True, "ok"
