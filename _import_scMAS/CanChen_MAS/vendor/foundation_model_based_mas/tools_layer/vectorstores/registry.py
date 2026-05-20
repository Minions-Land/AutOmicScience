from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_BIOMICS_ROOT = Path(
    os.getenv("BIOMICS_SOURCE_ROOT", WORKSPACE_ROOT / "BiOmics-master")
).resolve()
LOCAL_BRICK_ROOT = (PROJECT_ROOT / "BRICK").resolve()
LOCAL_NOTEBOOK_ROOT = (PROJECT_ROOT / "notebooks").resolve()
VECTORSTORE_ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


def _prefer_local_path(local_path: Path, fallback_path: Path) -> Path:
    return local_path if local_path.exists() else fallback_path


@dataclass(frozen=True, slots=True)
class VectorstoreSpec:
    name: str
    kind: str
    source_path: Path
    store_path: Path
    description: str
    builder_options: dict = field(default_factory=dict)


DEFAULT_VECTORSTORE_SPECS: dict[str, VectorstoreSpec] = {
    "BRICK_code_local": VectorstoreSpec(
        name="BRICK_code_local",
        kind="code",
        source_path=_prefer_local_path(LOCAL_BRICK_ROOT, (DEFAULT_BIOMICS_ROOT / "BRICK").resolve()),
        store_path=(VECTORSTORE_ARTIFACT_ROOT / "BRICK_code_local").resolve(),
        description="Local FAISS index built from migrated BRICK Python source code.",
        builder_options={"data_type": "folder"},
    ),
    "BRICK_notebook_local": VectorstoreSpec(
        name="BRICK_notebook_local",
        kind="notebook",
        source_path=_prefer_local_path(LOCAL_NOTEBOOK_ROOT, (DEFAULT_BIOMICS_ROOT / "notebooks").resolve()),
        store_path=(VECTORSTORE_ARTIFACT_ROOT / "BRICK_notebook_local").resolve(),
        description="Local FAISS index built from migrated BRICK notebooks.",
        builder_options={
            "include_outputs": True,
            "max_output_length": 400,
            "remove_newline": True,
        },
    ),
}


def get_vectorstore_spec(name: str) -> VectorstoreSpec:
    if name not in DEFAULT_VECTORSTORE_SPECS:
        raise KeyError(
            f"Unknown vectorstore '{name}'. Available names: {sorted(DEFAULT_VECTORSTORE_SPECS)}"
        )
    return DEFAULT_VECTORSTORE_SPECS[name]
