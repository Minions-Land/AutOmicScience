from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import ReferenceAssetPackage
from .io_utils import read_json


def load_reference_asset_package(package_path: str | Path) -> ReferenceAssetPackage:
    resolved = Path(package_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"reference_asset_package path not found: {resolved}")
    payload = read_json(resolved)
    package = ReferenceAssetPackage(**payload)
    return package


def reference_asset_payload(package: ReferenceAssetPackage) -> dict[str, Any]:
    return {
        "model_id": package.model_id,
        "reference_embeddings_path": str(Path(package.reference_embeddings_path).resolve()),
        "reference_obs_path": str(Path(package.reference_obs_path).resolve()),
        "coverage_json": str(Path(package.coverage_json).resolve()) if package.coverage_json else "",
        "source_manifest": str(Path(package.source_manifest).resolve()) if package.source_manifest else "",
        "reference_label_key": package.reference_label_key,
        "dataset_fingerprint": package.dataset_fingerprint,
    }

