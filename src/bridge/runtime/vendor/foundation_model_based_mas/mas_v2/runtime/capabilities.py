from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from mas_v2.contracts.schemas import ReferenceAssetPackage, RunProfile


PROJECT_ROOT = Path(
    os.environ.get("CANCHEN_MAS_FOUNDATION_MAS_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
CHECKPOINT_ROOT = Path(
    os.environ.get("CANCHEN_MAS_FOUNDATION_CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints" / "foundation_models")
).expanduser().resolve()
CAPABILITY_DIR = Path(os.environ.get("CANCHEN_MAS_VENDOR_CAPABILITY_DIR", PROJECT_ROOT / "config" / "capability"))
GF_CODE_DIR = Path(os.environ.get("CANCHEN_MAS_GENEFORMER_CODE_DIR", CHECKPOINT_ROOT / "geneformer" / "Geneformer_code_only"))
GF_WEIGHTS = Path(os.environ.get("CANCHEN_MAS_GENEFORMER_WEIGHTS", CHECKPOINT_ROOT / "geneformer" / "model.safetensors"))
NICHEFORMER_DIR = Path(os.environ.get("CANCHEN_MAS_NICHEFORMER_DIR", CHECKPOINT_ROOT / "nicheformer"))
SCGPT_CHECKPOINT_ROOT = Path(os.environ.get("CANCHEN_MAS_SCGPT_DIR", CHECKPOINT_ROOT / "scgpt"))


def load_capability_registry(capability_dir: str | Path = CAPABILITY_DIR) -> dict[str, dict[str, Any]]:
    root = Path(capability_dir).resolve()
    registry: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict) and payload.get("model_id"):
            registry[str(payload["model_id"])] = payload
    return registry


def _raw_model_asset_status(model_id: str) -> dict[str, Any]:
    normalized = model_id.strip().lower()
    if normalized == "geneformer":
        required = [GF_CODE_DIR, GF_WEIGHTS]
    elif normalized == "nicheformer":
        required = [NICHEFORMER_DIR / "nicheformer.ckpt", NICHEFORMER_DIR / "model.h5ad", NICHEFORMER_DIR / "gene_name_id_dict_gc104M.pkl"]
    elif normalized == "scgpt_generic":
        required = [SCGPT_CHECKPOINT_ROOT / "human" / "best_model.pt", SCGPT_CHECKPOINT_ROOT / "human" / "vocab.json", SCGPT_CHECKPOINT_ROOT / "human" / "args.json"]
    else:
        required = []
    missing = [str(path) for path in required if not Path(path).exists()]
    return {
        "status": "available" if not missing else "missing_assets",
        "missing_paths": missing,
    }


def _asset_package_from_source(profile: RunProfile, model_id: str) -> ReferenceAssetPackage | None:
    mapped = profile.input.reference_asset_packages.get(model_id)
    if mapped is not None:
        return mapped
    source = profile.input.reference_source
    if source.source_type != "reference_asset_package":
        return None
    if source.model_id.strip().lower() != model_id.strip().lower():
        return None
    return source


def build_asset_availability_registry(profile: RunProfile, capability_registry: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for model_id in capability_registry:
        package = _asset_package_from_source(profile, model_id)
        if package is not None:
            missing = [
                path
                for path in [
                    package.reference_embeddings_path,
                    package.reference_obs_path,
                    package.coverage_json,
                    package.source_manifest,
                ]
                if not Path(path).exists()
            ]
            registry[model_id] = {
                "reference_mode": "reference_asset_package",
                "status": "available" if not missing else "missing_assets",
                "missing_paths": missing,
                "package": json.loads(package.model_dump_json()),
            }
            continue
        raw_status = _raw_model_asset_status(model_id)
        raw_status["reference_mode"] = profile.input.reference_source.source_type
        registry[model_id] = raw_status
    return registry
