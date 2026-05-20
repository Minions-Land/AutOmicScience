from __future__ import annotations

from pathlib import Path
from typing import Any

from novaeve_bio import paths
from novaeve_bio.io import read_json


EXPECTED_EXISTING_VARIANTS = [
    "baseline_anchor_100k",
    "variant1_signal80_100k",
    "variant2_rare0p5pct_100k",
    "variant3_virtual_batch_100k",
    "variant4_missing_celltypes_100k",
]


def is_scdesign3_variant_dir(path: str | Path) -> bool:
    path = Path(path)
    return all((path / name).exists() for name in ("sim_counts.mtx", "sim_obs.csv", "sim_var.csv"))


def discover_existing_seaad_variants(root: str | Path = paths.EXISTING_SEAAD_SYNTHETIC_ROOT) -> list[dict[str, Any]]:
    root = Path(root)
    variants: list[dict[str, Any]] = []
    manifest_path = root / "variant_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        names = [item["dir"] for item in manifest.get("variants", [])]
    else:
        names = EXPECTED_EXISTING_VARIANTS
    for name in names:
        path = root / name
        if is_scdesign3_variant_dir(path):
            variants.append(
                {
                    "dataset_id": f"existing_seaad__{name}",
                    "source_id": "seaad_anchor_140gene",
                    "variant_id": name,
                    "path": str(path),
                    "orientation": "gene_by_cell",
                    "synthetic": True,
                    "existing": True,
                }
            )
    return variants


def discover_new_synthetic_variants(root: str | Path = paths.SYNTHETIC_DIR) -> list[dict[str, Any]]:
    root = Path(root)
    variants: list[dict[str, Any]] = []
    if not root.exists():
        return variants
    for source_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for variant_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            if not is_scdesign3_variant_dir(variant_dir):
                continue
            variants.append(
                {
                    "dataset_id": f"{source_dir.name}__{variant_dir.name}",
                    "source_id": source_dir.name,
                    "variant_id": variant_dir.name,
                    "path": str(variant_dir),
                    "orientation": "gene_by_cell",
                    "synthetic": True,
                    "existing": False,
                }
            )
    return variants
