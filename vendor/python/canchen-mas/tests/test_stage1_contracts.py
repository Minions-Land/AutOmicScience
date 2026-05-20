from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import io as spio
from scipy import sparse

from scmas import paths
from scmas.data.reference import write_seaad_donor_split
from scmas.eval.registry import artifact_exists, load_model_registry
from scmas.scdesign3.discover import discover_existing_seaad_variants, discover_new_synthetic_variants
from scmas.scdesign3.generate import preflight_scdesign3


def test_seaad_test_and_reference_donors_do_not_overlap(tmp_path, monkeypatch):
    donor_dir = tmp_path / "donor_h5ad"
    donor_dir.mkdir()
    donors = [
        "H20.33.001",
        "H21.33.040",
        "H20.33.015",
        "H20.33.004",
        *[f"REF{i:02d}" for i in range(20)],
    ]
    for donor in donors:
        (donor_dir / f"{donor}.h5ad").touch()
    monkeypatch.setattr(paths, "SEAAD_DONOR_H5AD_DIR", donor_dir)

    split = write_seaad_donor_split(output_path=tmp_path / "seaad_merfish_donor_split.json")
    assert set(split["test_donors"]).isdisjoint(set(split["reference_donors"]))
    assert split["test_donors"] == ["H20.33.001", "H21.33.040", "H20.33.015", "H20.33.004"]
    assert split["n_reference_donors"] >= 20


def test_existing_seaad_synthetic_variants_are_readable(tmp_path):
    manifest = {"variants": []}
    for name in ["baseline_anchor_100k", "variant1_signal80_100k", "variant2_rare0p5pct_100k", "variant3_virtual_batch_100k", "variant4_missing_celltypes_100k"]:
        variant_dir = tmp_path / name
        variant_dir.mkdir()
        spio.mmwrite(variant_dir / "sim_counts.mtx", sparse.coo_matrix(np.ones((2, 2), dtype=np.float32)))
        pd.DataFrame({"Supertype": ["Astro", "Neuron"]}).to_csv(variant_dir / "sim_obs.csv", index=False)
        pd.DataFrame({"feature_id": ["GENEA", "GENEB"]}).to_csv(variant_dir / "sim_var.csv", index=False)
        manifest["variants"].append({"dir": name})
    (tmp_path / "variant_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    variants = discover_existing_seaad_variants(root=tmp_path)
    assert len(variants) == 5
    for item in variants:
        root = Path(item["path"])
        assert (root / "sim_counts.mtx").exists()
        obs = pd.read_csv(root / "sim_obs.csv", nrows=5)
        var = pd.read_csv(root / "sim_var.csv", nrows=5)
        assert "Supertype" in obs.columns
        assert "feature_id" in var.columns


def test_new_synthetic_sources_have_required_variants_if_generated():
    variants = discover_new_synthetic_variants()
    if not variants:
        pytest.skip("No new scDesign3 variants have been generated yet.")
    by_source: dict[str, set[str]] = {}
    for item in variants:
        by_source.setdefault(item["source_id"], set()).add(item["variant_id"])
    required_tokens = ["baseline", "variant1", "variant2"]
    for source_id, names in by_source.items():
        joined = " ".join(names)
        for token in required_tokens:
            assert token in joined, f"{source_id} lacks {token}"


def test_model_registry_entries_have_artifacts_or_clear_reason():
    registry = load_model_registry()
    assert registry
    for spec in registry:
        ok, reason = artifact_exists(spec)
        assert isinstance(ok, bool)
        assert reason


def test_scdesign3_conda_environment_preflight():
    ok, reason = preflight_scdesign3("conda:scdesign3-pipeline")
    assert isinstance(ok, bool)
    assert reason
