from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


REPO_ROOT = Path(__file__).resolve().parents[2]

SCMAS_ROOT = _path_from_env("CANCHEN_MAS_ROOT", REPO_ROOT)
LEGACY_ROOT = _path_from_env(
    "CANCHEN_MAS_FOUNDATION_MAS_ROOT",
    SCMAS_ROOT / "vendor" / "foundation_model_based_mas",
)
LEGACY_DATA_ROOT = LEGACY_ROOT / "data"
SEA_AD_MJM_ROOT = _path_from_env("CANCHEN_MAS_SEAAD_MJM_ROOT", SCMAS_ROOT / "external" / "SEA-AD" / "MJM")

SC_DESIGN3_ROOT = LEGACY_ROOT / "FM_eval_hub" / "scDesign3-main"
SC_DESIGN3_PIPELINE = SC_DESIGN3_ROOT / "pipeline"
SC_DESIGN3_ANCHOR_RUNNER = SC_DESIGN3_PIPELINE / "run_scdesign3_anchor_donor_variants.py"
EXISTING_SEAAD_SYNTHETIC_ROOT = (
    SC_DESIGN3_ROOT / "pipeline_runs" / "anchor_donor_variants_140gene"
)

SEAAD_MERFISH_H5AD = _path_from_env(
    "CANCHEN_MAS_SEAAD_MERFISH_H5AD",
    SCMAS_ROOT / "data" / "raw" / "seaad_merfish.h5ad",
)
SEAAD_DONOR_H5AD_DIR = _path_from_env(
    "CANCHEN_MAS_SEAAD_DONOR_H5AD_DIR",
    SCMAS_ROOT / "data" / "raw" / "seaad_donor_h5ad",
)
SEAAD_MERFISH_GENES_JSON = SEA_AD_MJM_ROOT / "data" / "merfish_gene_names.json"
SEAAD_GLOBAL_NPZ = SEA_AD_MJM_ROOT / "data" / "global_data.npz"

DATA_DIR = SCMAS_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
TEST_DIR = DATA_DIR / "test"
PREPARED_SOURCE_DIR = DATA_DIR / "prepared_sources"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RUNS_DIR = SCMAS_ROOT / "runs"

REFERENCE_H5AD = REFERENCE_DIR / "scmas_human_mouse_reference.h5ad"
SEAAD_TEST_H5AD = TEST_DIR / "seaad_merfish_140gene_test.h5ad"
SEAAD_LABEL_MAPS_JSON = REFERENCE_DIR / "seaad_label_maps.json"
SEAAD_DONOR_SPLIT_JSON = REFERENCE_DIR / "seaad_merfish_donor_split.json"
SOURCE_MANIFEST_JSON = PREPARED_SOURCE_DIR / "source_manifest.json"

KUKANJA_MS_NPZ = _path_from_env("CANCHEN_MAS_KUKANJA_MS_NPZ", DATA_DIR / "query" / "kukanja_ms.npz")
KUKANJA_EAE_NPZ = _path_from_env("CANCHEN_MAS_KUKANJA_EAE_NPZ", DATA_DIR / "query" / "kukanja_eae.npz")

ALLEN_WHOLE_BRAIN_DIR = LEGACY_DATA_ROOT / "reference" / "Allen_whole_brain"
ALLEN_MOUSE_REFERENCE_DIR = LEGACY_DATA_ROOT / "allen_mouse_reference"
ALLEN_HUMAN_SMARTSEQ_DIR = LEGACY_DATA_ROOT / "allen_human_multiple_cortical_areas_smartseq"
HUMAN_WHOLE_BRAIN_DIR = LEGACY_DATA_ROOT / "reference" / "human_whole_brain"
MOUSE_WHOLE_BRAIN_DIR = LEGACY_DATA_ROOT / "reference" / "mouse_whole_brain"
SPINAL_DIR = LEGACY_DATA_ROOT / "reference" / "脊髓"

FOUNDATION_CHECKPOINT_ROOT = _path_from_env(
    "CANCHEN_MAS_FOUNDATION_CHECKPOINT_ROOT",
    SCMAS_ROOT / "checkpoints" / "foundation_models",
)
GENEFORMER_CHECKPOINT_DIR = _path_from_env(
    "CANCHEN_MAS_GENEFORMER_DIR",
    FOUNDATION_CHECKPOINT_ROOT / "geneformer",
)
SCGPT_CHECKPOINT_ROOT = _path_from_env(
    "CANCHEN_MAS_SCGPT_DIR",
    FOUNDATION_CHECKPOINT_ROOT / "scgpt",
)
NICHEFORMER_CHECKPOINT_DIR = _path_from_env(
    "CANCHEN_MAS_NICHEFORMER_DIR",
    FOUNDATION_CHECKPOINT_ROOT / "nicheformer",
)
UCE_4L_MODEL_DIR = _path_from_env(
    "CANCHEN_MAS_UCE_4L_DIR",
    FOUNDATION_CHECKPOINT_ROOT / "uce_4l",
)
UCE_33L_MODEL_DIR = _path_from_env(
    "CANCHEN_MAS_UCE_33L_DIR",
    FOUNDATION_CHECKPOINT_ROOT / "uce_33l",
)
UCE_MODEL_PY = _path_from_env(
    "CANCHEN_MAS_UCE_MODEL_PY",
    LEGACY_ROOT / "tools_layer" / "mcp_tools" / "UCE-main" / "model.py",
)
IMA_REFERENCE_H5AD = _path_from_env(
    "CANCHEN_MAS_IMA_REFERENCE_H5AD",
    DATA_DIR / "reference" / "IMA_sample.h5ad",
)
UCE_IMA_LEGACY_SCRIPT_DIR = _path_from_env(
    "CANCHEN_MAS_UCE_IMA_SCRIPT_DIR",
    LEGACY_ROOT / "scripts",
)

DEFAULT_TEST_DONORS = ["H20.33.001", "H21.33.040", "H20.33.015", "H20.33.004"]
