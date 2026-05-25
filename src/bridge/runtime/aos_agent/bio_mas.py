from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from aos_agent import paths
from aos_agent.io import ensure_dir, write_json, write_standard_bundle
from aos_agent.stage2.selector import profile_query, run_cross_species_plan, select_models


CORE_PYTHON_MODULES = ["numpy", "pandas", "scipy", "sklearn", "anndata", "scanpy", "yaml", "joblib"]
LLM_PYTHON_MODULES = ["openai", "httpx", "dotenv"]
FOUNDATION_PYTHON_MODULES = ["torch", "transformers", "safetensors"]
NOTEBOOK_PYTHON_MODULES = ["jupyter_client", "ipykernel"]
R_SCRIPT_CANDIDATES = [
    "Rscript",
]


def _module_status(module: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    status: dict[str, Any] = {"name": module, "available": spec is not None}
    if spec is None:
        return status
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "")
        if version:
            status["version"] = str(version)
    except Exception as exc:
        status["available"] = False
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def _path_status(name: str, path: str | Path, *, kind: str) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    status: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "path": str(p),
        "exists": bool(exists),
    }
    if exists:
        try:
            stat = p.stat()
            status["bytes"] = int(stat.st_size) if p.is_file() else None
        except OSError:
            pass
    return status


def _rscript_status(candidates: list[str] | None = None) -> dict[str, Any]:
    candidates = candidates or R_SCRIPT_CANDIDATES
    checked: list[dict[str, Any]] = []
    for candidate in candidates:
        resolved = shutil.which(candidate) if candidate == "Rscript" else candidate
        if not resolved:
            checked.append({"candidate": candidate, "available": False})
            continue
        exe = Path(resolved)
        if not exe.exists():
            checked.append({"candidate": candidate, "path": str(exe), "available": False})
            continue
        version = ""
        ok = True
        try:
            proc = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=10)
            version = (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr).strip() else ""
            ok = proc.returncode == 0
        except Exception as exc:
            ok = False
            version = f"{type(exc).__name__}: {exc}"
        checked.append({"candidate": candidate, "path": str(exe), "available": ok, "version": version})
        if ok:
            return {"available": True, "path": str(exe), "version": version, "checked": checked}
    return {"available": False, "path": "", "version": "", "checked": checked}


def bio_mas_preflight(
    *,
    output_dir: str | Path | None = None,
    rscript_path: str = "",
    include_foundation: bool = True,
    include_notebook: bool = True,
) -> dict[str, Any]:
    """Inspect local dependencies and external biological assets.

    This function does not download data or pretend missing assets are present.
    It reports which tiny/demo paths are usable locally and which real-data
    paths/checkpoints must be supplied by the operator.
    """
    out_dir = ensure_dir(output_dir or (Path("runs") / "bio_mas_preflight"))
    module_groups = {
        "core": [_module_status(m) for m in CORE_PYTHON_MODULES],
        "llm": [_module_status(m) for m in LLM_PYTHON_MODULES],
    }
    if include_foundation:
        module_groups["foundation"] = [_module_status(m) for m in FOUNDATION_PYTHON_MODULES]
    if include_notebook:
        module_groups["notebook"] = [_module_status(m) for m in NOTEBOOK_PYTHON_MODULES]

    external_assets = [
        _path_status("SEA-AD MERFISH h5ad", paths.SEAAD_MERFISH_H5AD, kind="real_data"),
        _path_status("SEA-AD donor h5ad dir", paths.SEAAD_DONOR_H5AD_DIR, kind="real_data"),
        _path_status("Kukanja MS npz", paths.KUKANJA_MS_NPZ, kind="real_query"),
        _path_status("Kukanja EAE npz", paths.KUKANJA_EAE_NPZ, kind="real_query"),
        _path_status("IMA reference h5ad", paths.IMA_REFERENCE_H5AD, kind="real_reference"),
    ]
    checkpoints = [
        _path_status("Geneformer checkpoint dir", paths.GENEFORMER_CHECKPOINT_DIR, kind="foundation_weight"),
        _path_status("scGPT checkpoint dir", paths.SCGPT_CHECKPOINT_ROOT, kind="foundation_weight"),
        _path_status("Nicheformer checkpoint dir", paths.NICHEFORMER_CHECKPOINT_DIR, kind="foundation_weight"),
        _path_status("UCE 4L checkpoint dir", paths.UCE_4L_MODEL_DIR, kind="foundation_weight"),
        _path_status("UCE 33L checkpoint dir", paths.UCE_33L_MODEL_DIR, kind="foundation_weight"),
        _path_status("UCE model.py", paths.UCE_MODEL_PY, kind="foundation_code"),
    ]
    default_tiny_root = Path("runs") / "bio_mas_tiny_data"
    tiny_paths = [
        _path_status("tiny query h5ad", default_tiny_root / "query_tiny.h5ad", kind="synthetic_tiny_demo"),
        _path_status(
            "tiny prepared source manifest",
            default_tiny_root / "prepared_sources" / "tiny_reference" / "source_manifest.json",
            kind="synthetic_tiny_demo",
        ),
    ]

    r_status = _rscript_status([rscript_path] if rscript_path else None)
    missing_core = [
        item["name"]
        for item in module_groups["core"]
        if not item.get("available")
    ]
    missing_foundation = [
        item["name"]
        for item in module_groups.get("foundation", [])
        if not item.get("available")
    ]
    missing_real_assets = [item["name"] for item in external_assets if not item.get("exists")]
    missing_checkpoints = [item["name"] for item in checkpoints if not item.get("exists")]

    recommended_python = sys.executable
    result = {
        "ok": not missing_core,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "recommended_env": {
            "AOS_PYTHON_BIN": recommended_python,
            "AOS_PYTHON_RUNTIME": str(paths.SCMAS_ROOT),
            "Rscript": r_status.get("path", ""),
        },
        "modules": module_groups,
        "rscript": r_status,
        "external_assets": external_assets,
        "foundation_checkpoints": checkpoints,
        "tiny_demo_assets": tiny_paths,
        "missing": {
            "core_python_modules": missing_core,
            "foundation_python_modules": missing_foundation,
            "real_data_assets": missing_real_assets,
            "foundation_checkpoints": missing_checkpoints,
            "rscript": [] if r_status.get("available") else ["Rscript"],
        },
        "notes": [
            "Tiny demo assets are synthetic and local only; they are not SEA-AD/Kukanja data and must not be used for scientific conclusions.",
            "Real SEA-AD/Kukanja datasets and foundation-model checkpoints are external assets. Provide them through AOS_MAS_* environment variables.",
            "Expression log1p kNN/prototype tools can run without torch/transformers. Geneformer/scGPT/Nicheformer/UCE require foundation dependencies and weights.",
        ],
    }
    report_path = out_dir / "bio_mas_preflight.json"
    write_json(result, report_path)
    result["report_path"] = str(report_path)
    return result


def _tiny_matrix(seed: int, cells_per_label: int) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i:03d}" for i in range(1, 33)]
    labels = ["astro", "neuron", "oligo"]
    rows: list[np.ndarray] = []
    obs_rows: list[dict[str, Any]] = []
    for label_idx, label in enumerate(labels):
        for local_idx in range(cells_per_label):
            baseline = rng.poisson(1.0, size=len(genes)).astype(np.float32)
            start = label_idx * 8
            baseline[start : start + 8] += rng.poisson(8.0, size=8).astype(np.float32)
            rows.append(baseline)
            obs_rows.append(
                {
                    "cell_id": f"ref_{label}_{local_idx:03d}",
                    "native_label": label,
                    "coarse_label": "glia" if label in {"astro", "oligo"} else "neuron",
                    "sample_id": f"donor_{label_idx + 1}",
                    "synthetic_tiny_demo": True,
                }
            )
    obs = pd.DataFrame(obs_rows).set_index("cell_id", drop=False)
    var = pd.DataFrame(
        {
            "feature_id": genes,
            "gene_symbol": genes,
            "synthetic_tiny_demo": True,
        },
        index=pd.Index(genes, name="feature_id"),
    )
    return sparse.csr_matrix(np.vstack(rows)), obs, var


def create_tiny_bio_demo(
    *,
    output_dir: str | Path | None = None,
    cells_per_label: int = 8,
    seed: int = 3028,
) -> dict[str, Any]:
    """Create a clearly marked synthetic tiny dataset for local smoke tests."""
    out_dir = ensure_dir(output_dir or (Path("runs") / "bio_mas_tiny_data"))
    prepared_root = ensure_dir(out_dir / "prepared_sources")
    source_dir = ensure_dir(prepared_root / "tiny_reference")
    artifact_root = ensure_dir(out_dir / "artifacts")
    manifest_dir = ensure_dir(artifact_root / "manifests")
    score_dir = ensure_dir(artifact_root / "scores")

    X_ref, obs_ref, var = _tiny_matrix(seed, cells_per_label)
    write_standard_bundle(
        counts_cell_by_gene=X_ref,
        obs=obs_ref,
        var=var,
        output_dir=source_dir,
        source_metadata={
            "source_id": "tiny_reference",
            "species": "synthetic",
            "label_column": "native_label",
            "coarse_label_column": "coarse_label",
            "sample_column": "sample_id",
            "synthetic_tiny_demo": True,
            "scientific_use": "smoke_test_only",
            "note": "Generated by AutOmicScience for local MAS smoke tests; not a real biological dataset.",
        },
    )
    counts_path = source_dir / "counts.mtx"
    if not counts_path.exists():
        raise FileNotFoundError(f"Tiny demo failed to write required Matrix Market counts file: {counts_path}")

    rng = np.random.default_rng(seed + 7)
    query_rows = []
    query_obs = []
    labels = ["astro", "neuron", "oligo"]
    for label_idx, label in enumerate(labels):
        for local_idx in range(max(2, cells_per_label // 2)):
            baseline = rng.poisson(1.0, size=X_ref.shape[1]).astype(np.float32)
            start = label_idx * 8
            baseline[start : start + 8] += rng.poisson(7.0, size=8).astype(np.float32)
            query_rows.append(baseline)
            query_obs.append(
                {
                    "cell_id": f"query_{label}_{local_idx:03d}",
                    "native_label": label,
                    "coarse_label": "glia" if label in {"astro", "oligo"} else "neuron",
                    "sample_id": "tiny_query",
                    "synthetic_tiny_demo": True,
                }
            )
    query = ad.AnnData(
        X=sparse.csr_matrix(np.vstack(query_rows)),
        obs=pd.DataFrame(query_obs).set_index("cell_id", drop=False),
        var=var.copy(),
    )
    query.uns["automic-science"] = {
        "synthetic_tiny_demo": True,
        "scientific_use": "smoke_test_only",
        "note": "Local synthetic tiny data; not SEA-AD/Kukanja and not valid for biological claims.",
    }
    query_path = out_dir / "query_tiny.h5ad"
    query.write_h5ad(query_path)

    source_manifest = {
        "schema_version": "automic-science.tiny.sources.v1",
        "synthetic_tiny_demo": True,
        "sources": [
            {
                "source_id": "tiny_reference",
                "source_path": str(source_dir),
                "species": "synthetic",
                "n_obs": int(obs_ref.shape[0]),
                "n_vars": int(var.shape[0]),
                "label_column": "native_label",
                "coarse_label_column": "coarse_label",
                "scientific_use": "smoke_test_only",
            }
        ],
    }
    write_json(source_manifest, manifest_dir / "synthetic_sources.json")
    score_csv = score_dir / "full_model_variant_scores.csv"
    score_csv.write_text(
        "\n".join(
            [
                "evaluation_type,model,source_group,variant,macro_f1,ratio_vs_baseline",
                "raw_label_transfer_no_training,expression_log1p_knn,tiny_reference,baseline_tiny,0.98,1.0",
                "raw_label_transfer_no_training,expression_log1p_knn,tiny_reference,signal80_tiny,0.95,0.97",
                "raw_label_transfer_no_training,expression_log1p_prototype,tiny_reference,baseline_tiny,0.94,1.0",
                "raw_label_transfer_no_training,expression_log1p_prototype,tiny_reference,signal80_tiny,0.91,0.96",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "automic-science.tiny_demo.v1",
        "synthetic_tiny_demo": True,
        "scientific_use": "smoke_test_only",
        "query_h5ad": str(query_path),
        "prepared_source_root": str(prepared_root),
        "source_dir": str(source_dir),
        "artifact_bundle": str(artifact_root),
        "capability_dir": str(Path("configs") / "capability"),
        "source_manifest": str(manifest_dir / "synthetic_sources.json"),
        "score_table": str(score_csv),
        "n_query_cells": int(query.n_obs),
        "n_reference_cells": int(obs_ref.shape[0]),
        "n_genes": int(var.shape[0]),
        "labels": labels,
        "warning": "This is generated synthetic tiny data for local smoke tests only; it is not a biological dataset.",
    }
    metadata_path = out_dir / "tiny_demo_manifest.json"
    write_json(metadata, metadata_path)
    metadata["manifest_path"] = str(metadata_path)
    return metadata


def run_tiny_bio_mas_demo(
    *,
    output_dir: str | Path | None = None,
    cells_per_label: int = 8,
    seed: int = 3028,
    top_k: int = 1,
) -> dict[str, Any]:
    """Run profile -> select -> execute over synthetic tiny demo data."""
    root = ensure_dir(output_dir or (Path("runs") / "bio_mas_tiny_demo"))
    demo = create_tiny_bio_demo(output_dir=root / "data", cells_per_label=cells_per_label, seed=seed)
    stage2_dir = ensure_dir(root / "stage2")
    profile = profile_query(
        dataset_id="tiny_synthetic_query",
        input_path=demo["query_h5ad"],
        output_dir=stage2_dir,
        max_cells=10_000,
        seed=seed,
    )
    selection = select_models(
        query_profile_path=profile["query_profile_path"],
        output_dir=stage2_dir,
        artifact_bundle=demo["artifact_bundle"],
        prepared_source_root=demo["prepared_source_root"],
        capability_dir=demo["capability_dir"],
        top_k=top_k,
        min_shared_genes=8,
        max_source_profile_cells=1000,
        max_query_cells=1000,
        max_reference_cells=1000,
        k=3,
        seed=seed,
        llm_mode="off",
        include_default_excluded=True,
        excluded_model_ids=[
            "geneformer_raw_knn",
            "scgpt_brain_raw_knn",
            "scgpt_human_raw_knn",
            "nicheformer_raw_knn",
            "uce_4l_raw_knn",
            "uce_33l_raw_knn",
        ],
    )
    execution = run_cross_species_plan(
        plan_path=selection["selected_execution_plan"],
        output_dir=root / "execution",
        max_query_cells=1000,
        max_reference_cells=1000,
        min_shared_genes=8,
        k=3,
        device="",
        batch_size=16,
    )
    n_metric_rows = int(execution.get("n_metric_rows", 0))
    n_prediction_rows = int(execution.get("n_prediction_rows", 0))
    n_skips = int(execution.get("n_skips", 0))
    result = {
        "ok": n_metric_rows > 0 and n_prediction_rows > 0,
        "synthetic_tiny_demo": True,
        "scientific_use": "smoke_test_only",
        "warning": "This run validates AutOmicScience MAS wiring only. It is not a biological benchmark.",
        "output_dir": str(root),
        "n_metric_rows": n_metric_rows,
        "n_prediction_rows": n_prediction_rows,
        "n_skips": n_skips,
        "tiny_demo": demo,
        "profile": profile,
        "selection": selection,
        "execution": execution,
    }
    write_json(result, root / "tiny_demo_run_summary.json")
    return result
