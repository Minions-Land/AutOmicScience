from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from novaeve_bio import paths
from novaeve_bio.io import ensure_dir, read_json, write_json


def _select_anchor_and_rare(prepared_source_dir: Path) -> tuple[str, str, list[str]]:
    manifest = read_json(prepared_source_dir / "source_manifest.json")
    obs = pd.read_csv(prepared_source_dir / "obs.csv", usecols=lambda c: c in {"sample_id", "native_label"})
    if "sample_id" not in obs.columns:
        obs["sample_id"] = "sample_0"
    if "native_label" not in obs.columns:
        obs["native_label"] = "unknown"
    anchor = obs["sample_id"].astype(str).value_counts().idxmax()
    label_counts = obs["native_label"].astype(str).value_counts()
    rare = label_counts[label_counts >= 5].sort_values().index[0] if (label_counts >= 5).any() else label_counts.index[-1]
    missing = label_counts.head(min(5, len(label_counts))).index.astype(str).tolist()
    return str(anchor), str(rare), missing


def build_generation_config(
    prepared_source_dir: str | Path,
    *,
    output_source_dir: str | Path | None = None,
    target_total: int = 20_000,
    n_cores: int = 8,
    include_batch_variants: bool = True,
    seed: int = 3028,
) -> dict[str, Any]:
    prepared_source_dir = Path(prepared_source_dir)
    manifest = read_json(prepared_source_dir / "source_manifest.json")
    output_source_dir = Path(output_source_dir or (paths.SYNTHETIC_DIR / manifest["source_id"]))
    anchor, rare, missing = _select_anchor_and_rare(prepared_source_dir)
    family_use = manifest.get("family_use", "nb")
    effective_n_cores = 1 if family_use == "gaussian" else int(n_cores)
    n_virtual_donors = 3 if int(target_total) < 500 else 10
    rare_proportion = 0.005 if int(target_total) >= 1000 else max(0.005, min(0.05, 10.0 / max(1, int(target_total))))

    variants = {
        "baseline": {"enabled": True},
        "variant1": {"enabled": True, "scale_factor": 0.8},
        "variant2": {"enabled": True, "rare_celltype": rare, "rare_proportion": rare_proportion, "seed": seed},
        "variant3": {
            "enabled": bool(include_batch_variants),
            "gene_fraction": 0.35,
            "log_shift_sd": 0.25,
            "global_scale_sd": 0.12,
            "seed": seed,
        },
        "variant4": {
            "enabled": bool(include_batch_variants),
            "donors": ["VDonor02", "VDonor03"],
            "missing_celltypes": missing,
            "gene_fraction": 0.35,
            "log_shift_sd": 0.25,
            "global_scale_sd": 0.12,
            "seed": seed,
            "batch_seed": 11,
        },
    }
    return {
        "data": {
            "input_format": "standardized_bundle",
            "input_path": str(prepared_source_dir),
        },
        "anchor_donor": {"column": "sample_id", "value": anchor},
        "simulation": {
            "assay_use": "counts",
            "celltype": "native_label",
            "pseudotime": None,
            "spatial": None,
            "other_covariates": None,
            "ncell": min(int(target_total), int(manifest["n_obs"])),
            "mu_formula": "native_label",
            "sigma_formula": "1",
            "family_use": family_use,
            "n_cores": effective_n_cores,
            "parallelization": "pbmcmapply",
            "usebam": False,
            "edf_flexible": False,
            "corr_formula": "native_label",
            "copula": "gaussian",
            "DT": family_use != "gaussian",
            "pseudo_obs": False,
            "important_feature": 0.8,
            "if_sparse": False,
            "fastmvn": False,
            "nonnegative": family_use != "gaussian",
            "nonzerovar": False,
            "n_rep": 1,
        },
        "generation_template": {
            "target_total": int(target_total),
            "virtual_donor_column": "sample_id",
            "anchor_backup_column": "anchor_sample_id",
            "virtual_donor_ids": [f"VDonor{i:02d}" for i in range(1, n_virtual_donors + 1)],
            "seed": seed,
        },
        "variants": variants,
        "output": {
            "output_dir": str(output_source_dir),
            "master_cache_path": str(output_source_dir / "anchor_master_model.rds"),
            "export_h5ad": False,
        },
    }


def write_generation_configs(
    *,
    prepared_source_root: str | Path = paths.PREPARED_SOURCE_DIR,
    config_root: str | Path | None = None,
    target_total: int = 20_000,
    n_cores: int = 8,
    seed: int = 3028,
) -> dict[str, Any]:
    prepared_source_root = Path(prepared_source_root)
    config_root = ensure_dir(config_root or (paths.SYNTHETIC_DIR / "_configs"))
    configs: list[dict[str, Any]] = []
    for source_dir in sorted(p for p in prepared_source_root.iterdir() if p.is_dir()):
        manifest_path = source_dir / "source_manifest.json"
        if not manifest_path.exists():
            continue
        source_manifest = read_json(manifest_path)
        obs = pd.read_csv(source_dir / "obs.csv", usecols=lambda c: c == "sample_id")
        include_batch = "sample_id" in obs.columns and obs["sample_id"].astype(str).nunique() >= 2
        cfg = build_generation_config(
            source_dir,
            output_source_dir=paths.SYNTHETIC_DIR / source_manifest["source_id"],
            target_total=target_total,
            n_cores=n_cores,
            include_batch_variants=include_batch,
            seed=seed,
        )
        config_path = config_root / f"{source_manifest['source_id']}.json"
        write_json(cfg, config_path)
        configs.append(
            {
                "source_id": source_manifest["source_id"],
                "config_path": str(config_path),
                "output_dir": cfg["output"]["output_dir"],
                "family_use": cfg["simulation"]["family_use"],
                "include_batch_variants": include_batch,
            }
        )
    manifest = {"configs": configs, "target_total": target_total, "n_cores": n_cores}
    write_json(manifest, config_root / "generation_config_manifest.json")
    return manifest


def resolve_rscript_command(rscript_path: str = "Rscript") -> list[str]:
    if rscript_path.startswith("conda:"):
        env = rscript_path.split(":", 1)[1]
        return ["conda", "run", "-n", env, "Rscript"]
    return [rscript_path]


def materialize_rscript_executable(rscript_path: str = "Rscript") -> str:
    if not rscript_path.startswith("conda:"):
        return rscript_path
    env = rscript_path.split(":", 1)[1]
    bin_dir = ensure_dir(paths.SYNTHETIC_DIR / "_bin")
    wrapper = bin_dir / f"Rscript_{env}.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"exec conda run -n {env} Rscript \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def preflight_scdesign3(rscript_path: str = "Rscript") -> tuple[bool, str]:
    rscript_cmd = resolve_rscript_command(rscript_path)
    exe = rscript_cmd[0]
    if shutil.which(exe) is None:
        return False, f"{exe} not found on PATH"
    if not paths.SC_DESIGN3_ANCHOR_RUNNER.exists():
        return False, f"Missing scDesign3 anchor runner: {paths.SC_DESIGN3_ANCHOR_RUNNER}"
    try:
        subprocess.run([*rscript_cmd, "--version"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return False, f"Rscript preflight failed via {' '.join(rscript_cmd)}: {exc.stderr or exc.stdout}"
    return True, "ok"


def run_generation_configs(
    config_manifest_path: str | Path,
    *,
    rscript_path: str = "Rscript",
    force_refit: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = read_json(config_manifest_path)
    ok, reason = preflight_scdesign3(rscript_path)
    rscript_executable = materialize_rscript_executable(rscript_path)
    rows: list[dict[str, Any]] = []
    for item in manifest.get("configs", []):
        cmd = [
            sys.executable,
            str(paths.SC_DESIGN3_ANCHOR_RUNNER),
            "--config",
            item["config_path"],
            "--rscript-path",
            rscript_executable,
        ]
        if force_refit:
            cmd.append("--force-refit")
        row = {**item, "command": " ".join(cmd)}
        if dry_run:
            row.update({"status": "dry_run", "reason": "not executed"})
        elif not ok:
            row.update({"status": "skipped", "reason": reason})
        else:
            try:
                subprocess.run(cmd, cwd=str(paths.SC_DESIGN3_ROOT), check=True)
                row.update({"status": "completed", "reason": ""})
            except subprocess.CalledProcessError as exc:
                row.update({"status": "failed", "reason": str(exc)})
        rows.append(row)
    output = {"preflight_ok": ok, "preflight_reason": reason, "runs": rows}
    write_json(output, paths.SYNTHETIC_DIR / "generation_run_manifest.json")
    return output
