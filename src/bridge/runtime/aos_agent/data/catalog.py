from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
from scipy import io as spio

from aos_agent import paths
from aos_agent.io import ensure_dir, read_json, write_json
from aos_agent.scdesign3.discover import discover_existing_seaad_variants, discover_new_synthetic_variants


REPORTS_DIR = paths.SCMAS_ROOT / "reports"


def _exists(path: str | Path) -> bool:
    return Path(path).exists()


def _h5ad_shape(path: str | Path) -> tuple[int | None, int | None]:
    path = Path(path)
    if not path.exists():
        return None, None
    adata = ad.read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs), int(adata.n_vars)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def _npz_shape(path: str | Path) -> tuple[int | None, int | None]:
    path = Path(path)
    if not path.exists():
        return None, None
    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        if "X" not in data:
            return None, None
        shape = data["X"].shape
    return int(shape[0]), int(shape[1]) if len(shape) > 1 else None


def _mtx_shape(path: str | Path, *, orientation: str = "gene_by_cell") -> tuple[int | None, int | None]:
    path = Path(path)
    if not path.exists():
        return None, None
    info = spio.mminfo(str(path))
    n_rows, n_cols = info[0], info[1]
    if orientation == "gene_by_cell":
        return int(n_cols), int(n_rows)
    return int(n_rows), int(n_cols)


def _row(
    *,
    dataset_id: str,
    species: str,
    source_path: str | Path,
    data_format: str,
    role: str,
    stage1_status: str,
    used_for_scdesign_current: bool = False,
    used_for_scdesign_planned: bool = False,
    used_for_reference_current: bool = False,
    used_for_reference_planned: bool = False,
    used_for_real_test: bool = False,
    used_for_synthetic_benchmark: bool = False,
    n_obs: int | None = None,
    n_vars: int | None = None,
    label_column: str = "",
    coarse_label_column: str = "",
    sample_column: str = "",
    donor_rule: str = "",
    generated_output: str = "",
    overlap_note: str = "",
    smoke_note: str = "",
) -> dict[str, Any]:
    path = Path(source_path)
    return {
        "dataset_id": dataset_id,
        "species": species,
        "source_path": str(source_path),
        "exists": _exists(path),
        "format": data_format,
        "role": role,
        "stage1_status": stage1_status,
        "used_for_scdesign_current": used_for_scdesign_current,
        "used_for_scdesign_planned": used_for_scdesign_planned,
        "used_for_reference_current": used_for_reference_current,
        "used_for_reference_planned": used_for_reference_planned,
        "used_for_real_test": used_for_real_test,
        "used_for_synthetic_benchmark": used_for_synthetic_benchmark,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "label_column": label_column,
        "coarse_label_column": coarse_label_column,
        "sample_column": sample_column,
        "donor_rule": donor_rule,
        "generated_output": generated_output,
        "overlap_note": overlap_note,
        "smoke_note": smoke_note,
    }


def _variant_id_label(variant_id: str) -> str:
    if variant_id.startswith("baseline"):
        return "baseline"
    if "signal80" in variant_id:
        return "signal80"
    if "rare0p5pct" in variant_id:
        return "rare0p5pct"
    if "virtual_batch" in variant_id:
        return "virtual_batch"
    if "missing_celltypes" in variant_id:
        return "missing_celltypes"
    return variant_id


def build_dataset_catalog(
    *,
    output_dir: str | Path = REPORTS_DIR,
    include_shape_probe: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []

    split_path = paths.SEAAD_DONOR_SPLIT_JSON
    split = read_json(split_path) if split_path.exists() else None
    if split is None:
        from aos_agent.data.reference import write_seaad_donor_split

        split = write_seaad_donor_split()

    n_obs, n_vars = _h5ad_shape(paths.SEAAD_TEST_H5AD) if include_shape_probe else (None, None)
    rows.append(
        _row(
            dataset_id="seaad_merfish_real_test_140gene",
            species="human",
            source_path=paths.SEAAD_TEST_H5AD,
            data_format="h5ad",
            role="real_test",
            stage1_status="built",
            used_for_real_test=True,
            n_obs=n_obs,
            n_vars=n_vars,
            label_column="Supertype",
            coarse_label_column="Subclass",
            sample_column="Donor ID",
            donor_rule="test donors: " + ",".join(split["test_donors"]),
            overlap_note="SEA-AD test donors are excluded from SEA-AD reference donors.",
        )
    )

    rows.append(
        _row(
            dataset_id="seaad_merfish_reference_donors",
            species="human",
            source_path=paths.SEAAD_MERFISH_H5AD,
            data_format="h5ad",
            role="reference_pool",
            stage1_status="planned_reference",
            used_for_reference_planned=True,
            used_for_scdesign_planned=True,
            label_column="Supertype",
            coarse_label_column="Subclass",
            sample_column="Donor ID",
            donor_rule="reference donors: " + ",".join(split["reference_donors"]),
            overlap_note="Donor-disjoint from seaad_merfish_real_test_140gene.",
        )
    )

    for dataset_id, path in (
        ("kukanja_ms", paths.KUKANJA_MS_NPZ),
        ("kukanja_eae", paths.KUKANJA_EAE_NPZ),
    ):
        n_obs, n_vars = _npz_shape(path) if include_shape_probe else (None, None)
        rows.append(
            _row(
                dataset_id=dataset_id,
                species="mouse",
                source_path=path,
                data_format="npz",
                role="real_test",
                stage1_status="test_only",
                used_for_real_test=True,
                n_obs=n_obs,
                n_vars=n_vars,
                overlap_note="Test-only. Not used for reference or scDesign3 generation.",
            )
        )

    for item in discover_existing_seaad_variants():
        n_obs, n_vars = _mtx_shape(Path(item["path"]) / "sim_counts.mtx") if include_shape_probe else (None, None)
        rows.append(
            _row(
                dataset_id=item["dataset_id"],
                species="human",
                source_path=item["path"],
                data_format="scdesign3_bundle",
                role="synthetic_benchmark_existing",
                stage1_status="reuse_only",
                used_for_synthetic_benchmark=True,
                n_obs=n_obs,
                n_vars=n_vars,
                label_column="Supertype",
                coarse_label_column="Subclass",
                sample_column="Donor ID/sample_id",
                donor_rule=_variant_id_label(item["variant_id"]),
                overlap_note="Already generated SEA-AD synthetic variants; no new SEA-AD generation is done in stage-1 smoke.",
            )
        )

    current_generated_sources = {item["source_id"] for item in discover_new_synthetic_variants()}
    current_generated_outputs: dict[str, list[str]] = {}
    for item in discover_new_synthetic_variants():
        current_generated_outputs.setdefault(item["source_id"], []).append(item["path"])

    for h5ad_path in sorted(paths.ALLEN_MOUSE_REFERENCE_DIR.glob("*.h5ad")):
        source_id = f"allen_mouse_reference_{h5ad_path.stem[:8]}"
        n_obs, n_vars = _h5ad_shape(h5ad_path) if include_shape_probe else (None, None)
        rows.append(
            _row(
                dataset_id=source_id,
                species="mouse",
                source_path=h5ad_path,
                data_format="h5ad",
                role="scdesign_source_pool; label_transfer_reference_pool",
                stage1_status="smoke_generated" if source_id in current_generated_sources else "planned_source",
                used_for_scdesign_current=source_id in current_generated_sources,
                used_for_scdesign_planned=True,
                used_for_reference_planned=True,
                n_obs=n_obs,
                n_vars=n_vars,
                label_column="cell_type",
                coarse_label_column="cell_type_annot",
                sample_column="donor_id",
                generated_output=";".join(current_generated_outputs.get(source_id, [])),
                overlap_note=(
                    "Smoke label-transfer reference is sampled from the same original h5ad "
                    "but excludes cells used as scDesign3 input."
                    if source_id in current_generated_sources
                    else "Planned source; no generated smoke artifact yet."
                ),
                smoke_note="current smoke source" if source_id in current_generated_sources else "",
            )
        )

    for h5ad_path in sorted(paths.HUMAN_WHOLE_BRAIN_DIR.glob("*.h5ad")):
        n_obs, n_vars = _h5ad_shape(h5ad_path) if include_shape_probe else (None, None)
        rows.append(
            _row(
                dataset_id=f"human_whole_brain_{h5ad_path.stem.replace('-', '_').lower()}",
                species="human",
                source_path=h5ad_path,
                data_format="h5ad",
                role="scdesign_source_pool; reference_pool",
                stage1_status="planned_source",
                used_for_scdesign_planned=True,
                used_for_reference_planned=True,
                n_obs=n_obs,
                n_vars=n_vars,
                label_column="cluster",
                coarse_label_column="supercluster",
                sample_column="donor_label/library_label",
                overlap_note="Not used as held-out real test in stage 1.",
            )
        )

    for h5ad_path in sorted(paths.MOUSE_WHOLE_BRAIN_DIR.glob("*.h5ad")):
        n_obs, n_vars = _h5ad_shape(h5ad_path) if include_shape_probe else (None, None)
        rows.append(
            _row(
                dataset_id=f"mouse_whole_brain_{h5ad_path.stem.replace('-', '_').lower()}",
                species="mouse",
                source_path=h5ad_path,
                data_format="h5ad",
                role="scdesign_source_pool; reference_pool",
                stage1_status="planned_source",
                used_for_scdesign_planned=True,
                used_for_reference_planned=True,
                n_obs=n_obs,
                n_vars=n_vars,
                label_column="supertype",
                coarse_label_column="class",
                sample_column="donor_label/library_label",
                overlap_note="Not used as held-out real test in stage 1.",
            )
        )

    spinal_rows = [
        (
            "spinal_gse190442",
            "human",
            paths.SPINAL_DIR / "GSE190442_aggregated_counts_postqc.csv.gz",
            "csv.gz",
            "subtype_annotation",
            "subtype_annotation",
            "sample",
            "planned_source",
        ),
        (
            "spinal_gse103892",
            "mouse",
            paths.SPINAL_DIR / "GSE103892_Expression_Count_Matrix.txt.gz",
            "txt.gz",
            "cell.type",
            "cell.type",
            "sample_cellbarcode",
            "planned_source",
        ),
        (
            "spinal_imputation_spinalcord",
            "unknown",
            paths.SPINAL_DIR / "imputation_spinalcord.h5ad",
            "h5ad",
            "",
            "",
            "",
            "phase2_unreviewed",
        ),
    ]
    for dataset_id, species, source_path, fmt, label_col, coarse_col, sample_col, status in spinal_rows:
        n_obs, n_vars = (None, None)
        if fmt == "h5ad" and include_shape_probe:
            n_obs, n_vars = _h5ad_shape(source_path)
        is_generated = dataset_id in current_generated_sources
        rows.append(
            _row(
                dataset_id=dataset_id,
                species=species,
                source_path=source_path,
                data_format=fmt,
                role="scdesign_source_pool; label_transfer_reference_pool" if status == "planned_source" else "phase2",
                stage1_status="smoke_generated" if is_generated else status,
                used_for_scdesign_current=is_generated,
                used_for_scdesign_planned=status == "planned_source",
                used_for_reference_planned=status == "planned_source",
                n_obs=n_obs,
                n_vars=n_vars,
                label_column=label_col,
                coarse_label_column=coarse_col,
                sample_column=sample_col,
                generated_output=";".join(current_generated_outputs.get(dataset_id, [])),
                overlap_note="Spinal source pool; not used as stage-1 held-out test.",
            )
        )

    for tar_path in sorted(paths.SPINAL_DIR.glob("GSE*_RAW.tar")):
        rows.append(
            _row(
                dataset_id=tar_path.stem.lower(),
                species="unknown",
                source_path=tar_path,
                data_format="raw_tar",
                role="phase2_raw",
                stage1_status="phase2_not_parsed",
                overlap_note="Raw archive intentionally excluded from stage-1 smoke.",
            )
        )

    rows.append(
        _row(
            dataset_id="seaad_mtg_snrna",
            species="human",
            source_path=paths.LEGACY_DATA_ROOT / "seaad_mtg_snrna" / "seaad_mtg_snrna.h5ad",
            data_format="h5ad",
            role="excluded_stage1",
            stage1_status="not_used_stage1",
            overlap_note="User requested SEA-AD snRNA not be used in this stage.",
        )
    )

    rows.append(
        _row(
            dataset_id="allen_human_multiple_cortical_areas_smartseq",
            species="human",
            source_path=paths.ALLEN_HUMAN_SMARTSEQ_DIR,
            data_format="directory",
            role="scdesign_source_pool; reference_pool",
            stage1_status="planned_source_optional",
            used_for_scdesign_planned=True,
            used_for_reference_planned=True,
            label_column="cell_type",
            coarse_label_column="class/subclass",
            sample_column="donor/sample",
            overlap_note="Optional planned source; not part of current smoke unless --include-smartseq is used.",
        )
    )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "dataset_catalog.csv"
    md_path = output_dir / "dataset_catalog.md"
    json_path = output_dir / "dataset_catalog.json"
    df.to_csv(csv_path, index=False)
    try:
        md_text = df.to_markdown(index=False)
    except ImportError:
        md_text = df.to_csv(index=False)
    md_path.write_text(md_text, encoding="utf-8")
    write_json({"rows": rows, "n_rows": len(rows)}, json_path)
    return {
        "n_rows": len(rows),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "json_path": str(json_path),
    }
