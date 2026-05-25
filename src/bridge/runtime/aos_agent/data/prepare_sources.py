from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from aos_agent import paths
from aos_agent.data.labels import load_merfish_genes
from aos_agent.data.reference import (
    _read_gene_by_cell_csv_sample,
    _read_smartseq_sample,
    _sample_h5ad,
    _set_unified_obs,
    annotate_whole_brain_obs,
    write_seaad_donor_split,
)
from aos_agent.io import ensure_dir, write_json, write_standard_bundle


def _write_adata_bundle(
    adata: ad.AnnData,
    output_dir: Path,
    *,
    source_id: str,
    species: str,
    family: str,
    label_column: str = "native_label",
    sample_column: str = "sample_id",
) -> dict[str, Any]:
    adata.X = sparse.csr_matrix(adata.X)
    family = _resolve_family(adata, family)
    var = adata.var.copy()
    if "feature_id" not in var.columns:
        var.insert(0, "feature_id", adata.var_names.astype(str))
    obs = adata.obs.copy()
    if label_column not in obs.columns:
        obs[label_column] = "unknown"
    if sample_column not in obs.columns:
        obs[sample_column] = "sample_0"
    write_standard_bundle(
        counts_cell_by_gene=adata.X,
        obs=obs,
        var=var,
        output_dir=output_dir,
        source_metadata={
            "source_id": source_id,
            "species": species,
            "family_use": family,
            "label_column": label_column,
            "sample_column": sample_column,
        },
    )
    return {
        "source_id": source_id,
        "path": str(output_dir),
        "species": species,
        "family_use": family,
        "label_column": label_column,
        "sample_column": sample_column,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }


def _limit_genes(adata: ad.AnnData, max_genes: int, *, force_genes: set[str] | None = None) -> ad.AnnData:
    if max_genes <= 0 or adata.n_vars <= max_genes:
        return adata
    force_genes = force_genes or set()
    x = sparse.csc_matrix(adata.X)
    scores = np.diff(x.indptr)
    forced = np.flatnonzero(pd.Index(adata.var_names.astype(str)).isin(force_genes))
    remaining_budget = max(0, int(max_genes) - int(len(forced)))
    ranked = np.argsort(scores)
    if remaining_budget:
        keep = np.unique(np.concatenate([forced, ranked[-remaining_budget:]]))
    else:
        keep = forced[:max_genes]
    keep.sort()
    return adata[:, keep].copy()


def _load_feature_panel(paths_or_files: list[str] | None) -> list[str]:
    if not paths_or_files:
        return []
    import json

    genes: list[str] = []
    for item in paths_or_files:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"feature panel not found: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                values = payload.get("genes") or payload.get("feature_names") or payload.get("features") or []
            else:
                values = payload
            genes.extend(str(x) for x in values)
        else:
            genes.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return sorted(dict.fromkeys(str(g).upper() for g in genes if str(g).strip()))


def _matrix_looks_like_counts(matrix: sparse.spmatrix, *, max_values: int = 200_000) -> bool:
    data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    data = np.asarray(data)
    if data.size == 0:
        return True
    if data.size > max_values:
        rng = np.random.default_rng(3028)
        data = data[rng.choice(data.size, size=max_values, replace=False)]
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    if float(data.min()) < 0:
        return False
    return bool(np.mean(np.isclose(data, np.round(data))) >= 0.98)


def _resolve_family(adata: ad.AnnData, requested: str) -> str:
    if requested == "gaussian":
        return "gaussian"
    if requested == "auto":
        return "nb" if _matrix_looks_like_counts(sparse.csr_matrix(adata.X)) else "gaussian"
    if requested == "nb" and not _matrix_looks_like_counts(sparse.csr_matrix(adata.X)):
        return "gaussian"
    return requested


def prepare_generation_sources(
    *,
    output_root: str | Path = paths.PREPARED_SOURCE_DIR,
    max_cells_per_source: int = 10_000,
    max_genes_per_source: int = 0,
    include_smartseq: bool = False,
    include_seaad_reference: bool = False,
    feature_panel_paths: list[str] | None = None,
    sources: list[str] | None = None,
    seed: int = 3028,
) -> dict[str, Any]:
    output_root = ensure_dir(output_root)
    wanted = set(sources or [])
    feature_panel = _load_feature_panel(feature_panel_paths)
    force_genes = set(feature_panel)

    def want(source_id: str) -> bool:
        return not wanted or source_id in wanted

    rows: list[dict[str, Any]] = []

    if include_seaad_reference and want("seaad_merfish_reference_donors"):
        split = write_seaad_donor_split()
        merfish_genes = load_merfish_genes()
        per_donor = max(1, max_cells_per_source // max(1, len(split["reference_donors"])))
        comps: list[ad.AnnData] = []
        for donor in split["reference_donors"]:
            donor_path = paths.SEAAD_DONOR_H5AD_DIR / f"{donor}.h5ad"
            if not donor_path.exists():
                continue
            comp = _sample_h5ad(
                donor_path,
                max_cells=per_donor,
                seed=seed,
                label_column="Supertype",
                feature_names=merfish_genes,
            )
            comp = _set_unified_obs(
                comp,
                species="human",
                source_dataset="seaad_merfish_reference_donors",
                source_path=donor_path,
                donor_col="Donor ID",
                sample_col="Donor ID",
                native_label_col="Supertype",
                coarse_label_col="Subclass",
            )
            comps.append(comp)
        if comps:
            adata = ad.concat(comps, join="outer", fill_value=0, merge="same")
            adata = _limit_genes(adata, max_genes_per_source, force_genes=force_genes)
            rows.append(
                _write_adata_bundle(
                    adata,
                    output_root / "seaad_merfish_reference_donors",
                    source_id="seaad_merfish_reference_donors",
                    species="human",
                    family="nb",
                )
            )

    for h5ad_path in sorted(paths.ALLEN_MOUSE_REFERENCE_DIR.glob("*.h5ad")):
        source_id = f"allen_mouse_reference_{h5ad_path.stem[:8]}"
        if not want(source_id):
            continue
        adata = _sample_h5ad(
            h5ad_path,
            max_cells=max_cells_per_source,
            seed=seed,
            label_column="cell_type",
            feature_names=feature_panel or None,
            fill_missing_features=False,
        )
        adata = _set_unified_obs(
            adata,
            species="mouse",
            source_dataset="allen_mouse_reference",
            source_path=h5ad_path,
            donor_col="donor_id",
            sample_col="slice",
            native_label_col="cell_type",
            coarse_label_col="cell_type_annot",
        )
        adata = _limit_genes(adata, max_genes_per_source, force_genes=force_genes)
        rows.append(_write_adata_bundle(adata, output_root / source_id, source_id=source_id, species="mouse", family="nb"))

    for h5ad_path in sorted(paths.HUMAN_WHOLE_BRAIN_DIR.glob("*.h5ad")):
        source_id = f"human_whole_brain_{h5ad_path.stem.replace('-', '_').lower()}"
        if not want(source_id):
            continue
        adata = _sample_h5ad(
            h5ad_path,
            max_cells=max_cells_per_source,
            seed=seed,
            label_column="anatomical_division_label",
            feature_names=feature_panel or None,
            fill_missing_features=False,
        )
        adata = annotate_whole_brain_obs(adata, species="human")
        adata = _set_unified_obs(
            adata,
            species="human",
            source_dataset="human_whole_brain",
            source_path=h5ad_path,
            donor_col="donor_label",
            sample_col="library_label",
            native_label_col="cluster",
            coarse_label_col="supercluster",
        )
        adata = _limit_genes(adata, max_genes_per_source, force_genes=force_genes)
        rows.append(_write_adata_bundle(adata, output_root / source_id, source_id=source_id, species="human", family="gaussian"))

    for h5ad_path in sorted(paths.MOUSE_WHOLE_BRAIN_DIR.glob("*.h5ad")):
        source_id = f"mouse_whole_brain_{h5ad_path.stem.replace('-', '_').lower()}"
        if not want(source_id):
            continue
        adata = _sample_h5ad(
            h5ad_path,
            max_cells=max_cells_per_source,
            seed=seed,
            label_column="anatomical_division_label",
            feature_names=feature_panel or None,
            fill_missing_features=False,
        )
        adata = annotate_whole_brain_obs(adata, species="mouse")
        adata = _set_unified_obs(
            adata,
            species="mouse",
            source_dataset="mouse_whole_brain",
            source_path=h5ad_path,
            donor_col="donor_label",
            sample_col="library_label",
            native_label_col="supertype",
            coarse_label_col="class",
        )
        adata = _limit_genes(adata, max_genes_per_source, force_genes=force_genes)
        rows.append(_write_adata_bundle(adata, output_root / source_id, source_id=source_id, species="mouse", family="gaussian"))

    if want("spinal_gse190442"):
        spinal_190442 = _read_gene_by_cell_csv_sample(
            counts_path=paths.SPINAL_DIR / "GSE190442_aggregated_counts_postqc.csv.gz",
            metadata_path=paths.SPINAL_DIR / "GSE190442_aggregated_metadata_postqc.csv.gz",
            metadata_sep=",",
            cell_id_col="Unnamed: 0",
            label_col="subtype_annotation",
            sample_col="sample",
            species="human",
            source_dataset="spinal_gse190442",
            max_cells=max_cells_per_source,
            seed=seed,
            max_genes=max_genes_per_source,
            feature_names=feature_panel or None,
        )
        spinal_190442 = _limit_genes(spinal_190442, max_genes_per_source, force_genes=force_genes)
        rows.append(_write_adata_bundle(spinal_190442, output_root / "spinal_gse190442", source_id="spinal_gse190442", species="human", family="nb"))

    if want("spinal_gse103892"):
        spinal_103892 = _read_gene_by_cell_csv_sample(
            counts_path=paths.SPINAL_DIR / "GSE103892_Expression_Count_Matrix.txt.gz",
            metadata_path=paths.SPINAL_DIR / "GSE103892_Sample_Cell_Cluster_Information.txt.gz",
            metadata_sep="\t",
            counts_sep="\t",
            cell_id_col="sample_cellbarcode",
            label_col="cell.type",
            sample_col="sample_cellbarcode",
            species="mouse",
            source_dataset="spinal_gse103892",
            max_cells=max_cells_per_source,
            seed=seed,
            max_genes=max_genes_per_source,
            feature_names=feature_panel or None,
        )
        spinal_103892 = _limit_genes(spinal_103892, max_genes_per_source, force_genes=force_genes)
        rows.append(_write_adata_bundle(spinal_103892, output_root / "spinal_gse103892", source_id="spinal_gse103892", species="mouse", family="nb"))

    smartseq = _read_smartseq_sample(
        max_cells=max_cells_per_source,
        seed=seed,
        include=include_smartseq,
        feature_names=feature_panel or None,
    )
    if smartseq is not None and want("allen_human_multiple_cortical_areas_smartseq"):
        smartseq = _limit_genes(smartseq, max_genes_per_source, force_genes=force_genes)
        rows.append(
            _write_adata_bundle(
                smartseq,
                output_root / "allen_human_multiple_cortical_areas_smartseq",
                source_id="allen_human_multiple_cortical_areas_smartseq",
                species="human",
                family="nb",
            )
        )

    manifest = {
        "prepared_source_root": str(output_root),
        "max_cells_per_source": max_cells_per_source,
        "max_genes_per_source": max_genes_per_source,
        "include_smartseq": include_smartseq,
        "include_seaad_reference": include_seaad_reference,
        "feature_panel_paths": list(feature_panel_paths or []),
        "feature_panel_genes": len(feature_panel),
        "selected_sources": sorted(wanted),
        "sources": rows,
    }
    write_json(manifest, paths.SOURCE_MANIFEST_JSON)
    write_json(manifest, output_root / "source_manifest.json")
    pd.DataFrame(rows).to_csv(output_root / "source_manifest.csv", index=False)
    return manifest
