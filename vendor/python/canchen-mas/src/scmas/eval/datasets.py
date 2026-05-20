from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import io as spio

from scmas import paths
from scmas.data.labels import build_seaad_label_maps, factorize_ids, load_merfish_genes, seaad_label_ids_from_obs
from scmas.io import ensure_dir, normalize_gene_name, read_standard_bundle, write_json
from scmas.scdesign3.discover import discover_existing_seaad_variants, discover_new_synthetic_variants


@dataclass
class EvalDataset:
    dataset_id: str
    kind: str
    path: str
    species: str = "human"
    synthetic: bool = False
    existing: bool = False
    source_id: str = ""
    variant_id: str = ""
    contract: str = "unknown"
    reason: str = ""


def discover_eval_datasets(include_new_synthetic: bool = True) -> list[EvalDataset]:
    datasets = [
        EvalDataset(
            dataset_id="seaad_merfish_test_donors_140gene",
            kind="h5ad",
            path=str(paths.SEAAD_TEST_H5AD),
            species="human",
            contract="seaad_140_h5ad",
        ),
        EvalDataset(
            dataset_id="kukanja_ms",
            kind="kukanja_npz",
            path=str(paths.KUKANJA_MS_NPZ),
            species="mouse",
            contract="kukanja_ms",
            reason="Kukanja label space is not SEA-AD class/subclass/supertype.",
        ),
        EvalDataset(
            dataset_id="kukanja_eae",
            kind="kukanja_npz",
            path=str(paths.KUKANJA_EAE_NPZ),
            species="mouse",
            contract="kukanja_eae",
            reason="Kukanja label space is not SEA-AD class/subclass/supertype.",
        ),
    ]
    for item in discover_existing_seaad_variants():
        datasets.append(
            EvalDataset(
                dataset_id=item["dataset_id"],
                kind="scdesign3_bundle",
                path=item["path"],
                species="human",
                synthetic=True,
                existing=True,
                source_id=item["source_id"],
                variant_id=item["variant_id"],
                contract="seaad_140_bundle",
            )
        )
    if include_new_synthetic:
        for item in discover_new_synthetic_variants():
            datasets.append(
                EvalDataset(
                    dataset_id=item["dataset_id"],
                    kind="scdesign3_bundle",
                    path=item["path"],
                    species="unknown",
                    synthetic=True,
                    existing=False,
                    source_id=item["source_id"],
                    variant_id=item["variant_id"],
                    contract="scdesign3_bundle",
                )
            )
    return datasets


def _write_npz_and_h5ad(
    *,
    dataset: EvalDataset,
    x: sparse.spmatrix,
    obs: pd.DataFrame,
    var_names: list[str],
    output_dir: Path,
    pseudo_min_donors: int = 10,
    max_cells: int = 0,
    seed: int = 3028,
) -> dict[str, Any]:
    if max_cells and max_cells > 0 and obs.shape[0] > max_cells:
        label_for_sampling = obs["Supertype"] if "Supertype" in obs.columns else pd.Series(np.arange(obs.shape[0]))
        from scmas.io import stratified_indices

        idx = stratified_indices(label_for_sampling, max_cells=max_cells, min_per_group=5, seed=seed)
        x = sparse.csr_matrix(x)[idx, :]
        obs = obs.iloc[idx].copy()

    maps = build_seaad_label_maps()
    y_class, y_subclass, y_supertype = seaad_label_ids_from_obs(obs, maps)
    donor_ids = np.arange(pseudo_min_donors, dtype=np.int64)
    shuffled = donor_ids.copy()
    np.random.seed(3028)
    np.random.shuffle(shuffled)
    test_donor_ids = shuffled[:6]
    non_test_donor_ids = shuffled[6:]
    batch_donor = np.resize(test_donor_ids, obs.shape[0]).astype(np.int64)

    spatial = np.zeros((obs.shape[0], 2), dtype=np.float32)
    if {"spatial_x", "spatial_y"}.issubset(obs.columns):
        spatial[:, 0] = pd.to_numeric(obs["spatial_x"], errors="coerce").fillna(0).to_numpy(np.float32)
        spatial[:, 1] = pd.to_numeric(obs["spatial_y"], errors="coerce").fillna(0).to_numpy(np.float32)
    elif {"x", "y"}.issubset(obs.columns):
        spatial[:, 0] = pd.to_numeric(obs["x"], errors="coerce").fillna(0).to_numpy(np.float32)
        spatial[:, 1] = pd.to_numeric(obs["y"], errors="coerce").fillna(0).to_numpy(np.float32)

    confidence = (
        pd.to_numeric(obs["Supertype confidence"], errors="coerce").fillna(1).to_numpy(np.float32)
        if "Supertype confidence" in obs.columns
        else np.ones(obs.shape[0], dtype=np.float32)
    )
    cps = (
        pd.to_numeric(obs["Continuous Pseudo-progression Score"], errors="coerce").fillna(0).to_numpy(np.float32)
        if "Continuous Pseudo-progression Score" in obs.columns
        else np.zeros(obs.shape[0], dtype=np.float32)
    )

    x = sparse.csr_matrix(x).astype(np.float32)
    is_dummy = np.zeros(obs.shape[0], dtype=bool)
    if len(non_test_donor_ids):
        dummy_n = len(non_test_donor_ids)
        x = sparse.vstack([x, sparse.csr_matrix((dummy_n, x.shape[1]), dtype=np.float32)]).tocsr()
        y_class = np.concatenate([y_class, np.repeat(y_class[:1], dummy_n)])
        y_subclass = np.concatenate([y_subclass, np.repeat(y_subclass[:1], dummy_n)])
        y_supertype = np.concatenate([y_supertype, np.repeat(y_supertype[:1], dummy_n)])
        batch_donor = np.concatenate([batch_donor, non_test_donor_ids.astype(np.int64)])
        spatial = np.vstack([spatial, np.zeros((dummy_n, 2), dtype=np.float32)])
        confidence = np.concatenate([confidence, np.zeros(dummy_n, dtype=np.float32)])
        cps = np.concatenate([cps, np.zeros(dummy_n, dtype=np.float32)])
        dummy_obs = pd.DataFrame(index=[f"__scmas_dummy_{idx}" for idx in range(dummy_n)])
        for col in obs.columns:
            dummy_obs[col] = obs[col].iloc[0] if len(obs) else ""
        dummy_obs["is_scmas_dummy"] = True
        obs = obs.copy()
        obs["is_scmas_dummy"] = False
        obs = pd.concat([obs, dummy_obs], axis=0)
        is_dummy = np.concatenate([is_dummy, np.ones(dummy_n, dtype=bool)])
    output_dir = ensure_dir(output_dir)
    npz_path = output_dir / f"{dataset.dataset_id}.npz"
    h5ad_path = output_dir / f"{dataset.dataset_id}.h5ad"
    meta = {
        "num_class": maps["num_class"],
        "num_subclass": maps["num_subclass"],
        "num_supertype": maps["num_supertype"],
        "num_donor": int(len(np.unique(batch_donor))),
        "class_labels": maps["class_labels"],
        "subclass_labels": maps["subclass_labels"],
        "supertype_labels": maps["supertype_labels"],
        "source_dataset_id": dataset.dataset_id,
    }
    np.savez_compressed(
        npz_path,
        X=x.toarray() if x.shape[1] <= 1000 else x,
        y_class=y_class,
        y_subclass=y_subclass,
        y_supertype=y_supertype,
        batch_donor=batch_donor,
        y_supertype_confidence=confidence,
        spatial=spatial,
        cps=cps,
        meta=meta,
        is_scmas_dummy=is_dummy,
    )

    h5ad_obs = obs.copy()
    h5ad_obs["source_cell_id"] = h5ad_obs.index.astype(str)
    h5ad = ad.AnnData(
        X=x,
        obs=h5ad_obs,
        var=pd.DataFrame(index=pd.Index(var_names, name="feature_id")),
    )
    h5ad.write_h5ad(h5ad_path)
    return {
        "dataset_id": dataset.dataset_id,
        "npz_path": str(npz_path),
        "h5ad_path": str(h5ad_path),
        "n_obs": int(x.shape[0]),
        "n_vars": int(x.shape[1]),
        "contract": "seaad_140_npz",
    }


def _reorder_to_merfish(x: sparse.spmatrix, var_names: list[str]) -> tuple[sparse.spmatrix, list[str], str]:
    merfish_genes = load_merfish_genes()
    upper_to_pos = {normalize_gene_name(g): idx for idx, g in enumerate(var_names)}
    missing = [gene for gene in merfish_genes if normalize_gene_name(gene) not in upper_to_pos]
    if len(missing) > 5:
        return x, var_names, f"gene panel mismatch: missing {len(missing)}/140 SEA-AD genes"
    x = sparse.csr_matrix(x)
    cols = []
    for gene in merfish_genes:
        pos = upper_to_pos.get(normalize_gene_name(gene))
        if pos is None:
            cols.append(sparse.csr_matrix((x.shape[0], 1), dtype=x.dtype))
        else:
            cols.append(x[:, pos])
    return sparse.hstack(cols, format="csr"), merfish_genes, ""


def prepare_dataset_for_seaad_models(
    dataset: EvalDataset,
    output_dir: str | Path,
    *,
    max_cells: int = 0,
    seed: int = 3028,
) -> dict[str, Any]:
    path = Path(dataset.path)
    if dataset.kind == "kukanja_npz":
        return {
            "dataset_id": dataset.dataset_id,
            "status": "skipped",
            "reason": dataset.reason,
            "contract": dataset.contract,
        }
    if not path.exists():
        return {"dataset_id": dataset.dataset_id, "status": "skipped", "reason": f"missing dataset path: {path}"}

    try:
        if dataset.kind == "h5ad":
            adata = ad.read_h5ad(path)
            var_names = [str(x) for x in adata.var_names]
            x, var_names, reason = _reorder_to_merfish(adata.X, var_names)
            if reason:
                return {"dataset_id": dataset.dataset_id, "status": "skipped", "reason": reason}
            return {
                **_write_npz_and_h5ad(
                    dataset=dataset,
                    x=x,
                    obs=adata.obs.copy(),
                    var_names=var_names,
                    output_dir=Path(output_dir),
                    max_cells=max_cells,
                    seed=seed,
                ),
                "status": "prepared",
            }

        if dataset.kind == "scdesign3_bundle":
            if (path / "sim_counts.mtx").exists():
                counts = spio.mmread(str(path / "sim_counts.mtx")).tocsr().transpose().tocsr()
                obs = pd.read_csv(path / "sim_obs.csv")
                var = pd.read_csv(path / "sim_var.csv")
                if "cell_id" in obs.columns:
                    obs = obs.set_index("cell_id", drop=False)
                if "feature_id" in var.columns:
                    var = var.set_index("feature_id", drop=False)
            else:
                counts, obs, var = read_standard_bundle(path, orientation="gene_by_cell")
            var_names = (
                var["feature_id"].astype(str).tolist()
                if "feature_id" in var.columns
                else var.index.astype(str).tolist()
            )
            x, var_names, reason = _reorder_to_merfish(counts, var_names)
            if reason:
                return {"dataset_id": dataset.dataset_id, "status": "skipped", "reason": reason}
            return {
                **_write_npz_and_h5ad(
                    dataset=dataset,
                    x=x,
                    obs=obs.copy(),
                    var_names=var_names,
                    output_dir=Path(output_dir),
                    max_cells=max_cells,
                    seed=seed,
                ),
                "status": "prepared",
            }
    except Exception as exc:  # noqa: BLE001
        return {"dataset_id": dataset.dataset_id, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    return {"dataset_id": dataset.dataset_id, "status": "skipped", "reason": f"unsupported kind: {dataset.kind}"}


def prepare_all_eval_datasets(
    output_dir: str | Path,
    include_new_synthetic: bool = True,
    *,
    max_cells: int = 0,
    seed: int = 3028,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    rows = [
        prepare_dataset_for_seaad_models(ds, output_dir, max_cells=max_cells, seed=seed)
        for ds in discover_eval_datasets(include_new_synthetic)
    ]
    manifest = {"prepared_dir": str(output_dir), "datasets": rows}
    write_json(manifest, Path(output_dir) / "prepared_eval_datasets.json")
    return manifest
