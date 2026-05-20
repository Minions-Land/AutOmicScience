from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from novaeve_bio import paths
from novaeve_bio.data.labels import build_seaad_label_maps, load_merfish_genes
from novaeve_bio.io import ensure_dir, normalize_gene_name, stratified_indices, write_json


def write_seaad_donor_split(
    test_donors: list[str] | None = None,
    output_path: str | Path = paths.SEAAD_DONOR_SPLIT_JSON,
) -> dict[str, Any]:
    test_donors = list(test_donors or paths.DEFAULT_TEST_DONORS)
    if paths.SEAAD_DONOR_H5AD_DIR.exists():
        donors = sorted(p.stem for p in paths.SEAAD_DONOR_H5AD_DIR.glob("*.h5ad"))
    else:
        adata = ad.read_h5ad(paths.SEAAD_MERFISH_H5AD, backed="r")
        try:
            donors = sorted({str(x) for x in adata.obs["Donor ID"].astype(str)})
        finally:
            adata.file.close()
    missing = sorted(set(test_donors).difference(donors))
    if missing:
        raise FileNotFoundError(f"SEA-AD test donors not found: {missing}")
    reference_donors = [donor for donor in donors if donor not in set(test_donors)]
    payload = {
        "test_donors": test_donors,
        "reference_donors": reference_donors,
        "n_test_donors": len(test_donors),
        "n_reference_donors": len(reference_donors),
        "has_overlap": bool(set(test_donors).intersection(reference_donors)),
    }
    write_json(payload, output_path)
    return payload


def _feature_symbols(adata: ad.AnnData) -> pd.Index:
    for column in ("feature_name", "gene_symbol", "gene_name"):
        if column in adata.var.columns:
            return pd.Index([normalize_gene_name(x) for x in adata.var[column].astype(str)])
    return pd.Index([normalize_gene_name(x) for x in adata.var_names.astype(str)])


def _deduplicate_var(adata: ad.AnnData) -> ad.AnnData:
    symbols = _feature_symbols(adata)
    keep = ~symbols.duplicated()
    adata = adata[:, keep].copy()
    symbols = symbols[keep]
    adata.var_names = symbols
    adata.var["feature_id"] = symbols
    return adata


def _align_to_feature_names(adata: ad.AnnData, feature_names: list[str]) -> ad.AnnData:
    desired = [normalize_gene_name(x) for x in feature_names]
    current = {normalize_gene_name(x): idx for idx, x in enumerate(adata.var_names.astype(str))}
    x = sparse.csr_matrix(adata.X)
    cols = []
    for gene in desired:
        idx = current.get(gene)
        if idx is None:
            cols.append(sparse.csr_matrix((adata.n_obs, 1), dtype=x.dtype))
        else:
            cols.append(x[:, idx])
    aligned = ad.AnnData(
        X=sparse.hstack(cols, format="csr"),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(desired, name=adata.var_names.name)),
        obsm={key: value.copy() for key, value in adata.obsm.items()},
    )
    aligned.var["feature_id"] = desired
    return aligned


def _filter_to_feature_names(adata: ad.AnnData, feature_names: list[str]) -> ad.AnnData:
    desired = {normalize_gene_name(x) for x in feature_names}
    symbols = _feature_symbols(adata)
    keep = symbols.isin(desired)
    out = adata[:, np.flatnonzero(keep)].copy()
    out.var_names = symbols[keep]
    out.var["feature_id"] = out.var_names.astype(str)
    return out


def _read_backed_csr_subset(path: Path, row_idx: np.ndarray, col_idx: np.ndarray, shape: tuple[int, int]) -> sparse.csr_matrix:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    col_idx = np.asarray(col_idx, dtype=np.int64)
    remap = np.full(shape[1], -1, dtype=np.int64)
    remap[col_idx] = np.arange(len(col_idx), dtype=np.int64)
    out_data: list[np.ndarray] = []
    out_indices: list[np.ndarray] = []
    out_indptr = [0]
    with h5py.File(path, "r") as handle:
        xgrp = handle["X"]
        data_ds = xgrp["data"]
        indices_ds = xgrp["indices"]
        indptr_ds = xgrp["indptr"]
        for row in row_idx:
            start = int(indptr_ds[int(row)])
            end = int(indptr_ds[int(row) + 1])
            cols = np.asarray(indices_ds[start:end], dtype=np.int64)
            mapped = remap[cols]
            keep = mapped >= 0
            if np.any(keep):
                out_data.append(np.asarray(data_ds[start:end])[keep])
                out_indices.append(mapped[keep].astype(np.int32, copy=False))
                out_indptr.append(out_indptr[-1] + int(np.sum(keep)))
            else:
                out_indptr.append(out_indptr[-1])
    data = np.concatenate(out_data) if out_data else np.asarray([], dtype=np.float32)
    indices = np.concatenate(out_indices) if out_indices else np.asarray([], dtype=np.int32)
    return sparse.csr_matrix((data, indices, np.asarray(out_indptr, dtype=np.int64)), shape=(len(row_idx), len(col_idx)))


def _h5ad_x_is_csr(path: Path) -> bool:
    with h5py.File(path, "r") as handle:
        return str(handle["X"].attrs.get("encoding-type", "")) == "csr_matrix"


def _sample_h5ad(
    path: Path,
    *,
    max_cells: int,
    seed: int,
    label_column: str | None = None,
    feature_names: list[str] | None = None,
    matrix_source: str = "X",
    fill_missing_features: bool = True,
) -> ad.AnnData:
    backed = ad.read_h5ad(path, backed="r")
    try:
        obs = backed.obs.copy()
        if label_column and label_column in obs.columns:
            idx = stratified_indices(obs[label_column], max_cells=max_cells, min_per_group=20, seed=seed)
        else:
            idx = np.arange(backed.n_obs, dtype=np.int64)
            if max_cells > 0 and max_cells < len(idx):
                rng = np.random.default_rng(seed)
                idx = np.sort(rng.choice(idx, size=max_cells, replace=False))
        if not isinstance(idx, slice) and len(idx) == backed.n_obs and np.array_equal(idx, np.arange(backed.n_obs, dtype=np.int64)):
            idx = slice(None)

        if feature_names:
            desired = {normalize_gene_name(g) for g in feature_names}
            symbols = _feature_symbols(backed)
            var_idx = np.flatnonzero(symbols.isin(desired))
        else:
            var_idx = slice(None)

        idx_is_slice = isinstance(idx, slice)
        var_is_slice = isinstance(var_idx, slice)
        if (
            matrix_source == "X"
            and not var_is_slice
            and not idx_is_slice
            and _h5ad_x_is_csr(path)
            and len(var_idx) != backed.n_vars
            and len(idx) != backed.n_obs
        ):
            x = _read_backed_csr_subset(path, np.asarray(idx, dtype=np.int64), np.asarray(var_idx, dtype=np.int64), backed.shape)
            sub = ad.AnnData(X=x, obs=obs.iloc[idx].copy(), var=backed.var.iloc[var_idx].copy())
        elif not var_is_slice and not idx_is_slice and len(var_idx) != backed.n_vars and len(idx) != backed.n_obs:
            # h5py-backed AnnData cannot fancy-index both axes in one read.
            # SEA-AD MERFISH uses this path with a small 140-gene panel, so
            # materializing the gene-filtered view first keeps memory bounded.
            sub = backed[:, var_idx].to_memory()[idx, :].copy()
        else:
            sub = backed[idx, var_idx].to_memory()
        if matrix_source.startswith("layer:"):
            layer_name = matrix_source.split(":", 1)[1]
            sub.X = sub.layers[layer_name]
        elif matrix_source == "raw":
            if sub.raw is None:
                raise ValueError(f"{path} has no raw matrix")
            sub = ad.AnnData(X=sub.raw.X, obs=sub.obs.copy(), var=sub.raw.var.copy())
        elif matrix_source != "X":
            raise ValueError(f"Unsupported matrix_source: {matrix_source}")
    finally:
        backed.file.close()
    sub = _deduplicate_var(sub)
    if feature_names:
        sub = _align_to_feature_names(sub, feature_names) if fill_missing_features else _filter_to_feature_names(sub, feature_names)
    return sub


def _set_unified_obs(
    adata: ad.AnnData,
    *,
    species: str,
    source_dataset: str,
    source_path: Path,
    donor_col: str | None = None,
    sample_col: str | None = None,
    native_label_col: str | None = None,
    coarse_label_col: str | None = None,
    split_role: str = "reference",
) -> ad.AnnData:
    obs = adata.obs.copy()
    donor_col = donor_col if donor_col in obs.columns else None
    sample_col = sample_col if sample_col in obs.columns else None
    native_label_col = native_label_col if native_label_col in obs.columns else None
    coarse_label_col = coarse_label_col if coarse_label_col in obs.columns else None

    obs["species"] = species
    obs["source_dataset"] = source_dataset
    obs["source_path"] = str(source_path)
    obs["donor_id"] = obs[donor_col].astype(str) if donor_col else "unknown"
    obs["sample_id"] = obs[sample_col].astype(str) if sample_col else obs["donor_id"].astype(str)
    obs["native_label"] = obs[native_label_col].astype(str) if native_label_col else "unknown"
    obs["coarse_label"] = obs[coarse_label_col].astype(str) if coarse_label_col else obs["native_label"]
    obs["split_role"] = split_role
    adata.obs = obs
    return adata


def _load_taxonomy_map(prefix: str) -> pd.DataFrame:
    root = paths.ALLEN_WHOLE_BRAIN_DIR
    membership = root / (
        "whb_cluster_to_cluster_annotation_membership.csv"
        if prefix == "whb"
        else "cluster_to_cluster_annotation_membership.csv"
    )
    df = pd.read_csv(
        membership,
        usecols=[
            "cluster_alias",
            "cluster_annotation_term_name",
            "cluster_annotation_term_set_name",
        ],
    )
    keep_names = {
        "whb": {"supercluster", "cluster", "subcluster", "neurotransmitter"},
        "wmb": {"class", "subclass", "supertype", "cluster", "neurotransmitter"},
    }[prefix]
    df = df[df["cluster_annotation_term_set_name"].isin(keep_names)].copy()
    out = df.pivot_table(
        index="cluster_alias",
        columns="cluster_annotation_term_set_name",
        values="cluster_annotation_term_name",
        aggfunc="first",
    )
    out.columns = [str(c) for c in out.columns]
    return out.reset_index()


def _join_cell_metadata_chunked(obs: pd.DataFrame, metadata_path: Path, usecols: list[str]) -> pd.DataFrame:
    wanted = set(obs.index.astype(str))
    hits: list[pd.DataFrame] = []
    for chunk in pd.read_csv(metadata_path, usecols=usecols, chunksize=500_000):
        chunk["cell_label"] = chunk["cell_label"].astype(str)
        sub = chunk[chunk["cell_label"].isin(wanted)]
        if not sub.empty:
            hits.append(sub)
    if not hits:
        return obs
    meta = pd.concat(hits, ignore_index=True).drop_duplicates("cell_label").set_index("cell_label")
    overlap = [column for column in meta.columns if column in obs.columns]
    if overlap:
        meta = meta.drop(columns=overlap)
    return obs.join(meta, how="left")


def annotate_whole_brain_obs(adata: ad.AnnData, *, species: str) -> ad.AnnData:
    if species == "human":
        meta_path = paths.ALLEN_WHOLE_BRAIN_DIR / "whb_cell_metadata.csv"
        prefix = "whb"
        usecols = ["cell_label", "donor_label", "cluster_alias", "anatomical_division_label"]
    else:
        meta_path = paths.ALLEN_WHOLE_BRAIN_DIR / "cell_metadata.csv"
        prefix = "wmb"
        usecols = ["cell_label", "donor_label", "cluster_alias", "anatomical_division_label"]
    available = set(pd.read_csv(meta_path, nrows=0).columns.astype(str))
    usecols = [column for column in usecols if column in available]
    obs = _join_cell_metadata_chunked(adata.obs.copy(), meta_path, usecols)
    if "cluster_alias" in obs.columns:
        tax = _load_taxonomy_map(prefix)
        obs = obs.merge(tax, how="left", on="cluster_alias", suffixes=("", "_tax"))
        obs.index = adata.obs.index
    adata.obs = obs
    return adata


def _read_gene_by_cell_csv_sample(
    *,
    counts_path: Path,
    metadata_path: Path,
    metadata_sep: str,
    counts_sep: str = ",",
    cell_id_col: str,
    label_col: str,
    sample_col: str,
    species: str,
    source_dataset: str,
    max_cells: int,
    seed: int,
    max_genes: int = 0,
    feature_names: list[str] | None = None,
) -> ad.AnnData:
    meta = pd.read_csv(metadata_path, sep=metadata_sep)
    if cell_id_col not in meta.columns:
        raise KeyError(f"{metadata_path} lacks {cell_id_col}")
    meta = meta.set_index(cell_id_col, drop=False)
    idx = stratified_indices(meta[label_col], max_cells=max_cells, min_per_group=20, seed=seed)
    selected_meta_ids = meta.iloc[idx].index.astype(str).tolist()

    header = pd.read_csv(counts_path, sep=counts_sep, nrows=0)
    first_col = header.columns[0]
    count_columns = set(header.columns.astype(str))
    id_to_count_col: dict[str, str] = {}
    for cell_id in selected_meta_ids:
        if cell_id in count_columns:
            id_to_count_col[cell_id] = cell_id
            continue
        dash_to_dot = cell_id.replace("-", ".")
        if dash_to_dot in count_columns:
            id_to_count_col[cell_id] = dash_to_dot
            continue
        dot_to_dash = cell_id.replace(".", "-")
        if dot_to_dash in count_columns:
            id_to_count_col[cell_id] = dot_to_dash
    if not id_to_count_col:
        raise ValueError(f"No selected metadata cell IDs matched columns in {counts_path}")

    count_to_meta_id = {v: k for k, v in id_to_count_col.items()}
    usecols = [first_col, *id_to_count_col.values()]
    usecols_set = set(usecols)

    desired_features = {normalize_gene_name(x) for x in feature_names or []}
    if desired_features:
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            counts_path,
            sep=counts_sep,
            usecols=lambda c: c in usecols_set,
            chunksize=512,
        ):
            symbols = chunk[first_col].astype(str).map(normalize_gene_name)
            sub = chunk[symbols.isin(desired_features)].copy()
            if not sub.empty:
                sub[first_col] = symbols[symbols.isin(desired_features)].to_numpy()
                chunks.append(sub)
        if not chunks:
            raise ValueError(f"No requested feature_names were found in {counts_path}")
        counts = pd.concat(chunks, ignore_index=True).drop_duplicates(first_col).set_index(first_col)
        x = sparse.csr_matrix(counts.transpose().to_numpy(dtype=np.float32))
        obs = meta.loc[[count_to_meta_id[str(c)] for c in counts.columns.astype(str)]].copy()
        obs.index = counts.columns.astype(str)
        var = pd.DataFrame(index=[normalize_gene_name(x) for x in counts.index.astype(str)])
    elif max_genes and max_genes > 0:
        # These GEO matrices are dense CSV.gz files with genes as rows and cells
        # as columns.  Reading all genes before subsetting can spend tens of
        # minutes converting a huge dense table.  Stream row chunks and keep only
        # the highest-detected genes needed by the caller.
        import heapq
        import itertools

        heap: list[tuple[float, int, int, str, np.ndarray]] = []
        counter = itertools.count()
        row_offset = 0
        for chunk in pd.read_csv(counts_path, sep=counts_sep, usecols=lambda c: c in usecols_set, chunksize=256):
            gene_names = chunk[first_col].astype(str).to_numpy()
            values = chunk.drop(columns=[first_col]).to_numpy(dtype=np.float32, copy=False)
            scores = np.count_nonzero(values, axis=1).astype(float)
            current_min = heap[0][0] if len(heap) >= max_genes else -1.0
            candidate_idx = np.flatnonzero(scores > current_min)
            if candidate_idx.size:
                candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
            for idx_in_chunk in candidate_idx:
                item = (
                    float(scores[idx_in_chunk]),
                    next(counter),
                    row_offset + int(idx_in_chunk),
                    str(gene_names[idx_in_chunk]),
                    np.asarray(values[idx_in_chunk], dtype=np.float32).copy(),
                )
                if len(heap) < max_genes:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
            row_offset += len(chunk)
        if not heap:
            raise ValueError(f"No genes were read from {counts_path}")
        selected = sorted(heap, key=lambda item: item[2])
        count_columns = [c for c in usecols if c != first_col]
        gene_names = [item[3] for item in selected]
        gene_by_cell = np.vstack([item[4] for item in selected])
        x = sparse.csr_matrix(gene_by_cell.T)
        obs = meta.loc[[count_to_meta_id[str(c)] for c in count_columns]].copy()
        obs.index = pd.Index(count_columns, dtype=str)
        var = pd.DataFrame(index=[normalize_gene_name(gene) for gene in gene_names])
    else:
        counts = pd.read_csv(counts_path, sep=counts_sep, usecols=lambda c: c in usecols_set)
        counts = counts.set_index(first_col)
        # Gene rows x selected cells columns -> cell x gene matrix.
        x = sparse.csr_matrix(counts.transpose().to_numpy(dtype=np.float32))
        obs = meta.loc[[count_to_meta_id[str(c)] for c in counts.columns.astype(str)]].copy()
        obs.index = counts.columns.astype(str)
        var = pd.DataFrame(index=[normalize_gene_name(x) for x in counts.index.astype(str)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata = _deduplicate_var(adata)
    return _set_unified_obs(
        adata,
        species=species,
        source_dataset=source_dataset,
        source_path=counts_path,
        donor_col=sample_col,
        sample_col=sample_col,
        native_label_col=label_col,
        coarse_label_col=label_col,
    )


def _read_smartseq_sample(
    max_cells: int,
    seed: int,
    include: bool,
    feature_names: list[str] | None = None,
) -> ad.AnnData | None:
    if not include:
        return None
    matrix_path = paths.ALLEN_HUMAN_SMARTSEQ_DIR / "matrix.csv"
    metadata_path = paths.ALLEN_HUMAN_SMARTSEQ_DIR / "metadata.csv"
    meta = pd.read_csv(metadata_path)
    meta = meta[meta["outlier_call"] == False].copy()  # noqa: E712
    idx = stratified_indices(meta["subclass_label"], max_cells=max_cells, min_per_group=10, seed=seed)
    wanted = set(meta.iloc[idx]["sample_name"].astype(str))

    rows: list[pd.DataFrame] = []
    usecols = None
    if feature_names:
        header = pd.read_csv(matrix_path, nrows=0).columns.astype(str).tolist()
        wanted_genes = {normalize_gene_name(x) for x in feature_names}
        usecols = ["sample_name", *[col for col in header if normalize_gene_name(col) in wanted_genes]]
    for chunk in pd.read_csv(matrix_path, chunksize=1000, usecols=usecols):
        sub = chunk[chunk["sample_name"].astype(str).isin(wanted)]
        if not sub.empty:
            rows.append(sub)
    if not rows:
        return None
    mat = pd.concat(rows, ignore_index=True).set_index("sample_name")
    obs = meta.set_index("sample_name").loc[mat.index].copy()
    var = pd.DataFrame(index=[normalize_gene_name(x) for x in mat.columns.astype(str)])
    adata = ad.AnnData(X=sparse.csr_matrix(mat.to_numpy(dtype=np.float32)), obs=obs, var=var)
    adata = _deduplicate_var(adata)
    return _set_unified_obs(
        adata,
        species="human",
        source_dataset="allen_human_multiple_cortical_areas_smartseq",
        source_path=matrix_path,
        donor_col="external_donor_name_label",
        sample_col="external_donor_name_label",
        native_label_col="cluster_label",
        coarse_label_col="subclass_label",
    )


def build_reference(
    *,
    output_path: str | Path = paths.REFERENCE_H5AD,
    max_cells_per_source: int = 10_000,
    max_cells_seaad_reference: int = 100_000,
    max_cells_seaad_test: int = 50_000,
    include_smartseq: bool = False,
    seed: int = 3028,
    write_test_h5ad: bool = True,
) -> dict[str, Any]:
    ensure_dir(paths.REFERENCE_DIR)
    ensure_dir(paths.TEST_DIR)
    build_seaad_label_maps()
    split = write_seaad_donor_split()
    merfish_genes = load_merfish_genes()

    components: list[ad.AnnData] = []
    component_rows: list[dict[str, Any]] = []

    seaad_ref_per_donor = max(1, max_cells_seaad_reference // max(1, len(split["reference_donors"])))
    for donor in split["reference_donors"]:
        donor_path = paths.SEAAD_DONOR_H5AD_DIR / f"{donor}.h5ad"
        if not donor_path.exists():
            continue
        comp = _sample_h5ad(
            donor_path,
            max_cells=seaad_ref_per_donor,
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
        components.append(comp)
        component_rows.append({"source_id": f"seaad_ref_{donor}", "n_obs": comp.n_obs, "n_vars": comp.n_vars})

    if write_test_h5ad:
        test_components: list[ad.AnnData] = []
        per_donor = max(1, max_cells_seaad_test // max(1, len(split["test_donors"])))
        for donor in split["test_donors"]:
            donor_path = paths.SEAAD_DONOR_H5AD_DIR / f"{donor}.h5ad"
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
                source_dataset="seaad_merfish_test_donors",
                source_path=donor_path,
                donor_col="Donor ID",
                sample_col="Donor ID",
                native_label_col="Supertype",
                coarse_label_col="Subclass",
                split_role="test",
            )
            test_components.append(comp)
        test = ad.concat(test_components, join="outer", fill_value=0, merge="same")
        test.X = sparse.csr_matrix(test.X)
        test.write_h5ad(paths.SEAAD_TEST_H5AD)

    for h5ad_path in sorted(paths.ALLEN_MOUSE_REFERENCE_DIR.glob("*.h5ad")):
        comp = _sample_h5ad(
            h5ad_path,
            max_cells=max_cells_per_source,
            seed=seed,
            label_column="cell_type",
            feature_names=None,
        )
        comp = _set_unified_obs(
            comp,
            species="mouse",
            source_dataset="allen_mouse_reference",
            source_path=h5ad_path,
            donor_col="donor_id",
            sample_col="slice",
            native_label_col="cell_type",
            coarse_label_col="cell_type_annot",
        )
        components.append(comp)
        component_rows.append({"source_id": f"allen_mouse_reference/{h5ad_path.name}", "n_obs": comp.n_obs, "n_vars": comp.n_vars})

    for h5ad_path in sorted(paths.HUMAN_WHOLE_BRAIN_DIR.glob("*.h5ad")):
        comp = _sample_h5ad(
            h5ad_path,
            max_cells=max_cells_per_source,
            seed=seed,
            label_column="anatomical_division_label",
            feature_names=None,
        )
        comp = annotate_whole_brain_obs(comp, species="human")
        comp = _set_unified_obs(
            comp,
            species="human",
            source_dataset="human_whole_brain",
            source_path=h5ad_path,
            donor_col="donor_label",
            sample_col="library_label",
            native_label_col="cluster",
            coarse_label_col="supercluster",
        )
        components.append(comp)
        component_rows.append({"source_id": f"human_whole_brain/{h5ad_path.name}", "n_obs": comp.n_obs, "n_vars": comp.n_vars})

    for h5ad_path in sorted(paths.MOUSE_WHOLE_BRAIN_DIR.glob("*.h5ad")):
        comp = _sample_h5ad(
            h5ad_path,
            max_cells=max_cells_per_source,
            seed=seed,
            label_column="anatomical_division_label",
            feature_names=None,
        )
        comp = annotate_whole_brain_obs(comp, species="mouse")
        comp = _set_unified_obs(
            comp,
            species="mouse",
            source_dataset="mouse_whole_brain",
            source_path=h5ad_path,
            donor_col="donor_label",
            sample_col="library_label",
            native_label_col="supertype",
            coarse_label_col="class",
        )
        components.append(comp)
        component_rows.append({"source_id": f"mouse_whole_brain/{h5ad_path.name}", "n_obs": comp.n_obs, "n_vars": comp.n_vars})

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
    )
    components.append(spinal_190442)
    component_rows.append({"source_id": "spinal_gse190442", "n_obs": spinal_190442.n_obs, "n_vars": spinal_190442.n_vars})

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
    )
    components.append(spinal_103892)
    component_rows.append({"source_id": "spinal_gse103892", "n_obs": spinal_103892.n_obs, "n_vars": spinal_103892.n_vars})

    smartseq = _read_smartseq_sample(max_cells=max_cells_per_source, seed=seed, include=include_smartseq)
    if smartseq is not None:
        components.append(smartseq)
        component_rows.append({"source_id": "allen_human_multiple_cortical_areas_smartseq", "n_obs": smartseq.n_obs, "n_vars": smartseq.n_vars})

    for comp in components:
        comp.X = sparse.csr_matrix(comp.X)

    merged = ad.concat(components, join="outer", fill_value=0, merge="same", label="reference_component")
    merged.X = sparse.csr_matrix(merged.X)
    merged.var["feature_symbol_upper"] = merged.var_names.astype(str)
    merged.uns["scmas_reference"] = {
        "seed": seed,
        "test_donors": split["test_donors"],
        "reference_donors": split["reference_donors"],
        "component_summary": component_rows,
    }
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    merged.write_h5ad(output_path)

    manifest = {
        "reference_h5ad": str(output_path),
        "seaad_test_h5ad": str(paths.SEAAD_TEST_H5AD) if write_test_h5ad else "",
        "n_obs": int(merged.n_obs),
        "n_vars": int(merged.n_vars),
        "components": component_rows,
        "split": split,
    }
    write_json(manifest, paths.REFERENCE_DIR / "reference_manifest.json")
    return manifest


def build_seaad_test_h5ad(
    *,
    output_path: str | Path = paths.SEAAD_TEST_H5AD,
    max_cells: int = 50_000,
    seed: int = 3028,
) -> dict[str, Any]:
    ensure_dir(paths.TEST_DIR)
    split = write_seaad_donor_split()
    merfish_genes = load_merfish_genes()
    test_components: list[ad.AnnData] = []
    per_donor = max(1, max_cells // max(1, len(split["test_donors"])))
    for donor in split["test_donors"]:
        donor_path = paths.SEAAD_DONOR_H5AD_DIR / f"{donor}.h5ad"
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
            source_dataset="seaad_merfish_test_donors",
            source_path=donor_path,
            donor_col="Donor ID",
            sample_col="Donor ID",
            native_label_col="Supertype",
            coarse_label_col="Subclass",
            split_role="test",
        )
        test_components.append(comp)
    test = ad.concat(test_components, join="outer", fill_value=0, merge="same")
    test.X = sparse.csr_matrix(test.X)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    test.write_h5ad(output_path)
    manifest = {
        "seaad_test_h5ad": str(output_path),
        "test_donors": split["test_donors"],
        "n_obs": int(test.n_obs),
        "n_vars": int(test.n_vars),
        "max_cells": int(max_cells),
        "seed": int(seed),
    }
    write_json(manifest, paths.TEST_DIR / "seaad_test_manifest.json")
    return manifest
