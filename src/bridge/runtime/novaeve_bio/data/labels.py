from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from novaeve_bio import paths
from novaeve_bio.io import ensure_dir, read_json, write_json


def _category_values(series: pd.Series) -> list[str]:
    if hasattr(series, "cat"):
        return [str(x) for x in sorted(series.cat.categories)]
    return sorted({str(x) for x in series.dropna().astype(str).tolist()})


def load_merfish_genes() -> list[str]:
    if paths.SEAAD_MERFISH_GENES_JSON.exists():
        return [str(x) for x in read_json(paths.SEAAD_MERFISH_GENES_JSON)]
    adata = ad.read_h5ad(paths.SEAAD_MERFISH_H5AD, backed="r")
    try:
        return [str(x) for x in adata.var_names if not str(x).startswith("Blank")]
    finally:
        adata.file.close()


def build_seaad_label_maps(
    h5ad_path: str | Path = paths.SEAAD_MERFISH_H5AD,
    output_path: str | Path = paths.SEAAD_LABEL_MAPS_JSON,
    force: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return read_json(output_path)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs[["Class", "Subclass", "Supertype", "Donor ID"]].copy()
    finally:
        adata.file.close()

    class_labels = _category_values(obs["Class"])
    subclass_labels = _category_values(obs["Subclass"])
    supertype_labels = _category_values(obs["Supertype"])
    donor_labels = _category_values(obs["Donor ID"])
    class_to_id = {label: idx for idx, label in enumerate(class_labels)}
    subclass_to_id = {label: idx for idx, label in enumerate(subclass_labels)}
    supertype_to_id = {label: idx for idx, label in enumerate(supertype_labels)}
    donor_to_id = {label: idx for idx, label in enumerate(donor_labels)}

    supertype_to_class: dict[str, str] = {}
    supertype_to_subclass: dict[str, str] = {}
    for supertype, sub in obs.groupby("Supertype", observed=True):
        supertype = str(supertype)
        supertype_to_class[supertype] = Counter(sub["Class"].astype(str)).most_common(1)[0][0]
        supertype_to_subclass[supertype] = Counter(sub["Subclass"].astype(str)).most_common(1)[0][0]

    payload = {
        "class_labels": class_labels,
        "subclass_labels": subclass_labels,
        "supertype_labels": supertype_labels,
        "donor_labels": donor_labels,
        "class_to_id": class_to_id,
        "subclass_to_id": subclass_to_id,
        "supertype_to_id": supertype_to_id,
        "donor_to_id": donor_to_id,
        "supertype_to_class": supertype_to_class,
        "supertype_to_subclass": supertype_to_subclass,
        "num_class": len(class_labels),
        "num_subclass": len(subclass_labels),
        "num_supertype": len(supertype_labels),
        "num_donor": len(donor_labels),
        "merfish_genes": load_merfish_genes(),
    }
    ensure_dir(output_path.parent)
    write_json(payload, output_path)
    return payload


def seaad_label_ids_from_obs(obs: pd.DataFrame, maps: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "Supertype" not in obs.columns:
        raise ValueError("SEA-AD-compatible scoring requires a Supertype column.")
    supertype = obs["Supertype"].astype(str)
    missing = sorted(set(supertype).difference(maps["supertype_to_id"]))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} Supertype labels are not in SEA-AD map: {preview}")

    if "Class" in obs.columns:
        class_label = obs["Class"].astype(str)
    else:
        class_label = supertype.map(maps["supertype_to_class"])
    if "Subclass" in obs.columns:
        subclass_label = obs["Subclass"].astype(str)
    else:
        subclass_label = supertype.map(maps["supertype_to_subclass"])

    y_class = class_label.map(maps["class_to_id"]).to_numpy(dtype=np.int64)
    y_subclass = subclass_label.map(maps["subclass_to_id"]).to_numpy(dtype=np.int64)
    y_supertype = supertype.map(maps["supertype_to_id"]).to_numpy(dtype=np.int64)
    return y_class, y_subclass, y_supertype


def factorize_ids(values: pd.Series | np.ndarray, min_unique: int = 0) -> np.ndarray:
    arr = pd.Series(values).astype(str).fillna("__missing__").to_numpy()
    unique = sorted(set(arr))
    if min_unique and len(unique) < min_unique:
        # Legacy foundation-model test wrappers construct train/val loaders even
        # in test mode. Pseudo batches avoid empty train/val splits for small
        # held-out donor panels without changing expression or labels.
        return (np.arange(len(arr)) % min_unique).astype(np.int64)
    mapping = {value: idx for idx, value in enumerate(unique)}
    return np.asarray([mapping[x] for x in arr], dtype=np.int64)
