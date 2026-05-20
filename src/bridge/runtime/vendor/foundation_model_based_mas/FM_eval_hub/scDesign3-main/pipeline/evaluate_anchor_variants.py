#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score


DEFAULT_VARIANTS = [
    "baseline_anchor_100k",
    "variant3_virtual_batch_100k",
    "variant4_missing_celltypes_100k",
]

DONOR_COLUMN_CANDIDATES = ["Donor ID", "Donor.ID", "donor_id", "donor"]
CELLTYPE_COLUMN_CANDIDATES = ["Supertype", "supertype", "cell_type", "celltype"]


@dataclass
class VariantData:
    name: str
    counts: sparse.csr_matrix  # cell x gene
    obs: pd.DataFrame
    var: pd.DataFrame
    summary: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_variant(variant_dir: Path) -> VariantData:
    counts_path = variant_dir / "sim_counts.mtx"
    obs_path = variant_dir / "sim_obs.csv"
    var_path = variant_dir / "sim_var.csv"
    summary_path = variant_dir / "variant_summary.json"

    if not counts_path.exists():
        raise FileNotFoundError(f"缺少 counts 文件: {counts_path}")
    if not obs_path.exists():
        raise FileNotFoundError(f"缺少 obs 文件: {obs_path}")
    if not var_path.exists():
        raise FileNotFoundError(f"缺少 var 文件: {var_path}")

    counts = spio.mmread(str(counts_path))
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    counts = counts.tocsr().transpose().tocsr()  # gene x cell -> cell x gene

    obs = pd.read_csv(obs_path)
    cell_id_col = "cell_id" if "cell_id" in obs.columns else obs.columns[0]
    obs = obs.set_index(cell_id_col)

    var = pd.read_csv(var_path)
    feature_id_col = "feature_id" if "feature_id" in var.columns else var.columns[0]
    var = var.set_index(feature_id_col)

    if counts.shape[0] != obs.shape[0]:
        raise ValueError(f"{variant_dir.name}: counts/obs 细胞数不一致")
    if counts.shape[1] != var.shape[0]:
        raise ValueError(f"{variant_dir.name}: counts/var 基因数不一致")

    summary = _read_json(summary_path) if summary_path.exists() else {}
    return VariantData(
        name=variant_dir.name,
        counts=counts,
        obs=obs,
        var=var,
        summary=summary,
    )


def _resolve_column(df: pd.DataFrame, requested: str, candidates: List[str]) -> str:
    if requested in df.columns:
        return requested
    for cand in [requested, *candidates]:
        if cand in df.columns:
            return cand
    normalized = {col.replace(" ", ".").lower(): col for col in df.columns}
    alt_keys = [requested, *candidates]
    for key in alt_keys:
        norm = key.replace(" ", ".").lower()
        if norm in normalized:
            return normalized[norm]
    raise KeyError(f"列不存在: {requested}；可用列包括: {list(df.columns[:20])}")


def _sample_indices(n: int, max_cells: int, seed: int) -> np.ndarray:
    if n <= max_cells:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_cells, replace=False)
    idx.sort()
    return idx


def _compute_embedding(counts: sparse.csr_matrix, n_components: int = 20) -> np.ndarray:
    libsize = np.asarray(counts.sum(axis=1)).ravel()
    libsize[libsize == 0] = 1.0
    norm = counts.multiply(1e4 / libsize[:, None])
    norm.data = np.log1p(norm.data)
    n_components = min(n_components, max(2, norm.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=1)
    return svd.fit_transform(norm)


def _safe_silhouette(embedding: np.ndarray, labels: Iterable[str]) -> float | None:
    labels = pd.Series(list(labels)).astype(str)
    if labels.nunique() < 2:
        return None
    counts = labels.value_counts()
    valid = labels.isin(counts[counts >= 2].index)
    if valid.sum() < 10 or labels[valid].nunique() < 2:
        return None
    return float(silhouette_score(embedding[valid.to_numpy()], labels[valid].to_numpy(), metric="euclidean"))


def _pseudobulk_metrics(
    counts: sparse.csr_matrix,
    obs: pd.DataFrame,
    donor_col: str,
    celltype_col: str,
    min_cells: int = 20,
) -> Dict[str, float]:
    donor_values = obs[donor_col].astype(str).to_numpy()
    celltype_values = obs[celltype_col].astype(str).to_numpy()
    groups = obs.groupby([donor_col, celltype_col]).indices

    per_group: Dict[tuple[str, str], np.ndarray] = {}
    for (donor, celltype), idx in groups.items():
        if len(idx) < min_cells:
            continue
        mean_vec = np.asarray(counts[idx].mean(axis=0)).ravel()
        per_group[(str(donor), str(celltype))] = np.log1p(mean_vec)

    if not per_group:
        return {
            "shared_celltypes": 0.0,
            "mean_ct_donor_l1": float("nan"),
            "mean_ct_donor_corr": float("nan"),
        }

    celltype_to_donors: Dict[str, List[np.ndarray]] = {}
    for (donor, celltype), vec in per_group.items():
        celltype_to_donors.setdefault(celltype, []).append(vec)

    l1_list: List[float] = []
    corr_list: List[float] = []
    shared = 0
    for celltype, vecs in celltype_to_donors.items():
        if len(vecs) < 2:
            continue
        shared += 1
        global_mean = np.mean(np.stack(vecs, axis=0), axis=0)
        for vec in vecs:
            l1_list.append(float(np.mean(np.abs(vec - global_mean))))
            if np.std(vec) == 0 or np.std(global_mean) == 0:
                corr_list.append(1.0)
            else:
                corr_list.append(float(np.corrcoef(vec, global_mean)[0, 1]))

    return {
        "shared_celltypes": float(shared),
        "mean_ct_donor_l1": float(np.mean(l1_list)) if l1_list else float("nan"),
        "mean_ct_donor_corr": float(np.mean(corr_list)) if corr_list else float("nan"),
    }


def _variant_metrics(
    data: VariantData,
    donor_col: str,
    celltype_col: str,
    max_cells_for_pca: int,
    seed: int,
) -> Dict[str, Any]:
    donor_col = _resolve_column(data.obs, donor_col, DONOR_COLUMN_CANDIDATES)
    celltype_col = _resolve_column(data.obs, celltype_col, CELLTYPE_COLUMN_CANDIDATES)
    sampled_idx = _sample_indices(data.counts.shape[0], max_cells_for_pca, seed)
    sampled_counts = data.counts[sampled_idx]
    sampled_obs = data.obs.iloc[sampled_idx].copy()
    embedding = _compute_embedding(sampled_counts)

    donor_counts = data.obs[donor_col].astype(str).value_counts().sort_index()
    celltype_counts = data.obs[celltype_col].astype(str).value_counts().sort_index()

    pseudo = _pseudobulk_metrics(data.counts, data.obs, donor_col=donor_col, celltype_col=celltype_col)
    return {
        "variant_name": data.name,
        "n_cells": int(data.obs.shape[0]),
        "n_genes": int(data.var.shape[0]),
        "n_donors": int(donor_counts.shape[0]),
        "n_celltypes": int(celltype_counts.shape[0]),
        "donor_balance_cv": float(donor_counts.std() / donor_counts.mean()) if donor_counts.mean() > 0 else float("nan"),
        "celltype_balance_cv": float(celltype_counts.std() / celltype_counts.mean()) if celltype_counts.mean() > 0 else float("nan"),
        "donor_silhouette_pca": _safe_silhouette(embedding, sampled_obs[donor_col]),
        "celltype_silhouette_pca": _safe_silhouette(embedding, sampled_obs[celltype_col]),
        **pseudo,
    }


def _variant4_missing_metrics(
    data: VariantData,
    donor_col: str,
    celltype_col: str,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    donor_col = _resolve_column(data.obs, donor_col, DONOR_COLUMN_CANDIDATES)
    celltype_col = _resolve_column(data.obs, celltype_col, CELLTYPE_COLUMN_CANDIDATES)
    donors = summary.get("donors", [])
    missing_celltypes = summary.get("missing_celltypes", [])
    if not donors or not missing_celltypes:
        return {}

    ct_table = pd.crosstab(data.obs[donor_col].astype(str), data.obs[celltype_col].astype(str))
    target_zero = 0
    target_total = 0
    other_positive = 0
    other_total = 0

    for donor in donors:
        for celltype in missing_celltypes:
            target_total += 1
            val = int(ct_table.get(celltype, pd.Series(dtype=int)).get(donor, 0))
            if val == 0:
                target_zero += 1

    other_donors = [d for d in ct_table.index.tolist() if d not in set(map(str, donors))]
    for donor in other_donors:
        for celltype in missing_celltypes:
            other_total += 1
            val = int(ct_table.get(celltype, pd.Series(dtype=int)).get(donor, 0))
            if val > 0:
                other_positive += 1

    return {
        "target_donors_zero_rate": float(target_zero / target_total) if target_total else float("nan"),
        "non_target_donors_positive_rate": float(other_positive / other_total) if other_total else float("nan"),
        "target_donors": donors,
        "missing_celltypes": missing_celltypes,
    }


def _compare_metrics(
    baseline: Dict[str, Any],
    other: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ["donor_silhouette_pca", "celltype_silhouette_pca", "mean_ct_donor_l1", "mean_ct_donor_corr"]:
        a = baseline.get(key)
        b = other.get(key)
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            out[f"{prefix}_{key}_delta_vs_baseline"] = None
        else:
            out[f"{prefix}_{key}_delta_vs_baseline"] = float(b - a)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Anchor Donor 方案下 variant3/4 是否满足设计目标")
    parser.add_argument("--run-dir", required=True, help="包含各 variant 子目录的运行目录")
    parser.add_argument("--donor-column", default="Donor ID", help="obs 中 donor 列名")
    parser.add_argument("--celltype-column", default="Supertype", help="obs 中 celltype 列名")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help="要评估的子目录，逗号分隔")
    parser.add_argument("--max-cells-for-pca", type=int, default=20000, help="PCA/silhouette 最大抽样细胞数")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--output-json", default=None, help="输出 JSON 路径")
    parser.add_argument("--output-tsv", default=None, help="输出 TSV 路径")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    variant_names = [x.strip() for x in args.variants.split(",") if x.strip()]
    variant_datas = [_load_variant(run_dir / name) for name in variant_names]

    result: Dict[str, Any] = {"run_dir": str(run_dir), "variants": {}, "comparisons": {}}
    metrics_map: Dict[str, Dict[str, Any]] = {}
    for data in variant_datas:
        metrics = _variant_metrics(
            data,
            donor_col=args.donor_column,
            celltype_col=args.celltype_column,
            max_cells_for_pca=args.max_cells_for_pca,
            seed=args.seed,
        )
        if data.name == "variant4_missing_celltypes_100k":
            metrics.update(_variant4_missing_metrics(data, args.donor_column, args.celltype_column, data.summary))
        metrics_map[data.name] = metrics
        result["variants"][data.name] = metrics

    baseline_name = "baseline_anchor_100k"
    if baseline_name in metrics_map:
        baseline = metrics_map[baseline_name]
        for name, metrics in metrics_map.items():
            if name == baseline_name:
                continue
            result["comparisons"][name] = _compare_metrics(baseline, metrics, prefix=name)

    output_json = Path(args.output_json) if args.output_json else run_dir / "anchor_variant_eval_metrics.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = []
    for name, metrics in result["variants"].items():
        row = {"variant_name": name, **metrics}
        flat_rows.append(row)
    flat_df = pd.DataFrame(flat_rows)
    output_tsv = Path(args.output_tsv) if args.output_tsv else run_dir / "anchor_variant_eval_metrics.tsv"
    flat_df.to_csv(output_tsv, sep="\t", index=False)

    print("[INFO] 评测完成")
    print(f"[INFO] JSON: {output_json}")
    print(f"[INFO] TSV : {output_tsv}")
    print("\n=== 关键指标 ===")
    for name, metrics in result["variants"].items():
        print(f"[{name}]")
        for key in [
            "donor_silhouette_pca",
            "celltype_silhouette_pca",
            "mean_ct_donor_l1",
            "mean_ct_donor_corr",
            "target_donors_zero_rate",
            "non_target_donors_positive_rate",
        ]:
            if key in metrics:
                print(f"  {key}: {metrics[key]}")
    if result["comparisons"]:
        print("\n=== 相对 baseline 的变化 ===")
        for name, metrics in result["comparisons"].items():
            print(f"[{name}]")
            for key, value in metrics.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
