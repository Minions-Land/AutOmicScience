#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
R_WORKER = SCRIPT_DIR / "run_scdesign3_controlled.R"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_feature_names_from_path(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    features: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        features.extend([part for part in parts if part])
    return features


def _normalize_feature_spec(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(x) for x in value if str(x).strip()]
    raise TypeError(f"不支持的 feature 规格类型: {type(value)!r}")


def _ensure_unique(index: pd.Index, name: str) -> pd.Index:
    if index.is_unique:
        return index
    seen: Dict[str, int] = {}
    new_values = []
    for raw_value in index.astype(str):
        value = str(raw_value)
        count = seen.get(value, 0)
        if count == 0:
            new_values.append(value)
        else:
            new_values.append(f"{value}-{count}")
        seen[value] = count + 1
    print(f"[WARN] {name} 存在重复，已自动改成唯一名称。", file=sys.stderr)
    return pd.Index(new_values)


def _to_csr(x: Any) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr()
    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError(f"输入矩阵必须是二维的，当前 shape={array.shape}")
    return sparse.csr_matrix(array)


def _filter_features(
    counts_cell_by_gene: sparse.csr_matrix,
    var: pd.DataFrame,
    data_cfg: Dict[str, Any],
) -> Tuple[sparse.csr_matrix, pd.DataFrame]:
    feature_names = pd.Index(var.index.astype(str))
    keep_mask = np.ones(len(feature_names), dtype=bool)

    include_features = _normalize_feature_spec(data_cfg.get("feature_include"))
    include_path = data_cfg.get("feature_include_path")
    if include_path:
        include_features.extend(_read_feature_names_from_path(Path(include_path)))
    if include_features:
        include_set = set(include_features)
        include_mask = feature_names.isin(include_set)
        missing = sorted(include_set.difference(set(feature_names)))
        if missing:
            preview = ", ".join(missing[:20])
            print(
                f"[WARN] feature_include 中有 {len(missing)} 个基因不在输入矩阵里，前20个：{preview}",
                file=sys.stderr,
            )
        if not include_mask.any():
            raise ValueError("feature_include 过滤后没有剩余基因，请检查基因名是否与 var_names 一致。")
        keep_mask &= include_mask

    exclude_features = _normalize_feature_spec(data_cfg.get("feature_exclude"))
    exclude_path = data_cfg.get("feature_exclude_path")
    if exclude_path:
        exclude_features.extend(_read_feature_names_from_path(Path(exclude_path)))
    if exclude_features:
        exclude_set = set(exclude_features)
        keep_mask &= ~feature_names.isin(exclude_set)

    if bool(data_cfg.get("drop_zero_variance_features", False)):
        mean = np.asarray(counts_cell_by_gene.mean(axis=0)).ravel()
        mean_sq = np.asarray(counts_cell_by_gene.power(2).mean(axis=0)).ravel()
        variance = mean_sq - np.square(mean)
        keep_mask &= variance > 0

    if not keep_mask.any():
        raise ValueError("feature 过滤后没有剩余基因。")

    if int(keep_mask.sum()) != len(keep_mask):
        kept_n = int(keep_mask.sum())
        removed_n = int((~keep_mask).sum())
        print(f"[INFO] feature 过滤：保留 {kept_n} 个基因，移除 {removed_n} 个基因。")
        counts_cell_by_gene = counts_cell_by_gene[:, keep_mask]
        var = var.iloc[keep_mask].copy()

    return counts_cell_by_gene, var


def _apply_obs_filters(
    counts_cell_by_gene: sparse.csr_matrix,
    obs: pd.DataFrame,
    data_cfg: Dict[str, Any],
) -> Tuple[sparse.csr_matrix, pd.DataFrame]:
    filters_cfg = data_cfg.get("obs_value_filters")
    if not filters_cfg:
        return counts_cell_by_gene, obs

    keep_mask = np.ones(obs.shape[0], dtype=bool)
    for column, allowed_values in filters_cfg.items():
        if column not in obs.columns:
            raise KeyError(f"obs_value_filters 指定的列不存在: {column}")
        allowed = _normalize_feature_spec(allowed_values)
        if not allowed:
            continue
        current_mask = obs[column].astype(str).isin({str(x) for x in allowed}).to_numpy()
        keep_mask &= current_mask

    if not keep_mask.any():
        raise ValueError("obs_value_filters 过滤后没有剩余细胞")

    filtered_counts = counts_cell_by_gene[keep_mask, :]
    filtered_obs = obs.iloc[keep_mask].copy()
    print(f"[INFO] obs 过滤：从 {obs.shape[0]} 个细胞中保留 {filtered_obs.shape[0]} 个细胞。")
    return filtered_counts, filtered_obs


def _apply_cell_sampling(
    counts_cell_by_gene: sparse.csr_matrix,
    obs: pd.DataFrame,
    data_cfg: Dict[str, Any],
) -> Tuple[sparse.csr_matrix, pd.DataFrame]:
    sampling_cfg = data_cfg.get("cell_sampling")
    if not sampling_cfg or not bool(sampling_cfg.get("enabled", False)):
        return counts_cell_by_gene, obs

    groupby = sampling_cfg.get("groupby")
    if not groupby:
        raise ValueError("cell_sampling.enabled=True 时必须提供 groupby")
    if groupby not in obs.columns:
        raise KeyError(f"cell_sampling.groupby 指定的列不存在: {groupby}")

    include_groups = sampling_cfg.get("include_groups")
    if include_groups:
        include_groups = {str(x) for x in include_groups}
        group_values = obs[groupby].astype(str)
        keep_group_mask = group_values.isin(include_groups).to_numpy()
        if not keep_group_mask.any():
            raise ValueError("cell_sampling.include_groups 过滤后没有任何细胞")
        counts_cell_by_gene = counts_cell_by_gene[keep_group_mask, :]
        obs = obs.iloc[keep_group_mask].copy()

    seed = int(sampling_cfg.get("seed", 1))
    rng = np.random.default_rng(seed)
    group_sizes = obs[groupby].astype(str).value_counts().sort_index()
    n_groups = int(group_sizes.shape[0])
    if n_groups == 0:
        raise ValueError("cell_sampling 未找到任何分组")

    min_per_group = sampling_cfg.get("min_per_group")
    max_per_group = sampling_cfg.get("max_per_group")
    target_total = sampling_cfg.get("target_total")
    exact_per_group = sampling_cfg.get("exact_per_group")

    if exact_per_group is not None:
        min_per_group = int(exact_per_group)
        max_per_group = int(exact_per_group)
    if min_per_group is None or max_per_group is None:
        raise ValueError("cell_sampling 需要提供 min_per_group/max_per_group 或 exact_per_group")

    min_per_group = int(min_per_group)
    max_per_group = int(max_per_group)
    if min_per_group < 0 or max_per_group < min_per_group:
        raise ValueError("cell_sampling 的 min/max 设置非法")

    mins = np.minimum(group_sizes.to_numpy(), min_per_group).astype(int)
    caps = np.minimum(group_sizes.to_numpy(), max_per_group).astype(int)
    capacities = caps - mins

    base_total = int(mins.sum())
    if target_total is None:
        target_total = int(caps.sum())
    else:
        target_total = int(target_total)
    if target_total < base_total:
        raise ValueError(
            f"cell_sampling.target_total={target_total} 小于各组最小采样总数 {base_total}"
        )

    extra_total = min(int(target_total - base_total), int(capacities.sum()))
    extras = np.zeros_like(mins)
    if extra_total > 0:
        expanded_group_ids = np.repeat(np.arange(n_groups), capacities)
        sampled_group_ids = rng.choice(expanded_group_ids, size=extra_total, replace=False)
        extras = np.bincount(sampled_group_ids, minlength=n_groups)

    target_per_group = mins + extras

    sampled_indices: list[np.ndarray] = []
    obs_groups = obs[groupby].astype(str)
    for idx, (group_name, target_n) in enumerate(zip(group_sizes.index.tolist(), target_per_group.tolist())):
        group_idx = np.flatnonzero(obs_groups.to_numpy() == group_name)
        if target_n <= 0:
            continue
        chosen = rng.choice(group_idx, size=int(target_n), replace=False)
        sampled_indices.append(np.sort(chosen))

    if not sampled_indices:
        raise ValueError("cell_sampling 没有抽到任何细胞")

    sampled_idx = np.concatenate(sampled_indices)
    sampled_idx.sort()
    sampled_counts = counts_cell_by_gene[sampled_idx, :]
    sampled_obs = obs.iloc[sampled_idx].copy()

    realized = sampled_obs[groupby].astype(str).value_counts().sort_index()
    print(
        f"[INFO] cell_sampling：按 {groupby} 分层抽样，"
        f"从 {obs.shape[0]} 个细胞中保留 {sampled_obs.shape[0]} 个细胞；"
        f"每组范围 {int(realized.min())}-{int(realized.max())}。"
    )
    return sampled_counts, sampled_obs


def _load_h5ad(
    input_path: Path,
    matrix_source: str = "auto",
    make_index_unique: bool = True,
    obsm_to_obs: Optional[Dict[str, list[str]]] = None,
) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    try:
        import anndata as ad  # type: ignore
    except ImportError as exc:
        raise ImportError("读取 h5ad 需要安装 Python 包 anndata。") from exc

    adata = ad.read_h5ad(input_path)

    selected_obs = adata.obs.copy()
    if obsm_to_obs:
        for obsm_key, column_names in obsm_to_obs.items():
            if obsm_key not in adata.obsm:
                raise KeyError(f"h5ad 中不存在 obsm['{obsm_key}']")
            arr = np.asarray(adata.obsm[obsm_key])
            if arr.ndim != 2:
                raise ValueError(f"obsm['{obsm_key}'] 必须是二维矩阵，当前 shape={arr.shape}")
            if arr.shape[1] != len(column_names):
                raise ValueError(
                    f"obsm['{obsm_key}'] 共有 {arr.shape[1]} 列，但配置了 {len(column_names)} 个目标列名"
                )
            for idx, col_name in enumerate(column_names):
                selected_obs[col_name] = arr[:, idx]
    if matrix_source == "auto":
        if "counts" in adata.layers:
            matrix_source = "layer:counts"
        elif adata.raw is not None:
            matrix_source = "raw"
        else:
            matrix_source = "X"

    if matrix_source.startswith("layer:"):
        layer_name = matrix_source.split(":", 1)[1]
        if layer_name not in adata.layers:
            raise KeyError(f"h5ad 中不存在 layer '{layer_name}'")
        matrix = _to_csr(adata.layers[layer_name])
        selected_var = adata.var.copy()
    elif matrix_source == "raw":
        if adata.raw is None:
            raise ValueError("matrix_source='raw'，但 h5ad 没有 raw 槽。")
        matrix = _to_csr(adata.raw.X)
        selected_var = adata.raw.var.copy()
    elif matrix_source == "X":
        matrix = _to_csr(adata.X)
        selected_var = adata.var.copy()
    else:
        raise ValueError(
            "matrix_source 只支持 auto / X / raw / layer:<name>，"
            f"当前收到: {matrix_source}"
        )

    obs_names = pd.Index(adata.obs_names.astype(str))
    var_names = pd.Index(selected_var.index.astype(str))
    if make_index_unique:
        obs_names = _ensure_unique(obs_names, "cell ids")
        var_names = _ensure_unique(var_names, "feature ids")
    elif (not obs_names.is_unique) or (not var_names.is_unique):
        raise ValueError("obs_names 或 var_names 不唯一，请先处理后再运行。")

    selected_obs.index = obs_names
    selected_var.index = var_names
    return matrix.tocsr(), selected_obs, selected_var


def _load_csv_matrix(
    counts_path: Path,
    obs_path: Path,
    var_path: Path,
    count_orientation: str,
) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    counts_df = pd.read_csv(counts_path, index_col=0)
    obs = pd.read_csv(obs_path, index_col=0)
    var = pd.read_csv(var_path, index_col=0)

    counts = sparse.csr_matrix(counts_df.to_numpy())
    if count_orientation == "gene_by_cell":
        counts = counts.transpose().tocsr()

    obs.index = _ensure_unique(pd.Index(obs.index.astype(str)), "cell ids")
    var.index = _ensure_unique(pd.Index(var.index.astype(str)), "feature ids")
    return counts, obs, var


def _copy_bundle(
    counts_path: Path,
    obs_path: Path,
    var_path: Path,
    prepared_dir: Path,
) -> Dict[str, str]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    target_counts = prepared_dir / "counts.mtx"
    target_obs = prepared_dir / "obs.csv"
    target_var = prepared_dir / "var.csv"
    shutil.copy2(counts_path, target_counts)
    shutil.copy2(obs_path, target_obs)
    shutil.copy2(var_path, target_var)
    return {
        "counts_path": str(target_counts),
        "obs_path": str(target_obs),
        "var_path": str(target_var),
        "count_orientation": "cell_by_gene",
        "feature_id_col": "feature_id",
        "cell_id_col": "cell_id",
    }


def _export_standard_bundle(
    counts_cell_by_gene: sparse.csr_matrix,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    prepared_dir: Path,
) -> Dict[str, str]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    counts_path = prepared_dir / "counts.mtx"
    obs_path = prepared_dir / "obs.csv"
    var_path = prepared_dir / "var.csv"

    obs_export = obs.copy()
    var_export = var.copy()
    obs_export.insert(0, "cell_id", obs_export.index.astype(str))
    var_export.insert(0, "feature_id", var_export.index.astype(str))

    spio.mmwrite(str(counts_path), counts_cell_by_gene.tocoo())
    obs_export.to_csv(obs_path, index=False)
    var_export.to_csv(var_path, index=False)

    return {
        "counts_path": str(counts_path),
        "obs_path": str(obs_path),
        "var_path": str(var_path),
        "count_orientation": "cell_by_gene",
        "feature_id_col": "feature_id",
        "cell_id_col": "cell_id",
    }


def _prepare_input_bundle(data_cfg: Dict[str, Any], prepared_dir: Path) -> Dict[str, Any]:
    input_format = data_cfg.get("input_format", "auto")
    input_path = data_cfg.get("input_path")

    if input_format == "auto":
        if input_path is None:
            raise ValueError("input_format='auto' 时必须提供 data.input_path。")
        p = Path(input_path)
        if p.is_file() and p.suffix.lower() == ".h5ad":
            input_format = "h5ad"
        elif p.is_file() and p.suffix.lower() == ".rds":
            input_format = "sce_rds"
        elif p.is_dir():
            if (p / "counts.mtx").exists() and (p / "obs.csv").exists() and (p / "var.csv").exists():
                input_format = "standardized_bundle"
            else:
                raise ValueError("目录输入 auto 检测失败：需要至少包含 counts.mtx、obs.csv、var.csv。")
        else:
            raise ValueError(f"无法自动识别输入格式: {p}")

    if input_format == "h5ad":
        if input_path is None:
            raise ValueError("h5ad 输入必须提供 data.input_path")
        counts, obs, var = _load_h5ad(
            Path(input_path),
            matrix_source=data_cfg.get("matrix_source", "auto"),
            make_index_unique=bool(data_cfg.get("make_index_unique", True)),
            obsm_to_obs=data_cfg.get("obsm_to_obs"),
        )
        counts, var = _filter_features(counts, var, data_cfg)
        counts, obs = _apply_obs_filters(counts, obs, data_cfg)
        counts, obs = _apply_cell_sampling(counts, obs, data_cfg)
        bundle = _export_standard_bundle(counts, obs, var, prepared_dir)
        return {
            "type": "standard_bundle",
            "bundle": bundle,
            "original_input": str(Path(input_path).resolve()),
        }

    if input_format == "standardized_bundle":
        root = Path(input_path)
        bundle = _copy_bundle(root / "counts.mtx", root / "obs.csv", root / "var.csv", prepared_dir)
        return {
            "type": "standard_bundle",
            "bundle": bundle,
            "original_input": str(root.resolve()),
        }

    if input_format == "matrix_market_triplet":
        counts_path = Path(data_cfg["counts_path"])
        obs_path = Path(data_cfg["obs_path"])
        var_path = Path(data_cfg["var_path"])
        counts = spio.mmread(str(counts_path))
        counts = _to_csr(counts)
        obs = pd.read_csv(obs_path, index_col=0)
        var = pd.read_csv(var_path, index_col=0)
        orientation = data_cfg.get("count_orientation", "cell_by_gene")
        if orientation == "gene_by_cell":
            counts = counts.transpose().tocsr()
        elif orientation != "cell_by_gene":
            raise ValueError("count_orientation 只支持 cell_by_gene 或 gene_by_cell")
        obs.index = _ensure_unique(pd.Index(obs.index.astype(str)), "cell ids")
        var.index = _ensure_unique(pd.Index(var.index.astype(str)), "feature ids")
        counts, var = _filter_features(counts, var, data_cfg)
        counts, obs = _apply_obs_filters(counts, obs, data_cfg)
        counts, obs = _apply_cell_sampling(counts, obs, data_cfg)
        bundle = _export_standard_bundle(counts, obs, var, prepared_dir)
        return {"type": "standard_bundle", "bundle": bundle}

    if input_format == "npz_triplet":
        counts_path = Path(data_cfg["counts_path"])
        obs_path = Path(data_cfg["obs_path"])
        var_path = Path(data_cfg["var_path"])
        counts = sparse.load_npz(counts_path).tocsr()
        orientation = data_cfg.get("count_orientation", "cell_by_gene")
        if orientation == "gene_by_cell":
            counts = counts.transpose().tocsr()
        elif orientation != "cell_by_gene":
            raise ValueError("count_orientation 只支持 cell_by_gene 或 gene_by_cell")
        obs = pd.read_csv(obs_path, index_col=0)
        var = pd.read_csv(var_path, index_col=0)
        obs.index = _ensure_unique(pd.Index(obs.index.astype(str)), "cell ids")
        var.index = _ensure_unique(pd.Index(var.index.astype(str)), "feature ids")
        counts, var = _filter_features(counts, var, data_cfg)
        counts, obs = _apply_obs_filters(counts, obs, data_cfg)
        counts, obs = _apply_cell_sampling(counts, obs, data_cfg)
        bundle = _export_standard_bundle(counts, obs, var, prepared_dir)
        return {"type": "standard_bundle", "bundle": bundle}

    if input_format == "csv_triplet":
        counts, obs, var = _load_csv_matrix(
            Path(data_cfg["counts_path"]),
            Path(data_cfg["obs_path"]),
            Path(data_cfg["var_path"]),
            data_cfg.get("count_orientation", "cell_by_gene"),
        )
        counts, var = _filter_features(counts, var, data_cfg)
        counts, obs = _apply_obs_filters(counts, obs, data_cfg)
        counts, obs = _apply_cell_sampling(counts, obs, data_cfg)
        bundle = _export_standard_bundle(counts, obs, var, prepared_dir)
        return {"type": "standard_bundle", "bundle": bundle}

    if input_format == "sce_rds":
        if input_path is None:
            raise ValueError("sce_rds 输入必须提供 data.input_path")
        return {"type": "sce_rds", "sce_rds_path": str(Path(input_path).resolve())}

    raise ValueError(f"不支持的 input_format: {input_format}")


def _package_output_to_h5ad(output_dir: Path, prepared_input: Dict[str, Any], n_rep: int) -> None:
    try:
        import anndata as ad  # type: ignore
    except ImportError as exc:
        raise ImportError("export_h5ad=True 时需要安装 anndata。") from exc

    if prepared_input.get("type") != "standard_bundle":
        print("[WARN] 当前仅对 standard_bundle 输入自动导出 h5ad，已跳过。", file=sys.stderr)
        return

    bundle = prepared_input["bundle"]
    var = pd.read_csv(bundle["var_path"])
    feature_id_col = bundle.get("feature_id_col", "feature_id")
    if feature_id_col in var.columns:
        var = var.set_index(feature_id_col)
    else:
        var = var.set_index(var.columns[0])

    metrics_path = output_dir / "model_metrics.json"
    if metrics_path.exists():
        metrics = _read_json(metrics_path)
    else:
        metrics = {}

    replicate_indices = range(1, max(1, n_rep) + 1)
    for rep_idx in replicate_indices:
        counts_file = output_dir / (f"sim_counts_rep{rep_idx}.mtx" if n_rep > 1 else "sim_counts.mtx")
        obs_file = output_dir / (f"sim_obs_rep{rep_idx}.csv" if n_rep > 1 else "sim_obs.csv")
        if not counts_file.exists() or not obs_file.exists():
            print(f"[WARN] 跳过 h5ad 打包：缺少 {counts_file.name} 或 {obs_file.name}", file=sys.stderr)
            continue

        counts = spio.mmread(str(counts_file)).tocsr().transpose().tocsr()
        obs = pd.read_csv(obs_file)
        cell_id_col = "cell_id" if "cell_id" in obs.columns else obs.columns[0]
        obs = obs.set_index(cell_id_col)

        adata = ad.AnnData(X=counts, obs=obs, var=var.copy())
        adata.uns["scdesign3_metrics"] = metrics
        out_path = output_dir / (f"sim_rep{rep_idx}.h5ad" if n_rep > 1 else "sim.h5ad")
        adata.write_h5ad(out_path)


def _run_r_worker(config_path: Path, rscript_path: str) -> None:
    cmd = [rscript_path, str(R_WORKER), "--config", str(config_path)]
    print("[INFO] 调用 R 脚本：", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python 调度 scDesign3 的受控生成流程。")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--rscript-path", default="Rscript", help="Rscript 可执行文件路径")
    parser.add_argument("--skip-r", action="store_true", help="只做输入标准化，不真正运行 R")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _read_json(config_path)

    output_cfg = config.setdefault("output", {})
    output_dir = Path(output_cfg.get("output_dir", PACKAGE_ROOT / "pipeline_runs" / config_path.stem)).resolve()
    prepared_dir = output_dir / "_prepared_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_input = _prepare_input_bundle(config.get("data", {}), prepared_dir)
    n_rep = int(config.get("simulation", {}).get("n_rep", 1))

    resolved_config = {
        "package_root": str(PACKAGE_ROOT),
        "prepared_input": prepared_input,
        "simulation": config.get("simulation", {}),
        "output": {
            **output_cfg,
            "output_dir": str(output_dir),
        },
    }

    resolved_config_path = output_dir / "resolved_config.json"
    _write_json(resolved_config, resolved_config_path)

    if not args.skip_r:
        _run_r_worker(resolved_config_path, args.rscript_path)

        if bool(output_cfg.get("export_h5ad", False)):
            _package_output_to_h5ad(output_dir, prepared_input, n_rep=n_rep)

    print(f"[INFO] 流程完成，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
