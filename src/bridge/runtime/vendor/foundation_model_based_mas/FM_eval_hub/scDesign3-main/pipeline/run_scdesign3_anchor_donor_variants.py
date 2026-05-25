#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import run_scdesign3_pipeline as base


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
R_WORKER = SCRIPT_DIR / "run_scdesign3_anchor_donor_variants.R"
VALID_VARIANTS = ["baseline", "variant1", "variant2", "variant3", "variant4"]


def _parse_variants_arg(raw: str | None) -> List[str]:
    if raw is None:
        return []
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts or parts == ["all"]:
        return []
    invalid = [item for item in parts if item not in VALID_VARIANTS]
    if invalid:
        raise ValueError(f"不支持的变体名: {', '.join(invalid)}；可选值: {', '.join(VALID_VARIANTS)}, all")
    return parts


def _run_r_worker(config_path: Path, rscript_path: str) -> None:
    cmd = [rscript_path, str(R_WORKER), "--config", str(config_path)]
    print("[INFO] 调用 R 脚本：", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _package_variant_h5ad(output_dir: Path, prepared_input: Dict[str, Any]) -> None:
    manifest_path = output_dir / "variant_manifest.json"
    if not manifest_path.exists():
        print(f"[WARN] 未找到 {manifest_path}，跳过 h5ad 打包。")
        return

    manifest = base._read_json(manifest_path)
    for variant in manifest.get("variants", []):
        variant_dir = output_dir / variant["dir"]
        n_rep = int(variant.get("n_rep", 1))
        if not variant_dir.exists():
            print(f"[WARN] 变体目录不存在，跳过：{variant_dir}")
            continue
        base._package_output_to_h5ad(variant_dir, prepared_input, n_rep=n_rep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor Donor 方案：单 donor 拟合 + 四变体生成")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--rscript-path", default="Rscript", help="Rscript 可执行文件路径")
    parser.add_argument(
        "--variants",
        default=None,
        help="只跑指定变体，逗号分隔；可选 baseline,variant1,variant2,variant3,variant4,all",
    )
    parser.add_argument("--force-refit", action="store_true", help="忽略已有 master cache，强制重新拟合")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = base._read_json(config_path)

    output_cfg = config.setdefault("output", {})
    output_dir = Path(
        output_cfg.get("output_dir", PACKAGE_ROOT / "pipeline_runs" / config_path.stem)
    ).resolve()
    prepared_dir = output_dir / "_prepared_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = _parse_variants_arg(args.variants)
    if selected_variants:
        config["selected_variants"] = selected_variants

    prepared_input = base._prepare_input_bundle(config.get("data", {}), prepared_dir)
    resolved_config: Dict[str, Any] = {
        "package_root": str(PACKAGE_ROOT),
        "prepared_input": prepared_input,
        "anchor_donor": config.get("anchor_donor", {}),
        "simulation": config.get("simulation", {}),
        "generation_template": config.get("generation_template", {}),
        "variants": config.get("variants", {}),
        "selected_variants": config.get("selected_variants", []),
        "output": {
            **output_cfg,
            "output_dir": str(output_dir),
        },
    }

    if args.force_refit:
        resolved_config.setdefault("output", {})["force_refit"] = True

    resolved_config_path = output_dir / "resolved_config.json"
    base._write_json(resolved_config, resolved_config_path)

    cache_path = resolved_config["output"].get("master_cache_path")
    if args.force_refit and cache_path:
        cache_file = Path(cache_path)
        if not cache_file.is_absolute():
            cache_file = (PACKAGE_ROOT / cache_file).resolve()
        if cache_file.exists():
            cache_file.unlink()
            print(f"[INFO] 已删除旧的 master cache：{cache_file}")

    _run_r_worker(resolved_config_path, args.rscript_path)

    if bool(output_cfg.get("export_h5ad", False)):
        _package_variant_h5ad(output_dir, prepared_input)

    print(f"[INFO] Anchor Donor 变体生成完成，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
