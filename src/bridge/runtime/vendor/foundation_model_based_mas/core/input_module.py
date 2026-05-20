from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

from .langsmith_compat import traceable, tracing_context
from .tracing import bootstrap_langsmith_from_env


class TracingConfig(BaseModel):
    enabled: bool = True
    project: str = "foundation_model_based_mas"
    tags: list[str] = Field(default_factory=lambda: ["mas", "input-module"])


class RuntimeConfig(BaseModel):
    env_path: str = ".env"
    tracing: TracingConfig = Field(default_factory=TracingConfig)


class TaskConfig(BaseModel):
    task_type: str
    task_request: str


class DatasetSplitConfig(BaseModel):
    path: str | None = None
    glob: str = "*.h5ad"


class DatasetSplitPathsConfig(BaseModel):
    train: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
    validation: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
    test: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)


class DatasetConfig(BaseModel):
    dataset_id: str
    description: str
    base_dir: str = "."
    paths: list[str] = Field(default_factory=list)
    globs: list[str] = Field(default_factory=list)
    npz_path: str | None = None
    h5ad_path: str | None = None
    active_split: Literal["train", "validation", "test"] = "test"
    split_paths: DatasetSplitPathsConfig | None = None
    structure_hint: str = "h5ad"
    merge_strategy: Literal["auto", "single_file", "concat_same_schema"] = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sources(self) -> "DatasetConfig":
        has_direct_pair = bool(self.npz_path or self.h5ad_path)
        has_legacy_sources = bool(self.paths or self.globs)
        has_split_sources = self.split_paths is not None
        if has_direct_pair:
            if not self.npz_path or not self.h5ad_path:
                raise ValueError("dataset.npz_path and dataset.h5ad_path must be provided together.")
            return self
        if not has_legacy_sources and not has_split_sources:
            raise ValueError(
                "Provide either dataset.npz_path + dataset.h5ad_path, "
                "dataset.split_paths, or legacy dataset.paths/dataset.globs."
            )
        if has_split_sources:
            if not self.split_paths.train.path:
                raise ValueError("dataset.split_paths.train.path must be provided.")
            if not self.split_paths.test.path:
                raise ValueError("dataset.split_paths.test.path must be provided.")
        return self


class MASInputConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    task: TaskConfig
    dataset: DatasetConfig


class InputModuleState(TypedDict):
    config_path: str
    raw_config: dict[str, Any]
    config: dict[str, Any]
    task_type: str
    task_request: str
    dataset_description: str
    named_input_paths: dict[str, str]
    split_resolved_paths: dict[str, list[str]]
    resolved_paths: list[str]
    bundled_path: str
    prepared_h5ad_path: str
    data_profile: dict[str, Any]
    input_manifest: dict[str, Any]


class InputModule:
    def __init__(self, *, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.logs_dir = self.project_root / "logs"
        self.outputs_dir = self.project_root / "outputs"
        self.input_outputs_dir = self.outputs_dir / "input_module"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.input_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.tracing_config = bootstrap_langsmith_from_env(self.project_root, default_project="foundation_model_based_mas_input")
        self.langsmith_enabled = bool(self.tracing_config.get("enabled", False))
        self.logger = logging.getLogger("foundation_model_based_mas.input_module")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh = logging.FileHandler(self.logs_dir / "mas_input_module.log", encoding="utf-8")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(InputModuleState)
        for node_name in [
            "load_config_node",
            "resolve_inputs_node",
            "bundle_inputs_node",
            "profile_inputs_node",
        ]:
            graph.add_node(node_name, getattr(self, node_name))
        graph.add_edge(START, "load_config_node")
        graph.add_edge("load_config_node", "resolve_inputs_node")
        graph.add_edge("resolve_inputs_node", "bundle_inputs_node")
        graph.add_edge("bundle_inputs_node", "profile_inputs_node")
        graph.add_edge("profile_inputs_node", END)
        return graph.compile()

    def _log_preview(self, node: str, payload: Any) -> None:
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = str(payload)
        if len(text) > 1200:
            text = text[:1200] + "...[truncated]"
        self.logger.info("[%s] %s", node, text)

    def _workspace(self, prefix: str) -> Path:
        workspace = self.input_outputs_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _read_config(self, config_path: str | Path) -> MASInputConfig:
        config_path = Path(config_path).resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return MASInputConfig.model_validate(payload)

    def _resolve_file_or_directory(
        self,
        *,
        config_path: str | Path,
        base_dir: str | Path,
        target_path: str | None,
        glob_pattern: str = "*.h5ad",
        required: bool,
    ) -> list[str]:
        config_path = Path(config_path).resolve()
        if not target_path:
            if required:
                raise FileNotFoundError("Required dataset split path is missing.")
            return []

        base_dir = Path(base_dir)
        if not base_dir.is_absolute():
            base_dir = (config_path.parent / base_dir).resolve()

        target = Path(target_path)
        if not target.is_absolute():
            target = (config_path.parent / target).resolve()
            if not target.exists():
                target = (base_dir / target_path).resolve()

        if not target.exists():
            raise FileNotFoundError(f"Input path does not exist: {target}")

        if target.is_file():
            return [str(target)]

        if not target.is_dir():
            raise ValueError(f"Input path must be a file or directory: {target}")

        matches = sorted(item.resolve() for item in target.glob(glob_pattern) if item.is_file())
        if required and not matches:
            raise FileNotFoundError(f"No files matched {glob_pattern} under required input directory: {target}")
        return [str(item) for item in matches]

    def _resolve_named_input_file(
        self,
        *,
        config_path: str | Path,
        base_dir: str | Path,
        target_path: str,
        input_name: str,
    ) -> str:
        config_path = Path(config_path).resolve()
        base_dir = Path(base_dir)
        if not base_dir.is_absolute():
            base_dir = (config_path.parent / base_dir).resolve()

        target = Path(target_path)
        if not target.is_absolute():
            target = (config_path.parent / target).resolve()
            if not target.exists():
                target = (base_dir / target_path).resolve()

        if not target.exists():
            raise FileNotFoundError(f"Configured {input_name} does not exist: {target}")
        if not target.is_file():
            raise ValueError(f"Configured {input_name} must be a file: {target}")
        return str(target.resolve())

    def _resolve_source_paths(self, config_path: str | Path, dataset: DatasetConfig) -> list[str]:
        config_path = Path(config_path).resolve()
        base_dir = Path(dataset.base_dir)
        if not base_dir.is_absolute():
            base_dir = (config_path.parent / base_dir).resolve()

        resolved: list[Path] = []
        for item in dataset.paths:
            path = Path(item)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Input file does not exist: {path}")
            resolved.append(path)

        for pattern in dataset.globs:
            matches = sorted(base_dir.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"Glob pattern did not match any files: {base_dir / pattern}")
            resolved.extend(item.resolve() for item in matches if item.is_file())

        deduped = list(dict.fromkeys(str(path) for path in resolved))
        if not deduped:
            raise FileNotFoundError("No input files were resolved from the dataset configuration.")
        return deduped

    def _resolve_named_input_paths(self, config_path: str | Path, dataset: DatasetConfig) -> dict[str, str]:
        if not dataset.npz_path or not dataset.h5ad_path:
            return {}
        resolved_npz_path = self._resolve_named_input_file(
            config_path=config_path,
            base_dir=dataset.base_dir,
            target_path=dataset.npz_path,
            input_name="dataset.npz_path",
        )
        resolved_h5ad_path = self._resolve_named_input_file(
            config_path=config_path,
            base_dir=dataset.base_dir,
            target_path=dataset.h5ad_path,
            input_name="dataset.h5ad_path",
        )
        return {
            "npz_path": resolved_npz_path,
            "h5ad_path": resolved_h5ad_path,
        }

    def _resolve_split_source_paths(self, config_path: str | Path, dataset: DatasetConfig) -> dict[str, list[str]]:
        if dataset.split_paths is None:
            return {}
        return {
            "train": self._resolve_file_or_directory(
                config_path=config_path,
                base_dir=dataset.base_dir,
                target_path=dataset.split_paths.train.path,
                glob_pattern=dataset.split_paths.train.glob,
                required=True,
            ),
            "validation": self._resolve_file_or_directory(
                config_path=config_path,
                base_dir=dataset.base_dir,
                target_path=dataset.split_paths.validation.path,
                glob_pattern=dataset.split_paths.validation.glob,
                required=False,
            ),
            "test": self._resolve_file_or_directory(
                config_path=config_path,
                base_dir=dataset.base_dir,
                target_path=dataset.split_paths.test.path,
                glob_pattern=dataset.split_paths.test.glob,
                required=True,
            ),
        }

    def _prepare_h5ad_with_source_ids(self, h5ad_path: str | Path, workspace: Path) -> str:
        import anndata as ad

        h5ad_path = Path(h5ad_path).resolve()
        backed = ad.read_h5ad(h5ad_path, backed="r")
        obs_columns = list(backed.obs.columns)
        if "source_cell_id" in obs_columns:
            self._log_preview("prepare_h5ad.reuse", {"prepared_h5ad_path": str(h5ad_path), "source_cell_id_present": True})
            return str(h5ad_path)

        adata = ad.read_h5ad(h5ad_path)
        adata.obs["source_cell_id"] = adata.obs_names.astype(str)
        out_path = workspace / f"prepared_{h5ad_path.stem}.h5ad"
        adata.write_h5ad(out_path, compression="gzip")
        self._log_preview("prepare_h5ad.write", {"input_h5ad_path": str(h5ad_path), "prepared_h5ad_path": str(out_path)})
        return str(out_path)

    def _concat_h5ad_files(self, resolved_paths: list[str], workspace: Path) -> str:
        import anndata as ad

        schemas: list[dict[str, Any]] = []
        var_name_reference: list[str] | None = None
        obs_columns_reference: list[str] | None = None
        for raw_path in resolved_paths:
            path = Path(raw_path).resolve()
            adata = ad.read_h5ad(path, backed="r")
            current_var_names = [str(x) for x in adata.var_names]
            current_obs_columns = [str(x) for x in adata.obs.columns]
            schemas.append({
                "path": str(path),
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "obs_columns": current_obs_columns,
            })
            if var_name_reference is None:
                var_name_reference = current_var_names
                obs_columns_reference = current_obs_columns
                continue
            if current_var_names != var_name_reference:
                raise ValueError(f"Input files do not share the same var_names schema: {path}")
            if current_obs_columns != (obs_columns_reference or []):
                raise ValueError(f"Input files do not share the same obs-column schema: {path}")

        merged_inputs: dict[str, Any] = {}
        ordered_keys: list[str] = []
        for idx, raw_path in enumerate(resolved_paths, start=1):
            path = Path(raw_path).resolve()
            key = f"part_{idx:03d}_{path.stem}"
            adata = ad.read_h5ad(path)
            adata.obs["mas_input_file"] = path.name
            merged_inputs[key] = adata
            ordered_keys.append(key)

        merged = ad.concat(
            merged_inputs,
            label="mas_input_source",
            keys=ordered_keys,
            index_unique="::",
            join="inner",
            merge="same",
        )
        out_path = workspace / "merged_input.h5ad"
        merged.write_h5ad(out_path, compression="gzip")
        self._log_preview("concat_h5ad_files", {"resolved_paths": resolved_paths, "merged_h5ad_path": str(out_path), "schemas": schemas})
        return str(out_path)

    def _bundle_inputs(
        self,
        config: MASInputConfig,
        resolved_paths: list[str],
        named_input_paths: dict[str, str],
        workspace: Path,
    ) -> tuple[str, str, dict[str, Any]]:
        if named_input_paths.get("npz_path") and named_input_paths.get("h5ad_path"):
            bundled_path = named_input_paths["h5ad_path"]
            prepared_h5ad_path = self._prepare_h5ad_with_source_ids(bundled_path, workspace)
            structure_hint = str(config.dataset.structure_hint).lower()
            manifest_structure_hint = "paired_npz_h5ad" if structure_hint == "auto" else structure_hint
            metadata = {
                **config.dataset.metadata,
                "npz_path": named_input_paths["npz_path"],
                "annotation_npz_path": named_input_paths["npz_path"],
                "h5ad_path": named_input_paths["h5ad_path"],
            }
            manifest = {
                "dataset_id": config.dataset.dataset_id,
                "input_mode": "direct_pair",
                "structure_hint": manifest_structure_hint,
                "resolved_paths": resolved_paths,
                "num_files": len(resolved_paths),
                "merge_strategy": config.dataset.merge_strategy,
                "merge_applied": False,
                "bundled_path": bundled_path,
                "prepared_h5ad_path": prepared_h5ad_path,
                "npz_path": named_input_paths["npz_path"],
                "h5ad_path": named_input_paths["h5ad_path"],
                "named_input_paths": dict(named_input_paths),
                "metadata": metadata,
            }
            return bundled_path, prepared_h5ad_path, manifest

        suffixes = sorted({Path(path).suffix.lower() for path in resolved_paths})
        structure_hint = str(config.dataset.structure_hint).lower()
        if len(suffixes) != 1:
            raise ValueError(f"All input files must share the same suffix. Got: {suffixes}")

        effective_suffix = suffixes[0]
        if structure_hint not in {"auto", "h5ad"} and structure_hint != effective_suffix.lstrip("."):
            raise ValueError(f"structure_hint={structure_hint} is inconsistent with file suffix {effective_suffix}.")
        if effective_suffix != ".h5ad":
            raise NotImplementedError(f"Only h5ad input is implemented for now. Got: {effective_suffix}")

        if len(resolved_paths) == 1:
            bundled_path = resolved_paths[0]
            merge_applied = False
        else:
            if config.dataset.merge_strategy == "single_file":
                raise ValueError("merge_strategy=single_file is inconsistent with multiple input files.")
            bundled_path = self._concat_h5ad_files(resolved_paths, workspace)
            merge_applied = True

        prepared_h5ad_path = self._prepare_h5ad_with_source_ids(bundled_path, workspace)
        manifest = {
            "dataset_id": config.dataset.dataset_id,
            "input_mode": "legacy_files",
            "structure_hint": effective_suffix.lstrip("."),
            "resolved_paths": resolved_paths,
            "num_files": len(resolved_paths),
            "merge_strategy": config.dataset.merge_strategy,
            "merge_applied": merge_applied,
            "bundled_path": bundled_path,
            "prepared_h5ad_path": prepared_h5ad_path,
            "metadata": config.dataset.metadata,
        }
        return bundled_path, prepared_h5ad_path, manifest

    def _sample_obs_preview(self, adata, preview_n: int = 5) -> dict[str, Any]:
        preview: dict[str, Any] = {}
        for column in list(adata.obs.columns)[: min(12, len(adata.obs.columns))]:
            series = adata.obs[column]
            payload: dict[str, Any] = {"dtype": str(series.dtype)}
            if hasattr(series.dtype, "categories"):
                categories = [str(x) for x in list(series.dtype.categories[:preview_n])]
                payload["preview_unique"] = categories
                payload["total_unique"] = int(len(series.dtype.categories))
            else:
                payload["preview_values"] = [str(x) for x in series.head(preview_n).tolist()]
            preview[str(column)] = payload
        return preview

    def _build_profile(self, config: MASInputConfig, prepared_h5ad_path: str, manifest: dict[str, Any]) -> dict[str, Any]:
        import anndata as ad

        adata = ad.read_h5ad(prepared_h5ad_path, backed="r")
        obsm_keys = list(adata.obsm.keys())
        uns_keys = list(adata.uns.keys())
        summary_preview = {
            "obs": self._sample_obs_preview(adata),
            "var": {"preview_var_names": [str(x) for x in adata.var_names[:12]]},
            "uns": {"keys": uns_keys[:20]},
            "obsm": {"keys": obsm_keys[:20]},
        }
        metadata = dict(config.dataset.metadata)
        profile = {
            "dataset_id": config.dataset.dataset_id,
            "dataset_description": config.dataset.description,
            "task_type": config.task.task_type,
            "task_request": config.task.task_request,
            "input_mode": manifest.get("input_mode", "legacy_files"),
            "active_split": manifest.get("active_split", "default"),
            "available_splits": manifest.get("available_splits", []),
            "split_file_counts": manifest.get("split_file_counts", {}),
            "resolved_paths": manifest["resolved_paths"],
            "num_input_files": manifest["num_files"],
            "structure_hint": manifest["structure_hint"],
            "merge_applied": manifest["merge_applied"],
            "bundled_path": manifest["bundled_path"],
            "source_h5ad_path": manifest.get("h5ad_path", manifest["bundled_path"]),
            "source_npz_path": manifest.get("npz_path", metadata.get("npz_path", "")),
            "working_h5ad_path": manifest["prepared_h5ad_path"],
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "obs_columns": [str(x) for x in adata.obs.columns[:25]],
            "var_preview": [str(x) for x in adata.var_names[:25]],
            "obsm_keys": obsm_keys[:25],
            "uns_keys": uns_keys[:25],
            "has_spatial": any("spatial" in key.lower() for key in obsm_keys + uns_keys),
            "species": metadata.get("species", "unknown"),
            "tissue_hint": metadata.get("tissue", metadata.get("organ", "unknown")),
            "modality": metadata.get("modality", metadata.get("data_modality", "unknown")),
            "dataset_metadata": metadata,
            "summary_preview": summary_preview,
        }
        return profile

    def load_config_node(self, state: InputModuleState) -> dict[str, Any]:
        config = self._read_config(state["config_path"])
        update = {
            "raw_config": config.model_dump(mode="python"),
            "config": config.model_dump(mode="python"),
            "task_type": config.task.task_type,
            "task_request": config.task.task_request,
            "dataset_description": config.dataset.description,
        }
        self._log_preview("load_config_node.update", update)
        return update

    def resolve_inputs_node(self, state: InputModuleState) -> dict[str, Any]:
        config = MASInputConfig.model_validate(state["config"])
        named_input_paths = self._resolve_named_input_paths(state["config_path"], config.dataset)
        split_resolved_paths: dict[str, list[str]] = {}
        if named_input_paths:
            resolved_paths = [named_input_paths["npz_path"], named_input_paths["h5ad_path"]]
        elif config.dataset.split_paths is not None:
            split_resolved_paths = self._resolve_split_source_paths(state["config_path"], config.dataset)
            resolved_paths = split_resolved_paths.get(config.dataset.active_split, [])
            if not resolved_paths:
                raise FileNotFoundError(
                    f"No files were resolved for active_split={config.dataset.active_split!r}."
                )
        else:
            resolved_paths = self._resolve_source_paths(state["config_path"], config.dataset)
        update = {
            "named_input_paths": named_input_paths,
            "split_resolved_paths": split_resolved_paths,
            "resolved_paths": resolved_paths,
        }
        self._log_preview("resolve_inputs_node.update", update)
        return update

    def bundle_inputs_node(self, state: InputModuleState) -> dict[str, Any]:
        config = MASInputConfig.model_validate(state["config"])
        workspace = self._workspace("input_bundle")
        bundled_path, prepared_h5ad_path, manifest = self._bundle_inputs(
            config,
            state["resolved_paths"],
            state["named_input_paths"],
            workspace,
        )
        if state["named_input_paths"]:
            active_split = "direct"
            available_splits = ["direct"]
            split_file_counts = {"direct": len(state["resolved_paths"])}
        elif state["split_resolved_paths"]:
            active_split = config.dataset.active_split
            available_splits = [key for key, value in state["split_resolved_paths"].items() if value]
            split_file_counts = {key: len(value) for key, value in state["split_resolved_paths"].items()}
        else:
            active_split = "default"
            available_splits = ["default"]
            split_file_counts = {"default": len(state["resolved_paths"])}
        update = {
            "bundled_path": bundled_path,
            "prepared_h5ad_path": prepared_h5ad_path,
            "input_manifest": {
                **manifest,
                "active_split": active_split,
                "available_splits": available_splits,
                "named_input_paths": state["named_input_paths"],
                "split_resolved_paths": state["split_resolved_paths"],
                "split_file_counts": split_file_counts,
            },
        }
        self._log_preview("bundle_inputs_node.update", update)
        return update

    def profile_inputs_node(self, state: InputModuleState) -> dict[str, Any]:
        config = MASInputConfig.model_validate(state["config"])
        profile = self._build_profile(config, state["prepared_h5ad_path"], state["input_manifest"])
        update = {"data_profile": profile}
        self._log_preview("profile_inputs_node.update", update)
        return update

    @traceable(name="MASInputModule.run", run_type="chain")
    def run(self, config_path: str | Path) -> dict[str, Any]:
        config_path = str(Path(config_path).resolve())
        initial_state: InputModuleState = {
            "config_path": config_path,
            "raw_config": {},
            "config": {},
            "task_type": "",
            "task_request": "",
            "dataset_description": "",
            "named_input_paths": {},
            "split_resolved_paths": {},
            "resolved_paths": [],
            "bundled_path": "",
            "prepared_h5ad_path": "",
            "data_profile": {},
            "input_manifest": {},
        }
        with tracing_context(enabled=self.langsmith_enabled):
            final_state = self.graph.invoke(
                initial_state,
                config={
                    "run_name": "MASInputModuleGraph",
                    "tags": ["mas", "input-module", "config-driven"],
                },
            )
        return final_state
