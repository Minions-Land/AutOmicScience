from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from tools_layer import get_function_tools
from tools_layer.mcp_tools import MCP_SERVER_CATALOG, create_client, get_local_tool

from .contracts import classify_path_execution, normalize_server_name, resolve_foundation_model_selection
from .langsmith_compat import traceable, tracing_context
from .tracing import bootstrap_langsmith_from_env


class ExecutorConfigModel(BaseModel):
    max_parallel_paths: int = Field(default=3, ge=1)
    artifact_root: str = "../outputs/executor_module"


EXECUTOR_RESERVED_TOOL_KEYS = {
    "mcp_tool",
    "mcp_server",
    "candidate_tool",
    "candidate_server",
    "selected_mcp_tool_name",
    "selected_model_id",
    "path_id",
    "path_type",
    "task_type",
}

def resolve_annotation_input_paths(
    *,
    prepared_h5ad_path: str,
    input_manifest: dict[str, Any],
    invocation_outline: dict[str, Any],
) -> dict[str, str]:
    resolved_paths = list(input_manifest.get("resolved_paths", []))
    split_resolved_paths = input_manifest.get("split_resolved_paths", {})
    metadata = input_manifest.get("metadata", {})

    candidate_npz_paths: list[str] = []
    for candidate in [
        invocation_outline.get("npz_path"),
        metadata.get("npz_path"),
        metadata.get("annotation_npz_path"),
    ]:
        if candidate:
            candidate_npz_paths.append(str(candidate))
    for item in resolved_paths:
        if str(item).lower().endswith(".npz"):
            candidate_npz_paths.append(str(item))
    for items in split_resolved_paths.values():
        for item in items:
            if str(item).lower().endswith(".npz"):
                candidate_npz_paths.append(str(item))

    npz_path = next((path for path in candidate_npz_paths if Path(path).exists()), "")

    candidate_h5ad_paths: list[str] = []
    for candidate in [
        invocation_outline.get("h5ad_path"),
        invocation_outline.get("cell_metadata_h5ad_path"),
        prepared_h5ad_path,
        metadata.get("h5ad_path"),
    ]:
        if candidate:
            candidate_h5ad_paths.append(str(candidate))
    for item in resolved_paths:
        if str(item).lower().endswith(".h5ad"):
            candidate_h5ad_paths.append(str(item))

    h5ad_path = next((path for path in candidate_h5ad_paths if Path(path).exists()), "")
    return {
        "npz_path": npz_path,
        "h5ad_path": h5ad_path or str(prepared_h5ad_path),
        "cell_metadata_h5ad_path": h5ad_path or str(prepared_h5ad_path),
    }

class ExecutorModule:
    def __init__(self, *, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tracing_config = bootstrap_langsmith_from_env(self.project_root, default_project="foundation_model_based_mas_executor")
        self.langsmith_enabled = bool(self.tracing_config.get("enabled", False))
        self.logger = logging.getLogger("foundation_model_based_mas.executor_module")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh = logging.FileHandler(self.logs_dir / "mas_executor_module.log", encoding="utf-8")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.capability_tool = get_function_tools(["read_capability_files_tool"])[0]

    def _discover_idle_gpu_devices(self) -> list[str]:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            self.logger.info("[executor.gpu_probe] skipped idle-GPU discovery: %s: %s", type(exc).__name__, exc)
            return []

        rows: list[tuple[int, int, int]] = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                gpu_index = int(parts[0])
                memory_used = int(parts[1])
                utilization = int(parts[2])
            except ValueError:
                continue
            rows.append((gpu_index, memory_used, utilization))
        rows.sort(key=lambda item: (item[2], item[1], item[0]))
        return [f"cuda:{gpu_index}" for gpu_index, _, _ in rows]

    def _assign_default_devices(
        self,
        runnable_paths: list[tuple[dict[str, Any], dict[str, Any], Path]],
    ) -> None:
        idle_devices = self._discover_idle_gpu_devices()
        if not idle_devices:
            return

        assigned_rows: list[dict[str, Any]] = []
        next_index = 0
        for path, capability, _ in runnable_paths:
            if path.get("path_type") != "foundation_model":
                continue
            invocation_outline = path.get("invocation_outline", {})
            outline_parameters = invocation_outline.get("parameters", {}) if isinstance(invocation_outline, dict) else {}
            explicit_device = (
                (outline_parameters.get("device") if isinstance(outline_parameters, dict) else None)
                or (invocation_outline.get("device") if isinstance(invocation_outline, dict) else None)
                or capability.get("executor_defaults", {}).get("device")
            )
            if explicit_device:
                continue
            assigned_device = idle_devices[next_index % len(idle_devices)]
            next_index += 1
            path["_executor_assigned_device"] = assigned_device
            assigned_rows.append(
                {
                    "path_id": path.get("path_id", ""),
                    "selected_model_id": path.get("selected_model_id", ""),
                    "assigned_device": assigned_device,
                }
            )

        if assigned_rows:
            self._log_preview("executor.gpu_assignment", assigned_rows)

    def _log_preview(self, node: str, payload: Any) -> None:
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = str(payload)
        if len(text) > 1200:
            text = text[:1200] + "...[truncated]"
        self.logger.info("[%s] %s", node, text)

    def _load_executor_config(self, config_path: str | Path) -> ExecutorConfigModel:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        return ExecutorConfigModel.model_validate(payload.get("executor", {}))

    def _resolve_artifact_root(self, config_path: str | Path, executor_config: ExecutorConfigModel) -> Path:
        artifact_root = Path(executor_config.artifact_root)
        if not artifact_root.is_absolute():
            artifact_root = (Path(config_path).resolve().parent / artifact_root).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        return artifact_root

    def _load_capability_index(self, config_path: str | Path) -> dict[str, dict[str, Any]]:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        planner_config = config.get("planner", {})
        capability_dir = Path(planner_config.get("capability_dir", "../config/capability"))
        if not capability_dir.is_absolute():
            capability_dir = (Path(config_path).resolve().parent / capability_dir).resolve()
        glob_pattern = str(planner_config.get("capability_glob", "**/*.yaml"))
        result = self.capability_tool.invoke(
            {"directory_path": str(capability_dir), "glob_pattern": glob_pattern}
        )
        if not result.get("success", False):
            raise RuntimeError(result.get("summary", "Failed to read capability files."))
        return result.get("model_index", {})

    async def _aget_mcp_tool(self, server_name: str, tool_name: str):
        client = create_client([server_name])
        tools = await client.get_tools()
        tool_map = {tool.name: tool for tool in tools}
        if tool_name not in tool_map:
            raise KeyError(f"Tool '{tool_name}' not found on MCP server '{server_name}'.")
        return tool_map[tool_name]

    async def _ainvoke_local_tool(self, server_name: str, tool_name: str, payload: dict[str, Any]) -> Any:
        tmp_dir = self.logs_dir / "executor_local_tool_calls"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        fd, request_path_raw = tempfile.mkstemp(prefix="request_", suffix=".json", dir=str(tmp_dir))
        os.close(fd)
        request_path = Path(request_path_raw)
        response_path = request_path.with_name(request_path.stem + "_response.json")
        stderr_path = request_path.with_name(request_path.stem + "_stderr.log")

        request_payload = {
            "server_name": server_name,
            "tool_name": tool_name,
            "payload": payload,
        }
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        runner_code = """
import json
import os
import sys
import traceback
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
request_path = Path(sys.argv[2]).resolve()
response_path = Path(sys.argv[3]).resolve()

pythonpath = os.environ.get("PYTHONPATH", "")
paths = [str(project_root)]
if pythonpath:
    paths.append(pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(paths)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools_layer.mcp_tools import get_local_tool

request = json.loads(request_path.read_text(encoding="utf-8"))

try:
    tool = get_local_tool(request["server_name"], request["tool_name"])
    result = tool(**request["payload"])
    payload = {"ok": True, "result": result}
except Exception:
    payload = {"ok": False, "error": traceback.format_exc()}

response_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)
"""

        env = dict(os.environ)
        pythonpath = env.get("PYTHONPATH", "")
        if str(self.project_root) not in pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = (
                f"{self.project_root}{os.pathsep}{pythonpath}" if pythonpath else str(self.project_root)
            )

        stderr_handle = open(stderr_path, "w", encoding="utf-8")
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                runner_code,
                str(self.project_root),
                str(request_path),
                str(response_path),
                cwd=str(self.project_root),
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr_handle,
            )
            returncode = await process.wait()
        finally:
            stderr_handle.close()

        response_payload: dict[str, Any] | None = None
        if response_path.exists():
            try:
                response_payload = json.loads(response_path.read_text(encoding="utf-8"))
            except Exception:
                response_payload = None

        if returncode == 0 and response_payload and response_payload.get("ok"):
            try:
                request_path.unlink(missing_ok=True)
                response_path.unlink(missing_ok=True)
                stderr_path.unlink(missing_ok=True)
            except Exception:
                pass
            return response_payload.get("result")

        stderr_tail = ""
        if stderr_path.exists():
            try:
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
                stderr_tail = stderr_text[-3000:]
            except Exception:
                stderr_tail = ""

        error_message = "Local tool subprocess failed."
        if response_payload and response_payload.get("error"):
            error_message = str(response_payload["error"])
        elif returncode != 0:
            error_message = f"Local tool subprocess exited with code {returncode}."
        if stderr_tail:
            error_message = f"{error_message}\n[stderr]\n{stderr_tail}"
        raise RuntimeError(error_message)

    async def _ainvoke_mcp_tool(self, server_name: str, tool_name: str, payload: dict[str, Any]) -> Any:
        force_local = os.getenv("MAS_EXECUTOR_FORCE_LOCAL_TOOL_CALL", "").strip().lower() in {"1", "true", "yes", "on"}
        disable_local = os.getenv("MAS_EXECUTOR_DISABLE_LOCAL_TOOL_CALL", "").strip().lower() in {"1", "true", "yes", "on"}
        if force_local or not disable_local:
            try:
                self.logger.info(
                    "[executor.local_%s] using local tool call for %s.%s",
                    "override" if force_local else "preferred",
                    server_name,
                    tool_name,
                )
                return await self._ainvoke_local_tool(server_name, tool_name, payload)
            except KeyError:
                if force_local:
                    raise

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                tool = await self._aget_mcp_tool(server_name, tool_name)
                return await tool.ainvoke(payload)
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "[executor.retry] failed %s.%s | attempt=%s/3 | error=%s: %s",
                    server_name,
                    tool_name,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                if attempt < 3:
                    await asyncio.sleep(attempt)
        self.logger.warning(
            "[executor.local_fallback] falling back to local tool call for %s.%s after MCP failure: %s: %s",
            server_name,
            tool_name,
            type(last_error).__name__ if last_error else "RuntimeError",
            last_error or "unknown MCP failure",
        )
        try:
            return await self._ainvoke_local_tool(server_name, tool_name, payload)
        except Exception:
            if last_error is not None:
                raise
            raise RuntimeError(f"Failed MCP call and local fallback: {server_name}.{tool_name}")

    def _order_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not steps:
            return []
        step_map = {str(step.get("step_id", f"step_{idx}")): dict(step) for idx, step in enumerate(steps)}
        incoming = {step_id: set(step_map[step_id].get("depends_on", [])) for step_id in step_map}
        ordered: list[dict[str, Any]] = []
        remaining = list(step_map.keys())
        while remaining:
            ready = [step_id for step_id in remaining if not incoming[step_id]]
            if not ready:
                return [step_map[step_id] for step_id in remaining]
            for step_id in ready:
                ordered.append(step_map[step_id])
                remaining.remove(step_id)
                for other_id in remaining:
                    incoming[other_id].discard(step_id)
        return ordered

    def _build_step_payload(
        self,
        *,
        path: dict[str, Any],
        capability: dict[str, Any],
        step: dict[str, Any],
        prepared_h5ad_path: str,
        input_manifest: dict[str, Any],
        path_dir: Path,
    ) -> dict[str, Any]:
        invocation_outline = self._sanitize_invocation_outline(dict(path.get("invocation_outline", {})))
        if "parameters" in invocation_outline and isinstance(invocation_outline["parameters"], dict):
            payload = dict(capability.get("executor_defaults", {}))
            payload.update(invocation_outline["parameters"])
        else:
            payload = dict(capability.get("executor_defaults", {}))
            payload.update(invocation_outline)
        payload.update(resolve_annotation_input_paths(
            prepared_h5ad_path=prepared_h5ad_path,
            input_manifest=input_manifest,
            invocation_outline=invocation_outline,
        ))
        payload["result_json_path"] = str(path_dir / f"{step['step_id']}_result.json")
        payload["prediction_npz_path"] = str(path_dir / f"{step['step_id']}_prediction_records.npz")
        payload["cell_metadata_h5ad_path"] = payload.get("cell_metadata_h5ad_path") or prepared_h5ad_path
        if not payload.get("device") and path.get("_executor_assigned_device"):
            payload["device"] = str(path["_executor_assigned_device"])

        if "h5ad_path" not in payload and Path(prepared_h5ad_path).suffix.lower() == ".h5ad":
            payload["h5ad_path"] = prepared_h5ad_path
        return payload

    def _sanitize_invocation_outline(self, invocation_outline: dict[str, Any]) -> dict[str, Any]:
        sanitized = {
            key: value
            for key, value in invocation_outline.items()
            if key not in EXECUTOR_RESERVED_TOOL_KEYS
        }
        if "parameters" in sanitized and isinstance(sanitized["parameters"], dict):
            sanitized["parameters"] = {
                key: value
                for key, value in sanitized["parameters"].items()
                if key not in EXECUTOR_RESERVED_TOOL_KEYS
            }
        return sanitized

    def _build_foundation_execution_steps(
        self,
        *,
        path: dict[str, Any],
        capability: dict[str, Any],
    ) -> list[dict[str, Any]]:
        selected_tool_name = str(path.get("selected_mcp_tool_name", "")).strip() or str(capability.get("mcp_tool_name", "")).strip()
        step_id = f"{str(path.get('path_id', 'foundation_model_path')).strip() or 'foundation_model_path'}_execute"
        return [
            {
                "step_id": step_id,
                "step_name": "execute_selected_mcp_tool",
                "step_type": "mcp_call",
                "candidate_server": str(capability.get("mcp_server_name", "")).strip(),
                "candidate_tool": selected_tool_name,
                "purpose": f"Execute the selected foundation-model MCP tool {selected_tool_name}.",
                "depends_on": [],
                "input_contract": {},
                "output_contract": {},
            }
        ]

    async def _run_path(
        self,
        *,
        path: dict[str, Any],
        capability: dict[str, Any],
        prepared_h5ad_path: str,
        input_manifest: dict[str, Any],
        path_dir: Path,
    ) -> dict[str, Any]:
        path_dir.mkdir(parents=True, exist_ok=True)
        ordered_steps = self._build_foundation_execution_steps(path=path, capability=capability)
        step_results: list[dict[str, Any]] = []
        final_prediction_artifact = ""
        final_result_json = ""
        started_at = time.time()

        try:
            for index, step in enumerate(ordered_steps, start=1):
                candidate_server = normalize_server_name(
                    str(step.get("candidate_server", "")).strip() or str(capability.get("mcp_server_name", "")).strip(),
                    MCP_SERVER_CATALOG,
                )
                candidate_tool = str(step.get("candidate_tool", "")).strip() or str(capability.get("mcp_tool_name", "")).strip()
                payload = self._build_step_payload(
                    path=path,
                    capability=capability,
                    step=step,
                    prepared_h5ad_path=prepared_h5ad_path,
                    input_manifest=input_manifest,
                    path_dir=path_dir,
                )
                if not payload.get("npz_path"):
                    raise FileNotFoundError(
                        "No npz_path could be resolved for annotation FM execution. "
                        "Provide it in invocation_outline, dataset metadata, or resolved input paths."
                    )

                step_started_at = time.time()
                raw_result = await self._ainvoke_mcp_tool(candidate_server, candidate_tool, payload)
                output_payload = raw_result.get("output", {}) if isinstance(raw_result, dict) else raw_result
                if isinstance(output_payload, dict) and output_payload.get("error"):
                    raise RuntimeError(str(output_payload["error"]))
                final_prediction_artifact = str(output_payload.get("prediction_artifact_path", "")) or payload["prediction_npz_path"]
                final_result_json = payload["result_json_path"]
                step_results.append(
                    {
                        "step_index": index,
                        "step_id": step.get("step_id", f"step_{index}"),
                        "candidate_server": candidate_server,
                        "candidate_tool": candidate_tool,
                        "payload": payload,
                        "duration_sec": round(time.time() - step_started_at, 3),
                        "result": raw_result,
                    }
                )

            path_result = {
                "path_id": path.get("path_id", ""),
                "path_type": path.get("path_type", ""),
                "task_type": path.get("task_type", ""),
                "selected_model_id": path.get("selected_model_id", ""),
                "selected_mcp_tool_name": path.get("selected_mcp_tool_name", ""),
                "status": "completed",
                "duration_sec": round(time.time() - started_at, 3),
                "path_dir": str(path_dir),
                "result_json_path": final_result_json,
                "prediction_artifact_path": final_prediction_artifact,
                "step_results": step_results,
            }
        except Exception as exc:
            path_result = {
                "path_id": path.get("path_id", ""),
                "path_type": path.get("path_type", ""),
                "task_type": path.get("task_type", ""),
                "selected_model_id": path.get("selected_model_id", ""),
                "selected_mcp_tool_name": path.get("selected_mcp_tool_name", ""),
                "status": "failed",
                "duration_sec": round(time.time() - started_at, 3),
                "path_dir": str(path_dir),
                "error": f"{type(exc).__name__}: {exc}",
                "step_results": step_results,
            }

        (path_dir / "path_result.json").write_text(
            json.dumps(path_result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path_result

    async def _run_paths_async(
        self,
        *,
        runnable_paths: list[tuple[dict[str, Any], dict[str, Any], Path]],
        prepared_h5ad_path: str,
        input_manifest: dict[str, Any],
        max_parallel_paths: int,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_parallel_paths)

        async def _guarded_run(path: dict[str, Any], capability: dict[str, Any], path_dir: Path) -> dict[str, Any]:
            async with semaphore:
                return await self._run_path(
                    path=path,
                    capability=capability,
                    prepared_h5ad_path=prepared_h5ad_path,
                    input_manifest=input_manifest,
                    path_dir=path_dir,
                )

        tasks = [_guarded_run(path, capability, path_dir) for path, capability, path_dir in runnable_paths]
        return await asyncio.gather(*tasks)

    @traceable(name="ExecutorModule.run", run_type="chain")
    def run(
        self,
        *,
        config_path: str | Path,
        task_type: str,
        prepared_h5ad_path: str,
        input_manifest: dict[str, Any],
        approved_paths: list[dict[str, Any]],
        ground_truth_label_key: str = "",
    ) -> dict[str, Any]:
        config_path = str(Path(config_path).resolve())
        executor_config = self._load_executor_config(config_path)
        capability_index = self._load_capability_index(config_path)
        artifact_root = self._resolve_artifact_root(config_path, executor_config)
        run_dir = artifact_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        runnable_paths: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
        skipped_paths: list[dict[str, Any]] = []

        for path in approved_paths:
            capability, _, resolved_model_id, resolved_tool_name = resolve_foundation_model_selection(
                path=path,
                capability_index=capability_index,
            )
            if resolved_model_id and not str(path.get("selected_model_id", "")).strip():
                path["selected_model_id"] = resolved_model_id
            if resolved_tool_name and not str(path.get("selected_mcp_tool_name", "")).strip():
                path["selected_mcp_tool_name"] = resolved_tool_name
            runnable, reason, capability = classify_path_execution(
                path=path,
                task_type=task_type,
                capability_index=capability_index,
                server_catalog=MCP_SERVER_CATALOG,
            )
            if not runnable:
                skipped_paths.append(
                    {
                        "path_id": path.get("path_id", ""),
                        "selected_model_id": path.get("selected_model_id", ""),
                        "selected_mcp_tool_name": path.get("selected_mcp_tool_name", ""),
                        "reason": reason,
                    }
                )
                continue
            runnable_paths.append((path, capability or {}, run_dir / str(path.get("path_id", "unnamed_path"))))

        self._assign_default_devices(runnable_paths)

        if not runnable_paths:
            execution_status = "unsupported_task" if task_type != "annotation" else "no_runnable_path"
            summary = {
                "execution_status": execution_status,
                "run_dir": str(run_dir),
                "executed_paths": [],
                "skipped_paths": skipped_paths,
                "path_results": [],
                "ground_truth_label_key": ground_truth_label_key,
            }
            (run_dir / "execution_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            self._log_preview("executor.run.summary", summary)
            return summary

        with tracing_context(enabled=self.langsmith_enabled):
            path_results = asyncio.run(
                self._run_paths_async(
                    runnable_paths=runnable_paths,
                    prepared_h5ad_path=prepared_h5ad_path,
                    input_manifest=input_manifest,
                    max_parallel_paths=executor_config.max_parallel_paths,
                )
            )

        executed_paths = [
            {"path_id": result.get("path_id", ""), "status": result.get("status", "unknown")}
            for result in path_results
        ]
        execution_status = "completed"
        if any(result.get("status") != "completed" for result in path_results):
            execution_status = "partial_failure"

        summary = {
            "execution_status": execution_status,
            "run_dir": str(run_dir),
            "executed_paths": executed_paths,
            "skipped_paths": skipped_paths,
            "path_results": path_results,
            "ground_truth_label_key": ground_truth_label_key,
        }
        (run_dir / "execution_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._log_preview("executor.run.summary", summary)
        return summary
