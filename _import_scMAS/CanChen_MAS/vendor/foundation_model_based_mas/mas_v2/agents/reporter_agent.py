from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from mas_v2.contracts.schemas import AnalysisResult, ModelSelectionPlan, RunProfile
from mas_v2.runtime.artifacts import RunWorkspace, register_artifacts
from mas_v2.runtime.logging import StructuredRunLogger


class ReportOutput(BaseModel):
    markdown_report: str


def _render_result_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Status | Unknown Rate | Accuracy | Macro F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| {model} | {status} | {unknown} | {acc} | {f1} |".format(
                model=result.get("model_id", ""),
                status=result.get("status", ""),
                unknown=result.get("unknown_rate", ""),
                acc=result.get("coarse_accuracy", ""),
                f1=result.get("coarse_macro_f1", ""),
            )
        )
    return "\n".join(lines)


class ReporterAgent:
    def _fallback_report(
        self,
        *,
        profile: RunProfile,
        intake_bundle: Any,
        plan: ModelSelectionPlan,
        model_results: list[dict[str, Any]],
    ) -> str:
        lines = [
            "# MAS V2 Report",
            "",
            "## Task Overview",
            f"- Dataset ID: {profile.input.dataset_id}",
            f"- Request: {profile.input.task_request}",
            "",
            "## Input Summary",
            f"- Reference source: {profile.input.reference_source.source_type}",
            f"- Query source: {profile.input.query_source.source_type}",
            f"- Query cells: {intake_bundle.query.n_obs}",
            f"- Query genes: {intake_bundle.query.n_vars}",
            f"- Query label keys: {', '.join(intake_bundle.query.candidate_label_keys)}",
            "",
            "## Planner Summary",
            f"- Selected models: {', '.join(item.model_id for item in plan.selected_models)}",
            f"- Judge reviews: {json.dumps(plan.judge_reviews, ensure_ascii=False)}",
            "",
            "## Result Summary",
            _render_result_table(model_results),
            "",
            "## Artifact Summary",
        ]
        for result in model_results:
            lines.append(f"- {result.get('model_id', '')}: {json.dumps(result.get('artifacts', {}), ensure_ascii=False)}")
        return "\n".join(lines)

    def run(
        self,
        *,
        profile: RunProfile,
        intake_bundle: Any,
        plan: ModelSelectionPlan,
        model_results: list[dict[str, Any]],
        workspace: RunWorkspace,
        logger: StructuredRunLogger,
    ) -> str:
        with logger.span("reporter_agent.run"):
            markdown = ""
            if profile.reporter.use_llm:
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    from llm_runtime import build_chat_model

                    llm = build_chat_model(prefix=profile.reporter.llm_prefix, default_temperature=0.0)
                    reporter = llm.with_structured_output(ReportOutput, method="function_calling")
                    out = reporter.invoke(
                        [
                            SystemMessage(
                                content="You are the MAS v2 reporter. Summarize why each model was selected, how data was adapted, and what the results imply. Be concrete and cite artifact paths."
                            ),
                            HumanMessage(
                                content=(
                                    f"Profile:\n{profile.model_dump_json(indent=2)}\n\n"
                                    f"Input bundle:\n{intake_bundle.model_dump_json(indent=2)}\n\n"
                                    f"Plan:\n{plan.model_dump_json(indent=2)}\n\n"
                                    f"Results:\n{json.dumps(model_results, ensure_ascii=False, indent=2, default=str)}"
                                )
                            ),
                        ]
                    )
                    markdown = out.markdown_report
                except Exception as exc:  # noqa: BLE001
                    logger.error("reporter_agent.llm_failed", exc)
            if not markdown:
                markdown = self._fallback_report(
                    profile=profile,
                    intake_bundle=intake_bundle,
                    plan=plan,
                    model_results=model_results,
                )

            report_path = workspace.report_dir / "final_report.md"
            report_path.write_text(markdown, encoding="utf-8")
            register_artifacts(workspace.registry_path, "reporter", {"final_report_md": str(report_path)})
            logger.event("reporter_agent.completed", payload={"report_path": str(report_path)})
            return markdown
