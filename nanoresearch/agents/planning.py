"""Planning agent — generates experiment blueprint from ideation output."""

from __future__ import annotations

import json
import logging
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.evolution.memory import MemoryType
from nanoresearch.idea_utils import (
    add_idea_aliases_to_blueprint,
    get_idea_candidates,
    get_idea_id,
    get_selected_idea,
    get_selected_idea_id,
)
from nanoresearch.schemas.experiment import ExperimentBlueprint
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Configurable limits
MAX_PAPERS_IN_SUMMARY = 15

PLANNING_SYSTEM_PROMPT = """You are a research experiment planner. Your task is to design a rigorous experiment plan based on the selected idea from the ideation phase.

CRITICAL RULES — evidence grounding:
- NEVER invent baseline performance numbers. Use ONLY numbers from the published evidence provided.
- If a metric value is not available in the evidence, set it to "N/A" — do NOT guess.
- For the proposed method, describe expected improvements as a "projected improvement range" (e.g. "5-10% improvement over best baseline"), NOT as exact numbers.
- Every baseline number MUST have a "performance_provenance" entry citing which paper it came from.
- Mark proposed-method values as projected: set "is_projected" to true for each metric.
- Reuse metric names EXACTLY. The keys in baseline "expected_performance" must exactly match the strings you put in the top-level "metrics" list.
- Make ablations traceable. Every ablation group and variant must explicitly mention one of the proposed_method.key_components or an exact method module name in its own name/description.

Design experiments that are:
- Reproducible with standard ML frameworks
- Include proper baselines for comparison (using REAL published numbers)
- Have clear evaluation metrics
- Include ablation studies to validate each component

Always respond in valid JSON format."""

BLUEPRINT_REVIEW_SYSTEM_PROMPT = """You are a pragmatic research-plan reviewer.

Your job is NOT to nitpick naming style. Your job is to detect whether an experiment blueprint has execution-blocking or evaluation-invalidating problems.

Review principles:
- Ignore cosmetic wording differences, capitalization differences, and harmless naming mismatches.
- Do NOT require exact string overlap between ablation names and method descriptions.
- Only mark a problem as fatal if it is likely to make execution, evaluation, or baseline comparison invalid.
- If a metric/baseline mismatch is only a naming alias issue (e.g. Accuracy vs accuracy, F1 vs F1 Score), treat it as non-fatal and mention it only as a warning.
- Prefer concise, actionable repair instructions.

Return ONLY valid JSON with this schema:
{
  "should_retry": bool,
  "fatal_issues": [{"type": str, "message": str, "repair_instruction": str}],
  "warnings": [{"type": str, "message": str}],
  "summary": str
}
"""


class PlanningAgent(BaseResearchAgent):
    stage = PipelineStage.PLANNING

    async def run(self, **inputs: Any) -> dict[str, Any]:
        if "ideation_output" not in inputs:
            raise ValueError("PlanningAgent requires 'ideation_output' in inputs")
        ideation_data: dict = inputs["ideation_output"]
        if not isinstance(ideation_data, dict):
            raise TypeError(
                f"Expected dict for ideation_output, got {type(ideation_data).__name__}"
            )
        self.log("Starting experiment planning")

        topic = ideation_data.get("topic", "")
        selected_hyp = get_selected_idea_id(ideation_data)
        rationale = ideation_data.get("rationale", "")
        adaptive_context = self.build_adaptive_context(
            "planning",
            topic=topic,
            text=json.dumps(ideation_data, ensure_ascii=False)[:4000],
            tags=[topic, selected_hyp, self.workspace.manifest.paper_mode.value],
        )
        retry_error = str(inputs.get("_retry_error", "")).strip()
        if retry_error:
            self.learn_from_trace(
                "planning",
                "planning_retry",
                retry_error,
                tags=[topic, selected_hyp, "retry"],
            )

        # Find the selected idea details while staying compatible with legacy fields.
        hyp_detail = ""
        hypotheses = get_idea_candidates(ideation_data)
        selected_idea = get_selected_idea(ideation_data)
        if selected_idea:
            hyp_detail = json.dumps(selected_idea, indent=2)
        if not hyp_detail and hypotheses:
            logger.warning(
                "Selected idea %r not found among %d idea candidates, using first",
                selected_hyp, len(hypotheses),
            )
            hyp_detail = json.dumps(hypotheses[0], indent=2)
            selected_hyp = get_idea_id(hypotheses[0], selected_hyp)

        # Summarize related gaps (limit to 10 to avoid prompt overflow)
        gaps_text = json.dumps(ideation_data.get("gaps", [])[:10], indent=2)

        # Summarize key papers
        papers = ideation_data.get("papers", [])
        papers_summary = "\n".join(
            f"- {p.get('title', '?')} ({p.get('year', '?')})"
            for p in papers[:MAX_PAPERS_IN_SUMMARY]
        )

        # Build evidence block from extracted metrics
        evidence_block = self._build_evidence_block(ideation_data)

        adaptive_prefix = f"{adaptive_context}\n\n" if adaptive_context else ""
        prompt = f"""{adaptive_prefix}Research Topic: "{topic}"

Selected Idea: {selected_hyp}
{hyp_detail}

Selection Rationale: {rationale}

Identified Research Gaps:
{gaps_text}

Key Related Papers:
{papers_summary}

{evidence_block}

Design a comprehensive experiment blueprint as JSON with:
1. "title": Experiment title
2. "idea_ref" or "hypothesis_ref": "{selected_hyp}"
3. "datasets": Array of datasets, each with:
   - "name", "description", "source_url", "size_info", "preprocessing_notes"
4. "baselines": Array of baseline methods, each with:
   - "name", "description", "reference_paper_id"
   - "expected_performance": dict of metric→value (use ONLY values from the evidence above, or "N/A")
   - "performance_provenance": dict of metric→source (e.g. "Abstract of arxiv:2401.00001")
   - "is_projected": dict of metric→bool (true only for proposed method projections)
5. "proposed_method": Object describing the proposed method with:
   - "name", "description", "key_components" (list), "architecture" (text description)
6. "metrics": Array of evaluation metrics, each with:
   - "name", "description", "higher_is_better" (bool), "primary" (bool)
7. "ablation_groups": Array of ablation studies, each with:
   - "group_name", "description", "variants" (array of dicts describing each variant)
8. "compute_requirements": Object with "gpu_type", "num_gpus", "estimated_hours", "memory_gb"
9. "evidence_summary": Brief summary of which published numbers were used
10. "data_provenance_note": Statement distinguishing published baseline numbers from projected improvements

IMPORTANT: Use ONLY the numbers from the PUBLISHED QUANTITATIVE EVIDENCE section.
If evidence is missing for a baseline-metric pair, use "N/A".
IMPORTANT: Do not invent cosmetic ablation names. If a variant is a sweep, include the controlled method component in the name, e.g. "retrieval top_k=3" instead of just "Top-3 retrieval", and "LoRA rank=8" instead of just "rank_8".

Return ONLY valid JSON."""

        # Retry loop: if blueprint validation fails, retry with error feedback
        max_planning_retries = 2
        last_validation_error = ""
        review: dict[str, Any] = {}
        for planning_attempt in range(max_planning_retries + 1):
            retry_prompt = prompt
            if last_validation_error:
                retry_prompt += (
                    f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                    f"Your previous JSON output failed schema validation:\n"
                    f"{last_validation_error}\n"
                    f"Please fix the issues and return valid JSON.\n"
                    f"--- END ERROR ---"
                )

            result = await self.generate_json(PLANNING_SYSTEM_PROMPT, retry_prompt)

            # Guard: generate_json may return a list instead of a dict
            if isinstance(result, list):
                if len(result) == 1 and isinstance(result[0], dict):
                    result = result[0]
                else:
                    last_validation_error = (
                        f"Expected JSON object, got list of length {len(result)}"
                    )
                    if planning_attempt < max_planning_retries:
                        logger.warning(
                            "Planning attempt %d/%d: %s",
                            planning_attempt + 1, max_planning_retries + 1,
                            last_validation_error,
                        )
                        continue
                    raise RuntimeError(last_validation_error)

            # Coerce fields the LLM may return as structured objects but schema expects as strings
            result = self._coerce_blueprint_fields(result)

            try:
                blueprint = ExperimentBlueprint.model_validate(result)
                review = await self._review_blueprint_with_llm(
                    topic=topic,
                    ideation_data=ideation_data,
                    blueprint_data=blueprint.model_dump(mode="json"),
                )
                fatal_issues = review.get("fatal_issues", [])
                if fatal_issues:
                    last_validation_error = self._format_blueprint_review_feedback(review)
                    if planning_attempt < max_planning_retries:
                        logger.warning(
                            "Planning attempt %d/%d LLM review requested retry: %s",
                            planning_attempt + 1, max_planning_retries + 1,
                            last_validation_error,
                        )
                        continue
                    logger.error(
                        "Blueprint LLM review still found fatal issues after %d attempts: %s",
                        max_planning_retries + 1,
                        last_validation_error,
                    )
                break  # Validation + LLM review succeeded or final attempt exhausted
            except Exception as exc:
                last_validation_error = str(exc)
                if planning_attempt < max_planning_retries:
                    logger.warning(
                        "Planning attempt %d/%d validation failed: %s",
                        planning_attempt + 1, max_planning_retries + 1,
                        last_validation_error,
                    )
                    continue
                logger.error("Blueprint validation failed after %d attempts: %s",
                             max_planning_retries + 1, exc)
                raise RuntimeError(
                    f"LLM output does not match ExperimentBlueprint schema: {exc}"
                ) from exc

        # Save output
        blueprint_payload = add_idea_aliases_to_blueprint(blueprint.model_dump(mode="json"))
        output_path = self.workspace.write_json(
            "plans/experiment_blueprint.json",
            blueprint_payload,
        )
        self.workspace.register_artifact(
            "experiment_blueprint", output_path, self.stage
        )
        if isinstance(review, dict) and review:
            review_path = self.workspace.write_json(
                "logs/blueprint_review.json",
                review,
            )
            self.workspace.register_artifact(
                "blueprint_review", review_path, self.stage,
            )
        primary_metrics = [m.name for m in blueprint.metrics if m.primary] or [m.name for m in blueprint.metrics[:3]]
        primary_metrics_text = ", ".join(primary_metrics)
        dataset_names = ", ".join(ds.name for ds in blueprint.datasets[:3])
        proposed_method_name = self._get_proposed_method_name(blueprint)
        self.remember_context(
            MemoryType.PROJECT_CONTEXT,
            f"Planning blueprint for {topic}: method={proposed_method_name}, datasets={dataset_names}, primary metrics={primary_metrics_text}",
            importance=0.78,
            tags=[topic, proposed_method_name, "planning"],
            source="experiment_blueprint",
            topic=topic,
        )
        self.remember_context(
            MemoryType.DECISION_HISTORY,
            f"Planning constraints for {topic}: include {len(blueprint.ablation_groups)} ablation groups and compute estimate {blueprint.compute_requirements.num_gpus}x {blueprint.compute_requirements.gpu_type} for {blueprint.compute_requirements.estimated_hours}h.",
            importance=0.72,
            tags=[topic, "ablation", "compute"],
            source="experiment_blueprint",
            topic=topic,
        )
        self.remember_promising_direction(
            topic=topic,
            ideation_output=ideation_data,
            planning_output=blueprint_payload,
            artifact_path="logs/promising_direction_summary_planning.json",
            source_stage="planning",
            source="experiment_blueprint",
        )
        planning_trace = (
            f"Blueprint for {topic}: method={proposed_method_name}; "
            f"datasets={[ds.name for ds in blueprint.datasets]}; "
            f"primary_metrics={primary_metrics}; "
            f"ablation_groups={len(blueprint.ablation_groups)}; "
            f"compute={blueprint.compute_requirements.num_gpus}x {blueprint.compute_requirements.gpu_type} "
            f"for {blueprint.compute_requirements.estimated_hours}h."
        )
        self.learn_from_trace(
            "planning",
            "planning_blueprint",
            planning_trace,
            tags=[topic, proposed_method_name, "planning", "blueprint"],
            confidence=0.68,
        )
        logger.info("[%s] Blueprint generated: %s", self.stage.value, blueprint.title)
        return blueprint_payload

    @staticmethod
    def _get_proposed_method_name(blueprint: ExperimentBlueprint) -> str:
        """Extract a stable proposed-method label from the validated blueprint."""
        method = blueprint.proposed_method
        if isinstance(method, dict):
            name = method.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return "proposed_method"

    @staticmethod
    def _build_evidence_block(ideation_data: dict) -> str:
        """Format extracted evidence for inclusion in the planning prompt."""
        evidence = ideation_data.get("evidence", {})
        metrics = evidence.get("extracted_metrics", [])

        if not metrics:
            return (
                "=== PUBLISHED QUANTITATIVE EVIDENCE ===\n"
                "No quantitative evidence was extracted from the literature review.\n"
                "For baseline expected_performance values, use well-known published results\n"
                "from the original papers of each baseline method on the target dataset.\n"
                "If you are confident about a baseline's published result on this dataset,\n"
                "include it. Otherwise use null (NOT 'N/A').\n"
                "=== END EVIDENCE ==="
            )

        lines = ["=== PUBLISHED QUANTITATIVE EVIDENCE ==="]
        for m in metrics[:100]:  # cap to avoid prompt overflow
            if not isinstance(m, dict):
                continue
            value = m.get("value", "?")
            unit = m.get("unit", "")
            unit_str = f" {unit}" if unit else ""
            lines.append(
                f"- {m.get('method_name', '?')} on {m.get('dataset', '?')}: "
                f"{m.get('metric_name', '?')} = {value}{unit_str} "
                f"(source: {m.get('paper_id', '?')} — \"{m.get('context', '')}\")"
            )

        notes = evidence.get("extraction_notes", "")
        if notes:
            lines.append(f"\nExtraction notes: {notes}")

        warnings = evidence.get("coverage_warnings", [])
        for w in warnings:
            lines.append(f"WARNING: {w}")

        lines.append("=== END EVIDENCE ===")
        return "\n".join(lines)

    async def _review_blueprint_with_llm(
        self,
        *,
        topic: str,
        ideation_data: dict,
        blueprint_data: dict,
    ) -> dict[str, Any]:
        """Run a lightweight LLM review for semantic blueprint validity.

        Fail-open on API errors so the planning stage does not become brittle.
        """
        selected_hyp = get_selected_idea_id(ideation_data)
        user_prompt = (
            f"Research topic: {topic}\n"
            f"Selected idea: {selected_hyp}\n\n"
            "Decide whether this experiment blueprint has any execution-blocking or "
            "evaluation-invalidating issues.\n\n"
            "Ideation output:\n"
            f"{json.dumps(ideation_data, ensure_ascii=False)[:6000]}\n\n"
            "Experiment blueprint:\n"
            f"{json.dumps(blueprint_data, ensure_ascii=False)[:12000]}"
        )
        try:
            review = await self.generate_json(
                BLUEPRINT_REVIEW_SYSTEM_PROMPT,
                user_prompt,
                stage_override=self.config.for_stage("review"),
            )
        except Exception as exc:
            logger.warning("Blueprint LLM review failed, accepting schema-valid blueprint: %s", exc)
            return {
                "should_retry": False,
                "fatal_issues": [],
                "warnings": [{"type": "review_unavailable", "message": str(exc)}],
                "summary": "LLM blueprint review unavailable; schema-valid blueprint accepted.",
            }

        if not isinstance(review, dict):
            return {
                "should_retry": False,
                "fatal_issues": [],
                "warnings": [{"type": "review_invalid", "message": f"Unexpected review type: {type(review).__name__}"}],
                "summary": "LLM blueprint review returned invalid shape; schema-valid blueprint accepted.",
            }

        fatal_issues = review.get("fatal_issues")
        warnings = review.get("warnings")
        if not isinstance(fatal_issues, list):
            review["fatal_issues"] = []
        if not isinstance(warnings, list):
            review["warnings"] = []
        review["should_retry"] = bool(review.get("should_retry")) and bool(review["fatal_issues"])
        review["summary"] = str(review.get("summary", "") or "").strip()
        return review

    @staticmethod
    def _format_blueprint_review_feedback(review: dict[str, Any]) -> str:
        fatal_issues = review.get("fatal_issues", [])
        lines: list[str] = []
        for idx, issue in enumerate(fatal_issues, start=1):
            if not isinstance(issue, dict):
                lines.append(f"{idx}. {issue}")
                continue
            issue_type = issue.get("type", "fatal_issue")
            message = issue.get("message", "")
            repair = issue.get("repair_instruction", "")
            line = f"{idx}. [{issue_type}] {message}".strip()
            if repair:
                line += f" Fix: {repair}"
            lines.append(line)
        return "\n".join(lines) if lines else str(review.get("summary", "") or "LLM blueprint review requested retry.")

    @staticmethod
    def _coerce_blueprint_fields(data: dict) -> dict:
        """Coerce LLM output fields that should be strings but came as dicts/lists.

        The LLM sometimes returns structured objects for fields like size_info,
        preprocessing_notes, evidence_summary. Convert them to JSON strings.
        """
        # Top-level string fields
        for key in ("evidence_summary", "data_provenance_note"):
            val = data.get(key)
            if val is not None and not isinstance(val, str):
                data[key] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)

        # Dataset-level string fields
        for ds in data.get("datasets", []):
            if not isinstance(ds, dict):
                continue
            for key in ("size_info", "preprocessing_notes", "description", "source_url"):
                val = ds.get(key)
                if val is not None and not isinstance(val, str):
                    ds[key] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)

        # Baseline-level string fields
        for bl in data.get("baselines", []):
            if not isinstance(bl, dict):
                continue
            for key in ("description",):
                val = bl.get(key)
                if val is not None and not isinstance(val, str):
                    bl[key] = str(val)

        # Proposed method string fields
        pm = data.get("proposed_method")
        if isinstance(pm, dict):
            for key in ("name", "description", "architecture"):
                val = pm.get(key)
                if val is not None and not isinstance(val, str):
                    pm[key] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
            # key_components: LLM may return a comma-separated string instead of list
            kc = pm.get("key_components")
            if isinstance(kc, str):
                pm["key_components"] = [s.strip() for s in kc.split(",") if s.strip()]

        # ComputeRequirements string fields
        cr = data.get("compute_requirements")
        if isinstance(cr, dict):
            for key in ("gpu_type", "notes"):
                val = cr.get(key)
                if val is not None and not isinstance(val, str):
                    cr[key] = str(val)
            # Numeric fields — LLM may return "4" or "8.0" as strings
            # Schema fields: num_gpus (int), estimated_hours (float), memory_gb (float)
            for key in ("num_gpus",):
                val = cr.get(key)
                if isinstance(val, str):
                    try:
                        cr[key] = int(val) if val.isdigit() else int(float(val))
                    except (ValueError, TypeError):
                        pass
            for key in ("estimated_hours", "memory_gb"):
                val = cr.get(key)
                if isinstance(val, str):
                    try:
                        cr[key] = float(val)
                    except (ValueError, TypeError):
                        pass

        return data
