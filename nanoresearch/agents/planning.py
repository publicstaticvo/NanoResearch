"""Planning agent — generates experiment blueprint from ideation output."""

from __future__ import annotations

import json
import logging
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.evolution.memory import MemoryType
from nanoresearch.schemas.experiment import ExperimentBlueprint
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Configurable limits
MAX_PAPERS_IN_SUMMARY = 15

PLANNING_SYSTEM_PROMPT = """You are a research experiment planner. Your task is to design a rigorous experiment plan based on the selected hypothesis from the ideation phase.

CRITICAL RULES — evidence grounding:
- NEVER invent baseline performance numbers. Use ONLY numbers from the published evidence provided.
- If a metric value is not available in the evidence, set it to null — do NOT guess or write "N/A".
- Prefer baseline methods whose published metric values are present in the retrieved literature evidence; avoid unverifiable baselines when comparable verified baselines exist.
- For the proposed method, describe expected improvements as a "projected improvement range" (e.g. "5-10% improvement over best baseline"), NOT as exact numbers.
- Every baseline number MUST have a "performance_provenance" entry citing which paper it came from.
- Mark proposed-method values as projected: set "is_projected" to true for each metric.

Design experiments that are:
- Reproducible with standard ML frameworks
- Include proper baselines for comparison (using REAL published numbers)
- Have clear evaluation metrics
- Include ablation studies to validate each component

Always respond in valid JSON format."""


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
        selected_hyp = ideation_data.get("selected_hypothesis", "")
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

        # Find the selected hypothesis details
        hyp_detail = ""
        hypotheses = ideation_data.get("hypotheses", [])
        for h in hypotheses:
            if h.get("hypothesis_id") == selected_hyp:
                hyp_detail = json.dumps(h, indent=2)
                break
        if not hyp_detail and hypotheses:
            logger.warning(
                "Selected hypothesis %r not found among %d hypotheses, using first",
                selected_hyp, len(hypotheses),
            )
            hyp_detail = json.dumps(hypotheses[0], indent=2)
            selected_hyp = hypotheses[0].get("hypothesis_id", selected_hyp)

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

Selected Hypothesis: {selected_hyp}
{hyp_detail}

Selection Rationale: {rationale}

Identified Research Gaps:
{gaps_text}

Key Related Papers:
{papers_summary}

{evidence_block}

Design a comprehensive experiment blueprint as JSON with:
1. "title": Experiment title
2. "hypothesis_ref": "{selected_hyp}"
3. "datasets": Array of datasets, each with:
   - "name", "description", "source_url", "size_info", "preprocessing_notes"
4. "baselines": Array of baseline methods, each with:
   - "name", "description", "reference_paper_id"
   - "expected_performance": dict of metric→value (use ONLY values from the evidence above, or null)
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
11. "experiment_matrix": Array of machine-checkable run specs. Include at minimum:
   - one required proposed run with role="proposed" and output_group="main_results"
   - at least two required measured baseline runs with role="baseline" and output_group="main_results"
   - at least two required ablation runs with role="ablation" and output_group="ablation_results"
   - one optimization/history run with role="optimization" and output_group="optimization_history"
   - one complexity run with role="complexity" and output_group="complexity_metrics"
   Each run spec MUST include: "run_id", "role", "method", "dataset", "metrics",
   "required", "output_group", "expected_artifacts", "failure_policy", and "config".
12. "required_artifacts": include the core artifacts needed to reproduce the run contract.
   Use canonical names such as "configs/experiment_matrix.json", "results/metrics.json",
   "results/run_manifest.json", and "results/final_metrics.json". Add diagnostic artifacts
   only when the topic or method genuinely produces them.
13. "minimum_success_criteria": object with "min_measured_baselines"=2,
   "min_ablation_runs"=2, "require_proposed"=true, and "required_metrics".
   Set "require_complexity" / "require_optimization_history" only when the topic actually
   defines those diagnostics as part of the experiment contract.

IMPORTANT: Use ONLY the numbers from the PUBLISHED QUANTITATIVE EVIDENCE section.
Choose baselines with reported OpenAlex/literature evidence when possible, but the experiment_matrix
means these baselines must be measured by generated code, not copied from papers.
If evidence is missing for a baseline-metric pair, use null and explain the gap in evidence_summary.

Return ONLY valid JSON."""

        # Retry loop: if blueprint validation fails, retry with error feedback
        max_planning_retries = 2
        last_validation_error = ""
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
            result = self._ensure_experiment_contract(result)

            try:
                blueprint = ExperimentBlueprint.model_validate(result)
                break  # Validation succeeded
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
        output_path = self.workspace.write_json(
            "plans/experiment_blueprint.json",
            blueprint.model_dump(mode="json"),
        )
        self.workspace.register_artifact(
            "experiment_blueprint", output_path, self.stage
        )
        primary_metrics = [m.name for m in blueprint.metrics if m.primary] or [m.name for m in blueprint.metrics[:3]]
        primary_metrics_text = ", ".join(primary_metrics)
        dataset_names = ", ".join(ds.name for ds in blueprint.datasets[:3])
        method_name = self._method_name(blueprint.proposed_method)
        self.remember_context(
            MemoryType.PROJECT_CONTEXT,
            f"Planning blueprint for {topic}: method={method_name}, datasets={dataset_names}, primary metrics={primary_metrics_text}",
            importance=0.78,
            tags=[topic, method_name, "planning"],
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
            planning_output=blueprint.model_dump(mode="json"),
            artifact_path="logs/promising_direction_summary_planning.json",
            source_stage="planning",
            source="experiment_blueprint",
        )
        planning_trace = (
            f"Blueprint for {topic}: method={method_name}; "
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
            tags=[topic, method_name, "planning", "blueprint"],
            confidence=0.68,
        )
        logger.info("[%s] Blueprint generated: %s", self.stage.value, blueprint.title)
        return blueprint.model_dump(mode="json")

    @staticmethod
    def _method_name(proposed_method: Any) -> str:
        """Return a stable method name from the schema's structured method dict."""
        if isinstance(proposed_method, dict):
            value = proposed_method.get("name") or proposed_method.get("method_name")
            return str(value).strip() if value else "proposed_method"
        value = getattr(proposed_method, "name", None)
        return str(value).strip() if value else "proposed_method"

    @staticmethod
    def _build_evidence_block(ideation_data: dict) -> str:
        """Format extracted evidence for inclusion in the planning prompt."""
        evidence = ideation_data.get("evidence", {})
        metrics = evidence.get("extracted_metrics", [])

        if not metrics:
            return (
                "=== PUBLISHED QUANTITATIVE EVIDENCE ===\n"
                "No quantitative evidence was extracted from the literature review.\n"
                "Do not invent, recall, or estimate baseline expected_performance values.\n"
                "Set missing baseline expected_performance values to null and keep\n"
                "published baseline context separate from measured experiment results.\n"
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

    @staticmethod
    def _ensure_experiment_contract(data: dict) -> dict:
        """Add a conservative execution contract if the planner omitted one."""
        if not isinstance(data, dict):
            return data

        metrics = [
            str(m.get("name") or "").strip()
            for m in data.get("metrics", [])
            if isinstance(m, dict) and str(m.get("name") or "").strip()
        ]
        if not metrics:
            metrics = ["accuracy"]
        datasets = [
            str(d.get("name") or "").strip()
            for d in data.get("datasets", [])
            if isinstance(d, dict) and str(d.get("name") or "").strip()
        ]
        dataset = datasets[0] if datasets else "dataset"
        proposed = data.get("proposed_method") if isinstance(data.get("proposed_method"), dict) else {}
        proposed_name = str(proposed.get("name") or "proposed_method").strip()

        matrix = data.get("experiment_matrix")
        if not isinstance(matrix, list):
            matrix = []
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in matrix:
            if not isinstance(raw, dict):
                continue
            run_id = str(raw.get("run_id") or "").strip()
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            raw.setdefault("metrics", metrics)
            raw.setdefault("dataset", dataset)
            raw.setdefault("required", True)
            raw.setdefault("expected_artifacts", ["results/metrics.json", "results/run_manifest.json"])
            raw.setdefault("failure_policy", "debug_then_degrade")
            raw.setdefault("config", {})
            normalized.append(raw)

        def add(run_id: str, role: str, method: str, output_group: str, config: dict[str, Any] | None = None) -> None:
            if run_id in seen:
                return
            seen.add(run_id)
            normalized.append({
                "run_id": run_id,
                "role": role,
                "method": method,
                "dataset": dataset,
                "metrics": metrics,
                "required": True,
                "output_group": output_group,
                "expected_artifacts": ["results/metrics.json", "results/run_manifest.json"],
                "failure_policy": "debug_then_degrade",
                "config": config or {},
            })

        roles = {str(r.get("role") or "").lower() for r in normalized if isinstance(r, dict)}
        if "proposed" not in roles:
            add("proposed_full", "proposed", proposed_name, "main_results", {"variant": "full"})

        baseline_specs = [b for b in data.get("baselines", []) if isinstance(b, dict)]
        for idx, baseline in enumerate(baseline_specs[:2], start=1):
            name = str(baseline.get("name") or f"baseline_{idx}").strip()
            add(f"baseline_{idx}_{PlanningAgent._slug(name)}", "baseline", name, "main_results", {"baseline_index": idx})
        while sum(1 for r in normalized if str(r.get("role") or "").lower() == "baseline") < 2:
            idx = sum(1 for r in normalized if str(r.get("role") or "").lower() == "baseline") + 1
            add(f"baseline_{idx}_simple", "baseline", f"SimpleBaseline{idx}", "main_results", {"baseline_index": idx})

        ablation_variants: list[tuple[str, dict[str, Any]]] = []
        for group in data.get("ablation_groups", []):
            if not isinstance(group, dict):
                continue
            for variant in group.get("variants", []) or []:
                if isinstance(variant, dict):
                    name = str(variant.get("name") or variant.get("variant_name") or variant.get("description") or "ablation").strip()
                    ablation_variants.append((name, variant))
        for idx, (name, variant) in enumerate(ablation_variants[:2], start=1):
            add(f"ablation_{idx}_{PlanningAgent._slug(name)}", "ablation", name, "ablation_results", variant)
        while sum(1 for r in normalized if str(r.get("role") or "").lower() == "ablation") < 2:
            idx = sum(1 for r in normalized if str(r.get("role") or "").lower() == "ablation") + 1
            add(f"ablation_{idx}_component", "ablation", f"AblationVariant{idx}", "ablation_results", {"ablation_index": idx})

        if not any(str(r.get("role") or "").lower() == "optimization" for r in normalized):
            add("optimization_history", "optimization", proposed_name, "optimization_history", {"track": "hyperparameter_search"})
        if not any(str(r.get("role") or "").lower() == "complexity" for r in normalized):
            add("complexity_profile", "complexity", proposed_name, "complexity_metrics", {"track": "runtime_parameter_profile"})

        data["experiment_matrix"] = normalized
        data.setdefault("required_artifacts", [
            "configs/experiment_matrix.json",
            "results/metrics.json",
            "results/run_manifest.json",
            "results/final_metrics.json",
        ])
        criteria = data.get("minimum_success_criteria")
        if not isinstance(criteria, dict):
            criteria = {}
        criteria.setdefault("min_measured_baselines", 2)
        criteria.setdefault("min_ablation_runs", 2)
        criteria.setdefault("require_proposed", True)
        criteria.setdefault("require_complexity", False)
        criteria.setdefault("require_optimization_history", False)
        criteria.setdefault("required_metrics", metrics)
        data["minimum_success_criteria"] = criteria
        return data

    @staticmethod
    def _slug(value: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
        return slug[:40] or "run"

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
            perf = bl.get("expected_performance")
            if isinstance(perf, dict):
                cleaned_perf = {}
                for metric_name, metric_value in perf.items():
                    if isinstance(metric_value, str) and metric_value.strip().lower() in {"n/a", "na", "not available", "unknown", "--", ""}:
                        cleaned_perf[metric_name] = None
                    else:
                        cleaned_perf[metric_name] = metric_value
                bl["expected_performance"] = cleaned_perf
            provenance = bl.get("performance_provenance")
            if isinstance(provenance, dict):
                bl["performance_provenance"] = {
                    str(k): str(v) for k, v in provenance.items()
                    if v is not None and str(v).strip().lower() not in {"n/a", "na", "unknown", ""}
                }

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
