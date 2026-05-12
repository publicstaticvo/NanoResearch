"""Stage-level paper structure planner for WritingAgent."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ._types import GroundingPacket

logger = logging.getLogger(__name__)

WRITING_STAGE_PLANNER_SYSTEM = """You are the stage planner for NanoResearch's paper-writing stage.
Return JSON only. Do not write paper prose.
Your job is to convert router policy guidance, measured artifacts, figures, citations, and user/style constraints into a concrete paper-structure plan for the section writers.

Non-negotiable rules:
- Never fabricate missing experiment results, baselines, ablations, timings, or citations.
- Related Work should normally be 2-3 dense paragraphs for positioning, not a long survey.
- Method should receive more detail than Related Work and must expose the technical mechanism.
- Experiments must interleave setup, table/figure references, and explanatory text; never plan a bare stack of floats.
- Missing artifact categories should be omitted or scoped as limitations, not filled with fake values and not phrased as engineering-status placeholders.
- Figure/table placements must keep method figures in Method and result figures/tables in Experiments; never place floats in Conclusion or References.
"""


class _WritingStagePlannerMixin:
    """Generate and consume a structured plan for section-by-section writing."""

    async def _generate_writing_stage_plan(
        self,
        *,
        ideation: dict[str, Any],
        blueprint: dict[str, Any],
        grounding: GroundingPacket,
        figure_output: dict[str, Any],
        core_ctx: dict[str, Any],
        adaptive_context: str,
        section_list: list[tuple[str, str, str, list[str]]],
        template_format: str,
        is_survey: bool,
    ) -> dict[str, Any]:
        fallback = self._fallback_writing_stage_plan(
            ideation=ideation,
            blueprint=blueprint,
            grounding=grounding,
            figure_output=figure_output,
            section_list=section_list,
            template_format=template_format,
            is_survey=is_survey,
        )
        if is_survey:
            self.workspace.write_json("plans/paper_structure_plan.json", fallback)
            return fallback

        figures = (figure_output or {}).get("figures", {}) if isinstance(figure_output, dict) else {}
        figure_summary = []
        if isinstance(figures, dict):
            for key, data in list(figures.items())[:10]:
                if isinstance(data, dict):
                    figure_summary.append({
                        "key": key,
                        "type": data.get("fig_type") or data.get("figure_type") or data.get("kind"),
                        "caption": data.get("caption") or data.get("title") or "",
                        "backend": data.get("source_backend") or data.get("backend") or data.get("actual_model"),
                    })

        payload = {
            "topic": ideation.get("topic", core_ctx.get("topic", "")),
            "template_format": template_format,
            "router_and_adaptive_context": adaptive_context[-5000:] if adaptive_context else "",
            "method": blueprint.get("proposed_method", {}),
            "datasets": blueprint.get("datasets", [])[:4],
            "metrics": blueprint.get("metrics", [])[:8],
            "baselines": blueprint.get("baselines", [])[:8],
            "ablation_groups": blueprint.get("ablation_groups", [])[:8],
            "grounding": grounding.to_output_dict(),
            "final_metrics": grounding.final_metrics,
            "main_results_count": len(grounding.main_results),
            "ablation_results_count": len(grounding.ablation_results),
            "has_main_table": bool(grounding.main_table_latex),
            "has_ablation_table": bool(grounding.ablation_table_latex),
            "evidence_gaps": grounding.evidence_gaps[:8],
            "figure_summary": figure_summary,
            "available_sections": [heading for heading, _label, _instr, _figs in section_list],
            "citation_count": len(core_ctx.get("cite_keys", {}) or {}),
        }
        prompt = """Create a paper_structure_plan JSON object with these keys:
{
  "section_budget": {"Introduction": "...", "Related Work": "...", "Method": "...", "Experiments": "...", "Conclusion": "..."},
  "section_goals": {"Introduction": ["..."], "Related Work": ["..."], "Method": ["..."], "Experiments": ["..."], "Conclusion": ["..."]},
  "related_work_axes": ["2-3 positioning axes"],
  "method_subsections": ["subsection titles or technical units"],
  "experiment_storyline": ["ordered prose/table/figure moves"],
  "finding_units": [{"claim": "artifact-backed finding", "evidence": ["table/figure keys"], "interpretation": "why it matters", "scope_limit": "what this run does not establish"}],
  "layout_constraints": [{"section": "Experiments", "rule": "one float at a time with prose before the next float"}],
  "figure_table_placement": [{"artifact": "fig/table key", "target_section": "Method or Experiments", "near_text_goal": "..."}],
  "required_claims": ["claims supported by current artifacts"],
  "forbidden_claims": ["claims not supported by current artifacts"],
  "review_checklist": ["checks the reviewer/revision stage must enforce"]
}

Return only JSON. Keep the plan specific to the artifacts. Do not include fake numbers.

Context JSON:
""" + json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            plan = await self.generate_json(WRITING_STAGE_PLANNER_SYSTEM, prompt)
            if not isinstance(plan, dict):
                raise TypeError(f"expected object, got {type(plan).__name__}")
            plan = self._normalize_writing_stage_plan(plan, fallback)
        except Exception as exc:
            logger.warning("Writing stage planning failed; using deterministic fallback: %s", exc)
            plan = fallback
        self.workspace.write_json("plans/paper_structure_plan.json", plan)
        try:
            self.workspace.register_artifact(
                "paper_structure_plan",
                self.workspace.path / "plans" / "paper_structure_plan.json",
                self.stage,
            )
        except Exception:
            pass
        return plan

    @staticmethod
    def _fallback_writing_stage_plan(
        *,
        ideation: dict[str, Any],
        blueprint: dict[str, Any],
        grounding: GroundingPacket,
        figure_output: dict[str, Any],
        section_list: list[tuple[str, str, str, list[str]]],
        template_format: str,
        is_survey: bool,
    ) -> dict[str, Any]:
        method = blueprint.get("proposed_method", {}) if isinstance(blueprint, dict) else {}
        method_name = method.get("name") if isinstance(method, dict) else "the proposed method"
        components = method.get("key_components", []) if isinstance(method, dict) else []
        if not isinstance(components, list):
            components = []
        method_units = [str(c) for c in components[:4] if str(c).strip()]
        if not method_units:
            method_units = ["Problem formulation", "Model design", "Training objective", "Complexity and deployment cost"]
        figures = (figure_output or {}).get("figures", {}) if isinstance(figure_output, dict) else {}
        placements: list[dict[str, str]] = []
        if isinstance(figures, dict):
            for key in figures:
                key_l = str(key).lower()
                target = "Method" if any(w in key_l for w in ("arch", "method", "framework", "overview", "pipeline")) else "Experiments"
                placements.append({
                    "artifact": str(key),
                    "target_section": target,
                    "near_text_goal": "Introduce the figure with artifact-grounded prose and interpret only what the figure supports.",
                })
        result_claim = "Report measured local results only from available run artifacts."
        if grounding.final_metrics:
            metric_bits = [f"{k}={v}" for k, v in list(grounding.final_metrics.items())[:3]]
            result_claim = "Report measured final metrics: " + ", ".join(metric_bits)
        if is_survey:
            related_budget = "3-5 paragraphs, survey-style synthesis"
            method_budget = "as required by survey mode"
            exp_budget = "omit experiments unless survey mode defines analysis"
        else:
            related_budget = "2-3 dense paragraphs; positioning only, not a long survey"
            method_budget = "4-5 subsections or 7-10 paragraphs; more detailed than Related Work"
            exp_budget = "setup, main table, result discussion, ablation if measured, complexity/optimization if measured; interleave prose and floats"
        return {
            "topic": ideation.get("topic", ""),
            "method_name": method_name or "the proposed method",
            "template_format": template_format,
            "section_budget": {
                "Introduction": "4-5 paragraphs with early contributions",
                "Related Work": related_budget,
                "Method": method_budget,
                "Experiments": exp_budget,
                "Conclusion": "2 concise paragraphs; no new results or citations",
            },
            "section_goals": {
                "Introduction": ["Motivate the problem concretely", "State the gap and contributions early", "Do not overclaim beyond artifacts"],
                "Related Work": ["Use 2-3 thematic positioning axes", "Acknowledge prior work fairly", "End by differentiating the proposed method"],
                "Method": ["Define notation", "Explain each core mechanism", "Use compact equations that fit page width", "Discuss complexity only when supported"],
                "Experiments": ["State protocol before results", "Place each table/figure near explanatory prose", "Use only measured artifacts for numbers", "Scope missing evidence academically"],
                "Conclusion": ["Summarize supported findings", "State limitations without engineering placeholders", "Avoid new claims"],
            },
            "related_work_axes": ["task and dataset context", "closest methodological baselines", "gap addressed by the proposed method"],
            "method_subsections": method_units,
            "experiment_storyline": ["experimental protocol", "main measured comparison", "ablation evidence when available", "optimization and complexity diagnostics when available", "scope of evidence"],
            "figure_table_placement": placements,
            "finding_units": [
                {
                    "claim": result_claim,
                    "evidence": ["tab:main_results"],
                    "interpretation": "Explain the main measured comparison as an artifact-grounded finding rather than a float description.",
                    "scope_limit": "Do not generalize beyond the executed artifacts and split unless repeated-run evidence exists.",
                }
            ],
            "layout_constraints": [
                {"section": "Experiments", "rule": "Interleave prose, table, prose, figure; never emit a bare stack of floats."},
                {"section": "Method", "rule": "Use equations selectively and surround each displayed equation with narrative explanation."},
                {"section": "Conclusion", "rule": "No figures or tables."},
            ],
            "required_claims": [result_claim],
            "forbidden_claims": [
                "Do not claim state of the art without comparable published evidence.",
                "Do not invent missing baseline, ablation, runtime, or complexity values.",
                "Do not describe absent artifacts as if they were measured.",
            ],
            "review_checklist": [
                "Related Work is not longer or more detailed than Method.",
                "Method contains concrete technical mechanisms and compact equations.",
                "Experiments contain explanatory prose around every result table or figure.",
                "No result figure/table is placed in Conclusion or References.",
                "All numeric claims are grounded in artifacts or explicitly published evidence.",
            ],
        }

    @staticmethod
    def _normalize_writing_stage_plan(plan: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(fallback)
        for key in (
            "section_budget", "section_goals", "related_work_axes", "method_subsections",
            "experiment_storyline", "figure_table_placement", "required_claims",
            "forbidden_claims", "review_checklist", "finding_units", "layout_constraints",
        ):
            value = plan.get(key)
            if value:
                normalized[key] = value
        for key, value in plan.items():
            if key not in normalized and value not in (None, "", [], {}):
                normalized[key] = value
        return normalized

    @staticmethod
    def _writing_plan_section_block(plan: dict[str, Any], heading: str) -> str:
        if not isinstance(plan, dict) or not plan:
            return ""
        budget = (plan.get("section_budget") or {}).get(heading, "") if isinstance(plan.get("section_budget"), dict) else ""
        goals = (plan.get("section_goals") or {}).get(heading, []) if isinstance(plan.get("section_goals"), dict) else []
        if isinstance(goals, str):
            goals = [goals]
        lines = ["=== PAPER STRUCTURE PLAN FOR THIS SECTION ==="]
        if budget:
            lines.append(f"Budget: {budget}")
        if goals:
            lines.append("Section goals:")
            lines.extend(f"- {g}" for g in goals[:8])
        if heading == "Related Work" and plan.get("related_work_axes"):
            lines.append("Related-work axes:")
            lines.extend(f"- {x}" for x in plan.get("related_work_axes", [])[:5])
        if heading == "Method" and plan.get("method_subsections"):
            lines.append("Required method units/subsections:")
            lines.extend(f"- {x}" for x in plan.get("method_subsections", [])[:8])
        if heading == "Experiments" and plan.get("experiment_storyline"):
            lines.append("Experiment storyline order:")
            lines.extend(f"- {x}" for x in plan.get("experiment_storyline", [])[:8])
        if heading == "Experiments" and plan.get("finding_units"):
            lines.append("Artifact-backed finding units:")
            for item in plan.get("finding_units", [])[:5]:
                if isinstance(item, dict):
                    lines.append(f"- claim: {item.get('claim', '')}; evidence: {item.get('evidence', '')}; scope: {item.get('scope_limit', '')}")
        if plan.get("layout_constraints"):
            constraints = [x for x in plan.get("layout_constraints", []) if not isinstance(x, dict) or heading.lower() in str(x.get('section', heading)).lower()]
            if constraints:
                lines.append("Layout constraints:")
                for item in constraints[:5]:
                    lines.append(f"- {item.get('rule', item) if isinstance(item, dict) else item}")
        placements = []
        for item in plan.get("figure_table_placement", []) or []:
            if not isinstance(item, dict):
                continue
            target = str(item.get("target_section") or "").lower()
            if heading.lower() in target or (heading == "Experiments" and "result" in target):
                placements.append(item)
        if placements:
            lines.append("Figure/table placement guidance:")
            for item in placements[:8]:
                lines.append(f"- {item.get('artifact', '')}: {item.get('near_text_goal', '')}")
        if plan.get("required_claims"):
            lines.append("Supported claims to preserve:")
            lines.extend(f"- {x}" for x in plan.get("required_claims", [])[:6])
        if plan.get("forbidden_claims"):
            lines.append("Forbidden unsupported claims:")
            lines.extend(f"- {x}" for x in plan.get("forbidden_claims", [])[:8])
        lines.append("=== END PAPER STRUCTURE PLAN ===")
        return "\n".join(lines)

    @staticmethod
    def _augment_section_instructions_with_plan(instructions: str, heading: str, plan: dict[str, Any]) -> str:
        block = _WritingStagePlannerMixin._writing_plan_section_block(plan, heading)
        if not block:
            return instructions
        section_specific = ""
        if heading == "Related Work":
            section_specific = (
                "\nPLAN OVERRIDE: Write Related Work as 2-3 dense paragraphs unless the plan explicitly says otherwise. "
                "Do not expand into a broad survey; use citations to position this paper against closest prior work."
            )
        elif heading == "Method":
            section_specific = (
                "\nPLAN OVERRIDE: Method must be technically substantive and should not be shorter than Related Work. "
                "Use subsections from the plan when appropriate, and keep display equations compact enough for a two-column paper."
            )
        elif heading == "Experiments":
            section_specific = (
                "\nPLAN OVERRIDE: Interleave paragraphs, tables, and figures. Every result float needs nearby explanatory prose. "
                "Do not write engineering-status placeholders for missing artifacts."
            )
        return f"{instructions}\n\n{block}{section_specific}"

    @staticmethod
    def _audit_paper_structure_against_plan(latex_content: str, plan: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        if not latex_content or not isinstance(plan, dict):
            return issues

        def section_text(name: str) -> str:
            pattern = re.compile(
                rf"\\section\*?\{{{re.escape(name)}\}}(.*?)(?=\\section\*?\{{|\\bibliography|\\end\{{document\}})",
                re.DOTALL,
            )
            match = pattern.search(latex_content)
            return match.group(1) if match else ""

        related = section_text("Related Work")
        method = section_text("Method")
        experiments = section_text("Experiments")
        if related:
            related_plain = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", "", related, flags=re.DOTALL)
            paragraph_count = len([p for p in re.split(r"\n\s*\n", related_plain) if len(p.strip()) > 120])
            if paragraph_count > 4:
                issues.append(f"Related Work appears long ({paragraph_count} dense paragraphs); plan expects compact positioning.")
        if related and method:
            related_words = len(re.findall(r"\b\w+\b", re.sub(r"\\cite\w*\{[^}]+\}", "", related)))
            method_words = len(re.findall(r"\b\w+\b", method))
            if method_words < max(350, int(0.85 * related_words)):
                issues.append("Method is too short relative to Related Work under the paper structure plan.")
        if method and len(re.findall(r"\\subsection\{", method)) < 2:
            issues.append("Method has fewer than two subsections; plan expects explicit technical units.")
        if experiments:
            float_count = len(re.findall(r"\\begin\{(?:figure|table)\*?\}", experiments))
            prose_words = len(re.findall(r"\b\w+\b", re.sub(r"\\begin\{(?:figure|table)\*?\}.*?\\end\{(?:figure|table)\*?\}", "", experiments, flags=re.DOTALL)))
            if float_count >= 2 and prose_words < 220 * float_count:
                issues.append("Experiments prose is thin relative to the number of result tables/figures.")
        for bad_section in ("Conclusion", "References"):
            text = section_text(bad_section)
            if re.search(r"\\begin\{figure\*?\}|\\begin\{table\*?\}", text):
                issues.append(f"{bad_section} contains figure/table floats; plan forbids result floats there.")
        return issues
