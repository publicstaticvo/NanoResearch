"""Stage gates with PROCEED / PIVOT / REJECT decisions and bounded rollback.

Inspired by AutoResearchClaw's gate-driven flow control.

Three gates fire between stages:

  * **SCREEN gate**  — after IDEATION, before PLANNING.
    Question: is the selected hypothesis worth turning into a plan?
  * **PLANNING gate** — after PLANNING, before EXPERIMENT / SETUP.
    Question: is the experiment blueprint executable and well-grounded?
  * **QUALITY gate** — after WRITING, before REVIEW.
    Question: is the draft good enough to spend review tokens on?

Each gate runs an LLM evaluator that returns a structured decision:

  * ``PROCEED`` — quality is acceptable, continue to the next stage.
  * ``PIVOT``   — significant gaps; re-run the **previous** stage with
    feedback so it can address them. The orchestrator hard-caps the
    total number of pivots across the entire pipeline at ``MAX_PIVOTS``
    (default 2). When the cap is hit, any further PIVOT is **forced to
    PROCEED** to break loops — borrowed from AutoResearchClaw's
    ``max_pivot=2`` anti-loop guard.
  * ``REJECT``  — fundamentally unrecoverable (e.g. nonsensical output);
    fail the pipeline.

The gate evaluator reuses the existing reflection LLM prompt
(quality_score / unmet_signals / suggestions) and adds a thin decision
layer on top, so we don't duplicate prompt engineering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# Hard cap on total pivots across the whole pipeline run.
# Borrowed from AutoResearchClaw's anti-loop guard.
MAX_PIVOTS = 2

# Quality threshold below which a stage is considered insufficient.
# Aligned with the existing ideation `coverage_score >= 8` convention.
GATE_QUALITY_THRESHOLD = 6  # below this -> PIVOT
GATE_REJECT_THRESHOLD = 1   # below this -> REJECT (irrecoverable)
# NOTE: lowered from 2 to 1 (2026-04-10, P1-E).
# Score=1 means "completely off-topic / empty / self-contradictory".
# Score=2 is bad but potentially recoverable via PIVOT — killing the
# pipeline at score=2 caused false REJECTs in --dev mode (§6.11.6).


# Stages where a gate fires AFTER the stage completes.
# Map: completed stage -> gate name (for logging / decision context).
# Stage values are uppercase strings (matching PipelineStage.value).
GATE_AFTER_STAGES: dict[str, str] = {
    "IDEATION": "SCREEN",
    "PLANNING": "PLANNING",
    "WRITING":  "QUALITY",
}


class GateDecision(str, Enum):
    """Outcome of a gate evaluation."""

    PROCEED = "PROCEED"
    PIVOT = "PIVOT"
    REJECT = "REJECT"


@dataclass
class GateResult:
    """Structured outcome of a gate check."""

    decision: GateDecision
    gate_name: str
    quality_score: int
    reason: str
    unmet_signals: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    forced: bool = False  # True if PIVOT was forced to PROCEED by max-pivot cap

    def to_feedback_dict(self) -> dict[str, Any]:
        """Serialize for cross-stage feedback (consumed by next agent run)."""
        return {
            "gate": self.gate_name,
            "decision": self.decision.value,
            "quality_score": self.quality_score,
            "reason": self.reason,
            "unmet_signals": list(self.unmet_signals),
            "suggestions": list(self.suggestions),
        }


_GATE_SYSTEM = """\
You are a research pipeline gate. A stage has just completed, and you must
decide whether the pipeline should PROCEED to the next stage, PIVOT (re-run
the previous stage with feedback), or REJECT (fundamentally unrecoverable
output).

Return JSON with exactly this shape:
{
  "quality_score": <integer 1-10>,
  "decision": "PROCEED" | "PIVOT" | "REJECT",
  "reason": "<one-sentence rationale>",
  "unmet_signals": ["<specific gap that justifies a non-PROCEED decision>", ...],
  "suggestions": ["<concrete fix the previous stage could apply on re-run>", ...]
}

Decision rules:
- PROCEED if the stage output is good enough to build the next stage on top of.
  Minor weaknesses are OK -- the next stage can compensate.
- PIVOT only if there are SPECIFIC, ACTIONABLE gaps that the previous stage
  could realistically fix in a re-run with the suggestions you provide.
  Do NOT PIVOT for vague concerns or stylistic preferences.
- REJECT only if the output is so broken that no re-run can save it
  (e.g. completely off-topic, empty, or self-contradictory at the core).

Be conservative: PROCEED is the default. PIVOT must be justified by concrete
unmet_signals, not by "could be better". REJECT is rare.

IMPORTANT -- dev mode: If the context mentions "Skipped stages (--dev mode)",
the pipeline is running WITHOUT real experiments. Placeholder language like
"pending results", "to be determined", or synthetic/illustrative data in
tables and figures is EXPECTED and must NOT be treated as a gap. Evaluate
only what the completed stages could realistically produce."""


async def evaluate_gate(
    *,
    stage_name: str,
    stage_result: dict[str, Any],
    accumulated: dict[str, Any],
    dispatcher,
    stage_config,
    pivot_count: int,
    max_pivots: int = MAX_PIVOTS,
    skip_stages: list[str] | None = None,
) -> GateResult:
    """Run the gate evaluator after ``stage_name`` completes.

    Returns a :class:`GateResult` describing the decision. If ``pivot_count``
    has already reached ``max_pivots``, any PIVOT decision is forcibly
    rewritten to PROCEED (with ``forced=True``) to break loops.
    """
    gate_name = GATE_AFTER_STAGES.get(stage_name.upper(), "")
    if not gate_name:
        # No gate for this stage -- treat as proceed.
        return GateResult(
            decision=GateDecision.PROCEED,
            gate_name="",
            quality_score=10,
            reason="no gate configured for this stage",
        )

    context = _build_gate_context(
        stage_name, gate_name, stage_result, accumulated,
        skip_stages=skip_stages or [],
    )

    # Default-PROCEED on any failure: gates must be fail-open so a
    # broken evaluator never blocks the pipeline.
    fallback = GateResult(
        decision=GateDecision.PROCEED,
        gate_name=gate_name,
        quality_score=10,
        reason="gate evaluator unavailable -- defaulting to PROCEED",
    )

    try:
        raw = await dispatcher.generate(
            stage_config, _GATE_SYSTEM, context, json_mode=True,
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return fallback
    except Exception as exc:
        logger.warning("[GATE %s] evaluator failed: %s", gate_name, exc)
        return fallback

    # Parse + clamp.
    score = payload.get("quality_score", 5)
    if isinstance(score, (int, float)):
        score = max(1, min(10, int(score)))
    else:
        score = 5

    raw_decision = str(payload.get("decision", "PROCEED")).upper().strip()
    if raw_decision not in {d.value for d in GateDecision}:
        # Decide from the score if the model didn't return a valid decision.
        if score < GATE_REJECT_THRESHOLD:
            raw_decision = GateDecision.REJECT.value
        elif score < GATE_QUALITY_THRESHOLD:
            raw_decision = GateDecision.PIVOT.value
        else:
            raw_decision = GateDecision.PROCEED.value

    decision = GateDecision(raw_decision)

    reason = str(payload.get("reason", "")).strip() or "(no reason)"
    unmet = [str(s) for s in payload.get("unmet_signals", []) if s]
    suggestions = [str(s) for s in payload.get("suggestions", []) if s]

    # Hard cap: if we've already pivoted MAX_PIVOTS times, force PROCEED.
    forced = False
    if decision == GateDecision.PIVOT and pivot_count >= max_pivots:
        logger.warning(
            "[GATE %s] PIVOT requested but pivot_count=%d >= max=%d -- "
            "forcing PROCEED to break loops",
            gate_name, pivot_count, max_pivots,
        )
        decision = GateDecision.PROCEED
        forced = True

    result = GateResult(
        decision=decision,
        gate_name=gate_name,
        quality_score=score,
        reason=reason,
        unmet_signals=unmet,
        suggestions=suggestions,
        forced=forced,
    )

    logger.info(
        "[GATE %s] decision=%s, score=%d/10, %d unmet, %d suggestions%s -- %s",
        gate_name, decision.value, score, len(unmet), len(suggestions),
        " (forced)" if forced else "",
        reason[:120],
    )
    return result


def _build_gate_context(
    stage_name: str,
    gate_name: str,
    stage_result: dict[str, Any],
    accumulated: dict[str, Any],
    *,
    skip_stages: list[str] | None = None,
) -> str:
    """Build a concise gate-evaluation context string for the LLM."""
    parts = [
        f"Gate: {gate_name}",
        f"Stage just completed: {stage_name}",
    ]

    topic = accumulated.get("topic", "")
    if topic:
        parts.append(f"Research topic: {topic}")

    # P1-E: inject --dev mode awareness so the evaluator doesn't penalise
    # placeholder language that is expected when experiment stages are skipped.
    if skip_stages:
        parts.append(f"Skipped stages (--dev mode): {', '.join(skip_stages)}")
        parts.append(
            "NOTE: Because experiment stages were skipped, the paper draft "
            "will contain placeholder language for results, tables, and "
            "ablation studies. This is EXPECTED and must NOT lower the score. "
            "Evaluate only the sections that could realistically be written "
            "without real experiment data (hypothesis, method design, writing "
            "quality, figure presence, structural completeness)."
        )

    if gate_name == "SCREEN":
        # IDEATION output: hypothesis viability check.
        ideation = stage_result.get("ideation_output", stage_result)
        if isinstance(ideation, dict):
            selected = ideation.get("selected_hypothesis", "")
            rationale = ideation.get("rationale", "")
            n_papers = len(ideation.get("papers", []) or [])
            n_evidence = len(
                (ideation.get("evidence", {}) or {}).get("extracted_metrics", []) or []
            )
            parts.append(f"Selected hypothesis: {selected}")
            parts.append(f"Selection rationale: {rationale[:400]}")
            parts.append(f"Literature collected: {n_papers} papers")
            parts.append(f"Quantitative evidence extracted: {n_evidence} metrics")

    elif gate_name == "PLANNING":
        bp = stage_result.get("experiment_blueprint", stage_result)
        if isinstance(bp, dict):
            method = (bp.get("proposed_method") or {}).get("name", "")
            metrics = [m.get("name", "") for m in (bp.get("metrics") or [])]
            datasets = [d.get("name", "") for d in (bp.get("datasets") or [])]
            baselines = [b.get("name", "") for b in (bp.get("baselines") or [])]
            ablations = bp.get("ablation_groups", []) or []
            parts.append(f"Method: {method}")
            parts.append(f"Metrics: {', '.join(metrics) or '(none)'}")
            parts.append(f"Datasets: {', '.join(datasets) or '(none)'}")
            parts.append(f"Baselines: {', '.join(baselines) or '(none)'}")
            parts.append(f"Ablation groups: {len(ablations)}")

    elif gate_name == "QUALITY":
        # WRITING output: skim the draft for placeholders / structural issues.
        writing = stage_result.get("writing_output", stage_result)
        if isinstance(writing, dict):
            consistency_issues = writing.get("consistency_issues", []) or []
            grounding = writing.get("grounding", {}) or {}
            parts.append(f"Draft completeness: {grounding.get('result_completeness', 'unknown')}")
            parts.append(f"Has real results: {grounding.get('has_real_results', False)}")
            parts.append(f"Consistency issues from writing stage: {len(consistency_issues)}")
            if consistency_issues:
                parts.append("Top consistency issues:")
                for issue in consistency_issues[:5]:
                    parts.append(f"  - {str(issue)[:200]}")

        # P1-E: inject objective paper artifacts so the gate doesn't rely
        # solely on the LLM judge's subjective score.
        workspace_dir = accumulated.get("_workspace_dir", "")
        if workspace_dir:
            _add_paper_artifact_info(parts, workspace_dir)

    return "\n".join(parts)


def _add_paper_artifact_info(parts: list[str], workspace_dir: str) -> None:
    """Append paper.tex / paper.pdf / figure stats to *parts* (best-effort)."""
    import os

    drafts = os.path.join(workspace_dir, "drafts")

    # paper.tex existence + line count
    tex_path = os.path.join(drafts, "paper.tex")
    if os.path.isfile(tex_path):
        try:
            with open(tex_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            n_lines = len(lines)
            # Count \includegraphics references
            fig_refs = sum(1 for l in lines if r"\includegraphics" in l)
            parts.append(f"paper.tex: {n_lines} lines, {fig_refs} \\includegraphics references")
        except OSError:
            pass
    else:
        parts.append("paper.tex: NOT FOUND")

    # paper.pdf existence + size
    pdf_path = os.path.join(drafts, "paper.pdf")
    if os.path.isfile(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        parts.append(f"paper.pdf: exists ({size_kb:.0f} KB)")
    else:
        parts.append("paper.pdf: NOT FOUND (compilation may have failed)")
