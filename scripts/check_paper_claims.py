#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []


def require_file(rel: str) -> Path:
    path = ROOT / rel
    if not path.is_file():
        errors.append(f"missing required file: {rel}")
    return path


def require_text(rel: str, needles: list[str]) -> None:
    path = require_file(rel)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    for needle in needles:
        if needle not in text:
            errors.append(f"{rel} missing implementation marker: {needle}")


def approx_equal(actual: float, expected: float, *, tol: float = 1e-6) -> bool:
    return abs(float(actual) - float(expected)) <= tol


# Pipeline and stage coverage.
require_text(
    "nanoresearch/pipeline/orchestrator.py",
    [
        "PipelineStage.IDEATION: IdeationAgent",
        "PipelineStage.PLANNING: PlanningAgent",
        "PipelineStage.EXPERIMENT: ExperimentAgent",
        "PipelineStage.WRITING: WritingAgent",
        "PipelineStage.REVIEW: ReviewAgent",
    ],
)

# Router, adaptive context, and selected memory/skill IDs.
require_text(
    "nanoresearch/router_policy.py",
    [
        "selected_memory_ids",
        "selected_skill_ids",
        "prompt_plan",
        "update_memory",
        "update_skill",
        "Run the trained SDPO router online",
        "class RouterUpdateManager",
    ],
)
require_text(
    "nanoresearch/agents/base.py",
    [
        "candidate_memory_count",
        "candidate_skill_count",
        "router_decision",
        "SDPO ROUTER PROMPT PLAN",
        "ROUTER EXPANDED STAGE PLAN",
    ],
)

# SDPO training/export path, not only inference.
require_text(
    "tools/export_router_sdpo_offpolicy.py",
    [
        "base_messages",
        "hindsight_messages",
        "target_text",
        "canonical_router_json",
        "validate_action_space",
    ],
)
require_text(
    "tools/train_router_sdpo_offpolicy.py",
    [
        "def compute_sdpo_step",
        "base_token_logprobs",
        "hindsight_token_logprobs",
        "advantage = (hindsight_token_logprobs - base_token_logprobs).detach()",
        "L_SDPO",
        "optimizer.step()",
    ],
)

# Memory / skill evolution.
require_text(
    "nanoresearch/evolution/memory.py",
    ["class MemoryStore", "def remember", "def retrieve_research", "def remember_research"],
)
require_text(
    "nanoresearch/evolution/skills.py",
    ["class SkillEvolutionStore", "def synthesize_nl_skill", "def review_skill_candidate", "MERGE_INTO_EXISTING", "def match_script_skills"],
)

# User profile and review isolation.
require_text(
    "nanoresearch/profile.py",
    ["render_profile_context", "render_router_hindsight_context", "router_hints"],
)
require_text(
    "nanoresearch/agents/review/__init__.py",
    ["REVIEW no longer participates in adaptive router", "self._adaptive_review_context = \"\""],
)

# Ideation evidence/novelty and planning correction loop.
require_text(
    "nanoresearch/agents/ideation.py",
    ["_extract_evidence", "_expand_via_citations", "_search_github_repos", "remember_promising_direction"],
)
require_text(
    "nanoresearch/agents/ideation_hypothesis.py",
    ["NOVELTY VERIFICATION", "generate_with_tools", "closest_existing_work"],
)
require_text(
    "nanoresearch/agents/planning.py",
    ["BLUEPRINT_REVIEW_SYSTEM_PROMPT", "_review_blueprint_with_llm", "should_retry", "blueprint_review.json"],
)

# Experiment, execution, analysis, writing evidence grounding.
require_text(
    "nanoresearch/agents/experiment/experiment_agent.py",
    ["Starting iterative experiment", "build_adaptive_context", "FeedbackAnalyzer", "remember_failed_direction"],
)
require_text(
    "nanoresearch/agents/execution/local_runner.py",
    ["_run_local_mode", "repair_launch_contract", "_run_local_quick_eval_loop"],
)
require_text(
    "nanoresearch/agents/writing/writing_agent.py",
    ["_build_grounding_packet", "paper_skeleton", "paper.tex", "learn_from_trace"],
)
require_text(
    "nanoresearch/agents/writing/grounding_tables.py",
    ["Do NOT fabricate proposed-method numbers", "_build_ablation_table_latex"],
)

# Data-integrity prompt guard for figure generation.
require_text(
    "nanoresearch/agents/figure_gen/evidence.py",
    ["Do NOT present them as verified experimental results", "Never disguise synthetic"],
)

if errors:
    for error in errors:
        print("ERROR:", error)
    raise SystemExit(1)
print("paper claim coverage checks passed")
