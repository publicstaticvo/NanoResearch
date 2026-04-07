"""User persona/profile management for NanoResearch."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ARCHETYPE = "resource_constrained_pragmatic"

ARCHETYPE_SEEDS: dict[str, dict[str, Any]] = {
    "nlp_conference": {
        "archetype_seed": "nlp_conference",
        "research_profile": {
            "domain": "NLP",
            "method_preference": "Prefer simple and interpretable methods with strong baselines.",
            "risk_preference": "moderate",
            "baseline_ablation_strictness": "high",
        },
        "resource_profile": {
            "gpu_budget": "1xA100 80GB",
            "wall_clock_budget": "3 days",
            "feasibility_bias": "Prefer methods feasible on small-to-medium compute budgets.",
        },
        "writing_profile": {
            "tone": "restrained academic",
            "claim_strength": "moderate",
            "section_organization": "Conference-style, direct and contribution-focused.",
        },
        "publication_profile": {
            "venue_style": "NeurIPS/ICLR conference",
            "latex_template_preference": "conference_template",
            "figure_style": "clean benchmark plots",
            "caption_style": "compact but informative",
        },
        "interaction_profile": {
            "priority_feedback": "Weak baselines, missing ablations, or overclaiming novelty.",
            "unacceptable_errors": "Ignoring compute limits or skipping rigorous comparisons.",
        },
    },
    "cv_visual_conference": {
        "archetype_seed": "cv_visual_conference",
        "research_profile": {
            "domain": "Computer Vision",
            "method_preference": "Accept visually compelling ideas if they remain benchmark-grounded.",
            "risk_preference": "moderate-high",
            "baseline_ablation_strictness": "high",
        },
        "resource_profile": {
            "gpu_budget": "2xA100 80GB",
            "wall_clock_budget": "4 days",
            "feasibility_bias": "Open to moderate compute if figures and evaluation are strong.",
        },
        "writing_profile": {
            "tone": "technical and direct",
            "claim_strength": "moderate",
            "section_organization": "Conference-style with strong experiment storytelling.",
        },
        "publication_profile": {
            "venue_style": "CV conference",
            "latex_template_preference": "conference_template",
            "figure_style": "qualitative + benchmark plotting",
            "caption_style": "compact but informative",
        },
        "interaction_profile": {
            "priority_feedback": "Weak visual evidence, unclear figure design, or incomplete benchmark comparisons.",
            "unacceptable_errors": "Poor figure readability or unsupported SOTA claims.",
        },
    },
    "ai4science_journal": {
        "archetype_seed": "ai4science_journal",
        "research_profile": {
            "domain": "AI for Science",
            "method_preference": "Prefer conservative, evidence-first methods with scientific plausibility.",
            "risk_preference": "low-moderate",
            "baseline_ablation_strictness": "high",
        },
        "resource_profile": {
            "gpu_budget": "1xA100 80GB",
            "wall_clock_budget": "5 days",
            "feasibility_bias": "Prefer reproducible designs with explicit assumptions and controlled compute.",
        },
        "writing_profile": {
            "tone": "highly restrained",
            "claim_strength": "conservative",
            "section_organization": "Journal-style with dense evidence and careful limitations.",
        },
        "publication_profile": {
            "venue_style": "Nature/Springer journal",
            "latex_template_preference": "nature_springer",
            "figure_style": "composite scientific figure",
            "caption_style": "self-contained dense",
        },
        "interaction_profile": {
            "priority_feedback": "Scientific plausibility, evidence gaps, or overstated generality.",
            "unacceptable_errors": "Overclaiming biological/physical conclusions or under-specifying data provenance.",
        },
    },
    "resource_constrained_pragmatic": {
        "archetype_seed": "resource_constrained_pragmatic",
        "research_profile": {
            "domain": "General CS/ML",
            "method_preference": "Prefer low-risk incremental methods with minimal engineering overhead.",
            "risk_preference": "low",
            "baseline_ablation_strictness": "high",
        },
        "resource_profile": {
            "gpu_budget": "1xA100 80GB",
            "wall_clock_budget": "24-48 hours",
            "feasibility_bias": "Strongly prefer small-budget, reproducible experiments.",
        },
        "writing_profile": {
            "tone": "concise academic",
            "claim_strength": "conservative",
            "section_organization": "Direct conference-style structure.",
        },
        "publication_profile": {
            "venue_style": "conference",
            "latex_template_preference": "conference_template",
            "figure_style": "standard benchmark plots",
            "caption_style": "compact",
        },
        "interaction_profile": {
            "priority_feedback": "Anything that increases cost without clear gain.",
            "unacceptable_errors": "Plans or drafts that ignore resource constraints.",
        },
    },
    "high_novelty_exploratory": {
        "archetype_seed": "high_novelty_exploratory",
        "research_profile": {
            "domain": "General CS/ML",
            "method_preference": "Prefer bold ideas if the evaluation plan remains falsifiable.",
            "risk_preference": "high",
            "baseline_ablation_strictness": "medium-high",
        },
        "resource_profile": {
            "gpu_budget": "2xA100 80GB",
            "wall_clock_budget": "5 days",
            "feasibility_bias": "Accept higher-risk proposals if novelty is clear and constraints are explicit.",
        },
        "writing_profile": {
            "tone": "confident but disciplined",
            "claim_strength": "moderate-high",
            "section_organization": "Concept-first with clear positioning.",
        },
        "publication_profile": {
            "venue_style": "top-tier conference",
            "latex_template_preference": "conference_template",
            "figure_style": "clear conceptual + benchmark figures",
            "caption_style": "informative",
        },
        "interaction_profile": {
            "priority_feedback": "Weak novelty framing or muddled contribution statements.",
            "unacceptable_errors": "Novelty claims unsupported by experiment or literature positioning.",
        },
    },
}


def get_nanoresearch_home() -> Path:
    configured = Path(
        (
            __import__("os").environ.get("NANORESEARCH_HOME")
            or str(Path.home() / ".nanoresearch")
        )
    )
    return configured


def get_profile_dir() -> Path:
    return get_nanoresearch_home() / "profile"


def get_profile_json_path() -> Path:
    return get_profile_dir() / "profile.json"


def get_profile_markdown_path() -> Path:
    return get_profile_dir() / "PROFILE.md"


def build_profile_seed(archetype: str) -> dict[str, Any]:
    archetype = archetype if archetype in ARCHETYPE_SEEDS else DEFAULT_ARCHETYPE
    seed = deepcopy(ARCHETYPE_SEEDS[archetype])
    now = datetime.now(timezone.utc).isoformat()
    seed.update(
        {
            "profile_id": f"profile-{archetype}",
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "router_hints": build_router_hints(seed),
        }
    )
    return seed


def build_router_hints(profile: dict[str, Any]) -> dict[str, Any]:
    publication = profile.get("publication_profile", {})
    writing = profile.get("writing_profile", {})
    resource = profile.get("resource_profile", {})
    return {
        "prefer_profile_over_sparse_memory": True,
        "writing_prompt_focus": {
            "tone": writing.get("tone", ""),
            "claim_strength": writing.get("claim_strength", ""),
            "venue_style": publication.get("venue_style", ""),
            "figure_style": publication.get("figure_style", ""),
            "caption_style": publication.get("caption_style", ""),
        },
        "planning_prompt_focus": {
            "resource_budget": resource.get("gpu_budget", ""),
            "feasibility_bias": resource.get("feasibility_bias", ""),
        },
    }


def load_user_profile() -> dict[str, Any] | None:
    path = get_profile_json_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_profile_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# NanoResearch User Profile",
        "",
        f"- `profile_id`: {profile.get('profile_id', '')}",
        f"- `archetype_seed`: {profile.get('archetype_seed', '')}",
        "",
        "## Summary",
        "",
        f"- Research direction: {profile.get('research_profile', {}).get('domain', '')}",
        f"- Method preference: {profile.get('research_profile', {}).get('method_preference', '')}",
        f"- Resource budget: {profile.get('resource_profile', {}).get('gpu_budget', '')}, {profile.get('resource_profile', {}).get('wall_clock_budget', '')}",
        f"- Writing tone: {profile.get('writing_profile', {}).get('tone', '')}",
        f"- Venue/style preference: {profile.get('publication_profile', {}).get('venue_style', '')}",
        f"- Template preference: {profile.get('publication_profile', {}).get('latex_template_preference', '')}",
        f"- Figure style: {profile.get('publication_profile', {}).get('figure_style', '')}",
        f"- Caption style: {profile.get('publication_profile', {}).get('caption_style', '')}",
        "",
        "## Feedback Priorities",
        "",
        f"- Most important feedback: {profile.get('interaction_profile', {}).get('priority_feedback', '')}",
        f"- Unacceptable mistakes: {profile.get('interaction_profile', {}).get('unacceptable_errors', '')}",
        "",
        "## Recommended Router Defaults",
        "",
        f"- Planning prompt focus: {profile.get('router_hints', {}).get('planning_prompt_focus', {})}",
        f"- Writing prompt focus: {profile.get('router_hints', {}).get('writing_prompt_focus', {})}",
    ]
    return "\n".join(lines) + "\n"


def save_user_profile(profile: dict[str, Any]) -> None:
    profile = deepcopy(profile)
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    profile["router_hints"] = build_router_hints(profile)
    profile.setdefault("profile_id", f"profile-{profile.get('archetype_seed', DEFAULT_ARCHETYPE)}")
    profile.setdefault("version", 1)
    profile.setdefault("created_at", profile["updated_at"])

    profile_dir = get_profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)
    get_profile_json_path().write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    get_profile_markdown_path().write_text(
        render_profile_markdown(profile),
        encoding="utf-8",
    )


def render_profile_context(task_type: str, profile: dict[str, Any] | None) -> str:
    if not profile:
        return ""
    task_type = task_type.lower().strip()
    research = profile.get("research_profile", {})
    resource = profile.get("resource_profile", {})
    writing = profile.get("writing_profile", {})
    publication = profile.get("publication_profile", {})
    interaction = profile.get("interaction_profile", {})

    lines = [
        "=== USER PROFILE ===",
        f"Archetype: {profile.get('archetype_seed', '')}",
    ]

    if task_type in {"planning", "literature", "ideation"}:
        lines.extend(
            [
                f"Research direction: {research.get('domain', '')}",
                f"Method preference: {research.get('method_preference', '')}",
                f"Risk preference: {research.get('risk_preference', '')}",
                f"Baseline/ablation strictness: {research.get('baseline_ablation_strictness', '')}",
                f"Resource budget: {resource.get('gpu_budget', '')}, {resource.get('wall_clock_budget', '')}",
                f"Feasibility bias: {resource.get('feasibility_bias', '')}",
            ]
        )
    elif task_type in {"writing", "review"}:
        lines.extend(
            [
                f"Writing tone: {writing.get('tone', '')}",
                f"Claim strength: {writing.get('claim_strength', '')}",
                f"Section organization: {writing.get('section_organization', '')}",
                f"Venue/style preference: {publication.get('venue_style', '')}",
                f"Template preference: {publication.get('latex_template_preference', '')}",
                f"Figure style: {publication.get('figure_style', '')}",
                f"Caption style: {publication.get('caption_style', '')}",
                f"Critical feedback preference: {interaction.get('priority_feedback', '')}",
            ]
        )
    else:
        lines.extend(
            [
                f"Method preference: {research.get('method_preference', '')}",
                f"Resource budget: {resource.get('gpu_budget', '')}, {resource.get('wall_clock_budget', '')}",
                f"Critical feedback preference: {interaction.get('priority_feedback', '')}",
            ]
        )
    return "\n".join(lines)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
