"""Base agent — common LLM call logic for all research agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

from nanoresearch.config import ResearchConfig, StageModelConfig
from nanoresearch.exceptions import LLMError
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.evolution.memory import MemoryScope, MemoryStore, MemoryType
from nanoresearch.evolution.memory_analyzer import MemoryEvolutionAnalyzer
from nanoresearch.profile import (
    load_user_profile,
    render_profile_context,
)
from nanoresearch.router_policy import RouterPolicyRunner, RouterUpdateManager
from nanoresearch.skills import UnifiedSkillMatcher

# Import all free functions from the helpers module so they remain accessible
# at their original locations (e.g. ``from nanoresearch.agents.base import detect_truncation``).
from nanoresearch.agents._base_helpers import (  # noqa: F401 — re-exports
    _VALID_JSON_ESCAPES,
    _LATEX_CMD_PREFIXES,
    _MAX_TOOL_RESULT_CHARS,
    _HEAD_CHARS,
    _TAIL_CHARS,
    _CONTEXT_COMPACT_THRESHOLD_CHARS,
    _PROTECTED_TAIL_TURNS,
    _truncate_tool_result,
    _compact_messages_if_needed,
    _fix_json_escapes,
    _extract_balanced_json_segment,
    _extract_json_candidates,
    _scan_json_fragment,
    _close_json_fragment,
    _trim_json_fragment,
    _repair_truncated_json,
    _json_error_msg,
    detect_truncation,
)

logger = logging.getLogger(__name__)

ROUTER_STAGE_PLAN_SYSTEM_PROMPT = """You are the stage planner that sits between NanoResearch's router and the stage execution model.

Your job is to expand the router's short hint into a concrete stage-specific plan that the downstream model can directly follow.

Requirements:
- Be operational, not high-level.
- Use the selected memories and skills explicitly when they are relevant.
- Respect the current stage only; do not re-plan the whole project.
- Keep the plan concise and execution-oriented.
- Do not mention hidden system internals or JSON ids.

Return plain text using EXACTLY these section headers:
Stage Goal:
Execution Plan:
Success Checks:
Avoid:
"""


class BaseResearchAgent(ABC):
    """Abstract base class for all NanoResearch agents."""

    stage: PipelineStage  # subclass must set this

    def __init__(self, workspace: Workspace, config: ResearchConfig) -> None:
        self.workspace = workspace
        self.config = config
        self._dispatcher = ModelDispatcher(config)
        static_skills_dir = getattr(config, "static_skills_dir", "") or None
        self._memory_store = MemoryStore(
            enabled=getattr(config, "memory_enabled", True),
            top_k=getattr(config, "memory_retrieval_top_k", 5),
            decay_factor=getattr(config, "memory_decay_factor", 0.08),
        )
        self._memory_analyzer = MemoryEvolutionAnalyzer(self._memory_store)
        self._skill_matcher = UnifiedSkillMatcher(
            Path(static_skills_dir) if static_skills_dir else None,
            retrieval_top_k=getattr(config, "skill_retrieval_top_k", 5),
            autorun_policy=getattr(config, "script_skill_autorun_policy", "safe_only"),
        )
        self._router_policy = RouterPolicyRunner(config)
        self._router_updates = RouterUpdateManager(
            config=config,
            memory_store=self._memory_store,
            memory_analyzer=self._memory_analyzer,
            skill_matcher=self._skill_matcher,
            workspace=self.workspace,
        )
        self._user_profile = load_user_profile() or {}

    def _safe_user_profile(self) -> dict[str, Any]:
        """Return a dict profile even if an upstream loader yields None."""
        if isinstance(self._user_profile, dict):
            return self._user_profile
        return {}

    def _remember_mutation_snapshot_entry(self, entry: dict[str, Any] | None) -> None:
        self._last_mutation_snapshot_entry = dict(entry) if isinstance(entry, dict) else None

    def consume_last_mutation_snapshot_entry(self) -> dict[str, Any] | None:
        entry = getattr(self, "_last_mutation_snapshot_entry", None)
        self._last_mutation_snapshot_entry = None
        return dict(entry) if isinstance(entry, dict) else None

    def _project_key(self, topic: str = "") -> str:
        raw = (topic or self.workspace.manifest.topic or self.workspace.manifest.session_id).strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
        return slug or self.workspace.manifest.session_id

    @staticmethod
    def _compact_router_text(value: Any, limit: int = 600) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + " ..."

    @staticmethod
    def _router_subsystem(task_type: str) -> str:
        task_type = (task_type or "").strip().lower()
        if task_type in {"ideation", "planning", "literature"}:
            return "method_generation"
        if task_type in {"experiment", "coding"}:
            return "code_implementation"
        if task_type == "writing":
            return "paper_writing"
        return "method_generation"

    @staticmethod
    def _router_plan_stage_name(task_type: str) -> str | None:
        task_type = (task_type or "").strip().lower()
        if task_type in {"ideation", "planning", "literature"}:
            return "method_plan"
        if task_type == "coding":
            return "coding_plan"
        if task_type == "writing":
            return "writing_plan"
        return None

    def _render_router_stage_plan(
        self,
        *,
        task_type: str,
        topic: str,
        blueprint: dict | None,
        profile_context: str,
        router_input: dict[str, Any],
        router_decision: dict[str, Any],
        selected_generic_memories: list[Any],
        selected_research_memories: list[Any],
        selected_static_matches: list[tuple[Any, int]],
        selected_nl_skills: list[Any],
        selected_script_skills: list[Any],
    ) -> tuple[str, dict[str, Any]]:
        if not bool(getattr(self.config, "router_planner_enabled", True)):
            return "", {}

        planner_stage = self._router_plan_stage_name(task_type)
        if not planner_stage:
            return "", {}

        cfg = self.config.for_stage(planner_stage)

        selected_memory_lines: list[str] = []
        for record in selected_generic_memories:
            selected_memory_lines.append(
                f"- generic::{record.memory_type.value}::{record.source or record.scope.value}: "
                f"{self._compact_router_text(record.content, 420)}"
            )
        for record in selected_research_memories:
            condition_bits = ", ".join(
                f"{key}={value}" for key, value in list(record.conditions.items())[:4]
            )
            suffix = f" | conditions: {condition_bits}" if condition_bits else ""
            selected_memory_lines.append(
                f"- research::{record.memory_kind.value}::{record.source_stage or record.source or 'research_memory'}: "
                f"{self._compact_router_text(record.content, 420)}{suffix}"
            )

        selected_skill_lines: list[str] = []
        for entry, _score in selected_static_matches:
            selected_skill_lines.append(
                f"- static::{entry.name}: {self._compact_router_text(entry.description or entry.name, 240)}"
            )
        for skill in selected_nl_skills:
            instructions = "; ".join(skill.instructions[:3]) if skill.instructions else (
                skill.rule_text or skill.description or skill.name
            )
            selected_skill_lines.append(
                f"- evolved::{skill.name or skill.stable_id or skill.skill_id}: "
                f"{self._compact_router_text(instructions, 240)}"
            )
        for skill in selected_script_skills:
            selected_skill_lines.append(
                f"- script::{skill.name}: {self._compact_router_text(skill.description or skill.name, 240)}"
            )

        blueprint_text = ""
        if blueprint:
            try:
                blueprint_text = json.dumps(blueprint, ensure_ascii=False)[:2500]
            except TypeError:
                blueprint_text = self._compact_router_text(str(blueprint), 2500)

        user_prompt = (
            f"Stage: {task_type}\n"
            f"Subsystem: {self._router_subsystem(task_type)}\n"
            f"Topic:\n{topic}\n\n"
            f"Profile Context:\n{profile_context or '(none)'}\n\n"
            f"Router Prompt Hint:\n{router_decision.get('prompt_plan', '') or '(empty)'}\n\n"
            f"Selected Memories:\n"
            + ("\n".join(selected_memory_lines) if selected_memory_lines else "- (none)")
            + "\n\nSelected Skills:\n"
            + ("\n".join(selected_skill_lines) if selected_skill_lines else "- (none)")
            + "\n\nBlueprint Context:\n"
            + (blueprint_text or "(none)")
            + "\n\nProduce the stage plan now."
        )

        timeout = cfg.timeout or self.config.timeout
        client = self._dispatcher._get_client(timeout, cfg.base_url, cfg.api_key)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": ROUTER_STAGE_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": cfg.max_tokens,
        }
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature

        t0 = time.monotonic()
        try:
            response = client.chat.completions.create(**kwargs)
            latency = (time.monotonic() - t0) * 1000
            usage = self._dispatcher._extract_usage(response)
            self._dispatcher._notify_usage("", usage, cfg.model, latency)
            if not response.choices:
                raise RuntimeError("planner model returned no choices")
            content = self._dispatcher._strip_think_blocks(
                response.choices[0].message.content or ""
            ).strip()
            if not content:
                raise RuntimeError("planner model returned empty plan text")
            payload = {
                "planner_stage": planner_stage,
                "planner_model": cfg.model,
                "planner_prompt": user_prompt,
                "plan_text": content,
                "router_input_task": router_input.get("task", ""),
            }
            return content, payload
        except Exception as exc:
            logger.warning(
                "Router stage planner failed for %s/%s via %s: %s",
                self.stage.value,
                task_type,
                cfg.model,
                exc,
            )
            return "", {
                "planner_stage": planner_stage,
                "planner_model": cfg.model,
                "planner_error": str(exc),
            }

    def _build_sdpo_router_blocks(
        self,
        task_type: str,
        *,
        topic: str,
        blueprint: dict | None,
        text: str,
        tags: list[str],
        template_format: str,
        include_script_recommendations: bool,
        project_key: str,
        profile_context: str,
    ) -> tuple[list[str], dict[str, Any]]:
        router_payload: dict[str, Any] = {}
        blocks: list[str] = []
        task_type_norm = (task_type or "").strip().lower()

        memory_records = self._memory_store.retrieve(
            task_type,
            topic=topic,
            tags=tags,
            text=text,
            project_key=project_key,
            top_k=getattr(self.config, "memory_retrieval_top_k", 5),
        ) if getattr(self.config, "memory_enabled", True) else []

        research_records: list[Any] = []
        if getattr(self.config, "memory_enabled", True) and getattr(self.config, "memory_evolution_enabled", True):
            research_conditions: dict[str, Any] = {
                "paper_mode": self.workspace.manifest.paper_mode.value,
            }
            if blueprint:
                research_conditions["has_blueprint"] = "yes"
            research_top_k = getattr(self.config, "direction_memory_top_k", 4)
            if task_type == "experiment":
                research_top_k = getattr(self.config, "strategy_memory_top_k", 4)
            research_records = self._memory_store.retrieve_research(
                task_type,
                topic=topic,
                tags=tags,
                text=text,
                conditions=research_conditions,
                project_key=project_key,
                top_k=research_top_k,
            )

        static_matches: list[tuple[Any, int]] = []
        if blueprint and task_type_norm in {"planning", "experiment", "coding"}:
            static_matches = self._skill_matcher._static.match(blueprint)
        elif task_type_norm in {"writing"}:
            static_matches = self._skill_matcher._static.match_writing_skills(topic=topic, template_format=template_format)

        text_payload = text
        if blueprint and not text_payload:
            try:
                text_payload = json.dumps(blueprint, ensure_ascii=False)
            except TypeError:
                text_payload = str(blueprint)

        domain = self._skill_matcher._domain_for_task(task_type)
        nl_matches = self._skill_matcher.evolution_store.match_nl_skills(
            domain,
            topic=topic,
            text=text_payload,
            tags=tags,
            top_k=getattr(self.config, "skill_retrieval_top_k", 5),
        ) if getattr(self.config, "skill_evolution_enabled", True) else []
        script_matches = self._skill_matcher.evolution_store.match_script_skills(
            domain,
            tags=tags,
            top_k=min(3, getattr(self.config, "skill_retrieval_top_k", 5)),
            autorun_policy=getattr(self.config, "script_skill_autorun_policy", "safe_only"),
        ) if getattr(self.config, "skill_evolution_enabled", True) and include_script_recommendations else []

        candidate_memory: list[dict[str, Any]] = []
        for record in memory_records:
            candidate_memory.append(
                {
                    "memory_id": record.memory_id,
                    "memory_type": record.memory_type.value,
                    "content": self._compact_router_text(record.content, 280),
                    "source": record.source or record.scope.value,
                }
            )
        for record in research_records:
            candidate_memory.append(
                {
                    "memory_id": record.memory_id,
                    "memory_type": record.memory_kind.value,
                    "content": self._compact_router_text(record.content, 280),
                    "source": record.source_stage or record.source or "research_memory",
                }
            )

        candidate_skills: list[dict[str, Any]] = []
        for entry, _score in static_matches:
            candidate_skills.append(
                {
                    "skill_id": entry.name,
                    "summary": self._compact_router_text(entry.description or entry.name, 180),
                    "source_kind": "static",
                }
            )
        for skill in nl_matches:
            candidate_skills.append(
                {
                    "skill_id": skill.stable_id or skill.skill_id,
                    "summary": self._compact_router_text(
                        skill.description or skill.when_to_use or skill.rule_text or skill.name,
                        180,
                    ),
                    "source_kind": "evolved_nl",
                }
            )
        for skill in script_matches:
            candidate_skills.append(
                {
                    "skill_id": skill.skill_id,
                    "summary": self._compact_router_text(skill.description or skill.name, 180),
                    "source_kind": "script",
                }
            )

        workspace_bits: list[str] = []
        if profile_context:
            workspace_bits.append(f"profile: {self._compact_router_text(profile_context, 320)}")
        if blueprint:
            workspace_bits.append(f"blueprint: {self._compact_router_text(json.dumps(blueprint, ensure_ascii=False), 420)}")
        if text:
            workspace_bits.append(f"current_context: {self._compact_router_text(text, 420)}")
        workspace_context = " | ".join(workspace_bits)

        profile = self._safe_user_profile()
        router_input = {
            "task": "Produce the hindsight-improved router decision after feedback."
            if task_type_norm in {"experiment", "writing"} and bool(text.strip())
            else "Produce the base router decision before tool execution.",
            "persona_id": str(profile.get("persona_id", "")),
            "round_id": self.workspace.manifest.session_id,
            "subsystem": self._router_subsystem(task_type_norm),
            "turn_id": f"{task_type_norm or 'stage'}-0",
            "profile_snapshot": profile,
            "task_spec": {
                "task_id": self.workspace.manifest.session_id,
                "topic": topic or self.workspace.manifest.topic,
                "task_brief": self._compact_router_text(topic or text or self.workspace.manifest.topic, 240),
                "stage_focus": task_type_norm,
            },
            "x": {
                "candidate_memory": candidate_memory,
                "candidate_skills": candidate_skills,
                "user_request": self._compact_router_text(topic or text or self.workspace.manifest.topic, 240),
                "workspace_context": workspace_context,
            },
        }
        decision = self._router_policy.decide(
            router_input,
            post_feedback=bool(text.strip()) and task_type_norm in {"experiment", "writing"},
        )

        selected_ids = set(decision.selected_memory_ids)
        selected_generic_memories = [record for record in memory_records if record.memory_id in selected_ids]
        selected_research_memories = [record for record in research_records if record.memory_id in selected_ids]

        if selected_research_memories:
            if task_type_norm in {"literature", "planning", "ideation"}:
                title = "DIRECTION MEMORY"
                instruction = (
                    "Use these router-selected direction summaries to prioritize feasible directions "
                    "and avoid repeating directions that failed under similar conditions."
                )
            elif task_type_norm == "experiment":
                title = "STRATEGY MEMORY"
                instruction = (
                    "Use these router-selected experiment strategies to improve data handling, preflight validation, "
                    "and training stability before making new implementation choices."
                )
            else:
                title = "RESEARCH MEMORY"
                instruction = "Use these router-selected research memories when they are directly relevant."
            lines = []
            for record in selected_research_memories:
                source = f" [{record.source_stage or record.source}]" if (record.source_stage or record.source) else ""
                condition_bits = ", ".join(f"{key}={value}" for key, value in list(record.conditions.items())[:4])
                evidence = f" | evidence: {record.evidence_summary}" if record.evidence_summary else ""
                trajectory = f" | trajectory: {'; '.join(record.trajectory_summary[:2])}" if record.trajectory_summary else ""
                uncertainty = f" | uncertainty: {record.uncertainty_note}" if record.uncertainty_note else ""
                suffix = f" | conditions: {condition_bits}" if condition_bits else ""
                lines.append(f"- ({record.memory_kind.value}){source} {record.content}{suffix}{evidence}{trajectory}{uncertainty}")
            blocks.append(
                f"\n\n=== {title} ===\n"
                f"{instruction}\n"
                + "\n".join(lines)
                + f"\n=== END {title} ===\n"
            )
            router_payload["selected_research_memory_ids"] = [record.memory_id for record in selected_research_memories]

        if selected_generic_memories:
            lines = []
            for record in selected_generic_memories:
                source = f" [{record.source}]" if record.source else ""
                lines.append(f"- ({record.memory_type.value}){source} {record.content}")
            blocks.append(
                "\n\n=== LONG-TERM RESEARCH MEMORY ===\n"
                "Use these router-selected durable preferences, prior decisions, and project facts when making choices.\n"
                + "\n".join(lines)
                + "\n=== END LONG-TERM RESEARCH MEMORY ===\n"
            )
            router_payload["selected_memory_ids"] = [record.memory_id for record in selected_generic_memories]

        selected_skill_ids = set(decision.selected_skill_ids)
        selected_static_matches = [
            item for item in static_matches if item[0].name in selected_skill_ids
        ]
        selected_nl_skills = [
            skill for skill in nl_matches if (skill.stable_id or skill.skill_id) in selected_skill_ids
        ]
        selected_script_skills = [
            skill for skill in script_matches if skill.skill_id in selected_skill_ids
        ]

        if selected_static_matches:
            static_ctx = self._skill_matcher._static.extract_context(selected_static_matches)
            if static_ctx.phase1_context:
                blocks.append(static_ctx.phase1_context)
            if include_script_recommendations and static_ctx.phase2_context:
                blocks.append(static_ctx.phase2_context)
            router_payload["selected_static_skills"] = [entry.name for entry, _ in selected_static_matches]

        if selected_nl_skills:
            lines = []
            for skill in selected_nl_skills:
                instructions = "; ".join(skill.instructions[:3]) if skill.instructions else skill.rule_text
                lines.append(
                    f"- [{skill.domain.value}/{skill.version}] {skill.name or skill.stable_id}: {instructions}"
                )
            blocks.append(
                "\n\n=== EVOLVED RESEARCH SKILLS ===\n"
                "Apply these router-selected reusable behavioral rules distilled from prior failures, retries, reviews, and artifact maintenance.\n"
                + "\n".join(lines)
                + "\n=== END EVOLVED RESEARCH SKILLS ===\n"
            )
            router_payload["selected_evolved_skill_ids"] = [
                skill.stable_id or skill.skill_id for skill in selected_nl_skills
            ]

        if selected_script_skills and include_script_recommendations:
            lines = []
            autorun_policy = getattr(self.config, "script_skill_autorun_policy", "safe_only")
            for skill in selected_script_skills:
                mode = "autorun" if skill.safe_to_autorun and autorun_policy != "off" else "recommended"
                lines.append(
                    f"- [{skill.category.value}/{mode}] {skill.name}: {skill.description} ({skill.script_path})"
                )
            blocks.append(
                "\n\n=== REGISTERED PYTHON SCRIPT SKILLS ===\n"
                "Prefer these router-selected tested low-risk automation hooks before asking the model to recreate repetitive setup or formatting work.\n"
                + "\n".join(lines)
                + "\n=== END REGISTERED PYTHON SCRIPT SKILLS ===\n"
            )
            router_payload["selected_script_skill_ids"] = [skill.skill_id for skill in selected_script_skills]

        if decision.prompt_plan:
            blocks.append(
                "\n\n=== SDPO ROUTER PROMPT PLAN ===\n"
                f"{decision.prompt_plan}\n"
                "=== END SDPO ROUTER PROMPT PLAN ===\n"
            )

        expanded_plan_text, expanded_plan_payload = self._render_router_stage_plan(
            task_type=task_type_norm,
            topic=topic,
            blueprint=blueprint,
            profile_context=profile_context,
            router_input=router_input,
            router_decision=decision.as_dict(),
            selected_generic_memories=selected_generic_memories,
            selected_research_memories=selected_research_memories,
            selected_static_matches=selected_static_matches,
            selected_nl_skills=selected_nl_skills,
            selected_script_skills=selected_script_skills,
        )
        if expanded_plan_text:
            blocks.append(
                "\n\n=== ROUTER EXPANDED STAGE PLAN ===\n"
                f"{expanded_plan_text}\n"
                "=== END ROUTER EXPANDED STAGE PLAN ===\n"
            )
        if expanded_plan_payload:
            router_payload["router_expanded_plan"] = expanded_plan_payload

        router_payload["router_input"] = router_input
        router_payload["router_decision"] = decision.as_dict()
        router_payload["candidate_memory_count"] = len(candidate_memory)
        router_payload["candidate_skill_count"] = len(candidate_skills)
        return blocks, router_payload

    def build_adaptive_context(
        self,
        task_type: str,
        *,
        topic: str = "",
        blueprint: dict | None = None,
        text: str = "",
        tags: list[str] | None = None,
        template_format: str = "",
        include_script_recommendations: bool = True,
    ) -> str:
        tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        blocks: list[str] = []
        payload: dict[str, Any] = {"task_type": task_type, "topic": topic, "tags": tags}
        try:
            project_key = self._project_key(topic)
            profile_context = render_profile_context(task_type, self._safe_user_profile())
            if profile_context:
                blocks.append(profile_context)
                payload["profile_context"] = profile_context
            if bool(getattr(self.config, "same_router_hindsight_sdpo_enabled", False)):
                sdpo_blocks, sdpo_payload = self._build_sdpo_router_blocks(
                    task_type,
                    topic=topic,
                    blueprint=blueprint,
                    text=text,
                    tags=tags,
                    template_format=template_format,
                    include_script_recommendations=include_script_recommendations,
                    project_key=project_key,
                    profile_context=profile_context,
                )
                blocks.extend(sdpo_blocks)
                payload["router_policy"] = sdpo_payload
            else:
                if getattr(self.config, "memory_enabled", True) and getattr(self.config, "memory_evolution_enabled", True):
                    research_conditions: dict[str, Any] = {
                        "paper_mode": self.workspace.manifest.paper_mode.value,
                    }
                    if blueprint:
                        research_conditions["has_blueprint"] = "yes"
                    research_top_k = getattr(self.config, "direction_memory_top_k", 4)
                    if task_type == "experiment":
                        research_top_k = getattr(self.config, "strategy_memory_top_k", 4)
                    research_context = self._memory_store.render_research_context(
                        task_type,
                        topic=topic,
                        tags=tags,
                        text=text,
                        conditions=research_conditions,
                        project_key=project_key,
                        top_k=research_top_k,
                    )
                    if research_context:
                        blocks.append(research_context)
                        payload["research_memory_context"] = research_context
                if getattr(self.config, "memory_enabled", True):
                    memory_context = self._memory_store.render_prompt_context(
                        task_type,
                        topic=topic,
                        tags=tags,
                        text=text,
                        project_key=project_key,
                        top_k=getattr(self.config, "memory_retrieval_top_k", 5),
                    )
                    if memory_context:
                        blocks.append(memory_context)
                        payload["memory_context"] = memory_context
                if getattr(self.config, "skill_evolution_enabled", True):
                    skill_context = self._skill_matcher.build_context(
                        task_type,
                        topic=topic,
                        blueprint=blueprint,
                        text=text,
                        tags=tags,
                        template_format=template_format,
                    )
                    if not include_script_recommendations:
                        combined = "\n\n".join(part for part in (skill_context.static_context, skill_context.evolved_context) if part)
                    else:
                        combined = skill_context.combined_context
                    if combined:
                        blocks.append(combined)
                        payload["matched_skills"] = skill_context.matched_skills
                        payload["skill_context"] = combined
        except Exception as exc:
            if bool(getattr(self.config, "same_router_hindsight_sdpo_enabled", False)):
                raise
            logger.warning("Failed to build adaptive context for %s/%s: %s", self.stage.value, task_type, exc)
        combined_context = "\n\n".join(blocks)
        if combined_context:
            payload["combined_context"] = combined_context
            try:
                self.workspace.write_json(f"logs/adaptive_context_{self.stage.value.lower()}_{task_type}.json", payload)
            except Exception as exc:
                logger.debug("Failed to persist adaptive context trace: %s", exc)
        return combined_context

    def wrap_with_adaptive_context(
        self,
        user_prompt: str,
        *,
        task_type: str,
        topic: str = "",
        blueprint: dict | None = None,
        text: str = "",
        tags: list[str] | None = None,
        template_format: str = "",
        include_script_recommendations: bool = True,
    ) -> str:
        """Prepend adaptive memory / skill / SDPO context to a user prompt."""
        adaptive_context = self.build_adaptive_context(
            task_type,
            topic=topic,
            blueprint=blueprint,
            text=text,
            tags=tags,
            template_format=template_format,
            include_script_recommendations=include_script_recommendations,
        )
        if not adaptive_context:
            return user_prompt
        return (
            "=== ADAPTIVE CONTEXT ===\n"
            f"{adaptive_context}\n"
            "=== END ADAPTIVE CONTEXT ===\n\n"
            f"{user_prompt}"
        )

    def remember_context(
        self,
        memory_type: MemoryType | str,
        content: str,
        *,
        importance: float = 0.6,
        tags: list[str] | None = None,
        source: str = "",
        scope: MemoryScope | str = MemoryScope.WORKSPACE_DERIVED,
        topic: str = "",
    ) -> None:
        try:
            self._router_updates.remember_context(
                stage_name=self.stage.value.lower(),
                project_key=self._project_key(topic),
                memory_type=memory_type,
                content=content,
                scope=scope,
                source=source,
                importance=importance,
                tags=tags,
            )
        except Exception as exc:
            logger.warning("Failed to write memory for %s: %s", self.stage.value, exc)

    def learn_from_trace(
        self,
        domain: str,
        trigger_pattern: str,
        source_trace: str,
        *,
        tags: list[str] | None = None,
        rule_text: str | None = None,
        confidence: float = 0.55,
    ) -> None:
        try:
            self._router_updates.learn_from_trace(
                stage_name=self.stage.value.lower(),
                domain=domain,
                trigger_pattern=trigger_pattern,
                source_trace=source_trace,
                rule_text=rule_text,
                confidence=confidence,
                tags=tags,
            )
        except Exception as exc:
            logger.warning("Failed to evolve skill for %s/%s: %s", self.stage.value, domain, exc)

    def remember_promising_direction(
        self,
        *,
        topic: str,
        ideation_output: dict | None = None,
        planning_output: dict | None = None,
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        try:
            payload = self._router_updates.remember_promising_direction(
                stage_name=self.stage.value.lower(),
                project_key=self._project_key(topic),
                topic=topic,
                ideation_output=ideation_output,
                planning_output=planning_output,
                source=source,
                source_stage=source_stage or self.stage.value.lower(),
                artifact_path=artifact_path,
            )
            return payload
        except Exception as exc:
            logger.warning("Failed to remember promising direction for %s: %s", self.stage.value, exc)
            return None

    def remember_failed_direction(
        self,
        *,
        topic: str,
        blueprint: dict | None = None,
        iteration_state: dict | None = None,
        failure_reason: str = "",
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        try:
            payload = self._router_updates.remember_failed_direction(
                stage_name=self.stage.value.lower(),
                project_key=self._project_key(topic),
                topic=topic,
                blueprint=blueprint,
                iteration_state=iteration_state,
                failure_reason=failure_reason,
                source=source,
                source_stage=source_stage or self.stage.value.lower(),
                artifact_path=artifact_path,
            )
            return payload
        except Exception as exc:
            logger.warning("Failed to remember failed direction for %s: %s", self.stage.value, exc)
            return None

    def remember_experiment_strategies(
        self,
        *,
        topic: str,
        blueprint: dict | None = None,
        iteration_state: dict | None = None,
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        try:
            payload = self._router_updates.remember_experiment_strategies(
                stage_name=self.stage.value.lower(),
                project_key=self._project_key(topic),
                topic=topic,
                blueprint=blueprint,
                iteration_state=iteration_state,
                source=source,
                source_stage=source_stage or self.stage.value.lower(),
                artifact_path=artifact_path,
            )
            return payload
        except Exception as exc:
            logger.warning("Failed to remember experiment strategies for %s: %s", self.stage.value, exc)
            return None

    @property
    def stage_config(self) -> StageModelConfig:
        return self.config.for_stage(self.stage.value)

    async def close(self) -> None:
        await self._dispatcher.close()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        stage_override: StageModelConfig | None = None,
    ) -> str:
        """Call the LLM configured for this agent's stage."""
        cfg = stage_override if stage_override is not None else self.stage_config
        return await self._dispatcher.generate(
            cfg, system_prompt, user_prompt, json_mode
        )

    async def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        json_mode: bool = False,
        stage_override: StageModelConfig | None = None,
    ) -> str:
        """Call the LLM with an image attachment (vision)."""
        cfg = stage_override if stage_override is not None else self.stage_config
        return await self._dispatcher.generate_with_image(
            cfg, system_prompt, user_prompt, image_bytes, mime_type, json_mode
        )

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        stage_override: StageModelConfig | None = None,
    ) -> dict | list:
        """Call LLM and parse the response as JSON.

        Handles LaTeX backslash sequences that break strict JSON parsing.
        """
        raw = await self.generate(
            system_prompt, user_prompt, json_mode=True,
            stage_override=stage_override,
        )
        last_attempt = raw.strip()
        for text in _extract_json_candidates(raw):
            last_attempt = text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            fixed = _fix_json_escapes(text)
            last_attempt = fixed
            try:
                return json.loads(fixed, strict=False)
            except json.JSONDecodeError:
                pass

            repaired = _repair_truncated_json(fixed)
            if repaired is not None and repaired != fixed:
                last_attempt = repaired
                try:
                    return json.loads(repaired, strict=False)
                except json.JSONDecodeError:
                    pass

        # All attempts failed
        logger.error(
            "JSON parse failed even after escape fixing. First 500 chars: %s",
            last_attempt[:500],
        )
        raise LLMError(
            f"LLM output is not valid JSON: "
            f"{_json_error_msg(last_attempt)}. "
            f"Raw output starts with: {raw[:200]!r}"
        ) from None

    async def generate_json_validated(
        self,
        system_prompt: str,
        user_prompt: str,
        model_class: type,
        stage_override: StageModelConfig | None = None,
    ) -> Any:
        """Call LLM, parse as JSON, and validate against a Pydantic model.

        On validation failure, feeds the error back to the LLM for one retry.
        Returns a validated Pydantic model instance.
        """
        raw_dict = await self.generate_json(
            system_prompt, user_prompt, stage_override=stage_override,
        )
        try:
            return model_class.model_validate(raw_dict)
        except Exception as first_exc:
            # Single retry: feed validation error back to LLM
            self.log(f"  JSON schema validation failed: {first_exc}, retrying...")
            retry_prompt = (
                f"Your previous JSON response had validation errors:\n"
                f"{first_exc}\n\n"
                f"Original request:\n{user_prompt}\n\n"
                f"Fix the JSON to match the required schema and try again."
            )
            try:
                raw_dict = await self.generate_json(
                    system_prompt, retry_prompt, stage_override=stage_override,
                )
                return model_class.model_validate(raw_dict)
            except Exception as retry_exc:
                logger.error(
                    "JSON validation failed after retry: %s", retry_exc,
                )
                raise LLMError(
                    f"JSON schema validation failed after retry: {retry_exc}"
                ) from retry_exc

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Any,  # ToolRegistry
        max_tool_rounds: int = 10,
        stage_override: StageModelConfig | None = None,
        reminder_text: str | None = None,
        reminder_interval: int = 3,
    ) -> str:
        """Run a ReAct loop: let the LLM call tools until it produces text."""
        cfg = stage_override if stage_override is not None else self.stage_config
        openai_tools = tools.to_openai_tools()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Track repeated failures to avoid infinite retry loops (OpenClaw pattern)
        _failure_counts: dict[str, int] = {}
        _MAX_IDENTICAL_FAILURES = 2

        for round_idx in range(max_tool_rounds):
            msg = await self._dispatcher.generate_with_tools(cfg, messages, openai_tools)

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return self._dispatcher._strip_think_blocks(msg.content or "")

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            if msg.content:
                assistant_msg["content"] = msg.content
            messages.append(assistant_msg)

            async def _execute_tool_call(tc):
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON in tool args for %s: %s", name, exc)
                    args = {}

                self.log(f"Tool call: {name}({args})")
                try:
                    result = await tools.call(name, args)
                    result_str = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    error_str = f"Error: {type(e).__name__}: {e}"
                    error_sig = type(e).__name__
                    try:
                        args_hash = hash(json.dumps(args, sort_keys=True, default=str))
                    except (TypeError, ValueError):
                        args_hash = hash(str(sorted(args.items())) if isinstance(args, dict) else str(args))
                    fail_key = f"{name}|{args_hash}|{error_sig}"
                    _failure_counts[fail_key] = _failure_counts.get(fail_key, 0) + 1
                    if _failure_counts[fail_key] >= _MAX_IDENTICAL_FAILURES:
                        error_str = (
                            f"[NON-RETRYABLE] {error_str} — "
                            f"This exact call has failed {_failure_counts[fail_key]} times. "
                            f"Do NOT retry with the same arguments. Try a different query or approach."
                        )
                    result_str = error_str

                result_content = _truncate_tool_result(result_str)
                return {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_content,
                }

            if len(tool_calls) > 1:
                tool_results = await asyncio.gather(
                    *(_execute_tool_call(tc) for tc in tool_calls),
                    return_exceptions=True,
                )
                for i, tr in enumerate(tool_results):
                    if isinstance(tr, Exception):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_calls[i].id,
                            "content": f"Error: {type(tr).__name__}: {tr}",
                        })
                    else:
                        messages.append(tr)
            else:
                messages.append(await _execute_tool_call(tool_calls[0]))

            _compact_messages_if_needed(messages)

            if (round_idx + 1) % reminder_interval == 0 and round_idx + 1 <= max_tool_rounds:
                _reminder = reminder_text or (
                    "[REMINDER] You are writing academic content for a top-tier venue. "
                    "Focus on producing the final output now. Use the information "
                    "gathered from tools to write high-quality content. "
                    "Do NOT continue searching indefinitely."
                )
                messages.append({"role": "system", "content": _reminder})

        self.log(f"Exceeded {max_tool_rounds} tool rounds, forcing final answer")
        final_msg = await self._dispatcher.generate_with_tools(cfg, messages, tools=None)
        if hasattr(final_msg, 'tool_calls') and final_msg.tool_calls:
            return self._dispatcher._strip_think_blocks(
                final_msg.content or "Agent completed but produced no text summary."
            )
        return self._dispatcher._strip_think_blocks(final_msg.content or "")

    @abstractmethod
    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute this agent's stage. Returns output data dict."""
        ...

    def log(self, msg: str) -> None:
        logger.info(f"[{self.stage.value}] {msg}")

    def save_log(self, filename: str, content: str) -> None:
        self.workspace.write_text(f"logs/{filename}", content)

    def _resolve_experiment_python(self) -> str:
        """Return the experiment Python path.

        Resolution order:
        1. config.experiment_python (user-managed environment)
        2. experiment/.venv python (auto-created venv)
        3. sys.executable (fallback)
        """
        import os
        import sys
        from pathlib import Path as _Path

        # Priority 1: user-specified python
        user_spec = (self.config.experiment_python or "").strip()
        if user_spec:
            from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
            mgr = RuntimeEnvironmentManager(self.config)
            resolved = mgr._resolve_user_python(user_spec)
            if resolved and _Path(resolved).exists():
                return resolved

        # Priority 2: experiment venv
        exp_dir = self.workspace.path / "experiment"
        if os.name == "nt":
            venv_py = exp_dir / ".venv" / "Scripts" / "python.exe"
        else:
            venv_py = exp_dir / ".venv" / "bin" / "python"
        if venv_py.exists():
            return str(venv_py)
        return sys.executable
