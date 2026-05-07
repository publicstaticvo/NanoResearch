"""Router-policy decisions for NanoResearch evo mode.

The router owns policy-level selection and write-back over memory and skills.
It can call a configured SDPO router backend, and otherwise uses the same
validated action schema with a deterministic fallback so evo runs remain
usable while still producing auditable router traces.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from openai import OpenAI

from nanoresearch.config import ResearchConfig
from nanoresearch.evolution.memory import MemoryScope, MemoryStore, MemoryType
from nanoresearch.evolution.memory_analyzer import MemoryEvolutionAnalyzer
from nanoresearch.evolution.skills import SkillDomain

logger = logging.getLogger(__name__)

PRE_ROUTER_SYSTEM = (
    "You are a router making pre-execution decisions for NanoResearch. "
    "Return JSON only with keys selected_memory_ids, selected_skill_ids, prompt_plan, update_memory, update_skill. "
    "Use only ids listed in x.candidate_memory and x.candidate_skills. "
    "Select a focused subset, not everything. "
    "When task constraints conflict with persona defaults, prioritize task constraints. "
    "Set update_memory and update_skill to null. "
    "Keep prompt_plan under 30 words."
)

POST_ROUTER_SYSTEM = (
    "You are a hindsight-improved router for NanoResearch. "
    "Return JSON only with keys selected_memory_ids, selected_skill_ids, prompt_plan, update_memory, update_skill. "
    "Improve retrieval and prompt planning after feedback. "
    "Write update_memory only for stable preferences or recurring constraints. "
    "Write update_skill only for reusable procedural rules. "
    "Keep each update to one short sentence."
)

_SKILL_DOMAIN_ALIASES: dict[str, SkillDomain] = {
    "literature": SkillDomain.LITERATURE,
    "ideation": SkillDomain.LITERATURE,
    "planning": SkillDomain.PLANNING,
    "setup": SkillDomain.EXPERIMENT,
    "experiment": SkillDomain.EXPERIMENT,
    "execution": SkillDomain.EXPERIMENT,
    "analysis": SkillDomain.EXPERIMENT,
    "coding": SkillDomain.CODING,
    "writing": SkillDomain.WRITING,
    "review": SkillDomain.REVIEW,
}


def _normalize_skill_domain(domain: str, *, stage_name: str = "") -> SkillDomain:
    key = str(domain or "").strip().lower()
    if key in _SKILL_DOMAIN_ALIASES:
        return _SKILL_DOMAIN_ALIASES[key]
    stage_key = str(stage_name or "").strip().lower()
    return _SKILL_DOMAIN_ALIASES.get(stage_key, SkillDomain.PLANNING)


@dataclass
class RouterDecision:
    selected_memory_ids: list[str]
    selected_skill_ids: list[str]
    prompt_plan: str
    update_memory: str | None
    update_skill: str | None
    backend: str
    raw_response: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_memory_ids": list(self.selected_memory_ids),
            "selected_skill_ids": list(self.selected_skill_ids),
            "prompt_plan": self.prompt_plan,
            "update_memory": self.update_memory,
            "update_skill": self.update_skill,
            "backend": self.backend,
            "raw_response": self.raw_response,
        }


class RouterPolicyRunner:
    """Run the SDPO router online, or produce a schema-compatible fallback."""

    def __init__(self, config: ResearchConfig) -> None:
        self._model_path = str(getattr(config, "router_sdpo_model_path", "") or "").strip()
        self._model_name = str(getattr(config, "router_sdpo_model_name", "") or "").strip()
        self._base_url = str(getattr(config, "router_sdpo_base_url", "") or "").strip()
        self._api_key = str(getattr(config, "router_sdpo_api_key", "") or "").strip()
        self._timeout = float(getattr(config, "router_sdpo_timeout", 120.0) or 120.0)
        self._temperature = float(getattr(config, "router_sdpo_temperature", 0.0) or 0.0)
        self._max_new_tokens = int(getattr(config, "router_sdpo_max_new_tokens", 256) or 256)

    @property
    def is_configured(self) -> bool:
        return bool(self._base_url or self._model_path)

    def backend_name(self) -> str:
        if self._base_url:
            return f"remote:{self._model_name or 'router-sdpo'}"
        if self._model_path:
            return f"local:{self._model_path}"
        return "deterministic-fallback"

    def decide(self, payload: dict[str, Any], *, post_feedback: bool = False) -> RouterDecision:
        if not self.is_configured:
            action = self._build_fallback_action(payload.get("x", {}))
            return RouterDecision(backend=self.backend_name(), raw_response=json.dumps(action), **action)
        try:
            raw = self._generate_remote(payload, post_feedback=post_feedback)
            action = self._validate_action(self._canonicalize_action(self._extract_action(raw)), payload.get("x", {}))
            return RouterDecision(backend=self.backend_name(), raw_response=raw, **action)
        except Exception as exc:
            logger.warning("Router backend failed; falling back to deterministic routing: %s", exc)
            action = self._build_fallback_action(payload.get("x", {}))
            return RouterDecision(backend=f"fallback-after-error:{self.backend_name()}", raw_response=json.dumps(action), **action)

    def _generate_remote(self, payload: dict[str, Any], *, post_feedback: bool) -> str:
        if not self._base_url:
            raise RuntimeError("Local router model path is configured, but this release only enables remote online router inference.")
        client = OpenAI(base_url=self._base_url, api_key=self._api_key or "EMPTY", timeout=self._timeout)
        kwargs: dict[str, Any] = {
            "model": self._model_name or self._model_path or "router-sdpo",
            "messages": [
                {"role": "system", "content": POST_ROUTER_SYSTEM if post_feedback else PRE_ROUTER_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            "max_tokens": self._max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._temperature > 0:
            kwargs["temperature"] = self._temperature
        response = client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError("Router endpoint returned no choices")
        return response.choices[0].message.content or ""

    @staticmethod
    def _extract_action(raw_response: str) -> dict[str, Any]:
        text = (raw_response or "").strip()
        if not text:
            raise RuntimeError("Router returned empty response")
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise

    @staticmethod
    def _canonicalize_action(action: dict[str, Any]) -> dict[str, Any]:
        def _ids(value: Any) -> list[str]:
            if not value:
                return []
            if isinstance(value, list):
                return [str(item) for item in value if str(item).strip()]
            return [str(value)]
        return {
            "selected_memory_ids": _ids(action.get("selected_memory_ids")),
            "selected_skill_ids": _ids(action.get("selected_skill_ids")),
            "prompt_plan": " ".join(str(action.get("prompt_plan") or "").split()),
            "update_memory": " ".join(str(action.get("update_memory") or "").split()) or None,
            "update_skill": " ".join(str(action.get("update_skill") or "").split()) or None,
        }

    @staticmethod
    def _validate_action(action: dict[str, Any], router_x: dict[str, Any]) -> dict[str, Any]:
        valid_memory_ids = {str(item.get("memory_id")) for item in router_x.get("candidate_memory", []) if item.get("memory_id")}
        valid_skill_ids = {str(item.get("skill_id")) for item in router_x.get("candidate_skills", []) if item.get("skill_id")}
        action["selected_memory_ids"] = [item for item in action["selected_memory_ids"] if item in valid_memory_ids]
        action["selected_skill_ids"] = [item for item in action["selected_skill_ids"] if item in valid_skill_ids]
        action["prompt_plan"] = " ".join(action["prompt_plan"].split()[:30]) or "Use retrieved context conservatively."
        return action

    @staticmethod
    def _build_fallback_action(router_x: dict[str, Any]) -> dict[str, Any]:
        memory = [str(item.get("memory_id")) for item in (router_x.get("candidate_memory") or [])[:3] if item.get("memory_id")]
        skills = [str(item.get("skill_id")) for item in (router_x.get("candidate_skills") or [])[:3] if item.get("skill_id")]
        return {
            "selected_memory_ids": memory,
            "selected_skill_ids": skills,
            "prompt_plan": "Use top-ranked memory and skills when they directly improve this stage.",
            "update_memory": None,
            "update_skill": None,
        }


class RouterUpdateManager:
    """Router-layer write-back for long-term memory and skill evolution."""

    def __init__(self, *, config: ResearchConfig, memory_store: MemoryStore, memory_analyzer: MemoryEvolutionAnalyzer, skill_matcher: Any, workspace: Any) -> None:
        self._config = config
        self._memory_store = memory_store
        self._memory_analyzer = memory_analyzer
        self._skill_matcher = skill_matcher
        self._workspace = workspace

    def remember_context(self, *, stage_name: str, project_key: str, memory_type: MemoryType | str, content: str, importance: float = 0.6, tags: list[str] | None = None, source: str = "", scope: MemoryScope | str = MemoryScope.WORKSPACE_DERIVED) -> None:
        if not getattr(self._config, "memory_enabled", True):
            return
        self._memory_store.remember(
            memory_type,
            content,
            scope=scope,
            source=source or f"router:{stage_name}:{self._workspace.manifest.session_id}",
            importance=importance,
            tags=tags,
            project_key=project_key,
            workspace_id=self._workspace.manifest.session_id,
        )

    def learn_from_trace(self, *, stage_name: str, domain: str, trigger_pattern: str, source_trace: str, tags: list[str] | None = None, rule_text: str | None = None, confidence: float = 0.55) -> dict[str, Any] | None:
        if not getattr(self._config, "skill_evolution_enabled", True):
            return None
        trace = (source_trace or "").strip()
        if not trace:
            return None
        lifecycle = self._skill_matcher.evolution_store.synthesize_nl_skill(
            domain=_normalize_skill_domain(domain, stage_name=stage_name),
            trigger_pattern=trigger_pattern,
            source_trace=trace,
            rule_text=rule_text,
            confidence=confidence,
            tags=tags,
            source_stage=stage_name,
        )
        if lifecycle is None:
            return None
        payload = lifecycle.model_dump(mode="json")
        self._workspace.write_json(f"logs/evolved_skill_{stage_name}_{trigger_pattern}.json", payload)
        return payload
