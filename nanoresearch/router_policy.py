"""Online SDPO router inference for adaptive memory/skill selection.

This module also owns router-layer memory / skill write-back so the adaptive
surface covers both retrieval selection and long-term updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
from typing import Any

import httpx
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
    "Keep prompt_plan under 30 words. Output one valid JSON object only."
)

POST_ROUTER_SYSTEM = (
    "You are a hindsight-improved router for NanoResearch. "
    "Return JSON only with keys selected_memory_ids, selected_skill_ids, prompt_plan, update_memory, update_skill. "
    "Use only ids listed in x.candidate_memory and x.candidate_skills plus any evolved ids already present in x.candidate_skills. "
    "Improve retrieval and prompt planning after feedback. "
    "When task constraints conflict with persona defaults, prioritize task constraints. "
    "Write update_memory only for stable preferences or recurring constraints. "
    "Write update_skill only for reusable procedural rules. "
    "Keep prompt_plan under 30 words. Keep each update to one short sentence. Output one valid JSON object only."
)

ROUTER_KEY_ORDER = (
    "selected_memory_ids",
    "selected_skill_ids",
    "prompt_plan",
    "update_memory",
    "update_skill",
)

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_CACHE_LOCK = threading.Lock()

_SKILL_DOMAIN_ALIASES: dict[str, SkillDomain] = {
    "literature": SkillDomain.LITERATURE,
    "ideation": SkillDomain.LITERATURE,
    "planning": SkillDomain.PLANNING,
    "setup": SkillDomain.EXPERIMENT,
    "experiment": SkillDomain.EXPERIMENT,
    "execution": SkillDomain.EXPERIMENT,
    "debug": SkillDomain.EXPERIMENT,
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
    if stage_key in _SKILL_DOMAIN_ALIASES:
        return _SKILL_DOMAIN_ALIASES[stage_key]
    return SkillDomain.PLANNING


def _looks_like_tokenizer_fast_parse_error(exc: BaseException) -> bool:
    text = str(exc or "")
    return "ModelWrapper" in text or "TokenizerFast.from_file" in text


def _oom_exception_types(torch_module: Any) -> tuple[type[BaseException], ...]:
    oom_type = getattr(torch_module, "OutOfMemoryError", None)
    if isinstance(oom_type, type) and issubclass(oom_type, BaseException):
        return (RuntimeError, oom_type)
    return (RuntimeError,)


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
    """Run the trained SDPO router online via local HF weights or an endpoint."""

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
        return "unconfigured"

    def decide(self, payload: dict[str, Any], *, post_feedback: bool = False) -> RouterDecision:
        if not self.is_configured:
            raise RuntimeError(
                "same_router_hindsight_sdpo_enabled=True but no SDPO router backend is configured. "
                "Set either router_sdpo_model_path or router_sdpo_base_url/router_sdpo_model_name."
            )

        system_prompt = POST_ROUTER_SYSTEM if post_feedback else PRE_ROUTER_SYSTEM
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        raw_response = self._generate(system_prompt, user_prompt)
        try:
            action = self._canonicalize_action(self._extract_action(raw_response))
            action = self._validate_action(action, payload.get("x", {}))
        except RuntimeError as exc:
            logger.warning(
                "SDPO router produced an unusable action for %s; falling back to top-ranked candidates: %s",
                self.backend_name(),
                exc,
            )
            action = self._build_fallback_action(payload.get("x", {}))
        return RouterDecision(
            selected_memory_ids=action["selected_memory_ids"],
            selected_skill_ids=action["selected_skill_ids"],
            prompt_plan=action["prompt_plan"],
            update_memory=action["update_memory"],
            update_skill=action["update_skill"],
            backend=self.backend_name(),
            raw_response=raw_response,
        )

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._base_url:
            return self._generate_remote(system_prompt, user_prompt)
        return self._generate_local(system_prompt, user_prompt)

    def _generate_remote(self, system_prompt: str, user_prompt: str) -> str:
        client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key or "EMPTY",
            timeout=httpx.Timeout(self._timeout, connect=15.0),
        )
        kwargs: dict[str, Any] = {
            "model": self._model_name or self._model_path,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        if self._temperature > 0:
            kwargs["temperature"] = self._temperature
        response = client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError("SDPO router endpoint returned no choices")
        return response.choices[0].message.content or ""

    def _generate_local(self, system_prompt: str, user_prompt: str) -> str:
        if not self._model_path:
            raise RuntimeError("Local SDPO router requested without router_sdpo_model_path")
        import torch

        tokenizer, model = self._load_local_model(self._model_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(tokenizer, "chat_template", None):
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **template_kwargs,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": self._temperature > 0,
        }
        if self._temperature > 0:
            generate_kwargs["temperature"] = self._temperature
            generate_kwargs["top_p"] = 0.95

        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)
        prompt_len = int(inputs["input_ids"].shape[1])
        completion_ids = output[0][prompt_len:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True)

    @staticmethod
    def _load_local_model(model_path: str) -> tuple[Any, Any]:
        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(model_path)
            if cached is not None:
                return cached
            import torch
            import tempfile
            import shutil
            from pathlib import Path
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            except Exception as exc:
                if not _looks_like_tokenizer_fast_parse_error(exc):
                    raise
                logger.warning(
                    "Fast tokenizer load failed for local SDPO router at %s; falling back to slow tokenizer: %s",
                    model_path,
                    exc,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=False,
                )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            load_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            except ValueError as exc:
                if "model type `qwen3`" not in str(exc).lower():
                    raise
                # Some cluster images still ship older Transformers builds that
                # understand Qwen2 but not the newer `qwen3` config tag. The
                # router checkpoints remain decoder-only causal LMs, so patching
                # only the local config/model type is enough to recover loading.
                with tempfile.TemporaryDirectory(prefix="router_qwen_compat_") as tmpdir:
                    tmp_path = Path(tmpdir)
                    for child in Path(model_path).iterdir():
                        target = tmp_path / child.name
                        if child.is_dir():
                            shutil.copytree(child, target)
                        else:
                            shutil.copy2(child, target)
                    config_path = tmp_path / "config.json"
                    if config_path.is_file():
                        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                        if str(config_payload.get("model_type") or "").strip().lower() == "qwen3":
                            config_payload["model_type"] = "qwen2"
                        architectures = config_payload.get("architectures")
                        if isinstance(architectures, list):
                            config_payload["architectures"] = [
                                "Qwen2ForCausalLM" if str(item) == "Qwen3ForCausalLM" else item
                                for item in architectures
                            ]
                        config_path.write_text(
                            json.dumps(config_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    model = AutoModelForCausalLM.from_pretrained(str(tmp_path), **load_kwargs)
            if torch.cuda.is_available():
                try:
                    model = model.to("cuda")
                except _oom_exception_types(torch) as exc:
                    if not RouterPolicyRunner._is_cuda_oom(exc):
                        raise
                    logger.warning(
                        "Falling back to CPU for local SDPO router at %s after CUDA OOM: %s",
                        model_path,
                        exc,
                    )
                    del model
                    torch.cuda.empty_cache()
                    cpu_load_kwargs = dict(load_kwargs)
                    cpu_load_kwargs.pop("torch_dtype", None)
                    model = AutoModelForCausalLM.from_pretrained(model_path, **cpu_load_kwargs)
            model.eval()
            _MODEL_CACHE[model_path] = (tokenizer, model)
            return tokenizer, model

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        text = str(exc).lower()
        return "out of memory" in text and "cuda" in text

    @staticmethod
    def _extract_action(raw_response: str) -> dict[str, Any]:
        text = (raw_response or "").strip()
        if not text:
            raise RuntimeError("SDPO router returned empty response")
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        brace_start = text.find("{")
        if brace_start < 0:
            raise RuntimeError(f"SDPO router did not return JSON: {text[:200]}")
        depth = 0
        for idx in range(brace_start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    fragment = text[brace_start:idx + 1]
                    try:
                        return json.loads(fragment)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"SDPO router returned invalid JSON: {fragment[:300]}") from exc
        raise RuntimeError(f"SDPO router JSON was not closed: {text[:300]}")

    @staticmethod
    def _canonicalize_action(action: dict[str, Any]) -> dict[str, Any]:
        canonical: dict[str, Any] = {}
        for key in ROUTER_KEY_ORDER:
            value = action.get(key)
            if key.endswith("_ids"):
                if not value:
                    canonical[key] = []
                elif isinstance(value, list):
                    canonical[key] = [str(item) for item in value if str(item).strip()]
                else:
                    canonical[key] = [str(value)]
            elif key.startswith("update_"):
                text = " ".join(str(value or "").split())
                canonical[key] = text or None
            else:
                canonical[key] = " ".join(str(value or "").split())
        return canonical

    @staticmethod
    def _validate_action(action: dict[str, Any], router_x: dict[str, Any]) -> dict[str, Any]:
        valid_memory_ids = {
            str(item.get("memory_id"))
            for item in (router_x.get("candidate_memory") or [])
            if item.get("memory_id")
        }
        valid_skill_ids = {
            str(item.get("skill_id"))
            for item in (router_x.get("candidate_skills") or [])
            if item.get("skill_id")
        }
        action["selected_memory_ids"] = [
            item for item in action["selected_memory_ids"] if item in valid_memory_ids
        ]
        action["selected_skill_ids"] = [
            item for item in action["selected_skill_ids"] if item in valid_skill_ids
        ]
        if len(action["prompt_plan"].split()) > 30:
            action["prompt_plan"] = " ".join(action["prompt_plan"].split()[:30])
        return action

    @staticmethod
    def _build_fallback_action(router_x: dict[str, Any]) -> dict[str, Any]:
        candidate_memory = router_x.get("candidate_memory") or []
        candidate_skills = router_x.get("candidate_skills") or []
        selected_memory_ids = [
            str(item.get("memory_id"))
            for item in candidate_memory[:3]
            if item.get("memory_id")
        ]
        selected_skill_ids = [
            str(item.get("skill_id"))
            for item in candidate_skills[:3]
            if item.get("skill_id")
        ]
        return {
            "selected_memory_ids": selected_memory_ids,
            "selected_skill_ids": selected_skill_ids,
            "prompt_plan": "Use top-ranked retrieved context conservatively.",
            "update_memory": None,
            "update_skill": None,
        }


class RouterUpdateManager:
    """Router-layer owner for memory / skill store updates."""

    def __init__(
        self,
        *,
        config: ResearchConfig,
        memory_store: MemoryStore,
        memory_analyzer: MemoryEvolutionAnalyzer,
        skill_matcher: Any,
        workspace: Any,
    ) -> None:
        self._config = config
        self._memory_store = memory_store
        self._memory_analyzer = memory_analyzer
        self._skill_matcher = skill_matcher
        self._workspace = workspace

    def remember_context(
        self,
        *,
        stage_name: str,
        project_key: str,
        memory_type: MemoryType | str,
        content: str,
        importance: float = 0.6,
        tags: list[str] | None = None,
        source: str = "",
        scope: MemoryScope | str = MemoryScope.WORKSPACE_DERIVED,
    ) -> None:
        if not getattr(self._config, "memory_enabled", True):
            return
        self._memory_store.remember(
            memory_type,
            content,
            scope=scope,
            source=source or f"{stage_name}:{self._workspace.manifest.session_id}",
            importance=importance,
            tags=tags,
            project_key=project_key,
            workspace_id=self._workspace.manifest.session_id,
        )

    def learn_from_trace(
        self,
        *,
        stage_name: str,
        domain: str,
        trigger_pattern: str,
        source_trace: str,
        tags: list[str] | None = None,
        rule_text: str | None = None,
        confidence: float = 0.55,
    ) -> dict[str, Any] | None:
        if not getattr(self._config, "skill_evolution_enabled", True):
            return None
        trace = (source_trace or "").strip()
        if not trace:
            return None
        normalized_domain = _normalize_skill_domain(domain, stage_name=stage_name)
        lifecycle = self._skill_matcher.evolution_store.synthesize_nl_skill(
            domain=normalized_domain,
            trigger_pattern=trigger_pattern,
            source_trace=trace,
            rule_text=rule_text,
            confidence=confidence,
            tags=tags,
            source_stage=stage_name,
        )
        if lifecycle is not None:
            payload = lifecycle.model_dump(mode="json")
            self._workspace.write_json(
                f"logs/evolved_skill_{stage_name}_{trigger_pattern}.json",
                payload,
            )
            return payload
        return None

    def remember_promising_direction(
        self,
        *,
        stage_name: str,
        project_key: str,
        topic: str,
        ideation_output: dict | None = None,
        planning_output: dict | None = None,
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        if not getattr(self._config, "memory_enabled", True) or not getattr(self._config, "memory_evolution_enabled", True):
            return None
        payload = self._memory_analyzer.summarize_promising_direction(
            topic=topic,
            paper_mode=self._workspace.manifest.paper_mode.value,
            ideation_output=ideation_output,
            planning_output=planning_output,
            source=source or f"{stage_name}:{self._workspace.manifest.session_id}",
            source_stage=source_stage or stage_name,
            project_key=project_key,
            workspace_id=self._workspace.manifest.session_id,
        )
        if payload and artifact_path:
            self._workspace.write_json(artifact_path, payload)
        return payload

    def remember_failed_direction(
        self,
        *,
        stage_name: str,
        project_key: str,
        topic: str,
        blueprint: dict | None = None,
        iteration_state: dict | None = None,
        failure_reason: str = "",
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        if not getattr(self._config, "memory_enabled", True) or not getattr(self._config, "memory_evolution_enabled", True):
            return None
        payload = self._memory_analyzer.summarize_failed_direction(
            topic=topic,
            paper_mode=self._workspace.manifest.paper_mode.value,
            blueprint=blueprint,
            iteration_state=iteration_state,
            failure_reason=failure_reason,
            source=source or f"{stage_name}:{self._workspace.manifest.session_id}",
            source_stage=source_stage or stage_name,
            project_key=project_key,
            workspace_id=self._workspace.manifest.session_id,
        )
        if payload and artifact_path:
            self._workspace.write_json(artifact_path, payload)
        return payload

    def remember_experiment_strategies(
        self,
        *,
        stage_name: str,
        project_key: str,
        topic: str,
        blueprint: dict | None = None,
        iteration_state: dict | None = None,
        artifact_path: str | None = None,
        source_stage: str = "",
        source: str = "",
    ) -> dict[str, Any] | None:
        if not getattr(self._config, "memory_enabled", True) or not getattr(self._config, "memory_evolution_enabled", True):
            return None
        payload = self._memory_analyzer.summarize_experiment_strategies(
            topic=topic,
            paper_mode=self._workspace.manifest.paper_mode.value,
            blueprint=blueprint,
            iteration_state=iteration_state,
            source=source or f"{stage_name}:{self._workspace.manifest.session_id}",
            source_stage=source_stage or stage_name,
            project_key=project_key,
            workspace_id=self._workspace.manifest.session_id,
        )
        if payload and artifact_path:
            self._workspace.write_json(artifact_path, payload)
        return payload
