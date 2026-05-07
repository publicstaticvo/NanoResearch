"""Evo pipeline orchestrator built on the deep 9-stage backbone."""

from __future__ import annotations

from typing import Any

from nanoresearch.evolution.memory import MemoryScope, MemoryType

from nanoresearch.pipeline.deep_orchestrator import DeepPipelineOrchestrator
from nanoresearch.schemas.manifest import PipelineMode


class EvoPipelineOrchestrator(DeepPipelineOrchestrator):
    """Deep pipeline with explicit skill, memory, and policy evolution enabled.

    The evo mode intentionally reuses the stable deep stage order and only
    changes the lifecycle contract: adaptive memory and skill evolution are
    required parts of the run, and the manifest records the run as ``evo``.
    """

    _PIPELINE_MODE = PipelineMode.EVO

    def __init__(self, workspace, config, progress_callback=None) -> None:
        # Keep deep behavior intact while making evo runs explicit and auditable.
        config.memory_enabled = True
        config.memory_evolution_enabled = True
        config.skill_evolution_enabled = True
        config.same_router_hindsight_sdpo_enabled = True
        config.ram_data_collection_enabled = True
        super().__init__(workspace, config, progress_callback)

    def _get_initial_results(self, topic: str) -> dict[str, Any]:
        results = super()._get_initial_results(topic)
        results["pipeline_mode"] = PipelineMode.EVO.value
        results["evolution_mode"] = {
            "backbone": PipelineMode.DEEP.value,
            "memory_enabled": bool(getattr(self.config, "memory_enabled", True)),
            "memory_evolution_enabled": bool(getattr(self.config, "memory_evolution_enabled", True)),
            "skill_evolution_enabled": bool(getattr(self.config, "skill_evolution_enabled", True)),
            "ram_enabled": bool(getattr(self.config, "ram_enabled", False)),
            "same_router_hindsight_sdpo_enabled": bool(getattr(self.config, "same_router_hindsight_sdpo_enabled", False)),
            "router_backend": getattr(getattr(self, "_router_policy", None), "backend_name", lambda: "agent-level")(),
        }
        return results

    async def _run_stage_with_retry(self, stage, topic: str, accumulated: dict) -> dict[str, Any]:
        result = await super()._run_stage_with_retry(stage, topic, accumulated)
        agent = self._agents.get(stage)
        router_updates = getattr(agent, "_router_updates", None)
        if router_updates is not None:
            feedback = self._derive_feedback(stage, result)
            project_key_fn = getattr(agent, "_project_key", None)
            project_key = project_key_fn(topic) if callable(project_key_fn) else self.workspace.manifest.session_id
            try:
                router_updates.remember_context(
                    stage_name=stage.value.lower(),
                    project_key=project_key,
                    memory_type=MemoryType.DECISION_HISTORY,
                    content=f"Evo stage feedback for {stage.value}: {feedback[:1200]}",
                    importance=0.68,
                    tags=[topic, stage.value.lower(), "evo", "feedback"],
                    source=f"evo:{stage.value.lower()}",
                    scope=MemoryScope.WORKSPACE_DERIVED,
                )
                if stage.value in {"EXECUTION", "REVIEW", "CODING", "WRITING"}:
                    router_updates.learn_from_trace(
                        stage_name=stage.value.lower(),
                        domain=stage.value.lower(),
                        trigger_pattern="evo_stage_feedback",
                        source_trace=feedback,
                        tags=[topic, stage.value.lower(), "evo", "feedback"],
                        confidence=0.58,
                    )
                self.workspace.write_json(
                    f"logs/evo_router_feedback_{stage.value.lower()}.json",
                    {
                        "stage": stage.value,
                        "feedback": feedback,
                        "project_key": project_key,
                        "memory_update": True,
                    },
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Evo router feedback update failed for %s: %s", stage.value, exc)
        return result
