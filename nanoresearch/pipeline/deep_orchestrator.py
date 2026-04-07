"""Deep 9-stage pipeline orchestrator."""

from __future__ import annotations

import logging
import shutil
from typing import Any

from nanoresearch.agents.analysis import AnalysisAgent
from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.coding import CodingAgent
from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.figure_gen import FigureAgent
from nanoresearch.agents.ideation import IdeationAgent
from nanoresearch.agents.planning import PlanningAgent
from nanoresearch.agents.review import ReviewAgent
from nanoresearch.agents.setup import SetupAgent
from nanoresearch.agents.writing import WritingAgent
from nanoresearch.pipeline.base_orchestrator import BaseOrchestrator
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage

logger = logging.getLogger(__name__)


class DeepPipelineOrchestrator(BaseOrchestrator):
    """Runs the deep 9-stage research pipeline with real experiments."""

    _STAGE_KEY_MAP: dict[PipelineStage, str] = {
        PipelineStage.IDEATION: "ideation_output",
        PipelineStage.PLANNING: "experiment_blueprint",
        PipelineStage.SETUP: "setup_output",
        PipelineStage.CODING: "coding_output",
        PipelineStage.EXECUTION: "execution_output",
        PipelineStage.ANALYSIS: "analysis_output",
        PipelineStage.FIGURE_GEN: "figure_gen_output",
        PipelineStage.WRITING: "writing_output",
        PipelineStage.REVIEW: "review_output",
    }

    _OUTPUT_FILE_MAP: dict[PipelineStage, str] = {
        PipelineStage.IDEATION: "papers/ideation_output.json",
        PipelineStage.PLANNING: "plans/experiment_blueprint.json",
        PipelineStage.SETUP: "plans/setup_output.json",
        PipelineStage.CODING: "plans/coding_output.json",
        PipelineStage.EXECUTION: "plans/execution_output.json",
        PipelineStage.ANALYSIS: "plans/analysis_output.json",
        PipelineStage.FIGURE_GEN: "drafts/figure_output.json",
        PipelineStage.WRITING: "drafts/paper_skeleton.json",
        PipelineStage.REVIEW: "drafts/review_output.json",
    }

    _PIPELINE_MODE = PipelineMode.DEEP

    def __init__(self, workspace, config, progress_callback=None) -> None:
        # Initialize RAM components before super().__init__ (which calls _build_agents)
        self._ram_module = None
        self._ram_collector = None
        if getattr(config, "ram_enabled", False):
            from nanoresearch.evolution.ram import RAMModule
            from nanoresearch.evolution.ram_data import RAMDataCollector

            self._ram_module = RAMModule(
                model_name_or_path=config.ram_model_name_or_path,
                backend=config.ram_backend,
                vllm_url=getattr(config, "ram_vllm_url", ""),
                max_new_tokens=getattr(config, "ram_max_new_tokens", 1024),
                temperature=getattr(config, "ram_temperature", 0.3),
                device=getattr(config, "ram_device", "auto"),
                enabled=True,
                checkpoint_path=getattr(config, "ram_checkpoint_path", ""),
            )
            self._ram_collector = RAMDataCollector(
                enabled=getattr(config, "ram_data_collection_enabled", True),
            )
        super().__init__(workspace, config, progress_callback)

    def _build_agents(self) -> dict[PipelineStage, BaseResearchAgent]:
        ram_kw: dict = {}
        if getattr(self, "_ram_module", None):
            ram_kw = {"ram_module": self._ram_module, "ram_collector": self._ram_collector}
        return {
            PipelineStage.IDEATION: IdeationAgent(self.workspace, self.config, **ram_kw),
            PipelineStage.PLANNING: PlanningAgent(self.workspace, self.config, **ram_kw),
            PipelineStage.SETUP: SetupAgent(self.workspace, self.config, **ram_kw),
            PipelineStage.CODING: CodingAgent(self.workspace, self.config, **ram_kw),
            PipelineStage.EXECUTION: ExecutionAgent(self.workspace, self.config, **ram_kw),
            PipelineStage.ANALYSIS: AnalysisAgent(self.workspace, self.config, **ram_kw),
            PipelineStage.FIGURE_GEN: FigureAgent(self.workspace, self.config, **ram_kw),
            PipelineStage.WRITING: WritingAgent(self.workspace, self.config, **ram_kw),
            PipelineStage.REVIEW: ReviewAgent(self.workspace, self.config, **ram_kw),
        }

    def _get_processing_stages(self) -> list[PipelineStage]:
        return PipelineStateMachine.processing_stages(PipelineMode.DEEP)

    def _get_initial_results(self, topic: str) -> dict[str, Any]:
        return {
            "topic": topic,
            "pipeline_mode": PipelineMode.DEEP.value,
        }

    def _post_pipeline(self, results: dict[str, Any]) -> None:
        """Export project and copy experiment code after successful completion."""
        try:
            export_path = self.workspace.export()
            logger.info("Exported project to: %s", export_path)
            results["export_path"] = str(export_path)

            exp_dir = self.workspace.path / "experiment"
            if exp_dir.exists():
                code_dest = export_path / "code"
                code_dest.mkdir(exist_ok=True)
                for file_path in (
                    list(exp_dir.glob("*.py"))
                    + list(exp_dir.glob("*.txt"))
                    + list(exp_dir.glob("*.slurm"))
                    + list(exp_dir.glob("*.json"))
                    + list(exp_dir.glob("*.yml"))
                    + list(exp_dir.glob("*.yaml"))
                    + list(exp_dir.glob("*.toml"))
                    + list(exp_dir.glob("*.cfg"))
                    + list(exp_dir.glob("*.sh"))
                ):
                    shutil.copy2(file_path, code_dest / file_path.name)

                results_src = exp_dir / "results"
                if results_src.exists():
                    results_dest = export_path / "results"
                    results_dest.mkdir(exist_ok=True)
                    for file_path in results_src.iterdir():
                        if file_path.is_file() and file_path.suffix in (
                            ".json", ".csv", ".log",
                        ):
                            shutil.copy2(file_path, results_dest / file_path.name)
        except Exception as exc:
            logger.warning("Export failed (non-fatal): %s", exc)

    async def _run_stage_with_retry(
        self, stage: PipelineStage, topic: str, accumulated: dict
    ) -> dict[str, Any]:
        """Run stage with retry, then complete RAM triple on success."""
        result = await super()._run_stage_with_retry(stage, topic, accumulated)
        # Complete RAM data collection triple
        if self._ram_module and self._ram_collector:
            agent = self._agents.get(stage)
            if agent and hasattr(agent, "complete_ram_triple"):
                quality = self._derive_quality_signal(stage, result)
                feedback_str = self._derive_feedback(stage, result)
                agent.complete_ram_triple(feedback_str, quality)
        return result

    def _derive_quality_signal(self, stage: PipelineStage, result: dict) -> float:
        """Derive quality signal from stage output for SDPO training."""
        stage_key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
        output = result.get(stage_key, {})
        if not isinstance(output, dict):
            return 0.5
        if stage == PipelineStage.EXECUTION:
            status = output.get("final_status") or output.get("experiment_status", "")
            if status == "success":
                return 1.0
            elif status in ("failed", "error"):
                return -1.0
            return 0.0
        elif stage == PipelineStage.REVIEW:
            score = output.get("overall_score", 0)
            if isinstance(score, (int, float)) and score > 0:
                return max(-1.0, min(1.0, (score - 5) / 5))
            return 0.0
        return 0.5

    def _derive_feedback(self, stage: PipelineStage, result: dict) -> str:
        """Derive feedback string from stage output for SDPO data collection."""
        stage_key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
        output = result.get(stage_key, {})
        if not isinstance(output, dict):
            return f"Stage {stage.value} completed."
        if stage == PipelineStage.EXECUTION:
            status = output.get("final_status") or output.get("experiment_status", "unknown")
            error = output.get("error_message", "")
            return f"Execution status: {status}" + (f"\nError: {error}" if error else "")
        elif stage == PipelineStage.REVIEW:
            review_text = output.get("review_text", "")
            return review_text[:2000] if review_text else "Review completed."
        elif stage == PipelineStage.CODING:
            files = output.get("generated_files", [])
            return f"Generated {len(files)} files: {', '.join(str(f) for f in files[:5])}"
        return f"Stage {stage.value} completed successfully."

    async def close(self) -> None:
        """Close agents and unload RAM model."""
        await super().close()
        if self._ram_module:
            self._ram_module.unload()

    def _prepare_inputs(
        self,
        stage: PipelineStage,
        topic: str,
        accumulated: dict,
        last_error: str,
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = {}

        if stage == PipelineStage.IDEATION:
            inputs["topic"] = topic

        elif stage == PipelineStage.PLANNING:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})

        elif stage == PipelineStage.SETUP:
            inputs["topic"] = topic
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        elif stage == PipelineStage.CODING:
            inputs["topic"] = topic
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            inputs["setup_output"] = accumulated.get("setup_output", {})

        elif stage == PipelineStage.EXECUTION:
            inputs["topic"] = topic
            inputs["coding_output"] = accumulated.get("coding_output", {})
            inputs["setup_output"] = accumulated.get("setup_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        elif stage == PipelineStage.ANALYSIS:
            inputs["execution_output"] = accumulated.get("execution_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        elif stage == PipelineStage.FIGURE_GEN:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            exec_output = accumulated.get("execution_output", {})
            analysis_output = accumulated.get("analysis_output", {})
            inputs["experiment_results"] = (
                exec_output.get("metrics")
                or (analysis_output.get("execution_output") or {}).get("metrics", {})
                or {}
            )
            inputs["experiment_analysis"] = analysis_output.get("analysis", {})
            inputs["experiment_summary"] = (
                analysis_output.get("experiment_summary")
                or exec_output.get("experiment_summary", "")
            )
            inputs["experiment_status"] = (
                exec_output.get("experiment_status")
                or exec_output.get("final_status", "pending")
            )
            # Pass ANALYSIS-generated figures so FIGURE_GEN can skip duplicates
            inputs["existing_figures"] = analysis_output.get("figures", {})
            # Pass survey blueprint for survey paper figure generation
            try:
                inputs["survey_blueprint"] = self.workspace.read_json("plans/survey_blueprint.json")
            except FileNotFoundError:
                inputs["survey_blueprint"] = {}

        elif stage == PipelineStage.WRITING:
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            # Merge figures from ANALYSIS + FIGURE_GEN so WRITING sees ALL of them.
            # ANALYSIS figures go first; FIGURE_GEN can override on key collision.
            analysis_figures = accumulated.get("analysis_output", {}).get("figures", {})
            fig_gen_figures = accumulated.get("figure_gen_output", {}).get("figures", {})
            merged_figures = {**analysis_figures, **fig_gen_figures}
            inputs["figure_output"] = {"figures": merged_figures} if merged_figures else {}
            inputs["template_format"] = self.config.template_format

            exec_output = accumulated.get("execution_output", {})
            analysis_output = accumulated.get("analysis_output", {})
            inputs["experiment_results"] = (
                exec_output.get("metrics")
                or (analysis_output.get("execution_output") or {}).get("metrics", {})
                or {}
            )
            inputs["experiment_analysis"] = analysis_output.get("analysis", {})
            inputs["experiment_summary"] = (
                analysis_output.get("experiment_summary")
                or exec_output.get("experiment_summary", "")
            )
            inputs["experiment_status"] = (
                exec_output.get("experiment_status")
                or exec_output.get("final_status", "pending")
            )

        elif stage == PipelineStage.REVIEW:
            writing_output = accumulated.get("writing_output", {})
            paper_tex = writing_output.get("paper_tex", "")
            if not paper_tex:
                tex_path = self.workspace.path / "drafts" / "paper.tex"
                if tex_path.exists():
                    paper_tex = tex_path.read_text(errors="replace")
            inputs["paper_tex"] = paper_tex
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            exec_output = accumulated.get("execution_output", {})
            analysis_output = accumulated.get("analysis_output", {})
            inputs["experiment_results"] = (
                exec_output.get("metrics")
                or (analysis_output.get("execution_output") or {}).get("metrics", {})
                or {}
            )
            inputs["experiment_analysis"] = analysis_output.get("analysis", {})
            inputs["experiment_status"] = (
                exec_output.get("experiment_status")
                or exec_output.get("final_status", "pending")
            )
            inputs["writing_grounding"] = writing_output.get("grounding", {})

        if last_error:
            inputs["_retry_error"] = last_error

        return inputs
