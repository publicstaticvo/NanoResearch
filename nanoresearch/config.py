"""Research configuration — per-stage model routing and global settings."""

from __future__ import annotations

from enum import Enum
import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

from nanoresearch.paths import get_config_path, get_private_endpoints_path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"


class ExecutionProfile(str, Enum):
    """High-level execution behavior presets for the unified pipeline."""

    FAST_DRAFT = "fast_draft"
    LOCAL_QUICK = "local_quick"
    CLUSTER_FULL = "cluster_full"


class WritingMode(str, Enum):
    """How aggressively the writing stage should use tools."""

    DIRECT = "direct"
    HYBRID = "hybrid"
    REACT = "react"


class StageModelConfig(BaseModel):
    """Configuration for a single pipeline stage."""

    model: str = DEFAULT_MODEL
    temperature: float | None = 0.3  # None = don't send (for models like Codex/o-series)
    max_tokens: int = 8192
    timeout: float | None = None  # per-stage override; None = use global

    # Image generation backend: "openai" (DALL-E) or "gemini" (native Gemini API)
    image_backend: str = "openai"
    # Per-stage base_url / api_key override (e.g. Gemini image API uses different endpoint)
    base_url: str | None = None
    api_key: str | None = None
    # Gemini-specific image options
    aspect_ratio: str = "1:1"
    image_size: str = "1024x1024"


class ResearchConfig(BaseModel):
    """Top-level research configuration."""

    base_url: str = ""
    api_key: str = ""
    timeout: float = 180.0

    ideation: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2", temperature=0.5,
            max_tokens=16384, timeout=600.0,
        )
    )
    planning: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2", temperature=0.2,
            max_tokens=16384, timeout=600.0,
        )
    )
    experiment: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2", temperature=0.1, timeout=600.0
        )
    )
    writing: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="pro/gpt-5.5", temperature=0.35,
            max_tokens=8192, timeout=300.0,
        )
    )
    code_gen: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="pro/gpt-5.5", temperature=0.1,
            max_tokens=16384, timeout=600.0,
        )
    )
    figure_prompt: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="pro/gpt-5.5", temperature=0.5,
            max_tokens=4096, timeout=300.0,
        )
    )
    figure_code: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="pro/gpt-5.5", temperature=0.1,
            max_tokens=16384, timeout=600.0,
        )
    )
    figure_gen: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="gpt-image-2",
            image_backend="openai",
            temperature=None,
            timeout=600.0,
        )
    )
    evidence_extraction: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model="deepseek-ai/DeepSeek-V3.2",
            temperature=0.1,
        )
    )
    review: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model=DEFAULT_MODEL,
            temperature=0.3,
            max_tokens=16384,
            timeout=300.0,
        )
    )
    revision: StageModelConfig = Field(
        default_factory=lambda: StageModelConfig(
            model=DEFAULT_MODEL,
            temperature=0.3,
            max_tokens=16384,
            timeout=600.0,
        )
    )
    skip_stages: list[str] = Field(default_factory=list)
    template_format: str = "neurips"
    max_retries: int = 2
    quick_eval_timeout: int = 3600  # seconds for quick-eval execution (60 min — includes dataset download)
    strict_experiment_contract: bool = False
    execution_profile: ExecutionProfile = ExecutionProfile.LOCAL_QUICK
    auto_upgrade_to_cluster_on_no_gpu: bool = False
    writing_mode: WritingMode = WritingMode.DIRECT
    deterministic_writing_fallback: bool = False
    deterministic_review_fallback: bool = True
    llm_writing_stage_planner: bool = False
    writing_tool_max_rounds: int = 2  # keep writing requests bounded on OpenAI-compatible endpoints
    auto_create_env: bool = True
    interactive_env_select: bool = False
    auto_download_resources: bool = True
    local_execution_timeout: int = 1800
    runtime_auto_install_enabled: bool = True
    runtime_auto_install_max_packages: int = 50
    runtime_auto_install_max_nltk_downloads: int = 50
    runtime_auto_install_allowlist: list[str] = Field(default_factory=list)
    github_clone_protocol: str = "ssh"  # "ssh" or "https"

    # Literature/search backends. Keep OpenAlex as the default public-release
    # backend; users can opt into arXiv/Semantic Scholar/PapersWithCode/web
    # in config.json when they have API keys and accept those dependencies.
    literature_sources: list[str] = Field(default_factory=lambda: ["openalex"])

    # Adaptive memory and skill evolution settings
    memory_enabled: bool = True
    memory_evolution_enabled: bool = True
    memory_retrieval_top_k: int = 5
    direction_memory_top_k: int = 4
    strategy_memory_top_k: int = 4
    memory_decay_factor: float = 0.08
    skill_evolution_enabled: bool = True
    skill_retrieval_top_k: int = 5
    script_skill_autorun_policy: str = "safe_only"
    static_skills_dir: str = ""
    static_skills_dirs: list[str] = Field(default_factory=list)
    vendored_skills_manifest: str = ""

    # Router-policy evolution settings. Evo mode enables the same-router
    # hindsight surface so feedback can update memory, skills, and planner
    # routing decisions across stages.
    same_router_hindsight_sdpo_enabled: bool = False
    router_planner_enabled: bool = True
    router_sdpo_model_path: str = ""
    router_sdpo_model_name: str = ""
    router_sdpo_base_url: str = ""
    router_sdpo_api_key: str = ""
    router_sdpo_max_new_tokens: int = 512
    router_sdpo_temperature: float = 0.0
    router_sdpo_timeout: float = 120.0

    # RAM (Reflection-Augmentation Model) settings
    ram_enabled: bool = False
    ram_model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    ram_backend: str = "hf"  # "hf" (HuggingFace Transformers) or "vllm" (vLLM HTTP API)
    ram_vllm_url: str = ""
    ram_max_new_tokens: int = 1024
    ram_temperature: float = 0.3
    ram_device: str = "auto"
    ram_data_collection_enabled: bool = True
    ram_checkpoint_path: str = ""  # LoRA adapter path (empty = base model)
    ram_subsystems: list[str] = Field(
        default_factory=lambda: ["method_gen", "code_impl", "paper_writing"]
    )

    # Environment backend for experiment execution.
    # "auto" — prefer conda/mamba when available, fall back to venv.
    # "conda" — force conda (error if not installed).
    # "venv"  — always use isolated venv (+ pip CUDA wheel pre-install).
    environment_backend: str = "auto"

    # Use an existing NAMED conda env instead of creating a per-session env.
    # When set, this env is used directly (shared across sessions).
    # Takes priority over environment_backend auto-detection.
    experiment_conda_env: str = ""  # e.g., "myenv"

    # Point to a user-managed Python environment. Accepts:
    #   - python executable path:  "D:/anaconda/envs/myenv/python.exe"
    #   - environment directory:   "D:/projects/.venv"  (auto-finds python inside)
    #   - conda env name:          "myenv"  (resolved via conda)
    # When set, skips all auto-creation/install — uses the environment as-is.
    # Takes highest priority over all other environment settings.
    experiment_python: str = ""

    # Multi-model review committee (optional).
    # Each entry: {"role": str, "focus": str, "model": str,
    #              "base_url": str|None, "api_key": str|None, "weight": float}
    # Empty list → single-model review (backward compatible).
    review_committee: list[dict] = Field(default_factory=list)

    # Cluster execution settings (optional — set in config.json under "research.cluster")
    cluster: dict = Field(default_factory=dict)  # {"enabled":true, "host":..., "user":..., ...}

    # Iteration settings for experiment agent
    experiment_max_rounds: int = 3           # maximum iteration rounds
    experiment_plateau_patience: int = 2     # consecutive rounds with < threshold improvement
    experiment_improvement_threshold: float = 0.005  # 0.5% minimum improvement

    # ReAct experiment mode: "pipeline" (default, hardcoded phases) or "react" (LLM-driven tools)
    experiment_mode: str = "pipeline"
    # Max tool-call rounds in react mode (each round = one LLM ↔ tool exchange)
    react_max_rounds: int = 80
    # SLURM settings for react mode (auto-detected if empty)
    slurm_partition: str = ""                # SLURM partition (auto-detected if empty)
    slurm_max_gpus: int = 2                 # max GPUs per job
    slurm_default_time: str = "30-00:00:00"  # default wall time (30 days)
    # Container settings for react mode (for clusters with old glibc)
    container_image: str = ""               # e.g., "docker://ubuntu:22.04" (clean base with glibc 2.35)
    container_path: str = ""                # e.g., "/path/to/ubuntu2204.sif"
    container_bind: str = "/mnt:/mnt"       # bind mounts for apptainer

    @classmethod
    def load(cls, path: Path | None = None) -> "ResearchConfig":
        """Load config from nanoresearch config file, then overlay env vars."""
        if path is None:
            path = get_config_path()

        research: dict = {}
        if path.is_file():
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot read config file {path}: {exc}"
                ) from exc
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Config file {path} contains invalid JSON: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Config file {path} must be a JSON object, got {type(data).__name__}"
                )
            research = data.get("research", {})

        cls._overlay_private_endpoints(research)

        try:
            cfg = cls.model_validate(research)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid config values in {path}: {exc}"
            ) from exc

        # Environment variable overrides (highest priority)
        if env_url := os.environ.get("NANORESEARCH_BASE_URL"):
            cfg.base_url = env_url.strip()
        if env_key := os.environ.get("NANORESEARCH_API_KEY"):
            cfg.api_key = env_key.strip()
        if env_timeout := os.environ.get("NANORESEARCH_TIMEOUT"):
            try:
                cfg.timeout = float(env_timeout)
            except ValueError:
                logger.warning(
                    "NANORESEARCH_TIMEOUT=%r is not a valid float, using default %.1f",
                    env_timeout, cfg.timeout,
                )

        if not cfg.base_url or not cfg.api_key:
            raise ValueError(
                "API credentials required. Set NANORESEARCH_BASE_URL and "
                "NANORESEARCH_API_KEY environment variables, or configure "
                "them in ~/.nanoresearch/config.json under 'research'."
            )
        return cfg

    @staticmethod
    def _overlay_private_endpoints(research: dict) -> None:
        """Apply local-only endpoint routing from ~/.nanoresearch/private_endpoints.json.

        This file is intentionally outside the repository and is never included
        in config snapshots. It lets local runs route code/chart prompt stages to
        a private text gateway and image generation to a separate image gateway.
        Missing or malformed files are ignored so public users can rely on the
        normal config.json/env-var path.
        """
        private_path = get_private_endpoints_path()
        if not private_path.is_file():
            return
        try:
            data = json.loads(private_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid private endpoints file %s: %s", private_path, exc)
            return
        if not isinstance(data, dict):
            return

        text_cfg = data.get("text_newapi")
        if isinstance(text_cfg, dict):
            base_url = str(text_cfg.get("base_url") or "").strip()
            api_key = str(text_cfg.get("api_key") or "").strip()
            models = text_cfg.get("models") if isinstance(text_cfg.get("models"), dict) else {}
            for stage in ("ideation", "planning", "experiment", "evidence_extraction", "code_gen", "figure_code", "figure_prompt", "writing", "review", "revision"):
                stage_cfg = research.setdefault(stage, {})
                if not isinstance(stage_cfg, dict):
                    stage_cfg = {}
                    research[stage] = stage_cfg
                if base_url:
                    stage_cfg["base_url"] = base_url
                if api_key:
                    stage_cfg["api_key"] = api_key
                model = str(models.get(stage) or models.get("default_text") or "").strip()
                if model:
                    stage_cfg["model"] = model

        image_cfg = data.get("image_service")
        if isinstance(image_cfg, dict):
            stage_cfg = research.setdefault("figure_gen", {})
            if not isinstance(stage_cfg, dict):
                stage_cfg = {}
                research["figure_gen"] = stage_cfg
            for key in ("base_url", "api_key", "model", "image_backend"):
                value = str(image_cfg.get(key) or "").strip()
                if value:
                    stage_cfg[key] = value

    def for_stage(self, stage_name: str) -> StageModelConfig:
        """Return model config for the given stage name."""
        stage_name = stage_name.lower()
        mapping = {
            "ideation": self.ideation,
            "planning": self.planning,
            "experiment": self.experiment,
            "writing": self.writing,
            "code_gen": self.code_gen,
            "figure_prompt": self.figure_prompt,
            "figure_code": self.figure_code,
            "figure_gen": self.figure_gen,
            "evidence_extraction": self.evidence_extraction,
            "review": self.review,
            "revision": self.revision,
        }
        if stage_name not in mapping:
            raise ValueError(f"Unknown stage: {stage_name}. Valid: {list(mapping)}")
        return mapping[stage_name]

    def prefers_cluster_execution(self) -> bool:
        """Whether the unified pipeline should prefer SLURM/cluster execution."""
        return self.execution_profile == ExecutionProfile.CLUSTER_FULL or bool(
            self.cluster and self.cluster.get("enabled")
        )

    def should_use_writing_tools(self, heading: str) -> bool:
        """Decide whether a section should use tool-augmented writing."""
        if self.writing_mode == WritingMode.DIRECT:
            return False
        if self.writing_mode == WritingMode.REACT:
            return True

        hybrid_sections = {"Introduction", "Related Work", "Method", "Experiments", "Results"}
        if self.execution_profile == ExecutionProfile.FAST_DRAFT:
            hybrid_sections = {"Introduction", "Related Work", "Method"}
        return heading.strip() in hybrid_sections

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot for manifest storage.

        Strips all API keys (global and per-stage) to prevent accidental leaks.
        """
        d = self.model_dump(mode="json")
        d.pop("api_key", None)  # don't persist global API key
        # Also strip per-stage api_key overrides
        for key, val in d.items():
            if isinstance(val, dict) and "api_key" in val:
                val.pop("api_key", None)
        # Strip api_key from review_committee entries
        for entry in d.get("review_committee", []):
            if isinstance(entry, dict):
                entry.pop("api_key", None)
        return d
