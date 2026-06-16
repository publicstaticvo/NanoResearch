"""Experiment stage data models: blueprint, datasets, metrics, ablations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ComputeRequirements(BaseModel):
    """Estimated compute resources needed for the experiment."""

    gpu_type: str = Field(default="", description="e.g. 'A100', 'V100'")
    num_gpus: int = Field(default=1, ge=0)
    estimated_hours: float = Field(default=0.0, ge=0.0)
    memory_gb: float = Field(default=0.0, ge=0.0)
    notes: str = ""

    @field_validator("gpu_type", "notes", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class Dataset(BaseModel):
    """A dataset to be used in experiments."""

    name: str
    description: str = ""
    source_url: str = ""
    size_info: str = ""
    preprocessing_notes: str = ""

    @field_validator("name", "description", "source_url", "size_info", "preprocessing_notes", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class Baseline(BaseModel):
    """A baseline method for comparison."""

    name: str
    description: str = ""
    reference_paper_id: str = ""
    expected_performance: dict[str, Any] = Field(default_factory=dict)
    performance_provenance: dict[str, str] = Field(
        default_factory=dict,
        description="metric_name → source description (e.g. 'Table 2 of arxiv:2401.00001')",
    )
    is_projected: dict[str, bool] = Field(
        default_factory=dict,
        description="metric_name → True if value is projected rather than published",
    )

    @field_validator("name", "description", "reference_paper_id", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class Metric(BaseModel):
    """An evaluation metric."""

    name: str
    description: str = ""
    higher_is_better: bool = True
    primary: bool = False

    @field_validator("name", "description", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class AblationGroup(BaseModel):
    """A group of ablation experiments."""

    group_name: str
    description: str = ""
    variants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Each variant is a dict describing what is removed/changed",
    )

    @field_validator("group_name", "description", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class ExperimentRunSpec(BaseModel):
    """One machine-checkable run required by the blueprint contract."""

    run_id: str
    role: str = Field(description="proposed, baseline, ablation, optimization, or complexity")
    method: str
    dataset: str = ""
    metrics: list[str] = Field(default_factory=list)
    required: bool = True
    output_group: str = "main_results"
    expected_artifacts: list[str] = Field(default_factory=lambda: ["results/metrics.json"])
    failure_policy: str = "debug_then_degrade"
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "role", "method", "dataset", "output_group", "failure_policy", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""

    @field_validator("metrics", "expected_artifacts", mode="before")
    @classmethod
    def _coerce_str_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        return [str(v)]


class MinimumSuccessCriteria(BaseModel):
    """Contract thresholds used to determine whether a run is complete enough."""

    min_measured_baselines: int = 2
    min_ablation_runs: int = 2
    require_proposed: bool = True
    require_complexity: bool = False
    require_optimization_history: bool = False
    required_metrics: list[str] = Field(default_factory=list)

    @field_validator("required_metrics", mode="before")
    @classmethod
    def _coerce_str_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        return [str(v)]


class ExperimentBlueprint(BaseModel):
    """Complete experiment plan generated by the planning agent."""

    title: str
    hypothesis_ref: str = Field(description="hypothesis_id from IdeationOutput")
    datasets: list[Dataset] = Field(default_factory=list)
    baselines: list[Baseline] = Field(default_factory=list)
    proposed_method: dict = Field(
        default_factory=dict,
        description="Structured description of the proposed method",
    )
    metrics: list[Metric] = Field(default_factory=list)
    ablation_groups: list[AblationGroup] = Field(default_factory=list)
    compute_requirements: ComputeRequirements = Field(
        default_factory=ComputeRequirements,
        description="Estimated compute needs (GPUs, hours, etc.)",
    )
    evidence_summary: str = Field(
        default="",
        description="Summary of quantitative evidence from literature used in this blueprint",
    )
    data_provenance_note: str = Field(
        default="",
        description="Note explaining which numbers are from published results vs projected",
    )
    experiment_matrix: list[ExperimentRunSpec] = Field(
        default_factory=list,
        description="Machine-checkable run matrix covering proposed, measured baselines, ablations, optimization, and complexity runs.",
    )
    required_artifacts: list[str] = Field(
        default_factory=lambda: [
            "configs/experiment_matrix.json",
            "results/metrics.json",
            "results/run_manifest.json",
            "results/final_metrics.json",
        ],
        description="Core artifacts that coding/execution should materialize; additional diagnostic artifacts are optional and topic-dependent.",
    )
    minimum_success_criteria: MinimumSuccessCriteria = Field(
        default_factory=MinimumSuccessCriteria,
        description="Minimum measured evidence required before making experimental claims.",
    )

    @field_validator("compute_requirements", mode="before")
    @classmethod
    def _coerce_compute_requirements(cls, v):
        if v is None:
            return ComputeRequirements()
        if isinstance(v, dict):
            return ComputeRequirements(**v)
        return v

    @field_validator("minimum_success_criteria", mode="before")
    @classmethod
    def _coerce_minimum_success_criteria(cls, v):
        if v is None:
            return MinimumSuccessCriteria()
        if isinstance(v, dict):
            return MinimumSuccessCriteria(**v)
        return v

    @field_validator("required_artifacts", mode="before")
    @classmethod
    def _coerce_required_artifacts(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        return [str(v)]

    @field_validator("title", "hypothesis_ref", "evidence_summary", "data_provenance_note", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class MetricResult(BaseModel):
    """A single metric measurement."""

    metric_name: str
    value: float
    std: float | None = None
    num_runs: int = 1


class MethodResult(BaseModel):
    """Results for one method on one dataset."""

    method_name: str
    dataset: str = ""
    is_proposed: bool = False
    metrics: list[MetricResult] = Field(default_factory=list)

    @field_validator("method_name", "dataset", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class AblationResult(BaseModel):
    """Results for one ablation variant."""

    variant_name: str
    metrics: list[MetricResult] = Field(default_factory=list)


class TrainingLogEntry(BaseModel):
    """One epoch of training log."""

    epoch: int
    train_loss: float | None = None
    val_loss: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentResults(BaseModel):
    """Structured experiment results from quick-eval."""

    main_results: list[MethodResult] = Field(default_factory=list)
    ablation_results: list[AblationResult] = Field(default_factory=list)
    training_log: list[TrainingLogEntry] = Field(default_factory=list)
