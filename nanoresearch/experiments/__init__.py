"""Experiment protocol, aggregation, and runner helpers for NanoResearch studies."""

from .deep_persona_runner import (
    DEFAULT_PERSONA_BRIEFS,
    build_assignment_topic,
    build_result_record,
    resolve_variant_runtime_settings,
    run_assignment,
    run_manifest,
)
from .elastic_scheduler import (
    default_worker_id,
    run_elastic_manifest,
)
from .router_persona_eval import (
    APPENDIX_VARIANTS,
    CORE_METRICS,
    DEFAULT_PERSONA_IDS,
    EFFICIENCY_METRICS,
    MAIN_VARIANTS,
    aggregate_experiment_results,
    build_experiment_manifest,
)

__all__ = [
    'APPENDIX_VARIANTS',
    'CORE_METRICS',
    'DEFAULT_PERSONA_BRIEFS',
    'DEFAULT_PERSONA_IDS',
    'EFFICIENCY_METRICS',
    'MAIN_VARIANTS',
    'aggregate_experiment_results',
    'build_assignment_topic',
    'build_experiment_manifest',
    'build_result_record',
    'default_worker_id',
    'resolve_variant_runtime_settings',
    'run_assignment',
    'run_elastic_manifest',
    'run_manifest',
]
