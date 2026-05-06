from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any, Iterable

from .canonical_baselines import lookup_canonical_baseline
from nanoresearch.idea_utils import get_selected_idea_id
from .router_persona_eval import VARIANT_BY_NAME

DEFAULT_PERSONA_BRIEFS = {
    'ai4science_journal_conservative': 'AI4Science journal persona that prefers conservative claims, careful ablations, and strong scientific grounding.',
    'ai4science_reproducibility_first': 'AI4Science persona that prioritizes reproducibility, exact reruns, and explicit experimental controls over flashy novelty.',
    'benchmark_maximalist_conference': 'Conference persona that values strong benchmark coverage, broad comparisons, and clear wins on established leaderboards.',
    'cv_fast_iteration_builder': 'Computer vision persona that prefers fast iteration loops, practical implementation choices, and quick empirical feedback.',
    'cv_visual_benchmark_heavy': 'Computer vision persona that cares about visual quality, benchmark-heavy evaluation, and polished quantitative reporting.',
    'journal_evidence_first_writer': 'Journal-writing persona that insists on evidence-first argumentation, measured claims, and high traceability from result to narrative.',
    'multimodal_systems_engineer': 'Multimodal systems persona that values robust system design, component interfaces, and implementation realism.',
    'nlp_conference_exploratory': 'NLP conference persona that welcomes exploratory ideas and broader idea search, while still expecting solid experiments.',
    'nlp_conference_pragmatic': 'NLP conference persona that prefers pragmatic, ablatable methods with clean implementation paths and reviewer-friendly framing.',
    'resource_constrained_repro_first': 'Compute-limited persona that prioritizes reproducibility, strict budget accounting, and lightweight experiment plans.',
}

PERSONA_PROFILE_SPECS = {
    'ai4science_journal_conservative': {
        'seed': 'ai4science_journal',
        'overrides': {
            'research_profile': {
                'method_preference': 'Prefer conservative, scientifically grounded methods with careful controls and restrained claims.',
                'risk_preference': 'low',
                'baseline_ablation_strictness': 'high',
            },
            'writing_profile': {'tone': 'highly restrained', 'claim_strength': 'conservative'},
        },
    },
    'ai4science_reproducibility_first': {
        'seed': 'ai4science_journal',
        'overrides': {
            'research_profile': {
                'method_preference': 'Prefer exact reruns, explicit controls, and reproducible ablations over speculative novelty.',
                'risk_preference': 'low',
                'baseline_ablation_strictness': 'very high',
            },
            'resource_profile': {'feasibility_bias': 'Prefer explicit reproducibility steps, deterministic settings, and auditable experiment plans.'},
            'interaction_profile': {'priority_feedback': 'Missing controls, weak reproducibility details, or hidden implementation variance.'},
        },
    },
    'benchmark_maximalist_conference': {
        'seed': 'high_novelty_exploratory',
        'overrides': {
            'research_profile': {'method_preference': 'Prefer methods that can support broad benchmark coverage and strong leaderboard-facing comparisons.'},
            'publication_profile': {'venue_style': 'benchmark-heavy conference', 'figure_style': 'dense benchmark tables and comparison plots'},
            'interaction_profile': {'priority_feedback': 'Weak benchmark coverage or insufficient comparison breadth.'},
        },
    },
    'cv_fast_iteration_builder': {
        'seed': 'cv_visual_conference',
        'overrides': {
            'resource_profile': {'gpu_budget': '1xA100 80GB', 'wall_clock_budget': '2 days', 'feasibility_bias': 'Prefer short iteration loops and implementation-practical experiments.'},
            'research_profile': {'method_preference': 'Prefer practical CV methods that can be validated quickly with strong debugging loops.'},
        },
    },
    'cv_visual_benchmark_heavy': {
        'seed': 'cv_visual_conference',
        'overrides': {
            'publication_profile': {'figure_style': 'high-quality qualitative panels plus benchmark-heavy quantitative plots'},
            'interaction_profile': {'priority_feedback': 'Weak qualitative evidence, weak visual storytelling, or thin benchmark coverage.'},
        },
    },
    'journal_evidence_first_writer': {
        'seed': 'ai4science_journal',
        'overrides': {
            'writing_profile': {'tone': 'evidence-first academic', 'claim_strength': 'conservative', 'section_organization': 'Journal-style with traceable evidence and explicit limitations.'},
            'interaction_profile': {'priority_feedback': 'Claims that outrun the evidence or weak result-to-narrative traceability.'},
        },
    },
    'multimodal_systems_engineer': {
        'seed': 'high_novelty_exploratory',
        'overrides': {
            'research_profile': {'domain': 'Multimodal Systems', 'method_preference': 'Prefer robust component interfaces, system decomposition, and implementation realism.'},
            'resource_profile': {'feasibility_bias': 'Prefer designs with clear system boundaries, measurable failure modes, and executable integration plans.'},
            'publication_profile': {'figure_style': 'system diagrams plus concise quantitative plots'},
        },
    },
    'nlp_conference_exploratory': {
        'seed': 'high_novelty_exploratory',
        'overrides': {
            'research_profile': {'domain': 'NLP', 'method_preference': 'Welcome broader idea search and exploratory directions so long as evaluation remains falsifiable.'},
            'publication_profile': {'venue_style': 'NLP conference'},
        },
    },
    'nlp_conference_pragmatic': {
        'seed': 'nlp_conference',
        'overrides': {
            'research_profile': {'method_preference': 'Prefer pragmatic NLP ideas with clean ablations, implementable plans, and reviewer-friendly framing.'},
            'interaction_profile': {'priority_feedback': 'Needlessly complex methods or ablations that do not clarify the core contribution.'},
        },
    },
    'resource_constrained_repro_first': {
        'seed': 'resource_constrained_pragmatic',
        'overrides': {
            'research_profile': {'method_preference': 'Prefer lightweight, reproducible methods with strict budget accounting.'},
            'resource_profile': {'feasibility_bias': 'Strongly prefer low-cost experiment plans with explicit compute accounting and rerun paths.'},
            'interaction_profile': {'priority_feedback': 'Anything that increases cost or variance without a clearly justified benefit.'},
        },
    },
}

SKIPPED_DEEP_STAGES = ('FIGURE_GEN', 'WRITING', 'REVIEW')
METHOD_TO_CODE_STAGES = ('IDEATION', 'PLANNING', 'SETUP', 'CODING', 'EXECUTION')
IMPLEMENTATION_STAGES = ('SETUP', 'CODING', 'EXECUTION')
_ALIGNMENT_SYSTEM_PROMPT = (
    'You are an expert research evaluator. Score how well the generated idea and experiment blueprint satisfy the stated user requirements. '
    'Return JSON only with keys alignment_score and feedback. '
    'Use a 1-10 integer scale: 1-2 means the output largely ignores or contradicts the requirements; '
    '3-4 means it addresses the task only superficially or misses major constraints; '
    '5-6 means it satisfies the main task but misses some important preferences, baselines, datasets, feasibility constraints, or ablations; '
    '7-8 means it satisfies the requirements well with only minor omissions; '
    '9-10 means it strongly satisfies the requirements with clear alignment to the user profile, dataset/baseline constraints, feasibility budget, and requested ablation/evaluation design. '
    'Be strict and concise.'
)
_NOVELTY_SYSTEM_PROMPT = (
    'You are an expert research evaluator. Score the novelty of the proposed idea relative to the provided baselines on a 1-10 scale. '
    'Use the following rubric: 1-2 = near-duplicate of the baselines with only superficial wording, hyperparameter, or training-detail changes; '
    '3-4 = weak incremental modification with high overlap in core method and contribution; '
    '5-6 = moderate incremental novelty with one clear local change such as a new module, loss, training strategy, or recombination of known components; '
    '7-8 = clearly recognizable novelty with a substantively different mechanism, method structure, or contribution logic relative to the baselines; '
    '9-10 = strong novelty with a non-trivial and clearly distinct core idea, not just module swapping or routine recombination. '
    'Judge primarily against the provided baselines, focus on the core mechanism rather than surface complexity, and do not over-score backbone swaps, tuning, regularization, or data augmentation. '
    'Return JSON only with keys novelty_score, closest_baseline, rationale.'
)
_STATUS_SUCCESS = {'success', 'completed', 'ok', 'passed'}


@dataclass(frozen=True)
class RunArtifacts:
    workspace_path: str
    topic: str
    ideation_output: dict[str, Any]
    blueprint: dict[str, Any]
    execution_output: dict[str, Any]
    analysis_output: dict[str, Any]
    cost_summary: dict[str, Any]


@dataclass(frozen=True)
class AssignmentOutcome:
    record: dict[str, Any]
    alignment_attempts: int
    workspace_paths: list[str]


def resolve_variant_runtime_settings(variant_name: str, assignment: dict[str, Any] | None = None) -> dict[str, Any]:
    if variant_name not in VARIANT_BY_NAME:
        raise ValueError(f'Unknown variant_name: {variant_name}')
    variant = VARIANT_BY_NAME[variant_name]
    settings = {
        'memory_enabled': bool(variant.memory_self_evolution),
        'memory_evolution_enabled': bool(variant.memory_self_evolution),
        'skill_evolution_enabled': bool(variant.skill_self_evolution),
        'same_router_hindsight_sdpo': bool(variant.same_router_hindsight_sdpo),
        'appendix_only': bool(variant.appendix_only),
    }
    flags = dict((assignment or {}).get('component_flags') or {})
    if 'memory_self_evolution' in flags:
        settings['memory_enabled'] = bool(flags['memory_self_evolution'])
        settings['memory_evolution_enabled'] = bool(flags['memory_self_evolution'])
    if 'skill_self_evolution' in flags:
        settings['skill_evolution_enabled'] = bool(flags['skill_self_evolution'])
    if 'same_router_hindsight_sdpo' in flags:
        settings['same_router_hindsight_sdpo'] = bool(flags['same_router_hindsight_sdpo'])
    if 'appendix_only' in flags:
        settings['appendix_only'] = bool(flags['appendix_only'])
    return settings


def build_assignment_topic(assignment: dict[str, Any]) -> str:
    question = dict(assignment.get('question') or {})
    persona_id = str(assignment.get('persona_id') or '').strip()
    persona_brief = str(
        question.get('persona_brief')
        or DEFAULT_PERSONA_BRIEFS.get(persona_id)
        or persona_id.replace('_', ' ')
    ).strip()
    baselines = '; '.join(str(item) for item in question.get('baselines', []) if str(item).strip())
    datasets = '; '.join(str(item) for item in question.get('datasets', []) if str(item).strip())
    background = str(question.get('background') or '').strip()
    problem_statement = str(question.get('problem_statement') or background).strip()
    user_requirements = str(question.get('user_requirements') or '').strip()
    extra_context = str(question.get('extra_context') or '').strip()

    lines = [
        f"Evaluation Question ID: {question.get('question_id', '')}",
        f"Persona Profile: {persona_brief}",
        f"Research Domain: {question.get('domain', '')}",
        f"Difficulty: {question.get('difficulty', '')}",
        f"Problem Statement: {problem_statement}",
        f"Background Context: {background}",
        f"Known Baselines: {baselines}",
        f"Evaluation Datasets: {datasets}",
        f"User Requirements: {user_requirements}",
    ]
    if extra_context:
        lines.append(f'Additional Context: {extra_context}')
    lines.append(
        'Task: Propose a new research idea, turn it into a rigorous experimental plan, implement the resulting experiment, and analyze the final outcome.'
    )
    return '\n'.join(lines)


def build_result_record(
    *,
    assignment: dict[str, Any],
    workspace_path: str,
    blueprint: dict[str, Any],
    experiment_output: dict[str, Any],
    analysis_output: dict[str, Any],
    cost_summary: dict[str, Any],
    alignment_judgment: dict[str, Any],
    novelty_judgment: dict[str, Any],
    alignment_token_to_pass: int,
) -> dict[str, Any]:
    primary_metric_name, higher_is_better = _select_primary_metric(blueprint, experiment_output, analysis_output)
    final_performance = _extract_final_performance(primary_metric_name, experiment_output, analysis_output)
    canonical_baseline = lookup_canonical_baseline(
        str((assignment.get('question') or {}).get('question_id') or assignment.get('question_id') or ''),
        primary_metric_name,
    )
    if canonical_baseline is not None:
        higher_is_better = bool(canonical_baseline.get('higher_is_better', higher_is_better))
        baseline_performance = _to_optional_float(canonical_baseline.get('baseline_value'))
    else:
        baseline_performance = _extract_baseline_performance(primary_metric_name, higher_is_better, blueprint)
    final_performance, baseline_performance = _normalize_metric_scale_pair(final_performance, baseline_performance)
    delta = _compute_delta(final_performance, baseline_performance, higher_is_better)
    implementation_success = _is_successful_implementation(experiment_output)

    return {
        'assignment_id': assignment.get('assignment_id'),
        'chain_id': assignment.get('chain_id'),
        'evolution_round': assignment.get('evolution_round'),
        'evolution_total_rounds': assignment.get('evolution_total_rounds'),
        'persona_id': assignment.get('persona_id'),
        'variant_name': assignment.get('variant_name'),
        'question_id': (assignment.get('question') or {}).get('question_id'),
        'workspace_path': workspace_path,
        'primary_metric_name': primary_metric_name,
        'novelty_score': _to_optional_float(novelty_judgment.get('novelty_score')),
        'novelty_closest_baseline': novelty_judgment.get('closest_baseline'),
        'alignment_score': _to_optional_float(alignment_judgment.get('alignment_score') or alignment_judgment.get('score')),
        'alignment_pass_at_1': bool(alignment_judgment.get('pass_at_1')),
        'alignment_feedback': str(alignment_judgment.get('feedback') or ''),
        'alignment_token_to_pass': int(alignment_token_to_pass),
        'plan_executability': bool(blueprint) and bool(experiment_output),
        'implementation_token_to_runnable': _sum_stage_tokens(cost_summary, IMPLEMENTATION_STAGES) if implementation_success else None,
        'implementation_success': implementation_success,
        'final_performance': final_performance,
        'baseline_performance': baseline_performance,
        'delta_over_baseline': delta,
        'total_tokens_from_method_to_code': _sum_stage_tokens(cost_summary, METHOD_TO_CODE_STAGES),
        'metadata': {
            'variant_runtime': resolve_variant_runtime_settings(str(assignment.get('variant_name') or ''), assignment),
            'cost_summary_path': str(Path(workspace_path) / 'logs' / 'cost_summary.json'),
            'blueprint_path': str(Path(workspace_path) / 'plans' / 'experiment_blueprint.json'),
            'execution_output_path': str(Path(workspace_path) / 'plans' / 'execution_output.json'),
            'analysis_output_path': str(Path(workspace_path) / 'plans' / 'analysis_output.json'),
            'experiment_status': _extract_status(experiment_output),
            'canonical_baseline_applied': canonical_baseline is not None,
            'canonical_baseline_name': canonical_baseline.get('baseline_name') if canonical_baseline else None,
            'canonical_baseline_metric_name': canonical_baseline.get('metric_name') if canonical_baseline else None,
            'canonical_baseline_provenance': canonical_baseline.get('provenance_uri') if canonical_baseline else None,
        },
    }


async def run_assignment(
    assignment: dict[str, Any],
    *,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    max_alignment_retries: int = 1,
    disable_ideation_retrieval: bool = False,
) -> AssignmentOutcome:
    settings = resolve_variant_runtime_settings(str(assignment.get('variant_name') or ''), assignment)
    ResearchConfig, Workspace, UnifiedPipelineOrchestrator, PipelineMode, PaperMode, build_profile_seed, save_user_profile = _load_runtime_symbols()
    assignment_slug = _slugify(str(assignment.get('assignment_id') or 'assignment'))
    assignment_root = Path(output_dir) / assignment_slug
    assignment_root.mkdir(parents=True, exist_ok=True)
    chain_slug = _slugify(str(assignment.get('chain_id') or assignment.get('assignment_id') or 'assignment'))
    chain_root = Path(output_dir) / '_chains' / chain_slug
    chain_root.mkdir(parents=True, exist_ok=True)
    nanoresearch_home = chain_root / 'nanoresearch_home'
    previous_home = os.environ.get('NANORESEARCH_HOME')
    os.environ['NANORESEARCH_HOME'] = str(nanoresearch_home)

    try:
        save_user_profile(_build_persona_profile(assignment, build_profile_seed))
        resolved_config_path = _resolve_assignment_config_path(assignment, config_path)
        base_config = ResearchConfig.load(Path(resolved_config_path) if resolved_config_path else None)
        if disable_ideation_retrieval:
            base_config.ideation_disable_retrieval = True

        alignment_feedback = ''
        alignment_token_total = 0
        attempts = 0
        workspace_paths: list[str] = []
        artifacts: RunArtifacts | None = None
        alignment_judgment: dict[str, Any] = {'alignment_score': 10.0, 'pass_at_1': True, 'feedback': ''}
        novelty_judgment: dict[str, Any] = {'novelty_score': None, 'closest_baseline': None}

        for attempt_index in range(1, max_alignment_retries + 2):
            attempts = attempt_index
            attempt_topic = build_assignment_topic(assignment)
            if alignment_feedback:
                attempt_topic += (
                    '\n\nAlignment Retry Feedback:\n'
                    f'{alignment_feedback}\n'
                    'Revise the idea and experiment plan so they satisfy the requirements more precisely.'
                )
            variant_config = _apply_variant_overrides(base_config, settings)
            workspace = Workspace.create(
                topic=attempt_topic,
                config_snapshot=_safe_model_dump(variant_config),
                root=assignment_root / 'workspaces',
                session_id=f'attempt-{attempt_index:02d}',
                pipeline_mode=PipelineMode.DEEP,
                paper_mode=PaperMode.ORIGINAL_RESEARCH,
            )
            _write_assignment_context(
                workspace=workspace,
                assignment=assignment,
                assignment_root=assignment_root,
                chain_root=chain_root,
                output_dir=Path(output_dir),
                attempt_index=attempt_index,
            )
            workspace_paths.append(str(workspace.path))
            orchestrator = UnifiedPipelineOrchestrator(workspace, variant_config)
            try:
                result = await orchestrator.run(attempt_topic)
            finally:
                await orchestrator.close()

            artifacts = RunArtifacts(
                workspace_path=str(workspace.path),
                topic=attempt_topic,
                ideation_output=_ensure_dict(result.get('ideation_output')) or _safe_read_json(workspace, 'papers/ideation_output.json'),
                blueprint=_ensure_dict(result.get('experiment_blueprint')) or _safe_read_json(workspace, 'plans/experiment_blueprint.json'),
                execution_output=_ensure_dict(result.get('execution_output')) or _safe_read_json(workspace, 'plans/execution_output.json'),
                analysis_output=_ensure_dict(result.get('analysis_output')) or _safe_read_json(workspace, 'plans/analysis_output.json'),
                cost_summary=_ensure_dict(result.get('cost_summary')) or _safe_read_json(workspace, 'logs/cost_summary.json'),
            )

            alignment_token_total += _sum_stage_tokens(artifacts.cost_summary, ('IDEATION', 'PLANNING'))
            alignment_judgment = _judge_alignment(base_config, assignment, artifacts.ideation_output, artifacts.blueprint)
            novelty_judgment = _judge_novelty(base_config, assignment, artifacts.ideation_output, artifacts.blueprint)

            if alignment_judgment.get('pass_at_1'):
                break
            alignment_feedback = str(alignment_judgment.get('feedback') or '').strip()

        if artifacts is None:
            raise RuntimeError(f'No pipeline attempt was executed for assignment {assignment.get("assignment_id")}')

        record = build_result_record(
            assignment=assignment,
            workspace_path=artifacts.workspace_path,
            blueprint=artifacts.blueprint,
            experiment_output=artifacts.execution_output,
            analysis_output=artifacts.analysis_output,
            cost_summary=artifacts.cost_summary,
            alignment_judgment=alignment_judgment,
            novelty_judgment=novelty_judgment,
            alignment_token_to_pass=alignment_token_total,
        )
        record['metadata']['alignment_attempts'] = attempts
        record['metadata']['workspace_paths'] = workspace_paths
        record['metadata']['nanoresearch_home'] = str(nanoresearch_home)
        record['metadata']['evolution_chain_id'] = str(assignment.get('chain_id') or assignment.get('assignment_id') or '')
        record['metadata']['config_path'] = str(resolved_config_path or '')
        record['metadata']['evolution_round'] = int(assignment.get('evolution_round') or 1)
        record['metadata']['evolution_total_rounds'] = int(assignment.get('evolution_total_rounds') or 1)
        record['metadata']['persona_profile_path'] = str(nanoresearch_home / 'profile' / 'profile.json')
        (assignment_root / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        return AssignmentOutcome(record=record, alignment_attempts=attempts, workspace_paths=workspace_paths)
    finally:
        if previous_home is None:
            os.environ.pop('NANORESEARCH_HOME', None)
        else:
            os.environ['NANORESEARCH_HOME'] = previous_home


async def run_manifest(
    assignments: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    max_alignment_retries: int = 1,
    disable_ideation_retrieval: bool = False,
) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    results_jsonl = output_path / 'results.jsonl'
    for assignment in assignments:
        outcome = await run_assignment(
            assignment,
            output_dir=output_path,
            config_path=config_path,
            max_alignment_retries=max_alignment_retries,
            disable_ideation_retrieval=disable_ideation_retrieval,
        )
        rows.append(outcome.record)
        with results_jsonl.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(outcome.record, ensure_ascii=False) + '\n')
    return rows


def _load_runtime_symbols():
    from nanoresearch.config import ResearchConfig
    from nanoresearch.pipeline.unified_orchestrator import UnifiedPipelineOrchestrator
    from nanoresearch.pipeline.workspace import Workspace
    from nanoresearch.profile import build_profile_seed, save_user_profile
    from nanoresearch.schemas.manifest import PaperMode, PipelineMode

    return ResearchConfig, Workspace, UnifiedPipelineOrchestrator, PipelineMode, PaperMode, build_profile_seed, save_user_profile


def _resolve_assignment_config_path(
    assignment: dict[str, Any],
    config_path: str | Path | None,
) -> Path | None:
    assignment_config = str(assignment.get('config_path') or '').strip()
    if assignment_config:
        return Path(assignment_config)
    if config_path:
        return Path(config_path)
    return None


def _apply_variant_overrides(base_config: Any, settings: dict[str, Any]) -> Any:
    config = base_config.model_copy(deep=True)
    config.memory_enabled = bool(settings['memory_enabled'])
    config.memory_evolution_enabled = bool(settings['memory_evolution_enabled'])
    config.skill_evolution_enabled = bool(settings['skill_evolution_enabled'])
    config.same_router_hindsight_sdpo_enabled = bool(settings['same_router_hindsight_sdpo'])
    merged_skips = list(dict.fromkeys([*(getattr(config, 'skip_stages', []) or []), *SKIPPED_DEEP_STAGES]))
    config.skip_stages = merged_skips
    return config


def _merge_nested_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_nested_dict(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _build_persona_profile(assignment: dict[str, Any], build_profile_seed) -> dict[str, Any]:
    persona_id = str(assignment.get('persona_id') or '').strip()
    question = dict(assignment.get('question') or {})
    spec = PERSONA_PROFILE_SPECS.get(persona_id, {'seed': 'resource_constrained_pragmatic', 'overrides': {}})
    profile = build_profile_seed(str(spec.get('seed') or 'resource_constrained_pragmatic'))
    overrides = spec.get('overrides') or {}
    if isinstance(overrides, dict):
        _merge_nested_dict(profile, overrides)
    domain = str(question.get('domain') or '').strip()
    if domain:
        profile.setdefault('research_profile', {})['domain'] = domain
    persona_brief = DEFAULT_PERSONA_BRIEFS.get(persona_id)
    if persona_brief:
        profile['persona_brief'] = persona_brief
    profile['profile_id'] = f'persona-{persona_id or "default"}'
    profile['persona_id'] = persona_id
    return profile


def _safe_model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, 'model_dump'):
        return model.model_dump(mode='json')
    if hasattr(model, 'dict'):
        return model.dict()
    return {}


def _write_assignment_context(
    *,
    workspace: Any,
    assignment: dict[str, Any],
    assignment_root: Path,
    chain_root: Path,
    output_dir: Path,
    attempt_index: int,
) -> None:
    question = dict(assignment.get('question') or {})
    workspace.write_json(
        'logs/deep_assignment_context.json',
        {
            'assignment_id': str(assignment.get('assignment_id') or ''),
            'chain_id': str(assignment.get('chain_id') or assignment.get('assignment_id') or ''),
            'persona_id': str(assignment.get('persona_id') or ''),
            'variant_name': str(assignment.get('variant_name') or ''),
            'question_id': str(question.get('question_id') or assignment.get('question_id') or ''),
            'evolution_round': int(assignment.get('evolution_round') or 1),
            'evolution_total_rounds': int(assignment.get('evolution_total_rounds') or 1),
            'workspace_path': str(workspace.path),
            'assignment_root': str(assignment_root),
            'chain_root': str(chain_root),
            'output_dir': str(output_dir),
            'attempt_index': int(attempt_index),
        },
    )


def _safe_read_json(workspace: Any, subpath: str) -> dict[str, Any]:
    try:
        data = workspace.read_json(subpath)
    except Exception:
        return {}
    return _ensure_dict(data)


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _slugify(value: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9._-]+', '-', value).strip('-').lower()
    return slug or 'assignment'


def _sum_stage_tokens(cost_summary: dict[str, Any], stage_names: Iterable[str]) -> int:
    stages = cost_summary.get('stages') if isinstance(cost_summary, dict) else {}
    if not isinstance(stages, dict):
        return 0
    total = 0
    for stage_name in stage_names:
        stage_row = stages.get(stage_name) or {}
        total += int(stage_row.get('total_tokens') or 0)
    return total


def _normalize_metric_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', (name or '').strip().lower())


def _metric_aliases(name: str | None) -> set[str]:
    text = str(name or '').strip()
    if not text:
        return set()
    normalized_text = _normalize_metric_name(text)
    aliases = {normalized_text}
    for paren in re.findall(r'\(([^()]*)\)', text):
        normalized = _normalize_metric_name(paren)
        if normalized:
            aliases.add(normalized)
    lowered = text.lower()
    if 'exact match' in lowered:
        aliases.add('em')
        aliases.add('exactmatch')
    if normalized_text == 'em':
        aliases.add('exactmatch')
    if normalized_text == 'accuracy':
        aliases.add('acc')
    if normalized_text == 'acc':
        aliases.add('accuracy')
    return {alias for alias in aliases if alias}


def _flatten_metric_payload(payload: Any) -> dict[str, float]:
    if isinstance(payload, dict):
        flat: dict[str, float] = {}
        for key, value in payload.items():
            numeric = _to_optional_float(value)
            if numeric is not None:
                flat[str(key)] = numeric
        return flat
    if isinstance(payload, list):
        flat: dict[str, float] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = item.get('metric_name') or item.get('name')
            value = _to_optional_float(item.get('value'))
            if name and value is not None:
                flat[str(name)] = value
        return flat
    return {}


def _metric_match_score(target: str, name: str) -> int:
    normalized = _normalize_metric_name(name)
    aliases = _metric_aliases(name)
    if not normalized and not aliases:
        return -1
    if target in aliases:
        return 10_000
    if normalized.endswith(target):
        score = 8_000
    elif target in normalized:
        score = 6_000
    else:
        return -1

    if 'finaltest' in normalized:
        score += 400
    elif normalized.startswith('test') or 'test' in normalized:
        score += 300
    elif 'final' in normalized:
        score += 200
    elif 'eval' in normalized:
        score += 100

    if 'dev' in normalized or 'val' in normalized:
        score -= 150
    if 'train' in normalized:
        score -= 300
    if 'best' in normalized:
        score -= 50
    return score


def _analysis_metric_sources(analysis_output: dict[str, Any]) -> list[dict[str, float]]:
    analysis = analysis_output.get('analysis') or {}
    comparison = analysis.get('comparison_with_baselines') or {}
    metric_sources = [_flatten_metric_payload(analysis.get('final_metrics'))]
    if isinstance(comparison, dict):
        metric_sources.append(_flatten_metric_payload(comparison.get('our_method')))
    return metric_sources


def _select_primary_metric(
    blueprint: dict[str, Any],
    experiment_output: dict[str, Any],
    analysis_output: dict[str, Any],
) -> tuple[str | None, bool]:
    metrics = blueprint.get('metrics') if isinstance(blueprint, dict) else []
    if isinstance(metrics, list):
        primary_candidates: list[tuple[str, bool]] = []
        for item in metrics:
            if not isinstance(item, dict):
                continue
            if item.get('primary'):
                name = str(item.get('name') or '').strip()
                if name:
                    primary_candidates.append((name, bool(item.get('higher_is_better', True))))
        for name, hib in primary_candidates:
            if _extract_final_performance(name, experiment_output, analysis_output) is not None:
                return name, hib
        if primary_candidates:
            return primary_candidates[0]
        for item in metrics:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name') or '').strip()
            if name:
                return name, bool(item.get('higher_is_better', True))
    combined = {}
    for metrics in _analysis_metric_sources(analysis_output):
        combined.update(metrics)
    combined.update(_flatten_metric_payload(experiment_output.get('experiment_results') or experiment_output.get('metrics')))
    if combined:
        first = next(iter(combined))
        return first, True
    return None, True


def _extract_final_performance(
    primary_metric_name: str | None,
    experiment_output: dict[str, Any],
    analysis_output: dict[str, Any],
) -> float | None:
    if not primary_metric_name:
        return None
    targets = _metric_aliases(primary_metric_name)
    if not targets:
        return None
    metric_sources = [
        *_analysis_metric_sources(analysis_output),
        _flatten_metric_payload(experiment_output.get('experiment_results') or experiment_output.get('metrics')),
    ]
    for metrics in metric_sources:
        for name, value in metrics.items():
            if _metric_aliases(name) & targets:
                return round(value, 6)
    best_value: float | None = None
    best_score = -1
    for metrics in metric_sources:
        for name, value in metrics.items():
            for target in targets:
                score = _metric_match_score(target, name)
                if score > best_score:
                    best_score = score
                    best_value = value
    if best_score >= 0 and best_value is not None:
        return round(best_value, 6)
    return None


def _extract_baseline_performance(
    primary_metric_name: str | None,
    higher_is_better: bool,
    blueprint: dict[str, Any],
) -> float | None:
    if not primary_metric_name:
        return None
    targets = _metric_aliases(primary_metric_name)
    candidates: list[float] = []
    for baseline in blueprint.get('baselines', []) if isinstance(blueprint, dict) else []:
        if not isinstance(baseline, dict):
            continue
        perf = baseline.get('expected_performance') or baseline.get('metrics') or {}
        if not isinstance(perf, dict):
            continue
        for name, value in perf.items():
            numeric = _to_optional_float(value)
            if numeric is None:
                continue
            if _metric_aliases(str(name)) & targets:
                candidates.append(numeric)
    if not candidates:
        return None
    best = max(candidates) if higher_is_better else min(candidates)
    return round(best, 6)


def _compute_delta(final_performance: float | None, baseline_performance: float | None, higher_is_better: bool) -> float | None:
    if final_performance is None or baseline_performance is None:
        return None
    if higher_is_better:
        return round(final_performance - baseline_performance, 6)
    return round(baseline_performance - final_performance, 6)


def _looks_like_fraction_metric(value: float | None) -> bool:
    return value is not None and 0.0 <= value <= 1.5


def _looks_like_percentage_metric(value: float | None) -> bool:
    return value is not None and 1.5 < value <= 100.0


def _normalize_metric_scale_pair(
    final_performance: float | None,
    baseline_performance: float | None,
) -> tuple[float | None, float | None]:
    """Normalize percentage-vs-fraction mismatches before delta computation.

    Generated blueprints often encode baseline expectations as percentages
    (e.g. 79.0) while execution outputs emit fractions (e.g. 0.79). We only
    rescale when one side clearly looks like a percentage and the other clearly
    looks like a fraction.
    """
    normalized_final = final_performance
    normalized_baseline = baseline_performance
    if _looks_like_fraction_metric(normalized_final) and _looks_like_percentage_metric(normalized_baseline):
        normalized_baseline = round(float(normalized_baseline) / 100.0, 6)
    elif _looks_like_percentage_metric(normalized_final) and _looks_like_fraction_metric(normalized_baseline):
        normalized_final = round(float(normalized_final) / 100.0, 6)
    return normalized_final, normalized_baseline


def _extract_status(experiment_output: dict[str, Any]) -> str:
    for key in ('experiment_status', 'final_status', 'status'):
        value = experiment_output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    code_execution = experiment_output.get('code_execution')
    if isinstance(code_execution, dict):
        value = code_execution.get('status')
        if isinstance(value, str) and value.strip():
            return value.strip()
    return 'unknown'


def _is_successful_implementation(experiment_output: dict[str, Any]) -> bool:
    result_contract = experiment_output.get('result_contract')
    if isinstance(result_contract, dict):
        contract_status = str(result_contract.get('status') or '').strip().lower()
        success_path = str(result_contract.get('success_path') or '').strip()
        failure_signals = list(result_contract.get('failure_signals', []) or [])
        execution_status = str(result_contract.get('execution_status') or '').strip().lower()
        final_status = str(result_contract.get('final_status') or '').strip().lower()
        satisfied_signals = set(str(item).strip().lower() for item in (result_contract.get('satisfied_signals') or []))
        artifact_inventory = result_contract.get('artifact_inventory') if isinstance(result_contract.get('artifact_inventory'), dict) else {}
        result_files = artifact_inventory.get('result_files') if isinstance(artifact_inventory.get('result_files'), list) else []
        metrics = experiment_output.get('metrics')
        raw_metrics = metrics.get('raw') if isinstance(metrics, dict) else None
        has_reusable_payload = any(
            bool(payload)
            for payload in (
                success_path,
                experiment_output.get('experiment_results'),
                raw_metrics,
                experiment_output.get('result_file_SUCCESS'),
                experiment_output.get('result_file__SUCCESS'),
                experiment_output.get('result_file_completion.json'),
                experiment_output.get('result_file_summary.json'),
                experiment_output.get('result_file_summary.csv'),
                experiment_output.get('result_file_aggregate.csv'),
                experiment_output.get('result_file_metrics.json'),
                experiment_output.get('result_file_run_finals.json'),
                experiment_output.get('result_file_summary_by_model_missingness.json'),
                result_files,
            )
        )
        if contract_status == 'success':
            return True
        if contract_status == 'partial' and success_path and not failure_signals:
            return True
        if (
            contract_status == 'partial'
            and success_path
            and execution_status in _STATUS_SUCCESS
            and final_status in _STATUS_SUCCESS
            and bool(satisfied_signals & {'training_log', 'result_files', 'metrics_signal'})
        ):
            return True
        # Some cluster runs finish cleanly and write reusable result artifacts, but
        # still miss a narrower parser-only signal such as `metrics_signal`.
        # Treat those as runnable success when the execution itself completed and
        # no explicit failure signals remain.
        if (
            contract_status == 'failed'
            and not failure_signals
            and (
                (execution_status in _STATUS_SUCCESS and final_status in _STATUS_SUCCESS)
                or bool(satisfied_signals & {'result_files', 'training_log', 'artifact_signal'})
                or bool(result_files)
            )
        ):
            return True if has_reusable_payload or (execution_status in _STATUS_SUCCESS and final_status in _STATUS_SUCCESS) else False
    status = _extract_status(experiment_output).lower()
    if status in _STATUS_SUCCESS:
        return True
    code_execution = experiment_output.get('code_execution')
    if isinstance(code_execution, dict):
        inner = str(code_execution.get('status') or '').strip().lower()
        if inner in _STATUS_SUCCESS:
            return True
    return False


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text or text.upper() == 'N/A':
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _judge_alignment(
    config: Any,
    assignment: dict[str, Any],
    ideation_output: dict[str, Any],
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    question = dict(assignment.get('question') or {})
    payload = {
        'persona_id': assignment.get('persona_id'),
        'domain': question.get('domain'),
        'difficulty': question.get('difficulty'),
        'background': question.get('background'),
        'baselines': question.get('baselines'),
        'datasets': question.get('datasets'),
        'user_requirements': question.get('user_requirements'),
        'selected_idea': get_selected_idea_id(ideation_output),
        'rationale': ideation_output.get('rationale'),
        'proposed_method': blueprint.get('proposed_method'),
        'blueprint_title': blueprint.get('title'),
        'ablation_groups': blueprint.get('ablation_groups'),
    }
    fallback = {'alignment_score': 10.0, 'pass_at_1': True, 'feedback': ''}
    response = _call_stage_json(config, stage_name='review', system_prompt=_ALIGNMENT_SYSTEM_PROMPT, payload=payload, max_tokens=512, fallback=fallback)
    alignment_score = _clamp_score(response.get('alignment_score', response.get('score')), default=10.0)
    response['alignment_score'] = alignment_score
    # Retain pass_at_1 for retry compatibility with older aggregation outputs.
    response['pass_at_1'] = bool(response.get('pass_at_1', alignment_score >= 7.0))
    response['feedback'] = str(response.get('feedback') or '')
    return response


def _clamp_score(value: Any, *, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = float(default)
    return max(1.0, min(10.0, score))


def _judge_novelty(
    config: Any,
    assignment: dict[str, Any],
    ideation_output: dict[str, Any],
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    question = dict(assignment.get('question') or {})
    payload = {
        'persona_id': assignment.get('persona_id'),
        'domain': question.get('domain'),
        'background': question.get('background'),
        'baselines': question.get('baselines'),
        'datasets': question.get('datasets'),
        'selected_idea': get_selected_idea_id(ideation_output),
        'rationale': ideation_output.get('rationale'),
        'proposed_method': blueprint.get('proposed_method'),
        'key_components': (blueprint.get('proposed_method') or {}).get('key_components'),
    }
    fallback = {'novelty_score': None, 'closest_baseline': None, 'rationale': ''}
    response = _call_stage_json(config, stage_name='review', system_prompt=_NOVELTY_SYSTEM_PROMPT, payload=payload, max_tokens=512, fallback=fallback)
    response['closest_baseline'] = response.get('closest_baseline')
    return response


def _call_stage_json(
    config: Any,
    *,
    stage_name: str,
    system_prompt: str,
    payload: dict[str, Any],
    max_tokens: int,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    try:
        stage_cfg = config.for_stage(stage_name)
        model = getattr(stage_cfg, 'model')
        base_url = getattr(stage_cfg, 'base_url', None) or getattr(config, 'base_url')
        api_key = getattr(stage_cfg, 'api_key', None) or getattr(config, 'api_key')
        request_payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            'max_tokens': max_tokens,
            'response_format': {'type': 'json_object'},
        }
        temperature = getattr(stage_cfg, 'temperature', None)
        if temperature is not None:
            request_payload['temperature'] = temperature
        req = urllib.request.Request(
            f"{str(base_url).rstrip('/')}/chat/completions",
            data=json.dumps(request_payload, ensure_ascii=False).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=getattr(stage_cfg, 'timeout', None) or getattr(config, 'timeout', 180)) as response:
            raw = json.loads(response.read().decode('utf-8'))
        content = raw['choices'][0]['message']['content']
        data = json.loads(content)
        return data if isinstance(data, dict) else deepcopy(fallback)
    except Exception:
        return deepcopy(fallback)
