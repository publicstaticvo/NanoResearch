from __future__ import annotations

import asyncio

from nanoresearch.experiments.deep_persona_runner import (
    DEFAULT_PERSONA_BRIEFS,
    _apply_variant_overrides,
    _resolve_assignment_config_path,
    build_assignment_topic,
    build_result_record,
    run_assignment,
    resolve_variant_runtime_settings,
)
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PaperMode, PipelineMode


def _question(question_id: str = 'q1') -> dict:
    return {
        'question_id': question_id,
        'domain': 'NLP',
        'difficulty': 'incremental_innovation',
        'background': 'Investigate a lightweight method for improving biomedical QA under strict compute limits.',
        'baselines': ['BioBERT', 'PubMedBERT'],
        'datasets': ['PubMedQA'],
        'user_requirements': 'Generate a novel idea, an executable plan, and keep the design reproducible.',
    }


def test_variant_settings_map_to_expected_runtime_flags() -> None:
    base = resolve_variant_runtime_settings('base_router')
    assert base == {
        'memory_enabled': False,
        'memory_evolution_enabled': False,
        'skill_evolution_enabled': False,
        'same_router_hindsight_sdpo': False,
        'appendix_only': False,
    }

    full = resolve_variant_runtime_settings('full_system')
    assert full == {
        'memory_enabled': True,
        'memory_evolution_enabled': True,
        'skill_evolution_enabled': True,
        'same_router_hindsight_sdpo': True,
        'appendix_only': False,
    }

    appendix = resolve_variant_runtime_settings('context_informed_generation')
    assert appendix['appendix_only'] is True
    assert appendix['same_router_hindsight_sdpo'] is False


def test_apply_variant_overrides_propagates_sdpo_flag_into_runtime_config() -> None:
    base_config = ResearchConfig(base_url='https://example.com', api_key='')
    settings = resolve_variant_runtime_settings('full_system')

    variant_config = _apply_variant_overrides(base_config, settings)

    assert variant_config.same_router_hindsight_sdpo_enabled is True
    assert variant_config.memory_enabled is True
    assert variant_config.memory_evolution_enabled is True
    assert variant_config.skill_evolution_enabled is True


def test_build_assignment_topic_includes_persona_and_question_context() -> None:
    assignment = {
        'assignment_id': 'resource_constrained_repro_first::base_router::q1',
        'persona_id': 'resource_constrained_repro_first',
        'variant_name': 'base_router',
        'question': _question(),
    }

    topic = build_assignment_topic(assignment)

    assert 'Persona Profile:' in topic
    assert DEFAULT_PERSONA_BRIEFS['resource_constrained_repro_first'] in topic
    assert 'Research Domain: NLP' in topic
    assert 'Known Baselines: BioBERT; PubMedBERT' in topic
    assert 'Evaluation Datasets: PubMedQA' in topic
    assert 'User Requirements:' in topic


def test_build_result_record_extracts_performance_and_stage_tokens() -> None:
    assignment = {
        'assignment_id': 'persona-a::full_system::q1',
        'persona_id': 'persona-a',
        'variant_name': 'full_system',
        'question': _question(),
    }
    blueprint = {
        'metrics': [
            {'name': 'Accuracy', 'primary': True, 'higher_is_better': True},
            {'name': 'F1', 'primary': False, 'higher_is_better': True},
        ],
        'baselines': [
            {'name': 'BioBERT', 'expected_performance': {'Accuracy': 0.62}},
            {'name': 'PubMedBERT', 'expected_performance': {'Accuracy': 0.65}},
        ],
    }
    experiment_output = {
        'experiment_results': {'Accuracy': 0.74, 'F1': 0.71},
        'experiment_status': 'success',
        'code_execution': {'status': 'success'},
    }
    analysis_output = {
        'analysis': {'final_metrics': {'Accuracy': 0.74, 'F1': 0.71}},
    }
    cost_summary = {
        'stages': {
            'IDEATION': {'total_tokens': 110},
            'PLANNING': {'total_tokens': 220},
            'SETUP': {'total_tokens': 330},
            'CODING': {'total_tokens': 440},
            'EXECUTION': {'total_tokens': 550},
            'ANALYSIS': {'total_tokens': 120},
        },
        'total_tokens': 1770,
    }
    alignment = {'alignment_score': 8.0, 'pass_at_1': True, 'feedback': 'Aligned with the request.'}
    novelty = {'novelty_score': 7.5, 'closest_baseline': 'PubMedBERT'}

    record = build_result_record(
        assignment=assignment,
        workspace_path='/tmp/ws',
        blueprint=blueprint,
        experiment_output=experiment_output,
        analysis_output=analysis_output,
        cost_summary=cost_summary,
        alignment_judgment=alignment,
        novelty_judgment=novelty,
        alignment_token_to_pass=330,
    )

    assert record['novelty_score'] == 7.5
    assert record['alignment_score'] == 8.0
    assert record['alignment_pass_at_1'] is True
    assert record['alignment_token_to_pass'] == 330
    assert record['plan_executability'] is True
    assert record['implementation_success'] is True
    assert record['implementation_token_to_runnable'] == 1320
    assert record['total_tokens_from_method_to_code'] == 1650
    assert record['final_performance'] == 0.74
    assert record['baseline_performance'] == 0.65
    assert record['delta_over_baseline'] == 0.09
    assert record['primary_metric_name'] == 'Accuracy'


def test_resolve_assignment_config_path_prefers_assignment_override(tmp_path) -> None:
    global_config = tmp_path / "global.json"
    assignment_config = tmp_path / "assignment.json"

    resolved = _resolve_assignment_config_path(
        {"config_path": str(assignment_config)},
        global_config,
    )

    assert resolved == assignment_config


def test_run_assignment_persists_assignment_context(tmp_path, monkeypatch) -> None:
    class FakeOrchestrator:
        def __init__(self, workspace, config) -> None:
            self.workspace = workspace
            self.config = config

        async def run(self, topic: str) -> dict:
            context = self.workspace.read_json('logs/deep_assignment_context.json')
            assert context['assignment_id'] == 'persona-a::base_router::q1'
            assert context['variant_name'] == 'base_router'
            return {
                'ideation_output': {'idea': 'x'},
                'experiment_blueprint': {
                    'metrics': [{'name': 'Accuracy', 'primary': True, 'higher_is_better': True}],
                    'baselines': [{'name': 'BioBERT', 'expected_performance': {'Accuracy': 0.62}}],
                },
                'execution_output': {
                    'experiment_results': {'Accuracy': 0.66},
                    'experiment_status': 'success',
                    'result_contract': {'status': 'success', 'success_path': 'structured_metrics_artifact'},
                },
                'analysis_output': {'analysis': {'final_metrics': {'Accuracy': 0.66}}},
                'cost_summary': {'stages': {'IDEATION': {'total_tokens': 10}, 'PLANNING': {'total_tokens': 20}}},
            }

        async def close(self) -> None:
            return None

    def _fake_runtime_symbols():
        return (
            ResearchConfig,
            Workspace,
            FakeOrchestrator,
            PipelineMode,
            PaperMode,
            lambda seed: {},
            lambda profile: None,
        )

    monkeypatch.setattr(
        'nanoresearch.experiments.deep_persona_runner._load_runtime_symbols',
        _fake_runtime_symbols,
    )
    monkeypatch.setattr(
        'nanoresearch.experiments.deep_persona_runner._judge_alignment',
        lambda *args, **kwargs: {'alignment_score': 8.0, 'pass_at_1': True, 'feedback': ''},
    )
    monkeypatch.setattr(
        'nanoresearch.experiments.deep_persona_runner._judge_novelty',
        lambda *args, **kwargs: {'novelty_score': 5.0, 'closest_baseline': 'BioBERT'},
    )

    assignment = {
        'assignment_id': 'persona-a::base_router::q1',
        'chain_id': 'persona-a::q1',
        'persona_id': 'resource_constrained_repro_first',
        'variant_name': 'base_router',
        'question': _question(),
    }
    outcome = asyncio.run(
        run_assignment(
            assignment,
            output_dir=tmp_path / 'runs',
            config_path=None,
            max_alignment_retries=0,
            disable_ideation_retrieval=True,
        )
    )

    context_path = (
        tmp_path
        / 'runs'
        / 'persona-a-base_router-q1'
        / 'workspaces'
        / 'attempt-01'
        / 'logs'
        / 'deep_assignment_context.json'
    )
    assert context_path.is_file()
    context = Workspace.load(context_path.parents[1]).read_json('logs/deep_assignment_context.json')
    assert context['chain_id'] == 'persona-a::q1'
    assert context['attempt_index'] == 1
    assert outcome.record['assignment_id'] == 'persona-a::base_router::q1'


def test_run_assignment_uses_assignment_level_config_path(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeConfig:
        def __init__(self) -> None:
            self.ideation_disable_retrieval = False

        def model_copy(self, deep: bool = False):
            clone = FakeConfig()
            clone.ideation_disable_retrieval = self.ideation_disable_retrieval
            clone.memory_enabled = getattr(self, "memory_enabled", True)
            clone.memory_evolution_enabled = getattr(self, "memory_evolution_enabled", True)
            clone.skill_evolution_enabled = getattr(self, "skill_evolution_enabled", True)
            clone.same_router_hindsight_sdpo_enabled = getattr(self, "same_router_hindsight_sdpo_enabled", False)
            clone.skip_stages = list(getattr(self, "skip_stages", []))
            return clone

        def model_dump(self, mode: str = "json") -> dict:
            return {"ideation_disable_retrieval": self.ideation_disable_retrieval}

        @classmethod
        def load(cls, path=None):
            captured["loaded_path"] = str(path) if path else ""
            cfg = cls()
            cfg.skip_stages = []
            cfg.memory_enabled = True
            cfg.memory_evolution_enabled = True
            cfg.skill_evolution_enabled = True
            cfg.same_router_hindsight_sdpo_enabled = False
            return cfg

    class FakeOrchestrator:
        def __init__(self, workspace, config) -> None:
            self.workspace = workspace
            self.config = config

        async def run(self, topic: str) -> dict:
            return {
                "ideation_output": {"idea": "x"},
                "experiment_blueprint": {
                    "metrics": [{"name": "Accuracy", "primary": True, "higher_is_better": True}],
                    "baselines": [{"name": "BioBERT", "expected_performance": {"Accuracy": 0.62}}],
                },
                "execution_output": {
                    "experiment_results": {"Accuracy": 0.66},
                    "experiment_status": "success",
                    "result_contract": {"status": "success"},
                },
                "analysis_output": {"analysis": {"final_metrics": {"Accuracy": 0.66}}},
                "cost_summary": {"stages": {"IDEATION": {"total_tokens": 10}, "PLANNING": {"total_tokens": 20}}},
            }

        async def close(self) -> None:
            return None

    def _fake_runtime_symbols():
        return (
            FakeConfig,
            Workspace,
            FakeOrchestrator,
            PipelineMode,
            PaperMode,
            lambda seed: {},
            lambda profile: None,
        )

    monkeypatch.setattr(
        "nanoresearch.experiments.deep_persona_runner._load_runtime_symbols",
        _fake_runtime_symbols,
    )
    monkeypatch.setattr(
        "nanoresearch.experiments.deep_persona_runner._judge_alignment",
        lambda *args, **kwargs: {"alignment_score": 8.0, "pass_at_1": True, "feedback": ""},
    )
    monkeypatch.setattr(
        "nanoresearch.experiments.deep_persona_runner._judge_novelty",
        lambda *args, **kwargs: {"novelty_score": 5.0, "closest_baseline": "BioBERT"},
    )

    assignment_config = tmp_path / "persona.json"
    assignment_config.write_text("{}", encoding="utf-8")
    assignment = {
        "assignment_id": "persona-a::base_router::q2",
        "chain_id": "persona-a::q2",
        "persona_id": "resource_constrained_repro_first",
        "variant_name": "base_router",
        "question": _question("q2"),
        "config_path": str(assignment_config),
    }

    outcome = asyncio.run(
        run_assignment(
            assignment,
            output_dir=tmp_path / "runs",
            config_path=tmp_path / "global.json",
            max_alignment_retries=0,
            disable_ideation_retrieval=False,
        )
    )

    assert captured["loaded_path"] == str(assignment_config)
    assert outcome.record["metadata"]["config_path"] == str(assignment_config)
