from __future__ import annotations

from nanoresearch.experiments.router_persona_eval import (
    APPENDIX_VARIANTS,
    DEFAULT_PERSONA_IDS,
    MAIN_VARIANTS,
    aggregate_experiment_results,
    build_experiment_manifest,
)


def _question(question_id: str, domain: str, difficulty: str) -> dict:
    return {
        "question_id": question_id,
        "domain": domain,
        "difficulty": difficulty,
        "background": f"{domain} background for {question_id}",
        "baselines": [f"{domain}-baseline-a", f"{domain}-baseline-b"],
        "datasets": [f"{domain}-dataset"],
        "user_requirements": "Generate a new idea and an implementation-oriented plan.",
    }


def _record(
    *,
    persona_id: str,
    variant_name: str,
    question_id: str,
    novelty_score: float,
    alignment_score: float,
    alignment_pass_at_1: bool,
    alignment_token_to_pass: int,
    plan_executability: bool,
    implementation_token_to_runnable: int | None,
    implementation_success: bool,
    final_performance: float | None,
    baseline_performance: float | None,
    total_tokens_from_method_to_code: int,
) -> dict:
    return {
        "persona_id": persona_id,
        "variant_name": variant_name,
        "question_id": question_id,
        "novelty_score": novelty_score,
        "alignment_score": alignment_score,
        "alignment_pass_at_1": alignment_pass_at_1,
        "alignment_token_to_pass": alignment_token_to_pass,
        "plan_executability": plan_executability,
        "implementation_token_to_runnable": implementation_token_to_runnable,
        "implementation_success": implementation_success,
        "final_performance": final_performance,
        "baseline_performance": baseline_performance,
        "total_tokens_from_method_to_code": total_tokens_from_method_to_code,
    }


def test_build_experiment_manifest_expands_questions_personas_and_variants() -> None:
    questions = [
        _question("q-nlp-1", "NLP", "incremental_innovation"),
        _question("q-cv-1", "CV", "nontrivial_recomposition"),
    ]

    manifest = build_experiment_manifest(questions, include_appendix_baseline=True)

    assert len(DEFAULT_PERSONA_IDS) == 10
    assert len(MAIN_VARIANTS) == 8
    assert len(APPENDIX_VARIANTS) == 1
    assert len(manifest) == len(questions) * len(DEFAULT_PERSONA_IDS) * 9

    first = manifest[0]
    assert first["persona_id"] in DEFAULT_PERSONA_IDS
    assert first["variant_name"] == "base_router"
    assert first["variant_label"] == "Base Router"
    assert first["question"]["question_id"] == "q-nlp-1"
    assert first["question"]["baselines"] == ["NLP-baseline-a", "NLP-baseline-b"]
    assert first["component_flags"] == {
        "memory_self_evolution": False,
        "skill_self_evolution": False,
        "same_router_hindsight_sdpo": False,
        "appendix_only": False,
    }


def test_aggregate_experiment_results_uses_persona_macro_average() -> None:
    records = [
        _record(
            persona_id="persona_a",
            variant_name="full_system",
            question_id="q1",
            novelty_score=10.0,
            alignment_score=9.0,
            alignment_pass_at_1=True,
            alignment_token_to_pass=100,
            plan_executability=True,
            implementation_token_to_runnable=200,
            implementation_success=True,
            final_performance=0.80,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=400,
        ),
        _record(
            persona_id="persona_a",
            variant_name="full_system",
            question_id="q2",
            novelty_score=8.0,
            alignment_score=7.0,
            alignment_pass_at_1=False,
            alignment_token_to_pass=150,
            plan_executability=True,
            implementation_token_to_runnable=250,
            implementation_success=True,
            final_performance=0.60,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=500,
        ),
        _record(
            persona_id="persona_b",
            variant_name="full_system",
            question_id="q1",
            novelty_score=0.0,
            alignment_score=5.0,
            alignment_pass_at_1=True,
            alignment_token_to_pass=50,
            plan_executability=True,
            implementation_token_to_runnable=100,
            implementation_success=True,
            final_performance=0.50,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=200,
        ),
        _record(
            persona_id="persona_a",
            variant_name="base_router",
            question_id="q1",
            novelty_score=4.0,
            alignment_score=4.0,
            alignment_pass_at_1=False,
            alignment_token_to_pass=180,
            plan_executability=True,
            implementation_token_to_runnable=260,
            implementation_success=True,
            final_performance=0.55,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=550,
        ),
        _record(
            persona_id="persona_b",
            variant_name="base_router",
            question_id="q1",
            novelty_score=1.0,
            alignment_score=2.0,
            alignment_pass_at_1=False,
            alignment_token_to_pass=90,
            plan_executability=True,
            implementation_token_to_runnable=120,
            implementation_success=True,
            final_performance=0.48,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=280,
        ),
    ]

    summary = aggregate_experiment_results(records)

    full_system = summary["macro_average"]["full_system"]["metrics"]
    assert full_system["Novelty"]["mean"] == 4.5
    assert full_system["User Req. Align."]["mean"] == 6.5
    assert full_system["Implementation success rate"]["mean"] == 1.0
    assert full_system["Delta over baseline"]["mean"] == 0.1

    persona_breakdown = summary["per_persona"]["persona_a"]["full_system"]["metrics"]
    assert persona_breakdown["novelty_mean"] == 9.0
    assert persona_breakdown["delta_over_baseline_runnable_only_mean"] == 0.2


def test_appendix_baseline_is_separated_from_main_table() -> None:
    records = [
        _record(
            persona_id="persona_a",
            variant_name="full_system",
            question_id="q1",
            novelty_score=9.0,
            alignment_score=9.0,
            alignment_pass_at_1=True,
            alignment_token_to_pass=100,
            plan_executability=True,
            implementation_token_to_runnable=180,
            implementation_success=True,
            final_performance=0.75,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=350,
        ),
        _record(
            persona_id="persona_b",
            variant_name="full_system",
            question_id="q1",
            novelty_score=8.0,
            alignment_score=8.0,
            alignment_pass_at_1=True,
            alignment_token_to_pass=90,
            plan_executability=True,
            implementation_token_to_runnable=170,
            implementation_success=True,
            final_performance=0.72,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=330,
        ),
        _record(
            persona_id="persona_a",
            variant_name="base_router",
            question_id="q1",
            novelty_score=4.0,
            alignment_score=4.0,
            alignment_pass_at_1=False,
            alignment_token_to_pass=160,
            plan_executability=True,
            implementation_token_to_runnable=240,
            implementation_success=True,
            final_performance=0.58,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=420,
        ),
        _record(
            persona_id="persona_b",
            variant_name="base_router",
            question_id="q1",
            novelty_score=5.0,
            alignment_score=5.0,
            alignment_pass_at_1=False,
            alignment_token_to_pass=140,
            plan_executability=True,
            implementation_token_to_runnable=210,
            implementation_success=True,
            final_performance=0.57,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=390,
        ),
        _record(
            persona_id="persona_a",
            variant_name="context_informed_generation",
            question_id="q1",
            novelty_score=6.0,
            alignment_score=7.0,
            alignment_pass_at_1=True,
            alignment_token_to_pass=110,
            plan_executability=True,
            implementation_token_to_runnable=200,
            implementation_success=True,
            final_performance=0.62,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=360,
        ),
        _record(
            persona_id="persona_b",
            variant_name="context_informed_generation",
            question_id="q1",
            novelty_score=6.5,
            alignment_score=7.5,
            alignment_pass_at_1=True,
            alignment_token_to_pass=120,
            plan_executability=True,
            implementation_token_to_runnable=190,
            implementation_success=True,
            final_performance=0.63,
            baseline_performance=0.50,
            total_tokens_from_method_to_code=370,
        ),
    ]

    summary = aggregate_experiment_results(records)

    main_methods = [row["method"] for row in summary["main_table"]["rows"]]
    assert "Context-informed Generation" not in main_methods
    assert summary["appendix_baselines"]["context_informed_generation"]["label"] == "Context-informed Generation"
    assert summary["appendix_baselines"]["context_informed_generation"]["metrics"]["Novelty"]["mean"] == 6.25
