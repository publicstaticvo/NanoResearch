from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Iterable


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    label: str
    memory_self_evolution: bool
    skill_self_evolution: bool
    same_router_hindsight_sdpo: bool
    appendix_only: bool = False


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source_key: str
    higher_is_better: bool


DEFAULT_PERSONA_IDS = [
    "ai4science_journal_conservative",
    "ai4science_reproducibility_first",
    "benchmark_maximalist_conference",
    "cv_fast_iteration_builder",
    "cv_visual_benchmark_heavy",
    "journal_evidence_first_writer",
    "multimodal_systems_engineer",
    "nlp_conference_exploratory",
    "nlp_conference_pragmatic",
    "resource_constrained_repro_first",
]

MAIN_VARIANTS = [
    ExperimentVariant("base_router", "Base Router", False, False, False),
    ExperimentVariant("memory_only", "+ Memory", True, False, False),
    ExperimentVariant("skill_only", "+ Skill", False, True, False),
    ExperimentVariant("sdpo_only", "+ SDPO", False, False, True),
    ExperimentVariant("memory_skill", "+ Memory + Skill", True, True, False),
    ExperimentVariant("memory_sdpo", "+ Memory + SDPO", True, False, True),
    ExperimentVariant("skill_sdpo", "+ Skill + SDPO", False, True, True),
    ExperimentVariant("full_system", "Full System", True, True, True),
]
APPENDIX_VARIANTS = [
    ExperimentVariant(
        "context_informed_generation",
        "Context-informed Generation",
        False,
        False,
        False,
        appendix_only=True,
    ),
]
ALL_VARIANTS = [*MAIN_VARIANTS, *APPENDIX_VARIANTS]
VARIANT_BY_NAME = {variant.name: variant for variant in ALL_VARIANTS}

CORE_METRICS = [
    MetricSpec("Novelty", "novelty_mean", True),
    MetricSpec("User Req. Align.", "alignment_score_mean", True),
    MetricSpec("Implementation success rate", "implementation_success_rate", True),
    MetricSpec("Delta over baseline", "delta_over_baseline_runnable_only_mean", True),
]

EFFICIENCY_METRICS = [
    MetricSpec("Alignment token-to-pass", "alignment_token_to_pass_mean", False),
    MetricSpec("Implementation token-to-runnable", "implementation_token_to_runnable_mean", False),
    MetricSpec("Total tokens", "total_tokens_from_method_to_code_mean", False),
]

_REQUIRED_QUESTION_FIELDS = {
    "question_id",
    "domain",
    "difficulty",
    "background",
    "baselines",
    "datasets",
    "user_requirements",
}


def build_experiment_manifest(
    test_questions: Iterable[dict[str, Any]],
    *,
    personas: Iterable[str] | None = None,
    include_appendix_baseline: bool = True,
    evolution_rounds: int = 1,
) -> list[dict[str, Any]]:
    normalized_questions = [_normalize_question(question) for question in test_questions]
    persona_list = [str(persona).strip() for persona in (personas or DEFAULT_PERSONA_IDS) if str(persona).strip()]
    variants = MAIN_VARIANTS + (APPENDIX_VARIANTS if include_appendix_baseline else [])
    rounds = max(1, int(evolution_rounds))

    manifest: list[dict[str, Any]] = []
    for persona_id in persona_list:
        for question in normalized_questions:
            for variant in variants:
                chain_id = f"{persona_id}::{variant.name}::{question['question_id']}"
                for evolution_round in range(1, rounds + 1):
                    manifest.append(
                        {
                            "assignment_id": f"{chain_id}::round{evolution_round:02d}",
                            "chain_id": chain_id,
                            "evolution_round": evolution_round,
                            "evolution_total_rounds": rounds,
                            "persona_id": persona_id,
                            "variant_name": variant.name,
                            "variant_label": variant.label,
                            "component_flags": {
                                "memory_self_evolution": variant.memory_self_evolution,
                                "skill_self_evolution": variant.skill_self_evolution,
                                "same_router_hindsight_sdpo": variant.same_router_hindsight_sdpo,
                                "appendix_only": variant.appendix_only,
                            },
                            "question": question,
                        }
                    )
    return manifest


def aggregate_experiment_results(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized_records = [_normalize_result_record(record) for record in records]
    grouped = _group_records(normalized_records)
    per_persona = _aggregate_per_persona(grouped)
    macro_average = _aggregate_macro_average(per_persona)

    main_table = _build_table(MAIN_VARIANTS, macro_average, CORE_METRICS)
    efficiency_table = _build_table(MAIN_VARIANTS, macro_average, EFFICIENCY_METRICS)
    appendix_baselines = _build_appendix_baselines(macro_average)
    ablation_contributions = _build_ablation_contributions(macro_average)

    question_ids = sorted({record["question_id"] for record in normalized_records})
    personas = sorted({record["persona_id"] for record in normalized_records})

    return {
        "record_count": len(normalized_records),
        "question_count": len(question_ids),
        "question_ids": question_ids,
        "personas": personas,
        "variant_registry": [asdict(variant) for variant in ALL_VARIANTS],
        "metric_registry": {
            "core": [asdict(metric) for metric in CORE_METRICS],
            "efficiency": [asdict(metric) for metric in EFFICIENCY_METRICS],
        },
        "per_persona": per_persona,
        "macro_average": macro_average,
        "main_table": main_table,
        "efficiency_table": efficiency_table,
        "appendix_baselines": appendix_baselines,
        "ablation_contributions": ablation_contributions,
    }


def _normalize_question(question: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(field for field in _REQUIRED_QUESTION_FIELDS if field not in question)
    if missing:
        raise ValueError(f"Question missing required fields: {', '.join(missing)}")

    normalized = dict(question)
    normalized["question_id"] = str(question["question_id"])
    normalized["domain"] = str(question["domain"])
    normalized["difficulty"] = str(question["difficulty"])
    normalized["background"] = str(question["background"])
    normalized["user_requirements"] = str(question["user_requirements"])
    normalized["baselines"] = [str(item) for item in question.get("baselines", [])]
    normalized["datasets"] = [str(item) for item in question.get("datasets", [])]
    if not normalized["baselines"]:
        raise ValueError(f"Question {normalized['question_id']} must provide at least one baseline")
    if not normalized["datasets"]:
        raise ValueError(f"Question {normalized['question_id']} must provide at least one dataset")
    return normalized


def _normalize_result_record(record: dict[str, Any]) -> dict[str, Any]:
    persona_id = str(record.get("persona_id", "")).strip()
    variant_name = str(record.get("variant_name", "")).strip()
    question_id = str(record.get("question_id", "")).strip()
    if not persona_id or not variant_name or not question_id:
        raise ValueError("Each result record needs persona_id, variant_name, and question_id")
    if variant_name not in VARIANT_BY_NAME:
        raise ValueError(f"Unknown variant_name: {variant_name}")

    final_performance = _to_optional_float(record.get("final_performance"))
    baseline_performance = _to_optional_float(record.get("baseline_performance"))
    delta_over_baseline = _to_optional_float(record.get("delta_over_baseline"))
    if delta_over_baseline is None and final_performance is not None and baseline_performance is not None:
        delta_over_baseline = final_performance - baseline_performance

    return {
        "persona_id": persona_id,
        "variant_name": variant_name,
        "question_id": question_id,
        "novelty_score": _to_optional_float(record.get("novelty_score")),
        "alignment_score": _to_optional_float(record.get("alignment_score", record.get("alignment_judge_score"))),
        "alignment_pass_at_1": _to_optional_bool(record.get("alignment_pass_at_1")),
        "alignment_token_to_pass": _to_optional_float(record.get("alignment_token_to_pass")),
        "plan_executability": _to_optional_bool(record.get("plan_executability")),
        "implementation_token_to_runnable": _to_optional_float(record.get("implementation_token_to_runnable")),
        "implementation_success": _to_optional_bool(record.get("implementation_success")),
        "final_performance": final_performance,
        "baseline_performance": baseline_performance,
        "delta_over_baseline": delta_over_baseline,
        "total_tokens_from_method_to_code": _to_optional_float(record.get("total_tokens_from_method_to_code")),
    }


def _group_records(records: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        persona_bucket = grouped.setdefault(record["persona_id"], {})
        persona_bucket.setdefault(record["variant_name"], []).append(record)
    return grouped


def _aggregate_per_persona(grouped: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    per_persona: dict[str, Any] = {}
    ordered_variants = [variant.name for variant in ALL_VARIANTS]
    for persona_id in sorted(grouped):
        per_persona[persona_id] = {}
        for variant_name in ordered_variants:
            variant_records = grouped[persona_id].get(variant_name, [])
            if not variant_records:
                continue
            per_persona[persona_id][variant_name] = {
                "label": VARIANT_BY_NAME[variant_name].label,
                "task_count": len(variant_records),
                "metrics": _aggregate_variant_records(variant_records),
            }
    return per_persona


def _aggregate_variant_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    novelty = _mean([record["novelty_score"] for record in records])
    alignment_score = _mean([record["alignment_score"] for record in records])
    alignment_pass = _mean_bool([record["alignment_pass_at_1"] for record in records])
    alignment_tokens = _mean([record["alignment_token_to_pass"] for record in records])
    plan_executability = _mean_bool([record["plan_executability"] for record in records])
    implementation_tokens = _mean([record["implementation_token_to_runnable"] for record in records])
    implementation_success = _mean_bool([record["implementation_success"] for record in records])
    runnable_final = _mean(
        [record["final_performance"] for record in records if record["implementation_success"] and record["final_performance"] is not None]
    )
    runnable_delta = _mean(
        [record["delta_over_baseline"] for record in records if record["implementation_success"] and record["delta_over_baseline"] is not None]
    )
    total_tokens = _mean([record["total_tokens_from_method_to_code"] for record in records])
    runnable_task_count = sum(1 for record in records if record["implementation_success"])

    return {
        "novelty_mean": novelty,
        "alignment_score_mean": alignment_score,
        "alignment_pass_at_1_rate": alignment_pass,
        "alignment_token_to_pass_mean": alignment_tokens,
        "plan_executability_rate": plan_executability,
        "implementation_token_to_runnable_mean": implementation_tokens,
        "implementation_success_rate": implementation_success,
        "final_performance_runnable_only_mean": runnable_final,
        "delta_over_baseline_runnable_only_mean": runnable_delta,
        "total_tokens_from_method_to_code_mean": total_tokens,
        "runnable_task_count": runnable_task_count,
    }


def _aggregate_macro_average(per_persona: dict[str, Any]) -> dict[str, Any]:
    macro_average: dict[str, Any] = {}
    for variant in ALL_VARIANTS:
        metric_summary: dict[str, Any] = {}
        for metric in [*CORE_METRICS, *EFFICIENCY_METRICS]:
            values = []
            for persona_results in per_persona.values():
                persona_variant = persona_results.get(variant.name)
                if not persona_variant:
                    continue
                value = persona_variant["metrics"].get(metric.source_key)
                if value is not None:
                    values.append(float(value))
            metric_summary[metric.name] = _summarize_values(values)
        macro_average[variant.name] = {
            "label": variant.label,
            "metrics": metric_summary,
        }
    return macro_average


def _build_table(
    variants: list[ExperimentVariant],
    macro_average: dict[str, Any],
    metric_specs: list[MetricSpec],
) -> dict[str, Any]:
    matrix_metrics = [{"name": metric.name, "higher_is_better": metric.higher_is_better} for metric in metric_specs]
    baselines: list[dict[str, Any]] = []
    proposed: dict[str, Any] | None = None
    for variant in variants:
        metrics = {
            metric.name: macro_average.get(variant.name, {}).get("metrics", {}).get(metric.name, {}).get("mean")
            for metric in metric_specs
        }
        method = {"name": variant.label, "metrics": metrics}
        if variant.name == "full_system":
            proposed = method
        else:
            baselines.append(method)
    if proposed is None:
        proposed = {"name": "Full System", "metrics": {metric.name: None for metric in metric_specs}}
    matrix = _build_comparison_matrix(baselines, proposed, matrix_metrics)
    return {
        **matrix,
        "latex": _comparison_matrix_to_latex(matrix),
    }


def _build_appendix_baselines(macro_average: dict[str, Any]) -> dict[str, Any]:
    appendix: dict[str, Any] = {}
    for variant in APPENDIX_VARIANTS:
        appendix[variant.name] = {
            "label": variant.label,
            "metrics": macro_average.get(variant.name, {}).get("metrics", {}),
        }
    return appendix


def _build_ablation_contributions(macro_average: dict[str, Any]) -> dict[str, Any]:
    full_metrics = {
        metric.name: macro_average.get("full_system", {}).get("metrics", {}).get(metric.name, {}).get("mean")
        for metric in CORE_METRICS
    }
    ablations = []
    for variant in MAIN_VARIANTS:
        if variant.name == "full_system":
            continue
        ablations.append(
            {
                "variant_name": variant.label,
                "metrics": {
                    metric.name: macro_average.get(variant.name, {}).get("metrics", {}).get(metric.name, {}).get("mean")
                    for metric in CORE_METRICS
                },
            }
        )
    return {
        metric.name: _quantify_ablation_contributions(full_metrics, ablations, metric.name, metric.higher_is_better)
        for metric in CORE_METRICS
    }


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 6)


def _mean_bool(values: Iterable[bool | None]) -> float | None:
    filtered = [1.0 if value else 0.0 for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 6)


def _summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "persona_count": 0, "ci": {"mean": None, "lower": None, "upper": None, "confidence": 0.95}}
    mean_value = sum(values) / len(values)
    std_value = 0.0
    if len(values) > 1:
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        std_value = math.sqrt(variance)
    ci = _bootstrap_ci(values)
    return {
        "mean": round(mean_value, 6),
        "std": round(std_value, 6),
        "persona_count": len(values),
        "ci": ci,
    }


def _to_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _bootstrap_ci(samples: list[float], n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42) -> dict[str, Any]:
    if len(samples) < 2:
        return {"mean": round(sum(samples) / len(samples), 6) if samples else None, "lower": None, "upper": None, "confidence": confidence}
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        boot = [rng.choice(samples) for _ in range(len(samples))]
        boot_means.append(sum(boot) / len(boot))
    boot_means.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    return {
        "mean": round(sum(samples) / len(samples), 6),
        "lower": round(boot_means[lo_idx], 6),
        "upper": round(boot_means[min(hi_idx, n_bootstrap - 1)], 6),
        "confidence": confidence,
    }


def _quantify_ablation_contributions(
    full_result: dict[str, Any],
    ablation_results: list[dict[str, Any]],
    primary_metric: str,
    higher_is_better: bool = True,
) -> list[dict[str, Any]]:
    full_score = full_result.get(primary_metric)
    if full_score is None or not isinstance(full_score, (int, float)):
        return []
    contributions = []
    for ablation in ablation_results:
        variant = ablation.get("variant_name", "unknown")
        metrics = ablation.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        ablated_score = metrics.get(primary_metric)
        if ablated_score is None or not isinstance(ablated_score, (int, float)):
            continue
        drop = full_score - ablated_score if higher_is_better else ablated_score - full_score
        relative = (drop / abs(full_score) * 100) if abs(full_score) > 1e-8 else 0.0
        contributions.append(
            {
                "component": variant,
                "full_model_score": round(full_score, 4),
                "without_component_score": round(ablated_score, 4),
                "absolute_drop": round(drop, 4),
                "relative_contribution_pct": round(relative, 2),
                "is_critical": relative > 10.0,
            }
        )
    contributions.sort(key=lambda item: item["absolute_drop"], reverse=True)
    return contributions


def _build_comparison_matrix(baselines: list[dict[str, Any]], proposed: dict[str, Any], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    all_methods = baselines + [proposed]
    headers = ["Method"] + [metric["name"] for metric in metrics]
    rows = []
    for method in all_methods:
        row = {"method": method.get("name", "Unknown"), "is_proposed": method is proposed}
        for metric in metrics:
            row[metric["name"]] = method.get("metrics", {}).get(metric["name"])
        rows.append(row)
    annotations: dict[str, str] = {}
    for metric in metrics:
        values = [(idx, row.get(metric["name"])) for idx, row in enumerate(rows) if isinstance(row.get(metric["name"]), (int, float))]
        if not values:
            continue
        values.sort(key=lambda item: item[1], reverse=metric.get("higher_is_better", True))
        annotations[f"{values[0][0]}:{metric['name']}"] = "best"
        if len(values) > 1:
            annotations[f"{values[1][0]}:{metric['name']}"] = "second"
    return {
        "headers": headers,
        "rows": rows,
        "annotations": annotations,
        "proposed_method_name": proposed.get("name", "Ours"),
    }



def _comparison_matrix_to_latex(matrix: dict[str, Any]) -> str:
    headers = matrix["headers"]
    rows = matrix["rows"]
    annotations = matrix["annotations"]
    col_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{_latex_escape_cell(header)}}}" for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row_index, row in enumerate(rows):
        cells = []
        method_name = _latex_escape_cell(row["method"])
        if row.get("is_proposed"):
            method_name = f"\\textbf{{{method_name}}} (Ours)"
        cells.append(method_name)
        for header in headers[1:]:
            value = row.get(header)
            if value is None:
                cells.append("--")
                continue
            formatted = f"{value:.2f}" if isinstance(value, float) and value < 1 else (f"{value:.1f}" if isinstance(value, float) else str(value))
            annotation = annotations.get(f"{row_index}:{header}")
            if annotation == "best":
                formatted = f"\\textbf{{{formatted}}}"
            elif annotation == "second":
                formatted = f"\\underline{{{formatted}}}"
            cells.append(formatted)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _latex_escape_cell(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("$", "\\$")
