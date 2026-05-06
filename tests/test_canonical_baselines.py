from __future__ import annotations

from nanoresearch.experiments.canonical_baselines import lookup_canonical_baseline
from tools.recompute_router_persona_canonical_deltas import recompute_record


def test_lookup_canonical_baseline_matches_alias() -> None:
    baseline = lookup_canonical_baseline("cv_small_image_cls", "Top-1 Accuracy")
    assert baseline is not None
    assert baseline["baseline_name"] == "ResNet-18"
    assert baseline["baseline_value"] == 0.949


def test_recompute_record_overrides_shared_baseline_and_delta() -> None:
    record = {
        "question_id": "tabular_budgeted_cls",
        "primary_metric_name": "Adult_accuracy",
        "final_performance": 0.8541,
        "baseline_performance": None,
        "delta_over_baseline": None,
        "metadata": {},
    }
    recomputed = recompute_record(record)
    assert recomputed["baseline_performance"] == 0.873
    assert recomputed["delta_over_baseline"] == -0.0189
    assert recomputed["raw_baseline_performance"] is None
    assert recomputed["raw_delta_over_baseline"] is None
    assert recomputed["metadata"]["canonical_baseline_applied"] is True
