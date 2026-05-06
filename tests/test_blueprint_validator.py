from nanoresearch.pipeline.blueprint_validator import validate_blueprint


def test_validator_accepts_semantic_ablation_reference_and_metric_aliases():
    blueprint = {
        "metrics": [
            {"name": "Accuracy", "primary": True, "higher_is_better": True},
            {"name": "F1 Score", "primary": False, "higher_is_better": True},
            {"name": "Parameter_count", "primary": False, "higher_is_better": False},
        ],
        "proposed_method": {
            "name": "Retrieval Gated LoRA",
            "description": "A retrieval-enhanced QA model with LoRA adapters and dynamic gating.",
            "architecture": "retrieval module -> gated fusion -> LoRA-adapted encoder",
            "key_components": ["retrieval module", "dynamic gating", "LoRA adapters"],
        },
        "ablation_groups": [
            {
                "group_name": "Retrieval depth",
                "description": "Vary the retrieval module top-k depth.",
                "variants": [
                    {"name": "Top-3 retrieval"},
                    {"name": "Top-5 retrieval"},
                ],
            },
            {
                "group_name": "LoRA rank",
                "description": "Sweep LoRA adapter rank.",
                "variants": [
                    {"name": "rank_4"},
                    {"name": "rank_8"},
                ],
            },
        ],
        "baselines": [
            {
                "name": "PubMed baseline",
                "expected_performance": {
                    "accuracy": 0.72,
                    "f1": 0.71,
                    "parameters": 110_000_000,
                },
            }
        ],
    }

    issues = validate_blueprint(blueprint)

    assert not any("doesn't reference anything" in issue for issue in issues)
    assert not any("not in the metrics list" in issue for issue in issues)


def test_validator_still_flags_unrelated_ablation_name():
    blueprint = {
        "metrics": [{"name": "Accuracy", "primary": True, "higher_is_better": True}],
        "proposed_method": {
            "name": "Retriever",
            "description": "A retrieval model with evidence fusion.",
            "architecture": "retrieval plus fusion",
            "key_components": ["retrieval", "fusion"],
        },
        "ablation_groups": [
            {
                "group_name": "Random sweep",
                "description": "A meaningless label that does not map to the method.",
                "variants": [{"name": "Blue setting"}],
            }
        ],
        "baselines": [],
    }

    issues = validate_blueprint(blueprint)

    assert any("doesn't reference anything" in issue for issue in issues)
