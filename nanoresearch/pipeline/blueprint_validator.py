"""Blueprint semantic validation — checks internal consistency after PLANNING."""

from __future__ import annotations

import logging
import re
from typing import Any

from nanoresearch import constants as nr_constants

logger = logging.getLogger(__name__)

_TOKEN_STOPWORDS = {
    "a", "an", "and", "as", "at", "based", "by", "default", "feature", "for",
    "full", "group", "high", "in", "is", "large", "low", "medium", "method",
    "model", "no", "of", "on", "only", "or", "ours", "proposed", "rank",
    "seed", "stage", "static", "strategy", "study", "sweep", "system", "test",
    "the", "threshold", "top", "variant", "variation", "vs", "with", "without",
}

_METRIC_ALIASES = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "f1": "f1",
    "f1score": "f1",
    "f1macro": "f1",
    "f1micro": "f1",
    "exactmatch": "em",
    "em": "em",
    "parameters": "parametercount",
    "parametercount": "parametercount",
    "parametercounts": "parametercount",
    "modelparameters": "parametercount",
    "trainableparameters": "parametercount",
    "latency": "latency",
    "inferencelatency": "latency",
    "inferencylatencyms": "latency",
    "latencyms": "latency",
    "latencymspersample": "latency",
    "trainingtokens": "trainingtokens",
    "tokens": "trainingtokens",
    "mmluscore": "mmlu",
    "mmluaccuracy": "mmlu",
    "mmlu": "mmlu",
    "mtbenchscore": "mtbench",
    "mt_bench_score": "mtbench",
    "scienceqaaccuracy": "scienceqa",
    "scienceqa": "scienceqa",
    "mmm uaccuracy": "mmmu",
    "mmmuaccuracy": "mmmu",
    "mmmu": "mmmu",
    "flops": "flops",
    "flopspersample": "flops",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _tokenize(value: str) -> set[str]:
    tokens = {
        token for token in _normalize_text(value).split()
        if token and token not in _TOKEN_STOPWORDS and not token.isdigit()
    }
    return tokens


def _normalize_metric_name(value: str) -> str:
    collapsed = re.sub(r"[^a-z0-9]+", "", value.lower())
    return _METRIC_ALIASES.get(collapsed, collapsed)


def validate_blueprint(blueprint: dict[str, Any]) -> list[str]:
    """Return a list of semantic issues found in the blueprint.

    An empty list means the blueprint passed all checks.
    """
    issues: list[str] = []

    # 1. Metrics list must be non-empty
    metrics = blueprint.get("metrics", [])
    if not metrics:
        issues.append("Blueprint has no evaluation metrics defined.")

    # 2. At least one primary metric
    has_primary = any(
        m.get("primary", False) for m in metrics if isinstance(m, dict)
    )
    if metrics and not has_primary:
        issues.append("No metric is marked as primary=True.")

    # 3. Metric direction consistency
    for m in metrics:
        if not isinstance(m, dict):
            continue
        name = m.get("name", "").lower()
        higher = m.get("higher_is_better", True)
        for pattern in nr_constants.LOWER_IS_BETTER_PATTERNS:
            if pattern in name and higher:
                issues.append(
                    f"Metric '{m.get('name')}' contains '{pattern}' but "
                    f"higher_is_better=True — likely should be False."
                )
                break

    # 4. Proposed method must have key_components
    pm = blueprint.get("proposed_method", {})
    if isinstance(pm, dict):
        kc = pm.get("key_components", [])
        if not kc:
            issues.append("proposed_method.key_components is empty.")

    # 5. Ablation variable names should reference key_components
    key_component_tokens: set[str] = set()
    if isinstance(pm, dict):
        for comp in pm.get("key_components", []):
            if isinstance(comp, str):
                key_component_tokens.update(_tokenize(comp))
    method_desc = ""
    if isinstance(pm, dict):
        method_desc = (
            pm.get("description", "") + " " +
            pm.get("architecture", "") + " " +
            " ".join(str(c) for c in pm.get("key_components", []))
        ).lower()
    method_tokens = _tokenize(method_desc) | key_component_tokens

    for ag in blueprint.get("ablation_groups", []):
        if not isinstance(ag, dict):
            continue
        group_name = ag.get("group_name", "")
        group_desc = ag.get("description", "")
        group_tokens = _tokenize(group_name) | _tokenize(group_desc)
        for variant in ag.get("variants", []):
            if not isinstance(variant, dict):
                continue
            var_name = variant.get("name", variant.get("variant_name", ""))
            variant_context = " ".join(
                str(variant.get(key, ""))
                for key in ("name", "variant_name", "description", "target", "component", "module")
            )
            variant_tokens = _tokenize(variant_context)

            # The old check required almost exact string overlap and produced many
            # false positives for sweeps like "Top-3 retrieval" or "LoRA rank 8".
            # A warning is only useful if neither the group nor the variant refers
            # to any method component in a normalized token space.
            if variant_tokens and method_tokens:
                if not ((variant_tokens & method_tokens) or (group_tokens & method_tokens)):
                    issues.append(
                        f"Ablation variant '{var_name}' in group "
                        f"'{group_name}' doesn't reference anything "
                        f"in the proposed method description."
                    )

    # 6. Baseline expected_performance metric names must match metrics list
    metric_names = {
        _normalize_metric_name(m.get("name", ""))
        for m in metrics if isinstance(m, dict) and m.get("name")
    }
    for bl in blueprint.get("baselines", []):
        if not isinstance(bl, dict):
            continue
        perf = bl.get("expected_performance", {})
        if isinstance(perf, dict):
            for metric_name in perf:
                normalized_metric = _normalize_metric_name(str(metric_name))
                if metric_names and normalized_metric not in metric_names:
                    issues.append(
                        f"Baseline '{bl.get('name')}' has performance for "
                        f"metric '{metric_name}' which is not in the "
                        f"metrics list: {metric_names}"
                    )

    return issues
