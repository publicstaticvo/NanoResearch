#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PRE_ROUTER_SYSTEM = (
    "You are a router making pre-execution decisions for NanoResearch. "
    "Return JSON only with keys selected_memory_ids, selected_skill_ids, prompt_plan, update_memory, update_skill. "
    "Use only ids listed in x.candidate_memory and x.candidate_skills. "
    "Select a focused subset, not everything. "
    "When task constraints conflict with persona defaults, prioritize task constraints. "
    "Set update_memory and update_skill to null. "
    "Keep prompt_plan under 30 words. Output one valid JSON object only."
)

POST_ROUTER_SYSTEM = (
    "You are a hindsight-improved router for NanoResearch. "
    "Return JSON only with keys selected_memory_ids, selected_skill_ids, prompt_plan, update_memory, update_skill. "
    "Use only ids listed in x.candidate_memory and x.candidate_skills plus any evolved ids already present in x.candidate_skills. "
    "Improve retrieval and prompt planning after seeing tool output and user feedback. "
    "When task constraints conflict with persona defaults, prioritize task constraints. "
    "Write update_memory only for stable preferences or recurring constraints. "
    "Write update_skill only for reusable procedural rules. "
    "Keep prompt_plan under 30 words. Keep each update to one short sentence. Output one valid JSON object only."
)

ROUTER_KEY_ORDER = [
    "selected_memory_ids",
    "selected_skill_ids",
    "prompt_plan",
    "update_memory",
    "update_skill",
]

STAGE_TURN_BUDGETS = {
    "method_generation": 5,
    "code_implementation": 10,
    "paper_writing": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export exact off-policy SDPO manifest for NanoResearch router.")
    parser.add_argument("--input-dir", action="append", required=True, help="Directory containing live_router_training_examples.jsonl")
    parser.add_argument("--output", required=True, help="Output manifest jsonl path")
    parser.add_argument("--stats-output", required=True, help="Output stats json path")
    parser.add_argument("--drop-report-output", required=True, help="Output drop-reasons json path")
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer/model path for exact token stats")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_path"] = str(path)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def parse_turn_int(turn_id: str | None) -> int:
    if not turn_id:
        return -1
    match = re.search(r"(\d+)$", str(turn_id))
    return int(match.group(1)) if match else -1


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def normalize_update(value: Any) -> str | None:
    text = normalize_text(value)
    return text or None


def normalize_id_list(value: Any) -> list[str]:
    if not value:
        return []
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


def canonical_router_action(action: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key in ROUTER_KEY_ORDER:
        if key.endswith("_ids"):
            canonical[key] = normalize_id_list(action.get(key))
        elif key.startswith("update_"):
            canonical[key] = normalize_update(action.get(key))
        else:
            canonical[key] = normalize_text(action.get(key))
    return canonical


def canonical_router_json(action: dict[str, Any]) -> str:
    return json.dumps(canonical_router_action(action), ensure_ascii=False, indent=2)


def render_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    chunks = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def tokenize_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def logical_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("persona_id", "")),
        str(row.get("task_id", "")),
        str(row.get("subsystem", "")),
        parse_turn_int(row.get("turn_id")),
    )


def best_attempt_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(row.get("attempt_no") or 0),
        str(row.get("_source_path", "")),
        int(row.get("_source_line") or 0),
    )


def get_profile_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("tool_trace", {}).get("tool_input", {}).get("profile_snapshot") or {})


def get_task_spec(row: dict[str, Any]) -> dict[str, Any]:
    x_task_spec = row.get("x", {}).get("task_spec")
    if isinstance(x_task_spec, dict) and x_task_spec:
        return dict(x_task_spec)
    tool_task_spec = row.get("tool_trace", {}).get("tool_input", {}).get("task_spec")
    if isinstance(tool_task_spec, dict) and tool_task_spec:
        return dict(tool_task_spec)
    return {
        "task_id": row.get("task_id"),
        "topic": row.get("task_topic"),
        "task_brief": "",
        "stage_focus": "",
    }


def candidate_id_set(items: list[dict[str, Any]], field: str) -> set[str]:
    values = set()
    for item in items:
        value = item.get(field)
        if value:
            values.add(str(value))
    return values


def validate_action_space(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    x = row.get("x", {})
    candidate_memory_ids = candidate_id_set(x.get("candidate_memory", []), "memory_id")
    candidate_skill_ids = candidate_id_set(x.get("candidate_skills", []), "skill_id")
    for key, valid_ids in (
        ("selected_memory_ids", candidate_memory_ids),
        ("selected_skill_ids", candidate_skill_ids),
    ):
        for action_name in ("y0", "y1"):
            action = canonical_router_action(row.get(action_name, {}))
            selected_ids = set(action.get(key, []))
            invalid = sorted(selected_ids - valid_ids)
            if invalid:
                reasons.append(f"{action_name}_{key}_outside_candidates")
    return reasons


def validate_row(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    tool_trace = row.get("tool_trace", {})
    y0 = canonical_router_action(row.get("y0", {}))
    y1 = canonical_router_action(row.get("y1", {}))

    if tool_trace.get("tool_status") != "ok":
        reasons.append("tool_status_not_ok")
    if tool_trace.get("tool_finish_reason") != "stop":
        reasons.append("tool_finish_reason_not_stop")
    if not y0.get("prompt_plan"):
        reasons.append("empty_y0_prompt_plan")
    if y0 == y1:
        reasons.append("y0_equals_y1")
    if parse_turn_int(row.get("turn_id")) < 1:
        reasons.append("missing_turn_id")
    if not get_profile_snapshot(row):
        reasons.append("missing_profile_snapshot")
    reasons.extend(validate_action_space(row))
    return reasons


def pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_pre_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Produce the base router decision before tool execution.",
        "persona_id": row["persona_id"],
        "round_id": row["round_id"],
        "subsystem": row["subsystem"],
        "turn_id": row["turn_id"],
        "stage_turn_budget": STAGE_TURN_BUDGETS[row["subsystem"]],
        "profile_snapshot": get_profile_snapshot(row),
        "task_spec": get_task_spec(row),
        "x": row["x"],
    }


def build_post_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Produce the hindsight-improved router decision after feedback.",
        "persona_id": row["persona_id"],
        "round_id": row["round_id"],
        "subsystem": row["subsystem"],
        "turn_id": row["turn_id"],
        "stage_turn_budget": STAGE_TURN_BUDGETS[row["subsystem"]],
        "profile_snapshot": get_profile_snapshot(row),
        "task_spec": get_task_spec(row),
        "x": row["x"],
        "y0": canonical_router_action(row["y0"]),
        "tool_trace": row["tool_trace"],
        "o": row["o"],
    }


def deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    selected: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    duplicate_rows = 0
    for row in rows:
        key = logical_key(row)
        if key in selected:
            duplicate_rows += 1
            if best_attempt_key(row) > best_attempt_key(selected[key]):
                selected[key] = row
        else:
            selected[key] = row
    kept_rows = sorted(
        selected.values(),
        key=lambda item: (
            str(item.get("persona_id", "")),
            str(item.get("task_id", "")),
            str(item.get("subsystem", "")),
            parse_turn_int(item.get("turn_id")),
        ),
    )
    return kept_rows, duplicate_rows


def export_manifest(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    drop_report_path = Path(args.drop_report_output)
    drop_report_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_rows: list[dict[str, Any]] = []
    for directory in args.input_dir:
        jsonl_path = Path(directory) / "live_router_training_examples.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing training examples file: {jsonl_path}")
        raw_rows.extend(read_jsonl(jsonl_path))

    deduped_rows, duplicate_rows = deduplicate_rows(raw_rows)

    drop_counts: Counter[str] = Counter()
    subsystem_counts: Counter[str] = Counter()
    persona_counts: Counter[str] = Counter()
    prompt_length_stats = {
        "base_prompt_tokens": [],
        "hindsight_prompt_tokens": [],
        "completion_tokens": [],
    }
    truncate_counts: Counter[str] = Counter()
    final_rows: list[dict[str, Any]] = []

    for row in deduped_rows:
        reasons = validate_row(row)
        if reasons:
            drop_counts.update(reasons)
            continue

        base_messages = [
            {"role": "system", "content": PRE_ROUTER_SYSTEM},
            {"role": "user", "content": pretty_json(build_pre_payload(row))},
        ]
        hindsight_messages = [
            {"role": "system", "content": POST_ROUTER_SYSTEM},
            {"role": "user", "content": pretty_json(build_post_payload(row))},
        ]
        target_text = canonical_router_json(row["y0"])

        base_prompt_tokens = tokenize_length(tokenizer, render_prompt_text(tokenizer, base_messages))
        hindsight_prompt_tokens = tokenize_length(tokenizer, render_prompt_text(tokenizer, hindsight_messages))
        completion_tokens = tokenize_length(tokenizer, target_text)

        base_prompt_truncated = base_prompt_tokens > args.max_prompt_length
        hindsight_prompt_truncated = hindsight_prompt_tokens > args.max_prompt_length
        if base_prompt_truncated:
            truncate_counts.update(["base_prompt_over_length"])
        if hindsight_prompt_truncated:
            truncate_counts.update(["hindsight_prompt_over_length"])
        if completion_tokens > args.max_completion_length:
            drop_counts.update(["completion_over_length"])
            continue

        turn = parse_turn_int(row["turn_id"])
        sample_id = f"{row['persona_id']}-{row['task_id']}-{row['subsystem']}-turn{turn}"
        manifest_row = {
            "sample_id": sample_id,
            "persona_id": row["persona_id"],
            "task_id": row["task_id"],
            "subsystem": row["subsystem"],
            "turn": turn,
            "attempt_no": int(row.get("attempt_no") or 0),
            "base_messages": base_messages,
            "hindsight_messages": hindsight_messages,
            "target_text": target_text,
            "base_prompt_tokens": base_prompt_tokens,
            "hindsight_prompt_tokens": hindsight_prompt_tokens,
            "completion_tokens": completion_tokens,
            "base_prompt_truncated": base_prompt_truncated,
            "hindsight_prompt_truncated": hindsight_prompt_truncated,
            "metadata": {
                "schema_version": row.get("schema_version"),
                "trajectory_id": row.get("trajectory_id"),
                "round_id": row.get("round_id"),
                "task_topic": row.get("task_topic"),
                "artifact_type": row.get("artifact_type"),
                "source_sample_id": row.get("source_sample_id"),
                "source_path": row.get("_source_path"),
                "source_line": row.get("_source_line"),
                "tool_model": row.get("tool_trace", {}).get("tool_model"),
                "tool_output_source": row.get("tool_trace", {}).get("tool_output_source"),
                "tool_finish_reason": row.get("tool_trace", {}).get("tool_finish_reason"),
                "critic_model": row.get("o", {}).get("critic_model"),
                "task_spec": get_task_spec(row),
                "y1": canonical_router_action(row["y1"]),
            },
        }
        final_rows.append(manifest_row)
        subsystem_counts.update([row["subsystem"]])
        persona_counts.update([row["persona_id"]])
        prompt_length_stats["base_prompt_tokens"].append(base_prompt_tokens)
        prompt_length_stats["hindsight_prompt_tokens"].append(hindsight_prompt_tokens)
        prompt_length_stats["completion_tokens"].append(completion_tokens)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    def stats(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 2),
        }

    stats_payload = {
        "raw_rows": len(raw_rows),
        "deduped_rows": len(deduped_rows),
        "duplicate_rows": duplicate_rows,
        "final_rows": len(final_rows),
        "subsystem_counts": dict(sorted(subsystem_counts.items())),
        "persona_counts": dict(sorted(persona_counts.items())),
        "length_stats": {name: stats(values) for name, values in prompt_length_stats.items()},
        "truncate_counts": dict(sorted(truncate_counts.items())),
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "input_dirs": args.input_dir,
    }
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    drop_report_path.write_text(json.dumps(dict(sorted(drop_counts.items())), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats_payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    export_manifest(args)


if __name__ == "__main__":
    main()
