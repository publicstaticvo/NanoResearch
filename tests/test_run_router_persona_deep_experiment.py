from __future__ import annotations

import argparse
import asyncio
import json

from tools.run_router_persona_deep_experiment import _main_async, load_completed_assignment_ids


def test_load_completed_assignment_ids_prefers_shallow_results_jsonl(tmp_path) -> None:
    root = tmp_path / "batch_root"
    shard = root / "persona__question"
    shard.mkdir(parents=True)
    (shard / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"assignment_id": "a::base_router::q::round01"}),
                json.dumps({"assignment_id": "a::full_system::q::round01"}),
            ]
        ),
        encoding="utf-8",
    )

    completed = load_completed_assignment_ids([str(root)])

    assert completed == {
        "a::base_router::q::round01",
        "a::full_system::q::round01",
    }


def test_load_completed_assignment_ids_falls_back_to_result_json_scan(tmp_path) -> None:
    root = tmp_path / "legacy_root"
    result_dir = root / "nested" / "attempt"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps({"assignment_id": "legacy::memory_skill::q::round02"}),
        encoding="utf-8",
    )

    completed = load_completed_assignment_ids([str(root)])

    assert completed == {"legacy::memory_skill::q::round02"}


def test_main_async_elastic_includes_output_dir_in_completed_scan(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    output_dir = tmp_path / "batch"
    output_dir.mkdir(parents=True)
    (output_dir / "results.jsonl").write_text(
        json.dumps({"assignment_id": "persona-a::base_router::q1::round01"}) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "assignment_id": "persona-a::base_router::q1::round02",
                "chain_id": "persona-a::base_router::q1",
                "evolution_round": 2,
                "evolution_total_rounds": 3,
                "persona_id": "persona-a",
                "variant_name": "base_router",
                "question": {"question_id": "q1"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    async def _fake_run_elastic_manifest(selected, **kwargs):
        captured["selected"] = selected
        captured["kwargs"] = kwargs
        return {"all_completed": False, "remaining_count": 1}

    monkeypatch.setattr(
        "tools.run_router_persona_deep_experiment.run_elastic_manifest",
        _fake_run_elastic_manifest,
    )

    args = argparse.Namespace(
        manifest=str(manifest_path),
        output_dir=str(output_dir),
        config=None,
        skip_completed_under=[],
        persona=[],
        variant=[],
        question=[],
        limit=0,
        max_alignment_retries=1,
        skip_sdpo_variants=False,
        disable_ideation_retrieval=False,
        elastic=True,
        scheduler_root="",
        worker_id="worker-x",
        heartbeat_seconds=60,
        claim_stale_seconds=1800,
        poll_seconds=30,
        failure_cooldown_seconds=600,
    )

    exit_code = asyncio.run(_main_async(args))

    assert exit_code == 0
    assert captured["selected"][0]["assignment_id"] == "persona-a::base_router::q1::round02"
    assert str(output_dir) in captured["kwargs"]["seed_completed_assignment_ids"]


def test_main_async_elastic_reuses_existing_scheduler_selection(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    output_dir = tmp_path / "batch"
    scheduler_root = output_dir / "_scheduler"
    scheduler_root.mkdir(parents=True)
    selected_path = scheduler_root / "selected_assignments.json"
    selected_payload = [
        {
            "assignment_id": "persona-a::base_router::q1::round01",
            "chain_id": "persona-a::base_router::q1",
            "evolution_round": 1,
            "evolution_total_rounds": 3,
            "persona_id": "persona-a",
            "variant_name": "base_router",
            "question": {"question_id": "q1"},
        },
        {
            "assignment_id": "persona-a::base_router::q1::round02",
            "chain_id": "persona-a::base_router::q1",
            "evolution_round": 2,
            "evolution_total_rounds": 3,
            "persona_id": "persona-a",
            "variant_name": "base_router",
            "question": {"question_id": "q1"},
        },
    ]
    selected_path.write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text("", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.jsonl").write_text(
        json.dumps({"assignment_id": "persona-a::base_router::q1::round01"}) + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    async def _fake_run_elastic_manifest(selected, **kwargs):
        captured["selected"] = selected
        captured["kwargs"] = kwargs
        return {"all_completed": False, "remaining_count": 1}

    monkeypatch.setattr(
        "tools.run_router_persona_deep_experiment.run_elastic_manifest",
        _fake_run_elastic_manifest,
    )

    args = argparse.Namespace(
        manifest=str(manifest_path),
        output_dir=str(output_dir),
        config=None,
        skip_completed_under=[],
        persona=[],
        variant=[],
        question=[],
        limit=0,
        max_alignment_retries=1,
        skip_sdpo_variants=False,
        disable_ideation_retrieval=False,
        elastic=True,
        scheduler_root="",
        worker_id="worker-y",
        heartbeat_seconds=60,
        claim_stale_seconds=1800,
        poll_seconds=30,
        failure_cooldown_seconds=600,
    )

    exit_code = asyncio.run(_main_async(args))

    assert exit_code == 0
    assert [row["assignment_id"] for row in captured["selected"]] == [
        "persona-a::base_router::q1::round01",
        "persona-a::base_router::q1::round02",
    ]
