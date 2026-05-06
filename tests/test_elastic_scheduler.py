from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nanoresearch.experiments.elastic_scheduler import (
    ElasticAssignmentScheduler,
    ElasticSchedulerConfig,
)


def _assignment(chain_id: str, round_id: int) -> dict:
    return {
        "assignment_id": f"{chain_id}::round{round_id:02d}",
        "chain_id": chain_id,
        "evolution_round": round_id,
        "evolution_total_rounds": 3,
        "persona_id": "persona-a",
        "variant_name": "full_system",
        "question": {"question_id": "q1"},
    }


def _build_scheduler(tmp_path: Path, assignments: list[dict], seed_completed: set[str] | None = None) -> ElasticAssignmentScheduler:
    return ElasticAssignmentScheduler(
        assignments,
        skipped_assignments=[],
        seed_completed_assignment_ids=seed_completed or set(),
        config=ElasticSchedulerConfig(
            output_dir=tmp_path / "batch",
            scheduler_root=tmp_path / "batch" / "_scheduler",
            worker_id="worker-01",
            heartbeat_seconds=5,
            claim_stale_seconds=30,
            poll_seconds=1,
            failure_cooldown_seconds=5,
        ),
    )


def test_elastic_scheduler_respects_round_dependencies_and_seed_completion(tmp_path) -> None:
    assignments = [
        _assignment("chain-a", 1),
        _assignment("chain-a", 2),
        _assignment("chain-a", 3),
    ]
    scheduler = _build_scheduler(
        tmp_path,
        assignments,
        seed_completed={assignments[0]["assignment_id"]},
    )
    scheduler._initialize_state()

    claimed_round2, _ = scheduler._claim_next_assignment("worker-01")
    assert claimed_round2 is not None
    assert claimed_round2["assignment_id"] == assignments[1]["assignment_id"]

    scheduler._mark_assignment_completed(
        "worker-01",
        claimed_round2,
        {"assignment_id": claimed_round2["assignment_id"], "workspace_path": "/tmp/ws2"},
    )

    claimed_round3, _ = scheduler._claim_next_assignment("worker-02")
    assert claimed_round3 is not None
    assert claimed_round3["assignment_id"] == assignments[2]["assignment_id"]

    status = json.loads((tmp_path / "batch" / "_scheduler" / "status.json").read_text(encoding="utf-8"))
    assert status["completed_count"] == 2
    assert status["remaining_count"] == 1


def test_elastic_scheduler_reclaims_stale_claims(tmp_path) -> None:
    assignments = [_assignment("chain-b", 1)]
    scheduler = _build_scheduler(tmp_path, assignments)
    scheduler._initialize_state()

    claimed, _ = scheduler._claim_next_assignment("worker-01")
    assert claimed is not None
    claim_path = scheduler._claim_path(claimed["assignment_id"])
    payload = json.loads(claim_path.read_text(encoding="utf-8"))
    payload["last_heartbeat_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    scheduler._write_json(claim_path, payload)

    reclaimed, _ = scheduler._claim_next_assignment("worker-02")
    assert reclaimed is not None
    assert reclaimed["assignment_id"] == claimed["assignment_id"]


def test_elastic_scheduler_failure_cooldown_blocks_immediate_reclaim(tmp_path) -> None:
    assignments = [_assignment("chain-c", 1)]
    scheduler = _build_scheduler(tmp_path, assignments)
    scheduler._initialize_state()

    claimed, _ = scheduler._claim_next_assignment("worker-01")
    assert claimed is not None
    scheduler._mark_assignment_failed("worker-01", claimed, RuntimeError("boom"))

    next_claim, snapshot = scheduler._claim_next_assignment("worker-02")
    assert next_claim is None
    assert snapshot["cooldown_blocked_count"] == 1
