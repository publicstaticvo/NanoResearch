from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any, Iterator

from .deep_persona_runner import run_assignment


_LOCK_ACQUIRE_TIMEOUT_SECONDS = 120.0
# The scheduler critical sections only perform small JSON/file updates.
# If a lock survives longer than the acquire timeout, it is very likely
# orphaned by a crashed worker on shared storage rather than legitimately held.
_LOCK_STALE_SECONDS = 90.0
_LOCK_POLL_INTERVAL_SECONDS = 0.2


@dataclass(frozen=True)
class ElasticSchedulerConfig:
    output_dir: Path
    scheduler_root: Path
    worker_id: str
    config_path: Path | None = None
    manifest_path: Path | None = None
    max_alignment_retries: int = 1
    disable_ideation_retrieval: bool = False
    heartbeat_seconds: int = 60
    claim_stale_seconds: int = 1800
    poll_seconds: int = 30
    failure_cooldown_seconds: int = 600
    round_wave_barrier: bool = True


def default_worker_id() -> str:
    hostname = socket.gethostname().split(".")[0]
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    pid = os.getpid()
    if slurm_job_id:
        return f"{hostname}-job{slurm_job_id}-pid{pid}"
    return f"{hostname}-pid{pid}"


class ElasticAssignmentScheduler:
    def __init__(
        self,
        assignments: list[dict[str, Any]],
        *,
        skipped_assignments: list[dict[str, Any]],
        seed_completed_assignment_ids: set[str],
        config: ElasticSchedulerConfig,
    ) -> None:
        self.assignments = list(assignments)
        self.skipped_assignments = list(skipped_assignments)
        self.seed_completed_assignment_ids = {str(item).strip() for item in seed_completed_assignment_ids if str(item).strip()}
        self.config = config

    async def run_worker(self) -> dict[str, Any]:
        self._initialize_state()
        worker_id = self.config.worker_id or default_worker_id()
        self._upsert_worker_state(
            worker_id,
            status="starting",
            current_assignment_id="",
            note="worker_start",
        )

        try:
            while True:
                assignment, snapshot = self._claim_next_assignment(worker_id)
                if assignment is None:
                    if bool(snapshot.get("all_completed")):
                        self._upsert_worker_state(
                            worker_id,
                            status="finished",
                            current_assignment_id="",
                            note="all_assignments_completed",
                        )
                        return snapshot
                    self._upsert_worker_state(
                        worker_id,
                        status="idle",
                        current_assignment_id="",
                        note=str(snapshot.get("waiting_reason") or "waiting"),
                    )
                    await asyncio.sleep(max(1, int(self.config.poll_seconds)))
                    continue

                assignment_id = str(assignment.get("assignment_id") or "")
                stop_event = asyncio.Event()
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(worker_id, assignment_id, stop_event)
                )
                try:
                    outcome = await run_assignment(
                        assignment,
                        output_dir=self.config.output_dir,
                        config_path=self.config.config_path,
                        max_alignment_retries=self.config.max_alignment_retries,
                        disable_ideation_retrieval=self.config.disable_ideation_retrieval,
                    )
                except Exception as exc:
                    stop_event.set()
                    await heartbeat_task
                    self._mark_assignment_failed(worker_id, assignment, exc)
                else:
                    stop_event.set()
                    await heartbeat_task
                    self._mark_assignment_completed(worker_id, assignment, outcome.record)
        finally:
            self._upsert_worker_state(
                worker_id,
                status="stopped",
                current_assignment_id="",
                note="worker_exit",
            )

    @property
    def _state_root(self) -> Path:
        return self.config.scheduler_root / "state"

    @property
    def _lock_path(self) -> Path:
        return self._state_root / "scheduler.lock"

    @property
    def _claims_root(self) -> Path:
        return self._state_root / "claims"

    @property
    def _workers_root(self) -> Path:
        return self._state_root / "workers"

    @property
    def _events_path(self) -> Path:
        return self._state_root / "events.jsonl"

    @property
    def _completed_index_path(self) -> Path:
        return self._state_root / "completed_index.json"

    @property
    def _failure_state_path(self) -> Path:
        return self._state_root / "failure_state.json"

    @property
    def _status_path(self) -> Path:
        return self.config.scheduler_root / "status.json"

    @property
    def _config_path(self) -> Path:
        return self.config.scheduler_root / "scheduler_config.json"

    @property
    def _selected_path(self) -> Path:
        return self.config.scheduler_root / "selected_assignments.json"

    @property
    def _skipped_path(self) -> Path:
        return self.config.scheduler_root / "skipped_assignments.json"

    @property
    def _manifest_copy_path(self) -> Path:
        return self.config.scheduler_root / "manifest.jsonl"

    @property
    def _results_path(self) -> Path:
        return self.config.output_dir / "results.jsonl"

    def _initialize_state(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.scheduler_root.mkdir(parents=True, exist_ok=True)
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._claims_root.mkdir(parents=True, exist_ok=True)
        self._workers_root.mkdir(parents=True, exist_ok=True)

        with self._locked():
            self._initialize_config_locked()
            completed_index = self._read_json(self._completed_index_path, {})
            if not isinstance(completed_index, dict):
                completed_index = {}

            changed = False
            now = self._now_iso()
            for assignment_id in sorted(self.seed_completed_assignment_ids):
                if assignment_id in completed_index:
                    continue
                completed_index[assignment_id] = {
                    "assignment_id": assignment_id,
                    "completed_at": now,
                    "source": "seed_completed_under",
                    "record_path": "",
                }
                changed = True
            if changed or not self._completed_index_path.exists():
                self._write_json(self._completed_index_path, completed_index)

            failure_state = self._read_json(self._failure_state_path, {})
            if not isinstance(failure_state, dict):
                failure_state = {}
                self._write_json(self._failure_state_path, failure_state)

            self._refresh_status_locked()

    def _initialize_config_locked(self) -> None:
        expected_signature = self._assignment_signature(self.assignments)
        payload = self._read_json(self._config_path, {})
        if payload:
            actual_signature = str(payload.get("assignment_signature") or "")
            if actual_signature != expected_signature:
                raise RuntimeError(
                    f"Elastic scheduler manifest mismatch at {self.config.scheduler_root}: "
                    f"expected signature {actual_signature!r}, got {expected_signature!r}"
                )
            return

        self._write_json(self._selected_path, self.assignments)
        self._write_json(self._skipped_path, self.skipped_assignments)
        self._write_manifest_jsonl(self._manifest_copy_path, self.assignments)
        self._write_json(
            self._config_path,
            {
                "created_at": self._now_iso(),
                "output_dir": str(self.config.output_dir),
                "scheduler_root": str(self.config.scheduler_root),
                "manifest_path": str(self.config.manifest_path or ""),
                "config_path": str(self.config.config_path or ""),
                "assignment_count": len(self.assignments),
                "assignment_signature": expected_signature,
                "heartbeat_seconds": int(self.config.heartbeat_seconds),
                "claim_stale_seconds": int(self.config.claim_stale_seconds),
                "poll_seconds": int(self.config.poll_seconds),
                "failure_cooldown_seconds": int(self.config.failure_cooldown_seconds),
                "round_wave_barrier": bool(self.config.round_wave_barrier),
            },
        )

    async def _heartbeat_loop(self, worker_id: str, assignment_id: str, stop_event: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=max(1, int(self.config.heartbeat_seconds)))
                break
            except (TimeoutError, asyncio.TimeoutError):
                try:
                    self._touch_claim(worker_id, assignment_id)
                except (TimeoutError, asyncio.TimeoutError):
                    # Missing a single heartbeat is acceptable because claims are
                    # only reclaimed after a much longer stale window. Treat lock
                    # contention here as transient instead of killing the worker.
                    continue

    def _touch_claim(self, worker_id: str, assignment_id: str) -> None:
        with self._locked():
            claim_path = self._claim_path(assignment_id)
            payload = self._read_json(claim_path, {})
            if not isinstance(payload, dict):
                return
            if str(payload.get("worker_id") or "") != worker_id:
                return
            now = self._now_iso()
            payload["last_heartbeat_at"] = now
            self._write_json(claim_path, payload)
            self._upsert_worker_state_locked(
                worker_id,
                status="busy",
                current_assignment_id=assignment_id,
                note="heartbeat",
            )
            self._refresh_status_locked()

    def _claim_next_assignment(self, worker_id: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        with self._locked():
            completed_index = self._read_json(self._completed_index_path, {})
            if not isinstance(completed_index, dict):
                completed_index = {}
            completed_ids = {str(item).strip() for item in completed_index.keys() if str(item).strip()}
            current_wave_round = self._current_wave_round(completed_ids)

            failure_state = self._read_json(self._failure_state_path, {})
            if not isinstance(failure_state, dict):
                failure_state = {}

            active_claims = self._load_active_claims_locked()

            for assignment in self.assignments:
                assignment_id = str(assignment.get("assignment_id") or "").strip()
                if not assignment_id:
                    continue
                if assignment_id in completed_ids:
                    continue
                if assignment_id in active_claims:
                    continue
                if self.config.round_wave_barrier and not self._wave_satisfied(assignment, current_wave_round):
                    continue
                if not self._prerequisites_satisfied(assignment, completed_ids):
                    continue
                if self._assignment_in_cooldown(assignment_id, failure_state):
                    continue

                now = self._now_iso()
                claim_payload = {
                    "assignment_id": assignment_id,
                    "chain_id": str(assignment.get("chain_id") or ""),
                    "worker_id": worker_id,
                    "claimed_at": now,
                    "last_heartbeat_at": now,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID", "").strip(),
                }
                try:
                    # Claim creation must be exclusive. Multiple workers can race here
                    # on shared storage, and ordinary overwrite semantics allow the
                    # later writer to silently steal the same assignment.
                    self._write_json_exclusive(self._claim_path(assignment_id), claim_payload)
                except FileExistsError:
                    continue
                self._upsert_worker_state_locked(
                    worker_id,
                    status="busy",
                    current_assignment_id=assignment_id,
                    note="claim_created",
                )
                self._append_jsonl(
                    self._events_path,
                    {
                        "ts": now,
                        "event": "assignment_claimed",
                        "worker_id": worker_id,
                        "assignment_id": assignment_id,
                    },
                )
                status = self._refresh_status_locked()
                return json.loads(json.dumps(assignment)), status

            waiting_reason = "waiting_for_other_workers"
            if self._has_remaining_assignments(completed_ids):
                if self.config.round_wave_barrier and self._has_wave_blocked_assignments(completed_ids):
                    waiting_reason = "waiting_on_round_barrier"
                elif self._has_dependency_blocked_assignments(completed_ids):
                    waiting_reason = "waiting_on_chain_dependency"
                elif self._has_cooldown_blocked_assignments(completed_ids, failure_state):
                    waiting_reason = "waiting_on_failure_cooldown"
            else:
                waiting_reason = "all_completed"

            self._upsert_worker_state_locked(
                worker_id,
                status="idle",
                current_assignment_id="",
                note=waiting_reason,
            )
            status = self._refresh_status_locked()
            status["waiting_reason"] = waiting_reason
            status["all_completed"] = not self._has_remaining_assignments(completed_ids)
            return None, status

    def _mark_assignment_completed(
        self,
        worker_id: str,
        assignment: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        assignment_id = str(assignment.get("assignment_id") or "").strip()
        if not assignment_id:
            return

        with self._locked():
            completed_index = self._read_json(self._completed_index_path, {})
            if not isinstance(completed_index, dict):
                completed_index = {}
            completed_index[assignment_id] = {
                "assignment_id": assignment_id,
                "completed_at": self._now_iso(),
                "source": "runtime",
                "record_path": str(
                    self.config.output_dir
                    / self._slugify(assignment_id)
                    / "result.json"
                ),
            }
            self._write_json(self._completed_index_path, completed_index)

            failure_state = self._read_json(self._failure_state_path, {})
            if isinstance(failure_state, dict) and assignment_id in failure_state:
                failure_state.pop(assignment_id, None)
                self._write_json(self._failure_state_path, failure_state)

            self._append_jsonl(self._results_path, record)
            self._unlink_if_exists(self._claim_path(assignment_id))
            self._append_jsonl(
                self._events_path,
                {
                    "ts": self._now_iso(),
                    "event": "assignment_completed",
                    "worker_id": worker_id,
                    "assignment_id": assignment_id,
                },
            )
            self._upsert_worker_state_locked(
                worker_id,
                status="idle",
                current_assignment_id="",
                note="assignment_completed",
            )
            self._refresh_status_locked()

    def _mark_assignment_failed(
        self,
        worker_id: str,
        assignment: dict[str, Any],
        exc: Exception,
    ) -> None:
        assignment_id = str(assignment.get("assignment_id") or "").strip()
        if not assignment_id:
            return

        with self._locked():
            failure_state = self._read_json(self._failure_state_path, {})
            if not isinstance(failure_state, dict):
                failure_state = {}
            previous = failure_state.get(assignment_id) if isinstance(failure_state.get(assignment_id), dict) else {}
            attempts = int(previous.get("attempts") or 0) + 1
            cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=max(1, int(self.config.failure_cooldown_seconds)))
            failure_state[assignment_id] = {
                "assignment_id": assignment_id,
                "attempts": attempts,
                "last_failed_at": self._now_iso(),
                "cooldown_until": cooldown_until.isoformat(),
                "last_error": f"{type(exc).__name__}: {exc}",
            }
            self._write_json(self._failure_state_path, failure_state)
            self._unlink_if_exists(self._claim_path(assignment_id))
            self._append_jsonl(
                self._events_path,
                {
                    "ts": self._now_iso(),
                    "event": "assignment_failed",
                    "worker_id": worker_id,
                    "assignment_id": assignment_id,
                    "attempts": attempts,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            self._upsert_worker_state_locked(
                worker_id,
                status="idle",
                current_assignment_id="",
                note=f"assignment_failed:{assignment_id}",
            )
            self._refresh_status_locked()

    def _load_active_claims_locked(self) -> dict[str, dict[str, Any]]:
        claims: dict[str, dict[str, Any]] = {}
        stale_paths: list[Path] = []
        now = datetime.now(timezone.utc)

        for path in sorted(self._claims_root.glob("*.json")):
            payload = self._read_json(path, {})
            if not isinstance(payload, dict):
                stale_paths.append(path)
                continue
            assignment_id = str(payload.get("assignment_id") or "").strip()
            if not assignment_id:
                stale_paths.append(path)
                continue
            heartbeat = self._parse_iso(str(payload.get("last_heartbeat_at") or ""))
            if heartbeat is None:
                stale_paths.append(path)
                continue
            age = (now - heartbeat).total_seconds()
            if age > max(1, int(self.config.claim_stale_seconds)):
                stale_paths.append(path)
                self._append_jsonl(
                    self._events_path,
                    {
                        "ts": self._now_iso(),
                        "event": "assignment_claim_reclaimed",
                        "assignment_id": assignment_id,
                        "worker_id": str(payload.get("worker_id") or ""),
                        "stale_seconds": round(age, 1),
                    },
                )
                continue
            claims[assignment_id] = payload

        for path in stale_paths:
            self._unlink_if_exists(path)
        return claims

    def _prerequisites_satisfied(self, assignment: dict[str, Any], completed_ids: set[str]) -> bool:
        round_id = int(assignment.get("evolution_round") or 1)
        if round_id <= 1:
            return True
        chain_id = str(assignment.get("chain_id") or "").strip()
        if not chain_id:
            return False
        previous_assignment_id = f"{chain_id}::round{round_id - 1:02d}"
        return previous_assignment_id in completed_ids

    def _current_wave_round(self, completed_ids: set[str]) -> int:
        remaining_rounds: list[int] = []
        for assignment in self.assignments:
            assignment_id = str(assignment.get("assignment_id") or "").strip()
            if not assignment_id or assignment_id in completed_ids:
                continue
            remaining_rounds.append(max(1, int(assignment.get("evolution_round") or 1)))
        return min(remaining_rounds) if remaining_rounds else 0

    @staticmethod
    def _wave_satisfied(assignment: dict[str, Any], current_wave_round: int) -> bool:
        if current_wave_round <= 0:
            return True
        return max(1, int(assignment.get("evolution_round") or 1)) == current_wave_round

    def _assignment_in_cooldown(self, assignment_id: str, failure_state: dict[str, Any]) -> bool:
        payload = failure_state.get(assignment_id)
        if not isinstance(payload, dict):
            return False
        cooldown_until = self._parse_iso(str(payload.get("cooldown_until") or ""))
        if cooldown_until is None:
            return False
        return cooldown_until > datetime.now(timezone.utc)

    def _has_remaining_assignments(self, completed_ids: set[str]) -> bool:
        for assignment in self.assignments:
            assignment_id = str(assignment.get("assignment_id") or "").strip()
            if assignment_id and assignment_id not in completed_ids:
                return True
        return False

    def _has_dependency_blocked_assignments(self, completed_ids: set[str]) -> bool:
        for assignment in self.assignments:
            assignment_id = str(assignment.get("assignment_id") or "").strip()
            if not assignment_id or assignment_id in completed_ids:
                continue
            if not self._prerequisites_satisfied(assignment, completed_ids):
                return True
        return False

    def _has_wave_blocked_assignments(self, completed_ids: set[str]) -> bool:
        current_wave_round = self._current_wave_round(completed_ids)
        if current_wave_round <= 0:
            return False
        for assignment in self.assignments:
            assignment_id = str(assignment.get("assignment_id") or "").strip()
            if not assignment_id or assignment_id in completed_ids:
                continue
            if not self._wave_satisfied(assignment, current_wave_round):
                return True
        return False

    def _has_cooldown_blocked_assignments(
        self,
        completed_ids: set[str],
        failure_state: dict[str, Any],
    ) -> bool:
        for assignment in self.assignments:
            assignment_id = str(assignment.get("assignment_id") or "").strip()
            if not assignment_id or assignment_id in completed_ids:
                continue
            if self._assignment_in_cooldown(assignment_id, failure_state):
                return True
        return False

    def _refresh_status_locked(self) -> dict[str, Any]:
        completed_index = self._read_json(self._completed_index_path, {})
        if not isinstance(completed_index, dict):
            completed_index = {}
        completed_ids = {str(item).strip() for item in completed_index.keys() if str(item).strip()}

        failure_state = self._read_json(self._failure_state_path, {})
        if not isinstance(failure_state, dict):
            failure_state = {}

        active_claims = self._load_active_claims_locked()
        current_wave_round = self._current_wave_round(completed_ids)
        remaining = 0
        eligible = 0
        wave_blocked = 0
        dependency_blocked = 0
        cooldown_blocked = 0
        claimed = 0
        for assignment in self.assignments:
            assignment_id = str(assignment.get("assignment_id") or "").strip()
            if not assignment_id or assignment_id in completed_ids:
                continue
            remaining += 1
            if assignment_id in active_claims:
                claimed += 1
                continue
            if self.config.round_wave_barrier and not self._wave_satisfied(assignment, current_wave_round):
                wave_blocked += 1
                continue
            if not self._prerequisites_satisfied(assignment, completed_ids):
                dependency_blocked += 1
                continue
            if self._assignment_in_cooldown(assignment_id, failure_state):
                cooldown_blocked += 1
                continue
            eligible += 1

        active_workers = 0
        idle_workers = 0
        busy_workers = 0
        now = datetime.now(timezone.utc)
        for path in sorted(self._workers_root.glob("*.json")):
            payload = self._read_json(path, {})
            if not isinstance(payload, dict):
                continue
            last_heartbeat = self._parse_iso(str(payload.get("last_heartbeat_at") or ""))
            if last_heartbeat is None:
                continue
            status = str(payload.get("status") or "").strip().lower()
            if status in {"finished", "stopped"}:
                continue
            if (now - last_heartbeat).total_seconds() > max(120, int(self.config.claim_stale_seconds)):
                continue
            active_workers += 1
            if status == "busy":
                busy_workers += 1
            else:
                idle_workers += 1

        status_payload = {
            "updated_at": self._now_iso(),
            "assignment_count": len(self.assignments),
            "completed_count": sum(
                1
                for assignment in self.assignments
                if str(assignment.get("assignment_id") or "").strip() in completed_ids
            ),
            "current_wave_round": current_wave_round,
            "remaining_count": remaining,
            "eligible_count": eligible,
            "claimed_count": claimed,
            "wave_blocked_count": wave_blocked,
            "dependency_blocked_count": dependency_blocked,
            "cooldown_blocked_count": cooldown_blocked,
            "active_worker_count": active_workers,
            "busy_worker_count": busy_workers,
            "idle_worker_count": idle_workers,
            "all_completed": remaining == 0,
        }
        self._write_json(self._status_path, status_payload)
        return status_payload

    def _upsert_worker_state(
        self,
        worker_id: str,
        *,
        status: str,
        current_assignment_id: str,
        note: str,
    ) -> None:
        with self._locked():
            self._upsert_worker_state_locked(
                worker_id,
                status=status,
                current_assignment_id=current_assignment_id,
                note=note,
            )
            self._refresh_status_locked()

    def _upsert_worker_state_locked(
        self,
        worker_id: str,
        *,
        status: str,
        current_assignment_id: str,
        note: str,
    ) -> None:
        path = self._worker_path(worker_id)
        existing = self._read_json(path, {})
        if not isinstance(existing, dict):
            existing = {}
        payload = {
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "").strip(),
            "started_at": str(existing.get("started_at") or self._now_iso()),
            "last_heartbeat_at": self._now_iso(),
            "status": status,
            "current_assignment_id": current_assignment_id,
            "note": note,
        }
        self._write_json(path, payload)

    def _worker_path(self, worker_id: str) -> Path:
        return self._workers_root / f"{self._slugify(worker_id)}.json"

    def _claim_path(self, assignment_id: str) -> Path:
        digest = hashlib.sha1(assignment_id.encode("utf-8")).hexdigest()[:12]
        return self._claims_root / f"{self._slugify(assignment_id)}-{digest}.json"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_dir = self._lock_path.with_suffix(".lockdir")
        owner_path = lock_dir / "owner.json"
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _LOCK_ACQUIRE_TIMEOUT_SECONDS
        acquired = False

        while not acquired:
            try:
                lock_dir.mkdir()
                owner_path.write_text(
                    json.dumps(
                        {
                            "created_at": self._now_iso(),
                            "hostname": socket.gethostname(),
                            "pid": os.getpid(),
                            "worker_id": self.config.worker_id,
                            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "").strip(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                acquired = True
            except FileExistsError:
                created_at = None
                if owner_path.is_file():
                    try:
                        payload = json.loads(owner_path.read_text(encoding="utf-8"))
                        created_at = self._parse_iso(str(payload.get("created_at") or ""))
                    except (OSError, json.JSONDecodeError, ValueError):
                        created_at = None
                if created_at is None:
                    try:
                        created_at = datetime.fromtimestamp(lock_dir.stat().st_mtime, timezone.utc)
                    except OSError:
                        created_at = None
                if created_at is not None:
                    age = (datetime.now(timezone.utc) - created_at).total_seconds()
                    if age > _LOCK_STALE_SECONDS:
                        shutil.rmtree(lock_dir, ignore_errors=True)
                        continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring elastic scheduler lock: {lock_dir}")
                time.sleep(_LOCK_POLL_INTERVAL_SECONDS)

        try:
            yield
        finally:
            try:
                owner_path.unlink(missing_ok=True)
            finally:
                try:
                    lock_dir.rmdir()
                except OSError:
                    shutil.rmtree(lock_dir, ignore_errors=True)

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

    @staticmethod
    def _write_json_exclusive(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o644)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @classmethod
    def _write_manifest_jsonl(cls, path: Path, assignments: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for assignment in assignments:
                handle.write(json.dumps(assignment, ensure_ascii=False) + "\n")

    @staticmethod
    def _unlink_if_exists(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = []
        for char in str(value):
            if char.isalnum() or char in "._-":
                cleaned.append(char)
            else:
                cleaned.append("-")
        slug = "".join(cleaned).strip("-").lower()
        return slug or "item"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _parse_iso(value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @classmethod
    def _assignment_signature(cls, assignments: list[dict[str, Any]]) -> str:
        assignment_ids = [str(item.get("assignment_id") or "") for item in assignments]
        payload = "\n".join(assignment_ids).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()


async def run_elastic_manifest(
    assignments: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    skipped_assignments: list[dict[str, Any]],
    seed_completed_assignment_ids: set[str],
    scheduler_root: str | Path,
    worker_id: str = "",
    config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    max_alignment_retries: int = 1,
    disable_ideation_retrieval: bool = False,
    heartbeat_seconds: int = 60,
    claim_stale_seconds: int = 1800,
    poll_seconds: int = 30,
    failure_cooldown_seconds: int = 600,
    round_wave_barrier: bool | None = None,
) -> dict[str, Any]:
    if round_wave_barrier is None:
        round_wave_barrier = str(os.environ.get("NANO_ELASTIC_ROUND_WAVE_BARRIER", "1")).strip().lower() not in {"0", "false", "no", "off"}
    scheduler = ElasticAssignmentScheduler(
        assignments,
        skipped_assignments=skipped_assignments,
        seed_completed_assignment_ids=seed_completed_assignment_ids,
        config=ElasticSchedulerConfig(
            output_dir=Path(output_dir),
            scheduler_root=Path(scheduler_root),
            worker_id=worker_id or default_worker_id(),
            config_path=Path(config_path) if config_path else None,
            manifest_path=Path(manifest_path) if manifest_path else None,
            max_alignment_retries=max(0, int(max_alignment_retries)),
            disable_ideation_retrieval=bool(disable_ideation_retrieval),
            heartbeat_seconds=max(1, int(heartbeat_seconds)),
            claim_stale_seconds=max(60, int(claim_stale_seconds)),
            poll_seconds=max(1, int(poll_seconds)),
            failure_cooldown_seconds=max(1, int(failure_cooldown_seconds)),
            round_wave_barrier=bool(round_wave_barrier),
        ),
    )
    return await scheduler.run_worker()
