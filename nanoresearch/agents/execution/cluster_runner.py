"""Cluster execution: SLURM job lifecycle, monitoring, and recovery."""
from __future__ import annotations

import asyncio
import atexit
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

from nanoresearch.agents.constants import CLUSTER_POLL_INTERVAL

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"
_ACTIVE_JOB_STATUSES = {
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
}
_TERMINAL_JOB_STATUSES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
    "TIMEOUT",
}
_WORKSPACE_STATE_RELATIVE_PATH = "logs/cluster_job_state.json"
_WORKSPACE_EVENTS_RELATIVE_PATH = "logs/cluster_job_events.jsonl"
_ASSIGNMENT_CONTEXT_RELATIVE_PATH = "logs/deep_assignment_context.json"
_ACTIVE_JOB_ID_FILENAME = "active_job_id.txt"
_DEFAULT_CLUSTER_STALL_TIMEOUT_SECONDS = 7200
_STATE_FILE_NAMES = {
    "cluster_job_state.json",
    "cluster_job_events.jsonl",
    _ACTIVE_JOB_ID_FILENAME,
}

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its descendants."""
    try:
        if _IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        else:
            try:
                result = subprocess.run(
                    ["pgrep", "-P", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for child in result.stdout.strip().split():
                    if child.strip():
                        _kill_process_tree(int(child))
            except Exception:  # noqa: BLE001
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    except Exception:  # noqa: BLE001
        pass


class _ClusterRunnerMixin:
    """Mixin implementing robust cluster child-job lifecycle management."""

    _cached_proxy_env: dict[str, str] | None = None
    _atexit_registered: bool = False
    _atexit_job_ids: set[str] = set()

    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _iso_from_epoch(epoch: float | None) -> str:
        if epoch is None:
            return ""
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat()

    @staticmethod
    def _registry_root() -> Path:
        override = os.environ.get("NANORESEARCH_EXECUTION_REGISTRY_ROOT", "").strip()
        if override:
            return Path(override)
        return Path.home() / ".nanoresearch" / "execution_registry" / "assignments"

    @staticmethod
    def _registry_path_for_assignment(assignment_id: str) -> Path:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", assignment_id).strip("-").lower() or "assignment"
        digest = hashlib.sha1(assignment_id.encode("utf-8")).hexdigest()[:12]
        filename = f"{cleaned[:80]}-{digest}.json"
        return _ClusterRunnerMixin._registry_root() / filename

    @staticmethod
    @contextmanager
    def _registry_lock(lock_path: Path) -> Iterator[None]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    @staticmethod
    def _read_json_path(path: Path, default: Any) -> Any:
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _write_json_path(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

    @staticmethod
    def _append_jsonl_path(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @classmethod
    def _ensure_atexit_cleanup_installed(cls) -> None:
        if cls._atexit_registered:
            return

        def _cleanup() -> None:
            for job_id in sorted(cls._atexit_job_ids):
                try:
                    subprocess.run(
                        ["scancel", job_id],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                except Exception:  # noqa: BLE001
                    continue

        atexit.register(_cleanup)
        cls._atexit_registered = True

    @classmethod
    def _register_atexit_job(cls, job_id: str) -> None:
        if not job_id:
            return
        cls._ensure_atexit_cleanup_installed()
        cls._atexit_job_ids.add(job_id)

    @classmethod
    def _unregister_atexit_job(cls, job_id: str) -> None:
        if not job_id:
            return
        cls._atexit_job_ids.discard(job_id)

    def _cluster_state_path(self) -> Path:
        return self.workspace.path / _WORKSPACE_STATE_RELATIVE_PATH

    def _cluster_events_path(self) -> Path:
        return self.workspace.path / _WORKSPACE_EVENTS_RELATIVE_PATH

    def _load_cluster_state(self) -> dict[str, Any]:
        payload = self._read_json_path(self._cluster_state_path(), {})
        return payload if isinstance(payload, dict) else {}

    def _write_cluster_state(self, payload: dict[str, Any]) -> None:
        self._write_json_path(self._cluster_state_path(), payload)

    def _record_cluster_event(self, event_type: str, **payload: Any) -> None:
        self._append_jsonl_path(
            self._cluster_events_path(),
            {
                "timestamp": self._now_utc_iso(),
                "event_type": event_type,
                **payload,
            },
        )

    def _load_assignment_context(self) -> dict[str, Any]:
        payload = self._read_json_path(
            self.workspace.path / _ASSIGNMENT_CONTEXT_RELATIVE_PATH,
            {},
        )
        return payload if isinstance(payload, dict) else {}

    def _active_job_tracker_path(self, code_dir: Path | None) -> Path | None:
        if code_dir is None:
            return None
        return code_dir / "logs" / _ACTIVE_JOB_ID_FILENAME

    def _write_active_job_tracker(self, code_dir: Path | None, job_id: str) -> None:
        tracker = self._active_job_tracker_path(code_dir)
        if tracker is None:
            return
        tracker.parent.mkdir(parents=True, exist_ok=True)
        tracker.write_text(job_id, encoding="utf-8")

    def _clear_active_job_tracker(self, code_dir: Path | None, *, job_id: str = "") -> None:
        tracker = self._active_job_tracker_path(code_dir)
        if tracker is None or not tracker.exists():
            return
        try:
            if job_id:
                tracked = tracker.read_text(encoding="utf-8").strip()
                if tracked and tracked != job_id:
                    return
            tracker.unlink(missing_ok=True)
        except OSError:
            return

    @staticmethod
    def _outer_slurm_job_id() -> str:
        return (
            os.environ.get("SLURM_JOB_ID", "").strip()
            or os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
        )

    def _build_cluster_state_payload(
        self,
        *,
        job_id: str,
        code_dir: Path,
        assignment_context: dict[str, Any],
        initial_status: str,
        source: str,
    ) -> dict[str, Any]:
        existing = self._load_cluster_state()
        now = self._now_utc_iso()
        latest_activity_at, latest_activity_path = self._latest_activity_snapshot(code_dir)
        return {
            "assignment_id": str(assignment_context.get("assignment_id") or existing.get("assignment_id") or ""),
            "chain_id": str(assignment_context.get("chain_id") or existing.get("chain_id") or ""),
            "workspace_path": str(self.workspace.path),
            "code_dir": str(code_dir),
            "outer_slurm_job_id": self._outer_slurm_job_id(),
            "child_slurm_job_id": job_id,
            "registered_at": str(existing.get("registered_at") or now),
            "last_heartbeat_at": now,
            "last_seen_status": initial_status,
            "latest_activity_at": latest_activity_at,
            "latest_activity_path": latest_activity_path,
            "source": source,
            "finalized_at": "",
            "terminal_status": "",
            "termination_reason": "",
        }

    def _sync_assignment_registry_from_state(self, state: dict[str, Any]) -> None:
        assignment_id = str(state.get("assignment_id") or "").strip()
        job_id = str(state.get("child_slurm_job_id") or "").strip()
        if not assignment_id or not job_id:
            return

        registry_path = self._registry_path_for_assignment(assignment_id)
        lock_path = registry_path.with_suffix(".lock")
        with self._registry_lock(lock_path):
            payload = self._read_json_path(registry_path, {"assignment_id": assignment_id, "records": []})
            records = payload.get("records") if isinstance(payload, dict) else []
            if not isinstance(records, list):
                records = []
            merged = {
                "assignment_id": assignment_id,
                "chain_id": str(state.get("chain_id") or ""),
                "workspace_path": str(state.get("workspace_path") or ""),
                "code_dir": str(state.get("code_dir") or ""),
                "outer_slurm_job_id": str(state.get("outer_slurm_job_id") or ""),
                "child_slurm_job_id": job_id,
                "registered_at": str(state.get("registered_at") or ""),
                "last_heartbeat_at": str(state.get("last_heartbeat_at") or ""),
                "last_seen_status": str(state.get("last_seen_status") or ""),
                "latest_activity_at": str(state.get("latest_activity_at") or ""),
                "latest_activity_path": str(state.get("latest_activity_path") or ""),
                "finalized_at": str(state.get("finalized_at") or ""),
                "terminal_status": str(state.get("terminal_status") or ""),
                "termination_reason": str(state.get("termination_reason") or ""),
                "source": str(state.get("source") or ""),
                "is_active": not bool(state.get("finalized_at")),
            }
            replaced = False
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                if str(record.get("child_slurm_job_id") or "").strip() == job_id:
                    records[index] = merged
                    replaced = True
                    break
            if not replaced:
                records.append(merged)
            payload = {
                "assignment_id": assignment_id,
                "updated_at": self._now_utc_iso(),
                "records": records,
            }
            self._write_json_path(registry_path, payload)

    def _register_active_cluster_job(
        self,
        *,
        job_id: str,
        code_dir: Path,
        assignment_context: dict[str, Any] | None,
        initial_status: str,
        source: str,
    ) -> None:
        payload = self._build_cluster_state_payload(
            job_id=job_id,
            code_dir=code_dir,
            assignment_context=assignment_context or {},
            initial_status=initial_status,
            source=source,
        )
        self._write_cluster_state(payload)
        self._write_active_job_tracker(code_dir, job_id)
        self._sync_assignment_registry_from_state(payload)
        self._register_atexit_job(job_id)
        self._record_cluster_event(
            "job_registered",
            job_id=job_id,
            status=initial_status,
            source=source,
            code_dir=str(code_dir),
            assignment_id=payload.get("assignment_id", ""),
        )

    def _update_tracked_cluster_job(
        self,
        job_id: str,
        *,
        last_seen_status: str,
        latest_activity_at: str = "",
        latest_activity_path: str = "",
    ) -> None:
        state = self._load_cluster_state()
        if str(state.get("child_slurm_job_id") or "").strip() != job_id:
            return
        previous_status = str(state.get("last_seen_status") or "").strip().upper()
        current_status = str(last_seen_status or "").strip().upper()
        state["last_heartbeat_at"] = self._now_utc_iso()
        state["last_seen_status"] = current_status
        if latest_activity_at:
            state["latest_activity_at"] = latest_activity_at
        if latest_activity_path:
            state["latest_activity_path"] = latest_activity_path
        self._write_cluster_state(state)
        self._sync_assignment_registry_from_state(state)
        if current_status and current_status != previous_status:
            self._record_cluster_event(
                "job_status_transition",
                job_id=job_id,
                previous_status=previous_status,
                current_status=current_status,
            )

    def _finalize_cluster_job(
        self,
        job_id: str,
        *,
        terminal_status: str,
        reason: str = "",
    ) -> None:
        state = self._load_cluster_state()
        tracked_job_id = str(state.get("child_slurm_job_id") or "").strip()
        if tracked_job_id and tracked_job_id != job_id:
            return
        if not state:
            state = {"child_slurm_job_id": job_id, "workspace_path": str(self.workspace.path)}
        code_dir_value = str(state.get("code_dir") or "").strip()
        code_dir = Path(code_dir_value) if code_dir_value else None
        state["child_slurm_job_id"] = job_id
        state["last_heartbeat_at"] = self._now_utc_iso()
        state["last_seen_status"] = str(terminal_status or state.get("last_seen_status") or "").strip().upper()
        state["finalized_at"] = self._now_utc_iso()
        state["terminal_status"] = str(terminal_status or "").strip().upper()
        state["termination_reason"] = str(reason or "")
        self._write_cluster_state(state)
        self._sync_assignment_registry_from_state(state)
        self._clear_active_job_tracker(code_dir, job_id=job_id)
        self._unregister_atexit_job(job_id)
        self._record_cluster_event(
            "job_finalized",
            job_id=job_id,
            terminal_status=state.get("terminal_status", ""),
            reason=reason,
        )

    def _latest_activity_snapshot(self, code_dir: Path) -> tuple[str, str]:
        latest_timestamp: float | None = None
        latest_path = ""
        for subdir_name in ("logs", "results", "checkpoints"):
            base = code_dir / subdir_name
            if not base.exists():
                continue
            try:
                candidates = list(base.rglob("*"))
            except OSError:
                continue
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                if candidate.name in _STATE_FILE_NAMES:
                    continue
                try:
                    mtime = candidate.stat().st_mtime
                except OSError:
                    continue
                if latest_timestamp is None or mtime > latest_timestamp:
                    latest_timestamp = mtime
                    latest_path = str(candidate.relative_to(code_dir))
        return self._iso_from_epoch(latest_timestamp), latest_path

    def _cluster_stall_timeout_seconds(self) -> int:
        configured = int(getattr(self.config, "cluster_stall_timeout_seconds", 0) or 0)
        return configured if configured > 0 else _DEFAULT_CLUSTER_STALL_TIMEOUT_SECONDS

    def _job_inactivity_seconds(self, code_dir: Path) -> tuple[int | None, str, str]:
        latest_activity_at, latest_activity_path = self._latest_activity_snapshot(code_dir)
        if not latest_activity_at:
            return None, "", latest_activity_path
        try:
            latest_epoch = datetime.fromisoformat(latest_activity_at).timestamp()
        except ValueError:
            return None, latest_activity_at, latest_activity_path
        inactivity = int(time.time() - latest_epoch)
        return inactivity, latest_activity_at, latest_activity_path

    async def _run_slurm_command(
        self,
        cmd: str,
        *,
        timeout: int = 120,
        retries: int = 3,
    ) -> dict[str, Any]:
        last_result: dict[str, Any] = {"returncode": -1, "stdout": "", "stderr": ""}
        for attempt in range(1, retries + 1):
            result = await self._run_shell(cmd, timeout=timeout)
            last_result = result
            combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".lower()
            if result.get("returncode", 1) == 0:
                return result
            if "reach max user active rpc limit" not in combined:
                return result
            delay = min(10 * attempt, 30)
            self.log(
                f"SLURM control command hit RPC limit on attempt {attempt}/{retries}, retrying in {delay}s"
            )
            await asyncio.sleep(delay)
        return last_result

    async def _cancel_slurm_job(self, job_id: str, *, reason: str) -> None:
        self._record_cluster_event(
            "job_cancel_requested",
            job_id=job_id,
            reason=reason,
        )
        result = await self._run_slurm_command(f"scancel {job_id}", retries=4)
        if result.get("returncode", 1) != 0:
            self.log(
                f"scancel {job_id} returned {result.get('returncode')}: "
                f"{result.get('stderr', '').strip() or result.get('stdout', '').strip()}"
            )

    async def _reclaim_duplicate_assignment_jobs(
        self,
        assignment_context: dict[str, Any] | None,
        *,
        exclude_job_id: str = "",
    ) -> list[str]:
        context = assignment_context or {}
        assignment_id = str(context.get("assignment_id") or "").strip()
        if not assignment_id:
            return []

        registry_path = self._registry_path_for_assignment(assignment_id)
        lock_path = registry_path.with_suffix(".lock")
        reclaimed: list[str] = []
        with self._registry_lock(lock_path):
            payload = self._read_json_path(registry_path, {"assignment_id": assignment_id, "records": []})
            records = payload.get("records") if isinstance(payload, dict) else []
            if not isinstance(records, list):
                records = []
            updated = False
            for record in records:
                if not isinstance(record, dict):
                    continue
                job_id = str(record.get("child_slurm_job_id") or "").strip()
                if not job_id or job_id == exclude_job_id:
                    continue
                if str(record.get("finalized_at") or "").strip():
                    continue

                status = await self._get_job_status(job_id)
                record["last_heartbeat_at"] = self._now_utc_iso()
                record["last_seen_status"] = status
                if status in _ACTIVE_JOB_STATUSES:
                    await self._cancel_slurm_job(
                        job_id,
                        reason=f"superseded_by:{self.workspace.path}",
                    )
                    record["finalized_at"] = self._now_utc_iso()
                    record["terminal_status"] = "CANCELLED"
                    record["termination_reason"] = f"superseded_by:{self.workspace.path}"
                    record["is_active"] = False
                    reclaimed.append(job_id)
                    updated = True
                elif status == "COMPLETED" or status in _TERMINAL_JOB_STATUSES:
                    record["finalized_at"] = str(record.get("finalized_at") or self._now_utc_iso())
                    record["terminal_status"] = status
                    record["is_active"] = False
                    updated = True
            if updated:
                payload = {
                    "assignment_id": assignment_id,
                    "updated_at": self._now_utc_iso(),
                    "records": records,
                }
                self._write_json_path(registry_path, payload)

        if reclaimed:
            self._record_cluster_event(
                "duplicate_assignment_jobs_reclaimed",
                assignment_id=assignment_id,
                reclaimed_job_ids=reclaimed,
            )
        return reclaimed

    async def _find_existing_job(self, code_dir: Path) -> tuple[str, str] | None:
        """Check whether a previously-submitted child job can be resumed."""
        candidates: list[str] = []
        state = self._load_cluster_state()
        tracked_job_id = str(state.get("child_slurm_job_id") or "").strip()
        if tracked_job_id:
            candidates.append(tracked_job_id)

        tracker = self._active_job_tracker_path(code_dir)
        if tracker is not None and tracker.exists():
            job_id = tracker.read_text(encoding="utf-8").strip()
            if job_id and job_id not in candidates:
                candidates.append(job_id)

        assignment_context = self._load_assignment_context()
        for job_id in candidates:
            if not job_id.isdigit():
                continue
            status = await self._get_job_status(job_id)
            inactivity, latest_activity_at, latest_activity_path = self._job_inactivity_seconds(code_dir)
            stall_timeout = self._cluster_stall_timeout_seconds()
            if (
                status in _ACTIVE_JOB_STATUSES
                and inactivity is not None
                and stall_timeout > 0
                and inactivity >= stall_timeout
            ):
                self.log(
                    f"Existing SLURM job {job_id} has no artifact updates for {inactivity}s "
                    f"(latest={latest_activity_path or '<unknown>'}), cancelling stale child job"
                )
                await self._cancel_slurm_job(job_id, reason="resume_stall_timeout")
                self._finalize_cluster_job(
                    job_id,
                    terminal_status="STALLED",
                    reason="resume_stall_timeout",
                )
                continue
            if status in _ACTIVE_JOB_STATUSES or status == "COMPLETED":
                self._register_active_cluster_job(
                    job_id=job_id,
                    code_dir=code_dir,
                    assignment_context=assignment_context,
                    initial_status=status,
                    source="resume_existing",
                )
                return (job_id, status)
            if status in _TERMINAL_JOB_STATUSES or status == "UNKNOWN":
                self._finalize_cluster_job(job_id, terminal_status=status or "UNKNOWN", reason="stale_tracker")
        return None

    async def _submit_job(
        self,
        slurm_script: str,
        *,
        code_dir: Path,
        assignment_context: dict[str, Any] | None = None,
    ) -> str:
        """Submit a SLURM batch job, register lifecycle state, and return the job ID."""
        if not Path(slurm_script).exists():
            raise RuntimeError(f"SLURM script not found: {slurm_script}")

        await self._reclaim_duplicate_assignment_jobs(assignment_context)
        result = await self._run_slurm_command(f"sbatch {shlex.quote(slurm_script)}", retries=4)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        match = re.search(r"Submitted batch job (\d+)", stdout)
        if not match:
            raise RuntimeError(
                f"Failed to submit SLURM job. stdout: {stdout}, stderr: {stderr}"
            )

        job_id = match.group(1)
        if not job_id.isdigit():
            raise RuntimeError(f"Extracted job ID is not numeric: {job_id!r}")

        self._register_active_cluster_job(
            job_id=job_id,
            code_dir=code_dir,
            assignment_context=assignment_context,
            initial_status="PENDING",
            source="sbatch_submit",
        )
        return job_id

    async def _monitor_job(self, job_id: str, code_dir: Path) -> str:
        """Poll SLURM until terminal state or no-progress stall is detected."""
        start_time = time.time()
        last_log_lines = 0
        stall_timeout = self._cluster_stall_timeout_seconds()
        pending_timeout = int(getattr(self.config, "cluster_pending_timeout_seconds", 0) or 0)

        while True:
            status = await self._get_job_status(job_id)
            latest_activity_at, latest_activity_path = self._latest_activity_snapshot(code_dir)
            self._update_tracked_cluster_job(
                job_id,
                last_seen_status=status,
                latest_activity_at=latest_activity_at,
                latest_activity_path=latest_activity_path,
            )

            log_files = sorted(code_dir.glob("logs/slurm_*.out"))
            if log_files:
                try:
                    content = log_files[-1].read_text(errors="replace")
                    lines = content.strip().split("\n")
                    if len(lines) > last_log_lines:
                        new_lines = lines[last_log_lines:]
                        for line in new_lines[-5:]:
                            self.log(f"[TRAIN] {line.strip()}")
                        last_log_lines = len(lines)
                except Exception:  # noqa: BLE001
                    pass

            if status == "COMPLETED" or status in _TERMINAL_JOB_STATUSES:
                return status

            elapsed = int(time.time() - start_time)
            if status == "PENDING":
                self.log(f"Job {job_id} pending... ({elapsed}s elapsed)")
                if pending_timeout > 0 and elapsed >= pending_timeout:
                    self.log(
                        f"Job {job_id} exceeded pending timeout ({pending_timeout}s), cancelling"
                    )
                    await self._cancel_slurm_job(job_id, reason="pending_timeout")
                    return "PENDING_TIMEOUT"
            elif status in _ACTIVE_JOB_STATUSES or status == "RUNNING":
                self.log(f"Job {job_id} running... ({elapsed}s elapsed)")
                if stall_timeout > 0 and latest_activity_at:
                    inactivity, _, _ = self._job_inactivity_seconds(code_dir)
                    if inactivity is None:
                        inactivity = 0
                    if inactivity >= stall_timeout:
                        self.log(
                            f"Job {job_id} has no artifact updates for {inactivity}s, cancelling as stalled"
                        )
                        await self._cancel_slurm_job(job_id, reason="stall_timeout")
                        return "STALLED"
            else:
                self.log(f"Job {job_id} status={status} ({elapsed}s elapsed)")

            await asyncio.sleep(CLUSTER_POLL_INTERVAL)

    async def _get_job_status(self, job_id: str) -> str:
        """Query SLURM for job status."""
        result = await self._run_shell(
            f"squeue -j {job_id} -h -o '%T' 2>/dev/null || "
            f"sacct -j {job_id} -n -o State -X 2>/dev/null"
        )
        stdout = result.get("stdout", "").strip()

        if not stdout:
            result2 = await self._run_shell(f"sacct -j {job_id} -n -o State -X")
            stdout = result2.get("stdout", "").strip()

        status = stdout.split("\n")[0].strip().upper() if stdout else "UNKNOWN"
        return status.rstrip("+").strip()

    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict[str, Any]:
        """Run a shell command asynchronously with proxy environment."""
        env = self._build_proxy_env()
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_process_tree(proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    def _build_proxy_env(self) -> dict[str, str]:
        if _ClusterRunnerMixin._cached_proxy_env is not None:
            env = {**os.environ}
            env.update(_ClusterRunnerMixin._cached_proxy_env)
            env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
            return env
        env = {**os.environ}
        proxy_url = env.get("https_proxy") or env.get("HTTPS_PROXY", "")
        proxy_overlay: dict[str, str] = {}
        if proxy_url:
            proxy_overlay = {
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
            }
            env.update(proxy_overlay)
        _ClusterRunnerMixin._cached_proxy_env = proxy_overlay
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return env

    async def _run_subprocess(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        env = self._build_proxy_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd) if cwd is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except PermissionError:
            return await asyncio.to_thread(
                self._run_subprocess_sync,
                command,
                cwd=cwd,
                timeout=timeout,
                env=env,
            )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_process_tree(proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    @staticmethod
    def _run_subprocess_sync(
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def close(self) -> None:
        state = self._load_cluster_state()
        job_id = str(state.get("child_slurm_job_id") or "").strip()
        if not job_id:
            return
        if str(state.get("finalized_at") or "").strip():
            self._unregister_atexit_job(job_id)
            return

        status = await self._get_job_status(job_id)
        if status in _ACTIVE_JOB_STATUSES:
            self.log(f"Cancelling active child SLURM job {job_id} during execution cleanup")
            await self._cancel_slurm_job(job_id, reason="execution_agent_close")
            terminal_status = "CANCELLED"
        elif status == "COMPLETED" or status in _TERMINAL_JOB_STATUSES:
            terminal_status = status
        else:
            terminal_status = "UNKNOWN"
        self._finalize_cluster_job(
            job_id,
            terminal_status=terminal_status,
            reason="execution_agent_close",
        )
