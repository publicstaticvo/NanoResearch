from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from nanoresearch.agents.execution._result_collector_helpers import _ResultCollectorHelpersMixin
from nanoresearch.agents.execution.cluster_runner import _ClusterRunnerMixin
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace


class DummyClusterRunner(_ClusterRunnerMixin):
    def __init__(self, workspace: Workspace, config: ResearchConfig) -> None:
        self.workspace = workspace
        self.config = config
        self.messages: list[str] = []
        self.shell_calls: list[str] = []
        self.status_sequences: dict[str, list[str] | str] = {}
        self.shell_outputs: dict[str, dict[str, object]] = {}

    def log(self, message: str) -> None:
        self.messages.append(message)

    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict[str, object]:
        self.shell_calls.append(cmd)
        for prefix, response in self.shell_outputs.items():
            if cmd.startswith(prefix):
                return dict(response)
        return {"returncode": 0, "stdout": "", "stderr": ""}

    async def _get_job_status(self, job_id: str) -> str:
        value = self.status_sequences.get(job_id, "UNKNOWN")
        if isinstance(value, list):
            if len(value) > 1:
                return value.pop(0)
            return value[0]
        return value


def _make_workspace(tmp_path: Path, name: str) -> Workspace:
    return Workspace.create(
        topic=f"topic-{name}",
        config_snapshot={},
        root=tmp_path / name,
        session_id="attempt-01",
    )


def _write_assignment_context(workspace: Workspace, assignment_id: str) -> dict[str, str]:
    payload = {
        "assignment_id": assignment_id,
        "chain_id": "chain-a",
        "workspace_path": str(workspace.path),
    }
    workspace.write_json("logs/deep_assignment_context.json", payload)
    return payload


def _read_registry_record(registry_root: Path, assignment_id: str) -> list[dict[str, object]]:
    path = _ClusterRunnerMixin._registry_path_for_assignment(assignment_id)
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def test_submit_job_reclaims_duplicate_assignment_jobs(tmp_path, monkeypatch) -> None:
    registry_root = tmp_path / "registry"
    monkeypatch.setenv("NANORESEARCH_EXECUTION_REGISTRY_ROOT", str(registry_root))

    ws1 = _make_workspace(tmp_path, "ws1")
    ws2 = _make_workspace(tmp_path, "ws2")
    code_dir1 = ws1.path / "code" / "exp"
    code_dir2 = ws2.path / "code" / "exp"
    code_dir1.mkdir(parents=True, exist_ok=True)
    code_dir2.mkdir(parents=True, exist_ok=True)
    script1 = code_dir1 / "run_train.slurm"
    script2 = code_dir2 / "run_train.slurm"
    script1.write_text("#!/bin/bash\n", encoding="utf-8")
    script2.write_text("#!/bin/bash\n", encoding="utf-8")

    assignment_context = _write_assignment_context(ws1, "persona-a::variant-a::q1")
    runner1 = DummyClusterRunner(ws1, ResearchConfig())
    runner1.shell_outputs["sbatch "] = {
        "returncode": 0,
        "stdout": "Submitted batch job 111\n",
        "stderr": "",
    }
    job1 = asyncio.run(
        runner1._submit_job(
            str(script1),
            code_dir=code_dir1,
            assignment_context=assignment_context,
        )
    )
    assert job1 == "111"

    assignment_context2 = _write_assignment_context(ws2, "persona-a::variant-a::q1")
    runner2 = DummyClusterRunner(ws2, ResearchConfig())
    runner2.status_sequences["111"] = "RUNNING"
    runner2.shell_outputs["scancel "] = {"returncode": 0, "stdout": "", "stderr": ""}
    runner2.shell_outputs["sbatch "] = {
        "returncode": 0,
        "stdout": "Submitted batch job 222\n",
        "stderr": "",
    }

    job2 = asyncio.run(
        runner2._submit_job(
            str(script2),
            code_dir=code_dir2,
            assignment_context=assignment_context2,
        )
    )

    assert job2 == "222"
    assert runner2.shell_calls[0] == "scancel 111"
    assert runner2.shell_calls[1].startswith("sbatch ")

    records = _read_registry_record(registry_root, "persona-a::variant-a::q1")
    record_by_job = {str(item["child_slurm_job_id"]): item for item in records}
    assert record_by_job["111"]["finalized_at"]
    assert record_by_job["111"]["terminal_status"] == "CANCELLED"
    assert record_by_job["222"]["finalized_at"] == ""

    runner2._finalize_cluster_job("222", terminal_status="COMPLETED", reason="test_cleanup")


def test_monitor_job_marks_stalled_when_artifacts_stop_updating(tmp_path) -> None:
    workspace = _make_workspace(tmp_path, "stall")
    runner = DummyClusterRunner(
        workspace,
        ResearchConfig(cluster_stall_timeout_seconds=1, cluster_pending_timeout_seconds=0),
    )
    assignment_context = _write_assignment_context(workspace, "persona-a::variant-a::q2")
    code_dir = workspace.path / "code" / "exp"
    logs_dir = code_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "slurm_333.out").write_text("epoch 1\n", encoding="utf-8")
    old_ts = time.time() - 120
    os.utime(logs_dir / "slurm_333.out", (old_ts, old_ts))
    runner._register_active_cluster_job(
        job_id="333",
        code_dir=code_dir,
        assignment_context=assignment_context,
        initial_status="RUNNING",
        source="test",
    )
    runner.status_sequences["333"] = "RUNNING"
    runner.shell_outputs["scancel "] = {"returncode": 0, "stdout": "", "stderr": ""}

    final_status = asyncio.run(runner._monitor_job("333", code_dir))

    assert final_status == "STALLED"
    assert "scancel 333" in runner.shell_calls
    runner._finalize_cluster_job("333", terminal_status="STALLED", reason="test_cleanup")


def test_find_existing_job_reclaims_stale_running_job(tmp_path) -> None:
    workspace = _make_workspace(tmp_path, "resume_stale")
    runner = DummyClusterRunner(
        workspace,
        ResearchConfig(cluster_stall_timeout_seconds=1, cluster_pending_timeout_seconds=0),
    )
    assignment_context = _write_assignment_context(workspace, "persona-a::variant-a::q-stale")
    code_dir = workspace.path / "code" / "exp"
    logs_dir = code_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "slurm_555.out"
    log_path.write_text("stale run\n", encoding="utf-8")
    old_ts = time.time() - 120
    os.utime(log_path, (old_ts, old_ts))
    runner._register_active_cluster_job(
        job_id="555",
        code_dir=code_dir,
        assignment_context=assignment_context,
        initial_status="RUNNING",
        source="test",
    )
    runner.status_sequences["555"] = "RUNNING"
    runner.shell_outputs["scancel "] = {"returncode": 0, "stdout": "", "stderr": ""}

    existing = asyncio.run(runner._find_existing_job(code_dir))

    assert existing is None
    assert "scancel 555" in runner.shell_calls
    state = json.loads((workspace.path / "logs" / "cluster_job_state.json").read_text(encoding="utf-8"))
    assert state["terminal_status"] == "STALLED"
    assert state["termination_reason"] == "resume_stall_timeout"


def test_close_cancels_active_child_job(tmp_path) -> None:
    workspace = _make_workspace(tmp_path, "close")
    runner = DummyClusterRunner(workspace, ResearchConfig())
    assignment_context = _write_assignment_context(workspace, "persona-a::variant-a::q3")
    code_dir = workspace.path / "code" / "exp"
    code_dir.mkdir(parents=True, exist_ok=True)
    runner._register_active_cluster_job(
        job_id="444",
        code_dir=code_dir,
        assignment_context=assignment_context,
        initial_status="RUNNING",
        source="test",
    )
    runner.status_sequences["444"] = "RUNNING"
    runner.shell_outputs["scancel "] = {"returncode": 0, "stdout": "", "stderr": ""}

    asyncio.run(runner.close())

    state = json.loads((workspace.path / "logs" / "cluster_job_state.json").read_text(encoding="utf-8"))
    assert state["terminal_status"] == "CANCELLED"
    assert state["finalized_at"]
    assert "scancel 444" in runner.shell_calls


def test_contract_allows_artifact_backed_terminal_recovery() -> None:
    contract = _ResultCollectorHelpersMixin._evaluate_experiment_contract(
        {
            "metrics": {
                "main_results": [
                    {
                        "method_name": "RecoveredRun",
                        "dataset": "Dummy",
                        "is_proposed": True,
                        "metrics": [{"metric_name": "Accuracy", "value": 0.81}],
                    }
                ],
                "ablation_results": [],
                "training_log": [],
            },
            "training_log": [{"epoch": 1, "train_loss": 0.4, "metrics": {}}],
            "checkpoints": ["/tmp/checkpoint.pt"],
            "stdout_log": "training finished",
            "stderr_log": "",
        },
        execution_backend="cluster",
        execution_status="failed",
        quick_eval_status="skipped",
        final_status="STALLED",
    )

    assert contract["status"] == "partial"
    assert contract["success_path"] == "artifact_backed_terminal_recovery"
