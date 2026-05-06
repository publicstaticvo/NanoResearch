"""Execution agent — submits SLURM jobs, monitors progress, debugs failures, collects results."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.debug import DebugAgent, MAX_DEBUG_ROUNDS
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.repair_journal import REPAIR_SNAPSHOT_JOURNAL_PATH
from nanoresearch.schemas.manifest import PipelineStage

from .cluster_runner import _ClusterRunnerMixin, _kill_process_tree
from .local_runner import _LocalRunnerMixin
from .repair import _RepairMixin, REMEDIATION_LEDGER_PATH
from .result_collector import _ResultCollectorMixin

__all__ = ["ExecutionAgent"]


class ExecutionAgent(
    _LocalRunnerMixin,
    _ClusterRunnerMixin,
    _RepairMixin,
    _ResultCollectorMixin,
    BaseResearchAgent,
):
    """Submits SLURM training jobs, monitors them, debugs failures, and collects results."""

    stage = PipelineStage.EXECUTION

    @property
    def stage_config(self):
        """Reuse experiment-stage model routing for execution-time reasoning."""
        return self.config.for_stage("experiment")

    @staticmethod
    def _resolve_cluster_slurm_script(code_dir: Path, slurm_script: str) -> str:
        candidates: list[Path] = []
        if slurm_script:
            candidates.append(Path(slurm_script))
        for rel_path in (
            "run_train.slurm",
            "train.slurm",
            "job.slurm",
            "job.sh",
            "run.sh",
        ):
            candidates.append(code_dir / rel_path)

        seen: set[str] = set()
        for candidate in candidates:
            normalized = str(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            if candidate.exists() and candidate.is_file():
                return normalized
        return slurm_script

    @staticmethod
    def _validate_cluster_launch_script(slurm_script: str, code_dir: Path) -> dict[str, Any]:
        launch_contract: dict[str, Any] = {
            "status": "passed",
            "target_kind": "slurm_script",
            "target": slurm_script,
            "resolved_target": slurm_script,
            "created_dirs": [],
            "warnings": [],
            "failures": [],
        }

        script_path = Path(slurm_script) if slurm_script else Path()
        if not slurm_script or not script_path.exists() or not script_path.is_file():
            launch_contract["status"] = "failed"
            launch_contract["failures"] = [f"SLURM script not found: {slurm_script or '<empty>'}"]
            return launch_contract

        for artifact_dir in ("logs", "results", "checkpoints"):
            target_dir = code_dir / artifact_dir
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                launch_contract["created_dirs"].append(str(target_dir))

        content = script_path.read_text(encoding="utf-8", errors="replace")
        stripped = content.lstrip("\ufeff\r\n\t ")
        if not stripped:
            launch_contract["status"] = "failed"
            launch_contract["failures"] = [f"SLURM script is empty: {script_path}"]
            return launch_contract

        if not stripped.startswith("#!"):
            launch_contract["status"] = "failed"
            launch_contract["failures"].append(
                f"SLURM script is missing a shebang on line 1: {script_path}"
            )

        first_nonempty = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
        if first_nonempty.startswith(("import ", "from ", "def ", "class ")):
            launch_contract["status"] = "failed"
            launch_contract["failures"].append(
                f"SLURM script appears to contain Python source instead of shell: {script_path}"
            )

        if "#SBATCH --output=" not in content:
            launch_contract["warnings"].append("SLURM script does not declare #SBATCH --output")
        if "#SBATCH --error=" not in content:
            launch_contract["warnings"].append("SLURM script does not declare #SBATCH --error")

        return launch_contract

    @staticmethod
    def _failed_result_contract(
        *,
        reason: str,
        execution_status: str,
        final_status: str,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "success_path": "",
            "execution_status": execution_status,
            "quick_eval_status": "skipped",
            "final_status": final_status,
            "satisfied_signals": [],
            "missing_signals": [reason],
            "failure_signals": [reason],
        }

    def _build_cluster_failure_result(
        self,
        *,
        code_dir: Path,
        remediation_ledger: list[dict[str, Any]],
        final_status: str,
        execution_status: str,
        reason: str,
        preflight: dict[str, Any] | None = None,
        launch_contract: dict[str, Any] | None = None,
        launch_contract_repair: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result_contract = self._failed_result_contract(
            reason=reason,
            execution_status=execution_status,
            final_status=final_status,
        )
        return {
            "job_id": "",
            "execution_backend": "cluster",
            "runtime_env": {
                "kind": "cluster",
                "profile": self.config.execution_profile.value,
                "partition": self.config.slurm_partition,
            },
            "final_status": final_status,
            "code_dir": str(code_dir),
            "debug_rounds": 0,
            "execution_status": execution_status,
            "quick_eval_status": "skipped",
            "experiment_status": "failed",
            "result_contract": result_contract,
            "experiment_results": {},
            "preflight": preflight or {},
            "launch_contract": launch_contract or {},
            "launch_contract_repair": launch_contract_repair or {},
            "remediation_ledger": list(remediation_ledger),
            "repair_snapshot_journal_path": self._repair_snapshot_journal_path(),
        }

    async def run(self, **inputs: Any) -> dict[str, Any]:
        coding_output: dict = inputs.get("coding_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})
        setup_output: dict = inputs.get("setup_output", {})
        topic: str = inputs.get("topic", "")

        code_dir = Path(coding_output.get("code_dir", ""))
        slurm_script = self._resolve_cluster_slurm_script(
            code_dir,
            str(coding_output.get("slurm_script", "") or ""),
        )

        if not code_dir.exists():
            raise RuntimeError(f"Code directory not found: {code_dir}")

        self.log(f"Starting execution in: {code_dir}")
        remediation_ledger: list[dict[str, Any]] = []

        # Create logs directory
        (code_dir / "logs").mkdir(exist_ok=True)
        (code_dir / "results").mkdir(exist_ok=True)

        cluster_available = bool(slurm_script) and shutil.which("sbatch") is not None

        # Auto-detect: if profile is local_quick but no local GPU and SLURM is
        # available, automatically upgrade to cluster execution.
        use_cluster = self.config.prefers_cluster_execution()
        if not use_cluster and cluster_available:
            try:
                import torch as _torch
                has_gpu = _torch.cuda.is_available() and _torch.cuda.device_count() > 0
            except Exception:
                has_gpu = False
            if not has_gpu:
                use_cluster = True
                self.log(
                    "No local GPU detected but sbatch is available — "
                    "auto-upgrading to cluster (SLURM) execution"
                )

        if not use_cluster or not cluster_available:
            if self.config.prefers_cluster_execution() and not cluster_available:
                self.log("Cluster execution requested but sbatch is unavailable, falling back to local mode")
            elif not slurm_script:
                self.log("No SLURM script produced by CODING, falling back to local mode")
            else:
                self.log(f"Execution profile '{self.config.execution_profile.value}' prefers local execution")
            final_result = await self._run_local_mode(
                code_dir,
                coding_output,
                experiment_blueprint,
                setup_output,
                topic,
                remediation_ledger=remediation_ledger,
            )
            self.workspace.write_json("plans/execution_output.json", final_result)
            return final_result

        auto_repair_enabled = self._execution_auto_repair_enabled()

        # Pre-flight: optionally fix common SLURM/runtime issues before first submission
        debug_agent = DebugAgent(self.workspace, self.config)
        if auto_repair_enabled:
            preflight_fixed = debug_agent._fix_common_slurm_issues(code_dir)
            if preflight_fixed:
                self.log("Pre-flight: fixed common SLURM script issues")
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="slurm_preflight_fix",
                    status="applied",
                    scope="cluster_preflight",
                    details={"code_dir": str(code_dir)},
                )

            python_preflight_fixes = debug_agent._fix_common_python_runtime_issues(code_dir)
            if python_preflight_fixes:
                self.log(
                    "Pre-flight: fixed recurring Python runtime issues in "
                    f"{python_preflight_fixes}"
                )
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="python_preflight_fix",
                    status="applied",
                    scope="cluster_preflight",
                    files=python_preflight_fixes,
                    details={"code_dir": str(code_dir)},
                )

        preflight = PreflightChecker(code_dir).run_all()
        self.workspace.write_json(
            "logs/execution_round_0_preflight.json",
            preflight.model_dump(),
        )
        self._append_remediation_entry(
            remediation_ledger,
            kind="preflight",
            status=preflight.overall_status,
            scope="cluster_preflight",
            round_number=0,
            details={
                "blocking_failures": list(preflight.blocking_failures),
                "warning_messages": list(preflight.warning_messages),
                "suggested_fixes": list(preflight.suggested_fixes),
            },
        )

        if preflight.overall_status == "failed":
            self.log("Cluster preflight failed; skipping SLURM submission")
            final_result = self._build_cluster_failure_result(
                code_dir=code_dir,
                remediation_ledger=remediation_ledger,
                final_status="PRECHECK_FAILED",
                execution_status="skipped",
                reason="cluster_preflight_failed",
                preflight=preflight.model_dump(),
            )
            final_result["remediation_ledger_path"] = self._persist_remediation_ledger(remediation_ledger)
            self.workspace.write_json("plans/execution_output.json", final_result)
            await debug_agent.close()
            return final_result

        # Pre-flight: local syntax/import check before wasting SLURM queue time
        local_ok, local_err = await self._local_preflight(code_dir)
        if not local_ok:
            self.log(f"Pre-flight import check failed, fixing before submission")
            # Run a mini debug loop locally (no SLURM submission)
            for pre_round in range(MAX_DEBUG_ROUNDS):
                debug_result = await debug_agent.run(
                    code_dir=str(code_dir),
                    stdout_log="",
                    stderr_log=local_err,
                    job_status="IMPORT_ERROR",
                    debug_round=pre_round + 1,
                    previous_fixes=[],
                )
                if not debug_result.get("needs_resubmit", False):
                    break
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="cluster_preflight_debug_fix",
                    status="applied",
                    scope="cluster_preflight",
                    cycle=pre_round + 1,
                    files=list(debug_result.get("fixed_files", []) or []),
                    details={
                        "diagnosis": debug_result.get("diagnosis", ""),
                        "patches": list(debug_result.get("patches", []) or []),
                    },
                )
                local_ok, local_err = await self._local_preflight(code_dir)
                if local_ok:
                    self.log(f"Pre-flight fixed after {pre_round + 1} round(s)")
                    break
            if not local_ok:
                self.log("Cluster local import preflight failed after debug loop; skipping SLURM submission")
                final_result = self._build_cluster_failure_result(
                    code_dir=code_dir,
                    remediation_ledger=remediation_ledger,
                    final_status="PRECHECK_FAILED",
                    execution_status="skipped",
                    reason="cluster_local_preflight_failed",
                    preflight=preflight.model_dump(),
                )
                final_result["stderr_log"] = local_err
                final_result["remediation_ledger_path"] = self._persist_remediation_ledger(remediation_ledger)
                self.workspace.write_json("plans/execution_output.json", final_result)
                await debug_agent.close()
                return final_result

        refreshed_preflight = PreflightChecker(code_dir).run_all()
        if refreshed_preflight.model_dump() != preflight.model_dump():
            preflight = refreshed_preflight
            self.workspace.write_json(
                "logs/execution_round_0_preflight.json",
                preflight.model_dump(),
            )

        launch_contract = self._validate_cluster_launch_script(slurm_script, code_dir)
        launch_contract_repair: dict[str, Any] = {
            "status": "skipped",
            "actions": [],
            "files_modified": [],
            "command": [slurm_script] if slurm_script else [],
            "initial_contract": launch_contract,
            "final_contract": launch_contract,
        }
        if auto_repair_enabled and launch_contract.get("status") == "failed":
            regenerated = debug_agent._build_fallback_slurm_wrapper(code_dir, Path(slurm_script))
            if regenerated:
                Path(slurm_script).write_text(regenerated, encoding="utf-8")
                launch_contract_repair = {
                    "status": "applied",
                    "actions": ["regenerated_slurm_wrapper"],
                    "files_modified": [slurm_script],
                    "command": [slurm_script],
                    "initial_contract": launch_contract,
                }
                launch_contract = self._validate_cluster_launch_script(slurm_script, code_dir)
                launch_contract_repair["final_contract"] = launch_contract

        self._record_launch_contract_repair_ledger(
            launch_contract_repair,
            remediation_ledger,
            round_number=0,
            scope="cluster_launch",
        )
        self._record_launch_contract_ledger(
            launch_contract,
            remediation_ledger,
            round_number=0,
            scope="cluster_launch",
        )
        self.workspace.write_json(
            "logs/execution_round_0_launch_contract.json",
            launch_contract,
        )
        self.workspace.write_json(
            "logs/execution_round_0_launch_contract_repair.json",
            launch_contract_repair,
        )
        if launch_contract.get("status") == "failed":
            self.log("Cluster launch contract failed; skipping SLURM submission")
            final_result = self._build_cluster_failure_result(
                code_dir=code_dir,
                remediation_ledger=remediation_ledger,
                final_status="PRECHECK_FAILED",
                execution_status="skipped",
                reason="cluster_launch_contract_failed",
                preflight=preflight.model_dump(),
                launch_contract=launch_contract,
                launch_contract_repair=launch_contract_repair,
            )
            final_result["remediation_ledger_path"] = self._persist_remediation_ledger(remediation_ledger)
            self.workspace.write_json("plans/execution_output.json", final_result)
            await debug_agent.close()
            return final_result

        # Debug loop: submit → monitor → if failed, debug & retry
        previous_fixes: list[dict] = []
        final_result = None
        assignment_context = self._load_assignment_context()

        for debug_round in range(MAX_DEBUG_ROUNDS + 1):
            # On first round, check for existing job from a previous run (resume)
            existing = await self._find_existing_job(code_dir) if debug_round == 0 else None
            if existing:
                job_id, existing_status = existing
                self.log(f"Found existing SLURM job {job_id} (status: {existing_status})")
                if existing_status == "COMPLETED":
                    final_status = "COMPLETED"
                else:  # RUNNING or PENDING
                    final_status = await self._monitor_job(job_id, code_dir)
                    self.log(f"Existing job {job_id} finished: {final_status}")
            else:
                # Submit new SLURM job
                job_id = await self._submit_job(
                    slurm_script,
                    code_dir=code_dir,
                    assignment_context=assignment_context,
                )
                self.log(f"Submitted SLURM job: {job_id}")
                # Monitor job until completion
                final_status = await self._monitor_job(job_id, code_dir)
                self.log(f"Job {job_id} finished with status: {final_status}")

            # Collect results
            results = await self._collect_results(code_dir, job_id, final_status)
            self.log(f"Collected results: {list(results.keys())}")
            self._finalize_cluster_job(
                job_id,
                terminal_status=final_status,
                reason="monitor_complete",
            )
            recovered_source = str(results.get("recovered_from") or "").strip()
            if recovered_source and (
                recovered_source == "slurm_logs" or results.get("metrics_artifact_materialized")
            ):
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="metrics_recovery",
                    status="applied",
                    scope="cluster_collect",
                    cycle=debug_round + 1,
                    details={
                        "source": recovered_source,
                        "job_id": job_id,
                        **(
                            {
                                "artifact_path": results.get("metrics_artifact_path", ""),
                                "artifact_materialized": True,
                            }
                            if results.get("metrics_artifact_materialized")
                            else {}
                        ),
                    },
                )
                if results.get("metrics_artifact_materialized"):
                    snapshot_entry = self.consume_last_mutation_snapshot_entry()
                    details = {
                        "source": recovered_source,
                        "job_id": job_id,
                        "artifact_path": str(results.get("metrics_artifact_path", "")),
                    }
                    if snapshot_entry:
                        details.update({
                            "snapshot_entry_id": snapshot_entry.get("entry_id"),
                            "snapshot_count": snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(snapshot_entry.get("snapshots", []) or []),
                        })
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="metrics_artifact_recovery",
                        status="applied",
                        scope="cluster_collect",
                        cycle=debug_round + 1,
                        files=[str(results.get("metrics_artifact_path", ""))],
                        details=details,
                    )

            if auto_repair_enabled and final_status != "COMPLETED":
                cluster_resume_fix = self._attempt_cluster_resume_repair(
                    code_dir,
                    final_status,
                    results,
                    setup_output,
                    scope="cluster_resume",
                )
                cluster_resume_snapshot_entry = self.consume_last_mutation_snapshot_entry()
                if cluster_resume_fix:
                    self.log(
                        "Applied deterministic cluster resume repair: "
                        f"{cluster_resume_fix}; resubmitting job"
                    )
                    details = None
                    if cluster_resume_snapshot_entry:
                        details = {
                            "snapshot_entry_id": cluster_resume_snapshot_entry.get("entry_id"),
                            "snapshot_count": cluster_resume_snapshot_entry.get("snapshot_count", 0),
                            "snapshot_journal_path": REPAIR_SNAPSHOT_JOURNAL_PATH,
                            "snapshots": list(cluster_resume_snapshot_entry.get("snapshots", []) or []),
                        }
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="resume_repair",
                        status="applied",
                        scope="cluster_resume",
                        cycle=debug_round + 1,
                        files=list(cluster_resume_fix),
                        details={
                            **(details or {}),
                            "job_id": job_id,
                            "job_status": final_status,
                        },
                    )
                    continue

            metrics = results.get("metrics") or {}
            execution_status = "success" if final_status == "COMPLETED" else "failed"
            result_contract = self._evaluate_experiment_contract(
                results,
                execution_backend="cluster",
                execution_status=execution_status,
                quick_eval_status="skipped",
                final_status=final_status,
            )
            experiment_status = str(result_contract.get("status", "failed"))
            self._append_remediation_entry(
                remediation_ledger,
                kind="result_contract_validation",
                status=experiment_status,
                scope="cluster_result",
                cycle=debug_round + 1,
                details={
                    "success_path": result_contract.get("success_path", ""),
                    "missing_signals": list(result_contract.get("missing_signals", []) or []),
                    "failure_signals": list(result_contract.get("failure_signals", []) or []),
                },
            )

            final_result = {
                "job_id": job_id,
                "execution_backend": "cluster",
                "runtime_env": {
                    "kind": "cluster",
                    "profile": self.config.execution_profile.value,
                    "partition": self.config.slurm_partition,
                },
                "remediation_ledger": list(remediation_ledger),
                "remediation_ledger_path": REMEDIATION_LEDGER_PATH,
                "repair_snapshot_journal_path": self._repair_snapshot_journal_path(),
                "final_status": final_status,
                "slurm_final_status": final_status,
                "code_dir": str(code_dir),
                "cluster_job_state_path": str(self.workspace.path / "logs" / "cluster_job_state.json"),
                "cluster_job_events_path": str(self.workspace.path / "logs" / "cluster_job_events.jsonl"),
                "debug_rounds": debug_round,
                "execution_status": execution_status,
                "quick_eval_status": "skipped",
                "experiment_status": experiment_status,
                "preflight": preflight.model_dump(),
                "launch_contract": launch_contract,
                "launch_contract_repair": launch_contract_repair,
                "result_contract": result_contract,
                "experiment_results": metrics,
                **results,
            }

            # If job succeeded or we've exhausted debug rounds, stop
            if final_status == "COMPLETED":
                if experiment_status in {"success", "partial"}:
                    self.log(
                        f"Job completed with result contract status {experiment_status} "
                        f"after {debug_round} debug round(s)"
                    )
                    break
                self.log(
                    "Job exited with code 0 but failed the explicit result contract. "
                    f"Missing={result_contract.get('missing_signals', [])}, "
                    f"failure_signals={result_contract.get('failure_signals', [])}"
                )
                final_status = "FAILED"
                final_result["final_status"] = "FAILED"
                final_result["experiment_status"] = "failed"
                final_result["result_contract"]["status"] = "failed"
                # Fall through to debug loop

            if debug_round >= MAX_DEBUG_ROUNDS:
                self.log(f"Max debug rounds ({MAX_DEBUG_ROUNDS}) reached, giving up")
                break

            # Job failed — enter debug loop
            self.log(f"Job failed, entering debug round {debug_round + 1}/{MAX_DEBUG_ROUNDS}")

            try:
                debug_result = await debug_agent.run(
                    code_dir=str(code_dir),
                    stdout_log=results.get("stdout_log", ""),
                    stderr_log=results.get("stderr_log", ""),
                    job_status=final_status,
                    debug_round=debug_round + 1,
                    previous_fixes=previous_fixes,
                )

                if not debug_result.get("needs_resubmit", False):
                    self.log("Debug agent determined no fix is possible, stopping")
                    break

                previous_fixes.append({
                    "diagnosis": debug_result.get("diagnosis", ""),
                    "patches": debug_result.get("patches", []),
                    "fixed_files": debug_result.get("fixed_files", []),
                })
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="cluster_debug_fix",
                    status="applied",
                    scope="cluster_debug",
                    cycle=debug_round + 1,
                    files=list(debug_result.get("fixed_files", []) or []),
                    details={
                        "diagnosis": debug_result.get("diagnosis", ""),
                        "patches": list(debug_result.get("patches", []) or []),
                        "job_status": final_status,
                    },
                )

                self.log(f"Debug round {debug_round + 1}: fixed {debug_result.get('fixed_files', [])}, resubmitting...")

            except Exception as e:
                self.log(f"Debug agent failed: {e}")
                break

        await debug_agent.close()

        if final_result is None:
            final_result = {
                "job_id": "",
                "execution_backend": "cluster",
                "runtime_env": {
                    "kind": "cluster",
                    "profile": self.config.execution_profile.value,
                    "partition": self.config.slurm_partition,
                },
                "final_status": "FAILED",
                "slurm_final_status": "FAILED",
                "code_dir": str(code_dir),
                "cluster_job_state_path": str(self.workspace.path / "logs" / "cluster_job_state.json"),
                "cluster_job_events_path": str(self.workspace.path / "logs" / "cluster_job_events.jsonl"),
                "debug_rounds": 0,
                "execution_status": "failed",
                "quick_eval_status": "skipped",
                "experiment_status": "failed",
                "preflight": preflight.model_dump() if "preflight" in locals() else {},
                "launch_contract": launch_contract if "launch_contract" in locals() else {},
                "launch_contract_repair": launch_contract_repair if "launch_contract_repair" in locals() else {},
                "experiment_results": {},
                "repair_snapshot_journal_path": self._repair_snapshot_journal_path(),
                "result_contract": self._failed_result_contract(
                    reason="cluster_execution_failed",
                    execution_status="failed",
                    final_status="FAILED",
                ),
            }

        final_result["remediation_ledger"] = list(remediation_ledger)
        final_result["remediation_ledger_path"] = self._persist_remediation_ledger(remediation_ledger)
        final_result["repair_snapshot_journal_path"] = self._repair_snapshot_journal_path()
        self.workspace.write_json("plans/execution_output.json", final_result)
        return final_result

    async def close(self) -> None:
        try:
            await _ClusterRunnerMixin.close(self)
        finally:
            await BaseResearchAgent.close(self)
