"""Result collector helpers: artifact collection, contract evaluation, log parsing."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.experiment import ExperimentAgent

logger = logging.getLogger(__name__)

RESULT_CONTRACT_CRASH_INDICATORS = (
    "RuntimeError",
    "Error(s) in loading",
    "Traceback",
    "CUDA out of memory",
    "OOM",
    "Killed",
    "Exception",
    "FileNotFoundError",
    "ModuleNotFoundError",
)


class _ResultCollectorHelpersMixin:
    """Mixin — artifact collection, contract evaluation, log metrics parsing."""

    def _collect_result_artifacts(self, code_dir: Path) -> dict[str, Any]:
        results: dict[str, Any] = {
            "metrics": {},
            "training_log": [],
        }

        config_matrix_path = code_dir / "configs" / "experiment_matrix.json"
        if config_matrix_path.exists():
            try:
                matrix_payload = json.loads(config_matrix_path.read_text(encoding="utf-8"))
                if isinstance(matrix_payload, dict):
                    results["experiment_matrix_config"] = matrix_payload
            except (OSError, json.JSONDecodeError):
                results["experiment_matrix_config"] = {}

        metrics_path = code_dir / "results" / "metrics.json"
        if metrics_path.exists():
            parsed_metrics = ExperimentAgent._parse_metrics_json(code_dir)
            if parsed_metrics:
                results["metrics"] = parsed_metrics
                results["training_log"] = list(parsed_metrics.get("training_log") or [])
                meta = parsed_metrics.get("_nanoresearch_meta")
                if isinstance(meta, dict) and str(meta.get("recovered_from") or "").strip():
                    results["recovered_from"] = str(meta.get("recovered_from") or "").strip()
            else:
                try:
                    results["metrics"] = {"raw": metrics_path.read_text()[:5000]}
                except OSError:
                    results["metrics"] = {}

        log_csv = code_dir / "results" / "training_log.csv"
        if log_csv.exists():
            try:
                with log_csv.open("r", encoding="utf-8", errors="replace") as handle:
                    results["training_log_csv"] = handle.read(10000)
            except OSError:
                results["training_log_csv"] = ""
            parsed_training_log = self._parse_training_log_csv(log_csv)
            if parsed_training_log and not results["training_log"]:
                results["training_log"] = parsed_training_log
            if parsed_training_log and not (
                isinstance(results.get("metrics"), dict)
                and any(
                    key in results["metrics"] for key in ("main_results", "ablation_results", "training_log")
                )
            ):
                materialized = self._materialize_recovered_metrics_artifact(
                    code_dir,
                    {
                        "main_results": [],
                        "ablation_results": [],
                        "training_log": parsed_training_log,
                    },
                    source="training_log_csv",
                    scope="result_artifacts",
                )
                results["metrics"] = materialized.get("metrics") or results.get("metrics", {})
                if results["metrics"]:
                    results["recovered_from"] = "training_log_csv"
                    results["training_log"] = list(results["metrics"].get("training_log") or parsed_training_log)
                if materialized.get("written"):
                    results["metrics_artifact_materialized"] = True
                    results["metrics_artifact_path"] = materialized.get("artifact_path", "")

        structured_json_files = {
            "run_manifest.json": "run_manifest",
            "final_metrics.json": "final_metrics",
            "pareto_front.json": "pareto_front",
        }
        for filename, key in structured_json_files.items():
            path = code_dir / "results" / filename
            if path.exists():
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                    results[key] = parsed if isinstance(parsed, (dict, list)) else {}
                except (OSError, json.JSONDecodeError):
                    results[key] = {}

        opt_csv = code_dir / "results" / "optimization_history.csv"
        if opt_csv.exists():
            try:
                results["optimization_history_csv"] = opt_csv.read_text(encoding="utf-8", errors="replace")[:10000]
            except OSError:
                results["optimization_history_csv"] = ""

        for results_file in (code_dir / "results").glob("*"):
            if results_file.is_file() and results_file.name not in (
                "metrics.json", "training_log.csv", "run_manifest.json",
                "final_metrics.json", "optimization_history.csv", "pareto_front.json",
            ):
                try:
                    content = results_file.read_text(errors="replace")[:5000]
                    results[f"result_file_{results_file.name}"] = content
                except Exception as exc:
                    logger.debug("Failed to read result artifact %s: %s", results_file, exc)

        checkpoints = (
            list((code_dir / "checkpoints").glob("*.pt"))
            if (code_dir / "checkpoints").exists()
            else []
        )
        results["checkpoints"] = [str(p) for p in checkpoints]
        return results

    def _collect_local_results(
        self,
        code_dir: Path,
        run_result: dict[str, Any],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            **self._collect_result_artifacts(code_dir),
            "stdout_log": str(run_result.get("stdout", ""))[-10000:],
            "stderr_log": str(run_result.get("stderr", ""))[-5000:],
        }
        if not results["metrics"] and results["stdout_log"]:
            results["parsed_metrics"] = self._parse_metrics_from_log(results["stdout_log"])
        return results

    @staticmethod
    def _metrics_satisfy_contract(metrics: dict[str, Any] | None) -> bool:
        if not isinstance(metrics, dict):
            return False
        main_results = metrics.get("main_results")
        if not isinstance(main_results, list):
            return False
        for item in main_results:
            if not isinstance(item, dict):
                continue
            metric_entries = item.get("metrics")
            if not isinstance(metric_entries, list):
                continue
            for metric in metric_entries:
                if not isinstance(metric, dict):
                    continue
                if str(metric.get("metric_name", "")).strip() and metric.get("value") is not None:
                    return True
        return False

    @staticmethod
    def _result_file_names(results: dict[str, Any]) -> list[str]:
        return sorted(
            key.removeprefix("result_file_")
            for key, value in results.items()
            if key.startswith("result_file_") and str(value or "").strip()
        )

    @classmethod
    def _detect_contract_failure_signals(
        cls,
        stdout_log: str,
        stderr_log: str,
    ) -> list[str]:
        combined = f"{stdout_log}\n{stderr_log}".lower()
        found: list[str] = []
        for indicator in RESULT_CONTRACT_CRASH_INDICATORS:
            if indicator.lower() in combined and indicator not in found:
                found.append(indicator)
        return found

    @classmethod
    def _evaluate_experiment_contract(
        cls,
        result_payload: dict[str, Any],
        *,
        execution_backend: str,
        execution_status: str,
        quick_eval_status: str,
        final_status: str,
    ) -> dict[str, Any]:
        metrics = result_payload.get("metrics") if isinstance(result_payload.get("metrics"), dict) else {}
        parsed_metrics = result_payload.get("parsed_metrics") if isinstance(result_payload.get("parsed_metrics"), dict) else {}
        training_log = result_payload.get("training_log") if isinstance(result_payload.get("training_log"), list) else []
        training_log_csv = str(result_payload.get("training_log_csv") or "").strip()
        checkpoints = result_payload.get("checkpoints") if isinstance(result_payload.get("checkpoints"), list) else []
        result_files = cls._result_file_names(result_payload)
        recovered_from = str(result_payload.get("recovered_from") or "").strip()
        stdout_log = str(result_payload.get("stdout_log") or "")
        stderr_log = str(result_payload.get("stderr_log") or "")

        matrix_config = result_payload.get("experiment_matrix_config")
        if not isinstance(matrix_config, dict):
            matrix_config = {}
        experiment_matrix = matrix_config.get("experiment_matrix")
        if not isinstance(experiment_matrix, list):
            experiment_matrix = []
        required_artifacts = matrix_config.get("required_artifacts")
        if not isinstance(required_artifacts, list):
            required_artifacts = []
        criteria = matrix_config.get("minimum_success_criteria")
        if not isinstance(criteria, dict):
            criteria = {}

        has_structured_metrics = cls._metrics_satisfy_contract(metrics)
        has_parsed_metrics = bool(parsed_metrics)
        has_training_trace = bool(training_log) or bool(training_log_csv)
        has_checkpoints = bool(checkpoints)
        has_result_files = bool(result_files)
        has_run_manifest = isinstance(result_payload.get("run_manifest"), (dict, list)) and bool(result_payload.get("run_manifest"))
        has_final_metrics = isinstance(result_payload.get("final_metrics"), dict) and bool(result_payload.get("final_metrics"))
        has_optimization_history = bool(str(result_payload.get("optimization_history_csv") or "").strip())
        has_pareto_front = isinstance(result_payload.get("pareto_front"), (dict, list)) and bool(result_payload.get("pareto_front"))
        failure_signals = cls._detect_contract_failure_signals(stdout_log, stderr_log)
        run_completed = final_status == "COMPLETED" or quick_eval_status in {"success", "partial"}
        has_recovered_artifact_support = (
            has_structured_metrics
            and bool(recovered_from)
            and run_completed
            and not failure_signals
            and (has_checkpoints or has_result_files)
            and (has_parsed_metrics or has_training_trace)
        )

        satisfied_signals: list[str] = []
        if has_structured_metrics and not recovered_from:
            satisfied_signals.append("structured_metrics_artifact")
        elif has_structured_metrics and recovered_from:
            satisfied_signals.append("structured_metrics_recovered")
        if has_parsed_metrics:
            satisfied_signals.append("parsed_metrics")
        if has_training_trace:
            satisfied_signals.append("training_log")
        if has_checkpoints:
            satisfied_signals.append("checkpoints")
        if has_result_files:
            satisfied_signals.append("result_files")
        if has_run_manifest:
            satisfied_signals.append("run_manifest")
        if has_final_metrics:
            satisfied_signals.append("final_metrics")
        if has_optimization_history:
            satisfied_signals.append("optimization_history")
        if has_pareto_front:
            satisfied_signals.append("pareto_front")

        main_results = metrics.get("main_results") if isinstance(metrics.get("main_results"), list) else []
        ablation_results = metrics.get("ablation_results") if isinstance(metrics.get("ablation_results"), list) else []
        measured_baseline_count = 0
        proposed_count = 0
        for entry in main_results:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").lower()
            is_proposed = bool(entry.get("is_proposed")) or role == "proposed"
            has_metrics = bool(entry.get("metrics"))
            if is_proposed and has_metrics:
                proposed_count += 1
            elif has_metrics:
                measured_baseline_count += 1
        measured_ablation_count = sum(
            1 for entry in ablation_results
            if isinstance(entry, dict) and bool(entry.get("metrics"))
        )

        required_run_ids = {
            str(run.get("run_id") or "").strip()
            for run in experiment_matrix
            if isinstance(run, dict) and run.get("required") and str(run.get("run_id") or "").strip()
        }
        manifest_payload = result_payload.get("run_manifest")
        manifest_entries = manifest_payload.get("runs") if isinstance(manifest_payload, dict) else manifest_payload
        if not isinstance(manifest_entries, list):
            manifest_entries = []
        completed_run_ids = {
            str(run.get("run_id") or "").strip()
            for run in manifest_entries
            if isinstance(run, dict)
            and str(run.get("run_id") or "").strip()
            and str(run.get("status") or "").lower() in {"success", "completed", "ok", "partial"}
        }
        missing_required_runs = sorted(required_run_ids - completed_run_ids)
        top_level_matrix_keys = {
            "project_name",
            "description",
            "datasets",
            "metrics",
            "experiment_matrix",
            "minimum_success_criteria",
            "required_artifacts",
        }
        matrix_schema_mismatch = (
            bool(required_run_ids)
            and bool(completed_run_ids)
            and not (required_run_ids & completed_run_ids)
            and bool(completed_run_ids & top_level_matrix_keys)
        )
        if matrix_schema_mismatch:
            failure_signals.append(
                "matrix_schema_mismatch: runner appears to execute top-level experiment_matrix.json keys "
                "instead of the nested experiment_matrix run list"
            )
        if (
            not run_completed
            and has_structured_metrics
            and required_run_ids
            and not missing_required_runs
        ):
            # Generated runners sometimes fail their own over-strict post-check
            # after writing complete measured artifacts. NanoResearch should use
            # its own artifact contract as the source of truth.
            run_completed = True

        failed_runs = [
            {
                "run_id": str(run.get("run_id") or ""),
                "status": str(run.get("status") or ""),
                "failure_reason": str(run.get("failure_reason") or run.get("error") or ""),
            }
            for run in manifest_entries
            if isinstance(run, dict) and str(run.get("status") or "").lower() not in {"success", "completed", "ok", "partial"}
        ]

        artifact_presence = {
            "configs/experiment_matrix.json": bool(experiment_matrix),
            "results/metrics.json": has_structured_metrics,
            "results/run_manifest.json": has_run_manifest,
            "results/final_metrics.json": has_final_metrics,
            "results/optimization_history.csv": has_optimization_history,
            "results/pareto_front.json": has_pareto_front,
        }
        missing_artifacts = [
            str(name) for name in required_artifacts
            if str(name) in artifact_presence and not artifact_presence[str(name)]
        ]

        success_path = ""
        status = "failed"
        contract_counts_ok = (
            proposed_count >= (1 if criteria.get("require_proposed", True) else 0)
            and measured_baseline_count >= int(criteria.get("min_measured_baselines", 0) or 0)
            and measured_ablation_count >= int(criteria.get("min_ablation_runs", 0) or 0)
            and (not criteria.get("require_optimization_history", False) or has_optimization_history)
            and (not criteria.get("require_complexity", False) or bool(metrics.get("complexity_metrics") or result_payload.get("final_metrics", {}).get("complexity_metrics") if isinstance(result_payload.get("final_metrics"), dict) else False))
        )

        if has_structured_metrics and not recovered_from and run_completed and not missing_artifacts and not missing_required_runs and contract_counts_ok:
            status = "success"
            success_path = "full_experiment_matrix_contract"
        elif has_structured_metrics and not recovered_from and run_completed:
            status = "partial"
            success_path = "structured_metrics_artifact_degraded_contract"
        elif has_recovered_artifact_support:
            status = "success"
            success_path = "structured_metrics_recovered"
        elif has_structured_metrics and recovered_from and run_completed:
            status = "partial"
            success_path = "structured_metrics_recovered"
        elif has_parsed_metrics and (has_training_trace or has_checkpoints or has_result_files) and run_completed:
            status = "partial"
            success_path = "parsed_metrics_with_artifacts"
        elif has_training_trace and has_checkpoints and final_status == "COMPLETED":
            status = "partial"
            success_path = "training_log_with_checkpoints"
        elif has_result_files and (has_training_trace or has_checkpoints) and final_status == "COMPLETED":
            status = "partial"
            success_path = "aux_results_with_artifacts"
        elif execution_backend == "cluster" and has_parsed_metrics and final_status == "COMPLETED":
            status = "partial"
            success_path = "parsed_metrics_only"

        missing_signals: list[str] = []
        if not (has_structured_metrics or has_parsed_metrics):
            missing_signals.append("metrics_signal")
        if not (has_training_trace or has_checkpoints or has_result_files or has_structured_metrics):
            missing_signals.append("artifact_signal")
        if missing_required_runs:
            missing_signals.append("required_run_coverage")
        if matrix_schema_mismatch:
            missing_signals.append("matrix_schema_mismatch")
        if missing_artifacts:
            missing_signals.append("required_artifacts")
        if measured_baseline_count < int(criteria.get("min_measured_baselines", 0) or 0):
            missing_signals.append("measured_baselines")
        if measured_ablation_count < int(criteria.get("min_ablation_runs", 0) or 0):
            missing_signals.append("ablation_runs")
        if criteria.get("require_optimization_history", False) and not has_optimization_history:
            missing_signals.append("optimization_history")
        if failure_signals:
            missing_signals.append("crash_free_logs")

        return {
            "version": "v1",
            "status": status,
            "execution_backend": execution_backend,
            "execution_status": execution_status,
            "quick_eval_status": quick_eval_status,
            "final_status": final_status,
            "recovered_from": recovered_from,
            "success_path": success_path,
            "satisfied_signals": satisfied_signals,
            "missing_signals": missing_signals,
            "failure_signals": failure_signals,
            "missing_required_runs": missing_required_runs,
            "missing_artifacts": missing_artifacts,
            "failed_runs": failed_runs,
            "matrix_schema_mismatch": matrix_schema_mismatch,
            "run_coverage": {
                "required_run_count": len(required_run_ids),
                "completed_run_count": len(completed_run_ids),
                "measured_baseline_count": measured_baseline_count,
                "measured_ablation_count": measured_ablation_count,
                "proposed_count": proposed_count,
            },
            "artifact_inventory": {
                "structured_metrics": has_structured_metrics,
                "parsed_metrics": has_parsed_metrics,
                "training_log_entries": len(training_log),
                "training_log_csv": bool(training_log_csv),
                "checkpoint_count": len(checkpoints),
                "result_files": result_files,
                "run_manifest": has_run_manifest,
                "final_metrics": has_final_metrics,
                "optimization_history_csv": has_optimization_history,
                "pareto_front": has_pareto_front,
            },
        }

    def _parse_metrics_from_log(self, log_text: str) -> dict:
        """Try to extract metrics from training log output."""
        metrics: dict[str, Any] = {}
        lines = log_text.split("\n")

        # Common patterns in training logs
        patterns = [
            # "Epoch 10: loss=0.123, accuracy=0.95"
            r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)",
            # "Test accuracy: 0.95"
            r"[Tt]est\s+(accuracy|acc)[=:\s]+([0-9.e-]+)",
            # "Best metric: 0.95"
            r"[Bb]est\s+(\w+)[=:\s]+([0-9.e-]+)",
            # "AUC: 0.95" / "F1: 0.85"
            r"(AUC|F1|RMSE|MAE|accuracy|precision|recall)[=:\s]+([0-9.e-]+)",
        ]

        epochs = []
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        if groups[0].isdigit():
                            continue
                        try:
                            metrics[groups[0]] = float(groups[1])
                        except ValueError:
                            metrics[groups[0]] = groups[1]

            # Track epoch losses
            epoch_match = re.search(
                r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)", line
            )
            if epoch_match:
                epochs.append({
                    "epoch": int(epoch_match.group(1)),
                    "loss": float(epoch_match.group(2)),
                })

        if epochs:
            metrics["epoch_losses"] = epochs
            metrics["final_loss"] = epochs[-1]["loss"]

        return metrics
