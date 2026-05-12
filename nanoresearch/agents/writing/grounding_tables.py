"""Grounding tables mixin: table/figure block building, table verification."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ._types import GroundingPacket
from . import _escape_latex_text

logger = logging.getLogger(__name__)




def _short_metric_name(name: str) -> str:
    """Compact metric labels for paper tables."""
    raw = str(name or "").strip()
    key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    aliases = {
        "balanced_accuracy": "BA",
        "test_balanced_accuracy": "Test BA",
        "cv_balanced_accuracy": "CV BA",
        "accuracy": "Acc",
        "test_accuracy": "Test Acc",
        "f1": "F1",
        "f1_score": "F1",
        "roc_auc": "AUC",
        "auc": "AUC",
        "selected_features": "#Feat",
        "selected_feature_count": "#Feat",
        "num_features": "#Feat",
        "feature_count": "#Feat",
        "nonzero_coefficients": "#Coef",
        "nonzero_coef_count": "#Coef",
        "fit_time_seconds": "Fit(s)",
        "fit_time": "Fit(s)",
        "predict_time_seconds": "Pred(s)",
        "predict_time": "Pred(s)",
        "runtime_seconds": "Time(s)",
        "pareto_front_size": "Pareto",
    }
    if key in aliases:
        return aliases[key]
    if "balanced_accuracy" in key and "pareto" in key:
        return "Pareto BA"
    if "balanced_accuracy" in key and "cross" in key:
        return "CV BA"
    if "balanced_accuracy" in key and "heldout" in key:
        return "Test BA"
    if key in {"heldout_accuracy", "test_accuracy"}:
        return "Test Acc"
    if "roc_auc" in key or key.endswith("auc"):
        return "AUC"
    if "f1" in key:
        return "F1"
    if "selected_feature" in key or key in {"num_features", "feature_count"}:
        return "#Feat"
    if "fit_time" in key:
        return "Fit(s)"
    if "predict_time" in key:
        return "Pred(s)"
    if "tree_count" in key:
        return "Trees"
    text = raw.replace("balanced_accuracy", "BA")
    text = text.replace("accuracy", "Acc")
    text = text.replace("selected_feature_count", "#Feat")
    text = text.replace("selected_features", "#Feat")
    text = text.replace("fit_time_seconds", "Fit(s)")
    text = text.replace("predict_time_seconds", "Pred(s)")
    text = text.replace("_", " ").strip()
    return text[:18] if len(text) > 18 else text


def _metric_priority(name: str) -> tuple[int, str]:
    key = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    order = [
        ("balanced_accuracy", 0),
        ("test_balanced_accuracy", 0),
        ("accuracy", 1),
        ("f1", 2),
        ("auc", 3),
        ("selected_feature", 4),
        ("num_features", 4),
        ("feature_count", 4),
        ("nonzero", 5),
        ("fit_time", 6),
        ("predict_time", 7),
        ("runtime", 8),
        ("pareto", 9),
    ]
    for needle, rank in order:
        if needle in key:
            return rank, key
    return 50, key

def _format_paper_number(value: Any) -> str:
    """Format numeric values for paper-facing prose/tables."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return f"{number:.4f}"


def _format_numbers_in_text(text: str) -> str:
    """Keep generated captions/prose from exposing raw float precision."""
    def repl(match: re.Match) -> str:
        raw = match.group(0)
        try:
            return f"{float(raw):.4f}"
        except ValueError:
            return raw
    return re.sub(r"(?<![A-Za-z0-9_])-?\d+\.\d{5,}(?![A-Za-z0-9_])", repl, str(text or ""))


class _GroundingTablesMixin:
    """Mixin — table/figure block building and table verification methods."""

    @staticmethod
    def _build_ablation_table_latex(
        ablation_results: list[dict],
        blueprint: dict,
    ) -> str:
        """Build a deterministic LaTeX ablation table from structured data."""
        if not ablation_results:
            return ""
        all_metrics: list[str] = []
        seen: set[str] = set()
        for entry in ablation_results:
            for m in entry.get("metrics", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("metric_name", "")
                if name and name not in seen:
                    all_metrics.append(name)
                    seen.add(name)
        if not all_metrics:
            return ""
        all_metrics = sorted(all_metrics, key=_metric_priority)[:6]
        n_metrics = len(all_metrics)
        col_spec = "@{}l" + "c" * n_metrics + "@{}"
        header_cells = " & ".join(_escape_latex_text(_short_metric_name(m)) for m in all_metrics)
        use_resizebox = n_metrics >= 4
        lines = [
            "\\begin{table}[htbp]", "\\centering", "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\caption{Ablation study. Each row removes or replaces one component.}",
            "\\label{tab:ablation}",
        ]
        if use_resizebox:
            lines.append("\\resizebox{\\linewidth}{!}{%")
        lines.extend([
            f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
            f"Variant & {header_cells} \\\\", "\\midrule",
        ])
        for entry in ablation_results:
            variant = _escape_latex_text(entry.get("variant_name", "?"))
            cells = []
            for metric_name in all_metrics:
                val_str = "--"
                for m in entry.get("metrics", []):
                    if isinstance(m, dict) and m.get("metric_name") == metric_name:
                        val = m.get("value")
                        if val is not None:
                            val_str = _format_paper_number(val)
                        break
                cells.append(val_str)
            lines.append(f"{variant} & {' & '.join(cells)} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        if use_resizebox:
            lines.append("}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    @staticmethod
    def _build_scaffold_main_table(blueprint: dict) -> str:
        """Build a table scaffold from blueprint when no real results exist."""
        baselines = blueprint.get("baselines", [])
        if not isinstance(baselines, list):
            baselines = []
        metrics_spec = blueprint.get("metrics", [])
        if not isinstance(metrics_spec, list):
            metrics_spec = []
        metric_names: list[str] = []
        for m in metrics_spec:
            if isinstance(m, dict):
                name = m.get("name", "") or m.get("metric_name", "")
            elif isinstance(m, str):
                name = m
            else:
                continue
            if name and name not in metric_names:
                metric_names.append(name)
        if not metric_names:
            return ""
        # Cap at 6 metric columns to avoid table overflow
        MAX_TABLE_COLS = 6
        if len(metric_names) > MAX_TABLE_COLS:
            metric_names = metric_names[:MAX_TABLE_COLS]
        baseline_names: list[str] = []
        for b in baselines:
            if isinstance(b, dict):
                name = b.get("name", "") or b.get("method", "")
            elif isinstance(b, str):
                name = b
            else:
                continue
            if name:
                baseline_names.append(name)
        if not baseline_names:
            baseline_names = ["Baseline 1", "Baseline 2"]
        proposed = (
            (blueprint.get("proposed_method") or {}).get("name", "")
            or blueprint.get("method_name", "") or "Ours"
        )
        datasets = blueprint.get("datasets", [])
        dataset_names = []
        for d in datasets:
            if isinstance(d, dict):
                name = d.get("name", "")
            elif isinstance(d, str):
                name = d
            else:
                continue
            if name:
                dataset_names.append(name)
        dataset_str = ", ".join(dataset_names[:3]) if dataset_names else "the benchmark"
        n = len(metric_names)
        col_spec = "@{}l" + "c" * n + "@{}"
        header = " & ".join(_escape_latex_text(m) for m in metric_names)
        lines = [
            "\\begin{table}[htbp]", "\\centering", "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            f"\\caption{{Main experimental results on {_escape_latex_text(dataset_str)}. "
            "Best results are in \\textbf{bold}. "
            "'--' indicates that the method was not evaluated in our experiments.}",
            "\\label{tab:main_results}",
        ]
        use_resizebox = n >= 5
        if use_resizebox:
            lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.extend([
            f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
            f"Method & {header} \\\\", "\\midrule",
        ])
        for bname in baseline_names:
            cells = " & ".join(["--"] * n)
            lines.append(f"{_escape_latex_text(bname)} & {cells} \\\\")
        lines.append("\\midrule")
        cells = " & ".join(["--"] * n)
        lines.append(f"{_escape_latex_text(proposed)} (Ours) & {cells} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        if use_resizebox:
            lines.append("}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    @staticmethod
    def _build_scaffold_ablation_table(blueprint: dict) -> str:
        """Build an ablation table scaffold from blueprint (no real data)."""
        metrics_spec = blueprint.get("metrics", [])
        if not isinstance(metrics_spec, list):
            metrics_spec = []
        metric_names: list[str] = []
        for m in metrics_spec:
            if isinstance(m, dict):
                name = m.get("name", "") or m.get("metric_name", "")
            elif isinstance(m, str):
                name = m
            else:
                continue
            if name and name not in metric_names:
                metric_names.append(name)
        if not metric_names:
            return ""
        # Cap at 6 metric columns to avoid table overflow
        MAX_TABLE_COLS = 6
        if len(metric_names) > MAX_TABLE_COLS:
            metric_names = metric_names[:MAX_TABLE_COLS]
        contributions = blueprint.get("contributions", [])
        if not isinstance(contributions, list):
            contributions = []
        variants: list[str] = []
        for c in contributions:
            if isinstance(c, str) and len(c) < 60:
                variants.append(f"w/o {c}")
            elif isinstance(c, dict):
                name = c.get("name", "") or c.get("component", "")
                if name:
                    variants.append(f"w/o {name}")
        if not variants:
            variants = ["w/o Component A", "w/o Component B"]
        proposed = (
            (blueprint.get("proposed_method") or {}).get("name", "")
            or blueprint.get("method_name", "") or "Full Model"
        )
        datasets = blueprint.get("datasets", [])
        dataset_names = []
        for d in datasets:
            if isinstance(d, dict):
                name = d.get("name", "")
            elif isinstance(d, str):
                name = d
            else:
                continue
            if name:
                dataset_names.append(name)
        dataset_str = ", ".join(dataset_names[:3]) if dataset_names else "the benchmark"
        n = len(metric_names)
        col_spec = "@{}l" + "c" * n + "@{}"
        header = " & ".join(metric_names)
        lines = [
            "\\begin{table}[htbp]", "\\centering", "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            f"\\caption{{Ablation study on {_escape_latex_text(dataset_str)}. Each row removes one component. "
            "No verified measured results are available for this table.}",
            "\\label{tab:ablation}",
        ]
        use_resizebox = n >= 5
        if use_resizebox:
            lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.extend([
            f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
            f"Variant & {header} \\\\", "\\midrule",
        ])
        cells = " & ".join(["--"] * n)
        for v in variants[:5]:
            lines.append(f"{_escape_latex_text(v)} & {cells} \\\\")
        lines.append("\\midrule")
        lines.append(f"{_escape_latex_text(proposed)} (Full) & {cells} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        if use_resizebox:
            lines.append("}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    @staticmethod
    def _build_real_results_context(
        experiment_results: dict, experiment_status: str
    ) -> str:
        """Build context block from real experiment results for writing prompts."""
        main_results = experiment_results.get("main_results", [])
        if not isinstance(main_results, list):
            main_results = []
        normalized_status = (experiment_status or "").lower()
        has_real = bool(
            experiment_results
            and normalized_status not in ("pending", "failed", "error", "unknown")
            and main_results
        )

        if has_real:
            lines = [
                "=== REAL EXPERIMENT RESULTS (MUST USE THESE EXACT NUMBERS) ===",
                "The following numbers come from actual experiments. Use them EXACTLY",
                "in tables, text, and analysis. Do NOT round, adjust, or fabricate.",
                "",
            ]
            for entry in main_results:
                if not isinstance(entry, dict):
                    continue
                method = entry.get("method_name", "?")
                dataset = entry.get("dataset", "?")
                is_proposed = entry.get("is_proposed", False)
                tag = " [PROPOSED]" if is_proposed else ""
                for metric in entry.get("metrics", []):
                    if not isinstance(metric, dict):
                        continue
                    val = metric.get("value", "?")
                    std = metric.get("std")
                    std_str = f" $\\pm$ {_format_paper_number(std)}" if std is not None else ""
                    lines.append(
                        f"  {method} on {dataset}: "
                        f"{metric.get('metric_name', '?')} = {_format_paper_number(val)}{std_str}{tag}"
                    )

            ablation = experiment_results.get("ablation_results", [])
            if not isinstance(ablation, list):
                ablation = []
            if ablation:
                lines.append("")
                lines.append("--- Ablation Results (real) ---")
                for entry in ablation:
                    if not isinstance(entry, dict):
                        continue
                    variant = entry.get("variant_name", "?")
                    for metric in entry.get("metrics", []):
                        if not isinstance(metric, dict):
                            continue
                        val = metric.get("value", "?")
                        lines.append(
                            f"  {variant}: {metric.get('metric_name', '?')} = {_format_paper_number(val)}"
                        )

            lines.append("=== END REAL EXPERIMENT RESULTS ===")
            return "\n".join(lines)
        else:
            return (
                "=== EXPERIMENT RESULTS: NOT AVAILABLE ===\n"
                "The proposed method's experiment did not produce results due to execution issues.\n\n"
                "ABSOLUTE RULES FOR WRITING WITHOUT RESULTS:\n"
                "- Do NOT fabricate results for the PROPOSED METHOD. Use '--' in its table cells.\n"
                "- Do NOT write ANY specific numbers or quantitative claims about the proposed method.\n"
                "- For BASELINE methods, you MAY fill in numbers from their original papers (cite source).\n"
                "- Do NOT generate \\begin{table} environments yourself. Tables are auto-injected.\n"
                "  Reference Table~\\ref{tab:main_results}. Reference Table~\\ref{tab:ablation} only if real ablation data exists.\n"
                "- Write a full Experiments section: datasets, metrics, baselines, setup, implementation.\n"
                "- Include: 'Due to technical issues during execution, quantitative results for our "
                "method are not available in this version.'\n"
                "=== END EXPERIMENT RESULTS ==="
            )

    @staticmethod
    def _build_experiment_analysis_context(
        experiment_analysis: dict, experiment_summary: str, experiment_status: str,
    ) -> str:
        """Build a compact narrative summary from execution analysis artifacts."""
        if not experiment_analysis and not experiment_summary:
            return ""
        lines = ["=== EXPERIMENT ANALYSIS SUMMARY ===", f"Status: {experiment_status}"]
        summary = str(experiment_analysis.get("summary", "")).strip()
        if summary:
            lines.append(f"Summary: {summary}")
        converged = experiment_analysis.get("converged")
        if converged is not None:
            lines.append(f"Converged: {converged}")
        for section, key in [("Final metrics snapshot:", "final_metrics"),
                              ("Key findings:", "key_findings"),
                              ("Limitations:", "limitations")]:
            data = experiment_analysis.get(key, {} if key == "final_metrics" else [])
            if isinstance(data, dict) and data:
                lines.append(section)
                for k, v in data.items():
                    lines.append(f"- {k}: {v}")
            elif isinstance(data, list) and data:
                lines.append(section)
                for item in data[:6]:
                    lines.append(f"- {item}")
        training_dynamics = experiment_analysis.get("training_dynamics")
        if training_dynamics:
            lines.append(f"Training dynamics: {training_dynamics}")
        cleaned_summary = experiment_summary.strip()
        if cleaned_summary:
            lines.append("Markdown experiment summary:")
            lines.append(cleaned_summary[:4000])
        lines.append("=== END EXPERIMENT ANALYSIS SUMMARY ===")
        return "\n".join(lines)

    @staticmethod
    def _build_baseline_comparison_context(grounding: "GroundingPacket | None") -> str:
        """Build context block from comparison_with_baselines analysis data."""
        if not grounding or not grounding.comparison_with_baselines:
            return ""
        lines = ["=== BASELINE COMPARISON (from experiment analysis) ===",
                  "Use these numbers for comparison tables and discussion.", ""]
        for method_name, metrics in grounding.comparison_with_baselines.items():
            if not isinstance(metrics, dict):
                continue
            tag = " [PROPOSED]" if method_name.lower() in ("our_method", "proposed", "ours") else ""
            metric_strs = [f"{k}={v}" for k, v in metrics.items() if v is not None]
            if metric_strs:
                lines.append(f"  {method_name}{tag}: {', '.join(metric_strs)}")
        lines.append("=== END BASELINE COMPARISON ===")
        return "\n".join(lines)

    @staticmethod
    def _build_grounding_status_context(grounding: "GroundingPacket | None") -> str:
        """Build a brief context block informing the LLM about evidence completeness."""
        if not grounding:
            return ""
        completeness_desc = {
            "full": "FULL -- complete experiment results are available. Use exact numbers.",
            "partial": (
                "PARTIAL -- experiment ran but did not fully converge. "
                "ONLY use numbers from REAL EXPERIMENT RESULTS provided below. "
                "For methods NOT run, use '--'. Do NOT use blueprint projected values. "
                "You MAY cite numbers from original published papers of baselines."
            ),
            "quick_eval": (
                "QUICK-EVAL ONLY -- results from a shortened evaluation run. "
                "Use these numbers but note they may not reflect full training. "
                "For methods NOT evaluated, use '--' in table cells."
            ),
            "none": (
                "NONE -- no experiment results available. "
                "ABSOLUTE BAN: Do NOT write ANY specific numbers or quantitative comparisons "
                "for the proposed method. Use ONLY qualitative language. "
                "For baselines, you MAY cite numbers from their original papers."
            ),
        }
        desc = completeness_desc.get(grounding.result_completeness, "UNKNOWN")
        lines = [
            f"=== RESULT COMPLETENESS: {grounding.result_completeness.upper()} ===",
            desc,
        ]
        if grounding.evidence_gaps:
            lines.append("Evidence gaps:")
            for gap in grounding.evidence_gaps:
                lines.append(f"  - {gap}")
        lines.append("=== END RESULT COMPLETENESS ===")
        return "\n".join(lines)

    # ---- figure/table blocks ------------------------------------------------

    @staticmethod
    def _find_table_span(content: str, label: str) -> tuple[int, int] | None:
        """Find span of ``\\begin{table}...\\end{table}`` enclosing *label*."""
        label_match = re.search(re.escape(label), content)
        if not label_match:
            return None
        before = content[:label_match.start()]
        tbl_start = before.rfind("\\begin{table}")
        if tbl_start < 0:
            return None
        after_match = re.search(r"\\end\{table\}", content[label_match.end():])
        if not after_match:
            return None
        tbl_end = label_match.end() + after_match.end()
        return (tbl_start, tbl_end)

    def _verify_and_inject_tables(self, content: str, grounding: GroundingPacket, heading: str) -> str:
        """Verify Experiments section has correct tables; inject if missing or wrong."""
        if heading.strip().lower() == "experiments":
            allowed_labels = {r"\label{tab:main_results}", r"\label{tab:ablation}"}

            def _keep_only_grounded_tables(match: re.Match) -> str:
                table_block = match.group(0)
                if any(label in table_block for label in allowed_labels):
                    return table_block
                self.log(f"  {heading}: removing ungrounded LLM-generated table")
                return ""

            content = re.sub(
                r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
                _keep_only_grounded_tables,
                content,
                flags=re.DOTALL,
            )

        for label_key, table_latex, kw_pattern in [
            (r"\label{tab:main_results}", grounding.main_table_latex,
             r'(?:main results|overall performance|comparison)'),
            (r"\label{tab:ablation}", grounding.ablation_table_latex,
             r'(?:ablation|component analysis)'),
        ]:
            if not table_latex:
                continue
            span = self._find_table_span(content, label_key)
            if span:
                if label_key == r"\label{tab:main_results}":
                    llm_table = content[span[0]:span[1]]
                    if self._table_metrics_match(llm_table, grounding):
                        self.log(f"  {heading}: LLM main table metrics match grounding")
                        continue
                    self.log(f"  {heading}: LLM table metrics MISMATCH, replacing with pre-built")
                else:
                    self.log(f"  {heading}: replacing LLM ablation table with pre-built")
                content = content[:span[0]] + table_latex + content[span[1]:]
            else:
                self.log(f"  {heading}: LLM omitted table {label_key}, injecting pre-built")
                insert_match = re.search(kw_pattern, content, re.IGNORECASE)
                if insert_match:
                    para_end = content.find('\n\n', insert_match.end())
                    if para_end == -1:
                        para_end = len(content)
                    content = content[:para_end] + "\n\n" + table_latex + "\n" + content[para_end:]
                else:
                    content += "\n\n" + table_latex
        return content

    @staticmethod
    def _table_metrics_match(
        llm_table: str, grounding: GroundingPacket,
    ) -> bool:
        """Check if an LLM-generated table's metric columns match the grounding packet."""
        expected_metrics: set[str] = set()
        for entry in grounding.main_results:
            for m in entry.get("metrics", []):
                if isinstance(m, dict):
                    name = m.get("metric_name", "")
                    if name:
                        expected_metrics.add(name.lower().strip())

        if not expected_metrics:
            return True

        llm_table_lower = llm_table.lower()
        found = sum(1 for m in expected_metrics if m in llm_table_lower)
        threshold = max(1, len(expected_metrics) // 2)
        return found >= threshold

    def _build_figure_blocks(self, blueprint: dict, figure_output: dict | None = None) -> dict[str, str]:
        """Pre-build LaTeX figure/table blocks to embed inline."""
        blocks: dict[str, str] = {}
        figures = (figure_output or {}).get("figures", {})
        figures_dir = self.workspace.path / "figures" if hasattr(self, "workspace") else None
        _full_kws = ("overview", "framework", "pipeline", "architecture", "model", "workflow", "diagram")
        if figures:
            for fig_key, fig_data in figures.items():
                if "error" in fig_data and "png_path" not in fig_data:
                    logger.warning("Skipping failed figure %s: %s", fig_key, fig_data.get("error", "?"))
                    continue
                caption = _escape_latex_text(_format_numbers_in_text(fig_data.get("caption", f"Figure: {fig_key}")))
                parts = fig_key.split("_", 1)
                label_suffix = parts[1] if len(parts) > 1 else fig_key
                include_name = self._resolve_figure_include(fig_key, fig_data, figures_dir)
                if include_name is None:
                    logger.warning("Figure %s: no valid file found on disk, skipping", fig_key)
                    continue
                is_full = any(kw in label_suffix.lower() for kw in _full_kws)
                fw = r"\textwidth" if is_full else r"0.75\textwidth"
                fh = r"0.32\textheight" if is_full else r"0.28\textheight"
                blocks[label_suffix] = (
                    f"\\begin{{figure}}[htbp]\n\\centering\n"
                    f"\\includegraphics[width={fw}, height={fh}, keepaspectratio]{{{include_name}}}\n"
                    f"\\caption{{{caption}}}\n\\label{{fig:{label_suffix}}}\n\\end{{figure}}"
                )
        else:
            if figures_dir and figures_dir.exists():
                for img in sorted(figures_dir.iterdir()):
                    if img.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg"):
                        stem = img.stem
                        readable = stem.replace("_", " ").replace("-", " ").title()
                        blocks[stem] = (
                            f"\\begin{{figure}}[htbp]\n\\centering\n"
                            f"\\includegraphics[width=0.75\\textwidth]{{{img.name}}}\n"
                            f"\\caption{{{_escape_latex_text(readable)}.}}\n"
                            f"\\label{{fig:{stem}}}\n\\end{{figure}}"
                        )
            if not blocks:
                logger.warning("No figures available -- paper will have no figure blocks")
        return blocks

    @staticmethod
    def _resolve_figure_include(
        fig_key: str, fig_data: dict, figures_dir: Path | None,
    ) -> str | None:
        """Resolve the actual filename to use in \\includegraphics."""
        candidates = [
            (fig_data.get("pdf_path"), f"{fig_key}.pdf"),
            (fig_data.get("png_path"), f"{fig_key}.png"),
            (None, f"{fig_key}.jpg"),
        ]
        for meta_path, default_name in candidates:
            if meta_path:
                p = Path(meta_path)
                if p.exists():
                    return p.name
            if figures_dir and (figures_dir / default_name).exists():
                return default_name
        return None


    @staticmethod
    def _metric_value(entry: dict, *names: str) -> Any:
        """Return a metric value from a normalized result entry."""
        wanted = {re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_") for n in names}
        metrics = entry.get("metrics", []) if isinstance(entry, dict) else []
        if isinstance(metrics, dict):
            metrics = [{"metric_name": k, "value": v} for k, v in metrics.items()]
        for metric in metrics or []:
            if not isinstance(metric, dict):
                continue
            key = re.sub(r"[^a-z0-9]+", "_", str(metric.get("metric_name") or "").lower()).strip("_")
            if key in wanted:
                return metric.get("value")
        for name in wanted:
            if name in entry:
                return entry.get(name)
        return None

    @staticmethod
    def _numeric_metric(entry: dict, *names: str) -> float | None:
        value = _GroundingTablesMixin._metric_value(entry, *names)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _result_name(entry: dict, default: str = "method") -> str:
        for key in ("method_name", "variant_name", "method", "model_name", "name", "run_id"):
            value = entry.get(key) if isinstance(entry, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip().replace("_", " ")
        return default

    @staticmethod
    def _figure_label_from_block(block: str) -> str:
        match = re.search(r"\\label\{fig:([^}]+)\}", block or "")
        return match.group(1) if match else ""

    @staticmethod
    def _figure_caption_from_block(block: str) -> str:
        match = re.search(r"\\caption\{(.*?)\}", block or "", flags=re.DOTALL)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""

    @staticmethod
    def _compose_experiments_section(
        grounding: GroundingPacket,
        figure_blocks: dict[str, str] | None,
        blueprint: dict | None = None,
        *,
        include_heading: bool = False,
    ) -> tuple[str, set[str]]:
        """Build an artifact-grounded Experiments section with text/table/figure interleaving.

        The composer is intentionally deterministic: it reports only measured
        artifacts already present in ``grounding`` and places explanatory prose
        around each table or figure so LaTeX floats do not collapse into a bare
        sequence of visuals.
        """
        blocks = dict(figure_blocks or {})
        used: set[str] = set()
        lines: list[str] = []
        if include_heading:
            lines.append(r"\section{Experiments}\label{sec:experiments}")
            lines.append("")

        dataset = "the evaluated dataset"
        if isinstance(blueprint, dict):
            datasets = blueprint.get("datasets") or []
            if isinstance(datasets, list) and datasets:
                first = datasets[0]
                if isinstance(first, dict):
                    dataset = str(first.get("name") or dataset).replace("_", " ")
                elif isinstance(first, str):
                    dataset = first.replace("_", " ")

        proposed = next(
            (e for e in grounding.main_results if isinstance(e, dict) and (e.get("is_proposed") or str(e.get("role", "")).lower() == "proposed")),
            grounding.main_results[0] if grounding.main_results else {},
        )
        proposed_name = _escape_latex_text(_GroundingTablesMixin._result_name(proposed, "the proposed method"))
        proposed_ba = _GroundingTablesMixin._numeric_metric(proposed, "heldout_balanced_accuracy", "test_balanced_accuracy", "balanced_accuracy")
        proposed_features = _GroundingTablesMixin._numeric_metric(proposed, "selected_feature_count", "selected_features", "nonzero_coefficient_count")
        baselines = [
            e for e in grounding.main_results
            if isinstance(e, dict) and not (e.get("is_proposed") or str(e.get("role", "")).lower() == "proposed")
        ]
        best_baseline = None
        best_baseline_ba = None
        for entry in baselines:
            score = _GroundingTablesMixin._numeric_metric(entry, "heldout_balanced_accuracy", "test_balanced_accuracy", "balanced_accuracy")
            if score is not None and (best_baseline_ba is None or score > best_baseline_ba):
                best_baseline = entry
                best_baseline_ba = score

        def result_name(entry: dict | None, fallback: str) -> str:
            return _escape_latex_text(_GroundingTablesMixin._result_name(entry or {}, fallback))

        def find_result(*needles: str) -> dict | None:
            lowered = [n.lower() for n in needles]
            for entry in grounding.main_results:
                if not isinstance(entry, dict):
                    continue
                name = _GroundingTablesMixin._result_name(entry, "").lower()
                if all(n in name for n in lowered):
                    return entry
            return None

        def metric(entry: dict | None, *names: str) -> float | None:
            return _GroundingTablesMixin._numeric_metric(entry or {}, *names)

        proposed_acc = metric(proposed, "heldout_accuracy", "test_accuracy", "accuracy")
        proposed_auc = metric(proposed, "heldout_roc_auc", "test_roc_auc", "roc_auc", "auc")
        proposed_fit = metric(proposed, "fit_time_seconds", "fit_time", "refit_time_seconds")
        proposed_pred = metric(proposed, "predict_time_seconds", "predict_time", "prediction_time_seconds")
        proposed_coef = metric(proposed, "nonzero_coefficient_count", "nonzero_coefficients")
        baseline_features = metric(best_baseline, "selected_feature_count", "selected_features", "feature_count", "num_features")
        logistic_baseline = find_result("logistic")
        forest_baseline = find_result("forest")

        def fmt_metric(value: float | None, suffix: str = "") -> str:
            return (_format_paper_number(value) + suffix) if value is not None else "the measured value"

        def main_result_discussion() -> str:
            if proposed_ba is None:
                return "The main comparison is interpreted only as an executed-run summary because the artifacts do not contain a primary held-out balanced-accuracy value for the proposed method."
            pieces = [
                f"The proposed row is best understood as a compactness-preserving operating point rather than a pure accuracy maximizer: it obtains held-out balanced accuracy {_format_paper_number(proposed_ba)}"
            ]
            if proposed_acc is not None:
                pieces.append(f"and held-out accuracy {_format_paper_number(proposed_acc)}")
            if proposed_auc is not None:
                pieces.append(f"with ROC--AUC {_format_paper_number(proposed_auc)}")
            sentence = " ".join(pieces) + "."
            if proposed_features is not None:
                sentence += f" The selected feature count is {_format_paper_number(int(proposed_features) if proposed_features.is_integer() else proposed_features)}, so the result exposes the accuracy cost of deploying a smaller diagnostic panel instead of only reporting predictive scores."
            if best_baseline is not None and best_baseline_ba is not None:
                delta = proposed_ba - best_baseline_ba
                sentence += f" Compared with {result_name(best_baseline, 'the strongest measured baseline')}, the balanced-accuracy difference is {_format_paper_number(delta)} under the same split and metric contract."
                if baseline_features is not None and proposed_features is not None:
                    feature_delta = proposed_features - baseline_features
                    sentence += f" The feature-count difference is {_format_paper_number(feature_delta)}, making the comparison explicitly about the accuracy--measurement trade-off."
            if logistic_baseline is not None and forest_baseline is not None:
                log_ba = metric(logistic_baseline, "heldout_balanced_accuracy", "test_balanced_accuracy", "balanced_accuracy")
                rf_ba = metric(forest_baseline, "heldout_balanced_accuracy", "test_balanced_accuracy", "balanced_accuracy")
                log_auc = metric(logistic_baseline, "heldout_roc_auc", "roc_auc", "auc")
                rf_auc = metric(forest_baseline, "heldout_roc_auc", "roc_auc", "auc")
                details = []
                if log_ba is not None:
                    details.append(f"full-feature logistic regression reaches balanced accuracy {_format_paper_number(log_ba)}")
                if rf_ba is not None:
                    details.append(f"random forest reaches {_format_paper_number(rf_ba)}")
                if log_auc is not None and rf_auc is not None:
                    details.append(f"their ROC--AUC values are {_format_paper_number(log_auc)} and {_format_paper_number(rf_auc)}")
                if details:
                    sentence += " The two baselines also clarify the comparison boundary: " + "; ".join(details) + "."
            sentence += " Since every numeric row comes from the local run artifacts, the discussion avoids importing literature-only scores that may use different datasets, preprocessing, or validation boundaries."
            return sentence

        def ablation_discussion() -> str:
            ablation_scores = []
            for entry in grounding.ablation_results:
                if isinstance(entry, dict):
                    score = metric(entry, "heldout_balanced_accuracy", "balanced_accuracy", "test_balanced_accuracy")
                    feat = metric(entry, "selected_feature_count", "selected_features", "feature_count", "num_features")
                    if score is not None:
                        ablation_scores.append((score, feat, _GroundingTablesMixin._result_name(entry, "variant")))
            if not ablation_scores:
                return "The ablation table is reported as an executed-variant inventory; unsupported component-level causal claims are intentionally avoided."
            best_score, best_feat, best_name = max(ablation_scores, key=lambda item: item[0])
            text = f"Among the measured variants, {_escape_latex_text(best_name)} gives the largest held-out balanced accuracy at {_format_paper_number(best_score)}."
            if proposed_ba is not None:
                text += f" Its difference from the proposed configuration is {_format_paper_number(best_score - proposed_ba)}, so the ablation should be read alongside the feature and complexity columns rather than as a single-number leaderboard."
            if best_feat is not None and proposed_features is not None:
                text += f" The same variant uses {_format_paper_number(int(best_feat) if best_feat.is_integer() else best_feat)} selected features versus {_format_paper_number(int(proposed_features) if proposed_features.is_integer() else proposed_features)} for the proposed method, which clarifies whether the apparent gain comes with a larger measurement budget."
            text += " This framing keeps the ablation aligned with the paper's design question: which component helps preserve accuracy while keeping the final model inspectable?"
            return text

        def complexity_discussion() -> str:
            parts = [
                "The optimization and complexity diagnostics separate one-time search cost from deployed-model cost, which is essential for lightweight tabular claims."
            ]
            if proposed_features is not None or proposed_coef is not None:
                desc = []
                if proposed_features is not None:
                    desc.append(f"{_format_paper_number(int(proposed_features) if proposed_features.is_integer() else proposed_features)} selected features")
                if proposed_coef is not None:
                    desc.append(f"{_format_paper_number(int(proposed_coef) if proposed_coef.is_integer() else proposed_coef)} nonzero coefficients")
                parts.append("For the final classifier, the artifacts record " + " and ".join(desc) + ", which directly determines the number of measurements and signed coefficients a user must inspect.")
            if proposed_fit is not None or proposed_pred is not None:
                timing = []
                if proposed_fit is not None:
                    timing.append(f"fit time {_format_paper_number(proposed_fit)} seconds")
                if proposed_pred is not None:
                    timing.append(f"prediction time {_format_paper_number(proposed_pred)} seconds")
                parts.append("The timing record reports " + " and ".join(timing) + "; these values describe the measured local execution rather than a hardware-independent theoretical guarantee.")
            parts.append("Consequently, the diagnostic figures are used to judge whether the search procedure produces a model that is simpler at deployment time, not merely whether it reaches a competitive validation score during search.")
            return " ".join(parts)

        def pop_fig(*keywords: str) -> tuple[str, str]:
            for key, block in list(blocks.items()):
                key_label = " ".join([
                    key.lower(),
                    _GroundingTablesMixin._figure_label_from_block(block).lower(),
                ])
                caption_text = _GroundingTablesMixin._figure_caption_from_block(block).lower()
                haystack = f"{key_label} {caption_text}"
                if any(kw in key_label for kw in ("method", "framework", "schematic", "architecture", "overview")):
                    continue
                if any(kw in haystack for kw in keywords):
                    used.add(key)
                    blocks.pop(key, None)
                    return key, block.strip()
            return "", ""

        def figure_ref(key: str, block: str) -> str:
            label = _GroundingTablesMixin._figure_label_from_block(block) or key
            return label if label.startswith("fig:") else f"fig:{label}"

        def add_figure_block(block: str, *, placement: str = "ht", width: str = "0.58\\linewidth") -> None:
            if not block:
                return
            block = block.strip()
            block = re.sub(r"\\begin\{figure\}\[[^]]*\]", rf"\\begin{{figure}}[{placement}]", block, count=1)
            block = re.sub(r"\\begin\{figure\}(?!\[)", rf"\\begin{{figure}}[{placement}]", block, count=1)
            block = re.sub(
                r"\\includegraphics\[[^]]*\]",
                lambda _m: f"\\includegraphics[width={width}, height=0.20\\textheight, keepaspectratio]",
                block,
                count=1,
            )
            lines.extend(["", _format_numbers_in_text(block), ""])

        def join_refs(refs: list[str]) -> str:
            if not refs:
                return ""
            if len(refs) == 1:
                return refs[0]
            if len(refs) == 2:
                return refs[0] + " and " + refs[1]
            return ", ".join(refs[:-1]) + ", and " + refs[-1]

        main_key, main_fig = pop_fig("main_results", "main result", "model_comparison", "performance_comparison")
        ablation_key, ablation_fig = pop_fig("ablation", "variant")
        trade_key, trade_fig = pop_fig("tradeoff", "trade-off", "sparsity", "pareto", "frontier")
        complexity_key, complexity_fig = pop_fig("complexity", "efficiency", "runtime", "latency", "cost", "profile")
        optimization_key, optimization_fig = pop_fig("optimization", "history", "convergence")

        lines.extend([
            "We evaluate the method using only measurements produced by the executed local pipeline. Literature and OpenAlex-retrieved papers are used for positioning, while the quantitative tables below are restricted to runs that share the same dataset split, preprocessing boundary, and metric definitions.",
            "",
            r"\subsection{Experimental Protocol}",
            f"The experiment uses {_escape_latex_text(dataset)} and compares {proposed_name} against locally executed full-feature logistic-regression and random-forest baselines when those runs are available in the artifacts. The protocol reports predictive metrics together with feature-count and timing measurements because the objective is not only accuracy, but also inspectability and lightweight execution.",
            "The proposed configuration is selected from a Pareto front rather than from a single validation score. This matters for interpretation: a model can improve held-out accuracy by retaining more features, but such a point may be less useful for a lightweight diagnostic setting than a slightly lower-scoring model with a smaller inspected feature set.",
            "All reported scores are treated as split-specific measurements from the current run. This means that the tables support within-run comparisons among methods evaluated under the same preprocessing and split contract, while broader statistical claims would require repeated seeds or external validation data.",
        ])

        if grounding.main_table_latex:
            lines.extend([
                "",
                r"\subsection{Main Results}",
                "Table~\\ref{tab:main_results} gives the primary measured comparison. The table intentionally keeps literature-only baselines out of the numeric rows, so every reported value comes from the same local evaluation contract.",
                "",
                grounding.main_table_latex,
                "",
            ])
            lines.append(main_result_discussion())
            lines.append("The result should therefore be read as a controlled trade-off rather than a leaderboard claim. Full-feature logistic regression remains the natural accuracy reference, random forest provides a nonlinear baseline, and the proposed sparse model tests whether a fixed-budget Pareto search can recover comparable held-out behavior with a substantially smaller feature subset.")

        if main_fig:
            main_ref = f"Figure~\\ref{{{figure_ref(main_key, main_fig)}}}"
            lines.append(f"{main_ref} supports the first finding: the proposed operating point should be judged jointly by held-out behavior and inspected feature count. The figure is interpreted with Table~\\ref{{tab:main_results}}, so the visual evidence and numeric comparison come from the same local run rather than from separate or literature-only measurements.")
            add_figure_block(main_fig, placement="ht", width="0.58\\linewidth")

        if grounding.ablation_results and grounding.ablation_table_latex:
            lines.extend([
                "",
                r"\subsection{Ablation Study}",
                "The ablation study checks whether alternative design choices move the method along the same accuracy--compactness frontier or change the operating point in a materially different way.",
                "",
                grounding.ablation_table_latex,
                "",
            ])
            ablation_scores = []
            for entry in grounding.ablation_results:
                if isinstance(entry, dict):
                    score = _GroundingTablesMixin._numeric_metric(entry, "heldout_balanced_accuracy", "balanced_accuracy", "test_balanced_accuracy")
                    if score is not None:
                        ablation_scores.append((score, _GroundingTablesMixin._result_name(entry, "variant")))
            lines.append(ablation_discussion())
            lines.append("The ablation results are especially important because they prevent the paper from attributing every score difference to the evolutionary search itself. The best-accuracy selection and random-search variants test two different alternatives: changing the Pareto selection rule and replacing the structured search procedure. Their rows show whether accuracy gains come from the intended sparse-selection mechanism or from relaxing the compactness constraint.")
            if ablation_fig:
                ablation_ref = f"Figure~\\ref{{{figure_ref(ablation_key, ablation_fig)}}}"
                lines.append(f"{ablation_ref} is read with Table~\\ref{{tab:ablation}} rather than as a separate result: it shows whether the strongest held-out score also requires a larger selected-feature budget. This is the key distinction for the lightweight use case, because an ablation that gains accuracy by expanding the measurement set may be less aligned with the target deployment setting than a slightly lower-scoring but more compact configuration.")
                add_figure_block(ablation_fig, placement="ht", width="0.58\\linewidth")

        if trade_fig or complexity_fig or optimization_fig:
            opt_bits = []
            pareto_size = metric(proposed, "pareto_front_size")
            opt_time = metric(proposed, "optimization_time_seconds")
            runtime = metric(proposed, "runtime_seconds")
            if pareto_size is not None:
                opt_bits.append(f"a Pareto front with {_format_paper_number(int(pareto_size) if pareto_size.is_integer() else pareto_size)} retained points")
            if opt_time is not None:
                opt_bits.append(f"optimization time {_format_paper_number(opt_time)} seconds")
            if runtime is not None:
                opt_bits.append(f"total runtime {_format_paper_number(runtime)} seconds")
            opt_sentence = "The run artifacts also record " + " and ".join(opt_bits) + "." if opt_bits else "The run artifacts provide complexity diagnostics for the final selected model."
            lines.extend([
                "",
                r"\subsection{Optimization and Complexity Analysis}",
                complexity_discussion(),
                opt_sentence + " These values are not used as hardware-independent speed claims; they document the local search envelope and help separate one-time model-selection cost from deployed-model cost.",
            ])
            primary_diagnostic = None
            if trade_fig:
                primary_diagnostic = (trade_key, trade_fig, "trade")
            elif complexity_fig:
                primary_diagnostic = (complexity_key, complexity_fig, "complexity")
            elif optimization_fig:
                primary_diagnostic = (optimization_key, optimization_fig, "optimization")
            if primary_diagnostic:
                diag_key, diag_fig, diag_kind = primary_diagnostic
                diag_ref = f"Figure~\\ref{{{figure_ref(diag_key, diag_fig)}}}"
                if diag_kind == "trade":
                    lines.append(f"{diag_ref} links held-out score to selected-feature count, so the accuracy comparison is read as an accuracy--compactness trade-off rather than a standalone leaderboard. The figure is interpreted together with Table~\\ref{{tab:main_results}} and Table~\\ref{{tab:ablation}}, because the table values identify which operating points come from the same executed split.")
                elif diag_kind == "complexity":
                    lines.append(f"{diag_ref} separates deployment-time compactness from one-time search overhead. This distinction matters for lightweight tabular use cases: the deployed classifier is judged by selected features, coefficients, and prediction cost, while the wrapper search is a training-time model-selection expense.")
                else:
                    lines.append(f"{diag_ref} reports the optimization trace from the same artifacts. We use it only to characterize the fixed-budget search behavior, not to claim hardware-independent convergence guarantees.")
                add_figure_block(diag_fig, placement="ht", width="0.58\\linewidth")
            omitted = [name for name, fig in (("complexity", complexity_fig), ("optimization", optimization_fig)) if fig and primary_diagnostic and fig != primary_diagnostic[1]]
            if omitted:
                lines.append("Additional diagnostic plots are treated as supporting artifacts rather than extra main-text figures, because the main paper already contains the table and figure evidence needed for the stated claims.")


        remaining_result_figs = []
        for key, block in list(blocks.items()):
            haystack = f"{key} {_GroundingTablesMixin._figure_caption_from_block(block)}".lower()
            key_label = key.lower() + " " + _GroundingTablesMixin._figure_label_from_block(block).lower()
            if any(kw in key_label for kw in ("method", "framework", "schematic", "architecture", "overview")):
                continue
            if any(kw in haystack for kw in ("result", "accuracy", "ablation", "complexity", "runtime", "pareto", "tradeoff", "history")):
                remaining_result_figs.append((key, block.strip()))
                used.add(key)
        for key, block in remaining_result_figs[:2]:
            extra_ref = f"Figure~\\ref{{{figure_ref(key, block)}}}"
            lines.append(f"{extra_ref} adds a supporting diagnostic for the executed experiment. It is used only to qualify the measured comparison already discussed above, and the paper does not derive claims beyond the quantities shown in the figure.")
            add_figure_block(block, placement="ht", width="0.58\\linewidth")

        if grounding.evidence_gaps:
            readable_gaps = []
            for gap in grounding.evidence_gaps:
                gap_l = str(gap).lower()
                if "missing" in gap_l or "no " in gap_l or "quick" in gap_l:
                    readable_gaps.append(str(gap))
            if readable_gaps:
                lines.extend([
                    "",
                    r"\subsection{Scope of Evidence}",
                    "The reported claims are scoped to the executed artifacts available to the writing stage. Additional repeated-split or external-validation evidence would be required before making broader statistical claims.",
                ])

        lines.extend([
            "",
            "Overall, the experimental section supports claims that are directly tied to the measured local protocol. The narrative therefore emphasizes verified comparisons, ablation behavior, and lightweight-complexity diagnostics without filling missing categories with inferred numbers.",
        ])
        return "\n".join(lines).strip() + "\n", used

    _TOOL_SECTIONS = frozenset({"Introduction", "Related Work", "Method", "Experiments"})
