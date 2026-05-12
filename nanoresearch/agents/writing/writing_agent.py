"""WritingAgent main run method and figure placement logic."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from nanoresearch.evolution.memory import MemoryType
from ._types import ContributionContract, GroundingPacket
from .import _check_global_consistency, PAPER_SECTIONS, PAPER_MODE_SECTIONS
from .section_writer import SURVEY_SECTION_PROMPTS
from .grounding_tables import _format_paper_number
from nanoresearch.schemas.paper import PaperSkeleton, Section

logger = logging.getLogger(__name__)


class _WritingAgentMixin:
    """Mixin — WritingAgent.run() and figure placement logic."""

    async def run(self, **inputs: Any) -> dict[str, Any]:
        ideation: dict = inputs.get("ideation_output", {})
        blueprint: dict = inputs.get("experiment_blueprint", {})
        figure_output: dict = inputs.get("figure_output", {})
        template_format: str = inputs.get("template_format", self.config.template_format)
        experiment_results: dict = inputs.get("experiment_results", {})
        experiment_analysis: dict = inputs.get("experiment_analysis", {})
        experiment_summary: str = inputs.get("experiment_summary", "")
        experiment_status: str = inputs.get("experiment_status", "pending")
        authors: list[str] = inputs.get("authors", None) or ["NanoResearch"]

        # Detect paper mode from ideation output
        paper_mode_str: str = ideation.get("paper_mode", "original_research")
        is_survey = paper_mode_str != "original_research"
        self.log(f"Paper mode: {paper_mode_str} (survey={is_survey})")

        # Select section list based on paper_mode
        if is_survey:
            # Resolve paper_mode string to PaperMode enum for PAPER_MODE_SECTIONS lookup
            from nanoresearch.schemas.manifest import PaperMode
            try:
                paper_mode_enum = PaperMode(paper_mode_str)
            except ValueError:
                paper_mode_enum = PaperMode.ORIGINAL_RESEARCH
            section_list = PAPER_MODE_SECTIONS.get(paper_mode_enum, PAPER_SECTIONS)
        else:
            section_list = PAPER_SECTIONS

        self.log("Starting paper writing")
        adaptive_context = self.build_adaptive_context(
            "writing",
            topic=ideation.get("topic", ""),
            blueprint=blueprint,
            text=json.dumps({
                "paper_mode": paper_mode_str,
                "topic": ideation.get("topic", ""),
                "selected_hypothesis": ideation.get("selected_hypothesis", ""),
            }, ensure_ascii=False),
            tags=[ideation.get("topic", ""), paper_mode_str, template_format],
            template_format=template_format,
            include_script_recommendations=True,
        )
        retry_error = str(inputs.get("_retry_error", "")).strip()
        if retry_error:
            self.learn_from_trace(
                "writing",
                "writing_retry",
                retry_error,
                tags=[ideation.get("topic", ""), paper_mode_str, "retry"],
            )

        # Step 0a: Build grounding packet
        grounding = self._build_grounding_packet(
            experiment_results, experiment_status,
            experiment_analysis, experiment_summary, blueprint,
        )
        self.log(
            f"Grounding: completeness={grounding.result_completeness}, "
            f"main_results={len(grounding.main_results)}, "
            f"ablations={len(grounding.ablation_results)}, "
            f"baselines={'yes' if grounding.comparison_with_baselines else 'no'}"
        )
        if grounding.evidence_gaps:
            self.log(f"Evidence gaps: {grounding.evidence_gaps}")

        # If execution produced no real metrics, write directly from the grounded
        # context. Tool-augmented literature loops are useful for full papers, but
        # in no-result smoke runs they can dominate runtime without adding usable
        # evidence for the proposed method.
        self._disable_writing_tools_for_stage = not grounding.has_real_results

        # Step 0b: Expand and build cite key mapping from papers.
        # Full-paper mode needs enough literature context for Introduction and
        # Related Work, but these citations are never used as measured results.
        ideation = await self._expand_citation_pool(ideation, blueprint, target_count=28)
        papers = ideation.get("papers", [])
        cite_keys = self._build_cite_keys(papers)
        bibtex = self._build_bibtex(papers, cite_keys)

        # Build per-section context primitives (P0-A). Keep this as a dict;
        # section-specific builders index structured fields from it.
        core_ctx = self._build_core_context(ideation, blueprint, cite_keys)
        if adaptive_context:
            core_ctx["adaptive_context"] = adaptive_context

        # Title & abstract need a broad context
        title_abstract_ctx = self._ctx_introduction(core_ctx, grounding=grounding)
        if adaptive_context:
            title_abstract_ctx = f"{title_abstract_ctx}\n\n{adaptive_context}"

        # Step 1: Generate title
        title = await self._generate_title(title_abstract_ctx)
        self.log(f"Title: {title}")

        # Step 2: Generate abstract
        abstract = await self._generate_abstract(title_abstract_ctx, grounding)
        self.log("Abstract generated")

        # Step 3: Build figures & table data from blueprint
        figure_blocks = self._build_figure_blocks(blueprint, figure_output)

        # Step 3b: Expand router/adaptive guidance into a concrete writing plan.
        paper_structure_plan = await self._generate_writing_stage_plan(
            ideation=ideation,
            blueprint=blueprint,
            grounding=grounding,
            figure_output=figure_output,
            core_ctx=core_ctx,
            adaptive_context=adaptive_context or "",
            section_list=section_list,
            template_format=template_format,
            is_survey=is_survey,
        )
        self.log("Paper structure plan generated")

        # Step 4: Generate each section independently, embed figures inline
        placed_figures: set[str] = set()

        # P0-B: Contribution contract
        contribution_contract: ContributionContract | None = None
        method_name = (blueprint.get("proposed_method") or {}).get("name", "")

        sections = []
        prior_sections_summary: list[str] = []
        for heading, label, section_instructions, fig_keys in section_list:
            # For surveys: use survey-specific prompts (stored separately) and skip experiment results
            if is_survey:
                instructions = SURVEY_SECTION_PROMPTS.get(label, section_instructions)
                # Surveys do not have experiment results — pass None/defaults to context builders
                ctx_experiment_results: dict | None = None
                ctx_experiment_status: str = "pending"
                ctx_experiment_analysis: dict | None = None
                ctx_experiment_summary: str = ""
            else:
                instructions = self._augment_section_instructions_with_plan(
                    section_instructions, heading, paper_structure_plan
                )
                ctx_experiment_results = experiment_results
                ctx_experiment_status = experiment_status
                ctx_experiment_analysis = experiment_analysis
                ctx_experiment_summary = experiment_summary

            self.log(f"Writing section: {heading}")

            _prior_content = {s.heading: s.content for s in sections}
            section_ctx = self._build_section_context(
                label, core_ctx, grounding=grounding,
                experiment_results=ctx_experiment_results,
                experiment_status=ctx_experiment_status,
                experiment_analysis=ctx_experiment_analysis,
                experiment_summary=ctx_experiment_summary,
                prior_sections=_prior_content,
            )
            if adaptive_context:
                section_ctx = f"{section_ctx}\n\n{adaptive_context}"
            plan_block = self._writing_plan_section_block(paper_structure_plan, heading)
            if plan_block:
                section_ctx = f"{section_ctx}\n\n{plan_block}"

            # Contribution contract is only for original research (not surveys)
            if not is_survey and contribution_contract and label != "sec:intro":
                contract_block = contribution_contract.for_section(label)
                if contract_block:
                    section_ctx = section_ctx + "\n\n" + contract_block

            remaining_figs = [k for k in figure_blocks if k not in placed_figures]
            fig_list_text = "\n".join(
                f"  - \\ref{{fig:{k}}}: {k}" for k in remaining_figs
            )
            placed_note = ""
            if placed_figures:
                placed_list = ", ".join(sorted(placed_figures))
                placed_note = (
                    f"\nFigures ALREADY placed in previous sections (do NOT include again): "
                    f"{placed_list}\n"
                )

            table_injection = ""
            if label == "sec:experiments":
                table_parts = []
                if grounding.main_table_latex and grounding.has_real_results:
                    header = "=== PRE-BUILT MAIN RESULTS TABLE (use this EXACTLY, do NOT rebuild) ==="
                    table_parts.append(
                        header + "\n" + grounding.main_table_latex + "\n=== END PRE-BUILT TABLE ==="
                    )
                if grounding.ablation_table_latex and grounding.has_real_results:
                    header = "=== PRE-BUILT ABLATION TABLE (use this EXACTLY, do NOT rebuild) ==="
                    table_parts.append(
                        header + "\n" + grounding.ablation_table_latex + "\n=== END PRE-BUILT TABLE ==="
                    )
                if table_parts:
                    table_injection = "\n\n" + "\n\n".join(table_parts)

            conclusion_binding = ""
            if label == "sec:conclusion":
                if is_survey:
                    # Surveys bind to key_challenges and future_directions instead of experiment results
                    ideation = inputs.get("ideation_output", {})
                    key_challenges = ideation.get("key_challenges", []) if isinstance(ideation, dict) else []
                    future_directions = ideation.get("future_directions", []) if isinstance(ideation, dict) else []
                    if key_challenges or future_directions:
                        challenges_str = "\n".join(f"  - {t}" for t in key_challenges) if key_challenges else "  (none provided)"
                        directions_str = "\n".join(f"  - {t}" for t in future_directions) if future_directions else "  (none provided)"
                        conclusion_binding = (
                            "\n\n=== CONCLUSION RESULT BINDING (SURVEY) ===\n"
                            "Key Challenges:\n" + challenges_str + "\n\n"
                            "Future Directions:\n" + directions_str + "\n\n"
                            "Use these to summarize open challenges and future research trajectories.\n"
                            "Do NOT cite specific experiment performance numbers.\n"
                            "=== END BINDING ==="
                        )
                elif grounding.has_real_results and grounding.final_metrics:
                    metric_strs = [f"{k}={_format_paper_number(v)}" for k, v in list(grounding.final_metrics.items())[:5]]
                    conclusion_binding = (
                        "\n\n=== CONCLUSION RESULT BINDING ===\n"
                        f"Real metrics to reference: {', '.join(metric_strs)}\n"
                        "Mention key results quantitatively when summarizing contributions. "
                        "Use the rounded display values above.\n"
                        "=== END BINDING ==="
                    )
                elif not grounding.has_real_results:
                    conclusion_binding = (
                        "\n\n=== CONCLUSION RESULT BINDING ===\n"
                        "No real experiment results. Do NOT cite specific performance numbers. "
                        "Focus on method design and future work.\n"
                        "=== END BINDING ==="
                    )

            context_with_figs = (
                f"{section_ctx}\n\n"
                f"=== AVAILABLE FIGURES (use \\ref{{fig:NAME}} to reference) ===\n"
                f"{fig_list_text}\n"
                f"{placed_note}"
                f"=== END FIGURES ==="
                f"{table_injection}"
                f"{conclusion_binding}"
            )

            deterministic_experiments = (
                label == "sec:experiments"
                and grounding.has_real_results
                and bool(grounding.main_table_latex)
            )
            if deterministic_experiments:
                content, composer_placed = self._compose_experiments_section(
                    grounding, figure_blocks, blueprint, include_heading=False,
                )
                placed_figures.update(composer_placed)
                self.log(
                    f"  Experiments composed from verified artifacts "
                    f"({len(composer_placed)} figures placed)"
                )
            else:
                content = await self._generate_section(
                    context_with_figs, heading, instructions, prior_sections_summary
                )

                # Post-generation table verification for Experiments
                if label == "sec:experiments" and (
                    grounding.main_table_latex or grounding.ablation_table_latex
                ):
                    content = self._verify_and_inject_tables(content, grounding, heading)

            # Detect figures the LLM already embedded
            llm_placed_labels = re.findall(
                r'\\begin\{figure\*?\}.*?\\label\{fig:([^}]+)\}.*?\\end\{figure\*?\}',
                content, re.DOTALL,
            )
            for fig_label in llm_placed_labels:
                if fig_label in figure_blocks and fig_label not in placed_figures:
                    placed_figures.add(fig_label)
                    self.log(f"  LLM already placed fig:{fig_label} in {heading}")
            llm_placed_files = re.findall(
                r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content,
            )
            for fname in llm_placed_files:
                stem = fname.rsplit(".", 1)[0]
                for fk in figure_blocks:
                    if fk in placed_figures:
                        continue
                    if fk in stem or stem.endswith(fk):
                        placed_figures.add(fk)
                        self.log(f"  LLM already included {fname} -> marking fig:{fk} as placed in {heading}")

            # Smart figure placement. Artifact-composed Experiments already
            # contains prose around each figure, so do not append more result
            # figures after the composer has ordered them.
            if not deterministic_experiments:
                content, placed_figures = self._place_section_figures(
                    content, label, heading, figure_blocks, placed_figures,
                )

            sections.append(Section(heading=heading, label=label, content=content))
            snippet = content[:200].replace("\n", " ").strip()
            prior_sections_summary.append(f"[{heading}]: {snippet}...")

            # P0-B: Extract contribution contract after Introduction (original research only)
            if not is_survey and label == "sec:intro" and not contribution_contract:
                contribution_contract = self._extract_contribution_contract(content, method_name)
                if contribution_contract.claims:
                    self.log(
                        f"Contribution contract: {len(contribution_contract.claims)} claims "
                        f"({', '.join(c.claim_type for c in contribution_contract.claims)})"
                    )
                else:
                    self.log("No contribution claims extracted from Introduction")

        # Fallback: distribute remaining figures
        remaining = [k for k in figure_blocks if k not in placed_figures]
        if remaining:
            self.log(f"Fallback placement for {len(remaining)} unplaced figures: {remaining}")
            section_hints = {
                "sec:experiments": ("result", "comparison", "performance", "main", "baseline",
                                    "ablation", "latency", "runtime", "complexity", "pareto",
                                    "history", "optimization", "tradeoff", "trade_off",
                                    "efficiency", "scalab", "accuracy", "loss"),
                "sec:method": ("architecture", "framework", "pipeline", "overview", "model",
                               "diagram", "workflow", "schematic", "method", "task", "motivation", "teaser",
                               "intuition", "illustration"),
            }
            for fk in remaining:
                target_label = "sec:experiments"
                for sec_label, keywords in section_hints.items():
                    if any(kw in fk for kw in keywords):
                        target_label = sec_label
                        break
                for sec in sections:
                    if sec.label == target_label:
                        sec.content += "\n\n" + figure_blocks[fk]
                        placed_figures.add(fk)
                        self.log(f"  Placed '{fk}' -> {target_label}")
                        break
                else:
                    for sec in sections:
                        if sec.label == "sec:experiments":
                            sec.content += "\n\n" + figure_blocks[fk]
                            placed_figures.add(fk)
                            self.log(f"  Placed '{fk}' -> sec:experiments (fallback)")
                            break

        # Post-assembly validation
        final_missing = [k for k in figure_blocks if k not in placed_figures]
        if final_missing:
            self.log(f"CRITICAL: {len(final_missing)} figures still unplaced after all passes: {final_missing}")
            for sec in sections:
                if sec.label == "sec:experiments":
                    for fk in final_missing:
                        sec.content += "\n\n" + figure_blocks[fk]
                        self.log(f"  Force-injected '{fk}' -> sec:experiments")
                    break

        self.log(f"Figure placement complete: {len(figure_blocks)} blocks, "
                 f"{len(placed_figures)} placed")

        # Per-section dedup
        self._dedup_section_figures(sections)

        # Step 5: Build skeleton
        skeleton = PaperSkeleton(
            title=title, authors=authors, abstract=abstract,
            sections=sections, figures=[],
            template_format=template_format, references_bibtex=bibtex,
        )

        # Step 6: Render LaTeX + sanitize
        latex_content = self._render_latex(skeleton)
        latex_content = self._sanitize_latex(latex_content)

        structure_issues = self._audit_paper_structure_against_plan(
            latex_content, paper_structure_plan
        )
        if structure_issues:
            self.log(f"Paper structure plan audit: {len(structure_issues)} issue(s)")
            for issue in structure_issues:
                self.log(f"  - {issue}")

        # Step 6b-pre: Full-document figure dedup
        latex_content = self._dedup_full_doc_figures(latex_content)

        # Step 6b: Final LaTeX-level figure validation
        latex_content = self._validate_figures_in_latex(latex_content, figure_output)

        # Step 6b.5: Ensure every placed figure has a natural LLM-written reference.
        latex_content = await self._ensure_llm_figure_references(
            latex_content, figure_output, grounding
        )
        latex_content = self._precompile_static_layout_audit(latex_content)

        # Step 6c: Resolve missing citations
        bibtex = await self._resolve_missing_citations(latex_content, bibtex)

        # Step 6d: Citation coverage validation
        citation_report = self._validate_citation_coverage(latex_content, ideation, cite_keys)
        if citation_report.get("missing_must_cites"):
            self.log(f"Must-cite enforcement: {len(citation_report['missing_must_cites'])} "
                     f"must-cite papers not referenced, injecting into Related Work")
            latex_content = self._inject_must_cites(
                latex_content, citation_report["missing_must_cites"], cite_keys, ideation
            )
            bibtex = await self._resolve_missing_citations(latex_content, bibtex)

        self._log_citation_report(citation_report)

        # Step 6d.5: Ensure full-paper literature coverage, then cleanup unused BibTeX entries.
        latex_content = self._ensure_minimum_citations(latex_content, ideation, cite_keys, min_refs=20)
        bibtex = await self._resolve_missing_citations(latex_content, bibtex)
        bibtex = self._cleanup_unused_bibtex(latex_content, bibtex)

        # Step 6e: Global consistency check
        consistency_issues = _check_global_consistency(latex_content, abstract, sections)
        if consistency_issues:
            self.log(f"Consistency check: {len(consistency_issues)} issue(s) found")
            for issue in consistency_issues:
                self.log(f"  - {issue}")

        # Save outputs
        tex_path = self.workspace.write_text("drafts/paper.tex", latex_content)
        bib_content = self._sanitize_bibtex(bibtex)
        bib_path = self.workspace.write_text("drafts/references.bib", bib_content)
        skeleton_path = self.workspace.write_json(
            "drafts/paper_skeleton.json", skeleton.model_dump(mode="json"),
        )

        self.workspace.register_artifact("paper_tex", tex_path, self.stage)
        self.workspace.register_artifact("references_bib", bib_path, self.stage)
        self.workspace.register_artifact("paper_skeleton", skeleton_path, self.stage)

        # Step 7: Try to compile PDF
        pdf_result = await self._compile_pdf(tex_path, template_format=template_format)

        result = {
            "tex_path": str(tex_path),
            "bib_path": str(bib_path),
            "paper_tex": latex_content,
            "grounding": grounding.to_output_dict(),
            "paper_structure_plan": paper_structure_plan,
            "paper_structure_issues": structure_issues,
            "consistency_issues": consistency_issues,
        }
        if "pdf_path" in pdf_result:
            result["pdf_path"] = pdf_result["pdf_path"]
            self.workspace.register_artifact(
                "paper_pdf", self.workspace.path / "drafts" / "paper.pdf", self.stage
            )
        else:
            result["pdf_error"] = pdf_result.get("error", "Unknown error")
            self.log(f"PDF compilation failed: {result['pdf_error']}")

        topic_name = ideation.get("topic", "unknown topic")
        pdf_ready = "yes" if "pdf_path" in result else "no"
        self.remember_context(
            MemoryType.PROJECT_CONTEXT,
            f"Writing completed for {topic_name} in mode {paper_mode_str} with template {template_format}. PDF={pdf_ready}.",
            importance=0.7,
            tags=[ideation.get("topic", ""), paper_mode_str, "writing", template_format],
            source="writing_output",
            topic=ideation.get("topic", ""),
        )
        writing_trace = (
            f"Writing completed for {topic_name}: paper_mode={paper_mode_str}; template={template_format}; "
            f"pdf_ready={pdf_ready}; consistency_issues={len(consistency_issues)}; "
            f"must_cite_missing={len(citation_report.get('missing_must_cites', []))}."
        )
        self.learn_from_trace(
            "writing",
            "writing_completion",
            writing_trace,
            tags=[ideation.get("topic", ""), paper_mode_str, "writing", template_format],
            confidence=0.64,
        )
        self.log("Writing stage complete")
        return result


    @staticmethod
    def _latex_without_figure_blocks(latex_content: str) -> str:
        return re.sub(
            r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
            "",
            latex_content,
            flags=re.DOTALL,
        )

    @staticmethod
    def _extract_figure_caption(figure_block: str) -> str:
        match = re.search(r"\\caption\{(.*?)\}\s*\\label", figure_block, re.DOTALL)
        if not match:
            match = re.search(r"\\caption\{(.*?)\}", figure_block, re.DOTALL)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip()

    @staticmethod
    def _nearby_section_text(latex_content: str, position: int, max_chars: int = 1800) -> str:
        start = max(0, position - max_chars)
        end = min(len(latex_content), position + max_chars)
        snippet = latex_content[start:end]
        snippet = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "", snippet, flags=re.DOTALL)
        snippet = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "", snippet, flags=re.DOTALL)
        return re.sub(r"\s+", " ", snippet).strip()[:max_chars]

    @staticmethod
    def _figure_artifact_summary(figure_output: dict | None, grounding: GroundingPacket) -> str:
        figures = (figure_output or {}).get("figures", {}) if isinstance(figure_output, dict) else {}
        figure_lines: list[str] = []
        if isinstance(figures, dict):
            for key, data in list(figures.items())[:8]:
                if not isinstance(data, dict):
                    continue
                figure_lines.append(
                    f"- {key}: type={data.get('fig_type') or data.get('figure_type') or data.get('kind') or 'unknown'}; "
                    f"caption={data.get('caption', '')}"
                )
        metric_lines: list[str] = []
        for entry in grounding.main_results[:4]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("method_name") or entry.get("variant_name") or entry.get("method") or entry.get("role") or "method"
            bits: list[str] = []
            for metric in entry.get("metrics", [])[:3]:
                if isinstance(metric, dict) and metric.get("value") is not None:
                    bits.append(f"{metric.get('metric_name', 'metric')}={_format_paper_number(metric.get('value'))}")
            if bits:
                metric_lines.append(f"- {name}: {', '.join(bits)}")
        ablation_lines: list[str] = []
        for entry in grounding.ablation_results[:3]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("variant_name") or entry.get("method_name") or entry.get("method") or "variant"
            bits: list[str] = []
            for metric in entry.get("metrics", [])[:2]:
                if isinstance(metric, dict) and metric.get("value") is not None:
                    bits.append(f"{metric.get('metric_name', 'metric')}={_format_paper_number(metric.get('value'))}")
            if bits:
                ablation_lines.append(f"- {name}: {', '.join(bits)}")
        return "\n".join([
            "Figures:", *(figure_lines or ["- no figure metadata"]),
            "Measured main results:", *(metric_lines or ["- no measured main results"]),
            "Measured ablations:", *(ablation_lines or ["- no measured ablations"]),
            "Evidence gaps:", *(f"- {gap}" for gap in grounding.evidence_gaps[:5]),
        ])

    async def _write_figure_reference_sentence(
        self,
        *,
        label: str,
        caption: str,
        nearby_text: str,
        artifact_summary: str,
    ) -> str:
        """Ask the configured writing model for a constrained figure-reference sentence."""
        system_prompt = (
            "You write concise LaTeX paper prose grounded strictly in provided artifacts. "
            "Do not invent numbers, baselines, datasets, or claims."
        )
        prompt = f"""Write 1-2 natural LaTeX sentences that introduce and interpret this figure.

Required exact reference: Figure~\\ref{{{label}}}
Caption: {caption or 'No caption provided'}

Artifact summary:
{artifact_summary[:5000]}

Nearby section text:
{nearby_text[:1800]}

Rules:
- The output MUST include the exact string Figure~\\ref{{{label}}}.
- Do not output a figure environment, table, bullet list, heading, or markdown.
- Do not invent numeric values. If you mention numbers, they must appear in the artifact summary.
- Keep it to at most 2 sentences.
- Output only the sentences."""
        try:
            sentence = ((await self.generate(system_prompt, prompt, stage_override=self.config.revision)) or "").strip()
        except Exception as exc:
            self.log(f"  Figure reference LLM failed for {label}: {exc}")
            return ""
        sentence = re.sub(r"```(?:latex|tex)?\s*|```", "", sentence).strip()
        sentence = re.sub(r"\s+", " ", sentence)
        if f"Figure~\\ref{{{label}}}" not in sentence:
            return ""
        if re.search(r"\\begin\{|\\end\{|\\section\{|\\caption\{|\\includegraphics", sentence):
            return ""
        parts = re.split(r"(?<=[.!?])\s+", sentence)
        return " ".join(parts[:2]).strip()

    async def _ensure_llm_figure_references(
        self,
        latex_content: str,
        figure_output: dict | None,
        grounding: GroundingPacket,
    ) -> str:
        """Insert constrained GPT-written references for figure blocks not cited in prose."""
        prose = self._latex_without_figure_blocks(latex_content)
        figure_matches = list(re.finditer(
            r"\\begin\{figure\*?\}.*?\\label\{(fig:[^}]+)\}.*?\\end\{figure\*?\}",
            latex_content,
            re.DOTALL,
        ))
        if not figure_matches:
            return latex_content
        artifact_summary = self._figure_artifact_summary(figure_output, grounding)
        inserts: list[tuple[int, str, str]] = []
        for match in figure_matches:
            label = match.group(1)
            ref_pat = re.compile(rf"\\(?:ref|autoref|cref)\{{{re.escape(label)}\}}")
            if ref_pat.search(prose):
                continue
            caption = self._extract_figure_caption(match.group(0))
            nearby = self._nearby_section_text(latex_content, match.start())
            sentence = await self._write_figure_reference_sentence(
                label=label,
                caption=caption,
                nearby_text=nearby,
                artifact_summary=artifact_summary,
            )
            if sentence:
                inserts.append((match.start(), label, sentence))
            else:
                self.log(f"  No safe LLM reference generated for {label}; leaving figure placement unchanged")
        for pos, label, sentence in reversed(inserts):
            latex_content = latex_content[:pos] + sentence + "\n\n" + latex_content[pos:]
            self.log(f"  Inserted LLM-written reference for {label}")
        return latex_content

    def _place_section_figures(
        self,
        content: str,
        label: str,
        heading: str,
        figure_blocks: dict[str, str],
        placed_figures: set[str],
    ) -> tuple[str, set[str]]:
        """Place figures by paper section; Intro and Conclusion stay figure-free."""
        _arch_kws = ("overview", "framework", "pipeline", "architecture", "model", "workflow", "diagram", "schematic", "method")
        _result_kws = (
            "result", "comparison", "performance", "main", "baseline",
            "ablation", "accuracy", "loss", "efficiency", "latency",
            "runtime", "complexity", "pareto", "history", "optimization",
            "tradeoff", "trade_off", "sparsity", "cost",
        )

        if label in {"sec:intro", "sec:conclusion"}:
            return content, placed_figures

        if label == "sec:method":
            for fk in list(figure_blocks.keys()):
                if fk not in placed_figures and any(kw in fk.lower() for kw in _arch_kws):
                    content = figure_blocks[fk] + "\n\n" + content.lstrip("\n")
                    placed_figures.add(fk)
                    self.log(f"  Placed method figure fig:{fk} at top of Method")
                    break

        if label == "sec:experiments":
            for fk in list(figure_blocks.keys()):
                if fk in placed_figures:
                    continue
                key_l = fk.lower()
                if any(kw in key_l for kw in _result_kws):
                    content, inserted = self._insert_figure_near_ref(content, fk, figure_blocks[fk])
                    if not inserted:
                        content += "\n\n" + figure_blocks[fk]
                    placed_figures.add(fk)

        # Insert remaining eligible figures near their references, but never leak
        # result figures into Method or method figures into Experiments.
        for fk, blk in figure_blocks.items():
            if fk in placed_figures:
                continue
            key_l = fk.lower()
            if label == "sec:method" and any(kw in key_l for kw in _result_kws):
                continue
            if label == "sec:experiments" and any(kw in key_l for kw in _arch_kws):
                continue
            content, inserted = self._insert_figure_near_ref(content, fk, blk)
            if inserted:
                placed_figures.add(fk)

        return content, placed_figures

    def _dedup_section_figures(self, sections: list[Section]) -> None:
        """Remove duplicate figure blocks across sections (keep first occurrence)."""
        seen_fig_labels: set[str] = set()
        seen_fig_files: set[str] = set()
        for sec in sections:
            def _dedup_figure(m: re.Match) -> str:
                block = m.group(0)
                label_m = re.search(r'\\label\{(fig:[^}]+)\}', block)
                lbl = label_m.group(1) if label_m else None
                if lbl and lbl in seen_fig_labels:
                    self.log(f"  Removed duplicate figure {lbl} from {sec.heading}")
                    return ""
                file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
                if file_m:
                    fname = file_m.group(1)
                    if fname in seen_fig_files:
                        self.log(f"  Removed duplicate figure file {fname} from {sec.heading}")
                        return ""
                    seen_fig_files.add(fname)
                if lbl:
                    seen_fig_labels.add(lbl)
                return block
            sec.content = re.sub(
                r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
                _dedup_figure, sec.content, flags=re.DOTALL,
            )
            sec.content = re.sub(r'\n{3,}', '\n\n', sec.content)

    def _dedup_full_doc_figures(self, latex_content: str) -> str:
        """Full-document figure dedup (safety net after assembly)."""
        seen_labels: set[str] = set()
        seen_files: set[str] = set()

        def _dedup_assembled(m: re.Match) -> str:
            block = m.group(0)
            lbl_m = re.search(r'\\label\{(fig:[^}]+)\}', block)
            if lbl_m:
                lbl = lbl_m.group(1)
                if lbl in seen_labels:
                    self.log(f"  Full-doc dedup: removed duplicate figure {lbl}")
                    return ""
                seen_labels.add(lbl)
            file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
            if file_m:
                fname = file_m.group(1)
                if fname in seen_files:
                    self.log(f"  Full-doc dedup: removed duplicate figure file {fname}")
                    return ""
                seen_files.add(fname)
            return block

        latex_content = re.sub(
            r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
            _dedup_assembled, latex_content, flags=re.DOTALL,
        )
        latex_content = re.sub(r'\n{3,}', '\n\n', latex_content)
        return latex_content
