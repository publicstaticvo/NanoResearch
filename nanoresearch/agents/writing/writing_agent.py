"""WritingAgent main run method and figure placement logic."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from ._types import ContributionContract, GroundingPacket
from .import _check_global_consistency, PAPER_SECTIONS, PAPER_MODE_SECTIONS
from .grounding_tables import infer_expected_section
from .section_writer import SURVEY_SECTION_PROMPTS
from nanoresearch.evolution.memory import MemoryType
from nanoresearch.schemas.paper import PaperSkeleton, Section

logger = logging.getLogger(__name__)


# A-5f: section-level orphan self-check (H1 front-load).
# Regex mirrors agents/review/consistency.py:147-149 so writing-side and
# review-side detections agree on what "orphan" means.
_A5F_LABEL_PATTERN = re.compile(r'\\label\{((?:fig|tab):[^}]+)\}')
_A5F_REF_PATTERN = re.compile(r'\\(?:(?:auto|[Cc])?ref)\{((?:fig|tab):[^}]+)\}')

def _check_float_consistency(
    content: str,
    *,
    global_labels: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(orphans, danglings)`` for a ``content`` string.

    ``orphans``  = section-local ``labels - refs``  (A-5f H1 scope).
    ``danglings`` = ``refs - global_labels``         (A-5g H2 scope).

    When ``global_labels`` is ``None`` the dangling pass is SKIPPED —
    not rendered as "section-local danglings", it is genuinely not
    run, and an empty list is returned in the second slot.  This
    signature deliberately fails loud: forgetting to pass
    ``global_labels`` at a call site that needed global detection
    gives zero danglings unambiguously, instead of a silent half-check
    against section-local labels that would produce large numbers of
    false positives for cross-section refs.
    """
    labels = set(_A5F_LABEL_PATTERN.findall(content))
    refs = set(_A5F_REF_PATTERN.findall(content))
    orphans = sorted(labels - refs)
    if global_labels is None:
        return orphans, []
    return orphans, sorted(refs - global_labels)


def _check_section_orphans(content: str) -> list[str]:
    """Return sorted list of ``fig:`` / ``tab:`` labels in ``content`` that
    have no matching ``\\ref`` / ``\\autoref`` / ``\\Cref`` / ``\\cref`` in
    the same string. Empty list means section is orphan-free.

    Thin wrapper over :func:`_check_float_consistency` preserved for
    call-site stability at the six A-5f integration points (pre / post
    / global).  A-5g callers invoke :func:`_check_float_consistency`
    directly with ``global_labels`` to get the dangling slot."""
    return _check_float_consistency(content)[0]


# A-5g: strict-form dangling ref regex for the limited-A fallback path.
# Matches ``Figure~\ref{fig:X}`` / ``Table~\Cref{tab:Y}`` / ``\autoref``
# / ``\cref`` variants when preceded by the ``Figure~`` or ``Table~``
# noun prefix.  Captures only these; other forms (bare ``\ref``,
# ``Figures~`` / ``Tables~`` plural prefix, space-separated noun, etc.)
# fall through to warning-only (default C) and surface through the
# A<->B collaboration channel as ``[layout/H2]``.  Capture group
# returns the label so the consumer can check it against the
# per-section dangling set before rewriting.
_A5G_STRICT_RE = re.compile(
    r'(?:Figure|Table)~\\(?:auto|[Cc])?ref\{((?:fig|tab):[^}]+)\}'
)


def _build_orphan_retry_feedback(orphans: list[str]) -> str:
    """Format a retry feedback block in the same style as _format_gate_feedback
    so the LLM treats it as authoritative pivot instruction."""
    joined = ", ".join(orphans)
    return (
        "\n\n=== ORPHAN RETRY FEEDBACK (MANDATORY FIX) ===\n"
        f"Your previous draft left these labels without any \\ref in this section: {joined}.\n"
        "Rewrite the section so every listed label is cited at least once "
        "(use Figure~\\ref{fig:X} for figures, Table~\\ref{tab:Y} for tables).\n"
        "Do not remove the labels; add refs in the prose instead.\n"
        "=== END RETRY FEEDBACK ==="
    )


def _inject_orphan_ref_stub(content: str, orphans: list[str]) -> str:
    """Fallback: append a minimal inline citation sentence for each still-orphan
    label to the end of the section. Used in A-5f post-injection stage, where
    `_verify_and_inject_tables` / `_place_section_figures` may have added a
    `\\label` without a matching `\\ref` — we can't retry LLM generation
    (it would discard injection), so we synthesize a safe reference stub.

    Templates are deliberately generic (no fabricated captions). Tables get
    Experiments-flavored wording since injected tables are almost always
    main_results or ablation in the Experiments section."""
    if not orphans:
        return content
    stubs = []
    for label in orphans:
        kind, _, key = label.partition(":")
        if kind == "fig":
            stubs.append(f"Figure~\\ref{{{label}}} visualizes this analysis.")
        elif kind == "tab":
            if "ablation" in key:
                stubs.append(
                    f"Table~\\ref{{{label}}} reports the ablation results "
                    "discussed above."
                )
            elif "main" in key or "result" in key:
                stubs.append(
                    f"Table~\\ref{{{label}}} summarizes the main quantitative "
                    "results."
                )
            else:
                stubs.append(
                    f"Table~\\ref{{{label}}} reports the corresponding "
                    "quantitative results."
                )
    if not stubs:
        return content
    return content.rstrip() + "\n\n" + " ".join(stubs) + "\n"


# ---------------------------------------------------------------------------
# A-5h: global-level counterpart of _inject_orphan_ref_stub used by the
# review agent's Step 3b (post-revision orphan check).  Scope differs:
# A-5f operates on a single section string during writing; A-5h operates
# on the whole paper.tex after the revision loop has completed and may
# have deleted the stub reference A-5f wrote during writing.
# ---------------------------------------------------------------------------
_A5H_SECTION_OPEN_RE = re.compile(r"\\section\{[^}]+\}")
_A5H_SECTION_STAR_RE = re.compile(r"\\section\*\{[^}]+\}")
_A5H_END_DOCUMENT_RE = re.compile(r"\\end\{document\}")
_A5H_ABSTRACT_END_RE = re.compile(r"\\end\{abstract\}")
_A5H_REF_RE_TMPL = r"\\(?:ref|autoref|[Cc]ref|eqref)\{{{}}}"
_A5H_LABEL_RE_TMPL = r"\\label\{{{}}}"
_A5H_AUTOFIX_MARKER = "% [A-5h auto-fixed] orphan ref stub injected post-revision"


def _a5h_render_stub_sentence(label: str) -> str:
    """Mirror of the inline templates in _inject_orphan_ref_stub above —
    kept as an in-module copy rather than shared helper to preserve the
    original single-section function untouched (Day 5 conservative
    refactor rule)."""
    kind, _, key = label.partition(":")
    if kind == "fig":
        return f"Figure~\\ref{{{label}}} visualizes this analysis."
    if kind == "tab":
        if "ablation" in key:
            return (
                f"Table~\\ref{{{label}}} reports the ablation results "
                "discussed above."
            )
        if "main" in key or "result" in key:
            return (
                f"Table~\\ref{{{label}}} summarizes the main quantitative "
                "results."
            )
        return (
            f"Table~\\ref{{{label}}} reports the corresponding "
            "quantitative results."
        )
    return ""


def _a5h_find_section_end(
    full_tex: str, section_start: int, section_end_after: int
) -> int:
    """Return the insert position for stub appended to the numbered
    section that opens at ``section_start``.

    Strategy: scan forward from ``section_end_after`` (= ``\\section{...}``
    closing brace) for the nearest hard boundary — another
    ``\\section{...}``, a ``\\section*{...}`` (e.g. Acknowledgments), or
    ``\\end{document}``. Returns len(full_tex) if no such boundary.
    """
    candidates = []
    for m in _A5H_SECTION_OPEN_RE.finditer(full_tex, pos=section_end_after):
        if m.start() != section_start:
            candidates.append(m.start())
            break
    for m in _A5H_SECTION_STAR_RE.finditer(full_tex, pos=section_end_after):
        candidates.append(m.start())
        break
    end_doc = _A5H_END_DOCUMENT_RE.search(full_tex, pos=section_end_after)
    if end_doc:
        candidates.append(end_doc.start())
    return min(candidates) if candidates else len(full_tex)


def _a5h_find_enclosing_section(full_tex: str, label_pos: int) -> tuple[int, int] | None:
    """Return (start, end_after) for the numbered ``\\section{...}`` that
    syntactically encloses ``label_pos`` (i.e. last section open whose
    start < label_pos). None if no numbered section precedes label_pos."""
    enclosing: re.Match[str] | None = None
    for m in _A5H_SECTION_OPEN_RE.finditer(full_tex):
        if m.start() < label_pos:
            enclosing = m
        else:
            break
    if enclosing is None:
        return None
    return (enclosing.start(), enclosing.end())


def _inject_orphan_ref_stub_global(full_tex: str, orphans: list[str]) -> str:
    """Full-paper variant of :func:`_inject_orphan_ref_stub`, invoked by
    ReviewAgent.run() Step 3b after ``_run_revision_loop`` has returned.

    For each still-orphan label (``fig:X`` / ``tab:X`` whose ``\\label``
    exists in ``full_tex`` but has no ``\\ref`` / ``\\autoref`` /
    ``\\cref`` / ``\\Cref`` / ``\\eqref``), locate the enclosing numbered
    section via ``\\section{...}`` scanning (``\\section*{...}`` is
    treated as a hard boundary, not a container, so Acknowledgments /
    References never receive stubs) and append a stub-ref sentence just
    before the next hard boundary of that section.

    Labels found inside the abstract or before the first ``\\section{...}``
    fall back to the Experiments section (or, failing that, the position
    immediately before ``\\end{document}``).

    Idempotent: a label whose ``\\ref`` already appears anywhere in the
    paper is left unchanged.
    """
    if not orphans:
        return full_tex

    abstract_end_m = _A5H_ABSTRACT_END_RE.search(full_tex)
    abstract_end_pos = abstract_end_m.end() if abstract_end_m else 0

    # Pre-compute the fallback target: Experiments section bounds.
    experiments_label_m = re.search(
        r"\\label\{sec:experiments\}", full_tex
    )
    fallback_bounds: tuple[int, int] | None = None
    if experiments_label_m:
        fallback_bounds = _a5h_find_enclosing_section(
            full_tex, experiments_label_m.start()
        )

    insertions: list[tuple[int, str]] = []
    for label in orphans:
        ref_re = re.compile(_A5H_REF_RE_TMPL.format(re.escape(label)))
        if ref_re.search(full_tex):
            # A \ref already exists — not orphan, skip.
            continue
        label_re = re.compile(_A5H_LABEL_RE_TMPL.format(re.escape(label)))
        label_match = label_re.search(full_tex)
        if not label_match:
            # Label not in tex at all — nothing to stub.
            continue
        label_pos = label_match.start()

        # Labels inside abstract or before first \section -> route to
        # Experiments fallback.
        if label_pos < abstract_end_pos:
            bounds = fallback_bounds
        else:
            bounds = _a5h_find_enclosing_section(full_tex, label_pos)
            if bounds is None:
                bounds = fallback_bounds

        if bounds is None:
            # No Experiments section either — append before \end{document}
            # or at EOF.
            end_doc = _A5H_END_DOCUMENT_RE.search(full_tex)
            insert_pos = end_doc.start() if end_doc else len(full_tex)
        else:
            sec_start, sec_end_after = bounds
            insert_pos = _a5h_find_section_end(full_tex, sec_start, sec_end_after)

        sentence = _a5h_render_stub_sentence(label)
        if not sentence:
            continue
        stub_block = (
            f"\n\n{_A5H_AUTOFIX_MARKER}\n{sentence}\n"
        )
        insertions.append((insert_pos, stub_block))

    if not insertions:
        return full_tex

    # Apply insertions in descending position so earlier offsets stay
    # valid for later insertions at smaller positions.
    out = full_tex
    for pos, stub in sorted(insertions, key=lambda t: -t[0]):
        out = out[:pos] + stub + out[pos:]
    return out


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
        self.report_substep("Building grounding packet...")

        # Self-evolution: pull adaptive context for the writing task type
        # before building the grounding packet.  See §6.13 / §1.5.
        self._adaptive_context = self.build_adaptive_context(
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

        # Step 0b: Build cite key mapping from papers
        papers = ideation.get("papers", [])
        cite_keys = self._build_cite_keys(papers)
        bibtex = self._build_bibtex(papers, cite_keys)

        # Build per-section context primitives (P0-A)
        core_ctx = self._build_core_context(ideation, blueprint, cite_keys)

        # Title & abstract need a broad context
        title_abstract_ctx = self._ctx_introduction(core_ctx, grounding=grounding)

        # Step 1: Generate title
        title = await self._generate_title(title_abstract_ctx)
        self.log(f"Title: {title}")

        # Step 2: Generate abstract
        abstract = await self._generate_abstract(title_abstract_ctx, grounding)
        self.log("Abstract generated")

        # Step 3: Build figures & table data from blueprint
        figure_blocks = self._build_figure_blocks(blueprint, figure_output)

        # Step 4: Generate each section independently, embed figures inline
        placed_figures: set[str] = set()

        # P0-B: Contribution contract
        contribution_contract: ContributionContract | None = None
        method_name = (blueprint.get("proposed_method") or {}).get("name", "")

        sections = []
        prior_sections_summary: list[str] = []

        # Classify sections into parallelizable phases
        # Phase 1 (serial): Introduction — needed for contribution contract
        # Phase 2 (parallel): Related Work + Method — independent, only need Intro summary
        # Phase 3 (serial): Experiments — needs Method context
        # Phase 4 (serial): Conclusion — needs all prior context
        _PARALLEL_LABELS = {"sec:related", "sec:method"}

        # Separate intro, parallelizable, and sequential sections
        intro_specs = []
        parallel_specs = []
        sequential_specs = []
        for spec in section_list:
            heading, label, section_instructions, fig_keys = spec
            if label == "sec:intro":
                intro_specs.append(spec)
            elif label in _PARALLEL_LABELS:
                parallel_specs.append(spec)
            else:
                sequential_specs.append(spec)

        # Helper: generate a single section (extracted from loop body)
        async def _gen_section(spec, prior_summaries, placed, existing_sections=None):
            heading, label, section_instructions, fig_keys = spec
            return await self._generate_one_section(
                spec, is_survey, inputs, core_ctx, grounding,
                experiment_results, experiment_status,
                experiment_analysis, experiment_summary,
                contribution_contract, method_name,
                figure_blocks, prior_summaries, placed,
                existing_sections=existing_sections,
            )

        # Phase 1: Introduction (serial)
        for spec in intro_specs:
            section, new_placed = await _gen_section(spec, prior_sections_summary, placed_figures)
            sections.append(section)
            placed_figures = new_placed
            snippet = section.content[:200].replace("\n", " ").strip()
            prior_sections_summary.append(f"[{section.heading}]: {snippet}...")
            # Extract contribution contract after Intro
            if not is_survey and section.label == "sec:intro" and not contribution_contract:
                contribution_contract = self._extract_contribution_contract(section.content, method_name)
                if contribution_contract.claims:
                    self.log(
                        f"Contribution contract: {len(contribution_contract.claims)} claims "
                        f"({', '.join(c.claim_type for c in contribution_contract.claims)})"
                    )

        # Phase 2: Related Work + Method in parallel
        if len(parallel_specs) >= 2:
            self.log("Generating Related Work + Method in parallel")
            frozen_summary = list(prior_sections_summary)  # snapshot for both
            frozen_placed = set(placed_figures)

            async def _gen_parallel(spec):
                return await _gen_section(spec, frozen_summary, set(frozen_placed))

            par_results = await asyncio.gather(
                *[_gen_parallel(spec) for spec in parallel_specs]
            )
            for section, new_placed in par_results:
                sections.append(section)
                placed_figures |= new_placed
                snippet = section.content[:200].replace("\n", " ").strip()
                prior_sections_summary.append(f"[{section.heading}]: {snippet}...")
        else:
            for spec in parallel_specs:
                section, new_placed = await _gen_section(spec, prior_sections_summary, placed_figures)
                sections.append(section)
                placed_figures = new_placed
                snippet = section.content[:200].replace("\n", " ").strip()
                prior_sections_summary.append(f"[{section.heading}]: {snippet}...")

        # Phase 3+4: Experiments, Conclusion (serial — pass existing sections for context)
        for spec in sequential_specs:
            section, new_placed = await _gen_section(
                spec, prior_sections_summary, placed_figures, existing_sections=sections,
            )
            sections.append(section)
            placed_figures = new_placed
            snippet = section.content[:200].replace("\n", " ").strip()
            prior_sections_summary.append(f"[{section.heading}]: {snippet}...")

        # Fallback: distribute remaining figures
        remaining = [k for k in figure_blocks if k not in placed_figures]
        if remaining:
            self.log(f"Fallback placement for {len(remaining)} unplaced figures: {remaining}")
            # Day 4 S4: fallback target_label now comes from the same
            # :data:`grounding_tables.SECTION_HINTS` table that sources
            # the `% nano:expected_section=` comment injected into the
            # figure block itself, so the three-way S4 check sees
            # expected == placement by construction for the fallback path.
            for fk in remaining:
                target_label = infer_expected_section(fk)
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

        # A-5f/global: final safety net. Fallback placement (L274-L317
        # above) and force-injection can append figure blocks to a section
        # AFTER _gen_section has returned — which means the pre/post
        # section-level orphan checks inside _gen_section never see these
        # late injections. This loop re-scans every finalized section and
        # applies the fallback stub for any surviving orphan. Day 2
        # Experiments fig:main_results escape was caused by exactly this
        # gap — LLM wasn't in the loop here, so only fallback stub can fix.
        for sec in sections:
            global_orphans = _check_section_orphans(sec.content)
            if not global_orphans:
                continue
            self.log(
                f"  [A-5f/global] {sec.label} post-assembly orphans: "
                f"{global_orphans} — applying fallback stub"
            )
            sec.content = _inject_orphan_ref_stub(sec.content, global_orphans)
            still = _check_section_orphans(sec.content)
            if still:
                logger.warning(
                    "[A-5f/global] fallback failed to clear %s orphans %s",
                    sec.label, still,
                )
            else:
                self.log(f"  [A-5f/global] {sec.label} cleared via stub")

        # A-5g/global: whole-paper dangling ref detection (H2 front-load),
        # counterpart of A-5f/global's orphan scan.  Architecture is
        # deliberately global-heavy:
        #   * [A-5g/pre] has NO code — the CHECKLIST #4 prompt defense is
        #     the only pre-stage line of defense.  Dangling detection
        #     requires the full global labels set, which is not yet
        #     assembled at that point; running a section-local approximation
        #     would produce large numbers of false positives on legitimate
        #     cross-section refs.
        #   * [A-5g/post] has NO code — A-5f/post's ``_inject_orphan_ref_stub``
        #     only refs labels that just entered this section during helper
        #     injection, so it is guaranteed not to produce new danglings.
        #   * [A-5g/global] owns the work: collect ``\label{(?:fig|tab):*}``
        #     across every finalized section, then scan each section for
        #     ``refs - global_labels``.
        # Fallback is a two-tier C + limited-A:
        #   * limited-A rewrites strict-form occurrences matched by
        #     :data:`_A5G_STRICT_RE` (``Figure~\ref{fig:X}`` /
        #     ``Table~\Cref{tab:Y}`` etc.) to grammatically-safe bare nouns
        #     (``the figure`` / ``the table``).
        #   * All other dangling refs fall through to logger.warning and
        #     surface via the A<->B layout-diagnosis channel as ``[layout/H2]``
        #     entries in ``review_output.json``.  We deliberately do NOT
        #     force-delete bare ``\ref`` occurrences — mangling prose in
        #     unpredictable contexts is a larger regression risk than
        #     leaving a ``??`` that A-5d's review tier already flags.
        global_labels: set[str] = set()
        for sec in sections:
            global_labels.update(_A5F_LABEL_PATTERN.findall(sec.content))

        for sec in sections:
            _, danglings = _check_float_consistency(
                sec.content, global_labels=global_labels
            )
            if not danglings:
                continue
            dangling_set = set(danglings)
            self.log(
                f"  [A-5g/global] {sec.label} dangling refs: "
                f"{sorted(dangling_set)}"
            )
            rewrites_applied: list[str] = []

            def _maybe_rewrite(m: re.Match) -> str:
                label = m.group(1)
                if label not in dangling_set:
                    return m.group(0)
                kind = label.split(":", 1)[0]
                rewrites_applied.append(label)
                return "the figure" if kind == "fig" else "the table"

            new_content = _A5G_STRICT_RE.sub(_maybe_rewrite, sec.content)
            if rewrites_applied:
                self.log(
                    f"  [A-5g/global] {sec.label} limited-A rewrote "
                    f"{len(rewrites_applied)} strict-form occurrence(s): "
                    f"{rewrites_applied}"
                )
                sec.content = new_content
            _, residual = _check_float_consistency(
                sec.content, global_labels=global_labels
            )
            for lbl in sorted(set(residual)):
                logger.warning(
                    "[A-5g/global] %s dangling ref left for [layout/H2]: %s",
                    sec.label, lbl,
                )

        # Step 5: Build skeleton
        skeleton = PaperSkeleton(
            title=title, authors=authors, abstract=abstract,
            sections=sections, figures=[],
            template_format=template_format, references_bibtex=bibtex,
        )

        # Step 6: Render LaTeX + sanitize
        self.report_substep("Rendering LaTeX...")
        latex_content = self._render_latex(skeleton)
        latex_content = self._sanitize_latex(latex_content)

        # Step 6b-pre: Full-document figure dedup
        latex_content = self._dedup_full_doc_figures(latex_content)

        # Step 6b: Final LaTeX-level figure validation
        latex_content = self._validate_figures_in_latex(latex_content, figure_output)

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

        # Step 6d.5: Cleanup unused BibTeX entries
        bibtex = self._cleanup_unused_bibtex(latex_content, bibtex)

        # Step 6e: Global consistency check
        consistency_issues = _check_global_consistency(latex_content, abstract, sections)
        if consistency_issues:
            self.log(f"Consistency check: {len(consistency_issues)} issue(s) found")
            for issue in consistency_issues:
                self.log(f"  - {issue}")

        # Step 6f: P1-A pre-compile LaTeX sanitiser (5-class LLM artifact cleanup)
        from nanoresearch.latex.fixer import validate_and_fix_latex
        latex_content = validate_and_fix_latex(latex_content, log_fn=self.log)

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
        self.report_substep("Compiling PDF...")
        pdf_result = await self._compile_pdf(tex_path, template_format=template_format)

        result = {
            "tex_path": str(tex_path),
            "bib_path": str(bib_path),
            "grounding": grounding.to_output_dict(),
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

        # Self-evolution: capture writing completion state + a writing trace
        # so future writing runs can recall what worked.  See §6.13 / §1.5.
        topic_name = ideation.get("topic", "unknown topic")
        pdf_ready = "yes" if "pdf_path" in result else "no"
        self.remember_context(
            MemoryType.PROJECT_CONTEXT,
            f"Writing completed for {topic_name} in mode {paper_mode_str} "
            f"with template {template_format}. PDF={pdf_ready}.",
            importance=0.7,
            tags=[ideation.get("topic", ""), paper_mode_str, "writing", template_format],
            source="writing_output",
            topic=ideation.get("topic", ""),
        )
        writing_trace = (
            f"Writing completed for {topic_name}: paper_mode={paper_mode_str}; "
            f"template={template_format}; pdf_ready={pdf_ready}; "
            f"consistency_issues={len(consistency_issues)}; "
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

    async def _generate_one_section(
        self,
        spec: tuple,
        is_survey: bool,
        inputs: dict,
        core_ctx: dict,
        grounding,
        experiment_results: dict,
        experiment_status: str,
        experiment_analysis: dict,
        experiment_summary: str,
        contribution_contract,
        method_name: str,
        figure_blocks: dict[str, str],
        prior_sections_summary: list[str],
        placed_figures: set[str],
        existing_sections: list | None = None,
    ) -> tuple[Section, set[str]]:
        """Generate a single section. Returns (Section, updated placed_figures)."""
        heading, label, section_instructions, fig_keys = spec
        placed_figures = set(placed_figures)  # local copy

        if is_survey:
            instructions = SURVEY_SECTION_PROMPTS.get(label, section_instructions)
            ctx_experiment_results = None
            ctx_experiment_status = "pending"
            ctx_experiment_analysis = None
            ctx_experiment_summary = ""
        else:
            instructions = section_instructions
            ctx_experiment_results = experiment_results
            ctx_experiment_status = experiment_status
            ctx_experiment_analysis = experiment_analysis
            ctx_experiment_summary = experiment_summary

        self.log(f"Writing section: {heading}")
        self.report_substep(f"Writing: {heading}")

        # Build prior_sections context from existing sections (for serial phases)
        _prior_content = {}
        if existing_sections:
            _prior_content = {s.heading: s.content for s in existing_sections}
        section_ctx = self._build_section_context(
            label, core_ctx, grounding=grounding,
            experiment_results=ctx_experiment_results,
            experiment_status=ctx_experiment_status,
            experiment_analysis=ctx_experiment_analysis,
            experiment_summary=ctx_experiment_summary,
            prior_sections=_prior_content,
        )

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
            if grounding.main_table_latex:
                if grounding.has_real_results:
                    header = "=== PRE-BUILT MAIN RESULTS TABLE (use this EXACTLY, do NOT rebuild) ==="
                else:
                    header = (
                        "=== SCAFFOLD MAIN RESULTS TABLE ===\n"
                        "Use this table structure. Fill baseline cells with numbers from "
                        "their original papers (cite sources). Keep proposed method cells as '--'."
                    )
                table_parts.append(
                    header + "\n" + grounding.main_table_latex + "\n=== END PRE-BUILT TABLE ==="
                )
            if grounding.ablation_table_latex:
                if grounding.has_real_results:
                    header = "=== PRE-BUILT ABLATION TABLE (use this EXACTLY, do NOT rebuild) ==="
                else:
                    header = (
                        "=== SCAFFOLD ABLATION TABLE ===\n"
                        "Use this table structure. Keep all cells as '--' since no "
                        "ablation data is available."
                    )
                table_parts.append(
                    header + "\n" + grounding.ablation_table_latex + "\n=== END PRE-BUILT TABLE ==="
                )
            if table_parts:
                table_injection = "\n\n" + "\n\n".join(table_parts)

        conclusion_binding = ""
        if label == "sec:conclusion":
            if is_survey:
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
                metric_strs = [f"{k}={v}" for k, v in list(grounding.final_metrics.items())[:5]]
                conclusion_binding = (
                    "\n\n=== CONCLUSION RESULT BINDING ===\n"
                    f"Real metrics to reference: {', '.join(metric_strs)}\n"
                    "Mention key results quantitatively when summarizing contributions. "
                    "Use the exact numbers above.\n"
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
            "=== ORPHAN-FREE CHECKLIST (MANDATORY, SELF-VERIFY BEFORE EMITTING) ===\n"
            "Before you finish this section, run this checklist on your own draft:\n"
            "  1. List every `\\label{fig:X}` and `\\label{tab:Y}` you included.\n"
            "  2. List every `\\ref{fig:*}` / `\\Cref{fig:*}` / `Table~\\ref{tab:*}` in your prose.\n"
            "  3. Confirm each label in step 1 has AT LEAST ONE matching ref in step 2,\n"
            "     within THIS SAME SECTION (not a later section).\n"
            "  4. Confirm every ref in step 2 resolves to a label that EITHER appears\n"
            "     in the AVAILABLE FIGURES list below (defined by another section) OR\n"
            "     in a `\\label{...}` you wrote in THIS section.\n"
            "If any label has no matching ref, rewrite the prose so every float is cited\n"
            "at least once BEFORE emitting. Do not leave orphans for downstream fix-up.\n"
            "If any ref resolves to NEITHER source above, prefer REWORDING the sentence\n"
            "to keep any data or claim intact; only delete the ref if the sentence has\n"
            "no standalone value without it — a ref with no label renders as `??` in\n"
            "the PDF.\n\n"
            "Positive example:\n"
            "  > As shown in Figure~\\ref{fig:overview}, our pipeline ...\n"
            "  > \\begin{figure}... \\label{fig:overview} \\end{figure}\n"
            "Negative example 1 — orphan (DO NOT produce this):\n"
            "  > \\begin{figure}... \\label{fig:overview} \\end{figure}\n"
            "  > [no \\ref{fig:overview} anywhere in this section]  <-- orphan, rejected\n"
            "Negative example 2 — dangling ref (DO NOT produce this):\n"
            "  > As reported in Table~\\ref{tab:perf_breakdown}, accuracy improves...\n"
            "  > [tab:perf_breakdown not in AVAILABLE FIGURES, not labeled here]\n"
            "  >   <-- dangling ref, renders as ?? in PDF, rejected\n"
            "=== END CHECKLIST ===\n\n"
            f"=== AVAILABLE FIGURES — YOU MUST CITE EACH ONE WITH \\ref{{fig:NAME}} ===\n"
            f"{fig_list_text}\n"
            f"{placed_note}"
            "Requirement: every figure listed above MUST appear in your prose as "
            "\\ref{fig:NAME} at least once. An uncited figure becomes an orphan "
            "figure in the final PDF, which reviewers explicitly flag as a defect.\n"
            f"=== END FIGURES ==="
            f"{table_injection}"
            f"{conclusion_binding}"
            "\n\n=== LATEX CROSS-REFERENCE RULE ===\n"
            "Do NOT use \\ref{sec:...} to reference subsections. "
            "Use natural language instead (e.g., 'in the ablation study' rather than "
            "'in Section~\\ref{sec:ablation}'). "
            "Only use \\ref{fig:...} for figures and \\ref{tab:...} for tables.\n"
            "IMPORTANT: the same orphan-prevention rule applies to TABLES. "
            "Every table you include (via \\begin{table}...\\label{tab:NAME}) MUST "
            "also be cited at least once in your prose as Table~\\ref{tab:NAME}. "
            "An uncited table is also an orphan and must be avoided.\n"
            "=== END RULE ==="
        )

        # P0-2: If the QUALITY gate sent us back via PIVOT, inject the
        # previous review's feedback so the LLM regenerates this section
        # informed by the specific issues that were flagged. The
        # orchestrator surfaces this via inputs["_gate_feedback"].
        gate_block = self._format_gate_feedback(inputs.get("_gate_feedback"))
        if gate_block:
            context_with_figs = context_with_figs + "\n\n" + gate_block

        # P0-3: high-leverage sections (Method, Experiments) go through the
        # self-convergence wrapper (LLM coverage_score >= 8 OR max 2 rounds,
        # gap-driven refinement). Other sections fall through to the plain
        # path because the extra LLM call isn't worth it for short / boilerplate
        # sections like Conclusion.
        content = await self._generate_section_with_convergence(
            context_with_figs, heading, instructions, prior_sections_summary
        )

        # A-5f pre-injection stage: writing-source self-check + single retry.
        # Detects \label{fig|tab:X} without a matching \ref that the LLM left in
        # its own draft, and reruns once with targeted pivot feedback.
        # Day 1 pipeline run revealed this stage alone is insufficient — helper
        # functions below (_verify_and_inject_tables / _place_section_figures)
        # can inject new \label env AFTER this check, creating orphans the pre
        # stage never sees. Post-injection stage (after figure placement) picks
        # those up via fallback stub.
        pre_orphans = _check_section_orphans(content)
        if pre_orphans:
            self.log(f"  [A-5f/pre] {label} orphans: {pre_orphans} — retrying once")
            retry_context = context_with_figs + _build_orphan_retry_feedback(pre_orphans)
            content = await self._generate_section_with_convergence(
                retry_context, heading, instructions, prior_sections_summary
            )
            remaining = _check_section_orphans(content)
            if remaining:
                logger.warning(
                    "[A-5f/pre] %s still orphan after retry: %s — will retry via post stage",
                    label, remaining,
                )
            else:
                self.log(f"  [A-5f/pre] {label} orphan retry cleared")

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

        # Smart figure placement
        content, placed_figures = self._place_section_figures(
            content, label, heading, figure_blocks, placed_figures,
        )

        # A-5f post-injection stage: catch \label env introduced by
        # _verify_and_inject_tables (e.g. tab:ablation when LLM cited it in
        # prose wording but omitted the table body) or _place_section_figures
        # (auto-placed figures the LLM didn't reference). Re-running the LLM
        # here would discard those injections, so we synthesize a minimal
        # \ref stub sentence instead via _inject_orphan_ref_stub.
        post_orphans = _check_section_orphans(content)
        if post_orphans:
            self.log(
                f"  [A-5f/post] {label} helper-injected orphans: {post_orphans} "
                "— applying fallback stub"
            )
            content = _inject_orphan_ref_stub(content, post_orphans)
            still = _check_section_orphans(content)
            if still:
                logger.warning(
                    "[A-5f/post] fallback failed to clear %s orphans %s",
                    label, still,
                )
            else:
                self.log(f"  [A-5f/post] {label} cleared via stub")

        # A-5g/post: DELIBERATELY NO CODE.  ``_inject_orphan_ref_stub`` above
        # only emits ``\ref{X}`` when ``X`` is already a section-local label
        # (that is how it enters the orphan set in the first place), so the
        # post stage cannot produce a dangling ref source.  Dangling detection
        # also requires the full global labels set, which is not yet
        # assembled at this per-section scope — the whole-paper scan happens
        # after ``_dedup_section_figures`` returns, where ``[A-5g/global]``
        # owns the detection + limited-A rewrite + fall-through warning path.

        return Section(heading=heading, label=label, content=content), placed_figures

    def _place_section_figures(
        self,
        content: str,
        label: str,
        heading: str,
        figure_blocks: dict[str, str],
        placed_figures: set[str],
    ) -> tuple[str, set[str]]:
        """Smart figure placement for a section. Returns (content, placed_figures)."""
        _arch_kws = ("overview", "framework", "pipeline", "architecture", "model")
        _intro_kws = ("qualitative", "example", "motivation", "task",
                       "illustration", "counterfactual", "demo", "teaser")

        if label == "sec:intro":
            intro_keywords = _intro_kws + ("intuition", "sample")
            for fk in list(figure_blocks.keys()):
                if fk in placed_figures:
                    continue
                if any(kw in fk for kw in intro_keywords):
                    content += "\n\n" + figure_blocks[fk]
                    placed_figures.add(fk)
                    break

        if label == "sec:method":
            arch_keywords = _arch_kws
            for fk in list(figure_blocks.keys()):
                if fk not in placed_figures:
                    continue
                if not any(kw in fk for kw in arch_keywords):
                    continue
                fig_pattern = re.compile(
                    r'\n*\\begin\{figure\*?\}.*?\\label\{fig:'
                    + re.escape(fk)
                    + r'\}.*?\\end\{figure\*?\}\n*',
                    re.DOTALL,
                )
                match = fig_pattern.search(content)
                if match and match.start() > 200:
                    content = content[:match.start()] + content[match.end():]
                    content = figure_blocks[fk] + "\n\n" + content.lstrip("\n")
                    self.log(f"  Moved LLM-placed fig:{fk} to top of Method")
                break

            for fk in list(figure_blocks.keys()):
                if fk in placed_figures:
                    continue
                if any(kw in fk for kw in arch_keywords):
                    content = figure_blocks[fk] + "\n\n" + content
                    placed_figures.add(fk)
                    break

        # Insert remaining figures near their \ref
        for fk, blk in figure_blocks.items():
            if fk in placed_figures:
                continue
            if label != "sec:method" and any(kw in fk for kw in _arch_kws):
                continue
            if label != "sec:intro" and any(kw in fk for kw in _intro_kws):
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
