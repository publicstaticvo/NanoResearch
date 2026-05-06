---
name: nanoresearch-writing
description: Draft a LaTeX research paper from all previous stage outputs
version: 0.1.0
---

# Writing Skill

## Purpose
Take all previous outputs (ideation, planning, experiment results) and produce a complete LaTeX paper draft with figures, tables, and bibliography.

## Tools Required
- `generate_latex`: Generate and assemble LaTeX source files for each paper section
- `compile_pdf`: Compile the LaTeX source into a PDF document
- `generate_figure`: Produce publication-quality figures from experiment results

## Input
- `ideation_output`: Path to `papers/ideation_output.json` from the ideation skill
- `experiment_blueprint`: Path to `papers/experiment_blueprint.json` from the planning skill
- `experiment_results`: Path to `experiments/` directory containing code and results from the experiment skill

## Process
1. Parse all upstream outputs to gather ideas, literature, experiment design, and results
2. Generate the paper outline following a standard structure (Abstract, Introduction, Related Work, Method, Experiments, Conclusion)
3. Draft the Abstract summarizing the problem, approach, and key findings
4. Draft the Introduction motivating the research question and stating contributions
5. Draft Related Work synthesizing the surveyed literature from the ideation stage
6. Draft the Method section describing the proposed approach in detail
7. Draft the Experiments section with dataset descriptions, baseline comparisons, and ablation results
8. Generate figures (performance plots, ablation charts, architecture diagrams) using `generate_figure`
9. Generate tables summarizing quantitative results
10. Draft the Conclusion with a summary of findings and future work directions
11. Compile the bibliography from all cited papers
12. Assemble the full LaTeX document using `generate_latex`
13. Compile to PDF using `compile_pdf` and verify the output

## Optional Paper Polish Mode
When `writing_paper_polish_enabled` is enabled, the writing stage applies an additional conference-paper polish contract before final output. This mode is intended for paper-facing case studies and camera-ready drafts, not for every fast batch run.

Polish mode should:
- Keep the paper in a dense conference format, with most added substance in Method and Experiments.
- Keep Discussion compact; by default it must be at most one paragraph.
- Preserve user-specified `writing_polish_required_sentences` exactly when provided.
- Avoid placeholder or softening phrases such as TODO, placeholder, illustrative, pilot, or not fully trained.
- Compile the PDF and write `drafts/paper_polish_report.json` with page count, reference count, Discussion length, required-sentence checks, and forbidden-term checks.
- Optionally export `drafts/figure1.pdf` when `writing_polish_export_figure1_pdf` is enabled.

Polish mode does not generate a 3x3 preview by default. Preview grids are a manual debugging artifact, not a standard NanoResearch writing output.

## Output
Produces `papers/draft/` directory containing:
- `main.tex`: Complete LaTeX source of the paper
- `references.bib`: Bibliography file with all citations
- `figures/`: Generated figures in PDF or PNG format
- `tables/`: LaTeX table source files
- `main.pdf`: Compiled PDF of the paper draft
- `paper_polish_report.json`: Optional validation report when paper polish mode is enabled
- `figure1.pdf`: Optional standalone first-figure export when requested
