"""LaTeX assembly: rendering, compilation, sanitization, figure handling."""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
from nanoresearch.latex import fixer as latex_fixer
from nanoresearch.schemas.paper import PaperSkeleton
from . import _escape_latex_text
from ._latex_helpers import (
    _sanitize_prose_line,
    _update_environment_stack,
    _strip_llm_thinking,
)

MAX_LATEX_FIX_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Sub-mixin imports
# ---------------------------------------------------------------------------
from .latex_figure_placement import _LaTeXFigurePlacementMixin
from .latex_bib_figures import _LaTeXBibFiguresMixin


class _LaTeXAssemblerMixin(
    _LaTeXFigurePlacementMixin,
    _LaTeXBibFiguresMixin,
):
    """Mixin -- LaTeX rendering, compilation, and sanitization.

    Inherits figure-placement methods from ``_LaTeXFigurePlacementMixin``
    and BibTeX/figure-validation methods from ``_LaTeXBibFiguresMixin``.
    """

    def _render_latex(self, skeleton: PaperSkeleton) -> str:
        """Render the paper skeleton to LaTeX string."""
        try:
            from mcp_server.tools.latex_gen import generate_full_paper
            data = skeleton.model_dump(mode="json")
            return generate_full_paper(data, skeleton.template_format)
        except ImportError:
            logger.debug("latex_gen module not available, using fallback")
            return self._fallback_latex(skeleton)
        except Exception as exc:
            logger.warning("LaTeX rendering failed, using fallback: %s", exc)
            return self._fallback_latex(skeleton)

    def _fallback_latex(self, skeleton: PaperSkeleton) -> str:
        """Generate LaTeX without templates as a fallback.

        Uses the NeurIPS 2025 document class by default.  The neurips_2025.sty
        file is copied into the compilation directory by ``_copy_style_files``.
        """
        lines = [
            r"\documentclass{article}",
            "",
            r"%% ---- NeurIPS 2025 Style ----",
            r"\usepackage[preprint]{neurips_2025}",
            "",
            r"%% ---- Standard Packages ----",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{booktabs}",
            r"\usepackage{xcolor}",
            r"\usepackage{float}",
            r"\usepackage{placeins}",  # allow text/table/figure interleaving across section boundaries
            r"\usepackage{multirow}",  # for multi-row table cells
            r"\graphicspath{{figures/}}",
            "",
            f"\\title{{{skeleton.title}}}",
            "\\author{{{}}}".format(" \\and ".join(skeleton.authors)),
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\begin{abstract}",
            _strip_llm_thinking(skeleton.abstract),
            r"\end{abstract}",
            "",
        ]

        for section in skeleton.sections:
            lines.append(f"\\section{{{section.heading}}}")
            if section.label:
                lines.append(f"\\label{{{section.label}}}")
            lines.append(_strip_llm_thinking(section.content))
            lines.append("")
            for sub in section.subsections:
                lines.append(f"\\subsection{{{sub.heading}}}")
                if sub.label:
                    lines.append(f"\\label{{{sub.label}}}")
                lines.append(_strip_llm_thinking(sub.content))
                lines.append("")

        lines.extend([
            r"\bibliographystyle{plainnat}",
            r"\bibliography{references}",
            "",
            r"\end{document}",
        ])
        return "\n".join(lines)

    async def _compile_pdf(
        self,
        tex_path,
        max_fix_attempts: int = MAX_LATEX_FIX_ATTEMPTS,
        template_format: str = "neurips",
    ) -> dict:
        """Compile LaTeX to PDF with automatic error-fix loop.

        If compilation fails, feed the error back to the LLM, apply the fix,
        and retry up to *max_fix_attempts* times.

        Safety features (OpenClaw-inspired):
        - Backs up original tex before fix loop; restores on total failure
        - Post-write verification: re-reads file to confirm write succeeded
        """
        import shutil

        self._copy_figures_to_drafts()
        self._copy_style_files(template_format)

        try:
            from mcp_server.tools.pdf_compile import compile_pdf
        except ImportError as exc:
            logger.warning("Cannot import pdf_compile: %s", exc)
            return {"error": f"PDF compiler module not available: {exc}"}

        tex_path = Path(tex_path)

        # Backup original tex before any fix attempts
        backup_path = tex_path.with_suffix('.tex.bak')
        try:
            shutil.copy2(tex_path, backup_path)
        except OSError:
            pass  # non-fatal

        result: dict = {}
        for attempt in range(max_fix_attempts + 1):
            result = await compile_pdf(str(tex_path))

            if "pdf_path" in result:
                if attempt > 0:
                    self.log(f"PDF compiled successfully after {attempt} fix(es)")
                return result

            error_msg = result.get("error", "Unknown compilation error")

            # Don't retry if the problem isn't fixable via LaTeX edits
            if "No LaTeX compiler found" in error_msg or "not found" in error_msg.lower():
                self.log("No LaTeX compiler available, skipping fix loop")
                return result

            if attempt >= max_fix_attempts:
                self.log(f"PDF compilation failed after {max_fix_attempts} fix attempts")
                return result

            # Ask LLM to fix the LaTeX
            self.log(f"PDF compilation failed (attempt {attempt + 1}), asking LLM to fix...")
            self.save_log(
                f"latex_compile_error_{attempt}.log", error_msg
            )

            try:
                current_tex = tex_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Cannot read tex file for fixing: %s", exc)
                return result

            fixed_tex = await self._fix_latex_errors(current_tex, error_msg)

            if fixed_tex and fixed_tex != current_tex:
                # Sanitize again after the LLM fix
                fixed_tex = self._sanitize_latex(fixed_tex)
                try:
                    tex_path.write_text(fixed_tex, encoding="utf-8")
                except OSError as exc:
                    logger.error("Cannot write fixed tex file: %s", exc)
                    return result
                # Post-write verification
                try:
                    verify = tex_path.read_text(encoding="utf-8")
                    if verify != fixed_tex:
                        self.log("  WARNING: post-write verification failed, reverting")
                        tex_path.write_text(current_tex, encoding="utf-8")
                        return result
                except OSError:
                    pass
                self.log(f"  Applied LLM fix (attempt {attempt + 1})")
            else:
                self.log("  LLM returned no changes, aborting fix loop")
                return result

        return result  # pragma: no cover

    async def _fix_latex_errors(self, tex_source: str, error_log: str) -> str | None:
        """Fix LaTeX compilation errors using a 2-level strategy.

        Level 1: Deterministic fixes (no LLM) -- via shared latex_fixer module.
        Level 2: Search-replace LLM fix -- LLM outputs {"old":"...","new":"..."} pairs.

        Inspired by OpenClaw's edit tool. NEVER sends full document for rewriting.
        """
        error_log = latex_fixer.truncate_error_log(error_log)

        error_lines = latex_fixer.extract_error_lines(error_log)
        error_line = error_lines[0] if error_lines else None

        tex_lines = tex_source.split('\n')
        error_lower = error_log.lower()

        # Level 1: Deterministic
        fixed = latex_fixer.deterministic_fix(
            tex_source, error_log, error_line, log_fn=self.log,
        )
        if fixed and fixed != tex_source:
            self.log("  Level 1: deterministic fix applied")
            return fixed

        targeted_hint = latex_fixer.classify_error(error_lower)

        # Level 2: Search-replace LLM fix
        result = await self._search_replace_llm_fix_writing(
            tex_source, tex_lines, error_line, error_log, targeted_hint
        )
        if result:
            return result

        self.log("  All fix levels exhausted, no fix found")
        return None

    async def _search_replace_llm_fix_writing(
        self, tex_source: str, tex_lines: list[str],
        error_line: int | None, error_log: str, targeted_hint: str,
    ) -> str | None:
        """Level 2 search-replace fix via shared latex_fixer module."""
        win_start, win_end, numbered = latex_fixer.build_error_snippet(
            tex_lines, error_line,
        )
        prompt = latex_fixer.build_search_replace_prompt(
            error_log, error_line, targeted_hint,
            win_start, win_end, numbered,
        )

        try:
            raw = (await self.generate(
                latex_fixer.SEARCH_REPLACE_SYSTEM_PROMPT, prompt,
            )) or ""
            edits = latex_fixer.parse_edit_json(raw)
            if not edits:
                self.log("  Level 2: LLM returned no valid edits")
                return None
            return latex_fixer.apply_edits(
                tex_source, edits, log_fn=self.log,
                search_window=(win_start, win_end),
            )
        except Exception as exc:
            self.log(f"  Level 2 search-replace fix failed: {exc}")
        return None

    # ---- LaTeX sanitization --------------------------------------------------

    @classmethod
    def _sanitize_latex(cls, text: str) -> str:
        """Fix common LLM output issues that break LaTeX compilation.

        Applies, in order:
        1. Unicode replacement (dashes, quotes)
        2. Percent-sign escaping
        3. Normalize float placement to flexible [htbp]/[tbp] (not [H])
        4. Auto-fix table overflow (inject \\small / \\tabcolsep / @{})
        5. Enforce max 3 contribution bullets in Introduction
        """
        # -- 0c. Truncate garbage before \documentclass --
        docclass_pos = text.find(r'\documentclass')
        if docclass_pos > 0:
            text = text[docclass_pos:]

        # -- 0d. Strip Markdown code fences --
        text = re.sub(r'```(?:latex|tex)?\s*\n', '', text)
        text = re.sub(r'\n```[ \t]*(?:\n|$)', '\n', text)

        # -- 0. Remove LLM artifact text --
        _LLM_ARTIFACT_PATTERNS = [
            r'I (?:now )?have sufficient \w+ to write.*',
            r'I have sufficient \w+.*',
            r'Let me (?:now )?(?:write|compose|draft|look up|check|verify).*',
            r'I will now (?:write|compose|draft|proceed).*',
            r'I see the paper ID.*',
            r'I (?:need|want) to (?:look up|check|find|verify|search).*',
            r'Based on (?:the|my) (?:analysis|research|review|context).*I (?:will|can|should).*',
            r'Now I (?:will|can|shall) (?:write|compose|draft).*',
            r'Here is the (?:completed?|final|written) (?:section|text|content).*:?\s*$',
            r'`[0-9a-f]{20,}`',
        ]
        for pat in _LLM_ARTIFACT_PATTERNS:
            text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # -- 0a. Remove internal retrieval/process wording from paper prose.
        # These counts are useful in logs, but read as implementation traces in the PDF.
        _TRACE_REPLACEMENTS = [
            (r'\bin the supplied related corpus,\s*', 'In the related literature, '),
            (r'In the supplied corpus,\s*evaluate', 'To our knowledge, existing related work does not directly evaluate'),
            (r'In the related literature,\s*evaluate', 'To our knowledge, existing related work does not directly evaluate'),
            (r'\bthe supplied corpus\b', 'the related literature'),
            (r'\bsupplied corpus\b', 'related literature'),
            (r'\bsupplied related corpus\b', 'related literature'),
            (r'The same corpus contains that jointly report', 'The same line of work rarely reports'),
            (r'The same report the complete', 'Prior work rarely reports the complete'),
            (r'\bretrieved context\b', 'supporting literature'),
            (r'\bavailable corpus\b', 'available literature'),
            (r'\bpipeline generated\b', 'generated'),
            (r'\bcurrent run did not produce\b', 'the experiments do not report'),
            (r'\b\d+\s*/\s*\d+\s+papers?\s+', ''),
            (r'\b\d+\s*/\s*\d+\s+papers?\b', 'the reviewed papers'),
        ]
        for pat, repl in _TRACE_REPLACEMENTS:
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)

        # Clean up partial replacements around numeric-decimal hypothesis text.
        text = re.sub(
            r'reported as unmet rather than achieved\.5 percentage points.*?while reducing',
            'reported as unmet rather than achieved, while reducing',
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r'\bexperiments tests\b', 'experiments test', text, flags=re.IGNORECASE)

        # -- 0b. Strip control characters (U+0000-U+001F except \n \r \t) --
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # -- 1. Unicode replacements --
        text = text.replace("\u2014", "---")  # em-dash
        text = text.replace("\u2013", "--")   # en-dash
        text = text.replace("\u2018", "`")    # left single quote
        text = text.replace("\u2019", "'")    # right single quote
        text = text.replace("\u201c", "``")   # left double quote
        text = text.replace("\u201d", "''")   # right double quote
        text = text.replace("\u2192", r"$\rightarrow$")
        text = text.replace("\u2190", r"$\leftarrow$")
        text = text.replace("\u2208", r"$\in$")
        text = text.replace("\u2209", r"$\notin$")
        text = text.replace("\u2264", r"$\leq$")
        text = text.replace("\u2265", r"$\geq$")
        text = text.replace("\u00d7", r"$\times$")
        text = text.replace("\u2248", r"$\approx$")
        text = text.replace("\u00b1", r"$\pm$")
        text = text.replace("\u221e", r"$\infty$")

        # Convert common Latin extended characters to LaTeX commands.
        # Conference templates with Times/T1 often drop these glyphs in author
        # names emitted by bibliography search (e.g., Polish names).
        _latex_unicode_replacements = {
            "\u00e1": r"{\'a}", "\u00c1": r"{\'A}",
            "\u00e9": r"{\'e}", "\u00c9": r"{\'E}",
            "\u00ed": r"{\'i}", "\u00cd": r"{\'I}",
            "\u00f3": r"{\'o}", "\u00d3": r"{\'O}",
            "\u00fa": r"{\'u}", "\u00da": r"{\'U}",
            "\u00e8": r"{\`e}", "\u00eb": r'{\"e}',
            "\u00ef": r'{\"i}', "\u00fc": r'{\"u}',
            "\u00dc": r'{\"U}', "\u00f6": r'{\"o}',
            "\u00d6": r'{\"O}', "\u00e4": r'{\"a}',
            "\u00c4": r'{\"A}', "\u00df": r"{\ss}",
            "\u00e7": r"{\c{c}}", "\u00c7": r"{\c{C}}",
            "\u00f1": r"{\~n}", "\u00d1": r"{\~N}",
            "\u0105": r"{\k{a}}", "\u0104": r"{\k{A}}",
            "\u0107": r"{\'c}", "\u0106": r"{\'C}",
            "\u0119": r"{\k{e}}", "\u0118": r"{\k{E}}",
            "\u0142": r"{\l}", "\u0141": r"{\L}",
            "\u0144": r"{\'n}", "\u0143": r"{\'N}",
            "\u015b": r"{\'s}", "\u015a": r"{\'S}",
            "\u017a": r"{\'z}", "\u0179": r"{\'Z}",
            "\u017c": r"{\.z}", "\u017b": r"{\.Z}",
        }
        for char, repl in _latex_unicode_replacements.items():
            text = text.replace(char, repl)

        # -- 2. Escape bare % after digits --
        lines = text.split("\n")
        fixed_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("%"):
                fixed_lines.append(line)
                continue
            if r'\url{' in line or r'\href{' in line:
                fixed_lines.append(line)
            else:
                fixed_line = re.sub(r'(?<!\\)(\d)%', r'\1\\%', line)
                fixed_lines.append(fixed_line)
        text = "\n".join(fixed_lines)

        # -- 2b. Escape prose-only special chars in lines / captions / titles --
        env_stack: list[str] = []
        sanitized_lines: list[str] = []
        for line in text.split("\n"):
            sanitized_lines.append(_sanitize_prose_line(line, env_stack))
            _update_environment_stack(line, env_stack)
        text = "\n".join(sanitized_lines)


        # -- 2b.2. Repair inline math commands that an LLM emitted in prose
        # without surrounding $...$.  This targets commands such as \mathcal,
        # \mathbf, \mathbb, and \boldsymbol while avoiding citation/ref commands.
        def _repair_inline_math_in_prose_line(line: str) -> str:
            stripped = line.lstrip()
            if not line.strip() or stripped.startswith('\\'):
                return line
            if '$' in line:
                return line
            math_cmds = (
                'mathcal', 'mathbb', 'mathbf', 'boldsymbol', 'operatorname',
                'mathrm', 'text', 'odot', 'oslash', 'in', 'subset', 'leq',
                'geq', 'prec', 'star', 'theta', 'lambda', 'tau', 'bar', 'ell',
            )
            start_re = re.compile(
                r'(\\(?:' + '|'.join(math_cmds) + r')\b|'
                r'(?<![A-Za-z])(?:[A-Z][A-Za-z]?|[a-z])\\[_^]|'
                r'(?<![A-Za-z])(?:[A-Z][A-Za-z]?|[a-z])\([^)]*\\(?:mathbf|mathcal|mathrm)|'
                r'(?<![A-Za-z])(?:[A-Z][A-Za-z]?|[a-z])\([^)]*\))'
            )
            allowed = set('\\{}_^[]=+-*/(),.|<>:;0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
            out = []
            pos = 0
            while pos < len(line):
                m = start_re.search(line, pos)
                if not m:
                    out.append(line[pos:])
                    break
                out.append(line[pos:m.start()])
                end = m.start()
                while end < len(line) and line[end] in allowed:
                    end += 1
                span = line[m.start():end]
                span = re.sub(r'\\_\{([^{}]+)\}', r'_{\1}', span)
                span = re.sub(r'\\_(?=\{|[A-Za-z0-9])', r'_', span)
                span = re.sub(r'\\\^\{\}\{([^{}]+)\}', r'^{\1}', span)
                span = re.sub(r'\^\{\}\{([^{}]+)\}', r'^{\1}', span)
                span = re.sub(r'\\\^\{\}', r'^{}', span)
                # Do not wrap plain English function calls accidentally.
                if '\\' in span or '_' in span or '^' in span or any(op in span for op in ('=', '<', '>', '|')):
                    out.append('$' + span + '$')
                else:
                    out.append(span)
                pos = end
            return ''.join(out)

        repaired_lines: list[str] = []
        in_math_env = False
        for line in text.split("\n"):
            if re.search(r'\\begin\{(?:equation|align|gather|multline)\*?\}', line):
                in_math_env = True
                repaired_lines.append(line)
                continue
            if re.search(r'\\end\{(?:equation|align|gather|multline)\*?\}', line):
                repaired_lines.append(line)
                in_math_env = False
                continue
            repaired_lines.append(line if in_math_env else _repair_inline_math_in_prose_line(line))
        text = "\n".join(repaired_lines)

        # -- 2b.5. Remove unresolved figure/table placeholders from prose.
        text = re.sub(r'\s*Figure~?\?\?[^.]*\.', '.', text)
        text = re.sub(r'\s*Table~?\?\?[^.]*\.', '.', text)

        # -- 2c. Fix \~{}\ref -> ~\ref --
        text = re.sub(
            r'\\~\{\}(\\(?:ref|eqref|cite[tp]?|pageref)\{)',
            r'~\1',
            text,
        )

        # -- 2c.2. Repair common set-notation omissions from LLM math.
        # Without escaped braces, LaTeX treats {0,1} as grouping and the PDF
        # renders as "0,1^d" instead of the intended binary set.
        text = re.sub(r'\\in\s*\{\s*0\s*,\s*1\s*\}', r'\\in\\{0,1\\}', text)
        text = re.sub(r'=\s*\{\s*j\s*:\s*m_j\s*=\s*1\s*\}', r'=\\{j: m_j=1\\}', text)
        text = re.sub(r'=\s*\{\s*j\s*:\s*([^{}]+?)\s*\}', r'=\\{j: \1\\}', text)

        # -- 2c.5. Compact common long binary-loss equations.
        # LLMs often emit full probability calls inside cross-entropy terms,
        # which overflows narrow conference columns. Keep the math equivalent
        # while using shorter symbols before LaTeX compilation.
        def _compact_long_equation(match: re.Match) -> str:
            block = match.group(0)
            if len(block) < 420:
                return block
            if not (r'\log p_{\boldsymbol{\theta}}' in block or r'\log\left(1-p_{\boldsymbol{\theta}}' in block):
                return block
            compact = block
            compact = compact.replace(r'p_{\boldsymbol{\theta}}(y_{i}=1\mid\mathbf{x}_{i},\mathbf{m})', r'p_i')
            compact = compact.replace(r'p_{\boldsymbol{\theta}}(y_i=1\mid\mathbf{x}_i,\mathbf{m})', r'p_i')
            compact = compact.replace(r'\boldsymbol{\theta}', r'\theta')
            compact = compact.replace(r'\mathbf{w}_{\mathbf{m}}', r'\mathbf{w}_m')
            compact = compact.replace(r'\mathcal{T}_{k}', r'T_k')
            return compact

        text = re.sub(
            r'\\begin\{equation\}.*?\\end\{equation\}',
            _compact_long_equation,
            text,
            flags=re.DOTALL,
        )
        text = text.replace(r'\inT_k', r'\in \mathcal{T}_{k}')
        text = text.replace(r'|T_k|', r'|\mathcal{T}_{k}|')

        # -- 2d. Avoid forced page breaks between abstract and Introduction.
        text = re.sub(
            r'(\\end\{abstract\})\s*(?:\\(?:clearpage|newpage|pagebreak)\s*)+(\\section\*?\{Introduction\})',
            r'\1\n\n\2',
            text,
            flags=re.IGNORECASE,
        )
        # -- 3. Normalize figure placement --
        text = re.sub(
            r'\\begin\{figure\}\s*\[[Hh]!?\]',
            r'\\begin{figure}[htbp]',
            text,
        )
        text = re.sub(
            r'\\begin\{figure\}(?!\[)',
            r'\\begin{figure}[htbp]',
            text,
        )
        text = re.sub(
            r'\\begin\{figure\*\}\s*\[[Hh]!?\]',
            r'\\begin{figure*}[tbp]',
            text,
        )
        text = re.sub(
            r'\\begin\{table\}\s*\[[Hh]!?\]',
            r'\\begin{table}[htbp]',
            text,
        )
        text = re.sub(
            r'\\begin\{table\*\}\s*\[[Hh]!?\]',
            r'\\begin{table*}[tbp]',
            text,
        )

        text = re.sub(
            r'\n(?:\\FloatBarrier\s*)?\\section\{Method\}',
            lambda _m: '\n\\FloatBarrier\n\\section{Method}',
            text,
        )

        # -- 4. Auto-fix table overflow --
        text = cls._fix_table_overflow(text)

        # -- 5. Enforce contribution limit --
        text = cls._enforce_contribution_limit(text)

        # -- 6. Collapse blank lines before/after math environments --
        _math_envs = r'(?:equation|align|gather|multline|eqnarray)\*?'
        text = re.sub(
            rf'\n[ \t]*\n([ \t]*\\begin\{{{_math_envs}\}})',
            r'\n\1',
            text,
        )
        text = re.sub(
            rf'(\\end\{{{_math_envs}\}})[ \t]*\n[ \t]*\n',
            r'\1\n',
            text,
        )

        # -- 7. Extract figure blocks from inside list environments --
        text = cls._extract_figures_from_lists(text)

        # -- 8. Relocate non-architecture figures out of Introduction --
        text = cls._relocate_intro_figures(text)

        # -- 8b. Avoid forced page breaks before Conclusion; flexible floats
        # should move naturally instead of creating large blank areas.
        text = re.sub(r'\n\\clearpage\n\s*(?=\\section\*?\{Conclusion\})', '\n\n', text)
        text = re.sub(r'(?:\n\\clearpage\s*){2,}', r'\n\\clearpage\n\n', text)

        # -- 9. Fix stray \end{document} inside body --
        text = cls._fix_end_document_placement(text)

        # -- 10. Relocate figures stranded after bibliography --
        text = cls._relocate_post_bib_figures(text)

        # -- 11. Precompile layout audit: compact floats, keep figures near prose,
        # and repair common formula lead-in fragments before the first PDF build.
        text = cls._precompile_static_layout_audit(text)

        # -- 12. Final float cleanup: remove page breaks that create blank space
        # before ordinary figures or Conclusion.
        text = re.sub(r'\n\\clearpage\n\s*(?=\\begin\{figure)', '\n', text)
        text = re.sub(r'\n\\clearpage\n\s*(?=\\section\*?\{Conclusion\})', '\n\n', text)

        # -- 13. Repair malformed inline math wrappers and escaped math
        # tokens that can be introduced when prose sanitization sees LLM math.
        # Normalize all inline math delimiters to $...$ first; mixing \(...\)
        # with $...$ is the main source of invalid spans such as
        # ``$\mathcal{X}\)`` and ``I_{$\mathrm{lr}}$``.
        text = text.replace(r'\(', '$').replace(r'\)', '$')
        text = text.replace(r'\($', '$').replace(r'\)\$', '$')
        text = text.replace(r'\$', '$')
        text = text.replace(r'_{\$', r'_{')
        text = re.sub(r'\$([^$\n]*?)\\\)([,.;:]?)', r'$\1$\2', text)
        text = re.sub(r'\\\(([^$\n]*?)\$', r'$\1$', text)
        text = re.sub(r'\${2,}', '$', text)

        def _repair_split_subscript_math(tex: str) -> str:
            # Common corruption: I_{$\mathrm{lr}}$ should be one inline math span.
            tex = re.sub(
                r'(?<![A-Za-z])([A-Za-z])_\{\$([^$\n]+?)\}\$',
                lambda m: f'${m.group(1)}_{{{m.group(2)}}}$',
                tex,
            )
            tex = re.sub(
                r'(?<![A-Za-z])([A-Za-z])_\{\$([^$\n]+)\$\}',
                lambda m: f'${m.group(1)}_{{{m.group(2)}}}$',
                tex,
            )
            return tex

        text = _repair_split_subscript_math(text)

        def _clean_inline_math(match: re.Match) -> str:
            span = match.group(1)
            span = span.replace(r'\(', '').replace(r'\)', '')
            span = re.sub(r'\\_\{([^{}]+)\}', r'_{\1}', span)
            span = re.sub(r'\\_(?=\{|[A-Za-z0-9])', r'_', span)
            span = re.sub(r'\\\^\{\}\{([^{}]+)\}', r'^{\1}', span)
            span = re.sub(r'\^\{\}\{([^{}]+)\}', r'^{\1}', span)
            span = re.sub(r'\\\^\{\}', r'^{}', span)
            return '$' + span + '$'

        text = re.sub(r'\$([^$\n]+)\$', _clean_inline_math, text)
        text = _repair_split_subscript_math(text)
        text = text.replace(r'\$', '$')
        text = re.sub(r'\${2,}', '$', text)
        text = re.sub(r'\$(,|\.|;|:)\$', r'$\1', text)
        text = re.sub(r'\$([^$\n]+)\$', _clean_inline_math, text)
        text = re.sub(r'(?:\n\\clearpage\s*){2,}', r'\n\\clearpage\n\n', text)

        def _round_long_decimal(match: re.Match) -> str:
            raw = match.group(0)
            try:
                return f"{float(raw):.4f}"
            except ValueError:
                return raw

        text = re.sub(r'(?<![A-Za-z])\d+\.\d{5,}(?![A-Za-z])', _round_long_decimal, text)

        # Repair escaped math fragments introduced by prose-level LaTeX escaping.
        text = text.replace(r'$r$\in', r'$r\in')
        text = text.replace(r'$\{($\mathcal', r'$\{(\mathcal')
        text = text.replace(r'\sigma($', r'\sigma(')
        text = re.sub(r'\\sim\$\\operatorname', r'\\sim\\operatorname', text)
        text = re.sub(
            r'\$([^$\n]{1,120}?)\sim\$\\operatorname\{([^{}]+)\}\(([^)]*)\)\$',
            lambda m: f'${m.group(1)}\\sim\\operatorname{{{m.group(2)}}}({m.group(3)})$',
            text,
        )
        text = re.sub(
            r'\$([A-Za-z][^$\n]{0,80}?)=\$([^$\n]{1,160})\$',
            lambda m: f'${m.group(1)}={m.group(2)}$',
            text,
        )
        text = re.sub(
            r'\\?_\{\$\\mathrm\{([^{}]+)\}\$\}',
            lambda m: r'_{\mathrm{' + m.group(1) + '}}',
            text,
        )
        text = re.sub(
            r'\\?_\{\$([^$\n{}]+)\}',
            lambda m: '_{' + m.group(1) + '}',
            text,
        )
        text = re.sub(
            r'\\?_\{\$(\\[A-Za-z]+\{[^{}]+\})\}',
            lambda m: '_{' + m.group(1) + '}',
            text,
        )
        text = text.replace(
            r'\max_{$\mathbf{m}\in\mathcal{F}\_{G}}A\_{\mathrm{cv}}',
            r'\max_{\mathbf{m}\in\mathcal{F}_{G}}A_{\mathrm{cv}}',
        )

        def _repair_inline_math_escapes(match: re.Match) -> str:
            span = match.group(1)
            span = span.replace(r'\_', '_')
            span = span.replace(r'\^{}', '^')
            span = span.replace(r'\{', '{').replace(r'\}', '}')
            return '$' + span + '$'

        text = re.sub(r'\$([^$\n]{1,500})\$', _repair_inline_math_escapes, text)
        text = text.replace(r'\widehat{B}(\mathbf{m})', r'\widehat{B}(\mathbf{m})')
        text = text.replace(r'\textbackslash{}|$\mathbf{m}\textbackslash{}|_{0}$', r'\|\mathbf{m}\|_{0}')
        text = re.sub(r'\\in\s*\{\s*0\s*,\s*1\s*\}', r'\\in\\{0,1\\}', text)
        text = re.sub(r'=\s*\{\s*0\s*,\s*1\s*\}', r'=\\{0,1\\}', text)
        text = re.sub(r'=\s*\{\s*j\s*:\s*m_j\s*=\s*1\s*\}', r'=\\{j: m_j=1\\}', text)

        # -- 14. Normalize figure refs and remove unresolved figure references
        # if a figure was dropped. Some LLM-written references omit the fig:
        # prefix even though the label itself is correct.
        fig_labels = set(re.findall(r'\\label\{(fig:[^}]+)\}', text))
        fig_suffixes = {label[4:]: label for label in fig_labels}

        def _normalize_ref_prefix(match: re.Match) -> str:
            label = match.group(1)
            if label in fig_suffixes:
                return r'\ref{' + fig_suffixes[label] + '}'
            return match.group(0)

        text = re.sub(r'\\ref\{(?!fig:|tab:|sec:|eq:)([^}]+)\}', _normalize_ref_prefix, text)

        def _missing_fig_ref(match: re.Match) -> str:
            label = match.group(1)
            if label in fig_labels:
                return match.group(0)
            if 'accuracy_sparsity' in label or 'optimization' in label:
                return 'the optimization-history artifact'
            return 'the corresponding experiment artifact'
        text = re.sub(r'Figure~\\ref\{(fig:[^}]+)\}', _missing_fig_ref, text)


        eq_labels = set(re.findall(r'\\label\{(eq:[^}]+)\}', text))

        def _missing_eq_ref(match: re.Match) -> str:
            label = match.group(1)
            if label in eq_labels:
                return match.group(0)
            if label == 'eq:mask_space':
                return 'the binary mask space'
            return 'the corresponding definition'

        text = re.sub(r'Eq\.~\\eqref\{(eq:[^}]+)\}', _missing_eq_ref, text)

        # -- 15a. Repair mixed prose/math lines where LLMs leave math
        # commands outside $...$ after an earlier inline math span on the same line.
        text = re.sub(r'(?<!\$)\\circ(?!\$)', r'$\\circ$', text)
        text = text.replace(
            r'\sigma(a)=(1+\exp(-a))\^{}{-1}',
            r'$\sigma(a)=(1+\exp(-a))^{-1}$',
        )
        text = re.sub(r'tolerance \\epsilon=([0-9.]+)', r'tolerance $\\epsilon=\1$', text)
        text = re.sub(r'(?<!\$)\\sigma\(\\cdot\)(?!\$)', r'$\\sigma(\\cdot)$', text)
        text = re.sub(r'\\widehat\{\$([^$\n]+)\$', lambda m: r'$\widehat{' + m.group(1) + r'}$', text)
        text = text.replace(r'$\widehat{\operatorname{BA}}(\mathbf{m})}$', r'$\widehat{\operatorname{BA}}(\mathbf{m})$')
        text = text.replace(r'\textbackslash{}|\mathbf{m}\textbackslash{}|_{0}', r'\|\mathbf{m}\|_{0}')
        text = re.sub(r'(?<!\$)([A-Za-z])=\\lVert\$([^$\n]+)\\rVert_\{([^{}]+)\}\$', lambda m: f'$' + m.group(1) + r'=\lVert' + m.group(2) + r'\rVert_{' + m.group(3) + r'}$', text)
        text = re.sub(r'\${2,}', '$', text)
        text = text.replace(r'\mathcal{O}(n\textbackslash{},d\textbackslash{},T_{\mathrm{lr}})', r'\mathcal{O}(n d T_{\mathrm{lr}})')
        text = text.replace(r'\mathcal{O}(M\textbackslash{},\bar{h})', r'\mathcal{O}(M\bar{h})')

        # -- 15. Split common overlong display equations. This catches long
        # optimization and Pareto-set equations that otherwise overflow narrow
        # conference columns after LLM rewriting.
        def _replace_equation_by_label(tex: str, label: str, replacement: str) -> str:
            pattern = (
                r'\\begin\{equation\}.*?'
                + re.escape(r'\label{' + label + '}')
                + r'.*?\\end\{equation\}'
            )
            return re.sub(pattern, lambda _m: replacement, tex, count=1, flags=re.DOTALL)

        logistic_replacement = r"""\begin{equation}
\begin{aligned}
\theta_{\mathbf{m}}^{\star}=\arg\min_{\mathbf{w}_{\mathbf{m}},b_{\mathbf{m}}}\quad
& -\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right] \\
& +\frac{\lambda}{2}\|\mathbf{w}_{\mathbf{m}}\|_2^2,\qquad
p_i=p_{\theta}(y_i=1\mid\mathbf{x}_i,\mathbf{m}).
\end{aligned}
\label{eq:logistic_objective}
\end{equation}"""
        pareto_replacement = r"""\begin{equation}
\begin{aligned}
\mathcal{P}=\{\mathbf{m}\in\bigcup_{t=0}^{T}P_t:\;&\nexists\,\mathbf{m}'\text{ such that }
f_1(\mathbf{m}')\ge f_1(\mathbf{m}),\\
&f_2(\mathbf{m}')\le f_2(\mathbf{m})\text{, and one inequality is strict}\}.
\end{aligned}
\label{eq:pareto_set}
\end{equation}"""
        final_refit_replacement = r"""\begin{equation}
\begin{aligned}
\widehat{\theta}=\arg\min_{\mathbf{w}_{\widehat{\mathbf{m}}},b_{\widehat{\mathbf{m}}}}\quad
& -\frac{1}{n_{\mathrm{tr}}}\sum_{i\in\mathcal{D}_{\mathrm{tr}}}\left[y_i\log \widehat{p}_i+(1-y_i)\log(1-\widehat{p}_i)\right] \\
& +\frac{\lambda}{2}\|\mathbf{w}_{\widehat{\mathbf{m}}}\|_2^2,\qquad
\widehat{p}_i=p_{\theta}(y_i=1\mid\mathbf{x}_i,\widehat{\mathbf{m}}).
\end{aligned}
\label{eq:final_refit}
\end{equation}"""
        text = _replace_equation_by_label(text, 'eq:logistic_objective', logistic_replacement)
        text = _replace_equation_by_label(text, 'eq:pareto_set', pareto_replacement)
        pareto_dominance_replacement = r"""\begin{equation}
\begin{aligned}
\mathbf{m}^{(a)}\prec \mathbf{m}^{(b)}\Longleftrightarrow
& f_{\mathrm{acc}}(\mathbf{m}^{(a)})\geq f_{\mathrm{acc}}(\mathbf{m}^{(b)}) \\
& \wedge\; f_{\mathrm{size}}(\mathbf{m}^{(a)})\leq f_{\mathrm{size}}(\mathbf{m}^{(b)}) \\
& \wedge\; \big[ f_{\mathrm{acc}}(\mathbf{m}^{(a)})> f_{\mathrm{acc}}(\mathbf{m}^{(b)}) \\
& \qquad\vee\; f_{\mathrm{size}}(\mathbf{m}^{(a)})< f_{\mathrm{size}}(\mathbf{m}^{(b)}) \big].
\end{aligned}
\label{eq:pareto_dominance}
\end{equation}"""
        text = _replace_equation_by_label(text, 'eq:pareto_dominance', pareto_dominance_replacement)
        logistic_loss_replacement = r"""\begin{equation}
\begin{aligned}
\mathcal{L}_{\mathrm{log}}(\theta;\mathbf{m}) =
& -\frac{1}{n_k}\sum_{i\in\mathcal{I}_k}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right] \\
& +\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2,\qquad
p_i=p_{\theta}(y=1\mid\mathbf{x}_i,\mathbf{m}).
\end{aligned}
\label{eq:logistic_loss}
\end{equation}"""
        text = _replace_equation_by_label(text, 'eq:final_refit', final_refit_replacement)
        text = _replace_equation_by_label(text, 'eq:logistic_loss', logistic_loss_replacement)
        lr_loss_replacement = r"""\begin{equation}
\begin{aligned}
\mathcal{L}_{\mathrm{lr}}(\boldsymbol{\theta};\mathbf{m},\mathcal{D}_{a}) =
& -\frac{1}{|\mathcal{D}_{a}|}
\sum_{(\mathbf{x}_{i},y_{i})\in\mathcal{D}_{a}}
\Big[y_i\log p_{\boldsymbol{\theta},\mathbf{m}}(y=1\mid\mathbf{x}_i) \\
& +(1-y_i)\log\big(1-p_{\boldsymbol{\theta},\mathbf{m}}(y=1\mid\mathbf{x}_i)\big)\Big]
+\frac{\lambda}{2}\|\mathbf{w}\|_2^2.
\end{aligned}
\label{eq:lr_loss}
\end{equation}"""
        text = _replace_equation_by_label(text, 'eq:lr_loss', lr_loss_replacement)
        fold_logistic_loss_replacement = r"""\begin{equation}
\begin{aligned}
\boldsymbol{\theta}_{k,\mathbf{m}}^{\star} = \arg\min_{\boldsymbol{\theta}}\quad
& -\frac{1}{|\mathcal{I}^{\mathrm{tr}}_{k}|}
\sum_{i\in\mathcal{I}^{\mathrm{tr}}_{k}}
\Big[y_i\log p_{\boldsymbol{\theta},\mathbf{m}}(y=1\mid\mathbf{x}_i) \\
& +(1-y_i)\log\big(1-p_{\boldsymbol{\theta},\mathbf{m}}(y=1\mid\mathbf{x}_i)\big)\Big]
+\frac{\lambda}{2}\|\mathbf{w}_{\mathbf{m}}\|_2^2.
\end{aligned}
\label{eq:fold_logistic_loss}
\end{equation}"""
        text = _replace_equation_by_label(text, 'eq:fold_logistic_loss', fold_logistic_loss_replacement)
        final_training_objective_replacement = r"""\begin{equation}
\begin{aligned}
\boldsymbol{\theta}^{\star}=\arg\min_{\boldsymbol{\theta}}\quad
& -\frac{1}{n}\sum_{i=1}^{n}
\Big[y_i\log p_{\boldsymbol{\theta},\mathbf{m}^{\star}}(y=1\mid\mathbf{x}_i) \\
& +(1-y_i)\log\big(1-p_{\boldsymbol{\theta},\mathbf{m}^{\star}}(y=1\mid\mathbf{x}_i)\big)\Big]
+\frac{\lambda}{2}\|\mathbf{w}_{\mathbf{m}^{\star}}\|_2^2.
\end{aligned}
\label{eq:final_training_objective}
\end{equation}"""
        text = _replace_equation_by_label(text, 'eq:final_training_objective', final_training_objective_replacement)

        def _humanize_identifier(raw: str) -> str:
            raw = raw.replace(r'\_', '_').strip('_')
            words = [w for w in raw.split('_') if w]
            if len(words) < 3:
                return raw.replace('_', ' ')
            special = {
                'nsga2': 'NSGA-II', 'nsga': 'NSGA', 'ii': 'II', 'roc': 'ROC',
                'auc': 'AUC', 'cv': 'CV', 'lr': 'LR', 'rf': 'RF', 'sklearn': 'sklearn',
            }
            mapped = [special.get(w.lower(), w.capitalize()) for w in words]
            phrase = ' '.join(mapped)
            phrase = phrase.replace('Fixed Budget', 'Fixed-Budget')
            phrase = phrase.replace('Full Feature', 'Full-Feature')
            phrase = phrase.replace('Feature Count', 'Feature-Count')
            phrase = phrase.replace('Multi Objective', 'Multi-Objective')
            phrase = phrase.replace('Logistic Regression', 'Logistic Regression')
            return phrase

        text = re.sub(
            r'\\texttt\{((?:[A-Za-z0-9]+\\_){2,}[A-Za-z0-9]+)\}',
            lambda m: _humanize_identifier(m.group(1)),
            text,
        )
        text = re.sub(
            r'(?<![A-Za-z0-9])(?:[A-Za-z0-9]+\\_){2,}[A-Za-z0-9]+(?![A-Za-z0-9])',
            lambda m: _humanize_identifier(m.group(0)),
            text,
        )
        text = re.sub(r'\b0/\d+\s+closest disease studies use', 'none of the closest disease studies use', text)
        text = re.sub(r'\b0/\d+\s+report', 'no surveyed papers report', text)

        # Keep result floats inside the experimental narrative. Without a
        # barrier, LaTeX can defer the final result figure past Conclusion or
        # even after References, which reads as a misplaced orphan figure.
        text = re.sub(
            r'\n(?:\\FloatBarrier\s*)?\\section\{Conclusion\}',
            lambda _m: '\n\\FloatBarrier\n\\section{Conclusion}',
            text,
            count=1,
        )

        # Final paper-quality cleanup after review rewrites. These patterns are
        # intentionally late because review can reintroduce malformed inline
        # math and generic figure transition sentences after writing.
        text = re.sub(
            r'Figure~\\ref\{fig:main_results\} visualizes the same measured comparison as Table~\\ref\{tab:main_results\}, making the relative accuracy and\s+compactness pattern easier to inspect\.',
            lambda _m: r'Figure~\ref{fig:main_results} and Table~\ref{tab:main_results} should be read as one measured comparison: the table gives the exact local scores, while the figure highlights whether the proposed operating point remains competitive while using fewer inspected features.',
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r'complexity by using \d+(?:\.\d+)?\% fewer features',
            'complexity by using fewer features',
            text,
        )
        text = re.sub(
            r'\\phi\\_\{\$([^}\n]+)\}((?:\([^$\n]*?\)))([.,;:]?)\$',
            lambda m: r'$\phi_{' + m.group(1) + r'}' + m.group(2) + r'$' + m.group(3),
            text,
        )
        text = re.sub(
            r'\\phi\\_\{\$([^$\n]+?)\}\$((?:\([^$\n]*?\))?)',
            lambda m: r'$\phi_{' + m.group(1) + r'}' + m.group(2) + r'$',
            text,
        )
        text = re.sub(
            r'(?<!\$)\\sigma\(a\)=1/\(1\+\\exp\(-a\)\)(?!\$)',
            lambda _m: r'$\sigma(a)=1/(1+\exp(-a))$',
            text,
        )
        text = text.replace(r'\phi\_{$\mathbf{m}}(\mathbf{x}).$', r'$\phi_{\mathbf{m}}(\mathbf{x})$.')
        text = text.replace(r'\phi\_{$\mathbf{m}}$', r'$\phi_{\mathbf{m}}$')
        text = text.replace(r', \rho is', r', $\rho$ is')
        text = text.replace(r' \rho is', r' $\rho$ is')
        text = text.replace(r'\mathcal{O}(K\textbackslash{},c_{\mathrm{LR}}(n,d))', r'\mathcal{O}(K\,c_{\mathrm{LR}}(n,d))')
        text = text.replace(r'Eq.~\eqref{eq:fold_logistic_loss}', 'the fold-level logistic objective')
        text = text.replace(r'Eq.~\eqref{eq:survivor_selection}', 'the NSGA-II survivor-selection rule')
        text = re.sub(
            r'\\begin\{equation\}.*?\\label\{eq:pareto_dominance\}\s*\\end\{equation\}',
            lambda _m: r'''\begin{equation}
\begin{aligned}
\mathbf{m}^{(a)}\prec \mathbf{m}^{(b)}\Longleftrightarrow
& f_{\mathrm{acc}}(\mathbf{m}^{(a)})\geq f_{\mathrm{acc}}(\mathbf{m}^{(b)}) \\
& \wedge\; f_{\mathrm{size}}(\mathbf{m}^{(a)})\leq f_{\mathrm{size}}(\mathbf{m}^{(b)}) \\
& \wedge\; \big[ f_{\mathrm{acc}}(\mathbf{m}^{(a)})> f_{\mathrm{acc}}(\mathbf{m}^{(b)}) \\
& \qquad\vee\; f_{\mathrm{size}}(\mathbf{m}^{(a)})< f_{\mathrm{size}}(\mathbf{m}^{(b)}) \big].
\end{aligned}
\label{eq:pareto_dominance}
\end{equation}''',
            text,
            count=1,
            flags=re.DOTALL,
        )
        text = text.replace(
            'the corresponding definition and the corresponding definition',
            'the fitted preprocessing and prediction definitions',
        )
        text = text.replace(
            'which component helps preserve accuracy while keeping the final model inspectable?',
            'which component helps preserve accuracy while keeping the final model inspectable.',
        )
        text = re.sub(r'(?<!\?)\b([Ww])hether ([^.?!]{20,180})\?', lambda m: m.group(1) + 'hether ' + m.group(2) + '.', text)
        text = re.sub(r'\${2,}', '$', text)

        return text
