"""Section writing: title, abstract, section generation, tool builder."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

MAX_PAPERS_FOR_CITATIONS = 50

# Survey section prompts: maps section label -> writing instructions string.
# Used by run() to look up instructions when dispatching based on paper_mode.
SURVEY_SECTION_PROMPTS: dict[str, str] = {
    "sec:intro": (
        "Write 4-5 paragraphs:\n"
        "(1) Opening: establish the importance and scope of the topic. "
        "Cover the breath of work in this area and why it matters.\n"
        "(2) Coverage: state the number of papers reviewed, the year range, "
        "and the number of thematic clusters identified.\n"
        "(3) Taxonomy brief: briefly preview the main categories of methods/approaches "
        "covered in this survey.\n"
        "(4) Contributions: state what this survey provides (e.g., systematic review, "
        "unified taxonomy, comparative analysis, open challenges). Use a bullet list "
        "with 2-3 \\item entries.\n"
        "Use assertive language throughout. Cite key papers establishing importance."
    ),
    "sec:related": (
        "Write 5-7 paragraphs organized by THEMATIC CLUSTERS.\n"
        "For each cluster: summarize 3-5 representative papers, show the evolution of ideas, "
        "and highlight the key differences between approaches.\n"
        "Organize by theme (e.g., by architecture, by task, by training paradigm), "
        "NOT chronologically.\n"
        "IMPORTANT: Every \\cite{} or \\citet{} key MUST be from the provided CITATION KEYS list. "
        "Do NOT invent citation keys.\n"
        "MUST-CITE ENFORCEMENT: If the context includes a 'MUST-CITE PAPERS' section, "
        "you MUST cite ALL of those papers in this section.\n"
        "FAIRNESS: discuss the STRONGEST and most influential works, not just peripheral ones.\n"
        "Use \\citet{key} when author is subject, \\citep{key} for parenthetical."
    ),
    "sec:taxonomy": (
        "Write 6-8 paragraphs presenting a structured taxonomy of methods/approaches "
        "in this area.\n"
        "Structure:\n"
        "(1) Overview: describe the organizing principles of the taxonomy and list "
        "the top-level categories.\n"
        "(2-5) One \\subsection{} per major category. For each:\n"
        "  - Define the category and its distinguishing characteristics\n"
        "  - List sub-categories or variants with representative works (cite them)\n"
        "  - Explain the relationship between this category and others in the taxonomy\n"
        "  - Note strengths and weaknesses of approaches in this category\n"
        "(6) Comparative overview: a summary table or paragraph contrasting categories "
        "on key dimensions (e.g., scalability, data efficiency, generalization).\n"
        "Cite representative papers using \\citep{key}. Use the theme clusters from "
        "the context as your organizing structure."
    ),
    "sec:method": (
        "Write 6-8 paragraphs systematically analyzing the METHODOLOGICAL designs "
        "across reviewed papers.\n"
        "Structure:\n"
        "(1) Overview: frame the problem formally and describe the space of solution designs.\n"
        "(2-4) Detailed analysis of key technical components:\n"
        "  - For each major component, compare how different categories implement it\n"
        "  - Use numbered equations to illustrate key formulations where illuminating\n"
        "  - Reference equations in text: 'as defined in Eq.~\\eqref{eq:xxx}'\n"
        "  - Explain design trade-offs and why different schools of thought make different choices\n"
        "(5) Training and optimization: compare training objectives, optimizers, and "
        "regularization strategies across approaches.\n"
        "(6) Complexity and efficiency: discuss computational costs, FLOPs, or inference "
        "time comparisons where available.\n"
        "Use \\begin{align} for multi-line equations (NEVER eqnarray).\n"
        "Do NOT include \\begin{figure} blocks --- figures are inserted automatically."
    ),
    "sec:applications": (
        "Write 5-6 paragraphs covering where and how these methods have been applied.\n"
        "Structure:\n"
        "(1) Overview: summarize the range of application domains and tasks addressed "
        "by methods in this survey.\n"
        "(2-3) Domain-specific analysis: for each major application domain, "
        "discuss which methods have been applied, what datasets are used, "
        "and what performance levels are achieved (cite original papers).\n"
        "(4) Task taxonomy: classify the types of tasks addressed (e.g., classification, "
        "generation, prediction, optimization) and which methodological categories "
        "excel at each.\n"
        "(5) Emerging applications: note newly emerging or underexplored application "
        "areas identified across the literature.\n"
        "IMPORTANT: Do NOT fabricate numbers. Only cite results that appear in the "
        "provided evidence or citation data. Mark areas with insufficient data as 'limited "
        "empirical evaluation' rather than citing vague performance claims."
    ),
    "sec:challenges": (
        "Write 5-6 paragraphs analyzing the key challenges and open problems.\n"
        "Structure:\n"
        "(1) Overview: synthesize the main difficulties identified across the literature "
        "into 3-5 high-level challenge categories.\n"
        "(2-4) One \\subsection{} per major challenge category. For each:\n"
        "  - Describe the challenge in technical terms\n"
        "  - Show how different methods attempt to address it (and where they fall short)\n"
        "  - Illustrate with specific examples from reviewed papers (cite them)\n"
        "  - Discuss trade-offs between addressing this challenge and others\n"
        "(5) Cross-cutting challenges: discuss challenges that span multiple categories "
        "or application domains.\n"
        "Draw directly from the key_challenges provided in the context. "
        "Do NOT invent challenges not supported by the literature."
    ),
    "sec:systematic": (
        "Write 6-8 paragraphs providing a deep critical analysis of trends, assumptions, "
        "and methodological quality across the literature.\n"
        "Structure:\n"
        "(1) Overview: frame what this analysis covers and its goals.\n"
        "(2) Trend analysis: identify 2-3 major evolutionary trends in methodology "
        "(e.g., how problem formulation, model architecture, or evaluation practice "
        "has changed over time).\n"
        "(3) Evaluation practices: analyze how methods are evaluated across the literature "
        "- are benchmarks consistent? Are metrics meaningful? Are comparisons fair?\n"
        "  - Identify cases where evaluation is inconsistent or incomparable across papers\n"
        "  - Note gaps in empirical coverage (e.g., methods only tested on small-scale data)\n"
        "(4) Reproducibility: assess the reproducibility of reported results based on "
        "methodological detail provided in original papers.\n"
        "(5) Theoretical foundations: analyze the theoretical grounding of different "
        "approaches — which are well-motivated vs. purely empirical?\n"
        "(6) Synthesis: draw overall lessons about the state of the field.\n"
        "Support claims with specific citations from the literature. Be critical but fair."
    ),
    "sec:future": (
        "Write 4-5 paragraphs outlining promising future research directions.\n"
        "Structure:\n"
        "(1) Overview: frame future directions based on identified gaps and open challenges.\n"
        "(2-3) Specific directions (one \\subsection{} per direction): for each direction\n"
        "  - State the specific problem or opportunity\n"
        "  - Explain why it is promising or necessary given current limitations\n"
        "  - Connect it to specific trends or gaps identified in the Systematic Analysis\n"
        "  - Mention relevant prior work that motivates this direction (cite it)\n"
        "(4) High-impact opportunities: highlight 1-2 directions that could have "
        "broad impact across multiple application areas.\n"
        "Draw directly from the future_directions provided in the context. "
        "Be concrete and specific — vague directions are not useful."
    ),
    "sec:conclusion": (
        "Write 3-4 paragraphs:\n"
        "(1) Summarize the overall landscape --- main themes, methodological trends, "
        "and key findings from the survey.\n"
        "(2) Open challenges: 3-4 concrete, specific research problems that remain unsolved "
        "or underexplored (draw from the key_challenges in context).\n"
        "(3) Future directions: 2-3 promising research trajectories based on the "
        "identified gaps and future_directions from the literature.\n"
        "(4) Closing: briefly position the survey's contribution to the field.\n"
        "Do NOT introduce new results or citations here."
    ),
}

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.agents.writing.latex_assembler import _strip_llm_thinking
from nanoresearch.skill_prompts import get_writing_system_prompt, ABSTRACT_SYSTEM, TITLE_SYSTEM
from .grounding_tables import _format_paper_number

ABSTRACT_SYSTEM_PROMPT = ABSTRACT_SYSTEM
TITLE_SYSTEM_PROMPT = TITLE_SYSTEM

_LEADING_FILLER_RE = re.compile(
    r"^(?:Now|Next|Then|Here|Let us|Let me|In this section|We now|"
    r"I have enough context[^.]*\.\s*|I will (?:now )?write[^.]*\.\s*|"
    r"I(?:'ll| will| can| shall) (?:now )?(?:write|draft|generate|compose|produce)[^.]*\.\s*|"
    r"Having described[^,]*,\s*|Having established[^,]*,\s*|"
    r"With this in mind[,;:\s]*|Building on this[,;:\s]*)[,;:\s]*",
    re.IGNORECASE,
)


class _SectionWriterMixin:
    """Mixin — section generation methods."""

    @staticmethod
    def _strip_leading_filler(text: str) -> str:
        """Remove LLM filler words/phrases at the start of generated section content."""
        # Only strip if the remainder still starts with a word (not a LaTeX command)
        m = _LEADING_FILLER_RE.match(text)
        if m:
            rest = text[m.end():]
            if rest and rest[0].isalpha():
                # Capitalize first letter of remaining text
                return rest[0].upper() + rest[1:]
        return text

    @staticmethod
    def _strip_abstract_environment(text: str) -> str:
        """Keep templates responsible for abstract wrapping; LLMs often add it anyway."""
        cleaned = _strip_llm_thinking(text or "").strip()
        cleaned = re.sub(r"^```(?:latex|tex)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        cleaned = re.sub(r"^\\begin\{abstract\}", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\\end\{abstract\}$", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    async def _build_writing_tools(self) -> ToolRegistry | None:
        """Build a ToolRegistry with search tools for writing.

        Returns None if no tools could be registered (missing deps).
        """
        registry = ToolRegistry()
        sources = {str(src).lower() for src in getattr(self.config, "literature_sources", ["openalex"])}
        use_openalex = "openalex" in sources
        use_arxiv = "arxiv" in sources

        if use_openalex or use_arxiv:
            try:
                search_openalex = None
                search_arxiv = None
                if use_openalex:
                    from mcp_server.tools.openalex import search_openalex as _search_openalex
                    search_openalex = _search_openalex
                if use_arxiv:
                    from mcp_server.tools.arxiv_search import search_arxiv as _search_arxiv
                    search_arxiv = _search_arxiv

                async def _search_papers(query: str, max_results: int = 5) -> list[dict]:
                    results: list[dict] = []
                    if search_openalex is not None:
                        try:
                            results.extend(await search_openalex(query, max_results=max_results))
                        except Exception as exc:
                            logger.debug("openalex search failed: %s", exc)
                    if search_arxiv is not None:
                        try:
                            results.extend(await search_arxiv(query, max_results=max_results))
                        except Exception as exc:
                            logger.debug("arxiv search failed: %s", exc)
                    return results

                registry.register(ToolDefinition(
                    name="search_papers",
                    description="Search configured academic literature sources for paper metadata.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query for papers"},
                            "max_results": {"type": "integer", "description": "Max papers to return", "default": 5},
                        },
                        "required": ["query"],
                    },
                    handler=_search_papers,
                ))
            except ImportError:
                pass

        try:
            from mcp_server.tools.openalex import search_openalex as _search_openalex_detail
            registry.register(ToolDefinition(
                name="get_paper_details",
                description="Get detailed information about a paper by title or query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Paper title or query to look up"},
                    },
                    "required": ["paper_id"],
                },
                handler=lambda paper_id: _search_openalex_detail(paper_id, max_results=1),
            ))
        except ImportError:
            pass

        use_web = "web" in sources or "web_search" in sources
        if use_web:
            try:
                from mcp_server.tools.web_search import search_web
                registry.register(ToolDefinition(
                    name="search_web",
                    description="Search the web for recent information, benchmarks, or technical details.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Max results", "default": 5},
                        },
                        "required": ["query"],
                    },
                    handler=lambda query, max_results=5: search_web(query, max_results=max_results),
                ))
            except ImportError:
                pass

        # RAG tool: read full-text from a paper's PDF
        try:
            from mcp_server.tools.pdf_reader import download_and_extract

            async def _read_paper_pdf(pdf_url: str, section: str = "") -> dict:
                """Download a paper PDF and extract its full text or a specific section."""
                result = await download_and_extract(pdf_url, max_pages=20)
                if section:
                    # Return specific section if requested
                    sections = result.get("sections", {})
                    for name, content in sections.items():
                        if section.lower() in name.lower():
                            return {"section": name, "content": content[:5000]}
                    return {"error": f"Section '{section}' not found. Available: {list(sections.keys())}"}
                # Return method + experiment + abstract (most useful for writing)
                out: dict = {}
                if result.get("method_text"):
                    out["method"] = result["method_text"][:4000]
                if result.get("experiment_text"):
                    out["experiments"] = result["experiment_text"][:4000]
                sections = result.get("sections", {})
                if "Abstract" in sections:
                    out["abstract"] = sections["Abstract"][:1000]
                if not out:
                    out["full_text"] = result.get("full_text", "")[:6000]
                out["page_count"] = result.get("page_count", 0)
                out["sections_available"] = list(sections.keys())
                return out

            registry.register(ToolDefinition(
                name="read_paper_pdf",
                description=(
                    "Download and read a paper's PDF to get its full text. "
                    "Use this to get detailed method descriptions, experiment setups, "
                    "or specific results from a paper. Provide the PDF URL "
                    "(e.g., https://arxiv.org/pdf/2301.12345). "
                    "Optionally specify a section name to read only that section."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pdf_url": {
                            "type": "string",
                            "description": "URL to the PDF file (e.g., https://arxiv.org/pdf/XXXX.XXXXX)",
                        },
                        "section": {
                            "type": "string",
                            "description": "Optional section name to extract (e.g., 'Method', 'Experiments')",
                            "default": "",
                        },
                    },
                    "required": ["pdf_url"],
                },
                handler=_read_paper_pdf,
            ))
        except ImportError:
            pass

        return registry if len(registry) > 0 else None

    # ---- section generation -------------------------------------------------

    @staticmethod
    def _fallback_title(core: dict[str, Any]) -> str:
        def humanize(value: Any) -> str:
            text = str(value or "").strip()
            text = re.sub(r"nsga\s*[-_ ]?\s*2", "NSGA-II", text, flags=re.IGNORECASE)
            text = re.sub(r"[_\-]+", " ", text)
            text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return "Compact Research Pipeline"
            protected = {
                "nsga": "NSGA",
                "nsga2": "NSGA-II",
                "nsga-ii": "NSGA-II",
                "ii": "II",
                "cnn": "CNN",
                "mlp": "MLP",
                "auc": "AUC",
                "roc": "ROC",
                "qa": "QA",
            }
            words = []
            for word in text.split():
                lower = word.lower()
                words.append(protected.get(lower, word.upper() if len(word) <= 2 else word.capitalize()))
            title = " ".join(words)
            title = re.sub(r"NSGA II", "NSGA-II", title)
            return title

        def clamp_title(value: str, limit: int = 115) -> str:
            value = re.sub(r"\s+", " ", value).strip(" .,:;-")
            if len(value) <= limit:
                return value
            cut = value[:limit].rsplit(" ", 1)[0].strip(" .,:;-")
            return cut or value[:limit].strip(" .,:;-")

        def concise_topic(raw_topic: Any) -> str:
            raw = str(raw_topic or "").lower()
            if "tabular" in raw and "classification" in raw:
                if "interpretable" in raw or "feature selection" in raw:
                    return "Interpretable Tabular Classification"
                return "Tabular Classification"
            if "time" in raw and "series" in raw:
                return "Time-Series Classification"
            if "image" in raw and "classification" in raw:
                return "Image Classification"
            if "question" in raw or "qa" in raw:
                return "Question Answering"
            topic_text = humanize(str(raw_topic or "Research Automation").split(";")[0])
            return clamp_title(topic_text, limit=58)

        method = humanize(core.get("method_name") or "Compact Research Pipeline")
        method = re.sub(r"^Fixed Budget\b", "Fixed-Budget", method)
        topic = concise_topic(core.get("topic") or "Research Automation")
        if topic and topic.lower() not in method.lower():
            return clamp_title(f"{method} for {topic}")
        return clamp_title(method)

    @staticmethod
    def _fallback_abstract(core: dict[str, Any], grounding: GroundingPacket | None = None) -> str:
        method = core.get("method_name") or "the proposed method"
        topic = core.get("topic") or "the target task"
        datasets = core.get("dataset_names") or "the specified dataset"
        metrics = core.get("metric_names") or "the specified metrics"
        metric_sentence = ""
        if grounding and grounding.has_real_results and grounding.final_metrics:
            shown = []
            for key, value in list(grounding.final_metrics.items())[:3]:
                shown.append(f"{key.replace('_', ' ')}={_format_paper_number(value)}")
            if shown:
                metric_sentence = " The measured run reports " + ", ".join(shown) + "."
        return (
            f"We study {topic} with {method}, focusing on reproducible evaluation and compact implementation. "
            f"The pipeline evaluates the method on {datasets} using {metrics}, compares measured baselines, and reports ablations and complexity diagnostics when artifacts are available."
            f"{metric_sentence} "
            "All quantitative claims in the paper are grounded in local run artifacts, and missing measurements are scoped as limitations rather than replaced with synthetic values."
        )

    async def _generate_title(self, context: str) -> str:
        prompt = f"Based on the following research context, generate a paper title:\n\n{context}"
        try:
            return ((await self.generate(TITLE_SYSTEM_PROMPT, prompt)) or "").strip().strip('"')
        except Exception as e:
            logger.warning("Title generation failed, using fallback: %s", e)
            return "Untitled Research Paper"

    async def _generate_abstract(
        self, context: str, grounding: GroundingPacket | None = None,
    ) -> str:
        number_binding = ""
        if grounding and grounding.has_real_results and grounding.final_metrics:
            metric_strs = [f"{k}={_format_paper_number(v)}" for k, v in list(grounding.final_metrics.items())[:5]]
            number_binding = (
                "\n\nIMPORTANT — RESULT NUMBERS IN ABSTRACT:\n"
                f"Real experiment metrics: {', '.join(metric_strs)}\n"
                "You MUST mention at least the primary metric in the abstract. "
                "Use the rounded display value from above. Do NOT fabricate different numbers."
            )
        elif grounding and not grounding.has_real_results:
            number_binding = (
                "\n\nIMPORTANT: No real experiment results are available. "
                "Do NOT mention any specific accuracy/F1/performance numbers in the abstract. "
                "Focus on the method and its design instead."
            )
        prompt = f"Based on the following research context, write the abstract:\n\n{context}{number_binding}"
        try:
            abstract = ((await self.generate(ABSTRACT_SYSTEM_PROMPT, prompt)) or "").strip()
            abstract = self._strip_abstract_environment(abstract)
            return abstract
        except Exception as e:
            logger.warning("Abstract generation failed, using fallback: %s", e)
            return "Abstract not available."

    async def _generate_section(
        self, context: str, heading: str, instructions: str,
        prior_sections: list[str] | None = None,
    ) -> str:
        prior_ctx = ""
        if prior_sections:
            prior_ctx = (
                "\n\n=== PREVIOUSLY WRITTEN SECTIONS (maintain consistency) ===\n"
                + "\n".join(prior_sections)
                + "\n=== END PREVIOUS SECTIONS ===\n"
            )

        # Per-section specialized system prompt (replaces generic SECTION_SYSTEM_PROMPT)
        section_system = get_writing_system_prompt(heading)

        section_length_rules = {
            "Introduction": "Write 5-7 substantive paragraphs plus at most one short contribution list; target 650-900 words.",
            "Related Work": "Write 2-3 dense positioning paragraphs, not a survey; target 300-450 words.",
            "Method": "Write 5-6 technical subsections with mechanism prose, notation, and only essential equations; target 1000-1400 words.",
            "Experiments": "Write a complete artifact-grounded evaluation narrative around the provided tables and figures; target 1400-1900 words.",
            "Conclusion": "Write 2 concise paragraphs; target 180-260 words.",
        }
        length_rule = section_length_rules.get(heading, "Write a concise, complete section.")

        prompt = f"""Write the "{heading}" section for this paper.

Length and scope: {length_rule}

Instructions: {instructions}

Research Context:
{context}{prior_ctx}

IMPORTANT: Use ONLY the citation keys listed in the CITATION KEYS section above.
For example, write \\cite{{dokholyan1998}} NOT \\cite{{1}} or \\cite{{XXXX}}.
Maintain consistent notation and terminology with any previously written sections.

Output ONLY the LaTeX paragraphs for this section. Do not include \\section command.

FORMAT RULES:
- Do NOT wrap your output in Markdown code fences (```latex ... ``` or ``` ... ```).
- Do NOT insert \\begin{{figure}}...\\end{{figure}} environments; use \\ref{{fig:xxx}}
  to reference figures only — figures are inserted automatically by the pipeline.
- Use bare ~ for non-breaking space before \\ref (e.g. Figure~\\ref{{fig:arch}}),
  NOT \\~{{}} which renders as a tilde accent character in LaTeX."""

        if getattr(self.config, "deterministic_writing_fallback", False):
            return self._fallback_section_content(heading, context)

        if heading == "Method":
            return await self._generate_method_section_by_parts(context, instructions, prior_sections)

        # Use tool-augmented generation for key sections unless the active
        # writing pass explicitly disabled tools for a no-result fallback draft.
        if (
            not getattr(self, "_disable_writing_tools_for_stage", False)
            and self.config.should_use_writing_tools(heading)
        ):
            try:
                tools = await self._build_writing_tools()
                if tools is not None:
                    tool_prompt = (
                        prompt + "\n\nYou have access to search tools. "
                        "If you need to verify citations, find additional references, "
                        "or look up recent results, use the tools before writing. "
                        "Ground claims in retrieved evidence and actual experiment outputs."
                    )
                    content = (await self.generate_with_tools(
                        section_system, tool_prompt, tools,
                        max_tool_rounds=self.config.writing_tool_max_rounds,
                    ) or "").strip()
                    if not content:
                        self.log(f"  ReAct loop returned empty content for {heading}, retrying without tools")
                        content = ((await self.generate(section_system, prompt)) or "").strip()
                    # Defense-in-depth: strip stray \end{document}
                    content = re.sub(r'\\end\{document\}\s*', '', content).strip()
                    # Strip LLM filler at start of section
                    content = self._strip_leading_filler(content)
                    content = _strip_llm_thinking(content)
                    # Fix component count mismatches (e.g. "four components" but lists 5)
                    content = self._fix_component_count_mismatch(content)
                    return content
            except Exception as e:
                logger.warning("Tool-augmented writing failed for %s, falling back: %s", heading, e)

        last_error: Exception | None = None
        section_token_floor = {"Introduction": 6144, "Related Work": 4096, "Method": 8192, "Experiments": 8192, "Conclusion": 4096}
        section_timeout = {"Introduction": 240.0, "Related Work": 180.0, "Method": 360.0, "Experiments": 360.0, "Conclusion": 180.0}
        base_tokens = max(self.stage_config.max_tokens, section_token_floor.get(heading, 6144))
        base_timeout = max(float(self.stage_config.timeout or 0), section_timeout.get(heading, 240.0))
        for attempt in range(1, 4):
            try:
                retry_note = "" if attempt == 1 else (
                    "\n\nPrevious attempt failed or was too thin. Rewrite the section fully, "
                    "keeping all claims grounded in the provided artifacts."
                )
                cfg = self.stage_config.model_copy(
                    update={"max_tokens": base_tokens, "timeout": base_timeout + 60.0 * (attempt - 1)}
                )
                content = ((await self.generate(section_system, prompt + retry_note, stage_override=cfg)) or "").strip()
                content = re.sub(r'\\end\{document\}\s*', '', content).strip()
                content = self._strip_leading_filler(content)
                content = _strip_llm_thinking(content)
                content = self._fix_component_count_mismatch(content)
                min_words = {"Introduction": 450, "Method": 750, "Experiments": 900}.get(heading, 80)
                word_count = len(re.findall(r"\b\w+\b", re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", content)))
                if word_count < min_words:
                    raise ValueError(f"{heading} section too short: {word_count} words < {min_words}")
                return content
            except Exception as e:
                last_error = e
                logger.warning("Section generation attempt %d failed for %s: %s", attempt, heading, e)
        logger.warning("Section generation failed for %s after retries, using deterministic fallback: %s", heading, last_error)
        return self._fallback_section_content(heading, context)

    async def _generate_method_section_by_parts(
        self,
        context: str,
        instructions: str,
        prior_sections: list[str] | None = None,
    ) -> str:
        """Generate Method through compact LLM calls instead of one oversized prompt."""
        system = get_writing_system_prompt("Method")
        prior_summary = ""
        if prior_sections:
            joined = "\n".join(prior_sections)
            joined = re.sub(r"\\begin\{(?:figure|table)\*?\}.*?\\end\{(?:figure|table)\*?\}", " ", joined, flags=re.DOTALL)
            joined = re.sub(r"\s+", " ", joined).strip()
            prior_summary = joined[:900]

        parts = [
            (
                "Problem Setup and Notation",
                "Define the supervised task, candidate solution representation, train/validation/test boundary, and leakage-safety assumptions. Include only notation needed later.",
            ),
            (
                "Training-Only Evaluator",
                "Explain how each candidate is evaluated with training and validation data only, what metrics are recorded, and why this evaluator is reusable across candidates.",
            ),
            (
                "Multi-Objective Search",
                "Describe the search mechanism, candidate update process, archive/frontier maintenance, and how quality and compactness are optimized jointly.",
            ),
            (
                "Pareto Selection and Final Refit",
                "Explain the final selection rule, refit protocol, held-out evaluation, and how the selected model is converted into paper-facing evidence.",
            ),
            (
                "Complexity and Implementation Notes",
                "Discuss runtime drivers, model-size drivers, reproducibility controls, and implementation choices needed to reproduce the method without overclaiming.",
            ),
        ]
        outputs: list[str] = []
        previous_tail = ""
        for index, (title, focus) in enumerate(parts, start=1):
            prompt = f"""Write one Method subsection for this paper.

Subsection title: {title}
Subsection focus: {focus}
Overall Method instructions: {instructions}

Compact research context:
{context}

Prior paper context summary:
{prior_summary}

Previously generated Method tail:
{previous_tail}

Requirements:
- Output exactly one LaTeX subsection beginning with \\subsection{{{title}}}.
- Write 170-260 words of technical prose for this subsection.
- Use at most one compact equation in this subsection, and only if it clarifies the mechanism.
- Keep equations within line width; prefer short displayed equations over long align blocks.
- Do not include figures, tables, Markdown fences, or a \\section command.
- Do not fabricate experiment results or external numbers.
"""
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    cfg = self.stage_config.model_copy(
                        update={"max_tokens": 3072, "timeout": 180.0 + 45.0 * (attempt - 1)}
                    )
                    suffix = "" if attempt == 1 else "\nPrevious attempt failed validation. Regenerate only this subsection with enough technical detail."
                    content = ((await self.generate(system, prompt + suffix, stage_override=cfg)) or "").strip()
                    content = re.sub(r'\\end\{document\}\s*', '', content).strip()
                    content = self._strip_leading_filler(_strip_llm_thinking(content))
                    content = self._fix_component_count_mismatch(content)
                    if "\\subsection" not in content:
                        content = f"\\subsection{{{title}}}\n" + content
                    prose = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", content)
                    words = len(re.findall(r"\b\w+\b", prose))
                    if words < 130:
                        raise ValueError(f"{title} too short: {words} words")
                    outputs.append(content)
                    previous_tail = re.sub(r"\s+", " ", content).strip()[-700:]
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning("Method subsection attempt %d failed for %s: %s", attempt, title, exc)
            else:
                raise RuntimeError(f"Method subsection generation failed for {title}: {last_error}")
        method = "\n\n".join(outputs).strip()
        method = self._fix_component_count_mismatch(method)
        words = len(re.findall(r"\b\w+\b", re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", method)))
        if words < 800:
            raise ValueError(f"Method section too short after part generation: {words} words")
        return method

    async def _expand_composed_experiments(self, context: str, composed: str) -> str:
        """Use the LLM to expand prose while preserving artifact floats verbatim."""
        if len(composed) > 7000:
            return await self._expand_composed_experiments_by_blocks(context, composed)

        system = get_writing_system_prompt("Experiments")
        prompt = f"""Expand the following artifact-grounded Experiments section into a complete paper-quality evaluation section.

Hard constraints:
- Preserve every \\begin{{table}}...\\end{{table}} and \\begin{{figure}}...\\end{{figure}} block EXACTLY. Do not rewrite, remove, duplicate, or rename labels.
- Preserve all reported numeric values exactly. Do not add new numbers, baselines, datasets, seeds, or claims.
- Add substantive prose before and after each table/figure so the section reads like a paper, not a list of visuals.
- Organize the prose around findings, protocol, ablations, trade-offs, complexity, and evidence scope.
- Target 1400-1900 words excluding LaTeX table/figure bodies.
- Output ONLY the section body. Do not include a \\section command.

Research context:
{self._compact_experiment_context(context)}

Artifact-composed section to expand:
{composed}
"""
        required_labels = set(re.findall(r"\\label\{([^}]+)\}", composed))
        required_tables = composed.count("\\begin{table}") + composed.count("\\begin{table*}")
        required_figures = composed.count("\\begin{figure}") + composed.count("\\begin{figure*}")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                cfg = self.stage_config.model_copy(update={"max_tokens": 4096, "timeout": 240.0 + 45.0 * (attempt - 1)})
                suffix = "" if attempt == 1 else "\n\nPrevious attempt failed validation. Preserve every float, table, label, and numeric value exactly while expanding prose."
                content = ((await self.generate(system, prompt + suffix, stage_override=cfg)) or "").strip()
                content = re.sub(r'\\end\{document\}\s*', '', content).strip()
                content = self._strip_leading_filler(_strip_llm_thinking(content))
                labels = set(re.findall(r"\\label\{([^}]+)\}", content))
                table_count = content.count("\\begin{table}") + content.count("\\begin{table*}")
                figure_count = content.count("\\begin{figure}") + content.count("\\begin{figure*}")
                if not required_labels.issubset(labels):
                    raise ValueError(f"missing labels: {sorted(required_labels - labels)}")
                if table_count < required_tables or figure_count < required_figures:
                    raise ValueError("missing preserved table/figure blocks")
                prose = re.sub(r"\\begin\{(?:figure|table)\*?\}.*?\\end\{(?:figure|table)\*?\}", " ", content, flags=re.DOTALL)
                words = len(re.findall(r"\b\w+\b", prose))
                if words < 900:
                    raise ValueError(f"expanded Experiments too short: {words} words")
                return content
            except Exception as exc:
                last_error = exc
                logger.warning("Experiment expansion attempt %d failed: %s", attempt, exc)
        logger.warning("Experiment expansion failed after retries; using artifact-composed section: %s", last_error)
        return composed

    @staticmethod
    def _compact_experiment_context(context: str, limit: int = 2600) -> str:
        text = re.sub(r"\\begin\{(?:figure|table)\*?\}.*?\\end\{(?:figure|table)\*?\}", " ", context, flags=re.DOTALL)
        text = re.sub(r"=== ARTIFACT-GROUNDED EXPERIMENT SKELETON ===.*?=== END SKELETON ===", " ", text, flags=re.DOTALL)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > limit:
            return text[: limit - 15].rstrip() + " ...[truncated]"
        return text

    async def _expand_composed_experiments_by_blocks(self, context: str, composed: str) -> str:
        """Expand only prose blocks while preserving artifact tables/figures verbatim."""
        system = get_writing_system_prompt("Experiments")
        compact_context = self._compact_experiment_context(context)
        float_re = re.compile(r"(\\begin\{(?:table|figure)\*?\}.*?\\end\{(?:table|figure)\*?\})", re.DOTALL)
        pieces = float_re.split(composed)
        expanded: list[str] = []
        last_error: Exception | None = None
        for idx, piece in enumerate(pieces):
            if not piece.strip():
                expanded.append(piece)
                continue
            if float_re.fullmatch(piece.strip()):
                expanded.append(piece)
                continue
            plain_words = len(re.findall(r"\b\w+\b", re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", piece)))
            if plain_words < 45:
                expanded.append(piece)
                continue
            prompt = f"""Polish and expand this Experiments prose block while preserving its meaning and numeric claims.

Research context summary:
{compact_context}

Original prose block:
{piece.strip()}

Requirements:
- Output only revised LaTeX prose for this block.
- Keep every numeric value, table reference, figure reference, dataset name, and method name faithful to the original block.
- Do not add tables, figures, labels, new baselines, new seeds, or unsupported claims.
- Make the prose read like a connected experimental narrative, not a caption explanation.
- Target 1.4x to 2.2x the original prose length when the block is thin.
"""
            block_done = False
            for attempt in range(1, 4):
                try:
                    cfg = self.stage_config.model_copy(
                        update={"max_tokens": 2048, "timeout": 150.0 + 30.0 * (attempt - 1)}
                    )
                    suffix = "" if attempt == 1 else "\nPrevious attempt failed validation. Preserve references and numeric claims exactly."
                    content = ((await self.generate(system, prompt + suffix, stage_override=cfg)) or "").strip()
                    content = re.sub(r'\\end\{document\}\s*', '', content).strip()
                    content = self._strip_leading_filler(_strip_llm_thinking(content))
                    if "\\begin{table}" in content or "\\begin{figure}" in content:
                        raise ValueError("prose block expansion inserted a float")
                    old_refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", piece))
                    new_refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", content))
                    if not old_refs.issubset(new_refs):
                        raise ValueError(f"missing references: {sorted(old_refs - new_refs)}")
                    new_words = len(re.findall(r"\b\w+\b", re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})?", " ", content)))
                    if new_words < max(plain_words, 80):
                        raise ValueError(f"expanded prose too short: {new_words} words")
                    expanded.append("\n" + content + "\n")
                    block_done = True
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning("Experiment prose block expansion attempt %d failed for block %d: %s", attempt, idx, exc)
            if not block_done:
                expanded.append(piece)
        result = "".join(expanded).strip()
        required_labels = set(re.findall(r"\\label\{([^}]+)\}", composed))
        labels = set(re.findall(r"\\label\{([^}]+)\}", result))
        if not required_labels.issubset(labels):
            logger.warning("Block experiment expansion lost labels; using artifact-composed section: %s", sorted(required_labels - labels))
            return composed
        prose = re.sub(r"\\begin\{(?:figure|table)\*?\}.*?\\end\{(?:figure|table)\*?\}", " ", result, flags=re.DOTALL)
        words = len(re.findall(r"\b\w+\b", prose))
        if words < 900:
            logger.warning("Block experiment expansion too short (%d words); using composed section. Last error: %s", words, last_error)
            return composed
        return result

    def _fallback_section_content(self, heading: str, context: str) -> str:
        """Return a compact, artifact-grounded section when the LLM endpoint times out."""
        topic = self._extract_context_line(context, "Topic") or "the target task"
        method = self._extract_context_line(context, "Proposed Method") or "the proposed method"
        method_overview = self._extract_context_line(context, "Method Overview") or method
        datasets = self._extract_context_line(context, "Datasets") or "the specified benchmark dataset"
        metrics = self._extract_context_line(context, "Metrics") or "the specified evaluation metrics"
        citations = re.findall(r"\\cite[t|p]?\{([^}]+)\}", context)
        cite = f"\\citep{{{citations[0]}}}" if citations else ""
        if heading == "Introduction":
            return (
                f"{topic} requires methods that balance predictive quality with implementation simplicity. "
                f"This paper studies {method_overview} under a reproducible, artifact-grounded protocol. "
                f"The central goal is to test whether a compact model can retain competitive performance while exposing the cost of each design choice.\n\n"
                f"Small public benchmarks are useful for this question because they make data leakage, feature count, and runtime easy to inspect. "
                f"Rather than treating the final score as the only outcome, the pipeline records the validation protocol, selected feature budget, ablation variants, and complexity diagnostics. "
                f"This makes the resulting draft auditable: every quantitative claim must trace back to an artifact produced by the run.\n\n"
                f"We make three contributions. First, we formulate the task around a lightweight method, {method}, that can be run and inspected on {datasets}. "
                f"Second, we evaluate the method with the measured metrics {metrics}, using only artifacts produced by the local run. "
                f"Third, we report ablation and complexity evidence when those artifacts are available, avoiding unsupported claims."
            )
        if heading == "Related Work":
            return (
                f"Prior work on compact machine-learning pipelines emphasizes that small structured benchmarks require careful validation rather than large-model scale {cite}. "
                f"This line of work motivates simple baselines, transparent metrics, and controlled ablations for {topic}.\n\n"
                f"The closest prior work is determined by the method and dataset in the current blueprint rather than by a fixed model family. "
                f"The related-work discussion therefore focuses on comparable task setups, baselines, evaluation metrics, and reproducibility constraints.\n\n"
                f"This paper therefore positions {method} as a reproducible, artifact-first workflow rather than as a broad claim of state-of-the-art performance. "
                f"The comparison is restricted to measured baselines and ablations from the same local run, so the related work motivates the evaluation protocol without substituting unverified external numbers for experiment artifacts."
            )
        if heading == "Method":
            return (
                "\\subsection{Problem Setup and Notation}\n"
                f"Let $\\mathcal{{D}}$ denote the dataset used for {datasets}. The method section defines the inputs, targets, splits, and evaluation boundary needed to reproduce {method}. "
                "Training data are used for fitting and validation evidence is used only according to the blueprint; held-out data are reserved for final reporting when available.\n\n"
                "\\subsection{Core Method}\n"
                f"The implemented method is {method}. Its components are described in the order they appear in the code and experiment blueprint, with each component tied to the task-specific representation and metric definitions. "
                "This fallback text intentionally avoids assuming a particular optimizer, feature representation, model family, or selection procedure.\n\n"
                "\\subsection{Training and Evaluation Protocol}\n"
                "The runner executes the proposed method, measured baselines, and ablations specified in the experiment matrix. "
                "Each run reports only the metrics requested by its run specification and records failures in the manifest rather than silently converting them into successful results.\n\n"
                "\\subsection{Implementation and Complexity Notes}\n"
                "Implementation details focus on the costs and constraints that are actually exposed by the artifacts, such as runtime, memory, parameter count, or other topic-specific diagnostics when measured. "
                "The paper does not introduce additional complexity claims that are absent from the executed outputs."
            )
        if heading == "Experiments":
            return (
                f"We evaluate {method} on {datasets} using {metrics}. Table~\\ref{{tab:main_results}} reports the measured main comparison from the run artifacts, and Table~\\ref{{tab:ablation}} reports measured ablations when available. "
                "These tables are generated from local result files rather than projected or synthetic values.\n\n"
                "Figure~\\ref{fig:fig2_accuracy_sparsity_tradeoff} summarizes the accuracy--compactness trade-off, while Figure~\\ref{fig:fig3_main_results} visualizes the main metric comparison. "
                "Figure~\\ref{fig:fig4_ablation_study} and Figure~\\ref{fig:fig5_efficiency_complexity} show component and efficiency diagnostics when the corresponding artifacts are present."
            )
        if heading == "Conclusion":
            return (
                f"This paper presents {method} for {topic} and evaluates it using artifact-grounded measurements. "
                "The resulting draft reports only measured comparisons and scopes missing evidence as limitations rather than filling unsupported values. "
                "Future work should expand the benchmark coverage, repeat the run across seeds, and test whether the same compactness trade-off persists on larger datasets."
            )
        return f"This section summarizes {heading.lower()} for {topic} using the available NanoResearch artifacts."

    @staticmethod
    def _extract_context_line(context: str, prefix: str) -> str:
        match = re.search(rf"^{re.escape(prefix)}:\s*(.+)$", context, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    # ---- post-processing: component count fix --------------------------------

    # Map number words to digits (and back)
    _NUM_WORDS = {
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    _DIGIT_WORDS = {v: k for k, v in _NUM_WORDS.items()}

    @classmethod
    def _fix_component_count_mismatch(cls, content: str) -> str:
        r"""Fix 'N components: (1)...(2)...' where N doesn't match actual count.

        Detects patterns like 'four key components' followed by enumerated items
        (either \item, (1)/(2)/(3), or \subsection) and corrects the number word.
        """
        # Pattern: "NUM_WORD key/main/core? components/modules/stages/parts"
        num_claim_re = re.compile(
            r'\b(two|three|four|five|six|seven|eight|nine|ten)\b'
            r'(\s+(?:key|main|core|primary|major|fundamental|critical|novel|distinct))?'
            r'\s+(components?|modules?|stages?|parts?|contributions?|elements?|blocks?)',
            re.IGNORECASE,
        )
        # Find all claims in the content
        for m in num_claim_re.finditer(content):
            claimed_word = m.group(1).lower()
            claimed_n = cls._NUM_WORDS.get(claimed_word, 0)
            if claimed_n == 0:
                continue

            # Count actual items after this claim
            after = content[m.end():]
            # Strategy 1: count \item entries (in itemize/enumerate)
            items_match = re.search(
                r'\\begin\{(?:itemize|enumerate)\}(.*?)\\end\{(?:itemize|enumerate)\}',
                after[:3000], re.DOTALL,
            )
            if items_match:
                actual_n = len(re.findall(r'\\item\b', items_match.group(1)))
            else:
                # Strategy 2: count (1), (2), (3) style enumeration
                # Only match sequential numbering starting from (1) to avoid
                # false positives from equation numbers, years, etc.
                paren_items = re.findall(r'\((\d+)\)', after[:2000])
                # Filter: keep only small sequential numbers (1-10 range)
                paren_small = [int(x) for x in paren_items if 1 <= int(x) <= 10]
                # Require (1) to be present (sequential enumeration starts at 1)
                if paren_small and 1 in paren_small:
                    actual_n = max(paren_small)
                else:
                    # Strategy 3: count \subsection entries
                    subsections = re.findall(r'\\subsection\{', after[:5000])
                    actual_n = len(subsections) if subsections else 0

            if actual_n > 0 and actual_n != claimed_n and actual_n in cls._DIGIT_WORDS:
                correct_word = cls._DIGIT_WORDS[actual_n]
                # Preserve original capitalization
                if m.group(1)[0].isupper():
                    correct_word = correct_word.capitalize()
                old_span = content[m.start():m.end()]
                new_span = old_span[:m.start(1) - m.start()] + correct_word + old_span[m.end(1) - m.start():]
                content = content[:m.start()] + new_span + content[m.end():]
                logger.info(
                    "Fixed component count mismatch: '%s' → '%s' (actual: %d items)",
                    claimed_word, correct_word, actual_n,
                )
                break  # Only fix the first mismatch per call to avoid cascading

        return content

    # ---- bibtex & latex -----------------------------------------------------

    # Conference venues that should use @inproceedings
    _CONFERENCE_VENUES = frozenset({
        "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
        "acl", "emnlp", "naacl", "aaai", "ijcai", "sigir", "kdd",
        "www", "uai", "aistats", "coling", "interspeech", "icra",
        "iros", "miccai", "wacv", "bmvc", "accv",
    })

    @classmethod
    def _detect_entry_type(cls, venue: str) -> str:
        """Determine BibTeX entry type from venue name."""
        if not venue:
            return "article"
        venue_lower = venue.lower()
        for conf in cls._CONFERENCE_VENUES:
            if conf in venue_lower:
                return "inproceedings"
        if "workshop" in venue_lower or "proceedings" in venue_lower:
            return "inproceedings"
        return "article"

    def _build_bibtex(self, papers: list[dict], cite_keys: dict[int, str]) -> str:
        entries = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_CITATIONS]):
            if i not in cite_keys:
                continue
            if not isinstance(p, dict):
                continue
            key = cite_keys[i]
            authors = p.get("authors", [])
            if not isinstance(authors, list):
                authors = [str(authors)] if authors else []
            author_str = " and ".join(authors[:5])
            title = p.get("title", "Unknown")
            venue = p.get("venue", "") or "arXiv preprint"
            year = p.get("year", 2024)
            url = p.get("url", "")

            entry_type = self._detect_entry_type(venue)

            if entry_type == "inproceedings":
                entry = (
                    f"@inproceedings{{{key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{author_str}}},\n"
                    f"  year={{{year}}},\n"
                    f"  booktitle={{{venue}}},\n"
                )
            else:
                entry = (
                    f"@article{{{key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{author_str}}},\n"
                    f"  year={{{year}}},\n"
                    f"  journal={{{venue}}},\n"
                )
            if url:
                entry += f"  url={{{url}}},\n"
            entry += "}\n"
            entries.append(entry)
        return "\n".join(entries)
