"""Ideation agent -- literature search, gap analysis, idea generation.

Split into 3 modules:
    ideation.py             -- IdeationAgent facade + run() + small helpers
    ideation_search.py      -- _IdeationSearchMixin (literature search/filter)
    ideation_hypothesis.py  -- _IdeationHypothesisMixin (tools, analysis, evidence)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.evolution.memory import MemoryType
from nanoresearch.experiments.canonical_baselines import load_canonical_baseline_registry
from nanoresearch.idea_utils import add_idea_aliases_to_ideation, get_selected_idea_id
from nanoresearch.schemas.evidence import EvidenceBundle, ExtractedMetric
from nanoresearch.schemas.ideation import IdeationOutput, PaperReference
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# --- Configurable limits (magic numbers extracted) ---
MAX_SEARCH_QUERIES = 5
MAX_RESULTS_PER_SEARCH = 10
MAX_PAPERS_FOR_ANALYSIS = 30          # was 50 -- reduced to save tokens
MAX_ABSTRACT_LENGTH = 500
MAX_GITHUB_REPOS = 5
MAX_GITHUB_QUERIES = 2

# Phase 4: Citation quality targets
TARGET_CITATION_COUNT = 30            # was 50 -- reduced to save tokens
MIN_HIGH_CITED_PAPERS = 8             # was 10 -- adjusted for smaller set
HIGH_CITATION_THRESHOLD = 100
TOP_K_FULL_TEXT = 4                   # was 8 -- PDF full-text is expensive

# Token budget limits for LLM prompts
MAX_METHOD_TEXT_PER_PAPER = 1000      # was 3000 -- method_text truncation
MAX_EXPERIMENT_TEXT_PER_PAPER = 1000  # was 3000 -- experiment_text truncation

# Lazy imports to avoid hard dependency on mcp_server at import time
_arxiv_search = None
_s2_search = None
_github_search = None
_oa_search = None
_import_lock = asyncio.Lock()


async def _get_arxiv_search():
    global _arxiv_search
    if _arxiv_search is None:
        async with _import_lock:
            if _arxiv_search is None:
                from mcp_server.tools.arxiv_search import search_arxiv
                _arxiv_search = search_arxiv
    return _arxiv_search


async def _get_s2_search():
    global _s2_search
    if _s2_search is None:
        async with _import_lock:
            if _s2_search is None:
                from mcp_server.tools.semantic_scholar import search_semantic_scholar
                _s2_search = search_semantic_scholar
    return _s2_search


async def _get_github_search():
    global _github_search
    if _github_search is None:
        async with _import_lock:
            if _github_search is None:
                from mcp_server.tools.github_search import search_repos
                _github_search = search_repos
    return _github_search


async def _get_oa_search():
    """Lazy import OpenAlex search (returns None if module unavailable)."""
    global _oa_search
    if _oa_search is None:
        async with _import_lock:
            if _oa_search is None:
                try:
                    from mcp_server.tools.openalex import search_openalex
                    _oa_search = search_openalex
                except ImportError:
                    _oa_search = False  # mark as unavailable
    return _oa_search if _oa_search else None


from nanoresearch.prompts import load_prompt as _load_prompt
from nanoresearch.skill_prompts import (
    IDEATION_QUERY_SYSTEM,
    IDEATION_ANALYSIS_SYSTEM,
    IDEATION_MUST_CITE_SYSTEM,
    IDEATION_EVIDENCE_SYSTEM,
)

# Legacy alias -- some internal methods still reference this.
IDEATION_SYSTEM_PROMPT = IDEATION_QUERY_SYSTEM

SEARCH_COVERAGE_SYSTEM_PROMPT = _load_prompt("ideation", "search_coverage")

# Import mixins (after constants are defined, since mixins import them)
from nanoresearch.agents.ideation_search import _IdeationSearchMixin      # noqa: E402
from nanoresearch.agents.ideation_hypothesis import _IdeationHypothesisMixin  # noqa: E402


class IdeationAgent(_IdeationSearchMixin, _IdeationHypothesisMixin, BaseResearchAgent):
    stage = PipelineStage.IDEATION

    @staticmethod
    def _normalize_string_list_field(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(
                    item.get("theme")
                    or item.get("challenge")
                    or item.get("direction")
                    or item.get("name")
                    or item.get("title")
                    or item.get("description")
                    or ""
                ).strip()
                if not text:
                    text = str(item).strip()
            else:
                text = str(item).strip() if item is not None else ""
            if text:
                normalized.append(text)
        return normalized

    @staticmethod
    def _extract_topic_field(topic: str, field_name: str) -> str:
        prefix = f"{field_name}:"
        for line in (topic or "").splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
        return ""

    @classmethod
    def _extract_topic_list_field(cls, topic: str, field_name: str) -> list[str]:
        raw = cls._extract_topic_field(topic, field_name)
        return [item.strip() for item in raw.split(";") if item.strip()]

    def _remember_literature_skill(
        self,
        topic: str,
        trigger_pattern: str,
        trace: str,
        *,
        confidence: float = 0.62,
        extra_tags: list[str] | None = None,
    ) -> None:
        tags = [topic, "literature", self.workspace.manifest.paper_mode.value]
        if extra_tags:
            tags.extend(extra_tags)
        self.learn_from_trace(
            "literature",
            trigger_pattern,
            trace,
            tags=tags,
            confidence=confidence,
        )

    def _record_retrieval_skill_summary(
        self,
        topic: str,
        *,
        queries: list[str],
        papers: list[dict[str, Any]],
        survey_papers: int,
        supplementary_rounds: int,
        must_cites: list[str],
        evidence: EvidenceBundle | None,
        reference_repos: list[dict[str, Any]],
        used_cache: bool,
    ) -> None:
        evidence = evidence or EvidenceBundle()
        query_preview = "; ".join(str(query).strip() for query in queries[:5] if str(query).strip())
        top_titles = "; ".join(
            str(paper.get("title") or "").strip()
            for paper in papers[:3]
            if str(paper.get("title") or "").strip()
        )
        warnings = "; ".join(str(item).strip() for item in evidence.coverage_warnings[:3] if str(item).strip())

        summary = (
            f"Literature retrieval for {topic}: generated {len(queries)} queries "
            f"({query_preview or 'none'}); retained {len(papers)} papers after filtering; "
            f"survey_papers={survey_papers}; supplementary_rounds={supplementary_rounds}; "
            f"must_cites={len(must_cites)}; extracted_metrics={len(evidence.extracted_metrics)}; "
            f"reference_repos={len(reference_repos)}; used_cache={used_cache}. "
            f"Representative papers: {top_titles or 'none'}. "
            f"Coverage warnings: {warnings or 'none'}."
        )
        self._remember_literature_skill(
            topic,
            "retrieval_pipeline_summary",
            summary,
            confidence=0.68 if len(papers) >= 10 else 0.58,
            extra_tags=["retrieval_pipeline"],
        )

        if supplementary_rounds > 0:
            self._remember_literature_skill(
                topic,
                "coverage_gap_supplement",
                (
                    f"Coverage self-evaluation for {topic} required {supplementary_rounds} supplementary search round(s). "
                    f"Use missing-direction feedback to broaden queries instead of trusting the first retrieval pass."
                ),
                confidence=0.74,
                extra_tags=["coverage_gap", "supplementary_search"],
            )

        if len(papers) < 8 or evidence.coverage_warnings:
            self._remember_literature_skill(
                topic,
                "retrieval_sparse_or_noisy",
                (
                    f"Retrieval for {topic} remained sparse/noisy: kept {len(papers)} papers, "
                    f"warnings={warnings or 'none'}. Prefer broader synonyms, benchmark names, survey/review queries, "
                    f"and baseline-specific OpenAlex lookups before drafting novelty claims."
                ),
                confidence=0.72,
                extra_tags=["sparse_retrieval"],
            )

        if evidence.extracted_metrics:
            metric_preview = "; ".join(
                f"{metric.method_name or 'method'} on {metric.dataset or 'dataset'}: {metric.metric_name}={metric.value}"
                for metric in evidence.extracted_metrics[:3]
            )
            self._remember_literature_skill(
                topic,
                "baseline_metric_grounding",
                (
                    f"For {topic}, literature retrieval extracted quantitative baseline evidence. "
                    f"Ground planning and novelty claims in retrieved metrics when available. Examples: {metric_preview}."
                ),
                confidence=0.76,
                extra_tags=["baseline_grounding", "quantitative_evidence"],
            )

    async def _retrieve_local_eval_context(
        self,
        topic: str,
    ) -> tuple[list[PaperReference], EvidenceBundle, list[str]]:
        """Build local-only eval context without any external retrieval."""
        question_id = self._extract_topic_field(topic, "Evaluation Question ID")
        baselines = self._extract_topic_list_field(topic, "Known Baselines")
        datasets = self._extract_topic_list_field(topic, "Evaluation Datasets")

        references: list[PaperReference] = []
        extracted_metrics: list[ExtractedMetric] = []
        coverage_warnings: list[str] = []
        extraction_notes = "Eval-fast mode used only local question context and canonical baseline registry; no external literature retrieval was performed."

        registry_entry = load_canonical_baseline_registry().get(question_id) if question_id else None
        if registry_entry:
            for idx, metric in enumerate(registry_entry.get("metrics") or [], start=1):
                baseline_name = str(metric.get("baseline_name") or "").strip()
                metric_name = str(metric.get("metric_name") or "").strip()
                provenance_title = str(metric.get("provenance_title") or baseline_name or f"{question_id} baseline").strip()
                provenance_uri = str(metric.get("provenance_uri") or "").strip()
                paper_id = f"local-baseline::{question_id}::{idx}"
                dataset_name = datasets[0] if datasets else question_id
                references.append(
                    PaperReference(
                        paper_id=paper_id,
                        title=provenance_title,
                        authors=["local_registry"],
                        abstract=f"Local canonical baseline entry for {question_id}.",
                        venue="local_registry",
                        url=provenance_uri,
                    )
                )
                if metric_name and metric.get("baseline_value") is not None:
                    extracted_metrics.append(
                        ExtractedMetric(
                            paper_id=paper_id,
                            paper_title=provenance_title,
                            dataset=dataset_name,
                            metric_name=metric_name,
                            value=metric.get("baseline_value"),
                            context=f"Local canonical baseline for {question_id}",
                            method_name=baseline_name,
                            higher_is_better=metric.get("higher_is_better"),
                        )
                    )
        else:
            coverage_warnings.append(
                "No canonical baseline registry entry matched this evaluation question; ideation used only the local manifest/task description."
            )

        if baselines and not extracted_metrics:
            extraction_notes += f" Declared baselines in task context: {', '.join(baselines[:6])}."

        evidence = EvidenceBundle(
            extracted_metrics=extracted_metrics,
            extraction_notes=extraction_notes,
            coverage_warnings=coverage_warnings,
        )
        return references, evidence, []

    async def _retrieve_baseline_evidence_openalex(
        self,
        topic: str,
    ) -> tuple[list[PaperReference], EvidenceBundle, list[str]]:
        """Lightweight baseline retrieval for eval-fast mode.

        Full literature search stays disabled, but we still try to retrieve a
        few OpenAlex papers tied to the declared baselines/datasets so planning
        can anchor baseline numbers to recent/public evidence.
        """
        search_oa = await _get_oa_search()
        if not search_oa:
            return [], EvidenceBundle(coverage_warnings=["OpenAlex unavailable in eval-fast mode"]), []

        baselines = self._extract_topic_list_field(topic, "Known Baselines")
        datasets = self._extract_topic_list_field(topic, "Evaluation Datasets")
        domain = self._extract_topic_field(topic, "Research Domain")
        problem_statement = self._extract_topic_field(topic, "Problem Statement")

        queries: list[str] = []
        all_papers: dict[str, dict[str, Any]] = {}

        for baseline in baselines[:6]:
            exact_query = f'"{baseline}"'
            if datasets:
                exact_query = f'{exact_query} {datasets[0]}'
            queries.append(exact_query)
            if domain:
                queries.append(f"{baseline} {domain}")

        if datasets:
            generic_candidates = [
                f"{datasets[0]} baseline {domain}".strip(),
                f"{datasets[0]} state of the art".strip(),
                f"{datasets[0]} benchmark".strip(),
            ]
            if "qa" in datasets[0].lower() or "question answering" in problem_statement.lower():
                generic_candidates.append(f"{datasets[0]} question answering".strip())
            for generic in generic_candidates:
                if generic and generic not in queries:
                    queries.append(generic)
        if problem_statement and datasets:
            task_hint = f"{datasets[0]} {problem_statement[:80]}".strip()
            if task_hint and task_hint not in queries:
                queries.append(task_hint)

        for query in queries[:10]:
            try:
                oa_results = await search_oa(query, max_results=5)
            except Exception as exc:
                logger.warning("[%s] OpenAlex baseline retrieval failed for '%s': %s", self.stage.value, query, exc)
                continue
            for paper in oa_results:
                key = self._dedup_key(paper)
                if key and key not in all_papers:
                    all_papers[key] = paper

        ranked = list(all_papers.values())
        ranked.sort(
            key=lambda paper: (
                int(paper.get("year") or 0),
                int(paper.get("citation_count") or 0),
            ),
            reverse=True,
        )
        ranked = ranked[:12]

        evidence = await self._extract_evidence(ranked) if ranked else EvidenceBundle()
        coverage_warnings = list(evidence.coverage_warnings)
        if ranked and not evidence.extracted_metrics:
            coverage_warnings.append(
                "OpenAlex baseline retrieval found papers, but no explicit quantitative baseline metrics were extracted from abstracts."
            )
        if not ranked:
            coverage_warnings.append("OpenAlex baseline retrieval found no candidate papers for the declared baselines.")
        evidence = EvidenceBundle(
            extracted_metrics=evidence.extracted_metrics,
            extraction_notes=evidence.extraction_notes,
            coverage_warnings=coverage_warnings,
        )

        references: list[PaperReference] = []
        for paper in ranked:
            try:
                references.append(
                    PaperReference(
                        paper_id=str(
                            paper.get("paper_id")
                            or paper.get("openalex_id")
                            or paper.get("arxiv_id")
                            or ""
                        ),
                        title=paper.get("title", ""),
                        authors=[str(author) for author in (paper.get("authors") or [])],
                        year=paper.get("year"),
                        abstract=paper.get("abstract", ""),
                        venue=paper.get("venue", ""),
                        citation_count=int(paper.get("citation_count", 0) or 0),
                        url=paper.get("url", ""),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed OpenAlex baseline paper: %s", exc)
        return references, evidence, queries

    async def run(self, **inputs: Any) -> dict[str, Any]:
        topic: str = inputs.get("topic", "")
        if not topic:
            raise ValueError("IdeationAgent requires a non-empty 'topic' in inputs")
        logger.info("[%s] Starting ideation for topic: %s", self.stage.value, topic)
        adaptive_context = self.build_adaptive_context(
            "literature",
            topic=topic,
            text=topic,
            tags=[topic, self.workspace.manifest.paper_mode.value],
        )
        retry_error = str(inputs.get("_retry_error", "")).strip()
        if retry_error:
            self.learn_from_trace(
                "literature",
                "ideation_retry",
                retry_error,
                tags=[topic, "retry", self.workspace.manifest.paper_mode.value],
            )

        if getattr(self.config, "ideation_disable_retrieval", False):
            self.log("Eval-fast mode: skipping all external literature retrieval and using local evaluation context only")
            output = await self._run_without_retrieval(topic, adaptive_context=adaptive_context)
            return self._persist_ideation_output(topic, output)

        # Check for cached search results (from a previous failed attempt)
        cache_path = self.workspace.path / "logs" / "ideation_search_cache.json"
        cached = None
        used_cache = False
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(cached, dict) or "papers" not in cached:
                    raise ValueError("invalid cache structure")
                self.log("Found cached search results from previous attempt, skipping search")
                used_cache = True
            except (json.JSONDecodeError, ValueError, OSError) as e:
                self.log(f"Search cache invalid ({e}), starting fresh")
                cached = None

        survey_paper_count = 0
        supplementary_rounds = 0
        if (cached is not None
                and isinstance(cached, dict)
                and isinstance(cached.get("queries"), list)
                and isinstance(cached.get("papers"), list)):
            queries = cached["queries"]
            papers = cached["papers"]
            survey_paper_count = sum(
                1 for paper in papers
                if "survey" in str(paper.get("title", "") or "").lower()
                or "review" in str(paper.get("title", "") or "").lower()
            )
            logger.info("[%s] Using cached: %d queries, %d papers",
                        self.stage.value, len(queries), len(papers))
            must_cites = cached.get("must_cites", [])
            if not must_cites:
                must_cites = await self._extract_must_cites(
                    [p for p in papers if "survey" in (p.get("title", "") or "").lower()
                     or "review" in (p.get("title", "") or "").lower()]
                )
        else:
            # Step 1: Generate search queries
            queries = await self._generate_queries(topic, adaptive_context=adaptive_context)
            logger.info("[%s] Generated %d search queries", self.stage.value, len(queries))

            # Step 2: Search literature
            papers = await self._search_literature(queries)
            logger.info("[%s] Retrieved %d papers", self.stage.value, len(papers))

            # Step 2b: Search for surveys and merge
            survey_papers = await self._search_surveys(topic)
            survey_paper_count = len(survey_papers)
            logger.info("[%s] Found %d survey papers", self.stage.value, len(survey_papers))
            existing_keys = {self._dedup_key(p) for p in papers}
            for sp in survey_papers:
                key = self._dedup_key(sp)
                if key and key not in existing_keys:
                    papers.append(sp)
                    existing_keys.add(key)

            # Step 2c: Rank and filter papers by citation quality
            papers = self._rank_and_filter_papers(papers, topic=topic)
            logger.info("[%s] After ranking/filtering: %d papers", self.stage.value, len(papers))

            # Step 2c2: Enrich papers from web/PwC with citation counts
            zero_cite = [p for p in papers if (p.get("citation_count", 0) or 0) == 0]
            if zero_cite:
                self.log(f"Enriching citation counts for {len(zero_cite)} papers")
                await self._enrich_citation_counts(zero_cite)
                papers = self._rank_and_filter_papers(papers, topic=topic)

            # Step 2c3: Citation graph expansion (snowball sampling)
            papers = await self._expand_via_citations(papers, top_k=5, max_new=15)
            logger.info("[%s] After citation expansion: %d papers", self.stage.value, len(papers))

            # Step 2d: Enrich top papers with full-text PDF reading
            papers = await self._enrich_with_full_text(papers)

            # Step 2d2: Search coverage self-evaluation (max 2 rounds)
            all_papers_dict = {self._dedup_key(p): p for p in papers}
            for _eval_round in range(2):
                coverage = await self._evaluate_search_coverage(topic, papers)
                score = coverage.get("coverage_score", 10)
                if score >= 8:
                    self.log(f"Search coverage: {score}/10 -- sufficient")
                    break
                missing = coverage.get("missing_directions", [])
                if not missing:
                    break
                supplementary_rounds += 1
                self.log(f"Search coverage: {score}/10 -- supplementing {len(missing)} directions")
                new_papers = await self._supplementary_search(missing, all_papers_dict)
                if new_papers:
                    papers.extend(new_papers)
                    for np in new_papers:
                        all_papers_dict[self._dedup_key(np)] = np
                    papers = self._rank_and_filter_papers(papers, topic=topic)
                    self.log(f"Added {len(new_papers)} papers from supplementary search")

            # Step 2e: Extract must-cite papers from surveys
            must_cites = await self._extract_must_cites(
                [p for p in papers if "survey" in (p.get("title", "") or "").lower()
                 or "review" in (p.get("title", "") or "").lower()]
            )
            if must_cites:
                logger.info("[%s] Identified %d must-cite papers",
                            self.stage.value, len(must_cites))
            else:
                must_cites = []

            # Cache search results for retry (including must_cites)
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"queries": queries, "papers": papers,
                                "must_cites": must_cites},
                               ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                self.log("Cached search results for potential retry")
            except Exception as e:
                logger.warning("Failed to cache search results: %s", e)

        # Step 3: LLM analysis -- gaps + idea candidates (with ReAct tool use)
        output = await self._analyze_and_hypothesize(
            topic, queries, papers, adaptive_context=adaptive_context
        )

        # Store must-cite titles and match to actual papers
        output.must_cites = must_cites
        if must_cites:
            mc_matches = self._match_must_cites_to_papers(must_cites, papers)
            output.must_cite_matches = mc_matches
            matched_count = sum(1 for m in mc_matches if m.get("matched"))
            self.log(f"Must-cite matching: {matched_count}/{len(must_cites)} matched to papers")

        # Step 4: Extract quantitative evidence from paper abstracts
        evidence = await self._extract_evidence(papers)
        output.evidence = evidence
        logger.info("[%s] Extracted %d metrics from literature",
                    self.stage.value, len(evidence.extracted_metrics))

        # Step 5: Search GitHub for reference implementations
        reference_repos = await self._search_github_repos(topic, queries)
        logger.info("[%s] Found %d reference GitHub repos",
                    self.stage.value, len(reference_repos))
        output.reference_repos = reference_repos
        self._record_retrieval_skill_summary(
            topic,
            queries=queries,
            papers=papers,
            survey_papers=survey_paper_count,
            supplementary_rounds=supplementary_rounds,
            must_cites=must_cites,
            evidence=evidence,
            reference_repos=reference_repos,
            used_cache=used_cache,
        )

        return self._persist_ideation_output(topic, output)

    def _persist_ideation_output(self, topic: str, output: IdeationOutput) -> dict[str, Any]:
        ideation_payload = add_idea_aliases_to_ideation(output.model_dump(mode="json"))
        output_path = self.workspace.write_json(
            "papers/ideation_output.json",
            ideation_payload,
        )
        self.workspace.register_artifact(
            "ideation_output", output_path, self.stage
        )
        gap_descriptions = [gap.description for gap in output.gaps[:3]]
        gap_summary = "; ".join(gap_descriptions)
        self.remember_context(
            MemoryType.PROJECT_CONTEXT,
            f"Ideation for {topic} selected {get_selected_idea_id(ideation_payload)} with rationale: {output.rationale}",
            importance=0.74,
            tags=[topic, "ideation", self.workspace.manifest.paper_mode.value],
            source="ideation_output",
            topic=topic,
        )
        if gap_summary:
            self.remember_context(
                MemoryType.DECISION_HISTORY,
                f"Key gaps for {topic}: {gap_summary}",
                importance=0.8,
                tags=[topic, "gaps", "literature"],
                source="ideation_output",
                topic=topic,
            )
        self.remember_promising_direction(
            topic=topic,
            ideation_output=ideation_payload,
            artifact_path="logs/promising_direction_summary_ideation.json",
            source_stage="ideation",
            source="ideation_output",
        )
        return ideation_payload

    async def _run_without_retrieval(self, topic: str, adaptive_context: str = "") -> IdeationOutput:
        baseline_papers, baseline_evidence, baseline_queries = await self._retrieve_local_eval_context(topic)
        adaptive_prefix = f"{adaptive_context}\n\n" if adaptive_context else ""
        prompt = f"""{adaptive_prefix}Research Topic:
{topic}

You are running in evaluation mode. Full literature review and GitHub retrieval are disabled.
Use the structured task context above to produce a compact ideation result for downstream planning.
If local baseline evidence was separately provided, treat it only as baseline context rather than a full literature survey.

Return valid JSON with:
- survey_summary: short literature-free framing of the task and constraints
- gaps: array with 2-4 items, each containing gap_id, description, supporting_refs, severity, quantitative_evidence, future_work_mention
- ideas or hypotheses: array with 2-4 idea candidates, each containing idea_id or hypothesis_id, statement, gap_refs, novelty_justification, feasibility_notes, closest_existing_work
- selected_idea or selected_hypothesis: one identifier from the list
- rationale: short explanation for the selection
- theme_clusters: optional array
- key_challenges: optional array
- future_directions: optional array

Rules:
- supporting_refs must be an empty list
- quantitative_evidence and future_work_mention must be empty strings unless directly implied by the topic text
- closest_existing_work may mention baselines named in the topic context
- Do not invent papers, URLs, metrics, or citations
- Keep the proposed ideas lightweight and reproducible when the task context asks for that

Return JSON only."""
        try:
            result = await self.generate_json(IDEATION_ANALYSIS_SYSTEM, prompt)
        except Exception as exc:
            logger.warning("[%s] No-retrieval ideation generation failed: %s", self.stage.value, exc)
            result = {}

        if not isinstance(result, dict):
            result = {}
        result = add_idea_aliases_to_ideation(result)

        hypotheses = result.get("hypotheses") if isinstance(result.get("hypotheses"), list) else []
        selected_hypothesis = str(result.get("selected_hypothesis") or "").strip()
        if not selected_hypothesis and hypotheses and isinstance(hypotheses[0], dict):
            selected_hypothesis = str(hypotheses[0].get("hypothesis_id") or "HYP-001")

        fallback = {
            "topic": topic,
            "search_queries": baseline_queries,
            "papers": [paper.model_dump(mode="json") for paper in baseline_papers],
            "survey_summary": str(
                result.get("survey_summary")
                or (
                    "Evaluation-mode ideation generated from local manifest context with local baseline registry support."
                    if baseline_papers
                    else "Evaluation-mode ideation generated from local manifest context without external retrieval."
                )
            ),
            "gaps": result.get("gaps") if isinstance(result.get("gaps"), list) else [
                {
                    "gap_id": "GAP-001",
                    "description": "Need a lightweight, reproducible method tailored to the stated task constraints.",
                    "supporting_refs": [],
                    "severity": "high",
                    "quantitative_evidence": "",
                    "future_work_mention": "",
                }
            ],
            "hypotheses": hypotheses or [
                {
                    "hypothesis_id": "HYP-001",
                    "statement": "A lightweight, ablatable method aligned with the provided baselines can improve the target task under the stated resource budget.",
                    "gap_refs": ["GAP-001"],
                    "novelty_justification": "Derived from the task constraints rather than external literature search.",
                    "feasibility_notes": "Designed for evaluation mode where downstream planning and implementation matter more than open-ended literature coverage.",
                    "closest_existing_work": "",
                }
            ],
            "selected_hypothesis": selected_hypothesis or "HYP-001",
            "rationale": str(result.get("rationale") or "Chosen for alignment with the explicit task constraints and downstream executability."),
            "evidence": baseline_evidence.model_dump(mode="json"),
            "reference_repos": [],
            "must_cites": [],
            "must_cite_matches": [],
            "theme_clusters": self._normalize_string_list_field(result.get("theme_clusters")),
            "key_challenges": self._normalize_string_list_field(result.get("key_challenges")),
            "future_directions": self._normalize_string_list_field(result.get("future_directions")),
        }
        return IdeationOutput.model_validate(fallback)

    async def _generate_queries(self, topic: str, adaptive_context: str = "") -> list[str]:
        adaptive_prefix = f"{adaptive_context}\n\n" if adaptive_context else ""
        prompt = f"""{adaptive_prefix}Given the research topic: "{topic}"

Generate {MAX_SEARCH_QUERIES} diverse search queries to find relevant academic papers.
Include queries for:
- Direct topic matches
- Related methods and techniques
- Benchmark datasets and evaluation approaches
- Recent surveys or reviews

Return JSON: {{"queries": ["query1", "query2", ...]}}"""

        try:
            result = await self.generate_json(IDEATION_SYSTEM_PROMPT, prompt)
            queries = result.get("queries", [])
            if queries:
                return queries
        except Exception as e:
            logger.warning("[%s] Query generation LLM call failed: %s", self.stage.value, e)
        # Fallback: use topic itself as a search query
        self.log("Using fallback queries derived from topic")
        return [topic, f"{topic} survey", f"{topic} benchmark"]

    def _dedup_key(self, paper: dict) -> str:
        """Return a deduplication key for a paper (prefer ID, fallback to title)."""
        for id_field in ("paper_id", "arxiv_id"):
            pid = (paper.get(id_field, "") or "").strip()
            if pid and pid != "unknown":
                return f"id:{pid}"
        return "title:" + (paper.get("title", "") or "").lower().strip()
