"""Citation management: BibTeX resolution, coverage checking, must-cite injection."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _CitationManagerMixin:
    """Mixin — citation management methods."""

    # Match \cite, \citet, \citep, \citeauthor, \citealp, \citealt, \Citet, etc.
    _CITE_KEY_RE = re.compile(r"\\[Cc]ite(?:t|p|author|year|alp|alt|num)?(?:\*)?(?:\[[^\]]*\])*\{([^}]+)\}")
    _BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)")


    async def _expand_citation_pool(
        self,
        ideation: dict,
        blueprint: dict,
        target_count: int = 28,
    ) -> dict:
        """Expand paper metadata with role-targeted OpenAlex searches.

        This is a paper-writing safeguard: full research papers need enough
        literature context, but measured result tables must remain local-run
        evidence only. The added papers are used for Introduction/Related Work
        citations and never as experimental measurements.
        """
        if not isinstance(ideation, dict):
            return ideation
        papers = ideation.setdefault("papers", [])
        if not isinstance(papers, list):
            papers = []
            ideation["papers"] = papers
        if len(papers) >= target_count:
            return ideation

        try:
            from mcp_server.tools.openalex import search_openalex
        except Exception as exc:
            self.log(f"Citation expansion skipped: OpenAlex unavailable ({exc})")
            return ideation

        def _as_text(value) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return " ".join(str(v) for v in value.values() if isinstance(v, (str, int, float)))
            if isinstance(value, list):
                return " ".join(_as_text(v) for v in value)
            return str(value or "")

        topic = str(ideation.get("topic") or blueprint.get("title") or "machine learning").strip()
        method = blueprint.get("proposed_method") if isinstance(blueprint, dict) else {}
        method_name = _as_text(method.get("name") if isinstance(method, dict) else "").strip()
        components = _as_text(method.get("key_components", []) if isinstance(method, dict) else [])
        datasets = _as_text(blueprint.get("datasets", []) if isinstance(blueprint, dict) else [])
        baselines = _as_text(blueprint.get("baselines", []) if isinstance(blueprint, dict) else [])

        role_queries: list[tuple[str, str]] = []
        base = topic or method_name or "machine learning"
        role_queries.extend([
            ("background", base),
            ("domain", f"{base} dataset benchmark evaluation"),
            ("method", f"{base} method architecture evaluation"),
            ("baseline", f"{base} baseline comparison"),
            ("evaluation_protocol", f"{base} reproducible evaluation protocol validation split"),
        ])
        if method_name:
            role_queries.append(("method", method_name))
        if components:
            role_queries.append(("method", f"{method_name} {components}".strip()))
        if datasets:
            role_queries.append(("domain", datasets[:240]))
        if baselines:
            role_queries.append(("baseline", baselines[:240]))

        # Broad fallback queries prevent thin reference lists when the initial
        # topic-specific search is too narrow.  These papers are citation
        # context only; they are not converted into measured baselines.
        broad_queries = [
            ("method", "machine learning method evaluation benchmark"),
            ("baseline", "baseline comparison machine learning experiment"),
            ("evaluation_protocol", "cross validation data leakage machine learning evaluation"),
            ("evaluation_protocol", "reproducible machine learning experimental protocol"),
            ("background", "reproducible machine learning experimental protocol"),
            ("background", "green AI efficient machine learning"),
        ]
        role_queries.extend(broad_queries)

        def _dedup_key(paper: dict) -> str:
            title = re.sub(r"\s+", " ", str(paper.get("title") or "").lower()).strip()
            year = str(paper.get("year") or "")
            if title:
                return f"{title}|{year}"
            return str(paper.get("url") or paper.get("doi") or "").lower()

        existing = {_dedup_key(p) for p in papers if isinstance(p, dict)}
        roles: dict[str, list[str]] = ideation.setdefault("citation_roles", {})
        if not isinstance(roles, dict):
            roles = {}
            ideation["citation_roles"] = roles

        added = 0
        for role, query in role_queries:
            if len(papers) >= target_count:
                break
            query = re.sub(r"\s+", " ", query).strip()
            if not query:
                continue
            try:
                results = await search_openalex(query, max_results=12)
            except Exception as exc:
                self.log(f"Citation expansion query failed [{role}]: {exc}")
                continue
            for paper in results or []:
                if len(papers) >= target_count:
                    break
                if not isinstance(paper, dict) or not paper.get("title"):
                    continue
                key = _dedup_key(paper)
                if not key or key in existing:
                    continue
                paper = dict(paper)
                paper.setdefault("citation_role", role)
                papers.append(paper)
                existing.add(key)
                added += 1
            roles.setdefault(role, [])

        # Backfill role labels for existing papers so injection can group them.
        for idx, paper in enumerate(papers):
            if not isinstance(paper, dict):
                continue
            role = str(paper.get("citation_role") or "background")
            roles.setdefault(role, [])
            marker = str(idx)
            if marker not in roles[role]:
                roles[role].append(marker)

        if added:
            self.log(f"Citation expansion added {added} OpenAlex papers (pool={len(papers)})")
        else:
            self.log(f"Citation expansion found no new papers (pool={len(papers)})")
        return ideation

    def _ensure_minimum_citations(
        self,
        latex: str,
        ideation: dict,
        cite_keys: dict[int, str],
        min_refs: int = 20,
    ) -> str:
        """Inject concise paper-facing related-work citations until min_refs.

        The injected text is literature context only. It does not introduce
        experimental numbers and does not affect measured-result tables.
        """
        cited: set[str] = set()
        for match in self._CITE_KEY_RE.finditer(latex):
            for key in match.group(1).split(","):
                key = key.strip()
                if key:
                    cited.add(key)
        if len(cited) >= min_refs:
            return latex

        papers = ideation.get("papers", []) if isinstance(ideation, dict) else []
        if not isinstance(papers, list):
            return latex

        role_to_keys: dict[str, list[str]] = {}
        for idx, paper in enumerate(papers):
            if idx not in cite_keys or not isinstance(paper, dict):
                continue
            key = cite_keys[idx]
            if key in cited:
                continue
            role = str(paper.get("citation_role") or "background")
            role_to_keys.setdefault(role, []).append(key)

        role_order = [
            "background", "domain", "feature_selection", "method",
            "evolutionary_optimization", "interpretability", "baseline",
            "evaluation_protocol",
        ]
        selected_by_role: dict[str, list[str]] = {}
        needed = max(0, min_refs - len(cited))
        # First give each role a small presence, then keep cycling through all
        # available roles until the paper text really cites the requested count.
        # The old one-pass cap at four papers per role could leave the final
        # manuscript with far fewer cited references than the expanded BibTeX pool.
        while needed > 0:
            progressed = False
            for role in role_order:
                keys = role_to_keys.get(role, [])
                if not keys:
                    continue
                key = keys.pop(0)
                selected_by_role.setdefault(role, []).append(key)
                needed -= 1
                progressed = True
                if needed <= 0:
                    break
            if not progressed:
                break
        if needed > 0:
            for role, keys in role_to_keys.items():
                if needed <= 0:
                    break
                if role in role_order:
                    continue
                while keys and needed > 0:
                    selected_by_role.setdefault(role, []).append(keys.pop(0))
                    needed -= 1

        if not selected_by_role:
            return latex

        role_phrases = {
            "background": "reproducible machine-learning and scientific-computing protocols",
            "domain": "domain-specific benchmark studies",
            "feature_selection": "feature-selection and sparse-modeling evaluations",
            "method": "methodologically related model-design studies",
            "evolutionary_optimization": "multi-objective and evolutionary optimization work",
            "interpretability": "interpretable and explainable-learning research",
            "baseline": "classical and modern baseline comparisons",
            "evaluation_protocol": "evaluation-protocol studies on leakage-safe validation",
        }
        clauses: list[str] = []
        for role in role_order:
            keys = selected_by_role.get(role)
            if not keys:
                continue
            cite = ",".join(keys)
            phrase = role_phrases.get(role, "related studies")
            clauses.append(f"{phrase}~\\citep{{{cite}}}")
        if not clauses:
            return latex
        if len(clauses) == 1:
            injection = (
                "This evaluation framing follows "
                + clauses[0]
                + ", treating transparent assumptions and comparable evaluation boundaries as part of the research contribution rather than as implementation details.\n"
            )
        else:
            first = "; ".join(clauses[:2])
            rest = "; ".join(clauses[2:4])
            injection = (
                "This evaluation framing follows "
                + first
                + ", treating transparent assumptions and comparable evaluation boundaries as part of the research contribution rather than as implementation details."
            )
            if rest:
                injection += (
                    " The same motivation is reinforced by "
                    + rest
                    + ", which motivates reporting benchmark design, preprocessing boundaries, and model complexity together with predictive scores."
                )
            injection += "\n"

        pattern = re.compile(
            r"(\\section\{(?:Related Works?|Prior Work|Literature Review|Background(?:\s+and\s+Related\s+Work)?)\}.*?)(?=\n\\section\{)",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(latex)
        if match:
            insert_pos = match.end(1)
            latex = latex[:insert_pos] + "\n\n" + injection + latex[insert_pos:]
        else:
            intro = re.search(r"(\\section\{Introduction\}.*?)(?=\n\\section\{)", latex, flags=re.DOTALL)
            if intro:
                insert_pos = intro.end(1)
                latex = latex[:insert_pos] + "\n\n" + injection + latex[insert_pos:]
            else:
                latex += "\n" + injection
        self.log(f"Merged extra related-work citations to reach >= {min_refs} citations")
        return latex

    async def _resolve_missing_citations(
        self, latex: str, bibtex: str
    ) -> str:
        """Find \\cite keys in LaTeX that are missing from the bib, and fill them.

        Strategy:
        1. Extract all cited keys from LaTeX.
        2. Extract all defined keys from bibtex.
        3. For each missing key, search Semantic Scholar by the key pattern
           (e.g. 'gu2022' → search 'Gu 2022') to find the real paper.
        4. If search fails, generate a stub entry so LaTeX compiles without [?].

        Returns the updated bibtex string with new entries appended.
        """
        # 1. Collect cited keys
        cited: set[str] = set()
        for m in self._CITE_KEY_RE.finditer(latex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cited.add(k)

        # 2. Collect bib keys
        defined: set[str] = set()
        for m in self._BIB_KEY_RE.finditer(bibtex):
            defined.add(m.group(1).strip())

        missing = cited - defined
        if not missing:
            return bibtex

        self.log(f"Resolving {len(missing)} missing citation(s): {sorted(missing)}")

        # 3. Try to resolve each missing key (skip if already added by a prior call)
        new_entries: list[str] = []
        for key in sorted(missing):
            # Double-check key isn't already in bibtex (guards against duplicate calls)
            if re.search(r'@\w+\s*\{\s*' + re.escape(key) + r'\s*,', bibtex):
                continue
            entry = await self._resolve_single_citation(key)
            new_entries.append(entry)

        # Append new entries to bibtex
        if new_entries:
            bibtex = bibtex.rstrip() + "\n\n" + "\n".join(new_entries)
            self.log(f"Added {len(new_entries)} bib entries")

        return bibtex

    async def _resolve_single_citation(self, key: str) -> str:
        """Resolve a single missing citation key to a bib entry.

        Parses the key pattern (e.g., 'gu2022', 'child2019b',
        'beltagy2020longformer') to extract author surname, year, and
        optional method/paper name, then searches OpenAlex.
        Falls back to a stub entry if search fails.
        """
        # Parse key: letters = surname, digits = year, optional trailing letters (method name)
        m = re.match(r"([a-z]+)(\d{4})([a-z]*)$", key, re.IGNORECASE)
        if m:
            surname = m.group(1).capitalize()
            year = m.group(2)
            method_hint = m.group(3)  # e.g. "longformer", "bigbird", "b"
            # Build search query: include method name if it's more than a
            # single disambiguator letter (e.g. "longformer" but not "b")
            if len(method_hint) > 1:
                query = f"{surname} {year} {method_hint}"
            else:
                query = f"{surname} {year}"
        else:
            # Unusual key format — use as-is for search
            surname = key
            year = ""
            method_hint = ""
            query = key

        # Try OpenAlex search (free, no API key application needed)
        try:
            from mcp_server.tools.openalex import search_openalex
            results = await search_openalex(query, max_results=5)
            # Find best match with prioritized matching
            best = None
            # Priority 1: year + author match + method hint in title
            for r in results:
                r_year = str(r.get("year", ""))
                r_authors = " ".join(r.get("authors", []))
                r_title = (r.get("title") or "").lower()
                if (year and r_year == year
                        and surname.lower() in r_authors.lower()):
                    if method_hint and len(method_hint) > 1:
                        if method_hint.lower() in r_title:
                            best = r
                            break
                    else:
                        best = r
                        break
            # Priority 2: year + author match (without method hint check)
            if not best:
                for r in results:
                    r_year = str(r.get("year", ""))
                    r_authors = " ".join(r.get("authors", []))
                    if year and r_year == year and surname.lower() in r_authors.lower():
                        best = r
                        break
            # Priority 3: year + method hint in title (author name may differ)
            if not best and method_hint and len(method_hint) > 1:
                for r in results:
                    r_year = str(r.get("year", ""))
                    r_title = (r.get("title") or "").lower()
                    if year and r_year == year and method_hint.lower() in r_title:
                        best = r
                        break
            # Priority 4: year match only (no blind first-result fallback)
            if not best and results:
                for r in results:
                    if year and str(r.get("year", "")) == year:
                        best = r
                        break

            if best:
                authors = best.get("authors", [])
                author_str = " and ".join(authors[:5]) if authors else surname
                title = best.get("title", "Unknown")
                venue = best.get("venue", "") or "arXiv preprint"
                r_year = best.get("year", year or 2024)
                entry_type = self._detect_entry_type(venue)
                venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
                return (
                    f"@{entry_type}{{{key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{author_str}}},\n"
                    f"  year={{{r_year}}},\n"
                    f"  {venue_field}={{{venue}}},\n"
                    f"}}\n"
                )
        except Exception as exc:
            logger.debug("OpenAlex search failed for citation key '%s': %s", key, exc)

        # BUG-22 fix: instead of a fake stub with title={Surname et al.},
        # generate an honest @misc entry without a fabricated title.
        self.log(f"  Unresolved entry for '{key}' (search failed)")
        return (
            f"@misc{{{key},\n"
            f"  author={{{surname}}},\n"
            f"  year={{{year or 2024}}},\n"
            f"  note={{Could not retrieve full metadata. "
            f"Please replace with the correct reference.}},\n"
            f"}}\n"
        )

    def _cleanup_unused_bibtex(self, latex: str, bibtex: str) -> str:
        """Remove BibTeX entries that are not cited anywhere in the LaTeX source.

        This prevents the .bib file from accumulating unused entries
        collected during ideation but never referenced in the final paper.
        """
        # 1. Collect all cited keys from LaTeX
        cited: set[str] = set()
        for m in self._CITE_KEY_RE.finditer(latex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cited.add(k)

        if not cited:
            return bibtex  # No citations at all — don't touch bib

        # 2. Parse bib entries and filter
        # Match @type{key, ... } blocks
        entry_re = re.compile(
            r'(@\w+\s*\{)\s*([^,\s]+)\s*,(.*?)(?=\n@|\Z)',
            re.DOTALL,
        )
        kept: list[str] = []
        removed_count = 0
        for m in entry_re.finditer(bibtex):
            key = m.group(2).strip()
            if key in cited:
                kept.append(m.group(0).rstrip())
            else:
                removed_count += 1

        if removed_count > 0:
            self.log(f"Removed {removed_count} unused BibTeX entries (kept {len(kept)})")
            return "\n\n".join(kept) + "\n"
        return bibtex

    # ---- citation coverage validation ----------------------------------------

    def _validate_citation_coverage(
        self, latex: str, ideation: dict, cite_keys: dict[int, str]
    ) -> dict:
        """Validate citation quality of the written paper.

        Checks:
        1. Total citation count
        2. Must-cite papers referenced
        3. Citation distribution across sections
        4. High-cited papers coverage
        """
        # 1. Extract all cited keys
        cited: set[str] = set()
        for m in self._CITE_KEY_RE.finditer(latex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cited.add(k)

        total_citations = len(cited)

        # 2. Check must-cite coverage
        must_cites = ideation.get("must_cites", [])
        must_cite_matches = ideation.get("must_cite_matches", [])
        papers = ideation.get("papers", [])

        missing_must_cites: list[dict] = []
        cited_must_cites: list[str] = []
        for mc in must_cite_matches:
            idx = mc.get("paper_index")
            matched = mc.get("matched", False)
            title = mc.get("title", "")
            if matched and idx is not None and idx in cite_keys:
                key = cite_keys[idx]
                if key in cited:
                    cited_must_cites.append(title)
                else:
                    missing_must_cites.append({"title": title, "cite_key": key, "paper_index": idx})
            else:
                # Unmatched must-cite — can't check, but flag it
                missing_must_cites.append({"title": title, "cite_key": None, "paper_index": None})

        # 3. Citation distribution by section
        section_cites: dict[str, int] = {}
        # Split by \section
        section_pattern = re.compile(r'\\section\{((?:[^{}]|\{[^{}]*\})+)\}')
        parts = section_pattern.split(latex)
        for i in range(1, len(parts), 2):
            sec_name = parts[i] if i < len(parts) else "Unknown"
            sec_content = parts[i + 1] if i + 1 < len(parts) else ""
            sec_cited = set()
            for m in self._CITE_KEY_RE.finditer(sec_content):
                for k in m.group(1).split(","):
                    k = k.strip()
                    if k:
                        sec_cited.add(k)
            section_cites[sec_name] = len(sec_cited)

        # 4. High-cited papers coverage
        high_cited_keys = set()
        for i, p in enumerate(papers):
            if not isinstance(p, dict):
                continue
            if (p.get("citation_count", 0) or 0) >= 100 and i in cite_keys:
                high_cited_keys.add(cite_keys[i])

        cited_high = high_cited_keys & cited
        uncited_high = high_cited_keys - cited

        return {
            "total_citations": total_citations,
            "must_cite_total": len(must_cites),
            "must_cite_found": len(cited_must_cites),
            "missing_must_cites": missing_must_cites,
            "cited_must_cites": cited_must_cites,
            "section_cites": section_cites,
            "high_cited_total": len(high_cited_keys),
            "high_cited_referenced": len(cited_high),
            "uncited_high_keys": sorted(uncited_high),
        }

    def _inject_must_cites(
        self, latex: str, missing: list[dict],
        cite_keys: dict[int, str], ideation: dict,
    ) -> str:
        """Inject missing must-cite papers into the Related Work section.

        Adds a brief sentence for each must-cite paper that was identified
        but not referenced in the paper.
        """
        papers = ideation.get("papers", [])

        # Build injection lines
        inject_lines = []
        for mc in missing:
            key = mc.get("cite_key")
            idx = mc.get("paper_index")
            if not key:
                continue  # Can't inject without a cite key

            # Get paper title for context
            title = mc.get("title", "")
            if idx is not None and idx < len(papers):
                p = papers[idx]
                if isinstance(p, dict):
                    title = p.get("title", title)

            inject_lines.append(f"\\citet{{{key}}} is also relevant to this line of research.")

        if not inject_lines:
            return latex

        injection = "\n".join(inject_lines)

        # Find the Related Work section and inject before its end
        # Handle common heading variants (Related Works, Prior Work, etc.)
        rw_pattern = re.compile(
            r'(\\section\{(?:Related Works?|Prior Work|Literature Review'
            r'|Background(?:\s+and\s+Related\s+Work)?)\}.*?)'
            r'(\\section\{)',
            re.DOTALL | re.IGNORECASE,
        )
        m = rw_pattern.search(latex)
        if m:
            # Insert before next \section
            insert_pos = m.start(2)
            latex = (
                latex[:insert_pos]
                + "\n\n" + injection + "\n\n"
                + latex[insert_pos:]
            )
            self.log(f"Injected {len(inject_lines)} must-cite references into Related Work")
        else:
            # Fallback: try inserting after Introduction
            intro_pattern = re.compile(
                r'(\\section\{Introduction\}.*?)(\\section\{)',
                re.DOTALL
            )
            m = intro_pattern.search(latex)
            if m:
                insert_pos = m.start(2)
                latex = (
                    latex[:insert_pos]
                    + "\n\n" + injection + "\n\n"
                    + latex[insert_pos:]
                )
                self.log(f"Injected {len(inject_lines)} must-cite references after Introduction")

        return latex

    def _log_citation_report(self, report: dict) -> None:
        """Log a citation quality report."""
        self.log("=== Citation Quality Report ===")
        self.log(f"  Total unique citations: {report.get('total_citations', 0)}")
        self.log(f"  Must-cite coverage: {report.get('must_cite_found', 0)}/{report.get('must_cite_total', 0)}")
        self.log(f"  High-cited papers referenced: {report.get('high_cited_referenced', 0)}/{report.get('high_cited_total', 0)}")

        section_cites = report.get("section_cites", {})
        if section_cites:
            self.log("  Per-section citations:")
            for sec, count in section_cites.items():
                self.log(f"    {sec}: {count}")

        uncited_high = report.get("uncited_high_keys", [])
        if uncited_high:
            self.log(f"  Uncited high-cited papers: {', '.join(uncited_high[:10])}")

        missing = report.get("missing_must_cites", [])
        if missing:
            self.log(f"  Missing must-cites: {[m['title'][:50] for m in missing[:5]]}")

        # Save report to workspace
        self.save_log("citation_quality_report.json",
                      json.dumps(report, indent=2, ensure_ascii=False, default=str))
        self.log("=== End Citation Report ===")

