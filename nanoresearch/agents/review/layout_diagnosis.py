"""PDF layout diagnosis mixin — calls the external latex-float-optimizer
tool to produce a structured rule-violation report after PDF compile.

Default-OFF: the tool is invoked only when env var
``NANORESEARCH_LATEX_OPTIMIZER_PATH`` points at the optimizer's ``main.py``.
Any failure (missing tool, timeout, non-zero exit) is logged and silently
swallowed — never blocks the review pipeline.

Output: ``drafts/pdf_diagnosis.json`` (registered as artifact ``pdf_diagnosis``).
Schema is the optimizer's v1.0 contract — see latex-float-optimizer/main.py.

Deployment note: the optimizer uses DocLayout-YOLO via huggingface_hub,
which by default issues an online version check on each run. Once the
model is cached locally, set ``HF_HUB_OFFLINE=1`` in the NanoResearch
environment so the subprocess uses the cache and survives flaky networks.
First-time use needs network to download the ~50 MB checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV_VAR = "NANORESEARCH_LATEX_OPTIMIZER_PATH"
_TIMEOUT_SECONDS = 240  # YOLO on CPU: ~3s/page × 10 pages + startup buffer

# Map optimizer rule_id -> (issue_type, severity).
# issue_type stays consistent with existing review schema vocabulary
# (ConsistencyIssue.issue_type free-form string).
_RULE_TO_ISSUE: dict[str, tuple[str, str]] = {
    "H1": ("orphan_float", "high"),
    "H2": ("dangling_ref", "high"),
    "H3": ("float_far_from_ref", "medium"),
    "S4": ("float_wrong_section", "medium"),
    "S5": ("page_too_many_floats", "low"),
}


class _LayoutDiagnosisMixin:
    """Mixin — invoke external latex-float-optimizer to emit pdf_diagnosis.json."""

    async def _run_layout_diagnosis(
        self, pdf_path: str | Path, tex_path: str | Path
    ) -> Optional[dict]:
        """Run the external optimizer and copy its diagnosis.json into the workspace.

        Returns the parsed diagnosis dict on success, or None when skipped/failed.
        """
        optimizer_main = os.environ.get(_ENV_VAR)
        if not optimizer_main:
            logger.debug(
                "Layout diagnosis skipped (%s not set). "
                "Set it to latex-float-optimizer/main.py to enable.",
                _ENV_VAR,
            )
            return None

        optimizer_path = Path(optimizer_main)
        if not optimizer_path.exists():
            self.log(
                f"Layout diagnosis: {_ENV_VAR}={optimizer_main} does not exist; skipping"
            )
            return None

        pdf_path = Path(pdf_path)
        tex_path = Path(tex_path)
        if not pdf_path.exists() or not tex_path.exists():
            self.log(
                "Layout diagnosis: pdf or tex missing "
                f"(pdf={pdf_path.exists()}, tex={tex_path.exists()}); skipping"
            )
            return None

        # Write directly into workspace so we avoid a copy step.
        out_path = self.workspace.path / "drafts" / "pdf_diagnosis.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(optimizer_path),
            "layout",
            str(pdf_path),
            "--tex", str(tex_path),
            "--json-out", str(out_path),
        ]
        self.log(f"Layout diagnosis: invoking {optimizer_path.name} ...")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(optimizer_path.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                self.log(
                    f"Layout diagnosis: timed out after {_TIMEOUT_SECONDS}s; skipping"
                )
                return None
        except Exception as e:
            self.log(f"Layout diagnosis: subprocess launch failed ({e}); skipping")
            return None

        if proc.returncode != 0:
            tail = (stderr_b or b"").decode("utf-8", errors="replace").strip()[-400:]
            self.log(
                f"Layout diagnosis: optimizer exited rc={proc.returncode}; "
                f"stderr tail: {tail or '(empty)'}"
            )
            return None

        if not out_path.exists():
            self.log("Layout diagnosis: optimizer succeeded but no JSON produced")
            return None

        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.log(f"Layout diagnosis: cannot parse {out_path.name} ({e})")
            return None

        rule_check = data.get("rule_check", {}) or {}
        summary = rule_check.get("summary", {}) or {}
        hard = summary.get("hard_violations_count", 0)
        soft = summary.get("soft_violations_count", 0)
        total = summary.get("violations_count", 0)
        self.log(
            f"Layout diagnosis: {total} violation(s) "
            f"({hard} hard, {soft} soft) — see drafts/pdf_diagnosis.json"
        )

        try:
            self.workspace.register_artifact(
                "pdf_diagnosis",
                out_path,
                self.stage,
            )
        except Exception as e:
            logger.debug("register_artifact(pdf_diagnosis) failed: %s", e)

        return data

    def _apply_diagnosis_to_review(
        self, diagnosis: Optional[dict[str, Any]], review: Any
    ) -> int:
        """Project diagnosis violations into review.consistency_issues.

        - ``diagnosis`` is the dict returned by ``_run_layout_diagnosis``
          (None if diagnosis was skipped/failed; treated as no-op).
        - ``review`` is the ReviewOutput pydantic model whose
          ``consistency_issues`` list is appended in-place.
        - De-dupes against existing entries by ``(issue_type, dedup_key)``
          where ``dedup_key`` is ``float_label`` if available else
          ``f"P{page_num}"``. This avoids double-reporting H1 when A-5d's
          post-revision orphan check has already flagged the same float.

        Returns the number of issues actually appended (post-dedup).
        """
        if not diagnosis:
            return 0

        # Lazy import to avoid a circular dep at module load.
        from nanoresearch.schemas.review import ConsistencyIssue

        violations = (
            (diagnosis.get("rule_check") or {}).get("violations") or []
        )
        if not violations:
            return 0

        def _key(issue_type: str, float_label: Optional[str], page_num: Any) -> tuple:
            tail = float_label if float_label else (f"P{page_num}" if page_num else "")
            return (issue_type, tail)

        # Build the existing-issue key set. A-5d uses
        # issue_type="orphan_float" with the orphan label embedded in the
        # description (e.g. "Orphan figure 'fig:foo' ..."), not as a
        # structured field — so for orphan_float we also scan descriptions.
        existing_keys: set[tuple] = set()
        for ci in getattr(review, "consistency_issues", []) or []:
            it = getattr(ci, "issue_type", "") or ""
            existing_keys.add((it, ""))  # coarse fallback
            if it == "orphan_float":
                desc = getattr(ci, "description", "") or ""
                # match labels like fig:xxx or tab:xxx
                import re as _re
                for m in _re.finditer(r"(fig|tab):[A-Za-z0-9_\-]+", desc):
                    existing_keys.add(("orphan_float", m.group(0)))

        appended = 0
        for v in violations:
            rid = v.get("rule_id", "")
            mapped = _RULE_TO_ISSUE.get(rid)
            if mapped is None:
                # Unknown rule id from a future optimizer schema — skip
                # silently rather than guess severity.
                continue
            issue_type, severity = mapped
            float_label = v.get("float_label")
            page_num = v.get("page_num")
            key = _key(issue_type, float_label, page_num)
            if key in existing_keys:
                continue

            locations: list[str] = []
            if page_num is not None:
                locations.append(f"PDF P{page_num}")
            if float_label:
                locations.append(float_label)

            description = v.get("detail") or v.get("rule_name") or rid
            suggestion = v.get("suggestion")
            if suggestion:
                description = f"{description} | suggestion: {suggestion}"

            # Day 4 S4: split the rid tag by whether float_rules emitted
            # a strict three-way violation ([strict-3way]) or a degraded
            # two-way one ([downgraded]/[downgraded-inference]). This
            # lets downstream consumers grep the description to separate
            # "semantic mismatch" from "missing-comment fallback" without
            # a new issue_type (dedup key stays (issue_type, float_label)).
            tag = rid
            if rid == "S4":
                if "[strict-3way]" in description:
                    tag = "S4-strict"
                elif "[downgraded" in description:  # matches [downgraded] and [downgraded-inference]
                    tag = "S4-degraded"

            review.consistency_issues.append(
                ConsistencyIssue(
                    issue_type=issue_type,
                    description=f"[layout/{tag}] {description}",
                    locations=locations,
                    severity=severity,
                )
            )
            existing_keys.add(key)
            appended += 1

        if appended:
            self.log(
                f"Layout diagnosis: appended {appended} consistency issue(s) "
                f"from PDF rule check"
            )
        return appended
