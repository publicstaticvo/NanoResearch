"""Optional paper-polish pass for the writing stage.

This module keeps the core WritingAgent focused on content generation while
adding a small, deterministic camera-ready quality gate when explicitly enabled.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FORBIDDEN_TERMS = [
    "TODO",
    "PLACEHOLDER",
    "illustrative",
    "representative case study",
    "not fully trained",
    "pilot",
]


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def paper_polish_enabled(config: Any) -> bool:
    """Return whether paper polish should run for this writing invocation."""
    return bool(getattr(config, "writing_paper_polish_enabled", False)) or _env_truthy(
        "NANO_WRITING_PAPER_POLISH"
    )


def target_page_range(config: Any) -> tuple[int, int]:
    """Return the desired inclusive PDF page range, with 0 meaning unspecified."""
    min_pages = int(getattr(config, "writing_polish_target_pages_min", 0) or 0)
    max_pages = int(getattr(config, "writing_polish_target_pages_max", 0) or 0)
    return min_pages, max_pages


def section_polish_guidance(label: str, config: Any) -> str:
    """Return extra section-writing guidance for the optional polish mode."""
    if not paper_polish_enabled(config):
        return ""

    min_pages, max_pages = target_page_range(config)
    page_goal = ""
    if min_pages and max_pages:
        page_goal = f"Target a final compiled paper of about {min_pages}-{max_pages} pages. "
    elif min_pages:
        page_goal = f"Target a final compiled paper of at least {min_pages} pages. "

    common = (
        "\n\n=== PAPER POLISH STYLE CONTRACT ===\n"
        f"{page_goal}Write in a dense conference-paper style, not as a short report. "
        "Keep claims concrete and evidence-facing. Avoid placeholder or softening phrases such as "
        "TODO, placeholder, illustrative, pilot, or not fully trained. "
        "Place technical and empirical substance in Method and Experiments rather than in a long Discussion.\n"
    )
    if label == "sec:method":
        return common + (
            "For Method, add enough concrete mechanism detail: formal notation, component rationale, "
            "implementation interface, complexity/resource discussion, and why simpler alternatives were rejected. "
            "Reference the architecture figure without letting the figure replace the method explanation.\n"
            "=== END PAPER POLISH STYLE CONTRACT ==="
        )
    if label == "sec:experiments":
        return common + (
            "For Experiments, include dataset protocol, baselines, matched training settings, main results, "
            "ablation interpretation, resource trade-off, diagnostics, and failure/error analysis. "
            "If more length is needed, expand experimental protocol and ablation interpretation, not generic discussion.\n"
            "=== END PAPER POLISH STYLE CONTRACT ==="
        )
    if label == "sec:related":
        return common + (
            "For Related Work, organize by technical themes and ensure the paper has enough citation density for a "
            "conference submission.\n"
            "=== END PAPER POLISH STYLE CONTRACT ==="
        )
    if label == "sec:conclusion":
        return common + (
            "For Conclusion, keep it compact and do not introduce new claims. If a Discussion section exists in the "
            "template or generated text, it must be at most one paragraph.\n"
            "=== END PAPER POLISH STYLE CONTRACT ==="
        )
    return common + "=== END PAPER POLISH STYLE CONTRACT ==="


def postprocess_latex_for_polish(latex: str, config: Any) -> str:
    """Apply deterministic LaTeX cleanup used by paper-polish mode."""
    if not paper_polish_enabled(config):
        return latex
    max_discussion = int(getattr(config, "writing_polish_discussion_max_paragraphs", 1) or 1)
    if max_discussion > 0:
        latex = _truncate_discussion_paragraphs(latex, max_discussion)
    return latex


def _truncate_discussion_paragraphs(latex: str, max_paragraphs: int) -> str:
    pattern = re.compile(
        r"(\\section\{Discussion\}(?:\\label\{[^}]+\})?\s*)(.*?)(?=\\section\{)",
        re.DOTALL,
    )

    def repl(match: re.Match) -> str:
        prefix, body = match.group(1), match.group(2)
        chunks = [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
        if len(chunks) <= max_paragraphs:
            return match.group(0)
        return prefix + "\n" + "\n\n".join(chunks[:max_paragraphs]) + "\n\n"

    return pattern.sub(repl, latex)


def build_polish_report(
    *,
    tex_path: Path,
    bib_path: Path,
    pdf_path: Path | None,
    config: Any,
) -> dict[str, Any]:
    """Validate polished paper artifacts and return a machine-readable report."""
    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    bib = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""
    min_pages, max_pages = target_page_range(config)
    page_count = _pdf_page_count(pdf_path) if pdf_path else None
    references_count = len(re.findall(r"@\w+\s*\{", bib))
    min_refs = int(getattr(config, "writing_polish_min_references", 0) or 0)
    required_sentences = list(getattr(config, "writing_polish_required_sentences", []) or [])
    forbidden_terms = list(getattr(config, "writing_polish_forbidden_terms", []) or DEFAULT_FORBIDDEN_TERMS)
    discussion_paragraphs = _discussion_paragraph_count(tex)

    report = {
        "enabled": True,
        "tex_path": str(tex_path),
        "bib_path": str(bib_path),
        "pdf_path": str(pdf_path) if pdf_path else "",
        "page_count": page_count,
        "target_pages_min": min_pages,
        "target_pages_max": max_pages,
        "page_count_ok": _page_count_ok(page_count, min_pages, max_pages),
        "references_count": references_count,
        "min_references": min_refs,
        "references_ok": references_count >= min_refs if min_refs else True,
        "discussion_paragraphs": discussion_paragraphs,
        "discussion_max_paragraphs": int(getattr(config, "writing_polish_discussion_max_paragraphs", 1) or 1),
        "discussion_ok": _discussion_ok(discussion_paragraphs, config),
        "required_sentences_present": {s: s in tex for s in required_sentences},
        "forbidden_terms_found": [t for t in forbidden_terms if re.search(re.escape(t), tex, re.IGNORECASE)],
    }
    report["required_sentences_ok"] = all(report["required_sentences_present"].values())
    report["forbidden_terms_ok"] = not report["forbidden_terms_found"]
    report["ok"] = all(
        bool(report[k])
        for k in [
            "page_count_ok",
            "references_ok",
            "discussion_ok",
            "required_sentences_ok",
            "forbidden_terms_ok",
        ]
    )
    return report


def write_polish_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_export_figure1_pdf(workspace_path: Path, tex_path: Path, config: Any) -> str:
    """Optionally export the first architecture/figure image as a standalone PDF."""
    if not bool(getattr(config, "writing_polish_export_figure1_pdf", False)):
        return ""
    try:
        from PIL import Image
    except Exception:
        return ""

    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    candidates = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    if not candidates:
        return ""
    preferred = None
    for name in candidates:
        lowered = name.lower()
        if any(k in lowered for k in ("architecture", "overview", "framework", "figure1")):
            preferred = name
            break
    preferred = preferred or candidates[0]
    src = _resolve_figure_path(workspace_path, preferred)
    if not src or not src.exists() or src.suffix.lower() == ".pdf":
        return str(src) if src else ""
    out = workspace_path / "drafts" / "figure1.pdf"
    try:
        im = Image.open(src).convert("RGB")
        im.save(out, "PDF", resolution=300.0)
    except Exception:
        return ""
    return str(out)


def _resolve_figure_path(workspace_path: Path, filename: str) -> Path | None:
    raw = Path(filename)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            workspace_path / "drafts" / raw,
            workspace_path / "figures" / raw,
            workspace_path / raw,
        ])
    for c in candidates:
        if c.exists():
            return c
    stem = raw.stem
    for base in [workspace_path / "drafts", workspace_path / "figures"]:
        if not base.exists():
            continue
        for ext in (".pdf", ".png", ".jpg", ".jpeg"):
            found = base / f"{stem}{ext}"
            if found.exists():
                return found
    return None


def _pdf_page_count(pdf_path: Path | None) -> int | None:
    if not pdf_path or not pdf_path.exists() or not shutil.which("pdfinfo"):
        return None
    try:
        proc = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    match = re.search(r"^Pages:\s*(\d+)", proc.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def _page_count_ok(page_count: int | None, min_pages: int, max_pages: int) -> bool:
    if page_count is None:
        return not min_pages and not max_pages
    if min_pages and page_count < min_pages:
        return False
    if max_pages and page_count > max_pages:
        return False
    return True


def _discussion_paragraph_count(tex: str) -> int:
    match = re.search(
        r"\\section\{Discussion\}(?:\\label\{[^}]+\})?\s*(.*?)(?=\\section\{|\\bibliographystyle|\\end\{document\})",
        tex,
        re.DOTALL,
    )
    if not match:
        return 0
    body = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", "", match.group(1), flags=re.DOTALL)
    return len([p for p in re.split(r"\n\s*\n", body.strip()) if p.strip()])


def _discussion_ok(count: int, config: Any) -> bool:
    max_paragraphs = int(getattr(config, "writing_polish_discussion_max_paragraphs", 1) or 1)
    return count <= max_paragraphs

