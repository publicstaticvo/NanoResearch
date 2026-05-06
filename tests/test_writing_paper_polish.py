from pathlib import Path

from nanoresearch.config import ResearchConfig
from nanoresearch.agents.writing.paper_polish import (
    build_polish_report,
    paper_polish_enabled,
    postprocess_latex_for_polish,
    section_polish_guidance,
    target_page_range,
)


def _config(**kwargs):
    return ResearchConfig(base_url="http://example.test", api_key="", **kwargs)


def test_paper_polish_guidance_and_target_range():
    cfg = _config(
        writing_paper_polish_enabled=True,
        writing_polish_target_pages_min=8,
        writing_polish_target_pages_max=9,
    )

    assert paper_polish_enabled(cfg)
    assert target_page_range(cfg) == (8, 9)
    guidance = section_polish_guidance("sec:experiments", cfg)
    assert "PAPER POLISH STYLE CONTRACT" in guidance
    assert "Method and Experiments" in guidance


def test_discussion_is_truncated_when_polish_enabled():
    cfg = _config(writing_paper_polish_enabled=True, writing_polish_discussion_max_paragraphs=1)
    tex = """
\\section{Method}
Body.
\\section{Discussion}\\label{sec:discussion}
First paragraph.

Second paragraph should be removed.

\\section{Conclusion}
Done.
"""

    polished = postprocess_latex_for_polish(tex, cfg)

    assert "First paragraph." in polished
    assert "Second paragraph should be removed." not in polished


def test_polish_report_checks_required_sentence_and_forbidden_terms(tmp_path: Path):
    cfg = _config(
        writing_paper_polish_enabled=True,
        writing_polish_required_sentences=["Exact required sentence."],
        writing_polish_min_references=1,
    )
    tex_path = tmp_path / "paper.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(
        "\\section{Introduction}\nExact required sentence.\n"
        "\\section{Discussion}\\label{sec:discussion}\nOne paragraph.\n"
        "\\section{Conclusion}\nDone.\n",
        encoding="utf-8",
    )
    bib_path.write_text("@article{a, title={A}}", encoding="utf-8")

    report = build_polish_report(tex_path=tex_path, bib_path=bib_path, pdf_path=None, config=cfg)

    assert report["required_sentences_ok"] is True
    assert report["references_ok"] is True
    assert report["discussion_ok"] is True
    assert report["forbidden_terms_ok"] is True
    assert "3x3" not in report

