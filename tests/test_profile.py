from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nanoresearch.cli import app
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.profile import (
    build_profile_seed,
    get_profile_json_path,
    get_profile_markdown_path,
    load_user_profile,
    render_profile_context,
    save_user_profile,
)


runner = CliRunner()


def _set_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / ".nanoresearch"))


def test_save_profile_writes_json_and_markdown(monkeypatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = build_profile_seed("ai4science_journal")
    profile["profile_id"] = "test-profile"
    save_user_profile(profile)

    json_path = get_profile_json_path()
    md_path = get_profile_markdown_path()

    assert json_path.is_file()
    assert md_path.is_file()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["archetype_seed"] == "ai4science_journal"
    assert "publication_profile" in loaded
    assert "Nature/Springer" in md_path.read_text(encoding="utf-8")


def test_render_profile_context_includes_writing_constraints(monkeypatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = build_profile_seed("ai4science_journal")
    profile["profile_id"] = "ctx-profile"
    save_user_profile(profile)

    loaded = load_user_profile()
    writing_context = render_profile_context("writing", loaded)
    planning_context = render_profile_context("planning", loaded)

    assert "USER PROFILE" in writing_context
    assert "Venue/style preference" in writing_context
    assert "Figure style" in writing_context
    assert "Research direction" in planning_context
    assert "Resource budget" in planning_context


def test_nano_init_creates_profile_files(monkeypatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)

    user_input = "\n".join(
        [
            "3",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "y",
        ]
    )

    result = runner.invoke(app, ["init"], input=user_input)

    assert result.exit_code == 0, result.output
    assert get_profile_json_path().is_file()
    assert get_profile_markdown_path().is_file()
    assert "Profile saved" in result.output


def test_skills_trace_reads_workspace_adaptive_context(monkeypatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)
    workspace = Workspace.create(topic="trace topic", config_snapshot={}, root=tmp_path / "research")
    workspace.write_json(
        "logs/adaptive_context_writing_writing.json",
        {
            "task_type": "writing",
            "topic": "trace topic",
            "candidate_static_skills": ["vendor.ml-paper-writing", "vendor.academic-plotting"],
            "matched_static_skills": ["vendor.ml-paper-writing"],
            "matched_evolved_skills": ["skill-local-writing-adaptation"],
            "matched_script_skills": ["figure_formatter"],
            "matched_skills": ["ml-paper-writing", "skill-local-writing-adaptation", "figure_formatter"],
        },
    )

    result = runner.invoke(app, ["skills", "trace", "--workspace", str(workspace.path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["workspace"] == str(workspace.path)
    assert len(payload["stages"]) == 1
    stage = payload["stages"][0]
    assert stage["task_type"] == "writing"
    assert stage["matched_static_skills"] == ["vendor.ml-paper-writing"]
    assert stage["matched_evolved_skills"] == ["skill-local-writing-adaptation"]
    assert stage["matched_script_skills"] == ["figure_formatter"]


def test_skills_inspect_reports_static_and_profile_matches(monkeypatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = build_profile_seed("cv_visual_conference")
    profile["profile_id"] = "inspect-profile"
    save_user_profile(profile)

    result = runner.invoke(
        app,
        [
            "skills",
            "inspect",
            "--stage",
            "writing",
            "--topic",
            "Write a NeurIPS paper on robust image classification.",
            "--template-format",
            "neurips",
            "--text",
            "Need academic writing, citations, latex, and benchmark figures.",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["stage"] == "writing"
    assert payload["profile_loaded"] is True
    assert "candidate_static_skills" in payload
    assert "matched_static_skills" in payload
    assert "matched_evolved_skills" in payload
    assert "matched_script_skills" in payload
    assert "vendor.ml-paper-writing" in payload["matched_static_skills"]


def test_nano_init_reviews_existing_profile(monkeypatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)
    save_user_profile(build_profile_seed("nlp_conference"))

    result = runner.invoke(app, ["init"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Existing profile found" in result.output
    assert "Keeping current profile unchanged" in result.output
