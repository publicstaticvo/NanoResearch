from __future__ import annotations

from pathlib import Path

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.profile import build_profile_seed, save_user_profile


class DummyAgent(BaseResearchAgent):
    stage = PipelineStage.WRITING

    async def run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def test_build_adaptive_context_reads_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / ".nanoresearch"))

    profile = build_profile_seed("cv_visual_conference")
    profile["profile_id"] = "cv-profile"
    save_user_profile(profile)

    workspace = Workspace.create(
        topic="cv robustness",
        config_snapshot={},
        root=tmp_path / "workspace",
    )
    config = ResearchConfig(
        base_url="http://example.com/v1",
        api_key="test-key",
    )
    agent = DummyAgent(workspace, config)

    context = agent.build_adaptive_context(
        "writing",
        topic="cv robustness",
        template_format="neurips",
    )

    assert "USER PROFILE" in context
    assert "cv_visual_conference" in context
    assert "Figure style" in context
