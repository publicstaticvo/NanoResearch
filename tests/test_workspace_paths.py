from pathlib import Path

from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage


def test_workspace_expands_home_before_relative_artifact_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    workspace = Workspace.create(
        topic="path normalization",
        root=Path("~/.nanoresearch/workspace/research"),
        session_id="session123",
    )
    train_py = workspace.path / "experiment" / "train.py"
    train_py.parent.mkdir(parents=True)
    train_py.write_text("print('ok')\n", encoding="utf-8")

    artifact = workspace.register_artifact("train", train_py, PipelineStage.CODING)

    assert workspace.path == tmp_path / ".nanoresearch" / "workspace" / "research" / "session123"
    assert artifact.path == "experiment/train.py"
