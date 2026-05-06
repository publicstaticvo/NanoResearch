from __future__ import annotations

from pathlib import Path

from nanoresearch.evolution.memory import MemoryStore
from nanoresearch.evolution.skills import SkillEvolutionStore


def test_memory_and_skill_stores_follow_nanoresearch_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NANORESEARCH_HOME", str(tmp_path / "nr-home"))

    memory_store = MemoryStore(enabled=True)
    skill_store = SkillEvolutionStore(enabled=True)

    assert memory_store.root == tmp_path / "nr-home" / "memory"
    assert skill_store.root == tmp_path / "nr-home" / "skills"
