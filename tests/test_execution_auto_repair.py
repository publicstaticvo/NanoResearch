from __future__ import annotations

from nanoresearch.config import ResearchConfig
from nanoresearch.agents.execution.repair import _RepairMixin


class _DummyRepair(_RepairMixin):
    def __init__(self, config: ResearchConfig) -> None:
        self.config = config


def test_execution_auto_repair_is_disabled_by_default() -> None:
    config = ResearchConfig(base_url='https://example.com', api_key='')

    assert config.execution_auto_repair_enabled is False
    assert _DummyRepair(config)._execution_auto_repair_enabled() is False


def test_execution_auto_repair_helper_reads_enabled_flag() -> None:
    config = ResearchConfig(
        base_url='https://example.com',
        api_key='',
        execution_auto_repair_enabled=True,
    )

    assert _DummyRepair(config)._execution_auto_repair_enabled() is True
