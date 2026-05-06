from __future__ import annotations

import asyncio
from pathlib import Path

from nanoresearch.agents.cluster_executor_ops import _ClusterExecutorOpsMixin
from nanoresearch.agents.coding_helpers import _CodingHelpersMixin
from nanoresearch.config import ResearchConfig


class _DummyCodingHelper(_CodingHelpersMixin):
    def __init__(self, config: ResearchConfig) -> None:
        self.config = config

    def _resolve_experiment_python(self) -> str:
        return "/usr/bin/python3"


class _DummyClusterOps(_ClusterExecutorOpsMixin):
    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.partition = "belt_road"
        self.gpus = 1
        self.quota_type = "auto"
        self.time_limit = ""
        self.container = ""
        self.conda_env = "base"
        self.local_mode = True

    def log(self, message: str) -> None:
        pass


def test_coding_helper_slurm_script_emits_explicit_mem_directive(tmp_path: Path) -> None:
    helper = _DummyCodingHelper(ResearchConfig(slurm_partition="belt_road", slurm_default_mem="64G"))
    script = asyncio.run(
        helper._generate_slurm_script(
            code_plan={"project_name": "demo"},
            blueprint={"compute_requirements": {"num_gpus": 1}},
            code_dir=tmp_path,
            train_command="python train.py",
        )
    )

    assert "#SBATCH --mem=64G" in script


def test_cluster_executor_sbatch_emits_explicit_mem_directive() -> None:
    ops = _DummyClusterOps(ResearchConfig(slurm_default_mem="64G"))
    script = ops._generate_sbatch_script("/tmp/nano_exp", "bash run_train.slurm")

    assert "#SBATCH --mem=64G" in script
