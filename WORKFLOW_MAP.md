# NanoResearch Agent Workflow Source

This directory contains the core NanoResearch agent implementation used by the release package. It includes the full workflow from user/profile initialization through ideation, planning, experimentation, writing, review, and self-evolution.

## Included Runtime Components

- `nanoresearch/pipeline/`: orchestration logic and multi-model execution wrappers.
- `nanoresearch/router_policy.py`, `nanoresearch/skills.py`, `nanoresearch/profile.py`: router, skill/memory/profile interfaces.
- `nanoresearch/agents/ideation*.py`: literature search, evidence extraction, hypothesis generation, novelty checking, and idea selection.
- `nanoresearch/agents/planning.py`: experiment-blueprint planning and validation-facing plan construction.
- `nanoresearch/agents/experiment/`, `nanoresearch/agents/coding*.py`, `nanoresearch/agents/execution/`: method implementation, generated code execution, repair, and result collection.
- `nanoresearch/agents/analysis/`: ablation, comparison, statistics, and training-dynamics analysis.
- `nanoresearch/agents/review/`: review, LaTeX compilation checks, citation checks, and layout diagnosis.
- `nanoresearch/agents/writing/`: paper-writing, grounding, citation management, LaTeX assembly, and paper polish.
- `nanoresearch/evolution/`: skill and memory update utilities for self-evolution.
- `nanoresearch/prompts/`, `nanoresearch/templates/`, `nanoresearch/schemas/`: prompt contracts, paper templates, and structured schemas.
- `mcp_server/`: local tool server components used by the agent workflow.
- `tests/`: focused tests covering router persona evaluation, execution repair, scheduler, and writing polish.



## Router and Self-Evolution Mechanism

The release includes the online router path rather than only the final paper-writing path.

- Router implementation: `nanoresearch/router_policy.py` defines the SDPO router interface. The router returns `selected_memory_ids`, `selected_skill_ids`, a compact `prompt_plan`, and optional hindsight updates.
- Router injection point: `nanoresearch/agents/base.py` builds the adaptive context shared by stage agents. It collects candidate memories and skills, asks the router for a focused subset, renders selected context blocks, appends the router prompt plan, and records `router_input`, `router_decision`, candidate counts, and selected asset IDs.
- Stage coverage: adaptive context is invoked by ideation, planning, experiment/coding/execution, and writing agents. The review agent is intentionally marked as not participating in adaptive retrieval in the current implementation.
- Selection policy: the router must choose only IDs present in candidate memory/skill lists and is instructed to select a focused subset rather than all available assets.

NanoResearch stores three reusable evolution channels:

- Memory evolution: `nanoresearch/evolution/memory.py` stores durable user/project/decision context and research memories such as promising directions, failed directions, data strategies, and training strategies. `memory_analyzer.py` distills trajectories into these reusable memories.
- Natural-language skill evolution: `nanoresearch/evolution/skills.py` stores reviewed procedural rules by domain, such as literature, planning, coding, experiment, writing, and review. These are retrieved as evolved behavioral rules and inserted into downstream prompts.
- Script-skill evolution: `nanoresearch/evolution/skills.py` also supports tested automation hooks for low-risk repeated operations such as environment setup, literature tracking, and figure formatting. These can be recommended or autorun depending on policy.

In short, the router decides what prior experience to reuse at each stage, while the evolution layer updates the reusable bank of memories, natural-language skills, and script skills after trajectories produce stable lessons.

## Release Notes

- Runtime secrets are not included. Configure providers through environment variables or local config outside this release pack.
- Cluster- or user-specific absolute paths have been replaced with portable placeholders.
- Generated caches such as `__pycache__` and `.pyc` files are intentionally omitted.
