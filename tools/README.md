# Router SDPO Training Tools

This directory contains the public training path for the NanoResearch hindsight router policy.

- `export_router_sdpo_offpolicy.py` converts live router traces (`live_router_training_examples.jsonl`) into an off-policy SDPO manifest with base and hindsight prompts.
- `train_router_sdpo_offpolicy.py` trains the router model with token-level SDPO advantages by comparing base-prompt and hindsight-prompt log probabilities for the same target router action.

Minimal usage:

```bash
python agent_workflow/tools/export_router_sdpo_offpolicy.py \
  --input-dir runs/router_trace_dir \
  --output artifacts/router_sdpo/train_manifest.jsonl \
  --stats-output artifacts/router_sdpo/export_stats.json \
  --drop-report-output artifacts/router_sdpo/drop_reasons.json \
  --tokenizer-path <base-router-model>

torchrun --nproc_per_node=8 agent_workflow/tools/train_router_sdpo_offpolicy.py \
  --model-path <base-router-model> \
  --manifest artifacts/router_sdpo/train_manifest.jsonl \
  --output-dir artifacts/router_sdpo/checkpoint
```

The release package does not include private traces or trained weights. It includes the exact export/training implementation needed to reproduce the planner-policy update path from public or locally collected router traces.
