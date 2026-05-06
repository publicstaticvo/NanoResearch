#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import types
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
def ensure_triton_ops_stub() -> None:
    """bitsandbytes 0.42 expects triton.ops, which is absent in Triton 3.x."""
    try:
        import triton.ops.matmul_perf_model  # type: ignore # noqa: F401
        return
    except Exception:
        ops_mod = sys.modules.get("triton.ops")
        if ops_mod is None:
            ops_mod = types.ModuleType("triton.ops")
            ops_mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules["triton.ops"] = ops_mod
        perf_mod = types.ModuleType("triton.ops.matmul_perf_model")

        def early_config_prune(*args: Any, **kwargs: Any) -> bool:
            return False

        def estimate_matmul_time(*args: Any, **kwargs: Any) -> float:
            return 0.0

        perf_mod.early_config_prune = early_config_prune  # type: ignore[attr-defined]
        perf_mod.estimate_matmul_time = estimate_matmul_time  # type: ignore[attr-defined]
        sys.modules["triton.ops.matmul_perf_model"] = perf_mod


def disable_deepspeed_discovery() -> None:
    """Prevent Transformers from auto-importing deepspeed in this FSDP-only job."""
    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str, package: str | None = None):  # type: ignore[override]
        if name == "deepspeed":
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = patched_find_spec  # type: ignore[assignment]


disable_deepspeed_discovery()
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


ensure_triton_ops_stub()
import bitsandbytes as bnb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NanoResearch router with exact off-policy SDPO.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def rank0_print(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    chunks = []
    for message in messages:
        chunks.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


class RouterSDPODataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        tokenizer: Any,
        max_prompt_length: int,
        max_completion_length: int,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.subsystem_counts: Counter[str] = Counter()
        self.persona_counts: Counter[str] = Counter()
        manifest_file = Path(manifest_path)
        with manifest_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                encoded = self._encode_sample(
                    tokenizer=tokenizer,
                    sample=sample,
                    max_prompt_length=max_prompt_length,
                    max_completion_length=max_completion_length,
                )
                self.examples.append(encoded)
                self.subsystem_counts.update([sample["subsystem"]])
                self.persona_counts.update([sample["persona_id"]])
        self.truncate_counts: Counter[str] = Counter(
            key
            for example in self.examples
            for key, flag in (
                ("base_prompt_truncated", example["base_prompt_truncated"]),
                ("hindsight_prompt_truncated", example["hindsight_prompt_truncated"]),
            )
            if flag
        )

    @staticmethod
    def _encode_text(tokenizer: Any, text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def _encode_sample(
        self,
        tokenizer: Any,
        sample: dict[str, Any],
        max_prompt_length: int,
        max_completion_length: int,
    ) -> dict[str, Any]:
        base_prompt_text = render_prompt_text(tokenizer, sample["base_messages"])
        hindsight_prompt_text = render_prompt_text(tokenizer, sample["hindsight_messages"])
        target_text = sample["target_text"]

        base_prompt_ids = self._encode_text(tokenizer, base_prompt_text)
        hindsight_prompt_ids = self._encode_text(tokenizer, hindsight_prompt_text)
        completion_ids = self._encode_text(tokenizer, target_text)

        base_prompt_truncated = len(base_prompt_ids) > max_prompt_length
        hindsight_prompt_truncated = len(hindsight_prompt_ids) > max_prompt_length
        if base_prompt_truncated:
            base_prompt_ids = base_prompt_ids[-max_prompt_length:]
        if hindsight_prompt_truncated:
            hindsight_prompt_ids = hindsight_prompt_ids[-max_prompt_length:]
        if len(completion_ids) > max_completion_length:
            raise ValueError(f"Completion too long for {sample['sample_id']}: {len(completion_ids)}")

        return {
            "sample_id": sample["sample_id"],
            "persona_id": sample["persona_id"],
            "task_id": sample["task_id"],
            "subsystem": sample["subsystem"],
            "turn": int(sample["turn"]),
            "attempt_no": int(sample.get("attempt_no", 0)),
            "base_input_ids": base_prompt_ids + completion_ids,
            "hindsight_input_ids": hindsight_prompt_ids + completion_ids,
            "base_prompt_len": len(base_prompt_ids),
            "hindsight_prompt_len": len(hindsight_prompt_ids),
            "completion_len": len(completion_ids),
            "base_prompt_truncated": base_prompt_truncated,
            "hindsight_prompt_truncated": hindsight_prompt_truncated,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class RouterSDPOCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _pad(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(seq) for seq in sequences)
        input_ids = []
        attention_masks = []
        for seq in sequences:
            padding = [self.pad_token_id] * (max_len - len(seq))
            input_ids.append(seq + padding)
            attention_masks.append([1] * len(seq) + [0] * len(padding))
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        base_input_ids, base_attention_mask = self._pad([item["base_input_ids"] for item in batch])
        hindsight_input_ids, hindsight_attention_mask = self._pad([item["hindsight_input_ids"] for item in batch])
        completion_lengths = torch.tensor([item["completion_len"] for item in batch], dtype=torch.long)
        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "subsystems": [item["subsystem"] for item in batch],
            "base_input_ids": base_input_ids,
            "base_attention_mask": base_attention_mask,
            "base_prompt_lens": torch.tensor([item["base_prompt_len"] for item in batch], dtype=torch.long),
            "hindsight_input_ids": hindsight_input_ids,
            "hindsight_attention_mask": hindsight_attention_mask,
            "hindsight_prompt_lens": torch.tensor([item["hindsight_prompt_len"] for item in batch], dtype=torch.long),
            "completion_lens": completion_lengths,
        }


def find_transformer_layer_classes(model: nn.Module) -> tuple[type[nn.Module], ...]:
    candidates: list[tuple[int, type[nn.Module]]] = []
    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            child_types = {type(child) for child in module}
            if len(child_types) == 1:
                candidates.append((len(module), next(iter(child_types))))
    if not candidates:
        raise RuntimeError("Could not infer transformer block class for FSDP auto-wrap.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return (candidates[0][1],)


def enable_activation_checkpointing(model: nn.Module, transformer_layer_classes: tuple[type[nn.Module], ...]) -> None:
    wrapper = lambda module: checkpoint_wrapper(  # noqa: E731
        module,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, transformer_layer_classes),
    )


def build_model(args: argparse.Namespace, local_rank: int) -> tuple[FSDP, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.resume_from or args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    transformer_layer_classes = find_transformer_layer_classes(model)
    enable_activation_checkpointing(model, transformer_layer_classes)

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    auto_wrap_policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(  # noqa: E731
        module,
        recurse,
        nonwrapped_numel,
        transformer_layer_cls=transformer_layer_classes,
    )

    fsdp_model = FSDP(
        model.to(local_rank),
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.device("cuda", local_rank),
        sync_module_states=True,
        use_orig_params=True,
    )
    return fsdp_model, tokenizer


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def extract_completion_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    completion_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    shift_labels = input_ids[:, 1:]
    gathered = torch.gather(shift_logprobs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    batch_size = input_ids.size(0)
    max_completion = int(completion_lens.max().item())
    token_logprobs = logits.new_zeros((batch_size, max_completion))
    token_mask = torch.zeros((batch_size, max_completion), dtype=torch.bool, device=logits.device)

    for idx in range(batch_size):
        prompt_len = int(prompt_lens[idx].item())
        completion_len = int(completion_lens[idx].item())
        valid_seq_len = int(attention_mask[idx].sum().item())
        start = prompt_len - 1
        end = min(start + completion_len, valid_seq_len - 1)
        current = gathered[idx, start:end]
        token_logprobs[idx, : current.size(0)] = current
        token_mask[idx, : current.size(0)] = True
    return token_logprobs, token_mask


def compute_sdpo_step(model: FSDP, batch: dict[str, Any], use_autocast: bool) -> tuple[torch.Tensor, dict[str, float]]:
    base_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
    with base_ctx:
        base_outputs = model(
            input_ids=batch["base_input_ids"],
            attention_mask=batch["base_attention_mask"],
            use_cache=False,
        )

    with torch.no_grad():
        hindsight_outputs = model(
            input_ids=batch["hindsight_input_ids"],
            attention_mask=batch["hindsight_attention_mask"],
            use_cache=False,
        )

    base_token_logprobs, base_mask = extract_completion_logprobs(
        logits=base_outputs.logits,
        input_ids=batch["base_input_ids"],
        attention_mask=batch["base_attention_mask"],
        prompt_lens=batch["base_prompt_lens"],
        completion_lens=batch["completion_lens"],
    )
    hindsight_token_logprobs, hindsight_mask = extract_completion_logprobs(
        logits=hindsight_outputs.logits,
        input_ids=batch["hindsight_input_ids"],
        attention_mask=batch["hindsight_attention_mask"],
        prompt_lens=batch["hindsight_prompt_lens"],
        completion_lens=batch["completion_lens"],
    )

    if not torch.equal(base_mask, hindsight_mask):
        raise RuntimeError("Base and hindsight completion masks diverged; target alignment is broken.")

    token_mask = base_mask
    advantage = (hindsight_token_logprobs - base_token_logprobs).detach()
    weighted_logprob = -(advantage * base_token_logprobs)
    loss_per_sample = (weighted_logprob * token_mask).sum(dim=1)
    loss = loss_per_sample.mean()

    token_count = int(token_mask.sum().item())
    metrics = {
        "loss_sum": float(loss_per_sample.detach().sum().item()),
        "sample_count": float(loss_per_sample.numel()),
        "token_count": float(token_count),
        "advantage_sum": float((advantage * token_mask).sum().item()),
        "positive_advantage_count": float(((advantage > 0) & token_mask).sum().item()),
        "base_logprob_sum": float((base_token_logprobs * token_mask).sum().item()),
        "hindsight_logprob_sum": float((hindsight_token_logprobs * token_mask).sum().item()),
        "max_abs_advantage": float(advantage.abs().max().item()) if token_count else 0.0,
    }
    return loss, metrics


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = sorted(metrics)
    tensor = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, tensor.tolist(), strict=True)}


def save_checkpoint(model: FSDP, tokenizer: Any, output_dir: Path, epoch: int, rank: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_config):
        state_dict = model.state_dict()
    if rank == 0:
        epoch_dir = output_dir / f"epoch-{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(epoch_dir, state_dict=state_dict)
        tokenizer.save_pretrained(epoch_dir)
    barrier()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    seed_everything(args.seed + rank)
    device = torch.device("cuda", local_rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    model, tokenizer = build_model(args, local_rank)
    dataset = RouterSDPODataset(
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )

    if rank == 0:
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(
                {
                    "samples": len(dataset),
                    "subsystem_counts": dict(sorted(dataset.subsystem_counts.items())),
                    "persona_counts": dict(sorted(dataset.persona_counts.items())),
                    "truncate_counts": dict(sorted(dataset.truncate_counts.items())),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=RouterSDPOCollator(tokenizer.pad_token_id),
    )

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    global_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps * world_size
    optimizer_steps_per_epoch = math.ceil(len(dataset) / global_batch_size)
    total_optimizer_steps = optimizer_steps_per_epoch * args.num_epochs
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    manifest_copy = output_dir / "train_manifest.jsonl"
    if rank == 0 and not manifest_copy.exists():
        manifest_copy.write_text(Path(args.manifest).read_text(encoding="utf-8"), encoding="utf-8")

    if rank == 0:
        (output_dir / "train_config.json").write_text(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "manifest": args.manifest,
                    "num_epochs": args.num_epochs,
                    "learning_rate": args.learning_rate,
                    "warmup_ratio": args.warmup_ratio,
                    "per_device_batch_size": args.per_device_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "world_size": world_size,
                    "global_batch_size": global_batch_size,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "total_optimizer_steps": total_optimizer_steps,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    "optimizer": "AdamW8bit",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    metrics_log = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    train_start = time.perf_counter()
    interrupted_recovered = bool(args.resume_from)

    for epoch in range(1, args.num_epochs + 1):
        sampler.set_epoch(epoch)
        window_metrics = {
            "loss_sum": 0.0,
            "sample_count": 0.0,
            "token_count": 0.0,
            "advantage_sum": 0.0,
            "positive_advantage_count": 0.0,
            "base_logprob_sum": 0.0,
            "hindsight_logprob_sum": 0.0,
            "max_abs_advantage": 0.0,
        }
        window_start = time.perf_counter()

        for step, raw_batch in enumerate(dataloader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            loss, metrics = compute_sdpo_step(model, batch, use_autocast=True)

            if args.gate_only and metrics["max_abs_advantage"] <= 0.0:
                raise RuntimeError("Gate failed: all token advantages are zero.")

            (loss / args.gradient_accumulation_steps).backward()

            for key in window_metrics:
                if key == "max_abs_advantage":
                    window_metrics[key] = max(window_metrics[key], metrics[key])
                else:
                    window_metrics[key] += metrics[key]

            should_step = (step % args.gradient_accumulation_steps == 0) or (step == len(dataloader))
            if not should_step:
                continue

            grad_norm = float(model.clip_grad_norm_(args.max_grad_norm).item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            reduced = reduce_metrics(window_metrics, device)
            elapsed = time.perf_counter() - window_start
            tokens_per_second = reduced["token_count"] / max(elapsed, 1e-6)
            averaged = {
                "epoch": epoch,
                "optimizer_step": global_step,
                "L_SDPO": reduced["loss_sum"] / max(reduced["sample_count"], 1.0),
                "mean_token_advantage": reduced["advantage_sum"] / max(reduced["token_count"], 1.0),
                "positive_advantage_ratio": reduced["positive_advantage_count"] / max(reduced["token_count"], 1.0),
                "mean_base_logprob": reduced["base_logprob_sum"] / max(reduced["token_count"], 1.0),
                "mean_hindsight_logprob": reduced["hindsight_logprob_sum"] / max(reduced["token_count"], 1.0),
                "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
                "token_count": int(reduced["token_count"]),
            }

            if rank == 0:
                append_jsonl(metrics_log, averaged)
                if global_step % args.log_every == 0:
                    rank0_print(
                        rank,
                        (
                            f"epoch={epoch} step={global_step} "
                            f"L_SDPO={averaged['L_SDPO']:.6f} "
                            f"adv={averaged['mean_token_advantage']:.6f} "
                            f"base_lp={averaged['mean_base_logprob']:.6f} "
                            f"hind_lp={averaged['mean_hindsight_logprob']:.6f} "
                            f"pos_adv={averaged['positive_advantage_ratio']:.4f} "
                            f"grad_norm={averaged['grad_norm']:.4f} "
                            f"tok/s={averaged['tokens_per_second']:.2f}"
                        ),
                    )

            window_metrics = {
                "loss_sum": 0.0,
                "sample_count": 0.0,
                "token_count": 0.0,
                "advantage_sum": 0.0,
                "positive_advantage_count": 0.0,
                "base_logprob_sum": 0.0,
                "hindsight_logprob_sum": 0.0,
                "max_abs_advantage": 0.0,
            }
            window_start = time.perf_counter()

            if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
                break

        save_checkpoint(model, tokenizer, output_dir, epoch, rank)
        if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
            break

    total_elapsed = time.perf_counter() - train_start
    if rank == 0:
        summary = {
            "samples": len(dataset),
            "num_epochs_requested": args.num_epochs,
            "optimizer_steps_completed": global_step,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "global_batch_size": global_batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "interrupted_recovered": interrupted_recovered,
            "gate_only": args.gate_only,
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
