"""Reflection-Augmentation Model (RAM) inference module.

Wraps a local language model (default: Qwen2.5-8B-Instruct) that generates
structured context augmentations for downstream Claude calls.  Supports:

  - HuggingFace Transformers backend (default)
  - vLLM HTTP serving backend
  - LoRA adapter loading
  - Lazy model loading (no GPU until first ``generate()`` call)
"""

from __future__ import annotations

import logging
import re
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .ram_prompts import get_ram_prompts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class RAMBackend(str, Enum):
    HF = "hf"
    VLLM = "vllm"


class RAMOutput(BaseModel):
    """Parsed output from a single RAM invocation."""

    diagnosis: str = ""
    augmentation: str = ""
    evolution_hints: list[str] = Field(default_factory=list)
    raw_text: str = ""
    input_text: str = ""  # Full prompt (for SDPO data collection)
    token_count: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

_TAG_RE = {
    "diagnosis": re.compile(
        r"<diagnosis>(.*?)</diagnosis>", re.DOTALL | re.IGNORECASE
    ),
    "augmentation": re.compile(
        r"<augmentation>(.*?)</augmentation>", re.DOTALL | re.IGNORECASE
    ),
    "evolution_hint": re.compile(
        r"<evolution_hint>(.*?)</evolution_hint>", re.DOTALL | re.IGNORECASE
    ),
}


def _parse_ram_output(text: str) -> dict[str, Any]:
    """Extract structured sections from RAM output text."""
    diagnosis = ""
    augmentation = ""
    hints: list[str] = []

    m = _TAG_RE["diagnosis"].search(text)
    if m:
        diagnosis = m.group(1).strip()

    m = _TAG_RE["augmentation"].search(text)
    if m:
        augmentation = m.group(1).strip()

    m = _TAG_RE["evolution_hint"].search(text)
    if m:
        raw_hint = m.group(1).strip()
        hints = [line.strip() for line in raw_hint.splitlines() if line.strip()]

    # Fallback: if no augmentation tag found, use entire text
    if not augmentation and text.strip():
        augmentation = text.strip()

    return {
        "diagnosis": diagnosis,
        "augmentation": augmentation,
        "evolution_hints": hints,
    }


# ---------------------------------------------------------------------------
# RAM Module
# ---------------------------------------------------------------------------


class RAMModule:
    """Local language model for reflection-augmentation.

    The model is lazily loaded on first ``generate()`` call. If ``enabled``
    is ``False``, ``generate()`` returns an empty :class:`RAMOutput` without
    ever allocating GPU memory.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path.
    backend : str
        ``"hf"`` for HuggingFace Transformers, ``"vllm"`` for vLLM HTTP API.
    vllm_url : str
        Base URL for vLLM server (only used when ``backend="vllm"``).
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    device : str
        PyTorch device string. ``"auto"`` selects GPU if available.
    enabled : bool
        Master toggle. When ``False``, ``generate()`` is a no-op.
    checkpoint_path : str
        Path to a LoRA adapter checkpoint. Empty string means no adapter.
    """

    def __init__(
        self,
        model_name_or_path: str = "/mnt/petrelfs/xujinhang/model/Qwen2.5-8B-Instruct",
        backend: str = "hf",
        vllm_url: str = "",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        device: str = "auto",
        enabled: bool = True,
        checkpoint_path: str = "",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.backend = RAMBackend(backend)
        self.vllm_url = vllm_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.enabled = enabled
        self.checkpoint_path = checkpoint_path

        # Populated by _lazy_load()
        self._model = None
        self._tokenizer = None
        self._loaded = False

    # ---- properties -------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ---- lazy loading -----------------------------------------------------

    def _lazy_load(self) -> None:
        """Load model + tokenizer on first use."""
        if self._loaded or not self.enabled:
            return

        if self.backend == RAMBackend.VLLM:
            # vLLM: no local model loading required
            self._loaded = True
            logger.info("RAM: using vLLM backend at %s", self.vllm_url)
            return

        logger.info("RAM: loading model %s ...", self.model_name_or_path)
        t0 = time.time()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            self._model = self._model.to(device)

        # Load LoRA adapter if specified
        if self.checkpoint_path:
            logger.info("RAM: loading LoRA adapter from %s", self.checkpoint_path)
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(
                self._model, self.checkpoint_path
            )
            self._model = self._model.merge_and_unload()

        self._model.eval()
        self._loaded = True
        logger.info("RAM: model loaded in %.1fs", time.time() - t0)

    # ---- generation -------------------------------------------------------

    def _build_messages(
        self,
        task_type: str,
        context: str,
        feedback: str,
        retrieved_skills: str,
        retrieved_memories: str,
    ) -> list[dict[str, str]]:
        """Build chat messages for the RAM model."""
        system_prompt, user_template = get_ram_prompts(task_type)
        user_msg = user_template.format(
            context=context or "(no context)",
            feedback=feedback or "(no prior feedback)",
            retrieved_skills=retrieved_skills or "(none)",
            retrieved_memories=retrieved_memories or "(none)",
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def generate(
        self,
        task_type: str,
        context: str,
        feedback: str = "",
        retrieved_skills: str = "",
        retrieved_memories: str = "",
    ) -> RAMOutput:
        """Generate a reflection-augmentation response.

        Parameters
        ----------
        task_type : str
            One of ``"method_gen"``, ``"code_impl"``, ``"paper_writing"``.
        context : str
            Current task context (topic, plan, etc.).
        feedback : str
            Downstream feedback from the previous attempt.
        retrieved_skills : str
            Pre-formatted skill context from the Skill Store.
        retrieved_memories : str
            Pre-formatted memory context from the Memory Store.

        Returns
        -------
        RAMOutput
            Parsed structured output. Empty if disabled.
        """
        if not self.enabled:
            return RAMOutput()

        messages = self._build_messages(
            task_type, context, feedback, retrieved_skills, retrieved_memories
        )

        if self.backend == RAMBackend.VLLM:
            return self._generate_vllm(messages)
        else:
            return self._generate_hf(messages)

    def _generate_hf(self, messages: list[dict[str, str]]) -> RAMOutput:
        """Generate using HuggingFace Transformers."""
        self._lazy_load()
        import torch

        t0 = time.time()

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                top_p=0.9 if self.temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_count = len(generated_ids)
        latency_ms = (time.time() - t0) * 1000

        parsed = _parse_ram_output(raw_text)
        return RAMOutput(
            diagnosis=parsed["diagnosis"],
            augmentation=parsed["augmentation"],
            evolution_hints=parsed["evolution_hints"],
            raw_text=raw_text,
            input_text=input_text,
            token_count=token_count,
            latency_ms=latency_ms,
        )

    def _generate_vllm(self, messages: list[dict[str, str]]) -> RAMOutput:
        """Generate using vLLM HTTP API."""
        self._lazy_load()
        import json as _json
        import urllib.request

        t0 = time.time()

        payload = {
            "model": self.model_name_or_path,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.vllm_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = _json.loads(resp.read())
        except Exception as exc:
            logger.error("vLLM API call failed: %s", exc)
            return RAMOutput()

        raw_text = result["choices"][0]["message"]["content"]
        token_count = result.get("usage", {}).get("completion_tokens", 0)
        latency_ms = (time.time() - t0) * 1000

        # Build input_text from messages for data collection
        input_text = "\n".join(
            f"<|{m['role']}|>\n{m['content']}" for m in messages
        )

        parsed = _parse_ram_output(raw_text)
        return RAMOutput(
            diagnosis=parsed["diagnosis"],
            augmentation=parsed["augmentation"],
            evolution_hints=parsed["evolution_hints"],
            raw_text=raw_text,
            input_text=input_text,
            token_count=token_count,
            latency_ms=latency_ms,
        )

    # ---- log-prob computation (for SDPO trainer, Step 8) ------------------

    def compute_logprobs(
        self, input_text: str, output_text: str
    ) -> "torch.Tensor":
        """Compute per-token log-probabilities for ``output_text`` given
        ``input_text`` as context.

        Used by the SDPO trainer for the original policy:
        ``log π_θ(y_i | x, y_{<i})``.
        """
        self._lazy_load()
        import torch

        full_text = input_text + output_text
        input_ids = self._tokenizer.encode(
            full_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(self._model.device)
        input_len = len(self._tokenizer.encode(input_text, truncation=True, max_length=4096))

        with torch.no_grad():
            logits = self._model(input_ids).logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[0, input_len - 1 : -1, :]  # predictions for output tokens
        shift_labels = input_ids[0, input_len:]  # actual output tokens

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

        return token_log_probs

    def compute_hindsight_logprobs(
        self, input_text: str, feedback: str, output_text: str
    ) -> "torch.Tensor":
        """Compute per-token log-probabilities under the hindsight policy.

        The hindsight policy conditions on feedback: ``log π_θ(y_i | x, o, y_{<i})``.
        """
        # Construct hindsight input: original input + feedback context
        hindsight_input = (
            input_text.rstrip()
            + "\n\n<hindsight_context> The following is a future user message. "
            "Use this to guide your answer: "
            + feedback
            + " </hindsight_context>\n"
        )
        return self.compute_logprobs(hindsight_input, output_text)

    # ---- cleanup ----------------------------------------------------------

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        # Try to free CUDA cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("RAM: model unloaded")
