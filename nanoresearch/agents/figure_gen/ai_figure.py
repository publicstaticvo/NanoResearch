"""AI image generation (OpenAI-compatible/Gemini) mixin."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

from ._constants import (
    AI_FIGURE_TEMPLATES,
    _clean_ai_image_caption,
    FIGURE_PROMPT_SYSTEM,
    MAX_IMAGE_PROMPT_LEN,
    MAX_IMAGE_RETRIES,
    MAX_OPTIMIZED_PROMPT_LEN,
    PROMPT_CORE_PRINCIPLES,
)

logger = logging.getLogger(__name__)


class _AiFigureMixin:
    """Mixin — AI-generated figure methods."""

    async def _generate_ai_figure(
        self,
        context: str,
        fig_key: str,
        filename_stem: str,
        description: str,
        ai_image_type: str = "generic",
        caption: str = "",
    ) -> dict[str, Any]:
        """Generate a single figure via the configured AI image model.

        Flow: generate prompt, call the configured image API with retries, and if all fail,
        LLM diagnoses error & optimizes prompt → retry → fallback to code chart.
        """
        # Ensure the caption used for the paper is short & academic, not the
        # verbose image-generation prompt.  The *description* is meant for
        # prompt generation; the *caption* goes into LaTeX.
        clean_caption = _clean_ai_image_caption(
            caption or description, title=fig_key.replace("_", " ").title(),
        )

        # Look up the template for this AI figure type
        template = AI_FIGURE_TEMPLATES.get(ai_image_type, AI_FIGURE_TEMPLATES["generic"])

        # Step 1: generate an image prompt. Method schematics use a short,
        # ASCII-only prompt because image2-compatible gateways are sensitive to
        # long prompts, Unicode math, and dense text-layout instructions.
        figure_prompt_config = self.config.for_stage("figure_prompt")

        def _short_method_prompt() -> str:
            method_match = re.search(r"Method:\s*([^\n]+)", context or "")
            dataset_match = re.search(r"Datasets:\s*([^\n]+)", context or "")
            metric_match = re.search(r"Metrics:\s*([^\n]+)", context or "")
            method_name = (method_match.group(1).strip() if method_match else "proposed method")[:80]
            dataset_name = (dataset_match.group(1).strip() if dataset_match else "dataset")[:80]
            metric_name = (metric_match.group(1).strip() if metric_match else "evaluation metrics")[:80]
            prompt = (
                "Create a polished vector-style Figure 1 method schematic for a machine learning paper, "
                "wide landscape composition, white background with subtle pale-blue panels and thin gray separators. "
                f"Method context: {method_name}. Do not render dataset names or metric names. "
                "Use visually distinct modules from left to right that match the actual method context: "
                "input data, preprocessing or representation, core model or algorithm, evaluation, and outputs. "
                "Make the central method module larger and label only topic-specific mechanisms that appear in the paper context. "
                "Show train/evaluation separation with a dashed boundary when applicable. "
                "Use muted blue, teal, amber, and green accents, consistent line weights, generous spacing, "
                "small icons, clean arrows, and short readable labels only. No title banner, no paragraphs, no 3D, no photorealism. "
                "The result should look like a carefully designed NeurIPS/ICML workflow diagram, not a generic box chart."
            )
            return re.sub(r"[^A-Za-z0-9 .,;:()/_+-]", "", prompt)[:1200]

        if ai_image_type in {"system_overview", "generic"} and any(
            kw in fig_key.lower() for kw in ("method", "framework", "overview", "architecture", "schematic")
        ):
            image_prompt = _short_method_prompt()
        else:
            user_prompt = (
                f"Research context:\n{context}\n\n"
                f"Figure description:\n{description}\n\n"
                f"=== REFERENCE TEMPLATE (adapt to match the research context above) ===\n"
                f"{template}\n"
                f"=== END TEMPLATE ===\n\n"
                f"Write a concise image generation prompt for this specific figure.\n"
                f"Use the actual method name, datasets, and metrics. Keep the prompt under 900 ASCII characters.\n"
                f"Require a clean 2D scientific diagram, white background, simple labels, no decorative elements."
            )
            try:
                image_prompt = await self._dispatcher.generate(
                    figure_prompt_config, FIGURE_PROMPT_SYSTEM, user_prompt
                )
            except Exception as e:
                logger.warning("LLM prompt generation failed for %s: %s", fig_key, e)
                image_prompt = f"Clean 2D academic diagram. Context: {description}"
            image_prompt = re.sub(r"[^A-Za-z0-9 .,;:()/_+-]", "", image_prompt.strip())[:900]

        self.log(f"  {fig_key} prompt generated ({len(image_prompt)} chars)")
        self.workspace.write_text(
            f"figures/{filename_stem}_prompt.txt", image_prompt
        )

        # Step 2: Generate image via configured image backend with retry loop.
        # The release pipeline is image2-first, but production gateways can be
        # unavailable or time out.  In that case we use configured compatible
        # image fallbacks and record the actual model instead of pretending the
        # fallback came from image2.
        figure_gen_config = self.config.for_stage("figure_gen")
        is_method_schematic = any(
            kw in fig_key.lower() for kw in ("method", "framework", "overview", "architecture", "schematic")
        )
        image_size = "1536x1024" if is_method_schematic else "1024x1024"
        last_error = ""
        prev_error = ""

        for attempt in range(MAX_IMAGE_RETRIES + 1):
            # Early-exit on repeated identical errors
            if attempt >= 1 and last_error and last_error == prev_error:
                self.log(f"  {fig_key} same image error repeated; skipping to diagnosis")
                break
            prev_error = last_error

            try:
                image_payload = await self._generate_image_with_backend_fallback(
                    figure_gen_config, image_prompt, prefer_image2=True, size=image_size,
                )
                if image_payload:
                    b64_image, used_config, backend_meta = image_payload
                    self.log(
                        f"  {fig_key} image generated on attempt {attempt + 1} "
                        f"with model={used_config.model}"
                    )
                    result = await self._save_figure_files(
                        fig_key, filename_stem,
                        clean_caption,
                        base64.b64decode(b64_image),
                    )
                    result["generation_prompt"] = image_prompt
                    result.update(backend_meta)
                    result["image_model"] = used_config.model
                    result["prompt_model"] = figure_prompt_config.model
                    return result
                last_error = "API returned no image data"
            except Exception as exc:
                last_error = str(exc)

            self.log(f"  {fig_key} attempt {attempt + 1}/{MAX_IMAGE_RETRIES + 1} failed: {last_error}")

        # Step 3: All retries failed — LLM diagnoses and optimizes the prompt
        self.log(f"  {fig_key} all {MAX_IMAGE_RETRIES + 1} attempts failed, running LLM diagnosis")
        optimized_prompt = await self._diagnose_and_optimize_prompt(
            fig_key, image_prompt, last_error, description,
        )

        if optimized_prompt:
            self.workspace.write_text(
                f"figures/{filename_stem}_prompt_optimized.txt", optimized_prompt
            )
            # Step 4: Retry with optimized prompt (2 more attempts)
            for opt_attempt in range(2):
                try:
                    image_payload = await self._generate_image_with_backend_fallback(
                        figure_gen_config, optimized_prompt, prefer_image2=True, size=image_size,
                    )
                    if image_payload:
                        b64_image, used_config, backend_meta = image_payload
                        self.log(
                            f"  {fig_key} succeeded with optimized prompt "
                            f"(attempt {opt_attempt + 1}, model={used_config.model})"
                        )
                        result = await self._save_figure_files(
                            fig_key, filename_stem,
                            clean_caption,
                            base64.b64decode(b64_image),
                        )
                        result["generation_prompt"] = optimized_prompt
                        result.update(backend_meta)
                        result["image_model"] = used_config.model
                        result["prompt_model"] = figure_prompt_config.model
                        return result
                except Exception as exc:
                    self.log(f"  {fig_key} optimized attempt {opt_attempt + 1} failed: {exc}")

        # Step 5: Final fallback — code-generated placeholder
        self.log(f"  {fig_key} all AI generation attempts exhausted, using fallback")
        result = await self._generate_fallback_chart(fig_key, filename_stem, clean_caption)
        result["generation_prompt"] = image_prompt
        result["source_backend"] = "code_fallback"
        result["image2_failure_reason"] = last_error[:1000]
        return result


    async def _generate_image_with_backend_fallback(
        self,
        base_config,
        prompt: str,
        prefer_image2: bool = True,
        size: str = "1024x1024",
    ) -> tuple[str, Any, dict[str, Any]] | None:
        """Generate an image with image2 first and audited compatible fallbacks.

        Returns (base64_png, used_config, metadata).  Metadata is deliberately
        precise: source_backend is ``image2`` only when the image2 model itself
        succeeds; otherwise it records the fallback model and the image2 error.
        """
        primary_model = str(base_config.model)
        primary_timeout = float(base_config.timeout or 300.0)
        if prefer_image2 and "gpt-image-2" in primary_model.lower():
            # image2 can legitimately take several minutes through compatible
            # gateways.  Waiting longer is better than silently degrading the
            # paper's main method schematic to a lower-quality fallback.
            primary_timeout = max(primary_timeout, 600.0)
            base_config = base_config.model_copy(update={"timeout": primary_timeout})
        configured = os.environ.get("NANORESEARCH_IMAGE_FALLBACK_MODELS", "").strip()
        fallback_models = [m.strip() for m in configured.split(",") if m.strip()]
        if prefer_image2 and "gpt-image-2" in primary_model.lower():
            for model in ("gpt-image-1-mini", "gpt-image-1"):
                if model not in fallback_models and model != primary_model:
                    fallback_models.append(model)

        model_sequence = [primary_model]
        for model in fallback_models:
            if model and model not in model_sequence:
                model_sequence.append(model)

        image2_error = ""
        errors: list[str] = []
        for idx, model in enumerate(model_sequence):
            used_config = base_config if model == primary_model else base_config.model_copy(
                update={"model": model, "timeout": min(primary_timeout, 120.0)}
            )
            try:
                b64_images = await self._dispatcher.generate_image(used_config, prompt=prompt, size=size)
                if b64_images:
                    if idx == 0 and "gpt-image-2" in model.lower():
                        meta = {
                            "source_backend": "image2",
                            "requested_backend": "image2",
                            "requested_model": primary_model,
                            "actual_model": model,
                            "timeout_seconds": float(used_config.timeout or primary_timeout),
                            "fallback_used": False,
                            "image2_failure_reason": "",
                        }
                    else:
                        meta = {
                            "source_backend": "image_fallback",
                            "requested_backend": "image2" if "gpt-image-2" in primary_model.lower() else primary_model,
                            "requested_model": primary_model,
                            "actual_model": model,
                            "timeout_seconds": float(used_config.timeout or primary_timeout),
                            "fallback_used": idx != 0,
                            "fallback_model": model if idx != 0 else None,
                            "image2_failure_reason": image2_error[:1000] if image2_error else "",
                        }
                    return b64_images[0], used_config, meta
                errors.append(f"{model}: no image data")
            except Exception as exc:
                msg = f"{model}: {exc}"
                errors.append(msg)
                if "gpt-image-2" in model.lower() and not image2_error:
                    image2_error = str(exc)
                self.log(f"  image model {model} failed: {str(exc)[:220]}")
                continue
        if errors:
            raise RuntimeError("; ".join(errors)[-1800:])
        return None

    async def _diagnose_and_optimize_prompt(
        self,
        fig_key: str,
        original_prompt: str,
        last_error: str,
        figure_description: str,
    ) -> str | None:
        """Use LLM to diagnose why image generation failed and produce a shorter, optimized prompt."""
        diagnosis_system = (
            "You are an expert at debugging AI image generation failures. "
            "Analyze the error and the original prompt, then produce an optimized "
            "prompt that avoids the issue while preserving the figure's scientific content."
        )
        diagnosis_user = (
            f"The configured image generation API failed for figure '{fig_key}'.\n\n"
            f"Error: {last_error[:500]}\n\n"
            f"Original prompt ({len(original_prompt)} characters):\n"
            f"---\n{original_prompt[:3000]}\n---\n\n"
            f"Figure purpose: {figure_description[:500]}\n\n"
            f"Common failure causes:\n"
            f"1. Prompt too long or complex (some image generation APIs work best with <1500 chars)\n"
            f"2. Too many specific layout instructions (pixel sizes, hex colors, font specs)\n"
            f"3. Requesting text rendering that the model can't do well\n"
            f"4. Content that triggers safety filters\n\n"
            f"Tasks:\n"
            f"1. Diagnose the most likely cause of failure\n"
            f"2. Write an OPTIMIZED prompt (800-1200 characters) that:\n"
            f"   - Preserves the SAME figure content (method names, components, data flow)\n"
            f"   - Uses simpler, more descriptive language\n"
            f"   - Removes pixel-level instructions, specific hex codes, font sizes\n"
            f"   - Describes WHAT to draw, not HOW to render it\n"
            f"   - Keeps it as a clean, 2D scientific diagram\n\n"
            f"Return JSON:\n"
            f'{{"diagnosis": "brief explanation", "optimized_prompt": "the new prompt"}}'
        )

        figure_prompt_config = self.config.for_stage("figure_prompt")
        try:
            result = await self.generate_json(
                diagnosis_system, diagnosis_user, stage_override=figure_prompt_config,
            )
            diagnosis = result.get("diagnosis", "unknown")
            optimized = result.get("optimized_prompt", "")
            self.log(f"  {fig_key} diagnosis: {diagnosis}")

            if not optimized:
                return None

            # Enforce length limit on optimized prompt
            if len(optimized) > MAX_OPTIMIZED_PROMPT_LEN:
                optimized = optimized[:MAX_OPTIMIZED_PROMPT_LEN].rsplit(" ", 1)[0]

            self.log(f"  {fig_key} optimized prompt: {len(optimized)} chars")
            return optimized
        except Exception as exc:
            self.log(f"  {fig_key} prompt diagnosis failed: {exc}")
            return None

    # -----------------------------------------------------------------------
    # Code-generated charts (executed in subprocess)
    # -----------------------------------------------------------------------
