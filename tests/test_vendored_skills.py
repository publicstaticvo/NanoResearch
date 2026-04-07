from __future__ import annotations

from pathlib import Path

from nanoresearch.evolution.skills import SkillDomain, SkillEvolutionStore
from nanoresearch.profile import build_profile_seed
from nanoresearch.skills import SkillMatcher, UnifiedSkillMatcher


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / "skills"
VENDORED_ROOT = SKILLS_ROOT / "vendor-ai-research"
MANIFEST_PATH = VENDORED_ROOT / "manifest.json"


def _matcher() -> SkillMatcher:
    return SkillMatcher([SKILLS_ROOT, VENDORED_ROOT], manifest_path=MANIFEST_PATH)


def test_vendored_manifest_indexes_subset() -> None:
    matcher = _matcher()

    registry = matcher.export_review_registry()
    skill_ids = {item["skill_id"] for item in registry}

    assert MANIFEST_PATH.is_file()
    assert "vendor.autoresearch" in skill_ids
    assert "vendor.ml-paper-writing" in skill_ids
    assert "vendor.peft-fine-tuning" in skill_ids


def test_stage_aware_static_retrieval_distinguishes_writing_and_experiment() -> None:
    matcher = _matcher()

    writing_matches = matcher.match_writing_skills(
        topic="Write a NeurIPS robustness paper with citations and latex figures.",
        template_format="neurips",
        extra_text="Need plotting and paper-writing guidance.",
        task_type="writing",
    )
    experiment_matches = matcher.match(
        {
            "title": "PEFT finetuning for robustness",
            "proposed_method": {
                "name": "QLoRA + evaluation harness",
                "description": "Fine-tune with LoRA, bf16, and evaluate with lm-eval-harness.",
                "key_components": ["peft", "qlora", "accelerate", "evaluation"],
            },
            "datasets": ["cifar10"],
            "metrics": ["accuracy"],
            "baselines": ["baseline lora"],
        },
        task_type="experiment",
    )

    writing_ids = {entry.source_id for entry, _ in writing_matches}
    experiment_ids = {entry.source_id for entry, _ in experiment_matches}

    assert "vendor.ml-paper-writing" in writing_ids
    assert "vendor.peft-fine-tuning" not in writing_ids
    assert "vendor.peft-fine-tuning" in experiment_ids
    assert "vendor.ml-paper-writing" not in experiment_ids


def test_unified_context_tracks_static_matches_with_profile() -> None:
    profile = build_profile_seed("cv_visual_conference")
    matcher = UnifiedSkillMatcher(
        skills_dirs=[SKILLS_ROOT, VENDORED_ROOT],
        manifest_path=MANIFEST_PATH,
        retrieval_top_k=4,
    )

    context = matcher.build_context(
        "writing",
        topic="Write a NeurIPS paper on robust image classification.",
        template_format="neurips",
        text="Need academic writing, latex, citations, and figure guidance.",
        profile=profile,
    )

    assert context.candidate_static_skills
    assert any(skill_id == "vendor.ml-paper-writing" for skill_id in context.matched_static_skills)
    assert any("ml-paper-writing" in skill_name for skill_name in context.matched_skills)
    assert context.matched_script_skills == []


def test_review_skill_candidate_discards_duplicate_static_skill(tmp_path: Path) -> None:
    matcher = _matcher()
    store = SkillEvolutionStore(root=tmp_path / "skill-store", enabled=True)
    store.set_static_skill_registry(matcher.export_review_registry())

    candidate = store.extract_skill_candidate(
        domain=SkillDomain.WRITING,
        trigger_pattern="paper_writing",
        source_trace="Need a publication-ready ML paper with LaTeX templates, verified citations, and conference-ready formatting.",
        name="Paper Writing Guidance",
        description="Publication-ready ML paper writing guidance.",
        when_to_use="Use when drafting ML papers for NeurIPS or ICML.",
        instructions=[
            "Use LaTeX templates and conference-specific formatting.",
            "Verify citations programmatically instead of writing BibTeX from memory.",
        ],
        tags=["writing", "latex", "citations"],
    )

    assert candidate is not None
    review = store.review_skill_candidate(candidate)

    assert review.decision.value == "discard"
    assert review.reviewed_against_static_skill_ids
    assert "vendor.ml-paper-writing" in review.reviewed_against_static_skill_ids


def test_review_skill_candidate_keeps_local_adaptation_beside_static_skill(tmp_path: Path) -> None:
    matcher = _matcher()
    store = SkillEvolutionStore(root=tmp_path / "skill-store", enabled=True)
    store.set_static_skill_registry(matcher.export_review_registry())

    candidate = store.extract_skill_candidate(
        domain=SkillDomain.WRITING,
        trigger_pattern="nature_dense_captions",
        source_trace="Nature-style paper revisions repeatedly needed dense self-contained captions with explicit axis descriptions and statistical notation.",
        name="Nature Dense Caption Adaptation",
        description="Local adaptation for Nature/Springer-style dense scientific captions.",
        when_to_use="Use when writing journal-style AI-for-Science figures with dense self-contained captions.",
        instructions=[
            "Write self-contained captions that define every panel, axis, and symbol in the figure.",
            "Include statistical qualifiers and data provenance cues directly in the caption when targeting Nature/Springer-style submissions.",
        ],
        tags=["writing", "nature", "captions"],
    )

    assert candidate is not None
    review = store.review_skill_candidate(candidate)

    assert review.decision.value == "add_new"
    assert review.reviewed_against_static_skill_ids
