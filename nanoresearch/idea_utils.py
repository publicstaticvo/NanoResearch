"""Helpers for normalizing legacy hypothesis fields to idea-centric accessors."""

from __future__ import annotations

from typing import Any


def get_idea_candidates(ideation_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    data = ideation_data if isinstance(ideation_data, dict) else {}
    ideas = data.get("ideas")
    if isinstance(ideas, list):
        return [item for item in ideas if isinstance(item, dict)]
    hypotheses = data.get("hypotheses")
    if isinstance(hypotheses, list):
        return [item for item in hypotheses if isinstance(item, dict)]
    return []


def get_selected_idea_id(ideation_data: dict[str, Any] | None) -> str:
    data = ideation_data if isinstance(ideation_data, dict) else {}
    selected = str(data.get("selected_idea") or data.get("selected_hypothesis") or "").strip()
    if selected:
        return selected
    ideas = get_idea_candidates(data)
    if ideas:
        first = ideas[0]
        return str(first.get("idea_id") or first.get("hypothesis_id") or "").strip()
    return ""


def get_idea_id(candidate: dict[str, Any] | None, fallback: str = "") -> str:
    data = candidate if isinstance(candidate, dict) else {}
    return str(data.get("idea_id") or data.get("hypothesis_id") or fallback).strip()


def get_selected_idea(ideation_data: dict[str, Any] | None) -> dict[str, Any]:
    selected = get_selected_idea_id(ideation_data)
    for idea in get_idea_candidates(ideation_data):
        if get_idea_id(idea) == selected:
            return idea
    ideas = get_idea_candidates(ideation_data)
    return ideas[0] if ideas else {}


def get_blueprint_idea_ref(blueprint_data: dict[str, Any] | None) -> str:
    data = blueprint_data if isinstance(blueprint_data, dict) else {}
    return str(data.get("idea_ref") or data.get("hypothesis_ref") or "").strip()


def add_idea_aliases_to_ideation(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload or {})
    ideas = get_idea_candidates(result)
    selected_idea = get_selected_idea_id(result)
    if ideas:
        aliased: list[dict[str, Any]] = []
        for idx, idea in enumerate(ideas, start=1):
            row = dict(idea)
            idea_id = get_idea_id(row, fallback=f"IDEA-{idx:03d}")
            row.setdefault("idea_id", idea_id)
            row.setdefault("hypothesis_id", idea_id)
            aliased.append(row)
        result["ideas"] = aliased
        result["hypotheses"] = aliased
    if selected_idea:
        result["selected_idea"] = selected_idea
        result["selected_hypothesis"] = selected_idea
    return result


def add_idea_aliases_to_blueprint(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload or {})
    idea_ref = get_blueprint_idea_ref(result)
    if idea_ref:
        result["idea_ref"] = idea_ref
        result["hypothesis_ref"] = idea_ref
    return result
