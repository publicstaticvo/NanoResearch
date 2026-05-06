from __future__ import annotations

from nanoresearch.profile import build_profile_seed, render_router_hindsight_context


def test_render_router_hindsight_context_for_planning_uses_router_policy_block() -> None:
    profile = build_profile_seed('resource_constrained_pragmatic')

    context = render_router_hindsight_context('planning', profile, enabled=True)

    assert 'SAME-ROUTER HINDSIGHT SDPO POLICY' in context
    assert 'router-policy decision step' in context
    assert 'Planning prompt focus:' in context


def test_render_router_hindsight_context_disabled_returns_empty_string() -> None:
    profile = build_profile_seed('resource_constrained_pragmatic')

    assert render_router_hindsight_context('planning', profile, enabled=False) == ''
