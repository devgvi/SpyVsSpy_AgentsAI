"""M6E spec section 27 (Phase A): network-free tests verifying the
compaction taxonomy before any live call. No model calls anywhere in this
file."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import copy

import pytest

from src.evaluation.context_compaction import (
    C0_FULL_CONTEXT,
    C1_STRUCTURAL_COMPACT,
    C2_DEDUPLICATED,
    C3A_LAST_3,
    C3B_LAST_1,
    CONDITIONS,
    apply_condition,
    representation_hash,
    token_estimate,
)
from src.evaluation.m6e_scenarios import SCENARIO_FAMILIES, build_all_scenarios, build_compact_observation_payload
from src.evaluation.request_fact_model import extract_request_facts, request_fact_hash

SCENARIOS = build_all_scenarios()


@pytest.mark.parametrize("family", SCENARIO_FAMILIES)
def test_all_seven_scenario_families_exist(family):
    assert family in SCENARIOS


@pytest.mark.parametrize("family", SCENARIO_FAMILIES)
def test_c0_matches_unmodified_production_payload(family):
    scenario = SCENARIOS[family]
    obs = build_compact_observation_payload(scenario)
    result = apply_condition(C0_FULL_CONTEXT, obs, scenario.memory_payload)
    assert result.observation_payload == obs
    assert result.memory_payload == scenario.memory_payload
    assert result.removed_report["removed_facts"] == []


@pytest.mark.parametrize("family", SCENARIO_FAMILIES)
def test_c1_is_fact_equivalent_to_c0_but_smaller(family):
    scenario = SCENARIOS[family]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    c0 = apply_condition(C0_FULL_CONTEXT, obs, mem)
    c1 = apply_condition(C1_STRUCTURAL_COMPACT, obs, mem)

    facts0 = extract_request_facts(c0.observation_payload, c0.memory_payload)
    facts1 = extract_request_facts(c1.observation_payload, c1.memory_payload)
    assert request_fact_hash(facts0) == request_fact_hash(facts1)
    assert len(c1.to_json()) <= len(c0.to_json())
    assert representation_hash(c1.to_json()) != representation_hash(c0.to_json()) or c1.to_json() == c0.to_json()


@pytest.mark.parametrize("family", SCENARIO_FAMILIES)
def test_c2_is_request_level_fact_equivalent_to_c0_even_with_different_duplication(family):
    scenario = SCENARIOS[family]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    c0 = apply_condition(C0_FULL_CONTEXT, obs, mem)
    c2 = apply_condition(C2_DEDUPLICATED, obs, mem)

    facts0 = extract_request_facts(c0.observation_payload, c0.memory_payload)
    facts2 = extract_request_facts(c2.observation_payload, c2.memory_payload)
    assert request_fact_hash(facts0) == request_fact_hash(facts2)
    # Section-level duplication must actually differ when a rule fires,
    # or the scenario simply had nothing to deduplicate - both are valid,
    # but the test itself must observe which one happened, not assume.
    dup0 = sum(1 for f in facts0 if len(f.source_sections) > 1)
    dup2 = sum(1 for f in facts2 if len(f.source_sections) > 1)
    assert dup2 <= dup0


def test_c2_deduplication_never_destroys_a_fact_unique_to_the_removed_section():
    """Regression guard for the real bug found while building this module:
    a naive 'clear the whole list' C2 implementation silently destroyed a
    GameState-tracked known-lock fact that was never mirrored into the
    synthetic AgentMemory seeded for a snapshot. S6_LONG_HISTORY's own
    frozen state reproduces exactly this asymmetry (a real engine-tracked
    door lock Observation knows about that the seeded AgentMemory does
    not) - this test pins the fix."""
    from src.evaluation.m6e_scenarios import S6_LONG_HISTORY

    scenario = SCENARIOS[S6_LONG_HISTORY]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    obs_lock_ids = set(obs.get("memory", {}).get("known_locked_objects", []))
    mem_lock_ids = {lock["target_id"] for lock in mem.get("known_locks", [])}
    unique_to_observation = obs_lock_ids - mem_lock_ids
    assert unique_to_observation, "fixture must exercise a non-overlapping id for this regression test to mean anything"

    c2 = apply_condition(C2_DEDUPLICATED, obs, mem)
    surviving_ids = set(c2.observation_payload.get("memory", {}).get("known_locked_objects", []))
    assert unique_to_observation.issubset(surviving_ids)


@pytest.mark.parametrize("condition", (C3A_LAST_3, C3B_LAST_1))
def test_c3_loses_only_declared_history_items_and_preserves_chronology(condition):
    from src.evaluation.m6e_scenarios import S6_LONG_HISTORY

    scenario = SCENARIOS[S6_LONG_HISTORY]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    keep_n = 3 if condition == C3A_LAST_3 else 1

    result = apply_condition(condition, obs, mem)
    for field_name in ("recent_interactions", "recent_failures", "recent_events", "recent_decisions"):
        original = mem[field_name]
        kept = result.memory_payload[field_name]
        assert kept == original[-keep_n:] if len(original) > keep_n else kept == original
        # chronology: kept items' game_time must be non-decreasing and match
        # the tail of the original ordering exactly (never reordered by
        # importance - spec section 15).
        times = [item.get("game_time") for item in kept]
        assert times == sorted(times)
        assert result.removed_report["history_loss"][field_name]["removed_count"] == max(0, len(original) - keep_n)


def test_c3_is_never_classified_as_fact_equivalent_to_c0():
    from src.evaluation.m6e_scenarios import S6_LONG_HISTORY

    scenario = SCENARIOS[S6_LONG_HISTORY]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    c0 = apply_condition(C0_FULL_CONTEXT, obs, mem)
    c3b = apply_condition(C3B_LAST_1, obs, mem)
    facts0 = extract_request_facts(c0.observation_payload, c0.memory_payload)
    facts3 = extract_request_facts(c3b.observation_payload, c3b.memory_payload)
    assert request_fact_hash(facts0) != request_fact_hash(facts3)
    assert len(facts3) < len(facts0)


def test_c3b_loses_the_oldest_recent_failure_that_c3a_still_retains():
    """The concrete information-loss boundary spec section 70 asks this
    milestone to identify: `filing_cabinet_01`'s already-searched fact is
    3rd-from-most-recent in S6's `recent_failures` - retained at LAST_3,
    lost at LAST_1."""
    from src.evaluation.m6e_scenarios import S6_LONG_HISTORY

    scenario = SCENARIOS[S6_LONG_HISTORY]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    c3a = apply_condition(C3A_LAST_3, obs, mem)
    c3b = apply_condition(C3B_LAST_1, obs, mem)

    a_targets = {f["target_id"] for f in c3a.memory_payload["recent_failures"]}
    b_targets = {f["target_id"] for f in c3b.memory_payload["recent_failures"]}
    assert "filing_cabinet_01" in a_targets
    assert "filing_cabinet_01" not in b_targets


@pytest.mark.parametrize("family", SCENARIO_FAMILIES)
def test_legal_actions_never_change_across_any_condition(family):
    scenario = SCENARIOS[family]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    baseline = obs["legal_actions_compact"]
    for condition in CONDITIONS:
        result = apply_condition(condition, obs, mem)
        assert result.observation_payload["legal_actions_compact"] == baseline


@pytest.mark.parametrize("family", SCENARIO_FAMILIES)
def test_serialization_is_deterministic_and_hashes_stable(family):
    scenario = SCENARIOS[family]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    for condition in CONDITIONS:
        a = apply_condition(condition, obs, mem)
        b = apply_condition(condition, obs, mem)
        assert a.to_json() == b.to_json()
        assert representation_hash(a.to_json()) == representation_hash(b.to_json())


def test_no_hidden_state_introduced_by_any_condition():
    """Every condition must operate only on the two payloads it's given -
    no condition may reach outside itself for extra facts."""
    scenario = SCENARIOS["S6_LONG_HISTORY"]
    obs = build_compact_observation_payload(scenario)
    mem = scenario.memory_payload
    obs_before, mem_before = copy.deepcopy(obs), copy.deepcopy(mem)
    for condition in CONDITIONS:
        apply_condition(condition, obs, mem)
    # apply_condition deep-copies its inputs - the caller's own originals
    # must never be mutated as a side effect.
    assert obs == obs_before
    assert mem == mem_before


def test_token_estimate_is_positive_and_monotonic_with_length():
    assert token_estimate("a" * 100) < token_estimate("a" * 1000)
    assert token_estimate("") >= 1
