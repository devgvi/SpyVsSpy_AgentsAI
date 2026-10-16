"""M5E.2 spec section 31: network-free tests for the controlled
competitive benchmark's config, seed pairing, technical-failure exclusion,
and aggregation logic."""

from __future__ import annotations

import json

from src.agents.registry import create_agent_runtime
from src.evaluation.competitive_aggregation import (
    aggregate_both_escaped_count,
    aggregate_first_escape_counts,
    aggregate_trap_totals,
    filter_technically_valid_episodes,
    is_technical_failure,
    three_way_seed_pairing,
)
from src.evaluation.compatibility import CompatibilityStatus
from src.evaluation.episode_runner import EpisodeResult, EpisodeStatus
from src.evaluation.spec import ExperimentSpec
from src.game import Game


# -- experiment config shape ----------------------------------------------

def _load(path: str) -> ExperimentSpec:
    with open(path, "r", encoding="utf-8") as f:
        return ExperimentSpec.from_json(f.read())


def test_m5e2_full_config_has_five_shared_seeds_and_frozen_num_ctx():
    spec = _load("experiments/m5e2_controlled_competitive_v1.json")
    assert list(spec.seeds) == [501, 502, 503, 504, 505]
    llm_variants = [v for v in spec.variants if v.provider == "ollama"]
    assert len(llm_variants) == 2
    for v in llm_variants:
        assert v.num_ctx == 16384
        assert v.model == "gemma4:latest"
        assert v.opponent_type == "scripted_active"


def test_m5e2_pilot_config_is_one_seed_same_variants_as_full():
    pilot = _load("experiments/m5e2_pilot.json")
    full = _load("experiments/m5e2_controlled_competitive_v1.json")
    assert list(pilot.seeds) == [501]
    assert {v.variant_id for v in pilot.variants} == {v.variant_id for v in full.variants}


def test_m5e2_config_variant_ids_are_native_strands_scripted():
    spec = _load("experiments/m5e2_controlled_competitive_v1.json")
    runtimes = {v.agent_runtime for v in spec.variants}
    assert runtimes == {"native_tool_like", "strands", "scripted"}


# -- num_ctx propagation into Game/OllamaClient ----------------------------

def test_num_ctx_reaches_the_native_ollama_client_via_game():
    game = Game(white_controller="llm", agent_runtime="native_tool_like", llm_provider="ollama", num_ctx=16384)
    assert game.bot_controller.controller.model_client.num_ctx == 16384


def test_num_ctx_none_by_default_preserves_prior_behavior():
    game = Game(white_controller="llm", agent_runtime="native_tool_like", llm_provider="ollama")
    assert game.bot_controller.controller.model_client.num_ctx is None


# -- Strands adapter version requirement -------------------------------------

def test_strands_runtime_is_adapter_v3():
    """Guards that M5E.1.1's watchdog fix stays present - the version was
    bumped again to v3 for M5E.4's added Bedrock model backend support,
    but the underlying `_invoke_strands` watchdog this test cares about is
    unchanged."""
    adapter = create_agent_runtime("strands", actor_id="white", agent_factory=lambda *a, **k: None)
    assert adapter.runtime_version == "strands-adapter-v3"


# -- matched seed generation -------------------------------------------------

def test_expand_produces_identical_seed_set_across_all_three_variants():
    spec = _load("experiments/m5e2_controlled_competitive_v1.json")
    episode_specs = spec.expand()
    by_variant: dict[str, set[int]] = {}
    for ep in episode_specs:
        by_variant.setdefault(ep.variant_id, set()).add(ep.seed)
    seed_sets = list(by_variant.values())
    assert all(s == seed_sets[0] for s in seed_sets)
    assert seed_sets[0] == {501, 502, 503, 504, 505}


# -- technical-failure exclusion / three-way pairing / aggregation --------

def _fake_episode(variant_id, seed, status, competitive_facts=None, decision_count=5):
    return EpisodeResult(
        experiment_id="m5e2-test",
        variant_id=variant_id,
        episode_id=f"m5e2-test__{variant_id}__seed{seed}",
        seed=seed,
        status=status,
        compatibility_status=CompatibilityStatus.SUPPORTED,
        decision_count=decision_count,
        game_facts={},
        competitive_facts=competitive_facts,
    )


def test_is_technical_failure_classifies_statuses_correctly():
    assert is_technical_failure(_fake_episode("A", 1, EpisodeStatus.ABORTED)) is True
    assert is_technical_failure(_fake_episode("A", 1, EpisodeStatus.TIMED_OUT)) is True
    assert is_technical_failure(_fake_episode("A", 1, EpisodeStatus.FAILED)) is True
    assert is_technical_failure(_fake_episode("A", 1, EpisodeStatus.COMPLETED)) is False
    assert is_technical_failure(_fake_episode("A", 1, EpisodeStatus.DECISION_LIMIT_REACHED)) is False


def test_filter_technically_valid_episodes_excludes_only_technical_failures():
    episodes = [
        _fake_episode("A", 1, EpisodeStatus.COMPLETED),
        _fake_episode("A", 2, EpisodeStatus.ABORTED),
        _fake_episode("A", 3, EpisodeStatus.DECISION_LIMIT_REACHED),
    ]
    valid = filter_technically_valid_episodes(episodes)
    assert {e.seed for e in valid} == {1, 3}


def test_three_way_seed_pairing_only_includes_seeds_present_in_all_three():
    episodes = [
        _fake_episode("A", 1, EpisodeStatus.COMPLETED),
        _fake_episode("B", 1, EpisodeStatus.COMPLETED),
        _fake_episode("C", 1, EpisodeStatus.COMPLETED),
        _fake_episode("A", 2, EpisodeStatus.COMPLETED),
        _fake_episode("B", 2, EpisodeStatus.COMPLETED),
        # seed 2 missing for C -> must not appear in the paired rows
    ]
    rows = three_way_seed_pairing(episodes, ("A", "B", "C"))
    assert [r["seed"] for r in rows] == [1]
    assert rows[0]["A"]["technical_status"] == "COMPLETED"


def test_three_way_seed_pairing_flags_technical_failure_per_cell():
    episodes = [
        _fake_episode("A", 1, EpisodeStatus.ABORTED),
        _fake_episode("B", 1, EpisodeStatus.COMPLETED),
        _fake_episode("C", 1, EpisodeStatus.COMPLETED),
    ]
    rows = three_way_seed_pairing(episodes, ("A", "B", "C"))
    assert rows[0]["A"]["is_technical_failure"] is True
    assert rows[0]["B"]["is_technical_failure"] is False


def test_aggregate_first_escape_counts_excludes_technical_failures():
    episodes = [
        _fake_episode("A", 1, EpisodeStatus.COMPLETED, competitive_facts={"first_escape_actor": "white_spy", "both_escaped": False}),
        _fake_episode("A", 2, EpisodeStatus.COMPLETED, competitive_facts={"first_escape_actor": "black_spy", "both_escaped": False}),
        _fake_episode("A", 3, EpisodeStatus.COMPLETED, competitive_facts={"first_escape_actor": None, "both_escaped": False}),
        _fake_episode("A", 4, EpisodeStatus.ABORTED, competitive_facts={"first_escape_actor": "white_spy", "both_escaped": True}),
    ]
    counts = aggregate_first_escape_counts(episodes)
    assert counts == {"agent_first": 1, "opponent_first": 1, "neither": 1}  # seed 4 (ABORTED) excluded


def test_aggregate_both_escaped_count_excludes_technical_failures():
    episodes = [
        _fake_episode("A", 1, EpisodeStatus.COMPLETED, competitive_facts={"both_escaped": True}),
        _fake_episode("A", 2, EpisodeStatus.COMPLETED, competitive_facts={"both_escaped": False}),
        _fake_episode("A", 3, EpisodeStatus.ABORTED, competitive_facts={"both_escaped": True}),
    ]
    assert aggregate_both_escaped_count(episodes) == 1


def test_aggregate_trap_totals_sums_correctly_and_excludes_technical_failures():
    episodes = [
        _fake_episode("A", 1, EpisodeStatus.COMPLETED, competitive_facts={
            "agent_traps_placed": 2, "opponent_traps_placed": 1,
            "agent_triggered_opponent_traps": 1, "opponent_triggered_agent_traps": 0,
            "agent_stuns": 0, "opponent_stuns": 1,
        }),
        _fake_episode("A", 2, EpisodeStatus.COMPLETED, competitive_facts={
            "agent_traps_placed": 3, "opponent_traps_placed": 2,
            "agent_triggered_opponent_traps": 0, "opponent_triggered_agent_traps": 1,
            "agent_stuns": 1, "opponent_stuns": 0,
        }),
        _fake_episode("A", 3, EpisodeStatus.ABORTED, competitive_facts={
            "agent_traps_placed": 99, "opponent_traps_placed": 99,
            "agent_triggered_opponent_traps": 99, "opponent_triggered_agent_traps": 99,
            "agent_stuns": 99, "opponent_stuns": 99,
        }),
    ]
    totals = aggregate_trap_totals(episodes)
    assert totals == {
        "agent_traps_placed": 5,
        "opponent_traps_placed": 3,
        "agent_triggered_opponent_traps": 1,
        "opponent_triggered_agent_traps": 1,
        "agent_stuns": 1,
        "opponent_stuns": 1,
    }


# -- historical outputs not overwritten -------------------------------------

def test_m5e2_output_directory_name_is_distinct_from_every_prior_milestone():
    spec = _load("experiments/m5e2_controlled_competitive_v1.json")
    prior_ids = {
        "m5d1-local-baseline-v1", "m5d3-corrected-mission-v1", "m5d6-corrected-mission-v2",
        "m5d7-runtime-representation-v1", "m5d8-gameplay-representation-v1",
        "m5e-active-opponent-v1", "m5e1-long-horizon-stability-v1",
        "m5e1_1-ollama-provider-stability-v1", "m5e1_1-gameplay-check-v1",
    }
    assert spec.experiment_id not in prior_ids
    assert spec.experiment_id == "m5e2-controlled-competitive-v1"


# -- episode-level technical-status classification (spec section 8) -------


def test_classify_episode_technical_status_gameplay_completion():
    from src.evaluation.competitive_aggregation import GAMEPLAY_COMPLETION, classify_episode_technical_status

    ep = _fake_episode("A", 1, EpisodeStatus.COMPLETED)
    ep.mission_success = True
    assert classify_episode_technical_status(ep) == GAMEPLAY_COMPLETION


def test_classify_episode_technical_status_deadline_completion():
    from src.evaluation.competitive_aggregation import DEADLINE_COMPLETION, classify_episode_technical_status

    ep = _fake_episode("A", 1, EpisodeStatus.COMPLETED)
    ep.mission_success = False
    assert classify_episode_technical_status(ep) == DEADLINE_COMPLETION


def test_classify_episode_technical_status_decision_cap_completion_when_mostly_valid():
    from src.evaluation.competitive_aggregation import DECISION_CAP_COMPLETION, classify_episode_technical_status
    from src.evaluation.failure_streak import compute_long_horizon_stability

    ep = _fake_episode("A", 1, EpisodeStatus.DECISION_LIMIT_REACHED)
    decisions = [{"decision_number": i, "final_model_action_valid": True} for i in range(1, 15)]
    stability = compute_long_horizon_stability(decisions, [{"provider_error_type": None}] * 14)
    assert classify_episode_technical_status(ep, stability) == DECISION_CAP_COMPLETION


def test_classify_episode_technical_status_reclassifies_as_technical_failure_when_majority_invalid():
    # The real M5E.2 finding: DECISION_LIMIT_REACHED whose budget was
    # actually consumed by persistent provider failures, not real
    # gameplay - must not be reported as ordinary decision-cap completion.
    from src.evaluation.competitive_aggregation import TECHNICAL_FAILURE, classify_episode_technical_status
    from src.evaluation.failure_streak import compute_long_horizon_stability

    ep = _fake_episode("A", 1, EpisodeStatus.DECISION_LIMIT_REACHED)
    decisions = [{"decision_number": i, "final_model_action_valid": (i <= 2)} for i in range(1, 16)]
    provider_calls = [{"provider_error_type": None}] * 2 + [
        {"provider_error_type": "provider_timeout", "provider_error_message_normalized": "timed out"}
    ] * 13
    stability = compute_long_horizon_stability(decisions, provider_calls)
    assert classify_episode_technical_status(ep, stability) == TECHNICAL_FAILURE


def test_classify_episode_technical_status_harness_level_failure_always_wins():
    from src.evaluation.competitive_aggregation import TECHNICAL_FAILURE, classify_episode_technical_status
    from src.evaluation.failure_streak import compute_long_horizon_stability

    ep = _fake_episode("A", 1, EpisodeStatus.ABORTED)
    decisions = [{"decision_number": i, "final_model_action_valid": True} for i in range(1, 15)]
    stability = compute_long_horizon_stability(decisions, [{"provider_error_type": None}] * 14)
    assert classify_episode_technical_status(ep, stability) == TECHNICAL_FAILURE


def test_filter_technically_valid_episodes_uses_richer_classification_when_given():
    from src.evaluation.competitive_aggregation import TECHNICAL_FAILURE, filter_technically_valid_episodes

    episodes = [
        _fake_episode("A", 1, EpisodeStatus.DECISION_LIMIT_REACHED),
        _fake_episode("A", 2, EpisodeStatus.DECISION_LIMIT_REACHED),
    ]
    status_map = {episodes[0].episode_id: "decision_cap_completion", episodes[1].episode_id: TECHNICAL_FAILURE}
    valid = filter_technically_valid_episodes(episodes, status_map)
    assert [e.seed for e in valid] == [1]
