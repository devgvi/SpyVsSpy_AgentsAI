"""M5E.4: tests for the pure runtime-comparison building blocks in
`src/evaluation/runtime_comparison.py` - no real Bedrock/Strands/Game
calls, matching every prior milestone's network-free test pattern."""

from __future__ import annotations

import pytest

from src.evaluation.compatibility import CompatibilityStatus
from src.evaluation.episode_runner import EpisodeResult, EpisodeStatus
from src.evaluation.failure_streak import compute_long_horizon_stability
from src.evaluation.runtime_comparison import (
    VARIANT_A_NATIVE,
    VARIANT_B_STRANDS,
    action_efficiency_metrics,
    balanced_run_order,
    build_variant_seed_row,
    loop_and_invalid_action_metrics,
    paired_seed_narrative,
    run_paired_seeds_with_stop_rules,
)
from src.llm_controller import PROVIDER_AUTH_ERROR, PROVIDER_SERVICE_ERROR


def _episode(seed=701, variant_id=VARIANT_A_NATIVE, mission_success=False, decision_count=10, metrics=None) -> EpisodeResult:
    return EpisodeResult(
        experiment_id="m5e4-test", variant_id=variant_id, episode_id=f"ep-{variant_id}-{seed}", seed=seed,
        status=EpisodeStatus.DECISION_LIMIT_REACHED, compatibility_status=CompatibilityStatus.SUPPORTED,
        mission_success=mission_success, escape_success=mission_success, decision_count=decision_count,
        metrics=metrics or {}, game_facts={}, opponent_game_facts=None, competitive_facts=None,
    )


# -- balanced run order (spec section 15) ------------------------------------


def test_balanced_run_order_alternates_starting_with_native():
    order = balanced_run_order([701, 702, 703, 704, 705])
    assert order[701] == (VARIANT_A_NATIVE, VARIANT_B_STRANDS)
    assert order[702] == (VARIANT_B_STRANDS, VARIANT_A_NATIVE)
    assert order[703] == (VARIANT_A_NATIVE, VARIANT_B_STRANDS)
    assert order[704] == (VARIANT_B_STRANDS, VARIANT_A_NATIVE)
    assert order[705] == (VARIANT_A_NATIVE, VARIANT_B_STRANDS)


def test_balanced_run_order_never_always_native_first():
    order = balanced_run_order([701, 702, 703, 704])
    first_variants = [pair[0] for pair in order.values()]
    assert first_variants != [VARIANT_A_NATIVE] * len(first_variants)


# -- action efficiency (spec section 26) --------------------------------------


def test_action_efficiency_reports_decision_count_when_mission_completed():
    ep = _episode(mission_success=True, decision_count=27)
    metrics = action_efficiency_metrics(ep)
    assert metrics["decisions_to_mission_completion"] == 27
    assert metrics["decisions_to_escape"] == 27


def test_action_efficiency_null_when_mission_not_completed():
    ep = _episode(mission_success=False, decision_count=40)
    metrics = action_efficiency_metrics(ep)
    assert metrics["decisions_to_mission_completion"] is None
    assert metrics["decisions_to_escape"] is None


def test_action_efficiency_honestly_nulls_unmeasured_sub_events():
    ep = _episode(mission_success=True)
    metrics = action_efficiency_metrics(ep)
    assert metrics["decisions_to_first_room_exit"] is None
    assert metrics["decisions_to_first_mission_item"] is None
    assert metrics["decisions_to_first_successful_interaction"] is None


# -- loop/repetition + invalid-action metrics (spec sections 27-28) ----------


def test_loop_and_invalid_action_metrics_reads_existing_compute_metrics_fields():
    ep = _episode(metrics={
        "repeated_action_count": 3, "recent_repetitions_count": 1, "revisited_failed_target_count": 0,
        "same_illegal_action_repeated": 2, "illegal_actions": 1, "out_of_range_model_actions": 0,
        "unknown_target_actions": 0, "engine_rejected_actions": 0, "stale_actions": 0,
    })
    result = loop_and_invalid_action_metrics(ep)
    assert result["repeated_action_count"] == 3
    assert result["same_illegal_action_repeated"] == 2
    assert result["illegal_actions"] == 1


def test_loop_and_invalid_action_metrics_defaults_to_zero_when_absent():
    ep = _episode(metrics={})
    result = loop_and_invalid_action_metrics(ep)
    assert all(v == 0 for v in result.values())


# -- per-seed row (spec section 47) -------------------------------------------


def test_build_variant_seed_row_pulls_every_required_field():
    ep = _episode(
        seed=701, variant_id=VARIANT_A_NATIVE, mission_success=True, decision_count=20,
        metrics={
            "repair_attempts": 1, "repair_successes": 1, "fallbacks": 0,
            "unique_rooms_visited": 3, "mission_items_acquired": 2, "successful_interactions": 4,
            "traps_placed": 1, "traps_disarmed": 0, "latency_ms": {"median": 3000.0, "max": 5000.0},
            "input_tokens_total": 1000, "output_tokens_total": 100, "tool_calls": 20,
        },
    )
    decisions = [{"final_model_action_valid": True, "recovery_outcome": "FIRST_PASS_VALID"}] * 19 + [
        {"final_model_action_valid": True, "recovery_outcome": "REPAIRED_VALID"}
    ]
    long_horizon = compute_long_horizon_stability(decisions, [])
    row = build_variant_seed_row(ep, VARIANT_A_NATIVE, "gameplay_complete", long_horizon, [])

    assert row["seed"] == 701
    assert row["variant_id"] == VARIANT_A_NATIVE
    assert row["mission_completed"] is True
    assert row["rooms_visited"] == 3
    assert row["mission_items_acquired"] == 2
    assert row["successful_interactions"] == 4
    assert row["traps_placed"] == 1
    assert row["disarms"] == 0
    assert row["median_latency_ms"] == 3000.0
    # 19 needed no repair, 1 did (matches the fixture's own single
    # REPAIRED_VALID decision) - genuine first-pass validity, not the
    # post-repair total (which would be 20).
    assert row["first_pass_valid"] == 19
    assert row["input_tokens"] == 1000
    assert row["tool_call_count"] == 20
    assert row["decisions_to_mission_completion"] == 20


def test_build_variant_seed_row_counts_provider_failures_from_calls():
    ep = _episode()
    long_horizon = compute_long_horizon_stability([{"final_model_action_valid": True}], [])
    calls = [{"provider_error_type": PROVIDER_SERVICE_ERROR}]
    row = build_variant_seed_row(ep, VARIANT_A_NATIVE, "technical_failure", long_horizon, calls)
    assert row["provider_failures"] == 1


# -- paired seed narrative (spec section 49) ----------------------------------


def test_paired_seed_narrative_reports_mission_completion_divergence():
    row_a = build_variant_seed_row(
        _episode(mission_success=True, decision_count=15, metrics={"unique_rooms_visited": 3, "mission_items_acquired": 2}),
        VARIANT_A_NATIVE, "gameplay_complete", compute_long_horizon_stability([{"final_model_action_valid": True}], []), [],
    )
    row_b = build_variant_seed_row(
        _episode(mission_success=False, decision_count=40, metrics={"unique_rooms_visited": 3, "mission_items_acquired": 2}),
        VARIANT_B_STRANDS, "decision_limit_reached", compute_long_horizon_stability([{"final_model_action_valid": True}], []), [],
    )
    narrative = paired_seed_narrative(row_a, row_b)
    assert narrative["seed"] == row_a["seed"]
    assert any("Native completed the mission" in fact for fact in narrative["facts"])


def test_paired_seed_narrative_reports_no_divergence_when_rows_match():
    metrics = {"unique_rooms_visited": 2, "mission_items_acquired": 1, "repair_attempts": 0, "fallbacks": 0}
    lh = compute_long_horizon_stability([{"final_model_action_valid": True}], [])
    row_a = build_variant_seed_row(_episode(mission_success=False, metrics=metrics), VARIANT_A_NATIVE, "decision_limit_reached", lh, [])
    row_b = build_variant_seed_row(_episode(mission_success=False, metrics=metrics), VARIANT_B_STRANDS, "decision_limit_reached", lh, [])
    narrative = paired_seed_narrative(row_a, row_b)
    assert narrative["facts"] == ["no notable divergence in the recorded counters"]


# -- paired stop rules (spec section 58) --------------------------------------


class _FakeSpec:
    def __init__(self, seed, variant_id):
        self.seed = seed
        self.variant_id = variant_id
        self.episode_id = f"m5e4-test__{variant_id}__seed{seed}"


def _episode_spec_for(seed, variant_id):
    return _FakeSpec(seed, variant_id)


def test_paired_run_covers_both_variants_for_every_seed_in_balanced_order():
    order = balanced_run_order([701, 702])
    outcome = run_paired_seeds_with_stop_rules(
        [701, 702], order, _episode_spec_for,
        run_one_episode=lambda spec: _episode(seed=spec.seed, variant_id=spec.variant_id),
        load_provider_calls=lambda episode_id: [],
        classify=lambda ep, calls: "decision_limit_reached",
    )
    ran = [(spec.seed, spec.variant_id) for spec, _ in outcome.episodes]
    assert ran == [(701, VARIANT_A_NATIVE), (701, VARIANT_B_STRANDS), (702, VARIANT_B_STRANDS), (702, VARIANT_A_NATIVE)]
    assert outcome.stop_reason is None
    assert outcome.tasks_not_run == []


def test_paired_run_stops_on_auth_expiry_and_preserves_completed_episodes():
    order = balanced_run_order([701, 702, 703])
    calls_by_episode = {f"m5e4-test__{VARIANT_B_STRANDS}__seed701": [{"provider_error_type": PROVIDER_AUTH_ERROR}]}
    outcome = run_paired_seeds_with_stop_rules(
        [701, 702, 703], order, _episode_spec_for,
        run_one_episode=lambda spec: _episode(seed=spec.seed, variant_id=spec.variant_id),
        load_provider_calls=lambda episode_id: calls_by_episode.get(episode_id, []),
        classify=lambda ep, calls: "decision_limit_reached",
    )
    assert outcome.stop_reason == "auth_token_expired_mid_run"
    ran = [(spec.seed, spec.variant_id) for spec, _ in outcome.episodes]
    assert ran == [(701, VARIANT_A_NATIVE), (701, VARIANT_B_STRANDS)]
    assert (702, VARIANT_B_STRANDS) in outcome.tasks_not_run


def test_paired_run_tracks_failure_signatures_independently_per_runtime():
    """Two Native failures with the same signature must trigger the
    stop; two failures split one-each between Native and Strands must
    NOT (spec section 58: "same runtime")."""
    order = balanced_run_order([701, 702])
    native_episode_ids = {f"m5e4-test__{VARIANT_A_NATIVE}__seed701", f"m5e4-test__{VARIANT_A_NATIVE}__seed702"}
    calls_by_episode = {eid: [{"provider_error_type": PROVIDER_SERVICE_ERROR}] for eid in native_episode_ids}

    def classify(ep, calls):
        return "technical_failure" if ep.variant_id == VARIANT_A_NATIVE else "decision_limit_reached"

    outcome = run_paired_seeds_with_stop_rules(
        [701, 702], order, _episode_spec_for,
        run_one_episode=lambda spec: _episode(seed=spec.seed, variant_id=spec.variant_id),
        load_provider_calls=lambda episode_id: calls_by_episode.get(episode_id, []),
        classify=classify,
    )
    assert outcome.stop_reason == "repeated_failure_signature"
    assert "runtime=" + VARIANT_A_NATIVE in outcome.stop_detail


def test_paired_run_split_failure_across_runtimes_does_not_stop():
    order = balanced_run_order([701, 702])
    calls_by_episode = {
        f"m5e4-test__{VARIANT_A_NATIVE}__seed701": [{"provider_error_type": PROVIDER_SERVICE_ERROR}],
        f"m5e4-test__{VARIANT_B_STRANDS}__seed701": [{"provider_error_type": PROVIDER_SERVICE_ERROR}],
    }

    def classify(ep, calls):
        return "technical_failure" if calls else "decision_limit_reached"

    outcome = run_paired_seeds_with_stop_rules(
        [701, 702], order, _episode_spec_for,
        run_one_episode=lambda spec: _episode(seed=spec.seed, variant_id=spec.variant_id),
        load_provider_calls=lambda episode_id: calls_by_episode.get(episode_id, []),
        classify=classify,
    )
    assert outcome.stop_reason is None
    assert len(outcome.episodes) == 4


def test_paired_seed_narrative_never_makes_subjective_claims():
    """Spec section 49: no speculation about hidden reasoning - every
    fact string must be built only from the objective counters, never a
    free-text judgment."""
    row_a = build_variant_seed_row(_episode(metrics={}), VARIANT_A_NATIVE, "technical_failure", compute_long_horizon_stability([], []), [])
    row_b = build_variant_seed_row(_episode(metrics={}), VARIANT_B_STRANDS, "gameplay_complete", compute_long_horizon_stability([], []), [])
    narrative = paired_seed_narrative(row_a, row_b)
    forbidden_words = {"smarter", "better", "confused", "intelligent", "strategy"}
    joined = " ".join(narrative["facts"]).lower()
    assert not any(word in joined for word in forbidden_words)
