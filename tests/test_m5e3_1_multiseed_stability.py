"""M5E.3.1: tests for the pure multi-seed stability building blocks in
`src/evaluation/multiseed_stability.py` - no real Bedrock/Game calls
anywhere, matching the network-free pattern every prior milestone's test
suite uses (spec section 35-36)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.evaluation.competitive_aggregation import TECHNICAL_FAILURE
from src.evaluation.episode_runner import EpisodeResult, EpisodeStatus
from src.evaluation.compatibility import CompatibilityStatus
from src.evaluation.failure_streak import compute_long_horizon_stability
from src.evaluation.multiseed_stability import (
    DECISION_LIMIT_REACHED,
    GAMEPLAY_COMPLETE,
    MISSION_DEADLINE_REACHED,
    build_aggregate_table,
    build_per_seed_row,
    classify_episode_for_m5e3_1,
    count_provider_failures_by_category,
    cumulative_bedrock_decisions,
    detect_sleep_contamination,
    episode_has_auth_expiry,
    failure_signature,
    is_technically_clean,
    resolve_output_dir,
    run_seeds_with_stop_rules,
)
from src.evaluation.spec import EpisodeSpec, ExperimentSpec, VariantSpec
from src.llm_controller import PROVIDER_AUTH_ERROR, PROVIDER_SERVICE_ERROR


def _episode(status=EpisodeStatus.DECISION_LIMIT_REACHED, mission_success=False, seed=601, error_type=None) -> EpisodeResult:
    return EpisodeResult(
        experiment_id="m5e3_1-test", variant_id="A_native_tool_like_bedrock", episode_id=f"ep-{seed}", seed=seed,
        status=status, compatibility_status=CompatibilityStatus.SUPPORTED, mission_success=mission_success,
        escape_success=mission_success, error_type=error_type, metrics={}, decision_count=1,
        game_facts={}, opponent_game_facts=None, competitive_facts=None,
    )


# -- 5-seed deterministic experiment generation ------------------------------


def test_five_seed_experiment_spec_deterministic_expansion():
    variant = VariantSpec(variant_id="A_native_tool_like_bedrock", agent_runtime="native_tool_like", provider="bedrock", model="m")
    spec = ExperimentSpec(experiment_id="m5e3_1-bedrock-multiseed-stability-v1", variants=(variant,), seeds=(601, 602, 603, 604, 605))
    episodes = spec.expand()
    assert [e.seed for e in episodes] == [601, 602, 603, 604, 605]
    assert len(set(e.episode_id for e in episodes)) == 5
    # Re-expanding the identical spec must produce the identical sequence.
    assert [e.episode_id for e in episodes] == [e.episode_id for e in spec.expand()]


# -- technical-status vocabulary mapping -------------------------------------


def test_classify_maps_decision_limit_reached_to_m5e3_1_label():
    episode = _episode(status=EpisodeStatus.DECISION_LIMIT_REACHED)
    long_horizon = compute_long_horizon_stability([{"final_model_action_valid": True}] * 15, [])
    assert classify_episode_for_m5e3_1(episode, long_horizon) == DECISION_LIMIT_REACHED
    assert is_technically_clean(DECISION_LIMIT_REACHED) is True


def test_classify_maps_completed_with_mission_success_to_gameplay_complete():
    episode = _episode(status=EpisodeStatus.COMPLETED, mission_success=True)
    long_horizon = compute_long_horizon_stability([{"final_model_action_valid": True}], [])
    assert classify_episode_for_m5e3_1(episode, long_horizon) == GAMEPLAY_COMPLETE


def test_classify_maps_completed_without_mission_success_to_mission_deadline_reached():
    episode = _episode(status=EpisodeStatus.COMPLETED, mission_success=False)
    long_horizon = compute_long_horizon_stability([{"final_model_action_valid": True}], [])
    assert classify_episode_for_m5e3_1(episode, long_horizon) == MISSION_DEADLINE_REACHED


def test_classify_maps_majority_invalid_decisions_to_technical_failure_even_at_decision_limit():
    """The M5E.2 gap this milestone's spec explicitly calls out (section
    10): a DECISION_LIMIT_REACHED episode whose budget was actually
    consumed by fallback/invalid decisions must not read as ordinary
    gameplay."""
    episode = _episode(status=EpisodeStatus.DECISION_LIMIT_REACHED)
    long_horizon = compute_long_horizon_stability([{"final_model_action_valid": False}] * 10 + [{"final_model_action_valid": True}] * 2, [])
    status = classify_episode_for_m5e3_1(episode, long_horizon)
    assert status == TECHNICAL_FAILURE
    assert is_technically_clean(status) is False


def test_classify_maps_aborted_to_technical_failure():
    episode = _episode(status=EpisodeStatus.ABORTED, error_type="max_runtime_failures_exceeded")
    assert classify_episode_for_m5e3_1(episode, None) == TECHNICAL_FAILURE


# -- provider failure taxonomy / auth-expiry detection -----------------------


def test_episode_has_auth_expiry_true_when_provider_auth_error_present():
    calls = [{"provider_error_type": None}, {"provider_error_type": PROVIDER_AUTH_ERROR}]
    assert episode_has_auth_expiry(calls) is True


def test_episode_has_auth_expiry_false_for_unrelated_failure_category():
    calls = [{"provider_error_type": PROVIDER_SERVICE_ERROR}]
    assert episode_has_auth_expiry(calls) is False


def test_count_provider_failures_by_category_never_merges_categories():
    calls = [
        {"provider_error_type": PROVIDER_SERVICE_ERROR}, {"provider_error_type": PROVIDER_SERVICE_ERROR},
        {"provider_error_type": PROVIDER_AUTH_ERROR}, {"provider_error_type": None},
    ]
    counts = count_provider_failures_by_category(calls)
    assert counts == {PROVIDER_SERVICE_ERROR: 2, PROVIDER_AUTH_ERROR: 1}


def test_failure_signature_none_for_a_non_failed_episode():
    assert failure_signature(_episode(), [], DECISION_LIMIT_REACHED) is None


def test_failure_signature_uses_dominant_provider_error_category():
    calls = [{"provider_error_type": PROVIDER_SERVICE_ERROR}, {"provider_error_type": PROVIDER_SERVICE_ERROR}]
    assert failure_signature(_episode(), calls, TECHNICAL_FAILURE) == PROVIDER_SERVICE_ERROR


def test_failure_signature_falls_back_to_harness_error_type_when_no_provider_calls():
    ep = _episode(error_type="max_runtime_failures_exceeded")
    assert failure_signature(ep, [], TECHNICAL_FAILURE) == "max_runtime_failures_exceeded"


# -- mid-run stop rules -------------------------------------------------------


def _make_specs(seeds):
    variant = VariantSpec(variant_id="A_native_tool_like_bedrock", agent_runtime="native_tool_like", provider="bedrock", model="m")
    spec = ExperimentSpec(experiment_id="m5e3_1-test", variants=(variant,), seeds=tuple(seeds))
    return spec.expand()


def test_mid_run_auth_error_stops_remaining_seeds_and_preserves_completed_ones():
    specs = _make_specs([601, 602, 603, 604, 605])
    calls_by_episode = {
        specs[0].episode_id: [],
        specs[1].episode_id: [{"provider_error_type": PROVIDER_AUTH_ERROR}],
    }

    def run_one(es):
        return _episode(seed=es.seed)

    outcome = run_seeds_with_stop_rules(
        specs, run_one_episode=run_one,
        load_provider_calls=lambda episode_id: calls_by_episode.get(episode_id, []),
        classify=lambda ep, calls: DECISION_LIMIT_REACHED,
    )
    assert outcome.seeds_run == [601, 602]
    assert outcome.seeds_not_run == [603, 604, 605]
    assert outcome.stop_reason == "auth_token_expired_mid_run"
    assert len(outcome.episodes) == 2


def test_repeated_failure_signature_stops_after_second_occurrence_not_first():
    specs = _make_specs([601, 602, 603, 604, 605])

    def run_one(es):
        return _episode(seed=es.seed)

    def classify(ep, calls):
        return TECHNICAL_FAILURE if ep.seed in (601, 603) else DECISION_LIMIT_REACHED

    calls_by_episode = {
        specs[0].episode_id: [{"provider_error_type": PROVIDER_SERVICE_ERROR}],
        specs[2].episode_id: [{"provider_error_type": PROVIDER_SERVICE_ERROR}],
    }

    outcome = run_seeds_with_stop_rules(
        specs, run_one_episode=run_one,
        load_provider_calls=lambda episode_id: calls_by_episode.get(episode_id, []),
        classify=classify,
    )
    # seed 601 fails once (not yet 2 occurrences - continues), seed 602
    # clean, seed 603 fails a second time with the SAME signature -> stop
    # before 604/605.
    assert outcome.seeds_run == [601, 602, 603]
    assert outcome.seeds_not_run == [604, 605]
    assert outcome.stop_reason == "repeated_failure_signature"


def test_single_isolated_failure_does_not_stop_remaining_seeds():
    specs = _make_specs([601, 602, 603])

    def run_one(es):
        return _episode(seed=es.seed)

    def classify(ep, calls):
        return TECHNICAL_FAILURE if ep.seed == 602 else DECISION_LIMIT_REACHED

    calls_by_episode = {specs[1].episode_id: [{"provider_error_type": PROVIDER_SERVICE_ERROR}]}

    outcome = run_seeds_with_stop_rules(
        specs, run_one_episode=run_one,
        load_provider_calls=lambda episode_id: calls_by_episode.get(episode_id, []),
        classify=classify,
    )
    assert outcome.seeds_run == [601, 602, 603]
    assert outcome.seeds_not_run == []
    assert outcome.stop_reason is None


def test_no_stop_rule_triggered_runs_every_seed():
    specs = _make_specs([601, 602, 603, 604, 605])
    outcome = run_seeds_with_stop_rules(
        specs, run_one_episode=lambda es: _episode(seed=es.seed),
        load_provider_calls=lambda episode_id: [],
        classify=lambda ep, calls: DECISION_LIMIT_REACHED,
    )
    assert outcome.seeds_run == [601, 602, 603, 604, 605]
    assert outcome.seeds_not_run == []
    assert outcome.stop_reason is None


# -- per-seed / aggregate reporting -------------------------------------------


def test_build_per_seed_row_and_aggregate_table():
    ep1 = _episode(seed=601, mission_success=True)
    ep1.metrics = {"latency_ms": {"median": 100.0, "max": 200.0}, "repair_attempts": 1, "fallbacks": 0, "input_tokens_total": 500, "output_tokens_total": 50}
    lh1 = compute_long_horizon_stability([{"final_model_action_valid": True, "recovery_outcome": "FIRST_PASS_VALID"}] * 5, [])

    ep2 = _episode(seed=602, mission_success=False)
    ep2.metrics = {"latency_ms": {"median": 300.0, "max": 400.0}, "repair_attempts": 0, "fallbacks": 1, "input_tokens_total": 700, "output_tokens_total": 80}
    lh2 = compute_long_horizon_stability(
        [{"final_model_action_valid": True, "recovery_outcome": "FIRST_PASS_VALID"}] * 4
        + [{"final_model_action_valid": False, "recovery_outcome": "FALLBACK"}],
        [],
    )

    row1 = build_per_seed_row(ep1, GAMEPLAY_COMPLETE, lh1, [])
    row2 = build_per_seed_row(ep2, DECISION_LIMIT_REACHED, lh2, [{"provider_error_type": PROVIDER_SERVICE_ERROR}])

    assert row1["seed"] == 601 and row1["decisions"] == 5 and row1["first_pass_valid"] == 5
    assert row2["decisions"] == 5 and row2["first_pass_valid"] == 4 and row2["provider_failures"] == 1

    agg = build_aggregate_table([row1, row2])
    assert agg["episodes_requested"] == 2
    assert agg["episodes_technically_clean"] == 2
    assert agg["total_decisions"] == 10
    assert agg["provider_failures"] == 1
    assert agg["total_input_tokens"] == 1200
    assert agg["total_output_tokens"] == 130


def test_aggregate_table_counts_technical_failures_separately():
    row_clean = {"technical_status": GAMEPLAY_COMPLETE, "decisions": 5, "provider_failures": 0, "runtime_failures": 0, "first_pass_valid": 5, "repairs": 0, "fallbacks": 0, "median_latency_ms": 100.0, "max_latency_ms": 200.0, "input_tokens": 10, "output_tokens": 5}
    row_failed = {"technical_status": TECHNICAL_FAILURE, "decisions": 2, "provider_failures": 2, "runtime_failures": 1, "first_pass_valid": 0, "repairs": 0, "fallbacks": 0, "median_latency_ms": None, "max_latency_ms": None, "input_tokens": 0, "output_tokens": 0}
    agg = build_aggregate_table([row_clean, row_failed])
    assert agg["episodes_technically_clean"] == 1
    assert agg["technical_failure_episodes"] == 1


def test_cumulative_bedrock_decisions_is_plain_arithmetic_over_documented_history():
    result = cumulative_bedrock_decisions(prior_clean_decisions=95, prior_total_decisions=95, new_clean_decisions=75, new_total_decisions=75)
    assert result["cumulative_clean"] == 170
    assert result["cumulative_total"] == 170


# -- sleep-log contamination --------------------------------------------------

_SAMPLE_LOG = """\
2026-09-20 15:00:00 +0200 Assertions          \tirrelevant line
2026-09-20 15:58:40 +0200 Sleep               \tEntering Sleep state due to 'Maintenance Sleep' 948 secs
2026-09-20 16:14:28 +0200 Wake                \tDarkWake to FullWake from Deep Idle 1010 secs
2026-09-20 20:00:00 +0200 DarkWake            \tDarkWake from Deep Idle
"""


def test_detect_sleep_contamination_true_when_event_falls_inside_window():
    start = datetime(2026, 9, 20, 15, 50, 0, tzinfo=timezone(timedelta(hours=2)))
    end = datetime(2026, 9, 20, 16, 20, 0, tzinfo=timezone(timedelta(hours=2)))
    report = detect_sleep_contamination(_SAMPLE_LOG, start, end)
    assert report.contaminated is True
    assert len(report.matching_events) == 2  # the Sleep at 15:58:40 and the Wake at 16:14:28


def test_detect_sleep_contamination_false_when_window_has_no_overlap():
    start = datetime(2026, 9, 20, 10, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    end = datetime(2026, 9, 20, 11, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    report = detect_sleep_contamination(_SAMPLE_LOG, start, end)
    assert report.contaminated is False
    assert report.matching_events == []


def test_detect_sleep_contamination_ignores_non_sleep_domain_lines():
    log = "2026-09-20 15:58:40 +0200 Assertions          \tPID 1 something unrelated\n"
    start = datetime(2026, 9, 20, 15, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    end = datetime(2026, 9, 20, 17, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    assert detect_sleep_contamination(log, start, end).contaminated is False


# -- rerun preservation (spec sections 20-21, 38) -----------------------------


def test_resolve_output_dir_returns_base_dir_when_no_prior_run_exists(tmp_path):
    base_dir = str(tmp_path / "m5e3_1-bedrock-multiseed-stability-v1")
    assert resolve_output_dir(base_dir) == base_dir


def test_resolve_output_dir_returns_a_distinct_rerun_dir_when_experiment_json_already_exists(tmp_path):
    base_dir = tmp_path / "m5e3_1-bedrock-multiseed-stability-v1"
    base_dir.mkdir()
    (base_dir / "experiment.json").write_text("{}")
    resolved = resolve_output_dir(str(base_dir), now=datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc))
    assert resolved != str(base_dir)
    assert resolved.startswith(str(base_dir) + "-rerun-")
    # The original directory/file must be untouched by merely resolving a path.
    assert (base_dir / "experiment.json").exists()
