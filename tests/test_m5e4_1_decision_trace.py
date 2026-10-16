"""M5E.4.1: tests for the per-decision trace / mission-progress building
blocks in `src/evaluation/decision_trace.py` - real `GameState` fixtures
where genuine engine state is needed, synthetic telemetry dicts
otherwise, never a real Game/LLM/Bedrock call."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src import constants
from src.evaluation.decision_trace import (
    BUDGET_CLEARLY_TOO_SHORT,
    BUDGET_SUFFICIENT_GAME_FRICTION,
    BUDGET_SUFFICIENT_INEFFICIENT_PLANNING,
    annotate_repeats,
    build_checkpoints,
    build_decision_trace_records,
    build_progress_snapshot,
    classify_action_effectiveness,
    classify_budget_conclusion,
    classify_progress_delta,
    decision_budget_breakdown,
    decision_to_progress_metrics,
    render_markdown_trace,
    stall_analysis,
)
from src.evaluation.synthetic_stability import _build_base_state


# -- progress snapshot ---------------------------------------------------


def test_build_progress_snapshot_has_required_fields():
    state = _build_base_state(seed=801)
    snapshot = build_progress_snapshot(state, constants.WHITE_SPY_ID)
    for key in ("unique_rooms_visited", "mission_items_acquired", "requirements_known", "exit_unlocked", "mission_success", "current_room"):
        assert key in snapshot


def test_build_progress_snapshot_exit_unlocked_reflects_door_state():
    state = _build_base_state(seed=801)
    snapshot_locked = build_progress_snapshot(state, constants.WHITE_SPY_ID)
    assert snapshot_locked["exit_unlocked"] is (state.door_states.get("exit_door") != "locked")
    state.door_states["exit_door"] = "unlocked"
    snapshot_unlocked = build_progress_snapshot(state, constants.WHITE_SPY_ID)
    assert snapshot_unlocked["exit_unlocked"] is True


# -- progress delta classification (spec section 14) ----------------------


def _snap(**overrides):
    base = {
        "unique_rooms_visited": 1, "knowledge_learned_count": 0, "requirements_known": [], "door_unlocked_count": 0,
        "mission_items_acquired": 0, "item_found_count": 0, "mission_progress_events": 0, "mission_success": False,
        "opponent_encounters": 0, "traps_placed": 0, "traps_triggered": 0, "traps_disarmed": 0, "enemy_traps_detected": 0,
    }
    base.update(overrides)
    return base


def test_classify_progress_delta_no_progress_when_nothing_changes():
    assert classify_progress_delta(_snap(), _snap()) == ["NO_PROGRESS"]


def test_classify_progress_delta_navigation():
    assert "NAVIGATION_PROGRESS" in classify_progress_delta(_snap(), _snap(unique_rooms_visited=2))


def test_classify_progress_delta_discovery():
    assert "DISCOVERY_PROGRESS" in classify_progress_delta(_snap(), _snap(requirements_known=["door_a"]))


def test_classify_progress_delta_requirement_via_item_count_not_list_comparison():
    """Regression: `mission_items_acquired` is an int count, not a list -
    must compare as ints, not silently no-op via a `len()` on a non-list."""
    assert "REQUIREMENT_PROGRESS" in classify_progress_delta(_snap(), _snap(mission_items_acquired=1))


def test_classify_progress_delta_item_progress():
    assert "ITEM_PROGRESS" in classify_progress_delta(_snap(), _snap(item_found_count=1))


def test_classify_progress_delta_mission_progress():
    assert "MISSION_PROGRESS" in classify_progress_delta(_snap(), _snap(mission_progress_events=1))


def test_classify_progress_delta_opponent_and_trap():
    deltas = classify_progress_delta(_snap(), _snap(opponent_encounters=1, traps_triggered=1))
    assert "OPPONENT_PROGRESS" in deltas
    assert "TRAP_PROGRESS" in deltas


def test_classify_progress_delta_can_report_multiple_categories_at_once():
    deltas = classify_progress_delta(_snap(), _snap(unique_rooms_visited=2, item_found_count=1))
    assert "NAVIGATION_PROGRESS" in deltas and "ITEM_PROGRESS" in deltas


# -- repeat detection (spec section 16) -----------------------------------


def test_annotate_repeats_counts_within_window():
    records = [
        {"proposed_action": "move_to", "proposed_target": "door_a"},
        {"proposed_action": "inspect", "proposed_target": "desk_01"},
        {"proposed_action": "move_to", "proposed_target": "door_a"},
        {"proposed_action": "move_to", "proposed_target": "door_a"},
    ]
    annotate_repeats(records, window=5)
    assert records[0]["repeat_count"] == 0
    assert records[2]["repeat_count"] == 1
    assert records[3]["repeat_count"] == 2


def test_annotate_repeats_respects_window_boundary():
    records = [{"proposed_action": "wait", "proposed_target": None} for _ in range(3)]
    annotate_repeats(records, window=1)
    assert records[0]["repeat_count"] == 0
    assert records[1]["repeat_count"] == 1  # only looks 1 back
    assert records[2]["repeat_count"] == 1


# -- action effectiveness (spec section 15) --------------------------------


def test_classify_action_effectiveness_invalid_when_fallback():
    record = {"action_source": "fallback", "final_action_valid": False, "repeat_count": 0, "engine_action_status": None}
    assert classify_action_effectiveness(record) == "invalid"


def test_classify_action_effectiveness_ineffective_when_repeated():
    record = {"action_source": "llm", "final_action_valid": True, "repeat_count": 2, "engine_action_status": "completed"}
    assert classify_action_effectiveness(record) == "ineffective"


def test_classify_action_effectiveness_ineffective_when_engine_failed():
    record = {"action_source": "llm", "final_action_valid": True, "repeat_count": 0, "engine_action_status": "failed"}
    assert classify_action_effectiveness(record) == "ineffective"


def test_classify_action_effectiveness_effective_when_valid_novel_and_completed():
    record = {"action_source": "llm", "final_action_valid": True, "repeat_count": 0, "engine_action_status": "completed"}
    assert classify_action_effectiveness(record) == "effective"


def test_classify_action_effectiveness_effective_even_with_no_engine_status_eg_wait():
    record = {"action_source": "llm", "final_action_valid": True, "repeat_count": 0, "engine_action_status": None}
    assert classify_action_effectiveness(record) == "effective"


# -- decision trace record joining (spec section 9) ------------------------


def _decision_event(n, decision_id, reason="explore", action="move_to", target="door_a", valid=True, repair=False):
    d = {
        "decision_number": n, "decision_id": decision_id, "game_time": float(n) * 0.5,
        "raw_response": {"action": action, "target_id": target, "reason": reason},
        "recovery_outcome": "FIRST_PASS_VALID" if valid else ("REPAIRED_VALID" if repair else "REPAIR_FAILED_FALLBACK"),
        "initial_error_type": None if valid else "illegal_target",
        "repair_attempted": repair,
        "repair_raw_response": {"action": "wait", "target_id": None, "reason": "repair"} if repair else {},
        "repair_valid": True if repair else None,
        "executed_action": {"action": action if valid else "wait", "target_id": target if valid else None},
        "final_model_action_valid": True,
        "action_source": "llm",
        "latency_ms": 1000.0, "input_tokens": 100, "output_tokens": 10,
    }
    return d


def test_build_decision_trace_records_joins_all_three_sources():
    decisions = [_decision_event(1, "d1"), _decision_event(2, "d2", action="inspect", target="desk_01")]
    lifecycle = [
        {"decision_id": "d1", "action_status": "completed", "action_terminal_reason": "target_reached"},
        {"decision_id": "d2", "action_status": "completed", "action_terminal_reason": "resolved"},
    ]
    snapshots = {0: _snap(), 1: _snap(unique_rooms_visited=2), 2: _snap(unique_rooms_visited=2, item_found_count=1)}
    records = build_decision_trace_records(decisions, lifecycle, snapshots, "ep1", 801, "native_tool_like")

    assert len(records) == 2
    assert records[0]["decision_number"] == 1
    assert records[0]["agent_stated_reason"] == "explore"
    assert records[0]["engine_action_status"] == "completed"
    assert "NAVIGATION_PROGRESS" in records[0]["progress_delta"]
    assert "ITEM_PROGRESS" in records[1]["progress_delta"]
    assert records[0]["action_effectiveness"] in ("effective", "ineffective", "invalid")


def test_build_decision_trace_records_never_overwrites_initial_with_repair():
    decisions = [_decision_event(1, "d1", action="move_to", target="locked_door", valid=False, repair=True)]
    records = build_decision_trace_records(decisions, [], {0: _snap(), 1: _snap()}, "ep1", 801, "strands")
    assert records[0]["proposed_action"] == "move_to"
    assert records[0]["proposed_target"] == "locked_door"
    assert records[0]["repair_action"] == "wait"
    assert records[0]["repair_reason"] == "repair"


def test_build_decision_trace_records_redacts_fake_secret_in_reason():
    decisions = [_decision_event(1, "d1", reason="my token is fake-bearer-token-do-not-use-abc123")]
    import src.evaluation.decision_trace as dt

    original = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "fake-bearer-token-do-not-use-abc123"
    try:
        records = build_decision_trace_records(decisions, [], {0: _snap(), 1: _snap()}, "ep1", 801, "native_tool_like")
    finally:
        if original is None:
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        else:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = original
    assert "fake-bearer-token-do-not-use-abc123" not in records[0]["agent_stated_reason"]


# -- checkpoints, budget breakdown, decision-to-progress --------------------


def test_build_checkpoints_produces_one_row_per_point_reached():
    records = [
        {"decision_number": i, "mission_progress_after": _snap(unique_rooms_visited=1 + i // 10, mission_items_acquired=i // 20),
         "action_effectiveness": "effective" if i % 3 else "ineffective", "progress_delta": ["NAVIGATION_PROGRESS"]}
        for i in range(1, 41)
    ]
    checkpoints = build_checkpoints(records)
    assert [c["decision"] for c in checkpoints] == [10, 20, 30, 40]
    assert checkpoints[0]["rooms_visited"] == 1 + 10 // 10  # value at decision 10, matching the fixture's own formula
    assert checkpoints[-1]["rooms_visited"] == 1 + 40 // 10


def test_build_checkpoints_skips_points_never_reached():
    records = [{"decision_number": i, "mission_progress_after": _snap(), "action_effectiveness": "effective", "progress_delta": ["NO_PROGRESS"]} for i in range(1, 6)]
    checkpoints = build_checkpoints(records)
    assert [c["decision"] for c in checkpoints] == [10, 20, 30, 40]  # clamps to latest available, still reports all points
    assert checkpoints[0]["decision"] == 10


def test_decision_budget_breakdown_categorizes_navigation_and_interaction():
    records = [
        {"proposed_action": "move_to", "progress_delta": ["NAVIGATION_PROGRESS"], "repeat_count": 0, "first_pass_valid": True},
        {"proposed_action": "inspect", "progress_delta": ["NO_PROGRESS"], "repeat_count": 0, "first_pass_valid": True},
        {"proposed_action": "search", "progress_delta": ["NO_PROGRESS"], "repeat_count": 1, "first_pass_valid": True},
        {"proposed_action": "wait", "progress_delta": ["NO_PROGRESS"], "repeat_count": 0, "first_pass_valid": False},
    ]
    breakdown = decision_budget_breakdown(records)
    assert breakdown["total_decisions"] == 4
    assert breakdown["navigation"] == 1
    assert breakdown["inspection_search_interaction"] == 2
    assert breakdown["invalid_repaired_fallback_actions"] == 1
    assert breakdown["repeated_or_no_progress_actions"] == 3  # inspect (no-progress) + search (repeat) + wait (no-progress)


def test_decision_to_progress_metrics_reports_first_occurrence_and_null_when_absent():
    records = [
        {"decision_number": 1, "progress_delta": ["NO_PROGRESS"], "action_effectiveness": "ineffective", "mission_progress_after": _snap()},
        {"decision_number": 2, "progress_delta": ["NAVIGATION_PROGRESS"], "action_effectiveness": "effective", "mission_progress_after": _snap()},
        {"decision_number": 3, "progress_delta": ["ITEM_PROGRESS"], "action_effectiveness": "effective", "mission_progress_after": _snap()},
    ]
    metrics = decision_to_progress_metrics(records)
    assert metrics["decision_to_first_room_exit"] == 2
    assert metrics["decision_to_first_mission_item"] == 3
    assert metrics["decision_to_mission_completion"] is None


# -- budget conclusion (spec section 46) ------------------------------------


def _record(n, delta, effectiveness="effective"):
    return {"decision_number": n, "progress_delta": delta, "action_effectiveness": effectiveness, "repeat_count": 0, "first_pass_valid": True}


def test_classify_budget_conclusion_clearly_too_short_when_low_repetition_and_still_progressing():
    records = [_record(i, ["NAVIGATION_PROGRESS"]) for i in range(1, 41)]
    breakdown = decision_budget_breakdown([{"proposed_action": "move_to", **r} for r in records])
    assert classify_budget_conclusion(records, breakdown) == BUDGET_CLEARLY_TOO_SHORT


def test_classify_budget_conclusion_sufficient_but_inefficient_when_heavy_repetition_dominates():
    records = [_record(i, ["NO_PROGRESS"], "ineffective") for i in range(1, 41)]
    for r in records:
        r["repeat_count"] = 1
    breakdown = decision_budget_breakdown([{"proposed_action": "move_to", **r} for r in records])
    assert classify_budget_conclusion(records, breakdown) == BUDGET_SUFFICIENT_INEFFICIENT_PLANNING


def test_classify_budget_conclusion_game_friction_when_steady_low_repetition_but_stalled_late():
    records = [_record(i, ["NAVIGATION_PROGRESS"] if i < 25 else ["NO_PROGRESS"]) for i in range(1, 41)]
    breakdown = decision_budget_breakdown([{"proposed_action": "move_to", **r} for r in records])
    assert classify_budget_conclusion(records, breakdown) == BUDGET_SUFFICIENT_GAME_FRICTION


def test_classify_budget_conclusion_never_uses_a_subjective_score():
    """The classification must be one of the five fixed labels, never a
    free-form judgment string."""
    from src.evaluation.decision_trace import BUDGET_MIXED_UNCLEAR, BUDGET_PROBABLY_TOO_SHORT

    valid_labels = {BUDGET_CLEARLY_TOO_SHORT, BUDGET_PROBABLY_TOO_SHORT, BUDGET_SUFFICIENT_INEFFICIENT_PLANNING, BUDGET_SUFFICIENT_GAME_FRICTION, BUDGET_MIXED_UNCLEAR}
    records = [_record(i, ["NO_PROGRESS"]) for i in range(1, 5)]
    breakdown = decision_budget_breakdown([{"proposed_action": "wait", **r} for r in records])
    assert classify_budget_conclusion(records, breakdown) in valid_labels


# -- stall analysis (spec section 45) ----------------------------------------


def test_stall_analysis_reports_no_stall_when_steady_progress():
    records = [{"decision_number": i, "progress_delta": ["NAVIGATION_PROGRESS"], "action_effectiveness": "effective", "repeat_signature": f"move_to:room{i}"} for i in range(1, 6)]
    assert "no obvious stall" in stall_analysis(records)


def test_stall_analysis_identifies_longest_repeated_run():
    records = [{"decision_number": 1, "progress_delta": ["NAVIGATION_PROGRESS"], "action_effectiveness": "effective", "repeat_signature": "move_to:door_a"}]
    for i in range(2, 8):
        records.append({"decision_number": i, "progress_delta": ["NO_PROGRESS"], "action_effectiveness": "ineffective", "repeat_signature": "move_to:locked_door"})
    result = stall_analysis(records)
    assert "D2-D7" in result
    assert "move_to:locked_door" in result


# -- markdown trace (spec section 10, 38) -----------------------------------


def test_render_markdown_trace_is_readable_and_never_leaks_a_secret():
    decisions = [_decision_event(1, "d1", reason="need to find the archive key")]
    records = build_decision_trace_records(decisions, [], {0: _snap(), 1: _snap(unique_rooms_visited=2)}, "ep1", 801, "native_tool_like")
    md = render_markdown_trace(records, "ep1", 801, "native_tool_like")
    assert "D01" in md
    assert "need to find the archive key" in md
    assert "AWS_BEARER_TOKEN_BEDROCK" not in md
    assert "bedrock-api-key" not in md
