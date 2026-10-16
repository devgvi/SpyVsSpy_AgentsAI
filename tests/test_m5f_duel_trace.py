"""M5F: multi-agent duel trace building blocks
(`src/evaluation/duel_trace.py`) - synthetic telemetry only, no real
Game/Bedrock calls."""

from __future__ import annotations

from src import constants
from src.evaluation.duel_trace import (
    annotate_opponent_visibility,
    build_actor_trace_records,
    contested_objective_events,
    extract_opponent_encounters,
    merge_duel_trace,
    mutual_same_room_encounters,
    reaction_to_sighting,
    render_duel_markdown,
)

BLACK = constants.BLACK_SPY_ID
WHITE = constants.WHITE_SPY_ID


def _decision(n, decision_id, actor_id, game_time, action="wait", target=None, opponent_visible=False, room="corridor", reason="thinking"):
    return {
        "decision_number": n, "decision_id": decision_id, "actor_id": actor_id, "game_time": game_time,
        "raw_response": {"action": action, "target_id": target, "reason": reason},
        "recovery_outcome": "FIRST_PASS_VALID", "initial_error_type": None, "repair_attempted": False,
        "repair_raw_response": {}, "repair_valid": None,
        "executed_action": {"action": action, "target_id": target},
        "final_model_action_valid": True, "action_source": "llm",
        "latency_ms": 500.0, "input_tokens": 50, "output_tokens": 5,
        "observation": {"opponent": {"visible": opponent_visible}, "self": {"room": room}},
    }


def _snap(current_room="corridor"):
    return {
        "unique_rooms_visited": 1, "current_room": current_room, "mission_success": False,
        "requirements_known": [], "door_unlocked_count": 0, "mission_items_acquired": 0,
        "item_found_count": 0, "mission_progress_events": 0, "opponent_encounters": 0,
        "traps_placed": 0, "traps_triggered": 0, "traps_disarmed": 0, "enemy_traps_detected": 0,
        "knowledge_learned_count": 0,
    }


# -- build_actor_trace_records / merge_duel_trace -----------------------------


def test_build_actor_trace_records_filters_by_actor_id_and_tags_them():
    decisions = [_decision(1, "b1", BLACK, 1.0), _decision(1, "w1", WHITE, 1.5)]
    snapshots = {0: _snap(), 1: _snap()}
    black_records = build_actor_trace_records(decisions, [], snapshots, "ep1", 900, "native_tool_like", BLACK)
    assert len(black_records) == 1
    assert black_records[0]["actor_id"] == BLACK
    assert black_records[0]["decision_id"] == "b1"


def test_merge_duel_trace_orders_by_game_time_across_actors():
    black_records = build_actor_trace_records([_decision(1, "b1", BLACK, 5.0)], [], {0: _snap(), 1: _snap()}, "ep1", 900, "native_tool_like", BLACK)
    white_records = build_actor_trace_records([_decision(1, "w1", WHITE, 2.0)], [], {0: _snap(), 1: _snap()}, "ep1", 900, "native_tool_like", WHITE)
    merged = merge_duel_trace(black_records, white_records)
    assert [r["decision_id"] for r in merged] == ["w1", "b1"]  # white's earlier game_time comes first


def test_render_duel_markdown_labels_each_actor_and_includes_rationale():
    records = build_actor_trace_records([_decision(1, "b1", BLACK, 1.0, reason="chasing white")], [], {0: _snap(), 1: _snap()}, "ep1", 900, "native_tool_like", BLACK)
    md = render_duel_markdown(records, "ep1", 900)
    assert "BLACK" in md
    assert "chasing white" in md


# -- opponent visibility annotation / encounters (spec sections 41-43) -------


def test_annotate_opponent_visibility_reads_the_real_stored_observation_field():
    decisions = [
        _decision(1, "b1", BLACK, 1.0, opponent_visible=False),
        _decision(2, "b2", BLACK, 2.0, opponent_visible=True),
    ]
    records = build_actor_trace_records(decisions, [], {0: _snap(), 1: _snap(), 2: _snap()}, "ep1", 900, "native_tool_like", BLACK)
    decisions_by_id = {d["decision_id"]: d for d in decisions}
    annotate_opponent_visibility(records, decisions_by_id)
    assert records[0]["opponent_visible_before"] is None
    assert records[0]["opponent_visible_after"] is False
    assert records[1]["opponent_visible_before"] is False
    assert records[1]["opponent_visible_after"] is True


def test_extract_opponent_encounters_detects_both_transition_directions():
    decisions = [
        _decision(1, "b1", BLACK, 1.0, opponent_visible=False),
        _decision(2, "b2", BLACK, 2.0, opponent_visible=True),
        _decision(3, "b3", BLACK, 3.0, opponent_visible=False),
    ]
    records = build_actor_trace_records(decisions, [], {0: _snap(), 1: _snap(), 2: _snap(), 3: _snap()}, "ep1", 900, "native_tool_like", BLACK)
    annotate_opponent_visibility(records, {d["decision_id"]: d for d in decisions})
    encounters = extract_opponent_encounters(records)
    transitions = [e["transition"] for e in encounters]
    assert "opponent_not_visible_to_visible" in transitions
    assert "opponent_visible_to_not_visible" in transitions


def test_reaction_to_sighting_reports_factual_changes_only():
    decisions = [
        _decision(1, "b1", BLACK, 1.0, action="move", target="desk_01"),
        _decision(2, "b2", BLACK, 2.0, action="move", target="cabinet_02"),
        _decision(3, "b3", BLACK, 3.0, action="place_trap", target="cabinet_02"),
    ]
    records = build_actor_trace_records(decisions, [], {0: _snap(), 1: _snap(), 2: _snap(), 3: _snap()}, "ep1", 900, "native_tool_like", BLACK)
    reactions = reaction_to_sighting(records, BLACK, sighting_decision_number=1, window=3)
    assert len(reactions) == 2
    assert reactions[0]["target_changed"] is True
    assert reactions[1]["trap_placed"] is True


def test_mutual_same_room_encounters_requires_both_visible_and_same_room_and_close_time():
    black_decisions = [_decision(1, "b1", BLACK, 10.0, opponent_visible=True, room="archives")]
    white_decisions = [_decision(1, "w1", WHITE, 10.5, opponent_visible=True, room="archives")]
    black_records = build_actor_trace_records(black_decisions, [], {0: _snap(), 1: _snap("archives")}, "ep1", 900, "native_tool_like", BLACK)
    white_records = build_actor_trace_records(white_decisions, [], {0: _snap(), 1: _snap("archives")}, "ep1", 900, "native_tool_like", WHITE)
    annotate_opponent_visibility(black_records, {d["decision_id"]: d for d in black_decisions})
    annotate_opponent_visibility(white_records, {d["decision_id"]: d for d in white_decisions})
    for r in black_records:
        r["current_room"] = "archives"
    for r in white_records:
        r["current_room"] = "archives"
    assert mutual_same_room_encounters(black_records, white_records) == 1


def test_mutual_same_room_encounters_zero_when_rooms_differ():
    black_decisions = [_decision(1, "b1", BLACK, 10.0, opponent_visible=True, room="archives")]
    white_decisions = [_decision(1, "w1", WHITE, 10.0, opponent_visible=True, room="reception")]
    black_records = build_actor_trace_records(black_decisions, [], {0: _snap(), 1: _snap("archives")}, "ep1", 900, "native_tool_like", BLACK)
    white_records = build_actor_trace_records(white_decisions, [], {0: _snap(), 1: _snap("reception")}, "ep1", 900, "native_tool_like", WHITE)
    annotate_opponent_visibility(black_records, {d["decision_id"]: d for d in black_decisions})
    annotate_opponent_visibility(white_records, {d["decision_id"]: d for d in white_decisions})
    assert mutual_same_room_encounters(black_records, white_records) == 0


# -- contested objectives (spec section 47) ------------------------------------


def test_contested_objective_events_detects_same_target_within_window():
    black_decisions = [_decision(1, "b1", BLACK, 10.0, action="move", target="archives_door")]
    white_decisions = [_decision(1, "w1", WHITE, 12.0, action="move", target="archives_door")]
    black_records = build_actor_trace_records(black_decisions, [], {0: _snap(), 1: _snap()}, "ep1", 900, "native_tool_like", BLACK)
    white_records = build_actor_trace_records(white_decisions, [], {0: _snap(), 1: _snap()}, "ep1", 900, "native_tool_like", WHITE)
    events = contested_objective_events(black_records, white_records, window_seconds=5.0)
    assert len(events) == 1
    assert events[0]["target"] == "archives_door"


def test_contested_objective_events_ignores_targets_outside_window():
    black_decisions = [_decision(1, "b1", BLACK, 10.0, action="move", target="archives_door")]
    white_decisions = [_decision(1, "w1", WHITE, 30.0, action="move", target="archives_door")]
    black_records = build_actor_trace_records(black_decisions, [], {0: _snap(), 1: _snap()}, "ep1", 900, "native_tool_like", BLACK)
    white_records = build_actor_trace_records(white_decisions, [], {0: _snap(), 1: _snap()}, "ep1", 900, "native_tool_like", WHITE)
    events = contested_objective_events(black_records, white_records, window_seconds=5.0)
    assert events == []
