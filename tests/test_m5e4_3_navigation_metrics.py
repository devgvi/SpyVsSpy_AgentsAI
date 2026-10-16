"""M5E.4.3: navigation-completion trace annotation
(`annotate_navigation_completion`/`navigation_completion_metrics` in
`src/evaluation/decision_trace.py`) - spec sections 27-30. Synthetic
records only, no real Game/Bedrock calls."""

from __future__ import annotations

from src.evaluation.decision_trace import (
    DOOR_NO_CHANGE,
    DOOR_TRAVERSED,
    annotate_door_sequences,
    annotate_navigation_completion,
    build_decision_trace_records,
    navigation_completion_metrics,
)


def _progress(current_room="corridor", door_state="open", connects=("corridor", "archives")):
    return {
        "current_room": current_room,
        "door_states": {"archives_door": door_state},
        "door_connects": {"archives_door": list(connects)},
    }


def _record(n, action, target, before, after, engine_terminal_reason=None):
    return {
        "decision_number": n, "final_action": action, "final_target": target,
        "proposed_action": action, "proposed_target": target,
        "mission_progress_before": before, "mission_progress_after": after,
        "engine_terminal_reason": engine_terminal_reason, "repeat_count": 0,
    }


# -- annotate_navigation_completion -------------------------------------------


def test_semantic_and_geometric_both_true_for_a_real_crossing():
    before = _progress(current_room="corridor")
    after = _progress(current_room="archives")
    records = [_record(1, "move", "archives_door", before, after, engine_terminal_reason="target_reached")]
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    assert records[0]["door_sequence_state"] == DOOR_TRAVERSED
    assert records[0]["geometric_target_reached"] is True
    assert records[0]["semantic_target_reached"] is True


def test_geometric_true_semantic_false_for_the_target_defect_itself():
    """The exact M5E.4.2 defect, now with an explicit engine signal
    (`doorway_crossing_incomplete`) instead of a silent false
    `target_reached`."""
    before = _progress(current_room="corridor")
    after = _progress(current_room="corridor")  # never actually crossed
    records = [_record(1, "move", "archives_door", before, after, engine_terminal_reason="doorway_crossing_incomplete")]
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    assert records[0]["door_sequence_state"] == DOOR_NO_CHANGE
    assert records[0]["geometric_target_reached"] is True
    assert records[0]["semantic_target_reached"] is False


def test_path_unreachable_is_neither_geometric_nor_semantic():
    before = _progress()
    after = _progress()
    records = [_record(1, "move", "archives_door", before, after, engine_terminal_reason="path_unreachable")]
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    assert records[0]["geometric_target_reached"] is False
    assert records[0]["semantic_target_reached"] is False


def test_non_door_move_semantic_equals_geometric():
    before = _progress()
    after = _progress()
    records = [_record(1, "move", "desk_01", before, after, engine_terminal_reason="target_reached")]
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    assert records[0]["door_sequence_state"] is None
    assert records[0]["geometric_target_reached"] is True
    assert records[0]["semantic_target_reached"] is True


def test_no_lifecycle_correlation_leaves_both_fields_none():
    records = [_record(1, "move", "archives_door", _progress(), _progress(), engine_terminal_reason=None)]
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    assert records[0]["geometric_target_reached"] is None
    assert records[0]["semantic_target_reached"] is None


# -- navigation_completion_metrics --------------------------------------------


def test_navigation_completion_metrics_crossing_success_rate():
    r1 = _record(1, "move", "archives_door", _progress(current_room="corridor"), _progress(current_room="archives"), "target_reached")
    r2 = _record(2, "move", "archives_door", _progress(current_room="corridor"), _progress(current_room="corridor"), "doorway_crossing_incomplete")
    records = [r1, r2]
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    metrics = navigation_completion_metrics(records)
    assert metrics["doorway_crossing_attempts"] == 2
    assert metrics["successful_room_crossings"] == 1
    assert metrics["failed_room_crossings"] == 1
    assert metrics["crossing_success_rate"] == 0.5
    assert metrics["geometric_reached_but_semantic_not_reached"] == 1


def test_navigation_completion_metrics_zero_attempts_is_null_rate_not_zero():
    records = [_record(1, "wait", None, _progress(), _progress(), None)]
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    metrics = navigation_completion_metrics(records)
    assert metrics["doorway_crossing_attempts"] == 0
    assert metrics["crossing_success_rate"] is None


def test_navigation_completion_metrics_counts_repeated_after_geometric_completion():
    r1 = _record(1, "move", "archives_door", _progress(current_room="corridor"), _progress(current_room="corridor"), "doorway_crossing_incomplete")
    r1["repeat_count"] = 0
    r2 = _record(2, "move", "archives_door", _progress(current_room="corridor"), _progress(current_room="corridor"), "doorway_crossing_incomplete")
    r2["repeat_count"] = 1
    records = [r1, r2]
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    metrics = navigation_completion_metrics(records)
    assert metrics["repeated_move_to_same_target_after_geometric_completion"] == 1


# -- end-to-end integration ---------------------------------------------------


def test_build_decision_trace_records_includes_navigation_completion_fields():
    decisions = [{
        "decision_number": 1, "decision_id": "d1", "game_time": 1.0,
        "raw_response": {"action": "move", "target_id": "archives_door", "reason": "cross the open door"},
        "recovery_outcome": "FIRST_PASS_VALID", "initial_error_type": None, "repair_attempted": False,
        "repair_raw_response": {}, "repair_valid": None,
        "executed_action": {"action": "move", "target_id": "archives_door"},
        "final_model_action_valid": True, "action_source": "llm",
        "latency_ms": 1000.0, "input_tokens": 100, "output_tokens": 10,
    }]
    lifecycle = [{"decision_id": "d1", "action_status": "completed", "action_terminal_reason": "target_reached"}]
    snapshots = {0: _progress(current_room="corridor"), 1: _progress(current_room="archives")}
    records = build_decision_trace_records(decisions, lifecycle, snapshots, "ep1", 801, "native_tool_like")
    assert records[0]["semantic_target_reached"] is True
    assert records[0]["movement_completion_reason"] == "target_reached"
