"""M5E.4.2: door-sequence trace fields and metrics
(`annotate_door_sequences`/`door_sequence_metrics`/`door_action_sequence_narrative`
in `src/evaluation/decision_trace.py`) - spec sections 24-25, 28-29, 36.
Synthetic records only, no real Game/Bedrock calls."""

from __future__ import annotations

from src.evaluation.decision_trace import (
    DOOR_NO_CHANGE,
    DOOR_OPENED,
    DOOR_RE_CLOSED,
    DOOR_REPEATED_OPEN_INTERACT,
    DOOR_TRAVERSED,
    annotate_door_sequences,
    build_decision_trace_records,
    door_action_sequence_narrative,
    door_sequence_metrics,
)


def _progress(current_room="corridor", door_state="closed", connects=("corridor", "archives")):
    return {
        "current_room": current_room,
        "door_states": {"archives_door": door_state},
        "door_connects": {"archives_door": list(connects)},
    }


def _record(n, action, target, before, after):
    return {
        "decision_number": n, "final_action": action, "final_target": target,
        "proposed_action": action, "proposed_target": target,
        "mission_progress_before": before, "mission_progress_after": after,
    }


# -- annotate_door_sequences (spec section 24) --------------------------------


def test_non_door_target_gets_all_none_door_fields():
    records = [_record(1, "inspect", "desk_01", _progress(), _progress())]
    annotate_door_sequences(records)
    assert records[0]["door_state_before"] is None
    assert records[0]["door_sequence_state"] is None


def test_interact_on_closed_door_classified_as_opened():
    before = _progress(door_state="closed")
    after = _progress(door_state="open")
    records = [_record(10, "interact", "archives_door", before, after)]
    annotate_door_sequences(records)
    assert records[0]["door_sequence_state"] == DOOR_OPENED
    assert records[0]["door_traversable"] is True
    assert records[0]["connected_room"] == ["corridor", "archives"]


def test_interact_on_open_door_reclassified_as_re_closed():
    before = _progress(door_state="open")
    after = _progress(door_state="closed")
    records = [_record(11, "interact", "archives_door", before, after)]
    annotate_door_sequences(records)
    assert records[0]["door_sequence_state"] == DOOR_RE_CLOSED


def test_interact_on_open_door_staying_open_is_repeated_open_interact():
    """The exact M5E.4.1-discovered defect pattern: INTERACT on an
    already-open door, engine 'resolves' it, door stays open."""
    before = _progress(door_state="open")
    after = _progress(door_state="open")
    records = [_record(12, "interact", "archives_door", before, after)]
    annotate_door_sequences(records)
    assert records[0]["door_sequence_state"] == DOOR_REPEATED_OPEN_INTERACT


def test_move_through_open_door_into_connected_room_is_traversed():
    before = _progress(current_room="corridor", door_state="open")
    after = _progress(current_room="archives", door_state="open")
    records = [_record(13, "move", "archives_door", before, after)]
    annotate_door_sequences(records)
    assert records[0]["door_sequence_state"] == DOOR_TRAVERSED


def test_move_via_unrelated_room_does_not_falsely_count_as_traversal():
    """Real bug caught by the M5E.4.2 pilot run: room_before must be one
    of THIS door's own two connected rooms - a move that changes room via
    a completely different route (e.g. ambassador_office -> corridor,
    while merely walking toward door_reception_corridor's position) must
    not be misclassified as traversing door_reception_corridor, whose
    connects pair is (reception, corridor), not (ambassador_office,
    corridor)."""
    before = {
        "current_room": "ambassador_office",
        "door_states": {"archives_door": "open"},
        "door_connects": {"archives_door": ["reception", "corridor"]},
    }
    after = {
        "current_room": "corridor",
        "door_states": {"archives_door": "open"},
        "door_connects": {"archives_door": ["reception", "corridor"]},
    }
    records = [_record(7, "move", "archives_door", before, after)]
    annotate_door_sequences(records)
    assert records[0]["door_sequence_state"] != DOOR_TRAVERSED
    assert records[0]["door_sequence_state"] == DOOR_NO_CHANGE


def test_move_to_door_without_crossing_is_no_change():
    """D21 from the real M5E.4.1 trace: MOVE_TO the door reaches it but
    current_room doesn't actually change yet - must not be misreported
    as a traversal."""
    before = _progress(current_room="corridor", door_state="open")
    after = _progress(current_room="corridor", door_state="open")
    records = [_record(14, "move", "archives_door", before, after)]
    annotate_door_sequences(records)
    assert records[0]["door_sequence_state"] == DOOR_NO_CHANGE


# -- door_sequence_metrics (spec sections 25, 28) -----------------------------


def test_door_sequence_metrics_computes_open_to_traverse_delay():
    records = [
        _record(10, "interact", "archives_door", _progress(door_state="closed"), _progress(door_state="open")),
        _record(11, "move", "archives_door", _progress(current_room="corridor", door_state="open"), _progress(current_room="archives", door_state="open")),
    ]
    annotate_door_sequences(records)
    metrics = door_sequence_metrics(records)
    assert metrics["doors_opened"] == 1
    assert metrics["doors_traversed"] == 1
    assert metrics["door_open_to_traverse_decisions"]["archives_door"] == 1
    assert metrics["open_to_traverse_success_rate"] == 1.0


def test_door_sequence_metrics_counts_repeated_interacts_before_traverse():
    records = [
        _record(10, "interact", "archives_door", _progress(door_state="closed"), _progress(door_state="open")),
        _record(11, "interact", "archives_door", _progress(door_state="open"), _progress(door_state="open")),
        _record(12, "interact", "archives_door", _progress(door_state="open"), _progress(door_state="open")),
        _record(13, "move", "archives_door", _progress(current_room="corridor", door_state="open"), _progress(current_room="archives", door_state="open")),
    ]
    annotate_door_sequences(records)
    metrics = door_sequence_metrics(records)
    assert metrics["repeated_open_door_interacts_before_traverse"]["archives_door"] == 2
    assert metrics["door_open_to_traverse_decisions"]["archives_door"] == 3


def test_door_sequence_metrics_null_delay_when_never_traversed():
    records = [_record(10, "interact", "archives_door", _progress(door_state="closed"), _progress(door_state="open"))]
    annotate_door_sequences(records)
    metrics = door_sequence_metrics(records)
    assert metrics["door_open_to_traverse_decisions"]["archives_door"] is None
    assert metrics["doors_traversed"] == 0
    assert metrics["open_to_traverse_success_rate"] == 0.0


def test_door_sequence_metrics_flags_closed_again_before_traverse():
    records = [
        _record(10, "interact", "archives_door", _progress(door_state="closed"), _progress(door_state="open")),
        _record(11, "interact", "archives_door", _progress(door_state="open"), _progress(door_state="closed")),
    ]
    annotate_door_sequences(records)
    metrics = door_sequence_metrics(records)
    assert metrics["closed_again_before_traverse"] == 1


def test_door_sequence_metrics_handles_no_doors_touched():
    records = [_record(1, "wait", None, _progress(), _progress())]
    annotate_door_sequences(records)
    metrics = door_sequence_metrics(records)
    assert metrics["doors_opened"] == 0
    assert metrics["open_to_traverse_success_rate"] is None


# -- door_action_sequence_narrative (spec section 41) --------------------------


def test_door_action_sequence_narrative_reports_one_line_per_door():
    records = [
        _record(10, "interact", "archives_door", _progress(door_state="closed"), _progress(door_state="open")),
        _record(11, "move", "archives_door", _progress(current_room="corridor", door_state="open"), _progress(current_room="archives", door_state="open")),
    ]
    annotate_door_sequences(records)
    lines = door_action_sequence_narrative(records)
    assert len(lines) == 1
    assert "archives_door" in lines[0]
    assert "D10" in lines[0] and "D11" in lines[0]
    assert DOOR_OPENED in lines[0] and DOOR_TRAVERSED in lines[0]


# -- end-to-end integration through build_decision_trace_records -------------


def test_build_decision_trace_records_includes_door_sequence_fields():
    decisions = [{
        "decision_number": 1, "decision_id": "d1", "game_time": 1.0,
        "raw_response": {"action": "interact", "target_id": "archives_door", "reason": "open the door"},
        "recovery_outcome": "FIRST_PASS_VALID", "initial_error_type": None, "repair_attempted": False,
        "repair_raw_response": {}, "repair_valid": None,
        "executed_action": {"action": "interact", "target_id": "archives_door"},
        "final_model_action_valid": True, "action_source": "llm",
        "latency_ms": 1000.0, "input_tokens": 100, "output_tokens": 10,
    }]
    snapshots = {0: _progress(door_state="closed"), 1: _progress(door_state="open")}
    records = build_decision_trace_records(decisions, [], snapshots, "ep1", 801, "native_tool_like")
    assert records[0]["door_sequence_state"] == DOOR_OPENED
    assert records[0]["door_state_before"] == "closed"
    assert records[0]["door_state_after"] == "open"
