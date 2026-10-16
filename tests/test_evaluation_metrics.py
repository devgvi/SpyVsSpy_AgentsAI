"""M5D spec section 34: "Metrics" - synthetic traces verify mission
success, decisions, first-pass validity, repairs, fallbacks, explicit
WAIT, latency aggregates, token totals, repetitions and trap counts.
compute_metrics is pure - no game/engine objects anywhere in this file.
"""

from src.evaluation.metrics import compute_metrics


def _decision(
    initial_valid=True,
    repair_attempted=False,
    repair_valid=None,
    final_valid=True,
    fallback=False,
    action="wait",
    target_id=None,
    latency_ms=1000.0,
    input_tokens=100,
    output_tokens=20,
    error_type=None,
    repair_error_type=None,
    tool_call_count=1,
):
    return {
        "initial_model_output_valid": initial_valid,
        "initial_error_type": error_type,
        "repair_attempted": repair_attempted,
        "repair_valid": repair_valid,
        "repair_error_type": repair_error_type,
        "final_model_action_valid": final_valid,
        "fallback_used": fallback,
        "validated_action": {"action": action, "target_id": target_id} if final_valid else None,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error_type": error_type,
        "runtime_metadata": {"tool_call_count": tool_call_count},
    }


def test_mission_success_and_progress_from_game_facts():
    game_facts = {
        "mission_success": True,
        "mission_items_required": 2,
        "mission_items_acquired": 2,
        "deadline_remaining": 42.0,
        "traps_placed": 0,
        "traps_triggered": 0,
        "enemy_traps_detected": 0,
        "traps_disarmed": 0,
        "stuns_suffered": 0,
        "opponent_encounters": 0,
        "unique_rooms_visited": 3,
        "door_unlocked_count": 1,
    }
    metrics = compute_metrics([], {}, game_facts)
    assert metrics["mission_success"] is True
    assert metrics["escape_success"] is True
    assert metrics["mission_progress_ratio"] == 1.0
    assert metrics["deadline_remaining"] == 42.0


def test_mission_progress_ratio_partial_and_none_when_no_requirements():
    assert compute_metrics([], {}, {"mission_items_required": 4, "mission_items_acquired": 1})["mission_progress_ratio"] == 0.25
    assert compute_metrics([], {}, {})["mission_progress_ratio"] is None


def test_decision_validity_and_repair_counts():
    events = [
        _decision(initial_valid=True, final_valid=True),  # clean first-pass
        _decision(initial_valid=False, repair_attempted=True, repair_valid=True, final_valid=True, error_type="missing_action"),
        _decision(initial_valid=False, repair_attempted=True, repair_valid=False, final_valid=False, fallback=True, error_type="unknown_action", repair_error_type="unknown_action"),
    ]
    metrics = compute_metrics(events, {}, {})

    assert metrics["decisions_requested"] == 3
    assert metrics["first_pass_valid"] == 1
    assert metrics["first_pass_invalid"] == 2
    assert metrics["repair_attempts"] == 2
    assert metrics["repair_successes"] == 1
    assert metrics["final_valid_llm_actions"] == 2
    assert metrics["fallbacks"] == 1
    assert metrics["invalid_actions"] == 1
    assert metrics["runtime_invocations"] == 5  # 3 decisions + 2 repairs


def test_explicit_wait_only_counted_when_genuinely_valid():
    events = [
        _decision(final_valid=True, action="wait"),
        _decision(final_valid=False, fallback=True, action=None),  # fallback wait must NOT count as explicit
    ]
    metrics = compute_metrics(events, {}, {})
    assert metrics["explicit_waits"] == 1
    assert metrics["fallbacks"] == 1


def test_illegal_and_stale_action_categories():
    events = [
        _decision(initial_valid=False, final_valid=False, fallback=True, error_type="illegal_target"),
        _decision(initial_valid=False, final_valid=False, fallback=True, error_type="stale_target"),
        _decision(initial_valid=True, final_valid=True),
    ]
    metrics = compute_metrics(events, {}, {})
    assert metrics["illegal_actions"] == 1
    assert metrics["stale_actions"] == 1


def test_latency_and_token_aggregates():
    events = [
        _decision(latency_ms=1000.0, input_tokens=100, output_tokens=10),
        _decision(latency_ms=3000.0, input_tokens=200, output_tokens=20),
        _decision(latency_ms=2000.0, input_tokens=300, output_tokens=30),
    ]
    metrics = compute_metrics(events, {}, {})
    stats = metrics["latency_ms"]
    assert stats["count"] == 3
    assert stats["mean"] == 2000.0
    assert stats["median"] == 2000.0
    assert stats["min"] == 1000.0
    assert stats["max"] == 3000.0
    assert metrics["input_tokens_total"] == 600
    assert metrics["output_tokens_total"] == 60


def test_latency_stats_empty_when_no_data():
    metrics = compute_metrics([], {}, {})
    assert metrics["latency_ms"] == {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    assert metrics["input_tokens_total"] is None


def test_move_actions_counted_from_decision_events():
    events = [
        _decision(action="move", target_id="door_1"),
        _decision(action="move", target_id="door_2"),
        _decision(action="inspect", target_id="desk_01"),
    ]
    metrics = compute_metrics(events, {}, {})
    assert metrics["move_actions"] == 2


def test_interaction_and_repetition_facts_from_memory_snapshot():
    memory_snapshot = {
        "recent_interactions": [
            {"action": "search", "target_id": "desk_01", "result": "found_item"},
            {"action": "search", "target_id": "desk_02", "result": "empty"},
            {"action": "inspect", "target_id": "cabinet_01", "result": "clear"},
        ],
        "recent_failures": [
            {"action": "interact", "target_id": "safe_01", "reason": "locked"},
            {"action": "interact", "target_id": "door_99", "reason": "unreachable"},
        ],
        "recent_repetitions": [
            {"action": "inspect", "target_id": "safe_01", "count": 3, "last_result": "blocked_locked"},
        ],
    }
    metrics = compute_metrics([], memory_snapshot, {})

    assert metrics["successful_searches"] == 1
    assert metrics["empty_searches"] == 1
    assert metrics["inspect_actions"] == 1
    assert metrics["search_actions"] == 2
    assert metrics["failed_interactions"] == 2
    assert metrics["locked_interaction_attempts"] == 1
    assert metrics["recent_repetitions_count"] == 1
    assert metrics["repeated_action_count"] == 2  # count=3 means 2 repeats
    assert metrics["revisited_failed_target_count"] == 1


def test_trap_and_competition_facts_from_game_facts():
    game_facts = {
        "traps_placed": 2,
        "traps_triggered": 1,
        "enemy_traps_detected": 1,
        "traps_disarmed": 1,
        "stuns_suffered": 1,
        "opponent_encounters": 3,
    }
    metrics = compute_metrics([], {}, game_facts)
    assert metrics["traps_placed"] == 2
    assert metrics["traps_triggered"] == 1
    assert metrics["enemy_traps_detected"] == 1
    assert metrics["traps_disarmed"] == 1
    assert metrics["stuns_suffered"] == 1
    assert metrics["opponent_encounters"] == 3


def test_compute_metrics_handles_none_inputs_gracefully():
    metrics = compute_metrics(None, None, None)
    assert metrics["decisions_requested"] == 0
    assert metrics["mission_success"] is None
