"""M5D.5: interaction recovery & planning contract (spec sections 13, 28).
No live LLM required anywhere in this file - deterministic fake-model/
fake-runtime scenarios only, per spec section 23's explicit instruction.

Covers the controlled Native-vs-Strands comparison spec section 2 asks
for: same Observation, same legal_actions, same shared validator, same
repair policy - only runtime-specific orchestration should differ. See
CHANGELOG.md's M5D.5 entry for the real live-telemetry trace this file's
scenarios are modeled on (Native repeating the identical rejected
`inspect(ambassador_safe)` proposal verbatim on repair, 7/8 times, vs.
Strands recovering to a different, legal target 3/9 times).
"""

from __future__ import annotations

from src.actions import ActionType
from src.evaluation.metrics import compute_metrics
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import (
    REPAIR_PROMPT_VERSION,
    LLMController,
    _recovery_outcome,
    _repair_instruction,
)
from src.observation import Observation, ObservedObject

WHITE = "white"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeNavGrid:
    """Minimal NavGrid stand-in so a repaired MOVE can genuinely start
    (ActionExecutor.start() needs nearest_approach_point/find_path) -
    always resolves a trivial one-step path, since these tests care about
    repair/validation outcomes, not pathfinding itself (see
    test_action_lifecycle.py's real-NavGrid tests for that)."""

    def nearest_approach_point(self, rect, max_distance, from_xy):
        return (from_xy[0] + 50.0, from_xy[1])

    def find_path(self, start_xy, goal_xy):
        return [goal_xy]


def _obj(id, **overrides):
    base = dict(
        id=id,
        type="desk",
        label=id,
        position=(10.0, 10.0),
        collision_size=(20.0, 20.0),
        interactable=True,
        trappable=False,
        searched=False,
        inspected=False,
        trap_status="unknown",
        trap_type=None,
        trap_is_mine=False,
        door_open=None,
        lock_type=None,
        locked=False,
        can_unlock=False,
    )
    base.update(overrides)
    return ObservedObject(**base)


def _make_observation(**overrides):
    base = dict(
        actor_id="white",
        seed=1,
        position=(5.0, 5.0),
        facing="east",
        stunned=False,
        stunned_seconds=0.0,
        busy=False,
        busy_seconds=0.0,
        mission_time_remaining=100.0,
        mission_complete=False,
        inventory=frozenset(),
        knowledge=frozenset(),
        required_items=(),
        trap_inventory={},
        nearby_object_id=None,
        items_found_here=frozenset(),
        objects=(),
        nav_grid=_FakeNavGrid(),
        log_event=lambda *a, **k: None,
        current_room="office",
        known_rooms=frozenset({"office"}),
        known_doors=frozenset(),
        known_locked_objects={},
        known_item_locations={},
        visible_other_actor=False,
        opponent=None,
        last_seen_opponent=None,
        revealed_opponent=None,
        revealed_opponent_seconds=0.0,
    )
    base.update(overrides)
    return Observation(**base)


def _run_native(responses, observation):
    telemetry = TelemetryLogger()
    controller = LLMController(WHITE, MockModelClient(responses), telemetry=telemetry)
    controller.get_action(observation)
    controller._thinking_thread.join(timeout=2.0)
    return controller, telemetry.events[-1]


# ---------------------------------------------------------------------------
# A - Far interaction: illegal -> one repair -> repaired MOVE_TO
# ---------------------------------------------------------------------------


def test_a_native_far_interaction_repairs_to_move():
    desk = _obj("desk_01")
    observation = _make_observation(objects=(desk,), nearby_object_id=None)
    controller, event = _run_native(
        [{"action": "inspect", "target_id": "desk_01"}, {"action": "move", "target_id": "desk_01"}], observation
    )

    assert event["initial_model_output_valid"] is False
    assert event["initial_error_type"] == "out_of_range"
    assert event["repair_attempted"] is True
    assert event["repair_valid"] is True
    assert event["recovery_outcome"] == "REPAIRED_VALID"
    assert controller.last_action.type == ActionType.MOVE


def test_a_strands_far_interaction_repairs_to_move():
    from tests.test_strands_agent import QueuedStrandsFactory, StrandsAgentAdapter

    desk = _obj("desk_01")
    observation = _make_observation(objects=(desk,), nearby_object_id=None)
    telemetry = TelemetryLogger()
    factory = QueuedStrandsFactory(
        [[{"action": "inspect", "target_id": "desk_01"}], [{"action": "move", "target_id": "desk_01"}]]
    )
    adapter = StrandsAgentAdapter(WHITE, agent_factory=factory, telemetry=telemetry)

    adapter.get_action(observation)
    if adapter._thinking_thread:
        adapter._thinking_thread.join(timeout=2.0)

    event = telemetry.events[-1]
    assert event["initial_model_output_valid"] is False
    assert event["initial_error_type"] == "out_of_range"
    assert event["repair_attempted"] is True
    assert event["repair_valid"] is True
    assert event["recovery_outcome"] == "REPAIRED_VALID"
    assert adapter.last_action.type == ActionType.MOVE


# ---------------------------------------------------------------------------
# B - Near object: first-pass valid, no repair
# ---------------------------------------------------------------------------


def test_b_near_object_first_pass_valid_no_repair():
    desk = _obj("desk_01")
    observation = _make_observation(objects=(desk,), nearby_object_id="desk_01")
    controller, event = _run_native([{"action": "inspect", "target_id": "desk_01"}], observation)

    assert event["initial_model_output_valid"] is True
    assert event["repair_attempted"] is False
    assert event["recovery_outcome"] == "FIRST_PASS_VALID"
    assert controller.last_action.type == ActionType.INSPECT


# ---------------------------------------------------------------------------
# C - Wrong nearby target: never silently redirected
# ---------------------------------------------------------------------------


def test_c_wrong_target_repairs_without_silent_redirect():
    statue = _obj("statue_01")
    desk = _obj("desk_02")
    observation = _make_observation(objects=(statue, desk), nearby_object_id="statue_01")
    controller, event = _run_native(
        [{"action": "inspect", "target_id": "desk_02"}, {"action": "inspect", "target_id": "statue_01"}], observation
    )

    assert event["initial_model_output_valid"] is False
    assert event["initial_error_type"] == "out_of_range"
    assert event["repair_valid"] is True
    # The repaired action targets the actor's real nearby object, not a
    # silent substitution of the original (wrong) target.
    assert controller.last_action.target_id == "statue_01"
    assert controller.last_action.target_id != "desk_02"


# ---------------------------------------------------------------------------
# D - Post-inspect SEARCH availability (recovery-context framing)
# ---------------------------------------------------------------------------


def test_d_search_becomes_legal_after_inspect_no_engine_macro_chaining():
    from src.llm_controller import enumerate_legal_actions

    desk_uninspected = _obj("desk_01", inspected=False)
    before = _make_observation(objects=(desk_uninspected,), nearby_object_id="desk_01")
    legal_before = {(e["action"], e.get("target_id")) for e in enumerate_legal_actions(before)}
    assert ("interact", "desk_01") in legal_before  # already legal - SEARCH never required prior INSPECT here

    # A fresh Observation reflecting the engine's own authoritative
    # inspected=True - never auto-chained by the engine itself (the model
    # must still issue INSPECT and then, on a later decision, SEARCH).
    desk_inspected = _obj("desk_01", inspected=True)
    after = _make_observation(objects=(desk_inspected,), nearby_object_id="desk_01")
    legal_after = {(e["action"], e.get("target_id")) for e in enumerate_legal_actions(after)}
    assert ("interact", "desk_01") in legal_after


# ---------------------------------------------------------------------------
# E - Repeated illegal proposal -> fallback after exactly one repair
# ---------------------------------------------------------------------------


def test_e_repeated_illegal_proposal_falls_back_after_one_repair():
    desk = _obj("desk_01")
    observation = _make_observation(objects=(desk,), nearby_object_id=None)
    controller, event = _run_native(
        [{"action": "inspect", "target_id": "desk_01"}, {"action": "inspect", "target_id": "desk_01"}], observation
    )

    assert event["initial_model_output_valid"] is False
    assert event["repair_attempted"] is True
    assert event["repair_valid"] is False
    assert event["fallback_used"] is True
    assert event["action_source"] == "fallback"
    assert event["recovery_outcome"] == "REPAIR_FAILED_FALLBACK"
    assert controller.last_action.type == ActionType.WAIT
    # Never more than one repair attempt (M5B.1, preserved - spec section 14).
    assert controller.repair_count == 1


def test_same_illegal_action_repeated_is_detected():
    """The exact live-observed Native pattern: initial and repair name the
    identical (action, target_id) pair - the model didn't reconsider at
    all."""
    events = [
        {
            "raw_response": {"action": "inspect", "target_id": "ambassador_safe"},
            "initial_proposal": {"action": "inspect", "target_id": "ambassador_safe"},
            "initial_error_type": "out_of_range",
            "repair_attempted": True,
            "repair_valid": False,
            "repair_raw_response": {"action": "inspect", "target_id": "ambassador_safe"},
            "fallback_used": True,
            "final_model_action_valid": False,
        }
    ]
    metrics = compute_metrics(events, {}, {})
    assert metrics["same_illegal_action_repeated"] == 1
    assert metrics["fallback_after_illegal"] == 1


# ---------------------------------------------------------------------------
# F - Stale repair: rejected against a state that has since changed
# ---------------------------------------------------------------------------


def test_f_stale_repair_rejected_against_changed_state():
    """Simulates spec section 6/13's staleness scenario: a proposal that
    was legal against observation A is re-validated (as repair inputs
    would be) against a *different*, later observation B where it no
    longer holds - the shared validator is a pure function of whatever
    Observation it's given, so it correctly rejects it there, with no
    stale execution. Genuine async staleness in production remains
    AgentMemoryTracker._check_pending_decision_failure's job one tick
    later, against a real new Observation - not something repair itself
    re-fetches (there's no new Observation available mid single decision
    request); this test locks in that the validator would correctly
    reject a stale repair *if* one were ever checked against updated
    state."""
    from src.llm_controller import validate_model_output, enumerate_legal_actions

    desk = _obj("desk_01")
    observation_a = _make_observation(objects=(desk,), nearby_object_id="desk_01")
    proposal = {"action": "inspect", "target_id": "desk_01"}
    result_a = validate_model_output(observation_a, proposal, enumerate_legal_actions(observation_a))
    assert result_a.valid is True

    # State changed: the actor is no longer near desk_01 (e.g. it moved to
    # continue a MOVE, or a different object became nearest).
    observation_b = _make_observation(objects=(desk,), nearby_object_id=None)
    result_b = validate_model_output(observation_b, proposal, enumerate_legal_actions(observation_b))
    assert result_b.valid is False
    assert result_b.error_type == "out_of_range"


# ---------------------------------------------------------------------------
# Memory never records false success
# ---------------------------------------------------------------------------


def test_memory_never_records_false_success_for_rejected_interaction():
    desk = _obj("desk_01", inspected=False)
    observation = _make_observation(objects=(desk,), nearby_object_id=None)
    controller, event = _run_native(
        [{"action": "inspect", "target_id": "desk_01"}, {"action": "inspect", "target_id": "desk_01"}], observation
    )

    assert event["fallback_used"] is True
    # No interaction was ever recorded as having actually happened.
    assert controller.memory.known_searched_objects == set()
    interactions = [r for r in controller.memory.recent_interactions if r.target_id == "desk_01"]
    assert interactions == []
    decisions = [d for d in controller.memory.recent_decisions if d.target_id == "desk_01"]
    assert all(d.valid is False for d in decisions)


# ---------------------------------------------------------------------------
# Runtime-neutral repair semantics
# ---------------------------------------------------------------------------


def test_repair_instruction_is_shared_and_runtime_neutral():
    instruction = _repair_instruction(
        "out_of_range",
        "inspect is not currently legal",
        proposed_action="inspect",
        proposed_target_id="ambassador_safe",
        near_object_id="desk_01",
        legal_action_count=4,
    )
    assert "inspect(ambassador_safe)" in instruction
    assert "out_of_range" in instruction
    assert "desk_01" in instruction
    assert "4" in instruction
    assert "do not repeat" in instruction.lower()
    # Never coaches mission strategy (spec section 15) - no suggested next
    # concrete action beyond "choose from legal_actions".
    assert "then" not in instruction.lower()


def test_native_and_strands_repair_receive_equivalent_context():
    from tests.test_strands_agent import QueuedStrandsFactory, StrandsAgentAdapter

    desk = _obj("desk_01")
    observation = _make_observation(objects=(desk,), nearby_object_id=None)

    native_telemetry = TelemetryLogger()
    native = LLMController(
        WHITE,
        MockModelClient([{"action": "inspect", "target_id": "desk_01"}, {"action": "wait"}]),
        telemetry=native_telemetry,
    )
    native.get_action(observation)
    native._thinking_thread.join(timeout=2.0)

    strands_telemetry = TelemetryLogger()
    factory = QueuedStrandsFactory([[{"action": "inspect", "target_id": "desk_01"}], [{"action": "wait"}]])
    strands = StrandsAgentAdapter(WHITE, agent_factory=factory, telemetry=strands_telemetry)
    strands.get_action(observation)
    if strands._thinking_thread:
        strands._thinking_thread.join(timeout=2.0)

    native_event = native_telemetry.events[-1]
    strands_event = strands_telemetry.events[-1]

    assert native_event["repair_context_version"] == strands_event["repair_context_version"] == REPAIR_PROMPT_VERSION
    assert native_event["repair_legal_action_count"] == strands_event["repair_legal_action_count"]
    assert native_event["repair_near_object_id"] == strands_event["repair_near_object_id"] is None
    assert native_event["initial_error_type"] == strands_event["initial_error_type"] == "out_of_range"


# ---------------------------------------------------------------------------
# Recovery telemetry / metrics
# ---------------------------------------------------------------------------


def test_recovery_telemetry_fields_present_and_correct():
    desk = _obj("desk_01")
    observation = _make_observation(objects=(desk,), nearby_object_id=None)
    controller, event = _run_native(
        [{"action": "inspect", "target_id": "desk_01"}, {"action": "move", "target_id": "desk_01"}], observation
    )

    assert event["initial_proposal"] == {"action": "inspect", "target_id": "desk_01"}
    assert event["repair_context_version"] == REPAIR_PROMPT_VERSION
    assert event["repair_legal_action_count"] == event["legal_actions_count"]
    assert event["repair_near_object_id"] is None
    assert event["recovery_outcome"] == "REPAIRED_VALID"


def test_recovery_outcome_helper_covers_all_synchronous_cases():
    assert _recovery_outcome(True, False, None) == "FIRST_PASS_VALID"
    assert _recovery_outcome(False, True, True) == "REPAIRED_VALID"
    assert _recovery_outcome(False, True, False) == "REPAIR_FAILED_FALLBACK"
    assert _recovery_outcome(False, False, None) == "REPAIR_FAILED_FALLBACK"
    assert _recovery_outcome(False, False, None, runtime_failure=True) == "RUNTIME_FAILURE"


def test_recovery_metrics_computed_with_explicit_denominators():
    decision_events = [
        {
            "final_model_action_valid": True,
            "repair_attempted": False,
            "initial_error_type": None,
        },
        {
            "final_model_action_valid": True,
            "repair_attempted": True,
            "repair_attempted": True,
            "repair_valid": True,
            "initial_error_type": "out_of_range",
            "initial_proposal": {"action": "inspect", "target_id": "ambassador_safe"},
            "repair_raw_response": {"action": "move", "target_id": "ambassador_safe"},
            "repair_near_object_id": None,
        },
        {
            "final_model_action_valid": False,
            "fallback_used": True,
            "repair_attempted": True,
            "repair_valid": False,
            "initial_error_type": "out_of_range",
            "initial_proposal": {"action": "inspect", "target_id": "ambassador_safe"},
            "repair_raw_response": {"action": "inspect", "target_id": "ambassador_safe"},
        },
    ]
    metrics = compute_metrics(decision_events, {}, {})

    assert metrics["illegal_first_pass_actions"] == 2
    assert metrics["repair_attempts"] == 2
    assert metrics["repair_successes"] == 1
    assert metrics["repair_success_rate"] == 0.5
    assert metrics["fallback_after_illegal"] == 1
    assert metrics["same_illegal_action_repeated"] == 1
    assert metrics["recovered_to_move"] == 1
