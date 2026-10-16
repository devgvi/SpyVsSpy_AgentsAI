"""M5D.4: action affordances & multi-step interaction planning (spec
sections 24-30, 39). No live LLM required anywhere in this file.

Central question this milestone answers: `legal_actions` must mean
"the actor can meaningfully attempt this right now," not merely "a
recognized action naming a real object somewhere" - see
`enumerate_legal_actions` in src/llm_controller.py and CHANGELOG.md's
M5D.4 entry for the live-telemetry audit (27/28 of M5D.3's attempted
interactions were exactly this gap: a legal-but-out-of-range target).
"""

from __future__ import annotations

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.metrics import compute_metrics
from src.items import load_items
from src.level import load_level
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController, enumerate_legal_actions, serialize_observation
from src.observation import Observation, ObservedObject, build_observation
from src.state import GameState

WHITE = constants.WHITE_SPY_ID


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
        nav_grid=None,
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


def _legal(observation):
    return {(e["action"], e.get("target_id")) for e in enumerate_legal_actions(observation)}


def _new_state(seed: int) -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=seed)


# ---------------------------------------------------------------------------
# Far vs. near object: proximity gates INSPECT/INTERACT, not MOVE
# ---------------------------------------------------------------------------


def test_far_object_move_legal_inspect_search_illegal():
    far_desk = _obj("desk_01")
    observation = _make_observation(objects=(far_desk,), nearby_object_id=None)
    legal = _legal(observation)

    assert ("move", "desk_01") in legal
    assert ("inspect", "desk_01") not in legal
    assert ("interact", "desk_01") not in legal


def test_near_object_inspect_and_interact_become_legal():
    near_desk = _obj("desk_01")
    observation = _make_observation(objects=(near_desk,), nearby_object_id="desk_01")
    legal = _legal(observation)

    assert ("inspect", "desk_01") in legal
    assert ("interact", "desk_01") in legal
    assert ("move", "desk_01") in legal  # still legal - reachability, not exclusive with proximity


def test_search_does_not_require_prior_inspection():
    """Actual game rule (GameState._search), not the spec's hypothetical
    example - spec section 3 explicitly says to use real rules. SEARCH
    (via INTERACT) and INSPECT are independent here: INSPECT reveals trap
    status without risk, SEARCH actually finds items but risks an
    undetected trap. Neither gates the other."""
    near_desk = _obj("desk_01", inspected=False)
    observation = _make_observation(objects=(near_desk,), nearby_object_id="desk_01")
    legal = _legal(observation)

    assert ("interact", "desk_01") in legal  # legal even though never inspected


# ---------------------------------------------------------------------------
# Illegal/invented actions rejected before engine execution (validator)
# ---------------------------------------------------------------------------


def test_illegal_far_inspect_rejected_before_engine_execution():
    far_desk = _obj("desk_01")
    observation = _make_observation(objects=(far_desk,), nearby_object_id=None)
    telemetry = TelemetryLogger()
    controller = LLMController(
        WHITE, MockModelClient([{"action": "inspect", "target_id": "desk_01"}, {"action": "wait"}]), telemetry=telemetry
    )

    controller.get_action(observation)
    controller._thinking_thread.join(timeout=2.0)

    decision_event = telemetry.events[-1]
    # The initial illegal proposal is rejected pre-execution regardless of
    # whether the bounded repair attempt then recovers a valid WAIT -
    # that's the point being tested, not the eventual fallback shape.
    assert decision_event["initial_model_output_valid"] is False
    # M5D.5 spec section 5: a real, known, proximity-gated target that
    # just isn't the actor's current nearby_object_id is specifically
    # "out_of_range" (split from the more generic "illegal_target").
    assert decision_event["initial_error_type"] == "out_of_range"
    # No action_lifecycle boundary for the rejected inspect proposal - it
    # never became an accepted action (spec section 16: never consume an
    # action lifecycle for a decision that was rejected pre-acceptance).
    assert not [e for e in telemetry.events if e["event_type"] == "action_lifecycle" and e.get("action") == "inspect"]


def test_invented_action_rejected():
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")
    telemetry = TelemetryLogger()
    controller = LLMController(WHITE, MockModelClient([{"action": "teleport", "target_id": "desk_01"}, {"action": "wait"}]), telemetry=telemetry)

    controller.get_action(observation)
    controller._thinking_thread.join(timeout=2.0)

    decision_event = telemetry.events[-1]
    assert decision_event["initial_model_output_valid"] is False
    assert decision_event["initial_error_type"] == "unknown_action"
    assert not [e for e in telemetry.events if e["event_type"] == "action_lifecycle" and e.get("action") == "teleport"]


def test_unknown_target_distinct_from_illegal_target():
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id=None)
    telemetry = TelemetryLogger()
    controller = LLMController(
        WHITE, MockModelClient([{"action": "inspect", "target_id": "no_such_object"}, {"action": "wait"}]), telemetry=telemetry
    )
    controller.get_action(observation)
    controller._thinking_thread.join(timeout=2.0)

    assert telemetry.events[-1]["initial_error_type"] == "unknown_target"


# ---------------------------------------------------------------------------
# No-op semantics: valid-empty vs. already-done vs. out-of-range
# ---------------------------------------------------------------------------


def test_valid_empty_search_is_a_real_execution_not_a_noop():
    state = _new_state(seed=701)
    actor = state.actors[WHITE]
    empty_desk = state.level.get_object("desk_01")
    actor.x, actor.y = empty_desk.position[0], empty_desk.position[1]
    state.update(0.0)  # recompute current_interaction_target at the new position
    observation = build_observation(state, WHITE)
    assert observation.nearby_object_id == "desk_01"

    action = GameAction(ActionType.INTERACT, target_id="desk_01")
    state.apply_action(WHITE, action, 1 / 30)

    own_events = [e["type"] for e in state.event_log if e.get("actor_id") == WHITE]
    assert "OBJECT_SEARCHED" in own_events  # a real execution happened


def test_already_searched_stays_legal_and_is_a_free_noop():
    """Documented decision (spec section 10/28): repeating SEARCH is
    intentionally allowed - GameState._search already makes it a free,
    harmless no-op with a distinct message, so it is not removed from
    legal_actions."""
    state = _new_state(seed=702)
    actor = state.actors[WHITE]
    desk = state.level.get_object("desk_01")
    actor.x, actor.y = desk.position[0], desk.position[1]
    state.update(0.0)

    state.apply_action(WHITE, GameAction(ActionType.INTERACT, target_id="desk_01"), 1 / 30)
    first_search_count = sum(1 for e in state.event_log if e["type"] == "OBJECT_SEARCHED")
    cooldown_after_first = actor.action_cooldown
    actor.action_cooldown = 0.0  # clear so a second attempt isn't blocked by is_busy

    state.apply_action(WHITE, GameAction(ActionType.INTERACT, target_id="desk_01"), 1 / 30)
    second_search_count = sum(1 for e in state.event_log if e["type"] == "OBJECT_SEARCHED")

    assert first_search_count == 1
    assert second_search_count == 1  # no duplicate OBJECT_SEARCHED - free no-op
    assert cooldown_after_first > 0.0
    assert actor.action_cooldown == 0.0  # repeat search charged no new cooldown

    observation = build_observation(state, WHITE)
    assert ("interact", "desk_01") in _legal(observation)  # still legal, not hidden


# ---------------------------------------------------------------------------
# PLACE_TRAP / DISARM preconditions
# ---------------------------------------------------------------------------


def test_place_trap_requires_prior_inspection_and_trap_inventory():
    trappable = _obj("desk_01", trappable=True, inspected=False)

    not_inspected = _make_observation(objects=(trappable,), nearby_object_id="desk_01", trap_inventory={"bomb": 1})
    assert ("place_trap", "desk_01") not in _legal(not_inspected)

    inspected_no_traps = _make_observation(
        objects=(_obj("desk_01", trappable=True, inspected=True),), nearby_object_id="desk_01", trap_inventory={"bomb": 0}
    )
    assert ("place_trap", "desk_01") not in _legal(inspected_no_traps)

    inspected_with_traps = _make_observation(
        objects=(_obj("desk_01", trappable=True, inspected=True),), nearby_object_id="desk_01", trap_inventory={"bomb": 1}
    )
    assert ("place_trap", "desk_01") in _legal(inspected_with_traps)


def test_disarm_requires_detection_or_ownership():
    undetected = _obj("desk_01", trap_status="unknown")
    assert ("disarm", "desk_01") not in _legal(_make_observation(objects=(undetected,), nearby_object_id="desk_01"))

    detected_enemy = _obj("desk_01", trap_status="detected", trap_type="bomb", trap_is_mine=False)
    assert ("disarm", "desk_01") in _legal(_make_observation(objects=(detected_enemy,), nearby_object_id="desk_01"))

    own_trap = _obj("desk_01", trap_status="detected", trap_type="bomb", trap_is_mine=True)
    assert ("disarm", "desk_01") in _legal(_make_observation(objects=(own_trap,), nearby_object_id="desk_01"))


def test_locked_object_interact_illegal_until_requirement_satisfied():
    locked_safe = _obj("safe_01", type="safe", lock_type="code", locked=True, can_unlock=False)
    assert ("interact", "safe_01") not in _legal(_make_observation(objects=(locked_safe,), nearby_object_id="safe_01"))

    unlockable_safe = _obj("safe_01", type="safe", lock_type="code", locked=True, can_unlock=True)
    assert ("interact", "safe_01") in _legal(_make_observation(objects=(unlockable_safe,), nearby_object_id="safe_01"))


# ---------------------------------------------------------------------------
# Engine defensive validation: target_id must match current proximity
# ---------------------------------------------------------------------------


def test_engine_rejects_mismatched_target_id_without_side_effect():
    state = _new_state(seed=703)
    actor = state.actors[WHITE]
    desk = state.level.get_object("desk_01")
    actor.x, actor.y = desk.position[0], desk.position[1]
    state.update(0.0)
    observation = build_observation(state, WHITE)
    assert observation.nearby_object_id == "desk_01"

    # A target_id that doesn't match what's actually nearby must not
    # execute against the (wrong) nearby object either - spec section 13:
    # engine authority is final, never a silent redirect to a different
    # object than the one named.
    state.apply_action(WHITE, GameAction(ActionType.INTERACT, target_id="ambassador_safe"), 1 / 30)
    assert not any(e["type"] == "OBJECT_SEARCHED" for e in state.event_log)


def test_target_id_none_preserves_scripted_and_human_controller_behavior():
    """M3/M4 regression guard (spec section 40): HumanController/
    ScriptedBotController never set target_id - GameAction(INTERACT) with
    no target must keep acting on whatever's nearest, unchanged."""
    state = _new_state(seed=704)
    actor = state.actors[WHITE]
    desk = state.level.get_object("desk_01")
    actor.x, actor.y = desk.position[0], desk.position[1]
    state.update(0.0)

    state.apply_action(WHITE, GameAction(ActionType.INTERACT), 1 / 30)
    assert any(e["type"] == "OBJECT_SEARCHED" and e["object_id"] == "desk_01" for e in state.event_log)


# ---------------------------------------------------------------------------
# Observation / legal-target consistency (no hidden target leakage)
# ---------------------------------------------------------------------------


def test_every_legal_target_is_a_real_known_object_no_hidden_leakage():
    state = _new_state(seed=705)
    observation = build_observation(state, WHITE)
    legal = enumerate_legal_actions(observation)
    known_ids = {obj.id for obj in observation.objects}

    for entry in legal:
        target_id = entry.get("target_id")
        if target_id is not None:
            assert target_id in known_ids, f"legal_actions named an object not present in Observation: {target_id}"


# ---------------------------------------------------------------------------
# Native / Strands parity
# ---------------------------------------------------------------------------


def test_native_and_strands_receive_identical_legal_actions():
    from tests.test_strands_agent import QueuedStrandsFactory, StrandsAgentAdapter

    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")

    native_telemetry = TelemetryLogger()
    native = LLMController(WHITE, MockModelClient([{"action": "wait"}]), telemetry=native_telemetry)
    native.get_action(observation)
    native._thinking_thread.join(timeout=2.0)

    strands_telemetry = TelemetryLogger()
    factory = QueuedStrandsFactory([[{"action": "wait"}]])
    strands = StrandsAgentAdapter(WHITE, agent_factory=factory, telemetry=strands_telemetry)
    strands.get_action(observation)
    if strands._thinking_thread:
        strands._thinking_thread.join(timeout=2.0)

    native_legal = {(e["action"], e.get("target_id")) for e in native_telemetry.events[-1]["legal_actions"]}
    strands_legal = {(e["action"], e.get("target_id")) for e in strands_telemetry.events[-1]["legal_actions"]}
    assert native_legal == strands_legal


# ---------------------------------------------------------------------------
# Interaction funnel metrics
# ---------------------------------------------------------------------------


def test_interaction_funnel_counts_each_stage_with_explicit_denominators():
    decision_events = [
        {  # proposed + legal + accepted + executed + successful
            "raw_response": {"action": "inspect", "target_id": "desk_01"},
            "final_model_action_valid": True,
            "validated_action": {"action": "inspect", "target_id": "desk_01"},
            "initial_error_type": None,
        },
        {  # proposed but out of range - never accepted
            "raw_response": {"action": "interact", "target_id": "far_object"},
            "final_model_action_valid": False,
            "validated_action": None,
            "initial_error_type": "out_of_range",
        },
        {  # proposed but named a nonexistent object
            "raw_response": {"action": "inspect", "target_id": "ghost"},
            "final_model_action_valid": False,
            "validated_action": None,
            "initial_error_type": "unknown_target",
        },
    ]
    action_lifecycle_events = [
        {"action": "inspect", "target_id": "desk_01", "action_status": "completed", "action_terminal_reason": "resolved"},
    ]
    memory_snapshot = {
        "recent_decisions": [{"selected_action": "inspect", "engine_result": "success"}],
        "recent_interactions": [],
    }

    metrics = compute_metrics(decision_events, memory_snapshot, {}, action_lifecycle_events)

    assert metrics["interaction_funnel"]["proposed"] == 3
    assert metrics["interaction_funnel"]["legal"] == 1
    assert metrics["interaction_funnel"]["accepted"] == 1
    assert metrics["interaction_funnel"]["executed"] == 1
    assert metrics["interaction_funnel"]["successful"] == 1
    assert metrics["out_of_range_model_actions"] == 1
    assert metrics["unknown_target_actions"] == 1
    assert metrics["interaction_success_rate"] == 1.0
