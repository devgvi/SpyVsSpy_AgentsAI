"""M5B: Agent decision memory & strategic context.

Covers MILESTONE doc section 14 - persistent facts, bounded failures,
decision correlation, no-leakage, and bounded growth over a long episode.
"""

from dataclasses import replace
import json

from src.llm.agent_memory import AgentMemory
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController
from src.observation import Observation, ObservedObject, OpponentView


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
        required_items=("secret_documents",),
        trap_inventory={},
        nearby_object_id=None,
        items_found_here=frozenset(),
        objects=(),
        nav_grid=None,
        log_event=lambda *args, **kwargs: None,
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


def _tick(controller, observation):
    """Advance one frame's worth of memory bookkeeping without necessarily
    issuing a new LLM decision (mirrors _update_memory_from_observation
    running every game.py frame, decisions only every few)."""
    controller._update_memory_from_observation(observation)


# ---------------------------------------------------------------------------
# Persistent facts (section 14: "Persistent facts")
# ---------------------------------------------------------------------------


def test_visited_rooms_persist_across_observations():
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    controller.get_action(_make_observation(current_room="reception"))
    controller.get_action(_make_observation(current_room="hallway", mission_time_remaining=90.0))
    controller.get_action(_make_observation(current_room="office", mission_time_remaining=80.0))

    assert controller.memory.exploration.visited_rooms == ["reception", "hallway", "office"]


def test_known_lock_requirements_persist():
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    safe = _obj("ambassador_safe", type="safe", lock_type="code", locked=True, can_unlock=False)
    obs = _make_observation(objects=(safe,), known_locked_objects={"ambassador_safe": "code"})

    controller.get_action(obs)

    assert "ambassador_safe" in controller.memory.locks.locks
    lock = controller.memory.locks.locks["ambassador_safe"]
    assert lock.lock_type == "code"
    assert lock.satisfied is False

    # Room changes, safe no longer in view - lock knowledge must still persist.
    controller.get_action(_make_observation(objects=(), current_room="hallway", mission_time_remaining=90.0))
    assert "ambassador_safe" in controller.memory.locks.locks


def test_known_searched_and_empty_objects_persist():
    controller = LLMController("white", MockModelClient({"action": "interact", "target_id": "desk_01"}))
    desk = _obj("desk_01")
    obs1 = _make_observation(objects=(desk,), nearby_object_id="desk_01")
    controller.get_action(obs1)

    # Engine resolves the interact: desk is now searched, nothing found.
    desk_after = _obj("desk_01", searched=True)
    obs2 = _make_observation(objects=(desk_after,), nearby_object_id="desk_01", mission_time_remaining=99.0)
    _tick(controller, obs2)

    assert "desk_01" in controller.memory.known_searched_objects
    assert "desk_01" in controller.memory.known_empty_objects

    # Object drops out of view entirely - the fact must still persist.
    controller.get_action(_make_observation(objects=(), current_room="hallway", mission_time_remaining=80.0))
    assert "desk_01" in controller.memory.known_searched_objects
    assert "desk_01" in controller.memory.known_empty_objects


def test_mission_item_progress_persists_and_updates():
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    controller.get_action(_make_observation(inventory=frozenset(), required_items=("passport",)))
    assert controller.memory.mission.have == []
    assert controller.memory.mission.missing == ["passport"]
    assert controller.memory.mission.ready_to_exit is False

    # Not mission_complete=True: that only flips after actually exiting, at
    # which point get_action's early-return guard stops updating memory at
    # all. ready_to_exit means "I currently hold everything required".
    controller.get_action(
        _make_observation(
            inventory=frozenset({"passport"}),
            required_items=("passport",),
            mission_time_remaining=90.0,
        )
    )
    assert controller.memory.mission.have == ["passport"]
    assert controller.memory.mission.missing == []
    assert controller.memory.mission.ready_to_exit is True


# ---------------------------------------------------------------------------
# Failures (section 14: "Failures")
# ---------------------------------------------------------------------------


def test_failed_interaction_on_locked_target_appears_in_recent_failures():
    # M5D.4: interact(locked-and-can't-unlock) is no longer offered as a
    # legal action at all (spec sections 9/29 - a missing key/code must
    # make the gated action unavailable, not merely a silent engine no-op),
    # so a MockModelClient response requesting it would now be rejected
    # before ever becoming an accepted/pending decision. This still tests
    # a real scenario _check_pending_decision_failure legitimately guards
    # against: a decision that *was* legal/accepted a moment earlier
    # (memory only learns the world changed one tick later) going stale by
    # the time its outcome is checked - simulated directly rather than
    # through a model response that the validator would now correctly
    # reject up front.
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    safe = _obj("ambassador_safe", type="safe", lock_type="code", locked=True, can_unlock=False)
    obs1 = _make_observation(objects=(safe,), nearby_object_id="ambassador_safe")
    controller._update_memory_from_observation(obs1)
    controller.memory.record_decision(
        game_time=obs1.mission_time_remaining,
        decision_id="dec-1",
        selected_action="interact",
        target_id="ambassador_safe",
        valid=True,
        action_source="llm",
    )
    controller._memory_tracker.set_pending_decision("dec-1", "interact", "ambassador_safe", "ambassador_safe")

    # One tick later: engine left the safe untouched (still locked) - the
    # pending decision should resolve to a recorded failure.
    obs2 = replace(obs1, mission_time_remaining=99.0)
    _tick(controller, obs2)

    assert len(controller.memory.recent_failures) == 1
    failure = controller.memory.recent_failures[0]
    assert failure.target_id == "ambassador_safe"
    assert failure.reason == "locked"

    decision = controller.memory.recent_decisions[-1]
    assert decision.engine_result == "blocked_locked"


def test_successful_retry_after_requirement_satisfied_updates_state():
    # M5D.4: same staleness simulation as the locked-target test above -
    # interact(locked-and-can't-unlock) is no longer a legal decision at
    # all, so the initial failure is set up directly rather than through a
    # model response the validator would now correctly reject.
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    safe = _obj("ambassador_safe", type="safe", lock_type="code", locked=True, can_unlock=False)
    obs0 = _make_observation(objects=(safe,), nearby_object_id="ambassador_safe")
    controller._update_memory_from_observation(obs0)
    controller.memory.record_decision(
        game_time=obs0.mission_time_remaining,
        decision_id="dec-1",
        selected_action="interact",
        target_id="ambassador_safe",
        valid=True,
        action_source="llm",
    )
    controller._memory_tracker.set_pending_decision("dec-1", "interact", "ambassador_safe", "ambassador_safe")
    _tick(controller, _make_observation(objects=(safe,), nearby_object_id="ambassador_safe", mission_time_remaining=99.0))
    assert len(controller.memory.recent_failures) == 1

    # Actor learns the code; safe becomes unlockable, then gets searched.
    safe_unlocked = _obj("ambassador_safe", type="safe", lock_type="code", locked=False, can_unlock=True)
    controller.get_action(
        _make_observation(objects=(safe_unlocked,), nearby_object_id="ambassador_safe", knowledge=frozenset({"safe_code"}), mission_time_remaining=90.0)
    )
    assert controller.memory.locks.locks["ambassador_safe"].satisfied is True

    safe_searched = _obj("ambassador_safe", type="safe", lock_type="code", locked=False, can_unlock=True, searched=True)
    _tick(
        controller,
        _make_observation(objects=(safe_searched,), nearby_object_id="ambassador_safe", knowledge=frozenset({"safe_code"}), mission_time_remaining=80.0),
    )
    assert "ambassador_safe" in controller.memory.known_searched_objects
    results = [i.result for i in controller.memory.recent_interactions if i.target_id == "ambassador_safe"]
    assert "empty" in results or "found_item" in results


def test_old_failures_expire_from_bounded_recent_history():
    memory = AgentMemory("white", max_recent_failures=3)
    for i in range(5):
        memory.record_failure(game_time=float(i), action="interact", target_id=f"door_{i}", reason="locked")

    assert len(memory.recent_failures) == 3
    remaining = [f.target_id for f in memory.recent_failures]
    assert remaining == ["door_2", "door_3", "door_4"]


# ---------------------------------------------------------------------------
# Decisions (section 14: "Decisions")
# ---------------------------------------------------------------------------


def test_validated_llm_decision_is_recorded():
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    controller.get_action(_make_observation())

    assert len(controller.memory.recent_decisions) == 1
    decision = controller.memory.recent_decisions[0]
    assert decision.valid is True
    assert decision.action_source == "llm"


def test_fallback_is_distinguished_from_llm_action():
    # M5B.1: repair is a real second model call (never a locally-fabricated
    # fix), so both the initial and the repair response must be genuinely
    # invalid to reach a real fallback.
    controller = LLMController("white", MockModelClient([{"action": "interact"}, {"foo": "bar"}]))
    controller.get_action(_make_observation())

    decision = controller.memory.recent_decisions[-1]
    assert decision.valid is False
    assert decision.action_source == "fallback"


def test_engine_result_is_correlated_to_decision_where_possible():
    controller = LLMController("white", MockModelClient({"action": "interact", "target_id": "desk_01"}))
    desk = _obj("desk_01")
    controller.get_action(_make_observation(objects=(desk,), nearby_object_id="desk_01"))

    desk_after = _obj("desk_01", searched=True)
    _tick(controller, _make_observation(objects=(desk_after,), nearby_object_id="desk_01", mission_time_remaining=99.0))

    decision = controller.memory.recent_decisions[-1]
    assert decision.engine_result == "success"


# ---------------------------------------------------------------------------
# Telemetry (section 12): decision_id correlation + memory metrics
# ---------------------------------------------------------------------------


def test_telemetry_decision_id_matches_memory_decision_id():
    logger = TelemetryLogger(path=None)
    controller = LLMController("white", MockModelClient({"action": "wait"}), telemetry=logger)
    controller.get_action(_make_observation())

    llm_event = next(e for e in logger.events if e["event_type"] == "llm_decision")
    memory_decision = controller.memory.recent_decisions[-1]
    assert llm_event["decision_id"] == memory_decision.decision_id


def test_telemetry_includes_memory_metrics():
    logger = TelemetryLogger(path=None)
    controller = LLMController("white", MockModelClient({"action": "wait"}), telemetry=logger)
    controller.get_action(_make_observation())

    llm_event = next(e for e in logger.events if e["event_type"] == "llm_decision")
    assert llm_event["memory_schema_version"] == AgentMemory.MEMORY_SCHEMA_VERSION
    assert llm_event["memory_size_chars"] > 0
    assert llm_event["visited_rooms_count"] >= 1


# ---------------------------------------------------------------------------
# No leakage (section 14: "No leakage")
# ---------------------------------------------------------------------------


def test_opponent_memory_never_exposes_live_off_room_position():
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    opponent = OpponentView(actor_id="black", position=(42.0, 17.0), facing="north", room="office", stunned=False)
    controller.get_action(_make_observation(visible_other_actor=True, opponent=opponent))

    compact = controller.memory.to_compact_dict()
    opponent_memory = compact["opponent_memory"]
    assert set(opponent_memory.keys()) == {"currently_visible", "last_seen_room", "last_seen_time"}
    assert "42" not in json.dumps(opponent_memory)
    assert "position" not in json.dumps(opponent_memory)

    # Opponent leaves the room: no live position, but last_seen persists (Scenario D).
    controller.get_action(_make_observation(visible_other_actor=False, mission_time_remaining=90.0))
    compact2 = controller.memory.to_compact_dict()
    assert compact2["opponent_memory"]["currently_visible"] is False
    assert compact2["opponent_memory"]["last_seen_room"] == "office"


def test_unknown_lock_requirement_is_never_fabricated():
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    mystery_door = _obj("mystery_door", type="door", lock_type="keycard", locked=True, can_unlock=False)
    # Deliberately no known_locked_objects entry - actor hasn't legitimately learned it.
    controller.get_action(_make_observation(objects=(mystery_door,)))

    lock = controller.memory.locks.locks["mystery_door"]
    assert lock.requires == "unknown"
    assert "keycard_master_code" not in json.dumps(controller.memory.to_compact_dict())


def test_undetected_trap_never_appears_in_memory():
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    trapped_but_unknown = _obj("cabinet_01", trappable=True, trap_status="unknown", trap_type=None)
    controller.get_action(_make_observation(objects=(trapped_but_unknown,)))
    _tick(controller, _make_observation(objects=(trapped_but_unknown,), mission_time_remaining=90.0))

    dump = json.dumps(controller.memory.to_compact_dict())
    assert "cabinet_01" not in dump  # never inspected, searched, locked, or failed on


def test_memory_never_leaks_generator_or_engine_internals():
    controller = LLMController("white", MockModelClient({"action": "wait"}))
    controller.get_action(_make_observation())
    dump = json.dumps(controller.memory.to_compact_dict())
    assert "seed" not in dump
    assert "nav_grid" not in dump


# ---------------------------------------------------------------------------
# Bounded growth (section 14: "Bounded growth")
# ---------------------------------------------------------------------------


def test_bounded_growth_over_long_synthetic_episode():
    # Rotate through a handful of targets so consecutive ticks on the same
    # object can produce real transitions (failures, successful searches),
    # not just novel objects that never generate a diff.
    target_ids = [f"safe_{i}" for i in range(4)]
    responses = [{"action": "interact", "target_id": target_ids[i % 4]} for i in range(120)]
    controller = LLMController("white", MockModelClient(responses))

    for i in range(120):
        target = target_ids[i % 4]
        locked = i % 3 != 0
        room = f"room_{i % 5}"
        obj = _obj(target, type="safe", lock_type="code", locked=locked, can_unlock=not locked, searched=not locked and i % 2 == 0)
        obs = _make_observation(
            objects=(obj,),
            nearby_object_id=target,
            current_room=room,
            known_rooms=frozenset({room}),
            mission_time_remaining=max(0.0, 200.0 - i),
        )
        controller.get_action(obs)

    memory = controller.memory
    assert len(memory.recent_events) <= memory.max_recent_events
    assert len(memory.recent_failures) <= memory.max_recent_failures
    assert len(memory.recent_decisions) <= memory.max_recent_decisions
    assert memory.get_memory_size_chars() < 20000

    # Persistent facts are allowed to keep growing with genuinely new info.
    assert len(memory.exploration.visited_rooms) == 5
