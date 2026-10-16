"""M5D.2: action-execution lifecycle regression tests (spec sections 16-17).

Covers, with no live LLM required:
- the required deterministic acceptance scenario (section 16): MOVE(desk) ->
  INSPECT(desk) -> INTERACT/search(desk) -> MOVE(door), using a real
  GameState/NavGrid so pathfinding is exercised for real, not mocked;
- MOVE spans many engine ticks as exactly one strategic decision, and no
  duplicate decision is requested while it progresses (the actual M5D.1
  root-cause regression guard - see action_lifecycle.py's module docstring
  for the full diagnostic trace);
- all three WAIT semantics (explicit / fallback / inference-pending) stay
  distinguishable;
- PATH_UNREACHABLE is a legitimate execution failure, not a validity error;
- decision/action/outcome correlation and boundary reasons are visible in
  telemetry, with no hidden-truth leakage into the lifecycle event schema;
- artificial async model latency does not duplicate decisions;
- model wall latency does not inflate mission (game) time in evaluation
  mode (Game._step's pre-existing should_pause_time logic, unchanged by
  M5D.2, but locked in here as a documented property of the fix);
- a harness decision cap stops new decisions, not the in-progress action.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import math

from src import constants
from src.actions import ActionType
from src.agents.action_lifecycle import ActionExecutor, PATH_UNREACHABLE
from src.items import load_items
from src.level import load_level
from src.llm.agent_memory import AgentMemory
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController
from src.observation import Observation, ObservedObject, build_observation
from src.state import GameState

WHITE = constants.WHITE_SPY_ID


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class SlowModelClient:
    def __init__(self, response, delay=0.3):
        self.response = response
        self.delay = delay
        self.calls = 0

    def generate_action(self, *args, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        return self.response


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
        current_room="reception",
        known_rooms=frozenset({"reception"}),
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


class _UnreachableNavGrid:
    """A NavGrid stand-in whose target genuinely cannot be reached - the
    same shape ScriptedBotController/ActionExecutor already expect."""

    def nearest_approach_point(self, rect, max_distance, from_xy):
        return (0.0, 0.0)

    def find_path(self, start_xy, goal_xy):
        return None


def _new_state(seed: int) -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=seed)


def _drive_tick(state: GameState, controller: LLMController, actor_id: str, dt: float = 1 / 30):
    observation = build_observation(state, actor_id)
    action = controller.get_action(observation)
    state.apply_action(actor_id, action, dt)
    state.update(dt)
    return observation, action


# ---------------------------------------------------------------------------
# Section 16: the required deterministic acceptance scenario
# ---------------------------------------------------------------------------


def test_full_acceptance_scenario_move_inspect_search_move():
    """Decision 1: MOVE_TO(desk) -> many ticks -> desk reached.
    Decision 2: INSPECT(desk) -> resolves.
    Decision 3: SEARCH(desk) (this codebase's INTERACT on a searchable
    object) -> resolves.
    Decision 4: MOVE_TO(door) -> many ticks -> door reached.
    No live LLM: MockModelClient replays a fixed, deterministic queue.
    """
    state = _new_state(seed=501)
    start_pos = (state.actors[WHITE].x, state.actors[WHITE].y)

    client = MockModelClient(
        response=[
            {"action": "move", "target_id": "desk_01"},
            {"action": "inspect", "target_id": "desk_01"},
            {"action": "interact", "target_id": "desk_01"},
            {"action": "move", "target_id": "door_corridor_main_office"},
        ]
    )
    telemetry = TelemetryLogger()
    controller = LLMController(WHITE, model_client=client, telemetry=telemetry, execution_mode="evaluation")
    controller._decision_interval_seconds = 0.02  # speed only; semantics unchanged

    ticks = 0
    for _ in range(4000):
        _drive_tick(state, controller, WHITE)
        ticks += 1
        if state.current_interaction_target.get(WHITE) == "door_corridor_main_office" and not controller._executor.in_progress:
            break
    else:
        raise AssertionError("door was never reached within the tick budget")

    # Real, multi-tick simulation - not a one-shot teleport.
    assert ticks > 60

    # Real position change (spec section 16's "real position changes").
    end_pos = (state.actors[WHITE].x, state.actors[WHITE].y)
    moved = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
    assert moved > 150.0

    # Authoritative interaction results, not model-claimed ones.
    own_events = [e["type"] for e in state.event_log if e.get("actor_id") == WHITE]
    assert "OBJECT_INSPECTED" in own_events
    assert "OBJECT_SEARCHED" in own_events

    # The root-cause regression guard: MOVE must not be re-decided on
    # almost every tick. Exactly two MOVE decisions were ever requested
    # (desk, then door), regardless of how many ticks each one spanned.
    move_decisions = [
        e
        for e in telemetry.events
        if e.get("event_type") == "llm_decision" and (e.get("validated_action") or {}).get("action") == "move"
    ]
    assert len(move_decisions) == 2

    inspect_decisions = [
        e
        for e in telemetry.events
        if e.get("event_type") == "llm_decision" and (e.get("validated_action") or {}).get("action") == "inspect"
    ]
    interact_decisions = [
        e
        for e in telemetry.events
        if e.get("event_type") == "llm_decision" and (e.get("validated_action") or {}).get("action") == "interact"
    ]
    assert len(inspect_decisions) == 1
    assert len(interact_decisions) == 1

    # Well below the ~1-decision-per-tick the pre-fix bug produced over a
    # comparable distance (M5D.1's diagnostic: 5 decisions / ~123px moved).
    assert controller.decisions_requested <= 8

    # decision_id -> action_id -> outcome correlation (spec section 12):
    # every MOVE decision has a matching action_lifecycle boundary event
    # sharing its decision_id.
    lifecycle_events = [e for e in telemetry.events if e.get("event_type") == "action_lifecycle"]
    move_decision_ids = {e["decision_id"] for e in move_decisions}
    lifecycle_decision_ids = {e["decision_id"] for e in lifecycle_events}
    assert move_decision_ids <= lifecycle_decision_ids
    move_lifecycle = [e for e in lifecycle_events if e["action"] == "move"]
    assert all(e["action_terminal_reason"] == "target_reached" for e in move_lifecycle)
    assert all(e["action_simulation_ticks"] > 1 for e in move_lifecycle)

    assert controller.actions_failed == 0


# ---------------------------------------------------------------------------
# Section 17: MOVE spans many ticks as one decision; no duplicate requests
# ---------------------------------------------------------------------------


def test_no_duplicate_decision_requests_while_move_progresses():
    state = _new_state(seed=502)
    client = MockModelClient(response=[{"action": "move", "target_id": "desk_01"}])
    telemetry = TelemetryLogger()
    controller = LLMController(WHITE, model_client=client, telemetry=telemetry, execution_mode="evaluation")
    controller._decision_interval_seconds = 0.02

    ticks_in_progress = 0
    requested_snapshot = None
    for _ in range(2000):
        _drive_tick(state, controller, WHITE)
        if controller._executor.in_progress:
            ticks_in_progress += 1
            if requested_snapshot is None:
                requested_snapshot = controller.decisions_requested
            else:
                assert controller.decisions_requested == requested_snapshot, (
                    "a new strategic decision was requested while an accepted MOVE "
                    "was still executing - this is the exact M5D.1 root cause"
                )
        if state.current_interaction_target.get(WHITE) == "desk_01":
            break
    else:
        raise AssertionError("desk was never reached within the tick budget")

    # Target reached before any subsequent normal decision (spec STOP
    # condition 7): still exactly one decision, right after arrival.
    assert ticks_in_progress > 20
    assert controller.decisions_requested == 1
    assert controller.decisions_completed == 1
    move_decisions = [
        e
        for e in telemetry.events
        if e.get("event_type") == "llm_decision" and (e.get("validated_action") or {}).get("action") == "move"
    ]
    assert len(move_decisions) == 1


# ---------------------------------------------------------------------------
# PATH_UNREACHABLE: a legitimate execution failure, distinct from an
# invalid model decision (spec section 13)
# ---------------------------------------------------------------------------


def test_path_unreachable_move_is_a_legitimate_execution_failure():
    executor = ActionExecutor()
    observation = _make_observation(objects=(_obj("far_desk"),), nav_grid=_UnreachableNavGrid())

    action, boundary, action_id = executor.start(observation, "dec-1", "move", "far_desk", None, 90.0)

    assert boundary == PATH_UNREACHABLE
    assert action.type == ActionType.WAIT
    assert executor.in_progress is False
    assert executor.actions_accepted == 1
    assert executor.actions_started == 0  # a MOVE that never actually started

    memory = AgentMemory(WHITE)
    telemetry = TelemetryLogger()
    executor.finish(
        memory=memory,
        telemetry=telemetry,
        episode_id="ep1",
        actor_id=WHITE,
        agent_runtime="native",
        agent_runtime_version="native-v1",
        observation=observation,
        boundary=boundary,
    )

    assert executor.actions_failed == 1
    assert len(memory.recent_failures) == 1
    assert memory.recent_failures[0].reason == "unreachable"

    event = telemetry.events[0]
    assert event["action_status"] == "failed"
    assert event["action_terminal_reason"] == "path_unreachable"
    assert event["decision_id"] == "dec-1"
    assert event["action_id"] == action_id


# ---------------------------------------------------------------------------
# Section 8: the three distinct WAIT semantics
# ---------------------------------------------------------------------------


def test_explicit_wait_is_an_accepted_action_with_a_lifecycle_boundary():
    client = MockModelClient({"action": "wait", "reason": "watching the corridor"})
    telemetry = TelemetryLogger()
    controller = LLMController(WHITE, model_client=client, telemetry=telemetry, execution_mode="evaluation")
    observation = _make_observation()

    controller.get_action(observation)
    controller._thinking_thread.join(timeout=2.0)

    assert controller.decisions_completed == 1
    decision_events = [e for e in telemetry.events if e["event_type"] == "llm_decision"]
    assert decision_events[-1]["action_source"] == "llm"
    assert decision_events[-1]["final_model_action_valid"] is True

    lifecycle_events = [e for e in telemetry.events if e["event_type"] == "action_lifecycle"]
    assert len(lifecycle_events) == 1
    assert lifecycle_events[0]["action_terminal_reason"] == "explicit_wait"
    assert controller.actions_accepted == 1


def test_fallback_wait_never_enters_the_action_lifecycle():
    # Both the initial attempt and the one bounded repair attempt return an
    # unrecognized action, so this can only resolve as a fallback.
    client = MockModelClient(response=[{"action": "not_a_real_action"}, {"action": "still_not_real"}])
    telemetry = TelemetryLogger()
    controller = LLMController(WHITE, model_client=client, telemetry=telemetry, execution_mode="evaluation")
    observation = _make_observation()

    controller.get_action(observation)
    controller._thinking_thread.join(timeout=2.0)

    assert controller.last_action.type == ActionType.WAIT
    decision_events = [e for e in telemetry.events if e["event_type"] == "llm_decision"]
    assert decision_events[-1]["action_source"] == "fallback"
    assert decision_events[-1]["final_model_action_valid"] is False

    # Spec section 8: fallback WAIT is safe execution after an invalid
    # model decision - it never becomes an "accepted action", so it never
    # produces an action_lifecycle boundary the way an explicit WAIT does.
    lifecycle_events = [e for e in telemetry.events if e["event_type"] == "action_lifecycle"]
    assert lifecycle_events == []
    assert controller.actions_accepted == 0


def test_inference_pending_wait_is_not_a_new_strategic_decision():
    slow_client = SlowModelClient({"action": "inspect", "target_id": "desk_01"}, delay=0.3)
    controller = LLMController(WHITE, model_client=slow_client, execution_mode="evaluation")
    observation = _make_observation(objects=(_obj("desk_01"),))

    action = controller.get_action(observation)
    assert action.type == ActionType.WAIT
    assert controller.decisions_requested == 1
    assert controller.status == "THINKING"

    # Polling again mid-flight with the same observation must not spawn a
    # second strategic decision while the first is still in flight.
    for _ in range(3):
        action_again = controller.get_action(observation)
        assert action_again.type == ActionType.WAIT
        assert controller.decisions_requested == 1

    controller._thinking_thread.join(timeout=2.0)
    assert controller.decisions_completed == 1


# ---------------------------------------------------------------------------
# Hidden-truth leakage: the lifecycle telemetry schema is fixed and small
# ---------------------------------------------------------------------------


def test_action_lifecycle_telemetry_has_no_hidden_truth_leakage():
    telemetry = TelemetryLogger()
    telemetry.emit_action_lifecycle(
        episode_id="ep1",
        actor_id=WHITE,
        decision_id="d1",
        action_id="a1",
        action="move",
        target_id="desk_01",
        action_status="completed",
        action_terminal_reason="target_reached",
        action_started_game_time=10.0,
        action_completed_game_time=12.0,
        action_simulation_ticks=42,
        agent_runtime="native",
        agent_runtime_version="native-v1",
    )
    event = telemetry.events[0]
    allowed_keys = {
        "event_type",
        "episode_id",
        "actor_id",
        "decision_id",
        "action_id",
        "action",
        "target_id",
        "action_status",
        "action_terminal_reason",
        "action_started_game_time",
        "action_completed_game_time",
        "action_duration_game_time",
        "action_simulation_ticks",
        "agent_runtime",
        "agent_runtime_version",
    }
    assert set(event.keys()) == allowed_keys
    assert "position" not in event
    assert "opponent" not in event
    assert "observation" not in event


# ---------------------------------------------------------------------------
# Wall latency vs. game time (Game._step's should_pause_time - pre-existing,
# unchanged by M5D.2, locked in here as a documented property of the fix)
# ---------------------------------------------------------------------------


def test_model_wall_latency_does_not_inflate_mission_time():
    from src.game import Game

    game = Game(seed=601, white_controller="llm", llm_provider="mock", llm_model="mock-model", llm_execution_mode="evaluation")
    game.bot_controller.controller.model_client = SlowModelClient({"action": "wait"}, delay=0.4)

    remaining_before = game.state.mission_time_remaining
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        # Real gameplay caps ticks to constants.FPS via Game.run()'s own
        # pygame Clock - an uncapped busy loop would simulate far more game
        # time per real second than actual play ever does, independent of
        # this test's own question. Mirror that real cap here so "did wall
        # latency inflate game time" isn't confounded by "did this test's
        # loop tick faster than a real frame rate."
        dt = game.clock.tick(constants.FPS) / 1000.0
        game._step(dt)

    remaining_after = game.state.mission_time_remaining
    # ~0.5s of real wall time passed while the model "thought", but
    # effective_dt is 0 for the whole window - the mission clock must not
    # have depleted by anywhere near that much.
    assert remaining_before - remaining_after < 0.2


# ---------------------------------------------------------------------------
# Decision cap: stops new decisions, lets the final accepted MOVE finish
# ---------------------------------------------------------------------------


def test_decision_cap_lets_the_final_accepted_move_reach_its_target():
    """Mirrors EpisodeRunner's own cap loop shape (spec section 15): check
    the completed-decision count *before* stepping, same as
    src/evaluation/episode_runner.py. A small model delay (matching M5D's
    own _SlowMockClient regression pattern - see
    test_decision_limit_counts_completed_decisions_not_decision_attempts in
    test_evaluation_runner.py) keeps a decision from completing within the
    exact same tick it was requested, so the cap check between ticks is
    meaningful rather than racing a same-tick fall-through.
    """
    from src.game import Game

    game = Game(seed=603, white_controller="llm", llm_provider="mock", llm_model="mock-model", llm_execution_mode="evaluation")
    telemetry = TelemetryLogger()
    game.bot_controller.controller.telemetry = telemetry
    # A delay well above the internal thread.join(timeout=0.05) poll means
    # a fall-through decision requested the instant decision 1's MOVE
    # finishes cannot itself complete within that same tick - so the cap
    # check between ticks below has a real chance to catch it before it's
    # counted, exactly like a real (non-instant) model would behave.
    game.bot_controller.controller.model_client = SlowModelClient({"action": "move", "target_id": "desk_01"}, delay=0.2)
    game.bot_controller.controller._decision_interval_seconds = 0.02

    max_decisions = 1
    ticks = 0
    for _ in range(4000):
        decisions_made = sum(1 for e in telemetry.events if e.get("event_type") == "llm_decision")
        if decisions_made >= max_decisions and not game.bot_controller.controller._executor.in_progress:
            break
        game._step(1 / 30)
        ticks += 1
    else:
        raise AssertionError("episode never settled within the tick budget")

    # Spec section 15: the cap stops *new* decisions, not the in-progress
    # action's own ticks - the single accepted MOVE must still have reached
    # its target, never cut off mid-path.
    assert game.state.current_interaction_target.get(WHITE) == "desk_01"
    assert sum(1 for e in telemetry.events if e.get("event_type") == "llm_decision") == 1
    assert ticks > 20
