"""Bugfix live flow, Part A: LIVE_ASYNC decision timing (A1-A10, §52).

Real engine, real controllers (Native LLMController and StrandsAgentAdapter);
only the model is replaced by a gated stand-in whose answer is released by
the test, so "the model is still thinking" is an exact, controllable state.
"""
from __future__ import annotations

import json
import os
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.actions import ActionType
from src.agents import live_async
from src.evaluation.navigation_audit import fresh_state, run_move
from src.llm.clients.base import ModelResponse
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController, LLMControllerConfig
from src.observation import build_observation

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID
DT = 1 / 60

# -- a model whose answer the test releases ---------------------------------


def _move_targets(payload: dict) -> list[str]:
    compact = payload.get("legal_actions_compact")
    if isinstance(compact, dict):
        return [t for t in compact.get("move", []) if t]
    return [a["target_id"] for a in payload.get("legal_actions") or [] if a.get("action") == "move" and a.get("target_id")]


class Gate:
    """Blocks each model call until released; `answer(payload)` builds the proposal."""

    def __init__(self, answer, auto: bool = False) -> None:
        self.answer = answer
        self.release = threading.Event()
        if auto:
            self.release.set()
        self.calls = 0
        self.payloads: list[dict] = []

    def propose(self, payload: dict) -> dict:
        self.calls += 1
        self.payloads.append(payload)
        assert self.release.wait(10), "test never released the model"
        return self.answer(payload)


class GatedClient:
    model_name = "gated"
    provider = "mock"

    def __init__(self, gate: Gate) -> None:
        self.gate = gate

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        proposal = self.gate.propose(json.loads(user_prompt))
        return ModelResponse(payload=proposal, raw_text=json.dumps(proposal), provider="mock", model="gated", latency_ms=1.0)


def native_agent(actor_id: str, gate: Gate, telemetry: TelemetryLogger | None = None, mode: str = live_async.LIVE_ASYNC):
    return LLMController(actor_id, model_client=GatedClient(gate), telemetry=telemetry, execution_mode=mode,
                         config=LLMControllerConfig(ready_response_guard=True))


def strands_agent(actor_id: str, gate: Gate, telemetry: TelemetryLogger | None = None, mode: str = live_async.LIVE_ASYNC):
    pytest.importorskip("strands")
    from src.agents.strands_agent import StrandsAgentAdapter, _StrandsInvocation

    adapter = StrandsAgentAdapter(actor_id, agent_factory=lambda *a, **k: None, telemetry=telemetry,
                                  execution_mode=mode, ready_response_guard=True)

    def invoke(model_payload, repair_instruction):
        proposal = gate.propose(json.loads(json.dumps(model_payload)))
        return _StrandsInvocation(proposal=proposal, proposal_count=1, stop_reason="end_turn", cycle_count=1,
                                  tool_call_count=1, input_tokens=1, output_tokens=1, latency_ms=1.0,
                                  error_type=None, error_message=None)

    adapter._invoke_strands = invoke
    return adapter


RUNTIMES = {"native": native_agent, "strands": strands_agent}


def wait_ready(agent, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and (getattr(agent, "_thinking_in_progress", False)
                                           or not getattr(agent, "_response_ready_uncommitted", False)):
        time.sleep(0.005)
    assert agent._response_ready_uncommitted and not agent._thinking_in_progress


def commits(telemetry: TelemetryLogger) -> list[dict]:
    return [e for e in telemetry.events if e.get("event_type") == "live_async_commit"]


# -- fixtures: a state where White stands at a piece of furniture -------------


def _furniture_in_room(state, actor_id=WHITE):
    room = build_observation(state, actor_id).current_room
    return next(o for o in state.level.interactables()
                if o.type != "door" and state.level.room_id_at(*o.center) == room)


@pytest.fixture
def at_furniture():
    """White next to a piece of furniture in its start room (engine near object)."""
    state = fresh_state()
    a = state.actors[WHITE]
    start = (a.x, a.y)
    obj = _furniture_in_room(state)
    row = run_move(state, obj.id, start, DT)
    assert row["near_object_id"] == obj.id, row
    return state, obj.id, start


# -- A4 / A5 / A6 / A9 / A10: stale classification at commit ------------------


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a4_stale_but_still_legal_is_applied(runtime, at_furniture):
    state, obj_id, _ = at_furniture
    telemetry = TelemetryLogger()
    gate = Gate(lambda p: {"action": "inspect", "target_id": obj_id, "reason": "look"})
    agent = RUNTIMES[runtime](WHITE, gate, telemetry)
    first = agent.get_action(build_observation(state, WHITE))
    assert first.type is ActionType.WAIT and agent._thinking_in_progress  # thinking: stands still
    # the world changes while the model thinks, the inspect stays legal
    other = state.actors[BLACK]
    other.x, other.y = state.actors[WHITE].x + 90, state.actors[WHITE].y
    state.update(0.3)
    gate.release.set()
    wait_ready(agent)
    action = agent.get_action(build_observation(state, WHITE))
    assert action.type is ActionType.INSPECT and action.target_id == obj_id
    [event] = commits(telemetry)
    assert event["stale_classification"] == live_async.STALE_BUT_STILL_LEGAL and not event["discarded_due_to_stale"]
    assert agent.live_async_stats.stale_but_still_legal == 1 and agent.last_call_committed_ready_response


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a5_stale_and_illegal_is_discarded_and_decided_again(runtime, at_furniture):
    state, obj_id, start = at_furniture
    telemetry = TelemetryLogger()
    gate = Gate(lambda p: {"action": "inspect", "target_id": obj_id, "reason": "look"})
    agent = RUNTIMES[runtime](WHITE, gate, telemetry)
    agent.get_action(build_observation(state, WHITE))
    assert agent.decisions_requested == 1
    # while the model thinks the spy is pushed back to where it started: out of range
    a = state.actors[WHITE]
    a.x, a.y = start
    state.update(0.0)
    obs = build_observation(state, WHITE)
    assert obs.nearby_object_id != obj_id
    gate.release.set()
    wait_ready(agent)
    action = agent.get_action(obs)
    # not applied: the engine never sees the inspect (no remote inspection)
    assert action.type is ActionType.WAIT and not agent._executor.in_progress
    state.apply_action(WHITE, action, DT)
    assert not any(e["type"] == "OBJECT_INSPECTED" for e in state.event_log)
    [event] = commits(telemetry)
    assert event["stale_classification"] == live_async.STALE_AND_ILLEGAL and event["discarded_due_to_stale"]
    assert event["stale_reason"] == f"inspect_target_{obj_id}_no_longer_legal"
    assert agent.live_async_stats.discarded_due_to_stale == 1
    assert not agent._response_ready_uncommitted and not agent.last_call_committed_ready_response
    # the discarded answer is consumed (HP09) and the very next frame asks again
    gate.release.clear()
    agent.get_action(obs)
    assert agent.decisions_requested == 2 and agent._thinking_in_progress
    gate.release.set()
    wait_ready(agent)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a6_no_duplicate_requests_while_in_flight_or_ready(runtime):
    state = fresh_state()
    gate = Gate(lambda p: {"action": "move", "target_id": _move_targets(p)[-1], "reason": "go"})
    agent = RUNTIMES[runtime](WHITE, gate)
    other = state.actors[BLACK]
    for i in range(30):  # the world keeps changing every frame
        other.x, other.y = state.actors[WHITE].x + 60 + i, state.actors[WHITE].y
        state.update(DT)
        agent.get_action(build_observation(state, WHITE))
    assert agent.decisions_requested == 1 and gate.calls == 1
    gate.release.set()
    wait_ready(agent)
    assert agent.decisions_requested == 1  # ready, not yet committed: still one
    action = agent.get_action(build_observation(state, WHITE))  # the commit
    assert action.type is ActionType.MOVE and agent.decisions_requested == 1 and agent._executor.in_progress


def test_a4_fresh_when_nothing_changed():
    state = fresh_state()
    telemetry = TelemetryLogger()
    gate = Gate(lambda p: {"action": "wait", "reason": "hold"})
    agent = native_agent(WHITE, gate, telemetry)
    obs = build_observation(state, WHITE)
    agent.get_action(obs)
    gate.release.set()
    wait_ready(agent)
    agent.get_action(obs)
    assert commits(telemetry)[0]["stale_classification"] == live_async.FRESH


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a4_a5_telemetry_fields(runtime, at_furniture):
    state, obj_id, _ = at_furniture
    telemetry = TelemetryLogger()
    gate = Gate(lambda p: {"action": "inspect", "target_id": obj_id, "reason": "look"}, auto=True)
    agent = RUNTIMES[runtime](WHITE, gate, telemetry)
    obs = build_observation(state, WHITE)
    agent.get_action(obs)
    if not commits(telemetry):  # not answered within the launch call's short join
        wait_ready(agent)
        agent.get_action(obs)
    [event] = commits(telemetry)
    for key in ("actor_id", "decision_boundary_id", "snapshot_id", "request_id", "request_start_game_time",
                "request_ready_game_time", "commit_game_time", "world_ticks_during_request",
                "actor_moved_during_opponent_request", "opponent_moved_during_request",
                "stale_classification", "stale_reason", "discarded_due_to_stale"):
        assert key in event, key
    assert event["decision_boundary_id"] == f"{WHITE}:b1" and event["request_id"] == event["decision_id"]


def test_a9_a10_native_and_strands_classify_identically(at_furniture):
    pytest.importorskip("strands")
    results = {}
    for runtime, build in RUNTIMES.items():
        state, obj_id, start = at_furniture
        state = fresh_state()
        a = state.actors[WHITE]
        run_move(state, obj_id, (a.x, a.y), DT)
        telemetry = TelemetryLogger()
        gate = Gate(lambda p: {"action": "inspect", "target_id": obj_id, "reason": "look"})
        agent = build(WHITE, gate, telemetry)
        agent.get_action(build_observation(state, WHITE))
        a.x, a.y = start
        state.update(0.0)
        gate.release.set()
        wait_ready(agent)
        action = agent.get_action(build_observation(state, WHITE))
        [event] = commits(telemetry)
        results[runtime] = (action.type, event["stale_classification"], event["stale_reason"],
                            sorted(k for k in event if k not in ("agent_runtime",)))
    assert results["native"] == results["strands"]


# -- A1 / A2 / A3 / A7 / A8: the game loop ------------------------------------


def _game(mode: str, white_gate: Gate, black_gate: Gate):
    from src.game import Game

    game = Game(seed=7, white_controller="llm", black_controller="llm", llm_provider="mock", llm_model="m",
                llm_execution_mode=mode, agent_runtime="native_tool_like")
    game.telemetry = TelemetryLogger()
    for agent, gate in ((game.bot_controller, white_gate), (game.black_llm_controller, black_gate)):
        agent.controller.model_client = GatedClient(gate)
        agent.controller.telemetry = game.telemetry
    return game


def _far_move(payload: dict) -> dict:
    targets = _move_targets(payload)
    door = [t for t in targets if t.startswith("door")] or targets
    return {"action": "move", "target_id": door[-1], "reason": "go"}


def _pos(game, actor_id):
    a = game.state.actors[actor_id]
    return (round(a.x, 2), round(a.y, 2))


def _steps(game, n):
    for _ in range(n):
        game._step(DT)


def _release_and_commit(game, gate, agent, max_frames=120):
    gate.release.set()
    for _ in range(max_frames):
        game._step(DT)
        if agent.live_async_stats.commits:
            return
        time.sleep(0.002)
    raise AssertionError("no commit")


@pytest.mark.parametrize("thinker", [BLACK, WHITE])
def test_a1_a2_one_spy_thinks_while_the_other_keeps_moving(thinker):
    mover = WHITE if thinker == BLACK else BLACK
    gates = {thinker: Gate(_far_move), mover: Gate(_far_move)}
    game = _game(live_async.LIVE_ASYNC, gates[WHITE], gates[BLACK])
    agents = {WHITE: game.bot_controller, BLACK: game.black_llm_controller}
    _steps(game, 2)  # both requests launched
    _release_and_commit(game, gates[mover], agents[mover])
    thinker_pos, mover_pos, clock = _pos(game, thinker), _pos(game, mover), game.state.mission_time_remaining
    _steps(game, 30)
    assert agents[thinker]._thinking_in_progress
    assert _pos(game, thinker) == thinker_pos  # the thinking spy stands still
    assert _pos(game, mover) != mover_pos  # the other one keeps walking
    assert game.state.mission_time_remaining < clock - 0.4  # and the game clock runs
    stats = agents[thinker].live_async_stats
    assert stats.world_ticks_during_request >= 30 and stats.opponent_move_ticks_during_request > 0
    gates[thinker].release.set()
    _release_and_commit(game, gates[thinker], agents[thinker])
    [event] = [e for e in game.telemetry.events if e.get("event_type") == "live_async_commit" and e["actor_id"] == thinker]
    assert event["opponent_moved_during_request"] and event["world_ticks_during_request"] >= 30


def test_a3_both_think_independently_and_commit_on_their_own():
    gates = {WHITE: Gate(_far_move), BLACK: Gate(_far_move)}
    game = _game(live_async.LIVE_ASYNC, gates[WHITE], gates[BLACK])
    white, black = game.bot_controller, game.black_llm_controller
    _steps(game, 2)
    assert white._thinking_in_progress and black._thinking_in_progress
    clock = game.state.mission_time_remaining
    _steps(game, 20)
    assert game.state.mission_time_remaining < clock - 0.25  # both think, nothing is frozen by design
    assert not game._barrier_active() and not game._held_decisions
    _release_and_commit(game, gates[WHITE], white)
    assert black._thinking_in_progress and black.live_async_stats.commits == 0  # no pair barrier
    white_pos = _pos(game, WHITE)
    _steps(game, 20)
    assert _pos(game, WHITE) != white_pos
    _release_and_commit(game, gates[BLACK], black)
    assert white.decisions_requested >= 1 and black.decisions_requested == 1


def test_a7_game_clock_advances_during_live_inference():
    gates = {WHITE: Gate(_far_move), BLACK: Gate(_far_move)}
    game = _game(live_async.LIVE_ASYNC, gates[WHITE], gates[BLACK])
    _steps(game, 2)
    clock = game.state.mission_time_remaining
    _steps(game, 60)
    assert game.bot_controller._thinking_in_progress and game.black_llm_controller._thinking_in_progress
    assert clock - game.state.mission_time_remaining == pytest.approx(60 * DT, abs=1e-6)
    for gate in gates.values():
        gate.release.set()


def test_a8_evaluation_mode_still_freezes_exactly_as_before():
    gates = {WHITE: Gate(_far_move), BLACK: Gate(_far_move)}
    game = _game("evaluation", gates[WHITE], gates[BLACK])
    white, black = game.bot_controller, game.black_llm_controller
    assert game._barrier_active() and white.execution_mode == "evaluation"
    _steps(game, 2)
    clock, positions = game.state.mission_time_remaining, (_pos(game, WHITE), _pos(game, BLACK))
    _steps(game, 60)
    assert game.state.mission_time_remaining == clock  # frozen while a model thinks
    assert (_pos(game, WHITE), _pos(game, BLACK)) == positions
    gates[WHITE].release.set()
    wait_ready(white)
    _steps(game, 10)  # White is ready, Black still thinking: the barrier holds White, the clock stays frozen
    assert game.state.mission_time_remaining == clock and WHITE in game._held_decisions
    gates[BLACK].release.set()
    wait_ready(black)
    _steps(game, 3)
    assert game.last_barrier_release is not None and game.state.mission_time_remaining < clock
    # evaluation mode never goes through the live path
    assert white.live_async_stats.requests == black.live_async_stats.requests == 0
    assert not [e for e in game.telemetry.events if e.get("event_type") == "live_async_commit"]


def test_live_async_is_opt_in():
    from src.game import Game

    game = Game(seed=7, white_controller="llm", black_controller="llm", llm_provider="mock", llm_model="m",
                agent_runtime="native_tool_like")
    assert game.llm_execution_mode == "evaluation" and game._barrier_active()
    assert not game.face_interaction_targets and not game.bot_controller._executor.close_approach
