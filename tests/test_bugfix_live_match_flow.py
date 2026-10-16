"""Bugfix live match flow V3 (Archive/BUGFIX_LIVE_MATCH_FLOW_EXIT_SPLITSCREEN_LIVES_V3.md).

RT  live wall-clock timer keeps running while models think
TR  trap ownership: own traps dangerous, own-trap memory, no self-reversal
SS  split screen: two cameras, one world
EX  the exit is a real doorway out of the building
END explicit terminal lifecycle: no requests after a spy is finished
L   three lives / trap hits
SC  M4E score + final ranking

All of it is live-only (LiveGame flags); the recorded rules are checked to
stay as they were.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time

import pygame
import pytest

from src import constants, exit_geometry
from src.actions import ActionType, GameAction
from src.agents import live_async
from src.agents.action_lifecycle import TARGET_REACHED
from src.evaluation.navigation_audit import fresh_state
from src.level import load_level
from src.llm.personality import render_live_mission_block
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import enumerate_legal_actions, serialize_observation
from src.observation import build_observation
from src.ranking import rank_match
from src.scoring import compute_score
from tests.test_bugfix_inspection_proximity import _trappable_in_start_room, act, events, move_to
from tests.test_bugfix_live_async import Gate, GatedClient
from tests.workshop_live_helpers import LegalMoveClient
from workshop.live.config import resolve_config
from workshop.live.overlay import THEME, THEME_TEXT_TOKENS, PANEL_ALPHA, TelemetryView, blend, contrast_ratio
from workshop.live.session import LiveGame, MatchRunner

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID
DT = 1 / 60
BASE = {"provider": "bedrock", "model": "m", "personality_preset": "rogue-vs-cautious", "encounters": "off",
        "spawn": "same_room"}  # fixtures: no fights, both spies start in the level room


class AlwaysSucceeds:
    """disarm_rng stand-in: every disarm roll succeeds."""

    def random(self) -> float:
        return 0.0


class FakeWall:
    """A controllable monotonic clock for the live wall-clock timer."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def live_game(gates: dict | None = None, telemetry=None, wall=None, **extra):
    gates = gates or {}

    def factory(spec, cfg):
        gate = gates.get(spec.actor_id)
        return GatedClient(gate) if gate is not None else LegalMoveClient(spec)

    return LiveGame(resolve_config({**BASE, **extra}), telemetry, client_factory=factory, wall_time_fn=wall)


def agents(game):
    return {BLACK: game.black_llm_controller, WHITE: game.bot_controller}


def calls(game):
    return {a: getattr(ag, "decisions_requested", 0) for a, ag in agents(game).items()}


def frames(game, wall, n, wall_step=0.1, dt=DT):
    for _ in range(n):
        wall.t += wall_step
        game._step(dt)


def settle(agent, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and agent._thinking_in_progress:
        time.sleep(0.005)


# RT ---------------------------------------------------------------------------

@pytest.mark.parametrize("thinkers", [(BLACK,), (WHITE,), (BLACK, WHITE)], ids=["RT1_black", "RT2_white", "RT3_both"])
def test_rt1_rt3_timer_decreases_while_thinking(thinkers):
    wall = FakeWall()
    gates = {a: Gate(lambda p: {"action": "wait", "reason": "x"}) for a in thinkers}
    game = live_game(gates, wall=wall)
    assert game.state.external_mission_clock
    frames(game, wall, 3)
    start = game.state.mission_time_remaining
    frames(game, wall, 50)  # 5 s of wall time
    for a in thinkers:
        assert agents(game)[a]._thinking_in_progress  # still waiting for its model
    assert game.state.mission_time_remaining == pytest.approx(start - 5.0, abs=1e-6)
    for gate in gates.values():
        gate.release.set()
    for a in thinkers:
        settle(agents(game)[a])


def test_rt4_rt5_pause_pauses_and_resume_does_not_jump():
    wall = FakeWall()
    game = live_game(wall=wall)
    frames(game, wall, 2)
    start = game.state.mission_time_remaining
    for _ in range(100):  # 10 s paused (the runner's paused frames)
        wall.t += 0.1
        game.pause_clock()
    assert game.state.mission_time_remaining == start
    frames(game, wall, 1)
    assert game.state.mission_time_remaining == pytest.approx(start - 0.1)


def test_rt_renderer_lag_does_not_distort_the_countdown():
    wall = FakeWall()
    game = live_game(wall=wall)
    frames(game, wall, 1)
    start = game.state.mission_time_remaining
    frames(game, wall, 1, wall_step=0.5, dt=0.1)  # a 0.5 s frame, world dt capped at 0.1
    assert game.state.mission_time_remaining == pytest.approx(start - 0.5)


def test_rt6_timeout_during_request_discards_the_response():
    wall = FakeWall()
    telemetry = TelemetryLogger(None)
    gate = Gate(lambda p: {"action": "wait", "reason": "late"})
    game = live_game({BLACK: gate}, telemetry=telemetry, wall=wall)
    frames(game, wall, 3)
    black = game.black_llm_controller
    assert black._thinking_in_progress
    game.state.mission_time_remaining = 0.2
    frames(game, wall, 3)  # the deadline passes while Black thinks
    assert game.state.lifecycle(BLACK) == constants.LIFECYCLE_TIMED_OUT
    gate.release.set()
    settle(black)
    frames(game, wall, 5)
    commit = [e for e in telemetry.events if e.get("event_type") == "live_async_commit" and e["actor_id"] == BLACK][-1]
    assert commit["stale_classification"] == live_async.DISCARDED_TERMINAL and commit["discarded_due_to_stale"]
    assert commit["time_remaining_at_request"] is not None and commit["time_remaining_at_commit"] == 0.0
    assert gate.calls == 1  # no new request
    assert black.live_async_stats.discarded_terminal == 1


def test_rt7_evaluation_timing_is_unchanged():
    state = fresh_state()
    assert not state.external_mission_clock and not state.live_lifecycle and not state.own_traps_dangerous
    before = state.mission_time_remaining
    state.update(1.0)
    assert state.mission_time_remaining == before - 1.0
    assert resolve_config(BASE).mission_timer == "wall_clock"
    assert not live_game(mission_timer="game_time").state.external_mission_clock
    assert not live_game(mission_clock="action_budget").state.external_mission_clock


# TR ---------------------------------------------------------------------------

@pytest.fixture
def live():
    return live_game().state


def _walk_away(state, obj, actor_id=WHITE):
    door = next(o for o in state.level.objects if o.type == "door" and o.connects
                and build_observation(state, actor_id).current_room in o.connects)
    move_to(state, door.id, actor_id=actor_id)
    assert state.current_interaction_target.get(actor_id) != obj.id


def test_tr1_owner_is_triggered_by_its_own_armed_trap(live):
    obj = _trappable_in_start_room(live)
    move_to(live, obj.id)
    act(live, ActionType.INSPECT, obj.id)
    live._object_entry(obj.id)["trap"] = None
    act(live, ActionType.PLACE_TRAP, obj.id, trap_type="bomb")  # engine accepts it (legal surface wants search first)
    act(live, ActionType.INTERACT, obj.id)
    triggered = events(live, "TRAP_TRIGGERED", obj.id)
    assert triggered and triggered[-1]["own_trap"] and triggered[-1]["trap_owner_id"] == WHITE
    assert live.lives[WHITE] == 2 and not events(live, "OBJECT_SEARCHED", obj.id)


def _search_then_trap(state, obj, actor_id=WHITE, trap="bomb"):
    move_to(state, obj.id, actor_id=actor_id)
    act(state, ActionType.INSPECT, obj.id, actor_id=actor_id)
    act(state, ActionType.INTERACT, obj.id, actor_id=actor_id)
    act(state, ActionType.PLACE_TRAP, obj.id, actor_id=actor_id, trap_type=trap)
    assert events(state, "TRAP_PLACED", obj.id)


def test_tr2_tr3_owner_remembers_and_sees_its_armed_trap(live):
    obj = _trappable_in_start_room(live)
    _search_then_trap(live, obj)
    obs = build_observation(live, WHITE)
    record = next(r for r in obs.self_placed_traps if r["target_id"] == obj.id)
    assert record["status"] == constants.OWN_TRAP_ARMED and record["trap_type"] == "bomb" and record["room"]
    payload = serialize_observation(obs, live_state=True, live_workshop=True)
    assert payload["self"]["own_traps"][0]["status"] == constants.OWN_TRAP_ARMED
    entry = next(o for o in payload["room"]["objects"] if o["id"] == obj.id)
    assert entry["trap_status"] == constants.OWN_TRAP_ARMED
    # not in the recorded payload
    assert "own_traps" not in serialize_observation(obs)["self"]


def test_tr4_tr10_owner_may_disarm_its_trap_after_leaving_and_coming_back(live):
    obj = _trappable_in_start_room(live)
    _search_then_trap(live, obj)
    _walk_away(live, obj)
    move_to(live, obj.id)
    obs = build_observation(live, WHITE)
    assert any(a["action"] == "disarm" and a["target_id"] == obj.id for a in enumerate_legal_actions(obs, reachable_only=True))
    act(live, ActionType.DISARM, obj.id)
    assert events(live, "TRAP_REMOVED", obj.id) and not events(live, "SELF_REVERSAL_NO_STATE_CHANGE")
    act(live, ActionType.INTERACT, obj.id)  # searching again is safe now
    assert not events(live, "TRAP_TRIGGERED")
    record = build_observation(live, WHITE).self_placed_traps[0]
    assert record["status"] == "REMOVED_BY_YOU" and not record["armed"]


def _enemy_detects(state, obj, actor_id=BLACK):
    black = state.actors[actor_id]
    white = state.actors[WHITE]
    black.x, black.y = white.x, white.y
    state.update(0.0)
    for _ in range(40):
        act(state, ActionType.INSPECT, obj.id, actor_id=actor_id)
        if events(state, "TRAP_DETECTED", obj.id):
            return
    pytest.fail("never detected")


def test_tr5_tr6_tr7_tr8_enemy_trap_detect_disarm_search_retrap(live):
    live.disarm_rng = AlwaysSucceeds()  # disarm odds are tested in test_live_trap_balance.py
    obj = _trappable_in_start_room(live)
    _search_then_trap(live, obj)
    _enemy_detects(live, obj)
    obs = build_observation(live, BLACK)
    legal = enumerate_legal_actions(obs, reachable_only=True)
    assert any(a["action"] == "disarm" and a["target_id"] == obj.id for a in legal)
    act(live, ActionType.DISARM, obj.id, actor_id=BLACK)
    disarmed = events(live, "TRAP_DISARMED", obj.id)[-1]
    assert disarmed["trap_owner_id"] == WHITE and disarmed["armed_before"] and not disarmed["armed_after"]
    act(live, ActionType.INTERACT, obj.id, actor_id=BLACK)
    assert not events(live, "TRAP_TRIGGERED") and any(e["actor_id"] == BLACK for e in events(live, "OBJECT_SEARCHED", obj.id))
    legal = enumerate_legal_actions(build_observation(live, BLACK), reachable_only=True)
    assert any(a["action"] == "place_trap" and a["target_id"] == obj.id for a in legal)
    act(live, ActionType.PLACE_TRAP, obj.id, actor_id=BLACK, trap_type="snare")
    assert events(live, "TRAP_PLACED", obj.id)[-1]["actor_id"] == BLACK
    # White's memory: its trap is gone (it does not learn who removed it)
    assert build_observation(live, WHITE).self_placed_traps[0]["status"] == "GONE"


def test_tr9_immediate_place_then_disarm_is_prevented_and_detected(live):
    obj = _trappable_in_start_room(live)
    move_to(live, obj.id)
    act(live, ActionType.INSPECT, obj.id)
    legal = enumerate_legal_actions(build_observation(live, WHITE), reachable_only=True)
    assert not any(a["action"] == "place_trap" for a in legal)  # search first, then trap
    act(live, ActionType.INTERACT, obj.id)
    act(live, ActionType.PLACE_TRAP, obj.id, trap_type="bomb")
    legal = enumerate_legal_actions(build_observation(live, WHITE), reachable_only=True)
    assert not any(a["action"] == "disarm" for a in legal)  # the pointless reversal is not offered
    act(live, ActionType.DISARM, obj.id)  # forced past the legal surface: the engine flags it
    assert events(live, "SELF_REVERSAL_NO_STATE_CHANGE", obj.id)


def test_tr9_recorded_legal_surface_is_unchanged():
    state = fresh_state()
    obj = _trappable_in_start_room(state)
    move_to(state, obj.id)
    act(state, ActionType.INSPECT, obj.id)
    legal = enumerate_legal_actions(build_observation(state, WHITE), reachable_only=True)
    assert any(a["action"] == "place_trap" for a in legal)
    act(state, ActionType.PLACE_TRAP, obj.id, trap_type="bomb")
    assert "trap_id" not in events(state, "TRAP_PLACED")[-1]


def test_tr11_trap_telemetry_follows_engine_order():
    telemetry = TelemetryLogger(None)
    game = live_game(telemetry=telemetry)
    state = game.state
    obj = _trappable_in_start_room(state)
    _search_then_trap(state, obj)
    _walk_away(state, obj)
    move_to(state, obj.id)
    act(state, ActionType.DISARM, obj.id)
    game._emit_engine_events()
    records = [e for e in telemetry.events if e.get("event_type") == "engine_event"]
    assert [r["engine_seq"] for r in records] == sorted(r["engine_seq"] for r in records)
    kinds = [r["engine_event_type"] for r in records if r["engine_event_type"].startswith("TRAP_")]
    assert kinds == ["TRAP_PLACED", "TRAP_REMOVED"]
    placed = next(r for r in records if r["engine_event_type"] == "TRAP_PLACED")
    assert placed["trap_owner_id"] == WHITE and placed["trap_id"] and placed["armed_after"]


# SS ---------------------------------------------------------------------------

def _rendered(game, screen=None):
    renderer = game.renderer
    if screen is not None:
        from workshop.live.overlay import LiveRenderer
        renderer = LiveRenderer(screen)
        game.renderer = renderer
    renderer.view, renderer.game = TelemetryView(game.live_config), game
    renderer.render(game.state, DT, None, None, game.bot_controller)
    return renderer


def test_ss1_ss2_each_pane_follows_its_spy():
    game = live_game()
    state = game.state
    state.actors[BLACK].x, state.actors[BLACK].y = 700, 960
    state.actors[WHITE].x, state.actors[WHITE].y = 1150, 560
    renderer = _rendered(game)
    w, h = renderer._half_world.get_size()
    for actor_id in (BLACK, WHITE):
        cx, cy = renderer._camera_for(state, actor_id, w, h)
        a = state.actors[actor_id]
        assert (cx + w / 2, cy + h / 2) == pytest.approx((a.x, a.y), abs=1)


def test_ss3_ss4_one_world_rendered_twice_and_not_mutated(monkeypatch):
    game = live_game()
    state = game.state
    seen = []
    original = game.renderer._render_view
    monkeypatch.setattr(game.renderer, "_render_view", lambda s, *a, **k: (seen.append(id(s)), original(s, *a, **k)))
    before = ({a: (x.x, x.y) for a, x in state.actors.items()}, len(state.event_log), state.mission_time_remaining)
    _rendered(game)
    assert seen == [id(state), id(state)]
    assert before == ({a: (x.x, x.y) for a, x in state.actors.items()}, len(state.event_log), state.mission_time_remaining)


@pytest.mark.parametrize("size", [(1680, 720), (1600, 900), (1024, 640)])
def test_ss5_ss6_panes_resize_and_overlays_stay_inside_their_pane(size):
    """Live feedback: the panes share the game view; every text panel is in
    the sidebar on the right, one section per spy, never over the game."""
    game = live_game()
    renderer = _rendered(game, pygame.Surface(size))
    bar, view = renderer.sidebar_rect(), renderer.game_rect()
    assert bar.right == size[0] and view.right == bar.left and view.width + bar.width == size[0]
    for actor_id in (BLACK, WHITE):
        pane = renderer.pane_rects[actor_id]
        assert pane.width == pytest.approx(view.width / 2, abs=1) and view.contains(pane)
        assert bar.contains(renderer.panel_rects[actor_id]) and bar.contains(renderer.header_rects[actor_id])
    assert not renderer.panel_rects[BLACK].colliderect(renderer.panel_rects[WHITE])
    assert live_game().screen.get_width() > constants.WINDOW_WIDTH  # the live window is wider


# EX ---------------------------------------------------------------------------

def test_ex1_exit_door_is_on_the_exterior_boundary():
    raw = load_level(constants.LEVEL_PATH)
    before = exit_geometry.audit(raw)
    assert before["walls_blocking_doorway"] and before["unsealed_perimeter_gaps"]  # the root cause, in the level file
    repaired = live_game().state.level
    audit = exit_geometry.audit(repaired)
    assert audit["on_perimeter"] and audit["inside_room"] == "ambassador_office"
    assert not audit["walls_blocking_doorway"] and not audit["unsealed_perimeter_gaps"]
    point = audit["exterior_point"]
    assert repaired.room_id_at(*point) is None  # one side a room, the other outside


def _in_ambassador_office(state, actor_id=BLACK, items=True):
    a = state.actors[actor_id]
    a.x, a.y = 720, 1380
    if items:
        for item in state.items.required:
            a.inventory.add(item)
    state.update(0.0)


def test_ex2_ex3_ex4_exit_reachable_needs_items_and_then_works():
    game = live_game()
    state = game.state
    _in_ambassador_office(state, items=False)
    boundary, obs = move_to(state, "exit_door", actor_id=BLACK)
    assert boundary == TARGET_REACHED and obs.nearby_object_id == "exit_door"
    act(state, ActionType.INTERACT, "exit_door", actor_id=BLACK)
    assert not state.mission_complete[BLACK] and events(state, "EXIT_ATTEMPTED")
    for item in state.items.required:
        state.actors[BLACK].inventory.add(item)
    assert state.lifecycle(BLACK) == constants.LIFECYCLE_OBJECTIVES_COMPLETE  # not terminal
    act(state, ActionType.INTERACT, "exit_door", actor_id=BLACK)
    assert state.mission_complete[BLACK]


def test_ex5_ex6_exit_walks_through_the_doorway_to_outside():
    game = live_game()
    state = game.state
    _in_ambassador_office(state)
    move_to(state, "exit_door", actor_id=BLACK)
    act(state, ActionType.INTERACT, "exit_door", actor_id=BLACK)
    assert state.lifecycle(BLACK) == constants.LIFECYCLE_ESCAPED and BLACK in state.exit_walking
    assert not game.match_over() or BLACK in state.exit_walking
    for _ in range(240):
        state.update(DT)
    black = state.actors[BLACK]
    door = state.level.get_object("exit_door").collision_rect
    assert state.left_building[BLACK] and black.x > door.right and state.level.room_id_at(black.x, black.y) is None
    assert events(state, "ACTOR_LEFT_BUILDING")
    # the doorway is passable now that the door is open (no wall behind it)
    assert not any(r.colliderect(door) for r in state.level.solid_rects(state.door_states))


def test_ex7_ex8_end1_end5_escaped_spy_stops_thinking_the_other_continues():
    # shared items: the other spy can still find its own copies and goes on.
    # With one copy of each item (the default) the other spy's mission is
    # impossible once Black left with them - tests/test_mission_impossible.py
    wall = FakeWall()
    game = live_game(wall=wall, items="shared")
    state = game.state
    _in_ambassador_office(state)
    state.mission_complete[BLACK] = True
    state.finish_order.append(BLACK)
    frames(game, wall, 3)
    before = calls(game)
    white = state.actors[WHITE]
    start = (white.x, white.y)
    frames(game, wall, 600)
    after = calls(game)
    assert after[BLACK] == before[BLACK]
    assert after[WHITE] >= 1 and (white.x, white.y) != start and not game.match_over()


# END --------------------------------------------------------------------------

def test_end2_no_request_after_timed_out():
    wall = FakeWall()
    game = live_game(wall=wall)
    frames(game, wall, 3)
    game.state.mission_time_remaining = 0.05
    frames(game, wall, 2)
    before = calls(game)
    frames(game, wall, 60)
    assert calls(game) == before
    assert all(game.state.lifecycle(a) == constants.LIFECYCLE_TIMED_OUT for a in (BLACK, WHITE))
    assert build_observation(game.state, WHITE).time_expired and game.match_over()


def _trigger(state, victim):
    obj = _trappable_in_start_room(state)
    entry = state._object_entry(obj.id)
    entry["trap"] = {"type": "snare", "owner_id": BLACK if victim == WHITE else WHITE, "armed": True}
    state.actors[victim].stunned_timer = 0.0
    state._trigger_trap(state.actors[victim], obj, entry, entry["trap"])


def test_end3_l6_no_request_after_eliminated():
    wall = FakeWall()
    game = live_game(wall=wall)
    for _ in range(3):
        _trigger(game.state, WHITE)
    game.state.actors[WHITE].stunned_timer = 0.0
    frames(game, wall, 3)
    before = calls(game)
    frames(game, wall, 60)
    assert calls(game)[WHITE] == before[WHITE] and game.state.lifecycle(WHITE) == constants.LIFECYCLE_ELIMINATED


def test_end4_in_flight_answer_discarded_after_escape():
    wall = FakeWall()
    telemetry = TelemetryLogger(None)
    gate = Gate(lambda p: {"action": "wait", "reason": "late"})
    game = live_game({BLACK: gate}, telemetry=telemetry, wall=wall)
    frames(game, wall, 3)
    assert game.black_llm_controller._thinking_in_progress
    game.state.mission_complete[BLACK] = True
    gate.release.set()
    settle(game.black_llm_controller)
    frames(game, wall, 5)
    commit = [e for e in telemetry.events if e.get("event_type") == "live_async_commit" and e["actor_id"] == BLACK][-1]
    assert commit["stale_classification"] == live_async.DISCARDED_TERMINAL and gate.calls == 1


def test_end6_both_terminal_ends_the_match():
    game = live_game()
    state = game.state
    assert not game.match_over()
    state.mission_complete[BLACK] = True
    assert not game.match_over()
    state.mission_time_remaining = 0.0
    assert game.match_over()
    state.exit_walking[BLACK] = [(0.0, 0.0)]
    assert not game.match_over()  # the walk out finishes first


def test_end7_sc9_no_model_requests_after_the_match_ends(tmp_path):
    class Runner(MatchRunner):
        def new_match(self):
            super().new_match()
            self.game.state.mission_time_remaining = 0.0  # both spies time out at once
            self.game.state.external_mission_clock = False

    runner = Runner(resolve_config(BASE), tmp_path, headless=True, max_wall_seconds=20,
                    client_factory=lambda spec, cfg: LegalMoveClient(spec))
    summary = runner.run()
    assert summary["stop_reason"] == "match_over"
    assert summary["pause"]["requests_launched_while_paused"] == 0
    assert all(a["decisions"] == 0 for a in summary["actors"].values())
    result = summary["match_result"]
    assert [r["rank"] for r in result["ranking"]] == [1, 1] and all(r["tie"] for r in result["ranking"])


# L ----------------------------------------------------------------------------

def test_l1_to_l5_lives_count_down_to_elimination(live):
    assert live.lives == {BLACK: 3, WHITE: 3} and live.trap_hits.get(WHITE, 0) == 0
    for expected in (2, 1, 0):
        _trigger(live, WHITE)
        assert live.lives[WHITE] == expected
    assert live.lifecycle(WHITE) == constants.LIFECYCLE_ELIMINATED
    trig = [e for e in live.event_log if e["type"] == "TRAP_TRIGGERED"]
    assert [(e["lives_before"], e["lives_after"]) for e in trig] == [(3, 2), (2, 1), (1, 0)]
    assert [e["eliminated"] for e in trig] == [False, False, True]
    assert [(e["trap_hits_before"], e["trap_hits_after"]) for e in trig] == [(0, 1), (1, 2), (2, 3)]


def test_l7_l8_detection_and_disarm_cost_no_life(live):
    live.disarm_rng = AlwaysSucceeds()  # disarm odds are tested in test_live_trap_balance.py
    obj = _trappable_in_start_room(live)
    _search_then_trap(live, obj)
    _enemy_detects(live, obj)
    act(live, ActionType.DISARM, obj.id, actor_id=BLACK)
    assert live.lives == {BLACK: 3, WHITE: 3}


def test_l9_own_trap_costs_a_life(live):
    obj = _trappable_in_start_room(live)
    _search_then_trap(live, obj)
    act(live, ActionType.INTERACT, obj.id)  # searching again with its own trap armed
    assert live.lives[WHITE] == 2 and live.trap_hits[WHITE] == 1
    assert build_observation(live, WHITE).self_placed_traps[0]["status"] == "TRIGGERED_ON_YOU"


def test_l10_own_trap_memory_survives_room_changes(live):
    obj = _trappable_in_start_room(live)
    _search_then_trap(live, obj)
    room = live.level.object_room(obj)
    white = live.actors[WHITE]
    white.x, white.y = 720, 1380  # the ambassador office
    live.update(DT)
    obs = build_observation(live, WHITE)
    assert obs.current_room != room
    assert obs.self_placed_traps[0]["target_id"] == obj.id and obs.self_placed_traps[0]["room"] == room
    assert obs.self_placed_traps[0]["status"] == constants.OWN_TRAP_ARMED


def test_l11_ui_counter_matches_engine():
    game = live_game()
    _trigger(game.state, BLACK)
    renderer = _rendered(game)
    snap = renderer.view.actor(game, BLACK)
    assert (snap.lives, snap.trap_hits) == (game.state.lives[BLACK], game.state.trap_hits[BLACK]) == (2, 1)
    lines = [t for t, _ in renderer.view.actor_lines(snap, detail=True)]
    assert "LIVES 2/3   TRAPS HIT 1   FIGHTS LOST 0" in lines


def test_l12_replay_preserves_the_trap_hit_and_life_sequence():
    def run():
        state = live_game(seed=4242).state
        obj = _trappable_in_start_room(state)
        _search_then_trap(state, obj)
        act(state, ActionType.INTERACT, obj.id)
        _trigger(state, WHITE)
        return [(e["type"], e.get("actor_id"), e.get("lives_left")) for e in state.event_log
                if e["type"] in ("TRAP_TRIGGERED", "LIFE_LOST", "ACTOR_ELIMINATED")]

    first = run()
    assert first and first == run()


# SC ---------------------------------------------------------------------------

def _escape(state, actor_id, at_elapsed):
    state.elapsed_time = at_elapsed
    state.mission_time_remaining = state.mission_time_total - at_elapsed
    for item in state.items.required:
        state.actors[actor_id].inventory.add(item)
    state._attempt_exit(state.actors[actor_id], state.level.get_object("exit_door"))


def test_sc1_to_sc4_both_escape_ranked_with_bonuses():
    state = live_game().state
    _escape(state, BLACK, 100.0)
    _escape(state, WHITE, 150.0)
    ranking = rank_match(state)
    assert all(r["mission_success"] for r in ranking)
    assert [r["actor_id"] for r in ranking] == [BLACK, WHITE] and [r["rank"] for r in ranking] == [1, 2]
    first, second = ranking
    assert first["score_first_escape"] == constants.SCORE_ESCAPE_FIRST and second["score_first_escape"] == 0
    assert first["score_time_bonus"] == pytest.approx(200.0 * constants.SCORE_REMAINING_TIME_BONUS_PER_SECOND)
    assert second["score_time_bonus"] == pytest.approx(150.0 * constants.SCORE_REMAINING_TIME_BONUS_PER_SECOND)
    assert first["score_total"] == compute_score(state, BLACK).total and first["escape_order"] == 1


def test_sc5_traps_give_no_points(live):
    before = compute_score(live, WHITE).total
    obj = _trappable_in_start_room(live)
    _search_then_trap(live, obj)
    assert compute_score(live, WHITE).total == before


def test_sc6_sc7_failure_penalty_and_deterministic_tie_break():
    state = live_game().state
    state.mission_time_remaining = 0.0
    ranking = rank_match(state)
    assert all(r["score_failure"] == constants.SCORE_FAILURE_TO_ESCAPE and r["failed"] for r in ranking)
    assert [r["rank"] for r in ranking] == [1, 1] and all(r["tie"] for r in ranking)
    _trigger(state, BLACK)  # same score: Black has fewer lives -> White ranks first
    ranking = rank_match(state)
    assert [r["actor_id"] for r in ranking] == [WHITE, BLACK] and [r["rank"] for r in ranking] == [1, 2]
    assert rank_match(state) == ranking


def test_sc8_final_screen_matches_telemetry():
    telemetry = TelemetryLogger(None)
    game = live_game(telemetry=telemetry)
    _escape(game.state, BLACK, 60.0)
    game.state.mission_time_remaining = 0.0
    before = calls(game)
    result = game.finish_match()
    assert calls(game) == before  # SC9: no model call
    event = next(e for e in telemetry.events if e.get("event_type") == "match_result")
    assert event["ranking"] == result["ranking"] and event["escape_order"] == [BLACK]
    renderer = _rendered(game)
    renderer.match_result = result
    text = "\n".join(t for t, _ in renderer.ranking_lines(result))
    for r in result["ranking"]:
        assert f"{r['rank']}. " in text and f"{r['score_total']:g} pts" in text
    for key in ("score_total", "score_mission", "score_escape", "score_first_escape", "score_other_survives",
                "score_time_bonus", "rank", "escape_order"):
        assert key in event["ranking"][0]


def test_sc_llm_receives_the_scoring_rules():
    block = render_live_mission_block("saboteur", lives=3, scoring=True)
    assert f"+{constants.SCORE_ESCAPE_FIRST:g}" in block and "Traps score nothing" in block
    assert "own armed trap" in block.lower() or "your own traps are dangerous" in block.lower()
    game = live_game()
    assert "Scoring:" in game.black_llm_controller.config.live_prompt_addendum


# UI contrast --------------------------------------------------------------------

def test_ui_theme_text_meets_contrast_target():
    panel = THEME["panel_background"]
    over_white = blend(panel, PANEL_ALPHA, (255, 255, 255))  # worst case: panel over the brightest floor
    for token in THEME_TEXT_TOKENS:
        assert contrast_ratio(THEME[token], panel) >= 4.5, token
        assert contrast_ratio(THEME[token], over_white) >= 4.5, token
    assert contrast_ratio(THEME["exit_sign_text"], THEME["exit_sign"]) >= 4.5


@pytest.mark.parametrize("runtime", ["native", "strands"])
def test_end4_native_and_strands_discard_a_post_terminal_answer_the_same_way(runtime):
    from tests.test_bugfix_live_async import native_agent, strands_agent, wait_ready

    state = fresh_state()
    gate = Gate(lambda p: {"action": "wait", "reason": "late"})
    telemetry = TelemetryLogger(None)
    agent = (native_agent if runtime == "native" else strands_agent)(WHITE, gate, telemetry=telemetry)
    agent.get_action(build_observation(state, WHITE))
    assert agent._thinking_in_progress
    state.mission_complete[WHITE] = True
    gate.release.set()
    wait_ready(agent)
    action = agent.get_action(build_observation(state, WHITE))
    assert action.type is ActionType.WAIT
    commit = [e for e in telemetry.events if e.get("event_type") == "live_async_commit"][-1]
    assert commit["stale_classification"] == live_async.DISCARDED_TERMINAL
    agent.get_action(build_observation(state, WHITE))
    assert gate.calls == 1


def test_end7_answer_arriving_after_match_end_is_recorded_discarded_terminal():
    wall = FakeWall()
    telemetry = TelemetryLogger(None)
    gate = Gate(lambda p: {"action": "wait", "reason": "late"})
    game = live_game({WHITE: gate}, telemetry=telemetry, wall=wall)
    frames(game, wall, 3)
    white = game.bot_controller
    assert white._thinking_in_progress
    game.state.mission_time_remaining = 0.0
    assert game.match_over()
    result = game.finish_match()
    assert result["requests_in_flight_at_match_end"] == [WHITE]
    gate.release.set()
    settle(white)
    game.finish_match()  # the runner calls it on every frame after the end
    commit = [e for e in telemetry.events if e.get("event_type") == "live_async_commit" and e["actor_id"] == WHITE][-1]
    assert commit["stale_classification"] == live_async.DISCARDED_TERMINAL and gate.calls == 1
