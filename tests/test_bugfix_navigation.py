"""Bugfix: MOVE_TO(object) must end where the intended interaction is legal.

Real level, real NavGrid, real Actor collisions, real engine range rule
(BUGFIX_NAVIGATION_AND_EXPLORATION_MEMORY.md §8-10). Frame steps cover the
normal 60 fps frame (1/60 s), the old test step (1/30 s), the live loop's
per-frame cap (0.1 s) and that cap at speed x2 (0.2 s).
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.actions import ActionType, GameAction
from src.agents.action_lifecycle import (
    APPROACH_INCOMPLETE,
    FAILED_MOVE_BOUNDARIES,
    MOVE_STALLED,
    PATH_UNREACHABLE,
    TARGET_REACHED,
    ActionExecutor,
    _passed_waypoint,
    ActiveAction,
)
from src.evaluation.navigation_audit import audit, fresh_state, run_move
from src.interaction_range import nearest_interactable_id
from src.llm_controller import enumerate_legal_actions
from src.observation import build_observation

WHITE = constants.WHITE_SPY_ID
DTS = (1 / 60, 1 / 30, 0.1, 0.2)


def _legal(state, action, target):
    return {"action": action, "target_id": target} in enumerate_legal_actions(build_observation(state, WHITE), reachable_only=True)


def _clear(state, start):
    import pygame
    w, h = constants.PLAYER_COLLISION_SIZE
    box = pygame.Rect(int(start[0] - w / 2), int(start[1] - h / 2), int(w), int(h))
    return not any(box.colliderect(r) for r in state.level.solid_rects(state.door_states))


def _move_ok(target, start, dt, doors_open=True):
    state = fresh_state(doors_open=doors_open)
    assert _clear(state, start), f"test start {start} overlaps a solid"
    row = run_move(state, target, start, dt)
    assert row["boundary"] == TARGET_REACHED, row
    assert row["near_object_id"] == target, row
    assert row["distance_to_target"] <= constants.INTERACTION_DISTANCE, row
    assert _legal(state, "interact", target) and _legal(state, "inspect", target) is (target != "exit_door" and "door" not in target)
    return row


@pytest.mark.parametrize("dt", DTS)
def test_n1_open_room_desk(dt):
    _move_ok("desk_01", (720.0, 1120.0), dt)


@pytest.mark.parametrize("dt", DTS)
def test_n2_object_against_a_wall(dt):
    # filing_cabinet_01 stands against the security room's north wall
    _move_ok("filing_cabinet_01", (720.0, 240.0), dt)
    # bookshelf_01 stands against the archives' west wall
    _move_ok("bookshelf_01", (400.0, 640.0), dt)


@pytest.mark.parametrize("dt", DTS)
def test_n3_object_next_to_another_obstacle_wins_the_range_check(dt):
    # side_table_01 is 20 px from the sofa: the approach point must be one
    # where the engine picks the side table, not the sofa
    row = _move_ok("side_table_01", (1040.0, 480.0), dt)
    assert row["near_object_id"] == "side_table_01"


@pytest.mark.parametrize("dt", DTS)
def test_n4_large_object_with_blocked_centre_is_approached_at_an_edge(dt):
    row = _move_ok("sofa_01", (1240.0, 500.0), dt)
    x, y = row["final"]
    assert not (952 < x < 1132 and 656 < y < 736)  # never inside the footprint


@pytest.mark.parametrize("dt", DTS)
def test_n5_door_approach_for_a_locked_door(dt):
    # locked door seen from the corridor: MOVE ends in range of the door itself
    state = fresh_state(doors_open=False)
    row = run_move(state, "door_security_corridor", (720.0, 560.0), dt)
    assert row["boundary"] == TARGET_REACHED and row["near_object_id"] == "door_security_corridor"


def test_n6_repeated_move_to_the_same_object_is_immediate():
    state = fresh_state()
    first = run_move(state, "desk_01", (720.0, 1120.0), 1 / 60)
    assert first["boundary"] == TARGET_REACHED
    executor = ActionExecutor()
    obs = build_observation(state, WHITE)
    _, boundary, _ = executor.start(obs, "d2", "move", "desk_01", None, obs.mission_time_remaining)
    assert boundary == TARGET_REACHED and executor.active is None


@pytest.mark.parametrize("start", [(600.0, 850.0), (840.0, 850.0), (500.0, 1100.0), (900.0, 1100.0), (720.0, 1150.0)])
@pytest.mark.parametrize("dt", (1 / 60, 0.1))
def test_n7_same_furniture_from_several_directions(start, dt):
    _move_ok("desk_01", start, dt)


@pytest.mark.parametrize("dt", DTS)
def test_n8_through_a_narrow_doorway_then_to_furniture(dt):
    # archives door (48 px wall gap): cross from the corridor, then reach a bookshelf
    state = fresh_state()
    row = run_move(state, "door_archives_corridor", (720.0, 560.0), dt)
    assert row["boundary"] == TARGET_REACHED
    assert state.level.room_id_at(*row["final"]) == "archives"
    row = run_move(state, "bookshelf_02", tuple(row["final"]), dt)
    assert row["boundary"] == TARGET_REACHED and row["near_object_id"] == "bookshelf_02"


def test_audit_every_interactable_from_every_room_at_every_frame_step():
    report = audit(per_room=2)
    assert report["by_class"].get("OK", 0) > 0
    assert report["failures"] == [], report["failures"][:3]
    assert report["legal_but_unreachable_from_white_spawn"] == []


def test_unreachable_target_fails_explicitly_and_is_not_offered_as_legal():
    # default doors: the security room is locked, so its furniture has no reachable approach
    state = fresh_state(doors_open=False)
    obs = build_observation(state, WHITE)
    assert not _legal(state, "move", "filing_cabinet_01")
    assert {"action": "move", "target_id": "filing_cabinet_01"} in enumerate_legal_actions(obs)  # the recorded (legacy) surface
    executor = ActionExecutor()
    _, boundary, _ = executor.start(obs, "d", "move", "filing_cabinet_01", None, obs.mission_time_remaining)
    assert boundary == PATH_UNREACHABLE
    assert executor.last_finished.failure_detail == "no_reachable_approach_point"
    nav = ActionExecutor.navigation_telemetry(executor.last_finished)
    assert nav["interaction_legal_after_move"] is False and nav["approach_point"] is None


def test_path_end_out_of_range_is_approach_incomplete_not_success(monkeypatch):
    state = fresh_state()
    obs = build_observation(state, WHITE)
    far = (obs.position[0] + 64.0, obs.position[1])  # a real, walkable, but useless end point
    monkeypatch.setattr(type(state.nav_grid), "reachable_approach_point", lambda self, *a, **k: far)
    executor = ActionExecutor()
    action, boundary, _ = executor.start(obs, "d", "move", "desk_01", None, obs.mission_time_remaining)
    for _ in range(600):
        if boundary is not None:
            break
        state.apply_action(WHITE, action, 1 / 60)
        state.update(1 / 60)
        action, boundary = executor.continue_(build_observation(state, WHITE))
    assert boundary == APPROACH_INCOMPLETE
    assert executor.last_finished.failure_detail == "target_not_in_interaction_range"


def test_no_progress_is_a_stall_in_game_time_not_in_ticks():
    state = fresh_state()
    executor = ActionExecutor()
    obs = build_observation(state, WHITE)
    action, boundary, _ = executor.start(obs, "d", "move", "desk_01", None, obs.mission_time_remaining)
    assert boundary is None
    # frozen clock (the other spy's model is thinking): no movement, no stall
    for _ in range(300):
        action, boundary = executor.continue_(build_observation(state, WHITE))
    assert boundary is None
    # clock running but the actor never moves (e.g. blocked): stall after 1 s of game time
    for _ in range(90):
        state.update(1 / 60)
        action, boundary = executor.continue_(build_observation(state, WHITE))
        if boundary is not None:
            break
    assert boundary == MOVE_STALLED and executor.last_finished.failure_detail == "no_progress"


def test_failed_move_boundaries_are_failures_in_telemetry_and_memory():
    from src.llm.agent_memory import AgentMemory
    from src.llm.telemetry import TelemetryLogger
    state = fresh_state(doors_open=False)
    obs = build_observation(state, WHITE)
    executor = ActionExecutor()
    executor.start(obs, "d", "move", "filing_cabinet_01", None, obs.mission_time_remaining)
    memory, telemetry = AgentMemory(WHITE), TelemetryLogger(None)
    executor.finish(memory, telemetry, "ep", WHITE, "native", "v", obs, PATH_UNREACHABLE)
    [event] = [e for e in telemetry.events if e["event_type"] == "action_lifecycle"]
    assert event["action_status"] == "failed"
    assert event["navigation"]["move_target_id"] == "filing_cabinet_01"
    assert event["navigation"]["interaction_range"] == constants.INTERACTION_DISTANCE
    assert FAILED_MOVE_BOUNDARIES >= {PATH_UNREACHABLE, APPROACH_INCOMPLETE, MOVE_STALLED}


def test_passed_waypoint_detects_overshoot():
    active = ActiveAction("a", "d", "move", "x", None, 0.0, path=[(0.0, 0.0), (32.0, 0.0), (64.0, 0.0)], path_index=1)
    assert not _passed_waypoint(active, (20.0, 0.0))
    assert _passed_waypoint(active, (44.0, 3.0))  # 12 px past, outside the 6 px tolerance


def test_small_steps_move_exactly_as_before():
    """Steps <= 8 px take one collision sub-step: identical to the old code."""
    from src.actor import Actor
    solids = fresh_state().level.solid_rects({})
    for dt in (1 / 60, 1 / 30):
        a, b = Actor(id="a", x=720.0, y=1120.0), Actor(id="b", x=720.0, y=1120.0)
        for _ in range(40):
            a.move((0.6, -0.8), dt, solids)
            # the old implementation, inlined
            b.moving = True
            b.update_facing((0.6, -0.8))
            b._move_axis(0.6 * b.speed * dt, 0, solids)
            b._move_axis(0, -0.8 * b.speed * dt, solids)
        assert (a.x, a.y, a.distance_travelled) == (b.x, b.y, b.distance_travelled)


def test_engine_and_navigation_share_one_range_rule():
    state = fresh_state()
    actor = state.actors[WHITE]
    candidates = [(o.id, o.collision_rect) for o in state.level.interactables()]
    for x in range(480, 1000, 23):
        for y in range(800, 1180, 19):
            actor.x, actor.y = float(x), float(y)
            assert state._find_nearest_interactable(actor) == nearest_interactable_id(actor.x, actor.y, candidates)


def test_reachable_only_is_off_by_default_so_recorded_payloads_do_not_change():
    state = fresh_state(doors_open=False)
    obs = build_observation(state, WHITE)
    legacy = enumerate_legal_actions(obs)
    live = enumerate_legal_actions(obs, reachable_only=True)
    assert set(map(str, live)) < set(map(str, legacy))
    assert all(e["action"] != "move" or e in legacy for e in live)
    assert [e for e in legacy if e["action"] != "move"] == [e for e in live if e["action"] != "move"]
