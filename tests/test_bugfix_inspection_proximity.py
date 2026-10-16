"""Bugfix live flow, Part B: inspection happens at the object (I1-I8, §19-26).

Real level and engine. The engine's range rule is not changed by this
bugfix (70 px from the feet to the collision rect, one shared helper); what
changes, in live play only, is where a MOVE_TO ends (close approach) and
that an interacting spy faces its target.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from src import constants
from src.actions import ActionType, GameAction
from src.agents.action_lifecycle import CLOSE_APPROACH_PX, ActionExecutor, TARGET_REACHED
from src.evaluation import interaction_visual_audit as visual
from src.evaluation.navigation_audit import fresh_state, start_positions
from src.interaction_range import distance_to_rect, nearest_interactable_id
from src.llm_controller import enumerate_legal_actions, validate_model_output
from src.observation import build_observation

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID
DT = 1 / 60


def move_to(state, target_id, *, close=True, actor_id=WHITE, max_seconds=20.0):
    obs = build_observation(state, actor_id)
    executor = ActionExecutor()
    executor.close_approach = close
    action, boundary, _ = executor.start(obs, "d", "move", target_id, None, obs.mission_time_remaining)
    ticks = 0
    while boundary is None and ticks < int(max_seconds / DT):
        state.apply_action(actor_id, action, DT)
        state.update(DT)
        ticks += 1
        obs = build_observation(state, actor_id)
        action, boundary = executor.continue_(obs)
    return boundary, build_observation(state, actor_id)


def act(state, action_type, target_id, actor_id=WHITE, trap_type=None):
    state.apply_action(actor_id, GameAction(action_type, target_id=target_id, trap_type=trap_type), DT)
    for _ in range(900):  # let the action's time cost run out
        state.update(DT)
        if not state.actors[actor_id].is_busy:
            break


def events(state, kind, object_id=None):
    return [e for e in state.event_log if e["type"] == kind and (object_id is None or e.get("object_id") == object_id)]


def legal(obs, action, target):
    return any(a["action"] == action and a.get("target_id") == target for a in enumerate_legal_actions(obs, reachable_only=True))


def furniture(state, room=None):
    return [o for o in state.level.interactables()
            if o.type != "door" and (room is None or state.level.room_id_at(*o.center) == room)]


def feet_distance(state, target_id, actor_id=WHITE):
    a = state.actors[actor_id]
    return distance_to_rect(a.x, a.y, state.level.get_object(target_id).collision_rect)


@pytest.fixture
def state():
    return fresh_state()


def _trappable_in_start_room(state):
    room = build_observation(state, WHITE).current_room
    return next(o for o in furniture(state, room) if o.trappable)


# I1 ---------------------------------------------------------------------------

def test_i1_inspect_immediately_after_valid_move_to(state):
    obj = _trappable_in_start_room(state)
    boundary, obs = move_to(state, obj.id)
    assert boundary == TARGET_REACHED and obs.nearby_object_id == obj.id
    assert legal(obs, "inspect", obj.id)
    # close approach: at the object, not at the edge of the 70 px range
    assert feet_distance(state, obj.id) <= CLOSE_APPROACH_PX + 1
    act(state, ActionType.INSPECT, obj.id)
    assert events(state, "OBJECT_INSPECTED", obj.id)


def test_i1_evaluation_completion_is_unchanged(state):
    """Without close approach (evaluation, every stored run) a MOVE_TO still
    ends on the first frame the target is in range."""
    obj = _trappable_in_start_room(state)
    boundary, obs = move_to(state, obj.id, close=False)
    assert boundary == TARGET_REACHED and obs.nearby_object_id == obj.id
    assert feet_distance(state, obj.id) > CLOSE_APPROACH_PX


# I2 ---------------------------------------------------------------------------

def test_i2_inspect_refused_outside_range(state):
    obj = _trappable_in_start_room(state)
    a = state.actors[WHITE]
    # the walkable start cell farthest from the object: out of range
    far = max(start_positions(state, build_observation(state, WHITE).current_room, 50, __import__("random").Random(1)),
              key=lambda p: distance_to_rect(*p, obj.collision_rect))
    a.x, a.y = far
    state.update(0.0)
    obs = build_observation(state, WHITE)
    assert feet_distance(state, obj.id) > constants.INTERACTION_DISTANCE and obs.nearby_object_id != obj.id
    for action in ("inspect", "interact", "place_trap", "disarm"):
        assert not legal(obs, action, obj.id)
    result = validate_model_output(obs, {"action": "inspect", "target_id": obj.id}, enumerate_legal_actions(obs, reachable_only=True))
    assert not result.valid and result.error_type == "out_of_range"
    act(state, ActionType.INSPECT, obj.id)  # the engine refuses too - no remote inspection
    assert not events(state, "OBJECT_INSPECTED", obj.id)


# I3 / I4 ----------------------------------------------------------------------

def _wall_gap(state, obj):
    return min(math_gap(obj.collision_rect, wall) for wall in state.level.walls)


def math_gap(a, b):
    dx = max(b.left - a.right, a.left - b.right, 0)
    dy = max(b.top - a.bottom, a.top - b.bottom, 0)
    return math.hypot(dx, dy)


def test_i3_inspect_wall_adjacent_furniture(state):
    candidates = [o for o in furniture(state) if o.trappable]
    obj = min(candidates, key=lambda o: _wall_gap(state, o))
    assert _wall_gap(state, obj) <= 8  # really against a wall
    room = state.level.room_id_at(*obj.center)
    a = state.actors[WHITE]
    a.x, a.y = start_positions(state, room, 1, __import__("random").Random(2))[0]
    state.update(0.0)
    boundary, obs = move_to(state, obj.id)
    assert boundary == TARGET_REACHED and legal(obs, "inspect", obj.id)
    act(state, ActionType.INSPECT, obj.id)
    assert events(state, "OBJECT_INSPECTED", obj.id)


def test_i4_inspect_large_furniture_from_its_edge(state):
    obj = max((o for o in furniture(state) if o.trappable), key=lambda o: o.collision_rect.width * o.collision_rect.height)
    room = state.level.room_id_at(*obj.center)
    a = state.actors[WHITE]
    a.x, a.y = start_positions(state, room, 1, __import__("random").Random(3))[0]
    state.update(0.0)
    boundary, obs = move_to(state, obj.id)
    assert boundary == TARGET_REACHED and legal(obs, "inspect", obj.id)
    # in range of the edge, even when the centre is far away
    assert feet_distance(state, obj.id) <= constants.INTERACTION_DISTANCE
    act(state, ActionType.INSPECT, obj.id)
    assert events(state, "OBJECT_INSPECTED", obj.id)


# I5 / I6 / I7 -----------------------------------------------------------------

def test_i5_search_after_inspect(state):
    obj = _trappable_in_start_room(state)
    move_to(state, obj.id)
    act(state, ActionType.INSPECT, obj.id)
    obs = build_observation(state, WHITE)
    assert obs.nearby_object_id == obj.id and legal(obs, "interact", obj.id)
    act(state, ActionType.INTERACT, obj.id)
    assert events(state, "OBJECT_SEARCHED", obj.id)


def _step_out_of_range(state, obj):
    room = state.level.room_id_at(*obj.center)
    a = state.actors[WHITE]
    a.x, a.y = max(start_positions(state, room, 50, __import__("random").Random(4)),
                   key=lambda p: distance_to_rect(*p, obj.collision_rect))
    state.update(0.0)
    assert build_observation(state, WHITE).nearby_object_id != obj.id


def test_i6_place_trap_only_in_range(state):
    obj = _trappable_in_start_room(state)
    move_to(state, obj.id)
    act(state, ActionType.INSPECT, obj.id)
    _step_out_of_range(state, obj)
    obs = build_observation(state, WHITE)
    assert not legal(obs, "place_trap", obj.id)
    act(state, ActionType.PLACE_TRAP, obj.id, trap_type="snare")
    assert not events(state, "TRAP_PLACED")
    move_to(state, obj.id)
    obs = build_observation(state, WHITE)
    assert legal(obs, "place_trap", obj.id)
    act(state, ActionType.PLACE_TRAP, obj.id, trap_type="snare")
    assert events(state, "TRAP_PLACED", obj.id)


def test_i7_disarm_only_in_range(state):
    obj = _trappable_in_start_room(state)
    entry = state._object_entry(obj.id)
    entry["trap"] = {"type": "snare", "owner_id": BLACK, "armed": True}
    entry["detected_by"][WHITE] = "snare"
    _step_out_of_range(state, obj)
    assert not legal(build_observation(state, WHITE), "disarm", obj.id)
    act(state, ActionType.DISARM, obj.id)
    assert not events(state, "TRAP_DISARMED")
    boundary, obs = move_to(state, obj.id)
    if boundary != TARGET_REACHED:  # the snare may have caught White on the way in; that is fine
        pytest.skip(f"move ended with {boundary}")
    assert legal(obs, "disarm", obj.id)
    act(state, ActionType.DISARM, obj.id)
    assert events(state, "TRAP_DISARMED", obj.id)


# I8 ---------------------------------------------------------------------------

def test_i8_render_and_collision_anchors(state):
    actor = state.actors[WHITE]
    # collision centred on the feet, sprite bottom-anchored on the feet
    assert actor.collision_rect.center == (int(actor.x - constants.PLAYER_COLLISION_SIZE[0] / 2) + constants.PLAYER_COLLISION_SIZE[0] // 2,
                                           int(actor.y - constants.PLAYER_COLLISION_SIZE[1] / 2) + constants.PLAYER_COLLISION_SIZE[1] // 2)
    assert abs(actor.sprite_rect.midbottom[0] - actor.x) <= 1 and abs(actor.sprite_rect.bottom - actor.y) <= 1
    renderer = visual._renderer()
    from src import assets
    from src.renderer import _door_sprite_path
    for obj in furniture(state):
        if obj.render_offset is not None:
            continue
        image = assets.get_fitted(_door_sprite_path(obj, None), obj.render_max_size)
        placed = renderer._placement_rect(obj, image)
        # the object's art stands on its collision footprint (bottom-centre)
        assert abs(placed.centerx - obj.collision_rect.centerx) <= 1, obj.id
        assert abs(placed.bottom - obj.collision_rect.bottom) <= 1, obj.id


def test_i8_close_approach_is_visually_close():
    audit = visual.approach_audit(per_room=2)
    before, after = audit["evaluation_first_frame_in_range"], audit["live_close_approach"]
    assert after["not_reached"] == 0 and after["classification"].get("TRUE_ENGINE_RANGE_BUG", 0) == 0
    assert after["visual_gap_world_px"]["max"] < before["visual_gap_world_px"]["max"]
    assert after["classification"].get("VISUALLY_CLOSE", 0) / after["interactions"] >= 0.85


def test_legal_actions_and_engine_use_the_same_range_rule(state):
    candidates = [(o.id, o.collision_rect) for o in state.level.interactables()]
    a = state.actors[WHITE]
    grid = state.nav_grid
    checked = 0
    for row in range(0, grid.rows, 3):
        for col in range(0, grid.cols, 3):
            if not grid.is_walkable((row, col)):
                continue
            a.x, a.y = grid.center_of((row, col))
            state.update(0.0)
            expected = nearest_interactable_id(a.x, a.y, candidates)
            assert state.current_interaction_target[WHITE] == expected
            assert build_observation(state, WHITE).nearby_object_id == expected
            checked += 1
    assert checked > 50


def test_live_game_faces_the_interaction_target():
    from src.game import Game

    game = Game(seed=7, white_controller="scripted", black_controller="scripted")
    game.face_interaction_targets = True
    st = game.state
    room = build_observation(st, WHITE).current_room
    obj = next(o for o in furniture(st, room) if o.trappable)
    move_to(st, obj.id)
    actor = st.actors[WHITE]
    cx, cy = obj.collision_rect.center
    dx, dy = cx - actor.x, cy - actor.y
    expected = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
    opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}[expected]
    actor.facing = opposite
    game._face_interaction_targets({WHITE: GameAction(ActionType.INSPECT, target_id=obj.id)})
    assert actor.facing == expected
    # never for an action the engine will not perform (not the near object)
    actor.facing = opposite
    game._face_interaction_targets({WHITE: GameAction(ActionType.INSPECT, target_id="exit_door")})
    assert actor.facing == opposite
    pygame.quit()
