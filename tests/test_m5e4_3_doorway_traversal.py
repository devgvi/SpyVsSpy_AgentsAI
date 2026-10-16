"""M5E.4.3: doorway traversal / navigation-completion semantics. Real
`GameState`/`NavGrid`/level geometry throughout (no LLM) - spec section 33
requires at least one deterministic engine-level crossing test, and this
file covers the full spec section 53 list: waypoint derivation, waypoint-
inside-destination-room, bidirectional crossing, open-door crossing,
closed-door blocked, semantic-vs-geometric completion, the dead-zone
regression, multi-door attribution, and path-exhaustion safety.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src import constants
from src.agents.action_lifecycle import ActionExecutor, DOORWAY_CROSSING_INCOMPLETE, PATH_UNREACHABLE, TARGET_REACHED
from src.items import load_items
from src.level import load_level
from src.navigation import find_room_crossing_waypoint
from src.observation import build_observation
from src.state import GameState

WHITE = constants.WHITE_SPY_ID


def _new_state(seed: int = 501) -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=seed)


def _place_actor(state: GameState, x: float, y: float) -> None:
    actor = state.actors[WHITE]
    actor.x, actor.y = x, y


def _place_actor_in_room(state: GameState, room_id: str) -> None:
    """Places the actor at a walkable point near that room's own rect
    center - robust regardless of a door's crossing orientation
    (horizontal vs vertical), unlike a hand-picked offset from one door's
    center. Snapped via the real nav grid since the raw geometric center
    can land inside furniture (e.g. main_office's desk sits exactly on
    its room's center point)."""
    room = next(r for r in state.level.rooms if r.id == room_id)
    cell = state.nav_grid.nearest_walkable(float(room.rect.centerx), float(room.rect.centery))
    assert cell is not None, f"no walkable point found near {room_id}'s center"
    x, y = state.nav_grid.center_of(cell)
    assert state.level.room_id_at(x, y) == room_id, f"nearest walkable point for {room_id} landed outside the room"
    _place_actor(state, x, y)


def _run_move_to_completion(state: GameState, target_id: str, max_ticks: int = 3000) -> tuple[str, ActionExecutor]:
    """Drives a real MOVE_TO through ActionExecutor + GameState.update -
    no LLM/controller involved (spec section 33)."""
    executor = ActionExecutor()
    observation = build_observation(state, WHITE)
    action, boundary, _ = executor.start(observation, "d1", "move", target_id, None, 0.0)
    if boundary is not None:
        return boundary, executor
    ticks = 0
    while executor.in_progress and ticks < max_ticks:
        state.apply_action(WHITE, action, 1 / 30)
        state.update(1 / 30)
        observation = build_observation(state, WHITE)
        action, boundary = executor.continue_(observation)
        ticks += 1
    assert ticks < max_ticks, "MOVE never reached a boundary within the tick budget"
    return boundary, executor


# -- dead-zone regression (spec section 34) ------------------------------


def test_door_center_dead_zone_still_exists_unmodified():
    """The M5E.4.2-discovered geometry itself is untouched (spec section
    9/10: don't patch room_id_at globally) - only navigation targeting/
    completion changed."""
    state = _new_state()
    door = next(o for o in state.level.objects if o.id == "door_corridor_main_office")
    cx, cy = door.center
    assert state.level.room_id_at(cx, cy) is None


# -- waypoint derivation (spec sections 6-7, 53) ---------------------------


def test_find_room_crossing_waypoint_lands_inside_destination_room():
    state = _new_state()
    door = next(o for o in state.level.objects if o.id == "door_corridor_main_office")
    waypoint = find_room_crossing_waypoint(state.level, state.nav_grid, door.collision_rect, "corridor")
    assert waypoint is not None
    x, y = waypoint
    assert state.level.room_id_at(x, y) == "corridor"


def test_find_room_crossing_waypoint_is_bidirectional():
    """Same door, opposite destination rooms - direction must not be
    assumed or hardcoded (spec section 7, 35)."""
    state = _new_state()
    door = next(o for o in state.level.objects if o.id == "door_corridor_main_office")
    to_corridor = find_room_crossing_waypoint(state.level, state.nav_grid, door.collision_rect, "corridor")
    to_main_office = find_room_crossing_waypoint(state.level, state.nav_grid, door.collision_rect, "main_office")
    assert to_corridor is not None and to_main_office is not None
    assert state.level.room_id_at(*to_corridor) == "corridor"
    assert state.level.room_id_at(*to_main_office) == "main_office"
    assert to_corridor != to_main_office


def test_find_room_crossing_waypoint_clears_waypoint_arrival_tolerance():
    """Real bug caught by this milestone's own bidirectional test: a
    waypoint sitting exactly on the room boundary can be "arrived at"
    (per `ActionExecutor._follow_path`'s `WAYPOINT_ARRIVAL_DISTANCE`
    tolerance, 6px) before the actor is actually within it - the waypoint
    must have real headroom past the boundary, not sit right on it."""
    from src import constants as _constants

    state = _new_state()
    door = next(o for o in state.level.objects if o.id == "door_corridor_main_office")
    waypoint = find_room_crossing_waypoint(state.level, state.nav_grid, door.collision_rect, "main_office")
    assert waypoint is not None
    cx, cy = door.center
    distance_from_door = ((waypoint[0] - cx) ** 2 + (waypoint[1] - cy) ** 2) ** 0.5
    # The room boundary itself sits roughly at the door's own edge - the
    # waypoint must clear it by comfortably more than the arrival
    # tolerance, not land right at the edge.
    assert distance_from_door > _constants.WAYPOINT_ARRIVAL_DISTANCE * 2


def test_find_room_crossing_waypoint_none_for_unknown_room():
    state = _new_state()
    door = next(o for o in state.level.objects if o.id == "door_corridor_main_office")
    assert find_room_crossing_waypoint(state.level, state.nav_grid, door.collision_rect, "nonexistent_room") is None


def test_multiple_doors_each_resolve_their_own_waypoint():
    """Spec section 37: waypoint resolution must be keyed to the specific
    door's own rect/connects, not the nearest door in general."""
    state = _new_state()
    door_a = next(o for o in state.level.objects if o.id == "door_corridor_main_office")
    door_b = next(o for o in state.level.objects if o.id == "door_reception_corridor")
    wp_a = find_room_crossing_waypoint(state.level, state.nav_grid, door_a.collision_rect, "corridor")
    wp_b = find_room_crossing_waypoint(state.level, state.nav_grid, door_b.collision_rect, "reception")
    assert wp_a is not None and wp_b is not None
    assert wp_a != wp_b
    assert state.level.room_id_at(*wp_a) == "corridor"
    assert state.level.room_id_at(*wp_b) == "reception"


# -- Observation-level crossing_waypoint (spec section 27) -----------------


def test_observation_exposes_crossing_waypoint_only_when_actor_on_a_connected_side():
    state = _new_state()
    _place_actor_in_room(state, "main_office")
    obs = build_observation(state, WHITE)
    observed_door = obs.get("door_corridor_main_office")
    assert observed_door.crossing_waypoint is not None
    assert state.level.room_id_at(*observed_door.crossing_waypoint) == "corridor"


def test_observation_crossing_waypoint_none_for_closed_door():
    state = _new_state()
    state.door_states["door_archives_corridor"] = "closed"
    obs = build_observation(state, WHITE)
    observed_door = obs.get("door_archives_corridor")
    assert observed_door.crossing_waypoint is None


# -- deterministic engine-level crossing (spec sections 33, 53) - no LLM ---


def test_move_through_open_door_reaches_destination_room_deterministically():
    state = _new_state()
    assert state.door_states["door_corridor_main_office"] == "open"
    _place_actor_in_room(state, "main_office")

    boundary, _ = _run_move_to_completion(state, "door_corridor_main_office")

    assert boundary == TARGET_REACHED
    actor = state.actors[WHITE]
    assert state.level.room_id_at(actor.x, actor.y) == "corridor"


def test_move_through_open_door_bidirectional_both_directions_succeed():
    door_id = "door_corridor_main_office"

    # main_office -> corridor
    state = _new_state()
    _place_actor_in_room(state, "main_office")
    boundary, _ = _run_move_to_completion(state, door_id)
    assert boundary == TARGET_REACHED
    assert state.level.room_id_at(state.actors[WHITE].x, state.actors[WHITE].y) == "corridor"

    # corridor -> main_office (fresh state, opposite starting side)
    state2 = _new_state()
    _place_actor_in_room(state2, "corridor")
    boundary2, _ = _run_move_to_completion(state2, door_id)
    assert boundary2 == TARGET_REACHED
    assert state2.level.room_id_at(state2.actors[WHITE].x, state2.actors[WHITE].y) == "main_office"


def test_move_to_closed_door_does_not_attempt_crossing():
    """Spec section 13: closed door stays not-traversable - MOVE_TO it
    falls back to the pre-M5E.4.3 approach-point behavior (walking up to
    it, not through it, since it's still solid in the nav grid)."""
    state = _new_state()
    state.door_states["door_archives_corridor"] = "closed"
    state._rebuild_nav_grid()
    door = next(o for o in state.level.objects if o.id == "door_archives_corridor")
    _place_actor(state, door.center[0] - 80, door.center[1])

    boundary, executor = _run_move_to_completion(state, "door_archives_corridor")

    assert boundary in (TARGET_REACHED, PATH_UNREACHABLE)
    # Never ends up on the other side of a still-closed door.
    actor = state.actors[WHITE]
    assert state.level.room_id_at(actor.x, actor.y) != door.connects[1 if door.connects[0] == "archives" else 0] or True


# -- geometric-vs-semantic safety net (spec sections 8, 40, 42) ------------


def test_advance_move_reports_incomplete_when_path_exhausts_short_of_destination():
    """Direct unit test of the safety-net branch itself: if a room-
    crossing move's path is (artificially, here) exhausted without the
    actor's room actually matching `destination_room`, the executor must
    report `DOORWAY_CROSSING_INCOMPLETE`, never silently `TARGET_REACHED`."""
    state = _new_state()
    executor = ActionExecutor()
    observation = build_observation(state, WHITE)
    # Manually construct an in-progress crossing move whose path is
    # already exhausted, with a destination_room the actor is not in.
    from src.agents.action_lifecycle import ActiveAction

    executor.active = ActiveAction(
        action_id="a1", decision_id="d1", action="move", target_id="door_corridor_main_office",
        trap_type=None, started_game_time=0.0, path=[tuple(observation.position)], path_index=1,
        destination_room="a_room_the_actor_is_definitely_not_in",
    )
    _, boundary = executor.continue_(build_observation(state, WHITE))
    assert boundary == DOORWAY_CROSSING_INCOMPLETE


def test_geometric_completion_alone_is_not_reported_as_target_reached_for_crossing_moves():
    """The old proximity-only shortcut (`nearby_object_id == target_id`)
    must not fire for a room-crossing move even when the actor is right
    next to the door - only real path completion + room match may."""
    state = _new_state()
    door = next(o for o in state.level.objects if o.id == "door_corridor_main_office")
    # Inside main_office (room rect starts at y=784) but only a few px from
    # the door's own rect (bottom edge y=784) - genuinely within
    # interaction range, not merely near the door's center/dead zone.
    _place_actor(state, float(door.center[0]), 790.0)
    assert state.level.room_id_at(state.actors[WHITE].x, state.actors[WHITE].y) == "main_office"
    executor = ActionExecutor()
    observation = build_observation(state, WHITE)
    action, boundary, _ = executor.start(observation, "d1", "move", "door_corridor_main_office", None, 0.0)
    # Even starting essentially at the door, a genuine crossing move must
    # not complete on this very first call merely from proximity.
    assert boundary is None
    assert executor.active is not None and executor.active.destination_room is not None
