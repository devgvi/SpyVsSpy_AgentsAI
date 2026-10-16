"""Spawn points (bugfix spawn/combat/visual polish, Part A-B).

`level` (the default everywhere except the live game): both spies start at
the level file's own spawns (`player_spawn` / `white_spy_spawn` - the same
room in level_01), exactly as every stored run and lab was recorded.

`different_rooms` (the live workshop default): Black and White start in two
different rooms, picked from a seeded RNG of their own (`Random("spawn:<seed>")`
- never the shared game `rng`, so dependency placement and INSPECT rolls
are untouched). Candidate rooms: every room the level-file spawn can walk to
with the doors as they start (no key needed to get out), except the room the
exit opens from. The point in each room is the walkable cell nearest the
room centre whose whole 3x3 cell neighbourhood is walkable and which is at
least SPAWN_OBJECT_CLEARANCE from every interactable object (not inside or
beside furniture, not on the exit). Each ordered pair of different rooms is
equally likely, so over many seeds neither colour is favoured.

`safe_respawn_point` (fight loser, Part B): the actor's own spawn if it is
safe - at least ENCOUNTER_REARM_DISTANCE from the opponent and not beside an
object carrying an armed trap - else the other candidate points in order.
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

from src import constants

if TYPE_CHECKING:  # pragma: no cover
    from src.state import GameState

LEVEL = "level"
DIFFERENT_ROOMS = "different_rooms"
SAME_ROOM = "same_room"  # live alias of LEVEL: both spies at the level file's spawns
SPAWN_POLICIES = (DIFFERENT_ROOMS, SAME_ROOM, LEVEL)
SPAWN_OBJECT_CLEARANCE = 64.0


def _clear_cell(state: "GameState", cell: tuple[int, int]) -> bool:
    grid = state.nav_grid
    row, col = cell
    return all(grid.is_walkable((row + dr, col + dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _far_from_objects(state: "GameState", point: tuple[float, float]) -> bool:
    from src.interaction_range import distance_to_rect

    return all(distance_to_rect(point[0], point[1], obj.collision_rect) >= SPAWN_OBJECT_CLEARANCE
               for obj in state.level.objects if obj.interactable)


def room_spawn_point(state: "GameState", room_id: str) -> tuple[float, float] | None:
    room = next((r for r in state.level.rooms if r.id == room_id), None)
    if room is None:
        return None
    grid = state.nav_grid
    cx, cy = room.rect.center
    best, best_d = None, None
    for row in range(grid.rows):
        for col in range(grid.cols):
            point = grid.center_of((row, col))
            if not room.rect.collidepoint(point) or not _clear_cell(state, (row, col)):
                continue
            if not _far_from_objects(state, point):
                continue
            d = math.dist(point, (cx, cy))
            if best_d is None or d < best_d:
                best, best_d = point, d
    return best


def candidate_rooms(state: "GameState") -> list[str]:
    from src.exit_geometry import exit_sides

    start = state.level.player_spawn
    reachable = state.nav_grid.reachable_cells(start)
    exit_room = (exit_sides(state.level) or {}).get("inside_room")
    rooms = []
    for room in state.level.rooms:
        if room.id == exit_room:
            continue
        point = room_spawn_point(state, room.id)
        if point is not None and state.nav_grid.cell_of(*point) in reachable:
            rooms.append(room.id)
    return rooms


def choose_spawns(state: "GameState", policy: str) -> dict[str, Any]:
    """The spawn record for `policy` (does not move anyone)."""
    black, white = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID
    level_points = {black: tuple(state.level.player_spawn), white: tuple(state.level.white_spy_spawn)}
    points = dict(level_points)
    candidates: list[str] = []
    if policy == DIFFERENT_ROOMS:
        candidates = candidate_rooms(state)
        if len(candidates) >= 2:
            pairs = [(a, b) for a in candidates for b in candidates if a != b]
            black_room, white_room = random.Random(f"spawn:{state.seed}").choice(pairs)
            points = {black: room_spawn_point(state, black_room), white: room_spawn_point(state, white_room)}
    return {
        "spawn_policy": policy,
        "candidate_rooms": candidates,
        "black_spawn_room": state.level.room_id_at(*points[black]),
        "white_spawn_room": state.level.room_id_at(*points[white]),
        "black_spawn_position": [round(points[black][0], 1), round(points[black][1], 1)],
        "white_spawn_position": [round(points[white][0], 1), round(points[white][1], 1)],
    }


def apply_spawns(state: "GameState", policy: str) -> dict[str, Any]:
    """Moves both spies to their spawn points and records them on the state."""
    record = choose_spawns(state, policy)
    for actor_id, key in ((constants.BLACK_SPY_ID, "black_spawn_position"), (constants.WHITE_SPY_ID, "white_spawn_position")):
        actor = state.actors[actor_id]
        actor.x, actor.y = record[key]
        actor.moving = False
    state.spawn_record = record
    state.spawn_points = {constants.BLACK_SPY_ID: tuple(record["black_spawn_position"]),
                          constants.WHITE_SPY_ID: tuple(record["white_spawn_position"])}
    state._log_event("SPAWNS_ASSIGNED", **record)
    return record


def _beside_armed_trap(state: "GameState", point: tuple[float, float]) -> bool:
    from src.interaction_range import distance_to_rect

    for obj in state.level.objects:
        if state.trap_on(obj.id) is not None and distance_to_rect(point[0], point[1], obj.collision_rect) \
                <= constants.INTERACTION_DISTANCE:
            return True
    return False


def safe_respawn_point(state: "GameState", actor_id: str) -> tuple[float, float]:
    """Walkable, clear of furniture, away from the opponent and from traps."""
    other = state.actors[next(a for a in state.actors if a != actor_id)]
    own = getattr(state, "spawn_points", {}).get(actor_id)
    points = [own] if own else []
    for room_id in candidate_rooms(state) + [r.id for r in state.level.rooms]:
        point = room_spawn_point(state, room_id)
        if point is not None and point not in points:
            points.append(point)
    level_point = tuple(state.level.player_spawn if actor_id == constants.BLACK_SPY_ID else state.level.white_spy_spawn)
    points.append(level_point)
    reachable = state.nav_grid.reachable_cells(level_point)
    for point in points:
        if point is None or state.nav_grid.cell_of(*point) not in reachable:
            continue
        if math.dist(point, (other.x, other.y)) < constants.ENCOUNTER_REARM_DISTANCE:
            continue
        if _beside_armed_trap(state, point):
            continue
        return point
    return own or level_point
