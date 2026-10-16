"""A coarse walkability grid built from level collision geometry, plus A*.

This is intentionally simple: a fixed-size grid (see constants.NAV_GRID_SIZE)
marks each cell blocked or walkable from the same wall/solid-object rects
the physics collision system already uses, so both spies navigate the same
space they collide in. Pathfinding is textbook 8-directional A* with a
corner-cut guard - nothing fancier is needed at this level's scale.
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import pygame

from src import constants
from src.interaction_range import distance_to_rect, nearest_interactable_id
from src.level import Level

# (row, col) neighbor offsets: 4 orthogonal + 4 diagonal, with per-step cost.
_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


@dataclass
class NavGrid:
    cell_size: int
    rows: int
    cols: int
    blocked: set[tuple[int, int]]
    # Bugfix (navigation/exploration): connected-component cache, keyed by the
    # snapped start cell. A NavGrid is rebuilt whenever a door changes state
    # (GameState._rebuild_nav_grid), so a cache on the instance never goes stale.
    _reach_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def reachable_cells(self, start_xy: tuple[float, float]) -> frozenset[tuple[int, int]]:
        """Every walkable cell A* could reach from `start_xy` (same snapping and
        corner-cut rule as `find_path`). Empty if the start cannot be snapped."""
        start = self.nearest_walkable(*start_xy)
        if start is None:
            return frozenset()
        cached = self._reach_cache.get(start)
        if cached is not None:
            return cached
        seen = {start}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for drow, dcol, _ in _NEIGHBORS:
                neighbor = (row + drow, col + dcol)
                if neighbor in seen or not self.is_walkable(neighbor):
                    continue
                if drow != 0 and dcol != 0 and (
                    not self.is_walkable((row + drow, col)) or not self.is_walkable((row, col + dcol))
                ):
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        result = frozenset(seen)
        self._reach_cache[start] = result
        return result

    def has_reachable_approach(
        self, rect: pygame.Rect, from_xy: tuple[float, float], max_distance: float = constants.INTERACTION_DISTANCE
    ) -> bool:
        """Cheap legality check: is any cell within `max_distance` of `rect`
        reachable from `from_xy`? (No nearest-object check - that refinement
        is only needed when a MOVE actually starts.) Cached per start cell."""
        start = self.nearest_walkable(*from_xy)
        if start is None:
            return False
        key = ("approach", start, (rect.left, rect.top, rect.width, rect.height), max_distance)
        cached = self._reach_cache.get(key)
        if cached is not None:
            return cached
        reachable = self.reachable_cells(from_xy)
        search = rect.inflate(max_distance * 2, max_distance * 2)
        top_row, left_col = self.cell_of(search.left, search.top)
        bottom_row, right_col = self.cell_of(search.right, search.bottom)
        found = False
        for row in range(max(0, top_row), min(self.rows - 1, bottom_row) + 1):
            for col in range(max(0, left_col), min(self.cols - 1, right_col) + 1):
                if (row, col) in reachable and distance_to_rect(*self.center_of((row, col)), rect) <= max_distance:
                    found = True
                    break
            if found:
                break
        self._reach_cache[key] = found
        return found

    def reachable_approach_point(
        self,
        target_id: str,
        rect: pygame.Rect,
        from_xy: tuple[float, float],
        interactables: list[tuple[str, pygame.Rect]] | None = None,
        max_distance: float = constants.INTERACTION_DISTANCE,
    ) -> tuple[float, float] | None:
        """The approach point a MOVE_TO(target) should walk to, or None if there
        is no reachable place from which the target can be interacted with.

        A candidate is a walkable cell centre that is (1) reachable from
        `from_xy`, (2) within `max_distance` of the target's collision rect,
        and (3) - when `interactables` is given - a point where the engine's
        own rule (`nearest_interactable_id`) picks this target, not a
        neighbouring object. Ranked like `nearest_approach_point`: closest to
        the target first, then closest to the actor. Falls back to (1)+(2)
        only if no cell satisfies (3), so a crowded target stays reachable
        rather than failing outright.
        """
        reachable = self.reachable_cells(from_xy)
        if not reachable:
            return None
        search = rect.inflate(max_distance * 2, max_distance * 2)
        top_row, left_col = self.cell_of(search.left, search.top)
        bottom_row, right_col = self.cell_of(search.right, search.bottom)
        best = best_key = None
        fallback = fallback_key = None
        for row in range(max(0, top_row), min(self.rows - 1, bottom_row) + 1):
            for col in range(max(0, left_col), min(self.cols - 1, right_col) + 1):
                cell = (row, col)
                if cell not in reachable:
                    continue
                cx, cy = self.center_of(cell)
                dist_to_rect = distance_to_rect(cx, cy, rect)
                if dist_to_rect > max_distance:
                    continue
                key = (dist_to_rect, math.dist((cx, cy), from_xy))
                if fallback_key is None or key < fallback_key:
                    fallback_key, fallback = key, (cx, cy)
                if interactables is not None and nearest_interactable_id(cx, cy, interactables, max_distance) != target_id:
                    continue
                if best_key is None or key < best_key:
                    best_key, best = key, (cx, cy)
        return best if best is not None else fallback

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_walkable(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and cell not in self.blocked

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int(y // self.cell_size), int(x // self.cell_size)

    def center_of(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return col * self.cell_size + self.cell_size / 2, row * self.cell_size + self.cell_size / 2

    def nearest_walkable(self, x: float, y: float, max_radius: int = 12) -> tuple[int, int] | None:
        """The walkable cell closest to (x, y), searching outward ring by
        ring. Used to snap an actor's continuous position, or a point near
        a piece of furniture, onto the grid.
        """
        start = self.cell_of(x, y)
        if self.is_walkable(start):
            return start

        start_row, start_col = start
        for radius in range(1, max_radius + 1):
            candidates = []
            for row in range(start_row - radius, start_row + radius + 1):
                for col in range(start_col - radius, start_col + radius + 1):
                    if max(abs(row - start_row), abs(col - start_col)) != radius:
                        continue
                    if self.is_walkable((row, col)):
                        candidates.append((row, col))
            if candidates:
                return min(candidates, key=lambda c: math.dist(self.center_of(c), (x, y)))
        return None

    def nearest_approach_point(
        self, rect: pygame.Rect, max_distance: float, from_xy: tuple[float, float]
    ) -> tuple[float, float] | None:
        """A walkable cell within `max_distance` of `rect` - i.e. a spot an
        actor could stand at to interact with an object occupying `rect`.

        Ranked primarily by distance to `rect` itself (hugging the target's
        boundary), with distance to `from_xy` only as a tie-break. This
        matters when objects sit close together (e.g. a sofa and a side
        table): picking the point nearest to the *actor's* start position
        first can land closer to a neighboring object than to the intended
        target, and since "what am I in range of" only ever tracks a
        single nearest object, that neighbor would win instead.
        """
        search = rect.inflate(max_distance * 2, max_distance * 2)
        top_row, left_col = self.cell_of(search.left, search.top)
        bottom_row, right_col = self.cell_of(search.right, search.bottom)

        best = None
        best_key = None
        for row in range(max(0, top_row), min(self.rows - 1, bottom_row) + 1):
            for col in range(max(0, left_col), min(self.cols - 1, right_col) + 1):
                cell = (row, col)
                if not self.is_walkable(cell):
                    continue
                cx, cy = self.center_of(cell)
                closest_x = max(rect.left, min(cx, rect.right))
                closest_y = max(rect.top, min(cy, rect.bottom))
                dist_to_rect = math.hypot(cx - closest_x, cy - closest_y)
                if dist_to_rect > max_distance:
                    continue
                key = (dist_to_rect, math.dist((cx, cy), from_xy))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (cx, cy)
        return best

    def find_path(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]] | None:
        """A* from start to goal (world coordinates). Returns a list of
        cell-center world positions from the first step to the goal, or
        None if unreachable. Both endpoints are snapped onto the grid.
        """
        start = self.nearest_walkable(*start_xy)
        goal = self.nearest_walkable(*goal_xy)
        if start is None or goal is None:
            return None
        if start == goal:
            return [self.center_of(goal)]

        open_heap = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start: 0.0}

        def heuristic(cell: tuple[int, int]) -> float:
            return math.dist(cell, goal)

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct(came_from, current)

            row, col = current
            for drow, dcol, cost in _NEIGHBORS:
                neighbor = (row + drow, col + dcol)
                if not self.is_walkable(neighbor):
                    continue
                if drow != 0 and dcol != 0:
                    # Corner-cut guard: both orthogonal neighbors of a
                    # diagonal step must also be walkable.
                    if not self.is_walkable((row + drow, col)) or not self.is_walkable((row, col + dcol)):
                        continue

                tentative = g_score[current] + cost
                if tentative < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    heapq.heappush(open_heap, (tentative + heuristic(neighbor), neighbor))

        return None  # no path exists

    def _reconstruct(self, came_from: dict, current: tuple[int, int]) -> list[tuple[float, float]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [self.center_of(cell) for cell in path]


def build_nav_grid(level: Level, door_states: dict[str, str], cell_size: int = constants.NAV_GRID_SIZE) -> NavGrid:
    room_w, room_h = level.size
    rows = math.ceil(room_h / cell_size)
    cols = math.ceil(room_w / cell_size)

    blocked: set[tuple[int, int]] = set()

    def block_rect(rect: pygame.Rect) -> None:
        min_row = max(0, rect.top // cell_size)
        max_row = min(rows - 1, (rect.bottom - 1) // cell_size)
        min_col = max(0, rect.left // cell_size)
        max_col = min(cols - 1, (rect.right - 1) // cell_size)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                blocked.add((row, col))

    for wall in level.walls:
        block_rect(wall)

    for obj in level.objects:
        if not obj.solid:
            continue
        if obj.type == "door" and door_states.get(obj.id) == "open":
            continue  # open doors don't block navigation
        block_rect(obj.collision_rect)

    return NavGrid(cell_size=cell_size, rows=rows, cols=cols, blocked=blocked)


# M5E.4.3: destination-side traversal waypoints - see M5E.4.3_DOORWAY_
# TRAVERSAL_NAVIGATION_COMPLETION_SEMANTICS.md. `Level.room_id_at` returns
# None in the real ~40-60px band straddling a door's own collision rect
# (the rect sits *between* the two rooms it connects, not inside either -
# see `Level.room_id_at`'s own gap and `GameState._update_room_memory`'s
# docstring, which already documented this same gap for a different
# reason). A MOVE_TO that stops at the door's rect therefore never
# actually lands inside the destination room. This derives a real point
# *inside* the destination room instead, from existing geometry only -
# never a hand-authored per-door/per-level coordinate.
_CROSSING_PROBE_STEP = 8.0
_CROSSING_PROBE_MAX_DISTANCE = 160.0
_CROSSING_PROBE_DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
# `ActionExecutor._follow_path` (action_lifecycle.py) considers a waypoint
# reached once within `constants.WAYPOINT_ARRIVAL_DISTANCE` (6px) of it -
# a waypoint sitting exactly on the room boundary can therefore be
# "arrived at" up to 6px short, landing right back in the dead zone this
# module exists to clear. Found live by this milestone's own test suite
# (a bidirectional crossing test failed in exactly this way before this
# margin existed) - push the candidate point further into the destination
# room first, with real headroom over that tolerance.
_CROSSING_SAFETY_MARGIN = 24.0


def find_room_crossing_waypoint(
    level: Level,
    nav_grid: "NavGrid",
    door_rect: pygame.Rect,
    destination_room: str,
) -> tuple[float, float] | None:
    """Probes outward from `door_rect`'s center in the four cardinal
    directions until a point genuinely inside `destination_room` (per
    `Level.room_id_at`, the same authoritative room-membership check
    everything else in this project uses - never a new, second notion of
    "room") is found, then snaps it onto the walkable nav grid. Direction
    is discovered, not assumed, so this works for either door orientation
    (a narrow door crossed horizontally, a wide door crossed vertically)
    without knowing which in advance. Returns `None` (never a guess) if no
    probe within `_CROSSING_PROBE_MAX_DISTANCE` lands in the destination
    room or the nearest such point isn't walkable - callers must fall back
    to the existing generic approach-point behavior in that case."""
    cx, cy = door_rect.center
    for dx, dy in _CROSSING_PROBE_DIRECTIONS:
        distance = _CROSSING_PROBE_STEP
        while distance <= _CROSSING_PROBE_MAX_DISTANCE:
            probe_x, probe_y = cx + dx * distance, cy + dy * distance
            if level.room_id_at(probe_x, probe_y) == destination_room:
                deep_x = cx + dx * (distance + _CROSSING_SAFETY_MARGIN)
                deep_y = cy + dy * (distance + _CROSSING_SAFETY_MARGIN)
                # Only use the deeper point if it's still genuinely inside
                # the destination room (a small room could push it back
                # out the far side) - otherwise the shallower, already-
                # confirmed point is the safer choice.
                target_x, target_y = (
                    (deep_x, deep_y) if level.room_id_at(deep_x, deep_y) == destination_room else (probe_x, probe_y)
                )
                cell = nav_grid.nearest_walkable(target_x, target_y)
                return nav_grid.center_of(cell) if cell is not None else None
            distance += _CROSSING_PROBE_STEP
    return None
