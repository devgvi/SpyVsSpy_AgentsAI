"""Exit geometry audit + live-only repair (bugfix live match flow V3, Part D).

Root cause found in data/level_01.json: `exit_door` (960,1332,48,100) is
drawn ON TOP of `wall_ambassador_office_4` (960,1232,48,300) - the east wall
of the ambassador office is one solid rect, so opening the exit door never
opened a passage and a spy that escaped stayed inside. Meanwhile the WEST
wall of the same room (`wall_ambassador_office_3_a` / `_3_b`) has a 100 px
gap at y 1332-1432 with no door in it: a hole in the building leading
nowhere.

The level file itself is left as it is: every stored M5/M6 replay and the
zero-call labs re-simulate on that exact geometry. The live game calls
`attach_exit_to_perimeter` on its own freshly loaded Level instead:

* walls overlapping the exit door are cut around it (a real doorway, closed
  by the door while it is closed - door collision is unchanged);
* a gap between two collinear wall segments that no door covers is sealed.

`exit_sides` / `exterior_point` give the inside room, the outside side and
the point just outside the doorway an escaping spy walks to.
"""
from __future__ import annotations

from typing import Any

import pygame

EXIT_ID = "exit_door"
EXTERIOR_DEPTH = 56  # px beyond the door's outer edge: the small exterior exit zone
_PROBE = 24


def _exit_rect(level) -> pygame.Rect | None:
    obj = level.get_object(EXIT_ID)
    return obj.collision_rect.copy() if obj is not None else None


def _vertical(rect: pygame.Rect) -> bool:
    return rect.height > rect.width  # the door sits in a vertical (east/west) wall


def exit_sides(level) -> dict[str, Any] | None:
    """Which side of the exit is a room (inside) and which is not (outside)."""
    rect = _exit_rect(level)
    if rect is None:
        return None
    if _vertical(rect):
        a = (rect.left - _PROBE, rect.centery)
        b = (rect.right + _PROBE, rect.centery)
    else:
        a = (rect.centerx, rect.top - _PROBE)
        b = (rect.centerx, rect.bottom + _PROBE)
    room_a, room_b = level.room_id_at(*a), level.room_id_at(*b)
    if (room_a is None) == (room_b is None):
        return {"inside_room": room_a or room_b, "outside_point": None, "on_perimeter": False}
    inside, outside = (room_a, b) if room_a is not None else (room_b, a)
    return {"inside_room": inside, "outside_probe": outside, "on_perimeter": True}


def exterior_point(level) -> tuple[float, float] | None:
    """Centre of the small exterior exit zone just outside the doorway."""
    rect, sides = _exit_rect(level), exit_sides(level)
    if rect is None or not sides or not sides.get("on_perimeter"):
        return None
    ox, oy = sides["outside_probe"]
    if _vertical(rect):
        x = rect.right + EXTERIOR_DEPTH if ox > rect.centerx else rect.left - EXTERIOR_DEPTH
        return float(x), float(rect.centery)
    y = rect.bottom + EXTERIOR_DEPTH if oy > rect.centery else rect.top - EXTERIOR_DEPTH
    return float(rect.centerx), float(y)


def exterior_zone(level) -> pygame.Rect | None:
    point = exterior_point(level)
    rect = _exit_rect(level)
    if point is None or rect is None:
        return None
    zone = pygame.Rect(0, 0, EXTERIOR_DEPTH * 2, max(rect.height, rect.width))
    if not _vertical(rect):
        zone.size = (max(rect.height, rect.width), EXTERIOR_DEPTH * 2)
    zone.center = (int(point[0]), int(point[1]))
    return zone


def walls_blocking_exit(level) -> list[pygame.Rect]:
    rect = _exit_rect(level)
    return [w for w in level.walls if rect is not None and w.colliderect(rect)]


def _door_covers(level, gap: pygame.Rect) -> bool:
    return any(obj.type == "door" and obj.collision_rect.colliderect(gap) for obj in level.objects)


def unsealed_perimeter_gaps(level, max_gap: int = 200) -> list[pygame.Rect]:
    """Gaps between two collinear wall segments that no door fills."""
    gaps: list[pygame.Rect] = []
    walls = sorted({(w.x, w.y, w.w, w.h) for w in level.walls})
    vertical: dict[tuple[int, int], list[tuple[int, int]]] = {}
    horizontal: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y, w, h in walls:
        if h > w:
            vertical.setdefault((x, w), []).append((y, y + h))
        else:
            horizontal.setdefault((y, h), []).append((x, x + w))
    for (x, w), spans in vertical.items():
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:]):
            if 0 < start - end <= max_gap:
                gap = pygame.Rect(x, end, w, start - end)
                if not _door_covers(level, gap) and not _covered_by_wall(level, gap):
                    gaps.append(gap)
    for (y, h), spans in horizontal.items():
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:]):
            if 0 < start - end <= max_gap:
                gap = pygame.Rect(end, y, start - end, h)
                if not _door_covers(level, gap) and not _covered_by_wall(level, gap):
                    gaps.append(gap)
    return gaps


def _covered_by_wall(level, gap: pygame.Rect) -> bool:
    return any(w.contains(gap) for w in level.walls)


def audit(level) -> dict[str, Any]:
    rect = _exit_rect(level)
    sides = exit_sides(level) or {}
    return {
        "exit_rect": list(rect) if rect is not None else None,
        "inside_room": sides.get("inside_room"),
        "on_perimeter": bool(sides.get("on_perimeter")),
        "walls_blocking_doorway": [list(w) for w in walls_blocking_exit(level)],
        "unsealed_perimeter_gaps": [list(g) for g in unsealed_perimeter_gaps(level)],
        "exterior_point": exterior_point(level),
    }


def attach_exit_to_perimeter(level) -> dict[str, Any]:
    """Live-only repair of the loaded Level (mutates `level.walls`)."""
    before = audit(level)
    rect = _exit_rect(level)
    if rect is None:
        return {"before": before, "after": before, "changed": False}
    new_walls: list[pygame.Rect] = []
    for wall in level.walls:
        if not wall.colliderect(rect):
            new_walls.append(wall)
            continue
        if _vertical(rect):
            if wall.top < rect.top:
                new_walls.append(pygame.Rect(wall.x, wall.top, wall.w, rect.top - wall.top))
            if wall.bottom > rect.bottom:
                new_walls.append(pygame.Rect(wall.x, rect.bottom, wall.w, wall.bottom - rect.bottom))
        else:
            if wall.left < rect.left:
                new_walls.append(pygame.Rect(wall.left, wall.y, rect.left - wall.left, wall.h))
            if wall.right > rect.right:
                new_walls.append(pygame.Rect(rect.right, wall.y, wall.right - rect.right, wall.h))
    level.walls[:] = new_walls
    for gap in unsealed_perimeter_gaps(level):
        level.walls.append(gap)
    after = audit(level)
    return {"before": before, "after": after, "changed": before != after}
