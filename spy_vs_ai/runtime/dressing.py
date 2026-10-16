"""Room dressing (FINAL-2 gameplay polish, spec 3-6, 30, 32-34): decorative props, renderer-only.

Props never enter the world state: they are not level objects, not in the navigation grid, not in any observation,
and no action can target them (DECORATIVE only - the level's own objects stay the INTERACTIVE / MISSION-RELEVANT
ones). The runtime renderer draws them (rugs on the floor, everything else y-sorted with the furniture).

Placement is deterministic per level (`random.Random("dressing:<level>:<room>")`), wall-hugging, and avoids:
  * the doors' clearance zones (wall art: their larger visual exclusion zones, as the level's paintings) and the
    exit,
  * the level's objects (and the room around them a spy uses to reach them),
  * the routes spies walk - shortest grid paths between every doorway, every object's approach point, every spawn
    point and the exit, inside each room (widened by a spy's half width),
  * the spawn points, and each other.
A room gets props up to a density target (3-5 notable objects in a medium / large room, counting the level's own),
with a themed mix per room, so rooms do not all look alike. Some rooms stay sparse.

`validate(level, props)` re-checks the result on an offline copy of the level where every prop is a solid object:
the level's own layout validator (door clearance, overlaps), visual exclusion, and that every doorway, every
interactable object and the exit stay exactly as reachable as without props, with the doors as they start and all
open - so the dressing never implies a blocked route even though, in the game, props block nothing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable

import pygame

from src.interactable import GameObject

DECORATIVE = "DECORATIVE"


@dataclass(frozen=True)
class PropKind:
    family: str
    sprites: tuple[str, ...]
    height: int                 # drawn height (world px), bottom-anchored on the footprint
    footprint: tuple[int, int]  # (width, depth) on the floor
    edges: tuple[str, ...] = ("top", "left", "right", "bottom")
    layer: str = "y"            # "y": y-sorted with the furniture; "floor": under everything (rugs)


def _p(*names: str) -> tuple[str, ...]:
    return tuple(f"assets/props/{n}.png" for n in names)


KINDS: dict[str, PropKind] = {k.family: k for k in (
    PropKind("plant", _p("plant_01", "plant_02", "plant_03", "plant_04", "plant_05", "plant_06"), 66, (46, 26)),
    PropKind("lamp", _p("lamp_table_01", "lamp_table_03", "lamp_table_05", "lamp_table_07"), 62, (32, 18)),
    PropKind("crate", _p("crate_closed_01", "crate_closed_02", "crate_cross_01", "crate_marked_01", "covered_crate"),
             58, (54, 34), ("top", "left", "right")),
    PropKind("round_table", _p("round_table_bottle", "round_table_cloth_flowers", "round_table_paper"), 64, (56, 30)),
    PropKind("small_table", _p("small_table_01", "side_table_bottle", "side_table_drawer"), 58, (54, 28)),
    PropKind("bust", _p("bust_man", "bust_woman_front", "bust_profile", "statue_pedestal_01", "pedestal_column"), 82,
             (36, 22), ("top", "left", "right")),
    PropKind("statue", _p("statue_woman_full", "lion_statue"), 92, (48, 28), ("top", "left", "right")),
    PropKind("phone", _p("phone_pedestal_01"), 68, (36, 22), ("top", "left", "right")),
    PropKind("painting", _p("painting_castle", "painting_drapery", "painting_flowers", "map_frame", "portrait_man",
                            "painting_spy_portrait"), 60, (66, 6), ("top",)),
    PropKind("clock", _p("grandfather_clock"), 104, (34, 20), ("top",)),
    PropKind("coat_stand", _p("coat_stand"), 96, (32, 20), ("top", "left", "right")),
    PropKind("radio", _p("radio_cabinet"), 62, (62, 28), ("top", "left", "right")),
    PropKind("books", _p("book_stack"), 40, (38, 24)),
    PropKind("typewriter", _p("typewriter_table"), 72, (48, 28), ("top", "left", "right")),
    PropKind("waste_bin", _p("waste_bin"), 42, (32, 20)),
    PropKind("briefcase", _p("briefcase"), 38, (38, 16)),
    PropKind("rug", _p("rug_persian"), 112, (200, 112), ("centre",), "floor"),
)}

# a themed mix per room (matched on the room id), so rooms differ; anything else gets the hall mix
THEMES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("archives", "records"), ("crate", "books", "lamp", "plant", "typewriter", "waste_bin", "painting")),
    (("library",), ("books", "lamp", "plant", "clock", "round_table", "painting", "bust")),
    (("security",), ("crate", "radio", "waste_bin", "coat_stand", "clock", "books")),
    (("reception",), ("plant", "coat_stand", "phone", "round_table", "painting", "briefcase", "lamp", "rug")),
    (("lobby",), ("plant", "bust", "coat_stand", "painting", "round_table", "statue", "rug")),
    (("main_office", "office"), ("typewriter", "waste_bin", "coat_stand", "lamp", "plant", "clock", "painting",
                                  "briefcase", "radio")),
    (("ambassador",), ("clock", "bust", "painting", "plant", "round_table", "lamp", "radio", "rug")),
    (("corridor", "hall", "atrium", "gallery"), ("painting", "plant", "bust", "clock", "coat_stand", "small_table",
                                                 "statue", "rug")),
)
DEFAULT_THEME = ("painting", "plant", "lamp", "small_table", "clock")

WALL_GAP = 8            # px between the wall face and a prop's footprint
SLOT_STEP = 16
OBJECT_CLEARANCE = 30   # px kept free around a level object (its approach routes are kept free separately)
WALL_ART = ("painting", "clock")  # kept out of the doors' (larger) visual exclusion zones, like the level's paintings
DOOR_MARGIN = 8         # px added around a door's clearance zone for every other prop
PROP_GAP = 14
SPAWN_CLEARANCE = 56
ROUTE_MARGIN = 14       # px either side of a walked route's cells (a spy's half width)
MAX_RUGS_PER_LEVEL = 2


@dataclass(frozen=True)
class Prop:
    id: str
    family: str
    sprite: str
    room: str
    footprint: tuple[int, int, int, int]  # x, y, w, h (world px)
    height: int
    layer: str
    category: str = DECORATIVE

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(*self.footprint)

    @property
    def visual_rect(self) -> pygame.Rect:
        """Roughly where the sprite is drawn (bottom-anchored on the footprint)."""
        r = self.rect
        if self.layer == "floor":
            return r
        width = max(r.width, int(self.height * 0.9))
        return pygame.Rect(r.centerx - width // 2, r.bottom - self.height, width, self.height)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "family": self.family, "sprite": self.sprite, "room": self.room,
                "footprint": list(self.footprint), "height": self.height, "layer": self.layer,
                "category": self.category}


def theme_for(room_id: str) -> tuple[str, ...]:
    for keys, kinds in THEMES:
        if any(k in room_id for k in keys):
            return kinds
    return DEFAULT_THEME


def density_target(room_rect: pygame.Rect) -> int:
    """Notable objects a room should show (spec 4): 3 small, 4 medium, 5 large."""
    area = room_rect.width * room_rect.height
    return 3 if area < 110_000 else 4 if area < 170_000 else 5


def _doors(level) -> list[GameObject]:
    return [o for o in level.objects if o.type == "door"]


def _room_objects(level, room) -> list[GameObject]:
    return [o for o in level.objects if o.type != "door" and room.rect.collidepoint(o.center)]


def _object_look(obj: GameObject) -> pygame.Rect:
    """The object's footprint and roughly where its sprite stands above it."""
    r = obj.collision_rect
    h = int(obj.render_height or max(r.height, 48) * 1.3)
    return r.union(pygame.Rect(r.x, r.bottom - h, r.width, h))


def _all_open(level) -> dict[str, str]:
    return {o.id: "open" for o in _doors(level)}


def _start_states(level) -> dict[str, str]:
    return {o.id: o.default_state for o in _doors(level)}


def spawn_points(level, grid) -> list[tuple[float, float]]:
    """The level spawns and every room's live spawn point (src.spawns.room_spawn_point)."""
    from src.spawns import room_spawn_point
    probe = SimpleNamespace(level=level, nav_grid=grid)
    points = [tuple(level.player_spawn), tuple(level.white_spy_spawn)]
    for room in level.rooms:
        point = room_spawn_point(probe, room.id)
        if point is not None:
            points.append(point)
    return points


def route_rects(level) -> list[pygame.Rect]:
    """The cells spies walk: shortest grid paths, inside each room, between its doorways' crossing points, its
    objects' approach points, the spawn points in it and the exit - widened by ROUTE_MARGIN px."""
    from src.navigation import build_nav_grid, find_room_crossing_waypoint
    grid = build_nav_grid(level, _all_open(level))
    spawns = spawn_points(level, grid)
    interactables = [(o.id, o.collision_rect) for o in level.interactables()]
    cells: set[tuple[int, int]] = set()
    for room in level.rooms:
        points: list[tuple[float, float]] = []
        for door in _doors(level):
            if door.connects and room.id in door.connects:
                p = find_room_crossing_waypoint(level, grid, door.collision_rect, room.id)
                if p is not None:
                    points.append(p)
            elif door.connects is None and room.rect.inflate(140, 140).colliderect(door.collision_rect):
                points.append(door.collision_rect.center)  # the exit, from its room
        center = room.rect.center
        for obj in _room_objects(level, room):
            if obj.interactable:
                p = grid.reachable_approach_point(obj.id, obj.collision_rect, center, interactables)
                if p is not None:
                    points.append(p)
        points += [p for p in spawns if room.rect.collidepoint(p)]
        points.append(center)
        for i, a in enumerate(points):
            for b in points[i + 1:]:
                path = grid.find_path(a, b) or []
                for x, y in [a, *path, b]:
                    cells.add(grid.cell_of(x, y))
    size = grid.cell_size
    out = []
    for row, col in cells:
        out.append(pygame.Rect(col * size, row * size, size, size).inflate(2 * ROUTE_MARGIN, 2 * ROUTE_MARGIN))
    return out


def _slots(room_rect: pygame.Rect, kind: PropKind) -> list[tuple[str, pygame.Rect]]:
    w, d = kind.footprint
    r = room_rect
    out: list[tuple[str, pygame.Rect]] = []
    if "centre" in kind.edges:
        out.append(("centre", pygame.Rect(r.centerx - w // 2, r.centery - d // 2, w, d)))
        return out
    for edge in kind.edges:
        if edge in ("top", "bottom"):
            y = r.top + WALL_GAP if edge == "top" else r.bottom - WALL_GAP - d
            for x in range(r.left + WALL_GAP, r.right - WALL_GAP - w + 1, SLOT_STEP):
                out.append((edge, pygame.Rect(x, y, w, d)))
        else:  # against a side wall (every sprite faces the viewer: the footprint stays the same)
            x = r.left + WALL_GAP if edge == "left" else r.right - WALL_GAP - w
            for y in range(r.top + 40, r.bottom - WALL_GAP - d + 1, SLOT_STEP):
                out.append((edge, pygame.Rect(x, y, w, d)))
    return out


def _sprite_for(kind: PropKind, rng: random.Random, used: set[str]) -> str:
    fresh = [s for s in kind.sprites if s not in used]
    return rng.choice(fresh or list(kind.sprites))


def plan(level, *, routes: list[pygame.Rect] | None = None) -> list[Prop]:
    """The decorative props for `level` (deterministic: same level file, same props)."""
    from src.layout_validation import door_clearance_rects, door_visual_exclusion_rects
    from src.navigation import build_nav_grid
    routes = route_rects(level) if routes is None else routes
    grid = build_nav_grid(level, _all_open(level))
    spawns = spawn_points(level, grid)
    exit_zone = [d.collision_rect.inflate(120, 120) for d in _doors(level) if d.connects is None]
    zones = {"wall_art": [z for door in _doors(level) for z in door_visual_exclusion_rects(door)] + exit_zone,
             "other": [z.inflate(2 * DOOR_MARGIN, 2 * DOOR_MARGIN) for door in _doors(level)
                       for z in door_clearance_rects(door)] + exit_zone}
    props: list[Prop] = []
    used_sprites: set[str] = set()
    has_rug = {r.id for r in level.rooms for o in _room_objects(level, r) if o.type == "rug"}
    for room in level.rooms:
        rng = random.Random(f"dressing:{level.id}:{room.id}")
        objects = _room_objects(level, room)
        notable = [o for o in objects if o.type != "rug"]
        # the level's own objects are small next to a room: they count, but a room still gets at least two props
        # (one in four rooms one fewer - some rooms stay sparse on purpose)
        jitter = rng.choice((0, 0, 0, 1))
        wanted = max(2, min(4, density_target(room.rect) + 2 - len(notable))) - jitter
        blockers = [o.collision_rect.inflate(OBJECT_CLEARANCE * 2, OBJECT_CLEARANCE * 2) for o in notable]
        looks = [_object_look(o).inflate(8, 8) for o in notable]
        theme = list(theme_for(room.id))
        rng.shuffle(theme)
        placed_here: list[Prop] = []
        counts: dict[str, int] = {}
        for family in theme * 2:
            if len(placed_here) >= wanted:
                break
            kind = KINDS[family]
            if family == "rug" and (room.id in has_rug or counts.get("rug")
                                    or sum(p.family == "rug" for p in props) >= MAX_RUGS_PER_LEVEL):
                continue  # one rug design: a few rooms only, or every room looks the same
            if counts.get(family, 0) >= (1 if family in ("clock", "rug", "statue", "phone", "radio", "typewriter") else 2):
                continue
            slots = _slots(room.rect, kind)
            rng.shuffle(slots)
            # corners first: props read best against two walls, and keep the middle of a wall free
            slots.sort(key=lambda s: -_corner_score(room.rect, s[1]))
            for _edge, rect in slots:
                if kind.layer == "floor":
                    ok = not any(rect.colliderect(o.collision_rect) for o in objects if o.type == "rug")
                else:
                    exclusions = zones["wall_art" if family in WALL_ART else "other"]
                    ok = _fits(rect, kind, room.rect, blockers, looks, exclusions, routes, spawns,
                               [p.rect.inflate(PROP_GAP * 2, PROP_GAP * 2) for p in props if p.layer != "floor"],
                               [p.visual_rect.inflate(8, 8) for p in props if p.layer != "floor"], level)
                if not ok:
                    continue
                sprite = _sprite_for(kind, rng, used_sprites)
                used_sprites.add(sprite)
                prop = Prop(id=f"prop_{room.id}_{len(placed_here) + 1:02d}", family=family, sprite=sprite,
                            room=room.id, footprint=(rect.x, rect.y, rect.width, rect.height), height=kind.height,
                            layer=kind.layer)
                placed_here.append(prop)
                props.append(prop)
                counts[family] = counts.get(family, 0) + 1
                break
    report = validate(level, props, routes=routes)
    if not report["ok"]:  # drop whatever breaks a check (never expected; kept as a guarantee)
        props = [p for p in props if p.id not in set(report["offending"])]
    return props


def _corner_score(room_rect: pygame.Rect, rect: pygame.Rect) -> float:
    dx = min(rect.left - room_rect.left, room_rect.right - rect.right)
    dy = min(rect.top - room_rect.top, room_rect.bottom - rect.bottom)
    return -min(dx, dy) - 0.25 * max(dx, dy)


def _fits(rect, kind, room_rect, blockers, looks, exclusions, routes, spawns, others, other_looks, level) -> bool:
    if not room_rect.contains(rect):
        return False
    visual = Prop("p", kind.family, "", "", (rect.x, rect.y, rect.width, rect.height), kind.height, kind.layer).visual_rect
    if any(rect.colliderect(b) for b in blockers) or any(visual.colliderect(lk) for lk in looks):
        return False
    if any(rect.colliderect(z) or visual.colliderect(z) for z in exclusions):
        return False
    if any(rect.colliderect(r) for r in routes):
        return False
    if any(math.dist(p, rect.center) < SPAWN_CLEARANCE + max(rect.width, rect.height) / 2 for p in spawns):
        return False
    if any(rect.colliderect(o) for o in others) or any(visual.colliderect(o) for o in other_looks):
        return False  # props never stand in front of each other
    return not any(rect.colliderect(w) for w in level.walls)


# -- validation (spec 30) ------------------------------------------------------------------------------------

def _as_object(prop: Prop) -> GameObject:
    x, y, w, h = prop.footprint
    return GameObject(id=prop.id, type="prop", label=prop.family, position=(x, y), collision_size=(w, h), solid=True,
                      interactable=False, interaction=None, message=None)


def _reachability(level, door_states: dict[str, str]) -> dict[str, bool]:
    """What a spy can reach from either level spawn: each doorway (both sides), each interactable object and the
    exit (its approach)."""
    from src.navigation import build_nav_grid, find_room_crossing_waypoint
    grid = build_nav_grid(level, door_states)
    starts = [tuple(level.player_spawn), tuple(level.white_spy_spawn)]
    reach = set().union(*(grid.reachable_cells(s) for s in starts))
    out: dict[str, bool] = {}
    for obj in level.objects:
        if obj.type == "door" and obj.connects:
            for room in obj.connects:
                p = find_room_crossing_waypoint(level, grid, obj.collision_rect, room)
                out[f"door:{obj.id}:{room}"] = p is not None and grid.cell_of(*p) in reach
        if obj.interactable:
            out[f"object:{obj.id}"] = any(grid.has_reachable_approach(obj.collision_rect, s) for s in starts)
    return out


def validate(level, props: Iterable[Prop], *, routes: list[pygame.Rect] | None = None) -> dict[str, Any]:
    from src.layout_validation import door_clearance_rects, door_visual_exclusion_rects, validate_level
    from src.navigation import build_nav_grid
    props = list(props)
    solid = [p for p in props if p.layer != "floor"]
    dressed = copy.copy(level)
    dressed.objects = list(level.objects) + [_as_object(p) for p in solid]
    problems: list[str] = []
    offending: set[str] = set()
    prop_ids = {p.id for p in solid}
    for error in validate_level(dressed):
        if error.object_id in prop_ids or any(pid in error.reason for pid in prop_ids):
            problems.append(f"{error.object_id}: {error.reason}")
            offending.add(error.object_id)
            offending.update(pid for pid in prop_ids if pid in error.reason)
    visual_zones = [(d.id, z) for d in _doors(level) for z in door_visual_exclusion_rects(d)]
    clearance = [(d.id, z) for d in _doors(level) for z in door_clearance_rects(d)]
    for p in solid:
        for door_id, z in (visual_zones if p.family in WALL_ART else clearance):
            if p.rect.colliderect(z) or p.visual_rect.colliderect(z):
                problems.append(f"{p.id}: in the {'visual exclusion' if p.family in WALL_ART else 'clearance'} zone "
                                f"of {door_id}")
                offending.add(p.id)
        room = next((r for r in level.rooms if r.id == p.room), None)
        if room is None or not room.rect.contains(p.rect) or any(p.rect.colliderect(w) for w in level.walls):
            problems.append(f"{p.id}: not inside its room")
            offending.add(p.id)
        for o in level.objects:
            if o.type not in ("door", "rug") and p.rect.colliderect(o.collision_rect):
                problems.append(f"{p.id}: overlaps {o.id}")
                offending.add(p.id)
    grid = build_nav_grid(level, _all_open(level))
    for point in spawn_points(level, grid):
        for p in solid:
            if p.rect.inflate(SPAWN_CLEARANCE, SPAWN_CLEARANCE).collidepoint(point):
                problems.append(f"{p.id}: on a spawn point {tuple(round(v) for v in point)}")
                offending.add(p.id)
    if routes:
        for p in solid:
            if any(p.rect.colliderect(r) for r in routes):
                problems.append(f"{p.id}: on a walked route")
                offending.add(p.id)
    reach: dict[str, Any] = {}
    exit_ok = True
    for name, states in (("doors as they start", _start_states(level)), ("all doors open", _all_open(level))):
        before, after = _reachability(level, states), _reachability(dressed, states)
        lost = sorted(k for k, ok in before.items() if ok and not after.get(k))
        reach[name] = {"checked": len(before), "reachable": sum(before.values()),
                       "reachable_with_props": sum(1 for k, ok in before.items() if ok and after.get(k)), "lost": lost}
        if lost:
            problems.append(f"{name}: no longer reachable: {', '.join(lost)}")
        if name == "all doors open":
            exit_ok = bool(after.get("object:exit_door"))
    if not exit_ok:
        problems.append("the exit is not reachable with every door open")
    return {"ok": not problems, "problems": problems, "offending": sorted(offending), "props": len(props),
            "reachability": reach, "exit_reachable": exit_ok,
            "rooms": {r.id: sum(1 for p in props if p.room == r.id) for r in level.rooms}}


# -- per-level cache ---------------------------------------------------------------------------------------

_CACHE: dict[str, list[Prop]] = {}


def level_key(level) -> str:
    body = json.dumps({"id": level.id, "size": list(level.size), "walls": [list(w) for w in level.walls],
                       "objects": [[o.id, o.type, list(o.position), list(o.collision_size), o.solid, o.interactable]
                                   for o in level.objects],
                       "rooms": [[r.id, list(r.rect)] for r in level.rooms]}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def props_for(level) -> list[Prop]:
    key = level_key(level)
    if key not in _CACHE:
        _CACHE[key] = plan(level)
    return _CACHE[key]


def summary(level, props: list[Prop]) -> dict[str, Any]:
    by_room = {}
    for room in level.rooms:
        own = [o for o in _room_objects(level, room) if o.type != "rug"]
        mine = [p for p in props if p.room == room.id]
        by_room[room.id] = {"level_objects": len(own), "props": len(mine), "families": [p.family for p in mine],
                            "target": density_target(room.rect)}
    return by_room


__all__ = ["Prop", "PropKind", "KINDS", "THEMES", "plan", "validate", "props_for", "route_rects", "summary",
           "spawn_points", "density_target", "theme_for", "DECORATIVE", "level_key"]
