"""Loads a level definition from JSON.

The room and its objects are data, not code. Adding or moving furniture
means editing data/level_01.json, not touching rendering or game logic.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pygame

from src.interactable import GameObject


@dataclass(frozen=True)
class Room:
    """A named area of the embassy (Milestone 4A). Purely static metadata -
    stable room IDs for M4B's dependency graph and M4C's per-actor
    knowledge (`known_rooms`, `current_room`) to key off later; nothing in
    M4A itself reads this beyond `Level.room_id_at` and the F3 debug
    overlay.
    """

    id: str
    label: str
    rect: pygame.Rect


@dataclass
class Level:
    id: str
    name: str
    size: tuple[int, int]
    player_spawn: tuple[float, float]
    white_spy_spawn: tuple[float, float]
    walls: list[pygame.Rect] = field(default_factory=list)
    objects: list[GameObject] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    # M5F.3A: traps present at game initialization, independent of either
    # spy (spec sections 1-7, 33) - generic level data, never a
    # Security-room-specific concept in Python. Each entry:
    # {"id", "target_id", "trap_type", "owner", "hidden"}; `owner` is
    # expected to be "ENVIRONMENT" but this field intentionally doesn't
    # validate that, since ownership meaning lives entirely in
    # `GameState.new_game`'s consumption of it, not in level loading.
    initial_traps: list[dict] = field(default_factory=list)

    def get_object(self, object_id: str) -> GameObject | None:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def solid_rects(self, door_states: dict[str, str] | None = None) -> list[pygame.Rect]:
        """All rects an actor collides with: walls plus solid objects.

        An open door doesn't block movement - the same rule
        navigation.build_nav_grid already applies when deciding which
        cells are walkable, so a door's passability agrees between
        physics and pathfinding. `door_states` is optional only so this
        method still works with no state at all (there are no doors, or
        the caller doesn't care); pass it whenever doors exist and any of
        them might be open (Milestone 4A's inter-room doors always do -
        M3's doors were dead ends nobody ever needed to walk through, so
        this never mattered before).
        """
        rects = list(self.walls)
        for obj in self.objects:
            if not obj.solid:
                continue
            if obj.type == "door" and door_states is not None and door_states.get(obj.id) == "open":
                continue
            rects.append(obj.collision_rect)
        return rects

    def interactables(self) -> list[GameObject]:
        return [obj for obj in self.objects if obj.interactable]

    def room_id_at(self, x: float, y: float) -> str | None:
        for room in self.rooms:
            if room.rect.collidepoint(x, y):
                return room.id
        return None

    def doors_between(self, room_a: str, room_b: str) -> list[str]:
        """Ids of the doors whose `connects` joins exactly these two rooms."""
        pair = {room_a, room_b}
        return [obj.id for obj in self.objects if obj.type == "door" and obj.connects is not None and set(obj.connects) == pair]

    def object_room(self, obj: GameObject) -> str | None:
        """The room an ordinary object stands in (by its footprint centre).
        Doors belong to no single room - use `connects` for them."""
        if obj.type == "door":
            return None
        return self.room_id_at(*obj.center)

    def room_exits(self, room_id: str) -> list[str]:
        """Doors between `room_id` and another room, in level order (exit_door has no `connects`)."""
        return [obj.id for obj in self.objects if obj.type == "door" and obj.connects is not None and room_id in obj.connects]


def load_level(path: str) -> Level:
    data = json.loads(Path(path).read_text())

    walls = [pygame.Rect(*w["rect"]) for w in data.get("walls", [])]

    objects = [
        GameObject(
            id=o["id"],
            type=o["type"],
            label=o.get("label", o["type"]),
            position=tuple(o["position"]),
            collision_size=tuple(o["collision_size"]),
            solid=o.get("solid", False),
            interactable=o.get("interactable", False),
            interaction=o.get("interaction"),
            message=o.get("message"),
            sprite=o.get("sprite"),
            sprite_open=o.get("sprite_open"),
            sprite_locked=o.get("sprite_locked"),
            contains=list(o.get("contains", [])),
            render_width=o.get("render_width"),
            render_height=o.get("render_height"),
            render_offset=tuple(o["render_offset"]) if "render_offset" in o else None,
            default_state=o.get("default_state", "closed"),
            connects=tuple(o["rooms"]) if "rooms" in o else None,
            lock_type=o.get("lock_type"),
            requires=o.get("requires"),
        )
        for o in data.get("objects", [])
    ]

    rooms = [Room(id=r["id"], label=r.get("label", r["id"]), rect=pygame.Rect(*r["rect"])) for r in data.get("rooms", [])]

    initial_traps = list(data.get("initial_traps", []))

    return Level(
        id=data["id"],
        name=data["name"],
        size=tuple(data["size"]),
        player_spawn=tuple(data["player_spawn"]),
        white_spy_spawn=tuple(data["white_spy_spawn"]),
        walls=walls,
        objects=objects,
        rooms=rooms,
        initial_traps=initial_traps,
    )
