"""Validates a level's physical layout - Mileston_4A2.md's corrective
milestone over M4A/M4A.1. Two checks, both grounded in the same
`collision_rect`/`position` data physics, navigation and rendering already
share (section 3's "one coherent world model" - nothing here is a second,
approximate geometry):

  1. no blocking (solid) object may overlap a door's doorway clearance
     zone - the protected rectangle either side of every opening (section
     8) - and a painting specifically must never sit in one even though it
     doesn't physically block (section 15);
  2. no two blocking objects may overlap each other (section 19).

This runs once at world creation (see `Game.__init__`), not per frame -
section 37 is explicit that layout validation belongs at level load, not
in the game loop. This project has no separate prod/dev build, so "fail
loudly in debug/development" (section 17) means always: `log_validation`
prints a `[LAYOUT]` PASS/ERROR line per room and raises if anything failed,
so a broken level is caught at startup, not discovered mid-playtest.
"""

from dataclasses import dataclass

import pygame

from src.interactable import GameObject
from src.level import Level

CLEARANCE_PAD = 32  # added to the door's own opening width/height
CLEARANCE_DEPTH = 96  # how far the clearance zone extends into each room

# M4A.3 (M4A_3_LIVE_PLAY_DIAGNOSTICS_DOOR_POLISH.md sections 18-22):
# composition, not collision - deliberately larger than the physical
# clearance zone above. A large wall decoration could technically clear
# CLEARANCE_PAD/CLEARANCE_DEPTH (no collision violation) while still
# visually crowding the doorway; this zone exists to catch that even
# though nothing here ever touches collision/navigation.
VISUAL_EXCLUSION_PAD = 60
VISUAL_EXCLUSION_DEPTH = 140


def _door_zone_rects(door: GameObject, pad: float, depth: float) -> list[pygame.Rect]:
    """The two rectangles either side of `door`'s opening, sized by `pad`
    (added to the opening's own width/height) and `depth` (how far into
    each room). Orientation is read straight from the door's own
    collision_size (wide-short -> horizontal wall, tall-narrow -> vertical
    wall) - the same authoritative rect physics/navigation/rendering
    already use for this door, not a separately-tracked approximation.
    """
    gap = door.collision_rect
    horizontal = gap.width > gap.height

    if horizontal:
        cw = gap.width + pad
        cx = gap.centerx - cw / 2
        return [
            pygame.Rect(cx, gap.top - depth, cw, depth),
            pygame.Rect(cx, gap.bottom, cw, depth),
        ]

    ch = gap.height + pad
    cy = gap.centery - ch / 2
    return [
        pygame.Rect(gap.left - depth, cy, depth, ch),
        pygame.Rect(gap.right, cy, depth, ch),
    ]


def door_clearance_rects(door: GameObject) -> list[pygame.Rect]:
    """The two protected (collision/navigation) rectangles either side of
    `door`'s opening - see `_door_zone_rects`."""
    return _door_zone_rects(door, CLEARANCE_PAD, CLEARANCE_DEPTH)


def door_visual_exclusion_rects(door: GameObject) -> list[pygame.Rect]:
    """The two, larger, visual-only rectangles either side of `door`'s
    opening - composition, not collision. See `VISUAL_EXCLUSION_PAD`.
    """
    return _door_zone_rects(door, VISUAL_EXCLUSION_PAD, VISUAL_EXCLUSION_DEPTH)


@dataclass(frozen=True)
class LayoutError:
    room_id: str
    object_id: str
    reason: str


def validate_level(level: Level) -> list[LayoutError]:
    """Returns every layout error found (empty list = clean). Rugs and
    other floor decoration are explicitly allowed to overlap a clearance
    zone (section 9) and are never checked here.

    A painting gets two separate checks, deliberately not merged into
    one: the ordinary (small) clearance zone, same as any blocking object,
    even though `solid=False` (M4A.2's rule, from
    Mileston_4A2.md section 15) - and the larger visual exclusion zone
    (M4A.3's rule, composition rather than collision - a painting could
    clear the first check while still visually crowding the doorway).
    Reported separately so a failure names which rule actually tripped.
    """
    errors: list[LayoutError] = []
    doors = [o for o in level.objects if o.type == "door"]
    blocking = [o for o in level.objects if o.solid and o.type != "door"]
    paintings = [o for o in level.objects if o.type == "painting"]

    def room_of(obj: GameObject) -> str:
        return level.room_id_at(*obj.center) or "?"

    for door in doors:
        for clearance in door_clearance_rects(door):
            for obj in blocking:
                if clearance.colliderect(obj.collision_rect):
                    errors.append(LayoutError(room_of(obj), obj.id, f"overlaps doorway clearance for {door.id}"))
            for obj in paintings:
                if clearance.colliderect(obj.collision_rect):
                    errors.append(LayoutError(room_of(obj), obj.id, f"painting overlaps doorway clearance for {door.id}"))

        for exclusion in door_visual_exclusion_rects(door):
            for obj in paintings:
                if exclusion.colliderect(obj.collision_rect):
                    errors.append(
                        LayoutError(room_of(obj), obj.id, f"painting overlaps visual exclusion zone for {door.id}")
                    )

    for i, a in enumerate(blocking):
        for b in blocking[i + 1 :]:
            if a.collision_rect.colliderect(b.collision_rect):
                errors.append(LayoutError(room_of(a), a.id, f"collision overlaps {b.id}"))

    return errors


def log_validation(level: Level) -> None:
    errors = validate_level(level)
    by_room: dict[str, list[LayoutError]] = {room.id: [] for room in level.rooms}
    for error in errors:
        by_room.setdefault(error.room_id, []).append(error)

    for room in level.rooms:
        room_errors = by_room.get(room.id, [])
        if room_errors:
            for error in room_errors:
                print(f"[LAYOUT ERROR] room={error.room_id} object={error.object_id} reason={error.reason}")
        else:
            print(f"[LAYOUT] {room.label}: PASS")

    if errors:
        raise AssertionError(f"{len(errors)} layout validation error(s) - see [LAYOUT ERROR] lines above")
