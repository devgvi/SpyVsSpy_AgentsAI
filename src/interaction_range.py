"""The one interaction-range rule (BUGFIX navigation/exploration, §7).

An actor can interact with an object when the distance from the actor's
feet point (x, y) - which is also the centre of its 28x28 collision box -
to the nearest point of the object's *collision* rect (never its sprite) is
at most `constants.INTERACTION_DISTANCE` (70 px), and that object is the
single nearest such interactable. Ties go to the object listed later in the
level (the historical `<=` comparison). Walls are not considered.

`GameState._find_nearest_interactable` (the engine, authoritative) and the
navigation approach-point choice both call these functions, so the place a
MOVE_TO aims for is judged by exactly the rule the engine applies afterwards.
"""
from __future__ import annotations

import math
from typing import Iterable

from src import constants


def distance_to_rect(x: float, y: float, rect) -> float:
    closest_x = max(rect.left, min(x, rect.right))
    closest_y = max(rect.top, min(y, rect.bottom))
    return math.hypot(x - closest_x, y - closest_y)


def nearest_interactable_id(
    x: float, y: float, candidates: Iterable[tuple[str, object]], reach: float = constants.INTERACTION_DISTANCE
) -> str | None:
    """`candidates`: (object_id, collision_rect) for every interactable, in level order."""
    nearest_id = None
    nearest_dist = reach
    for object_id, rect in candidates:
        dist = distance_to_rect(x, y, rect)
        if dist <= nearest_dist:
            nearest_dist = dist
            nearest_id = object_id
    return nearest_id


def range_margin(x: float, y: float, target_id: str, candidates: Iterable[tuple[str, object]]) -> float:
    """How much closer the target is than the next nearest interactable (inf if alone)."""
    target = math.inf
    other = math.inf
    for object_id, rect in candidates:
        dist = distance_to_rect(x, y, rect)
        if object_id == target_id:
            target = dist
        else:
            other = min(other, dist)
    return other - target
