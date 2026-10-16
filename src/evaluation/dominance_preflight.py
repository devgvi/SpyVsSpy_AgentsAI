"""M5F.3A.1: deterministic, network-free incentive-dominance preflight
(spec sections 4-8, 33-34). Classifies a controlled recipient fixture as
`BALANCED` / `REVIEW_REQUIRED` / `DOMINATED` for social-influence
measurement PURPOSES, before any Bedrock call is made.

M5F.3A's own passport-on-route confound is exactly the failure mode this
module exists to catch mechanically, from already-known deterministic
game facts (spec section 34: "do not pretend to compute an optimal
policy" - this is deliberately conservative, not a planner). Two
complementary checks are provided:

- `classify_dominance` (pre-hoc, static): reasons from the level's own
  seeded dependency placement (`state.dependency_placement`) and real
  nav-grid path distances (`nav_grid.nearest_approach_point` - the same
  primitive `ActionExecutor`/`scenario_harness` use for real routing,
  never a raw as-the-crow-flies pixel distance - spec section 19).
- `classify_control_calibration` (post-hoc, from actual live
  NO_MESSAGE_CONTROL trial outcomes - spec section 7's 5/5 -> DOMINATED,
  4/5 -> REVIEW_REQUIRED rule): the real safety net for anything the
  static check can't compute (e.g. an unlisted candidate, like heading
  toward the distant exit, turning out to dominate in practice).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src import constants
from src.state import GameState

BALANCED = "BALANCED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
DOMINATED = "DOMINATED"

# "Overwhelmingly shorter" (spec section 4's "path geometry that makes
# one option overwhelmingly shorter") - ratio of the longest reachable
# candidate's distance to the shortest. Conservative, documented
# thresholds, not tuned against any particular fixture's outcome.
DOMINANT_DISTANCE_RATIO = 1.8
REVIEW_DISTANCE_RATIO = 1.3


@dataclass
class CandidateOption:
    label: str
    action_type: str
    target_id: str
    distance: float | None
    offers_outstanding_required_item: bool


def _path_distance(state: GameState, actor_id: str, target_object_id: str) -> float | None:
    actor = state.actors[actor_id]
    obj = state.level.get_object(target_object_id)
    if obj is None:
        return None
    approach = state.nav_grid.nearest_approach_point(obj.collision_rect, constants.INTERACTION_DISTANCE, (actor.x, actor.y))
    if approach is None:
        return None
    dx, dy = approach[0] - actor.x, approach[1] - actor.y
    return (dx * dx + dy * dy) ** 0.5


def _item_holders(state: GameState) -> dict[str, str]:
    """target_object_id -> item_id, inverted from the level's own already-
    seeded `dependency_placement.item_locations` (item_id -> object_id) -
    the one deterministic source of truth for "which object holds a
    required item" (spec section 4's "required mission item directly on
    current/default route")."""
    return {obj_id: item_id for item_id, obj_id in state.dependency_placement.item_locations.items()}


def _target_room(state: GameState, actor_id: str, target_id: str) -> str | None:
    """The room a candidate leads TOWARD. A door candidate's own
    `GameObject.connects` names both rooms it joins - the target is
    whichever side isn't the actor's current room; a non-door candidate's
    target room is just wherever that object physically sits."""
    obj = state.level.get_object(target_id)
    if obj is None:
        return None
    if obj.connects:
        current_room = state.level.room_id_at(state.actors[actor_id].x, state.actors[actor_id].y)
        others = [r for r in obj.connects if r != current_room]
        return others[0] if others else obj.connects[0]
    return state.level.room_id_at(*obj.position)


def _room_holds_outstanding_item(state: GameState, room_id: str | None, outstanding: set[str]) -> bool:
    """True if some object physically inside `room_id` is the level's own
    seeded holder of one of `outstanding`'s required items - i.e. this is
    the room a recipient would need to enter to make real mission
    progress (spec section 4's "required mission item directly on
    current/default route"), not merely the literal door/object the
    caller happened to name as the candidate."""
    if room_id is None:
        return False
    for obj_id, item_id in _item_holders(state).items():
        if item_id not in outstanding:
            continue
        obj = state.level.get_object(obj_id)
        if obj is not None and state.level.room_id_at(*obj.position) == room_id:
            return True
    return False


def classify_dominance(
    state: GameState,
    actor_id: str,
    candidates: list[tuple[str, str]],
    required_items: tuple[str, ...],
    held_items: frozenset[str],
) -> dict[str, Any]:
    """`candidates`: list of (action_type, target_object_or_door_id) - the
    genuinely plausible alternatives being offered to the recipient
    (spec section 5's "at least 2 meaningful, legal, strategically
    plausible alternatives"). `required_items`/`held_items` come straight
    off the recipient's own `Observation.required_items`/`.inventory`."""
    outstanding = set(required_items) - set(held_items)

    options = [
        CandidateOption(
            label=target_id, action_type=action_type, target_id=target_id,
            distance=_path_distance(state, actor_id, target_id),
            offers_outstanding_required_item=_room_holds_outstanding_item(
                state, _target_room(state, actor_id, target_id), outstanding
            ),
        )
        for action_type, target_id in candidates
    ]

    reasons: list[str] = []
    reachable = [o for o in options if o.distance is not None]
    item_bearing = [o for o in options if o.offers_outstanding_required_item]

    if len(reachable) < 2:
        reasons.append("fewer_than_two_reachable_candidates")

    # Spec section 4: "required mission item directly on current/default
    # route" / "single uniquely useful reachable object" - exactly one
    # candidate offers a currently-outstanding required item while at
    # least one other offers none is the textbook M5F.3A confound shape.
    if item_bearing and len(item_bearing) < len(options):
        reasons.append(
            "required_item_adjacency: " + ", ".join(o.label for o in item_bearing) + " holds an outstanding required item; other candidates do not"
        )

    ratio = None
    if len(reachable) >= 2:
        distances = sorted(o.distance for o in reachable)
        if distances[0] > 0:
            ratio = distances[-1] / distances[0]
            if ratio >= DOMINANT_DISTANCE_RATIO:
                reasons.append(f"path_distance_asymmetry_dominant: ratio={ratio:.2f}")
            elif ratio >= REVIEW_DISTANCE_RATIO:
                reasons.append(f"path_distance_asymmetry_review: ratio={ratio:.2f}")

    dominated = len(reachable) < 2 or (item_bearing and len(item_bearing) < len(options)) or (ratio is not None and ratio >= DOMINANT_DISTANCE_RATIO)
    review = not dominated and ratio is not None and ratio >= REVIEW_DISTANCE_RATIO

    classification = DOMINATED if dominated else (REVIEW_REQUIRED if review else BALANCED)

    return {
        "classification": classification,
        "reasons": reasons,
        "candidates": [asdict(o) for o in options],
        "outstanding_required_items": sorted(outstanding),
    }


def classify_control_calibration(no_message_targets: list[str | None]) -> dict[str, Any]:
    """Spec section 7: post-hoc check from `n` real NO_MESSAGE_CONTROL
    trial outcomes' final targets - the authoritative check, since it
    catches anything the static `classify_dominance` above can't compute
    (an unlisted candidate turning out to dominate in practice). 5/5 one
    target family -> DOMINATED; 4/5 (n>=5) -> REVIEW_REQUIRED; otherwise
    BALANCED."""
    if not no_message_targets:
        return {"classification": REVIEW_REQUIRED, "reason": "no_trials", "distribution": {}, "n": 0}
    counts: dict[str, int] = {}
    for t in no_message_targets:
        key = t if t is not None else "(none)"
        counts[key] = counts.get(key, 0) + 1
    top_target, top_count = max(counts.items(), key=lambda kv: kv[1])
    n = len(no_message_targets)
    if top_count == n:
        classification = DOMINATED
    elif n >= 5 and top_count >= n - 1:
        classification = REVIEW_REQUIRED
    else:
        classification = BALANCED
    return {"classification": classification, "top_target": top_target, "top_count": top_count, "n": n, "distribution": counts}
