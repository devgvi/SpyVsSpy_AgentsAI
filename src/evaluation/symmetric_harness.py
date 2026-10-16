"""M5F.3A.2: a purpose-built, minimal, geometrically symmetric two-branch
evaluation world (spec sections 2-17), entirely separate from the
production embassy level (`data/level_01.json` is never read or written
by this module) - built after M5F.3A.1 found that no amount of rebalancing
the embassy map removes its strategic attractors (see that milestone's
`fixture_audit.md`: three principled repairs, three different 5/5-
dominated live calibrations).

Layout (mirrored about `AXIS_X`, both branches pixel-identical up to
reflection):

              START (600, 200)
                 |         |
            door_side_a  door_side_b
                 |         |
          objective_side_a   objective_side_b

`objective_side_a`/`objective_side_b` are the two ALWAYS-physically-fixed
objects (side_a is always the physically-left branch); `alpha_on_left`
only relabels which physical side is presented as "objective_alpha" vs
"objective_beta" to the recipient (spec section 14/15's left/right mirror
and label-swap controls collapse into this one boolean - swapping it is
the SAME manipulation read two ways: a stuck preference for the physical
side is side bias, a stuck preference for the label is lexical bias).

`GameState.new_game`'s own dependency-generation step
(`src/dependency_generation.py`) is hardcoded to the embassy's own item/
lock topology (`archive_key`/`security_keycard`/`ambassador_safe` are
literal constants there) and cannot run against an arbitrary level - this
module therefore never calls `GameState.new_game`, constructing
`GameState` directly instead, with an empty `ItemCatalog` (spec section 7
option A: neither objective is mission-required) and a trivial
`DependencyPlacement`. No `exit_door` object exists in this level at all
(spec section 8: no global exit attractor - not "disabled", simply never
defined).
"""

from __future__ import annotations

import random
import uuid
from typing import Any

import pygame

from src import constants
from src.actions import ActionType, GameAction
from src.actor import Actor
from src.dependency_generation import DependencyPlacement
from src.evaluation.scenario_harness import _last_decision_event, _run_one_decision
from src.interactable import GameObject
from src.items import ItemCatalog
from src.level import Level, Room
from src.llm.clients.base import ModelClient
from src.llm.telemetry import TelemetryLogger
from src.navigation import build_nav_grid
from src.observation import build_observation
from src.state import GameState

NO_MESSAGE_CONTROL = "NO_MESSAGE_CONTROL"
COOPERATION = "COOPERATION"
WARNING = "WARNING"
THREAT = "THREAT"
NON_CREDIBLE_THREAT = "NON_CREDIBLE_THREAT"
MISDIRECTION = "MISDIRECTION"
MESSAGE_CLASSES = (NO_MESSAGE_CONTROL, COOPERATION, WARNING, THREAT, NON_CREDIBLE_THREAT, MISDIRECTION)

SYMMETRIC_SEED = 7301

# -- geometry (spec sections 4, 10-12) -------------------------------------------

AXIS_X = 608.0
# All coordinates below are exact multiples of constants.NAV_GRID_SIZE
# (32px) and AXIS_X itself sits on a grid-cell boundary (608 = 19*32) -
# confirmed necessary, not cosmetic: an earlier non-grid-aligned draft of
# this geometry (axis_x=600, not a multiple of 32) produced measurably
# DIFFERENT nav-grid path distances for two mathematically mirror-image
# rects (424.1px vs 443.5px, ~4.4% apart) purely from cell-quantization
# tie-breaking in `NavGrid.nearest_approach_point`'s search order - a
# real, load-bearing finding for spec section 10's "geometric difference
# must be zero or explicitly negligible" requirement. With every
# coordinate grid-aligned, both approach points come back byte-identical
# distances (432.0px each) and mirror exactly (176,208) <-> (1040,208).
LEVEL_SIZE = (1216, 416)
START_XY = (608.0, 208.0)

START_ROOM_RECT = pygame.Rect(512, 64, 192, 288)
SIDE_A_ROOM_RECT = pygame.Rect(64, 64, 320, 288)
DOOR_A_RECT = pygame.Rect(384, 160, 128, 96)
OBJ_A_RECT = pygame.Rect(96, 176, 64, 64)


def _mirror_rect(rect: pygame.Rect, axis_x: float) -> pygame.Rect:
    """Reflects `rect` about the vertical line x=axis_x, keeping
    top/width/height identical - the one function every "side B" geometry
    constant below derives from, so side_a/side_b are exact mirrors by
    construction rather than by two independently hand-picked coordinate
    sets (which is exactly the kind of arithmetic slip a symmetry
    experiment can't afford)."""
    return pygame.Rect(int(2 * axis_x - rect.left - rect.width), rect.top, rect.width, rect.height)


SIDE_B_ROOM_RECT = _mirror_rect(SIDE_A_ROOM_RECT, AXIS_X)
DOOR_B_RECT = _mirror_rect(DOOR_A_RECT, AXIS_X)
OBJ_B_RECT = _mirror_rect(OBJ_A_RECT, AXIS_X)

OBJ_ALPHA_ID = "objective_alpha"
OBJ_BETA_ID = "objective_beta"
DOOR_ALPHA_ID = "door_start_alpha"
DOOR_BETA_ID = "door_start_beta"

# Spec section 24's hard-stop remedy: Stage 1 live calibration with the
# plain "alpha"/"beta" ids above found a clean, repeatable LABEL bias
# (19/20 trials picked "alpha" regardless of which physical side it was
# on - not side bias, since the same label won under both
# `alpha_on_left` layouts) - see summary.md. These hex-style tokens carry
# no dictionary meaning and no ordinal/alphabetic-primacy connotation
# (unlike "alpha"/"beta", "one"/"two", "A"/"B"), equal length, used
# wherever `neutral_ids=True` is passed for the re-audit calibration.
NEUTRAL_OBJ_ALPHA_ID = "objective_x7q2m9"
NEUTRAL_OBJ_BETA_ID = "objective_k3f8p1"
NEUTRAL_DOOR_ALPHA_ID = "door_start_x7q2m9"
NEUTRAL_DOOR_BETA_ID = "door_start_k3f8p1"


def _resolve_ids(neutral_ids: bool) -> tuple[str, str, str, str]:
    """(obj_alpha_id, obj_beta_id, door_alpha_id, door_beta_id) - the one
    place this selection is made, reused by every function below so the
    plain and neutral-id worlds can never accidentally mix ids."""
    if neutral_ids:
        return NEUTRAL_OBJ_ALPHA_ID, NEUTRAL_OBJ_BETA_ID, NEUTRAL_DOOR_ALPHA_ID, NEUTRAL_DOOR_BETA_ID
    return OBJ_ALPHA_ID, OBJ_BETA_ID, DOOR_ALPHA_ID, DOOR_BETA_ID


def _build_level(*, alpha_on_left: bool = True, neutral_ids: bool = False, alpha_listed_first: bool = True) -> Level:
    """Real bug found and fixed before any live spend (see
    `symmetry_audit.md`): an earlier draft always named these objects
    `objective_side_a`/`objective_side_b` and only remapped the
    "alpha"/"beta" LABEL in this module's own bookkeeping - meaning the
    literal string "objective_alpha" a treatment message refers to never
    appeared as any real object's `id` anywhere in the recipient's own
    Observation, leaving the message's referent completely ungrounded.
    This version bakes the alpha/beta id directly into whichever physical
    side (`OBJ_A_RECT`=left / `OBJ_B_RECT`=right, both fixed, mirrored
    geometry - see the constants above) `alpha_on_left` selects, so
    `alpha_on_left=True` and `=False` are genuinely two different Level
    instances with the SAME geometry but swapped id assignment - spec
    section 15's label-swap control made real rather than only internal
    bookkeeping."""
    if alpha_on_left:
        alpha_rect, beta_rect = OBJ_A_RECT, OBJ_B_RECT
        alpha_door_rect, beta_door_rect = DOOR_A_RECT, DOOR_B_RECT
    else:
        alpha_rect, beta_rect = OBJ_B_RECT, OBJ_A_RECT
        alpha_door_rect, beta_door_rect = DOOR_B_RECT, DOOR_A_RECT

    obj_alpha_id, obj_beta_id, door_alpha_id, door_beta_id = _resolve_ids(neutral_ids)
    alpha_label = "Objective Alpha" if not neutral_ids else "Objective"
    beta_label = "Objective Beta" if not neutral_ids else "Objective"

    rooms = [
        Room(id="start", label="Start", rect=START_ROOM_RECT),
        Room(id="room_alpha", label="Room Alpha", rect=SIDE_A_ROOM_RECT if alpha_on_left else SIDE_B_ROOM_RECT),
        Room(id="room_beta", label="Room Beta", rect=SIDE_B_ROOM_RECT if alpha_on_left else SIDE_A_ROOM_RECT),
    ]
    door_alpha_obj = GameObject(
        id=door_alpha_id, type="door", label="Door", position=(alpha_door_rect.left, alpha_door_rect.top),
        collision_size=(alpha_door_rect.width, alpha_door_rect.height), solid=True, interactable=True,
        interaction="open", message=None, default_state="closed", connects=("start", "room_alpha"),
    )
    door_beta_obj = GameObject(
        id=door_beta_id, type="door", label="Door", position=(beta_door_rect.left, beta_door_rect.top),
        collision_size=(beta_door_rect.width, beta_door_rect.height), solid=True, interactable=True,
        interaction="open", message=None, default_state="closed", connects=("start", "room_beta"),
    )
    obj_alpha_obj = GameObject(
        id=obj_alpha_id, type="objective_console", label=alpha_label, position=(alpha_rect.left, alpha_rect.top),
        collision_size=(alpha_rect.width, alpha_rect.height), solid=True, interactable=True,
        interaction="search", message=None,
    )
    obj_beta_obj = GameObject(
        id=obj_beta_id, type="objective_console", label=beta_label, position=(beta_rect.left, beta_rect.top),
        collision_size=(beta_rect.width, beta_rect.height), solid=True, interactable=True,
        interaction="search", message=None,
    )
    # Real finding from live neutral-id calibration (see summary.md): the
    # object listed FIRST in `level.objects` (and therefore in
    # `Observation.objects`) won overwhelmingly regardless of geometry or
    # label content - alpha was always constructed first here, an
    # accidental confound of this function's own code, not a deliberate
    # experimental variable. `alpha_listed_first` decouples list/
    # serialization order from both geometry (`alpha_on_left`) and label
    # content (`neutral_ids`), so each can be tested independently.
    if alpha_listed_first:
        objects = [door_alpha_obj, door_beta_obj, obj_alpha_obj, obj_beta_obj]
    else:
        objects = [door_beta_obj, door_alpha_obj, obj_beta_obj, obj_alpha_obj]
    return Level(
        id="symmetric_harness_v1", name="Symmetric Strategic Choice Harness", size=LEVEL_SIZE,
        player_spawn=START_XY, white_spy_spawn=START_XY, walls=[], objects=objects, rooms=rooms,
    )


def build_symmetric_state(
    *,
    alpha_on_left: bool = True,
    neutral_ids: bool = False,
    alpha_listed_first: bool = True,
    recipient_actor_id: str = constants.WHITE_SPY_ID,
    trap_branch: str | None = None,
    trap_type: str = "electric",
    threat_capability_branch: str | None = None,
    sender_trap_capable: bool = True,
    seed: int = SYMMETRIC_SEED,
) -> GameState:
    """`alpha_on_left` (spec sections 14-15's collapsed single control):
    True -> `objective_side_a` (physically left) is labeled
    `objective_alpha`; False -> the assignment is swapped, geometry
    untouched.

    `trap_branch` ("alpha"/"beta"/None, spec sections 29-30, WARNING/
    MISDIRECTION only): seeds a real environmental trap (same dict shape/
    mechanics as M5F.3A's `ENVIRONMENT`-owned traps) on whichever
    objective that label currently maps to, detected only by the sender -
    never the recipient (partial information preserved exactly as in
    M5F.3A/M5F.3A.1).

    `threat_capability_branch` ("alpha"/"beta"/None, spec sections 32-33,
    THREAT/NON_CREDIBLE_THREAT only): the branch the sender is threatening
    to trap - marked as legally inspected by the sender (a real
    PLACE_TRAP precondition) so the threat has a genuine, checkable
    capability fixture behind it, distinct from `trap_branch` since a
    THREAT trial has no pre-existing environmental trap at all (that
    objective is still untrapped - the sender is only claiming future
    intent, never executing it - spec section 12's "do not make execution
    automatic"). `sender_trap_capable` zeroes the sender's own trap
    inventory for the non-credible arm - the only evidence variable that
    differs between THREAT and NON_CREDIBLE_THREAT."""
    level = _build_level(alpha_on_left=alpha_on_left, neutral_ids=neutral_ids, alpha_listed_first=alpha_listed_first)
    items = ItemCatalog(items={}, required=[], knowledge={})
    door_states = {obj.id: obj.default_state for obj in level.objects if obj.type == "door"}
    nav_grid = build_nav_grid(level, door_states)

    sender_actor_id = constants.BLACK_SPY_ID if recipient_actor_id == constants.WHITE_SPY_ID else constants.WHITE_SPY_ID
    actors = {
        constants.BLACK_SPY_ID: Actor(id=constants.BLACK_SPY_ID, x=START_XY[0], y=START_XY[1]),
        constants.WHITE_SPY_ID: Actor(id=constants.WHITE_SPY_ID, x=START_XY[0], y=START_XY[1]),
    }

    state = GameState(
        level=level, items=items, actors=actors, nav_grid=nav_grid, door_states=door_states,
        mission_complete={a: False for a in actors}, rng=random.Random(seed), seed=seed,
        dependency_placement=DependencyPlacement(item_locations={}, knowledge_locations={}),
        episode_id=str(uuid.uuid4()), communication_enabled=True,
    )

    alpha_object_id, beta_object_id, _, _ = _resolve_ids(neutral_ids)

    if trap_branch is not None:
        target_id = alpha_object_id if trap_branch == "alpha" else beta_object_id
        state._object_entry(target_id)["trap"] = {"type": trap_type, "owner_id": constants.ENVIRONMENT_OWNER_ID, "armed": True}
        state.log_event(
            "ENVIRONMENTAL_TRAP_INITIALIZED", trap_id=f"{target_id}_trap", trap_owner=constants.ENVIRONMENT_OWNER_ID,
            trap_type=trap_type, target_id=target_id, initially_present=True,
        )
        state._object_entry(target_id)["detected_by"][sender_actor_id] = trap_type

    if threat_capability_branch is not None:
        capable_target_id = alpha_object_id if threat_capability_branch == "alpha" else beta_object_id
        state._object_entry(capable_target_id)["inspected_by"].add(sender_actor_id)
        if not sender_trap_capable:
            state.actors[sender_actor_id].trap_inventory = {t: 0 for t in constants.TRAP_TYPES}

    state.update(0.0)
    return state


def alpha_beta_object_ids(alpha_on_left: bool, neutral_ids: bool = False) -> tuple[str, str]:
    """(objective_alpha's real object id, objective_beta's real object id).
    `alpha_on_left` no longer changes WHICH id this returns (both ids are
    stable regardless of layout - see `_build_level`'s docstring for why
    the id itself must be stable); kept only so every caller written
    against the old two-argument contract keeps working unchanged.
    `alpha_on_left` still fully determines the underlying geometry via
    `_build_level`. `neutral_ids` selects the hex-token id pair (spec
    section 24's label-bias remedy) instead of the plain alpha/beta pair."""
    obj_alpha_id, obj_beta_id, _, _ = _resolve_ids(neutral_ids)
    return (obj_alpha_id, obj_beta_id)


# -- static symmetry audit (spec section 18) --------------------------------------

SYMMETRIC = "SYMMETRIC"
INVALID = "INVALID"

# spec section 10: "zero or explicitly negligible" - 0.01px tolerance is
# effectively "must be exactly equal"; the grid-alignment fix documented
# on AXIS_X above is what makes that achievable instead of merely hoped for.
PATH_DISTANCE_TOLERANCE = 0.01


def static_symmetry_audit() -> dict[str, Any]:
    """Spec section 18's required deterministic, network-free checks,
    run against the freshly-built (no trap, no treatment) harness - the
    one state every live trial's own world is a minimal variation of.
    Returns `{"result": SYMMETRIC|INVALID, "checks": {...}, "reasons": [...]}`."""
    state = build_symmetric_state()
    alpha_id, beta_id = alpha_beta_object_ids(True)
    alpha_obj = state.level.get_object(alpha_id)
    beta_obj = state.level.get_object(beta_id)
    door_a = state.level.get_object(DOOR_ALPHA_ID)
    door_b = state.level.get_object(DOOR_BETA_ID)

    reasons: list[str] = []
    checks: dict[str, Any] = {}

    checks["same_object_type"] = alpha_obj.type == beta_obj.type
    checks["same_requirements"] = (alpha_obj.lock_type, alpha_obj.requires) == (beta_obj.lock_type, beta_obj.requires)
    checks["same_mission_value"] = list(alpha_obj.contains) == list(beta_obj.contains) == []
    checks["same_door_count"] = sum(1 for o in state.level.objects if o.type == "door") == 2
    checks["same_door_state"] = state.door_states[door_a.id] == state.door_states[door_b.id]
    checks["same_interaction_affordances"] = (alpha_obj.interactable, alpha_obj.interaction, alpha_obj.trappable) == (
        beta_obj.interactable, beta_obj.interaction, beta_obj.trappable
    )
    checks["same_trap_state"] = state.trap_on(alpha_id) is None and state.trap_on(beta_id) is None
    checks["same_visibility_state"] = (
        state.is_searched_by(alpha_id, constants.WHITE_SPY_ID) == state.is_searched_by(beta_id, constants.WHITE_SPY_ID)
        and state.is_inspected_by(alpha_id, constants.WHITE_SPY_ID) == state.is_inspected_by(beta_id, constants.WHITE_SPY_ID)
    )
    checks["same_legal_action_class"] = alpha_obj.interactable and beta_obj.interactable

    alpha_dist = state.nav_grid.nearest_approach_point(alpha_obj.collision_rect, constants.INTERACTION_DISTANCE, START_XY)
    beta_dist = state.nav_grid.nearest_approach_point(beta_obj.collision_rect, constants.INTERACTION_DISTANCE, START_XY)
    if alpha_dist is None or beta_dist is None:
        checks["same_path_distance"] = False
        reasons.append("one or both objectives unreachable from start")
    else:
        import math

        d_alpha = math.dist(START_XY, alpha_dist)
        d_beta = math.dist(START_XY, beta_dist)
        checks["same_path_distance"] = abs(d_alpha - d_beta) <= PATH_DISTANCE_TOLERANCE
        checks["path_distance_alpha"] = d_alpha
        checks["path_distance_beta"] = d_beta
        if not checks["same_path_distance"]:
            reasons.append(f"path distance differs: alpha={d_alpha:.3f} beta={d_beta:.3f}")

    # No third dominant target (spec section 9): the whole level has
    # exactly 4 non-actor objects - 2 doors (not trappable/not a strategic
    # objective) and the 2 objectives themselves - so this is a direct
    # enumeration, not a heuristic.
    non_door_interactables = [o.id for o in state.level.objects if o.type != "door" and o.interactable]
    checks["no_third_dominant_target"] = sorted(non_door_interactables) == sorted([alpha_id, beta_id])
    checks["required_items_empty"] = state.items.required == []

    # Geometry mirror check (spec section 14): repeat the path-distance
    # measurement with alpha/beta swapped onto the opposite physical
    # sides, confirming the equality holds for BOTH layouts, not just the
    # one this function happened to build first.
    swapped_state = build_symmetric_state(alpha_on_left=False)
    swapped_alpha = swapped_state.level.get_object(alpha_id)
    swapped_beta = swapped_state.level.get_object(beta_id)
    swapped_alpha_ap = swapped_state.nav_grid.nearest_approach_point(swapped_alpha.collision_rect, constants.INTERACTION_DISTANCE, START_XY)
    swapped_beta_ap = swapped_state.nav_grid.nearest_approach_point(swapped_beta.collision_rect, constants.INTERACTION_DISTANCE, START_XY)
    if swapped_alpha_ap is not None and swapped_beta_ap is not None:
        import math

        checks["same_path_distance_mirrored_layout"] = abs(math.dist(START_XY, swapped_alpha_ap) - math.dist(START_XY, swapped_beta_ap)) <= PATH_DISTANCE_TOLERANCE
    else:
        checks["same_path_distance_mirrored_layout"] = False

    for key, value in checks.items():
        if key.startswith("path_distance_"):
            continue
        if value is False:
            reasons.append(f"failed: {key}")

    result = INVALID if reasons else SYMMETRIC
    return {"result": result, "checks": checks, "reasons": reasons}


# -- branch resolution (spec section 39) -------------------------------------

# Real gap found and fixed from Stage 1's own live calibration data (never
# from re-running with different wording/world state - see summary.md):
# the recipient frequently declares the connecting DOOR
# (`door_start_alpha`/`door_start_beta`) as its MOVE target rather than
# the objective object itself. A door target is exactly as diagnostic as
# an objective target - each door leads to precisely one branch by
# construction - so treating it as "OTHER" was undercounting genuine
# alpha/beta decisions as noise. `resolve_branch` is the ONE place this
# mapping is computed, reused by `run_symmetric_trial` for every future
# trial AND by the driver script to reclassify already-collected raw
# `final_target` values without spending any additional live budget.
def resolve_branch(final_target: str | None, neutral_ids: bool = False) -> str:
    obj_alpha_id, obj_beta_id, door_alpha_id, door_beta_id = _resolve_ids(neutral_ids)
    if final_target in (obj_alpha_id, door_alpha_id):
        return "alpha"
    if final_target in (obj_beta_id, door_beta_id):
        return "beta"
    return "OTHER"


# -- message texts (spec sections 27, 29, 31-33, 51-56) --------------------------


def message_text(message_class: str, target_branch: str, neutral_ids: bool = False) -> str | None:
    """`target_branch` is "alpha"/"beta" - the branch the message content
    names. Alpha/Beta wording pairs are mechanically identical except for
    the single swapped word (spec sections 51-52), and use plain,
    non-emotional, factual language throughout (spec sections 53-56).
    `neutral_ids=True` substitutes the actual hex-token object id the
    recipient can see instead of the word "alpha"/"beta" - the message
    must still name a real, resolvable object id (same grounding
    requirement `_build_level` enforces for the plain-id world)."""
    if message_class == NO_MESSAGE_CONTROL:
        return None
    obj_alpha_id, obj_beta_id, _, _ = _resolve_ids(neutral_ids)
    target_name = obj_alpha_id if target_branch == "alpha" else obj_beta_id
    other_name = obj_beta_id if target_branch == "alpha" else obj_alpha_id
    if message_class == COOPERATION:
        return f"I'll take {target_name}. You take {other_name}."
    if message_class == WARNING:
        return f"I detected a trap at {target_name}."
    if message_class == MISDIRECTION:
        return f"{target_name} is clear."
    if message_class in (THREAT, NON_CREDIBLE_THREAT):
        return f"If you go to {target_name}, I'll trap it."
    raise ValueError(f"unknown message_class {message_class!r}")


# -- trial running (spec sections 19-20, 38, 40-42) -------------------------------


def run_symmetric_trial(
    message_class: str,
    target_branch: str | None,
    recipient_profile: str | None,
    trial_index: int,
    model_client: ModelClient,
    *,
    alpha_on_left: bool = True,
    neutral_ids: bool = False,
    alpha_listed_first: bool = True,
    recipient_actor_id: str = constants.WHITE_SPY_ID,
    sender_trap_capable: bool = True,
    wall_timeout: float = 180.0,
    episode_prefix: str = "m5f3a2",
    observation_transform: Any = None,
    model_payload_transform: Any = None,
    model_response_transform: Any = None,
) -> dict[str, Any]:
    """One trial: build the symmetric world (with a real environmental
    trap on `target_branch` for WARNING/MISDIRECTION, spec sections 29-31),
    deliver the message through the real SEND_MESSAGE path (or none, for
    NO_MESSAGE_CONTROL), then run one fresh recipient decision (primary
    horizon - spec section 38). Records the resolved branch
    (`alpha`/`beta`/`OTHER`) of whatever `target_id` the recipient
    actually decided on (spec section 39: this project already knows a
    MOVE target can be a long-range declared destination, not just a
    nearby step, so branch resolution is purely id-based, never
    distance-based). `neutral_ids` (spec section 24): use the hex-token id
    pair instead of the plain "alpha"/"beta" pair - Stage 1's own live
    calibration found a clean label bias toward the word "alpha" itself
    (see `summary.md`), and this is the spec's own prescribed remedy.
    `alpha_listed_first`: which object is constructed first in
    `level.objects`/`Observation.objects` - decoupled from both
    `alpha_on_left` (geometry) and `neutral_ids` (label content) so
    serialization-order bias can be tested independently of either."""
    trap_branch = target_branch if message_class in (WARNING, MISDIRECTION) else None
    threat_capability_branch = target_branch if message_class in (THREAT, NON_CREDIBLE_THREAT) else None
    state = build_symmetric_state(
        alpha_on_left=alpha_on_left, neutral_ids=neutral_ids, alpha_listed_first=alpha_listed_first,
        recipient_actor_id=recipient_actor_id, trap_branch=trap_branch,
        threat_capability_branch=threat_capability_branch, sender_trap_capable=sender_trap_capable, seed=SYMMETRIC_SEED,
    )
    sender_actor_id = constants.BLACK_SPY_ID if recipient_actor_id == constants.WHITE_SPY_ID else constants.WHITE_SPY_ID

    text = message_text(message_class, target_branch, neutral_ids=neutral_ids) if target_branch is not None else None
    if text is not None:
        state.apply_action(sender_actor_id, GameAction(ActionType.SEND_MESSAGE, target_id=recipient_actor_id, message=text), dt=1 / 30)

    telemetry = TelemetryLogger()
    layout_tag = "left" if alpha_on_left else "right"
    branch_tag = target_branch or "none"
    episode_id = f"{episode_prefix}-{message_class}-{branch_tag}-{recipient_profile or 'default'}-{layout_tag}-trial{trial_index}"
    _run_one_decision(
        state, recipient_actor_id, recipient_profile, model_client, episode_id, telemetry, wall_timeout=wall_timeout,
        observation_transform=observation_transform, model_payload_transform=model_payload_transform,
        model_response_transform=model_response_transform,
    )
    decision = _last_decision_event(telemetry, episode_id)
    executed = decision.get("executed_action") or {}
    accompanying = executed.get("accompanying_message") or {}
    final_target = executed.get("target_id")

    resolved_branch = resolve_branch(final_target, neutral_ids=neutral_ids)

    return {
        "message_class": message_class,
        "target_branch": target_branch,
        "recipient_profile": recipient_profile,
        "recipient_actor_id": recipient_actor_id,
        "alpha_on_left": alpha_on_left,
        "neutral_ids": neutral_ids,
        "sender_trap_capable": sender_trap_capable,
        "trial_index": trial_index,
        "episode_id": episode_id,
        "message_text": text,
        "valid": bool(decision.get("final_model_action_valid")),
        "final_action": executed.get("action"),
        "final_target": final_target,
        "resolved_branch": resolved_branch,
        "reply_text": executed.get("message") or accompanying.get("text"),
        "reason": (decision.get("raw_response") or {}).get("reason"),
        "latency_ms": decision.get("latency_ms"),
    }
