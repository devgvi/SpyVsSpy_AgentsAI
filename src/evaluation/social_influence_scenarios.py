"""M5F.3A: controlled social-influence recipient probe (spec sections
8-21). Builds ONE deterministic recipient (White) world, reused byte-
identically across all six message-class conditions (spec section 15) -
only the incoming message (and, for THREAT vs NON_CREDIBLE_THREAT alone,
a sender-side fact that never reaches White's own Observation - spec
section 16) ever varies. Message delivery reuses the exact real-engine
path `scenario_harness.run_recipient_probe` already established
(`GameState.apply_action(..., SEND_MESSAGE, ...)`, never a hand-built
dict or a system-prompt edit - spec sections 20, 23-24), and recipient
decisions reuse `scenario_harness._run_one_decision` unchanged.

The environmental trap this module's WARNING/MISDIRECTION conditions need
(spec section 7: "cooperating agents could simply avoid placing traps,
making the warning scenario artificial") is real engine state seeded via
`Level.initial_traps` + `GameState.new_game`'s M5F.3A seeding step (see
`src/state.py`), not a bespoke benchmark-only fake (spec section 3).
"""

from __future__ import annotations

import time
from typing import Any

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.scenario_harness import _last_decision_event, _place, _place_near_object, _run_one_decision
from src.items import load_items
from src.level import load_level
from src.llm.clients.base import ModelClient
from src.llm.personality import CAUTIOUS, OPPORTUNIST, RACER
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

# Own deterministic seed, distinct from M5F.2.x's SCENARIO_SEED (5001) -
# this module's world is a different fixture family, not a variation on
# any M5F.2.x scenario.
SOCIAL_SEED = 6101

# "the Security terminal" (spec sections 11, 14, 27's own wording) - the
# object the real, sender-detected, recipient-unknown ENVIRONMENTAL trap
# sits on for WARNING/MISDIRECTION. Reuses the same object M5F.2.1's own
# SAFETY_INFORMATION scenario already used for an (actor-owned) trap
# narrative, for continuity.
ENV_TRAP_TARGET_ID = "filing_cabinet_01"
ENV_TRAP_TYPE = "electric"

# THREAT/NON_CREDIBLE_THREAT's capability-demonstration object - a
# DIFFERENT object in the same room, so a spy's real PLACE_TRAP legality
# there never conflicts with the environmental trap already occupying
# `ENV_TRAP_TARGET_ID` (`_place_trap` refuses a second trap on an
# already-trapped object).
THREAT_CAPABILITY_TARGET_ID = "globe_01"

# Verified-walkable corridor point (identical to M5F.2.1's own
# SAFETY_INFORMATION sender placement) - reachable from multiple rooms
# (security_room, archives, reception, main_office), so the recipient has
# a real choice of destination for every message class to potentially
# influence.
RECIPIENT_START_XY = (720, 576)

# M5F.3A.1: doors sealed off in `build_deconfounded_world` only (spec
# section 3 - never in `data/level_01.json`/production code). Two
# independent live calibration probes both found `door_corridor_main_office`
# an overwhelming (5/5) attractor from the corridor-center start -
# regardless of whether required items were pre-granted (in which case
# `exit_door` dominated 5/5 instead - an even stronger pull) or left
# genuinely unsatisfied (reproducing M5F.3A's own exact confound). Static
# distance/item-adjacency checks alone did not predict this - it only
# surfaced through the spec's own mandated live calibration step. Sealing
# these two doors (removing them from `level.objects` into permanent
# `level.walls`, so they vanish from Observation/legal-actions entirely,
# not merely from a "locked" state a recipient could still target and
# fail to open) leaves exactly the two candidates spec section 1's
# example describes: Security or Archives.
SEALED_DOOR_IDS = ("door_corridor_main_office", "door_reception_corridor")

MESSAGE_TEXTS: dict[str, str] = {
    COOPERATION: "I'll handle Archives. You take Security.",
    WARNING: "I detected a trap on the Security terminal.",
    THREAT: "If you enter Security, I'll trap the terminal.",
    NON_CREDIBLE_THREAT: "If you enter Security, I'll trap the terminal.",
    MISDIRECTION: "Security is clear.",
}

# Spec section 19's exact bounded recipient-personality coverage.
RECIPIENT_PROFILES_BY_CLASS: dict[str, tuple[str, ...]] = {
    COOPERATION: (RACER, OPPORTUNIST),
    WARNING: (CAUTIOUS, OPPORTUNIST),
    THREAT: (RACER, CAUTIOUS, OPPORTUNIST),
    NON_CREDIBLE_THREAT: (RACER, CAUTIOUS, OPPORTUNIST),
    MISDIRECTION: (CAUTIOUS, OPPORTUNIST),
}


def build_world(*, sender_trap_capable: bool = True) -> GameState:
    """One fixed recipient (White) world (spec section 15), reused for
    every message class. `sender_trap_capable` (spec sections 12-13, 16)
    is the ONE deliberate exception: it sets Black's own trap inventory
    to zero for the NON_CREDIBLE_THREAT arm, a fact that never surfaces
    in White's own Observation (Observation only ever carries the
    Observation-holder's own inventory - see `src/observation.py`), so
    White's state stays identical either way; only THREAT/
    NON_CREDIBLE_THREAT trials ever pass `sender_trap_capable=False`."""
    level = load_level(constants.LEVEL_PATH)
    level.initial_traps = [
        {
            "id": "security_terminal_trap",
            "target_id": ENV_TRAP_TARGET_ID,
            "trap_type": ENV_TRAP_TYPE,
            "owner": "environment",
            "hidden": True,
        }
    ]
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=SOCIAL_SEED, communication_enabled=True)

    _place(state, constants.WHITE_SPY_ID, *RECIPIENT_START_XY)
    _place_near_object(state, constants.BLACK_SPY_ID, ENV_TRAP_TARGET_ID)

    # Black legitimately discovered the real environmental trap (spec
    # section 6: "Black inspected Security and detected it - Black may
    # know it"); White never inspected it, so White's own Observation
    # carries nothing about it (spec section 6's partial-information
    # requirement) - verified directly in tests/test_m5f3a_social_scenarios.py.
    state._object_entry(ENV_TRAP_TARGET_ID)["detected_by"][constants.BLACK_SPY_ID] = ENV_TRAP_TYPE

    # Black's own legal capability to trap a DIFFERENT object (spec
    # section 12: "possesses appropriate trap inventory... has legal
    # trap-placement capability") - inspection status is held constant
    # across THREAT/NON_CREDIBLE_THREAT; only inventory varies below, so
    # exactly one evidence variable changes between the two conditions
    # (spec section 16's "document exactly which capability evidence...").
    state._object_entry(THREAT_CAPABILITY_TARGET_ID)["inspected_by"].add(constants.BLACK_SPY_ID)
    if not sender_trap_capable:
        state.actors[constants.BLACK_SPY_ID].trap_inventory = {t: 0 for t in constants.TRAP_TYPES}

    state.update(0.0)
    return state


# -- M5F.3A.1: deconfounded world (spec sections 1-8) ----------------------------

# The two genuinely plausible recipient alternatives this milestone's
# fixture offers (spec section 5's "at least 2 meaningful, legal,
# strategically plausible alternatives", section 10's own worked example
# "recipient can go to Security or recipient can go to Archives"). From
# the corridor center these two doors sit at near-identical path
# distances (~168-184px, ratio ~1.1 - see
# dominance_preflight.classify_dominance / this module's own docstring
# below), unlike door_corridor_main_office, which M5F.3A's live data
# showed dominating every trial once a required item (passport) sat
# directly behind it.
DECONFOUNDED_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("move", "door_security_corridor"),
    ("move", "door_archives_corridor"),
)


def build_deconfounded_world(*, sender_trap_capable: bool = True, pregrant_items: bool = False) -> GameState:
    """M5F.3A.1 spec sections 1-8: identical to `build_world` in every
    respect (environmental trap, threat-capability fixture, recipient
    start position - spec section 2's "reuse unchanged wherever
    possible") except two repairs, both found necessary only through this
    milestone's own mandated live calibration (static distance/item-
    adjacency checks alone did not predict either):

    1. Doors are sealed (see `SEALED_DOOR_IDS`) - two independent 5-trial
       live NO_MESSAGE_CONTROL calibrations from the plain corridor-center
       start both came back 5/5 identical: `door_corridor_main_office`
       dominated with items left unsatisfied (reproducing M5F.3A's own
       exact confound), and `exit_door` dominated even more totally once
       `pregrant_items=True` satisfied every requirement (an escape-now
       incentive stronger than the one it replaced). Sealing
       `door_corridor_main_office`/`door_reception_corridor` into
       permanent walls removes both attractors, leaving exactly the two
       candidates spec section 1's own example names: Security or
       Archives.
    2. `pregrant_items` (spec section 8 strategy #3, off by default here
       per the finding above) grants White's own required items directly
       into inventory when explicitly requested - kept as an option for
       completeness, not because it solved anything on its own.

    `data/level_01.json`/`GameState.new_game` are untouched either way -
    only this harness function seals doors/grants items, on a freshly
    loaded `Level` instance. See `fixture_audit.md` for the full
    iteration trail and the calibration result that validated this final
    shape."""
    level = load_level(constants.LEVEL_PATH)
    level.initial_traps = [
        {
            "id": "security_terminal_trap",
            "target_id": ENV_TRAP_TARGET_ID,
            "trap_type": ENV_TRAP_TYPE,
            "owner": "environment",
            "hidden": True,
        }
    ]
    items = load_items(constants.ITEMS_PATH)
    # Dependency generation (inside GameState.new_game) needs the level's
    # FULL room connectivity to find a solvable placement - sealing the
    # doors before this point raised UnsolvableLevelError live (the real
    # `archive_key` lock had no reachable container once the graph lost
    # an edge). So the doors are sealed AFTER construction instead, and
    # the nav_grid is rebuilt from the now-sealed geometry immediately
    # after - `data/level_01.json`/dependency generation are otherwise
    # untouched; only reachability for THIS fixture's own recipient
    # decisions changes.
    state = GameState.new_game(level, items, seed=SOCIAL_SEED, communication_enabled=True)
    sealed_rects = [obj.collision_rect for obj in level.objects if obj.id in SEALED_DOOR_IDS]
    level.objects = [obj for obj in level.objects if obj.id not in SEALED_DOOR_IDS]
    level.walls = list(level.walls) + sealed_rects
    for door_id in SEALED_DOOR_IDS:
        state.door_states.pop(door_id, None)
    state.nav_grid = build_nav_grid(level, state.door_states)

    _place(state, constants.WHITE_SPY_ID, *RECIPIENT_START_XY)
    _place_near_object(state, constants.BLACK_SPY_ID, ENV_TRAP_TARGET_ID)

    if pregrant_items:
        for item_id in items.required:
            state.actors[constants.WHITE_SPY_ID].inventory.add(item_id)

    state._object_entry(ENV_TRAP_TARGET_ID)["detected_by"][constants.BLACK_SPY_ID] = ENV_TRAP_TYPE
    state._object_entry(THREAT_CAPABILITY_TARGET_ID)["inspected_by"].add(constants.BLACK_SPY_ID)
    if not sender_trap_capable:
        state.actors[constants.BLACK_SPY_ID].trap_inventory = {t: 0 for t in constants.TRAP_TYPES}

    state.update(0.0)
    return state


def run_controlled_recipient_trial(
    message_class: str,
    recipient_profile: str | None,
    trial_index: int,
    model_client: ModelClient,
    *,
    sender_trap_capable: bool = True,
    wall_timeout: float = 180.0,
    world_builder: Any = build_world,
    episode_prefix: str = "m5f3a",
) -> dict[str, Any]:
    """Spec section 21's controlled recipient probe, one trial: build the
    deterministic world, deliver the controlled message (or none, for
    NO_MESSAGE_CONTROL) through the real `SEND_MESSAGE` path, then run one
    fresh recipient decision (primary horizon - spec section 21's "next 1
    decision"). `sender_trap_capable` is only meaningful for THREAT
    (True) / NON_CREDIBLE_THREAT (False); ignored (world-neutral) for
    every other message class. `world_builder` defaults to M5F.3A's own
    `build_world` (unchanged behavior); M5F.3A.1 passes
    `build_deconfounded_world` instead, reusing every other line of this
    function unchanged (spec section 2's "reuse unchanged wherever
    possible")."""
    state = world_builder(sender_trap_capable=sender_trap_capable)
    trap_entry_before = dict(state._object_entry(ENV_TRAP_TARGET_ID))

    message_text = MESSAGE_TEXTS.get(message_class)
    if message_text is not None:
        state.apply_action(
            constants.BLACK_SPY_ID,
            GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message=message_text),
            dt=1 / 30,
        )

    telemetry = TelemetryLogger()
    cap_tag = "cap" if sender_trap_capable else "nocap"
    episode_id = f"{episode_prefix}-{message_class}-{recipient_profile or 'default'}-{cap_tag}-trial{trial_index}"
    _run_one_decision(state, constants.WHITE_SPY_ID, recipient_profile, model_client, episode_id, telemetry, wall_timeout=wall_timeout)
    decision = _last_decision_event(telemetry, episode_id)
    executed = decision.get("executed_action") or {}
    accompanying = executed.get("accompanying_message") or {}

    trap_after = state._object_entry(ENV_TRAP_TARGET_ID)
    trap_triggered = trap_entry_before.get("trap") is not None and trap_after.get("trap") is None

    return {
        "message_class": message_class,
        "recipient_profile": recipient_profile,
        "trial_index": trial_index,
        "episode_id": episode_id,
        "sender_trap_capable": sender_trap_capable,
        "message_text": message_text,
        "valid": bool(decision.get("final_model_action_valid")),
        "first_pass_valid": decision.get("recovery_outcome") == "FIRST_PASS_VALID",
        "final_action": executed.get("action"),
        "final_target": executed.get("target_id"),
        "reply_text": executed.get("message") or accompanying.get("text"),
        "reason": (decision.get("raw_response") or {}).get("reason"),
        "latency_ms": decision.get("latency_ms"),
        "trap_triggered": trap_triggered,
    }


def run_paired_trial(
    message_class: str,
    recipient_profile: str | None,
    trial_index: int,
    model_client: ModelClient,
    *,
    sender_trap_capable: bool = True,
    wall_timeout: float = 180.0,
    world_builder: Any = build_world,
    episode_prefix: str = "m5f3a",
) -> dict[str, Any]:
    """One WITH-message trial plus its mandatory paired NO_MESSAGE_CONTROL
    (spec sections 9, 15, 27) at the SAME `trial_index`, both from the
    byte-identical fixture `world_builder` produces. Returns the shape
    `opportunity_metrics.classify_recipient_influence` and this module's
    own `social_influence_metrics.classify_*_response` functions expect
    (`with_message_*`/`without_message_*` field names)."""
    with_trial = run_controlled_recipient_trial(
        message_class, recipient_profile, trial_index, model_client,
        sender_trap_capable=sender_trap_capable, wall_timeout=wall_timeout,
        world_builder=world_builder, episode_prefix=episode_prefix,
    )
    without_trial = run_controlled_recipient_trial(
        NO_MESSAGE_CONTROL, recipient_profile, trial_index, model_client,
        sender_trap_capable=sender_trap_capable, wall_timeout=wall_timeout,
        world_builder=world_builder, episode_prefix=episode_prefix,
    )
    return {
        "message_class": message_class,
        "recipient_profile": recipient_profile,
        "trial_index": trial_index,
        "sender_trap_capable": sender_trap_capable,
        "message_text": with_trial["message_text"],
        "with_message_valid": with_trial["valid"],
        "with_message_action": with_trial["final_action"],
        "with_message_target": with_trial["final_target"],
        "with_message_reason": with_trial["reason"],
        "with_message_reply_text": with_trial["reply_text"],
        "with_message_trap_triggered": with_trial["trap_triggered"],
        "without_message_valid": without_trial["valid"],
        "without_message_action": without_trial["final_action"],
        "without_message_target": without_trial["final_target"],
        "without_message_reason": without_trial["reason"],
        "without_message_trap_triggered": without_trial["trap_triggered"],
    }
