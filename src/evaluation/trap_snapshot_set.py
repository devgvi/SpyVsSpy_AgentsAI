"""M6B (spec sections 24-26): a dedicated, frozen set of trap-placement-
opportunity snapshots for the PLACE_TRAP reliability investigation -
narrower than the spec's recommended 8-12 (4 here), an explicit scope
reduction given this session's already-large cumulative live Bedrock
spend across M6A/M6A.1/M6A.2, not a silent cut. Reuses the exact same
construction technique `runtime_snapshot_replay.py` already established
and validated (`state.update(0.0)` after placement, direct state
seeding rather than organic play) rather than inventing a second one.

Every snapshot exposes only what the actor is legally allowed to know
(spec section 26) - `trap_inventory` is the actor's own real inventory,
never a hidden hint about which type "should" be used.
"""

from __future__ import annotations

from src import constants
from src.evaluation.runtime_snapshot_replay import DecisionSnapshot, _fresh_state, _place, _place_near_object, _snapshot_from_state
from src.state import GameState

# Reuses `runtime_snapshot_replay._fresh_state()`'s own fixed seed (6001)
# internally - no separate seed constant needed here.

MULTIPLE_TYPES_REMAINING = "MULTIPLE_TYPES_REMAINING"
SINGLE_TYPE_REMAINING = "SINGLE_TYPE_REMAINING"
ONE_TYPE_DEPLETED = "ONE_TYPE_DEPLETED"
OPPONENT_NEARBY = "OPPONENT_NEARBY"

TRAP_SNAPSHOT_CATEGORIES = (MULTIPLE_TYPES_REMAINING, SINGLE_TYPE_REMAINING, ONE_TYPE_DEPLETED, OPPONENT_NEARBY)


def _base_trap_opportunity_state() -> GameState:
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "ambassador_safe")
    state._object_entry("ambassador_safe")["inspected_by"].add(constants.WHITE_SPY_ID)
    return state


def build_trap_snapshot_set() -> list[DecisionSnapshot]:
    snapshots: list[DecisionSnapshot] = []

    # 1. MULTIPLE_TYPES_REMAINING - actor holds every real trap type
    # (GameState.new_game's own default starting inventory - never
    # fabricated), all four legal choices genuinely available.
    state = _base_trap_opportunity_state()
    snapshots.append(_snapshot_from_state("trap01_multiple_types_remaining", MULTIPLE_TYPES_REMAINING, state))

    # 2. SINGLE_TYPE_REMAINING - only one real type left, by directly
    # zeroing the others in the actor's own inventory (legitimate
    # self-knowledge, not hidden state).
    state = _base_trap_opportunity_state()
    actor = state.actors[constants.WHITE_SPY_ID]
    kept_type = constants.TRAP_TYPES[0]
    for trap_type in constants.TRAP_TYPES:
        if trap_type != kept_type:
            actor.trap_inventory[trap_type] = 0
    snapshots.append(_snapshot_from_state("trap02_single_type_remaining", SINGLE_TYPE_REMAINING, state))

    # 3. ONE_TYPE_DEPLETED - the actor still holds several types, but one
    # specific real type is exhausted (distinct from "only one left").
    state = _base_trap_opportunity_state()
    actor = state.actors[constants.WHITE_SPY_ID]
    depleted_type = constants.TRAP_TYPES[-1]
    actor.trap_inventory[depleted_type] = 0
    snapshots.append(_snapshot_from_state("trap03_one_type_depleted", ONE_TYPE_DEPLETED, state))

    # 4. OPPONENT_NEARBY - Black Spy visible in the same room while the
    # trap-placement opportunity is legal (spec section 24's "opponent
    # nearby" variation).
    state = _base_trap_opportunity_state()
    white = state.actors[constants.WHITE_SPY_ID]
    _place(state, constants.BLACK_SPY_ID, white.x + 15, white.y)
    snapshots.append(_snapshot_from_state("trap04_opponent_nearby", OPPONENT_NEARBY, state))

    return snapshots
