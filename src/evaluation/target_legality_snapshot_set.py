"""M6B.2 (spec section 28, Phase B): a dedicated, frozen set of PLACE_TRAP
target-legality snapshots - one per real, actor-visible dimension that can
make a target illegal (uninspected, non-trappable, out of range, legal).
`already_trapped`/`unknown_target` are deliberately NOT snapshot-based (spec
section 7): a pre-existing hidden trap is never exposed to `legal_actions`
at all (the engine alone rejects it at execution time - see
`GameState._place_trap`), and an unknown target is just a fabricated
`target_id` string, not a game-state variation. Reuses the exact same
construction technique `trap_snapshot_set.py`/`runtime_snapshot_replay.py`
already established (`state.update(0.0)` after placement, direct state
seeding) rather than inventing a second one.
"""

from __future__ import annotations

from src import constants
from src.evaluation.runtime_snapshot_replay import DecisionSnapshot, _fresh_state, _place, _place_near_object, _snapshot_from_state
from src.state import GameState

UNINSPECTED_TRAPPABLE_NEARBY = "UNINSPECTED_TRAPPABLE_NEARBY"
INSPECTED_TRAPPABLE_NEARBY = "INSPECTED_TRAPPABLE_NEARBY"
NON_TRAPPABLE_NEARBY = "NON_TRAPPABLE_NEARBY"
NO_TRAP_INVENTORY_NEARBY = "NO_TRAP_INVENTORY_NEARBY"

TARGET_LEGALITY_SNAPSHOT_CATEGORIES = (
    UNINSPECTED_TRAPPABLE_NEARBY, INSPECTED_TRAPPABLE_NEARBY, NON_TRAPPABLE_NEARBY, NO_TRAP_INVENTORY_NEARBY,
)


def build_target_legality_snapshot_set() -> list[DecisionSnapshot]:
    snapshots: list[DecisionSnapshot] = []

    # 1. UNINSPECTED_TRAPPABLE_NEARBY - the exact M6B.1 outlier-episode
    # shape (seed 9605): actor stands next to a trappable object, holds
    # traps, but has never inspected THIS object - place_trap must not be
    # offered (see enumerate_legal_actions), and validate_model_output
    # must classify a place_trap proposal on it as `target_not_inspected`,
    # never a generic `illegal_target`. Uses `ambassador_safe` (not a
    # generic furniture item) with an opponent placed visibly nearby -
    # this is the EXACT object/opponent-proximity shape M6B's own
    # trap_snapshot_set.py OPPONENT_NEARBY category already proved live
    # (Native 3/3, Strands 5/5 place_trap attempts elicited - the only
    # populated cell of that milestone's 4 snapshots). Two earlier live
    # pilots of this snapshot using a generic object (`sofa_01`, with and
    # without an opponent present) elicited ZERO place_trap proposals from
    # either runtime across 40 single-decision trials total - a strategic
    # target (the safe, this level's actual mission-relevant object) turns
    # out to matter, not just opponent presence, confirming the object
    # choice - not merely proximity motivation - was the missing variable.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "ambassador_safe")
    white = state.actors[constants.WHITE_SPY_ID]
    _place(state, constants.BLACK_SPY_ID, white.x + 15, white.y)
    snapshots.append(_snapshot_from_state("tl01_uninspected_trappable_nearby", UNINSPECTED_TRAPPABLE_NEARBY, state))

    # 2. INSPECTED_TRAPPABLE_NEARBY - the legal case: inspected, trappable,
    # traps held - place_trap IS offered and IS the sole legal target. Same
    # object/opponent-nearby configuration as above (byte-identical, in
    # fact, to M6B's own proven OPPONENT_NEARBY snapshot), for the same
    # reason.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "ambassador_safe")
    state._object_entry("ambassador_safe")["inspected_by"].add(constants.WHITE_SPY_ID)
    white = state.actors[constants.WHITE_SPY_ID]
    _place(state, constants.BLACK_SPY_ID, white.x + 15, white.y)
    snapshots.append(_snapshot_from_state("tl02_inspected_trappable_nearby", INSPECTED_TRAPPABLE_NEARBY, state))

    # 3. NON_TRAPPABLE_NEARBY - a door: interactable, in range, even
    # inspected (inspection is irrelevant here) - still never trappable at
    # all (Interactable.trappable excludes every door by type, spec
    # section 4's "Engine remains authoritative"). Expect
    # `target_not_trappable`, never `target_not_inspected`.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "door_main_office_ambassador")
    snapshots.append(_snapshot_from_state("tl03_non_trappable_nearby", NON_TRAPPABLE_NEARBY, state))

    # 4. NO_TRAP_INVENTORY_NEARBY - inspected, trappable, in range, but the
    # actor holds zero traps of any type - place_trap correctly never
    # offered at all (enumerate_legal_actions's `has_any_trap` gate), and a
    # place_trap proposal here fails at the earlier, trap_type-level check
    # (`trap_type_not_in_inventory`/`missing_target`) - never reaches the
    # target-legality branch this milestone's new categories live in. Kept
    # here as the boundary case proving the two dimensions (trap_type vs.
    # target) stay properly separated even when both would independently
    # fail.
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "sofa_01")
    state._object_entry("sofa_01")["inspected_by"].add(constants.WHITE_SPY_ID)
    actor = state.actors[constants.WHITE_SPY_ID]
    for trap_type in constants.TRAP_TYPES:
        actor.trap_inventory[trap_type] = 0
    snapshots.append(_snapshot_from_state("tl04_no_trap_inventory_nearby", NO_TRAP_INVENTORY_NEARBY, state))

    return snapshots
