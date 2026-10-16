"""M6G spec sections 14-19: the conflict taxonomy (C0-C5), built only from
existing Engine mechanics - no new Engine code (spec section 46).

Each scenario is a placement `setup_fn` plus one scripted, model-facing
decision per actor (`{"action", "target_id", "trap_type"}` - the exact
shape a model returns). Phase B routes those scripted decisions through
the same `convert_response_to_action` -> `GameState.apply_action` path the
live phases use, so Engine semantics are measured on the identical
pipeline, just without model variability (spec section 21).

Mechanics used (read from `src/state.py`, then pinned by Phase B tests):
- `searched_by` / `inspected_by` are per-actor sets inside the shared
  object entry (state.py 878-888): both actors can search the same object.
- `door_states[id]` is shared world truth (state.py 703-731): once one
  actor unlocks a door, the other actor's INTERACT runs the cheaper OPEN
  branch instead (0.3 s instead of 1.0 s, no keycard check).
- `object_states[id]["trap"]` is one shared slot (state.py 644-667): a
  second PLACE_TRAP on an occupied slot is refused silently.
- DISARM clears that shared slot (state.py 669-701): a second DISARM finds
  nothing and is refused silently - a different mechanic from C4's.
- There is no actor-actor collision (`Actor._move_axis` only tests level
  geometry), so C3 is a negative control, not a forced conflict.

`side_effect_visibility` records whether the FIRST action's effect is
visible in the OTHER actor's observation afterwards (checked, not assumed,
by the Phase B tests). That is what separates a sequential-information
effect (S3/S4 second actor can see and react) from an apply-time
invalidation it cannot see coming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B
from src.evaluation.runtime_snapshot_replay import _place_near_object
from src.state import GameState

C0_NON_CONFLICTING = "C0_NON_CONFLICTING_ACTIONS"
C1_SAME_TARGET = "C1_SAME_TARGET_INTERACTION"
C2_SHARED_DOOR = "C2_SHARED_DOOR_OR_OBJECT"
C3_SHARED_SPACE = "C3_BOTH_MOVE_TO_SHARED_SPACE"
C4_TRAP_CONFLICT = "C4_TRAP_OR_INTERACTION_CONFLICT"
C5_INVALIDATES = "C5_ONE_ACTION_INVALIDATES_OTHER"

CONFLICT_FAMILIES = (C0_NON_CONFLICTING, C1_SAME_TARGET, C2_SHARED_DOOR, C3_SHARED_SPACE, C4_TRAP_CONFLICT, C5_INVALIDATES)

VISIBILITY_NONE = "NONE"
VISIBILITY_OBSERVABLE = "OBSERVABLE"
VISIBILITY_HIDDEN = "HIDDEN"


@dataclass(frozen=True)
class ConflictScenario:
    family: str
    scenario_id: str
    description: str
    setup_fn: Callable[[GameState], None]
    raw_a: dict[str, Any]
    raw_b: dict[str, Any]
    shared_state: str
    side_effect_visibility: str
    # Predeclared from code reading before Phase B ran; Phase B's tests
    # check each one against the running Engine.
    expected_apply_order_effect: bool
    expected_second_observation_changes: bool
    negative_control: bool = False

    def scripted_raw(self, actor_id: str) -> dict[str, Any]:
        return dict(self.raw_a if actor_id == ACTOR_A else self.raw_b)


def _setup_c0(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "desk_01")
    _place_near_object(state, ACTOR_B, "locker_01")


def _setup_c1(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "desk_01")
    _place_near_object(state, ACTOR_B, "desk_01")


def _setup_c2(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "door_security_corridor")
    _place_near_object(state, ACTOR_B, "door_security_corridor")
    state.actors[ACTOR_A].inventory.add("security_keycard")
    state.actors[ACTOR_B].inventory.add("security_keycard")


def _setup_c3(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "door_corridor_main_office")
    _place_near_object(state, ACTOR_B, "door_corridor_main_office")


def _setup_c4(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "ambassador_safe")
    _place_near_object(state, ACTOR_B, "ambassador_safe")
    entry = state._object_entry("ambassador_safe")
    entry["inspected_by"] = {ACTOR_A, ACTOR_B}


def _setup_c5(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "ambassador_safe")
    _place_near_object(state, ACTOR_B, "ambassador_safe")
    entry = state._object_entry("ambassador_safe")
    entry["inspected_by"] = {ACTOR_A, ACTOR_B}
    entry["trap"] = {"type": "bomb", "owner_id": ACTOR_A, "armed": True}
    entry["detected_by"] = {ACTOR_B: "bomb"}


def _raw(action: str, target_id: str | None = None, trap_type: str | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {"action": action, "target_id": target_id, "reason": "m6g scripted pair"}
    if trap_type is not None:
        raw["trap_type"] = trap_type
    return raw


def build_conflict_scenarios() -> dict[str, ConflictScenario]:
    return {
        C0_NON_CONFLICTING: ConflictScenario(
            C0_NON_CONFLICTING, "c0_non_conflicting",
            "A searches desk_01 (main_office), B searches locker_01 (security_room): disjoint objects, different rooms.",
            _setup_c0, _raw("interact", "desk_01"), _raw("interact", "locker_01"),
            "none", VISIBILITY_NONE, expected_apply_order_effect=False, expected_second_observation_changes=False,
            negative_control=True,
        ),
        C1_SAME_TARGET: ConflictScenario(
            C1_SAME_TARGET, "c1_same_target_interaction",
            "A and B both search desk_01 from the same spot: same object, but searched_by is per-actor.",
            _setup_c1, _raw("interact", "desk_01"), _raw("interact", "desk_01"),
            "object_states['desk_01']['searched_by'] (per-actor set)", VISIBILITY_NONE,
            expected_apply_order_effect=False, expected_second_observation_changes=False,
        ),
        C2_SHARED_DOOR: ConflictScenario(
            C2_SHARED_DOOR, "c2_shared_door_or_object",
            "A and B both hold the keycard and both INTERACT with the locked door_security_corridor.",
            _setup_c2, _raw("interact", "door_security_corridor"), _raw("interact", "door_security_corridor"),
            "door_states['door_security_corridor'] (shared)", VISIBILITY_OBSERVABLE,
            expected_apply_order_effect=True, expected_second_observation_changes=True,
        ),
        C3_SHARED_SPACE: ConflictScenario(
            C3_SHARED_SPACE, "c3_both_move_to_shared_space",
            "A and B both MOVE one tick toward the same door_corridor_main_office; no actor-actor collision exists.",
            _setup_c3, _raw("move", "door_corridor_main_office"), _raw("move", "door_corridor_main_office"),
            "none (positions only; Actor.move never checks the other actor)", VISIBILITY_OBSERVABLE,
            expected_apply_order_effect=False, expected_second_observation_changes=True, negative_control=True,
        ),
        C4_TRAP_CONFLICT: ConflictScenario(
            C4_TRAP_CONFLICT, "c4_trap_or_interaction_conflict",
            "A and B have both inspected ambassador_safe and both PLACE_TRAP(bomb) on it.",
            _setup_c4, _raw("place_trap", "ambassador_safe", "bomb"), _raw("place_trap", "ambassador_safe", "bomb"),
            "object_states['ambassador_safe']['trap'] (shared single slot, hidden from non-owners)", VISIBILITY_HIDDEN,
            expected_apply_order_effect=True, expected_second_observation_changes=False,
        ),
        C5_INVALIDATES: ConflictScenario(
            C5_INVALIDATES, "c5_one_action_invalidates_other",
            "A's bomb is on ambassador_safe and B has detected it; A (owner) and B (detector) both DISARM.",
            _setup_c5, _raw("disarm", "ambassador_safe"), _raw("disarm", "ambassador_safe"),
            "object_states['ambassador_safe']['trap'] (shared; visible to its owner and to the detector)",
            VISIBILITY_OBSERVABLE, expected_apply_order_effect=True, expected_second_observation_changes=True,
        ),
    }
