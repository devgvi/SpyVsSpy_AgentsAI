"""M6G: two-actor frozen snapshot construction and canonical hashes (spec
sections 4, 7-8, 22).

`build_duel_snapshot()` mirrors `Game._step`'s own ordering (game.py
421-449: `build_observation` for both actors back-to-back, before either
`apply_action`): both Observations come from the identical, unmutated
`GameState`. The same-snapshot invariant is still never *assumed* by the
scheduling harness - `decision_scheduling.py` records the world hash each
actor was actually snapshotted from and compares them (spec section 8).

Placement reuses `runtime_snapshot_replay.py`'s helpers and its documented
`state.update(0.0)` requirement (without it `current_interaction_target` -
and so every INTERACT/INSPECT/PLACE_TRAP/DISARM legality gate - stays
empty however close an actor is placed).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from src import constants
from src.evaluation.runtime_snapshot_replay import _fresh_state
from src.llm_controller import enumerate_legal_actions, serialize_observation
from src.observation import Observation, build_observation
from src.state import GameState

# Spec vocabulary: S0-S4 are written in terms of actors A and B. A is
# Black and B is White throughout M6G, which makes S0 match production's
# own default (Black decided first, Black applied first - game.py:421-449).
# Identity counterbalancing comes from the conditions themselves: S1
# reverses who is called first, S2 who is applied first, S4 who acts first.
ACTOR_A = constants.BLACK_SPY_ID
ACTOR_B = constants.WHITE_SPY_ID
ACTORS = (ACTOR_A, ACTOR_B)


def other_actor(actor_id: str) -> str:
    return ACTOR_B if actor_id == ACTOR_A else ACTOR_A


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def fresh_scenario_state(setup_fn: Callable[[GameState], None] | None = None) -> GameState:
    """Deterministic pre-action world: fixed seed (`_fresh_state`, seed
    6001, SOCIAL_OFF), scenario placement, then the mandatory
    `update(0.0)` - zero dt, so no game time passes."""
    state = _fresh_state(communication_enabled=False)
    if setup_fn is not None:
        setup_fn(state)
    state.update(0.0)
    return state


@dataclass(frozen=True)
class DuelSnapshot:
    state: GameState
    observations: dict[str, Observation]
    legal_actions: dict[str, list[dict[str, Any]]]
    world_state_hash: str

    @property
    def observation_a(self) -> Observation:
        return self.observations[ACTOR_A]

    @property
    def observation_b(self) -> Observation:
        return self.observations[ACTOR_B]

    @property
    def legal_actions_a(self) -> list[dict[str, Any]]:
        return self.legal_actions[ACTOR_A]

    @property
    def legal_actions_b(self) -> list[dict[str, Any]]:
        return self.legal_actions[ACTOR_B]


def build_duel_snapshot(setup_fn: Callable[[GameState], None] | None = None) -> DuelSnapshot:
    state = fresh_scenario_state(setup_fn)
    wsh = world_state_hash(state)
    observations = {actor_id: build_observation(state, actor_id) for actor_id in ACTORS}
    legal = {actor_id: enumerate_legal_actions(observations[actor_id]) for actor_id in ACTORS}
    return DuelSnapshot(state, observations, legal, wsh)


def _round(value: float) -> float:
    return round(float(value), 6)


def world_state_snapshot(state: GameState) -> dict[str, Any]:
    """Spec section 7's world state: the load-bearing, mutable world truth
    an action can change and a later action can depend on.

    Included: per-actor position, inventory, trap inventory, knowledge and
    stun/busy/alarm timers; every object's searched_by/inspected_by/trap/
    detected_by; door states; both clocks; escape bookkeeping; messages;
    and the shared RNG state (INSPECT draws from it, so apply order can
    change which actor gets which draw).

    Excluded on purpose: `event_log` (hashed separately and
    order-sensitively by `event_log_hash`), `nav_grid` (derived from
    door_states), and the bookkeeping `update()` recomputes from positions
    (current_interaction_target, visited_rooms, known_locked_objects,
    last_seen_opponent, _opponent_visible), plus presentation-only fields
    (facing, moving, distance_travelled, Black-only HUD status text).
    Keeping those out means S3/S4's extra zero-dt `update()` between the
    two applies can't masquerade as a scheduling difference by itself.
    """

    def actor_snapshot(actor_id: str) -> dict[str, Any]:
        a = state.actors[actor_id]
        return {
            "x": round(a.x, 3),
            "y": round(a.y, 3),
            "inventory": sorted(a.inventory.items),
            "trap_inventory": dict(sorted(a.trap_inventory.items())),
            "knowledge": sorted(a.knowledge),
            "stunned_timer": _round(a.stunned_timer),
            "action_cooldown": _round(a.action_cooldown),
            "alarm_reveal_timer": _round(a.alarm_reveal_timer),
        }

    return {
        "actors": {aid: actor_snapshot(aid) for aid in sorted(state.actors)},
        "object_states": {
            oid: {
                "searched_by": sorted(entry["searched_by"]),
                "inspected_by": sorted(entry["inspected_by"]),
                "trap": entry["trap"],
                "detected_by": dict(sorted(entry["detected_by"].items())),
            }
            for oid, entry in sorted(state.object_states.items())
        },
        "door_states": dict(sorted(state.door_states.items())),
        "mission_time_remaining": _round(state.mission_time_remaining),
        "elapsed_time": _round(state.elapsed_time),
        "mission_complete": dict(sorted(state.mission_complete.items())),
        "finish_order": list(state.finish_order),
        "messages": list(state.messages),
        "rng_state": hashlib.sha256(repr(state.rng.getstate()).encode("utf-8")).hexdigest(),
    }


def world_state_hash(state: GameState) -> str:
    return _hash(world_state_snapshot(state))


def event_log_hash(state: GameState, since_index: int = 0) -> str:
    """Order-sensitive hash of events logged from `since_index` onward
    (spec section 22). The event log carries no timestamps or tick
    numbers, so list order is the only ordering information it has."""
    return _hash(state.event_log[since_index:])


def game_clock(state: GameState) -> dict[str, float]:
    """Spec section 11: the two clocks a decision could illegitimately
    advance. Recorded before and after every model call."""
    return {"elapsed_time": _round(state.elapsed_time), "mission_time_remaining": _round(state.mission_time_remaining)}


def snapshot_id(scenario_family: str, world_hash: str) -> str:
    """Identifies the world a decision was made from, not the trial:
    two decisions share a snapshot_id exactly when they observed the same
    world state."""
    return f"{scenario_family}@{world_hash[:16]}"


def observation_hash(observation: Observation) -> str:
    return _hash(serialize_observation(observation))


def legal_actions_hash(legal_actions: list[dict[str, Any]]) -> str:
    return _hash(legal_actions)


def memory_hash(memory_payload: dict[str, Any]) -> str:
    return _hash(memory_payload)


def personality_hash(personality_profile: str | None) -> str:
    return _hash(personality_profile)


def json_hash(value: Any) -> str:
    return _hash(value)
