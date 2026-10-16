"""M6G.2: commitment policy as an evaluation-level harness object.

DECISION POLICY - what each model selected - is held fixed. COMMITMENT
POLICY - which already-accepted action is applied first - is the only
variable (spec sections 1, 3). A policy returns an order over the two
actor ids and nothing else: it never sees, rewrites, repairs, re-prompts or
re-decides a decision (spec section 14).

    P0 CURRENT_ANSWER_ORDER   whichever accepted response is ready first
                              applies first (production's real loop; two
                              responses ready on the same loop tick apply
                              Black then White - Game._step's fixed order)
    P1 BARRIER_ANSWER_ORDER   hold both, then apply in P0's order
    P2 FIXED_BLACK_FIRST      hold both, Black then White
    P3 FIXED_WHITE_FIRST      hold both, White then Black
    P4 ALTERNATING_FIRST      hold both; the first-commit actor alternates
                              with the logical decision boundary index,
                              from a counterbalanced initial parity

P5 (seeded balanced order) is not defined: nothing here justified adding
randomness (spec sections 4, 55).

"Production" above is the loop M6G.2 measured. Since M6G.3 the production
default is P1 (Game(commit_policy="barrier_answer_order"), with answers
visible on one tick committed in publication order) and P0 is
Game(commit_policy="answer_order").

Replay (`replay_under_policy`) maps each policy onto M6G.1's
`replay_pair`, so every policy runs the same captured decisions through the
same production validation, ActionExecutor and Engine: P0 is M6G.1's T0,
the barrier policies are its one-tick T1A/T1B in the policy's order.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Callable

from src.evaluation.conflict_scenarios import VISIBILITY_HIDDEN, VISIBILITY_OBSERVABLE, ConflictScenario
from src.evaluation.decision_scheduling import DECISION_TICK_DT, native_render
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS, json_hash, world_state_snapshot
from src.evaluation.runtime_snapshot_replay import _place_near_object
from src.evaluation.temporal_semantics import (
    APPLY_APPLIED,
    APPLY_INTENTIONAL_WAIT,
    APPLY_MOVE_UNREACHABLE,
    L0_NON_CONFLICTING,
    L1_SHARED_DOOR,
    L2_SHARED_OBJECT,
    L3_TRAP_SLOT_CONFLICT,
    L4_DISARM_CONFLICT,
    L5_ACTION_INVALIDATES_OTHER,
    STALE_AND_INVALID,
    T0,
    T1A,
    T1B,
    _cost,
    _decide_from_raw,
    _outcome,
    actor_self_state,
    build_temporal_scenarios,
    own_effect_view,
    replay_pair,
)
from src.state import GameState

P0 = "P0_CURRENT_ANSWER_ORDER"
P1 = "P1_BARRIER_ANSWER_ORDER"
P2 = "P2_FIXED_BLACK_FIRST"
P3 = "P3_FIXED_WHITE_FIRST"
P4 = "P4_ALTERNATING_FIRST"
POLICY_IDS = (P0, P1, P2, P3, P4)
BARRIER_POLICIES = (P1, P2, P3, P4)


@dataclass(frozen=True)
class CommitmentPolicy:
    policy_id: str
    waits_for_both: bool
    first_commit_rule: str
    depends_on_latency: bool
    depends_on_actor_id: bool
    deterministic: bool

    def order(self, decision_pair: dict[str, Any] | None, response_ready_order: tuple[str, str] | None,
              decision_boundary_index: int = 0, seed: int = 0) -> tuple[str, str]:
        """(first, second) commit actor. `response_ready_order` is the order
        the loop saw the two responses become ready at tick resolution, or
        None when both became ready on the same tick; `seed` is P4's
        initial parity (0: Black first at boundary 0, 1: White first).
        `decision_pair` is accepted for the spec's signature and never
        read: the order may not depend on what was decided."""
        del decision_pair
        if self.policy_id in (P0, P1):
            return tuple(response_ready_order) if response_ready_order else (ACTOR_A, ACTOR_B)
        if self.policy_id == P2:
            return ACTOR_A, ACTOR_B
        if self.policy_id == P3:
            return ACTOR_B, ACTOR_A
        if self.policy_id == P4:
            first = ACTORS[(decision_boundary_index + seed) % 2]
            return first, ACTOR_B if first == ACTOR_A else ACTOR_A
        raise ValueError(self.policy_id)


POLICIES: dict[str, CommitmentPolicy] = {
    P0: CommitmentPolicy(P0, False, "first response ready applies first (same-tick readiness: Black, then White)", True, False, False),
    P1: CommitmentPolicy(P1, True, "hold both; apply in P0's response-ready order", True, False, False),
    P2: CommitmentPolicy(P2, True, "hold both; Black first", False, True, True),
    P3: CommitmentPolicy(P3, True, "hold both; White first", False, True, True),
    P4: CommitmentPolicy(P4, True, "hold both; first actor alternates with the decision-boundary index from a counterbalanced parity",
                         False, False, True),
}


def policy_table() -> list[dict[str, Any]]:
    """Spec section 66. `deterministic` means the order is a function of the
    policy inputs alone; P0/P1 depend on provider latency, which is not."""
    notes = {
        P0: "production behavior; first commit follows provider latency plus the <=50 ms request head start of the first-called actor",
        P1: "same order as P0, so the same world semantics; only the request lifecycle can differ",
        P2: "actor-prioritized control (maximum actor-order exposure), not a candidate by virtue of being deterministic",
        P3: "actor-prioritized control, mirror of P2",
        P4: "depends on the actor id only through the boundary parity, which is counterbalanced across trials",
    }
    return [{**asdict(p), "note": notes[pid]} for pid, p in POLICIES.items()]


# -- scenarios (spec sections 10, 28) -------------------------------------------------

C0 = "C0_NON_CONFLICTING"
C1 = "C1_SAME_TARGET"
C2 = "C2_SHARED_DOOR"
C4 = "C4_SHARED_TRAP_SLOT"
C5 = "C5_DISARM_RACE"
C5M = "C5M_DISARM_RACE_MIRRORED"
C6 = "C6_SEARCH_AFTER_TRAP"
C6M = "C6M_SEARCH_AFTER_TRAP_MIRRORED"
SCENARIO_IDS = (C0, C1, C2, C4, C5, C5M, C6, C6M)
# M6G.2 id -> the M6G.1 L family whose setup/pair it is (mirrors are new)
SCENARIO_TO_FAMILY = {C0: L0_NON_CONFLICTING, C1: L2_SHARED_OBJECT, C2: L1_SHARED_DOOR, C4: L3_TRAP_SLOT_CONFLICT,
                      C5: L4_DISARM_CONFLICT, C6: L5_ACTION_INVALIDATES_OTHER}
FAMILY_TO_SCENARIO = {v: k for k, v in SCENARIO_TO_FAMILY.items()}
MIRROR_OF = {C5M: C5, C6M: C6}


def _setup_c5_mirrored(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "ambassador_safe")
    _place_near_object(state, ACTOR_B, "ambassador_safe")
    entry = state._object_entry("ambassador_safe")
    entry["inspected_by"] = {ACTOR_A, ACTOR_B}
    entry["trap"] = {"type": "bomb", "owner_id": ACTOR_B, "armed": True}
    entry["detected_by"] = {ACTOR_A: "bomb"}


def _setup_c6_mirrored(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "desk_01")
    _place_near_object(state, ACTOR_B, "desk_01")
    state._object_entry("desk_01")["inspected_by"] = {ACTOR_B}


def build_policy_scenarios() -> dict[str, ConflictScenario]:
    """M6G.1's L scenarios under the M6G.2 ids, plus identity mirrors of the
    two asymmetric conflicts (C5: who owns the trap; C6: who places it), so
    an actor-specific role cannot pass for a commitment-policy effect."""
    l_scen = build_temporal_scenarios()
    out = {sid: _renamed(l_scen[fam], sid) for sid, fam in SCENARIO_TO_FAMILY.items()}
    c5, c6 = l_scen[L4_DISARM_CONFLICT], l_scen[L5_ACTION_INVALIDATES_OTHER]
    out[C5M] = ConflictScenario(
        C5M, "c5m_disarm_race_mirrored", "C5 with roles swapped: White owns the bomb on ambassador_safe, Black has detected it; both DISARM.",
        _setup_c5_mirrored, dict(c5.raw_b), dict(c5.raw_a), "object_states['ambassador_safe']['trap'] (shared)", VISIBILITY_OBSERVABLE,
        expected_apply_order_effect=True, expected_second_observation_changes=True,
    )
    out[C6M] = ConflictScenario(
        C6M, "c6m_search_after_trap_mirrored", "C6 with roles swapped: White (inspected desk_01) places a bomb on it; Black searches desk_01.",
        _setup_c6_mirrored, dict(c6.raw_b), dict(c6.raw_a), "object_states['desk_01']['trap'] (shared slot, hidden from Black)",
        VISIBILITY_HIDDEN, expected_apply_order_effect=True, expected_second_observation_changes=False,
    )
    return {sid: out[sid] for sid in SCENARIO_IDS}


def _renamed(s: ConflictScenario, sid: str) -> ConflictScenario:
    return ConflictScenario(sid, s.scenario_id, s.description, s.setup_fn, dict(s.raw_a), dict(s.raw_b), s.shared_state,
                            s.side_effect_visibility, s.expected_apply_order_effect, s.expected_second_observation_changes,
                            s.negative_control)


# -- replay under a policy (spec sections 15-20) ----------------------------------------


def replay_under_policy(scenario: ConflictScenario, raws: dict[str, dict[str, Any]], policy_id: str, *,
                        response_ready_order: tuple[str, str] | None, decision_boundary_index: int = 0, parity: int = 0,
                        pair_id: str, decision_boundary_id: str, model_id: str = "replay",
                        expected_request_hashes: dict[str, str] | None = None,
                        render_fn: Callable[..., tuple[str, str]] = native_render) -> dict[str, Any]:
    """One captured pair under one policy, zero model calls. P0 commits the
    first-ready decision on a frozen tick and the second on the next live
    tick (M6G.1's T0; a same-tick tie is production's one-tick Black-then-
    White commit); barrier policies commit both on one live tick in the
    policy's order (M6G.1's T1A/T1B)."""
    policy = POLICIES[policy_id]
    first, second = policy.order(raws, response_ready_order, decision_boundary_index, parity)
    kwargs = dict(pair_id=pair_id, model_id=model_id, expected_request_hashes=expected_request_hashes, render_fn=render_fn)
    if policy_id == P0 and response_ready_order:
        row = replay_pair(scenario, raws, T0, arrival_order=(first, second), **kwargs)
    else:
        row = replay_pair(scenario, raws, T1A if first == ACTOR_A else T1B, **kwargs)
    if row["apply_order"] != [first, second]:
        raise RuntimeError(f"{pair_id}/{policy_id}: replay applied {row['apply_order']}, policy ordered {[first, second]}")
    ready_rank = {a: (response_ready_order.index(a) + 1) if response_ready_order else None for a in ACTORS}
    recs = row["records"]
    row.update({
        "commitment_policy": policy_id,
        "p4_parity": parity if policy_id == P4 else None,
        "decision_boundary_id": decision_boundary_id,
        "decision_boundary_index": decision_boundary_index,
        "response_ready_order": list(response_ready_order) if response_ready_order else None,
        "response_ready_tie": response_ready_order is None,
        "first_commit_actor": first,
        "second_commit_actor": second,
        "commit_rank": {first: 1, second: 2},
        "response_ready_rank": ready_rank,
        "first_ready_is_first_commit": (response_ready_order[0] == first) if response_ready_order else None,
        "world_hash_after_first_commit": row["world_hash_after_first_apply"],
        "first_action_cost": recs[first]["time_cost"],
        "second_action_cost": recs[second]["time_cost"],
        "first_result": f"{recs[first]['apply_outcome']}/{recs[first]['consequence']}",
        "second_result": f"{recs[second]['apply_outcome']}/{recs[second]['consequence']}",
        "decision_hashes": {a: recs[a]["decision_hash"] for a in ACTORS},
    })
    return row


# -- commitment effect classification (spec sections 30-31) ----------------------------

NO_COMMITMENT_POLICY_EFFECT = "NO_COMMITMENT_POLICY_EFFECT"
COST_ONLY_COMMITMENT_EFFECT = "COST_ONLY_COMMITMENT_EFFECT"
SHARED_RESOURCE_COMMITMENT_EFFECT = "SHARED_RESOURCE_COMMITMENT_EFFECT"
STALE_VALIDITY_COMMITMENT_EFFECT = "STALE_VALIDITY_COMMITMENT_EFFECT"
WORLD_OUTCOME_COMMITMENT_EFFECT = "WORLD_OUTCOME_COMMITMENT_EFFECT"
MIXED = "MIXED"


def commitment_effect(a_first: dict[str, Any], b_first: dict[str, Any]) -> dict[str, Any]:
    """The same decision pair committed Black-first vs White-first. Every
    policy's outcome for a single boundary equals one of these two, so this
    is what commitment policy can decide for the pair.

    identical world and events                      -> NO_COMMITMENT_POLICY_EFFECT
    the second decision is refused / changed in an order:
      same action on the same target, second refused
      in both orders (the first takes the one slot)  -> SHARED_RESOURCE_COMMITMENT_EFFECT
      otherwise (the first changes whether or how
      the second takes effect)                       -> STALE_VALIDITY_COMMITMENT_EFFECT
    both stay valid in both orders:
      only clocks/cooldowns differ (effect hash =)   -> COST_ONLY_COMMITMENT_EFFECT
      any other world difference                     -> WORLD_OUTCOME_COMMITMENT_EFFECT

    Validity is checked before cost: in a disarm race the world ends up the
    same either way (the trap is gone), but who is refused is decided by
    the order - that is not a mere cost difference.
    """
    same_world = (a_first["world_state_hash_after"], a_first["event_multiset_hash"]) == (
        b_first["world_state_hash_after"], b_first["event_multiset_hash"])
    same_effect = a_first["effect_state_hash_after"] == b_first["effect_state_hash_after"]
    da = a_first["records"][ACTOR_A]["decision"]
    db = a_first["records"][ACTOR_B]["decision"]
    same_action_same_target = da["action"] == db["action"] and da["target_id"] == db["target_id"] and da["action"] not in ("wait", "move")
    second_invalid = [r["records"][r["apply_order"][1]]["staleness_class"] == STALE_AND_INVALID for r in (a_first, b_first)]
    if same_world:
        label = NO_COMMITMENT_POLICY_EFFECT
    elif same_action_same_target and all(second_invalid):
        label = SHARED_RESOURCE_COMMITMENT_EFFECT
    elif any(second_invalid):
        label = STALE_VALIDITY_COMMITMENT_EFFECT
    elif same_effect:
        label = COST_ONLY_COMMITMENT_EFFECT
    else:
        label = WORLD_OUTCOME_COMMITMENT_EFFECT
    return {
        "classification": label,
        "same_world": same_world,
        "same_effect_state": same_effect,
        "costs_black_first": {a: a_first["records"][a]["time_cost"] for a in ACTORS},
        "costs_white_first": {a: b_first["records"][a]["time_cost"] for a in ACTORS},
        "second_stale_and_invalid": {"black_first": second_invalid[0], "white_first": second_invalid[1]},
        "contested_black_first": a_first["contested_effects"],
        "contested_white_first": b_first["contested_effects"],
    }


def scenario_effect(labels: list[str]) -> str:
    distinct = sorted(set(labels))
    return distinct[0] if len(distinct) == 1 else MIXED


# -- counterfactual on any snapshot (boundaries after the first) -------------------------


def counterfactual_on_state(state: GameState, actor_id: str, raw: dict[str, Any], *, family: str) -> dict[str, Any]:
    """M6G.1's fresh counterfactual generalized to an arbitrary snapshot: the
    decision applied alone to a copy of the world it was made from, on one
    live tick. Legality at the snapshot and the reference cost/effect.

    A state whose apply_action/update were replaced on the instance (a
    probe's wrappers) is refused: the copy would keep the wrapper closures
    and apply to whatever game they wrap instead of to the copy."""
    overridden = [n for n in ("apply_action", "update") if n in vars(state)]
    if overridden:
        raise ValueError(f"counterfactual on a state with instance-level {overridden}: pass an unpatched copy")
    state = copy.deepcopy(state)
    d = _decide_from_raw(state, actor_id, raw, family=family, model_id="counterfactual", provider_name="bedrock")
    before = world_state_snapshot(state)
    events_before = len(state.event_log)
    cooldown_before = state.actors[actor_id].action_cooldown
    state.apply_action(actor_id, d.first_action, DECISION_TICK_DT)
    after = world_state_snapshot(state)
    types = [e["type"] for e in state.event_log[events_before:] if e.get("actor_id") == actor_id]
    outcome = _outcome(d, DECISION_TICK_DT, json_hash(before) != json_hash(after), types)
    return {
        "effective_outcome": outcome,
        "engine_valid": outcome in (APPLY_APPLIED, APPLY_INTENTIONAL_WAIT) or (d.decision["action"] == "move" and outcome != APPLY_MOVE_UNREACHABLE),
        "time_cost": _cost(cooldown_before, state.actors[actor_id].action_cooldown),
        "target_state_after": own_effect_view(after, actor_id, d.decision),
        "self_state_after": actor_self_state(after, actor_id),
        "event_types": types,
    }


# -- request lifecycle (spec sections 32-35) ------------------------------------------------

IDLE = "IDLE"
REQUEST_IN_FLIGHT = "REQUEST_IN_FLIGHT"
RESPONSE_READY_UNCOMMITTED = "RESPONSE_READY_UNCOMMITTED"
COMMITTED = "COMMITTED"
REQUEST_STATES = (IDLE, REQUEST_IN_FLIGHT, RESPONSE_READY_UNCOMMITTED, COMMITTED)

# How each state maps onto the production controllers (LLMController and
# StrandsAgentAdapter keep the same fields). Only REQUEST_IN_FLIGHT is an
# explicit code state; the other three are implied by it plus the order of
# the background thread and the next get_action call.
REQUEST_STATE_CODE_MAP = {
    IDLE: "_thinking_in_progress False and no validated decision waiting (nothing requested yet, or the last decision has been returned)",
    REQUEST_IN_FLIGHT: "_thinking_in_progress True: status THINKING, _request_action_background running (_thinking_thread)",
    RESPONSE_READY_UNCOMMITTED: "no explicit field: the thread has set last_action, incremented decisions_completed and cleared "
                                "_thinking_in_progress, but no get_action has returned the decision to Game._step yet (production) "
                                "or the barrier has not released it (P1-P4)",
    COMMITTED: "no explicit field: get_action returned the decision's first GameAction and GameState.apply_action ran it",
}

# What a polled get_action did with the decision trigger (the observation
# digest, excluding the time field), classified from outside.
ATTEMPT_LAUNCHED = "REQUEST_LAUNCHED"
ATTEMPT_ABSORBED = "CHANGE_ABSORBED_IN_FLIGHT"
ATTEMPT_THROTTLED = "CHANGE_THROTTLED"
ATTEMPT_NO_CHANGE = "NO_CHANGE"
ATTEMPT_BUSY = "NOT_EVALUATED_BUSY"
ATTEMPT_EXECUTING = "NOT_EVALUATED_EXECUTING"
HELD_BY_BARRIER = "HELD_BY_BARRIER"


def hp09_violation(prior_state: str, launched: bool) -> bool:
    """HP09: an actor must not launch a replacement request for a logical
    decision boundary while a prior response for it is ready but
    uncommitted."""
    return launched and prior_state == RESPONSE_READY_UNCOMMITTED
