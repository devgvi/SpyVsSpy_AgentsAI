"""M6G.1: stale decisions, commitment order and action durations as
recorded, testable harness semantics.

Five moments are kept apart (spec section 1): SNAPSHOT (observation built),
READY (decision available), COMMIT (handed to the execution path), APPLY
(Engine processes it) and COMPLETE (the modeled duration has elapsed).

Everything reuses production code unchanged: `validate_model_output` for
decisions, `ActionExecutor.start/continue_` for turning a decision into
per-tick GameActions (MOVE paths are computed from the snapshot
observation, exactly as the production background thread does),
`GameState.apply_action` / `update` for the Engine. Nothing here adds a
mechanic, re-decides a stale decision, or rewrites one (spec sections
20-21, 44).

Commit conditions replayed over a captured decision pair (zero model
calls):

    T0  CURRENT_ANSWER_ORDER_COMMIT  first-ready decision applies on a
                                     frozen tick (dt=0, the other actor is
                                     still thinking), idle frozen ticks
                                     follow, the second applies on the
                                     first live tick - the production
                                     loop's tick structure
    T1A BARRIER_APPLY_A_THEN_B       both decisions held, then applied on
    T1B BARRIER_APPLY_B_THEN_A       one live tick in the stated order
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from src.actions import ActionType, GameAction
from src.agents.action_lifecycle import PATH_UNREACHABLE, ActionExecutor
from src.evaluation.conflict_scenarios import (
    C0_NON_CONFLICTING,
    C1_SAME_TARGET,
    C2_SHARED_DOOR,
    C4_TRAP_CONFLICT,
    C5_INVALIDATES,
    VISIBILITY_HIDDEN,
    ConflictScenario,
    build_conflict_scenarios,
)
from src.evaluation.decision_scheduling import (
    APPLY_APPLIED,
    APPLY_FAILURE_EVENT_ONLY,
    APPLY_INTENTIONAL_WAIT,
    APPLY_REFUSED_NO_OP,
    DECISION_TICK_DT,
    _decision_key,
    _is_listed,
    build_request,
    classify_apply_outcome,
    native_render,
)
from src.evaluation.duel_snapshot import (
    ACTOR_A,
    ACTOR_B,
    ACTORS,
    event_log_hash,
    fresh_scenario_state,
    game_clock,
    json_hash,
    snapshot_id,
    world_state_hash,
    world_state_snapshot,
)
from src.evaluation.runtime_snapshot_replay import _place_near_object
from src.evaluation.runtime_variance_analysis import content_hash
from src.llm.personality import DEFAULT
from src.llm_controller import enumerate_legal_actions, validate_model_output
from src.observation import build_observation
from src.state import GameState

MOMENTS = ("snapshot", "ready", "commit", "apply", "complete")

# spec section 8
INSTANTANEOUS = "INSTANTANEOUS"
DURATION_AS_COST_ONLY = "DURATION_AS_COST_ONLY"
EFFECT_AT_START_WITH_BUSY_PERIOD = "EFFECT_AT_START_WITH_BUSY_PERIOD"
EFFECT_AT_COMPLETION = "EFFECT_AT_COMPLETION"
CONTINUOUS_OR_MULTI_TICK = "CONTINUOUS_OR_MULTI_TICK"
UNKNOWN = "UNKNOWN"

# spec section 5 (RELEVANT_STATE_STALE is the union of the last two)
FRESH = "FRESH"
WORLD_STALE_IRRELEVANT = "WORLD_STALE_IRRELEVANT"
RELEVANT_STATE_STALE = "RELEVANT_STATE_STALE"
STALE_AND_INVALID = "STALE_AND_INVALID"
STALE_BUT_VALID = "STALE_BUT_VALID"
# Listed at the snapshot but already refused by the Engine there - a
# legality-surface gap (M6G audit section 7), not staleness.
INVALID_AT_SNAPSHOT = "INVALID_AT_SNAPSHOT"

# spec section 36
ACCEPTED_UNCHANGED = "ACCEPTED_UNCHANGED"
REFUSED = "REFUSED"
NO_OP = "NO_OP"
DIFFERENT_COST = "DIFFERENT_COST"
DIFFERENT_WORLD_EFFECT = "DIFFERENT_WORLD_EFFECT"
FALLBACK = "FALLBACK"

# spec section 26
NO_TEMPORAL_CONFLICT = "NO_TEMPORAL_CONFLICT"
APPLICATION_ORDER_ONLY = "APPLICATION_ORDER_ONLY"
STALE_BUT_VALID_EFFECT = "STALE_BUT_VALID_EFFECT"
STALE_AND_INVALID_EFFECT = "STALE_AND_INVALID_EFFECT"
DURATION_COST_ASYMMETRY = "DURATION_COST_ASYMMETRY"
IN_FLIGHT_CONFLICT = "IN_FLIGHT_CONFLICT"
COMPLETION_ORDER_EFFECT = "COMPLETION_ORDER_EFFECT"
MIXED = "MIXED"

T0 = "T0_CURRENT_ANSWER_ORDER_COMMIT"
T1A = "T1A_BARRIER_APPLY_A_THEN_B"
T1B = "T1B_BARRIER_APPLY_B_THEN_A"
COMMIT_CONDITIONS = (T0, T1A, T1B)

# A MOVE committed on a frozen tick has no displacement yet; its effect
# starts on the first live tick. Not a refusal.
APPLY_COMMITTED_FROZEN = "COMMITTED_FROZEN"
APPLY_MOVE_UNREACHABLE = "MOVE_PATH_UNREACHABLE"

# -- scenarios (spec section 23) ----------------------------------------------

L0_NON_CONFLICTING = "L0_NON_CONFLICTING"
L1_SHARED_DOOR = "L1_SHARED_DOOR"
L2_SHARED_OBJECT = "L2_SHARED_OBJECT"
L3_TRAP_SLOT_CONFLICT = "L3_TRAP_SLOT_CONFLICT"
L4_DISARM_CONFLICT = "L4_DISARM_CONFLICT"
L5_ACTION_INVALIDATES_OTHER = "L5_ACTION_INVALIDATES_OTHER"
L_FAMILIES = (L0_NON_CONFLICTING, L1_SHARED_DOOR, L2_SHARED_OBJECT, L3_TRAP_SLOT_CONFLICT, L4_DISARM_CONFLICT, L5_ACTION_INVALIDATES_OTHER)

# M6G conflict family each L scenario reuses unchanged (L5 is new).
L_TO_M6G = {
    L0_NON_CONFLICTING: C0_NON_CONFLICTING,
    L1_SHARED_DOOR: C2_SHARED_DOOR,
    L2_SHARED_OBJECT: C1_SAME_TARGET,
    L3_TRAP_SLOT_CONFLICT: C4_TRAP_CONFLICT,
    L4_DISARM_CONFLICT: C5_INVALIDATES,
}


def _setup_l5(state: GameState) -> None:
    _place_near_object(state, ACTOR_A, "desk_01")
    _place_near_object(state, ACTOR_B, "desk_01")
    state._object_entry("desk_01")["inspected_by"] = {ACTOR_A}


def build_temporal_scenarios() -> dict[str, ConflictScenario]:
    """L0-L4 are M6G's own scenarios (same setups, same scripted pairs);
    L5 is the one new pair: a different-type invalidation, where A's hidden
    PLACE_TRAP turns B's SEARCH of the same object into a trap trigger."""
    m6g = build_conflict_scenarios()
    out: dict[str, ConflictScenario] = {}
    for fam, m6g_fam in L_TO_M6G.items():
        s = m6g[m6g_fam]
        out[fam] = ConflictScenario(
            fam, s.scenario_id, s.description, s.setup_fn, s.raw_a, s.raw_b, s.shared_state, s.side_effect_visibility,
            s.expected_apply_order_effect, s.expected_second_observation_changes, s.negative_control,
        )
    out[L5_ACTION_INVALIDATES_OTHER] = ConflictScenario(
        L5_ACTION_INVALIDATES_OTHER, "l5_trap_then_search",
        "A (inspected desk_01) places a bomb on it; B searches desk_01 - A's hidden trap pre-empts B's search if applied first.",
        _setup_l5,
        {"action": "place_trap", "target_id": "desk_01", "trap_type": "bomb", "reason": "m6g.1 scripted pair"},
        {"action": "interact", "target_id": "desk_01", "reason": "m6g.1 scripted pair"},
        "object_states['desk_01']['trap'] (shared slot, hidden from B)", VISIBILITY_HIDDEN,
        expected_apply_order_effect=True, expected_second_observation_changes=False,
    )
    return out


def m6g_family_for(family: str) -> str | None:
    return L_TO_M6G.get(family)


# -- relevant state (spec section 4) ------------------------------------------


def _entry(world: dict[str, Any], object_id: str) -> dict[str, Any]:
    return world["object_states"].get(object_id, {"searched_by": [], "inspected_by": [], "trap": None, "detected_by": {}})


def relevant_state(world: dict[str, Any], actor_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    """The part of the world a decision's outcome can depend on - never the
    whole world, so unrelated change is not counted as staleness.

    door INTERACT  -> that door's state
    search         -> the target's trap slot + whether THIS actor searched it
    INSPECT        -> trap slot, own inspection, own detection
    PLACE_TRAP     -> trap slot, own inspection, own count of that trap type
    DISARM         -> trap slot, own detection
    MOVE           -> every door state (the only world facts that shape a path)
    WAIT           -> nothing
    """
    action, target = decision.get("action"), decision.get("target_id")
    if action == "wait" or action is None:
        return {}
    if action == "move":
        return {"door_states": world["door_states"]}
    if target in world["door_states"]:
        return {"door_state": world["door_states"][target]}
    entry = _entry(world, target)
    trap = entry["trap"]
    if action == "interact":
        return {"trap": trap, "searched_by_self": actor_id in entry["searched_by"]}
    if action == "inspect":
        return {"trap": trap, "inspected_by_self": actor_id in entry["inspected_by"], "detected_by_self": entry["detected_by"].get(actor_id)}
    if action == "place_trap":
        trap_type = decision.get("trap_type")
        return {"trap": trap, "inspected_by_self": actor_id in entry["inspected_by"],
                "own_traps_of_type": world["actors"][actor_id]["trap_inventory"].get(trap_type)}
    if action == "disarm":
        return {"trap": trap, "detected_by_self": entry["detected_by"].get(actor_id)}
    return {"target": target, "entry": entry}


def relevant_state_hash(world: dict[str, Any], actor_id: str, decision: dict[str, Any]) -> str:
    return json_hash(relevant_state(world, actor_id, decision))


def own_effect_view(world: dict[str, Any], actor_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    """What this decision's own effect looks like from its actor's side,
    for comparing a stale apply with its fresh counterfactual. A MOVE's
    effect is its position (compared through `actor_self_state`), so its
    view of the target is empty."""
    if decision.get("action") == "move":
        return {}
    return relevant_state(world, actor_id, decision)


def target_state(world: dict[str, Any], target_id: str | None) -> Any:
    if target_id is None:
        return None
    if target_id in world["door_states"]:
        return {"door_state": world["door_states"][target_id]}
    return _entry(world, target_id)


def actor_self_state(world: dict[str, Any], actor_id: str) -> dict[str, Any]:
    a = world["actors"][actor_id]
    return {"x": a["x"], "y": a["y"], "inventory": a["inventory"], "knowledge": a["knowledge"],
            "trap_inventory": a["trap_inventory"], "stunned": a["stunned_timer"] > 0}


def effect_state_hash(world: dict[str, Any]) -> str:
    """World truth without clocks and self timers: equal exactly when two
    schedules produced the same effects, even if one ran longer."""
    stripped = json.loads(json.dumps(world, default=str))
    stripped.pop("mission_time_remaining", None)
    stripped.pop("elapsed_time", None)
    for a in stripped["actors"].values():
        for k in ("stunned_timer", "action_cooldown", "alarm_reveal_timer"):
            a.pop(k, None)
    return json_hash(stripped)


def classify_staleness(*, world_changed: bool, relevant_changed: bool, engine_valid_at_snapshot: bool, legal_at_apply: bool) -> str:
    if not engine_valid_at_snapshot:
        return INVALID_AT_SNAPSHOT
    if not legal_at_apply:
        return STALE_AND_INVALID
    if relevant_changed:
        return STALE_BUT_VALID
    if world_changed:
        return WORLD_STALE_IRRELEVANT
    return FRESH


def classify_consequence(record: dict[str, Any], counterfactual: dict[str, Any]) -> str:
    if not record["first_pass_valid"]:
        return FALLBACK
    if record["decision"]["action"] == "wait":
        return NO_OP
    if record["apply_outcome"] in (APPLY_REFUSED_NO_OP, APPLY_FAILURE_EVENT_ONLY, APPLY_MOVE_UNREACHABLE):
        return REFUSED
    same_effect = (
        record["target_state_after"] == counterfactual["target_state_after"]
        and record["self_state_after"] == counterfactual["self_state_after"]
        and record["effective_outcome"] == counterfactual["effective_outcome"]
    )
    if same_effect and record["time_cost"] == counterfactual["time_cost"]:
        return ACCEPTED_UNCHANGED
    if same_effect:
        return DIFFERENT_COST
    return DIFFERENT_WORLD_EFFECT


# -- decisions ------------------------------------------------------------------


@dataclass
class _Decided:
    actor_id: str
    raw: dict[str, Any]
    first_pass_valid: bool
    validation_error: str | None
    decision: dict[str, Any]
    listed_at_snapshot: bool
    executor: ActionExecutor
    first_action: GameAction
    start_boundary: str | None
    provider_request_hash: str
    agent_request_semantic_hash: str


def _decide_from_raw(state: GameState, actor_id: str, raw: dict[str, Any], *, family: str, model_id: str, provider_name: str,
                     render_fn: Callable[..., tuple[str, str]] = native_render, personality_profile: str | None = DEFAULT) -> _Decided:
    """The READY moment for a captured decision: production validation
    against the actor's own snapshot, then `ActionExecutor.start` with the
    snapshot observation - what the production background thread does
    right before it signals the decision ready."""
    observation = build_observation(state, actor_id)
    legal = enumerate_legal_actions(observation)
    request = build_request(actor_id, observation, legal, scenario_family=family, model_id=model_id, provider_name=provider_name,
                            personality_profile=personality_profile)
    system_prompt, user_prompt = render_fn(request)
    validation = validate_model_output(observation, raw or {}, legal)
    decision = _decision_key(validation.validated_action if validation.valid else None)
    message = (validation.validated_action or {}).get("message") if validation.valid else None
    executor = ActionExecutor()
    first_action, boundary, _ = executor.start(
        observation, f"{family}:{actor_id}", decision["action"], decision["target_id"], decision["trap_type"], state.mission_time_remaining,
        message=message,
    )
    return _Decided(
        actor_id, dict(raw or {}), bool(validation.valid), validation.error_type, decision, _is_listed(decision, legal),
        executor, first_action, boundary, content_hash(system_prompt, user_prompt), request.request_semantic_hash(),
    )


def _outcome(decided: _Decided, dt: float, world_changed: bool, event_types: list[str]) -> str:
    if decided.decision["action"] == "move":
        if decided.start_boundary == PATH_UNREACHABLE:
            return APPLY_MOVE_UNREACHABLE
        if dt == 0.0:
            return APPLY_COMMITTED_FROZEN
    return classify_apply_outcome(decided.decision["action"], world_changed, event_types)


def fresh_counterfactual(scenario: ConflictScenario, actor_id: str, raw: dict[str, Any], *, model_id: str = "counterfactual",
                         provider_name: str = "bedrock") -> dict[str, Any]:
    """The same decision applied alone to its own snapshot on one live tick:
    what it would have done had nothing changed in between. Gives both the
    Engine's legality at the snapshot and the reference effect/cost."""
    state = fresh_scenario_state(scenario.setup_fn)
    d = _decide_from_raw(state, actor_id, raw, family=scenario.family, model_id=model_id, provider_name=provider_name)
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


def _cost(cooldown_before: float, cooldown_after: float) -> float:
    return round(cooldown_after, 6) if cooldown_before == 0 and cooldown_after > 0 else 0.0


# -- the schedule --------------------------------------------------------------


def _tick_actions(decided: dict[str, _Decided], pending: dict[str, bool], state: GameState, committed: set[str]) -> dict[str, GameAction]:
    """What each actor's controller returns this tick: the decision on its
    commit tick, then the executor's continuation (MOVE in progress) or
    WAIT (instant action done / busy) - never a new decision."""
    actions: dict[str, GameAction] = {}
    for actor_id in ACTORS:
        d = decided[actor_id]
        if pending.get(actor_id):
            actions[actor_id] = d.first_action
        elif actor_id in committed and d.executor.in_progress:
            action, _ = d.executor.continue_(build_observation(state, actor_id))
            actions[actor_id] = action
        else:
            actions[actor_id] = GameAction(ActionType.WAIT)
    return actions


def replay_pair(
    scenario: ConflictScenario,
    raws: dict[str, dict[str, Any]],
    condition: str,
    *,
    arrival_order: tuple[str, str] | None = None,
    pair_id: str = "pair",
    model_id: str = "replay",
    provider_name: str = "bedrock",
    expected_request_hashes: dict[str, str] | None = None,
    render_fn: Callable[..., tuple[str, str]] = native_render,
    idle_frozen_ticks: int = 1,
    observed_schedule: list[tuple[float, list[str]]] | None = None,
) -> dict[str, Any]:
    """Replay one captured decision pair under a commit condition. Both
    decisions are validated and started from the one frozen snapshot
    before anything is applied; no model is called; a stale decision is
    applied as decided (spec sections 20-21).

    `observed_schedule` (T0 only) replaces the idealized T0 tick plan with
    the one a live trial actually ran - [(dt, [actors committing])] from
    the first commit tick to the second - so a live outcome can be
    reproduced exactly, including a handoff tick that ran live."""
    if condition == T0 and arrival_order is None and observed_schedule is None:
        raise ValueError("T0 needs the observed or logical arrival order")
    state = fresh_scenario_state(scenario.setup_fn)
    snapshot = world_state_snapshot(state)
    snapshot_hash = json_hash(snapshot)
    events_start = len(state.event_log)
    decided = {a: _decide_from_raw(state, a, raws[a], family=scenario.family, model_id=model_id, provider_name=provider_name, render_fn=render_fn)
               for a in ACTORS}
    if expected_request_hashes:
        for a in ACTORS:
            if expected_request_hashes.get(a) not in (None, decided[a].provider_request_hash):
                raise RuntimeError(f"{pair_id}: rebuilt request for {a} is not the request the decision was made for")

    # Commit plan: a list of ticks, each (dt, actors committing this tick).
    if condition == T0 and observed_schedule is not None:
        plan = [(float(dt), set(committing)) for dt, committing in observed_schedule]
        apply_sequence = [a for _, committing in observed_schedule for a in ACTORS if a in committing]
        if sorted(apply_sequence) != sorted(ACTORS):
            raise ValueError(f"{pair_id}: observed schedule must commit each actor exactly once")
        arrival_order = tuple(apply_sequence)
    elif condition == T0:
        first, second = arrival_order
        plan = [(0.0, {first})] + [(0.0, set())] * idle_frozen_ticks + [(DECISION_TICK_DT, {second})]
        apply_sequence = [first, second]
    elif condition in (T1A, T1B):
        order = (ACTOR_A, ACTOR_B) if condition == T1A else (ACTOR_B, ACTOR_A)
        plan = [(DECISION_TICK_DT, set(order))]
        apply_sequence = list(order)
    else:
        raise ValueError(condition)

    committed: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    world_hash_after_first_apply: str | None = None
    tick = 0
    for dt, committing in plan:
        tick += 1
        pending = {a: a in committing for a in ACTORS}
        actions = _tick_actions(decided, pending, state, committed)
        # Within one tick the production loop applies Black then White; a
        # barrier applies in its own stated order.
        within_tick = apply_sequence if len(committing) == 2 else list(ACTORS)
        for actor_id in within_tick:
            before = world_state_snapshot(state)
            clock = game_clock(state)
            events_before = len(state.event_log)
            cooldown_before = state.actors[actor_id].action_cooldown
            listed_at_apply = None
            if pending[actor_id]:
                # agent-visible legality at apply time (measurement only)
                listed_at_apply = _is_listed(decided[actor_id].decision, enumerate_legal_actions(build_observation(state, actor_id)))
            state.apply_action(actor_id, actions[actor_id], dt)
            after = world_state_snapshot(state)
            new_events = state.event_log[events_before:]
            if pending[actor_id]:
                d = decided[actor_id]
                types = [e["type"] for e in new_events if e.get("actor_id") == actor_id]
                outcome = _outcome(d, dt, json_hash(before) != json_hash(after), types)
                records[actor_id] = {
                    "actor_id": actor_id,
                    "decision_id": f"{pair_id}:{actor_id}",
                    "snapshot_id": snapshot_id(scenario.family, snapshot_hash),
                    "snapshot_world_hash": snapshot_hash,
                    "snapshot_relevant_hash": relevant_state_hash(snapshot, actor_id, d.decision),
                    "apply_world_hash": json_hash(before),
                    "apply_relevant_hash": relevant_state_hash(before, actor_id, d.decision),
                    "apply_game_clock": clock,
                    "apply_tick": tick,
                    "apply_dt": dt,
                    "apply_index": apply_sequence.index(actor_id),
                    "raw_model_output": d.raw,
                    "first_pass_valid": d.first_pass_valid,
                    "validation_error": d.validation_error,
                    "decision": d.decision,
                    "decision_hash": json_hash(d.decision),
                    "provider_request_hash": d.provider_request_hash,
                    "agent_request_semantic_hash": d.agent_request_semantic_hash,
                    "listed_at_snapshot": d.listed_at_snapshot,
                    "listed_at_apply": listed_at_apply,
                    "target_state_before_apply": target_state(before, d.decision["target_id"]),
                    "other_actor_busy_at_apply": before["actors"][ACTOR_B if actor_id == ACTOR_A else ACTOR_A]["action_cooldown"] > 0,
                    "apply_outcome": outcome,
                    "apply_event_types": types,
                    "time_cost": _cost(cooldown_before, state.actors[actor_id].action_cooldown),
                    "busy_after_apply": state.actors[actor_id].action_cooldown > 0,
                    "stunned_after_apply": state.actors[actor_id].stunned_timer > 0,
                    # this action's own effect, read right after it applied
                    # (MOVE is re-read at the end: its effect starts on the
                    # first live tick)
                    "target_state_after": own_effect_view(after, actor_id, d.decision),
                    "self_state_after": actor_self_state(after, actor_id),
                }
                if not committed:
                    world_hash_after_first_apply = json_hash(after)
                committed.add(actor_id)
            steps.append({"tick": tick, "dt": dt, "actor_id": actor_id, "committed_decision": pending[actor_id],
                          "action": actions[actor_id].type.value, "world_hash_after": json_hash(after),
                          "target_states_after": {a: target_state(after, decided[a].decision["target_id"]) for a in ACTORS},
                          "cooldowns_after": {a: after["actors"][a]["action_cooldown"] for a in ACTORS}})
        state.update(dt)

    final = world_state_snapshot(state)
    for actor_id, rec in records.items():
        d = decided[actor_id]
        if d.decision["action"] == "move":
            rec["target_state_after"] = own_effect_view(final, actor_id, d.decision)
            rec["self_state_after"] = actor_self_state(final, actor_id)
        rec["target_state_final"] = target_state(final, d.decision["target_id"])
        rec["effective_outcome"] = (APPLY_APPLIED if rec["apply_outcome"] == APPLY_COMMITTED_FROZEN else rec["apply_outcome"])
        rec["legal_at_apply"] = rec["effective_outcome"] in (APPLY_APPLIED, APPLY_INTENTIONAL_WAIT)
        cf = fresh_counterfactual(scenario, actor_id, d.raw)
        rec["counterfactual"] = cf
        rec["legal_at_snapshot"] = cf["engine_valid"]
        rec["world_changed"] = rec["apply_world_hash"] != snapshot_hash
        rec["relevant_state_changed"] = rec["apply_relevant_hash"] != rec["snapshot_relevant_hash"]
        rec["relevant_state_stale"] = rec["relevant_state_changed"]
        rec["stale_distance"] = 1 if rec["relevant_state_changed"] else 0
        rec["staleness_class"] = classify_staleness(world_changed=rec["world_changed"], relevant_changed=rec["relevant_state_changed"],
                                                    engine_valid_at_snapshot=cf["engine_valid"], legal_at_apply=rec["legal_at_apply"])
        rec["consequence"] = classify_consequence(rec, cf)
    events = state.event_log[events_start:]
    return {
        "pair_id": pair_id,
        "scenario_family": scenario.family,
        "condition": condition,
        "arrival_order": list(arrival_order) if arrival_order else None,
        "apply_order": apply_sequence,
        "snapshot_world_hash": snapshot_hash,
        # spec section 25: before / after the first apply / after the pair
        "world_hash_before": snapshot_hash,
        "world_hash_after_first_apply": world_hash_after_first_apply,
        "world_hash_after_pair": json_hash(final),
        "world_state_hash_after": json_hash(final),
        "effect_state_hash_after": effect_state_hash(final),
        "event_log_hash": event_log_hash(state, events_start),
        "event_multiset_hash": json_hash(sorted(json.dumps(e, sort_keys=True, default=str) for e in events)),
        "events": events,
        "contested_effects": [(e["type"], e.get("actor_id")) for e in events if e["type"] in {"DOOR_UNLOCKED", "TRAP_PLACED", "TRAP_REMOVED", "TRAP_DISARMED", "TRAP_TRIGGERED"}],
        "final_game_clock": game_clock(state),
        "records": records,
        "steps": steps,
    }


def world_outcome(row: dict[str, Any]) -> tuple[str, str]:
    return row["world_state_hash_after"], row["event_multiset_hash"]


# -- busy-period / completion variants (spec section 24) ----------------------------


def run_busy_variant(scenario: ConflictScenario, first: str, *, second_waits_for_completion: bool, max_ticks: int = 2000) -> dict[str, Any]:
    """First actor applies; the second applies either while the first is
    still inside its busy period (same tick) or only after that period has
    fully elapsed on live ticks. Identical second-actor outcomes mean the
    first action carries no in-flight state: its effect was complete at
    apply time."""
    second = ACTOR_B if first == ACTOR_A else ACTOR_A
    raws = {ACTOR_A: scenario.raw_a, ACTOR_B: scenario.raw_b}
    state = fresh_scenario_state(scenario.setup_fn)
    decided = {a: _decide_from_raw(state, a, raws[a], family=scenario.family, model_id="busy", provider_name="bedrock") for a in ACTORS}
    trace: list[dict[str, Any]] = []

    def apply(actor_id: str) -> dict[str, Any]:
        before = world_state_snapshot(state)
        events_before = len(state.event_log)
        cd_before = state.actors[actor_id].action_cooldown
        state.apply_action(actor_id, decided[actor_id].first_action, DECISION_TICK_DT)
        after = world_state_snapshot(state)
        types = [e["type"] for e in state.event_log[events_before:] if e.get("actor_id") == actor_id]
        rec = {"actor_id": actor_id, "outcome": classify_apply_outcome(decided[actor_id].decision["action"], json_hash(before) != json_hash(after), types),
               "event_types": types, "time_cost": _cost(cd_before, state.actors[actor_id].action_cooldown),
               "target_state_before": target_state(before, decided[actor_id].decision["target_id"]),
               "target_state_after": target_state(after, decided[actor_id].decision["target_id"]),
               "first_actor_busy_at_apply": before["actors"][first]["action_cooldown"] > 0,
               "game_clock": game_clock(state)}
        trace.append(rec)
        return rec

    first_rec = apply(first)
    effect_after_first = effect_state_hash(world_state_snapshot(state))
    busy_ticks = 0
    completion_events: list[str] = []
    if second_waits_for_completion:
        while (state.actors[first].action_cooldown > 0 or state.actors[first].stunned_timer > 0) and busy_ticks < max_ticks:
            events_before = len(state.event_log)
            state.update(DECISION_TICK_DT)
            completion_events += [e["type"] for e in state.event_log[events_before:]]
            busy_ticks += 1
    effect_at_completion = effect_state_hash(world_state_snapshot(state))
    second_rec = apply(second)
    return {
        "scenario_family": scenario.family,
        "first_actor": first,
        "second_actor": second,
        "second_waits_for_completion": second_waits_for_completion,
        "first": first_rec,
        "second": second_rec,
        "first_busy_ticks_until_complete": busy_ticks,
        "events_during_first_busy_period": completion_events,
        "first_effect_unchanged_through_completion": effect_after_first == effect_at_completion,
        "effect_state_hash_after_pair": effect_state_hash(world_state_snapshot(state)),
    }


# -- Phase A probes (spec sections 6-9, 22, 28-29) ------------------------------------


def probe_action(label: str, setup_fn: Callable[[GameState], None] | None, actor_id: str, raw: dict[str, Any],
                 *, communication_enabled: bool = False, max_ticks: int = 3000) -> dict[str, Any]:
    """Measures, on the real Engine, when an action changes the world and
    what its duration does: effect at apply, busy period length, whether
    frozen time drains it, whether anything happens at completion, and
    whether the actor can act while busy."""
    from src.evaluation.runtime_snapshot_replay import _fresh_state

    state = _fresh_state(communication_enabled=communication_enabled)
    if setup_fn is not None:
        setup_fn(state)
    state.update(0.0)
    d = _decide_from_raw(state, actor_id, raw, family=label, model_id="probe", provider_name="bedrock")
    before = world_state_snapshot(state)
    events_before = len(state.event_log)
    state.apply_action(actor_id, d.first_action, DECISION_TICK_DT)
    after = world_state_snapshot(state)
    types = [e["type"] for e in state.event_log[events_before:] if e.get("actor_id") == actor_id]
    actor = state.actors[actor_id]
    cost = actor.action_cooldown
    stun = actor.stunned_timer
    effect_at_apply = effect_state_hash(before) != effect_state_hash(after)
    observation = build_observation(state, actor_id)
    observation_busy = observation.busy or observation.stunned

    # frozen time: 30 zero-dt updates must not shorten the busy period
    for _ in range(30):
        state.update(0.0)
    frozen_cooldown = actor.action_cooldown
    frozen_stun = actor.stunned_timer

    # a second action while busy/stunned is dropped silently (a one-tick
    # MOVE is the most visible probe: any displacement would show)
    locked_out = cost > 0 or stun > 0
    probe_before = world_state_hash(state)
    state.apply_action(actor_id, GameAction(ActionType.MOVE, direction=(1.0, 0.0)), DECISION_TICK_DT)
    dropped_while_busy = locked_out and world_state_hash(state) == probe_before

    # run the duration out on live ticks; watch for completion-time effects
    effect_before_completion = effect_state_hash(world_state_snapshot(state))
    ticks = 0
    events_during: list[str] = []
    move_ticks = 0
    positions_changed_per_tick = []
    if d.decision["action"] == "move":
        action = d.first_action
        while d.executor.in_progress and move_ticks < max_ticks:
            x0, y0 = actor.x, actor.y
            action, _ = d.executor.continue_(build_observation(state, actor_id))
            state.apply_action(actor_id, action, DECISION_TICK_DT)
            state.update(DECISION_TICK_DT)
            if action.type is ActionType.MOVE:
                positions_changed_per_tick.append((actor.x, actor.y) != (x0, y0))
            move_ticks += 1
    while (actor.action_cooldown > 0 or actor.stunned_timer > 0) and ticks < max_ticks:
        n = len(state.event_log)
        state.update(DECISION_TICK_DT)
        events_during += [e["type"] for e in state.event_log[n:]]
        ticks += 1
    effect_after_completion = effect_state_hash(world_state_snapshot(state))
    return {
        "label": label,
        "decision": d.decision,
        "apply_outcome": classify_apply_outcome(d.decision["action"], json_hash(before) != json_hash(after), types),
        "event_types_at_apply": types,
        "world_effect_at_apply": effect_at_apply,
        "time_cost_seconds": round(cost, 6),
        "stun_seconds": round(stun, 6),
        "busy_after_apply": locked_out,
        "busy_kind": "action_cooldown" if cost > 0 else ("stunned" if stun > 0 else None),
        "observation_reports_busy": observation_busy,
        "busy_unchanged_by_frozen_updates": frozen_cooldown == cost and frozen_stun == stun,
        "second_action_dropped_while_busy": dropped_while_busy,
        "busy_ticks_until_complete": ticks,
        "events_during_busy_and_at_completion": events_during,
        "world_effect_at_completion": effect_after_completion != effect_before_completion and d.decision["action"] != "move",
        "multi_tick_ticks": move_ticks,
        "moved_on_every_live_tick": all(positions_changed_per_tick) if positions_changed_per_tick else None,
        "path_waypoints": len(d.executor.last_finished.path) if d.executor.last_finished is not None and d.decision["action"] == "move" else None,
    }


def classify_temporal(probe: dict[str, Any]) -> str:
    """Spec section 8, from measurements only:

    MOVE spanning several live ticks          -> CONTINUOUS_OR_MULTI_TICK
    world changes when the duration ends      -> EFFECT_AT_COMPLETION
    world changes at apply, then a lockout    -> EFFECT_AT_START_WITH_BUSY_PERIOD
    a lockout with no world change at all     -> DURATION_AS_COST_ONLY
    no lockout                                -> INSTANTANEOUS
    """
    if probe["decision"]["action"] == "move":
        return CONTINUOUS_OR_MULTI_TICK if probe["multi_tick_ticks"] > 1 else INSTANTANEOUS
    if probe["world_effect_at_completion"]:
        return EFFECT_AT_COMPLETION
    if probe["busy_after_apply"]:
        return EFFECT_AT_START_WITH_BUSY_PERIOD if probe["world_effect_at_apply"] else DURATION_AS_COST_ONLY
    if probe["time_cost_seconds"] == 0 and probe["stun_seconds"] == 0:
        return INSTANTANEOUS
    return UNKNOWN
