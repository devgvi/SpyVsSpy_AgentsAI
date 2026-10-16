"""M6G spec sections 3-13, 18-19: decision scheduling as an explicit,
testable harness variable instead of implicit loop behavior (Q6).

The three axes are kept independent (spec section 3):

    A. snapshot timing   - FROZEN (both actors observe one pre-action world)
                           or SEQUENTIAL (the second actor observes the world
                           after the first actor's action was applied)
    B. model-call order  - which actor's decider is invoked first
    C. application order - which actor's decision reaches apply_action first

`SchedulePlan` declares all three; `run_condition()` executes a plan and
records every step it actually took (`SchedulingResult.trace`) so a test
can compare the executed schedule with the declared one.

Nothing here touches `Game._step` (spec section 45). Apply order is varied
only by calling `GameState.apply_action()` directly, in plan order.

Reused unchanged, never re-implemented:
- `agent_contract.build_agent_request()` (M6F canonical AgentRequest) and
  `native_model_payload()`/`native_system_prompt()` for rendering;
- `llm_controller.validate_model_output()` (the production strict gate);
- `llm_controller.enumerate_legal_actions()` for legality at snapshot and
  at apply time.

Harness choices, fixed for every condition:
- Game time is frozen during inference (spec section 11): nothing here
  advances either clock, the only `update()` calls use dt=0, and every
  action is applied with the same fixed `DECISION_TICK_DT`.
- No repair (spec section 36): an invalid first-pass output falls back to
  WAIT, and that fallback is recorded, never hidden.
- MOVE is applied as one tick toward its target. Multi-tick action
  semantics are M6G.1's subject (spec section 48).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.agent_contract import AgentRequest, build_agent_request, native_model_payload, native_system_prompt
from src.evaluation.conflict_scenarios import ConflictScenario
from src.evaluation.duel_snapshot import (
    ACTORS,
    event_log_hash,
    fresh_scenario_state,
    game_clock,
    json_hash,
    legal_actions_hash,
    memory_hash,
    observation_hash,
    personality_hash,
    snapshot_id,
    world_state_hash,
)
from src.evaluation.runtime_variance_analysis import content_hash
from src.llm.agent_memory import AgentMemory
from src.llm.personality import DEFAULT
from src.llm_controller import enumerate_legal_actions, validate_model_output
from src.observation import Observation, build_observation
from src.state import GameState

DECISION_TICK_DT = 1.0 / constants.FPS

S0_FROZEN_SNAPSHOT_BASELINE = "S0_FROZEN_SNAPSHOT_BASELINE"
S1_REVERSED_CALL = "S1_FROZEN_SNAPSHOT_REVERSED_CALL"
S2_REVERSED_APPLY = "S2_FROZEN_SNAPSHOT_REVERSED_APPLY"
S3_SEQUENTIAL_A_FIRST = "S3_FULLY_SEQUENTIAL_A_FIRST"
S4_SEQUENTIAL_B_FIRST = "S4_FULLY_SEQUENTIAL_B_FIRST"

SCHEDULING_CONDITIONS = (S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL, S2_REVERSED_APPLY, S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST)

SNAPSHOT_FROZEN = "FROZEN_SHARED_SNAPSHOT"
SNAPSHOT_SEQUENTIAL = "SEQUENTIAL_REBUILT_AFTER_PRIOR_APPLY"

APPLY_APPLIED = "APPLIED"
APPLY_INTENTIONAL_WAIT = "INTENTIONAL_WAIT"
APPLY_REFUSED_NO_OP = "REFUSED_NO_OP"
APPLY_FAILURE_EVENT_ONLY = "FAILURE_EVENT_ONLY"
APPLY_ALTERNATE_EFFECT = "ALTERNATE_EFFECT"
APPLY_OUTCOMES_VALID_AT_APPLY = frozenset({APPLY_APPLIED, APPLY_INTENTIONAL_WAIT})
# `_trigger_trap` replaces the intended search with a stun (state.py
# 544-614): the action did something, just not what was decided.
ALTERNATE_EFFECT_EVENT_TYPES = frozenset({"TRAP_TRIGGERED", "ACTOR_STUNNED"})

# Provider-side failures abort a live run instead of being scored as a
# model decision; malformed or tool-less model output is a first-pass
# validity failure instead (it falls back to WAIT like production does).
MODEL_OUTPUT_ERROR_TYPES = frozenset({"provider_invalid_response", "no_tool_selected"})


@dataclass(frozen=True)
class SchedulePlan:
    condition: str
    snapshot_timing: str
    call_order: tuple[str, str]
    apply_order: tuple[str, str]

    def __post_init__(self) -> None:
        if sorted(self.call_order) != sorted(ACTORS) or sorted(self.apply_order) != sorted(ACTORS):
            raise ValueError(f"{self.condition}: call/apply order must name each actor exactly once")
        if self.snapshot_timing == SNAPSHOT_SEQUENTIAL and self.call_order != self.apply_order:
            raise ValueError(f"{self.condition}: a sequential schedule applies each decision right after making it")
        if self.snapshot_timing not in (SNAPSHOT_FROZEN, SNAPSHOT_SEQUENTIAL):
            raise ValueError(f"{self.condition}: unknown snapshot timing {self.snapshot_timing!r}")

    def declared_steps(self) -> list[tuple[str, str]]:
        if self.snapshot_timing == SNAPSHOT_FROZEN:
            return (
                [("snapshot", a) for a in ACTORS]
                + [("request", a) for a in ACTORS]
                + [("decide", a) for a in self.call_order]
                + [("apply", a) for a in self.apply_order]
            )
        steps: list[tuple[str, str]] = []
        for a in self.call_order:
            steps += [("snapshot", a), ("request", a), ("decide", a), ("apply", a)]
        return steps

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "snapshot_timing": self.snapshot_timing,
            "call_order": list(self.call_order),
            "apply_order": list(self.apply_order),
            "declared_steps": [list(s) for s in self.declared_steps()],
        }


_A, _B = ACTORS
SCHEDULE_PLANS: dict[str, SchedulePlan] = {
    S0_FROZEN_SNAPSHOT_BASELINE: SchedulePlan(S0_FROZEN_SNAPSHOT_BASELINE, SNAPSHOT_FROZEN, (_A, _B), (_A, _B)),
    S1_REVERSED_CALL: SchedulePlan(S1_REVERSED_CALL, SNAPSHOT_FROZEN, (_B, _A), (_A, _B)),
    S2_REVERSED_APPLY: SchedulePlan(S2_REVERSED_APPLY, SNAPSHOT_FROZEN, (_A, _B), (_B, _A)),
    S3_SEQUENTIAL_A_FIRST: SchedulePlan(S3_SEQUENTIAL_A_FIRST, SNAPSHOT_SEQUENTIAL, (_A, _B), (_A, _B)),
    S4_SEQUENTIAL_B_FIRST: SchedulePlan(S4_SEQUENTIAL_B_FIRST, SNAPSHOT_SEQUENTIAL, (_B, _A), (_B, _A)),
}


# -- deciders -----------------------------------------------------------------


class Decider(Protocol):
    """`render` builds the exact model-visible (system, user) text for a
    request without calling anything; `call` performs one decision. Kept
    separate so both actors' requests can be rendered and hashed before
    either call happens (spec section 12)."""

    runtime: str

    def render(self, request: AgentRequest) -> tuple[str, str]: ...

    def call(self, request: AgentRequest, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


def native_render(request: AgentRequest) -> tuple[str, str]:
    """Native's exact provider-facing text: the same system prompt and
    sort_keys JSON every M6-family Native driver has sent since M6D."""
    return native_system_prompt(request), json.dumps(native_model_payload(request), sort_keys=True)


class ScriptedDecider:
    """Network-free decider for Phase B and tests. `policy` is either a
    fixed `{actor_id: raw_output}` map or a callable over the request."""

    runtime = "scripted"

    def __init__(
        self,
        policy: dict[str, dict[str, Any]] | Callable[[AgentRequest], dict[str, Any]],
        delay_seconds: float = 0.0,
        render_fn: Callable[[AgentRequest], tuple[str, str]] = native_render,
    ) -> None:
        self._policy = policy
        self._delay = delay_seconds
        self._render = render_fn
        self.calls: list[str] = []

    def render(self, request: AgentRequest) -> tuple[str, str]:
        return self._render(request)

    def call(self, request: AgentRequest, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self._delay:
            time.sleep(self._delay)
        self.calls.append(request.actor_id)
        raw = self._policy(request) if callable(self._policy) else dict(self._policy[request.actor_id])
        return {"raw": raw, "latency_ms": None, "input_tokens": None, "output_tokens": None, "error_type": None}


class ReplayDecider:
    """S2 with live decisions (spec sections 13, 23): returns decisions an
    earlier frozen-snapshot trial already made, so only the application
    order changes. Refuses to run if the rebuilt request is not the one the
    reused decision was made for."""

    def __init__(
        self,
        source_trial_id: str,
        runtime: str,
        outputs: dict[str, dict[str, Any]],
        expected_request_hashes: dict[str, str],
        render_fn: Callable[[AgentRequest], tuple[str, str]] = native_render,
    ) -> None:
        self.source_trial_id = source_trial_id
        self.runtime = runtime
        self._outputs = outputs
        self._expected = expected_request_hashes
        self._render = render_fn

    def render(self, request: AgentRequest) -> tuple[str, str]:
        return self._render(request)

    def call(self, request: AgentRequest, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        expected = self._expected[request.actor_id]
        if request.request_semantic_hash() != expected:
            raise RuntimeError(f"replay request mismatch for {request.actor_id}: decision was made for a different request")
        out = dict(self._outputs[request.actor_id])
        # No model call happened here - cost belongs to the source trial.
        out.update({"latency_ms": None, "input_tokens": None, "output_tokens": None})
        return out


# -- records ------------------------------------------------------------------


@dataclass
class ActorRecord:
    actor_id: str
    decision_id: str
    snapshot_id: str
    decision_call_index: int
    apply_index: int
    # spec section 7
    world_state_hash_before_snapshot: str
    observation_hash: str
    legal_actions_hash: str
    memory_hash: str
    personality_hash: str
    # spec section 12, three strictness levels: the whole canonical
    # AgentRequest, M6F's semantic hash, and the exact provider-facing text
    agent_request_hash: str
    agent_request_semantic_hash: str
    provider_request_hash: str
    provider_request_characters: int
    # spec section 11
    game_clock_before_call: dict[str, float]
    game_clock_after_call: dict[str, float]
    world_unchanged_during_call: bool
    # decision (visible fields only - spec section 51)
    runtime: str
    raw_model_output: dict[str, Any]
    visible_rationale: str | None
    first_pass_valid: bool
    validation_error: str | None
    fallback_to_wait: bool
    decision: dict[str, Any]
    normalized_action: dict[str, Any]
    # spec section 18
    legal_at_snapshot: bool
    listed_legal_at_apply_time: bool
    legal_at_apply_time: bool
    apply_outcome: str
    apply_event_types: list[str]
    world_state_hash_before_apply: str
    world_state_hash_after_apply: str
    # cost (spec section 40 - reported separately from every conclusion)
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    provider_error_type: str | None
    provider_error_message: str | None = None
    decision_reused_from: str | None = None
    # the request this actor would have received at the pre-action snapshot
    # (built for measurement only, never sent); differs from
    # provider_request_hash only when a sequential schedule showed it the
    # first actor's effect
    pre_action_provider_request_hash: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulingResult:
    trial_id: str
    condition: str
    plan: dict[str, Any]
    scenario_family: str
    runtime: str
    trace: list[list[str]]
    records: dict[str, ActorRecord]
    same_snapshot: bool
    first_actor: str
    second_actor: str
    world_changed_before_second_snapshot: bool
    second_actor_saw_first_action: bool
    world_state_hash_pre_action: str
    world_state_hash_after: str
    event_log_hash: str
    events: list[dict[str, Any]]
    decisions_reused_from: str | None = None

    def executed_steps(self) -> list[tuple[str, str]]:
        return [(step, actor) for step, actor in self.trace]

    def to_row(self, **extra: Any) -> dict[str, Any]:
        row = {
            "trial_id": self.trial_id,
            "condition": self.condition,
            "plan": self.plan,
            "scenario_family": self.scenario_family,
            "runtime": self.runtime,
            "trace": self.trace,
            "same_snapshot": self.same_snapshot,
            "first_actor": self.first_actor,
            "second_actor": self.second_actor,
            "world_changed_before_second_snapshot": self.world_changed_before_second_snapshot,
            "second_actor_saw_first_action": self.second_actor_saw_first_action,
            "world_state_hash_pre_action": self.world_state_hash_pre_action,
            "world_state_hash_after": self.world_state_hash_after,
            "event_log_hash": self.event_log_hash,
            "events": self.events,
            "decisions_reused_from": self.decisions_reused_from,
            "records": {actor_id: asdict(rec) for actor_id, rec in self.records.items()},
        }
        row.update(extra)
        return row


# -- steps --------------------------------------------------------------------


@dataclass
class _Prepared:
    actor_id: str
    world_hash: str
    snapshot_id: str
    observation: Observation
    legal_actions: list[dict[str, Any]]
    request: AgentRequest | None = None
    system_prompt: str = ""
    user_prompt: str = ""


def build_request(
    actor_id: str,
    observation: Observation,
    legal_actions: list[dict[str, Any]],
    *,
    scenario_family: str,
    model_id: str,
    provider_name: str,
    personality_profile: str | None = DEFAULT,
) -> AgentRequest:
    """One canonical request per (actor, snapshot). episode_id and metadata
    carry nothing condition-specific, so the same actor at the same
    snapshot gets an identical AgentRequest object in every condition -
    not only an identical semantic hash. Memory is a fresh, empty
    within-episode AgentMemory (spec section 33); no cross-episode social
    memory transform exists on this path at all."""
    return build_agent_request(
        actor_id=actor_id,
        episode_id=f"m6g:{scenario_family}",
        observation=observation,
        legal_actions=legal_actions,
        memory_payload=AgentMemory(actor_id).to_compact_dict(),
        personality_profile=personality_profile,
        model_id=model_id,
        provider_name=provider_name,
        metadata={},
    )


def _snapshot(state: GameState, actor_id: str, scenario_family: str, trace: list[list[str]]) -> _Prepared:
    wsh = world_state_hash(state)
    observation = build_observation(state, actor_id)
    legal = enumerate_legal_actions(observation)
    trace.append(["snapshot", actor_id])
    return _Prepared(actor_id, wsh, snapshot_id(scenario_family, wsh), observation, legal)


def _prepare_request(prepared: _Prepared, decider: Decider, ctx: dict[str, Any], trace: list[list[str]]) -> None:
    prepared.request = build_request(
        prepared.actor_id, prepared.observation, prepared.legal_actions,
        scenario_family=ctx["scenario_family"], model_id=ctx["model_id"], provider_name=ctx["provider_name"],
        personality_profile=ctx["personality_profile"],
    )
    prepared.system_prompt, prepared.user_prompt = decider.render(prepared.request)
    trace.append(["request", prepared.actor_id])


def _decision_key(validated: dict[str, Any] | None) -> dict[str, Any]:
    if not validated:
        return {"action": "wait", "target_id": None, "trap_type": None}
    action = validated.get("action")
    return {
        "action": action,
        "target_id": validated.get("target_id") if action != "wait" else None,
        "trap_type": validated.get("trap_type") if action == "place_trap" else None,
    }


def _decide(state: GameState, prepared: _Prepared, decider: Decider, trace: list[list[str]]) -> dict[str, Any]:
    clock_before = game_clock(state)
    world_before = world_state_hash(state)
    output = decider.call(prepared.request, prepared.system_prompt, prepared.user_prompt)
    clock_after = game_clock(state)
    world_after = world_state_hash(state)
    trace.append(["decide", prepared.actor_id])

    raw = output.get("raw") if isinstance(output.get("raw"), dict) else {}
    validation = validate_model_output(prepared.observation, raw, prepared.legal_actions)
    if validation.valid and validation.action is not None:
        executed = validation.action
        decision = _decision_key(validation.validated_action)
    else:
        executed = GameAction(ActionType.WAIT)
        decision = _decision_key(None)
    return {
        "output": output,
        "raw": raw,
        "validation": validation,
        "executed": executed,
        "decision": decision,
        "clock_before": clock_before,
        "clock_after": clock_after,
        "world_unchanged_during_call": world_before == world_after,
    }


def _is_listed(decision: dict[str, Any], legal_actions: list[dict[str, Any]]) -> bool:
    if decision["action"] == "wait":
        return any(e.get("action") == "wait" for e in legal_actions)
    return any(e.get("action") == decision["action"] and e.get("target_id") == decision["target_id"] for e in legal_actions)


def classify_apply_outcome(action: str, world_changed: bool, actor_event_types: list[str]) -> str:
    """Spec section 19, measured per apply: what the Engine actually did.
    Every handler re-checks its own preconditions against the live state
    and returns early on failure without logging or charging anything
    (state.py 644-731), so "nothing changed, nothing logged" is a silent
    refusal. `_attempt_exit` is the one handler that logs a failure
    (EXIT_ATTEMPTED) - FAILURE_EVENT_ONLY exists for it."""
    if action == "wait":
        return APPLY_INTENTIONAL_WAIT
    if ALTERNATE_EFFECT_EVENT_TYPES & set(actor_event_types):
        return APPLY_ALTERNATE_EFFECT
    if world_changed:
        return APPLY_APPLIED
    if actor_event_types:
        return APPLY_FAILURE_EVENT_ONLY
    return APPLY_REFUSED_NO_OP


def _apply(state: GameState, prepared: _Prepared, decided: dict[str, Any], trace: list[list[str]]) -> dict[str, Any]:
    actor_id = prepared.actor_id
    # Agent-visible legality at apply time: what a fresh observation of the
    # live world would list right now. Built for measurement only.
    live_legal = enumerate_legal_actions(build_observation(state, actor_id))
    listed = _is_listed(decided["decision"], live_legal)
    world_before = world_state_hash(state)
    events_before = len(state.event_log)
    state.apply_action(actor_id, decided["executed"], DECISION_TICK_DT)
    new_events = state.event_log[events_before:]
    world_after = world_state_hash(state)
    trace.append(["apply", actor_id])
    actor_event_types = [e["type"] for e in new_events if e.get("actor_id") == actor_id]
    outcome = classify_apply_outcome(decided["decision"]["action"], world_before != world_after, actor_event_types)
    return {
        "listed_legal_at_apply_time": listed,
        "legal_at_apply_time": outcome in APPLY_OUTCOMES_VALID_AT_APPLY,
        "apply_outcome": outcome,
        "apply_event_types": actor_event_types,
        "world_before": world_before,
        "world_after": world_after,
    }


def _normalized_action_dict(action: GameAction) -> dict[str, Any]:
    return {
        "type": action.type.value,
        "target_id": action.target_id,
        "trap_type": action.trap_type,
        "direction": [round(v, 6) for v in action.direction] if action.direction is not None else None,
    }


def _record(
    prepared: _Prepared,
    decided: dict[str, Any],
    applied: dict[str, Any],
    *,
    trial_id: str,
    call_index: int,
    apply_index: int,
    runtime: str,
    reused_from: str | None,
    pre_action_provider_hash: str,
) -> ActorRecord:
    request = prepared.request
    output = decided["output"]
    validation = decided["validation"]
    raw = decided["raw"]
    return ActorRecord(
        actor_id=prepared.actor_id,
        decision_id=f"{trial_id}:{prepared.actor_id}",
        snapshot_id=prepared.snapshot_id,
        decision_call_index=call_index,
        apply_index=apply_index,
        world_state_hash_before_snapshot=prepared.world_hash,
        observation_hash=observation_hash(prepared.observation),
        legal_actions_hash=legal_actions_hash(prepared.legal_actions),
        memory_hash=memory_hash(request.memory_payload),
        personality_hash=personality_hash(request.personality_policy),
        agent_request_hash=json_hash(request.as_dict()),
        agent_request_semantic_hash=request.request_semantic_hash(),
        provider_request_hash=content_hash(prepared.system_prompt, prepared.user_prompt),
        provider_request_characters=len(prepared.system_prompt) + len(prepared.user_prompt),
        game_clock_before_call=decided["clock_before"],
        game_clock_after_call=decided["clock_after"],
        world_unchanged_during_call=decided["world_unchanged_during_call"],
        runtime=runtime,
        raw_model_output=raw,
        visible_rationale=raw.get("reason") if isinstance(raw.get("reason"), str) else None,
        first_pass_valid=bool(validation.valid),
        validation_error=validation.error_type,
        fallback_to_wait=not validation.valid,
        decision=decided["decision"],
        normalized_action=_normalized_action_dict(decided["executed"]),
        legal_at_snapshot=_is_listed(decided["decision"], prepared.legal_actions),
        listed_legal_at_apply_time=applied["listed_legal_at_apply_time"],
        legal_at_apply_time=applied["legal_at_apply_time"],
        apply_outcome=applied["apply_outcome"],
        apply_event_types=applied["apply_event_types"],
        world_state_hash_before_apply=applied["world_before"],
        world_state_hash_after_apply=applied["world_after"],
        latency_ms=output.get("latency_ms"),
        input_tokens=output.get("input_tokens"),
        output_tokens=output.get("output_tokens"),
        provider_error_type=output.get("error_type"),
        provider_error_message=output.get("error_message"),
        decision_reused_from=reused_from,
        pre_action_provider_request_hash=pre_action_provider_hash,
        extra={k: output[k] for k in ("tool_call_count", "proposal_count", "stop_reason") if k in output},
    )


# -- conditions ----------------------------------------------------------------


def run_condition(
    condition: str | SchedulePlan,
    scenario: ConflictScenario,
    decider: Decider,
    *,
    trial_id: str,
    model_id: str,
    provider_name: str,
    personality_profile: str | None = DEFAULT,
) -> SchedulingResult:
    plan = condition if isinstance(condition, SchedulePlan) else SCHEDULE_PLANS[condition]
    ctx = {
        "scenario_family": scenario.family,
        "model_id": model_id,
        "provider_name": provider_name,
        "personality_profile": personality_profile,
    }
    reused_from = getattr(decider, "source_trial_id", None)
    trace: list[list[str]] = []
    state = fresh_scenario_state(scenario.setup_fn)
    pre_action_hash = world_state_hash(state)
    events_start = len(state.event_log)

    prepared: dict[str, _Prepared] = {}
    decided: dict[str, dict[str, Any]] = {}
    applied: dict[str, dict[str, Any]] = {}
    call_index: dict[str, int] = {}
    apply_index: dict[str, int] = {}
    pre_action_provider_hash: dict[str, str] = {}

    if plan.snapshot_timing == SNAPSHOT_FROZEN:
        for actor_id in ACTORS:
            prepared[actor_id] = _snapshot(state, actor_id, scenario.family, trace)
        # Both requests exist before either call (spec section 12).
        for actor_id in ACTORS:
            _prepare_request(prepared[actor_id], decider, ctx, trace)
            pre_action_provider_hash[actor_id] = content_hash(prepared[actor_id].system_prompt, prepared[actor_id].user_prompt)
        # Both decisions exist before either apply (spec section 13).
        for idx, actor_id in enumerate(plan.call_order):
            decided[actor_id] = _decide(state, prepared[actor_id], decider, trace)
            call_index[actor_id] = idx
        for idx, actor_id in enumerate(plan.apply_order):
            applied[actor_id] = _apply(state, prepared[actor_id], decided[actor_id], trace)
            apply_index[actor_id] = idx
    else:
        second = plan.call_order[1]
        # Measurement only: the second actor's request at the pre-action
        # snapshot, rendered and hashed but never sent and not a schedule
        # step. It is the baseline for "did the rebuilt snapshot show the
        # second actor anything new?" (spec sections 9, 24).
        cf = _Prepared(second, pre_action_hash, snapshot_id(scenario.family, pre_action_hash), build_observation(state, second), [])
        cf.legal_actions = enumerate_legal_actions(cf.observation)
        _prepare_request(cf, decider, ctx, [])
        pre_action_provider_hash[second] = content_hash(cf.system_prompt, cf.user_prompt)
        for idx, actor_id in enumerate(plan.call_order):
            if idx == 1:
                # zero-dt update: refreshes interaction targets/visibility
                # from the post-apply world without advancing either clock
                state.update(0.0)
            prepared[actor_id] = _snapshot(state, actor_id, scenario.family, trace)
            _prepare_request(prepared[actor_id], decider, ctx, trace)
            if idx == 0:
                pre_action_provider_hash[actor_id] = content_hash(prepared[actor_id].system_prompt, prepared[actor_id].user_prompt)
            decided[actor_id] = _decide(state, prepared[actor_id], decider, trace)
            call_index[actor_id] = idx
            applied[actor_id] = _apply(state, prepared[actor_id], decided[actor_id], trace)
            apply_index[actor_id] = idx

    records = {
        actor_id: _record(
            prepared[actor_id], decided[actor_id], applied[actor_id], trial_id=trial_id,
            call_index=call_index[actor_id], apply_index=apply_index[actor_id], runtime=decider.runtime,
            reused_from=reused_from, pre_action_provider_hash=pre_action_provider_hash[actor_id],
        )
        for actor_id in ACTORS
    }
    first_actor, second_actor = plan.apply_order
    second_rec = records[second_actor]
    return SchedulingResult(
        trial_id=trial_id,
        condition=plan.condition,
        plan=plan.as_dict(),
        scenario_family=scenario.family,
        runtime=decider.runtime,
        trace=trace,
        records=records,
        same_snapshot=records[first_actor].world_state_hash_before_snapshot == records[second_actor].world_state_hash_before_snapshot,
        first_actor=first_actor,
        second_actor=second_actor,
        world_changed_before_second_snapshot=second_rec.world_state_hash_before_snapshot != pre_action_hash,
        second_actor_saw_first_action=second_rec.provider_request_hash != second_rec.pre_action_provider_request_hash,
        world_state_hash_pre_action=pre_action_hash,
        world_state_hash_after=world_state_hash(state),
        event_log_hash=event_log_hash(state, events_start),
        events=list(state.event_log[events_start:]),
        decisions_reused_from=reused_from,
    )


def replay_with_reversed_apply(source: SchedulingResult | dict[str, Any], scenario: ConflictScenario, *, trial_id: str, model_id: str,
                               provider_name: str, render_fn: Callable[[AgentRequest], tuple[str, str]] = native_render,
                               personality_profile: str | None = DEFAULT) -> SchedulingResult:
    """S2 from an S0 trial's own decisions: same pre-action state (rebuilt
    and hash-checked), same decisions, only the application order flips.
    `source` may be the live result or its stored JSONL row."""
    row = source.to_row() if isinstance(source, SchedulingResult) else source
    if row["plan"]["snapshot_timing"] != SNAPSHOT_FROZEN:
        raise ValueError("apply-order replay needs a frozen-snapshot source trial")
    records = row["records"]
    outputs = {actor_id: {"raw": rec["raw_model_output"], "error_type": rec["provider_error_type"]} for actor_id, rec in records.items()}
    decider = ReplayDecider(
        row["trial_id"], row["runtime"], outputs,
        {actor_id: rec["agent_request_semantic_hash"] for actor_id, rec in records.items()},
        render_fn=render_fn,
    )
    result = run_condition(S2_REVERSED_APPLY, scenario, decider, trial_id=trial_id, model_id=model_id, provider_name=provider_name,
                           personality_profile=personality_profile)
    if result.world_state_hash_pre_action != row["world_state_hash_pre_action"]:
        raise RuntimeError("apply-order replay rebuilt a different pre-action world")
    for actor_id, rec in result.records.items():
        if rec.decision != records[actor_id]["decision"] or rec.provider_request_hash != records[actor_id]["provider_request_hash"]:
            raise RuntimeError(f"apply-order replay changed {actor_id}'s request or decision")
    return result
