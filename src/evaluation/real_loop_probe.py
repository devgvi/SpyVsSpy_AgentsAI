"""M6G.1 spec sections 12, 17-19, 30: the production game loop, observed
from outside.

`run_real_loop_trial` builds a real `Game` with two LLM-controlled actors
(Native `native_tool_like` or Strands), places them in a temporal scenario
and steps the real `Game._step` until both actors' first decisions have
been committed. Nothing in the loop, the controllers or the Engine is
changed. Observation happens through pass-through wrappers installed on
one Game instance:

- the model call (Native: the controller's `model_client`; Strands:
  `_invoke_strands`) - request text hash, wall-clock start/end, payload;
- the controller's background request (`_request_action_background`) -
  when the decision became READY;
- each actor's `get_action` - the snapshot world when a request starts and
  the tick a ready decision is COMMITTED;
- `GameState.apply_action` - the world immediately before/after every
  APPLY.

Each wrapper calls the original and returns its result unchanged. The
loop's own threads are the only concurrency; no race is added
(spec section 45). Stepping stops as soon as both first decisions are
committed, so no second-round model call is made.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.conflict_scenarios import ConflictScenario
from src.evaluation.decision_scheduling import DECISION_TICK_DT, _decision_key, _is_listed, classify_apply_outcome
from src.evaluation.duel_snapshot import (
    ACTOR_A,
    ACTOR_B,
    ACTORS,
    event_log_hash,
    fresh_scenario_state,
    game_clock,
    json_hash,
    snapshot_id,
    world_state_snapshot,
)
from src.evaluation.runtime_variance_analysis import content_hash
from src.evaluation.temporal_semantics import (
    APPLY_APPLIED,
    APPLY_INTENTIONAL_WAIT,
    actor_self_state,
    classify_consequence,
    classify_staleness,
    effect_state_hash,
    fresh_counterfactual,
    own_effect_view,
    relevant_state_hash,
    target_state,
)
from src.llm_controller import enumerate_legal_actions, validate_model_output
from src.observation import build_observation

SEED = 6001  # the same world `runtime_snapshot_replay._fresh_state` builds


def _action_dict(action: GameAction) -> dict[str, Any]:
    return {
        "type": action.type.value,
        "target_id": action.target_id,
        "trap_type": action.trap_type,
        "direction": [round(v, 6) for v in action.direction] if action.direction is not None else None,
    }


class _TimingClient:
    """Pass-through around a Native controller's model client."""

    def __init__(self, inner: Any, actor_id: str, log: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._actor_id = actor_id
        self._log = log

    def generate_action(self, system_prompt, user_prompt, output_schema, *args, **kwargs):
        start_wall, start = time.time(), time.monotonic()
        response = self._inner.generate_action(system_prompt, user_prompt, output_schema, *args, **kwargs)
        end = time.monotonic()
        self._log.append({
            "actor_id": self._actor_id, "call_start_wall": start_wall, "call_end_wall": time.time(),
            "call_latency_ms": round((end - start) * 1000.0, 2), "provider_request_hash": content_hash(system_prompt, user_prompt),
            "payload": response.payload if isinstance(response.payload, dict) else {}, "error_type": response.error_type,
            "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        })
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _wrap_strands_invoke(adapter: Any, actor_id: str, log: list[dict[str, Any]]) -> None:
    original = adapter._invoke_strands

    def wrapped(model_payload, repair_instruction=None):
        start_wall, start = time.time(), time.monotonic()
        invocation = original(model_payload, repair_instruction)
        end = time.monotonic()
        prompt = dict(model_payload)
        if repair_instruction:
            from src.llm_controller import REPAIR_PROMPT_VERSION

            prompt["repair_instruction"] = repair_instruction
            prompt["repair_prompt_version"] = REPAIR_PROMPT_VERSION
        log.append({
            "actor_id": actor_id, "call_start_wall": start_wall, "call_end_wall": time.time(),
            "call_latency_ms": round((end - start) * 1000.0, 2),
            "provider_request_hash": content_hash(adapter._effective_system_prompt, json.dumps(prompt, sort_keys=True)),
            "payload": invocation.proposal if isinstance(invocation.proposal, dict) else {}, "error_type": invocation.error_type,
            "input_tokens": invocation.input_tokens, "output_tokens": invocation.output_tokens,
        })
        return invocation

    adapter._invoke_strands = wrapped


def _wrap_request(owner: Any, actor_id: str, log: list[dict[str, Any]]) -> None:
    original = owner._request_action_background

    def wrapped(observation, observation_payload):
        entry = {"actor_id": actor_id, "request_start_wall": time.time(), "_start": time.monotonic()}
        try:
            original(observation, observation_payload)
        finally:
            entry["ready_wall"] = time.time()
            entry["request_to_ready_ms"] = round((time.monotonic() - entry.pop("_start")) * 1000.0, 2)
            log.append(entry)

    owner._request_action_background = wrapped


def _controller(adapter: Any) -> Any:
    return adapter.__dict__.get("controller", adapter)


GATE_TIMEOUT_SECONDS = 30.0


def _wait(delay_seconds: float, gate: threading.Event | None) -> None:
    """Answer no earlier than `delay_seconds` after the call started and,
    with a gate, not before the gate opens."""
    start = time.monotonic()
    if gate is not None:
        gate.wait(timeout=GATE_TIMEOUT_SECONDS)
    remaining = delay_seconds - (time.monotonic() - start)
    if remaining > 0:
        time.sleep(remaining)


def scripted_native_client(raw: dict[str, Any], delay_seconds: float, gate: threading.Event | None = None):
    """A Native model client that answers `raw` after `delay_seconds` -
    zero network. The delay only sets which answer is ready first; a gate
    (see `run_real_loop_trial(gates=...)`) makes that order independent of
    machine load."""
    from src.llm.clients.mock_client import MockModelClient

    class _Delayed(MockModelClient):
        def generate_action(self, *args, **kwargs):
            _wait(delay_seconds, gate)
            return super().generate_action(*args, **kwargs)

    return _Delayed(response=[dict(raw)])


def scripted_strands_invoke(raw: dict[str, Any], delay_seconds: float, gate: threading.Event | None = None):
    """Stands in for one Strands SDK invocation that called the
    choose_action tool once with `raw` - zero network. Everything around
    it (the adapter's validation, repair, executor, thinking lifecycle) is
    production code."""
    from src.agents.strands_agent import _StrandsInvocation

    def invoke(model_payload, repair_instruction=None):
        _wait(delay_seconds, gate)
        return _StrandsInvocation(proposal=dict(raw), proposal_count=1, stop_reason="tool_use", cycle_count=1, tool_call_count=1,
                                  input_tokens=None, output_tokens=None, latency_ms=delay_seconds * 1000.0, error_type=None,
                                  error_message=None)

    return invoke


def build_game(scenario: ConflictScenario, *, runtime: str, provider: str, model_id: str, region: str | None,
               decision_call_order: str = "black_first", llm_timeout: float = 60.0, commit_policy: str | None = None):
    """The real Game for the M6G.1/M6G.2 real-loop probes. Those probes
    measure the pre-M6G.3 answer-order loop (P0/T0) and layer their own
    evaluation barrier on top of it, so `commit_policy` defaults to the
    legacy "answer_order" here; M6G.3's probe passes the production default."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from src.game import COMMIT_POLICY_ANSWER_ORDER, Game

    game = Game(
        white_controller="llm", black_controller="llm", llm_provider=provider, llm_model=model_id, llm_region=region,
        agent_runtime=runtime, seed=SEED, decision_call_order=decision_call_order, llm_timeout=llm_timeout,
        black_personality="default", white_personality="default", communication_enabled=False,
        commit_policy=commit_policy or COMMIT_POLICY_ANSWER_ORDER,
    )
    scenario.setup_fn(game.state)
    game.state.update(0.0)
    return game


def run_real_loop_trial(
    scenario: ConflictScenario,
    *,
    trial_id: str,
    runtime: str = "native_tool_like",
    provider: str = "bedrock",
    model_id: str,
    region: str | None = None,
    decision_call_order: str = "black_first",
    client_overrides: dict[str, Any] | None = None,
    strands_invoke_overrides: dict[str, Any] | None = None,
    max_wall_seconds: float = 120.0,
    tick_sleep_seconds: float = 0.005,
    force_handoff: tuple[str, ...] = (),
    force_handoff_min_age_s: float = 0.0,
    force_ready_after_opponent_commit: tuple[str, ...] = (),
    gates: dict[str, threading.Event] | None = None,
    release_after_opponent_commit: tuple[str, ...] = (),
) -> dict[str, Any]:
    """One real-loop trial. `client_overrides` ({actor_id: client}) swaps
    in a scripted model client for a Native actor; `strands_invoke_
    overrides` ({actor_id: fn(payload, repair_instruction)}) does the same
    for a Strands actor's single SDK invocation (see `scripted_strands_
    invoke`). Everything else - the adapter's own thinking/digest/executor
    lifecycle, validation, repair, the Engine - is production code. Live
    runs pass neither.

    Two test-only knobs (never used in live runs) pin down interleavings
    the loop otherwise hits by chance. They add no concurrency; they remove
    the nondeterminism:

    - `force_handoff`: once every other actor has committed, a listed
      actor's thinking thread is joined right after its `get_action`
      returned "still thinking", so its decision becomes ready between
      `get_action` and the loop's thinking-flag check (one live tick runs
      before it commits). `force_handoff_min_age_s` delays that until the
      request is at least that old - past the controllers' 0.5 s
      re-request throttle, the in-flight `get_action` has already taken
      the opponent's change into its digest, so the handoff is not mixed
      with the re-request path below.
    - `force_ready_after_opponent_commit`: a listed actor's thinking thread
      is joined right after the opponent's commit was applied, so its
      decision becomes ready after the world changed but before its own
      next `get_action` (the controller then sees a new observation
      digest with no request in flight).

    `gates` ({actor_id: Event}, shared with that actor's scripted client)
    hold a scripted answer back until the probe opens the gate: at a
    forcing knob's moment, or - `release_after_opponent_commit` - right
    after the opponent's commit was applied. This fixes which answer is
    ready first independently of machine load. Every gate is opened when
    the trial ends."""
    game = build_game(scenario, runtime=runtime, provider=provider, model_id=model_id, region=region,
                      decision_call_order=decision_call_order)
    adapters = {ACTOR_A: game.black_llm_controller, ACTOR_B: game.bot_controller}
    pre_world = world_state_snapshot(game.state)
    harness_world = world_state_snapshot(fresh_scenario_state(scenario.setup_fn))
    events_start = len(game.state.event_log)

    call_log: list[dict[str, Any]] = []
    request_log: list[dict[str, Any]] = []
    get_log: list[dict[str, Any]] = []
    apply_log: list[dict[str, Any]] = []
    tick = {"n": 0}
    committed: dict[str, int] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    ctrls = {actor_id: _controller(adapter) for actor_id, adapter in adapters.items()}

    def open_gate(actor_id: str) -> None:
        if gates and actor_id in gates:
            gates[actor_id].set()

    def force_ready(actor_id: str) -> None:
        open_gate(actor_id)
        thread = getattr(ctrls[actor_id], "_thinking_thread", None)
        if thread is not None:
            thread.join()

    for actor_id, adapter in adapters.items():
        ctrl = ctrls[actor_id]
        if runtime == "strands":
            if strands_invoke_overrides and actor_id in strands_invoke_overrides:
                adapter._invoke_strands = strands_invoke_overrides[actor_id]
            _wrap_strands_invoke(adapter, actor_id, call_log)
        else:
            if client_overrides and actor_id in client_overrides:
                ctrl.model_client = client_overrides[actor_id]
            ctrl.model_client = _TimingClient(ctrl.model_client, actor_id, call_log)
        _wrap_request(ctrl, actor_id, request_log)

        def make_get(adapter=adapter, ctrl=ctrl, actor_id=actor_id):
            original = adapter.get_action

            def wrapped(observation):
                world = world_state_snapshot(game.state)
                requested_before = ctrl.decisions_requested
                thinking_before = bool(getattr(ctrl, "_thinking_in_progress", False))
                completed_before = ctrl.decisions_completed
                action = original(observation)
                requested = ctrl.decisions_requested > requested_before
                entry = {"tick": tick["n"], "actor_id": actor_id, "returned": _action_dict(action), "requested": requested,
                         "thinking_before": thinking_before, "ready_at_call": not thinking_before and completed_before >= 1,
                         "thinking_after": bool(getattr(ctrl, "_thinking_in_progress", False)),
                         "decisions_completed": ctrl.decisions_completed, "wall": time.time()}
                if requested and actor_id not in snapshots:
                    snapshots[actor_id] = {"tick": tick["n"], "world": world, "observation": observation,
                                           "legal": enumerate_legal_actions(observation), "requested_mono": time.monotonic()}
                # COMMIT = the tick the first decision's own action is handed
                # to the Engine. Until that decision is ready the controller
                # can only return WAIT, so the first non-WAIT return is the
                # commit; a WAIT decision commits on the first call that
                # started with the decision already ready. (Reading the flag
                # only after the call would misdate a thread that finishes
                # inside `get_action` after it has already chosen WAIT.)
                if actor_id in snapshots and actor_id not in committed and (action.type is not ActionType.WAIT or entry["ready_at_call"]):
                    committed[actor_id] = tick["n"]
                    entry["commit"] = True
                get_log.append(entry)
                # (only ever the actor's first decision: a fresh request its
                # own commit tick started must stay in flight)
                if actor_id in force_handoff and actor_id not in committed and entry["thinking_after"] and all(
                        o in committed for o in ACTORS if o != actor_id and o not in force_handoff) and (
                        time.monotonic() - snapshots[actor_id]["requested_mono"] >= force_handoff_min_age_s):
                    force_ready(actor_id)
                return action

            return wrapped

        adapter.get_action = make_get()

    original_apply = game.state.apply_action

    def wrapped_apply(actor_id, action, dt):
        is_commit = committed.get(actor_id) == tick["n"] and not any(r["actor_id"] == actor_id and r["commit"] for r in apply_log)
        before = world_state_snapshot(game.state) if (is_commit or action.type is not ActionType.WAIT) else None
        listed = None
        if is_commit:
            listed = enumerate_legal_actions(build_observation(game.state, actor_id))
        events_before = len(game.state.event_log)
        cooldown_before = game.state.actors[actor_id].action_cooldown
        clock = game_clock(game.state)
        original_apply(actor_id, action, dt)
        after = world_state_snapshot(game.state) if before is not None else None
        apply_log.append({
            "tick": tick["n"], "actor_id": actor_id, "action": _action_dict(action), "dt": dt, "commit": is_commit,
            "clock_before": clock, "world_before": before, "world_after": after, "live_legal": listed,
            "events": game.state.event_log[events_before:], "cooldown_before": cooldown_before,
            "cooldown_after": game.state.actors[actor_id].action_cooldown,
        })
        if is_commit:
            for other in ACTORS:
                if other == actor_id or other in committed:
                    continue
                if other in force_ready_after_opponent_commit:
                    force_ready(other)
                elif other in release_after_opponent_commit:
                    open_gate(other)

    game.state.apply_action = wrapped_apply

    started = datetime.now(timezone.utc).isoformat()
    t_start = time.monotonic()
    tick_log: list[dict[str, Any]] = []
    while time.monotonic() - t_start < max_wall_seconds:
        tick["n"] += 1
        clock_before, wall_start = game.state.mission_time_remaining, time.time()
        game._step(DECISION_TICK_DT)
        tick_log.append({"tick": tick["n"], "wall_start": wall_start, "wall_end": time.time(),
                         "clock_before": clock_before, "clock_after": game.state.mission_time_remaining})
        if len(committed) == 2:
            break
        time.sleep(tick_sleep_seconds)
    for actor_id in ACTORS:  # never leave a scripted answer blocked behind a gate
        open_gate(actor_id)
    final_world = world_state_snapshot(game.state)
    finished = datetime.now(timezone.utc).isoformat()
    extra_requests = sum(1 for e in get_log if e["requested"]) - len(snapshots)
    row = _assemble(scenario, trial_id, runtime, decision_call_order, started, finished, pre_world, harness_world,
                    final_world, call_log, request_log, get_log, apply_log, snapshots, committed, extra_requests,
                    timed_out=len(committed) < 2)
    row.update(_freeze_audit(tick_log, request_log, row))
    row.update(_commit_schedule(apply_log, committed))
    events = game.state.event_log[events_start:]
    first_commit = next((a for a in apply_log if a["commit"]), None)
    # A request after an actor's first one: which actor, on which tick, and
    # whether that tick was also its own commit (its stale first decision
    # applied as the in-flight action while a fresh decision is requested).
    first_request_tick = {a: s["tick"] for a, s in snapshots.items()}
    row["extra_request_log"] = [
        {"actor_id": e["actor_id"], "tick": e["tick"], "on_own_commit_tick": committed.get(e["actor_id"]) == e["tick"],
         "opponent_committed_before": any(t < e["tick"] for a, t in committed.items() if a != e["actor_id"]),
         "returned": e["returned"]["type"]}
        for e in get_log if e["requested"] and e["tick"] != first_request_tick.get(e["actor_id"])
    ]
    row.update({
        "world_hash_before": json_hash(pre_world),
        "world_hash_after_first_apply": json_hash(first_commit["world_after"]) if first_commit else None,
        "world_hash_after_pair": row["world_state_hash_after"],
        "effect_state_hash_after": effect_state_hash(final_world),
        "event_log_hash": event_log_hash(game.state, events_start),
        "event_multiset_hash": json_hash(sorted(json.dumps(e, sort_keys=True, default=str) for e in events)),
        "events": events,
    })
    for actor_id, rec in row["records"].items():
        if not rec.get("incomplete"):
            rec["stale_distance"] = _stale_distance(apply_log, snapshots[actor_id], actor_id, rec)
    return row


def _stale_distance(apply_log: list[dict[str, Any]], snap: dict[str, Any], actor_id: str, rec: dict[str, Any]) -> int:
    """Spec section 3: relevant world revisions between this decision's
    snapshot and its own apply - each apply (the other actor's commit, a
    MOVE step) that changed the decision's relevant state counts once."""
    decision = rec["decision"]
    prev = relevant_state_hash(snap["world"], actor_id, decision)
    revisions = 0
    for a in apply_log:
        if a["actor_id"] == actor_id and a["commit"]:
            break
        if a["world_after"] is None or a["tick"] < snap["tick"]:
            continue
        h = relevant_state_hash(a["world_after"], actor_id, decision)
        if h != prev:
            revisions, prev = revisions + 1, h
    return max(revisions, 1) if rec["relevant_state_changed"] else revisions


def _commit_schedule(apply_log: list[dict[str, Any]], committed: dict[str, int]) -> dict[str, Any]:
    """The tick plan the loop actually ran up to the second commit, as
    `replay_pair(observed_schedule=...)` takes it: from the first commit
    (or an earlier live tick, if one ran) to the second commit. Runs of
    frozen idle ticks collapse to one - a dt=0 tick where nobody commits
    changes nothing, since before its commit an actor only WAITs and after
    it may only WAIT or continue a MOVE (`reapplied_instant_actions_after_
    commit` checks the latter). Live idle ticks are kept one by one."""
    tick_dt: dict[int, float] = {}
    for a in apply_log:
        tick_dt.setdefault(a["tick"], a["dt"])
    out: dict[str, Any] = {"observed_commit_schedule": None, "frozen_idle_ticks_between_commits": None,
                           "live_idle_ticks_before_second_commit": None, "reapplied_instant_actions_after_commit": None}
    if len(committed) < 2:
        return out
    first_tick, last_tick = min(committed.values()), max(committed.values())
    live_before = [t for t, dt in tick_dt.items() if dt > 0.0 and t < last_tick and not any(c == t for c in committed.values())]
    schedule: list[list[Any]] = []
    frozen_idle = 0
    for t in range(min([first_tick] + live_before), last_tick + 1):
        dt = tick_dt.get(t, 0.0)
        committing = [a for a in ACTORS if committed[a] == t]
        if not committing and dt == 0.0:
            frozen_idle += first_tick < t
            if not schedule or (not schedule[-1][1] and schedule[-1][0] == 0.0):
                continue
        schedule.append([dt, committing])
    reapplied = sum(1 for a in apply_log if a["actor_id"] in committed and a["tick"] > committed[a["actor_id"]]
                    and a["action"]["type"] not in ("wait", "move"))
    out.update({"observed_commit_schedule": schedule, "frozen_idle_ticks_between_commits": frozen_idle,
                "live_idle_ticks_before_second_commit": len(live_before), "reapplied_instant_actions_after_commit": reapplied})
    return out


def _freeze_audit(tick_log: list[dict[str, Any]], request_log: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    """Spec section 11/25: did game time move while a model call was in
    flight? A tick counts as "during inference" only if it started after a
    request began and ended before that decision was ready. The tick in
    which the last decision becomes ready is a handoff tick: the loop reads
    the thinking flag after `get_action`, so if the thread finishes inside
    that window the tick runs live and the decision commits one tick later
    (at most one 1/60 s tick of game time, independent of latency)."""
    windows = [(r["request_start_wall"], r["ready_wall"]) for r in request_log if "ready_wall" in r]
    advanced = [t for t in tick_log if t["clock_after"] < t["clock_before"]]
    during = [t for t in advanced if any(s <= t["wall_start"] and t["wall_end"] <= e for s, e in windows)]
    # the handoff is about the pair's own decisions: an extra request that
    # starts later must not push the "last ready" moment past the handoff
    first_ready = {}
    for r in sorted((r for r in request_log if "ready_wall" in r), key=lambda r: r["request_start_wall"]):
        first_ready.setdefault(r["actor_id"], r["ready_wall"])
    last_ready = max(first_ready.values(), default=None)
    records = row["records"]
    second_tick = max((r.get("apply_tick") or 0) for r in records.values()) if row["complete"] else None
    handoff = [t for t in advanced if second_tick is not None and t["tick"] < second_tick and last_ready is not None and t["wall_end"] >= last_ready]
    return {
        "ticks_advanced_during_inference": len(during),
        "game_time_advanced_during_inference_s": round(sum(t["clock_before"] - t["clock_after"] for t in during), 6),
        "handoff_ticks_before_second_apply": len(handoff),
        "game_time_before_second_apply_s": round(sum(t["clock_before"] - t["clock_after"] for t in advanced
                                                     if second_tick is not None and t["tick"] < second_tick), 6),
    }


def _final_raw(calls: list[dict[str, Any]], observation, legal) -> tuple[dict[str, Any], bool, str | None]:
    """The payload production actually executed: the first call if it
    validated, else the repair call if that validated, else a fallback."""
    for c in calls:
        v = validate_model_output(observation, c["payload"], legal)
        if v.valid:
            return c["payload"], True, None
    last = calls[-1]["payload"] if calls else {}
    v = validate_model_output(observation, last, legal)
    return last, False, v.error_type


def _assemble(scenario, trial_id, runtime, call_order, started, finished, pre_world, harness_world, final_world, call_log,
              request_log, get_log, apply_log, snapshots, committed, extra_requests, timed_out) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for actor_id in ACTORS:
        calls = sorted((c for c in call_log if c["actor_id"] == actor_id), key=lambda c: c["call_start_wall"])
        requests = sorted((r for r in request_log if r["actor_id"] == actor_id), key=lambda r: r["request_start_wall"])
        first_request = requests[0] if requests else {}
        first_calls = [c for c in calls if not first_request or c["call_start_wall"] <= first_request.get("ready_wall", float("inf"))]
        snap = snapshots.get(actor_id)
        apply = next((a for a in apply_log if a["actor_id"] == actor_id and a["commit"]), None)
        rec: dict[str, Any] = {"actor_id": actor_id, "model_calls_for_first_decision": len(first_calls),
                               "extra_requests_before_pair_complete": 0}
        if snap is None or apply is None:
            rec["incomplete"] = True
            records[actor_id] = rec
            continue
        raw, first_pass_valid, error = _final_raw(first_calls, snap["observation"], snap["legal"])
        decision = _decision_key(validate_model_output(snap["observation"], raw, snap["legal"]).validated_action) if first_pass_valid else _decision_key(None)
        before, after = apply["world_before"], apply["world_after"]
        types = [e["type"] for e in apply["events"] if e.get("actor_id") == actor_id]
        outcome = classify_apply_outcome(decision["action"], json_hash(before) != json_hash(after), types)
        if decision["action"] == "move" and apply["dt"] == 0.0:
            outcome = "COMMITTED_FROZEN"
        effective = APPLY_APPLIED if outcome == "COMMITTED_FROZEN" else outcome
        cf = fresh_counterfactual(scenario, actor_id, raw)
        rec.update({
            "request_start_wall": first_request.get("request_start_wall"),
            "response_ready_wall": first_request.get("ready_wall"),
            "response_latency_ms": first_request.get("request_to_ready_ms"),
            "model_call_latency_ms": [c["call_latency_ms"] for c in first_calls],
            "input_tokens": sum(c["input_tokens"] or 0 for c in first_calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in first_calls),
            "provider_error_types": [c["error_type"] for c in first_calls if c["error_type"]],
            "provider_request_hash": first_calls[0]["provider_request_hash"] if first_calls else None,
            "decision_id": f"{trial_id}:{actor_id}",
            "snapshot_id": snapshot_id(scenario.family, json_hash(snap["world"])),
            "snapshot_tick": snap["tick"],
            "snapshot_world_hash": json_hash(snap["world"]),
            "snapshot_relevant_hash": relevant_state_hash(snap["world"], actor_id, decision),
            "commit_tick": committed.get(actor_id),
            "apply_tick": apply["tick"],
            "apply_dt": apply["dt"],
            "apply_game_clock": apply["clock_before"],
            "apply_world_hash": json_hash(before),
            "apply_relevant_hash": relevant_state_hash(before, actor_id, decision),
            "raw_model_output": raw,
            "first_pass_valid": first_pass_valid,
            "validation_error": error,
            "decision": decision,
            "decision_hash": json_hash(decision),
            "listed_at_snapshot": _is_listed(decision, snap["legal"]),
            "listed_at_apply": _is_listed(decision, apply["live_legal"] or []),
            "committed_action": apply["action"],
            "apply_outcome": outcome,
            "effective_outcome": effective,
            "apply_event_types": types,
            "time_cost": round(apply["cooldown_after"], 6) if apply["cooldown_before"] == 0 and apply["cooldown_after"] > 0 else 0.0,
            "busy_after_apply": apply["cooldown_after"] > 0,
            # this action's own effect, read right after it applied - a MOVE
            # is read at the end instead (its displacement starts on the
            # first live tick), the same convention as `replay_pair`
            "target_state_after": own_effect_view(final_world if decision["action"] == "move" else after, actor_id, decision),
            "self_state_after": actor_self_state(final_world if decision["action"] == "move" else after, actor_id),
            "target_state_before_apply": target_state(before, decision["target_id"]),
            "other_actor_busy_at_apply": before["actors"][ACTOR_B if actor_id == ACTOR_A else ACTOR_A]["action_cooldown"] > 0,
            "counterfactual": cf,
        })
        # The probe's commit tick must be the tick the decision's own first
        # GameAction was applied (a MOVE may start as WAIT when already at
        # its target); False would mean the commit was mis-detected.
        committed_type = apply["action"]["type"]
        rec["commit_matches_decision"] = (committed_type == "wait" if decision["action"] == "wait" else
                                          committed_type in ("move", "wait") if decision["action"] == "move" else
                                          committed_type == decision["action"])
        rec["legal_at_snapshot"] = cf["engine_valid"]
        rec["legal_at_apply"] = effective in (APPLY_APPLIED, APPLY_INTENTIONAL_WAIT)
        rec["world_changed"] = rec["apply_world_hash"] != rec["snapshot_world_hash"]
        rec["relevant_state_changed"] = rec["apply_relevant_hash"] != rec["snapshot_relevant_hash"]
        rec["staleness_class"] = classify_staleness(world_changed=rec["world_changed"], relevant_changed=rec["relevant_state_changed"],
                                                    engine_valid_at_snapshot=cf["engine_valid"], legal_at_apply=rec["legal_at_apply"])
        rec["consequence"] = classify_consequence(rec, cf) if decision["action"] != "move" else ("ACCEPTED_UNCHANGED" if rec["legal_at_apply"] else "REFUSED")
        records[actor_id] = rec

    complete = all(not r.get("incomplete") for r in records.values())
    ready_order = sorted(ACTORS, key=lambda a: records[a].get("response_ready_wall") or float("inf")) if complete else None
    apply_order = sorted(ACTORS, key=lambda a: (records[a].get("apply_tick") or 10**9, 0 if a == ACTOR_A else 1)) if complete else None
    if complete:
        for rank, a in enumerate(ready_order, start=1):
            records[a]["ready_rank"] = rank
        for idx, a in enumerate(apply_order):
            records[a]["apply_index"] = idx
    clocks = [r.get("apply_game_clock") for r in records.values()]
    return {
        "trial_id": trial_id,
        "scenario_family": scenario.family,
        "runtime": runtime,
        "decision_call_order": call_order,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "timed_out": timed_out,
        "complete": complete,
        "pre_action_world_hash": json_hash(pre_world),
        "pre_action_matches_harness_snapshot": json_hash(pre_world) == json_hash(harness_world),
        "world_state_hash_after": json_hash(final_world),
        "final_world": final_world,
        "ticks": max((e["tick"] for e in get_log), default=0),
        "same_tick_commit": complete and records[ACTOR_A]["apply_tick"] == records[ACTOR_B]["apply_tick"],
        "ready_order": ready_order,
        "apply_order": apply_order,
        "ready_order_equals_apply_order": complete and ready_order == apply_order,
        "game_clock_at_start": pre_world["mission_time_remaining"],
        "game_clock_at_each_apply": clocks,
        "game_time_frozen_until_second_apply": complete and all(c and c["mission_time_remaining"] == pre_world["mission_time_remaining"] for c in clocks),
        "extra_requests_before_pair_complete": extra_requests,
        "records": records,
        "contested_effects": [(e["type"], e.get("actor_id")) for a in apply_log for e in a["events"]
                              if e["type"] in {"DOOR_UNLOCKED", "TRAP_PLACED", "TRAP_REMOVED", "TRAP_DISARMED", "TRAP_TRIGGERED"}],
    }


def mission_time() -> float:
    return constants.MISSION_TIME
