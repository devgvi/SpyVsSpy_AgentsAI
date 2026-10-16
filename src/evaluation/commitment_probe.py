"""M6G.2 spec sections 32-38, 49-51: the production game loop under an
explicit commitment policy, observed from outside.

`run_policy_trial` builds a real Game with two LLM actors (M6G.1's
`build_game`) and steps the real `Game._step`. Under P0 nothing is changed:
the loop commits whichever response is ready first, exactly as M6G.1
measured it. Under a barrier policy (P1-P4) an evaluation-level commit
scheduler sits between the loop and each controller:

- a ready decision is HELD: that actor's get_action is not called, so the
  controller neither returns the decision nor re-evaluates its observation,
  until the other actor's decision for the same boundary is ready too;
- while a decision is held the tick runs at dt=0 (apply_action and update
  see dt=0), so barrier waiting advances no game time (spec section 49);
- on the release tick both get_action calls run and both decisions commit
  on one live tick, in the policy's order (Game._step applies Black then
  White; the apply wrapper defers Black's apply until White's when the
  policy puts White first);
- with more than one boundary, the next one opens only when both actors
  are free again: a free actor is kept waiting (not polled) while the other
  finishes its action on live ticks, then both are polled on the same tick
  and request from the same snapshot.

The controllers' request/digest/throttle logic, validation, repair, the
ActionExecutor and the Engine are all production code. The scheduler
chooses only WHEN a controller is polled and in which order two same-tick
commits reach GameState.apply_action; it never touches a decision.

Every polled get_action is also classified from outside against the
controller's own decision trigger (the observation digest without the time
field, the 0.5 s re-request throttle), and every request is tracked
through REQUEST_IN_FLIGHT -> RESPONSE_READY_UNCOMMITTED -> COMMITTED, so a
replacement request launched while a response is ready but uncommitted
(HP09) is caught wherever it happens.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.actions import ActionType, GameAction
from src.evaluation.commitment_policy import (
    ATTEMPT_ABSORBED,
    ATTEMPT_BUSY,
    ATTEMPT_EXECUTING,
    ATTEMPT_LAUNCHED,
    ATTEMPT_NO_CHANGE,
    ATTEMPT_THROTTLED,
    COMMITTED,
    IDLE,
    P0,
    P1,
    POLICIES,
    REQUEST_IN_FLIGHT,
    RESPONSE_READY_UNCOMMITTED,
    counterfactual_on_state,
    hp09_violation,
)
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
from src.evaluation.real_loop_probe import (
    _action_dict,
    _commit_schedule,
    _controller,
    _final_raw,
    _TimingClient,
    _wrap_request,
    _wrap_strands_invoke,
    build_game,
)
from src.evaluation.temporal_semantics import (
    APPLY_APPLIED,
    APPLY_INTENTIONAL_WAIT,
    actor_self_state,
    classify_consequence,
    classify_staleness,
    effect_state_hash,
    own_effect_view,
    relevant_state_hash,
    target_state,
)
from src.llm_controller import enumerate_legal_actions, serialize_observation, validate_model_output
from src.observation import build_observation

CONTESTED = {"DOOR_UNLOCKED", "TRAP_PLACED", "TRAP_REMOVED", "TRAP_DISARMED", "TRAP_TRIGGERED"}


PATCHED_STATE_METHODS = ("apply_action", "update")


def unpatched_copy(state):
    """A deep copy of the live GameState for counterfactuals, without the
    probe's instance-level apply_action/update wrappers. deepcopy keeps
    function attributes by reference, so a plain copy would still route
    apply_action through the wrapper to the LIVE game (and its apply log)."""
    c = copy.deepcopy(state)
    for name in PATCHED_STATE_METHODS:
        vars(c).pop(name, None)
    return c


def observation_digest(observation, live_state: bool = True) -> str:
    """The controllers' own decision trigger (LLMController.get_action and
    StrandsAgentAdapter.get_action compute exactly this). `live_state`
    mirrors their `live_state_representation` (default True since the
    navigation/exploration bugfix); pass False for a controller built with
    the recorded M5/M6 representation."""
    payload = dict(serialize_observation(observation, live_state=live_state))
    payload.pop("time", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class _Plan:
    tick: int
    hold: set = field(default_factory=set)
    freeze: bool = False
    release: bool = False
    opened: bool = False


@dataclass
class _Boundary:
    index: int
    open_tick: int
    launches: dict = field(default_factory=dict)       # actor -> launch info
    ready_seen_tick: dict = field(default_factory=dict)
    commits: dict = field(default_factory=dict)        # actor -> commit info
    order: tuple | None = None                          # the scheduler's release order (barrier policies)
    release_tick: int | None = None
    hold_ticks: set = field(default_factory=set)
    complete_tick: int | None = None
    end_world: dict | None = None


class _Scheduler:
    def __init__(self, policy_id: str, boundaries: int, parity: int, game, ctrls: dict[str, Any]) -> None:
        self.policy_id = policy_id
        self.policy = POLICIES[policy_id]
        self.barrier = self.policy.waits_for_both
        if not self.barrier and boundaries != 1:
            raise ValueError("P0 runs the production loop, which has no synchronized second boundary")
        self.n_boundaries = boundaries
        self.parity = parity
        self.game = game
        self.ctrls = ctrls
        self.boundaries = [_Boundary(0, 1)]
        self.phase = "deciding"
        self.consumed = {a: 0 for a in ACTORS}   # decisions_completed value consumed by the last commit
        self.align_hold_ticks: set = set()
        self._plan: _Plan | None = None

    @property
    def cur(self) -> _Boundary:
        return self.boundaries[-1]

    def thinking(self, a: str) -> bool:
        return bool(getattr(self.ctrls[a], "_thinking_in_progress", False))

    def is_ready(self, a: str) -> bool:
        b = self.cur
        return a in b.launches and not self.thinking(a) and self.ctrls[a].decisions_completed >= b.launches[a]["requested_count"]

    def is_free(self, a: str) -> bool:
        actor = self.game.state.actors[a]
        executor = getattr(self.ctrls[a], "_executor", None)
        return (not self.thinking(a) and actor.action_cooldown <= 0 and actor.stunned_timer <= 0
                and not (executor is not None and executor.in_progress))

    def response_state(self, a: str) -> str:
        if self.thinking(a):
            return REQUEST_IN_FLIGHT
        if self.ctrls[a].decisions_completed > self.consumed[a]:
            return RESPONSE_READY_UNCOMMITTED
        return COMMITTED if self.consumed[a] > 0 else IDLE

    def expects_launch(self, a: str) -> bool:
        return self.phase == "deciding" and a not in self.cur.launches

    def plan(self, t: int) -> _Plan:
        if self._plan is not None and self._plan.tick == t:
            return self._plan
        p = _Plan(t)
        if self.barrier and self.phase == "deciding":
            b = self.cur
            ready = {a: self.is_ready(a) for a in ACTORS}
            if len(b.launches) == 2 and all(ready.values()):
                for a in ACTORS:
                    b.ready_seen_tick.setdefault(a, t)
                p.release, b.release_tick, b.order = True, t, self._order(b)
            else:
                p.hold = {a for a in ACTORS if ready[a]}
                for a in p.hold:
                    b.ready_seen_tick.setdefault(a, t)
                if p.hold:
                    p.freeze = True
                    b.hold_ticks.add(t)
        elif self.barrier and self.phase == "aligning":
            free = {a: self.is_free(a) for a in ACTORS}
            if all(free.values()):
                self.boundaries.append(_Boundary(len(self.boundaries), t))
                self.phase, p.opened = "deciding", True
            else:
                p.hold = {a for a in ACTORS if free[a]}
                if p.hold:
                    self.align_hold_ticks.add(t)
        self._plan = p
        return p

    def _order(self, b: _Boundary) -> tuple[str, str]:
        if self.policy_id == P1:  # P0's order: ready tick first, same tick -> Black then White
            ta, tb = b.ready_seen_tick[ACTOR_A], b.ready_seen_tick[ACTOR_B]
            ready = None if ta == tb else ((ACTOR_A, ACTOR_B) if ta < tb else (ACTOR_B, ACTOR_A))
            return self.policy.order(None, ready, b.index, self.parity)
        return self.policy.order(None, None, b.index, self.parity)

    def on_commit(self, a: str, t: int) -> None:
        b = self.cur
        self.consumed[a] = self.ctrls[a].decisions_completed
        if len(b.commits) == 2:
            b.complete_tick = t
            self.phase = "aligning" if len(self.boundaries) < self.n_boundaries else "done"

    def white_first_release(self, t: int) -> bool:
        b = self.cur if self.cur.release_tick == t else next((x for x in self.boundaries if x.release_tick == t), None)
        return b is not None and b.order == (ACTOR_B, ACTOR_A)


def run_policy_trial(
    scenario: ConflictScenario,
    policy_id: str,
    *,
    trial_id: str,
    runtime: str = "native_tool_like",
    provider: str = "bedrock",
    model_id: str,
    region: str | None = None,
    decision_call_order: str = "black_first",
    boundaries: int = 1,
    p4_parity: int = 0,
    client_overrides: dict[str, Any] | None = None,
    strands_invoke_overrides: dict[str, Any] | None = None,
    gates: dict[str, threading.Event] | None = None,
    release_after_opponent_step: tuple[str, ...] = (),
    force_ready_after_opponent_step: tuple[str, ...] = (),
    ready_ticks_after_opponent_step: dict[str, int] | None = None,
    force_ready_at_tick: int | None = None,
    max_wall_seconds: float = 150.0,
    tick_sleep_seconds: float = 0.005,
    join_inflight_seconds: float = 90.0,
) -> dict[str, Any]:
    """One trial of `policy_id` on the production loop. Scripted clients
    (`client_overrides` / `strands_invoke_overrides`, with optional
    `gates`) stand in for the model in network-free tests; live runs pass
    none of them.

    Test-only knobs pin interleavings down deterministically:
    `release_after_opponent_step` opens an actor's gate, and
    `force_ready_after_opponent_step` also joins its request thread, at the
    moment the opponent's first decision is handed to the loop - its commit
    under P0, its first held tick under a barrier. `force_ready_at_tick`
    makes every in-flight answer ready at the end of that tick.
    `ready_ticks_after_opponent_step` {actor: k} makes the actor's answer
    ready at the end of the k-th tick after that moment (k >= 1), or later
    if its controller has not yet absorbed the observation change it saw
    in flight: the actor has been polled in flight after the opponent's
    step, its answer never lands between its own poll and Game._step's
    thinking-flag check (M6G.1's handoff window), and it is never consumed
    with a stale stored digest - the ordinary no-race interleaving, pinned
    independently of machine load."""
    if ready_ticks_after_opponent_step and min(ready_ticks_after_opponent_step.values()) < 1:
        raise ValueError("ready_ticks_after_opponent_step needs k >= 1")
    game = build_game(scenario, runtime=runtime, provider=provider, model_id=model_id, region=region,
                      decision_call_order=decision_call_order)
    adapters = {ACTOR_A: game.black_llm_controller, ACTOR_B: game.bot_controller}
    ctrls = {a: _controller(ad) for a, ad in adapters.items()}
    sched = _Scheduler(policy_id, boundaries, p4_parity, game, ctrls)
    pre_world = world_state_snapshot(game.state)
    harness_world = world_state_snapshot(fresh_scenario_state(scenario.setup_fn))
    events_start = len(game.state.event_log)

    call_log: list[dict[str, Any]] = []
    request_log: list[dict[str, Any]] = []
    get_log: list[dict[str, Any]] = []
    apply_log: list[dict[str, Any]] = []
    tick_log: list[dict[str, Any]] = []
    tick = {"n": 0}
    stepped_for: set[str] = set()   # actors whose first decision has been handed to the loop (for the test knobs)
    knob_fired: set[str] = set()
    ready_at_end_of_tick: dict[str, int] = {}
    last_poll: dict[str, tuple[str, str | None]] = {}   # actor -> (observation digest, controller's stored digest) at its last poll

    def open_gate(a: str) -> None:
        if gates and a in gates:
            gates[a].set()

    def join_request(a: str, timeout: float | None = None) -> None:
        thread = getattr(ctrls[a], "_thinking_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def fire_opponent_knobs(stepped_actor: str) -> None:
        for other in ACTORS:
            if other == stepped_actor or other in knob_fired:
                continue
            if other in force_ready_after_opponent_step:
                knob_fired.add(other)
                open_gate(other)
                join_request(other)
            elif ready_ticks_after_opponent_step and other in ready_ticks_after_opponent_step:
                knob_fired.add(other)
                ready_at_end_of_tick[other] = tick["n"] + ready_ticks_after_opponent_step[other]
            elif other in release_after_opponent_step:
                knob_fired.add(other)
                open_gate(other)

    for a, adapter in adapters.items():
        ctrl = ctrls[a]
        if runtime == "strands":
            if strands_invoke_overrides and a in strands_invoke_overrides:
                adapter._invoke_strands = strands_invoke_overrides[a]
            _wrap_strands_invoke(adapter, a, call_log)
        else:
            if client_overrides and a in client_overrides:
                ctrl.model_client = client_overrides[a]
            ctrl.model_client = _TimingClient(ctrl.model_client, a, call_log)
        _wrap_request(ctrl, a, request_log)

        def make_get(adapter=adapter, ctrl=ctrl, a=a):
            original = adapter.get_action

            def wrapped(observation):
                t = tick["n"]
                p = sched.plan(t)
                prior = sched.response_state(a)
                b = sched.cur
                if a in p.hold:
                    get_log.append({"tick": t, "actor_id": a, "held": True, "boundary": b.index, "prior_state": prior, "wall": time.time()})
                    return GameAction(ActionType.WAIT)
                expects = sched.expects_launch(a)
                world = world_state_snapshot(game.state) if expects else None
                state_copy = unpatched_copy(game.state) if expects else None
                stored_before = getattr(ctrl, "_last_observation_digest", None)
                digest = observation_digest(observation)
                busy = bool(observation.busy or observation.stunned or observation.mission_complete)
                executor = getattr(ctrl, "_executor", None)
                executing = bool(executor is not None and executor.in_progress)
                requested_before, completed_before = ctrl.decisions_requested, ctrl.decisions_completed
                thinking_before = sched.thinking(a)
                wall = time.time()
                action = original(observation)
                launched = ctrl.decisions_requested > requested_before
                stored_after = getattr(ctrl, "_last_observation_digest", None)
                last_poll[a] = (digest, stored_after)
                if busy:
                    attempt = ATTEMPT_BUSY
                elif launched:
                    attempt = ATTEMPT_LAUNCHED
                elif executing and stored_after == stored_before:
                    attempt = ATTEMPT_EXECUTING
                elif digest == stored_before:
                    attempt = ATTEMPT_NO_CHANGE
                elif stored_after != stored_before:
                    attempt = ATTEMPT_ABSORBED
                else:
                    attempt = ATTEMPT_THROTTLED
                entry = {"tick": t, "actor_id": a, "held": False, "boundary": b.index, "prior_state": prior, "attempt": attempt,
                         "launched": launched, "returned": _action_dict(action), "thinking_before": thinking_before,
                         "thinking_after": sched.thinking(a), "wall": wall, "release_tick": p.release}
                if launched:
                    boundary_launch = expects and sched.phase == "deciding" and a not in b.launches
                    entry.update({"launch_kind": "boundary" if boundary_launch else "extra", "hp09_violation": hp09_violation(prior, True),
                                  "requested_count": ctrl.decisions_requested})
                    if boundary_launch:
                        b.launches[a] = {"tick": t, "requested_count": ctrl.decisions_requested, "world": world, "observation": observation,
                                         "legal": enumerate_legal_actions(observation), "state_copy": state_copy, "wall": wall,
                                         "get_index": len(get_log), "opened_tick": b.open_tick}
                # COMMIT of this boundary's decision (M6G.1's rule): the first
                # non-WAIT return, or the first call that began with the
                # decision already ready. A barrier lets it through only on
                # its release tick.
                if a in b.launches and a not in b.commits:
                    ready_at_call = not thinking_before and completed_before >= b.launches[a]["requested_count"]
                    if action.type is not ActionType.WAIT or ready_at_call:
                        if sched.barrier and not p.release:
                            action = GameAction(ActionType.WAIT)
                            entry["early_return_held"] = True
                        else:
                            b.commits[a] = {"tick": t, "wall": wall, "get_index": len(get_log), "action": _action_dict(action),
                                            "consumed_count": ctrl.decisions_completed}
                            entry["commit"] = True
                            sched.on_commit(a, t)
                get_log.append(entry)
                return action

            return wrapped

        adapter.get_action = make_get()

    original_apply = game.state.apply_action
    original_update = game.state.update
    deferred: dict[str, Any] = {}

    def commit_boundary_of(a: str, t: int) -> _Boundary | None:
        return next((bb for bb in sched.boundaries if a in bb.commits and bb.commits[a]["tick"] == t
                     and not bb.commits[a].get("applied")), None)

    def do_apply(a: str, action: GameAction, dt: float, bnd: _Boundary | None) -> None:
        is_commit = bnd is not None
        before = world_state_snapshot(game.state) if (is_commit or action.type is not ActionType.WAIT) else None
        listed = enumerate_legal_actions(build_observation(game.state, a)) if is_commit else None
        events_before = len(game.state.event_log)
        cooldown_before = game.state.actors[a].action_cooldown
        clock = game_clock(game.state)
        original_apply(a, action, dt)
        after = world_state_snapshot(game.state) if before is not None else None
        entry = {"tick": tick["n"], "seq": len(apply_log), "actor_id": a, "action": _action_dict(action), "dt": dt, "commit": is_commit,
                 "commit_boundary": bnd.index if bnd else None, "clock_before": clock, "world_before": before, "world_after": after,
                 "live_legal": listed, "events": game.state.event_log[events_before:], "cooldown_before": cooldown_before,
                 "cooldown_after": game.state.actors[a].action_cooldown}
        apply_log.append(entry)
        if bnd is not None:
            bnd.commits[a]["applied"] = True
            if policy_id == P0 and bnd.index == 0 and a not in stepped_for:
                stepped_for.add(a)
                fire_opponent_knobs(a)

    def wrapped_apply(a, action, dt):
        t = tick["n"]
        p = sched.plan(t)
        if p.freeze:
            dt = 0.0
        bnd = commit_boundary_of(a, t)
        if a == ACTOR_A and bnd is not None and p.release and sched.white_first_release(t):
            deferred["black"] = (action, dt, bnd)   # the policy puts White first on this release tick
            return
        do_apply(a, action, dt, bnd)
        if a == ACTOR_B and "black" in deferred:
            do_apply(ACTOR_A, *deferred.pop("black"))

    def wrapped_update(dt):
        t = tick["n"]
        p = sched.plan(t) if sched._plan is not None and sched._plan.tick == t else None
        if p is not None and p.freeze:
            dt = 0.0
        original_update(dt)
        for bb in sched.boundaries:
            if bb.complete_tick == t and bb.end_world is None:
                bb.end_world = world_state_snapshot(game.state)
        if sched.barrier and p is not None and sched.cur.index == 0:
            for a in p.hold:
                if a not in stepped_for:
                    stepped_for.add(a)
                    fire_opponent_knobs(a)
        if force_ready_at_tick is not None and t == force_ready_at_tick:
            for a in ACTORS:
                open_gate(a)
                join_request(a)
        for a in [x for x, at in ready_at_end_of_tick.items() if at <= t]:
            # Wait until the controller has absorbed the last observation change it saw in flight. Inside its
            # 0.5 s re-request throttle it does not store the new digest, and an answer consumed only after the
            # throttle expires would re-trigger a request (the HP09 path) - live answers take seconds, so live
            # changes are always absorbed first.
            seen, stored = last_poll.get(a, (None, None))
            if seen is not None and seen != stored:
                continue
            del ready_at_end_of_tick[a]
            open_gate(a)
            join_request(a)

    game.state.apply_action = wrapped_apply
    game.state.update = wrapped_update

    started = datetime.now(timezone.utc).isoformat()
    t_start = time.monotonic()
    while time.monotonic() - t_start < max_wall_seconds:
        tick["n"] += 1
        clock_before, wall_start = game.state.mission_time_remaining, time.time()
        game._step(DECISION_TICK_DT)
        plan = sched._plan if sched._plan is not None and sched._plan.tick == tick["n"] else _Plan(tick["n"])
        tick_log.append({"tick": tick["n"], "wall_start": wall_start, "wall_end": time.time(), "clock_before": clock_before,
                         "clock_after": game.state.mission_time_remaining, "held": sorted(plan.hold), "freeze": plan.freeze,
                         "release": plan.release, "opened": plan.opened, "phase_after": sched.phase})
        if sched.phase == "done":
            break
        time.sleep(tick_sleep_seconds)
    timed_out = sched.phase != "done"
    for a in ACTORS:  # never leave a scripted answer blocked behind a gate
        open_gate(a)
    finished = datetime.now(timezone.utc).isoformat()
    final_world = world_state_snapshot(game.state)
    events = game.state.event_log[events_start:]
    events_hash = event_log_hash(game.state, events_start)
    # Requests still in flight (a replacement launched on a commit tick) are
    # waited for - without stepping the game - so their calls and tokens
    # are logged; they can no longer be consumed.
    join_deadline = time.monotonic() + join_inflight_seconds
    for a in ACTORS:
        join_request(a, max(0.0, join_deadline - time.monotonic()))

    row = _assemble(scenario, policy_id, trial_id, runtime, decision_call_order, boundaries, p4_parity, started, finished, pre_world,
                    harness_world, final_world, events, call_log, request_log, get_log, apply_log, tick_log, sched, timed_out)
    row["event_log_hash"] = events_hash
    row["live_state_untouched_by_assembly"] = (json_hash(world_state_snapshot(game.state)) == json_hash(final_world)
                                               and len(game.state.event_log) == events_start + len(events))
    return row


# -- assembly ---------------------------------------------------------------------------------


def _requests(actor_id: str, get_log, request_log, call_log, sched: _Scheduler) -> list[dict[str, Any]]:
    launches = [e for e in get_log if e["actor_id"] == actor_id and not e["held"] and e["launched"]]
    runs = sorted((r for r in request_log if r["actor_id"] == actor_id), key=lambda r: r["request_start_wall"])
    calls = sorted((c for c in call_log if c["actor_id"] == actor_id), key=lambda c: c["call_start_wall"])
    consumed_counts = {c["consumed_count"]: (bb.index, c["tick"]) for bb in sched.boundaries for a, c in bb.commits.items() if a == actor_id}
    out = []
    for i, launch in enumerate(launches):
        run = runs[i] if i < len(runs) else None
        window_end = run["ready_wall"] if run else float("inf")
        mine = [c for c in calls if (run is None or c["call_start_wall"] >= run["request_start_wall"] - 1e-3) and c["call_start_wall"] <= window_end]
        calls = [c for c in calls if c not in mine]
        consumed = consumed_counts.get(launch["requested_count"])
        out.append({
            "actor_id": actor_id, "request_index": i + 1, "launch_tick": launch["tick"], "launch_kind": launch["launch_kind"],
            "boundary": launch["boundary"], "prior_state": launch["prior_state"], "hp09_violation": launch["hp09_violation"],
            "request_start_wall": run["request_start_wall"] if run else None, "ready_wall": run["ready_wall"] if run else None,
            "request_to_ready_ms": run["request_to_ready_ms"] if run else None, "ready": run is not None,
            "consumed": consumed is not None, "consumed_by_boundary": consumed[0] if consumed else None,
            "consumed_at_tick": consumed[1] if consumed else None,
            "model_calls": len(mine), "input_tokens": sum(c["input_tokens"] or 0 for c in mine),
            "output_tokens": sum(c["output_tokens"] or 0 for c in mine),
            "provider_error_types": [c["error_type"] for c in mine if c["error_type"]],
            "payloads": [c["payload"] for c in mine],
            "provider_request_hash": mine[0]["provider_request_hash"] if mine else None,
        })
    return out


def _decision_record(scenario, trial_id, b: _Boundary, a: str, req: dict[str, Any] | None, apply: dict[str, Any] | None,
                     end_world: dict[str, Any] | None) -> dict[str, Any]:
    launch = b.launches.get(a)
    rec: dict[str, Any] = {"actor_id": a, "boundary": b.index}
    if launch is None or apply is None or req is None:
        rec["incomplete"] = True
        return rec
    calls = [{"payload": p} for p in req["payloads"]]
    raw, first_pass_valid, error = _final_raw(calls, launch["observation"], launch["legal"])
    decision = _decision_key(validate_model_output(launch["observation"], raw, launch["legal"]).validated_action) if first_pass_valid else _decision_key(None)
    before, after = apply["world_before"], apply["world_after"]
    types = [e["type"] for e in apply["events"] if e.get("actor_id") == a]
    outcome = classify_apply_outcome(decision["action"], json_hash(before) != json_hash(after), types)
    if decision["action"] == "move" and apply["dt"] == 0.0:
        outcome = "COMMITTED_FROZEN"
    effective = APPLY_APPLIED if outcome == "COMMITTED_FROZEN" else outcome
    cf = counterfactual_on_state(launch["state_copy"], a, raw, family=scenario.family)
    view_world = (end_world or after) if decision["action"] == "move" else after
    rec.update({
        "decision_id": f"{trial_id}:b{b.index}:{a}",
        "request_index": req["request_index"],
        "snapshot_id": snapshot_id(scenario.family, json_hash(launch["world"])),
        "snapshot_tick": launch["tick"],
        "snapshot_world_hash": json_hash(launch["world"]),
        "snapshot_relevant_hash": relevant_state_hash(launch["world"], a, decision),
        "request_start_wall": req["request_start_wall"],
        "response_ready_wall": req["ready_wall"],
        "response_latency_ms": req["request_to_ready_ms"],
        "model_calls": req["model_calls"],
        "input_tokens": req["input_tokens"],
        "output_tokens": req["output_tokens"],
        "provider_error_types": req["provider_error_types"],
        "provider_request_hash": req["provider_request_hash"],
        "raw_model_output": raw,
        "first_pass_valid": first_pass_valid,
        "validation_error": error,
        "decision": decision,
        "decision_hash": json_hash(decision),
        "listed_at_snapshot": _is_listed(decision, launch["legal"]),
        "listed_at_apply": _is_listed(decision, apply["live_legal"] or []),
        "commit_tick": b.commits[a]["tick"],
        "commit_wall": b.commits[a]["wall"],
        "apply_seq": apply["seq"],
        "apply_dt": apply["dt"],
        "apply_game_clock": apply["clock_before"],
        "apply_world_hash": json_hash(before),
        "apply_relevant_hash": relevant_state_hash(before, a, decision),
        "committed_action": apply["action"],
        "apply_outcome": outcome,
        "effective_outcome": effective,
        "apply_event_types": types,
        "time_cost": round(apply["cooldown_after"], 6) if apply["cooldown_before"] == 0 and apply["cooldown_after"] > 0 else 0.0,
        "busy_after_apply": apply["cooldown_after"] > 0,
        "target_state_after": own_effect_view(view_world, a, decision),
        "self_state_after": actor_self_state(view_world, a),
        "target_state_before_apply": target_state(before, decision["target_id"]),
        "other_actor_busy_at_apply": before["actors"][ACTOR_B if a == ACTOR_A else ACTOR_A]["action_cooldown"] > 0,
        "counterfactual": cf,
    })
    committed_type = apply["action"]["type"]
    rec["commit_matches_decision"] = (committed_type == "wait" if decision["action"] == "wait" else
                                      committed_type in ("move", "wait") if decision["action"] == "move" else committed_type == decision["action"])
    rec["legal_at_snapshot"] = cf["engine_valid"]
    rec["legal_at_apply"] = effective in (APPLY_APPLIED, APPLY_INTENTIONAL_WAIT)
    rec["world_changed"] = rec["apply_world_hash"] != rec["snapshot_world_hash"]
    rec["relevant_state_changed"] = rec["apply_relevant_hash"] != rec["snapshot_relevant_hash"]
    rec["staleness_class"] = classify_staleness(world_changed=rec["world_changed"], relevant_changed=rec["relevant_state_changed"],
                                                engine_valid_at_snapshot=cf["engine_valid"], legal_at_apply=rec["legal_at_apply"])
    rec["consequence"] = classify_consequence(rec, cf) if decision["action"] != "move" else ("ACCEPTED_UNCHANGED" if rec["legal_at_apply"] else "REFUSED")
    return rec


def _assemble(scenario, policy_id, trial_id, runtime, call_order, n_boundaries, parity, started, finished, pre_world, harness_world,
              final_world, events, call_log, request_log, get_log, apply_log, tick_log, sched: _Scheduler, timed_out) -> dict[str, Any]:
    requests = {a: _requests(a, get_log, request_log, call_log, sched) for a in ACTORS}
    boundaries_out = []
    for b in sched.boundaries:
        applies = {a: next((x for x in apply_log if x["actor_id"] == a and x["commit_boundary"] == b.index), None) for a in ACTORS}
        reqs = {a: next((r for r in requests[a] if r["boundary"] == b.index and r["launch_kind"] == "boundary"), None) for a in ACTORS}
        records = {a: _decision_record(scenario, trial_id, b, a, reqs[a], applies[a], b.end_world) for a in ACTORS}
        complete = all(not r.get("incomplete") for r in records.values())
        out: dict[str, Any] = {"boundary": b.index, "decision_boundary_id": f"{trial_id}:b{b.index}", "complete": complete,
                               "open_tick": b.open_tick, "records": records}
        if complete:
            apply_order = sorted(ACTORS, key=lambda a: records[a]["apply_seq"])
            ready_order = sorted(ACTORS, key=lambda a: records[a]["response_ready_wall"])
            same_tick = records[ACTOR_A]["commit_tick"] == records[ACTOR_B]["commit_tick"]
            launch_ticks = {a: b.launches[a]["tick"] for a in ACTORS}
            t0 = min(b.launches[a]["wall"] for a in ACTORS)
            commit_walls = sorted(records[a]["commit_wall"] for a in ACTORS)
            ready_walls = sorted(records[a]["response_ready_wall"] for a in ACTORS)
            contested = [(e["type"], e.get("actor_id")) for x in applies.values() for e in x["events"] if e["type"] in CONTESTED]
            for rank, a in enumerate(apply_order, start=1):
                records[a]["commit_rank"] = rank
            for rank, a in enumerate(ready_order, start=1):
                records[a]["response_ready_rank"] = rank
            # The tick-resolution ready order the loop acted on. Under P0 a
            # decision commits on the first tick its readiness is seen, so
            # the commit ticks are that order; a barrier records the tick it
            # first saw each decision ready.
            first_seen = b.ready_seen_tick if sched.barrier else {a: records[a]["commit_tick"] for a in ACTORS}
            tie = first_seen.get(ACTOR_A) == first_seen.get(ACTOR_B)
            loop_ready_order = None if tie else sorted(ACTORS, key=lambda a: first_seen[a])
            # P0's rule is M6G.1's: the ready order, same-tick readiness Black first
            expected = tuple(sched._order(b)) if sched.barrier else ((ACTOR_A, ACTOR_B) if same_tick else tuple(ready_order))
            out.update({
                "first_commit_actor": apply_order[0], "second_commit_actor": apply_order[1], "apply_order": apply_order,
                "ready_order": ready_order, "loop_ready_order": loop_ready_order, "ready_tie_same_tick": tie,
                "first_ready_is_first_commit": ready_order[0] == apply_order[0],
                "policy_order": list(expected), "policy_order_respected": list(expected) == apply_order,
                "same_tick_commit": same_tick, "release_tick": b.release_tick, "hold_ticks": len(b.hold_ticks),
                "synchronized_snapshot": launch_ticks[ACTOR_A] == launch_ticks[ACTOR_B]
                and records[ACTOR_A]["snapshot_world_hash"] == records[ACTOR_B]["snapshot_world_hash"],
                "launch_ticks": launch_ticks,
                "contested_effects": contested,
                "world_hash_before": records[apply_order[0]]["snapshot_world_hash"],
                "world_hash_after_first_commit": json_hash(applies[apply_order[0]]["world_after"]),
                "world_hash_after_pair": json_hash(b.end_world) if b.end_world else None,
                "effect_state_hash_after_pair": effect_state_hash(b.end_world) if b.end_world else None,
                "event_multiset_hash_pair": json_hash(sorted(json.dumps(e, sort_keys=True, default=str) for x in applies.values() for e in x["events"])),
                "first_action_cost": records[apply_order[0]]["time_cost"],
                "second_action_cost": records[apply_order[1]]["time_cost"],
                "first_result": f"{records[apply_order[0]]['apply_outcome']}/{records[apply_order[0]]['consequence']}",
                "second_result": f"{records[apply_order[1]]['apply_outcome']}/{records[apply_order[1]]['consequence']}",
                "wall_ms": {
                    "to_first_ready": round((ready_walls[0] - t0) * 1000.0, 1), "to_both_ready": round((ready_walls[1] - t0) * 1000.0, 1),
                    "to_first_commit": round((commit_walls[0] - t0) * 1000.0, 1), "to_pair_commit": round((commit_walls[1] - t0) * 1000.0, 1),
                },
                "game_time_at_commits": [records[a]["apply_game_clock"]["mission_time_remaining"] for a in apply_order],
            })
            if b.index == 0:
                committed0 = {a: records[a]["commit_tick"] for a in ACTORS}
                out.update(_commit_schedule([x for x in apply_log if x["tick"] <= max(committed0.values())], committed0))
        boundaries_out.append(out)

    all_requests = [r for a in ACTORS for r in requests[a]]
    windows = [(r["request_start_wall"], r["ready_wall"]) for r in all_requests if r["ready"]]
    advanced = [t for t in tick_log if t["clock_after"] < t["clock_before"]]
    hold_ticks = {t for b in sched.boundaries for t in b.hold_ticks}
    last_commit_tick = max((c["tick"] for b in sched.boundaries for c in b.commits.values()), default=None)
    b0 = sched.boundaries[0]
    b0_ready = [r["ready_wall"] for r in all_requests if r["boundary"] == 0 and r["launch_kind"] == "boundary" and r["ready"]]
    b0_pair_tick = b0.complete_tick
    lifecycle = {
        "requests_launched": len(all_requests),
        "boundary_requests": sum(r["launch_kind"] == "boundary" for r in all_requests),
        "extra_requests": sum(r["launch_kind"] == "extra" for r in all_requests),
        "hp09_violations": sum(bool(r["hp09_violation"]) for r in all_requests),
        "responses_ready": sum(r["ready"] for r in all_requests),
        "responses_consumed": sum(r["consumed"] for r in all_requests),
        "unused_responses": sum(r["ready"] and not r["consumed"] for r in all_requests),
        "in_flight_at_end": sum(not r["ready"] for r in all_requests),
        "provider_calls": sum(r["model_calls"] for r in all_requests),
        "unused_calls": sum(r["model_calls"] for r in all_requests if not r["consumed"]),
        "unused_input_tokens": sum(r["input_tokens"] for r in all_requests if not r["consumed"]),
        "unused_output_tokens": sum(r["output_tokens"] for r in all_requests if not r["consumed"]),
        "attempts": _count(e.get("attempt") or "HELD_BY_BARRIER" for e in get_log),
        "early_returns_held": sum(bool(e.get("early_return_held")) for e in get_log),
        "held_polls": sum(e["held"] for e in get_log),
    }
    freeze = {
        "ticks_advanced_during_inference": sum(1 for t in advanced if any(s <= t["wall_start"] and t["wall_end"] <= e for s, e in windows)),
        "ticks_advanced_while_a_decision_was_held": sum(1 for t in advanced if t["tick"] in hold_ticks),
        "game_time_advanced_while_held_s": round(sum(t["clock_before"] - t["clock_after"] for t in advanced if t["tick"] in hold_ticks), 6),
        "hold_ticks": len(hold_ticks),
        "handoff_ticks_before_boundary0_pair": sum(1 for t in advanced if b0_pair_tick is not None and t["tick"] < b0_pair_tick
                                                   and b0_ready and t["wall_end"] >= max(b0_ready)),
        "game_time_before_boundary0_pair_s": round(sum(t["clock_before"] - t["clock_after"] for t in advanced
                                                       if b0_pair_tick is not None and t["tick"] < b0_pair_tick), 6),
        "align_ticks": len(sched.align_hold_ticks),
        "game_time_advanced_while_aligning_s": round(sum(t["clock_before"] - t["clock_after"] for t in advanced if t["tick"] in sched.align_hold_ticks), 6),
    }
    return {
        "trial_id": trial_id,
        "commitment_policy": policy_id,
        "scenario_id": scenario.family,
        "runtime": runtime,
        "decision_call_order": call_order,
        "boundaries_planned": n_boundaries,
        "p4_parity": parity if policy_id.startswith("P4") else None,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "timed_out": timed_out,
        "complete": not timed_out and all(b["complete"] for b in boundaries_out) and len(boundaries_out) == n_boundaries,
        "pre_action_world_hash": json_hash(pre_world),
        "pre_action_matches_harness_snapshot": json_hash(pre_world) == json_hash(harness_world),
        "world_state_hash_after": json_hash(final_world),
        "effect_state_hash_after": effect_state_hash(final_world),
        "event_multiset_hash": json_hash(sorted(json.dumps(e, sort_keys=True, default=str) for e in events)),
        "ticks": len(tick_log),
        "last_commit_tick": last_commit_tick,
        "boundaries": boundaries_out,
        "requests": requests,
        "lifecycle": lifecycle,
        "freeze_audit": freeze,
        "lifecycle_events": [
            {k: e.get(k) for k in ("tick", "actor_id", "boundary", "prior_state", "attempt", "launched", "launch_kind", "hp09_violation",
                                   "early_return_held", "commit")}
            for e in get_log if not e["held"] and (e["launched"] or e["attempt"] in (ATTEMPT_ABSORBED, ATTEMPT_THROTTLED)
                                                   or e.get("early_return_held") or e.get("commit"))
        ],
    }


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


# -- scripted model stand-ins (zero network; tests only) --------------------------------------


def scripted_sequence_native_client(steps: list[tuple[dict[str, Any], float, threading.Event | None]]):
    """A Native model client answering call i with steps[i] = (raw, delay,
    gate); after the last step it keeps answering the last raw."""
    from src.evaluation.real_loop_probe import _wait
    from src.llm.clients.mock_client import MockModelClient

    class _Sequence(MockModelClient):
        def __init__(self) -> None:
            super().__init__(response=[dict(s[0]) for s in steps])
            self._i = 0

        def generate_action(self, *args, **kwargs):
            raw, delay, gate = steps[min(self._i, len(steps) - 1)]
            self._i += 1
            _wait(delay, gate)
            if not self.responses:
                self.responses.append(dict(raw))
            return super().generate_action(*args, **kwargs)

    return _Sequence()


def scripted_sequence_strands_invoke(steps: list[tuple[dict[str, Any], float, threading.Event | None]]):
    """The Strands counterpart: invocation i calls choose_action with steps[i]."""
    from src.agents.strands_agent import _StrandsInvocation
    from src.evaluation.real_loop_probe import _wait

    state = {"i": 0}

    def invoke(model_payload, repair_instruction=None):
        raw, delay, gate = steps[min(state["i"], len(steps) - 1)]
        state["i"] += 1
        _wait(delay, gate)
        return _StrandsInvocation(proposal=dict(raw), proposal_count=1, stop_reason="tool_use", cycle_count=1, tool_call_count=1,
                                  input_tokens=64, output_tokens=8, latency_ms=delay * 1000.0, error_type=None, error_message=None)

    return invoke
