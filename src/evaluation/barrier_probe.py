"""M6G.3: the PRODUCTION loop observed from outside.

`run_production_trial` builds a real Game with two LLM actors
(`real_loop_probe.build_game`) under the production commit policy (the
promoted P1 barrier by default, or the legacy "answer_order" loop for the
pre-promotion reference) and steps the real `Game._step`. Everything here is
a pass-through observer: nothing is held, reordered, frozen or re-decided by
the probe. It records

- every request: launch tick, `decision_boundary_id` (the Game's own), the
  actor's prior lifecycle state, ready wall time, the controller's
  `response_ready_at`, hand-over tick, commit tick and rank, consumed /
  cancelled / unused / beyond the trial horizon, calls and tokens;
- every apply: actor, action, dt, world before/after, events, cooldowns;
- every tick: game clock before/after, whether a decision was outstanding
  (in flight, ready but not handed over, or held by the barrier), the
  barrier's release record.

A trial ends once each actor has committed `rounds` decisions.

Test-only knob (scripted stand-ins, zero network): `ready_schedule` -
groups of (actor, request number) whose answers are made ready together at
the end of one tick, group after group, `gap_ticks` ticks after the previous
group was handed over, and (with `absorb`) only once the actor's controller
has stored the observation change it saw in flight - which pins the ready
order, ties, and the M6G.1 race timing (gap 0, no absorb) independently of
machine load. Live runs pass none of it.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from src.actions import ActionType
from src.evaluation.commitment_policy import counterfactual_on_state
from src.evaluation.commitment_probe import CONTESTED, observation_digest, unpatched_copy
from src.evaluation.conflict_scenarios import ConflictScenario
from src.evaluation.decision_scheduling import DECISION_TICK_DT, _decision_key, _is_listed, classify_apply_outcome
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS, event_log_hash, fresh_scenario_state, game_clock, json_hash, world_state_snapshot
from src.evaluation.real_loop_probe import _action_dict, _controller, _final_raw, _TimingClient, _wrap_request, _wrap_strands_invoke, build_game
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
from src.game import DEFAULT_COMMIT_POLICY
from src.llm_controller import enumerate_legal_actions, validate_model_output

IDLE, IN_FLIGHT, READY_UNCOMMITTED, HELD, COMMITTED = "IDLE", "REQUEST_IN_FLIGHT", "RESPONSE_READY_UNCOMMITTED", "HELD_BY_BARRIER", "COMMITTED"


def run_production_trial(
    scenario: ConflictScenario,
    *,
    trial_id: str,
    runtime: str = "native_tool_like",
    provider: str = "bedrock",
    model_id: str,
    region: str | None = None,
    decision_call_order: str = "black_first",
    commit_policy: str = DEFAULT_COMMIT_POLICY,
    rounds: int = 1,
    client_overrides: dict[str, Any] | None = None,
    strands_invoke_overrides: dict[str, Any] | None = None,
    gates: dict[tuple[str, int], threading.Event] | None = None,
    ready_schedule: list[list[tuple[str, int]]] | None = None,
    gap_ticks: int = 1,
    absorb: bool = True,
    max_wall_seconds: float = 150.0,
    tick_sleep_seconds: float = 0.005,
    join_inflight_seconds: float = 90.0,
) -> dict[str, Any]:
    game = build_game(scenario, runtime=runtime, provider=provider, model_id=model_id, region=region,
                      decision_call_order=decision_call_order, commit_policy=commit_policy)
    adapters = {ACTOR_A: game.black_llm_controller, ACTOR_B: game.bot_controller}
    ctrls = {a: _controller(ad) for a, ad in adapters.items()}
    barrier = game._barrier_active()
    pre_world = world_state_snapshot(game.state)
    harness_world = world_state_snapshot(fresh_scenario_state(scenario.setup_fn))
    events_start = len(game.state.event_log)

    call_log: list[dict[str, Any]] = []
    run_log: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    applies: list[dict[str, Any]] = []
    ticks: list[dict[str, Any]] = []
    requests: dict[str, list[dict[str, Any]]] = {a: [] for a in ACTORS}
    commits: dict[str, int] = {a: 0 for a in ACTORS}
    last_poll: dict[str, tuple[str, str | None]] = {}
    tick = {"n": 0}
    sched = {"next": 0, "last_handover_tick": None, "released": []}
    legacy_boundary = {"id": 0, "open": False}
    horizon_reached: dict[str, int | None] = {a: None for a in ACTORS}

    def gate_for(a: str, index: int) -> threading.Event | None:
        return (gates or {}).get((a, index))

    def outstanding(a: str) -> bool:
        return bool(ctrls[a]._thinking_in_progress or ctrls[a].response_ready_uncommitted or a in game._held_decisions)

    def prior_state(a: str) -> str:
        if a in game._held_decisions:
            return HELD
        if ctrls[a]._thinking_in_progress:
            return IN_FLIGHT
        if ctrls[a].response_ready_uncommitted:
            return READY_UNCOMMITTED
        return COMMITTED if commits[a] else IDLE

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
        _wrap_request(ctrl, a, run_log)

        def make_get(adapter=adapter, ctrl=ctrl, a=a):
            original = adapter.get_action

            def wrapped(observation):
                t = tick["n"]
                prior = prior_state(a)
                digest = observation_digest(observation)
                clock = game_clock(game.state)
                requested_before = ctrl.decisions_requested
                cancelled_before = ctrl.ready_responses_cancelled
                ready_at_before = ctrl.response_ready_at
                wall = time.time()
                action = original(observation)
                launched = ctrl.decisions_requested > requested_before
                handed_over = bool(ctrl.last_call_committed_ready_response)
                last_poll[a] = (digest, ctrl._last_observation_digest)
                entry = {"tick": t, "actor_id": a, "prior_state": prior, "launched": launched, "handed_over": handed_over,
                         "returned": _action_dict(action), "wall": wall}

                def mark(req: dict[str, Any], ready_at: float | None) -> None:
                    req.update({"handed_over_tick": t, "handed_over_wall": wall, "response_ready_at": ready_at,
                                "handed_over_action": _action_dict(action)})
                    entry["request_index"] = req["request_index"]

                pending = [r for r in requests[a] if r["handed_over_tick"] is None and not r.get("cancelled")]
                # a hand-over of an earlier ready request - every path except an answer inside the launching join
                if handed_over and not (launched and prior != READY_UNCOMMITTED) and pending:
                    mark(pending[-1], ready_at_before if launched else ctrl.response_ready_at)
                if ctrl.ready_responses_cancelled > cancelled_before and pending:
                    pending[-1].update({"cancelled": True, "cancel_reason": ctrl.last_ready_response_cancel_reason, "cancelled_tick": t})
                if launched:
                    # get_action never mutates the world, so the state right after the call is the snapshot
                    world = world_state_snapshot(game.state)
                    state_copy = unpatched_copy(game.state)
                    index = len(requests[a]) + 1
                    requests[a].append({
                        "actor_id": a, "request_index": index, "launch_tick": t, "launch_wall": wall, "prior_state": prior,
                        "hp09_violation": prior in (READY_UNCOMMITTED, HELD), "beyond_horizon": horizon_reached[a] is not None,
                        "decision_boundary_id": None, "world": world, "state_copy": state_copy, "observation": observation,
                        "legal": enumerate_legal_actions(observation), "clock_at_launch": clock,
                        "handed_over_tick": None, "committed_tick": None, "commit_seq": None,
                    })
                    entry["request_index"] = index
                    if handed_over and prior != READY_UNCOMMITTED:   # answered inside the launching call's join
                        mark(requests[a][-1], ctrl.response_ready_at)
                calls.append(entry)
                return action

            return wrapped

        adapter.get_action = make_get()

    original_apply = game.state.apply_action
    original_update = game.state.update

    def wrapped_apply(a, action, dt):
        # snapshots only where a commit can happen (a handed-over decision is pending) or the world can change
        pending = any(r["handed_over_tick"] is not None and r["committed_tick"] is None for r in requests[a])
        capture = pending or action.type is not ActionType.WAIT
        before = world_state_snapshot(game.state) if capture else None
        events_before = len(game.state.event_log)
        cooldown_before = game.state.actors[a].action_cooldown
        clock = game_clock(game.state)
        original_apply(a, action, dt)
        applies.append({"tick": tick["n"], "seq": len(applies), "actor_id": a, "action": _action_dict(action), "dt": dt, "wall": time.time(),
                        "clock_before": clock, "world_before": before, "world_after": world_state_snapshot(game.state) if capture else None,
                        "events": game.state.event_log[events_before:], "cooldown_before": cooldown_before,
                        "cooldown_after": game.state.actors[a].action_cooldown})

    def release_group(group: list[tuple[str, int]]) -> None:
        for a, index in group:
            g = gate_for(a, index)
            if g is not None:
                g.set()
            thread = getattr(ctrls[a], "_thinking_thread", None)
            if thread is not None and thread.is_alive():
                thread.join()

    def run_schedule(t: int) -> None:
        if not ready_schedule or sched["next"] >= len(ready_schedule):
            return
        group = ready_schedule[sched["next"]]
        if not all(len(requests[a]) >= index for a, index in group):
            return
        if sched["next"] > 0:
            previous = ready_schedule[sched["next"] - 1]
            handed = [requests[a][index - 1]["handed_over_tick"] for a, index in previous]
            if any(h is None for h in handed) or t < max(handed) + gap_ticks:
                return
        if absorb and any(last_poll.get(a, (None, None))[0] != last_poll.get(a, (None, None))[1] for a, _ in group
                          if last_poll.get(a) is not None and ctrls[a]._thinking_in_progress):
            return
        release_group(group)
        sched["released"].append({"group": [list(x) for x in group], "tick": t})
        sched["next"] += 1

    def wrapped_update(dt):
        t = tick["n"]
        waiting = {a: outstanding(a) for a in ACTORS}
        original_update(dt)
        # boundary ids: the Game's own under the barrier; the same rule applied from outside for the legacy loop
        for a in ACTORS:
            for r in requests[a]:
                if r["decision_boundary_id"] is None:
                    if barrier:
                        r["decision_boundary_id"] = game.request_boundary_ids.get(a)
                    else:
                        if not legacy_boundary["open"]:
                            legacy_boundary["id"] += 1
                            legacy_boundary["open"] = True
                        r["decision_boundary_id"] = legacy_boundary["id"]
        tick_applies = [x for x in applies if x["tick"] == t]
        committed_now = False
        for a in ACTORS:
            for r in requests[a]:
                if r["handed_over_tick"] is not None and r["committed_tick"] is None and a not in game._held_decisions:
                    mine = next((x for x in tick_applies if x["actor_id"] == a), None)
                    r.update({"committed_tick": t, "commit_seq": mine["seq"] if mine else None})
                    commits[a] += 1
                    committed_now = True
                    if commits[a] >= rounds and horizon_reached[a] is None:
                        horizon_reached[a] = t
        if committed_now:
            end_worlds[t] = world_state_snapshot(game.state)
        if not barrier and not any(outstanding(a) for a in ACTORS):
            legacy_boundary["open"] = False
        run_schedule(t)
        ticks.append({"tick": t, "dt": dt, "waiting": waiting, "held_after": sorted(game._held_decisions),
                      "release": dict(game.last_barrier_release) if game.last_barrier_release and
                      game.last_barrier_release is not ticks_state["last_release"] else None})
        ticks_state["last_release"] = game.last_barrier_release

    ticks_state = {"last_release": None}
    end_worlds: dict[int, dict[str, Any]] = {}
    game.state.apply_action = wrapped_apply
    game.state.update = wrapped_update

    started = datetime.now(timezone.utc).isoformat()
    t_start = time.monotonic()
    while time.monotonic() - t_start < max_wall_seconds:
        tick["n"] += 1
        clock_before, wall_start = game.state.mission_time_remaining, time.time()
        game._step(DECISION_TICK_DT)
        ticks[-1].update({"clock_before": clock_before, "clock_after": game.state.mission_time_remaining,
                          "wall_start": wall_start, "wall_end": time.time()})
        if all(commits[a] >= rounds for a in ACTORS):
            break
        time.sleep(tick_sleep_seconds)
    timed_out = not all(commits[a] >= rounds for a in ACTORS)
    for key in list((gates or {})):
        gates[key].set()
    finished = datetime.now(timezone.utc).isoformat()
    final_world = world_state_snapshot(game.state)
    events = game.state.event_log[events_start:]
    events_hash = event_log_hash(game.state, events_start)
    deadline = time.monotonic() + join_inflight_seconds
    for a in ACTORS:
        thread = getattr(ctrls[a], "_thinking_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(max(0.0, deadline - time.monotonic()))
    return _assemble(scenario, trial_id, runtime, decision_call_order, commit_policy, barrier, rounds, started, finished, pre_world,
                     harness_world, final_world, events, events_hash, call_log, run_log, calls, applies, ticks, requests, end_worlds,
                     sched, timed_out)


# -- assembly -----------------------------------------------------------------------------


def _request_rows(a: str, reqs: list[dict[str, Any]], run_log: list[dict[str, Any]], call_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = sorted((r for r in run_log if r["actor_id"] == a), key=lambda r: r["request_start_wall"])
    calls = sorted((c for c in call_log if c["actor_id"] == a), key=lambda c: c["call_start_wall"])
    out = []
    for i, req in enumerate(reqs):
        run = runs[i] if i < len(runs) else None
        window_end = run["ready_wall"] if run else float("inf")
        mine = [c for c in calls if (run is None or c["call_start_wall"] >= run["request_start_wall"] - 1e-3) and c["call_start_wall"] <= window_end]
        calls = [c for c in calls if c not in mine]
        consumed = req["committed_tick"] is not None
        cancelled = bool(req.get("cancelled"))
        ready = run is not None
        out.append({**{k: v for k, v in req.items() if k not in ("world", "state_copy", "observation", "legal")},
                    "request_start_wall": run["request_start_wall"] if run else None, "ready_wall": run["ready_wall"] if run else None,
                    "request_to_ready_ms": run["request_to_ready_ms"] if run else None, "ready": ready, "consumed": consumed,
                    "cancelled": cancelled,
                    "unused": ready and not consumed and not cancelled and not req["beyond_horizon"],
                    "in_flight_at_end": not ready,
                    "model_calls": len(mine), "input_tokens": sum(c["input_tokens"] or 0 for c in mine),
                    "output_tokens": sum(c["output_tokens"] or 0 for c in mine),
                    "provider_error_types": [c["error_type"] for c in mine if c["error_type"]],
                    "payloads": [c["payload"] for c in mine],
                    "provider_request_hash": mine[0]["provider_request_hash"] if mine else None,
                    "_req": req})
    return out


def _record(scenario, a: str, row: dict[str, Any], apply: dict[str, Any]) -> dict[str, Any]:
    req = row["_req"]
    raw, first_pass_valid, error = _final_raw([{"payload": p} for p in row["payloads"]], req["observation"], req["legal"])
    decision = (_decision_key(validate_model_output(req["observation"], raw, req["legal"]).validated_action) if first_pass_valid
                else _decision_key(None))
    before, after = apply["world_before"], apply["world_after"]
    types = [e["type"] for e in apply["events"] if e.get("actor_id") == a]
    outcome = classify_apply_outcome(decision["action"], json_hash(before) != json_hash(after), types)
    if decision["action"] == "move" and apply["dt"] == 0.0:
        outcome = "COMMITTED_FROZEN"
    effective = APPLY_APPLIED if outcome == "COMMITTED_FROZEN" else outcome
    cf = counterfactual_on_state(req["state_copy"], a, raw, family=scenario.family)
    rec: dict[str, Any] = {
        "actor_id": a, "request_index": row["request_index"], "decision_boundary_id": row["decision_boundary_id"],
        "raw_model_output": raw, "first_pass_valid": first_pass_valid, "validation_error": error,
        "decision": decision, "decision_hash": json_hash(decision),
        "snapshot_world_hash": json_hash(req["world"]), "snapshot_relevant_hash": relevant_state_hash(req["world"], a, decision),
        "apply_world_hash": json_hash(before), "apply_relevant_hash": relevant_state_hash(before, a, decision),
        "committed_action": apply["action"], "apply_dt": apply["dt"], "apply_outcome": outcome, "effective_outcome": effective,
        "apply_event_types": types,
        "time_cost": round(apply["cooldown_after"], 6) if apply["cooldown_before"] == 0 and apply["cooldown_after"] > 0 else 0.0,
        "target_state_after": own_effect_view(after, a, decision), "self_state_after": actor_self_state(after, a),
        "target_state_before_apply": target_state(before, decision["target_id"]),
        "counterfactual": cf, "listed_at_snapshot": _is_listed(decision, req["legal"]),
        "game_time_before_wait": req["clock_at_launch"]["mission_time_remaining"],
        "game_time_after_wait": apply["clock_before"]["mission_time_remaining"],
    }
    rec["game_time_waited_s"] = round(rec["game_time_before_wait"] - rec["game_time_after_wait"], 6)
    rec["legal_at_snapshot"] = cf["engine_valid"]
    rec["legal_at_apply"] = effective in (APPLY_APPLIED, APPLY_INTENTIONAL_WAIT)
    rec["world_changed"] = rec["apply_world_hash"] != rec["snapshot_world_hash"]
    rec["relevant_state_changed"] = rec["apply_relevant_hash"] != rec["snapshot_relevant_hash"]
    rec["staleness_class"] = classify_staleness(world_changed=rec["world_changed"], relevant_changed=rec["relevant_state_changed"],
                                                engine_valid_at_snapshot=cf["engine_valid"], legal_at_apply=rec["legal_at_apply"])
    rec["consequence"] = classify_consequence(rec, cf) if decision["action"] != "move" else ("ACCEPTED_UNCHANGED" if rec["legal_at_apply"] else "REFUSED")
    return rec


def _assemble(scenario, trial_id, runtime, call_order, commit_policy, barrier, rounds, started, finished, pre_world, harness_world,
              final_world, events, events_hash, call_log, run_log, calls, applies, ticks, requests, end_worlds, sched, timed_out) -> dict[str, Any]:
    rows = {a: _request_rows(a, requests[a], run_log, call_log) for a in ACTORS}
    by_seq = {x["seq"]: x for x in applies}
    for a in ACTORS:
        for r in rows[a]:
            if r["consumed"] and r["commit_seq"] is not None and by_seq[r["commit_seq"]]["world_before"] is not None:
                r["record"] = _record(scenario, a, r, by_seq[r["commit_seq"]])
                r["commit_wall"] = by_seq[r["commit_seq"]]["wall"]
    boundaries = []
    ids = sorted({r["decision_boundary_id"] for a in ACTORS for r in rows[a] if r["decision_boundary_id"] is not None})
    for bid in ids:
        mine = [r for a in ACTORS for r in rows[a] if r["decision_boundary_id"] == bid]
        committed = [r for r in mine if r["consumed"] and "record" in r]
        out: dict[str, Any] = {"decision_boundary_id": bid, "requests": {a: sum(r["actor_id"] == a for r in mine) for a in ACTORS},
                               "committed": {a: sum(r["actor_id"] == a for r in committed) for a in ACTORS},
                               "decisions": len(committed), "pair": len({r["actor_id"] for r in committed}) == 2 and len(committed) == 2}
        if committed:
            ready = sorted(committed, key=lambda r: (r["response_ready_at"] is None, r["response_ready_at"] or 0.0, ACTORS.index(r["actor_id"])))
            commit = sorted(committed, key=lambda r: r["commit_seq"])
            t0 = min(r["launch_wall"] for r in mine)
            for rank, r in enumerate(ready, start=1):
                r["ready_rank"] = rank
            for rank, r in enumerate(commit, start=1):
                r["commit_rank"] = rank
            first_apply, last_apply = by_seq[commit[0]["commit_seq"]], by_seq[commit[-1]["commit_seq"]]
            end_world = end_worlds.get(commit[-1]["committed_tick"])
            out.update({
                "ready_order": [r["actor_id"] for r in ready], "commit_order": [r["actor_id"] for r in commit],
                "first_ready_is_first_commit": ready[0]["actor_id"] == commit[0]["actor_id"],
                "ready_wall_order": [r["actor_id"] for r in sorted(committed, key=lambda r: r["ready_wall"] or 0.0)],
                "exact_ready_tie": len(committed) == 2 and committed[0]["response_ready_at"] == committed[1]["response_ready_at"],
                "handed_over_same_tick": len({r["handed_over_tick"] for r in committed}) == 1,
                "same_tick_commit": len({r["committed_tick"] for r in committed}) == 1,
                "launch_ticks": {r["actor_id"]: r["launch_tick"] for r in committed},
                "committed_ticks": {r["actor_id"]: r["committed_tick"] for r in committed},
                "synchronized_snapshot": len({r["record"]["snapshot_world_hash"] for r in committed}) == 1,
                "records": {r["actor_id"]: r["record"] for r in committed},
                "world_hash_before": commit[0]["record"]["snapshot_world_hash"],
                "world_hash_after_first_commit": json_hash(first_apply["world_after"]),
                "world_hash_after_pair": json_hash(end_world) if end_world else None,
                "effect_state_hash_after_pair": effect_state_hash(end_world) if end_world else None,
                "event_log_hash_pair": json_hash([json.dumps(e, sort_keys=True, default=str) for r in commit for e in by_seq[r["commit_seq"]]["events"]]),
                "contested_effects": [(e["type"], e.get("actor_id")) for r in commit for e in by_seq[r["commit_seq"]]["events"] if e["type"] in CONTESTED],
                "first_action_cost": commit[0]["record"]["time_cost"], "second_action_cost": commit[-1]["record"]["time_cost"] if len(commit) > 1 else None,
                "first_result": f"{commit[0]['record']['apply_outcome']}/{commit[0]['record']['consequence']}",
                "second_result": f"{commit[-1]['record']['apply_outcome']}/{commit[-1]['record']['consequence']}" if len(commit) > 1 else None,
                "game_time_waited_s": {r["actor_id"]: r["record"]["game_time_waited_s"] for r in committed},
                "first_commit_on_unchanged_snapshot": not commit[0]["record"]["world_changed"],
                "wall_ms": {
                    "to_first_ready": round(((ready[0]["ready_wall"] or t0) - t0) * 1000.0, 1),
                    "to_both_ready": round((max(r["ready_wall"] or t0 for r in committed) - t0) * 1000.0, 1),
                    "to_first_commit": round((first_apply["wall"] - t0) * 1000.0, 1),
                    "to_pair_commit": round((last_apply["wall"] - t0) * 1000.0, 1),
                },
            })
        boundaries.append(out)

    all_rows = [r for a in ACTORS for r in rows[a]]
    horizon = [r for r in all_rows if not r["beyond_horizon"]]
    waiting_live = [t for t in ticks if any(t["waiting"].values()) and t["dt"] > 0]
    lifecycle = {
        "requests_launched": len(horizon), "beyond_horizon_requests": len(all_rows) - len(horizon),
        "duplicate_requests": sum(bool(r["hp09_violation"]) for r in horizon),
        "responses_ready": sum(r["ready"] for r in horizon), "responses_consumed": sum(r["consumed"] for r in horizon),
        "responses_cancelled": sum(r["cancelled"] for r in horizon), "unused_responses": sum(r["unused"] for r in horizon),
        "in_flight_at_end": sum(r["in_flight_at_end"] for r in horizon),
        "provider_calls": sum(r["model_calls"] for r in all_rows),
        "unused_calls": sum(r["model_calls"] for r in horizon if r["unused"]),
        "unused_input_tokens": sum(r["input_tokens"] for r in horizon if r["unused"]),
        "unused_output_tokens": sum(r["output_tokens"] for r in horizon if r["unused"]),
    }
    freeze = {
        "waiting_ticks": sum(1 for t in ticks if any(t["waiting"].values())),
        "ticks_advanced_while_waiting": len(waiting_live),
        "game_time_advanced_while_waiting_s": round(sum(t["clock_before"] - t["clock_after"] for t in waiting_live), 6),
        "held_ticks": sum(1 for t in ticks if t["held_after"]),
    }
    for a in ACTORS:
        for r in rows[a]:
            r.pop("_req", None)
    return {
        "trial_id": trial_id, "commit_policy": commit_policy, "barrier_active": barrier, "scenario_id": scenario.family,
        "runtime": runtime, "decision_call_order": call_order, "rounds": rounds,
        "started_at_utc": started, "finished_at_utc": finished, "timed_out": timed_out,
        "complete": not timed_out and all(sum(r["consumed"] for r in rows[a]) >= rounds for a in ACTORS),
        "pre_action_world_hash": json_hash(pre_world), "pre_action_matches_harness_snapshot": json_hash(pre_world) == json_hash(harness_world),
        "world_state_hash_after": json_hash(final_world), "effect_state_hash_after": effect_state_hash(final_world),
        "event_multiset_hash": json_hash(sorted(json.dumps(e, sort_keys=True, default=str) for e in events)),
        "event_log_hash": events_hash, "ticks": len(ticks),
        "requests": rows, "boundaries": boundaries, "lifecycle": lifecycle, "freeze_audit": freeze,
        "releases": [t["release"] | {"tick": t["tick"]} for t in ticks if t.get("release")],
        "ready_schedule_released": sched["released"],
        "lifecycle_events": [{k: c.get(k) for k in ("tick", "actor_id", "prior_state", "launched", "handed_over", "request_index")}
                             for c in calls if c["launched"] or c["handed_over"]],
    }


# -- scripted stand-ins -----------------------------------------------------------------------


def scripted_requests(runtime: str, per_actor: dict[str, list[list[dict[str, Any]]]], *, delay: float = 0.02,
                      gated: bool = True, delays: dict[tuple[str, int], float] | None = None,
                      ungated: tuple[tuple[str, int], ...] = ()) -> tuple[dict[str, Any], dict[tuple[str, int], threading.Event]]:
    """Scripted model stand-ins (zero network) answering request i of each
    actor with its payload list (the first call, then repairs), each
    request's first call behind its own gate (except the `ungated` ones)
    and no earlier than its delay (`delays[(actor, i)]`, else `delay`)
    after the call started. Returns (the kwargs for run_production_trial,
    gates)."""
    from src.evaluation.commitment_probe import scripted_sequence_native_client, scripted_sequence_strands_invoke

    gates: dict[tuple[str, int], threading.Event] = {}
    kwargs: dict[str, Any] = {}
    steps: dict[str, list] = {}
    for a, reqs in per_actor.items():
        steps[a] = []
        for i, payloads in enumerate(reqs, start=1):
            g = threading.Event() if gated and (a, i) not in ungated else None
            if g is not None:
                gates[(a, i)] = g
            d = (delays or {}).get((a, i), delay)
            for j, p in enumerate(payloads):
                steps[a].append((p, d, g if j == 0 else None))
    if runtime == "strands":
        kwargs["strands_invoke_overrides"] = {a: scripted_sequence_strands_invoke(s) for a, s in steps.items()}
    else:
        kwargs["client_overrides"] = {a: scripted_sequence_native_client(s) for a, s in steps.items()}
    return kwargs, gates
