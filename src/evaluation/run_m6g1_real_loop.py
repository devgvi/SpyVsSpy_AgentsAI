"""M6G.1 Phases C and E (spec sections 12, 17, 30, 37-39): the production
game loop, observed with `real_loop_probe`.

Live (`--runtime native|strands`): each trial builds a real `Game` with two
LLM actors on Bedrock, places them in L1 (shared locked door: the
stale-but-valid case) or L0 (non-conflicting control), and steps the real
loop until both first decisions are committed - one model call per actor
unless production repair fires. Scenarios alternate trial by trial;
`decision_call_order` alternates by repetition so neither identity always
gets the 50 ms request head start. Every completed pair is then replayed
with zero calls: under the exact tick plan the loop ran (must reproduce
the live world), idealized T0 in the observed order, T0 reversed, T1A and
T1B. Native writes `real_loop_trials.jsonl`; Strands writes
`strands_trials.jsonl` and runs only behind the RP01-RP18 gate. Rows are
appended as trials finish and the runner resumes. Any provider-side
failure stops the run.

Scripted (`--scripted`, zero network): the same production loop for BOTH
runtimes with the model answer replaced by each L scenario's fixed
decision after a fixed delay (slow 0.35 s / fast 0.08 s, both
assignments; the slow answer is also held until the fast commit has been
applied). This carries the stale-and-invalid cases (L3/L4/L5) through
each runtime's own adapter lifecycle without asking a live model to
generate trap actions (spec sections 11, 39) -
`real_loop_scripted_trials.jsonl`.

    python -m src.evaluation.run_m6g1_real_loop --scripted
    caffeinate -i python -m src.evaluation.run_m6g1_real_loop --runtime native
    caffeinate -i python -m src.evaluation.run_m6g1_real_loop --runtime strands
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import sys
import threading
from datetime import datetime, timezone

from src.evaluation.decision_scheduling import MODEL_OUTPUT_ERROR_TYPES
from src.evaluation.duel_snapshot import ACTOR_A, ACTORS
from src.evaluation.m6g_live import append_jsonl, read_jsonl, run_live_preflight, verify_sleep_assertion
from src.evaluation.real_loop_probe import run_real_loop_trial, scripted_native_client, scripted_strands_invoke
from src.evaluation.run_m6g1_phase_b import OUT_DIR, PROBE_MODEL_ID, out_path
from src.evaluation.temporal_semantics import (
    L0_NON_CONFLICTING,
    L1_SHARED_DOOR,
    L_FAMILIES,
    T0,
    T1A,
    T1B,
    build_temporal_scenarios,
    replay_pair,
)

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", PROBE_MODEL_ID)
REPS = 10
SCENARIOS = (L1_SHARED_DOOR, L0_NON_CONFLICTING)
RUNTIMES = {"native": "native_tool_like", "strands": "strands"}
FILES = {"native": "real_loop_trials.jsonl", "strands": "strands_trials.jsonl"}
SCRIPTED_FILE = "real_loop_scripted_trials.jsonl"
SCRIPTED_DELAYS = {"slow": 0.35, "fast": 0.08}
REPLAY_LABELS = ("T0_observed_schedule", "T0_observed_arrival", "T0_reversed_arrival", "T1A", "T1B")


def planned_trials(reps: int = REPS) -> list[tuple[int, str, str]]:
    plan = []
    for rep in range(reps):
        order = "black_first" if rep % 2 == 0 else "white_first"
        families = SCENARIOS if rep % 2 == 0 else tuple(reversed(SCENARIOS))
        for family in families:
            plan.append((rep, family, order))
    return plan


def scripted_plan() -> list[tuple[str, str, str]]:
    """(runtime, family, slow actor): every L scenario, both runtimes, both
    latency assignments."""
    return [(runtime, family, slow) for runtime in RUNTIMES for family in L_FAMILIES for slow in ACTORS]


def provider_failure(row: dict) -> str | None:
    for rec in row["records"].values():
        for err in rec.get("provider_error_types") or []:
            if err not in MODEL_OUTPUT_ERROR_TYPES:
                return f"{rec['actor_id']}: {err}"
    return None


def _strands_render(region: str):
    from src.evaluation.m6g_live import StrandsDecider

    return StrandsDecider(region, MODEL_ID).render


def replays_for(row: dict, scenario, render_fn=None) -> dict[str, dict]:
    """Zero-call replays of one real-loop pair. `T0_observed_schedule`
    re-runs the exact tick plan the live loop ran (reproduces the live
    world bit-for-bit, clocks included); `T0_observed_arrival` is the
    idealized T0 in the observed order (or the same-tick Black-then-White
    commit), `T0_reversed_arrival` the other order; T1A/T1B the barrier."""
    raws = {a: row["records"][a]["raw_model_output"] for a in ACTORS}
    kwargs = {"pair_id": row["trial_id"], "model_id": MODEL_ID}
    if render_fn is not None:
        kwargs["render_fn"] = render_fn
    arrival = tuple(row["apply_order"])
    live_condition = T1A if row["same_tick_commit"] else T0
    plans = (("T0_observed_schedule", T0, None, row["observed_commit_schedule"]),
             ("T0_observed_arrival", live_condition, None if row["same_tick_commit"] else arrival, None),
             ("T0_reversed_arrival", T0, (arrival[1], arrival[0]), None),
             ("T1A", T1A, None, None), ("T1B", T1B, None, None))
    out = {}
    for label, cond, order, schedule in plans:
        r = replay_pair(scenario, raws, cond, arrival_order=order, observed_schedule=schedule, **kwargs)
        r.update({"source": f"m6g1_real_loop_{row['runtime']}", "replay": label, "runtime": row["runtime"], "source_trial_id": row["trial_id"]})
        out[label] = r
    return out


def live_replay_checks(row: dict, replays: dict[str, dict]) -> dict:
    """Live vs replay. The exact schedule must reproduce the live world and
    event multiset; the idealized T0 must match it on effects (clock-free:
    a handoff tick only moves clocks and cooldowns, unless a MOVE was under
    way); the idealized T0 must equal the same-order barrier."""
    same_order = replays["T1A"] if row["apply_order"][0] == ACTOR_A else replays["T1B"]
    exact, ideal = replays["T0_observed_schedule"], replays["T0_observed_arrival"]
    return {
        "live_matches_t0_replay": (exact["world_state_hash_after"], exact["event_multiset_hash"]) == (
            row["world_state_hash_after"], row["event_multiset_hash"]),
        "live_effect_matches_ideal_t0": ideal["effect_state_hash_after"] == row["effect_state_hash_after"],
        "live_world_matches_ideal_t0": ideal["world_state_hash_after"] == row["world_state_hash_after"],
        "t0_replay_matches_same_order_barrier": (ideal["world_state_hash_after"], ideal["event_multiset_hash"]) == (
            same_order["world_state_hash_after"], same_order["event_multiset_hash"]),
        "replay_outcomes": {label: {"world_state_hash_after": r["world_state_hash_after"], "effect_state_hash_after": r["effect_state_hash_after"],
                                    "event_multiset_hash": r["event_multiset_hash"], "apply_order": r["apply_order"],
                                    "contested_effects": r["contested_effects"]} for label, r in replays.items()},
    }


def _print_row(tag: str, row: dict) -> None:
    recs = row["records"]
    print(f"[{tag}] ready={row['ready_order']} apply={row['apply_order']} "
          f"lat={[recs[a].get('response_latency_ms') for a in ACTORS]} "
          f"decisions={[recs[a].get('decision', {}).get('action') for a in ACTORS]} "
          f"stale={[recs[a].get('staleness_class') for a in ACTORS]} frozen={row['game_time_frozen_until_second_apply']} "
          f"handoff={row.get('handoff_ticks_before_second_apply')} live==T0replay={row.get('live_matches_t0_replay')} "
          f"effect==idealT0={row.get('live_effect_matches_ideal_t0')}", flush=True)


def run(runtime: str, region: str, reps: int = REPS) -> int:
    scenarios = build_temporal_scenarios()
    path = out_path(FILES[runtime])
    render = _strands_render(region) if runtime == "strands" else None
    done = {(r["rep"], r["scenario_family"]) for r in read_jsonl(path)}
    for rep, family, call_order in planned_trials(reps):
        if (rep, family) in done:
            continue
        trial_id = f"{'C' if runtime == 'native' else 'E'}:real_loop:{runtime}:{family}:r{rep:02d}"
        row = run_real_loop_trial(scenarios[family], trial_id=trial_id, runtime=RUNTIMES[runtime], provider="bedrock",
                                  model_id=MODEL_ID, region=region, decision_call_order=call_order)
        row.pop("final_world")
        failure = provider_failure(row) if row["complete"] else None
        if failure:
            append_jsonl(out_path("provider_events.jsonl"), {"phase": f"real_loop_{runtime}_abort", "trial_id": trial_id, "failure": failure,
                                                            "at_utc": datetime.now(timezone.utc).isoformat()})
            print(f"[ABORT] provider failure in {trial_id}: {failure}", file=sys.stderr)
            return 2
        row.update({"rep": rep, "phase": "C" if runtime == "native" else "E", "model_id": MODEL_ID, "decision_source": "live_model"})
        if row["complete"]:
            replays = replays_for(row, scenarios[family], render)
            row.update(live_replay_checks(row, replays))
            for r in replays.values():
                append_jsonl(out_path("real_loop_barrier_replays.jsonl"), r)
        append_jsonl(path, row)
        _print_row(f"{runtime} r{rep:02d} {family[:2]} {call_order}", row)
    return 0


def run_scripted(region: str = "us-east-1") -> list[dict]:
    """Zero network: see the module docstring."""
    scenarios = build_temporal_scenarios()
    renders = {"native": None, "strands": _strands_render(region)}
    rows = []
    for runtime, family, slow in scripted_plan():
        s = scenarios[family]
        raws = {a: s.scripted_raw(a) for a in ACTORS}
        delays = {a: SCRIPTED_DELAYS["slow" if a == slow else "fast"] for a in ACTORS}
        # the slow answer is held until the fast commit has been applied, so
        # the intended ready order holds whatever the machine load
        gates = {slow: threading.Event()}
        if runtime == "native":
            overrides = {"client_overrides": {a: scripted_native_client(raws[a], delays[a], gates.get(a)) for a in ACTORS}}
            provider, model_id = "mock", "scripted"
        else:
            # The adapter is built for Bedrock exactly as live; its one SDK
            # invocation is replaced, so no request leaves the process.
            overrides = {"strands_invoke_overrides": {a: scripted_strands_invoke(raws[a], delays[a], gates.get(a)) for a in ACTORS}}
            provider, model_id = "bedrock", MODEL_ID
        row = run_real_loop_trial(s, trial_id=f"S:real_loop:{runtime}:{family}:slow_{slow}", runtime=RUNTIMES[runtime], provider=provider,
                                  model_id=model_id, region=region, gates=gates, release_after_opponent_commit=(slow,), **overrides)
        row.pop("final_world")
        row.update({"phase": "C_scripted" if runtime == "native" else "E_scripted", "decision_source": "scripted", "slow_actor": slow,
                    "scripted_delays_s": delays, "runtime_key": runtime})
        if row["complete"]:
            row.update(live_replay_checks(row, replays_for(row, s, renders[runtime])))
        rows.append(row)
        _print_row(f"scripted {runtime} {family[:2]} slow={slow}", row)
    with open(out_path(SCRIPTED_FILE), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=tuple(RUNTIMES))
    parser.add_argument("--scripted", action="store_true", help="zero-network scripted real-loop trials for both runtimes")
    parser.add_argument("--region", default=None)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.scripted:
        rows = run_scripted()
        bad = [r["trial_id"] for r in rows if not (r["complete"] and r["live_matches_t0_replay"])]
        print(f"[DONE] scripted real loop: {len(rows)} trials, not reproduced by replay: {bad}")
        return 1 if bad else 0
    if args.runtime is None:
        parser.error("--runtime or --scripted is required")

    from src.llm.clients.bedrock_client import resolve_bedrock_region

    if args.runtime == "strands":
        from src.evaluation.run_m6g_strands_replay import runtime_parity_gate

        gate = runtime_parity_gate()
        append_jsonl(out_path("provider_events.jsonl"), {"phase": "runtime_parity_gate_rp01_rp18", "at_utc": datetime.now(timezone.utc).isoformat(), **gate})
        print(f"[RP GATE] {json.dumps(gate)}")
        if not gate["ok"]:
            print("[FATAL] RP01-RP18 gate not clean - refusing to start Strands live calls.", file=sys.stderr)
            return 1
    region = resolve_bedrock_region(args.region)
    if not verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1
    preflight = run_live_preflight(MODEL_ID, region)
    append_jsonl(out_path("provider_events.jsonl"), {"phase": f"preflight_real_loop_{args.runtime}", "at_utc": datetime.now(timezone.utc).isoformat(), **preflight})
    print(f"[PREFLIGHT] {json.dumps(preflight, default=str)}")
    if not preflight["ok"]:
        return 1
    if args.preflight_only:
        return 0
    start = datetime.now(timezone.utc)
    status = run(args.runtime, region, args.reps)
    print(f"[DONE] {args.runtime} status={status} ({start.isoformat()} - {datetime.now(timezone.utc).isoformat()})")
    return status


if __name__ == "__main__":
    sys.exit(main())
