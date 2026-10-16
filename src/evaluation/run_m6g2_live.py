"""M6G.2 Phase E (spec sections 37-44, 49-51): minimal live validation of
commitment policies on the production loop.

Each trial is one real Game with two LLM actors on Bedrock, run under one
commitment policy by `commitment_probe.run_policy_trial`:

    P0  production loop, first decision boundary only
    P1  barrier in P0's order, first boundary only
    P4  barrier, alternating first commit over TWO synchronized boundaries
        (per-trial balance), initial parity counterbalanced across trials

Scenarios: C2 shared door (primary: the only live conflict with no
tool-schema confound) and C0 non-conflict control (Native only). Within a
repetition the three policies run back to back in a rotating order;
scenario order and `decision_call_order` alternate by repetition, and P4's
parity follows (rep // 2) % 2 so it is not tied to the call order.

Native writes native_live_trials.jsonl; Strands (after the Native analysis,
behind RP01-RP18) writes strands_live_trials.jsonl. Every trial's lifecycle
events go to request_lifecycle_events.jsonl. Resumable; any provider-side
failure stops the run.

    caffeinate -i python -m src.evaluation.run_m6g2_live --runtime native
    caffeinate -i python -m src.evaluation.run_m6g2_live --runtime strands
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import sys
from datetime import datetime, timezone

from src.evaluation.commitment_policy import C0, C2, P0, P1, P2, P3, P4, build_policy_scenarios, replay_under_policy
from src.evaluation.commitment_probe import run_policy_trial
from src.evaluation.decision_scheduling import MODEL_OUTPUT_ERROR_TYPES
from src.evaluation.duel_snapshot import ACTORS
from src.evaluation.m6g_live import append_jsonl, read_jsonl, run_live_preflight, verify_sleep_assertion
from src.evaluation.run_m6g2_replay import MODEL_ID, OUT_DIR, out_path

LIVE_POLICIES = (P0, P1, P4)
BOUNDARIES = {P0: 1, P1: 1, P4: 2}
PLAN = {"native": {C2: 8, C0: 4}, "strands": {C2: 8}}
RUNTIMES = {"native": "native_tool_like", "strands": "strands"}
FILES = {"native": "native_live_trials.jsonl", "strands": "strands_live_trials.jsonl"}
LIFECYCLE_FILE = "request_lifecycle_events.jsonl"


def planned_trials(runtime: str) -> list[dict]:
    plan = []
    reps = max(PLAN[runtime].values())
    for rep in range(reps):
        rotation = LIVE_POLICIES[rep % 3:] + LIVE_POLICIES[:rep % 3]
        scenarios = [s for s in (C2, C0) if s in PLAN[runtime] and rep < PLAN[runtime][s]]
        if rep % 2:
            scenarios.reverse()
        for sid in scenarios:
            for policy in rotation:
                plan.append({
                    "trial_id": f"{'E' if runtime == 'strands' else 'D'}:{runtime}:{sid}:{policy}:r{rep:02d}",
                    "rep": rep, "scenario_id": sid, "policy": policy, "boundaries": BOUNDARIES[policy],
                    "decision_call_order": "black_first" if rep % 2 == 0 else "white_first",
                    "p4_parity": (rep // 2) % 2 if policy == P4 else 0,
                })
    return plan


def expected_calls(runtime: str) -> int:
    return sum(2 * t["boundaries"] for t in planned_trials(runtime))


def live_plan() -> dict:
    """live_call_plan.json (spec section 39), frozen before the first call."""
    per_runtime = {}
    for rk in RUNTIMES:
        trials = planned_trials(rk)
        counts: dict[str, dict[str, int]] = {}
        for t in trials:
            counts.setdefault(t["scenario_id"], {}).setdefault(t["policy"], 0)
            counts[t["scenario_id"]][t["policy"]] += 1
        per_runtime[rk] = {"trials": len(trials), "trials_per_scenario_policy": counts, "expected_model_calls": expected_calls(rk),
                           "max_model_calls": 2 * expected_calls(rk), "plan": trials}
    return {
        "frozen_before_live_calls": "Written after the zero-call replay matrix and the network-free harness tests, before any new "
                                    "provider call. Trial list, counts, interleaving, stop rules and the analysis rules below were not "
                                    "changed after seeing live data.",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "provider": "bedrock", "region": "us-east-1",
        "auth": "AWS_PROFILE=gl-primitives (SigV4); m6g_live.run_live_preflight before each phase; caffeinate required",
        "policies_live": {P0: "production loop, boundary 0 only", P1: "barrier in P0's order, boundary 0 only",
                          P4: "barrier, alternating first commit over two synchronized boundaries"},
        "policies_replay_only": {P2: "zero-call replay", P3: "zero-call replay"},
        "scenarios_live": {C2: "shared door, primary live conflict (no tool-schema confound)", C0: "non-conflict control, Native only"},
        "scenarios_replay_only": "C1, C4, C5, C5M, C6, C6M: trap/disarm/search cases stay scripted (trap_type confound, spec section 12)",
        "personalities": "DEFAULT vs DEFAULT", "social": "SOCIAL_OFF (cross-episode social memory off)",
        "interleaving": "within a repetition the three policies run back to back in a rotating order (rep % 3); scenario order and "
                        "decision_call_order alternate with rep % 2; P4 parity = (rep // 2) % 2, so parity and call order are crossed",
        "order_native_then_strands": "Native first; Strands only after the Native analysis and a clean RP01-RP18 gate",
        "stop_rule": "any provider-side error (not a model-output validation error) or an incomplete (timed-out) trial aborts the "
                     "phase; the aborting trial goes to provider_events.jsonl only, so a resumed run repeats it",
        "runtimes": per_runtime,
        "analysis_rules": {
            "latency_commit_coupling": "P(first-ready actor is the first-committed actor), by ready wall time; P0/P1 expected 1.0 except "
                                       "same-tick readiness (Black first); P2/P3/P4 independent of it by definition",
            "actor_priority": "Black/White first-commit counts per policy; P4 also per trial (two boundaries)",
            "world_outcome_sensitivity": "zero-call replay: Black-first vs White-first outcome of the same decision pair "
                                         "(commitment_effect classification); live: the door unlock follows the first commit",
            "barrier_preserves_world_semantics": "replay P0 == P1 on every captured pair, and every live barrier trial's boundary-0 "
                                                 "world equals its zero-call replay under the same policy and ready order",
            "barrier_removes_request_race": "zero HP09 violations and zero unused responses in live barrier trials, the race observed "
                                            "under P0 (live, here or in M6G.1), and the forced-interleaving tests (both runtimes)",
            "request_waste": "provider calls launched vs responses consumed per trial; unused responses and their tokens",
            "wall_clock": "ms from the boundary's first request to the first commit and to the pair commit; game time must not advance "
                          "while a decision is held",
            "runtime_replication": "per policy/scenario, Native vs Strands: same qualitative result -> COMMITMENT_EFFECT_RUNTIME_"
                                   "INDEPENDENT; a rule that holds on one runtime and not the other -> _DEPENDENT; only one runtime "
                                   "observed -> not classified",
            "no_global_score": "latency independence, actor balance, world stability, responsiveness and request waste are reported "
                               "side by side, never combined",
        },
    }


def provider_failure(row: dict) -> str | None:
    """The phase stop rule: a provider-side error on any request (model-output
    validation errors are data, not failures), or a trial that did not finish."""
    for a in ACTORS:
        for req in row["requests"][a]:
            for err in req["provider_error_types"]:
                if err not in MODEL_OUTPUT_ERROR_TYPES:
                    return f"{a}: {err}"
    if not row["complete"]:
        return "incomplete trial" + (" (timed out)" if row.get("timed_out") else "")
    return None


def replay_check(row: dict, scenario, render_fn=None) -> dict:
    """Boundary 0 of a live trial against the same decisions replayed under
    the same policy, in the order the loop saw them ready (zero calls)."""
    b = row["boundaries"][0]
    if not b["complete"]:
        return {"replay_matches_live": None}
    raws = {a: b["records"][a]["raw_model_output"] for a in ACTORS}
    ready = tuple(b["loop_ready_order"]) if b["loop_ready_order"] else None
    kwargs = {"render_fn": render_fn} if render_fn else {}
    r = replay_under_policy(scenario, raws, row["commitment_policy"], response_ready_order=ready, parity=row["p4_parity"] or 0,
                            pair_id=row["trial_id"], decision_boundary_id=b["decision_boundary_id"], model_id=MODEL_ID, **kwargs)
    return {
        "replay_matches_live": r["world_state_hash_after"] == b["world_hash_after_pair"],
        "replay_effect_matches_live": r["effect_state_hash_after"] == b["effect_state_hash_after_pair"],
        "replay_first_commit_actor": r["first_commit_actor"],
        "replay_contested_effects": r["contested_effects"],
    }


# -- zero-call reproduction of live trials ---------------------------------------------------

REPRO_FILES = {"native": "native_live_reproductions.jsonl", "strands": "strands_live_reproductions.jsonl"}
REPRO_FAST_S, REPRO_SLOW_S, REPRO_RACE_S = 0.08, 0.35, 0.6
# Fields that come from the counterfactual (the decision applied alone to a
# copy of its snapshot). Live rows written before the unpatched-copy fix
# carry wrong values here; the verified reproduction supplies them.
DERIVED_RECORD_FIELDS = ("counterfactual", "legal_at_snapshot", "staleness_class", "consequence")
VERIFIED_RECORD_FIELDS = ("decision_hash", "snapshot_world_hash", "snapshot_relevant_hash", "apply_world_hash", "apply_relevant_hash",
                          "committed_action", "commit_matches_decision", "apply_outcome", "time_cost", "target_state_after", "self_state_after")
VERIFIED_BOUNDARY_FIELDS = ("apply_order", "world_hash_before", "world_hash_after_first_commit", "world_hash_after_pair",
                            "effect_state_hash_after_pair", "event_multiset_hash_pair", "contested_effects", "same_tick_commit")
VERIFIED_ROW_FIELDS = ("pre_action_world_hash", "world_state_hash_after", "effect_state_hash_after", "event_multiset_hash")


def _norm(v):
    return json.loads(json.dumps(v, sort_keys=True, default=str))


def reproduce(row: dict, scenario, *, region: str = "us-east-1") -> dict:
    """Zero-call reproduction of one live trial through the same probe: same
    scenario, policy, parity, call order and boundaries, with scripted
    stand-ins answering every recorded model call in order and the
    boundary-0 ready order pinned to the one the loop saw (a same-tick tie
    is forced onto one tick; an HP09 race is forced the way it happened).
    Every world, decision and apply field must come out identical; then
    the reproduction's counterfactual-derived fields hold for the live
    decisions."""
    import threading

    from src.evaluation.commitment_probe import scripted_sequence_native_client, scripted_sequence_strands_invoke

    bs = row["boundaries"]
    order = bs[0].get("loop_ready_order")
    racers = tuple(a for a in ACTORS if any(r["launch_kind"] == "extra" and r["hp09_violation"] for r in row["requests"][a]))
    kwargs: dict = {}
    if racers:
        gated = racers
        kwargs["force_ready_after_opponent_step"] = racers
    elif order is None:
        gated = ACTORS
        kwargs["force_ready_at_tick"] = 1
    else:
        # the slower answer lands at a tick boundary one tick after the opponent's step: after the loop has
        # polled it in flight once, and never inside the handoff window (a live tick the recorded trial did not have)
        gated = (order[1],)
        kwargs["ready_ticks_after_opponent_step"] = {order[1]: 1}
    gates = {a: threading.Event() for a in gated}
    steps = {}
    for a in ACTORS:
        payloads = [p for req in row["requests"][a] for p in req["payloads"]]
        slow = REPRO_RACE_S if a in racers else REPRO_SLOW_S
        steps[a] = [(p, slow if (i == 0 and a in gates) else REPRO_FAST_S, gates.get(a) if i == 0 else None) for i, p in enumerate(payloads)]
    if row["runtime"] == "strands":
        kwargs["strands_invoke_overrides"] = {a: scripted_sequence_strands_invoke(steps[a]) for a in ACTORS}
        provider, model_id = "bedrock", MODEL_ID
    else:
        kwargs["client_overrides"] = {a: scripted_sequence_native_client(steps[a]) for a in ACTORS}
        provider, model_id = "mock", "scripted"
    rep = run_policy_trial(scenario, row["commitment_policy"], trial_id=row["trial_id"], runtime=row["runtime"], provider=provider,
                           model_id=model_id, region=region, decision_call_order=row["decision_call_order"],
                           boundaries=row["boundaries_planned"], p4_parity=row["p4_parity"] or 0, gates=gates, **kwargs)
    mismatches = [f"row.{k}" for k in VERIFIED_ROW_FIELDS if _norm(row.get(k)) != _norm(rep.get(k))]
    if len(rep["boundaries"]) != len(bs) or not rep["complete"]:
        mismatches.append("boundaries/complete")
    out_boundaries = []
    for b, rb in zip(bs, rep["boundaries"]):
        mismatches += [f"b{b['boundary']}.{k}" for k in VERIFIED_BOUNDARY_FIELDS if _norm(b.get(k)) != _norm(rb.get(k))]
        recs = {}
        for a in ACTORS:
            live_rec, rep_rec = b["records"][a], rb["records"].get(a, {})
            mismatches += [f"b{b['boundary']}.{a}.{k}" for k in VERIFIED_RECORD_FIELDS if _norm(live_rec.get(k)) != _norm(rep_rec.get(k))]
            recs[a] = {k: rep_rec.get(k) for k in DERIVED_RECORD_FIELDS}
            recs[a]["live_value_differed"] = [k for k in DERIVED_RECORD_FIELDS if _norm(live_rec.get(k)) != _norm(rep_rec.get(k))]
        out_boundaries.append({"boundary": b["boundary"], "records": recs, "first_result": rb.get("first_result"),
                               "second_result": rb.get("second_result")})
    return {
        "trial_id": row["trial_id"], "runtime": row["runtime"], "runtime_key": row.get("runtime_key"),
        "commitment_policy": row["commitment_policy"], "scenario_id": row["scenario_id"], "model_calls": 0,
        "pinned": {"boundary0_ready_order": order, "hp09_racers": list(racers), "same_tick_tie": order is None and not racers},
        "reproduced": not mismatches, "mismatches": mismatches,
        "reproduction_hp09_violations": rep["lifecycle"]["hp09_violations"],
        "live_state_untouched_by_assembly": rep["live_state_untouched_by_assembly"],
        "event_log_hash": rep["event_log_hash"], "live_event_log_hash_differed": row.get("event_log_hash") != rep["event_log_hash"],
        "boundaries": out_boundaries,
    }


def run_reproductions(runtime: str, region: str = "us-east-1") -> dict:
    scenarios = build_policy_scenarios()
    rows = read_jsonl(out_path(FILES[runtime]))
    out = [reproduce(r, scenarios[r["scenario_id"]], region=region) for r in rows]
    with open(out_path(REPRO_FILES[runtime]), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    return {"trials": len(out), "reproduced": sum(r["reproduced"] for r in out),
            "not_reproduced": {r["trial_id"]: r["mismatches"] for r in out if not r["reproduced"]},
            "records_with_corrected_fields": sum(bool(rec["live_value_differed"]) for r in out for b in r["boundaries"]
                                                 for rec in b["records"].values()),
            "event_log_hash_corrected": sum(r["live_event_log_hash_differed"] for r in out)}


def run(runtime: str, region: str) -> int:
    scenarios = build_policy_scenarios()
    path = out_path(FILES[runtime])
    done = {r["trial_id"] for r in read_jsonl(path)}
    render = None
    if runtime == "strands":
        from src.evaluation.m6g_live import StrandsDecider

        render = StrandsDecider(region, MODEL_ID).render
    for t in planned_trials(runtime):
        if t["trial_id"] in done:
            continue
        row = run_policy_trial(scenarios[t["scenario_id"]], t["policy"], trial_id=t["trial_id"], runtime=RUNTIMES[runtime],
                               provider="bedrock", model_id=MODEL_ID, region=region, decision_call_order=t["decision_call_order"],
                               boundaries=t["boundaries"], p4_parity=t["p4_parity"])
        failure = provider_failure(row)
        if failure:
            append_jsonl(out_path("provider_events.jsonl"), {
                "phase": f"live_{runtime}_abort", "trial_id": t["trial_id"], "failure": failure, "timed_out": row["timed_out"],
                "lifecycle": row["lifecycle"], "freeze_audit": row["freeze_audit"],
                "provider_error_types": {a: [e for r in row["requests"][a] for e in r["provider_error_types"]] for a in ACTORS},
                "at_utc": datetime.now(timezone.utc).isoformat()})
            print(f"[ABORT] provider failure in {t['trial_id']}: {failure}", file=sys.stderr)
            return 2
        row.update({"rep": t["rep"], "phase": "E" if runtime == "strands" else "D", "model_id": MODEL_ID, "runtime_key": runtime})
        row.update(replay_check(row, scenarios[t["scenario_id"]], render))
        for e in row.pop("lifecycle_events"):
            append_jsonl(out_path(LIFECYCLE_FILE), {"trial_id": t["trial_id"], "runtime": runtime, "commitment_policy": t["policy"], **e})
        append_jsonl(path, row)
        bs = row["boundaries"]
        print(f"[{runtime} {t['trial_id']}] "
              + " | ".join(f"b{b['boundary']}: ready={b.get('ready_order')} commit={b.get('apply_order')} "
                           f"{[b['records'][a].get('decision', {}).get('action') for a in ACTORS]} contested={b.get('contested_effects')}"
                           for b in bs)
              + f" | requests={row['lifecycle']['requests_launched']} unused={row['lifecycle']['unused_responses']} "
                f"hp09={row['lifecycle']['hp09_violations']} held_dt={row['freeze_audit']['ticks_advanced_while_a_decision_was_held']} "
                f"replay==live={row.get('replay_matches_live')}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=tuple(RUNTIMES))
    parser.add_argument("--region", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--write-plan", action="store_true", help="freeze live_call_plan.json (refuses to overwrite)")
    parser.add_argument("--reproduce", action="store_true", help="zero-call reproduction of every recorded live trial of --runtime")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.reproduce:
        if args.runtime is None:
            parser.error("--reproduce needs --runtime")
        summary = run_reproductions(args.runtime)
        print(f"[REPRODUCE] {json.dumps(summary)}")
        return 0 if not summary["not_reproduced"] else 1
    if args.write_plan:
        from src.evaluation.m6g_live import write_json

        if os.path.exists(out_path("live_call_plan.json")):
            print("[FATAL] live_call_plan.json already frozen", file=sys.stderr)
            return 1
        write_json(out_path("live_call_plan.json"), live_plan())
        print(f"[PLAN] frozen: native {expected_calls('native')} calls, strands {expected_calls('strands')} calls")
        return 0
    if args.runtime is None:
        parser.error("--runtime or --write-plan is required")
    if not os.path.exists(out_path("live_call_plan.json")):
        print("[FATAL] freeze live_call_plan.json (--write-plan) before any live call", file=sys.stderr)
        return 1
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
    append_jsonl(out_path("provider_events.jsonl"), {"phase": f"preflight_live_{args.runtime}", "at_utc": datetime.now(timezone.utc).isoformat(), **preflight})
    print(f"[PREFLIGHT] {json.dumps(preflight, default=str)}")
    if not preflight["ok"]:
        return 1
    if args.preflight_only:
        return 0
    start = datetime.now(timezone.utc)
    status = run(args.runtime, region)
    print(f"[DONE] {args.runtime} status={status} ({start.isoformat()} - {datetime.now(timezone.utc).isoformat()})")
    return status


if __name__ == "__main__":
    sys.exit(main())
