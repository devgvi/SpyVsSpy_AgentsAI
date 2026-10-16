"""M6G.3 Phase B (spec sections 19-25, 40): minimal live validation of the
PROMOTED production scheduler.

Each trial is one real production Game with two LLM actors on Bedrock under
the default commit policy (P1 BARRIER_ANSWER_ORDER + the HP09 guard),
observed by `barrier_probe.run_production_trial` - a pass-through observer:
nothing is held, reordered, frozen or re-decided by the harness. A trial
runs until each actor has committed K=2 decisions: the first decision
boundary from the scenario's shared snapshot, then continuous play (the
second decisions start whenever each actor is free again, as in a game).

Plan: Native 10 x C2 (shared door) + 5 x C0 (non-conflict), then - after
the Native analysis and a clean RP01-RP18 gate - Strands the same.
Scenarios interleaved (C2, C0, C2 blocks), decision_call_order alternating
with the scenario repetition. SOCIAL_OFF, cross-episode memory off,
DEFAULT vs DEFAULT.

After each trial, with zero calls, boundary 1 is checked three ways from
its recorded decisions and ready order: reproduced through the promoted
loop, run through the legacy "answer_order" loop (pre-promotion P0), and
replayed under P0 semantics (commitment_policy.replay_under_policy).

Writes native_live_trials.jsonl / strands_live_trials.jsonl,
request_lifecycle_events.jsonl, provider_events.jsonl. Resumable; any
provider-side failure or incomplete trial stops the run.

    python -m src.evaluation.run_m6g3_live --write-plan
    caffeinate -i python -m src.evaluation.run_m6g3_live --runtime native
    caffeinate -i python -m src.evaluation.run_m6g3_live --runtime strands
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from typing import Any

from src.evaluation.barrier_probe import run_production_trial
from src.evaluation.commitment_policy import C0, C2, P0, build_policy_scenarios, replay_under_policy
from src.evaluation.decision_scheduling import MODEL_OUTPUT_ERROR_TYPES
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS
from src.evaluation.m6g_live import append_jsonl, read_jsonl, run_live_preflight, verify_sleep_assertion, write_json
from src.evaluation.run_m6g3_replay import MODEL_ID, OUT_DIR, _p0_summary, _summary, _trial, out_path
from src.game import COMMIT_POLICY_ANSWER_ORDER, COMMIT_POLICY_BARRIER, DEFAULT_COMMIT_POLICY

RUNTIMES = {"native": "native_tool_like", "strands": "strands"}
FILES = {"native": "native_live_trials.jsonl", "strands": "strands_live_trials.jsonl"}
LIFECYCLE_FILE = "request_lifecycle_events.jsonl"
PLAN = {C2: 10, C0: 5}
BLOCK = (C2, C0, C2)
ROUNDS = 2
TRIAL_MAX_WALL_S = 240.0
PRODUCTION_FILES = ("src/game.py", "src/llm_controller.py", "src/agents/strands_agent.py")
CHECK_KEYS = ("commit_order", "world_hash_before", "world_hash_after_first_commit", "world_hash_after_pair", "effect_state_hash_after_pair",
              "event_log_hash_pair", "contested_effects", "first_action_cost", "second_action_cost", "first_result", "second_result")


def planned_trials(runtime: str) -> list[dict[str, Any]]:
    trials, reps = [], {sid: 0 for sid in PLAN}
    while any(reps[sid] < n for sid, n in PLAN.items()):
        for sid in BLOCK:
            if reps[sid] >= PLAN[sid]:
                continue
            rep = reps[sid]
            reps[sid] += 1
            trials.append({
                "trial_id": f"{'M' if runtime == 'strands' else 'L'}:{runtime}:{sid}:r{rep:02d}", "runtime_key": runtime, "rep": rep,
                "scenario_id": sid, "decision_call_order": "black_first" if rep % 2 == 0 else "white_first", "rounds": ROUNDS,
                "commit_policy": DEFAULT_COMMIT_POLICY,
            })
    return trials


def expected_calls(runtime: str) -> int:
    return sum(2 * t["rounds"] for t in planned_trials(runtime))


def max_calls(runtime: str) -> int:
    """Cap: every in-horizon decision repaired once, plus one beyond-horizon request per actor per trial."""
    return 2 * expected_calls(runtime) + 2 * len(planned_trials(runtime))


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def live_plan() -> dict[str, Any]:
    """live_call_plan.json (spec section 19), frozen before the first call."""
    return {
        "frozen_before_live_calls": "Written after the network-free production tests and the zero-call replay equivalence passed, "
                                    "before any provider call. Trial list, counts, order, stop rules and the analysis rules below are "
                                    "not changed after seeing live data.",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "provider": "bedrock", "region": "us-east-1",
        "auth": "AWS_PROFILE=gl-primitives (SigV4); m6g_live.run_live_preflight before each phase; caffeinate required",
        "commit_policy": f"{DEFAULT_COMMIT_POLICY} (the promoted production default: P1 barrier in response-ready order + HP09 guard)",
        "production_code_sha256": {p: _file_sha256(p) for p in PRODUCTION_FILES},
        "harness": "barrier_probe.run_production_trial: pass-through observer of the real Game._step; no ready schedule, no gates, "
                   "no scripted model, no evaluation barrier",
        "rounds_per_actor": ROUNDS,
        "episode_validation": "boundary 1 = both actors decide from the scenario's shared snapshot; later decisions start whenever each "
                              "actor is free again (continuous play). A trial ends once each actor has committed 2 decisions; a request "
                              "launched after an actor's own 2nd commit is beyond the trial horizon and reported separately",
        "scenarios": {C2: "shared locked door, primary conflict (1.0 s unlock vs 0.3 s re-open)", C0: "non-conflict control"},
        "personalities": "DEFAULT vs DEFAULT", "social": "SOCIAL_OFF (cross-episode social memory off)",
        "order": "blocks of (C2, C0, C2); decision_call_order black_first on even and white_first on odd scenario repetitions",
        "order_native_then_strands": "Native first; Strands only after the Native analysis and a clean RP01-RP18 gate",
        "stop_rule": "any provider-side error (not a model-output validation error), an incomplete (timed-out) trial, or the call cap "
                     "aborts the phase; the aborting trial goes to provider_events.jsonl only, so a resumed run repeats it",
        "runtimes": {rk: {"trials": len(planned_trials(rk)), "trials_per_scenario": dict(PLAN), "expected_model_calls": expected_calls(rk),
                          "max_model_calls": max_calls(rk), "plan": planned_trials(rk)} for rk in RUNTIMES},
        "analysis_rules": {
            "lifecycle_gate": "per actor and boundary: zero requests launched while a decision was ready-uncommitted or held (HP09) and "
                              "zero unused responses within the trial horizon",
            "game_time_gate": "zero live ticks while any decision was in flight, ready-uncommitted or held; game_time_waited_s == 0 for "
                              "every committed decision",
            "ready_order_gate": "first-ready (response_ready_at) actor == first-commit actor for every paired boundary, except an exact "
                                "timestamp tie (documented Black-then-White fallback)",
            "world_outcome": "boundary 1 world, effect, pair events, costs and commit order reproduced with zero calls through the "
                             "promoted loop, and equal to the legacy loop and to the P0 replay for the same ready order; a same-tick "
                             "hand-over with White published first is the documented exception (P0 applied it Black first)",
            "wall_clock": "median ms from the boundary's first request to the first commit and to the pair commit, boundary 1, vs "
                          "M6G.2's live P0 trials on the same runtime and scenario",
            "no_balance_claim": "P1 keeps latency-derived ready-order priority; no fairness or staleness claim",
        },
    }


def experiment() -> dict[str, Any]:
    return {
        "milestone": "M6G.3", "title": "Response-Barrier Production Promotion & Episode Validation",
        "spec_file": "M6G.3_RESPONSE_BARRIER_PRODUCTION_PROMOTION_AND_EPISODE_VALIDATION.md",
        "follows": "M6G.2 (evaluation_runs/m6g2-commitment-policy-counterbalancing-v1/), M6G.1, M6G - all read-only here",
        "question": "Can P1 BARRIER_ANSWER_ORDER plus the HP09 invariant become the production scheduler without changing world semantics?",
        "candidate": "P1_BARRIER_ANSWER_ORDER", "model_id": MODEL_ID, "provider": "bedrock", "region": "us-east-1",
        "auth": "AWS named profile gl-primitives (SigV4 credential chain via AWS_PROFILE)", "temperature": 0.0,
        "runtimes": {"native": "production Game -> NativeToolLikeAgentAdapter -> LLMController -> BedrockClient",
                     "strands": "production Game -> StrandsAgentAdapter -> Strands Agent on BedrockModel; behind RP01-RP18"},
        "communication_policy": "SOCIAL_OFF (Game(communication_enabled=False))",
        "cross_episode_social_memory": "OFF", "personality": {ACTOR_A: "default", ACTOR_B: "default"},
        "production_change": {
            "Game(commit_policy=...)": f"{COMMIT_POLICY_BARRIER!r} (default) | {COMMIT_POLICY_ANSWER_ORDER!r} (legacy P0, historical harnesses)",
            "barrier": "Game._step/_barrier_commit: a handed-over decision is held (WAIT, zero game time) while the other LLM decision is "
                       "outstanding; when none is, every held decision commits on one tick in response_ready_at order",
            "hp09_guard": "LLMController/StrandsAgentAdapter.get_action: a ready decision is handed over before any new request",
            "design": "promotion_design.md (written before the production edit)",
        },
        "evidence": {
            "tests": "tests/test_m6g3_response_barrier.py (network-free: T0-T6, both HP09 routes, both runtimes)",
            "replay": "run_m6g3_replay: 250 captured pairs through the promoted loop vs the legacy loop vs P0 replay (replay_equivalence.jsonl)",
            "live": "run_m6g3_live: the promoted loop on Bedrock (native_live_trials.jsonl, strands_live_trials.jsonl)",
        },
        "not_done_by_design": [
            "no P2/P3/P4 or random order in production", "no stale revalidate/cancel/redecide", "no action-semantics change",
            "no memory, context, prompt, repair, fallback or tool-schema change", "no new model, provider or runtime",
            "no fix of the known no-path MOVE issue (separate known issue)",
        ],
        "harness_modules": ["src/evaluation/barrier_probe.py", "src/evaluation/run_m6g3_replay.py", "src/evaluation/run_m6g3_live.py",
                            "src/evaluation/run_m6g3_analysis.py", "tests/test_m6g3_response_barrier.py"],
        "regression_baseline_entering_milestone": "1848/1854 passing; 6 known environment-only failures (1 AWS region, 1 sampler timing, "
                                                  "4 Ollama qwen:latest)",
    }


def provider_failure(row: dict[str, Any]) -> str | None:
    """The stop rule: a provider-side error on any request (model-output
    validation errors are data, not failures), or a trial that did not finish."""
    for a in ACTORS:
        for req in row["requests"][a]:
            for err in req["provider_error_types"]:
                if err not in MODEL_OUTPUT_ERROR_TYPES:
                    return f"{a}: {err}"
    if not row["complete"]:
        return "incomplete trial" + (" (timed out)" if row.get("timed_out") else "")
    return None


def _norm(v: Any) -> Any:
    return json.loads(json.dumps(v, sort_keys=True, default=str))


def boundary1_checks(row: dict[str, Any], scenario) -> dict[str, Any]:
    """Zero calls: boundary 1's recorded decisions and ready order through the
    promoted loop (reproduction), the legacy loop (pre-promotion P0) and
    the P0 replay semantics."""
    b = row["boundaries"][0] if row["boundaries"] else {}
    if not b.get("pair"):
        return {"b1_checked": False}
    order, same_tick = b["ready_order"], b["handed_over_same_tick"]
    p = {"pair_id": row["trial_id"], "source_trial_id": row["trial_id"], "runtime": row["runtime_key"],
         "raws": {a: b["records"][a]["raw_model_output"] for a in ACTORS}}
    groups = [[(order[0], 1), (order[1], 1)]] if same_tick else [[(order[0], 1)], [(order[1], 1)]]
    post = _summary(_trial(scenario, p, COMMIT_POLICY_BARRIER, groups))
    pre = _summary(_trial(scenario, p, COMMIT_POLICY_ANSWER_ORDER, groups))
    rep = _p0_summary(replay_under_policy(scenario, p["raws"], P0, response_ready_order=None if same_tick else tuple(order),
                                          pair_id=row["trial_id"], decision_boundary_id=f"{row['trial_id']}:b1", model_id=MODEL_ID))
    live = _norm({k: b.get(k) for k in CHECK_KEYS})
    live_decisions = {a: b["records"][a]["decision_hash"] for a in ACTORS}

    def diff(other: dict[str, Any]) -> list[str]:
        o = _norm({k: other.get(k) for k in CHECK_KEYS})
        return sorted(k for k in CHECK_KEYS if o[k] != live[k]) + (["decision_hashes"] if other["decision_hashes"] != live_decisions else [])

    rep_view = {"commit_order": rep["apply_order"], "world_hash_after_pair": rep["world_state_hash_after"],
                "effect_state_hash_after_pair": rep["effect_state_hash_after"], "first_action_cost": rep["first_action_cost"],
                "second_action_cost": rep["second_action_cost"], "contested_effects": rep["contested_effects"],
                "first_result": rep["first_result"], "second_result": rep["second_result"]}
    rep_keys = tuple(rep_view)
    rep_diff = sorted(k for k in rep_keys if _norm(rep_view[k]) != live[k])
    return {
        "b1_checked": True, "b1_same_tick_handover": same_tick, "b1_true_ready_order": order,
        "b1_white_published_first_same_tick": same_tick and order[0] == ACTOR_B,
        "b1_reproduced": not diff(post), "b1_reproduction_mismatches": diff(post),
        "b1_legacy_equal": not diff(pre), "b1_legacy_mismatches": diff(pre), "b1_legacy_commit_order": pre["commit_order"],
        "b1_p0_replay_equal": not rep_diff, "b1_p0_replay_mismatches": rep_diff,
        "b1_reproduction_lifecycle": post["lifecycle"], "b1_check_model_calls": 0,
    }


def _calls_so_far(runtime: str) -> int:
    return sum(row["lifecycle"]["provider_calls"] for row in read_jsonl(out_path(FILES[runtime])))


def run(runtime: str, region: str) -> int:
    scenarios = build_policy_scenarios()
    path = out_path(FILES[runtime])
    done = {r["trial_id"] for r in read_jsonl(path)}
    cap = max_calls(runtime)
    for t in planned_trials(runtime):
        if t["trial_id"] in done:
            continue
        if _calls_so_far(runtime) >= cap:
            append_jsonl(out_path("provider_events.jsonl"), {"phase": f"live_{runtime}_abort", "trial_id": t["trial_id"],
                                                            "failure": f"call cap {cap} reached", "at_utc": datetime.now(timezone.utc).isoformat()})
            print(f"[ABORT] call cap {cap} reached before {t['trial_id']}", file=sys.stderr)
            return 2
        s = scenarios[t["scenario_id"]]
        row = run_production_trial(s, trial_id=t["trial_id"], runtime=RUNTIMES[runtime], provider="bedrock", model_id=MODEL_ID, region=region,
                                   decision_call_order=t["decision_call_order"], commit_policy=t["commit_policy"], rounds=t["rounds"],
                                   max_wall_seconds=TRIAL_MAX_WALL_S)
        failure = provider_failure(row)
        if failure:
            append_jsonl(out_path("provider_events.jsonl"), {
                "phase": f"live_{runtime}_abort", "trial_id": t["trial_id"], "failure": failure, "timed_out": row["timed_out"],
                "lifecycle": row["lifecycle"], "freeze_audit": row["freeze_audit"],
                "provider_error_types": {a: [e for r in row["requests"][a] for e in r["provider_error_types"]] for a in ACTORS},
                "at_utc": datetime.now(timezone.utc).isoformat()})
            print(f"[ABORT] provider failure in {t['trial_id']}: {failure}", file=sys.stderr)
            return 2
        row.update({"rep": t["rep"], "phase": "B", "model_id": MODEL_ID, "runtime_key": runtime, "planned": t})
        row.update(boundary1_checks(row, s))
        for e in row.pop("lifecycle_events"):
            append_jsonl(out_path(LIFECYCLE_FILE), {"trial_id": t["trial_id"], "runtime": runtime, "commit_policy": t["commit_policy"], **e})
        append_jsonl(path, row)
        lc, fz = row["lifecycle"], row["freeze_audit"]
        print(f"[{runtime} {t['trial_id']}] "
              + " | ".join(f"b{b['decision_boundary_id']}: ready={b.get('ready_order')} commit={b.get('commit_order')} "
                           f"{[b['records'][a]['decision']['action'] for a in ACTORS if a in b.get('records', {})]}"
                           f" contested={b.get('contested_effects')}" for b in row["boundaries"])
              + f" | requests={lc['requests_launched']}+{lc['beyond_horizon_requests']} dup={lc['duplicate_requests']} "
                f"unused={lc['unused_responses']} live_ticks_waiting={fz['ticks_advanced_while_waiting']} "
                f"b1 repro={row.get('b1_reproduced')} legacy={row.get('b1_legacy_equal')} p0={row.get('b1_p0_replay_equal')}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=tuple(RUNTIMES))
    parser.add_argument("--region", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--write-plan", action="store_true", help="freeze live_call_plan.json (refuses to overwrite)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.write_plan:
        if os.path.exists(out_path("live_call_plan.json")):
            print("[FATAL] live_call_plan.json already frozen", file=sys.stderr)
            return 1
        write_json(out_path("live_call_plan.json"), live_plan())
        if not os.path.exists(out_path("experiment.json")):
            write_json(out_path("experiment.json"), experiment())
        print(f"[PLAN] frozen: native {expected_calls('native')} calls (cap {max_calls('native')}), "
              f"strands {expected_calls('strands')} calls (cap {max_calls('strands')})")
        return 0
    if args.runtime is None:
        parser.error("--runtime or --write-plan is required")
    if not os.path.exists(out_path("live_call_plan.json")):
        print("[FATAL] freeze live_call_plan.json (--write-plan) before any live call", file=sys.stderr)
        return 1
    with open(out_path("live_call_plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    changed = [p for p, h in plan["production_code_sha256"].items() if _file_sha256(p) != h]
    if changed:
        print(f"[FATAL] production code changed since the plan was frozen: {changed}", file=sys.stderr)
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
