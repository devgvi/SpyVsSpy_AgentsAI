"""M6G.3 analysis (spec sections 26-30, 44-49): gates, metrics, the
promotion decision and the required tables.

Reads (all written by M6G.3 itself unless noted, read-only):
    replay_equivalence.jsonl, native/strands_live_trials.jsonl,
    provider_events.jsonl, live_call_plan.json
    M6G.2 native/strands_live_trials.jsonl and request_waste_metrics.json
    (the "before" wall clock and request lifecycle), M6G.1 live trials
    (the race observed live before)

Runs, with zero model calls, the targeted HP09 race (both routes, both
runtimes, legacy vs promoted loop) so the lifecycle table carries measured
targeted numbers next to the live ones.

Writes semantic_equivalence_metrics.json, request_waste_metrics.json,
wall_clock_metrics.json, runtime_validation.json, promotion_decision.json,
summary.json, and splices the section 45-48 tables into summary.md between
the TABLES markers.

    python -m src.evaluation.run_m6g3_analysis [--regression-log PATH]
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

from src.evaluation.commitment_policy import C0, C2, P0, P1
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS
from src.evaluation.m6g_live import read_jsonl, write_json
from src.evaluation.run_m6g3_live import FILES, planned_trials
from src.evaluation.run_m6g3_replay import KIND_EQUIVALENCE, KIND_TIE_TRUE_ORDER, OUT_DIR, out_path

M6G2_DIR = os.path.join("evaluation_runs", "m6g2-commitment-policy-counterbalancing-v1")
M6G1_DIR = os.path.join("evaluation_runs", "m6g1-long-running-actions-v1")
RUNTIME_KEYS = ("native", "strands")
CANDIDATE = "P1_BARRIER_ANSWER_ORDER"
CONFIRMED = "P1_PRODUCTION_PROMOTION_CONFIRMED"
BLOCKED = "P1_PRODUCTION_PROMOTION_BLOCKED"
SEMANTIC_FAILED = "P1_SEMANTIC_EQUIVALENCE_FAILED"
LIFECYCLE_NOT_CONFIRMED = "P1_LIFECYCLE_FIX_NOT_CONFIRMED"
MIXED = "MIXED"
REQUIRED_CHECKS = ("complete", "commit_order", "world", "effect_state", "pair_event_hash", "costs")
MOVE_FIRST_EXPLANATION = ("MOVE first commit: the legacy loop committed it on a frozen tick (COMMITTED_FROZEN, no movement until the next "
                          "live tick); the barrier commits it on the live release tick, where it moves at once. The world after the pair "
                          "is identical; only the first commit's intermediate state and outcome label differ")
# the six known environment-only failures entering M6G.3 (spec section 42)
KNOWN_FAILURE_PATTERNS = (
    r"tests/test_bedrock_client\.py::test_game_construction_fails_clearly_with_no_region_configured",
    r"tests/test_endurance\.py::test_background_resource_sampler_collects_multiple_samples_and_stops_cleanly",
    r"tests/test_evaluation_m5d3\.py::",
    r"tests/test_evaluation_m5d6\.py::",
    r"tests/test_gameplay_representation\.py::test_m5d8",
)
TABLES_BEGIN, TABLES_END = "<!-- M6G3_TABLES_BEGIN -->", "<!-- M6G3_TABLES_END -->"


def _stats(values: list[float]) -> dict[str, Any]:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    q = statistics.quantiles(v, n=4, method="inclusive") if len(v) > 1 else [v[0]] * 3
    return {"n": len(v), "median": round(statistics.median(v), 1), "p25": round(q[0], 1), "p75": round(q[2], 1),
            "min": round(v[0], 1), "max": round(v[-1], 1), "mean": round(statistics.fmean(v), 1)}


def _short(h: str | None) -> str:
    return h[:10] if h else "-"


def _pct(k: int, n: int) -> str:
    return f"{k}/{n}"


# -- semantic equivalence (spec sections 17, 26) ------------------------------------------


def _explain(row: dict[str, Any]) -> str | None:
    failed = set(row["failed_checks"])
    if not failed:
        return None
    post, pre = row["post"], row["pre"]
    first = (post.get("commit_order") or [None])[0]
    move_first = (first is not None and post["records"].get(first, {}).get("decision", {}).get("action") == "move"
                  and pre["records"].get(first, {}).get("apply_dt") == 0.0)
    if failed <= {"results", "world_after_first_commit"} and move_first and row["equivalent"]:
        return MOVE_FIRST_EXPLANATION
    return "UNEXPLAINED"


def semantic_equivalence(replay: list[dict[str, Any]], live: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    eq = [r for r in replay if r["kind"] == KIND_EQUIVALENCE]
    ties = [r for r in replay if r["kind"] == KIND_TIE_TRUE_ORDER]
    checks = sorted({k for r in eq for k in r["checks"]})
    diffs = [{"pair_id": r["pair_id"], "source": r["source"], "runtime": r["runtime"], "scenario_id": r["scenario_id"],
              "ready_order": r["response_ready_order"], "failed_checks": r["failed_checks"], "explanation": _explain(r)}
             for r in eq if r["failed_checks"]]
    recorded = [r for r in eq if "m6g2_recorded_p0" in r["checks"]]
    tie_rows = [{"pair_id": r["pair_id"], "source": r["source"], "runtime": r["runtime"], "scenario_id": r["scenario_id"],
                 "true_ready_order": r["true_ready_order"], "barrier_commit_order": r["post"]["commit_order"],
                 "p0_commit_order": r["p0_replay"]["apply_order"], "world_differs_from_p0": r["world_differs_from_p0"],
                 "contested_effects": r["post"]["contested_effects"], "lifecycle_clean": r["post_lifecycle_clean"]} for r in ties]
    live_out = {}
    for rk, rows in live.items():
        if not rows:
            continue
        checked = [r for r in rows if r.get("b1_checked")]
        white_same_tick = [r for r in checked if r["b1_white_published_first_same_tick"]]
        live_out[rk] = {
            "trials": len(rows), "b1_checked": len(checked), "b1_reproduced": sum(r["b1_reproduced"] for r in checked),
            "b1_legacy_equal": sum(r["b1_legacy_equal"] for r in checked), "b1_p0_replay_equal": sum(r["b1_p0_replay_equal"] for r in checked),
            "b1_same_tick_handovers": sum(r["b1_same_tick_handover"] for r in checked),
            "b1_white_published_first_same_tick": len(white_same_tick),
            "b1_unexplained_legacy_differences": [r["trial_id"] for r in checked if not r["b1_legacy_equal"]
                                                  and not r["b1_white_published_first_same_tick"]],
            "b1_not_reproduced": {r["trial_id"]: r["b1_reproduction_mismatches"] for r in checked if not r["b1_reproduced"]},
            "b1_check_model_calls": sum(r["b1_check_model_calls"] for r in checked),
        }
    replay_pass = all(r["equivalent"] for r in eq) and all(d["explanation"] != "UNEXPLAINED" for d in diffs)
    live_pass = all(v["b1_reproduced"] == v["b1_checked"] == v["trials"] and not v["b1_unexplained_legacy_differences"]
                    for v in live_out.values())
    return {
        "replay": {
            "pairs": len(eq), "model_calls": sum(r["model_calls"] for r in eq),
            "by_source": dict(sorted(Counter(f"{r['source']}:{r['runtime']}" for r in eq).items())),
            "by_scenario": dict(sorted(Counter(r["scenario_id"] for r in eq).items())),
            "required_checks": list(REQUIRED_CHECKS),
            "check_pass_counts": {k: sum(bool(r["checks"].get(k)) for r in eq if k in r["checks"]) for k in checks},
            "equivalent_pairs": sum(r["equivalent"] for r in eq), "all_checks_pass_pairs": sum(r["all_checks_pass"] for r in eq),
            "supporting_check_differences": diffs,
            "m6g2_recorded_p0": {"compared": len(recorded), "equal": sum(r["checks"]["m6g2_recorded_p0"] for r in recorded)},
            "same_tick_ties": {
                "pairs": sum(r["tie"] for r in eq), "replayed_black_published_first_equivalent": sum(r["equivalent"] for r in eq if r["tie"]),
                "true_order_rows": len(ties), "true_order_white_first": sum(r["true_ready_order"] == [ACTOR_B, ACTOR_A] for r in ties),
                "barrier_commit_order_is_true_ready_order": sum(r["commit_order_is_true_ready_order"] for r in ties),
                "commit_order_differs_from_p0": sum(r["differs_from_p0"] for r in ties),
                "world_differs_from_p0": sum(r["world_differs_from_p0"] for r in ties), "rows": tie_rows,
                "rule": "the barrier commits answers that reach the loop on one tick in publication order (spec sections 10-11); P0 "
                        "applied them Black first. Intended, reported separately, not an equivalence failure",
            },
        },
        "live": live_out,
        "gate": {
            "semantic_equivalence_pass": replay_pass and live_pass, "replay_pass": replay_pass, "live_pass": live_pass,
            "rule": "for identical captured decisions and response-ready order the promoted loop's world, effect state, pair events, costs "
                    "and commit order equal the pre-promotion loop's and the P0 replay's; supporting differences must be explained",
            "explained_exceptions": sorted({d["explanation"] for d in diffs if d["explanation"]}) + [
                "same-tick hand-overs commit in publication order, not Black first (reported under same_tick_ties / "
                "b1_white_published_first_same_tick)"],
        },
    }


# -- request lifecycle and waste (spec sections 14, 27) ------------------------------------------


def targeted_race() -> list[dict[str, Any]]:
    """M6G.1's race timings forced through both loops (zero calls): the
    second answer lands after its 0.5 s window, next to the first decision's
    commit (route 1) or after an in-window poll saw it (route 2)."""
    from src.evaluation.barrier_probe import run_production_trial, scripted_requests
    from src.evaluation.commitment_policy import build_policy_scenarios
    from src.evaluation.run_m6g3_replay import MODEL_ID, REGION
    from src.game import COMMIT_POLICY_ANSWER_ORDER, COMMIT_POLICY_BARRIER

    s = build_policy_scenarios()[C2]
    out = []
    for runtime in ("native_tool_like", "strands"):
        backend = ({"provider": "bedrock", "model_id": MODEL_ID, "region": REGION} if runtime == "strands"
                   else {"provider": "mock", "model_id": "mock-model", "region": None})
        for route, gap in (("route1_answer_next_to_change", 0), ("route2_throttle", 1)):
            for policy in (COMMIT_POLICY_ANSWER_ORDER, COMMIT_POLICY_BARRIER):
                kwargs, gates = scripted_requests(runtime, {a: [[s.scripted_raw(a)]] for a in ACTORS}, delays={(ACTOR_B, 1): 0.6})
                row = run_production_trial(s, trial_id=f"targeted:{runtime}:{route}:{policy}", runtime=runtime, commit_policy=policy,
                                           gates=gates, ready_schedule=[[(ACTOR_A, 1)], [(ACTOR_B, 1)]], gap_ticks=gap, absorb=False,
                                           **backend, **kwargs)
                lc = row["lifecycle"]
                out.append({"runtime": "strands" if runtime == "strands" else "native", "route": route, "commit_policy": policy,
                            "requests_launched": lc["requests_launched"], "duplicate_requests": lc["duplicate_requests"],
                            "responses_consumed": lc["responses_consumed"], "unused_responses": lc["unused_responses"],
                            "unused_output_tokens": lc["unused_output_tokens"],
                            "live_ticks_while_waiting": row["freeze_audit"]["ticks_advanced_while_waiting"], "model_calls": 0})
    return out


def _request_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        for a in ACTORS:
            for r in row["requests"][a]:
                out.append({"trial_id": row["trial_id"], "scenario_id": row["scenario_id"], "actor_id": a, **{k: r.get(k) for k in (
                    "request_index", "decision_boundary_id", "prior_state", "hp09_violation", "beyond_horizon", "ready", "consumed",
                    "cancelled", "unused", "in_flight_at_end", "model_calls", "input_tokens", "output_tokens", "ready_rank", "commit_rank",
                    "request_start_wall", "ready_wall", "launch_tick", "handed_over_tick", "committed_tick")},
                            "game_time_before_wait": (r.get("record") or {}).get("game_time_before_wait"),
                            "game_time_after_wait": (r.get("record") or {}).get("game_time_after_wait"),
                            "game_time_waited_s": (r.get("record") or {}).get("game_time_waited_s")})
    return out


def lifecycle_table(live: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Spec section 46: one row per runtime/trial/boundary/actor."""
    table = []
    for rk, rows in live.items():
        for row in rows:
            groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
            for r in _request_rows([row]):
                groups[(r["decision_boundary_id"], r["actor_id"])].append(r)
            for (bid, a), reqs in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, ACTORS.index(kv[0][1]))):
                table.append({"runtime": rk, "trial_id": row["trial_id"], "decision_boundary_id": bid, "actor_id": a,
                              "requests_launched": len(reqs), "responses_ready": sum(r["ready"] for r in reqs),
                              "responses_consumed": sum(r["consumed"] for r in reqs), "unused_responses": sum(bool(r["unused"]) for r in reqs),
                              "duplicate_request": any(r["hp09_violation"] for r in reqs), "beyond_horizon": all(r["beyond_horizon"] for r in reqs),
                              "cancelled": sum(r["cancelled"] for r in reqs)})
    return table


def request_waste(live: dict[str, list[dict[str, Any]]], targeted: list[dict[str, Any]]) -> dict[str, Any]:
    by_runtime = {}
    for rk, rows in live.items():
        if not rows:
            continue
        reqs = _request_rows(rows)
        horizon = [r for r in reqs if not r["beyond_horizon"]]
        beyond = [r for r in reqs if r["beyond_horizon"]]
        by_runtime[rk] = {
            "trials": len(rows), "boundaries": sum(len(r["boundaries"]) for r in rows),
            "paired_boundaries": sum(b["pair"] for r in rows for b in r["boundaries"]),
            "requests_launched": len(horizon), "duplicate_requests": sum(bool(r["hp09_violation"]) for r in horizon),
            "responses_ready": sum(r["ready"] for r in horizon), "responses_consumed": sum(r["consumed"] for r in horizon),
            "responses_cancelled": sum(r["cancelled"] for r in horizon), "unused_responses": sum(bool(r["unused"]) for r in horizon),
            "in_flight_at_end": sum(r["in_flight_at_end"] for r in horizon),
            "provider_calls": sum(r["model_calls"] for r in reqs), "provider_calls_in_horizon": sum(r["model_calls"] for r in horizon),
            "input_tokens": sum(r["input_tokens"] or 0 for r in reqs), "output_tokens": sum(r["output_tokens"] or 0 for r in reqs),
            "unused_input_tokens": sum(r["input_tokens"] or 0 for r in horizon if r["unused"]),
            "unused_output_tokens": sum(r["output_tokens"] or 0 for r in horizon if r["unused"]),
            "beyond_horizon": {"requests": len(beyond), "consumed": sum(r["consumed"] for r in beyond),
                               "ready_not_consumed_at_trial_end": sum(r["ready"] and not r["consumed"] for r in beyond),
                               "duplicate_requests": sum(bool(r["hp09_violation"]) for r in beyond), "provider_calls": sum(r["model_calls"] for r in beyond),
                               "output_tokens": sum(r["output_tokens"] or 0 for r in beyond)},
            "trials_with_duplicates": sum(r["lifecycle"]["duplicate_requests"] > 0 for r in rows),
            "prior_state_at_launch": dict(sorted(Counter(r["prior_state"] for r in reqs).items())),
        }
    before = {}
    waste2 = os.path.join(M6G2_DIR, "request_waste_metrics.json")
    if os.path.exists(waste2):
        with open(waste2, encoding="utf-8") as f:
            m2 = json.load(f)
        for rk in RUNTIME_KEYS:
            p0 = m2["by_runtime"].get(rk, {}).get(P0)
            if p0:
                before[f"m6g2_live_p0:{rk}"] = {k: p0[k] for k in ("trials", "requests_launched", "extra_requests", "hp09_violations",
                                                                     "unused_responses", "unused_output_tokens")}
    for rk, fname in (("native", "real_loop_trials.jsonl"), ("strands", "strands_trials.jsonl")):
        rows = read_jsonl(os.path.join(M6G1_DIR, fname))
        if rows:
            before[f"m6g1_live_p0:{rk}"] = {"trials": len(rows), "trials_with_extra_request": sum(r["extra_requests_before_pair_complete"] > 0 for r in rows),
                                            "extra_requests": sum(r["extra_requests_before_pair_complete"] for r in rows)}
    legacy = [t for t in targeted if t["commit_policy"] == "answer_order"]
    promoted = [t for t in targeted if t["commit_policy"] == "barrier_answer_order"]
    live_ok = all(v["duplicate_requests"] == 0 and v["unused_responses"] == 0 and v["beyond_horizon"]["duplicate_requests"] == 0
                  for v in by_runtime.values())
    targeted_ok = (all(t["duplicate_requests"] == 0 and t["unused_responses"] == 0 for t in promoted)
                   and all(t["duplicate_requests"] == 1 and t["unused_responses"] == 1 for t in legacy))
    return {
        "by_runtime": by_runtime, "before": before, "targeted_race": targeted,
        "gate": {"request_lifecycle_pass": live_ok and targeted_ok and bool(by_runtime), "live_pass": live_ok, "targeted_pass": targeted_ok,
                 "rule": "zero requests launched while a decision was ready-uncommitted or held, and zero unused responses within the "
                         "trial horizon, live and in the targeted race; the legacy loop must still show the race (the test discriminates)"},
        "notes": {
            "beyond_horizon": "a request an actor launches after its own K-th commit belongs to a boundary past the trial horizon; it is "
                              "reported separately (consumed within the trial, or ready/in flight when the trial ended), never as unused",
            "before": "M6G.2 live P0 had no race in 20 trials; M6G.1 saw it live in 2/20 Strands trials; the forced targeted race is the "
                      "discriminating evidence (spec section 16)",
        },
    }


# -- wall clock and game time (spec sections 12-13, 28, 30) -----------------------------------------

WALL_KEYS = ("to_first_ready", "to_both_ready", "to_first_commit", "to_pair_commit")


def _b0_wall(rows: list[dict[str, Any]], policy: str, sid: str) -> list[dict[str, float]]:
    return [r["boundaries"][0]["wall_ms"] for r in rows if r["commitment_policy"] == policy and r["scenario_id"] == sid
            and r["boundaries"] and r["boundaries"][0].get("wall_ms")]


def wall_clock(live: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    m2 = {rk: read_jsonl(os.path.join(M6G2_DIR, FILES[rk])) for rk in RUNTIME_KEYS}
    out: dict[str, Any] = {}
    for rk, rows in live.items():
        if not rows:
            continue
        per = {}
        for sid in (C2, C0):
            after = [r["boundaries"][0]["wall_ms"] for r in rows if r["scenario_id"] == sid and r["boundaries"] and r["boundaries"][0]["pair"]]
            before = _b0_wall(m2[rk], P0, sid)
            ref = _b0_wall(m2[rk], P1, sid)

            def block(ws):
                d = {f"{k}_ms": _stats([w[k] for w in ws]) for k in WALL_KEYS}
                d["first_commit_after_first_ready_ms"] = _stats([w["to_first_commit"] - w["to_first_ready"] for w in ws])
                d["pair_commit_after_both_ready_ms"] = _stats([w["to_pair_commit"] - w["to_both_ready"] for w in ws])
                return d

            entry = {"after_m6g3_p1_production_b1": block(after), "before_m6g2_p0_live_b0": block(before) if before else None,
                     "reference_m6g2_p1_evaluation_barrier_b0": block(ref) if ref else None}
            if before and after:
                a, b = entry["after_m6g3_p1_production_b1"], entry["before_m6g2_p0_live_b0"]
                entry["median_delta_after_minus_before_ms"] = {
                    k: round(a[f"{k}_ms"]["median"] - b[f"{k}_ms"]["median"], 1) for k in WALL_KEYS}
                entry["median_commit_lag_delta_ms"] = round(a["first_commit_after_first_ready_ms"]["median"]
                                                            - b["first_commit_after_first_ready_ms"]["median"], 1)
            per[sid] = entry
        committed = [rec for r in rows for b in r["boundaries"] for rec in b.get("records", {}).values()]
        per["game_time"] = {
            "live_ticks_while_a_decision_was_outstanding": sum(r["freeze_audit"]["ticks_advanced_while_waiting"] for r in rows),
            "game_time_advanced_while_waiting_s": round(sum(r["freeze_audit"]["game_time_advanced_while_waiting_s"] for r in rows), 6),
            "waiting_ticks": sum(r["freeze_audit"]["waiting_ticks"] for r in rows), "held_ticks": sum(r["freeze_audit"]["held_ticks"] for r in rows),
            "committed_decisions": len(committed), "committed_with_zero_game_time_wait": sum(rec["game_time_waited_s"] == 0.0 for rec in committed),
            "max_game_time_waited_s": max((rec["game_time_waited_s"] for rec in committed), default=None),
        }
        out[rk] = per
    gt_ok = bool(out) and all(v["game_time"]["live_ticks_while_a_decision_was_outstanding"] == 0
                              and v["game_time"]["committed_with_zero_game_time_wait"] == v["game_time"]["committed_decisions"]
                              for v in out.values())
    return {
        "by_runtime": out,
        "gate": {"game_time_freeze_pass": gt_ok,
                 "rule": "no live tick while any decision was in flight, ready-uncommitted or held; every committed decision waited 0 s "
                         "of game time between its request and its commit"},
        "notes": {
            "before": "M6G.2 live P0 boundary 0 on the same runtime and scenario (a different day's provider latency); the commit lag "
                      "(first commit minus first ready) isolates the scheduler's share",
            "expected_tradeoff": "under the barrier the first commit waits for the slower response: first_commit_after_first_ready "
                                 "~= the gap between the two answers; P0 committed the first answer within a tick",
        },
    }


# -- ready order (spec sections 10-11, 29) -----------------------------------------------------------


def ready_order(live: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for rk, rows in live.items():
        if not rows:
            continue
        pairs = [(r["trial_id"], b) for r in rows for b in r["boundaries"] if b["pair"]]
        out[rk] = {
            "paired_boundaries": len(pairs), "first_ready_is_first_commit": sum(b["first_ready_is_first_commit"] for _, b in pairs),
            "exact_ready_ties": sum(b["exact_ready_tie"] for _, b in pairs), "same_tick_handovers": sum(b["handed_over_same_tick"] for _, b in pairs),
            "violations": [f"{t}:b{b['decision_boundary_id']}" for t, b in pairs if not b["first_ready_is_first_commit"] and not b["exact_ready_tie"]],
            "first_ready_actor": dict(Counter(b["ready_order"][0] for _, b in pairs)),
            "b1_first_commit_actor": dict(Counter(r["boundaries"][0]["commit_order"][0] for r in rows if r["boundaries"])),
            "single_actor_boundaries": sum(not b["pair"] for r in rows for b in r["boundaries"]),
        }
    ok = bool(out) and all(not v["violations"] for v in out.values())
    return {"by_runtime": out, "gate": {"ready_order_preserved": ok,
                                        "rule": "first-ready (response_ready_at) actor == first-commit actor on every paired boundary, "
                                                "except an exact timestamp tie"}}


# -- runtime validation (spec sections 22-23) ------------------------------------------------------


def runtime_validation(live: dict[str, list[dict[str, Any]]], events: list[dict[str, Any]], sem: dict, waste: dict, clock: dict,
                       order: dict) -> dict[str, Any]:
    gates = [e for e in events if e.get("phase") == "runtime_parity_gate_rp01_rp18"]
    gate = gates[-1] if gates else None
    per = {}
    for rk in RUNTIME_KEYS:
        rows = live.get(rk) or []
        planned = len(planned_trials(rk))
        w, s, c, o = (waste["by_runtime"].get(rk), sem["live"].get(rk), clock["by_runtime"].get(rk), order["by_runtime"].get(rk))
        checks = {
            "all_planned_trials_complete": len(rows) == planned and all(r["complete"] for r in rows),
            "no_duplicate_requests": bool(w) and w["duplicate_requests"] == 0 and w["beyond_horizon"]["duplicate_requests"] == 0,
            "no_unused_responses": bool(w) and w["unused_responses"] == 0,
            "zero_game_time_wait": bool(c) and c["game_time"]["live_ticks_while_a_decision_was_outstanding"] == 0
                                   and c["game_time"]["committed_with_zero_game_time_wait"] == c["game_time"]["committed_decisions"],
            "ready_order_preserved": bool(o) and not o["violations"],
            "b1_world_matches_p1_expectation": bool(s) and s["b1_reproduced"] == s["trials"] and not s["b1_unexplained_legacy_differences"],
        }
        if rk == "strands":
            checks["rp01_rp18_gate_clean"] = bool(gate and gate.get("ok"))
        model_errors = Counter(e for r in rows for a in ACTORS for req in r["requests"][a] for e in req["provider_error_types"])
        per[rk] = {"trials": len(rows), "planned": planned, "checks": checks, "validated": bool(rows) and all(checks.values()),
                   "model_output_errors": dict(model_errors),
                   "first_pass_invalid_committed": sum(1 for r in rows for b in r["boundaries"] for rec in b.get("records", {}).values()
                                                       if not rec["first_pass_valid"])}
    if not (per["native"]["trials"] and per["strands"]["trials"]):
        replication = "NOT_CLASSIFIED_ONE_RUNTIME_ONLY"
    elif per["native"]["validated"] and per["strands"]["validated"]:
        replication = "P1_PRODUCTION_BEHAVIOR_RUNTIME_INDEPENDENT"
    else:
        replication = "P1_PRODUCTION_BEHAVIOR_RUNTIME_DEPENDENT"
    return {"rp01_rp18_gate": gate, "by_runtime": per, "replication": replication,
            "rule": "the same gate results on both runtimes -> RUNTIME_INDEPENDENT; a gate that holds on one runtime only -> _DEPENDENT"}


# -- regression (spec section 42) ------------------------------------------------------------------


def regression(log_path: str | None) -> dict[str, Any] | None:
    if not log_path:
        return None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    tail = [ln for ln in text.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    last = tail[-1] if tail else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", last)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", last)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) errors?", last)) else 0
    failures = sorted(set(re.findall(r"^FAILED (\S+)", text, flags=re.M)) | set(re.findall(r"^ERROR (\S+)", text, flags=re.M)))
    new = [f for f in failures if not any(re.match(p, f) for p in KNOWN_FAILURE_PATTERNS)]
    return {"passed": passed, "failed": failed, "errors": errors, "total": passed + failed + errors, "summary_line": last.strip(" ="),
            "failures": failures, "known_failures": [f for f in failures if f not in new], "new_failures": new,
            "baseline": "1848/1854 (6 known environment-only failures)"}


# -- decision and tables (spec sections 44-49) -----------------------------------------------------------


def decide(sem, waste, clock, order, rv, reg) -> dict[str, Any]:
    semantic = sem["gate"]["semantic_equivalence_pass"]
    lifecycle = waste["gate"]["request_lifecycle_pass"]
    game_time = clock["gate"]["game_time_freeze_pass"]
    ready = order["gate"]["ready_order_preserved"]
    native, strands = rv["by_runtime"]["native"]["validated"], rv["by_runtime"]["strands"]["validated"]
    new_regressions = None if reg is None else len(reg["new_failures"])
    pending = [k for k, v in (("strands_live", rv["by_runtime"]["strands"]["trials"] > 0), ("regression_log", reg is not None)) if not v]
    if not semantic:
        cls = SEMANTIC_FAILED
    elif not (lifecycle and game_time):
        cls = LIFECYCLE_NOT_CONFIRMED
    elif pending:
        cls = None
    elif ready and native and strands and new_regressions == 0:
        cls = CONFIRMED
    elif native != strands:
        cls = MIXED
    else:
        cls = BLOCKED
    return {"classification": cls, "pending": pending, "semantic": semantic, "lifecycle": lifecycle, "game_time": game_time, "ready": ready,
            "native": native, "strands": strands, "new_regressions": new_regressions}


def promotion_decision(d: dict, sem: dict, waste: dict, clock: dict, order: dict, rv: dict, reg: dict | None) -> dict[str, Any]:
    c2 = {rk: v.get(C2, {}) for rk, v in clock["by_runtime"].items()}
    lag = {rk: (v.get("after_m6g3_p1_production_b1", {}).get("first_commit_after_first_ready_ms", {}).get("median"),
                (v.get("before_m6g2_p0_live_b0") or {}).get("first_commit_after_first_ready_ms", {}).get("median")) for rk, v in c2.items()}
    ties = sem["replay"]["same_tick_ties"]
    return {
        "candidate": CANDIDATE,
        "production_promoted": d["classification"] == CONFIRMED,
        "classification": d["classification"],
        "pending": d["pending"],
        "semantic_equivalence_pass": d["semantic"],
        "request_lifecycle_pass": d["lifecycle"],
        "game_time_freeze_pass": d["game_time"],
        "ready_order_preserved": d["ready"],
        "native_validated": d["native"],
        "strands_validated": d["strands"],
        "new_regressions": d["new_regressions"],
        "regression": reg,
        "production_default": "Game(commit_policy='barrier_answer_order'); legacy P0 = Game(commit_policy='answer_order')",
        "known_tradeoffs": [
            "the first visible action waits for the slower response: median first-commit lag after the first answer "
            + ", ".join(f"{rk} C2 {a} ms (P0 before: {b} ms)" for rk, (a, b) in lag.items()),
            f"answers that reach the loop on one tick commit in publication order, not Black first (spec sections 10-11): "
            f"{ties['true_order_white_first']}/{ties['true_order_rows']} captured same-tick ties had White published first, "
            f"{ties['world_differs_from_p0']} of them changed the shared door's owner vs P0",
            "a MOVE committed first moves on the release tick instead of sitting one tick COMMITTED_FROZEN; the world after the pair is "
            f"identical ({sum(x['explanation'] == MOVE_FIRST_EXPLANATION for x in sem['replay']['supporting_check_differences'])}"
            f"/{sem['replay']['pairs']} replay pairs)",
            "a held actor is not polled: its memory and MOVE executor see the other actor's effect only after its own commit "
            "(telemetry and later AgentRequests, not the pair outcome)",
        ],
        "known_open_risks": [
            "stale decisions are unchanged: the second decision of a shared-state conflict is still applied to a changed world without "
            "revalidation (no STALE_REVALIDATE / STALE_CANCEL / STALE_REDECIDE)",
            "the no-path MOVE offered as legal (M6G.2) is unchanged and separate",
            "after the first boundary, decisions desynchronize (different action costs): most later boundaries hold one actor; an "
            "actor free again may launch its next request while the other is still busy (beyond-horizon requests here)",
            "a slow or hung provider call holds the other actor's decision until it answers or llm_timeout expires (zero game time, "
            "wall-clock only)",
            "the barrier applies only between two LLM agents in evaluation mode; single-LLM and playable games keep per-decision "
            "commits (the HP09 guard still applies there)",
            "the ready-response cancellation path (busy/stunned at hand-over) is exercised by tests only",
        ],
        "evidence": {
            "replay_pairs_equivalent": f"{sem['replay']['equivalent_pairs']}/{sem['replay']['pairs']}",
            "live_trials": {rk: v["trials"] for rk, v in rv["by_runtime"].items()},
            "live_duplicates": {rk: v["duplicate_requests"] for rk, v in waste["by_runtime"].items()},
            "live_unused": {rk: v["unused_responses"] for rk, v in waste["by_runtime"].items()},
            "targeted_race": {f"{t['runtime']}:{t['route']}:{t['commit_policy']}": f"dup {t['duplicate_requests']}, unused {t['unused_responses']}"
                              for t in waste["targeted_race"]},
            "rp01_rp18_gate_ok": (rv["rp01_rp18_gate"] or {}).get("ok"),
        },
        "next_step": ("M6 FINAL SYNTHESIS + AGENT ENGINEERING PLAYBOOK (not started automatically)" if d["classification"] == CONFIRMED
                      else "do not promote; isolate the smallest failing scheduler invariant"),
        "scheduling_track": "CLOSED for workshop scope" if d["classification"] == CONFIRMED else "OPEN",
    }


def _md(rows: list[list[Any]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tables(sem, waste, clock, order, rv, live, replay) -> str:
    parts = []
    # section 45 - before / after
    def med(which: str, key: str) -> str:
        vals = []
        for rk in RUNTIME_KEYS:
            blk = clock["by_runtime"].get(rk, {}).get(C2, {}).get(which) or {}
            m = (blk.get(key) or {}).get("median")
            vals.append(f"{'Native' if rk == 'native' else 'Strands'} {m} ms")
        return ", ".join(vals)

    ties = sem["replay"]["same_tick_ties"]
    tgt = {(t["runtime"], t["route"], t["commit_policy"]): t for t in waste["targeted_race"]}
    legacy_dup = sum(t["duplicate_requests"] for t in waste["targeted_race"] if t["commit_policy"] == "answer_order")
    promoted_dup = sum(t["duplicate_requests"] for t in waste["targeted_race"] if t["commit_policy"] == "barrier_answer_order")
    live_dup = sum(v["duplicate_requests"] for v in waste["by_runtime"].values())
    live_unused = sum(v["unused_responses"] for v in waste["by_runtime"].values())
    gt = {rk: v["game_time"] for rk, v in clock["by_runtime"].items()}
    ba = [
        ["snapshot semantics", "both actors decide from one pre-action snapshot (boundary 1)", "same", "no"],
        ["decision outputs", "one decision per request", f"same (request hashes equal pre/post on {sem['replay']['check_pass_counts'].get('requests', 0)}/{sem['replay']['pairs']} replay pairs)", "no"],
        ["commit order", "first answer the loop sees first; same-tick hand-overs Black first",
         "first-ready first (response_ready_at); same-tick hand-overs in publication order; exact tie Black first",
         f"only for same-tick hand-overs ({ties['true_order_white_first']}/{ties['true_order_rows']} captured ties White-first)"],
        ["world outcome (same ready order)", "-", f"identical on {sem['replay']['equivalent_pairs']}/{sem['replay']['pairs']} replay pairs; live b1 "
         + ", ".join(f"{rk} {v['b1_reproduced']}/{v['trials']}" for rk, v in sem["live"].items()), "no"],
        ["duplicate-request possibility", f"yes: targeted race {legacy_dup} duplicates in {len([t for t in waste['targeted_race'] if t['commit_policy'] == 'answer_order'])} forced runs; live M6G.1 Strands 2/20",
         f"no: targeted {promoted_dup}; live {live_dup}", "yes"],
        ["unused-response possibility", "yes: the replacement request's answer (1 per forced race run)",
         f"no: targeted {sum(t['unused_responses'] for t in waste['targeted_race'] if t['commit_policy'] == 'barrier_answer_order')}; "
         f"live {live_unused}", "yes"],
        ["wall time to first commit (C2 boundary 1, median)", med("before_m6g2_p0_live_b0", "to_first_commit_ms") + " (M6G.2 live P0)",
         med("after_m6g3_p1_production_b1", "to_first_commit_ms") + " (waits for the slower answer)", "yes (slower)"],
        ["first commit after the first answer (C2, median)", med("before_m6g2_p0_live_b0", "first_commit_after_first_ready_ms"),
         med("after_m6g3_p1_production_b1", "first_commit_after_first_ready_ms"), "yes"],
        ["wall time to pair commit (C2, median)", med("before_m6g2_p0_live_b0", "to_pair_commit_ms"),
         med("after_m6g3_p1_production_b1", "to_pair_commit_ms"), "no (set by the slower answer either way)"],
        ["game-time advancement while deciding", "0 while a request is in flight (M5F freeze); an answer published after its own "
         "poll could let one live tick run before its hand-over (M6G.1's handoff tick)",
         "0 while in flight, ready-uncommitted or held: " + ", ".join(f"{rk} {v['live_ticks_while_a_decision_was_outstanding']} live ticks" for rk, v in gt.items()),
         "yes: the hand-over tick is frozen too"],
        ["stale-decision semantics", "second decision applied to the changed world, no revalidation", "same", "no"],
    ]
    parts.append("### Before / after (spec section 45)\n\n" + _md(ba, ["property", "before (P0)", "after (P1 production)", "changed?"]))
    # section 46 - lifecycle (aggregate + targeted)
    agg = [[rk, v["trials"], v["boundaries"], v["paired_boundaries"], v["requests_launched"], v["responses_ready"], v["responses_consumed"],
            v["unused_responses"], v["duplicate_requests"], v["beyond_horizon"]["requests"], v["provider_calls"]] for rk, v in waste["by_runtime"].items()]
    parts.append("### Request lifecycle (spec section 46)\n\nLive, per runtime (requests within each trial's horizon; beyond-horizon "
                 "requests counted separately):\n\n" + _md(agg, ["runtime", "trials", "boundaries", "paired", "requests launched",
                                                              "responses ready", "responses consumed", "unused", "duplicates",
                                                              "beyond horizon", "provider calls"]))
    trows = [[t["runtime"], t["route"], t["commit_policy"], t["requests_launched"], t["responses_consumed"], t["unused_responses"],
              t["duplicate_requests"], t["live_ticks_while_waiting"]] for t in waste["targeted_race"]]
    parts.append("Targeted race (zero calls, M6G.1 timings forced):\n\n" + _md(trows, ["runtime", "route", "commit policy", "requests launched",
                                                                                     "responses consumed", "unused", "duplicates", "live ticks waiting"]))
    # section 47 - replay (grouped)
    eq = [r for r in replay if r["kind"] == KIND_EQUIVALENCE]
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in eq:
        order_s = "tie" if r["response_ready_order"] is None else "B→W" if r["response_ready_order"][0] == ACTOR_A else "W→B"
        groups[(r["source"], r["runtime"], r["scenario_id"], order_s, r["pre"]["world_hash_after_pair"], r["post"]["world_hash_after_pair"])].append(r)
    rrows = []
    for (src, rt, sid, o, pre_h, post_h), rs in sorted(groups.items(), key=lambda kv: kv[0][:4]):
        rrows.append([f"{src} ({len(rs)})", rt, sid, o, _short(pre_h), _short(post_h),
                      _pct(sum(r["checks"]["pair_event_hash"] for r in rs), len(rs)), _pct(sum(r["checks"]["costs"] for r in rs), len(rs)),
                      _pct(sum(r["checks"]["commit_order"] for r in rs), len(rs))])
    parts.append("### Replay equivalence (spec section 47)\n\nGrouped by source, runtime, scenario, ready order and world hashes "
                 "(pre = legacy loop, post = promoted loop; every row's P0 replay hash equals both):\n\n"
                 + _md(rrows, ["source trials (n)", "runtime", "scenario", "ready order", "pre-promotion world", "post-promotion world",
                               "event hash equal", "costs equal", "commit order equal"]))
    # section 48 - live
    lrows = []
    for rk, rows in live.items():
        for r in rows:
            b1 = r["boundaries"][0] if r["boundaries"] else {}
            waited = sum(v for b in r["boundaries"] for v in b.get("game_time_waited_s", {}).values())
            lrows.append([rk, r["scenario_id"], r["trial_id"], (b1.get("ready_order") or ["-"])[0], (b1.get("commit_order") or ["-"])[0],
                          r["lifecycle"]["duplicate_requests"], r["lifecycle"]["unused_responses"], f"{waited:.3f} s",
                          f"{[tuple(e) for e in b1.get('contested_effects', [])] or 'no contest'}; repro {r.get('b1_reproduced')}, "
                          f"= legacy {r.get('b1_legacy_equal')}", len(r["boundaries"]), r["lifecycle"]["beyond_horizon_requests"]])
    parts.append("### Live validation (spec section 48)\n\n" + _md(lrows, ["runtime", "scenario", "trial", "first ready (b1)", "first commit (b1)",
                                                                        "extra requests", "unused responses", "game-time wait",
                                                                        "world outcome (b1)", "boundaries", "beyond horizon"]))
    # appendix - lifecycle per actor and boundary (section 46 in full)
    lt = lifecycle_table(live)
    arows = [[t["runtime"], t["trial_id"], t["decision_boundary_id"], t["actor_id"], t["requests_launched"], t["responses_ready"],
              t["responses_consumed"], t["unused_responses"], "yes" if t["duplicate_request"] else "no", "yes" if t["beyond_horizon"] else ""]
             for t in lt]
    parts.append("### Appendix: lifecycle per trial, boundary and actor\n\n" + _md(arows, ["runtime", "trial", "boundary", "actor",
                                                                                          "requests launched", "responses ready",
                                                                                          "responses consumed", "unused", "duplicate?",
                                                                                          "beyond horizon"]))
    return "\n\n".join(parts)


def splice(md_path: str, block: str) -> None:
    body = f"{TABLES_BEGIN}\n\n{block}\n\n{TABLES_END}"
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            text = f.read()
        if TABLES_BEGIN in text and TABLES_END in text:
            text = text[:text.index(TABLES_BEGIN)] + body + text[text.index(TABLES_END) + len(TABLES_END):]
        else:
            text = text.rstrip() + "\n\n## Tables\n\n" + body + "\n"
    else:
        text = "# M6G.3 — Response-Barrier Production Promotion & Episode Validation\n\n## Tables\n\n" + body + "\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)


def run(regression_log: str | None = None) -> dict[str, Any]:
    replay = read_jsonl(out_path("replay_equivalence.jsonl"))
    live = {rk: read_jsonl(out_path(FILES[rk])) for rk in RUNTIME_KEYS}
    events = read_jsonl(out_path("provider_events.jsonl"))
    sem = semantic_equivalence(replay, live)
    targeted = targeted_race()
    waste = request_waste(live, targeted)
    clock = wall_clock(live)
    order = ready_order(live)
    rv = runtime_validation(live, events, sem, waste, clock, order)
    reg = regression(regression_log)
    d = decide(sem, waste, clock, order, rv, reg)
    decision = promotion_decision(d, sem, waste, clock, order, rv, reg)
    write_json(out_path("semantic_equivalence_metrics.json"), sem)
    write_json(out_path("request_waste_metrics.json"), waste | {"lifecycle_table": lifecycle_table(live)})
    write_json(out_path("wall_clock_metrics.json"), clock)
    write_json(out_path("runtime_validation.json"), rv | {"ready_order": order})
    write_json(out_path("promotion_decision.json"), decision)
    summary = {
        "milestone": "M6G.3", "candidate": CANDIDATE, "classification": d["classification"], "pending": d["pending"],
        "gates": {k: decision[k] for k in ("semantic_equivalence_pass", "request_lifecycle_pass", "game_time_freeze_pass", "ready_order_preserved",
                                           "native_validated", "strands_validated", "new_regressions")},
        "replay": {k: sem["replay"][k] for k in ("pairs", "equivalent_pairs", "all_checks_pass_pairs", "model_calls", "by_source")}
                  | {"m6g2_recorded_p0": sem["replay"]["m6g2_recorded_p0"],
                     "same_tick_ties": {k: v for k, v in sem["replay"]["same_tick_ties"].items() if k not in ("rows", "rule")}},
        "live": {rk: {"trials": v["trials"], "requests_launched": v["requests_launched"], "duplicates": v["duplicate_requests"],
                      "unused": v["unused_responses"], "beyond_horizon_requests": v["beyond_horizon"]["requests"], "provider_calls": v["provider_calls"],
                      "paired_boundaries": v["paired_boundaries"], "boundaries": v["boundaries"]} for rk, v in waste["by_runtime"].items()},
        "ready_order": {rk: {k: v[k] for k in ("paired_boundaries", "first_ready_is_first_commit", "exact_ready_ties", "same_tick_handovers")}
                        for rk, v in order["by_runtime"].items()},
        "wall_clock_c2_median_ms": {rk: {"after": {k: (v.get(C2, {}).get("after_m6g3_p1_production_b1", {}).get(f"{k}_ms") or {}).get("median") for k in WALL_KEYS},
                                         "before": {k: ((v.get(C2, {}).get("before_m6g2_p0_live_b0") or {}).get(f"{k}_ms") or {}).get("median") for k in WALL_KEYS}}
                                    for rk, v in clock["by_runtime"].items()},
        "game_time": {rk: v["game_time"] for rk, v in clock["by_runtime"].items()},
        "rp01_rp18_gate_ok": (rv["rp01_rp18_gate"] or {}).get("ok"), "runtime_replication": rv["replication"],
        "regression": None if reg is None else {k: reg[k] for k in ("passed", "total", "summary_line", "new_failures")},
    }
    write_json(out_path("summary.json"), summary)
    splice(out_path("summary.md"), tables(sem, waste, clock, order, rv, live, replay))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-log", default=None, help="pytest -rf output of the full regression run")
    args = parser.parse_args()
    summary = run(args.regression_log)
    print(json.dumps({k: summary[k] for k in ("classification", "pending", "gates", "live", "ready_order", "rp01_rp18_gate_ok")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
