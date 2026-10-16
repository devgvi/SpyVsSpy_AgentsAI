"""M6G.3 golden replay equivalence (spec sections 17-18): every cheap
captured decision pair through the PROMOTED production scheduler, zero
model calls.

For each pair (identical world state, accepted decision pair and
response-ready order) three pre/post measurements are compared:

    post  the production Game loop under the default commit policy (P1
          BARRIER_ANSWER_ORDER + the HP09 guard), scripted stand-ins
          answering the captured decisions in the captured ready order
    pre   the same production loop under the legacy "answer_order" policy
          (the pre-promotion loop, P0), same stand-ins and ready order
    P0    the pre-promotion replay semantics (commitment_policy.replay_under_policy
          P0 = M6G.1's T0), plus the P0 row M6G.2 recorded for the pair

Required equivalence (spec section 17): world outcome, event hash, action
costs and commit order identical; the decisions and the snapshot they were
made from are checked too.

Corpus (read-only):

    scripted            the 8 M6G.2 scenarios' fixed pairs, both ready orders
    m6g_reuse           M6G's 110 live pairs (logical arrival order)
    m6g1_live           M6G.1's 40 live real-loop pairs (observed ready order)
    m6g1_scripted_loop  M6G.1's 24 scripted real-loop pairs (every L scenario)
    m6g2_live           M6G.2's 60 live trials, boundary 0 (loop-visible order)

A pair whose two answers became visible on one tick (a "tie") is replayed
with both made ready on one tick, Black published first - P0's same-tick
rule. Where the tie's true sub-tick order is known it is also replayed in
that order: the promoted barrier commits the earlier-published answer
first (spec sections 10-11; no actor priority), which is the one intended
difference from P0 and is reported separately, never counted as an
equivalence failure.

Writes replay_manifest.json (before replaying) and replay_equivalence.jsonl.

    python -m src.evaluation.run_m6g3_replay
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.evaluation.barrier_probe import run_production_trial, scripted_requests
from src.evaluation.commitment_policy import P0, SCENARIO_IDS, build_policy_scenarios, replay_under_policy
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS
from src.evaluation.m6g_live import read_jsonl, write_json
from src.game import COMMIT_POLICY_ANSWER_ORDER, COMMIT_POLICY_BARRIER

OUT_DIR = os.path.join("evaluation_runs", "m6g3-response-barrier-promotion-v1")
M6G2_DIR = os.path.join("evaluation_runs", "m6g2-commitment-policy-counterbalancing-v1")
MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"
KIND_EQUIVALENCE = "equivalence"
KIND_TIE_TRUE_ORDER = "tie_true_order"
PAIR_KEYS = ("world_hash_before", "world_hash_after_first_commit", "world_hash_after_pair", "effect_state_hash_after_pair",
             "event_log_hash_pair", "contested_effects", "first_action_cost", "second_action_cost", "first_result", "second_result")


def out_path(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def _probe_runtime(runtime: str) -> str:
    return "strands" if runtime == "strands" else "native_tool_like"


def _backend(runtime: str) -> dict[str, Any]:
    # scripted stand-ins answer every call; Strands still needs its Bedrock model object built (no call is made)
    if runtime == "strands":
        return {"provider": "bedrock", "model_id": MODEL_ID, "region": REGION}
    return {"provider": "mock", "model_id": "mock-model", "region": None}


def _order(value) -> list[str] | None:
    return list(value) if value else None


def corpus() -> list[dict[str, Any]]:
    """Every captured pair with its provenance, raws, loop-visible ready
    order (None = same-tick tie) and, where known, the true ready order."""
    from src.evaluation.run_m6g2_replay import captured_pairs

    scenarios = build_policy_scenarios()
    pairs: list[dict[str, Any]] = []
    for sid in SCENARIO_IDS:
        s = scenarios[sid]
        for order in ((ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A)):
            pairs.append({
                "pair_id": f"scripted:{sid}:{order[0]}_ready_first", "source": "scripted", "source_milestone": "M6G.2",
                "source_file": "src/evaluation/commitment_policy.py (build_policy_scenarios)", "source_trial_id": f"scripted:{sid}",
                "runtime": "native", "scenario_id": sid, "raws": {a: s.scripted_raw(a) for a in ACTORS},
                "response_ready_order": list(order), "true_ready_order": list(order), "ready_order_basis": "hypothetical: both orders",
                "decision_hashes": None, "snapshot_hashes": None, "m6g2_replay_key": (f"scripted:{sid}", tuple(order)),
            })
    for p in captured_pairs():
        order = _order(p["response_ready_order"])
        true = order
        if order is None and p.get("logical_ready_ms"):
            ms = p["logical_ready_ms"]
            true = sorted(ACTORS, key=lambda a: ms[a]) if ms[ACTOR_A] != ms[ACTOR_B] else None
        pairs.append({
            "pair_id": p["pair_id"], "source": p["source"], "source_milestone": p["source_milestone"], "source_file": p["source_file"],
            "source_trial_id": p["source_trial_id"], "runtime": p["runtime"], "scenario_id": p["scenario_id"], "raws": p["raws"],
            "response_ready_order": order, "true_ready_order": true, "ready_order_basis": p["ready_order_basis"],
            "decision_hashes": p["decision_hashes"], "snapshot_hashes": p["snapshot_hashes"],
            "m6g2_replay_key": (p["pair_id"], tuple(order) if order else None),
        })
    for fname, runtime in (("native_live_trials.jsonl", "native"), ("strands_live_trials.jsonl", "strands")):
        for row in read_jsonl(os.path.join(M6G2_DIR, fname)):
            if not row["complete"] or not row["boundaries"] or not row["boundaries"][0].get("complete", True):
                continue
            b0 = row["boundaries"][0]
            recs = b0["records"]
            pairs.append({
                "pair_id": f"m6g2:{row['trial_id']}:b0", "source": "m6g2_live", "source_milestone": "M6G.2",
                "source_file": f"{M6G2_DIR}/{fname}", "source_trial_id": row["trial_id"], "source_policy": row["commitment_policy"],
                "runtime": runtime, "scenario_id": row["scenario_id"], "raws": {a: recs[a]["raw_model_output"] for a in ACTORS},
                "response_ready_order": _order(b0["loop_ready_order"]), "true_ready_order": _order(b0["ready_order"]),
                "ready_order_basis": "observed in the M6G.2 live loop: loop-visible order (same tick = tie); true order by ready wall time",
                "decision_hashes": {a: recs[a]["decision_hash"] for a in ACTORS},
                "snapshot_hashes": {a: recs[a]["snapshot_world_hash"] for a in ACTORS},
                "m6g2_replay_key": None,
            })
    return pairs


def replay_manifest(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by = Counter(f"{p['source']}:{p['runtime']}" for p in pairs)
    ties = [p for p in pairs if p["response_ready_order"] is None]
    return {
        "written_before_replay_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_calls": 0,
        "n_pairs": len(pairs),
        "by_source": dict(sorted(by.items())),
        "by_scenario": dict(sorted(Counter(p["scenario_id"] for p in pairs).items())),
        "same_tick_ties": len(ties),
        "ties_with_known_true_order": sum(p["true_ready_order"] is not None for p in ties),
        "ties_true_order_white_first": sum(p["true_ready_order"] == [ACTOR_B, ACTOR_A] for p in ties),
        "measurements": {
            "post": "production Game loop, commit_policy=barrier_answer_order (the promoted default)",
            "pre": "production Game loop, commit_policy=answer_order (the pre-promotion loop, P0)",
            "p0_replay": "commitment_policy.replay_under_policy(P0) - M6G.1 T0 semantics, recomputed",
            "m6g2_recorded_p0": "the P0 row M6G.2 recorded for the same pair and ready order (replay_trials.jsonl), where one exists",
        },
        "ready_schedule": "scripted stand-ins answer the captured decisions; answers are made ready in the pair's loop-visible order, "
                          "the second one tick after the first was handed over and only once its controller stored any observation "
                          "change it saw in flight; a tie is made ready on one tick, Black published first",
        "required_equal": ["commit order", "world hash after the pair", "effect-state hash", "pair event hash", "first/second action cost",
                           "first/second result", "decision hashes (and = the source's)", "snapshot hash (= the source's)"],
        "tie_true_order": "ties with a known true order are also replayed through the barrier in that order and reported as the "
                          "intended same-tick refinement (spec sections 10-11), not as equivalence failures",
        "request_hash_check": "not repeated: M6G.2's reuse manifest already rebuilt and hash-checked every M6G request",
        "pairs": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items() if k not in ("raws",)} for p in pairs],
    }


def _m6g2_p0_rows() -> dict[tuple, dict[str, Any]]:
    out = {}
    for r in read_jsonl(os.path.join(M6G2_DIR, "replay_trials.jsonl")):
        if r["commitment_policy"] == P0:
            key = (r["pair_id"], tuple(r["response_ready_order"]) if r["response_ready_order"] else None)
            out[key] = {k: r[k] for k in ("world_state_hash_after", "effect_state_hash_after", "event_log_hash", "apply_order",
                                          "first_action_cost", "second_action_cost", "first_result", "second_result")}
    return out


def _groups(order: list[str] | None) -> list[list[tuple[str, int]]]:
    if order:
        return [[(order[0], 1)], [(order[1], 1)]]
    return [[(ACTOR_A, 1), (ACTOR_B, 1)]]


def _trial(scenario, p: dict[str, Any], policy: str, groups) -> dict[str, Any]:
    runtime = _probe_runtime(p["runtime"])
    kwargs, gates = scripted_requests(runtime, {a: [[p["raws"][a]]] for a in ACTORS})
    return run_production_trial(scenario, trial_id=f"{p['pair_id']}:{policy}", runtime=runtime, commit_policy=policy, gates=gates,
                                ready_schedule=groups, **_backend(p["runtime"]), **kwargs)


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    b = row["boundaries"][0] if row["boundaries"] else {}
    recs = b.get("records", {})
    return {
        "complete": row["complete"], "pair": bool(b.get("pair")), "commit_order": b.get("commit_order"), "ready_order": b.get("ready_order"),
        **{k: b.get(k) for k in PAIR_KEYS},
        "same_tick_commit": b.get("same_tick_commit"), "first_ready_is_first_commit": b.get("first_ready_is_first_commit"),
        "game_time_waited_s": b.get("game_time_waited_s"), "trial_event_log_hash": row["event_log_hash"],
        "decision_hashes": {a: recs[a]["decision_hash"] for a in recs},
        "snapshot_hashes": {a: recs[a]["snapshot_world_hash"] for a in recs},
        "request_hashes": {a: [r["provider_request_hash"] for r in row["requests"][a]] for a in ACTORS},
        "records": {a: {k: recs[a][k] for k in ("decision", "apply_outcome", "consequence", "staleness_class", "time_cost", "legal_at_apply",
                                                "apply_dt", "first_pass_valid")} for a in recs},
        "lifecycle": row["lifecycle"], "freeze_audit": row["freeze_audit"], "releases": row["releases"],
        "pre_action_matches_harness_snapshot": row["pre_action_matches_harness_snapshot"],
    }


def _p0_summary(rep: dict[str, Any]) -> dict[str, Any]:
    return {k: rep[k] for k in ("apply_order", "world_state_hash_after", "effect_state_hash_after", "event_log_hash", "first_action_cost",
                                "second_action_cost", "first_result", "second_result")} | {
        "decision_hashes": {a: rep["records"][a]["decision_hash"] for a in ACTORS}, "contested_effects": [list(e) for e in rep["contested_effects"]]}


def _lifecycle_clean(s: dict[str, Any]) -> bool:
    lc, fz = s["lifecycle"], s["freeze_audit"]
    return (lc["duplicate_requests"] == 0 and lc["unused_responses"] == 0 and lc["in_flight_at_end"] == 0
            and lc["requests_launched"] == 2 == lc["responses_consumed"] and fz["ticks_advanced_while_waiting"] == 0)


def compare(p: dict[str, Any], post: dict[str, Any], pre: dict[str, Any], p0: dict[str, Any], recorded: dict[str, Any] | None) -> dict[str, Any]:
    expected = p["response_ready_order"] or [ACTOR_A, ACTOR_B]
    checks = {
        "complete": post["complete"] and pre["complete"] and post["pair"] and pre["pair"],
        "commit_order": post["commit_order"] == pre["commit_order"] == p0["apply_order"] == expected,
        "world": post["world_hash_after_pair"] == pre["world_hash_after_pair"] == p0["world_state_hash_after"],
        "effect_state": post["effect_state_hash_after_pair"] == pre["effect_state_hash_after_pair"] == p0["effect_state_hash_after"],
        "world_after_first_commit": post["world_hash_after_first_commit"] == pre["world_hash_after_first_commit"],
        "pair_event_hash": post["event_log_hash_pair"] == pre["event_log_hash_pair"],
        "trial_event_log_hash": post["trial_event_log_hash"] == pre["trial_event_log_hash"] == p0["event_log_hash"],
        "contested_effects": [list(e) for e in post["contested_effects"]] == [list(e) for e in pre["contested_effects"]] == p0["contested_effects"],
        "costs": (post["first_action_cost"], post["second_action_cost"]) == (pre["first_action_cost"], pre["second_action_cost"])
                 == (p0["first_action_cost"], p0["second_action_cost"]),
        "results": (post["first_result"], post["second_result"]) == (pre["first_result"], pre["second_result"])
                   == (p0["first_result"], p0["second_result"]),
        "decisions": post["decision_hashes"] == pre["decision_hashes"] == p0["decision_hashes"]
                     and (p["decision_hashes"] is None or post["decision_hashes"] == p["decision_hashes"]),
        "snapshot": post["pre_action_matches_harness_snapshot"] and pre["pre_action_matches_harness_snapshot"]
                    and len(set(post["snapshot_hashes"].values())) == 1
                    and (p["snapshot_hashes"] is None or set(post["snapshot_hashes"].values()) == set(p["snapshot_hashes"].values())),
        "requests": post["request_hashes"] == pre["request_hashes"],
        "post_lifecycle_clean": _lifecycle_clean(post),
        "post_zero_game_time_wait": set((post["game_time_waited_s"] or {}).values()) == {0.0},
    }
    if recorded is not None:
        checks["m6g2_recorded_p0"] = (recorded["world_state_hash_after"], recorded["effect_state_hash_after"], recorded["event_log_hash"],
                                      recorded["apply_order"], recorded["first_action_cost"], recorded["second_action_cost"]) == (
            p0["world_state_hash_after"], p0["effect_state_hash_after"], p0["event_log_hash"], p0["apply_order"],
            p0["first_action_cost"], p0["second_action_cost"])
    # spec section 17's four required equalities; the rest are supporting checks
    required = ("complete", "commit_order", "world", "effect_state", "pair_event_hash", "costs")
    return {"checks": checks, "equivalent": all(checks[k] for k in required), "all_checks_pass": all(checks.values()),
            "failed_checks": sorted(k for k, v in checks.items() if not v)}


def run() -> dict[str, Any]:
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = build_policy_scenarios()
    pairs = corpus()
    write_json(out_path("replay_manifest.json"), replay_manifest(pairs))
    recorded = _m6g2_p0_rows()
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for i, p in enumerate(pairs, start=1):
        s = scenarios[p["scenario_id"]]
        order = p["response_ready_order"]
        post = _summary(_trial(s, p, COMMIT_POLICY_BARRIER, _groups(order)))
        pre = _summary(_trial(s, p, COMMIT_POLICY_ANSWER_ORDER, _groups(order)))
        rep = replay_under_policy(s, p["raws"], P0, response_ready_order=tuple(order) if order else None, pair_id=p["pair_id"],
                                  decision_boundary_id=f"{p['source_trial_id']}:b0", model_id=MODEL_ID)
        p0 = _p0_summary(rep)
        rec = recorded.get(p["m6g2_replay_key"]) if p["m6g2_replay_key"] else None
        cmp = compare(p, post, pre, p0, rec)
        base = {k: p[k] for k in ("pair_id", "source", "source_milestone", "source_file", "source_trial_id", "runtime", "scenario_id",
                                  "response_ready_order", "true_ready_order", "ready_order_basis")} | {
            "source_policy": p.get("source_policy"), "tie": order is None, "model_calls": 0}
        rows.append(base | {"kind": KIND_EQUIVALENCE, "post": post, "pre": pre, "p0_replay": p0, "m6g2_recorded_p0": rec} | cmp)
        true = p["true_ready_order"]
        if order is None and true is not None:
            tie = _summary(_trial(s, p, COMMIT_POLICY_BARRIER, [[(true[0], 1), (true[1], 1)]]))
            rows.append(base | {
                "kind": KIND_TIE_TRUE_ORDER, "post": tie, "p0_replay": p0,
                "commit_order_is_true_ready_order": tie["commit_order"] == true,
                "differs_from_p0": (tie["world_hash_after_pair"], tie["commit_order"]) != (p0["world_state_hash_after"], p0["apply_order"]),
                "world_differs_from_p0": tie["world_hash_after_pair"] != p0["world_state_hash_after"],
                "post_lifecycle_clean": _lifecycle_clean(tie),
            })
        if i % 25 == 0:
            print(f"  {i}/{len(pairs)} pairs ({time.monotonic() - started:.0f} s)", flush=True)
    with open(out_path("replay_equivalence.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    eq = [r for r in rows if r["kind"] == KIND_EQUIVALENCE]
    ties = [r for r in rows if r["kind"] == KIND_TIE_TRUE_ORDER]
    return {
        "pairs": len(pairs), "equivalent": sum(r["equivalent"] for r in eq), "all_checks_pass": sum(r["all_checks_pass"] for r in eq),
        "failed": [(r["pair_id"], r["failed_checks"]) for r in eq if not r["all_checks_pass"]],
        "tie_true_order_rows": len(ties), "tie_rows_differing_from_p0": sum(r["differs_from_p0"] for r in ties),
        "seconds": round(time.monotonic() - started, 1),
    }


if __name__ == "__main__":
    out = run()
    print(f"{out['pairs']} pairs: {out['equivalent']} equivalent, {out['all_checks_pass']} pass every check; "
          f"tie true-order rows {out['tie_true_order_rows']} ({out['tie_rows_differing_from_p0']} differ from P0); {out['seconds']} s")
    for pair_id, failed in out["failed"]:
        print(f"  {pair_id}: {failed}")
