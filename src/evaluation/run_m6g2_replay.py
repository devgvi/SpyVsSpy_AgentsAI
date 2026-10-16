"""M6G.2 Phases A-C (spec sections 8-31): the zero-call replay matrix.

Every captured decision pair is replayed under every commitment policy with
the decisions held fixed - the primary world-semantics instrument (spec
section 13). Sources, all read-only:

    scripted            the 8 M6G.2 scenarios' fixed pairs (C0, C1, C2, C4,
                        C5, C5M, C6, C6M); no live trial, so P0/P1 run under
                        both hypothetical ready orders
    m6g1_scripted_loop  M6G.1's 24 scripted real-loop trials (both runtimes,
                        every L scenario, both latency assignments): scripted
                        decisions with a ready order the real loop produced
    m6g_reuse           M6G's 110 live pairs (M6G.1's logical arrival order;
                        rebuilt requests hash-checked against what M6G sent)
    m6g1_live           M6G.1's 40 live real-loop pairs (observed ready order)

Policies per captured pair: P0 and P1 in the recorded ready order, P2, P3,
and P4 at both parities (the counterbalanced assignment - parity alternating
with the pair's index within its source and scenario - is flagged).

Writes reuse_manifest.json, replay_call_plan.json (before replaying),
scenario_manifest.json, policy_manifest.json, replay_trials.jsonl.

    python -m src.evaluation.run_m6g2_replay
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from src.evaluation.commitment_policy import (
    ATTEMPT_ABSORBED,
    ATTEMPT_BUSY,
    ATTEMPT_EXECUTING,
    ATTEMPT_LAUNCHED,
    ATTEMPT_NO_CHANGE,
    ATTEMPT_THROTTLED,
    FAMILY_TO_SCENARIO,
    HELD_BY_BARRIER,
    MIRROR_OF,
    P0,
    P1,
    P2,
    P3,
    P4,
    REQUEST_STATE_CODE_MAP,
    SCENARIO_IDS,
    SCENARIO_TO_FAMILY,
    build_policy_scenarios,
    policy_table,
    replay_under_policy,
)
from src.evaluation.decision_scheduling import native_render
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS, fresh_scenario_state, json_hash, world_state_snapshot
from src.evaluation.m6g_live import read_jsonl, write_json

OUT_DIR = os.path.join("evaluation_runs", "m6g2-commitment-policy-counterbalancing-v1")
M6G_DIR = os.path.join("evaluation_runs", "m6g-decision-scheduling-v1")
M6G1_DIR = os.path.join("evaluation_runs", "m6g1-long-running-actions-v1")
MODEL_ID = "us.anthropic.claude-sonnet-4-6"
CAPTURED_POLICIES = ((P0, 0), (P1, 0), (P2, 0), (P3, 0), (P4, 0), (P4, 1))


def out_path(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def _rk(runtime: str) -> str:
    return "strands" if runtime == "strands" else "native"


def _m6g_snapshot_hashes() -> dict[str, dict[str, str]]:
    out = {}
    for fname in ("native_call_order_trials.jsonl", "native_snapshot_timing_trials.jsonl", "strands_trials.jsonl"):
        for row in read_jsonl(os.path.join(M6G_DIR, fname)):
            out[row["trial_id"]] = {a: row["records"][a]["world_state_hash_before_snapshot"] for a in ACTORS}
    return out


def captured_pairs() -> list[dict[str, Any]]:
    """Every reusable live or real-loop pair, with the provenance the reuse
    manifest needs (spec section 9)."""
    from src.evaluation.run_m6g1_reuse import captured_pairs as m6g_pairs
    from src.evaluation.run_m6g1_reuse import logical_arrival

    pairs: list[dict[str, Any]] = []
    snap = _m6g_snapshot_hashes()
    for p in m6g_pairs():
        order, ready_b, ready_w = logical_arrival(p["latency_ms"][ACTOR_A], p["latency_ms"][ACTOR_B])
        pairs.append({
            "pair_id": p["pair_id"], "source": "m6g_reuse", "source_milestone": "M6G", "source_file": p["source_file"],
            "source_trial_id": p["source_trial_id"], "runtime": p["runtime"], "scenario_id": FAMILY_TO_SCENARIO[p["scenario_family"]],
            "family": p["scenario_family"], "raws": p["raws"], "response_ready_order": list(order) if order else None,
            "ready_order_basis": "logical: Black at its M6G latency, White at 50 ms + its latency; same tick (<16.7 ms) = tie",
            "response_latency_ms": p["latency_ms"], "logical_ready_ms": {ACTOR_A: round(ready_b, 2), ACTOR_B: round(ready_w, 2)},
            "provider_request_hashes": p["provider_request_hashes"], "agent_request_semantic_hashes": p["agent_request_semantic_hashes"],
            "decision_hashes": p["decision_hashes"], "snapshot_hashes": snap.get(p["source_trial_id"]),
            "request_hash_check": True,
        })
    for fname, runtime in (("real_loop_trials.jsonl", "native"), ("strands_trials.jsonl", "strands")):
        for row in read_jsonl(os.path.join(M6G1_DIR, fname)):
            if not row["complete"]:
                continue
            recs = row["records"]
            pairs.append({
                "pair_id": f"m6g1:{row['trial_id']}", "source": "m6g1_live", "source_milestone": "M6G.1",
                "source_file": f"{M6G1_DIR}/{fname}", "source_trial_id": row["trial_id"], "runtime": runtime,
                "scenario_id": FAMILY_TO_SCENARIO[row["scenario_family"]], "family": row["scenario_family"],
                "raws": {a: recs[a]["raw_model_output"] for a in ACTORS},
                "response_ready_order": None if row["same_tick_commit"] else row["ready_order"],
                "ready_order_basis": "observed in the production loop (ready wall time; same-tick readiness = tie)",
                "response_latency_ms": {a: recs[a]["response_latency_ms"] for a in ACTORS},
                "provider_request_hashes": {a: recs[a]["provider_request_hash"] for a in ACTORS},
                "agent_request_semantic_hashes": None,
                "decision_hashes": {a: recs[a]["decision_hash"] for a in ACTORS},
                "snapshot_hashes": {a: recs[a]["snapshot_world_hash"] for a in ACTORS},
                "request_hash_check": False,
                "apply_order_observed": row["apply_order"],
            })
    for row in read_jsonl(os.path.join(M6G1_DIR, "real_loop_scripted_trials.jsonl")):
        if not row["complete"]:
            continue
        recs = row["records"]
        pairs.append({
            "pair_id": f"m6g1:{row['trial_id']}", "source": "m6g1_scripted_loop", "source_milestone": "M6G.1",
            "source_file": f"{M6G1_DIR}/real_loop_scripted_trials.jsonl", "source_trial_id": row["trial_id"],
            "runtime": _rk(row["runtime"]), "scenario_id": FAMILY_TO_SCENARIO[row["scenario_family"]], "family": row["scenario_family"],
            "raws": {a: recs[a]["raw_model_output"] for a in ACTORS},
            "response_ready_order": None if row["same_tick_commit"] else row["ready_order"],
            "ready_order_basis": "scripted answer delays (slow 350 ms / fast 80 ms) through the production loop",
            "response_latency_ms": {a: recs[a]["response_latency_ms"] for a in ACTORS},
            "provider_request_hashes": {a: recs[a]["provider_request_hash"] for a in ACTORS},
            "agent_request_semantic_hashes": None,
            "decision_hashes": {a: recs[a]["decision_hash"] for a in ACTORS},
            "snapshot_hashes": {a: recs[a]["snapshot_world_hash"] for a in ACTORS},
            "request_hash_check": False,
            "apply_order_observed": row["apply_order"],
        })
    # counterbalanced P4 parity: alternate with the pair's index within its source/runtime/scenario group
    groups: dict[tuple, int] = defaultdict(int)
    for p in pairs:
        key = (p["source"], p["runtime"], p["scenario_id"])
        p["p4_assigned_parity"] = groups[key] % 2
        groups[key] += 1
    return pairs


def _render_for(runtime: str):
    if runtime == "strands":
        from src.evaluation.m6g_live import StrandsDecider

        return StrandsDecider("us-east-1", MODEL_ID).render
    return native_render


def _compact(row: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    keep = ("pair_id", "scenario_family", "condition", "apply_order", "snapshot_world_hash", "world_hash_before", "world_hash_after_first_apply",
            "world_hash_after_pair", "world_state_hash_after", "effect_state_hash_after", "event_log_hash", "event_multiset_hash",
            "contested_effects", "final_game_clock", "records", "commitment_policy", "p4_parity", "decision_boundary_id",
            "decision_boundary_index", "response_ready_order", "response_ready_tie", "first_commit_actor", "second_commit_actor",
            "commit_rank", "response_ready_rank", "first_ready_is_first_commit", "world_hash_after_first_commit", "first_action_cost",
            "second_action_cost", "first_result", "second_result", "decision_hashes")
    out = {k: row[k] for k in keep}
    out["n_events"] = len(row["events"])
    out.update(extra)
    return out


def replay_plan(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    counts = defaultdict(int)
    for p in pairs:
        counts[p["source"]] += 1
    n_captured_rows = len(pairs) * len(CAPTURED_POLICIES)
    return {
        "written_before_replay_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_calls": 0,
        "sources": {
            "scripted": {"pairs": len(SCENARIO_IDS), "policies": "P0 and P1 under both hypothetical ready orders, P2, P3, P4 at both parities",
                         "rows": len(SCENARIO_IDS) * 8},
            **{src: {"pairs": n, "policies": "P0 and P1 in the recorded ready order, P2, P3, P4 at both parities", "rows": n * len(CAPTURED_POLICIES)}
               for src, n in sorted(counts.items())},
        },
        "total_rows": len(SCENARIO_IDS) * 8 + n_captured_rows,
        "invariants_checked": [
            "each actor's decision hash identical across every policy replay of a pair and equal to the source's decision hash",
            "each pair's snapshot hash identical across policies and equal to the source's snapshot hash",
            "M6G pairs: each rebuilt request hashes to exactly the provider-facing request M6G sent (else the pair is rejected)",
            "P0 and P1 replays of the same pair give the same world hash and event multiset",
            "the replay applied the order the policy chose",
        ],
        "p4_parity_assignment": "every pair is replayed at both parities; the counterbalanced assignment alternates with the pair's index "
                                "within (source, runtime, scenario) and is the one P4 aggregates use",
    }


def scenario_manifest() -> dict[str, Any]:
    out = {}
    for sid, s in build_policy_scenarios().items():
        st = fresh_scenario_state(s.setup_fn)
        out[sid] = {
            "m6g1_family": SCENARIO_TO_FAMILY.get(sid), "mirror_of": MIRROR_OF.get(sid), "description": s.description,
            "shared_state": s.shared_state, "side_effect_visibility": s.side_effect_visibility,
            "scripted_decisions": {a: s.scripted_raw(a) for a in ACTORS}, "snapshot_world_hash": json_hash(world_state_snapshot(st)),
            "live": sid in ("C0_NON_CONFLICTING", "C2_SHARED_DOOR"),
            "trap_cases_scripted_only": sid.startswith(("C4", "C5", "C6")),
        }
    return out


def policy_manifest() -> dict[str, Any]:
    return {
        "policies": policy_table(),
        "p5_seeded_balanced_order": "omitted: nothing justified randomness; deterministic counterbalancing (P4) covers balance",
        "policy_may_choose_only": "which already-accepted action applies first",
        "policy_may_not": ["rewrite an action", "change a target", "repair again", "re-prompt", "re-decide", "change the legal action set"],
        "request_states": REQUEST_STATE_CODE_MAP,
        "attempt_kinds": {
            ATTEMPT_LAUNCHED: "get_action started a model request (decisions_requested incremented)",
            ATTEMPT_ABSORBED: "the observation digest changed while a request was in flight; the stored digest was refreshed, no request",
            ATTEMPT_THROTTLED: "the digest changed but the 0.5 s re-request throttle held; no request, stored digest not refreshed",
            ATTEMPT_NO_CHANGE: "the digest equals the stored one; the controller returned its last action or the in-flight WAIT",
            ATTEMPT_BUSY: "the actor was busy/stunned: get_action returned WAIT before looking at the digest",
            ATTEMPT_EXECUTING: "a MOVE was in progress: get_action continued it before looking at the digest",
            HELD_BY_BARRIER: "a barrier policy held the actor's ready decision (or kept a free actor waiting for a synchronized boundary); "
                             "get_action was not called",
        },
    }


def run() -> dict[str, Any]:
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = build_policy_scenarios()
    pairs = captured_pairs()
    write_json(out_path("replay_call_plan.json"), replay_plan(pairs))
    write_json(out_path("scenario_manifest.json"), scenario_manifest())
    write_json(out_path("policy_manifest.json"), policy_manifest())

    rows: list[dict[str, Any]] = []
    renders: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    violations: list[str] = []

    # scripted pairs: no live ready order, so P0/P1 under both hypothetical orders
    for sid, s in scenarios.items():
        raws = {a: s.scripted_raw(a) for a in ACTORS}
        variants = [(P0, (ACTOR_A, ACTOR_B), 0), (P0, (ACTOR_B, ACTOR_A), 0), (P1, (ACTOR_A, ACTOR_B), 0), (P1, (ACTOR_B, ACTOR_A), 0),
                    (P2, None, 0), (P3, None, 0), (P4, None, 0), (P4, None, 1)]
        for policy, ready, parity in variants:
            r = replay_under_policy(s, raws, policy, response_ready_order=ready, parity=parity, pair_id=f"scripted:{sid}",
                                    decision_boundary_id=f"scripted:{sid}:b0", model_id=MODEL_ID)
            rows.append(_compact(r, {"source": "scripted", "runtime": None, "scenario_id": sid, "ready_order_hypothetical": ready is not None,
                                     "p4_assigned": None, "response_latency_ms": None}))

    for p in pairs:
        s = scenarios[p["scenario_id"]]
        render = renders.setdefault(p["runtime"], _render_for(p["runtime"]))
        ready = tuple(p["response_ready_order"]) if p["response_ready_order"] else None
        pair_rows = []
        for policy, parity in CAPTURED_POLICIES:
            r = replay_under_policy(s, p["raws"], policy, response_ready_order=ready, parity=parity, pair_id=p["pair_id"],
                                    decision_boundary_id=f"{p['source_trial_id']}:b0", model_id=MODEL_ID,
                                    expected_request_hashes=p["provider_request_hashes"] if p["request_hash_check"] else None, render_fn=render)
            pair_rows.append(_compact(r, {"source": p["source"], "runtime": p["runtime"], "scenario_id": p["scenario_id"],
                                          "source_trial_id": p["source_trial_id"], "ready_order_hypothetical": False,
                                          "p4_assigned": (parity == p["p4_assigned_parity"]) if policy == P4 else None,
                                          "response_latency_ms": p["response_latency_ms"]}))
        # spec section 15: the decision pair and the snapshot are the same under every policy
        dec = {json_hash(r["decision_hashes"]) for r in pair_rows}
        snaps = {r["snapshot_world_hash"] for r in pair_rows}
        dec_ok = len(dec) == 1 and pair_rows[0]["decision_hashes"] == p["decision_hashes"]
        snap_ok = len(snaps) == 1 and (p["snapshot_hashes"] is None or set(p["snapshot_hashes"].values()) == snaps)
        p0, p1 = pair_rows[0], pair_rows[1]
        p0_eq_p1 = (p0["world_state_hash_after"], p0["event_multiset_hash"]) == (p1["world_state_hash_after"], p1["event_multiset_hash"])
        if not (dec_ok and snap_ok and p0_eq_p1):
            violations.append(f"{p['pair_id']}: decisions {dec_ok} snapshot {snap_ok} P0==P1 {p0_eq_p1}")
        rows += pair_rows
        manifest.append({k: p[k] for k in ("pair_id", "source", "source_milestone", "source_file", "source_trial_id", "runtime", "scenario_id",
                                           "family", "response_ready_order", "ready_order_basis", "response_latency_ms",
                                           "provider_request_hashes", "agent_request_semantic_hashes", "decision_hashes", "snapshot_hashes",
                                           "p4_assigned_parity")}
                        | {"actor_ids": list(ACTORS), "request_hash_checked": p["request_hash_check"], "decision_hashes_reproduced": dec_ok,
                           "snapshot_hashes_reproduced": snap_ok})

    by_source = defaultdict(int)
    for m in manifest:
        by_source[f"{m['source']}:{m['runtime']}"] += 1
    write_json(out_path("reuse_manifest.json"), {"n_pairs": len(manifest), "by_source": dict(sorted(by_source.items())), "pairs": manifest,
                                                 "invariant_violations": violations})
    with open(out_path("replay_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    return {"pairs": len(pairs), "rows": len(rows), "violations": violations}


if __name__ == "__main__":
    out = run()
    print(f"{out['pairs']} captured pairs, {out['rows']} replay rows, invariant violations: {out['violations'] or 'none'}")
