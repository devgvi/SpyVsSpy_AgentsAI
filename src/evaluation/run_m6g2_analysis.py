"""M6G.2 analysis (spec sections 22-36, 49-53, 66-74, 81).

Reads the zero-call replay matrix, the live trials of each runtime with
their zero-call reproductions, and the request-lifecycle events, and writes

    commitment_policy_metrics.json  per policy and scenario: first-commit
                                    distribution, latency coupling, world-
                                    outcome sensitivity, commitment effect (25-32)
    staleness_metrics.json          second-decision staleness, who bears it (24, 33, 70)
    request_waste_metrics.json      launched / consumed / unused calls and tokens (32-38, 71)
    wall_clock_metrics.json         time to first and to pair commit, game time held (49-51, 72)
    runtime_replication.json        Native vs Strands per policy (41-43, 73)
    promotion_decision.json         the seven properties side by side; no promotion (53-54)
    summary.json                    overall classification (74), next-step gate (81)

and splices the required tables (66-73) into summary.md between two marker
lines. Nothing is combined into a score (spec section 52).

Live counterfactual-derived fields (legal_at_snapshot, staleness_class,
consequence) are taken from the verified zero-call reproduction of each
trial (`run_m6g2_live --reproduce`): the Native rows were recorded before
the unpatched-copy fix in commitment_probe, so theirs are wrong. A live row
without a verified reproduction keeps its fields only if the fixed probe
wrote it (`live_state_untouched_by_assembly`); otherwise it is left out of
the staleness counts.

    python -m src.evaluation.run_m6g2_analysis
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import copy
import json
import statistics
from collections import Counter, defaultdict
from typing import Any

from src.evaluation.commitment_policy import (
    C0,
    C2,
    COST_ONLY_COMMITMENT_EFFECT,
    MIRROR_OF,
    NO_COMMITMENT_POLICY_EFFECT,
    P0,
    P1,
    P2,
    P3,
    P4,
    POLICIES,
    POLICY_IDS,
    SCENARIO_IDS,
    commitment_effect,
    policy_table,
    scenario_effect,
)
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS
from src.evaluation.m6g_live import read_jsonl, write_json
from src.evaluation.run_m6g2_live import DERIVED_RECORD_FIELDS, LIFECYCLE_FILE, LIVE_POLICIES, REPRO_FILES, expected_calls
from src.evaluation.run_m6g2_live import FILES as LIVE_FILES
from src.evaluation.run_m6g2_replay import M6G1_DIR, MODEL_ID, OUT_DIR, out_path
from src.evaluation.scheduling_analysis import decision_label
from src.evaluation.temporal_semantics import DIFFERENT_COST, INVALID_AT_SNAPSHOT, REFUSED, STALE_AND_INVALID, STALE_BUT_VALID

# spec section 74
LATENCY_COUPLED = "LATENCY_COUPLED_COMMITMENT_CONFIRMED"
BARRIER_PRESERVES = "BARRIER_PRESERVES_WORLD_SEMANTICS"
BARRIER_REMOVES_RACE = "BARRIER_REMOVES_REQUEST_RACE"
FIXED_PRIORITY = "FIXED_ACTOR_PRIORITY_EFFECT"
ALTERNATING_BALANCED = "ALTERNATING_PRIORITY_COUNTERBALANCED"
WORLD_EFFECT = "COMMITMENT_POLICY_WORLD_EFFECT_SUPPORTED"
COST_EFFECT = "COMMITMENT_POLICY_COST_EFFECT_SUPPORTED"
STALENESS_EFFECT = "COMMITMENT_POLICY_STALENESS_EFFECT_SUPPORTED"
RACE_SUPPORTED = "REQUEST_LIFECYCLE_RACE_SUPPORTED"
RT_INDEPENDENT = "COMMITMENT_EFFECT_RUNTIME_INDEPENDENT"
RT_DEPENDENT = "COMMITMENT_EFFECT_RUNTIME_DEPENDENT"
OVERALL_MIXED = "MIXED"
RT_NOT_CLASSIFIED = "NOT_CLASSIFIED_ONE_RUNTIME_ONLY"

SHORT = {P0: "P0", P1: "P1", P2: "P2", P3: "P3", P4: "P4"}
WHO = {ACTOR_A: "Black", ACTOR_B: "White"}
# events that hand a shared resource to one actor (M6G/M6G.1 contested effects)
RESOURCE_EVENTS = ("DOOR_UNLOCKED", "TRAP_PLACED", "TRAP_REMOVED", "TRAP_DISARMED")
FORCED_RACE_TESTS = [
    "tests/test_m6g2_commitment_policy.py::test_hp09_race_happens_under_p0_and_the_barrier_removes_it[native_tool_like]",
    "tests/test_m6g2_commitment_policy.py::test_hp09_race_happens_under_p0_and_the_barrier_removes_it[strands]",
]
TABLES_BEGIN = "<!-- BEGIN GENERATED TABLES (run_m6g2_analysis.py) -->"
TABLES_END = "<!-- END GENERATED TABLES -->"


# -- small helpers -----------------------------------------------------------------------


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _pct(k: int, n: int) -> str:
    return f"{k}/{n}" + (f" ({k / n:.0%})" if n else "")


def _stats(xs: list[float | None]) -> dict[str, Any] | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0]] * 3
    return {"n": len(xs), "min": round(min(xs), 1), "p25": round(q[0], 1), "median": round(statistics.median(xs), 1),
            "p75": round(q[2], 1), "max": round(max(xs), 1), "mean": round(statistics.fmean(xs), 1)}


def _med(stats: dict[str, Any] | None) -> str:
    return "-" if not stats else f"{stats['median']:.0f}"


def _short(h: str | None, n: int = 10) -> str:
    return (h or "")[:n]


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pair_label(records: dict[str, Any]) -> str:
    """"black decision / white decision" (no pipes: it goes into markdown tables)."""
    return " / ".join(decision_label(records[a]) for a in ACTORS)


def _shares_target(records: dict[str, Any]) -> bool:
    """Both decisions act on one object or door (MOVE holds no resource: M6G C3)."""
    a, b = records[ACTOR_A]["decision"], records[ACTOR_B]["decision"]
    return all(d["action"] not in ("wait", "move") for d in (a, b)) and a["target_id"] is not None and a["target_id"] == b["target_id"]


def _same_resource(records: dict[str, Any]) -> bool:
    """Same action on the same target: the two compete for one resource."""
    a, b = records[ACTOR_A]["decision"], records[ACTOR_B]["decision"]
    return _shares_target(records) and a["action"] == b["action"]


def _winner(row: dict[str, Any]) -> str | None:
    for e in row.get("contested_effects") or []:
        if e[0] in RESOURCE_EVENTS:
            return e[1]
    return None


def _contested(row: dict[str, Any]) -> str:
    effects = [tuple(e) for e in row.get("contested_effects") or []]
    return ", ".join(f"{t}:{WHO.get(a, a)}" for t, a in effects) if effects else "none"


def _outcome_label(row: dict[str, Any]) -> str:
    return (f"{_contested(row)}; 2nd {row['second_result']}; cost {row['first_action_cost']:g}/{row['second_action_cost']:g} s")


def _stale_actor(row: dict[str, Any]) -> str | None:
    return next((a for a in ACTORS if row["records"][a]["staleness_class"] == STALE_AND_INVALID), None)


# -- loading -------------------------------------------------------------------------------


def corrected_live(rows: list[dict[str, Any]], repro: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Live rows with the counterfactual-derived fields of every boundary
    record (and the order-sensitive event_log_hash) taken from the verified
    zero-call reproduction. Call payloads are dropped from the copy."""
    out = []
    for row in rows:
        rep = repro.get(row["trial_id"])
        r = copy.deepcopy(row)
        for a in ACTORS:
            for req in r["requests"][a]:
                req.pop("payloads", None)
        r["_reproduced"] = bool(rep and rep["reproduced"])
        r["_derived_source"] = ("zero_call_reproduction" if r["_reproduced"]
                                else "live_probe_after_fix" if row.get("live_state_untouched_by_assembly") else None)
        if r["_reproduced"]:
            for b, rb in zip(r["boundaries"], rep["boundaries"]):
                for a in ACTORS:
                    b["records"][a].update({k: rb["records"][a][k] for k in DERIVED_RECORD_FIELDS})
                b["first_result"], b["second_result"] = rb["first_result"], rb["second_result"]
            r["event_log_hash"] = rep["event_log_hash"]
        out.append(r)
    return out


def load() -> dict[str, Any]:
    live_raw = {rk: read_jsonl(out_path(name)) for rk, name in LIVE_FILES.items()}
    repro = {rk: {r["trial_id"]: r for r in read_jsonl(out_path(name))} for rk, name in REPRO_FILES.items()}
    return {
        "replay": read_jsonl(out_path("replay_trials.jsonl")),
        "reuse": _json(out_path("reuse_manifest.json")),
        "plan": _json(out_path("live_call_plan.json")),
        "live_raw": live_raw,
        "repro": repro,
        "live": {rk: corrected_live(rows, repro[rk]) for rk, rows in live_raw.items()},
        "lifecycle_events": read_jsonl(out_path(LIFECYCLE_FILE)),
        "provider_events": read_jsonl(out_path("provider_events.jsonl")),
        "m6g1_summary": _json(os.path.join(M6G1_DIR, "summary.json")),
    }


def live_boundaries(live: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Every complete live decision boundary, flattened with its trial's keys."""
    out = []
    for rk, rows in live.items():
        for row in rows:
            for b in row["boundaries"]:
                if not b["complete"]:
                    continue
                out.append({**b, "trial_id": row["trial_id"], "runtime_key": rk, "scenario_id": row["scenario_id"],
                            "commitment_policy": row["commitment_policy"], "p4_parity": row["p4_parity"], "rep": row["rep"],
                            "decision_call_order": row["decision_call_order"], "derived_ok": bool(row["_derived_source"]),
                            "replay_matches_live": row.get("replay_matches_live") if b["boundary"] == 0 else None})
    return out


def _captured(replay: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    """Captured pairs under `policy`; P4 at the counterbalanced parity only."""
    return [r for r in replay if r["source"] != "scripted" and r["commitment_policy"] == policy and (policy != P4 or r["p4_assigned"])]


def _scripted(replay: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    """Scripted pairs under `policy` (P0/P1 in both hypothetical ready orders, P4 at both parities)."""
    return [r for r in replay if r["source"] == "scripted" and r["commitment_policy"] == policy]


POPULATIONS = (("replay_captured", _captured), ("replay_scripted", _scripted))


# -- zero-call replay: invariants, policy blocks, commitment effect (spec 15-31) ------------


def replay_invariants(replay: list[dict[str, Any]], reuse: dict[str, Any]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in replay:
        by_pair[r["pair_id"]].append(r)
    dec_same = snap_same = p0p1 = 0
    for rows in by_pair.values():
        dec_same += len({json.dumps(r["decision_hashes"], sort_keys=True) for r in rows}) == 1
        snap_same += len({r["snapshot_world_hash"] for r in rows}) == 1
        same_order = [(x, y) for x in rows if x["commitment_policy"] == P0 for y in rows
                      if y["commitment_policy"] == P1 and x["response_ready_order"] == y["response_ready_order"]]
        p0p1 += bool(same_order) and all((x["world_state_hash_after"], x["event_multiset_hash"]) == (y["world_state_hash_after"], y["event_multiset_hash"])
                                         for x, y in same_order)
    applied_policy_order = 0
    for r in replay:
        ready = tuple(r["response_ready_order"]) if r["response_ready_order"] else None
        expected = POLICIES[r["commitment_policy"]].order(None, ready, r["decision_boundary_index"], r["p4_parity"] or 0)
        applied_policy_order += r["apply_order"] == list(expected)
    return {
        "model_calls": 0, "pairs": len(by_pair), "rows": len(replay),
        "rows_by_source": dict(sorted(Counter(r["source"] for r in replay).items())),
        "decision_hashes_identical_across_policies": dec_same,
        "snapshot_hashes_identical_across_policies": snap_same,
        "p0_equals_p1_same_ready_order": p0p1,
        "rows_applying_the_policy_order": applied_policy_order,
        "reuse_manifest": {"pairs": reuse["n_pairs"], "by_source": reuse["by_source"], "invariant_violations": reuse["invariant_violations"]},
    }


def _stale_block(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(recs)
    cls = Counter(r["staleness_class"] for r in recs)
    cons = Counter(r["consequence"] for r in recs)
    return {
        "n": n, "staleness": dict(sorted(cls.items())), "consequence": dict(sorted(cons.items())),
        "relevant_state_stale_rate": _rate(cls[STALE_BUT_VALID] + cls[STALE_AND_INVALID], n),
        "stale_but_valid_rate": _rate(cls[STALE_BUT_VALID], n),
        "stale_and_invalid_rate": _rate(cls[STALE_AND_INVALID], n),
        "refusal_rate": _rate(cons[REFUSED], n),
        "refusals_invalid_at_snapshot": sum(r["consequence"] == REFUSED and r["staleness_class"] == INVALID_AT_SNAPSHOT for r in recs),
        "different_cost_rate": _rate(cons[DIFFERENT_COST], n),
    }


def _policy_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    firsts = Counter(r["first_commit_actor"] for r in rows)
    ordered = [r for r in rows if r.get("response_ready_order")]
    coupled = sum(r["response_ready_order"][0] == r["first_commit_actor"] for r in ordered)
    contests = [r for r in rows if _same_resource(r["records"]) and _winner(r)]
    return {
        "n": n,
        "first_commit": {a: firsts.get(a, 0) for a in ACTORS},
        "first_commit_rate": {a: _rate(firsts.get(a, 0), n) for a in ACTORS},
        "latency_commit_coupling": {"pairs_with_ready_order": len(ordered), "first_ready_is_first_commit": coupled,
                                    "rate": _rate(coupled, len(ordered)), "same_tick_ties": n - len(ordered)},
        "resource_contests": len(contests),
        "resource_to_first_commit": sum(_winner(r) == r["first_commit_actor"] for r in contests),
        "resource_winner": {a: sum(_winner(r) == a for r in contests) for a in ACTORS},
        "second_decision": _stale_block([r["records"][r["second_commit_actor"]] for r in rows]),
        "world_outcomes": dict(Counter(_outcome_label(r) for r in rows).most_common()),
    }


def replay_policy_metrics(replay: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pop, select in POPULATIONS:
        out[pop] = {}
        for p in POLICY_IDS:
            rows = select(replay, p)
            per = {sid: _policy_block([r for r in rows if r["scenario_id"] == sid]) for sid in SCENARIO_IDS if any(r["scenario_id"] == sid for r in rows)}
            out[pop][p] = {"_all_scenarios": _policy_block(rows), **per}
    return out


def commitment_effects(replay: list[dict[str, Any]]) -> dict[str, Any]:
    """World-outcome sensitivity (spec sections 30-32): every decision pair
    committed Black-first (P2) vs White-first (P3). Any policy's outcome for
    one boundary is one of the two, so this is all a policy can decide."""
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in replay:
        if r["commitment_policy"] in (P2, P3):
            by_pair[r["pair_id"]][r["commitment_policy"]] = r
    pairs = []
    for pid, d in sorted(by_pair.items()):
        bf, wf = d[P2], d[P3]
        eff = commitment_effect(bf, wf)
        stale = {"black_first": _stale_actor(bf), "white_first": _stale_actor(wf)}
        pairs.append({
            "pair_id": pid, "source": bf["source"], "runtime": bf["runtime"], "scenario_id": bf["scenario_id"],
            "decision_pair": _pair_label(bf["records"]), "classification": eff["classification"],
            "final_state_changes": not eff["same_world"], "effect_state_changes": not eff["same_effect_state"],
            "cost_changes": eff["costs_black_first"] != eff["costs_white_first"],
            "refusal_changes": any((bf["records"][a]["consequence"] == REFUSED) != (wf["records"][a]["consequence"] == REFUSED) for a in ACTORS),
            "stale_and_invalid_actor": stale,
            "stale_and_invalid_is_second_commit": all(stale[k] in (None, r["second_commit_actor"]) for k, r in (("black_first", bf), ("white_first", wf))),
            "resource_winner": {"black_first": _winner(bf), "white_first": _winner(wf)} if _same_resource(bf["records"]) else None,
            "outcome_black_first": _outcome_label(bf), "outcome_white_first": _outcome_label(wf),
            **{k: eff[k] for k in ("costs_black_first", "costs_white_first", "second_stale_and_invalid")},
        })
    by_scenario = {}
    for sid in SCENARIO_IDS:
        ps = [p for p in pairs if p["scenario_id"] == sid]
        if not ps:
            continue
        labels = [p["classification"] for p in ps]
        contests = [p for p in ps if p["resource_winner"] and p["resource_winner"]["black_first"]]
        by_scenario[sid] = {
            "pairs": len(ps), "by_source": dict(sorted(Counter(p["source"] for p in ps).items())),
            "mirror_of": MIRROR_OF.get(sid),
            "labels": dict(sorted(Counter(labels).items())), "scenario_effect": scenario_effect(labels),
            "scripted_pair_effect": next((p["classification"] for p in ps if p["source"] == "scripted"), None),
            "effect_by_decision_pair": {dp: dict(Counter(p["classification"] for p in ps if p["decision_pair"] == dp))
                                        for dp in sorted({p["decision_pair"] for p in ps})},
            "final_state_changes": sum(p["final_state_changes"] for p in ps),
            "effect_state_changes": sum(p["effect_state_changes"] for p in ps),
            "cost_changes": sum(p["cost_changes"] for p in ps),
            "refusal_changes": sum(p["refusal_changes"] for p in ps),
            "stale_and_invalid_moves_with_order": sum(p["stale_and_invalid_actor"]["black_first"] != p["stale_and_invalid_actor"]["white_first"] for p in ps),
            "stale_and_invalid_is_second_commit": f"{sum(p['stale_and_invalid_is_second_commit'] for p in ps)}/{len(ps)}",
            "resource_to_first_commit_both_orders": f"{sum(p['resource_winner'] == {'black_first': ACTOR_A, 'white_first': ACTOR_B} for p in contests)}/{len(contests)}",
            "outcomes": {o: n for o, n in Counter(f"Black-first: {p['outcome_black_first']} || White-first: {p['outcome_white_first']}" for p in ps).most_common()},
        }
    return {"pairs": pairs, "by_scenario": by_scenario}


def _staleness_for(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "second_commit": _stale_block([r["records"][r["second_commit_actor"]] for r in pairs]),
        "first_commit": _stale_block([r["records"][r["first_commit_actor"]] for r in pairs]),
        "by_actor": {a: {"as_second_commit": sum(r["second_commit_actor"] == a for r in pairs), **_stale_block([r["records"][a] for r in pairs])}
                     for a in ACTORS},
        "stale_and_invalid_borne_by": {a: sum(r["records"][a]["staleness_class"] == STALE_AND_INVALID for r in pairs) for a in ACTORS},
        "by_scenario": {sid: _stale_block([r["records"][r["second_commit_actor"]] for r in pairs if r["scenario_id"] == sid])
                        for sid in SCENARIO_IDS if any(r["scenario_id"] == sid for r in pairs)},
    }


def staleness_metrics(replay: list[dict[str, Any]], lbs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {pop: {p: _staleness_for(select(replay, p)) for p in POLICY_IDS} for pop, select in POPULATIONS}
    out["live"] = {}
    for rk in LIVE_FILES:
        bs = [b for b in lbs if b["runtime_key"] == rk]
        if not bs:
            continue
        out["live"][rk] = {p: _staleness_for([b for b in bs if b["commitment_policy"] == p and b["derived_ok"]]) for p in LIVE_POLICIES
                           if any(b["commitment_policy"] == p for b in bs)}
        out["live"][rk]["_excluded_boundaries_without_verified_derived_fields"] = sum(not b["derived_ok"] for b in bs)
    out["notes"] = [
        "Rates are over second-committed decisions unless stated (first_commit, by_actor); the first decision of a boundary is "
        "applied to the world it was made from, so it is FRESH unless something else changed.",
        "A policy does not change how often the second decision is stale - in a shared-state conflict the second one always is - "
        "it changes WHICH actor's decision is second and so who bears the consequence (by_actor, stale_and_invalid_borne_by).",
        "INVALID_AT_SNAPSHOT (listed by the agent's legal list but refused by the Engine on its own snapshot) is a legality-surface "
        "gap (M6G audit section 7), not staleness; refusals_invalid_at_snapshot counts them inside refusal_rate.",
    ]
    return out


# -- live: latency coupling, actor priority, P4 balance (spec 26-27, 37-44) ------------------


def _p4_block(trials: list[dict[str, Any]]) -> dict[str, Any]:
    per_trial = []
    for t in trials:
        bs = [b for b in t["boundaries"] if b["complete"]]
        firsts = [b["first_commit_actor"] for b in bs]
        expected = [ACTORS[(b["boundary"] + (t["p4_parity"] or 0)) % 2] for b in bs]
        per_trial.append({"trial_id": t["trial_id"], "parity": t["p4_parity"], "first_commit": firsts, "first_ready": [b["ready_order"][0] for b in bs],
                          "follows_parity": firsts == expected, "balanced": len(bs) == 2 and set(firsts) == set(ACTORS),
                          "boundary1_synchronized_snapshot": len(bs) == 2 and bool(bs[1]["synchronized_snapshot"])})
    cross = Counter(f"first-ready {WHO[r]} -> first-commit {WHO[c]}" for pt in per_trial for r, c in zip(pt["first_ready"], pt["first_commit"]))
    return {"trials": len(per_trial), "balanced_trials": sum(pt["balanced"] for pt in per_trial),
            "follows_parity": sum(pt["follows_parity"] for pt in per_trial),
            "boundary1_synchronized": sum(pt["boundary1_synchronized_snapshot"] for pt in per_trial),
            "initial_parity": dict(sorted(Counter(str(pt["parity"]) for pt in per_trial).items())),
            "first_ready_vs_first_commit": dict(sorted(cross.items())), "per_trial": per_trial}


def _live_block(trials: list[dict[str, Any]], bs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(bs)
    firsts = Counter(b["first_commit_actor"] for b in bs)
    ties = [b for b in bs if b["ready_tie_same_tick"]]
    non_tie = [b for b in bs if not b["ready_tie_same_tick"]]
    contests = [b for b in bs if _same_resource(b["records"]) and _winner(b)]
    b0 = [b for b in bs if b["boundary"] == 0]
    out = {
        "trials": len(trials), "complete": sum(t["complete"] for t in trials), "boundaries": n,
        "first_commit": {a: firsts.get(a, 0) for a in ACTORS}, "first_commit_rate": {a: _rate(firsts.get(a, 0), n) for a in ACTORS},
        "first_commit_boundary0": {a: sum(b["first_commit_actor"] == a for b in b0) for a in ACTORS},
        "first_ready": {a: sum(b["ready_order"][0] == a for b in bs) for a in ACTORS},
        # the first-called actor gets the <=50 ms request head start; alternated by repetition
        "by_call_order": {o: {"boundaries": sum(b["decision_call_order"] == o for b in bs),
                              "black_first_ready": sum(b["decision_call_order"] == o and b["ready_order"][0] == ACTOR_A for b in bs),
                              "black_first_commit": sum(b["decision_call_order"] == o and b["first_commit_actor"] == ACTOR_A for b in bs)}
                          for o in ("black_first", "white_first")},
        "latency_commit_coupling": {
            "first_ready_is_first_commit": sum(b["first_ready_is_first_commit"] for b in bs), "boundaries": n,
            "rate": _rate(sum(b["first_ready_is_first_commit"] for b in bs), n),
            "same_tick_ties": len(ties), "ties_black_first": sum(b["first_commit_actor"] == ACTOR_A for b in ties),
            "non_tie_coupled": sum(b["first_ready_is_first_commit"] for b in non_tie), "non_tie": len(non_tie),
        },
        "policy_order_respected": sum(b["policy_order_respected"] for b in bs),
        "same_tick_pair_commit": sum(b["same_tick_commit"] for b in bs),
        "resource_contests": len(contests),
        "resource_to_first_commit": sum(_winner(b) == b["first_commit_actor"] for b in contests),
        "resource_to_first_ready": sum(_winner(b) == b["ready_order"][0] for b in contests),
        "resource_winner": {a: sum(_winner(b) == a for b in contests) for a in ACTORS},
        "replay_matches_live_boundary0": sum(t.get("replay_matches_live") is True for t in trials),
        "reproduced_zero_call": sum(t["_reproduced"] for t in trials),
        "first_pass_valid": sum(b["records"][a]["first_pass_valid"] for b in bs for a in ACTORS), "decisions": 2 * n,
        "decision_pairs": {f"b{k}": dict(Counter(_pair_label(b["records"]) for b in bs if b["boundary"] == k).most_common())
                           for k in sorted({b["boundary"] for b in bs})},
        "ready_gap_ms": _stats([b["wall_ms"]["to_both_ready"] - b["wall_ms"]["to_first_ready"] for b in bs]),
    }
    if any(t["commitment_policy"] == P4 for t in trials):
        out["p4"] = _p4_block([t for t in trials if t["commitment_policy"] == P4])
    return out


def live_commitment(live: dict[str, list[dict[str, Any]]], lbs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rk in LIVE_FILES:
        rows = live[rk]
        if not rows:
            out[rk] = {"status": "not run"}
            continue
        blk: dict[str, Any] = {"status": "done", "trials": len(rows)}
        for p in LIVE_POLICIES:
            ts = [t for t in rows if t["commitment_policy"] == p]
            bs = [b for b in lbs if b["runtime_key"] == rk and b["commitment_policy"] == p]
            per = {sid: _live_block([t for t in ts if t["scenario_id"] == sid], [b for b in bs if b["scenario_id"] == sid])
                   for sid in (C2, C0) if any(t["scenario_id"] == sid for t in ts)}
            blk[p] = {"_all_scenarios": _live_block(ts, bs), **per}
        out[rk] = blk
    return out


def latency_table_rows(lbs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spec section 68, grouped: identical (first-ready, first-commit, winner) boundaries collapse into one row."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for b in lbs:
        key = (b["runtime_key"], b["scenario_id"], b["commitment_policy"], b["boundary"], b["ready_order"][0], b["first_commit_actor"],
               _shares_target(b["records"]), _winner(b) if _same_resource(b["records"]) else None, bool(b["ready_tie_same_tick"]))
        groups[key].append(b)
    rows = []
    for (rk, sid, p, k, fr, fc, shared, winner, tie), bs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rows.append({"runtime": rk, "scenario_id": sid, "policy": p, "boundary": k, "first_ready": fr, "first_commit": fc, "same": fr == fc,
                     "same_tick_tie": tie, "shared_conflict": shared, "resource_winner": winner, "n": len(bs),
                     "trials": [b["trial_id"].rsplit(":", 1)[-1] for b in bs]})
    return rows


# -- live: request lifecycle and waste (spec 32-38, 71) ------------------------------------


def request_waste(live: dict[str, list[dict[str, Any]]], events: list[dict[str, Any]]) -> dict[str, Any]:
    by_runtime: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for rk in LIVE_FILES:
        rows = live[rk]
        if not rows:
            by_runtime[rk] = {"status": "not run"}
            continue
        blk: dict[str, Any] = {"status": "done"}
        for p in LIVE_POLICIES:
            ts = [t for t in rows if t["commitment_policy"] == p]
            reqs = [req for t in ts for a in ACTORS for req in t["requests"][a]]
            unused = [req for req in reqs if req["ready"] and not req["consumed"]]
            attempts: Counter = Counter()
            for t in ts:
                attempts.update(t["lifecycle"]["attempts"])
            blk[p] = {
                "trials": len(ts),
                "provider_calls_launched": sum(req["model_calls"] for req in reqs),
                "requests_launched": len(reqs),
                "boundary_requests": sum(req["launch_kind"] == "boundary" for req in reqs),
                "extra_requests": sum(req["launch_kind"] == "extra" for req in reqs),
                "hp09_violations": sum(bool(req["hp09_violation"]) for req in reqs),
                "trials_with_hp09": sum(t["lifecycle"]["hp09_violations"] > 0 for t in ts),
                "responses_ready": sum(req["ready"] for req in reqs),
                "responses_consumed": sum(req["consumed"] for req in reqs),
                "unused_responses": len(unused),
                "in_flight_at_end": sum(not req["ready"] for req in reqs),
                "unused_provider_calls": sum(req["model_calls"] for req in reqs if not req["consumed"]),
                "unused_input_tokens": sum(req["input_tokens"] for req in reqs if not req["consumed"]),
                "unused_output_tokens": sum(req["output_tokens"] for req in reqs if not req["consumed"]),
                "input_tokens": sum(req["input_tokens"] for req in reqs), "output_tokens": sum(req["output_tokens"] for req in reqs),
                "early_returns_held": sum(t["lifecycle"]["early_returns_held"] for t in ts),
                "held_polls": sum(t["lifecycle"]["held_polls"] for t in ts),
                "attempts": dict(sorted(attempts.items())),
            }
            for a in ACTORS:
                for k in sorted({req["boundary"] for t in ts for req in t["requests"][a]}):
                    rs = [req for t in ts for req in t["requests"][a] if req["boundary"] == k]
                    table.append({"runtime": rk, "policy": p, "actor": a, "boundary": k, "requests": len(rs),
                                  "extra_requests": sum(r["launch_kind"] == "extra" for r in rs), "hp09": sum(bool(r["hp09_violation"]) for r in rs),
                                  "responses_ready": sum(r["ready"] for r in rs), "responses_committed": sum(r["consumed"] for r in rs),
                                  "unused": sum(r["ready"] and not r["consumed"] for r in rs),
                                  "extra_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rs if not r["consumed"])})
            for t in ts:
                for a in ACTORS:
                    for req in t["requests"][a]:
                        if req["launch_kind"] == "extra":
                            extra.append({"trial_id": t["trial_id"], "runtime": rk, "policy": p, "actor_id": a, "tick": req["launch_tick"],
                                          "decision_boundary_id": f"{t['trial_id']}:b{req['boundary']}", "prior_state": req["prior_state"],
                                          "request_launched": True, "result_consumed": req["consumed"], "hp09_violation": req["hp09_violation"],
                                          "tokens": req["input_tokens"] + req["output_tokens"]})
        by_runtime[rk] = blk
    ev = Counter((e["runtime"], e["commitment_policy"], e.get("attempt") or ("HELD" if not e.get("launched") else "LAUNCH")) for e in events)
    return {
        "by_runtime": by_runtime, "table_71": table, "extra_request_attempts": extra,
        "lifecycle_event_counts": {f"{rk}/{SHORT[p]}/{att}": n for (rk, p, att), n in sorted(ev.items())},
        "notes": [
            "A response is consumed when its decision is committed. Requests still in flight at the end of a trial were joined "
            "(without stepping the game) so their calls and tokens count as unused.",
            "extra_request_attempts lists every request launched after the boundary's own request, with the actor, tick, boundary "
            "and the prior response state (spec section 35).",
        ],
    }


# -- live: wall clock and game-time freeze (spec 49-51, 72) --------------------------------


def wall_clock(live: dict[str, list[dict[str, Any]]], lbs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rk in LIVE_FILES:
        rows = live[rk]
        if not rows:
            out[rk] = {"status": "not run"}
            continue
        blk: dict[str, Any] = {"status": "done"}
        for p in LIVE_POLICIES:
            ts = [t for t in rows if t["commitment_policy"] == p]
            per_b = {}
            for k in sorted({b["boundary"] for b in lbs if b["runtime_key"] == rk and b["commitment_policy"] == p}):
                kb = [b for b in lbs if b["runtime_key"] == rk and b["commitment_policy"] == p and b["boundary"] == k]
                ws = [b["wall_ms"] for b in kb]
                per_b[f"b{k}"] = {
                    "boundaries": len(ws),
                    "first_commit_on_unchanged_snapshot": f"{sum(not b['records'][b['first_commit_actor']]['world_changed'] for b in kb)}/{len(kb)}",
                    "to_first_ready_ms": _stats([w["to_first_ready"] for w in ws]),
                    "to_both_ready_ms": _stats([w["to_both_ready"] for w in ws]),
                    "to_first_commit_ms": _stats([w["to_first_commit"] for w in ws]),
                    "to_pair_commit_ms": _stats([w["to_pair_commit"] for w in ws]),
                    "first_commit_after_first_ready_ms": _stats([w["to_first_commit"] - w["to_first_ready"] for w in ws]),
                    "pair_commit_after_both_ready_ms": _stats([w["to_pair_commit"] - w["to_both_ready"] for w in ws]),
                }
            fa = [t["freeze_audit"] for t in ts]
            game = {k: round(sum(f[k] for f in fa), 6) for k in ("ticks_advanced_during_inference", "ticks_advanced_while_a_decision_was_held",
                                                                 "game_time_advanced_while_held_s", "hold_ticks", "handoff_ticks_before_boundary0_pair",
                                                                 "game_time_before_boundary0_pair_s", "align_ticks", "game_time_advanced_while_aligning_s")}
            # The tick audit covers held ticks and ticks inside a request window. This closes the rest: the first commit of every
            # boundary must meet exactly the world (clocks included) its decision was made from.
            pb = [b for b in lbs if b["runtime_key"] == rk and b["commitment_policy"] == p]
            unchanged = sum(not b["records"][b["first_commit_actor"]]["world_changed"] for b in pb)
            game["first_commit_on_unchanged_snapshot"] = f"{unchanged}/{len(pb)}"
            game["game_time_advanced_during_inference_or_barrier"] = bool(game["ticks_advanced_during_inference"] or game["ticks_advanced_while_a_decision_was_held"]
                                                                          or unchanged < len(pb))
            game["early_returns_held"] = sum(t["lifecycle"]["early_returns_held"] for t in ts)
            blk[p] = {"trials": len(ts), **per_b, "game_time": game}
        out[rk] = blk
    out["notes"] = [
        "Times are wall-clock ms from the boundary's first request launch. Under P0 the first commit follows the first ready answer "
        "within one loop tick; under a barrier it waits for the second answer, so first_commit_after_first_ready_ms is the barrier's "
        "wall-clock cost (the gap between the two answers) and the pair still completes when the slower answer is ready.",
        "game_time_advanced_while_aligning_s (P4 only) is the first boundary's actions finishing on live ticks before the second "
        "synchronized boundary opens: no decision is in flight or held then, so it is not inference or barrier time.",
        "first_commit_on_unchanged_snapshot: the first commit of each boundary met exactly the world, clocks included, its decision "
        "was made from - so no tick between the snapshot and the commit advanced game time, including a tick on which an answer "
        "landed after that tick's hold decision (early_returns_held: the controller's early return was replaced by WAIT and the "
        "same decision committed at the release).",
    ]
    return out


# -- runtime replication (spec 41-43, 73) --------------------------------------------------


def _rules(policy: str, ts: list[dict[str, Any]], bs: list[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, Any], str]:
    b0 = [b for b in bs if b["boundary"] == 0]
    contests = [b for b in bs if _same_resource(b["records"]) and _winner(b)]
    derived = [b for b in contests if b["derived_ok"]]
    second_sbv = sum(b["records"][b["second_commit_actor"]]["staleness_class"] == STALE_BUT_VALID
                     and b["records"][b["second_commit_actor"]]["consequence"] == DIFFERENT_COST for b in derived)
    fa = [t["freeze_audit"] for t in ts]
    rules = {
        "commit_order_follows_the_policy": all(b["policy_order_respected"] for b in bs),
        "live_world_equals_zero_call_replay": all(t.get("replay_matches_live") is True for t in ts),
        "no_game_time_during_inference": all(f["ticks_advanced_during_inference"] == 0 for f in fa),
    }
    if contests:
        rules["shared_resource_to_first_commit"] = all(_winner(b) == b["first_commit_actor"] for b in contests)
    if derived:
        rules["second_door_decision_stale_but_valid_different_cost"] = second_sbv == len(derived)
    if policy in (P0, P1):
        rules["first_ready_is_first_commit"] = all(b["first_ready_is_first_commit"] or (b["ready_tie_same_tick"] and b["first_commit_actor"] == ACTOR_A)
                                                  for b in bs)
    if POLICIES[policy].waits_for_both:
        rules["pair_commits_on_one_tick"] = all(b["same_tick_commit"] for b in bs)
        rules["no_game_time_while_held"] = all(f["ticks_advanced_while_a_decision_was_held"] == 0 for f in fa)
        rules["no_replacement_request_no_unused_response"] = all(t["lifecycle"]["extra_requests"] == 0 and t["lifecycle"]["unused_responses"] == 0 for t in ts)
    p4 = _p4_block(ts) if policy == P4 else None
    if p4:
        rules["every_trial_balanced_black_and_white_first"] = p4["balanced_trials"] == p4["trials"]
        rules["second_boundary_from_one_shared_snapshot"] = p4["boundary1_synchronized"] == p4["trials"]
    incidence = {
        "trials": len(ts), "boundaries": len(bs), "black_first": sum(b["first_commit_actor"] == ACTOR_A for b in bs),
        "trials_with_hp09": sum(t["lifecycle"]["hp09_violations"] > 0 for t in ts),
        "unused_responses": sum(t["lifecycle"]["unused_responses"] for t in ts),
        "first_pass_valid": f"{sum(b['records'][a]['first_pass_valid'] for b in bs for a in ACTORS)}/{2 * len(bs)}",
        "median_ready_gap_ms": (_stats([b["wall_ms"]["to_both_ready"] - b["wall_ms"]["to_first_ready"] for b in bs]) or {}).get("median"),
    }
    text = (f"{len(ts)} trials: policy order {sum(b['policy_order_respected'] for b in bs)}/{len(bs)}, "
            f"replay==live {sum(t.get('replay_matches_live') is True for t in ts)}/{len(ts)}"
            + (f", resource to first commit {sum(_winner(b) == b['first_commit_actor'] for b in contests)}/{len(contests)}" if contests else "")
            + (f", 2nd STALE_BUT_VALID/DIFFERENT_COST {second_sbv}/{len(derived)}" if derived else "")
            + (f", one-tick pair {sum(b['same_tick_commit'] for b in bs)}/{len(bs)}" if POLICIES[policy].waits_for_both else "")
            + (f", balanced trials {p4['balanced_trials']}/{p4['trials']}" if p4 else "")
            + f", Black first {incidence['black_first']}/{len(bs)}, HP09 trials {incidence['trials_with_hp09']}/{len(ts)}")
    return rules, incidence, text


def runtime_replication(live: dict[str, list[dict[str, Any]]], lbs: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for p in LIVE_POLICIES:
        for sid in (C2, C0):
            per: dict[str, Any] = {}
            for rk in LIVE_FILES:
                ts = [t for t in live[rk] if t["commitment_policy"] == p and t["scenario_id"] == sid]
                if ts:
                    per[rk] = _rules(p, ts, [b for b in lbs if b["runtime_key"] == rk and b["commitment_policy"] == p and b["scenario_id"] == sid])
            if not per:
                continue
            name = f"{SHORT[p]}/{sid}"
            if len(per) < 2:
                only = next(iter(per))
                comparisons[name] = {"native": per.get("native", (None, None, "not run"))[2], "strands": per.get("strands", (None, None, "not run"))[2],
                                     "rules": {k: {only: v} for k, v in per[only][0].items()}, "classification": RT_NOT_CLASSIFIED}
                continue
            shared = sorted(set(per["native"][0]) & set(per["strands"][0]))
            dependent = [k for k in shared if per["native"][0][k] != per["strands"][0][k]]
            comparisons[name] = {
                "native": per["native"][2], "strands": per["strands"][2],
                "rules": {k: {"native": per["native"][0][k], "strands": per["strands"][0][k]} for k in shared},
                "dependent_rules": dependent,
                "incidence_not_classified": {"native": per["native"][1], "strands": per["strands"][1]},
                "classification": RT_DEPENDENT if dependent else RT_INDEPENDENT,
            }
    classified = [c["classification"] for c in comparisons.values() if c["classification"] != RT_NOT_CLASSIFIED]
    overall = (RT_NOT_CLASSIFIED if not classified else RT_INDEPENDENT if all(c == RT_INDEPENDENT for c in classified)
               else RT_DEPENDENT if all(c == RT_DEPENDENT for c in classified) else OVERALL_MIXED)
    return {
        "status": "done" if live["strands"] else "strands not run",
        "overall": overall, "comparisons": comparisons,
        "confound_controls": "same policy, scenario, model/provider settings (Bedrock, us.anthropic.claude-sonnet-4-6, temperature 0), "
                             "DEFAULT vs DEFAULT, SOCIAL_OFF, production request construction per runtime (M6F RP01-RP18 parity gate "
                             "before Strands)",
        "rule": "a rule that holds on both runtimes (or fails on both) -> COMMITMENT_EFFECT_RUNTIME_INDEPENDENT; holds on one and not "
                "the other -> COMMITMENT_EFFECT_RUNTIME_DEPENDENT; rates (Black-first share, HP09 incidence, ready gap) are reported "
                "as incidence and not classified; a scenario observed on one runtime only is not classified",
    }


# -- overall classification (spec 74), promotion (53-54), next-step gate (81) -------------------


def _by_runtime(items: list[dict[str, Any]], pred) -> str:
    parts = []
    for rk in LIVE_FILES:
        xs = [x for x in items if x.get("runtime_key") == rk]
        if xs:
            parts.append(f"{rk} {sum(bool(pred(x)) for x in xs)}/{len(xs)}")
    return ", ".join(parts) or "none"


def overall_classification(data: dict[str, Any], inv: dict[str, Any], effects: dict[str, Any], lbs: list[dict[str, Any]],
                           repl: dict[str, Any]) -> dict[str, Any]:
    labels: list[str] = []
    ev: dict[str, str] = {}
    live, replay = data["live"], data["replay"]
    trials = [dict(t, runtime_key=rk) for rk, rows in live.items() for t in rows]
    barrier_trials = [t for t in trials if POLICIES[t["commitment_policy"]].waits_for_both]
    p0_trials = [t for t in trials if t["commitment_policy"] == P0]
    runtimes_run = [rk for rk in LIVE_FILES if live[rk]]

    # latency -> commit coupling (live P0)
    p0b = [b for b in lbs if b["commitment_policy"] == P0]
    ties = [b for b in p0b if b["ready_tie_same_tick"]]
    non_tie = [b for b in p0b if not b["ready_tie_same_tick"]]
    doors = [b for b in p0b if _same_resource(b["records"]) and _winner(b)]
    if non_tie and all(b["first_ready_is_first_commit"] for b in non_tie) and all(b["first_commit_actor"] == ACTOR_A for b in ties):
        labels.append(LATENCY_COUPLED)
    ev[LATENCY_COUPLED] = (f"live P0: first-ready == first-commit {_by_runtime(non_tie, lambda b: b['first_ready_is_first_commit'])} boundaries "
                           f"(+{len(ties)} same-tick ties, Black first {sum(b['first_commit_actor'] == ACTOR_A for b in ties)}); shared door "
                           f"unlocked by the first-ready actor {_by_runtime(doors, lambda b: _winner(b) == b['ready_order'][0])}; M6G.1 (spec): "
                           f"40/40 and 20/20")

    # barrier preserves world semantics
    p1b = [b for b in lbs if b["commitment_policy"] == P1]
    held = sum(t["freeze_audit"]["ticks_advanced_while_a_decision_was_held"] for t in barrier_trials)
    if (inv["p0_equals_p1_same_ready_order"] == inv["pairs"] and p1b and all(b["policy_order_respected"] for b in p1b)
            and all(t.get("replay_matches_live") is True for t in barrier_trials) and held == 0):
        labels.append(BARRIER_PRESERVES)
    ev[BARRIER_PRESERVES] = (f"zero-call replay: P0 == P1 (same decisions, same ready order) on {inv['p0_equals_p1_same_ready_order']}/{inv['pairs']} "
                             f"pairs; live P1 committed in P0's order {_by_runtime(p1b, lambda b: b['policy_order_respected'])}; live barrier "
                             f"world == zero-call replay {_by_runtime(barrier_trials, lambda t: t.get('replay_matches_live') is True)} trials; "
                             f"game time advanced while a decision was held: {held} ticks")

    # barrier removes the request race
    m6g1_race = data["m6g1_summary"].get("stale_decision_applied_while_fresh_request_started", {}).get("trials", {})
    p0_race = [t for t in p0_trials if t["lifecycle"]["hp09_violations"] > 0]
    race_seen = bool(p0_race) or sum(m6g1_race.values()) > 0
    clean = all(t["lifecycle"]["hp09_violations"] == 0 and t["lifecycle"]["extra_requests"] == 0 and t["lifecycle"]["unused_responses"] == 0
                for t in barrier_trials)
    both_runtimes = all(any(t["runtime_key"] == rk for t in barrier_trials) for rk in LIVE_FILES)
    if barrier_trials and clean and race_seen and both_runtimes:
        labels.append(BARRIER_REMOVES_RACE)
    ev[BARRIER_REMOVES_RACE] = (f"live barrier trials with no replacement request, no HP09 violation and no unused response: "
                                f"{_by_runtime(barrier_trials, lambda t: t['lifecycle']['extra_requests'] == 0 and t['lifecycle']['unused_responses'] == 0)}"
                                f"{'' if both_runtimes else ' (Strands pending)'}; race under P0: M6G.2 live {_by_runtime(p0_trials, lambda t: t['lifecycle']['hp09_violations'] > 0)}, "
                                f"M6G.1 live {m6g1_race}; mechanism pinned on both runtimes by {', '.join(FORCED_RACE_TESTS)} (the held decision is "
                                f"not applied, so the waiting actor's observation digest does not change and nothing re-triggers a request)"
                                + ("" if p0_race else "; live incidence alone does not discriminate here - the race needs two answers within "
                                                      "about one loop tick and did not recur under P0 in this run - so the removal rests on "
                                                      "the forced interleavings, with the live barrier trials as the no-regression check"))

    # fixed actor priority
    p2 = _captured(replay, P2) + _scripted(replay, P2)
    p3 = _captured(replay, P3) + _scripted(replay, P3)
    effect_scen = [sid for sid, s in effects["by_scenario"].items() if s["scenario_effect"] != NO_COMMITMENT_POLICY_EFFECT]
    doors2 = [r for r in p2 if r["scenario_id"] == C2 and _same_resource(r["records"])]
    doors3 = [r for r in p3 if r["scenario_id"] == C2 and _same_resource(r["records"])]
    if p2 and all(r["first_commit_actor"] == ACTOR_A for r in p2) and all(r["first_commit_actor"] == ACTOR_B for r in p3) and effect_scen:
        labels.append(FIXED_PRIORITY)
    ev[FIXED_PRIORITY] = (f"replay: P2 Black first {_pct(sum(r['first_commit_actor'] == ACTOR_A for r in p2), len(p2))}, P3 White first "
                          f"{_pct(sum(r['first_commit_actor'] == ACTOR_B for r in p3), len(p3))}; shared door unlocked by Black under P2 "
                          f"{sum(_winner(r) == ACTOR_A for r in doors2)}/{len(doors2)}, by White under P3 {sum(_winner(r) == ACTOR_B for r in doors3)}/{len(doors3)}; "
                          f"order-sensitive scenarios: {', '.join(effect_scen) or 'none'} - the fixed order hands every one of them to one identity")

    # alternating priority counterbalanced
    p4_trials = [t for t in trials if t["commitment_policy"] == P4]
    p4 = _p4_block(p4_trials) if p4_trials else None
    p4_rep = _policy_block(_captured(replay, P4))
    p4_doors = [b for b in lbs if b["commitment_policy"] == P4 and _same_resource(b["records"]) and _winner(b)]
    parity_even = p4 is not None and all(
        sum(t["p4_parity"] == 0 for t in p4_trials if t["runtime_key"] == rk) == sum(t["p4_parity"] == 1 for t in p4_trials if t["runtime_key"] == rk)
        for rk in runtimes_run)
    if (p4 and p4["balanced_trials"] == p4["trials"] and p4["follows_parity"] == p4["trials"] and parity_even
            and all(_winner(b) == b["first_commit_actor"] for b in p4_doors)):
        labels.append(ALTERNATING_BALANCED)
    ev[ALTERNATING_BALANCED] = (f"live P4: every trial one Black-first and one White-first boundary {_by_runtime(p4_trials, lambda t: _p4_block([t])['balanced_trials'])}, "
                                f"first commit follows the boundary parity {_by_runtime(p4_trials, lambda t: _p4_block([t])['follows_parity'])}, "
                                f"initial parity {p4['initial_parity'] if p4 else '-'}; first-ready vs first-commit {p4['first_ready_vs_first_commit'] if p4 else '-'}; "
                                f"door unlocked by the first commit {_by_runtime(p4_doors, lambda b: _winner(b) == b['first_commit_actor'])}; replay "
                                f"(counterbalanced parity) Black first {_pct(p4_rep['first_commit'][ACTOR_A], p4_rep['n'])}. Not a claim of global superiority")

    # world / cost / staleness effects of commitment order (zero-call replay)
    bys = effects["by_scenario"]
    world = {sid: f"{s['effect_state_changes']}/{s['pairs']}" for sid, s in bys.items() if s["effect_state_changes"]}
    cost = {sid: f"{s['cost_changes']}/{s['pairs']}" for sid, s in bys.items() if COST_ONLY_COMMITMENT_EFFECT in s["labels"]}
    stale = {sid: f"{s['stale_and_invalid_moves_with_order']}/{s['pairs']}" for sid, s in bys.items() if s["stale_and_invalid_moves_with_order"]}
    if world:
        labels.append(WORLD_EFFECT)
    ev[WORLD_EFFECT] = f"pairs whose effect state (not just clocks) differs Black-first vs White-first: {world or 'none'}"
    if cost:
        labels.append(COST_EFFECT)
    ev[COST_EFFECT] = f"pairs where only the action costs move with the order (first unlocks for 1.0 s, second re-opens for 0.3 s): {cost or 'none'}"
    if stale:
        labels.append(STALENESS_EFFECT)
    ev[STALENESS_EFFECT] = (f"pairs where the order decides whose decision is STALE_AND_INVALID: {stale or 'none'}; the stale rate itself does "
                            f"not move - in a shared-state conflict the second decision is always the stale one")

    # the request-lifecycle race itself
    if race_seen:
        labels.append(RACE_SUPPORTED)
    ev[RACE_SUPPORTED] = (f"HP09 violations live under P0: M6G.2 {_by_runtime(p0_trials, lambda t: t['lifecycle']['hp09_violations'] > 0)} trials, "
                          f"M6G.1 {m6g1_race}; forced interleaving reproduces it deterministically on both runtimes")

    if repl["overall"] in (RT_INDEPENDENT, RT_DEPENDENT):
        labels.append(repl["overall"])
    elif repl["overall"] == OVERALL_MIXED:
        labels.append(OVERALL_MIXED)
    ev["runtime_replication"] = f"{repl['overall']}: " + "; ".join(f"{k} {v['classification']}" for k, v in repl["comparisons"].items())
    return {"labels": labels, "evidence": ev}


def promotion_decision(rpm: dict[str, Any], stale: dict[str, Any], live_c: dict[str, Any], waste: dict[str, Any], clock: dict[str, Any],
                       repl: dict[str, Any], effects: dict[str, Any]) -> dict[str, Any]:
    """Spec sections 53-54: the seven properties per policy, side by side.
    Nothing is promoted and nothing is ranked."""
    done = [rk for rk in LIVE_FILES if live_c[rk].get("status") == "done"]
    props = {}
    for p in POLICY_IDS:
        pol, cap, st = POLICIES[p], rpm["replay_captured"][p]["_all_scenarios"], stale["replay_captured"][p]
        live_p = {rk: live_c[rk][p]["_all_scenarios"] for rk in done if p in live_c[rk]}
        props[p] = {
            "latency_dependence": {"by_definition": pol.depends_on_latency,
                                   "replay_first_ready_is_first_commit": cap["latency_commit_coupling"]["rate"],
                                   "live_first_ready_is_first_commit": {rk: b["latency_commit_coupling"]["rate"] for rk, b in live_p.items()} or "not run live"},
            "actor_priority": {"depends_on_actor_id": pol.depends_on_actor_id, "replay_first_commit_rate": cap["first_commit_rate"],
                               "live_first_commit_rate": {rk: b["first_commit_rate"] for rk, b in live_p.items()} or "not run live",
                               **({"live_p4_balanced_trials": {rk: f"{b['p4']['balanced_trials']}/{b['p4']['trials']}" for rk, b in live_p.items()}}
                                  if p == P4 and live_p else {})},
            "conflict_semantics": {"rule": "under every policy the first commit takes the contested effect; the policy only decides which actor "
                                           "commits first", "scenario_effect": {sid: s["scenario_effect"] for sid, s in effects["by_scenario"].items()}},
            "stale_decision_consequences": {"second_commit": {k: st["second_commit"][k] for k in ("n", "staleness", "consequence")},
                                            "stale_and_invalid_borne_by": st["stale_and_invalid_borne_by"]},
            "request_waste": ({rk: {k: waste["by_runtime"][rk][p][k] for k in ("provider_calls_launched", "responses_consumed", "unused_responses",
                                                                                 "unused_input_tokens", "unused_output_tokens", "hp09_violations")}
                               for rk in done if p in waste["by_runtime"][rk]}
                              or "not run live (P2/P3 use the P1/P4 barrier unchanged; only the order differs)"),
            "wall_clock_responsiveness": ({rk: {"median_to_first_commit_ms": (clock[rk][p]["b0"]["to_first_commit_ms"] or {}).get("median"),
                                                "median_to_pair_commit_ms": (clock[rk][p]["b0"]["to_pair_commit_ms"] or {}).get("median"),
                                                "median_first_commit_after_first_ready_ms": (clock[rk][p]["b0"]["first_commit_after_first_ready_ms"] or {}).get("median"),
                                                "game_time_during_inference_or_barrier": clock[rk][p]["game_time"]["game_time_advanced_during_inference_or_barrier"]}
                                           for rk in done if p in clock[rk]}
                                          or "not run live"),
            "runtime_replication": ({k: v["classification"] for k, v in repl["comparisons"].items() if k.startswith(SHORT[p] + "/")}
                                    or "replay only (no runtime-specific code on the replay path)"),
        }
    return {
        "promoted": False,
        "production_scheduler_changed": False,
        "decision": "NO_PROMOTION: M6G.2 collects evidence only (spec sections 49, 53, 54). Any production commitment policy is a separate, "
                    "explicitly started milestone.",
        "no_global_score": "latency dependence, actor priority, conflict semantics, stale-decision consequences, request waste, wall-clock "
                           "responsiveness and runtime replication are reported side by side and never combined (spec section 52)",
        "properties_by_policy": props,
    }


def next_step_gate(labels: list[str], effects: dict[str, Any], repl: dict[str, Any]) -> dict[str, Any]:
    stale_remains = any(s["stale_and_invalid_moves_with_order"] for s in effects["by_scenario"].values())
    candidates = []
    if BARRIER_PRESERVES in labels and BARRIER_REMOVES_RACE in labels:
        candidates.append("P1: production's world semantics unchanged and no replacement requests (an engineering change only)")
    if ALTERNATING_BALANCED in labels:
        candidates.append("P4: first commit independent of latency and no persistent actor priority (a gameplay-policy change: it moves who "
                          "wins contested resources relative to P0)")
    return {
        "started": False,
        "note": "no follow-up milestone starts automatically (spec sections 54, 81); the options below only record which trigger conditions "
                "the evidence meets",
        "options": [
            {"option": "M6G.3 - Commitment Policy Production Promotion & Episode Validation",
             "condition": "one policy has desirable explicit properties and no unresolved semantic regressions",
             "condition_met": bool(candidates) and repl["overall"] != RT_DEPENDENT, "evidence": candidates},
            {"option": "M6G.3 - Stale Decision Revalidation / Cancellation Policies",
             "condition": "stale decisions remain the main unresolved risk",
             "condition_met": stale_remains,
             "evidence": "every policy still applies stale decisions unchanged; it only chooses whose decision is second (scripted C4/C5/C6: the "
                         "second decision is refused or springs the trap)"},
            {"option": "focused scheduler lifecycle fix",
             "condition": "the lifecycle race is the only remaining issue",
             "condition_met": RACE_SUPPORTED in labels and not stale_remains,
             "evidence": "the race is supported and a barrier removes it" + ("; stale decisions also remain, so it is not the only issue" if stale_remains else "")},
            {"option": "move to the next major harness variable",
             "condition": "commitment semantics are sufficiently understood",
             "condition_met": LATENCY_COUPLED in labels and BARRIER_PRESERVES in labels and repl["overall"] == RT_INDEPENDENT,
             "evidence": "ordering semantics are characterized; the stale-decision consequence is a separate open item"},
        ],
    }


# -- bookkeeping ---------------------------------------------------------------------------


def reproduction_summary(data: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for rk in LIVE_FILES:
        rows, rep = data["live_raw"][rk], data["repro"][rk]
        if not rows:
            continue
        out[rk] = {
            "live_trials": len(rows), "reproductions": len(rep), "reproduced": sum(r["reproduced"] for r in rep.values()),
            "not_reproduced": {tid: r["mismatches"] for tid, r in rep.items() if not r["reproduced"]},
            "records_with_corrected_derived_fields": sum(bool(rec["live_value_differed"]) for r in rep.values() for b in r["boundaries"]
                                                         for rec in b["records"].values()),
            "event_log_hash_corrected": sum(r["live_event_log_hash_differed"] for r in rep.values()),
            "rows_written_by_the_fixed_probe": sum(bool(r.get("live_state_untouched_by_assembly")) for r in rows),
            "model_calls": 0,
        }
    return out


def live_call_count(data: dict[str, Any]) -> dict[str, Any]:
    calls = {rk: sum(t["lifecycle"]["provider_calls"] for t in rows) for rk, rows in data["live_raw"].items() if rows}
    aborted = sum((e.get("lifecycle") or {}).get("provider_calls", 0) for e in data["provider_events"] if str(e.get("phase", "")).endswith("_abort"))
    preflight = sum(1 for e in data["provider_events"] if str(e.get("phase", "")).startswith("preflight"))
    return {"trial_calls": calls, "planned_trial_calls": {rk: expected_calls(rk) for rk in LIVE_FILES}, "aborted_trial_calls": aborted,
            "preflight_connectivity_calls": preflight, "total": sum(calls.values()) + aborted + preflight,
            "replay_and_reproduction_calls": 0}


def incidental_observations(lbs: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [{"trial_id": b["trial_id"], "boundary": b["boundary"], "actor_id": a, "decision": decision_label(b["records"][a]),
              "listed_at_snapshot": b["records"][a]["listed_at_snapshot"], "counterfactual_outcome": b["records"][a]["counterfactual"]["effective_outcome"],
              "apply_outcome": b["records"][a]["apply_outcome"], "commit_position": 1 if b["first_commit_actor"] == a else 2}
             for b in lbs if b["derived_ok"] for a in ACTORS if b["records"][a]["staleness_class"] == INVALID_AT_SNAPSHOT]
    return {
        "invalid_at_snapshot_live_decisions": cases,
        "reading": "decisions the agent's legal list offered but the Engine refused on their own snapshot (not staleness, and the same in "
                   "either commit position): a legality-surface gap as in the M6G audit (section 7), outside M6G.2's scope" if cases else "none",
    }


# -- required tables (spec 66-73) ------------------------------------------------------------


def replay_table_rows(replay: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spec section 67, grouped: rows with the same decisions, commit order and outcome collapse; the policies that produced
    each group are listed with their row counts."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in replay:
        key = (r["scenario_id"], _pair_label(r["records"]), r["first_commit_actor"], r["second_commit_actor"], r["first_result"],
               r["second_result"], r["first_action_cost"], r["second_action_cost"], r["world_state_hash_after"], r["event_log_hash"])
        groups[key].append(r)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: (SCENARIO_IDS.index(kv[0][0]), kv[0][1], kv[0][2] != ACTOR_A, str(kv[0][8]))):
        sid, pair, first, second, r1, r2, c1, c2, world, event = key
        rows.append({"scenario_id": sid, "decision_pair": pair, "first_commit": first, "second_commit": second, "first_result": r1,
                     "second_result": r2, "first_cost": c1, "second_cost": c2, "world_hash": world, "event_hash": event,
                     "policies": {SHORT[p]: n for p, n in sorted(Counter(r["commitment_policy"] for r in rs).items())},
                     "sources": dict(sorted(Counter(r["source"] for r in rs).items())), "pairs": len({r["pair_id"] for r in rs}), "rows": len(rs)})
    return rows


def _yn(v: bool) -> str:
    return "yes" if v else "no"


def required_tables(inv, replay_rows, effects, lat_rows, rpm, live_c, stale, waste, clock, repl) -> str:
    L: list[str] = []
    L += ["### Policy table (spec section 66)", "", "| Policy | Wait for both? | First-commit rule | Depends on latency? | Depends on actor id? | Deterministic? |",
          "|---|---|---|---|---|---|"]
    for r in policy_table():
        L.append(f"| {r['policy_id']} | {_yn(r['waits_for_both'])} | {r['first_commit_rule']} | {_yn(r['depends_on_latency'])} | "
                 f"{_yn(r['depends_on_actor_id'])} | {_yn(r['deterministic'])} |")

    L += ["", "### Replay table (spec section 67) - zero-call replay, grouped", "",
          f"{inv['rows']} rows over {inv['pairs']} decision pairs; rows with the same decisions, commit order and outcome are grouped. "
          "Every row is in `replay_trials.jsonl`.", "",
          "| Decision pair (Black / White) | Scenario | Policies (rows) | First commit | Second commit | First result | Second result | First cost | Second cost | World hash | Event hash |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in replay_rows:
        pol = ", ".join(f"{p}×{n}" for p, n in g["policies"].items())
        L.append(f"| {g['decision_pair']} | {g['scenario_id']} | {pol} ({g['pairs']} pairs) | {WHO[g['first_commit']]} | {WHO[g['second_commit']]} | "
                 f"{g['first_result']} | {g['second_result']} | {g['first_cost']:g} s | {g['second_cost']:g} s | `{_short(g['world_hash'])}` | `{_short(g['event_hash'])}` |")

    L += ["", "### Commitment effect by scenario (spec sections 30-32) - Black-first vs White-first on the same pair", "",
          "| Scenario | Pairs (sources) | Effect by decision pair | Scenario effect | Final state changes | Cost changes | Refusal changes | Whose decision is stale-and-invalid moves with the order | Contested resource to the first commit, both orders |",
          "|---|---|---|---|---|---|---|---|---|"]
    for sid, s in effects["by_scenario"].items():
        src = ", ".join(f"{k} {v}" for k, v in s["by_source"].items())
        by_dp = "; ".join(f"{dp}: " + ", ".join(f"{lab} ×{n}" for lab, n in labs.items()) for dp, labs in s["effect_by_decision_pair"].items())
        L.append(f"| {sid} | {s['pairs']} ({src}) | {by_dp} | {s['scenario_effect']} | {s['final_state_changes']}/{s['pairs']} | "
                 f"{s['cost_changes']}/{s['pairs']} | {s['refusal_changes']}/{s['pairs']} | {s['stale_and_invalid_moves_with_order']}/{s['pairs']} | "
                 f"{s['resource_to_first_commit_both_orders']} |")

    L += ["", "### Latency-coupling table (spec section 68) - live trials, grouped", "",
          "| Runtime | Scenario | Policy | Boundary | First-ready actor | First-commit actor | Same? | Shared conflict? | Resource winner | n (reps) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in lat_rows:
        tie = " (same tick)" if g["same_tick_tie"] else ""
        L.append(f"| {g['runtime']} | {g['scenario_id']} | {SHORT[g['policy']]} | b{g['boundary']} | {WHO[g['first_ready']]}{tie} | {WHO[g['first_commit']]} | "
                 f"{_yn(g['same'])} | {_yn(g['shared_conflict'])} | {WHO.get(g['resource_winner'], '-')} | {g['n']} ({', '.join(g['trials'])}) |")
    if not lat_rows:
        L.append("| (no live trials yet) | | | | | | | | | |")

    L += ["", "### Actor-priority table (spec section 69)", "", "| Policy | Source | Black first | White first | Actor-counterbalanced? |", "|---|---|---|---|---|"]
    counter_note = {P0: "no - follows latency (call order alternated to spread the 50 ms head start)",
                    P1: "no - follows latency, as P0", P2: "no - Black always first (read with P3 as a mirror pair)",
                    P3: "no - White always first (read with P2 as a mirror pair)", P4: "yes - alternates per boundary from a counterbalanced parity"}
    for p in POLICY_IDS:
        cap = rpm["replay_captured"][p]["_all_scenarios"]
        L.append(f"| {SHORT[p]} | replay, captured pairs{' (assigned parity)' if p == P4 else ''} | {_pct(cap['first_commit'][ACTOR_A], cap['n'])} | "
                 f"{_pct(cap['first_commit'][ACTOR_B], cap['n'])} | {counter_note[p]} |")
        for rk in LIVE_FILES:
            blk = live_c[rk].get(p, {}).get("_all_scenarios") if live_c[rk].get("status") == "done" else None
            if not blk:
                continue
            extra = f"; balanced trials {blk['p4']['balanced_trials']}/{blk['p4']['trials']}, initial parity {blk['p4']['initial_parity']}" if "p4" in blk else ""
            L.append(f"| {SHORT[p]} | live {rk} ({blk['boundaries']} boundaries) | {_pct(blk['first_commit'][ACTOR_A], blk['boundaries'])} | "
                     f"{_pct(blk['first_commit'][ACTOR_B], blk['boundaries'])} | {counter_note[p]}{extra} |")

    L += ["", "### Staleness table (spec section 70) - second-committed decisions", "",
          "| Policy | Source | n | Relevant-state stale | Stale-but-valid | Stale-and-invalid | Refusal (invalid at snapshot) | Different cost | Stale-and-invalid borne by Black / White |",
          "|---|---|---|---|---|---|---|---|---|"]

    def srow(p: str, src: str, s: dict[str, Any]) -> str:
        sc, by = s["second_commit"], s["stale_and_invalid_borne_by"]
        f = lambda v: "-" if v is None else f"{v:.0%}"  # noqa: E731
        return (f"| {SHORT[p]} | {src} | {sc['n']} | {f(sc['relevant_state_stale_rate'])} | {f(sc['stale_but_valid_rate'])} | {f(sc['stale_and_invalid_rate'])} | "
                f"{f(sc['refusal_rate'])} ({sc['refusals_invalid_at_snapshot']}) | {f(sc['different_cost_rate'])} | {by[ACTOR_A]} / {by[ACTOR_B]} |")
    for pop, label in (("replay_captured", "replay, captured"), ("replay_scripted", "replay, scripted")):
        for p in POLICY_IDS:
            L.append(srow(p, label, stale[pop][p]))
    for rk, blk in stale["live"].items():
        for p in LIVE_POLICIES:
            if p in blk:
                L.append(srow(p, f"live {rk}", blk[p]))

    L += ["", "### Request-lifecycle table (spec section 71)", "",
          "| Runtime | Policy | Actor | Decision boundary | Requests (extra, HP09) | Responses ready | Responses committed | Unused responses | Extra tokens |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in waste["table_71"]:
        L.append(f"| {r['runtime']} | {SHORT[r['policy']]} | {WHO[r['actor']]} | b{r['boundary']} | {r['requests']} ({r['extra_requests']}, {r['hp09']}) | "
                 f"{r['responses_ready']} | {r['responses_committed']} | {r['unused']} | {r['extra_tokens']} |")
    if not waste["table_71"]:
        L.append("| (no live trials yet) | | | | | | | | |")

    L += ["", "### Wall-clock table (spec section 72)", "",
          "| Runtime | Policy | Boundary | n | Median to first ready (ms) | Median to first commit (ms) | Median to pair commit (ms) | Median first commit after first ready (ms) | Game time advanced during inference / barrier? |",
          "|---|---|---|---|---|---|---|---|---|"]
    for rk in LIVE_FILES:
        if clock[rk].get("status") != "done":
            continue
        for p in LIVE_POLICIES:
            blk = clock[rk].get(p)
            if not blk:
                continue
            g = blk["game_time"]
            for bk in sorted(k for k in blk if k.startswith("b")):
                w = blk[bk]
                gt = (f"{_yn(g['game_time_advanced_during_inference_or_barrier'])} ({g['ticks_advanced_during_inference']} inference ticks, "
                      f"{g['ticks_advanced_while_a_decision_was_held']} held ticks advanced; first commit on its unchanged snapshot "
                      f"{w['first_commit_on_unchanged_snapshot']})")
                L.append(f"| {rk} | {SHORT[p]} | {bk} | {w['boundaries']} | {_med(w['to_first_ready_ms'])} | {_med(w['to_first_commit_ms'])} | "
                         f"{_med(w['to_pair_commit_ms'])} | {_med(w['first_commit_after_first_ready_ms'])} | {gt} |")

    L += ["", "### Runtime table (spec section 73)", "", "| Policy / scenario | Native result | Strands result | Classification |", "|---|---|---|---|"]
    for name, c in repl["comparisons"].items():
        L.append(f"| {name} | {c['native']} | {c['strands']} | {c['classification']} |")
    if not repl["comparisons"]:
        L.append("| (no live trials yet) | | | |")
    return "\n".join(L)


# -- entry point ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    data = load()
    replay = data["replay"]
    lbs = live_boundaries(data["live"])
    inv = replay_invariants(replay, data["reuse"])
    rpm = replay_policy_metrics(replay)
    effects = commitment_effects(replay)
    stale = staleness_metrics(replay, lbs)
    live_c = live_commitment(data["live"], lbs)
    lat_rows = latency_table_rows(lbs)
    waste = request_waste(data["live"], data["lifecycle_events"])
    clock = wall_clock(data["live"], lbs)
    repl = runtime_replication(data["live"], lbs)
    overall = overall_classification(data, inv, effects, lbs, repl)
    gate = next_step_gate(overall["labels"], effects, repl)
    promotion = promotion_decision(rpm, stale, live_c, waste, clock, repl, effects)
    reproductions = reproduction_summary(data)
    calls = live_call_count(data)
    incidental = incidental_observations(lbs)
    replay_rows = replay_table_rows(replay)

    write_json(out_path("commitment_policy_metrics.json"), {
        "policy_table": policy_table(), "replay_invariants": inv, "replay": rpm,
        "commitment_effect_by_scenario": effects["by_scenario"], "commitment_effect_pairs": effects["pairs"],
        "live": live_c, "live_reproductions": reproductions, "latency_table_68": lat_rows, "replay_table_67": replay_rows,
    })
    write_json(out_path("staleness_metrics.json"), stale)
    write_json(out_path("request_waste_metrics.json"), waste)
    write_json(out_path("wall_clock_metrics.json"), clock)
    write_json(out_path("runtime_replication.json"), repl)
    write_json(out_path("promotion_decision.json"), promotion)

    def live_brief(rk: str) -> dict[str, Any]:
        if live_c[rk].get("status") != "done":
            return {"status": "not run"}
        out = {}
        for p in LIVE_POLICIES:
            b, w = live_c[rk][p]["_all_scenarios"], waste["by_runtime"][rk][p]
            out[SHORT[p]] = {"trials": b["trials"], "boundaries": b["boundaries"], "policy_order_respected": f"{b['policy_order_respected']}/{b['boundaries']}",
                             "first_ready_is_first_commit": f"{b['latency_commit_coupling']['first_ready_is_first_commit']}/{b['boundaries']}",
                             "black_first": f"{b['first_commit'][ACTOR_A]}/{b['boundaries']}",
                             "resource_to_first_commit": f"{b['resource_to_first_commit']}/{b['resource_contests']}",
                             "replay_matches_live": f"{b['replay_matches_live_boundary0']}/{b['trials']}",
                             "reproduced_zero_call": f"{b['reproduced_zero_call']}/{b['trials']}",
                             "hp09_violations": w["hp09_violations"], "unused_responses": w["unused_responses"],
                             "game_time_during_inference_or_barrier": clock[rk][p]["game_time"]["game_time_advanced_during_inference_or_barrier"]}
        return out

    summary = {
        "milestone": "M6G.2",
        "title": "Commitment Policy Counterbalancing",
        "overall_classification": overall["labels"],
        "classification_evidence": overall["evidence"],
        "model_id": MODEL_ID, "provider": "bedrock", "communication_policy": "SOCIAL_OFF", "personality": "default vs default",
        "counts": {"replay_pairs": inv["pairs"], "replay_rows": inv["rows"], "replay_rows_by_source": inv["rows_by_source"],
                   "live_trials": {rk: len(rows) for rk, rows in data["live_raw"].items()}, "live_boundaries": {rk: sum(b["runtime_key"] == rk for b in lbs) for rk in LIVE_FILES},
                   "lifecycle_events": len(data["lifecycle_events"])},
        "new_live_model_calls": calls,
        "replay_invariants": {k: inv[k] for k in ("decision_hashes_identical_across_policies", "snapshot_hashes_identical_across_policies",
                                                  "p0_equals_p1_same_ready_order", "rows_applying_the_policy_order")} | {"pairs": inv["pairs"], "rows": inv["rows"]},
        "commitment_effect_by_scenario": {sid: {"scenario_effect": s["scenario_effect"], "labels": s["labels"], "scripted_pair": s["scripted_pair_effect"]}
                                          for sid, s in effects["by_scenario"].items()},
        "live": {rk: live_brief(rk) for rk in LIVE_FILES},
        "live_reproductions": reproductions,
        "harness_bug_fixed_during_milestone": {
            "what": "commitment_probe deep-copied the live GameState for each decision's counterfactual; deepcopy kept the probe's "
                    "instance-level apply_action/update wrappers, so the counterfactual applied to the LIVE game (after the loop) "
                    "instead of the copy",
            "affected": "live records' counterfactual, legal_at_snapshot, staleness_class and consequence (and the boundary result "
                        "strings), and the trial's order-sensitive event_log_hash. Not affected: commit order, timing, world and effect "
                        "hashes, event multiset, costs, request lifecycle, and the zero-call replay matrix (a fresh state per replay)",
            "fix": "commitment_probe.unpatched_copy strips the wrappers from the copy; counterfactual_on_state refuses a patched state; "
                   "event_log_hash is taken before assembly; rows record live_state_untouched_by_assembly",
            "correction": "every live trial is re-run with zero calls through the fixed probe (run_m6g2_live --reproduce) with scripted "
                          "stand-ins answering the recorded payloads; world, decision and apply fields must match exactly, then the "
                          "reproduction supplies the derived fields",
            "reproductions": reproductions,
        },
        "incidental_observations": incidental,
        "runtime_replication": {"status": repl["status"], "overall": repl["overall"],
                                "comparisons": {k: v["classification"] for k, v in repl["comparisons"].items()}},
        "promotion": {"promoted": False, "production_scheduler_changed": False},
        "next_step_gate": gate,
        "notes": [
            "World semantics come from zero-call replay of captured pairs: the same decisions, snapshots and Engine under every policy, "
            "so provider/model variance cannot appear as a policy effect.",
            "Live calls validate the real-loop barrier and the request lifecycle only; live decisions are never compared across policies "
            "as distributions.",
            "Trap, disarm and search-after-trap cases use scripted decisions (M6G C4 trap_type confound); P2/P3 are replay-only.",
            "No global policy score is computed.",
        ],
    }
    write_json(out_path("summary.json"), summary)
    summary["_tables_markdown"] = required_tables(inv, replay_rows, effects, lat_rows, rpm, live_c, stale, waste, clock, repl)
    splice_generated_tables(out_path("summary.md"), summary["_tables_markdown"])
    return summary


def splice_generated_tables(path: str, tables: str) -> bool:
    """summary.md is written by hand except for the required tables, which
    are regenerated between two marker lines so they always match the
    data. A file without the markers is left untouched."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start, end = text.find(TABLES_BEGIN), text.find(TABLES_END)
    if start < 0 or end < start:
        return False
    body = text[: start + len(TABLES_BEGIN)] + "\n\n" + tables.replace("### ", "#### ") + "\n\n" + text[end:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return True


if __name__ == "__main__":
    out = run()
    tables = out.pop("_tables_markdown")
    print(json.dumps({k: out[k] for k in ("overall_classification", "classification_evidence", "counts", "new_live_model_calls", "replay_invariants",
                                          "commitment_effect_by_scenario", "live", "live_reproductions", "incidental_observations",
                                          "runtime_replication", "next_step_gate")}, indent=2, default=str))
    print()
    print(tables)
    print(f"\n(outputs in {OUT_DIR})")
