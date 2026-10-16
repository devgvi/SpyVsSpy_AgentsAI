"""M6G analysis (spec sections 23-24, 37-43, 55-61): reads every trial file
in `evaluation_runs/m6g-decision-scheduling-v1/` and writes the metrics
files plus `summary.json`. Analysis only - no model calls, no
LLM-as-a-Judge; every number comes from recorded hashes, decisions, apply
outcomes and event logs. Prints the spec's required tables (sections
55-60) as markdown.

    python -m src.evaluation.run_m6g_analysis
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import statistics
from collections import Counter
from typing import Any

from src.evaluation.conflict_scenarios import CONFLICT_FAMILIES, build_conflict_scenarios
from src.evaluation.decision_scheduling import (
    APPLY_FAILURE_EVENT_ONLY,
    APPLY_REFUSED_NO_OP,
    S0_FROZEN_SNAPSHOT_BASELINE,
    S1_REVERSED_CALL,
    S2_REVERSED_APPLY,
    S3_SEQUENTIAL_A_FIRST,
    S4_SEQUENTIAL_B_FIRST,
    SCHEDULE_PLANS,
    SCHEDULING_CONDITIONS,
)
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS
from src.evaluation.m6g_live import out_path, read_jsonl, write_json
from src.evaluation.scheduling_analysis import (
    ACTOR_ORDER_EFFECT,
    APPLICATION_ORDER_EFFECT,
    APPLY_TIME_INVALIDATION_EFFECT,
    MIXED,
    MODEL_CALL_ORDER_EFFECT,
    NO_SCHEDULING_EFFECT,
    SEQUENTIAL_INFORMATION_EFFECT,
    apply_time_invalidated,
    classify_actor_order,
    classify_apply_order_pair,
    classify_call_order,
    classify_snapshot_timing,
    compare_distributions,
    contested_effects,
    contested_label,
    decision_label,
    differing_layers,
    distribution,
    event_multiset_hash,
    first_difference,
    intent_label,
    modal,
    rate,
    request_identity,
    world_outcome,
)
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
PROVIDER_NAME = "bedrock"

S0, S1, S2, S3, S4 = S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL, S2_REVERSED_APPLY, S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST
SHORT = {S0: "S0", S1: "S1", S2: "S2", S3: "S3", S4: "S4"}

# Predeclared in call_plan.json before any live call.
PROVIDER_CONTROL_TRIGGER = "modal decision differs between S0 and S1, or TVD(S0, S1) >= 0.3, for an actor"

# Written after the analysis; every number is also in the metrics files.
KEY_FINDING = (
    "Snapshot fairness holds wherever the schedule is explicit: every frozen-snapshot trial with live decisions (60 S0, 30 S1, "
    "30 S2 replays on Native; 20 on Strands) observed one pre-action world, each actor's request was byte-identical across S0 and "
    "S1, and the production Game._step builds both observations before either apply. Model-call order had no effect (C0/C2/C4, "
    "TVD <= 0.2, p >= 0.72; the null replicates on Strands). "
    "Application order matters wherever the Engine has single-owner shared state: C2 (who pays the 1.0 s unlock vs the 0.3 s "
    "re-open), C4 and C5 (first mover wins, the loser is refused silently), and live in 10/10 C2 decision pairs. Snapshot timing "
    "is the largest behavioral lever: when the second actor can see the first actor's effect (C2) it abandons its baseline "
    "decision in 20/20 Native and 20/20 Strands trials, for both identities; when it cannot (C0, C4) nothing detectable changes. "
    "In live LLM play the production loop applies same-snapshot decisions in model completion order, so provider latency - not "
    "the hard-coded Black-first order - decides shared-state conflicts, even though the game-time freeze holds."
)
NOTES = {
    "provider_control": "C4/white_spy: 10 isolated repeats of the byte-identical request gave 10/10 fallback WAIT (place_trap without trap_type), "
                        "vs 4/10 and 5/10 in the interleaved Phase C cells and 7/10 in Phase D's S0. The spread is provider-side non-stationarity "
                        "for a fixed request, not a call-order effect (S0 vs S1 TVD 0.2, p 0.72); the paired, interleaved design is what keeps it "
                        "out of the call-order comparison.",
    "c4_live_conflict": "The C4 trap race never reached the Engine live: every place_trap intent omitted trap_type (validation error "
                        "missing_target) and fell back to WAIT under this no-repair harness, so C4's apply-order and invalidation evidence is "
                        "Phase B's deterministic pair plus the production-loop test.",
    "runtime_replication": "S0 vs S4 is PARTIAL_REPLICATION: on both runtimes Black abandons the interact once it sees the open door, but "
                           "Strands' Black then heads for door_archives_corridor (10/10) where Native's goes through door_security_corridor (9/10). "
                           "Strands ran about 7.5 h after the Native phases; Native's S0 baseline was identical across Phase C and Phase D.",
}


def _short(h: str | None, n: int = 12) -> str:
    return (h or "")[:n]


def _select(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]


def _labels(rows: list[dict[str, Any]], actor_id: str) -> list[str]:
    return [decision_label(r["records"][actor_id]) for r in rows]


def _by_rep(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {r["rep"]: r for r in rows}


def _contested_summary(row: dict[str, Any]) -> str:
    return contested_label(row)


# -- Phase B -------------------------------------------------------------------


def _order_summary(row: dict[str, Any]) -> dict[str, Any]:
    ra, rb = row["records"][ACTOR_A], row["records"][ACTOR_B]
    return {
        "condition": row["condition"],
        "apply_order": row["plan"]["apply_order"],
        "action_a": ra["decision"],
        "action_b": rb["decision"],
        "a_legal_at_snapshot": ra["legal_at_snapshot"],
        "b_legal_at_snapshot": rb["legal_at_snapshot"],
        "a_valid_at_apply": ra["legal_at_apply_time"],
        "b_valid_at_apply": rb["legal_at_apply_time"],
        "a_listed_at_apply": ra["listed_legal_at_apply_time"],
        "b_listed_at_apply": rb["listed_legal_at_apply_time"],
        "a_apply_outcome": ra["apply_outcome"],
        "b_apply_outcome": rb["apply_outcome"],
        "a_events": ra["apply_event_types"],
        "b_events": rb["apply_event_types"],
        "world_state_hash_after": row["world_state_hash_after"],
        "event_log_hash": row["event_log_hash"],
        "event_multiset_hash": event_multiset_hash(row),
        "contested_effects": contested_effects(row),
    }


def deterministic_application_order_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios = build_conflict_scenarios()
    out = {}
    for family in CONFLICT_FAMILIES:
        s = scenarios[family]
        by_cond = {r["condition"]: r for r in rows if r["scenario_family"] == family}
        s0, s1, s2, s3, s4 = (by_cond[c] for c in (S0, S1, S2, S3, S4))
        labels = classify_apply_order_pair(s0, s2)
        out[family] = {
            "same_starting_state": len({r["world_state_hash_pre_action"] for r in (s0, s1, s2, s3, s4)}) == 1,
            "description": s.description,
            "shared_state": s.shared_state,
            "side_effect_visibility": s.side_effect_visibility,
            "negative_control": s.negative_control,
            "predeclared_expected_apply_order_effect": s.expected_apply_order_effect,
            "observed_labels": labels,
            "expectation_held": (APPLICATION_ORDER_EFFECT in labels) == s.expected_apply_order_effect,
            "orders": {"a_first": _order_summary(s0), "b_first": _order_summary(s2)},
            "first_difference_s0_vs_s2": first_difference(s0, s2),
            "differing_layers_s0_vs_s2": differing_layers(s0, s2),
            "event_log_order_only_difference": s0["event_log_hash"] != s2["event_log_hash"] and event_multiset_hash(s0) == event_multiset_hash(s2)
            and s0["world_state_hash_after"] == s2["world_state_hash_after"],
            "call_order_s0_vs_s1": {"identical_outcome": world_outcome(s0) == world_outcome(s1) and s0["event_log_hash"] == s1["event_log_hash"],
                                    "first_difference": first_difference(s0, s1)},
            "sequential_with_fixed_decisions": {
                "S3_world_matches_S0": s3["world_state_hash_after"] == s0["world_state_hash_after"],
                "S4_world_matches_S2": s4["world_state_hash_after"] == s2["world_state_hash_after"],
                "S3_world_changed_before_second_snapshot": s3["world_changed_before_second_snapshot"],
                "S4_world_changed_before_second_snapshot": s4["world_changed_before_second_snapshot"],
                "S3_second_actor_saw_first_action": s3["second_actor_saw_first_action"],
                "S4_second_actor_saw_first_action": s4["second_actor_saw_first_action"],
                "predeclared_second_observation_changes": s.expected_second_observation_changes,
                "S3_second_actor_decision_still_valid": s3["records"][s3["second_actor"]]["first_pass_valid"],
                "S4_second_actor_decision_still_valid": s4["records"][s4["second_actor"]]["first_pass_valid"],
                "S3_second_actor_validation_error": s3["records"][s3["second_actor"]]["validation_error"],
                "S4_second_actor_validation_error": s4["records"][s4["second_actor"]]["validation_error"],
            },
        }
    return out


# -- live helpers --------------------------------------------------------------


def _cell_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for actor_id in ACTORS:
        recs = [r["records"][actor_id] for r in rows]
        out[actor_id] = {
            "n": len(recs),
            "action_distribution": distribution(r["decision"]["action"] for r in recs),
            "target_distribution": distribution(str(r["decision"]["target_id"]) for r in recs),
            "decision_distribution": distribution(decision_label(r) for r in recs),
            "intent_distribution": distribution(intent_label(r) for r in recs),
            "first_pass_validity": rate(r["first_pass_valid"] for r in recs),
            "repair_rate": 0.0,  # no repair on this path by design (spec section 36)
            "fallback_rate": rate(r["fallback_to_wait"] for r in recs),
            "validation_errors": distribution(r["validation_error"] for r in recs if r["validation_error"]),
        }
    return out


def _cell_execution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recs = [r["records"][a] for r in rows for a in ACTORS]
    acted = [r for r in recs if r["decision"]["action"] != "wait"]
    same_target = [
        r["records"][ACTOR_A]["decision"]["action"] != "wait" and r["records"][ACTOR_B]["decision"]["action"] != "wait"
        and r["records"][ACTOR_A]["decision"]["target_id"] == r["records"][ACTOR_B]["decision"]["target_id"]
        for r in rows
    ]
    return {
        "n_trials": len(rows),
        "apply_time_invalidation_rate": rate(apply_time_invalidated(r) for r in recs),
        "no_op_refusal_rate": rate(r["apply_outcome"] in (APPLY_REFUSED_NO_OP, APPLY_FAILURE_EVENT_ONLY) for r in acted),
        "same_target_rate": rate(same_target),
        "apply_outcomes": distribution(r["apply_outcome"] for r in recs),
        "contested_effects": distribution(_contested_summary(r) for r in rows),
        "world_outcomes": len({world_outcome(r) for r in rows}),
    }


def _paired_divergence(base: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    b, o = _by_rep(base), _by_rep(other)
    reps = sorted(set(b) & set(o))
    pairs = [(b[k], o[k]) for k in reps]
    return {
        "n_pairs": len(pairs),
        "world_state_divergence_rate": rate(x["world_state_hash_after"] != y["world_state_hash_after"] for x, y in pairs),
        "event_log_divergence_rate": rate(event_multiset_hash(x) != event_multiset_hash(y) for x, y in pairs),
        "event_log_order_divergence_rate": rate(x["event_log_hash"] != y["event_log_hash"] for x, y in pairs),
        "first_difference_distribution": distribution(first_difference(x, y) for x, y in pairs),
    }


# -- Phase C: call order ------------------------------------------------------


def provider_control_triggers(c_rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    triggers = []
    for family in sorted({r["scenario_family"] for r in c_rows}):
        s0, s1 = _select(c_rows, scenario_family=family, condition=S0), _select(c_rows, scenario_family=family, condition=S1)
        for actor_id in ACTORS:
            comp = compare_distributions(_labels(s0, actor_id), _labels(s1, actor_id))
            if comp["modal_x"] != comp["modal_y"] or comp["tvd"] >= 0.3:
                triggers.append((family, actor_id))
    return triggers


def call_order_metrics(c_rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    triggers = set(provider_control_triggers(c_rows))
    for family in sorted({r["scenario_family"] for r in c_rows}):
        s0, s1 = _select(c_rows, scenario_family=family, condition=S0), _select(c_rows, scenario_family=family, condition=S1)
        cls = classify_call_order(s0, s1)
        per_actor_resolution = {}
        for actor_id in ACTORS:
            ctrl = [c for c in controls if c["scenario_family"] == family and c["actor_id"] == actor_id]
            entry: dict[str, Any] = {"control_triggered": (family, actor_id) in triggers, "control_n": len(ctrl)}
            if ctrl:
                ctrl_labels = [c["decision_label"] for c in ctrl]
                entry["control_distribution"] = distribution(ctrl_labels)
                entry["control_request_matches_phase_c"] = all(
                    c["provider_request_hash"] in cls["per_actor"][actor_id]["request_identity"]["provider_request_hash"] for c in ctrl
                )
                entry["s0_vs_control"] = compare_distributions(_labels(s0, actor_id), ctrl_labels)
                entry["s1_vs_control"] = compare_distributions(_labels(s1, actor_id), ctrl_labels)
                pooled = _labels(s0, actor_id) + _labels(s1, actor_id) + ctrl_labels
                entry["pooled_distribution"] = distribution(pooled)
                entry["pooled_modal_share"] = round(Counter(pooled).most_common(1)[0][1] / len(pooled), 4)
            per_actor_resolution[actor_id] = entry
        supported = [a for a in ACTORS if cls["per_actor"][a]["supported"]]
        resolved = cls["classification"]
        for a in supported:
            res = per_actor_resolution[a]
            if res.get("control_n") and not res["s0_vs_control"]["supported"] and not res["s1_vs_control"]["supported"]:
                resolved = NO_SCHEDULING_EFFECT  # the isolated same-request spread covers the S0/S1 gap
        out[family] = {
            **cls,
            "call_positions": {
                "S0": {ACTOR_A: 1, ACTOR_B: 2},
                "S1": {ACTOR_A: 2, ACTOR_B: 1},
            },
            "paired": _paired_divergence(s0, s1),
            "behavior": {"S0": _cell_behavior(s0), "S1": _cell_behavior(s1)},
            "execution": {"S0": _cell_execution(s0), "S1": _cell_execution(s1)},
            "provider_control": per_actor_resolution,
            "classification_after_provider_control": resolved,
        }
    return out


# -- Phase C S2 replays: live apply order ------------------------------------


def live_application_order_metrics(c_rows: list[dict[str, Any]], replays: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for family in sorted({r["scenario_family"] for r in replays}):
        sources = {r["trial_id"]: r for r in _select(c_rows, scenario_family=family, condition=S0)}
        pairs = [(sources[r["decisions_reused_from"]], r) for r in replays if r["scenario_family"] == family]
        labelled = []
        for s0, s2 in pairs:
            labels = classify_apply_order_pair(s0, s2)
            labelled.append({
                "rep": s0["rep"],
                "decision_a": decision_label(s0["records"][ACTOR_A]),
                "decision_b": decision_label(s0["records"][ACTOR_B]),
                "labels": labels,
                "a_first": {"a_valid_at_apply": s0["records"][ACTOR_A]["legal_at_apply_time"], "b_valid_at_apply": s0["records"][ACTOR_B]["legal_at_apply_time"],
                            "contested": _contested_summary(s0), "world": _short(s0["world_state_hash_after"])},
                "b_first": {"a_valid_at_apply": s2["records"][ACTOR_A]["legal_at_apply_time"], "b_valid_at_apply": s2["records"][ACTOR_B]["legal_at_apply_time"],
                            "contested": _contested_summary(s2), "world": _short(s2["world_state_hash_after"])},
                "first_difference": first_difference(s0, s2),
            })
        by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in labelled:
            by_pair.setdefault((item["decision_a"], item["decision_b"]), []).append(item)
        out[family] = {
            "n_pairs": len(pairs),
            "decisions_reused_not_recomputed": all(r["decisions_reused_from"] for _, r in pairs),
            "application_order_effect_rate": rate(APPLICATION_ORDER_EFFECT in i["labels"] for i in labelled),
            "apply_time_invalidation_effect_rate": rate(APPLY_TIME_INVALIDATION_EFFECT in i["labels"] for i in labelled),
            "execution": {"S0": _cell_execution([s0 for s0, _ in pairs]), "S2_replay": _cell_execution([s2 for _, s2 in pairs])},
            "by_decision_pair": {
                f"{a} | {b}": {
                    "n": len(items),
                    "labels": distribution(",".join(i["labels"]) for i in items),
                    "a_first": items[0]["a_first"],
                    "b_first": items[0]["b_first"],
                }
                for (a, b), items in sorted(by_pair.items(), key=lambda kv: -len(kv[1]))
            },
            "pairs": labelled,
        }
    return out


# -- Phase D: snapshot timing -------------------------------------------------


def snapshot_timing_metrics(d_rows: list[dict[str, Any]], c_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for family in sorted({r["scenario_family"] for r in d_rows}):
        s0, s3, s4 = (_select(d_rows, scenario_family=family, condition=c) for c in (S0, S3, S4))
        s0_c = _select(c_rows, scenario_family=family, condition=S0)
        entry = {
            "S3_vs_S0": classify_snapshot_timing(s0, s3),
            "S4_vs_S0": classify_snapshot_timing(s0, s4),
            "S3_vs_S4": classify_actor_order(s3, s4),
            "first_actor_request_identity": {
                "S3_black_first": request_identity(s0 + s3, ACTOR_A)["identical"],
                "S4_white_first": request_identity(s0 + s4, ACTOR_B)["identical"],
            },
            "first_actor_decision_vs_s0": {
                "S3_black": compare_distributions(_labels(s0, ACTOR_A), _labels(s3, ACTOR_A)),
                "S4_white": compare_distributions(_labels(s0, ACTOR_B), _labels(s4, ACTOR_B)),
            },
            "paired": {"S3": _paired_divergence(s0, s3), "S4": _paired_divergence(s0, s4)},
            "second_actor_saw_first_action_rate": {SHORT[c]: rate(r["second_actor_saw_first_action"] for r in rows) for c, rows in ((S0, s0), (S3, s3), (S4, s4))},
            "same_snapshot_rate": {SHORT[c]: rate(r["same_snapshot"] for r in rows) for c, rows in ((S0, s0), (S3, s3), (S4, s4))},
            "behavior": {"S0": _cell_behavior(s0), "S3": _cell_behavior(s3), "S4": _cell_behavior(s4)},
            "execution": {"S0": _cell_execution(s0), "S3": _cell_execution(s3), "S4": _cell_execution(s4)},
            "snapshot_hashes": {
                c: {"first_actor": sorted({_short(r["records"][r["first_actor"]]["world_state_hash_before_snapshot"]) for r in rows}),
                    "second_actor": sorted({_short(r["records"][r["second_actor"]]["world_state_hash_before_snapshot"]) for r in rows})}
                for c, rows in ((S0, s0), (S3, s3), (S4, s4))
            },
        }
        if s0_c:
            entry["baseline_stability_phase_c_vs_phase_d_s0"] = {
                a: compare_distributions(_labels(s0_c, a), _labels(s0, a)) for a in ACTORS
            }
        out[family] = entry
    return out


def same_snapshot_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for cond in SCHEDULING_CONDITIONS:
        cell = [r for r in rows if r["condition"] == cond]
        if cell:
            out[cond] = {
                "n": len(cell),
                "same_snapshot_rate": rate(r["same_snapshot"] for r in cell),
                "second_actor_saw_first_action_rate": rate(r["second_actor_saw_first_action"] for r in cell),
                "world_changed_before_second_snapshot_rate": rate(r["world_changed_before_second_snapshot"] for r in cell),
            }
    return out


# -- runtime replication --------------------------------------------------------


def _effect_present(label_list: list[str]) -> bool:
    return any(label != NO_SCHEDULING_EFFECT for label in label_list)


def runtime_replication(c_rows: list[dict[str, Any]], d_rows: list[dict[str, Any]], e_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not e_rows:
        return {"status": "not_run"}
    family = e_rows[0]["scenario_family"]
    native_s0_c, native_s1 = _select(c_rows, scenario_family=family, condition=S0), _select(c_rows, scenario_family=family, condition=S1)
    native_s0_d, native_s3, native_s4 = (_select(d_rows, scenario_family=family, condition=c) for c in (S0, S3, S4))
    st_s0, st_s1, st_s3, st_s4 = (_select(e_rows, condition=c) for c in (S0, S1, S3, S4))

    comparisons = {}
    n_call, s_call = classify_call_order(native_s0_c, native_s1), classify_call_order(st_s0, st_s1)
    comparisons["call_order_S0_vs_S1"] = {"native": n_call["classification"], "strands": s_call["classification"],
                                          "native_detail": {a: {k: n_call["per_actor"][a][k] for k in ("distribution_x", "distribution_y", "tvd", "p_value")} for a in ACTORS},
                                          "strands_detail": {a: {k: s_call["per_actor"][a][k] for k in ("distribution_x", "distribution_y", "tvd", "p_value")} for a in ACTORS}}
    for name, n_seq, s_seq in (("snapshot_timing_S0_vs_S3", native_s3, st_s3), ("snapshot_timing_S0_vs_S4", native_s4, st_s4)):
        n_t, s_t = classify_snapshot_timing(native_s0_d, n_seq), classify_snapshot_timing(st_s0, s_seq)
        comparisons[name] = {
            "native": n_t["classification"], "strands": s_t["classification"],
            "native_second_actor": n_t["second_actor_decision_vs_s0"]["distribution_y"],
            "strands_second_actor": s_t["second_actor_decision_vs_s0"]["distribution_y"],
            "native_saw_rate": n_t["second_actor_saw_first_action_rate"], "strands_saw_rate": s_t["second_actor_saw_first_action_rate"],
        }
    for comp in comparisons.values():
        native_list = [comp["native"]] if isinstance(comp["native"], str) else comp["native"]
        strands_list = [comp["strands"]] if isinstance(comp["strands"], str) else comp["strands"]
        n_eff, s_eff = _effect_present(native_list), _effect_present(strands_list)
        if not n_eff and not s_eff:
            comp["classification"] = "SCHEDULING_EFFECT_RUNTIME_INDEPENDENT"  # the null replicates
        elif n_eff and s_eff:
            same_modal = modal(list(Counter(comp.get("native_second_actor", {})).elements())) == modal(list(Counter(comp.get("strands_second_actor", {})).elements()))
            comp["classification"] = "SCHEDULING_EFFECT_RUNTIME_INDEPENDENT" if sorted(native_list) == sorted(strands_list) and same_modal else "PARTIAL_REPLICATION"
        elif n_eff:
            comp["classification"] = "NO_REPLICATION"
        else:
            comp["classification"] = "SCHEDULING_EFFECT_RUNTIME_DEPENDENT"

    per_condition = {}
    for cond, n_rows, s_rows in ((S0, native_s0_c + native_s0_d, st_s0), (S1, native_s1, st_s1), (S3, native_s3, st_s3), (S4, native_s4, st_s4)):
        per_condition[cond] = {
            a: {"native": distribution(_labels(n_rows, a)), "strands": distribution(_labels(s_rows, a)),
                "native_modal": modal(_labels(n_rows, a)), "strands_modal": modal(_labels(s_rows, a))}
            for a in ACTORS
        }
    strands_behavior = {c: _cell_behavior(rows) for c, rows in ((S0, st_s0), (S1, st_s1), (S3, st_s3), (S4, st_s4))}
    return {"scenario": family, "comparisons": comparisons, "per_condition_decisions": per_condition, "strands_behavior": strands_behavior,
            "strands_same_snapshot": same_snapshot_rates(e_rows)}


# -- cost -----------------------------------------------------------------------


def cost_metrics(live_rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in live_rows:
        for rec in r["records"].values():
            if rec.get("decision_reused_from"):
                continue
            cells.setdefault(f"{r['runtime']}|{r['phase']}|{r['scenario_family']}|{SHORT[r['condition']]}", []).append(rec)
    for c in controls:
        cells.setdefault(f"native|C_PROVIDER_CONTROL|{c['scenario_family']}|{c['actor_id']}", []).append(c)

    def summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
        lat = [r["latency_ms"] for r in recs if r.get("latency_ms") is not None]
        inp = [r["input_tokens"] for r in recs if r.get("input_tokens") is not None]
        outp = [r["output_tokens"] for r in recs if r.get("output_tokens") is not None]
        return {
            "n_calls": len(recs),
            "latency_ms_mean": round(statistics.mean(lat), 1) if lat else None,
            "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
            "input_tokens_mean": round(statistics.mean(inp), 1) if inp else None,
            "output_tokens_mean": round(statistics.mean(outp), 1) if outp else None,
            "input_tokens_total": sum(inp),
            "output_tokens_total": sum(outp),
        }

    per_cell = {k: summarize(v) for k, v in sorted(cells.items())}
    by_runtime: dict[str, list[dict[str, Any]]] = {}
    for k, v in cells.items():
        by_runtime.setdefault(k.split("|")[0], []).extend(v)
    return {"per_cell": per_cell, "per_runtime": {k: summarize(v) for k, v in sorted(by_runtime.items())},
            "total_live_calls": sum(len(v) for v in cells.values())}


# -- overall ----------------------------------------------------------------------


def overall_classification(det: dict[str, Any], call: dict[str, Any], live_apply: dict[str, Any], timing: dict[str, Any],
                           fairness: dict[str, Any], replication: dict[str, Any]) -> list[str]:
    labels = []
    frozen_ok = all(fairness.get(c, {}).get("same_snapshot_rate", 1.0) == 1.0 for c in (S0, S1, S2))
    identical = all(v["requests_identical_across_conditions"] for v in call.values()) if call else False
    if frozen_ok and identical:
        labels.append("SNAPSHOT_FAIRNESS_CONFIRMED")
    call_effect = any(v["classification_after_provider_control"] == MODEL_CALL_ORDER_EFFECT for v in call.values())
    labels.append("MODEL_CALL_ORDER_EFFECT_SUPPORTED" if call_effect else "NO_MODEL_CALL_ORDER_EFFECT")
    if any(APPLICATION_ORDER_EFFECT in v["observed_labels"] for v in det.values()) or any(v["application_order_effect_rate"] > 0 for v in live_apply.values()):
        labels.append("APPLICATION_ORDER_EFFECT_SUPPORTED")
    if any(SEQUENTIAL_INFORMATION_EFFECT in v[k]["classification"] for v in timing.values() for k in ("S3_vs_S0", "S4_vs_S0")):
        labels.append("SEQUENTIAL_INFORMATION_EFFECT_SUPPORTED")
    if any(APPLY_TIME_INVALIDATION_EFFECT in v["observed_labels"] for v in det.values()) or any(
        v["apply_time_invalidation_effect_rate"] > 0 for v in live_apply.values()
    ) or any(APPLY_TIME_INVALIDATION_EFFECT in v[k]["classification"] for v in timing.values() for k in ("S3_vs_S0", "S4_vs_S0")):
        labels.append("APPLY_TIME_INVALIDATION_EFFECT_SUPPORTED")
    comps = replication.get("comparisons", {})
    if comps:
        # Aggregation: an effect that is absent on one runtime (NO_REPLICATION)
        # or present only on Strands (RUNTIME_DEPENDENT) makes the scheduling
        # result runtime-dependent. PARTIAL_REPLICATION - the same effect on
        # both runtimes, with a different re-targeted follow-up action - does
        # not; that residual is reported per comparison.
        classes = {c["classification"] for c in comps.values()}
        dependent = classes & {"SCHEDULING_EFFECT_RUNTIME_DEPENDENT", "NO_REPLICATION"}
        labels.append("SCHEDULING_EFFECT_RUNTIME_DEPENDENT" if dependent else "SCHEDULING_EFFECT_RUNTIME_INDEPENDENT")
    effect_kinds = [x for x in labels if x.endswith("_SUPPORTED")]
    if len(effect_kinds) > 1:
        labels.append(MIXED)
    return labels


# -- required tables (markdown) -------------------------------------------------------


def _dist_str(d: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items())


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def required_tables(det, call, live_apply, timing, fairness, replication) -> str:
    lines: list[str] = []
    lines += ["### Scheduling table (spec section 55)", "",
              "| Condition | Snapshot timing | Decision call order | Application order | Same snapshot? | Second actor sees first action? |",
              "|---|---|---|---|---|---|"]
    for cond in SCHEDULING_CONDITIONS:
        plan = SCHEDULE_PLANS[cond]
        f = fairness.get(cond)
        same = f"{f['same_snapshot_rate']:.0%} (n={f['n']})" if f else "-"
        saw = f"{f['second_actor_saw_first_action_rate']:.0%}" if f else "-"
        lines.append(f"| {cond} | {plan.snapshot_timing} | {' -> '.join(plan.call_order)} | {' -> '.join(plan.apply_order)} | {same} | {saw} |")

    lines += ["", "### Deterministic conflict table (spec section 56)", "",
              "| Scenario | Action_A | Action_B | Apply order | A valid at apply? | B valid at apply? | Final world hash | Event-log hash | Classification |",
              "|---|---|---|---|---|---|---|---|---|"]
    for family, m in det.items():
        for key in ("a_first", "b_first"):
            o = m["orders"][key]
            act_a = decision_label({"decision": o["action_a"]})
            act_b = decision_label({"decision": o["action_b"]})
            lines.append(f"| {family} | {act_a} | {act_b} | {' -> '.join(o['apply_order'])} | {o['a_valid_at_apply']} ({o['a_apply_outcome']}) | "
                         f"{o['b_valid_at_apply']} ({o['b_apply_outcome']}) | `{_short(o['world_state_hash_after'])}` | `{_short(o['event_log_hash'])}` | "
                         f"{', '.join(m['observed_labels'])} |")

    lines += ["", "### Call-order table (spec section 57)", "",
              "| Scenario | Actor | AgentRequest hash (S0 = S1) | S0 call position | S0 decisions | S1 call position | S1 decisions | TVD / p | Classification |",
              "|---|---|---|---|---|---|---|---|---|"]
    for family, m in call.items():
        for actor_id in ACTORS:
            pa = m["per_actor"][actor_id]
            ident = pa["request_identity"]
            h = _short(ident["agent_request_semantic_hash"][0]) + (" (identical)" if ident["identical"] else " (DIFFERS)")
            lines.append(f"| {family} | {actor_id} | `{h}` | {m['call_positions']['S0'][actor_id]} | {_dist_str(pa['distribution_x'])} | "
                         f"{m['call_positions']['S1'][actor_id]} | {_dist_str(pa['distribution_y'])} | {pa['tvd']} / {_p(pa['p_value'])} | "
                         f"{m['classification_after_provider_control']} |")

    lines += ["", "### Snapshot-timing table (spec section 58)", "",
              "| Scenario | Condition | First actor | Second actor | Snapshot hashes (first / second) | Second actor saw first action | Second-agent decisions | World outcome (contested effects) | Classification |",
              "|---|---|---|---|---|---|---|---|---|"]
    for family, m in timing.items():
        for cond, key in ((S0, None), (S3, "S3_vs_S0"), (S4, "S4_vs_S0")):
            plan = SCHEDULE_PLANS[cond]
            hashes = m["snapshot_hashes"][cond]
            exec_ = m["execution"][SHORT[cond]]
            saw = f"{m['second_actor_saw_first_action_rate'][SHORT[cond]]:.0%}"
            if key is None:
                second = plan.apply_order[1]
                decisions = m["behavior"]["S0"][second]["decision_distribution"]
                cls = "baseline"
            else:
                t = m[key]
                second = t["second_actor"]
                decisions = t["second_actor_decision_vs_s0"]["distribution_y"]
                cls = ", ".join(t["classification"]) + f" (TVD {t['second_actor_decision_vs_s0']['tvd']}, p {_p(t['second_actor_decision_vs_s0']['p_value'])})"
            lines.append(f"| {family} | {SHORT[cond]} | {plan.apply_order[0]} | {plan.apply_order[1]} | "
                         f"`{','.join(hashes['first_actor'])}` / `{','.join(hashes['second_actor'])}` | {saw} | {_dist_str(decisions)} | "
                         f"{_dist_str(exec_['contested_effects'])} | {cls} |")

    lines += ["", "### Apply-order table (spec section 59; live S0 decisions replayed as S2)", "",
              "| Scenario | Decision A | Decision B | n | Apply order | Apply-time validity (A / B) | Final world state (contested effects, world hash) | Classification |",
              "|---|---|---|---|---|---|---|---|"]
    for family, m in live_apply.items():
        for pair, info in m["by_decision_pair"].items():
            a, b = pair.split(" | ")
            for key, order in (("a_first", f"{ACTOR_A} -> {ACTOR_B}"), ("b_first", f"{ACTOR_B} -> {ACTOR_A}")):
                o = info[key]
                lines.append(f"| {family} | {a} | {b} | {info['n']} | {order} | {o['a_valid_at_apply']} / {o['b_valid_at_apply']} | "
                             f"{o['contested']}, `{o['world']}` | {_dist_str(info['labels'])} |")

    lines += ["", "### Actor counterbalance table (spec section 60)", "",
              "| Scenario | Order source | First actor identity | Second actor identity | Same starting state? | Outcome |",
              "|---|---|---|---|---|---|"]
    for family, m in det.items():
        same_start = "yes" if m["same_starting_state"] else "NO"
        for key, first, second in (("a_first", ACTOR_A, ACTOR_B), ("b_first", ACTOR_B, ACTOR_A)):
            o = m["orders"][key]
            outcome = ", ".join(f"{t}:{a}" for t, a in o["contested_effects"]) or "no contested effect"
            lines.append(f"| {family} | Phase B scripted ({'S0' if key == 'a_first' else 'S2'}) | {first} | {second} | {same_start} | "
                         f"{outcome}; A {o['a_apply_outcome']}, B {o['b_apply_outcome']} |")
    for family, m in timing.items():
        ao = m["S3_vs_S4"]
        same_start = "yes" if ao["same_starting_state"] else "NO"
        for cond, first, second, key in ((S3, ACTOR_A, ACTOR_B, "contested_effects_s3"), (S4, ACTOR_B, ACTOR_A, "contested_effects_s4")):
            decisions = m["behavior"][SHORT[cond]][second]["decision_distribution"]
            lines.append(f"| {family} | Phase D live ({SHORT[cond]}) | {first} | {second} | {same_start} | "
                         f"contested: {_dist_str(ao[key])}; second actor: {_dist_str(decisions)} |")
    comps = replication.get("comparisons", {})
    if comps:
        lines += ["", "### Runtime replication (spec section 43)", "",
                  "| Comparison | Native | Strands | Classification |", "|---|---|---|---|"]
        for name, comp in comps.items():
            native = comp["native"] if isinstance(comp["native"], str) else ", ".join(comp["native"])
            strands = comp["strands"] if isinstance(comp["strands"], str) else ", ".join(comp["strands"])
            lines.append(f"| {name} | {native} | {strands} | {comp['classification']} |")
    return "\n".join(lines)


# -- entry point ------------------------------------------------------------------------


def run() -> dict[str, Any]:
    scripted = read_jsonl(out_path("scripted_pair_trials.jsonl"))
    c_rows = read_jsonl(out_path("native_call_order_trials.jsonl"))
    replays = read_jsonl(out_path("native_apply_order_replays.jsonl"))
    controls = read_jsonl(out_path("provider_controls.jsonl"))
    d_rows = read_jsonl(out_path("native_snapshot_timing_trials.jsonl"))
    e_rows = read_jsonl(out_path("strands_trials.jsonl"))

    det = deterministic_application_order_metrics(scripted)
    call = call_order_metrics(c_rows, controls) if c_rows else {}
    live_apply = live_application_order_metrics(c_rows, replays) if replays else {}
    timing = snapshot_timing_metrics(d_rows, c_rows) if d_rows else {}
    fairness = same_snapshot_rates(c_rows + replays + d_rows)
    replication = runtime_replication(c_rows, d_rows, e_rows)
    cost = cost_metrics(c_rows + d_rows + e_rows, controls)
    overall = overall_classification(det, call, live_apply, timing, fairness, replication)

    write_json(out_path("application_order_metrics.json"), {"deterministic_phase_b": det, "live_phase_c_s2_replays": live_apply})
    write_json(out_path("call_order_metrics.json"), call)
    write_json(out_path("snapshot_timing_metrics.json"), {"per_scenario": timing, "same_snapshot_rate_by_condition_native": fairness})
    write_json(out_path("runtime_replication.json"), replication)
    write_json(out_path("cost_metrics.json"), cost)

    clocks_frozen = all(
        rec["game_clock_before_call"] == rec["game_clock_after_call"] and rec["world_unchanged_during_call"]
        for r in c_rows + d_rows + e_rows for rec in r["records"].values()
    )
    summary = {
        "milestone": "M6G",
        "overall_classification": overall,
        "key_finding": KEY_FINDING,
        "notes": NOTES,
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "trial_counts": {"phase_b_scripted": len(scripted), "phase_c_native": len(c_rows), "phase_c_s2_replays": len(replays),
                         "provider_controls": len(controls), "phase_d_native": len(d_rows), "strands": len(e_rows)},
        "live_calls": cost["total_live_calls"],
        "game_time_frozen_during_every_live_call": clocks_frozen,
        "same_snapshot_rate_by_condition_native": fairness,
        "deterministic_application_order": {f: m["observed_labels"] for f, m in det.items()},
        "predeclared_expectations_held": all(m["expectation_held"] for m in det.values()),
        "call_order": {f: {"classification": m["classification"], "after_provider_control": m["classification_after_provider_control"],
                           "requests_identical": m["requests_identical_across_conditions"]} for f, m in call.items()},
        "live_application_order": {f: {"application_order_effect_rate": m["application_order_effect_rate"],
                                       "apply_time_invalidation_effect_rate": m["apply_time_invalidation_effect_rate"], "n_pairs": m["n_pairs"]}
                                   for f, m in live_apply.items()},
        "snapshot_timing": {f: {k: m[k]["classification"] for k in ("S3_vs_S0", "S4_vs_S0", "S3_vs_S4")} for f, m in timing.items()},
        "runtime_replication": {k: v["classification"] for k, v in replication.get("comparisons", {}).items()},
        "cost_per_runtime": cost["per_runtime"],
    }
    write_json(out_path("summary.json"), summary)
    summary["_tables_markdown"] = required_tables(det, call, live_apply, timing, fairness, replication)
    return summary


if __name__ == "__main__":
    out = run()
    tables = out.pop("_tables_markdown")
    import json

    print(json.dumps(out, indent=2, default=str))
    print()
    print(tables)
