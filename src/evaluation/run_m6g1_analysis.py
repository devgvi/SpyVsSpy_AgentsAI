"""M6G.1 analysis (spec sections 26-36, 40, 56-62, 70): zero model calls.

Reads every M6G.1 output and writes:

    staleness_metrics.json       per source / condition / scenario (35-36, 57)
    commitment_metrics.json      answer-order telemetry, freeze audit, door,
                                 barrier replays (27, 30-34, 58-60)
    runtime_replication.json     Native vs Strands (37, 40, 61)
    summary.json                 overall classification (62), next-step gate (70)
    answer_order_telemetry.jsonl regenerated: M6G logical rows + observed
                                 real-loop rows
    action_temporal_semantics.md generated from the Phase A probes (6, 28-29, 56)

and prints the required tables (spec sections 56-61). Every comparison
across commit orders holds the decision pair fixed (captured-decision
replay), so provider/model variance cannot show up as an answer-order
effect (STOP condition 40).

    python -m src.evaluation.run_m6g1_analysis
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
import statistics
from collections import Counter, defaultdict
from typing import Any

from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS
from src.evaluation.m6g_live import read_jsonl, write_json
from src.evaluation.run_m6g1_phase_b import OUT_DIR, PROBE_MODEL_ID, out_path
from src.evaluation.scheduling_analysis import decision_label, distribution
from src.evaluation.temporal_semantics import (
    ACCEPTED_UNCHANGED,
    APPLICATION_ORDER_ONLY,
    COMPLETION_ORDER_EFFECT,
    CONTINUOUS_OR_MULTI_TICK,
    DIFFERENT_COST,
    DIFFERENT_WORLD_EFFECT,
    DURATION_AS_COST_ONLY,
    DURATION_COST_ASYMMETRY,
    EFFECT_AT_COMPLETION,
    EFFECT_AT_START_WITH_BUSY_PERIOD,
    FALLBACK,
    FRESH,
    IN_FLIGHT_CONFLICT,
    INVALID_AT_SNAPSHOT,
    L0_NON_CONFLICTING,
    L1_SHARED_DOOR,
    L3_TRAP_SLOT_CONFLICT,
    L4_DISARM_CONFLICT,
    L5_ACTION_INVALIDATES_OTHER,
    L_FAMILIES,
    MIXED,
    NO_OP,
    NO_TEMPORAL_CONFLICT,
    REFUSED,
    STALE_AND_INVALID,
    STALE_AND_INVALID_EFFECT,
    STALE_BUT_VALID,
    STALE_BUT_VALID_EFFECT,
    WORLD_STALE_IRRELEVANT,
)

STALENESS_CLASSES = (FRESH, WORLD_STALE_IRRELEVANT, STALE_BUT_VALID, STALE_AND_INVALID, INVALID_AT_SNAPSHOT)
CONSEQUENCES = (ACCEPTED_UNCHANGED, REFUSED, NO_OP, DIFFERENT_COST, DIFFERENT_WORLD_EFFECT, FALLBACK)

# spec section 40
RT_INDEPENDENT = "TEMPORAL_EFFECT_RUNTIME_INDEPENDENT"
RT_DEPENDENT = "TEMPORAL_EFFECT_RUNTIME_DEPENDENT"
RT_PARTIAL = "PARTIAL_REPLICATION"
RT_NONE = "NO_REPLICATION"
INCIDENCE_GAP = 0.3  # an effect that shows in both runtimes at rates this far apart replicates only partially

LIVE_FILES = {"native": "real_loop_trials.jsonl", "strands": "strands_trials.jsonl"}
SCRIPTED_LOOP_FILE = "real_loop_scripted_trials.jsonl"
TELEMETRY_FILE = "answer_order_telemetry.jsonl"
LOGICAL = "logical_from_m6g_latency"
OBSERVED = "observed_real_loop_wall_clock"


# -- small helpers -----------------------------------------------------------------------


def _rk(runtime: str) -> str:
    return "strands" if runtime == "strands" else "native"


def _short(h: str | None, n: int = 10) -> str:
    return (h or "")[:n]


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _pct(k: int, n: int) -> str:
    return f"{k}/{n}" + (f" ({k / n:.0%})" if n else "")


def _stats(xs: list[float]) -> dict[str, Any] | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0]] * 3
    return {"n": len(xs), "min": round(min(xs), 1), "p25": round(q[0], 1), "median": round(statistics.median(xs), 1), "p75": round(q[2], 1),
            "max": round(max(xs), 1), "mean": round(statistics.fmean(xs), 1)}


def _other(actor_id: str) -> str:
    return ACTOR_B if actor_id == ACTOR_A else ACTOR_A


def _outcome(row: dict[str, Any]) -> tuple[str, str]:
    return row["world_state_hash_after"], row["event_multiset_hash"]


def _contested(row: dict[str, Any]) -> str:
    effects = [tuple(e) for e in row.get("contested_effects") or []]
    return ", ".join(f"{t}:{a}" for t, a in effects) if effects else "none"


def _shares_target(decisions: dict[str, dict[str, Any]]) -> bool:
    """Both decisions act on the same object/door - the only way two
    decisions can compete for shared state (MOVE holds no resource: M6G C3)."""
    a, b = decisions[ACTOR_A], decisions[ACTOR_B]
    acting = all(d["action"] not in ("wait", "move") for d in (a, b))
    return acting and a["target_id"] is not None and a["target_id"] == b["target_id"]


def _pair_label(decisions: dict[str, dict[str, Any]]) -> str:
    """"black decision / white decision" (no pipes: it goes into markdown tables)."""
    return " / ".join(decision_label({"decision": decisions[a]}) for a in ACTORS)


def _commit_rule_holds(row: dict[str, Any]) -> bool:
    """The production commit rule: decisions ready on different ticks apply
    in response-ready order; decisions ready on one tick apply Black then
    White (Game._step's fixed apply order)."""
    if row["same_tick_commit"]:
        return row["apply_order"] == [ACTOR_A, ACTOR_B]
    return row["ready_order"] == row["apply_order"]


# -- loading ---------------------------------------------------------------------------


def load() -> dict[str, Any]:
    scripted = read_jsonl(out_path("scripted_temporal_trials.jsonl"))
    telemetry = read_jsonl(out_path(TELEMETRY_FILE))
    with open(out_path("duration_semantics.json"), encoding="utf-8") as f:
        probes = json.load(f)["probes"]
    with open(out_path("m6g_reuse_manifest.json"), encoding="utf-8") as f:
        reuse_manifest = json.load(f)
    live = {rk: [r for r in read_jsonl(out_path(name)) if r.get("decision_source", "live_model") == "live_model"] for rk, name in LIVE_FILES.items()}
    return {
        "probes": probes,
        "scripted_pairs": [r for r in scripted if "records" in r],
        "busy": [r for r in scripted if r["variant"].startswith("busy_")],
        "reuse_replays": read_jsonl(out_path("barrier_replay_trials.jsonl")),
        "reuse_manifest": reuse_manifest,
        "logical_telemetry": [t for t in telemetry if t.get("timing_source") == LOGICAL],
        "live": live,
        "live_replays": read_jsonl(out_path("real_loop_barrier_replays.jsonl")),
        "scripted_loop": read_jsonl(out_path(SCRIPTED_LOOP_FILE)),
        "provider_events": read_jsonl(out_path("provider_events.jsonl")),
    }


# -- action temporal semantics (spec sections 6-9, 28-29, 56) -----------------------------

# Apply-time behavior per action, each backed by a scripted scenario: the
# decision is never re-validated against the agent's legal list at commit
# (production validates once, against the snapshot); the Engine's own rules
# decide at apply.
REVALIDATION = {
    "INTERACT door: locked": "Engine reads the door's state at apply: locked -> unlock+open (1.0 s), open -> re-open (0.3 s) [L1]",
    "INTERACT door: already open": "Engine reads the door's state at apply [L1]",
    "INTERACT search (desk_01)": "Engine reads the trap slot at apply: an armed enemy trap springs instead [L5]",
    "INTERACT search springs": "Engine reads the trap slot at apply [L5]",
    "INTERACT exit_door": "Engine checks mission items at apply",
    "INSPECT": "Engine evaluates the target at apply",
    "PLACE_TRAP": "Engine refuses if the slot was filled after the snapshot [L3]",
    "DISARM": "Engine refuses if the trap is gone at apply [L4]",
    "MOVE": "path planned once from the snapshot nav grid, never re-planned; Engine collides per tick",
    "WAIT": "n/a",
    "SEND_MESSAGE": "SOCIAL_ON only (not used in any M6G.1 run)",
}


def _note(label: str, table: dict[str, str]) -> str:
    return next((v for k, v in table.items() if label.startswith(k)), "")


def action_semantics(probes: list[dict[str, Any]], busy: list[dict[str, Any]]) -> dict[str, Any]:
    busy_identical = {}
    for fam in sorted({b["scenario_family"] for b in busy}):
        for first in ACTORS:
            rows = {b["second_waits_for_completion"]: b for b in busy if b["scenario_family"] == fam and b["first_actor"] == first}
            if len(rows) == 2:
                key = lambda b: (b["second"]["outcome"], b["second"]["event_types"], b["second"]["time_cost"], b["second"]["target_state_after"])  # noqa: E731
                busy_identical[f"{fam}:{first}_first"] = {
                    "second_outcome_identical_during_vs_after_busy": key(rows[False]) == key(rows[True]),
                    "first_effect_unchanged_through_completion": rows[True]["first_effect_unchanged_through_completion"],
                    "events_during_first_busy_period": rows[True]["events_during_first_busy_period"],
                    "first_busy_ticks_until_complete": rows[True]["first_busy_ticks_until_complete"],
                    "effect_state_identical_during_vs_after_busy": rows[False]["effect_state_hash_after_pair"] == rows[True]["effect_state_hash_after_pair"],
                }
    no_in_flight = bool(busy_identical) and all(v["second_outcome_identical_during_vs_after_busy"] and v["first_effect_unchanged_through_completion"]
                                                 for v in busy_identical.values())
    rows = []
    for p in probes:
        action = p["decision"]["action"]
        is_move = action == "move"
        if is_move:
            cost = f"{p['multi_tick_ticks']} live ticks, no cost"
        elif p["stun_seconds"]:
            cost = f"stun {p['stun_seconds']:g} s"
        else:
            cost = f"{p['time_cost_seconds']:g} s" if p["time_cost_seconds"] else "0"
        busy_kind = p["busy_kind"]
        if is_move:
            timing = "each live tick (no displacement on frozen ticks)"
        elif p["world_effect_at_completion"]:
            timing = "at completion"
        elif p["world_effect_at_apply"]:
            timing = "at apply (start); nothing during or at completion"
        elif p["busy_after_apply"]:
            timing = "none - cost only (event only)"
        else:
            timing = "none"
        if busy_kind == "stunned":
            timing += "; ACTOR_RECOVERED event at stun end"
        if is_move:
            conflict = "no resource claim (M6G C3: no order effect); door changes mid-move do not re-plan the path"
        elif p["busy_after_apply"]:
            conflict = "no in-flight state: effect complete at apply; opponent outcome identical during/after the busy period" if no_in_flight else "see busy variants"
        else:
            conflict = "n/a (no duration)"
        rows.append({
            "action": p["label"],
            "action_type": action,
            "duration_cost": cost,
            "time_cost_seconds": p["time_cost_seconds"],
            "stun_seconds": p["stun_seconds"],
            "busy_state": busy_kind or "no",
            "actor_busy": p["busy_after_apply"],
            "world_effect_at_start": p["world_effect_at_apply"] and not is_move,
            "world_effect_at_completion": p["world_effect_at_completion"],
            "continuous_effect": is_move,
            "world_effect_timing": timing,
            "apply_time_revalidation": _note(p["label"], REVALIDATION),
            "can_conflict_while_in_flight": conflict,
            "can_target_change_during_action": ("doors can change mid-move; the committed path is not re-planned" if is_move else
                                                "yes - the opponent can act on it during the busy period; this action's effect is already complete"
                                                if p["busy_after_apply"] and p["world_effect_at_apply"] else
                                                "yes - the opponent can act on it during the busy period; this action changed nothing"
                                                if p["busy_after_apply"] else "n/a"),
            "busy_ticks_until_complete": p["busy_ticks_until_complete"],
            "busy_unchanged_by_frozen_updates": p["busy_unchanged_by_frozen_updates"],
            "observation_reports_busy": p["observation_reports_busy"],
            "second_action_dropped_while_busy": p["second_action_dropped_while_busy"],
            "events_at_apply": p["event_types_at_apply"],
            "events_during_busy_and_at_completion": p["events_during_busy_and_at_completion"],
            "classification": p["classification"],
        })
    classes = Counter(r["classification"] for r in rows)
    return {
        "actions": rows,
        "classification_counts": dict(classes),
        "busy_variants": busy_identical,
        "no_in_flight_state": no_in_flight,
        "any_effect_at_completion": any(p["world_effect_at_completion"] for p in probes) or classes.get(EFFECT_AT_COMPLETION, 0) > 0,
        "frozen_updates_never_drain_busy": all(p["busy_unchanged_by_frozen_updates"] for p in probes),
        "busy_semantics": {
            "busy_start": "the apply tick (action_cooldown / stunned_timer set by GameState.apply_action)",
            "busy_end": "after time_cost seconds of LIVE ticks; frozen (dt=0) ticks never drain it",
            "can_observe_while_busy": all(p["observation_reports_busy"] for p in probes if p["busy_after_apply"]),
            "can_decide_while_busy": "no - LLMController/StrandsAgentAdapter.get_action return WAIT without a model request while observation.busy or .stunned",
            "can_receive_another_action": "apply_action accepts it and drops it silently" if all(
                p["second_action_dropped_while_busy"] for p in probes if p["busy_after_apply"]) else "see probes",
            "can_be_affected_by_opponent": "the opponent can change the shared state the busy actor just changed; the busy actor's completed action is unaffected",
        },
    }


def action_semantics_markdown(sem: dict[str, Any]) -> str:
    lines = ["# M6G.1 — Action Temporal Semantics", "",
             "Generated by `src/evaluation/run_m6g1_analysis.py` from `duration_semantics.json` (Phase A probes on the real Engine, "
             "zero model calls) and the Phase B busy-period variants in `scripted_temporal_trials.jsonl`. Nothing here was inferred "
             "from constants alone: every value is a measurement of `GameState.apply_action` / `update`.", "",
             "Moments kept apart: SNAPSHOT (observation built) → READY (decision available) → COMMIT (handed to the execution path) → "
             "APPLY (Engine processes it) → COMPLETE (modeled duration elapsed). For every current action, APPLY is where the world "
             "changes; COMPLETE only ends the actor's lockout.", "",
             "## Required action-semantics table (spec section 56)", "",
             "| Action | Duration / cost | Busy state? | World-effect timing | Apply-time revalidation? | Can conflict while in flight? | Classification |",
             "|---|---|---|---|---|---|---|"]
    for r in sem["actions"]:
        lines.append(f"| {r['action']} | {r['duration_cost']} | {r['busy_state']} | {r['world_effect_timing']} | {r['apply_time_revalidation']} | "
                     f"{r['can_conflict_while_in_flight']} | {r['classification']} |")
    lines += ["", "## Per-action audit (spec section 6)", "",
              "| Action | Duration / cost | Actor busy? | World effect at start? | World effect at completion? | Continuous effect? | "
              "Can target change during action? | Busy ticks to complete | Frozen ticks drain busy? |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sem["actions"]:
        lines.append(f"| {r['action']} | {r['duration_cost']} | {r['actor_busy']} | {r['world_effect_at_start']} | {r['world_effect_at_completion']} | "
                     f"{r['continuous_effect']} | {r['can_target_change_during_action']} | {r['busy_ticks_until_complete']} | "
                     f"{'no' if r['busy_unchanged_by_frozen_updates'] else 'YES'} |")
    lines += ["", "## Effect timing (spec section 28)", "",
              "For every duration-bearing action the world mutates **before** the duration (at apply). Nothing mutates **during** it, "
              "and nothing mutates **after** it: `world_effect_at_completion` is false for every probe. The stun from a sprung trap "
              "logs one `ACTOR_RECOVERED` event when it ends, with no world change. MOVE is the one multi-tick action: it moves on "
              "every live tick along a path fixed at decision time, and it has no cost or lockout.", "",
              "## Busy-state semantics (spec section 29)", ""]
    for k, v in sem["busy_semantics"].items():
        lines.append(f"- **{k.replace('_', ' ')}:** {v}")
    lines += ["", "## Busy-period variants (spec section 24)", "",
              "The second actor applies either inside the first actor's busy period (same tick) or only after it has fully elapsed on "
              "live ticks. Identical outcomes mean the first action carries no in-flight state.", "",
              "| Scenario / first actor | Second outcome identical? | First effect unchanged through completion? | Events during busy period | Busy ticks |",
              "|---|---|---|---|---|"]
    for k, v in sem["busy_variants"].items():
        lines.append(f"| {k} | {v['second_outcome_identical_during_vs_after_busy']} | {v['first_effect_unchanged_through_completion']} | "
                     f"{', '.join(v['events_during_first_busy_period']) or 'none'} | {v['first_busy_ticks_until_complete']} |")
    counts = sem["classification_counts"]
    lines += ["", "## Classification", "",
              "Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ".", "",
              f"- `{EFFECT_AT_START_WITH_BUSY_PERIOD}`: the world changes at apply, then the actor is locked out for the duration. "
              "The lockout is a cost to the actor, not an action in flight.",
              f"- `{DURATION_AS_COST_ONLY}`: a lockout with no world change (re-opening an already-open door).",
              f"- `{CONTINUOUS_OR_MULTI_TICK}`: MOVE only.",
              "- `INSTANTANEOUS`: WAIT and a failed exit attempt.",
              f"- `{EFFECT_AT_COMPLETION}` and true in-flight state: **none**. "
              f"No-in-flight-state check over all busy variants: {sem['no_in_flight_state']}. "
              "M6G.1 invents no in-flight mechanism (spec sections 7, 44).", ""]
    return "\n".join(lines)


# -- staleness (spec sections 5, 35-36, 57) ---------------------------------------------


def _entry(source: str, condition: str, row: dict[str, Any], actor_id: str) -> dict[str, Any]:
    rec = row["records"][actor_id]
    cf = rec.get("counterfactual") or {}
    return {
        "source": source, "condition": condition, "scenario_family": row["scenario_family"],
        "pair_id": row.get("pair_id") or row.get("trial_id"), "actor_id": actor_id,
        "apply_position": row["apply_order"].index(actor_id) + 1,
        "decision": decision_label(rec), "action": rec["decision"]["action"],
        "staleness_class": rec["staleness_class"], "consequence": rec["consequence"],
        "relevant_state_changed": rec["relevant_state_changed"], "world_changed": rec["world_changed"],
        "legal_at_snapshot": rec["legal_at_snapshot"], "legal_at_apply": rec["legal_at_apply"],
        "listed_at_snapshot": rec["listed_at_snapshot"], "listed_at_apply": rec["listed_at_apply"],
        "apply_outcome": rec["apply_outcome"], "time_cost": rec["time_cost"], "counterfactual_time_cost": cf.get("time_cost"),
        "stale_distance": rec.get("stale_distance"),
        "snapshot_world_hash": rec["snapshot_world_hash"], "apply_world_hash": rec["apply_world_hash"],
        "snapshot_relevant_hash": rec["snapshot_relevant_hash"], "apply_relevant_hash": rec["apply_relevant_hash"],
    }


def staleness_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in data["scripted_pairs"]:
        out += [_entry("scripted_phase_b", r["variant"], r, a) for a in r["apply_order"]]
    for r in data["reuse_replays"]:
        out += [_entry(f"m6g_reuse_{r['runtime']}", r["replay"], r, a) for a in r["apply_order"]]
    for rk, rows in data["live"].items():
        for r in rows:
            if r["complete"]:
                out += [_entry(f"real_loop_live_{rk}", "T0_live", r, a) for a in r["apply_order"]]
    for r in data["scripted_loop"]:
        if r["complete"]:
            out += [_entry(f"real_loop_scripted_{_rk(r['runtime'])}", "T0_live_scripted_decisions", r, a) for a in r["apply_order"]]
    return out


def _stale_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(entries)
    classes = Counter(e["staleness_class"] for e in entries)
    relevant = [e for e in entries if e["relevant_state_changed"]]
    stale_invalid = [e for e in entries if e["staleness_class"] == STALE_AND_INVALID]
    return {
        "n_decisions": n,
        "staleness_classes": {c: classes.get(c, 0) for c in STALENESS_CLASSES},
        "relevant_state_stale": len(relevant),
        "relevant_state_stale_rate": _rate(len(relevant), n),
        "stale_but_valid": classes.get(STALE_BUT_VALID, 0),
        "stale_but_valid_rate": _rate(classes.get(STALE_BUT_VALID, 0), n),
        "stale_and_invalid": classes.get(STALE_AND_INVALID, 0),
        "stale_and_invalid_rate": _rate(classes.get(STALE_AND_INVALID, 0), n),
        "stale_and_invalid_still_listed_at_apply": sum(1 for e in stale_invalid if e["listed_at_apply"]),
        "consequences_all": {c: sum(1 for e in entries if e["consequence"] == c) for c in CONSEQUENCES},
        "consequences_relevant_stale": {c: sum(1 for e in relevant if e["consequence"] == c) for c in CONSEQUENCES},
        "stale_distance": dict(sorted(Counter(e["stale_distance"] for e in entries if e["stale_distance"] is not None).items())),
    }


def staleness_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in entries:
        grouped[e["source"]][e["condition"]][e["scenario_family"]].append(e)
    by_source: dict[str, Any] = {}
    for source, conds in grouped.items():
        by_source[source] = {}
        for cond, fams in conds.items():
            cell = {fam: {"all_decisions": _stale_summary(es), "second_applied": _stale_summary([e for e in es if e["apply_position"] == 2])}
                    for fam, es in sorted(fams.items())}
            flat = [e for es in fams.values() for e in es]
            cell["_all_scenarios"] = {"all_decisions": _stale_summary(flat), "second_applied": _stale_summary([e for e in flat if e["apply_position"] == 2])}
            by_source[source][cond] = cell
    not_fresh = [e for e in entries if e["apply_position"] == 1 and e["staleness_class"] != FRESH]
    return {
        "definition": "stale = the decision's relevant state (per action type, never the whole world) changed between its snapshot and its "
                      "apply; legal_at_snapshot = the Engine accepts the decision applied alone to its snapshot; legal_at_apply = the Engine "
                      "applied its intended effect when it actually ran",
        "first_applied_decisions_not_fresh": len(not_fresh),
        "first_applied_not_fresh_examples": [{k: e[k] for k in ("source", "condition", "pair_id", "actor_id", "staleness_class", "consequence")}
                                             for e in not_fresh[:5]],
        "by_source": by_source,
    }


def staleness_table_rows(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    scripted = [e for e in entries if e["source"] == "scripted_phase_b" and e["condition"].startswith("T1")]
    grouped: dict[tuple, list] = defaultdict(list)
    for e in entries:
        if e["source"].startswith(("real_loop_live", "m6g_reuse")) and e["condition"] in ("T0_live", "T0_logical_arrival"):
            key = (e["source"], e["scenario_family"], e["apply_position"], e["decision"], e["staleness_class"], e["consequence"],
                   e["relevant_state_changed"], e["legal_at_snapshot"], e["legal_at_apply"], e["apply_outcome"])
            grouped[key].append(e)
    live = []
    for key, es in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], -len(kv[1]))):
        live.append({"source": key[0], "scenario_family": key[1], "apply_position": key[2], "decision": key[3], "n": len(es),
                     "staleness_class": key[4], "consequence": key[5], "relevant_state_changed": key[6], "legal_at_snapshot": key[7],
                     "legal_at_apply": key[8], "apply_outcome": key[9],
                     "actors": dict(Counter(e["actor_id"] for e in es)),
                     "snapshot_hashes": sorted({_short(e["snapshot_world_hash"]) for e in es}),
                     "apply_hashes": sorted({_short(e["apply_world_hash"]) for e in es})})
    return {"scripted_t1": scripted, "live_and_reuse_t0_grouped": live}


# -- temporal conflict classification (spec section 26) ------------------------------------


def temporal_conflict(t1a: dict[str, Any], t1b: dict[str, Any], busy_ok: bool | None = True, completion_ok: bool = True) -> dict[str, Any]:
    labels: list[str] = []
    if busy_ok is False:
        labels.append(IN_FLIGHT_CONFLICT)
    if not completion_ok:
        labels.append(COMPLETION_ORDER_EFFECT)
    for row in (t1a, t1b):
        second = row["records"][row["apply_order"][1]]
        cls, cons = second["staleness_class"], second["consequence"]
        if cls == STALE_AND_INVALID:
            labels.append(STALE_AND_INVALID_EFFECT)
        elif cls == STALE_BUT_VALID and cons == DIFFERENT_COST:
            labels.append(DURATION_COST_ASYMMETRY)
        elif cls == STALE_BUT_VALID and cons == DIFFERENT_WORLD_EFFECT:
            labels.append(STALE_BUT_VALID_EFFECT)
    order_sensitive = _outcome(t1a) != _outcome(t1b)
    if not labels and order_sensitive:
        labels.append(APPLICATION_ORDER_ONLY)
    distinct = sorted(set(labels))
    return {"classification": NO_TEMPORAL_CONFLICT if not distinct else distinct[0] if len(distinct) == 1 else MIXED,
            "labels": distinct, "order_sensitive_outcome": order_sensitive}


def scripted_conflicts(data: dict[str, Any], sem: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for fam in L_FAMILIES:
        rows = {r["variant"]: r for r in data["scripted_pairs"] if r["scenario_family"] == fam}
        busy = [v for k, v in sem["busy_variants"].items() if k.startswith(fam + ":")]
        busy_ok = all(v["second_outcome_identical_during_vs_after_busy"] for v in busy) if busy else None
        completion_ok = all(v["first_effect_unchanged_through_completion"] for v in busy) if busy else True
        t1a, t1b = rows["T1A_barrier_A_then_B"], rows["T1B_barrier_B_then_A"]
        tc = temporal_conflict(t1a, t1b, busy_ok, completion_ok)
        tc.update({
            "t0_a_first_equals_t1a": _outcome(rows["T0_arrival_A_first"]) == _outcome(t1a),
            "t0_b_first_equals_t1b": _outcome(rows["T0_arrival_B_first"]) == _outcome(t1b),
            "second_under_t1a": {k: t1a["records"][t1a["apply_order"][1]][k] for k in ("actor_id", "staleness_class", "consequence", "listed_at_apply")},
            "second_under_t1b": {k: t1b["records"][t1b["apply_order"][1]][k] for k in ("actor_id", "staleness_class", "consequence", "listed_at_apply")},
            "contested_t1a": _contested(t1a), "contested_t1b": _contested(t1b),
            "busy_variants_identical": busy_ok,
        })
        out[fam] = tc
    return out


# -- barrier replays (spec sections 14-15, 32-33, 38-39, 59) -------------------------------


def _pairs_from_replays(replays: list[dict[str, Any]], key: str) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in replays:
        pairs[r[key]][r["replay"]] = r
    return pairs


def barrier_analysis(data: dict[str, Any]) -> dict[str, Any]:
    sets = []
    for pid, reps in _pairs_from_replays(data["reuse_replays"], "pair_id").items():
        sets.append((f"m6g_reuse_{reps['T1A']['runtime']}", pid, reps, "T0_logical_arrival"))
    for pid, reps in _pairs_from_replays(data["live_replays"], "source_trial_id").items():
        sets.append((f"real_loop_live_{_rk(reps['T1A']['runtime'])}", pid, reps, "T0_observed_arrival"))
    pairs = []
    for source, pid, reps, t0_label in sets:
        t0, t0_rev, t1a, t1b = reps[t0_label], reps["T0_reversed_arrival"], reps["T1A"], reps["T1B"]
        same = t1a if t0["apply_order"][0] == ACTOR_A else t1b
        other = t1b if same is t1a else t1a
        decisions = {a: t0["records"][a]["decision"] for a in ACTORS}
        tc = temporal_conflict(t1a, t1b)
        pairs.append({
            "source": source, "pair_id": pid, "scenario_family": t0["scenario_family"], "decision_pair": _pair_label(decisions),
            "shares_target": _shares_target(decisions),
            "t0_arrival_order": t0["apply_order"], "t0_outcome": _contested(t0), "t0_world": _short(t0["world_state_hash_after"]),
            "t1a_outcome": _contested(t1a), "t1a_world": _short(t1a["world_state_hash_after"]),
            "t1b_outcome": _contested(t1b), "t1b_world": _short(t1b["world_state_hash_after"]),
            "t0_equals_same_order_barrier": _outcome(t0) == _outcome(same),
            "t0_reversed_equals_other_barrier": _outcome(t0_rev) == _outcome(other),
            "order_sensitive": tc["order_sensitive_outcome"],
            "temporal_conflict": tc["classification"],
            "live_exact_reproduction": (reps.get("T0_observed_schedule") is not None),
        })
    summary: dict[str, Any] = {}
    for source in sorted({p["source"] for p in pairs}):
        mine = [p for p in pairs if p["source"] == source]
        summary[source] = {}
        for fam in sorted({p["scenario_family"] for p in mine}) + ["_all_scenarios"]:
            ps = [p for p in mine if fam == "_all_scenarios" or p["scenario_family"] == fam]
            summary[source][fam] = {
                "n_pairs": len(ps),
                "t0_equals_same_order_barrier": sum(p["t0_equals_same_order_barrier"] for p in ps),
                "t0_reversed_equals_other_barrier": sum(p["t0_reversed_equals_other_barrier"] for p in ps),
                "order_sensitive_pairs": sum(p["order_sensitive"] for p in ps),
                "order_sensitive_pairs_sharing_a_target": sum(p["order_sensitive"] and p["shares_target"] for p in ps),
                "order_sensitive_pairs_not_sharing_a_target": sum(p["order_sensitive"] and not p["shares_target"] for p in ps),
                "temporal_conflict": distribution(p["temporal_conflict"] for p in ps),
                "decision_pairs": distribution(p["decision_pair"] for p in ps),
            }
    all_explained = bool(pairs) and all(p["t0_equals_same_order_barrier"] and p["t0_reversed_equals_other_barrier"] for p in pairs)
    return {"pairs": pairs, "summary": summary, "n_pairs": len(pairs), "arrival_adds_nothing_beyond_application_order": all_explained}


def barrier_table_rows(barrier: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list] = defaultdict(list)
    for p in barrier["pairs"]:
        grouped[(p["source"], p["scenario_family"], p["decision_pair"], " -> ".join(p["t0_arrival_order"]), p["t0_outcome"],
                 p["t1a_outcome"], p["t1b_outcome"])].append(p)
    rows = []
    for key, ps in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], -len(kv[1]))):
        rows.append({"source": key[0], "scenario_family": key[1], "decision_pair": key[2], "t0_arrival_order": key[3], "n": len(ps),
                     "t0_outcome": key[4], "t1a_outcome": key[5], "t1b_outcome": key[6],
                     "explained": sum(p["t0_equals_same_order_barrier"] and p["t0_reversed_equals_other_barrier"] for p in ps),
                     "pair_ids": [p["pair_id"] for p in ps][:3]})
    return rows


# -- door (spec sections 10, 27, 60) ---------------------------------------------------


def _door_row(label: str, row: dict[str, Any]) -> dict[str, Any] | None:
    first, second = row["apply_order"]
    rf, rs = row["records"][first], row["records"][second]
    df, ds = rf["decision"], rs["decision"]
    is_door = lambda r: isinstance(r.get("target_state_before_apply"), dict) and "door_state" in r["target_state_before_apply"]  # noqa: E731
    if not (df["action"] == ds["action"] == "interact" and df["target_id"] == ds["target_id"] and is_door(rf) and is_door(rs)):
        return None
    final = rs.get("target_state_final") or rs.get("target_state_after")
    return {
        "source": label, "pair_id": row.get("pair_id") or row.get("trial_id"), "order": f"{first} -> {second}",
        "door_state_before_first": rf["target_state_before_apply"]["door_state"],
        "door_state_before_second": rs["target_state_before_apply"]["door_state"],
        "first_cost": rf["time_cost"], "second_cost": rs["time_cost"],
        "first_events": rf["apply_event_types"], "second_events": rs["apply_event_types"],
        "first_actor_busy_at_second_apply": rs["other_actor_busy_at_apply"],
        "final_door_state": (final or {}).get("door_state"),
        "second_staleness": rs["staleness_class"], "second_consequence": rs["consequence"],
    }


def door_analysis(data: dict[str, Any], sem: dict[str, Any]) -> dict[str, Any]:
    scripted = [d for d in (_door_row(f"scripted_phase_b:{r['variant']}", r) for r in data["scripted_pairs"]
                            if r["scenario_family"] == L1_SHARED_DOOR) if d]
    live = []
    for rk, rows in data["live"].items():
        live += [d for d in (_door_row(f"real_loop_live_{rk}", r) for r in rows if r["complete"] and r["scenario_family"] == L1_SHARED_DOOR) if d]
    reuse = [d for d in (_door_row(f"m6g_reuse_{r['runtime']}", r) for r in data["reuse_replays"]
                         if r["replay"] == "T0_logical_arrival" and r["scenario_family"] == L1_SHARED_DOOR) if d]
    loop_scripted = [d for d in (_door_row(f"real_loop_scripted_{_rk(r['runtime'])}", r) for r in data["scripted_loop"]
                                 if r["complete"] and r["scenario_family"] == L1_SHARED_DOOR) if d]
    everything = scripted + live + reuse + loop_scripted
    rule = lambda d: (d["door_state_before_first"] == "locked" and d["door_state_before_second"] == "open"  # noqa: E731
                      and d["first_cost"] == 1.0 and d["second_cost"] == 0.3 and d["final_door_state"] == "open")
    grouped: dict[tuple, list] = defaultdict(list)
    for d in everything:
        grouped[(d["source"].split(":")[0], d["order"], d["door_state_before_first"], d["door_state_before_second"], d["first_cost"],
                 d["second_cost"], d["first_actor_busy_at_second_apply"], d["final_door_state"], d["second_staleness"], d["second_consequence"])].append(d)
    table = [{"source": k[0], "order": k[1], "n": len(v), "door_state_before_first": k[2], "door_state_before_second": k[3], "first_cost": k[4],
              "second_cost": k[5], "first_actor_busy_at_second_apply": k[6], "final_door_state": k[7], "second_staleness": k[8],
              "second_consequence": k[9]} for k, v in sorted(grouped.items())]
    busy = {k: v for k, v in sem["busy_variants"].items() if k.startswith(L1_SHARED_DOOR)}
    pure_order = bool(everything) and all(rule(d) for d in everything) and all(v["second_outcome_identical_during_vs_after_busy"] for v in busy.values())
    return {
        "rows": everything,
        "table": table,
        "n_door_contests": {"scripted_phase_b": len(scripted), "real_loop_live": len(live), "m6g_reuse_t0": len(reuse), "real_loop_scripted": len(loop_scripted)},
        "all_follow_order_rule": all(rule(d) for d in everything),
        "busy_variants": busy,
        "asymmetry_is_pure_application_order_semantics": pure_order,
        "answer": ("Pure application-order semantics. Whoever is applied first finds the door locked: the unlock-and-open happens at "
                   "that apply and costs 1.0 s of lockout. Whoever is applied second finds it already open: the re-open costs 0.3 s "
                   "and changes nothing in the world. The first actor is still inside its 1.0 s lockout when the second applies, and "
                   "the second actor's outcome is identical whether it applies during that lockout or after it. There is no in-flight "
                   "door state, so the duration is not a long-running action.") if pure_order else
                  "NOT a pure application-order effect - inspect door rows",
    }


# -- commitment / answer order (spec sections 12, 17, 30-31, 34, 58) --------------------------


def _trial_summary(row: dict[str, Any]) -> dict[str, Any]:
    recs = row["records"]
    first, second = row["apply_order"]
    decisions = {a: recs[a]["decision"] for a in ACTORS}
    contested = [tuple(e) for e in row.get("contested_effects") or []]
    holder = contested[0][1] if contested else None
    return {
        "trial_id": row["trial_id"], "runtime": _rk(row["runtime"]), "scenario_family": row["scenario_family"],
        "decision_call_order": row["decision_call_order"],
        "black_latency_ms": recs[ACTOR_A]["response_latency_ms"], "white_latency_ms": recs[ACTOR_B]["response_latency_ms"],
        "latency_diff_black_minus_white_ms": round(recs[ACTOR_A]["response_latency_ms"] - recs[ACTOR_B]["response_latency_ms"], 2),
        "ready_diff_black_minus_white_ms": round((recs[ACTOR_A]["response_ready_wall"] - recs[ACTOR_B]["response_ready_wall"]) * 1000.0, 2),
        "first_ready": row["ready_order"][0], "first_applied": first, "same_tick_commit": row["same_tick_commit"],
        "commit_rule_holds": _commit_rule_holds(row), "ready_order_equals_apply_order": row["ready_order_equals_apply_order"],
        "decisions": _pair_label(decisions), "shared_resource": _shares_target(decisions),
        "second_relevant_stale": recs[second]["relevant_state_changed"], "second_staleness": recs[second]["staleness_class"],
        "second_consequence": recs[second]["consequence"],
        "contested_effects": contested, "contested_holder": holder,
        "first_ready_holds_contested": (holder == row["ready_order"][0]) if holder else None,
        "world_outcome": f"{_contested(row)}; effect {_short(row.get('effect_state_hash_after'))}",
        "handoff_ticks": row.get("handoff_ticks_before_second_apply"),
    }


def _answer_order_block(trials: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trials)
    contests = [t for t in trials if t["shared_resource"]]
    held = [t for t in contests if t["first_ready_holds_contested"] is not None]
    by_call_order = {}
    for order in ("black_first", "white_first"):
        ts = [t for t in trials if t["decision_call_order"] == order]
        by_call_order[order] = {"n": len(ts), "black_ready_first": sum(t["first_ready"] == ACTOR_A for t in ts)}
    return {
        "n_trials": n,
        "black_response_ready_first": sum(t["first_ready"] == ACTOR_A for t in trials),
        "white_response_ready_first": sum(t["first_ready"] == ACTOR_B for t in trials),
        "black_response_ready_first_rate": _rate(sum(t["first_ready"] == ACTOR_A for t in trials), n),
        "white_response_ready_first_rate": _rate(sum(t["first_ready"] == ACTOR_B for t in trials), n),
        "same_tick_commits": sum(t["same_tick_commit"] for t in trials),
        "ready_order_equals_apply_order": sum(t["ready_order_equals_apply_order"] for t in trials),
        "ready_order_equals_apply_order_rate": _rate(sum(t["ready_order_equals_apply_order"] for t in trials), n),
        "commit_rule_holds": sum(t["commit_rule_holds"] for t in trials),
        "black_latency_ms": _stats([t["black_latency_ms"] for t in trials]),
        "white_latency_ms": _stats([t["white_latency_ms"] for t in trials]),
        "latency_diff_black_minus_white_ms": _stats([t["latency_diff_black_minus_white_ms"] for t in trials]),
        "abs_latency_diff_ms": _stats([abs(t["latency_diff_black_minus_white_ms"]) for t in trials]),
        "ready_diff_black_minus_white_ms": _stats([t["ready_diff_black_minus_white_ms"] for t in trials]),
        "abs_ready_diff_ms": _stats([abs(t["ready_diff_black_minus_white_ms"]) for t in trials]),
        "black_ready_first_by_decision_call_order": by_call_order,
        "second_relevant_state_stale": sum(t["second_relevant_stale"] for t in trials),
        "second_relevant_state_stale_rate": _rate(sum(t["second_relevant_stale"] for t in trials), n),
        "shared_resource_pairs": len(contests),
        "p_first_response_gets_contested_resource": _rate(sum(bool(t["first_ready_holds_contested"]) for t in held), len(held)),
        "n_contests_with_contested_effect": len(held),
        "p_second_response_relevant_state_stale_given_shared_resource": _rate(sum(t["second_relevant_stale"] for t in contests), len(contests)),
        "second_staleness": distribution(t["second_staleness"] for t in trials),
        "second_consequence": distribution(t["second_consequence"] for t in trials),
        "decision_pairs": distribution(t["decisions"] for t in trials),
    }


def extra_request_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trials where an actor started a second model request before the pair
    was complete. Rows recorded after `extra_request_log` was added carry
    it; earlier rows are read from their tick record: the second decision
    committed one tick after the first, on a frozen tick, while the first
    actor sat in its lockout (so could not request) - only the second
    actor's own fresh request can have kept the clock frozen. The mechanism
    is pinned by test_decision_ready_right_after_the_opponent_commit_..."""
    out = []
    for r in rows:
        if not r["complete"] or not r["extra_requests_before_pair_complete"]:
            continue
        if r.get("extra_request_log") is not None:
            out.append({"trial_id": r["trial_id"], "requests": r["extra_request_log"], "source": "recorded"})
            continue
        first, second = r["apply_order"]
        recs = r["records"]
        schedule = r.get("observed_commit_schedule") or []
        stale_and_rerequested = (recs[second]["commit_tick"] == recs[first]["commit_tick"] + 1 and bool(schedule)
                                 and schedule[-1][0] == 0.0 and recs[first]["busy_after_apply"])
        out.append({"trial_id": r["trial_id"],
                    "requests": [{"actor_id": second, "tick": recs[second]["commit_tick"], "on_own_commit_tick": True,
                                  "opponent_committed_before": True, "returned": recs[second]["committed_action"]["type"]}]
                    if stale_and_rerequested else [],
                    "source": "inferred from the tick record" if stale_and_rerequested else "unexplained"})
    return out


def _integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if r["complete"]]
    recs = [rec for r in complete for rec in r["records"].values()]
    extra = extra_request_cases(rows)
    return {
        "extra_request_trials": extra,
        "stale_decision_applied_while_fresh_request_started": sum(
            1 for c in extra for q in c["requests"] if q.get("on_own_commit_tick") and q.get("opponent_committed_before")),
        "n_trials": len(rows), "n_complete": len(complete), "timed_out": sum(r["timed_out"] for r in rows),
        "pre_action_matches_harness_snapshot": sum(r["pre_action_matches_harness_snapshot"] for r in rows),
        "ticks_advanced_during_inference_total": sum(r["ticks_advanced_during_inference"] for r in complete),
        "game_time_advanced_during_inference_s_total": round(sum(r["game_time_advanced_during_inference_s"] for r in complete), 6),
        "trials_with_handoff_tick": sum(1 for r in complete if r["handoff_ticks_before_second_apply"]),
        "max_handoff_ticks": max((r["handoff_ticks_before_second_apply"] for r in complete), default=0),
        "max_game_time_before_second_apply_s": max((r["game_time_before_second_apply_s"] for r in complete), default=0.0),
        "game_time_frozen_until_second_apply": sum(r["game_time_frozen_until_second_apply"] for r in complete),
        "commit_matches_decision": sum(rec["commit_matches_decision"] for rec in recs), "n_records": len(recs),
        "extra_requests_before_pair_complete": sum(r["extra_requests_before_pair_complete"] for r in complete),
        "reapplied_instant_actions_after_commit": sum(r["reapplied_instant_actions_after_commit"] or 0 for r in complete),
        "live_matches_t0_replay_exact_schedule": sum(bool(r.get("live_matches_t0_replay")) for r in complete),
        "live_effect_matches_ideal_t0": sum(bool(r.get("live_effect_matches_ideal_t0")) for r in complete),
        "live_world_matches_ideal_t0": sum(bool(r.get("live_world_matches_ideal_t0")) for r in complete),
        "ideal_t0_matches_same_order_barrier": sum(bool(r.get("t0_replay_matches_same_order_barrier")) for r in complete),
        "model_calls_for_first_decision": dict(Counter(rec["model_calls_for_first_decision"] for rec in recs)),
        "first_pass_valid": sum(rec["first_pass_valid"] for rec in recs),
    }


def _cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recs = [rec for r in rows if r["complete"] for rec in r["records"].values()]
    lat = [x for rec in recs for x in rec.get("model_call_latency_ms") or []]
    first = sum(rec["model_calls_for_first_decision"] for rec in recs)
    extra = sum(r["extra_requests_before_pair_complete"] for r in rows if r["complete"])
    return {"model_calls": first + extra, "model_calls_for_first_decisions": first,
            "extra_requests_started": extra,
            "extra_requests_note": "each extra request is one more model call, started inside the trial on a background thread "
                                   "that the runner process kept alive after the trial ended; its answer was never applied or logged",
            "mean_call_latency_ms": round(statistics.fmean(lat), 1) if lat else None,
            "mean_input_tokens_per_decision": round(statistics.fmean([rec["input_tokens"] for rec in recs]), 1) if recs else None,
            "mean_output_tokens_per_decision": round(statistics.fmean([rec["output_tokens"] for rec in recs]), 1) if recs else None}


def _logical_block(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for rk in sorted({t["runtime"] for t in telemetry}):
        ts = [t for t in telemetry if t["runtime"] == rk]
        block: dict[str, Any] = {}
        for fam in sorted({t["scenario_family"] for t in ts}) + ["_all_scenarios"]:
            cell = [t for t in ts if fam == "_all_scenarios" or t["scenario_family"] == fam]
            contests, held = [], []
            for t in cell:
                decisions = t["decisions"]
                if _shares_target(decisions):
                    contests.append(t)
                    effects = [tuple(e) for e in t["contested_effects"]]
                    if effects:
                        held.append(effects[0][1] == t["ready_order"][0])
            block[fam] = {
                "n_pairs": len(cell),
                "black_response_ready_first": sum(t["first_ready"] == ACTOR_A for t in cell),
                "white_response_ready_first": sum(t["first_ready"] == ACTOR_B for t in cell),
                "same_tick_ties": sum(t["same_tick_tie"] for t in cell),
                "latency_diff_black_minus_white_ms": _stats([t["latency_diff_black_minus_white_ms"] for t in cell]),
                "second_relevant_state_stale": sum(t["second_relevant_state_stale"] for t in cell),
                "second_relevant_state_stale_rate": _rate(sum(t["second_relevant_state_stale"] for t in cell), len(cell)),
                "shared_resource_pairs": len(contests),
                "p_first_response_gets_contested_resource": _rate(sum(held), len(held)),
                "p_second_response_relevant_state_stale_given_shared_resource": _rate(sum(t["second_relevant_state_stale"] for t in contests), len(contests)),
                "second_staleness": distribution(t["second_staleness"] for t in cell),
            }
        out[rk] = block
    out["note"] = ("Logical arrival: Black ready at its M6G latency, White at 50 ms (the loop's get_action join) + its latency; the two "
                   "M6G calls ran one after the other, so their latencies never overlapped. Apply order equals this arrival order by "
                   "construction here - the real loop measures it.")
    return out


def commitment_metrics(data: dict[str, Any]) -> dict[str, Any]:
    live: dict[str, Any] = {}
    trials: dict[str, list[dict[str, Any]]] = {}
    for rk, rows in data["live"].items():
        if not rows:
            live[rk] = {"status": "not_run"}
            continue
        ts = [_trial_summary(r) for r in rows if r["complete"]]
        trials[rk] = ts
        live[rk] = {"status": "done", "integrity": _integrity(rows), "cost": _cost(rows),
                    "answer_order": {fam: _answer_order_block([t for t in ts if fam == "_all_scenarios" or t["scenario_family"] == fam])
                                     for fam in sorted({t["scenario_family"] for t in ts}) + ["_all_scenarios"]}}
    scripted = {}
    for rk in ("native", "strands"):
        rows = [r for r in data["scripted_loop"] if _rk(r["runtime"]) == rk]
        complete = [r for r in rows if r["complete"]]
        if rows:
            scripted[rk] = {"integrity": _integrity(rows), "n": len(complete),
                            "commit_rule_holds": sum(_commit_rule_holds(r) for r in complete),
                            "fast_actor_applied_first": sum(r["apply_order"][0] != r["slow_actor"] for r in complete),
                            "same_tick_commits": sum(r["same_tick_commit"] for r in complete),
                            "second_decision": distribution(_second_labels(complete))}
    return {"real_loop_live": live, "real_loop_scripted": scripted, "m6g_logical": _logical_block(data["logical_telemetry"]), "_trials": trials}


def regenerate_telemetry(data: dict[str, Any]) -> list[dict[str, Any]]:
    """answer_order_telemetry.jsonl = M6G logical rows (as written by the
    reuse runner) + one observed row per live real-loop trial."""
    rows = list(data["logical_telemetry"])
    for rk, live_rows in data["live"].items():
        for r in live_rows:
            if not r["complete"]:
                continue
            recs = r["records"]
            second = r["apply_order"][1]
            rows.append({
                "pair_id": r["trial_id"], "timing_source": OBSERVED, "runtime": rk, "scenario_family": r["scenario_family"],
                "decision_call_order": r["decision_call_order"],
                "request_start_wall_time": {a: recs[a]["request_start_wall"] for a in ACTORS},
                "response_ready_wall_time": {a: recs[a]["response_ready_wall"] for a in ACTORS},
                "response_latency_ms": {a: recs[a]["response_latency_ms"] for a in ACTORS},
                "latency_diff_black_minus_white_ms": round(recs[ACTOR_A]["response_latency_ms"] - recs[ACTOR_B]["response_latency_ms"], 2),
                "ready_rank": {a: recs[a]["ready_rank"] for a in ACTORS},
                "first_ready": r["ready_order"][0], "ready_order": r["ready_order"], "apply_order": r["apply_order"],
                "same_tick_tie": r["same_tick_commit"], "ready_order_equals_apply_order": r["ready_order_equals_apply_order"],
                "commit_ticks": {a: recs[a]["commit_tick"] for a in ACTORS},
                "game_clock_at_each_apply": r["game_clock_at_each_apply"],
                "handoff_ticks_before_second_apply": r["handoff_ticks_before_second_apply"],
                "decisions": {a: recs[a]["decision"] for a in ACTORS},
                "second_staleness": recs[second]["staleness_class"], "second_consequence": recs[second]["consequence"],
                "second_relevant_state_stale": recs[second]["relevant_state_changed"],
                "contested_effects": r["contested_effects"], "world_state_hash_after": r["world_state_hash_after"],
            })
    with open(out_path(TELEMETRY_FILE), "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps(t, sort_keys=True, default=str) + "\n")
    return rows


# -- runtime replication (spec sections 37-40, 61) --------------------------------------------


def _second_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [f"{r['records'][r['apply_order'][1]]['staleness_class']}/{r['records'][r['apply_order'][1]]['consequence']}" for r in rows]


def _classify(native_n: int, strands_n: int, semantics_agree: bool, native_rate: float | None = None, strands_rate: float | None = None) -> str:
    if native_n == 0 or strands_n == 0:
        return RT_NONE
    if not semantics_agree:
        return RT_DEPENDENT
    if native_rate is not None and strands_rate is not None and abs(native_rate - strands_rate) >= INCIDENCE_GAP:
        return RT_PARTIAL
    return RT_INDEPENDENT


def _same_pair_same_outcome(n_rows: list[dict[str, Any]], s_rows: list[dict[str, Any]]) -> bool:
    """Where both runtimes produced the same decision pair in the same apply
    order, the stale decision's class/consequence and the world effects
    must match."""
    def key(r):
        return (r["scenario_family"], _pair_label({a: r["records"][a]["decision"] for a in ACTORS}), tuple(r["apply_order"]))

    def value(r):
        # the idealized-T0 effect for this pair and order: independent of a
        # handoff tick, which only this one live trial happened to hit
        return tuple(_second_labels([r])), (r.get("replay_outcomes") or {}).get("T0_observed_arrival", {}).get("effect_state_hash_after")

    n_by: dict[tuple, set] = defaultdict(set)
    for r in n_rows:
        n_by[key(r)].add(value(r))
    return all(value(r) in n_by[key(r)] for r in s_rows if key(r) in n_by)


def runtime_replication(data: dict[str, Any], commitment: dict[str, Any]) -> dict[str, Any]:
    n_live = [r for r in data["live"]["native"] if r["complete"]]
    s_live = [r for r in data["live"]["strands"] if r["complete"]]
    comps: dict[str, dict[str, Any]] = {}

    # 1) answer-order commit (the loop's own rule), live + scripted
    def rule(rows):
        return sum(_commit_rule_holds(r) for r in rows), len(rows)

    sc = {rk: [r for r in data["scripted_loop"] if r["complete"] and _rk(r["runtime"]) == rk] for rk in ("native", "strands")}
    nh, nn = rule(n_live + sc["native"])
    sh, sn = rule(s_live + sc["strands"])
    comps["answer_order_commit"] = {
        "native": f"commit rule holds {_pct(nh, nn)} (live {rule(n_live)[0]}/{len(n_live)}, scripted {rule(sc['native'])[0]}/{len(sc['native'])})",
        "strands": f"commit rule holds {_pct(sh, sn)} (live {rule(s_live)[0]}/{len(s_live)}, scripted {rule(sc['strands'])[0]}/{len(sc['strands'])})",
        "classification": _classify(len(n_live), len(s_live), nh == nn and sh == sn),
    }

    # 2) shared-door timing (L1, live)
    def doors(rows):
        ds = [d for d in (_door_row("x", r) for r in rows if r["scenario_family"] == L1_SHARED_DOOR) if d]
        ok = sum(d["door_state_before_first"] == "locked" and d["door_state_before_second"] == "open" and d["first_cost"] == 1.0
                 and d["second_cost"] == 0.3 for d in ds)
        return ds, ok, len([r for r in rows if r["scenario_family"] == L1_SHARED_DOOR])

    nd, nok, nl1 = doors(n_live)
    sd, sok, sl1 = doors(s_live)
    comps["shared_door_timing_L1"] = {
        "native": f"door contest {_pct(len(nd), nl1)}; first 1.0 s unlock / second 0.3 s re-open in {nok}/{len(nd)}",
        "strands": f"door contest {_pct(len(sd), sl1)}; first 1.0 s unlock / second 0.3 s re-open in {sok}/{len(sd)}",
        "classification": _classify(len(nd), len(sd), nok == len(nd) and sok == len(sd), _rate(len(nd), nl1), _rate(len(sd), sl1)),
    }

    # 3) stale-but-valid (the L1 second-applied decision)
    n1 = [r for r in n_live if r["scenario_family"] == L1_SHARED_DOOR]
    s1 = [r for r in s_live if r["scenario_family"] == L1_SHARED_DOOR]
    nsbv = [x for x in _second_labels(n1) if x.startswith(STALE_BUT_VALID)]
    ssbv = [x for x in _second_labels(s1) if x.startswith(STALE_BUT_VALID)]
    comps["stale_but_valid_L1_second_decision"] = {
        "native": ", ".join(f"{k} {v}" for k, v in distribution(_second_labels(n1)).items()) or "none",
        "strands": ", ".join(f"{k} {v}" for k, v in distribution(_second_labels(s1)).items()) or "none",
        "classification": _classify(len(nsbv), len(ssbv), _same_pair_same_outcome(n1, s1), _rate(len(nsbv), len(n1)), _rate(len(ssbv), len(s1))),
    }

    # 4) stale-and-invalid (scripted decisions through each runtime's real loop)
    def invalid_cells(rows):
        return {(r["scenario_family"], r["slow_actor"]): _second_labels([r])[0] for r in rows
                if r["scenario_family"] in (L3_TRAP_SLOT_CONFLICT, L4_DISARM_CONFLICT, L5_ACTION_INVALIDATES_OTHER)}

    ni, si = invalid_cells(sc["native"]), invalid_cells(sc["strands"])
    n_inv = sum(v.startswith(STALE_AND_INVALID) for v in ni.values())
    s_inv = sum(v.startswith(STALE_AND_INVALID) for v in si.values())
    comps["stale_and_invalid_L3_L4_L5_scripted_real_loop"] = {
        "native": f"STALE_AND_INVALID second decision in {n_inv}/{len(ni)} cells: " + ", ".join(f"{k} {v}" for k, v in distribution(ni.values()).items()),
        "strands": f"STALE_AND_INVALID second decision in {s_inv}/{len(si)} cells: " + ", ".join(f"{k} {v}" for k, v in distribution(si.values()).items()),
        "classification": _classify(n_inv, s_inv, ni == si),
        "note": "decisions scripted (no live trap generation: spec sections 11, 39); each runtime's own adapter lifecycle, validation and executor ran",
    }

    # 5) non-conflict control (L0, live)
    def control(rows):
        rs = [r for r in rows if r["scenario_family"] == L0_NON_CONFLICTING]
        reps = _pairs_from_replays([x for x in data["live_replays"] if x["scenario_family"] == L0_NON_CONFLICTING], "source_trial_id")
        non_sharing = [r for r in rs if not _shares_target({a: r["records"][a]["decision"] for a in ACTORS})]
        order_sensitive = sum(1 for r in non_sharing if r["trial_id"] in reps and _outcome(reps[r["trial_id"]]["T1A"]) != _outcome(reps[r["trial_id"]]["T1B"]))
        stale = sum(r["records"][r["apply_order"][1]]["relevant_state_changed"] for r in non_sharing)
        return rs, non_sharing, order_sensitive, stale

    n0, n0ns, n0os, n0st = control(n_live)
    s0, s0ns, s0os, s0st = control(s_live)
    comps["non_conflict_control_L0"] = {
        "native": f"{len(n0ns)}/{len(n0)} pairs share no target; order-sensitive {n0os}, second relevant-stale {n0st}",
        "strands": f"{len(s0ns)}/{len(s0)} pairs share no target; order-sensitive {s0os}, second relevant-stale {s0st}",
        "classification": _classify(len(n0ns), len(s0ns), n0os == 0 and s0os == 0 and n0st == 0 and s0st == 0),
    }

    # 6) arrival adds nothing beyond application order (live captured pairs)
    def explained(rows):
        return sum(bool(r.get("t0_replay_matches_same_order_barrier")) for r in rows), len(rows)

    ne, nt = explained(n_live)
    se, st = explained(s_live)
    comps["arrival_timing_adds_nothing_beyond_application_order"] = {
        "native": f"idealized T0 == same-order barrier {_pct(ne, nt)}", "strands": f"idealized T0 == same-order barrier {_pct(se, st)}",
        "classification": _classify(nt, st, ne == nt and se == st),
    }

    if not s_live:
        return {"status": "not_run", "comparisons": comps, "overall": None}
    labels = [c["classification"] for c in comps.values()]
    if RT_DEPENDENT in labels:
        overall = RT_DEPENDENT
    elif all(x == RT_INDEPENDENT for x in labels):
        overall = RT_INDEPENDENT
    elif all(x == RT_NONE for x in labels):
        overall = RT_NONE
    else:
        overall = RT_PARTIAL
    def rerequested(rows):
        return sum(1 for c in extra_request_cases(rows) for q in c["requests"] if q.get("on_own_commit_tick"))

    def _closest(rk):
        block = commitment["real_loop_live"].get(rk, {}).get("answer_order", {}).get("_all_scenarios", {})
        return (block.get("abs_ready_diff_ms") or {}).get("min")

    return {
        "status": "done", "comparisons": comps, "overall": overall,
        "additional_observations": {
            "stale_decision_applied_while_fresh_request_started": {
                "native_live_trials": rerequested(n_live), "strands_live_trials": rerequested(s_live),
                "classification": "not classified (not in the frozen comparison set)",
                "note": "both adapters share the code path and behave identically when the interleaving is forced (test on both "
                        "runtimes); live incidence follows how close the two answers become ready - the closest Strands pair was "
                        f"{_closest('strands')} ms apart, the closest Native pair {_closest('native')} ms",
            },
        },
        "semantics_agree_wherever_both_observed": RT_DEPENDENT not in labels,
        "every_comparison_observed_in_both": RT_NONE not in labels,
        "rule": f"per comparison: {RT_NONE} if one runtime never produced the situation; {RT_DEPENDENT} if the temporal semantics differ; "
                f"{RT_PARTIAL} if they agree but the situation's incidence differs by >= {INCIDENCE_GAP}; else {RT_INDEPENDENT}. Overall: "
                f"{RT_DEPENDENT} if any comparison is; {RT_INDEPENDENT} if all are; otherwise {RT_PARTIAL}.",
    }


# -- overall classification (spec sections 62-66, 70) ----------------------------------------


def overall_classification(sem, conflicts, commitment, barrier, door, replication, entries) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    live_trials = [t for ts in commitment["_trials"].values() for t in ts]
    contests = [t for t in live_trials if t["shared_resource"] and t["first_ready_holds_contested"] is not None]
    rule_ok = bool(live_trials) and all(t["commit_rule_holds"] for t in live_trials)
    evidence["ANSWER_ORDER_IS_EFFECTIVE_PRIORITY"] = {
        "live_trials": len(live_trials), "commit_rule_holds": sum(t["commit_rule_holds"] for t in live_trials),
        "live_contests_with_contested_effect": len(contests), "first_ready_held_contested_effect": sum(t["first_ready_holds_contested"] for t in contests),
        "supported": rule_ok and bool(contests) and all(t["first_ready_holds_contested"] for t in contests),
    }
    scripted_ok = all(c["t0_a_first_equals_t1a"] and c["t0_b_first_equals_t1b"] for c in conflicts.values())
    evidence["ANSWER_ORDER_ONLY_SELECTS_APPLICATION_ORDER"] = {
        "captured_pairs": barrier["n_pairs"], "captured_pairs_explained": barrier["arrival_adds_nothing_beyond_application_order"],
        "scripted_t0_equals_same_order_barrier_all_scenarios": scripted_ok,
        "supported": barrier["arrival_adds_nothing_beyond_application_order"] and scripted_ok,
    }

    def count(pred):
        c = Counter(f"{e['source']}:{e['condition']}" for e in entries if pred(e))
        return dict(sorted(c.items()))

    changed = lambda e: e["relevant_state_changed"] and e["consequence"] not in (ACCEPTED_UNCHANGED, NO_OP)  # noqa: E731
    sbv = lambda e: e["staleness_class"] == STALE_BUT_VALID and e["consequence"] in (DIFFERENT_COST, DIFFERENT_WORLD_EFFECT)  # noqa: E731
    sai = lambda e: e["staleness_class"] == STALE_AND_INVALID  # noqa: E731
    for label, pred in (("STALE_DECISION_EFFECT_SUPPORTED", changed), ("STALE_BUT_VALID_EFFECT_SUPPORTED", sbv), ("STALE_AND_INVALID_EFFECT_SUPPORTED", sai)):
        by = count(pred)
        evidence[label] = {"decisions_by_source": by, "supported": sum(by.values()) > 0}
    evidence["DURATION_IS_COST_ONLY"] = {
        "classification_counts": sem["classification_counts"], "no_in_flight_state": sem["no_in_flight_state"],
        "any_effect_at_completion": sem["any_effect_at_completion"], "door_asymmetry_is_pure_application_order": door["asymmetry_is_pure_application_order_semantics"],
        "supported": sem["no_in_flight_state"] and not sem["any_effect_at_completion"] and door["asymmetry_is_pure_application_order_semantics"],
        "caveat": "timed actions are EFFECT_AT_START_WITH_BUSY_PERIOD: the world changes at apply and the duration is paid as an actor "
                  "lockout - a cost, not an action in flight",
    }
    evidence["TRUE_IN_FLIGHT_ACTION_SEMANTICS_SUPPORTED"] = {"supported": not sem["no_in_flight_state"]}
    evidence["COMPLETION_ORDER_EFFECT_SUPPORTED"] = {"supported": sem["any_effect_at_completion"]}
    if replication.get("status") == "done":
        evidence["TEMPORAL_EFFECT_RUNTIME_INDEPENDENT"] = {"supported": replication["semantics_agree_wherever_both_observed"] and replication["every_comparison_observed_in_both"],
                                                          "overall": replication["overall"]}
        evidence["TEMPORAL_EFFECT_RUNTIME_DEPENDENT"] = {"supported": not replication["semantics_agree_wherever_both_observed"]}
    labels = [k for k, v in evidence.items() if v["supported"]]
    return {"labels": labels, "evidence": evidence}


def next_step_gate(overall: dict[str, Any], staleness: dict[str, Any]) -> dict[str, Any]:
    ev = overall["evidence"]
    live_second = [c["second_applied"] for src, conds in staleness["by_source"].items() if src.startswith("real_loop_live")
                   for c in [conds.get("T0_live", {}).get("_all_scenarios")] if c]
    stale = sum(s["relevant_state_stale"] for s in live_second)
    legal_changed = sum(s["consequences_relevant_stale"][DIFFERENT_COST] + s["consequences_relevant_stale"][DIFFERENT_WORLD_EFFECT] for s in live_second)
    conditions = {
        "answer_order_only_selects_application_order": ev["ANSWER_ORDER_ONLY_SELECTS_APPLICATION_ORDER"]["supported"],
        "actions_are_cost_only": ev["DURATION_IS_COST_ONLY"]["supported"],
        "true_in_flight_actions_exist": ev["TRUE_IN_FLIGHT_ACTION_SEMANTICS_SUPPORTED"]["supported"],
        "live_second_decisions_relevant_stale": stale,
        "live_stale_decisions_legal_with_changed_cost_or_effect": legal_changed,
    }
    if conditions["answer_order_only_selects_application_order"] and conditions["actions_are_cost_only"]:
        nxt = "M6G.2 — Commitment Policy Counterbalancing"
    elif conditions["true_in_flight_actions_exist"]:
        nxt = "M6G.2 — In-Flight Conflict & Completion Ordering"
    else:
        nxt = "stale-decision policy milestone or next harness variable"
    return {"conditions": conditions, "natural_next_step": nxt, "started": False,
            "note": "spec section 70: recorded only - no follow-up milestone is started"}


# -- tables (spec sections 56-61) ----------------------------------------------------------


def required_tables(sem, stale_rows, commitment, barrier_rows, door, replication) -> str:
    L: list[str] = []
    L += ["### Action-semantics table (spec section 56)", "",
          "| Action | Duration / cost | Busy state? | World-effect timing | Apply-time revalidation? | Can conflict while in flight? | Classification |",
          "|---|---|---|---|---|---|---|"]
    for r in sem["actions"]:
        L.append(f"| {r['action']} | {r['duration_cost']} | {r['busy_state']} | {r['world_effect_timing']} | {r['apply_time_revalidation']} | "
                 f"{r['can_conflict_while_in_flight']} | {r['classification']} |")

    L += ["", "### Staleness table (spec section 57) — scripted pairs, barrier orders", "",
          "| Scenario | Order | Actor (position) | Decision | Snapshot hash | Apply hash | Relevant state changed? | Legal at snapshot? | Legal at apply? | Staleness class | Final result |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in stale_rows["scripted_t1"]:
        order = "A→B" if e["condition"].startswith("T1A") else "B→A"
        L.append(f"| {e['scenario_family']} | {order} | {e['actor_id']} ({e['apply_position']}) | {e['decision']} | `{_short(e['snapshot_world_hash'])}` | "
                 f"`{_short(e['apply_world_hash'])}` | {e['relevant_state_changed']} | {e['legal_at_snapshot']} | {e['legal_at_apply']} | "
                 f"{e['staleness_class']} | {e['consequence']} ({e['apply_outcome']}, cost {e['time_cost']:g} s, listed at apply: {e['listed_at_apply']}) |")
    L += ["", "### Staleness table (spec section 57) — live real loop and M6G reuse (T0), grouped", "",
          "| Source | Scenario | Position | Decision | n | Snapshot hash | Apply hash | Relevant state changed? | Legal at snapshot? | Legal at apply? | Staleness class | Final result |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in stale_rows["live_and_reuse_t0_grouped"]:
        L.append(f"| {g['source']} | {g['scenario_family']} | {g['apply_position']} | {g['decision']} | {g['n']} | `{','.join(g['snapshot_hashes'])}` | "
                 f"`{','.join(g['apply_hashes'][:3])}{'…' if len(g['apply_hashes']) > 3 else ''}` | {g['relevant_state_changed']} | {g['legal_at_snapshot']} | "
                 f"{g['legal_at_apply']} | {g['staleness_class']} | {g['consequence']} ({g['apply_outcome']}) |")

    L += ["", "### Answer-order table (spec section 58) — live real loop", "",
          "| Trial | Black latency (ms) | White latency (ms) | First ready | First applied | Shared resource? | Second stale? | World outcome |",
          "|---|---|---|---|---|---|---|---|"]
    for rk, ts in commitment["_trials"].items():
        for t in ts:
            tie = " (same tick)" if t["same_tick_commit"] else ""
            L.append(f"| {t['trial_id']} | {t['black_latency_ms']:.0f} | {t['white_latency_ms']:.0f} | {t['first_ready']} | {t['first_applied']}{tie} | "
                     f"{t['shared_resource']} | {t['second_relevant_stale']} ({t['second_staleness']}) | {t['world_outcome']} |")
    if not commitment["_trials"]:
        L.append("| (live real loop not run yet) | | | | | | | |")

    L += ["", "### Barrier table (spec section 59) — captured decision pairs, grouped", "",
          "Pairs with the same decisions, arrival order and outcomes are grouped; every pair is listed with its id in "
          "`commitment_metrics.json` (`barrier_pairs`).", "",
          "| Source | Decision pair ids | Scenario | Decision pair (Black / White) | T0 arrival order | n | T0 outcome | T1A A→B outcome | T1B B→A outcome | Difference explained by application order? |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b in barrier_rows:
        ids = f"`{b['pair_ids'][0]}`" + (f" +{b['n'] - 1}" if b["n"] > 1 else "")
        L.append(f"| {b['source']} | {ids} | {b['scenario_family']} | {b['decision_pair']} | {b['t0_arrival_order']} | {b['n']} | {b['t0_outcome']} | "
                 f"{b['t1a_outcome']} | {b['t1b_outcome']} | {b['explained']}/{b['n']} |")

    L += ["", "### Door table (spec section 60)", "",
          "| Source | Order | n | Door before first | Door before second | First cost | Second cost | Busy / in-flight state? | Final door state |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in door["table"]:
        busy = f"first actor in 1.0 s lockout at second apply: {d['first_actor_busy_at_second_apply']}; no in-flight door state"
        L.append(f"| {d['source']} | {d['order']} | {d['n']} | {d['door_state_before_first']} | {d['door_state_before_second']} | {d['first_cost']:g} s | "
                 f"{d['second_cost']:g} s | {busy} | {d['final_door_state']} |")

    comps = replication.get("comparisons") or {}
    if replication.get("status") == "done":
        L += ["", "### Runtime table (spec section 61)", "", "| Scenario | Native result | Strands result | Classification |", "|---|---|---|---|"]
        for name, c in comps.items():
            L.append(f"| {name} | {c['native']} | {c['strands']} | {c['classification']} |")
    return "\n".join(L)


# -- entry point ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    data = load()
    sem = action_semantics(data["probes"], data["busy"])
    with open(out_path("action_temporal_semantics.md"), "w", encoding="utf-8") as f:
        f.write(action_semantics_markdown(sem))
    entries = staleness_entries(data)
    staleness = staleness_metrics(entries)
    stale_rows = staleness_table_rows(entries)
    conflicts = scripted_conflicts(data, sem)
    barrier = barrier_analysis(data)
    door = door_analysis(data, sem)
    commitment = commitment_metrics(data)
    telemetry = regenerate_telemetry(data)
    replication = runtime_replication(data, commitment)
    overall = overall_classification(sem, conflicts, commitment, barrier, door, replication, entries)
    gate = next_step_gate(overall, staleness)

    write_json(out_path("staleness_metrics.json"), {**staleness, "table_57_scripted_t1": stale_rows["scripted_t1"],
                                                     "table_57_live_and_reuse_t0_grouped": stale_rows["live_and_reuse_t0_grouped"]})
    trials = commitment.pop("_trials")
    write_json(out_path("commitment_metrics.json"), {**commitment, "live_trials_table_58": trials, "barrier_replays": barrier["summary"],
                                                      "barrier_pairs": barrier["pairs"], "arrival_adds_nothing_beyond_application_order":
                                                      barrier["arrival_adds_nothing_beyond_application_order"],
                                                      "door": {k: v for k, v in door.items() if k != "rows"}, "door_rows": door["rows"],
                                                      "scripted_temporal_conflicts": conflicts})
    write_json(out_path("runtime_replication.json"), replication)
    commitment["_trials"] = trials

    live_calls: dict[str, Any] = {rk: blk.get("cost", {}).get("model_calls", 0) for rk, blk in commitment["real_loop_live"].items()
                                  if blk.get("status") == "done"}
    live_calls["preflight_connectivity_calls"] = sum(1 for e in data["provider_events"] if str(e.get("phase", "")).startswith("preflight"))
    live_calls["total"] = sum(v for v in live_calls.values())
    stale_rerequest = {rk: blk["integrity"]["stale_decision_applied_while_fresh_request_started"]
                       for rk, blk in commitment["real_loop_live"].items() if blk.get("status") == "done"}
    summary = {
        "milestone": "M6G.1",
        "title": "Long-Running Actions, Stale Decisions & Commitment Semantics",
        "overall_classification": overall["labels"],
        "classification_evidence": overall["evidence"],
        "model_id": PROBE_MODEL_ID, "provider": "bedrock", "communication_policy": "SOCIAL_OFF", "personality": "default vs default",
        "counts": {"phase_a_probes": len(data["probes"]), "phase_b_scripted_rows": len(data["scripted_pairs"]) + len(data["busy"]),
                   "m6g_reused_pairs": data["reuse_manifest"]["n_pairs"], "m6g_reuse_replays": len(data["reuse_replays"]),
                   "real_loop_live_trials": {rk: len(rows) for rk, rows in data["live"].items()},
                   "real_loop_live_replays": len(data["live_replays"]), "real_loop_scripted_trials": len(data["scripted_loop"]),
                   "answer_order_telemetry_rows": len(telemetry)},
        "new_live_model_calls": live_calls,
        "stale_decision_applied_while_fresh_request_started": {
            "trials": stale_rerequest,
            "mechanism": "the slow decision became ready after the opponent's commit changed its observation but before its own next "
                         "get_action: the controller saw a new observation digest with nothing in flight, started a fresh request, and "
                         "returned the stale decision as the in-flight action - which the Engine applied (on a frozen tick, since the "
                         "fresh request was now in flight). Window: about one loop tick; hit only when the two answers land within "
                         "that window of each other. Same code path in LLMController and StrandsAgentAdapter; pinned by a test on both.",
        },
        "action_temporal_semantics": {"classification_counts": sem["classification_counts"], "no_in_flight_state": sem["no_in_flight_state"],
                                      "any_effect_at_completion": sem["any_effect_at_completion"],
                                      "frozen_updates_never_drain_busy": sem["frozen_updates_never_drain_busy"]},
        "scripted_temporal_conflicts": {f: c["classification"] for f, c in conflicts.items()},
        "door": {"answer": door["answer"], "n_door_contests": door["n_door_contests"], "all_follow_order_rule": door["all_follow_order_rule"]},
        "barrier": {"captured_pairs": barrier["n_pairs"], "arrival_adds_nothing_beyond_application_order": barrier["arrival_adds_nothing_beyond_application_order"],
                    "by_source": {s: v["_all_scenarios"] for s, v in barrier["summary"].items()}},
        "real_loop_live": {rk: ({"integrity": blk["integrity"], "answer_order": blk["answer_order"]["_all_scenarios"]} if blk.get("status") == "done" else blk)
                           for rk, blk in commitment["real_loop_live"].items()},
        "real_loop_scripted": commitment["real_loop_scripted"],
        "runtime_replication": {"status": replication.get("status"), "overall": replication.get("overall"),
                                "comparisons": {k: v["classification"] for k, v in (replication.get("comparisons") or {}).items()}},
        "next_step_gate": gate,
        "notes": [
            "Every order/arrival comparison replays the same captured decision pair, so provider/model variance cannot appear as an "
            "answer-order effect; live real-loop decisions are never compared across conditions as distributions.",
            "Real-loop requests carry the production controller's own memory, so they are not byte-identical to the M6G harness "
            "requests and are not pooled with M6G decision distributions.",
            "The stale-and-invalid cases (L3/L4/L5) use scripted decisions - through the Engine (Phase B) and through each runtime's "
            "real loop - because live trap placement is confounded by missing trap_type (M6G C4).",
        ],
    }
    write_json(out_path("summary.json"), summary)
    summary["_tables_markdown"] = required_tables(sem, stale_rows, commitment, barrier_table_rows(barrier), door, replication)
    splice_generated_tables(out_path("summary.md"), summary["_tables_markdown"])
    return summary


TABLES_BEGIN = "<!-- BEGIN GENERATED TABLES (run_m6g1_analysis.py) -->"
TABLES_END = "<!-- END GENERATED TABLES -->"


def splice_generated_tables(path: str, tables: str) -> bool:
    """summary.md is written by hand except for the required tables, which
    are regenerated here between two marker lines so they always match
    the data. A file without the markers is left untouched."""
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
    print(json.dumps({k: out[k] for k in ("overall_classification", "counts", "new_live_model_calls", "scripted_temporal_conflicts", "door",
                                          "barrier", "real_loop_live", "real_loop_scripted", "runtime_replication", "next_step_gate")},
                     indent=2, default=str))
    print()
    print(tables)
    print(f"\n(outputs in {OUT_DIR})")
