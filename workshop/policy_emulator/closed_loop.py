"""Stage 3 milestone 3: closed-loop behavioural fidelity - Phase A vs B1 vs B2. No provider call.

Three sources, never mixed:

    phase_a       the frozen Phase-A LLM matches (CLEAN decisions + their match telemetry, read only)
    table-v1      B1 surrogates on both sides, closed loop
    logistic-v1   B2 surrogates on both sides, closed loop

One "spy record" per spy and match: its decisions (features, action, coverage,
confidence) and its trajectory (engine events, fights, terminal). The same
extraction runs on both telemetry kinds; only the decision events differ
(Phase A: CLEAN rows of the dataset; closed loop: `surrogate_decision`).
Offline replay values (milestone 2) are never mixed in.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from workshop.policy_emulator import dataset as ds
from workshop.policy_emulator import schema
from workshop.policy_emulator.evaluation import OPPORTUNITIES
from workshop.runners.common import PROJECT_ROOT, WorkshopError

ANALYSIS_VERSION = "closed-loop-analysis-v2"
PHASE_A_DIR = "workshop_runs/cross-model-phase-a-v2-sonnet55-vs-gpt61sol"
DATASET = "phase-a-v2"
CONDITION_KEYS = ("seed", "mission_preset", "encounters", "lives", "spawn", "items", "puzzle", "decision_timing",
                  "mission_clock", "mission_timer", "commit_policy", "runtime")
ACTORS = ("black_spy", "white_spy")
PHASE_A = "phase_a"
B1, B2 = "table-v1", "logistic-v1"
LABELS = {PHASE_A: "Phase A (real LLM)", B1: "B1 table (closed loop)", B2: "B2 logistic (closed loop)",
          "tree-v1": "B3 tree (closed loop)", "frequency-v1": "B0 frequency (closed loop)",
          "hybrid": "Hybrid R2 (B2 + real-LLM escalation)"}
SHORT = {PHASE_A: "Phase A", B1: "B1", B2: "B2", "tree-v1": "B3", "frequency-v1": "B0", "hybrid": "Hybrid"}
REQUIRED_ITEMS = ("passport", "secret_documents")
TERMINAL_EVENTS = ("MISSION_COMPLETED", "ACTOR_ELIMINATED", "ACTOR_TIMED_OUT", "MISSION_IMPOSSIBLE")
LIFECYCLES = ("ESCAPED", "MISSION_IMPOSSIBLE", "ELIMINATED", "TIMED_OUT")
COVERAGE_LABELS = ("HIGH", "MEDIUM", "LOW", "OOD")
APPROACH_FIGHT_WINDOW_S = 10.0
HIGH_CONFIDENCE, LOW_CONFIDENCE = 0.8, 0.5
MIN_OPPORTUNITIES = 5   # an identity x opportunity cell enters the MAE only with this many opportunities in both sources
BOOTSTRAP_RESAMPLES = 1000
VISITATION: dict[str, Callable[[dict], Any]] = {
    "room": lambda d: d["features"].get("current_room"),
    "opponent alert": lambda d: bool(d["features"].get("opponent_alert_present")),
    "health band": lambda d: d["features"].get("health_band"),
    "life balance": lambda d: d["features"].get("life_balance"),
    "time pressure": lambda d: d["features"].get("time_pressure"),
    "inspect opportunity": lambda d: bool(d["features"].get("inspect_legal")),
    "trap opportunity": lambda d: bool(d["features"].get("place_trap_legal")),
    "search opportunity": lambda d: bool(d["features"].get("search_legal")),
    "mission complete": lambda d: bool(d["features"].get("mission_complete")),
    "legal-action set": lambda d: "|".join(sorted(d["legal"])),
    "coverage": lambda d: d.get("coverage") or "none",
}
# (key, label, kind): kind "count" / "rate" (bool per spy) / "time" (None when never reached) / "per_min"
TRAJECTORY = [
    ("traps_placed", "traps placed", "count"), ("traps_on_opponent", "traps triggered on the opponent", "count"),
    ("trap_hits", "traps triggered on self", "count"), ("inspections", "inspections", "count"),
    ("searches", "searches", "count"), ("approaches", "deliberate approaches", "count"),
    ("avoids", "avoid actions", "count"), ("fights", "fights", "count"),
    ("approach_fights", f"fights within {APPROACH_FIGHT_WINDOW_S:g} s of own approach", "count"),
    ("hearts_lost", "hearts lost", "count"), ("items_collected", "objective items collected", "count"),
    ("rooms_visited", "rooms visited", "count"),
    ("mission_complete", "mission completion", "rate"), ("escaped", "escape", "rate"),
    ("mission_impossible", "MISSION_IMPOSSIBLE", "rate"), ("eliminated", "ELIMINATED", "rate"),
    ("timed_out", "TIMEOUT", "rate"), ("decisions_per_min", "decisions per simulated minute", "per_min"),
    ("t_first_objective", "time to first objective item (s)", "time"),
    ("t_objectives_complete", "time to mission complete (s)", "time"),
    ("t_terminal", "time to terminal (s)", "time"),
]
REQUIRED_DECISION_FIELDS = ("controller_mode", "policy", "emulator", "target_emulator", "intended_action",
                            "intended_target", "probs", "confidence", "coverage_label", "coverage_score",
                            "coverage_reasons", "policy_rng_seed", "environment_seed", "mission_time_remaining",
                            "mission_elapsed", "decision_index")
NULLABLE_FIELDS = ("intended_target",)  # a wait has no target
SEQUENCES = ("MOVE -> MOVE", "INSPECT -> SEARCH", "SEARCH -> PLACE_TRAP", "APPROACH -> fight", "AVOID -> resume mission")
MISSION_CATEGORIES = ("MOVE_PROGRESS", "SEARCH", "INSPECT", "EXIT")


# -- reading ---------------------------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def phase_a_configurations(root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """The 48 Phase-A configurations, in Phase-A order, from the frozen manifest and run.json files (read only)."""
    manifest = json.loads((root / PHASE_A_DIR / "manifest.json").read_text())
    configs = []
    for m in manifest["matches"]:
        if m.get("status") != "done":
            raise WorkshopError(f"Phase-A match {m['name']} is not done")
        run = json.loads((root / m["run_dir"] / "run.json").read_text())
        cfg = run["configuration"]
        if m["black"] != m["white"]:
            raise WorkshopError(f"{m['name']}: Phase A pairs the same personality on both sides")
        conditions = {k: cfg[k] for k in CONDITION_KEYS}
        if conditions["seed"] != m["seed"]:
            raise WorkshopError(f"{m['name']}: run.json seed {conditions['seed']} != manifest seed {m['seed']}")
        black = schema.MODELS[m["black_model_key"]]["name"]
        white = schema.MODELS[m["white_model_key"]]["name"]
        configs.append({
            "config_id": f"{m['seed']}-{m['black']}-{m['orientation']}", "seed": m["seed"], "seed_index": m["seed_index"],
            "personality": m["black"], "orientation": m["orientation"],
            "black_policy": f"{black}:{m['black']}", "white_policy": f"{white}:{m['white']}",
            "phase_a_match": m["name"], "phase_a_run_dir": m["run_dir"], "phase_a_game_version": run.get("game_version"),
            "conditions": conditions,
        })
    return configs


def _mission_clock(events: list[dict], limit: float) -> Callable[[float], float]:
    """World game time -> mission seconds used, from the engine events that carry both."""
    anchors = sorted((e["game_time"], limit - e["mission_time_remaining"]) for e in events
                     if isinstance(e.get("game_time"), int | float) and isinstance(e.get("mission_time_remaining"), int | float))

    def convert(world: float) -> float:
        if not anchors:
            return world
        before = [a for a in anchors if a[0] <= world]
        after = [a for a in anchors if a[0] >= world]
        if before and after and after[0][0] > before[-1][0]:
            (g0, m0), (g1, m1) = before[-1], after[0]
            return m0 + (m1 - m0) * (world - g0) / (g1 - g0)
        g, m = before[-1] if before else after[0]
        return m + (world - g)
    return convert


def spy_records(records: list[dict], summary: dict, decisions: dict[str, list[dict]], meta: dict[str, Any],
                decision_counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """One record per spy: decisions + trajectory, identical extraction for every source."""
    limit = float((summary.get("mission") or {}).get("time_limit_s") or 300.0)
    events = [r for r in records if r.get("event_type") == "engine_event"]
    fights_raw = [r for r in records if r.get("event_type") == "spy_fight"]
    result = next((r for r in records if r.get("event_type") == "match_result"), None) or {}
    ranking = {x["actor_id"]: x for x in result.get("ranking", [])}
    to_mission = _mission_clock(events, limit)
    timeline = sorted(({"t": round(limit - e["mission_time_remaining"], 3), "type": e.get("engine_event_type"),
                        "actor": e.get("actor_id"), "item": e.get("item_id"), "owner": e.get("trap_owner_id"),
                        "own_trap": e.get("own_trap"), "room": e.get("room")} for e in events
                       if isinstance(e.get("mission_time_remaining"), int | float)), key=lambda x: x["t"])
    fights = [{"t": round(to_mission(float(f["game_time"])), 3), "winner": f.get("winner_id"), "loser": f.get("loser_id")}
              for f in fights_raw if isinstance(f.get("game_time"), int | float)]
    match_end = max([x["t"] for x in timeline] + [0.0])
    out = []
    for me in ACTORS:
        opp = ACTORS[1 - ACTORS.index(me)]
        mine = [x for x in timeline if x["actor"] == me]
        held, complete_t, first_t, collected = set(), None, None, 0
        for x in timeline:
            if x["type"] == "ITEM_FOUND" and x["actor"] == me and x["item"] in REQUIRED_ITEMS:
                held.add(x["item"])
                collected += 1
                first_t = first_t if first_t is not None else x["t"]
            elif x["type"] == "ITEM_STOLEN" and x["item"] in REQUIRED_ITEMS:
                if x["actor"] == me:
                    held.add(x["item"])
                    collected += 1
                    first_t = first_t if first_t is not None else x["t"]
                else:
                    held.discard(x["item"])
            if complete_t is None and held >= set(REQUIRED_ITEMS):
                complete_t = x["t"]
        end_t = next((x["t"] for x in mine if x["type"] in TERMINAL_EVENTS), None)
        lifecycle = (ranking.get(me) or {}).get("lifecycle") or ((summary.get("actors") or {}).get(me) or {}).get("lifecycle")
        active = end_t if end_t is not None else match_end
        ds_me = sorted(decisions.get(me, []), key=lambda d: (d["t"], d.get("decision_index") or 0))
        my_fights = [f for f in fights if me in (f["winner"], f["loser"])]
        approach_ts = [d["t"] for d in ds_me if d["action"] == "approach_opponent"]
        rooms = {d["features"].get("current_room") for d in ds_me if d["features"].get("current_room")}
        rooms |= {x["room"] for x in mine if x["type"] == "ACTOR_RESPAWNED" and x["room"]}
        n_all = (decision_counts or {}).get(me, len(ds_me))
        terminal = ((summary.get("actors") or {}).get(me) or {}).get("terminal") or {}
        policy = meta["policies"][me]
        model, personality = schema.parse_policy_key(policy)
        out.append({
            **{k: meta.get(k) for k in ("source", "emulator", "replicate", "config_id", "seed", "seed_index",
                                         "orientation", "match_id", "run_dir")},
            "actor_id": me, "side": "black" if me == "black_spy" else "white", "policy": policy, "model": model,
            "personality": personality, "opponent_policy": meta["policies"][opp],
            "decisions": ds_me, "n_decisions": n_all, "lifecycle": lifecycle,
            "escaped": lifecycle == "ESCAPED", "mission_impossible": lifecycle == "MISSION_IMPOSSIBLE",
            "eliminated": lifecycle == "ELIMINATED", "timed_out": lifecycle == "TIMED_OUT",
            "mission_complete": bool((ranking.get(me) or {}).get("objectives_complete")) or lifecycle == "ESCAPED",
            "t_first_objective": first_t, "t_objectives_complete": complete_t, "t_terminal": end_t,
            "active_mission_s": round(active, 3), "match_mission_s": round(match_end, 3),
            "traps_placed": sum(1 for x in mine if x["type"] == "TRAP_PLACED"),
            "traps_on_opponent": sum(1 for x in timeline if x["type"] == "TRAP_TRIGGERED" and x["owner"] == me
                                     and x["actor"] == opp),
            "trap_hits": sum(1 for x in mine if x["type"] == "TRAP_TRIGGERED"),
            "inspections": sum(1 for x in mine if x["type"] == "OBJECT_INSPECTED"),
            "searches": sum(1 for x in mine if x["type"] == "OBJECT_SEARCHED"),
            "approaches": len(approach_ts), "avoids": sum(1 for d in ds_me if d["action"] == "avoid_opponent"),
            "fights": len(my_fights), "fights_won": sum(1 for f in my_fights if f["winner"] == me),
            "approach_fights": sum(1 for f in my_fights if any(f["t"] - APPROACH_FIGHT_WINDOW_S <= t <= f["t"]
                                                               for t in approach_ts)),
            "fight_times": [f["t"] for f in my_fights],
            "hearts_lost": sum(1 for x in mine if x["type"] == "LIFE_LOST"),
            "items_collected": collected, "rooms_visited": len(rooms),
            "decisions_per_min": round(60.0 * n_all / active, 3) if active > 0 else None,
            "decisions_after_terminal": sum(1 for d in ds_me if end_t is not None and d["t"] > end_t + 1e-6),
            "requests_after_terminal": terminal.get("requests_after_terminal"),
        })
    return out


def _phase_a_decision(row: dict, coverage: dict[str, str]) -> dict[str, Any]:
    return {"t": float(row.get("mission_time_used_s") or 0.0), "decision_index": row.get("decision_number"),
            "features": row["features"], "action": row["intended_action"], "target": row.get("intended_target"),
            "category": row.get("category"), "legal": list(row.get("legal_action_names") or []),
            "coverage": coverage.get(row["decision_id"]), "confidence": None, "probs": None, "valid": True,
            "decision_id": row["decision_id"]}


def load_phase_a(root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    """Phase-A spy records: CLEAN decisions + match telemetry, read only. Coverage per decision is the
    held-out coverage of milestone 2 (estimator fitted without that decision's seed)."""
    clean = ds.load(DATASET)
    replay = root / "workshop/policy_reports" / DATASET / "replay_table-v1.jsonl"
    coverage = {}
    if replay.is_file():
        for r in _jsonl(replay):
            if r.get("coverage"):
                coverage[r["decision_id"]] = r["coverage"]["coverage"]
    by_match: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in clean:
        by_match[(r["match_id"], r["actor_id"])].append(_phase_a_decision(r, coverage))
    spies = []
    for c in phase_a_configurations(root):
        run_dir = root / c["phase_a_run_dir"]
        records = _jsonl(run_dir / "telemetry_match01.jsonl")
        summary = json.loads((run_dir / "summary.json").read_text())
        counts = Counter(r.get("actor_id") for r in records if r.get("event_type") == "llm_decision")
        meta = {**c, "source": PHASE_A, "emulator": PHASE_A, "replicate": 0, "match_id": c["phase_a_match"],
                "run_dir": c["phase_a_run_dir"], "policies": {"black_spy": c["black_policy"], "white_spy": c["white_policy"]}}
        spies.extend(spy_records(records, summary, {a: by_match.get((c["phase_a_match"], a), []) for a in ACTORS}, meta,
                                 dict(counts)))
    return spies


def _surrogate_decision(e: dict, limit: float) -> dict[str, Any]:
    cov = e.get("coverage") or {}
    return {"t": float(e.get("mission_elapsed") if e.get("mission_elapsed") is not None
                       else limit - e["mission_time_remaining"]),
            "decision_index": e.get("decision_index") or e.get("decision_number"), "features": e.get("features") or {},
            "action": e.get("intended_action") or e.get("sampled_action"), "executed": e.get("sampled_action"),
            "target": e.get("intended_target", e.get("target_id")), "category": e.get("category"),
            "legal": list(e.get("legal_actions") or []), "coverage": e.get("coverage_label") or cov.get("coverage"),
            "coverage_score": e.get("coverage_score", cov.get("score")),
            "coverage_reasons": e.get("coverage_reasons", cov.get("reasons")), "confidence": e.get("confidence"),
            "probs": e.get("probs"), "valid": e.get("valid"), "unsupported": e.get("unsupported_masked") or [],
            "decision_id": e.get("decision_id"), "emulator": e.get("emulator"), "target_emulator": e.get("target_emulator"),
            "policy_rng_seed": e.get("policy_rng_seed"), "environment_seed": e.get("environment_seed")}


def load_closed_loop(session_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Closed-loop spy records and the per-match integrity records of one session."""
    manifest = json.loads((session_root / "manifest.json").read_text())
    spies, matches = [], []
    for match_id, rec in sorted(manifest.get("matches", {}).items()):
        run_dir = session_root / rec["rel_dir"]
        if rec.get("status") != "done" or not (run_dir / "match.json").is_file():
            matches.append({"match_id": match_id, "status": rec.get("status"), "rel_dir": rec["rel_dir"]})
            continue
        match = json.loads((run_dir / "match.json").read_text())
        run = json.loads((run_dir / "run.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        records = _jsonl(run_dir / "telemetry_match01.jsonl")
        entry = run["entry"]
        limit = float((summary.get("mission") or {}).get("time_limit_s") or 300.0)
        decisions = {a: [_surrogate_decision(e, limit) for e in records
                         if e.get("event_type") == "surrogate_decision" and e.get("actor_id") == a] for a in ACTORS}
        meta = {**entry, "source": entry["emulator"], "run_dir": str(run_dir.relative_to(PROJECT_ROOT))
                if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
                "policies": {"black_spy": entry["black_policy"], "white_spy": entry["white_policy"]}}
        spies.extend(spy_records(records, summary, decisions, meta))
        llm_events = sum(1 for e in records if e.get("event_type") in ("llm_decision", "provider_call"))
        raw = [e for e in records if e.get("event_type") == "surrogate_decision"]
        incomplete = sum(1 for e in raw if any(k not in e or (e[k] is None and k not in NULLABLE_FIELDS)
                                               for k in REQUIRED_DECISION_FIELDS))
        mismatched = sum(1 for e in raw if e.get("emulator") != entry["emulator"]
                         or e.get("target_emulator") != match.get("target_policy")
                         or e.get("policy_rng_seed") != entry["policy_rng_seed"] or e.get("environment_seed") != entry["seed"])
        matches.append({**match, "status": "done", "rel_dir": rec["rel_dir"], "config_id": entry["config_id"],
                        "replicate": entry["replicate"], "conditions": entry["conditions"],
                        "llm_or_provider_events": llm_events,
                        "configuration": {k: (run.get("configuration") or {}).get(k) for k in CONDITION_KEYS},
                        "decisions_missing_coverage": sum(1 for a in ACTORS for d in decisions[a] if not d["coverage"]),
                        "decisions_missing_confidence": sum(1 for a in ACTORS for d in decisions[a]
                                                            if not isinstance(d["confidence"], int | float)),
                        "surrogate_decisions": sum(len(v) for v in decisions.values()),
                        "decisions_missing_fields": incomplete, "decisions_mismatched_provenance": mismatched})
    return spies, matches


# -- small statistics --------------------------------------------------------------------------------

def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _sd(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.pstdev(xs), 4) if len(xs) > 1 else None


def _dist(values: list[Any]) -> dict[str, float]:
    c = Counter(str(v) for v in values)
    n = sum(c.values())
    return {k: round(v / n, 4) for k, v in sorted(c.items(), key=lambda x: -x[1])} if n else {}


def tvd(p: dict[str, float], q: dict[str, float]) -> float | None:
    if not p or not q:
        return None
    return round(0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q)), 4)


def jsd(p: dict[str, float], q: dict[str, float]) -> float | None:
    if not p or not q:
        return None
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a):
        return sum(a.get(k, 0.0) * math.log(a.get(k, 0.0) / m[k], 2) for k in keys if a.get(k, 0.0) > 0)
    return round(0.5 * kl(p) + 0.5 * kl(q), 4)


def rates(decisions: list[dict]) -> dict[str, dict[str, Any]]:
    """Opportunity-normalized rates over realised (intended) actions - the milestone-2 definitions."""
    out = {}
    for name, (cond, action) in OPPORTUNITIES.items():
        pool = [d for d in decisions if cond(d["features"])]
        k = sum(1 for d in pool if d["action"] == action)
        out[name] = {"n": len(pool), "k": k, "rate": round(k / len(pool), 4) if pool else None}
    return out


def _decisions(spies: list[dict]) -> list[dict]:
    return [d for s in spies for d in s["decisions"]]


def _by(spies: list[dict], key: str) -> dict[Any, list[dict]]:
    out: dict[Any, list[dict]] = defaultdict(list)
    for s in spies:
        out[s[key]].append(s)
    return dict(sorted(out.items(), key=lambda x: str(x[0])))


# -- the sections -----------------------------------------------------------------------------------

def opportunity_fidelity(groups: dict[str, list[dict]]) -> dict[str, Any]:
    ref = groups[PHASE_A]
    out: dict[str, Any] = {"pooled": {}, "per_identity": {}, "mae": {}, "per_personality": {}, "per_model": {}}
    for src, spies in groups.items():
        out["pooled"][src] = rates(_decisions(spies))
        out["per_identity"][src] = {k: rates(_decisions(v)) for k, v in _by(spies, "policy").items()}
        out["per_personality"][src] = {k: rates(_decisions(v)) for k, v in _by(spies, "personality").items()}
        out["per_model"][src] = {k: rates(_decisions(v)) for k, v in _by(spies, "model").items()}
    for src in groups:
        if src == PHASE_A:
            continue
        out["mae"][src] = opportunity_mae(out["per_identity"][src], out["per_identity"][PHASE_A])
    out["reference_identities"] = sorted(out["per_identity"][PHASE_A])
    return out


def opportunity_mae(src: dict[str, dict], ref: dict[str, dict]) -> dict[str, Any]:
    errors, by_opp, skipped = [], defaultdict(list), 0
    for ident, opps in ref.items():
        for name, r in opps.items():
            s = (src.get(ident) or {}).get(name) or {}
            if r["n"] >= MIN_OPPORTUNITIES and (s.get("n") or 0) >= MIN_OPPORTUNITIES:
                e = abs(s["rate"] - r["rate"])
                errors.append(e)
                by_opp[name].append(e)
            else:
                skipped += 1
    return {"mae": _mean(errors), "cells": len(errors), "cells_skipped": skipped,
            "by_opportunity": {k: _mean(v) for k, v in by_opp.items()},
            "max_error": round(max(errors), 4) if errors else None}


def state_visitation(groups: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {"dims": {}, "distance": {}, "per_identity_distance": {}}
    for dim, fn in VISITATION.items():
        out["dims"][dim] = {src: _dist([fn(d) for d in _decisions(spies)]) for src, spies in groups.items()}
    for src in groups:
        if src == PHASE_A:
            continue
        per_dim = {dim: tvd(v[src], v[PHASE_A]) for dim, v in out["dims"].items() if dim != "coverage"}
        vals = [x for x in per_dim.values() if x is not None]
        out["distance"][src] = {"per_dim_tvd": per_dim, "mean_tvd": _mean(vals),
                                "coverage_tvd": tvd(out["dims"]["coverage"][src], out["dims"]["coverage"][PHASE_A])}
        ref_by, src_by = _by(groups[PHASE_A], "policy"), _by(groups[src], "policy")
        per_ident = {}
        for ident in ref_by:
            vals = [tvd(_dist([fn(d) for d in _decisions(src_by.get(ident, []))]), _dist([fn(d) for d in _decisions(ref_by[ident])]))
                    for dim, fn in VISITATION.items() if dim != "coverage"]
            per_ident[ident] = _mean([v for v in vals if v is not None])
        out["per_identity_distance"][src] = per_ident
    out["rooms_visited"] = {src: {"mean": _mean([s["rooms_visited"] for s in spies]),
                                  "dist": _dist([s["rooms_visited"] for s in spies])} for src, spies in groups.items()}
    return out


def _metric_values(spies: list[dict], key: str, kind: str) -> list[float]:
    vals = [s[key] for s in spies]
    if kind == "rate":
        return [1.0 if v else 0.0 for v in vals]
    return [float(v) for v in vals if v is not None]


def trajectory(groups: dict[str, list[dict]], scale_spies: list[dict] | None = None) -> dict[str, Any]:
    """Per-metric means, and the per-identity |closed loop - Phase A| divided by the metric's standard
    deviation over all Phase-A spy records (`scale_spies`), averaged over identities."""
    scale_spies = scale_spies if scale_spies is not None else groups[PHASE_A]
    out: dict[str, Any] = {"metrics": {}, "standardized_mae": {}, "per_identity": {}, "scales": {}}
    for key, label, kind in TRAJECTORY:
        row = {}
        for src, spies in groups.items():
            vals = _metric_values(spies, key, kind)
            row[src] = {"mean": _mean(vals), "sd": _sd(vals), "n": len(vals),
                        **({"reached": f"{len(vals)}/{len(spies)}"} if kind == "time" else {})}
        out["metrics"][key] = {"label": label, "kind": kind, **row}
    ref_by = _by(groups[PHASE_A], "policy")
    for src, spies in groups.items():
        src_by = _by(spies, "policy")
        out["per_identity"][src] = {ident: {key: _mean(_metric_values(src_by.get(ident, []), key, kind))
                                            for key, _, kind in TRAJECTORY} for ident in ref_by}
        if src == PHASE_A:
            continue
        diffs = {}
        for key, _, kind in TRAJECTORY:
            scale = _sd(_metric_values(scale_spies, key, kind)) or None
            out["scales"][key] = scale
            per = []
            for ident in ref_by:
                a, b = out["per_identity"][src][ident][key], out["per_identity"][PHASE_A][ident][key]
                if a is not None and b is not None and scale:
                    per.append(abs(a - b) / scale)
            diffs[key] = _mean(per)
        out["standardized_mae"][src] = {"per_metric": diffs, "mean": _mean([v for v in diffs.values() if v is not None])}
    return out


def terminals(groups: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {"pooled": {}, "tvd": {}, "per_identity": {}, "per_identity_tvd": {}, "per_side": {}}
    for src, spies in groups.items():
        out["pooled"][src] = {lc: sum(1 for s in spies if s["lifecycle"] == lc) for lc in LIFECYCLES}
        out["pooled"][src]["other"] = sum(1 for s in spies if s["lifecycle"] not in LIFECYCLES)
        out["per_identity"][src] = {k: _dist([s["lifecycle"] for s in v]) for k, v in _by(spies, "policy").items()}
        out["per_side"][src] = {k: _dist([s["lifecycle"] for s in v]) for k, v in _by(spies, "side").items()}
    ref = _dist([s["lifecycle"] for s in groups[PHASE_A]])
    for src, spies in groups.items():
        if src == PHASE_A:
            continue
        out["tvd"][src] = tvd(_dist([s["lifecycle"] for s in spies]), ref)
        out["per_identity_tvd"][src] = {k: tvd(v, out["per_identity"][PHASE_A].get(k, {}))
                                        for k, v in out["per_identity"][src].items()}
        out["per_identity_tvd"][src]["mean"] = _mean([v for v in out["per_identity_tvd"][src].values() if v is not None])
    return out


def _third(d: dict, spy: dict) -> str:
    span = spy["active_mission_s"] or 0.0
    f = d["t"] / span if span > 0 else 0.0
    return "early" if f < 1 / 3 else "middle" if f < 2 / 3 else "late"


def coverage_drift(groups: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src, spies in groups.items():
        thirds: dict[str, list[str]] = {"early": [], "middle": [], "late": []}
        for s in spies:
            for d in s["decisions"]:
                thirds[_third(d, s)].append(d["coverage"] or "none")
        overall = [c for v in thirds.values() for c in v]
        low_ood = {k: round(sum(1 for c in v if c in ("LOW", "OOD")) / len(v), 4) if v else None for k, v in thirds.items()}
        out[src] = {"overall": _dist(overall), "by_third": {k: _dist(v) for k, v in thirds.items()},
                    "low_ood_share": round(sum(1 for c in overall if c in ("LOW", "OOD")) / len(overall), 4) if overall else None,
                    "ood_share": round(sum(1 for c in overall if c == "OOD") / len(overall), 4) if overall else None,
                    "low_ood_by_third": low_ood, "n_by_third": {k: len(v) for k, v in thirds.items()},
                    "drift_late_minus_early": round(low_ood["late"] - low_ood["early"], 4)
                    if low_ood["late"] is not None and low_ood["early"] is not None else None,
                    "per_identity_low_ood": {k: _share_low_ood(_decisions(v)) for k, v in _by(spies, "policy").items()}}
    return out


def _share_low_ood(decisions: list[dict]) -> float | None:
    return round(sum(1 for d in decisions if d["coverage"] in ("LOW", "OOD")) / len(decisions), 4) if decisions else None


def _band(c: float | None) -> str | None:
    if c is None:
        return None
    return "HIGH" if c >= HIGH_CONFIDENCE else "LOW" if c < LOW_CONFIDENCE else "MEDIUM"


def confidence_coverage(groups: dict[str, list[dict]]) -> dict[str, Any]:
    out = {}
    for src, spies in groups.items():
        if src == PHASE_A:
            continue
        ds_ = _decisions(spies)
        cells = Counter((_band(d["confidence"]), d["coverage"]) for d in ds_)
        quad = {"HIGH confidence + HIGH coverage": sum(v for (b, c), v in cells.items() if b == "HIGH" and c == "HIGH"),
                "HIGH confidence + LOW/OOD coverage": sum(v for (b, c), v in cells.items() if b == "HIGH" and c in ("LOW", "OOD")),
                "LOW confidence + HIGH coverage": sum(v for (b, c), v in cells.items() if b == "LOW" and c == "HIGH"),
                "LOW confidence + LOW/OOD coverage": sum(v for (b, c), v in cells.items() if b == "LOW" and c in ("LOW", "OOD"))}
        out[src] = {"n": len(ds_), "quadrants": quad,
                    "quadrant_rates": {k: round(v / len(ds_), 4) if ds_ else None for k, v in quad.items()},
                    "grid": {f"{b} confidence | {c}": v for (b, c), v in sorted(cells.items(), key=lambda x: (str(x[0][0]), str(x[0][1])))}}
    return out


def low_ood_stratification(groups: dict[str, list[dict]]) -> dict[str, Any]:
    out = {}
    for src, spies in groups.items():
        ds_ = _decisions(spies)
        strata = {"HIGH/MEDIUM": [d for d in ds_ if d["coverage"] in ("HIGH", "MEDIUM")],
                  "LOW/OOD": [d for d in ds_ if d["coverage"] in ("LOW", "OOD")]}
        out[src] = {name: {"n": len(v), "rates": rates(v), "actions": _dist([d["action"] for d in v]),
                           "mean_confidence": _mean([d["confidence"] for d in v])} for name, v in strata.items()}
        out[src]["action_tvd_between_strata"] = tvd(out[src]["HIGH/MEDIUM"]["actions"], out[src]["LOW/OOD"]["actions"])
    return out


def sequences(groups: dict[str, list[dict]]) -> dict[str, Any]:
    out = {}
    for src, spies in groups.items():
        counts = {k: [0, 0] for k in SEQUENCES}  # [pattern, opportunities]
        for s in spies:
            ds_ = s["decisions"]
            for a, b in zip(ds_, ds_[1:]):
                if a["category"] == "MOVE_PROGRESS" and a["action"] == "move":
                    counts["MOVE -> MOVE"][1] += 1
                    counts["MOVE -> MOVE"][0] += int(b["action"] == "move")
                if a["category"] == "INSPECT":
                    counts["INSPECT -> SEARCH"][1] += 1
                    counts["INSPECT -> SEARCH"][0] += int(b["category"] == "SEARCH")
                if a["category"] == "SEARCH":
                    counts["SEARCH -> PLACE_TRAP"][1] += 1
                    counts["SEARCH -> PLACE_TRAP"][0] += int(b["category"] == "PLACE_TRAP")
                if a["category"] == "AVOID":
                    counts["AVOID -> resume mission"][1] += 1
                    counts["AVOID -> resume mission"][0] += int(b["category"] in MISSION_CATEGORIES)
            for d in ds_:
                if d["action"] == "approach_opponent":
                    counts["APPROACH -> fight"][1] += 1
                    counts["APPROACH -> fight"][0] += int(any(d["t"] <= f <= d["t"] + APPROACH_FIGHT_WINDOW_S
                                                              for f in s["fight_times"]))
        out[src] = {k: {"k": v[0], "n": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None} for k, v in counts.items()}
    return out


KEY_METRICS = (("escaped", "rate"), ("eliminated", "rate"), ("n_decisions", "count"), ("traps_placed", "count"),
               ("inspections", "count"), ("searches", "count"), ("fights", "count"), ("hearts_lost", "count"),
               ("active_mission_s", "count"))


def stratify(groups: dict[str, list[dict]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ref_by = _by(groups[PHASE_A], key)
    for src, spies in groups.items():
        src_by = _by(spies, key)
        out[src] = {}
        for value in ref_by:
            members = src_by.get(value, [])
            row = {m: _mean(_metric_values(members, m, kind)) for m, kind in KEY_METRICS}
            row["spies"] = len(members)
            row["low_ood_share"] = _share_low_ood(_decisions(members))
            if src != PHASE_A:
                row["opportunity_mae"] = opportunity_mae({k: rates(_decisions(v)) for k, v in _by(members, "policy").items()},
                                                         {k: rates(_decisions(v)) for k, v in _by(ref_by[value], "policy").items()})["mae"]
                row["terminal_tvd"] = tvd(_dist([s["lifecycle"] for s in members]), _dist([s["lifecycle"] for s in ref_by[value]]))
            out[src][str(value)] = row
    return out


PAIRED_METRICS = (("match_mission_s", "mission duration (s)"), ("n_decisions", "decisions"),
                  ("traps_placed", "traps placed"), ("inspections", "inspections"), ("searches", "searches"),
                  ("fights", "fights"), ("low_ood", "LOW/OOD fraction"))


def _value(s: dict, key: str) -> float | None:
    if key == "low_ood":
        return _share_low_ood(s["decisions"])
    v = s.get(key)
    return float(v) if isinstance(v, int | float) else None


def paired(groups: dict[str, list[dict]], a: str = B1, b: str = B2) -> dict[str, Any]:
    if a not in groups or b not in groups:
        return {}
    ref = {(s["config_id"], s["actor_id"]): s for s in groups[PHASE_A]}
    idx_a = {(s["replicate"], s["config_id"], s["actor_id"]): s for s in groups[a]}
    idx_b = {(s["replicate"], s["config_id"], s["actor_id"]): s for s in groups[b]}
    keys = sorted(set(idx_a) & set(idx_b))
    out: dict[str, Any] = {"pairs": len(keys), "metrics": {}}
    for key, label in PAIRED_METRICS:
        diffs, closer_a, closer_b, ties = [], 0, 0, 0
        for k in keys:
            va, vb = _value(idx_a[k], key), _value(idx_b[k], key)
            if va is None or vb is None:
                continue
            diffs.append(vb - va)
            r = ref.get((k[1], k[2]))
            vr = _value(r, key) if r else None
            if vr is not None:
                da, db = abs(va - vr), abs(vb - vr)
                closer_a += int(da < db)
                closer_b += int(db < da)
                ties += int(da == db)
        n = len(diffs)
        sd = statistics.pstdev(diffs) if n > 1 else 0.0
        out["metrics"][key] = {"label": label, "n": n, "mean_diff_b_minus_a": _mean(diffs),
                               "se": round(sd / math.sqrt(n), 4) if n > 1 else None,
                               "b_higher": sum(1 for x in diffs if x > 0), "equal": sum(1 for x in diffs if x == 0),
                               "b_lower": sum(1 for x in diffs if x < 0),
                               "closer_to_phase_a": {SHORT[a]: closer_a, SHORT[b]: closer_b, "tie": ties}}
    same = sum(1 for k in keys if idx_a[k]["lifecycle"] == idx_b[k]["lifecycle"])
    out["terminal"] = {"same_lifecycle": same, "different": len(keys) - same,
                       "transitions": dict(Counter(f"{idx_a[k]['lifecycle']} -> {idx_b[k]['lifecycle']}"
                                                   for k in keys if idx_a[k]["lifecycle"] != idx_b[k]["lifecycle"]).most_common()),
                       "matches_phase_a": {SHORT[a]: sum(1 for k in keys if (ref.get((k[1], k[2])) or {}).get("lifecycle") == idx_a[k]["lifecycle"]),
                                           SHORT[b]: sum(1 for k in keys if (ref.get((k[1], k[2])) or {}).get("lifecycle") == idx_b[k]["lifecycle"])}}
    return out


def divergence_examples(groups: dict[str, list[dict]], limit: int = 6) -> dict[str, Any]:
    ref = {(s["config_id"], s["actor_id"]): s for s in groups[PHASE_A]}
    out: dict[str, Any] = {}
    for src, spies in groups.items():
        if src == PHASE_A:
            continue
        loops = []
        for s in spies:
            run, best, best_at = 1, 1, None
            for x, y in zip(s["decisions"], s["decisions"][1:]):
                run = run + 1 if (x["action"], x["target"]) == (y["action"], y["target"]) else 1
                if run > best:
                    best, best_at = run, (y["action"], y["target"], round(y["t"], 1))
            loops.append((best, s, best_at))
        loops.sort(key=lambda x: -x[0])
        mism = []
        for s in spies:
            r = ref.get((s["config_id"], s["actor_id"]))
            if r and r["lifecycle"] != s["lifecycle"]:
                mism.append({"config_id": s["config_id"], "side": s["side"], "policy": s["policy"],
                             "phase_a": r["lifecycle"], "closed_loop": s["lifecycle"],
                             "decisions": f"{r['n_decisions']} -> {s['n_decisions']}",
                             "hearts_lost": f"{r['hearts_lost']} -> {s['hearts_lost']}", "run_dir": s["run_dir"]})
        gaps = sorted(((abs(s["n_decisions"] - ref[(s["config_id"], s["actor_id"])]["n_decisions"]), s)
                       for s in spies if (s["config_id"], s["actor_id"]) in ref), key=lambda x: -x[0])
        hc = sorted((d | {"_spy": s} for s in spies for d in s["decisions"]
                     if d["coverage"] in ("LOW", "OOD") and (d["confidence"] or 0) >= HIGH_CONFIDENCE),
                    key=lambda d: (-(d["confidence"] or 0), d["coverage"] != "OOD"))
        out[src] = {
            "longest_repeats": [{"config_id": s["config_id"], "side": s["side"], "policy": s["policy"], "repeats": n,
                                 "decision": at, "run_dir": s["run_dir"]} for n, s, at in loops[:limit]],
            "terminal_mismatches": len(mism), "terminal_mismatch_examples": mism[:limit],
            "terminal_mismatch_transitions": dict(Counter(f"{m['phase_a']} -> {m['closed_loop']}" for m in mism).most_common()),
            "largest_decision_gaps": [{"config_id": s["config_id"], "side": s["side"], "policy": s["policy"],
                                       "phase_a": ref[(s["config_id"], s["actor_id"])]["n_decisions"],
                                       "closed_loop": s["n_decisions"], "lifecycle": s["lifecycle"]} for _, s in gaps[:limit]],
            "high_confidence_low_ood": len(hc),
            "high_confidence_low_ood_examples": [_brief(d) for d in _one_per_spy(hc)[:limit]],
        }
    return out


def _one_per_spy(decisions: list[dict]) -> list[dict]:
    """The first (most confident) example of each spy and match, so the examples are not one match."""
    seen, out = set(), []
    for d in decisions:
        key = (d["_spy"]["config_id"], d["_spy"]["actor_id"], d["_spy"]["replicate"])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _brief(d: dict) -> dict[str, Any]:
    f, s = d["features"], d["_spy"]
    return {"config_id": s["config_id"], "side": s["side"], "policy": s["policy"], "t": round(d["t"], 1),
            "action": d["action"], "target": d["target"], "confidence": d["confidence"], "coverage": d["coverage"],
            "reasons": (d.get("coverage_reasons") or [])[:3],
            "state": {k: f.get(k) for k in ("current_room", "opponent_alert_present", "health_band", "life_balance",
                                            "time_pressure", "mission_complete", "recent_action_1")}}


def integrity(matches: list[dict], spies: list[dict], plan_size: int | None = None) -> dict[str, Any]:
    done = [m for m in matches if m.get("status") == "done"]
    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for m in done:
        by_cfg[(m["replicate"], m["config_id"])].append(m)
    pairing = []
    for key, ms in sorted(by_cfg.items()):
        if len(ms) < 2:
            continue
        env = {m["environment_seed"] for m in ms}
        prng = {m["policy_rng_seed"] for m in ms}
        tgt = {m["target_policy"] for m in ms}
        cond = {json.dumps(m["configuration"], sort_keys=True) for m in ms}
        pairing.append({"replicate": key[0], "config_id": key[1], "emulators": sorted(m["emulator"] for m in ms),
                        "same_environment_seed": len(env) == 1, "same_policy_rng_seed": len(prng) == 1,
                        "same_target_policy": len(tgt) == 1, "same_conditions": len(cond) == 1})
    hashes = defaultdict(set)
    for m in done:
        for actor, a in (m.get("artifacts") or {}).items():
            hashes[(a["policy"], a["emulator"], "action")].add(a["emulator_artifact_hash"])
            hashes[(a["policy"], a["target"], "target")].add(a["target_artifact_hash"])
    sims = [m.get("game_time_elapsed") or 0 for m in done]
    walls = [m.get("wall_seconds") or 0 for m in done]
    return {
        "planned": plan_size, "done": len(done), "not_done": [m["match_id"] for m in matches if m.get("status") != "done"],
        "provider_calls": sum(int(m.get("provider_calls") or 0) for m in done),
        "llm_or_provider_events": sum(int(m.get("llm_or_provider_events") or 0) for m in done),
        "network_attempts": sum(len(m.get("network_attempts") or []) for m in done),
        "unexpected_writes": sorted({w for m in done for w in m.get("unexpected_writes") or []}),
        "stop_reasons": dict(Counter(m.get("stop_reason") for m in done)),
        "requests_after_terminal": sum(int(v or 0) for m in done for v in (m.get("requests_after_terminal") or {}).values()),
        "decisions_after_terminal": sum(s["decisions_after_terminal"] for s in spies),
        "surrogate_decisions": sum(int(m.get("surrogate_decisions") or 0) for m in done),
        "decisions_missing_coverage": sum(int(m.get("decisions_missing_coverage") or 0) for m in done),
        "decisions_missing_confidence": sum(int(m.get("decisions_missing_confidence") or 0) for m in done),
        "decisions_missing_fields": sum(int(m.get("decisions_missing_fields") or 0) for m in done),
        "decisions_mismatched_provenance": sum(int(m.get("decisions_mismatched_provenance") or 0) for m in done),
        "required_decision_fields": list(REQUIRED_DECISION_FIELDS),
        "pairing": {"pairs": len(pairing), "all_same_environment_seed": all(p["same_environment_seed"] for p in pairing),
                    "all_same_policy_rng_seed": all(p["same_policy_rng_seed"] for p in pairing),
                    "all_same_target_policy": all(p["same_target_policy"] for p in pairing),
                    "all_same_conditions": all(p["same_conditions"] for p in pairing)},
        "artifact_hashes_consistent": all(len(v) == 1 for v in hashes.values()),
        "target_policies": sorted({m.get("target_policy") for m in done}),
        "game_code_hashes": sorted({m.get("game_code_hash") for m in done}),
        "phase_a_game_versions": sorted({m.get("phase_a_game_version") for m in done if m.get("phase_a_game_version")}),
        "lifecycles": dict(Counter(v for m in done for v in (m.get("lifecycle") or {}).values())),
        "done_per_replicate": {str(k): v for k, v in sorted(Counter(m["replicate"] for m in done).items())},
        "simulated_seconds": round(sum(sims), 1), "wall_seconds": round(sum(walls), 1),
        "decisions": sum(int(v or 0) for m in done for v in (m.get("decisions") or {}).values()),
    }


def phase_a_reference(groups: dict[str, list[dict]]) -> dict[str, Any]:
    pa = groups[PHASE_A]
    return {"matches": len({s["config_id"] for s in pa}), "spies": len(pa),
            "clean_decisions": len(_decisions(pa)), "all_decisions": sum(s["n_decisions"] for s in pa),
            "per_identity": {k: {"spies": len(v), "clean_decisions": len(_decisions(v)),
                                 "escaped": sum(1 for s in v if s["escaped"])} for k, v in _by(pa, "policy").items()},
            "coverage_note": "Phase-A coverage labels are milestone 2's held-out labels (estimator fitted without "
                             "the decision's seed); closed-loop labels come from estimators fitted on all six seeds."}


DISTANCE_KEYS = ("opportunity-rate MAE", "state-visitation distance (mean TVD)", "trajectory-metric MAE (standardized)",
                 "terminal-distribution distance (TVD)", "coverage drift (LOW/OOD share)",
                 "coverage drift (LOW/OOD excess over Phase A)", "coverage drift (late - early LOW/OOD)")


def distance_vector(sections: dict[str, Any], emulators: list[str]) -> dict[str, Any]:
    out = {}
    for e in emulators:
        cov = sections["coverage"][e]
        out[e] = {"opportunity-rate MAE": sections["opportunity"]["mae"][e]["mae"],
                  "state-visitation distance (mean TVD)": sections["visitation"]["distance"][e]["mean_tvd"],
                  "trajectory-metric MAE (standardized)": sections["trajectory"]["standardized_mae"][e]["mean"],
                  "terminal-distribution distance (TVD)": sections["terminals"]["tvd"][e],
                  "coverage drift (LOW/OOD share)": cov["low_ood_share"],
                  "coverage drift (LOW/OOD excess over Phase A)": round(abs(cov["low_ood_share"] - sections["coverage"][PHASE_A]["low_ood_share"]), 4),
                  "coverage drift (late - early LOW/OOD)": cov["drift_late_minus_early"]}
    return out


def strata_counts(sections: dict[str, Any], emulators: list[str]) -> dict[str, Any]:
    """Criterion 4: in how many seed and side strata each emulator has the lower opportunity MAE."""
    a, b = SHORT[emulators[0]], SHORT[emulators[1]]
    counts = {a: 0, b: 0, "tie": 0}
    for key in ("seed", "side"):
        for value, row in sections["stratification"][key][emulators[0]].items():
            ra, rb = row.get("opportunity_mae"), sections["stratification"][key][emulators[1]].get(value, {}).get("opportunity_mae")
            if ra is None or rb is None:
                continue
            counts[a if ra < rb else b if rb < ra else "tie"] += 1
    counts["closer"] = a if counts[a] > counts[b] else b if counts[b] > counts[a] else "tie"
    return counts


def analyse(session_root: Path, *, root: Path = PROJECT_ROOT, plan_size: int | None = None,
            with_hybrid: bool = True, repro_roots: list[Path] | None = None) -> dict[str, Any]:
    phase_a = load_phase_a(root)
    cl_spies, matches = load_closed_loop(session_root)
    emulators = [e for e in (B1, B2, "tree-v1", "frequency-v1") if any(s["source"] == e for s in cl_spies)]
    configs_run = {s["config_id"] for s in cl_spies}
    groups = {PHASE_A: [s for s in phase_a if s["config_id"] in configs_run] if configs_run else phase_a}
    for e in emulators:
        groups[e] = [s for s in cl_spies if s["source"] == e]
    sections: dict[str, Any] = {
        "integrity": integrity(matches, cl_spies, plan_size),
        "phase_a": phase_a_reference(groups),
        "opportunity": opportunity_fidelity(groups),
        "visitation": state_visitation(groups),
        "coverage": coverage_drift(groups),
        "confidence_coverage": confidence_coverage(groups),
        "low_ood": low_ood_stratification(groups),
        "trajectory": trajectory(groups, phase_a),
        "terminals": terminals(groups),
        "sequences": sequences(groups),
        "stratification": {k: stratify(groups, k) for k in ("seed", "side", "personality", "model", "policy", "orientation")},
        "paired": paired(groups, *emulators[:2]) if len(emulators) >= 2 else {},
        "divergence": divergence_examples(groups),
    }
    sections["distance"] = distance_vector(sections, emulators)
    if len(emulators) >= 2 and groups[PHASE_A]:
        from workshop.policy_emulator import closed_loop_stability as cs
        stab = cs.replicate_stability(groups, emulators, phase_a)
        boot = cs.bootstrap_runs(groups, emulators, phase_a, mode="run")
        sections.update({
            "stability": stab, "bootstrap": boot,
            "bootstrap_configuration": cs.bootstrap_runs(groups, emulators, phase_a, mode="configuration"),
            "paired_runs": cs.paired_runs(groups, emulators), "per_replicate": cs.per_replicate(groups, emulators),
            "repetition": cs.repetition(groups, emulators),
            "opportunity_by": {k: cs.opportunity_by(groups, k) for k in ("replicate", "personality", "policy", "seed", "side")},
            "terminals_by": {k: cs.terminals_by(groups, k) for k in ("replicate", "personality", "policy", "seed", "side")},
            "strata": strata_counts(sections, emulators)})
        sections["recommendation"] = cs.decide(stab, boot, sections["strata"], emulators)
        sections["hybrid"] = cs.hybrid_readiness(groups, emulators) if with_hybrid else {}
    replicates = sorted({s["replicate"] for s in cl_spies})
    sections["replicates"] = replicates
    sections["reproducibility"] = [compare_sessions(session_root, r) for r in (repro_roots or [])]
    sections["emulators"] = emulators
    sections["configurations"] = sorted(configs_run)
    sections["analysis_version"] = ANALYSIS_VERSION
    return sections


def _trace(run_dir: Path) -> tuple[list, list]:
    records = _jsonl(run_dir / "telemetry_match01.jsonl")
    decisions = [(e["actor_id"], e.get("decision_index"), e.get("mission_time_remaining"), e.get("intended_action"),
                  e.get("intended_target"), e.get("trap_type")) for e in records if e.get("event_type") == "surrogate_decision"]
    engine = [(e.get("engine_event_type"), e.get("actor_id"), e.get("mission_time_remaining")) for e in records
              if e.get("event_type") == "engine_event"]
    return decisions, engine


def compare_sessions(root_a: Path, root_b: Path) -> dict[str, Any]:
    """Same configuration + same seeds in two sessions: identical surrogate decisions and engine events?"""
    ma = json.loads((root_a / "manifest.json").read_text()).get("matches", {})
    mb = json.loads((root_b / "manifest.json").read_text()).get("matches", {})
    rows = []
    for match_id in sorted(set(ma) & set(mb)):
        if ma[match_id].get("status") != "done" or mb[match_id].get("status") != "done":
            continue
        da, ea = _trace(root_a / ma[match_id]["rel_dir"])
        db, eb = _trace(root_b / mb[match_id]["rel_dir"])
        rows.append({"match_id": match_id, "decisions": len(da), "identical_decisions": da == db,
                     "identical_engine_events": ea == eb})
    return {"compared": len(rows), "identical": sum(1 for r in rows if r["identical_decisions"] and r["identical_engine_events"]),
            "matches": rows, "session": root_b.name}
