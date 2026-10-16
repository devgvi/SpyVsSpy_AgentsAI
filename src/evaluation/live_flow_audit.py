"""Live-flow / inspection / trap audit for one or more live runs
(BUGFIX_LIVE_FLOW_INSPECTION_AND_TRAP_BEHAVIOR.md §17, §24, §27-28, §50).

Reads telemetry_match*.jsonl (+ summary.json) of workshop live runs. Reports:

- flow: share of wall time the whole world was frozen (both spies stopped
  and the clock not advancing), approximated from the monitor's busy-step
  share and the decision latencies; per decision, whether the other spy
  could move while this one was thinking (never, under the evaluation barrier);
- inspection: for every INSPECT / INTERACT / PLACE_TRAP / DISARM decision,
  the engine distance from the actor's feet to the target's collision rect
  and whether the target was the engine's near object;
- traps: per decision T0 NO_TRAP_INVENTORY / T1 TRAP_ACTION_NOT_LEGAL /
  T2 TRAP_ACTION_LEGAL_BUT_NOT_SELECTED / T3 TRAP_SELECTED_BUT_FAILED /
  PLACED, plus the personality the payload carried.

    python -m src.evaluation.live_flow_audit RUN_DIR [...] [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any

from src import constants
from src.level import load_level

INTERACT_ACTIONS = ("inspect", "interact", "place_trap", "disarm")


def _rects() -> dict[str, Any]:
    level = load_level(constants.LEVEL_PATH)
    return {o.id: o.collision_rect for o in level.objects}


def _dist(x: float, y: float, rect) -> float:
    cx = max(rect.left, min(x, rect.right))
    cy = max(rect.top, min(y, rect.bottom))
    return math.hypot(x - cx, y - cy)


def analyse(run_dir: Path) -> dict[str, Any]:
    rects = _rects()
    events = [json.loads(line) for f in sorted(run_dir.glob("telemetry_match*.jsonl")) for line in f.open()]
    decisions = [e for e in events if e["event_type"] == "llm_decision"]
    lifecycle = {e["decision_id"]: e for e in events if e["event_type"] == "action_lifecycle"}
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}

    # -- inspection distances
    distances: dict[str, list[float]] = collections.defaultdict(list)
    near_mismatch = 0
    for d in decisions:
        act = d.get("executed_action") or {}
        name, target = act.get("action"), act.get("target_id")
        if name not in INTERACT_ACTIONS or target not in rects:
            continue
        x, y = d["observation"]["self"]["position"]
        distances[name].append(round(_dist(x, y, rects[target]), 1))
        if d["observation"]["self"].get("near_object_id") != target:
            near_mismatch += 1
    move_end = [e["navigation"]["distance_to_target_after_move"] for e in lifecycle.values()
                if e.get("action") == "move" and e.get("navigation") and e["navigation"].get("distance_to_target_after_move") is not None
                and not str(e["target_id"]).startswith("door") and e["target_id"] != "exit_door"]

    # -- traps
    cases = collections.Counter()
    personalities = collections.Counter()
    per_actor_traps: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    trap_records = []
    for d in decisions:
        obs = d["observation"]
        inv = sum((obs.get("self", {}).get("trap_inventory") or {}).values()) if isinstance(obs.get("self", {}).get("trap_inventory"), dict) else None
        legal = d.get("legal_actions") or obs.get("legal_actions") or []
        trap_legal = any(a.get("action") == "place_trap" for a in legal)
        act = d.get("executed_action") or d.get("validated_action") or {}
        chose = (d.get("initial_proposal") or {}).get("action") == "place_trap" or act.get("action") == "place_trap"
        personalities[str(d.get("personality_profile"))] += 1
        if chose and act.get("action") == "place_trap" and d.get("valid"):
            case = "PLACED"
        elif chose:
            case = "T3_TRAP_SELECTED_BUT_FAILED"
        elif inv == 0:
            case = "T0_NO_TRAP_INVENTORY"
        elif not trap_legal:
            case = "T1_TRAP_ACTION_NOT_LEGAL"
        else:
            case = "T2_TRAP_ACTION_LEGAL_BUT_NOT_SELECTED"
        cases[case] += 1
        per_actor_traps[d.get("actor_id")][case] += 1
        # §27: one record per decision
        inventory = obs.get("self", {}).get("trap_inventory")
        exploration = (obs.get("exploration") or {})
        room = obs.get("self", {}).get("room")
        trap_records.append({
            "actor_id": d.get("actor_id"), "decision_id": d.get("decision_id"),
            "trap_inventory_count": inv, "trap_inventory_in_payload": isinstance(inventory, dict),
            "place_trap_legal": trap_legal,
            "legal_trap_types": sorted(t for t, n in inventory.items() if n > 0) if isinstance(inventory, dict) else None,
            "legal_trap_targets": sorted(a.get("target_id") for a in legal if a.get("action") == "place_trap"),
            "personality": d.get("personality_profile"), "room": room,
            "room_exploration_status": ((exploration.get("rooms") or {}).get(room) or {}).get("status"),
            "place_trap_selected": chose, "repair_used": bool(d.get("repair_attempted")),
            "fallback_used": bool(d.get("fallback_used")), "case": case,
        })

    # -- flow
    stab = (summary.get("stability") or [{}])[0]
    steps, busy = stab.get("steps"), stab.get("busy_steps")
    live_commits = [e for e in events if e["event_type"] == "live_async_commit"]
    timing = stab.get("decision_timing") or summary.get("decision_timing") or ("live_async" if live_commits else "evaluation")
    # evaluation: every frame with a call outstanding is frozen (clock + both
    # spies); live_async: only frames where the clock did not move although
    # a spy was thinking (should be 0)
    frozen = busy if timing == "evaluation" else stab.get("steps_frozen_while_thinking")
    latencies = [d["latency_ms"] for d in decisions if isinstance(d.get("latency_ms"), (int, float))]

    def stats(values):
        return None if not values else {"n": len(values), "median": statistics.median(values), "max": max(values),
                                        "share_over_70": round(sum(v > 70 for v in values) / len(values), 3),
                                        "share_over_40": round(sum(v > 40 for v in values) / len(values), 3)}

    return {
        "run": str(run_dir),
        "commit_policy": summary.get("commit_policy"),
        "decision_timing": timing,
        "decisions": len(decisions),
        "flow": {
            "frames": steps, "frames_with_a_model_call_outstanding": busy,
            "frames_world_frozen": frozen,
            "share_of_frames_world_frozen": round(frozen / steps, 3) if steps and frozen is not None else None,
            "live_async_commits": dict(collections.Counter(e["stale_classification"] for e in live_commits)),
            "live_async_discarded_stale": sum(1 for e in live_commits if e.get("discarded_due_to_stale")),
            "decisions_where_opponent_moved_during_request": sum(1 for e in live_commits if e.get("opponent_moved_during_request")),
            "duplicate_requests": sum(a.get("duplicate_requests", 0) for a in (stab.get("actors") or {}).values()),
            "unused_responses": sum(a.get("unused_responses", 0) for a in (stab.get("actors") or {}).values()),
            "wall_seconds": summary.get("wall_seconds"), "game_seconds": summary.get("game_time_elapsed"),
            "game_seconds_per_wall_second": round(summary["game_time_elapsed"] / summary["wall_seconds"], 3) if summary.get("wall_seconds") else None,
            "median_decision_latency_ms": statistics.median(latencies) if latencies else None,
        },
        "inspection": {
            "distance_at_interaction_px": {k: stats(v) for k, v in distances.items()},
            "interactions_with_target_not_near_object": near_mismatch,
            "furniture_move_end_distance_px": stats(move_end),
            "engine_interaction_distance_px": constants.INTERACTION_DISTANCE,
        },
        "encounters": encounter_analysis(events, decisions, lifecycle),
        "mission_time": mission_time_analysis(events, decisions, summary),
        "traps": {"cases": dict(cases), "personality_in_payload": dict(personalities),
                  "cases_by_actor": {a: dict(c) for a, c in sorted(per_actor_traps.items())},
                  "personality_by_actor": {a: v.get("personality") for a, v in (summary.get("actors") or {}).items()},
                  "engine_trap_events": summary.get("engine_events"),
                  "records": trap_records},
    }


ENCOUNTER_ACTIONS = ("approach_opponent", "avoid_opponent")
OPTIONAL_ACTIONS = ("inspect", "place_trap", "disarm", "approach_opponent", "avoid_opponent", "send_message")


def _action_cost(act: dict, decision: dict | None = None) -> float:
    name = act.get("action")
    quoted = (((decision or {}).get("observation") or {}).get("action_costs") or {}).get(name)
    if isinstance(quoted, dict) and isinstance(quoted.get(act.get("target_id")), (int, float)):
        return float(quoted[act.get("target_id")])  # the cost the engine quoted for this run's rules
    if name == "inspect":
        return constants.TIME_COST_INSPECT
    if name == "place_trap":
        return constants.TRAP_PLACEMENT_COST.get(act.get("trap_type"), 0.0)
    if name == "disarm":
        return constants.TIME_COST_DISARM
    if name == "send_message":
        return constants.TIME_COST_SEND_MESSAGE
    return 0.0


def mission_time_analysis(events: list[dict], decisions: list[dict], summary: dict) -> dict[str, Any]:
    """Mission time budget V2 §32/§38-40: per spy, optional actions chosen and
    the mission time they cost, the mission outcome and the time left at the
    exit; and, where the decisions carried time facts, the optional-action
    share per urgency band."""
    per_actor: dict[str, dict] = {}
    for d in decisions:
        act = d.get("executed_action") or {}
        name = act.get("action") or "wait"
        a = per_actor.setdefault(d.get("actor_id"), {"decisions": 0, "optional_actions": 0, "optional_time_s": 0.0,
                                                     "by_urgency": {}})
        a["decisions"] += 1
        if isinstance(d.get("latency_ms"), (int, float)):
            a["model_wait_s"] = round(a.get("model_wait_s", 0.0) + d["latency_ms"] / 1000.0, 1)
        optional = name in OPTIONAL_ACTIONS
        if optional:
            a["optional_actions"] += 1
            a["optional_time_s"] = round(a["optional_time_s"] + _action_cost(act, d), 1)
        urgency = ((d.get("observation") or {}).get("time_budget") or {}).get("urgency")
        if urgency:
            band = a["by_urgency"].setdefault(urgency, {"decisions": 0, "optional": 0, "exit_moves": 0, "summaries": 0})
            band["decisions"] += 1
            band["optional"] += int(optional)
            band["exit_moves"] += int(name == "move" and act.get("target_id") == "exit_door")
            band["summaries"] += int(bool(act.get("reason")))
    # re-checked with the current checker (the run-time flag is kept separately)
    from src.mission_time import rationale_fact_check
    budgets = {d.get("decision_id"): (d.get("observation") or {}).get("time_budget") for d in decisions}
    commits = [e for e in events if e.get("event_type") == "live_async_commit" and e.get("decision_summary")]
    mismatch_details = [
        {"actor_id": e.get("actor_id"), "summary": e["decision_summary"], "detail": problems}
        for e in commits if (problems := rationale_fact_check(e["decision_summary"], budgets.get(e.get("decision_id"))))
    ]
    mismatches = len(mismatch_details)
    for actor_id, a in per_actor.items():
        info = (summary.get("actors") or {}).get(actor_id) or {}
        a["personality"] = info.get("personality")
        a["outcome"] = info.get("outcome")
        a["time_used_s"] = info.get("time_used_s")
        a["time_remaining_s"] = info.get("time_remaining_s")
        a["optional_share"] = round(a["optional_actions"] / a["decisions"], 3) if a["decisions"] else None
    return {"mission": summary.get("mission"), "by_actor": per_actor, "decision_summaries": len(commits),
            "rationale_fact_mismatches": mismatches, "rationale_fact_mismatch_examples": mismatch_details[:10],
            "rationale_fact_mismatch_flag_at_runtime": sum(
                1 for e in events if e.get("event_type") == "live_async_commit" and e.get("rationale_fact_mismatch"))}


def _chosen(d: dict) -> str | None:
    act = (d.get("executed_action") or d.get("validated_action") or {}).get("action")
    return act or (d.get("initial_proposal") or {}).get("action")


def encounter_analysis(events: list[dict], decisions: list[dict], lifecycle: dict[str, dict]) -> dict[str, Any]:
    """V2 Part D §65: E0-E5 per decision made while the opponent was visible.

    E0 OPPONENT_NEVER_VISIBLE (run level) / E1 VISIBLE_NO_CAPABILITY /
    E2 CHOSE_IGNORE (any other action) / E3 APPROACH_NAVIGATION_FAILED /
    E4 CONTACT_FIGHT (a fight followed an APPROACH) / E5 AVOID_SEPARATED;
    plus APPROACH_NO_CONTACT (lost sight / time limit) and
    AVOID_NOT_SEPARATED, so every approach/avoid is accounted for."""
    fights = [e for e in events if e.get("event_type") == "spy_fight"]
    fight_times = [f.get("game_time") for f in fights]
    cases = collections.Counter()
    per_actor: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for d in decisions:
        obs = d.get("observation") or {}
        facts = obs.get("encounter")
        visible = facts.get("opponent_visible") if facts else (obs.get("opponent") or {}).get("visible")
        if not visible:
            continue
        actor = d.get("actor_id")
        per_actor[actor]["visible_decisions"] += 1
        legal = d.get("legal_actions") or []
        capable = any(a.get("action") in ENCOUNTER_ACTIONS for a in legal)
        chosen = _chosen(d)
        per_actor[actor][{"approach_opponent": "APPROACH", "avoid_opponent": "AVOID"}.get(chosen, "IGNORE")] += 1
        if not capable:
            case = "E1_VISIBLE_NO_CAPABILITY"
        elif chosen not in ENCOUNTER_ACTIONS:
            case = "E2_CHOSE_IGNORE"
        else:
            lc = lifecycle.get(d.get("decision_id")) or {}
            reason = lc.get("action_terminal_reason")
            nav = lc.get("navigation") or {}
            if chosen == "approach_opponent":
                if lc.get("action_status") == "failed":
                    case = "E3_APPROACH_NAVIGATION_FAILED"
                elif reason == "target_reached" and any(
                        t is not None and abs((constants.MISSION_TIME - lc.get("action_completed_game_time", -99)) - t) <= 0.5
                        for t in fight_times):
                    case = "E4_CONTACT_FIGHT"
                else:
                    case = f"APPROACH_NO_CONTACT:{reason}"
            else:
                last = (obs.get("opponent") or {}).get("last_seen") or {}
                opp = last.get("position")
                start, final = nav.get("start_position"), nav.get("final_position")
                if opp and start and final and math.dist(final, opp) > math.dist(start, opp):
                    case = "E5_AVOID_SEPARATED"
                else:
                    case = f"AVOID_NOT_SEPARATED:{reason}"
        cases[case] += 1
    commits = [e for e in events if e.get("event_type") == "live_async_commit"]
    invalidated = 0
    for c in commits:
        if not c.get("discarded_due_to_stale") or c.get("commit_game_time") is None:
            continue
        elapsed = constants.MISSION_TIME - c["commit_game_time"]
        if any(t is not None and 0 <= elapsed - t <= constants.FIGHT_LOSER_STUN_SECONDS + 1 for t in fight_times):
            invalidated += 1
    return {
        "run_case": "E0_OPPONENT_NEVER_VISIBLE" if not per_actor else "OPPONENT_VISIBLE",
        "cases": dict(cases),
        "by_actor": {a: dict(c) for a, c in sorted(per_actor.items())},
        "fights": len(fights),
        "wins": dict(collections.Counter(f.get("winner_id") for f in fights)),
        "fight_rolls": [f.get("rng_roll") for f in fights],
        "pending_decisions_invalidated_after_fight": invalidated,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.live_flow_audit")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    rows = [analyse(Path(r)) for r in args.runs]
    brief = [{**r, "traps": {k: v for k, v in r["traps"].items() if k != "records"}} for r in rows]
    print(json.dumps(brief, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
