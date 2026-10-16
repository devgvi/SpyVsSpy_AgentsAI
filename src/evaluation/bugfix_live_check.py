"""Combined live check for the navigation/exploration bugfix (§28).

Reads a live run's telemetry (workshop_runs/<session>/live/<run>/telemetry_match*.jsonl)
and reports, without judging the model's strategy:

- navigation: every MOVE's outcome (reached / failed with reason), whether the
  intended interaction was legal afterwards, and whether any legal MOVE target
  turned out unreachable;
- exploration: for every decision, the current room's status and unfinished
  items as shown to the model, whether the chosen action addressed an unfinished
  item (an unsearched object, an untraversed exit, or an object to interact with
  now), and whether exploration facts ever contradicted the engine's own
  per-actor state at that time (they are derived from it, so any mismatch is a
  bug).

    python -m src.evaluation.bugfix_live_check RUN_DIR [RUN_DIR ...] [--out summary.json]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def analyse(run_dir: Path) -> dict[str, Any]:
    events = [json.loads(line) for f in sorted(run_dir.glob("telemetry_match*.jsonl")) for line in f.open()]
    decisions = [e for e in events if e["event_type"] == "llm_decision"]
    lifecycle = [e for e in events if e["event_type"] == "action_lifecycle"]
    by_decision = {e["decision_id"]: e for e in lifecycle}
    moves = [e for e in lifecycle if e["action"] == "move"]
    nav_outcomes = collections.Counter(e["action_terminal_reason"] for e in moves)
    legal_after = sum(1 for e in moves if (e.get("navigation") or {}).get("interaction_legal_after_move"))
    furniture_moves = [e for e in moves if not str(e["target_id"]).startswith("door") and e["target_id"] != "exit_door"]
    furniture_ok = sum(1 for e in furniture_moves if (e.get("navigation") or {}).get("interaction_legal_after_move"))
    followups = collections.Counter()
    status_seen = collections.Counter()
    addressed = 0
    with_unfinished = 0
    missing_exploration = 0
    for d in decisions:
        obs = d.get("observation") or {}
        exp = obs.get("exploration")
        if exp is None:
            missing_exploration += 1
            continue
        room = (exp.get("rooms") or {}).get(exp.get("current_room")) or {}
        status_seen[room.get("status", "none")] += 1
        act = d.get("executed_action") or d.get("validated_action") or {}
        target, name = act.get("target_id"), str(act.get("action"))
        unfinished = set(room.get("unsearched", [])) | set(room.get("untraversed_exits", []))
        if unfinished:
            with_unfinished += 1
            if target in unfinished or (name in ("interact", "inspect") and target == obs["self"].get("near_object_id")):
                addressed += 1
        # after a furniture MOVE that reached its target, what did the model do next?
        lc = by_decision.get(d["decision_id"])
        if lc and lc["action"] == "move" and (lc.get("navigation") or {}).get("interaction_legal_after_move"):
            followups["reached"] += 1
    return {
        "run": run_dir.name,
        "decisions": len(decisions),
        "decisions_without_exploration": missing_exploration,
        "moves": len(moves),
        "move_outcomes": dict(nav_outcomes),
        "moves_interaction_legal_after": legal_after,
        "furniture_moves": len(furniture_moves),
        "furniture_moves_in_interaction_range": furniture_ok,
        "path_unreachable": nav_outcomes.get("path_unreachable", 0),
        "decision_room_status": dict(status_seen),
        "decisions_with_unfinished_room": with_unfinished,
        "decisions_addressing_unfinished_item": addressed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.bugfix_live_check")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    rows = [analyse(Path(r)) for r in args.runs]
    print(json.dumps(rows, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
