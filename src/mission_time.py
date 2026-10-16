"""Time as a first-class agent fact (bugfix mission time budget V2, §12-23, §31E-F).

Built from the actor's own Observation only, for the live payload:

* `time_budget`: limit, used, remaining (this actor's own clock), the
  required items still missing, the exit (known? walking ETA over rooms this
  actor has visited - never a route through rooms it has not seen), the
  minimum known time to finish and exit, the slack left over, and a
  deterministic urgency band;
* `action_costs`: the real engine time cost of every legal action
  (src/action_costs.py) and the walking ETA of MOVE targets in known rooms.

Minimum known required time (a lower bound, not a prediction): walking to
the exit + opening it + for every required item not yet held, one search
plus the pickup (its location may be unknown, so its walk is not counted).

Urgency (documented, derived from slack relative to that known work):
    CRITICAL  slack < max(10 s, 0.25 x required)  - one detour could lose the mission
    TIGHT     slack < max(30 s, 1.0 x required)   - no room for the known work twice
    RELAXED   otherwise
When the exit ETA is not known (route through unexplored rooms), the band
falls back to the share of the limit left: RELAXED >= 50 %, TIGHT >= 20 %,
else CRITICAL.

Also here: the neutral facts caption the overlay shows next to (never
instead of) the model's own decision summary, and the check that flags a
summary whose time claims contradict the facts (RATIONALE_FACT_MISMATCH).
"""
from __future__ import annotations

import math
import re
from typing import Any

from src import action_costs, constants

RELAXED, TIGHT, CRITICAL = "RELAXED", "TIGHT", "CRITICAL"
URGENCY_BANDS = (RELAXED, TIGHT, CRITICAL)
CRITICAL_MIN_S, CRITICAL_FACTOR = 10.0, 0.25
TIGHT_MIN_S, TIGHT_FACTOR = 30.0, 1.0
EXIT_ID = "exit_door"


def _path_length(start: tuple[float, float], path: list[tuple[float, float]]) -> float:
    total, (px, py) = 0.0, start
    for x, y in path:
        total += math.hypot(x - px, y - py)
        px, py = x, y
    return total


def _known_route(observation: Any, path: list[tuple[float, float]], also: tuple = ()) -> bool:
    """Every point of the path lies in a room this actor has visited (or in a
    doorway); `also`: rooms that may be entered anyway - the far side of the
    very door being walked through, which the actor sees from its doorway."""
    room_at = getattr(observation, "room_at", None)
    if room_at is None:
        return False
    known = set(observation.known_rooms) | ({observation.current_room} if observation.current_room else set()) | set(also)
    return all((room_at(x, y) in known) or room_at(x, y) is None for x, y in path)


def walk_eta(observation: Any, target_id: str) -> float | None:
    """Walking seconds to where a MOVE_TO(target) would end, over known rooms only."""
    from src.agents.action_lifecycle import choose_approach_point

    target = observation.get(target_id)
    if target is None:
        return None
    if target.type == "door" and target.crossing_waypoint is not None:
        goal = target.crossing_waypoint
    else:
        goal = choose_approach_point(observation, target)
    if goal is None:
        return None
    if math.dist(goal, observation.position) < 1.0:
        return 0.0
    path = observation.nav_grid.find_path(observation.position, goal)
    also: tuple = ()
    if target.type == "door" and target.connects:
        known = set(observation.known_rooms) | {observation.current_room}
        if known & set(target.connects):  # a door of a known room: its far side may be entered
            also = tuple(r for r in target.connects if r not in known)
    if not path or not _known_route(observation, path, also):
        return None
    return round(_path_length(observation.position, path) / constants.PLAYER_SPEED, 1)


LIVE_TIGHT_SHARE = 0.5  # live: half the mission gone -> at least TIGHT
LIVE_CRITICAL_SHARE = 0.25  # live: the last quarter -> CRITICAL
_ORDER = {"RELAXED": 0, "TIGHT": 1, "CRITICAL": 2}


def urgency_for(remaining: float, required: float | None, limit: float, share_floor: bool = False) -> str:
    """`share_floor` (live game): the known required time is only a lower
    bound (unexplored rooms, unfound items), so it alone said RELAXED all
    match in live run 20261001-114245 (480 s, 115 decisions, all RELAXED).
    The share of the mission still left sets a floor: TIGHT once half is
    gone, CRITICAL in the last quarter."""
    share = remaining / limit if limit else 0.0
    if required is None:
        return RELAXED if share >= 0.5 else TIGHT if share >= 0.2 else CRITICAL
    slack = remaining - required
    if slack < max(CRITICAL_MIN_S, CRITICAL_FACTOR * required):
        level = CRITICAL
    elif slack < max(TIGHT_MIN_S, TIGHT_FACTOR * required):
        level = TIGHT
    else:
        level = RELAXED
    if share_floor:
        floor = CRITICAL if share < LIVE_CRITICAL_SHARE else TIGHT if share < LIVE_TIGHT_SHARE else RELAXED
        level = max(level, floor, key=_ORDER.__getitem__)
    return level


def time_budget_facts(observation: Any) -> dict[str, Any]:
    limit = float(observation.mission_time_limit)
    remaining = max(0.0, float(observation.mission_time_remaining))
    missing = [i for i in observation.required_items if i not in observation.inventory]
    exit_obj = observation.get(EXIT_ID)
    exit_eta = walk_eta(observation, EXIT_ID) if exit_obj is not None else None
    required = None
    if exit_eta is not None:
        required = round(exit_eta + action_costs.interact_cost(exit_obj)
                         + len(missing) * (constants.TIME_COST_SEARCH + constants.TIME_COST_PICKUP), 1)
    slack = None if required is None else round(remaining - required, 1)
    return {
        "clock": observation.mission_clock_mode,
        "time_limit_s": int(round(limit)),
        "time_used_s": int(round(limit - remaining)),
        "time_remaining_s": int(round(remaining)),
        "required_items_missing": missing,
        "exit": {"known": exit_obj is not None, "eta_s": exit_eta,
                 "route": "known" if exit_eta is not None else "through rooms you have not explored"},
        "minimum_known_required_time_s": required,
        "slack_s": slack,
        "urgency": urgency_for(remaining, required, limit, share_floor=True),
        "note": "a lower bound: finding a missing item may need more searching and walking",
    }


def action_cost_facts(observation: Any, legal_actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Engine time costs for the legal actions (seconds of mission time)."""
    costs: dict[str, Any] = {}
    for entry in legal_actions:
        action, target_id = entry.get("action"), entry.get("target_id")
        obj = observation.get(target_id) if target_id else None
        if action == "inspect" and obj is not None:
            costs.setdefault("inspect", {})[target_id] = action_costs.inspect_cost(observation)
        elif action == "interact" and obj is not None:
            costs.setdefault("interact", {})[target_id] = action_costs.interact_cost(obj)
        elif action == "place_trap":
            held = sorted(t for t, n in observation.trap_inventory.items() if n > 0)
            costs.setdefault("place_trap", {})[target_id] = {t: action_costs.place_trap_cost(t) for t in held}
        elif action == "disarm":
            costs.setdefault("disarm", {})[target_id] = action_costs.disarm_cost()
            if obj is not None and getattr(obj, "trap_is_mine", False):
                costs.setdefault("disarm_only_removes_your_own_trap", []).append(target_id)
            else:
                chances = getattr(observation, "disarm_success_probability", None)
                trap_type = getattr(obj, "trap_type", None) if obj is not None else None
                if chances and trap_type in chances:
                    # an enemy trap you detected: the chance this disarm works; on failure it goes off on you
                    costs.setdefault("disarm_success_chance", {})[target_id] = chances[trap_type]
        elif action == "send_message":
            costs["send_message"] = action_costs.send_message_cost()
        elif action == "move" and target_id:
            costs.setdefault("move_eta_s", {})[target_id] = walk_eta(observation, target_id)
        elif action in ("approach_opponent", "avoid_opponent"):
            loser = (f"loses one life and is knocked out {constants.FIGHT_KO_SECONDS:g} s, then respawns elsewhere"
                     if getattr(observation, "fight_ko_respawn", False)
                     else f"stunned {constants.FIGHT_LOSER_STUN_SECONDS:g} s")
            costs[action] = f"walking time only; contact may start a 50/50 fight, the loser is {loser}"
    if "interact" in costs:
        costs["search_item_pickup_extra_s"] = action_costs.PICKUP_EXTRA
    costs["note"] = "seconds of your mission time; move_eta_s null = route through rooms you have not explored"
    return costs


def annotate_live_payload(observation: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Adds `time_budget` and `action_costs` (live payloads, at request time)."""
    out = dict(payload)
    out["time_budget"] = time_budget_facts(observation)
    out["action_costs"] = action_cost_facts(observation, payload.get("legal_actions") or [])
    return out


def chosen_action_cost(action_dict: dict[str, Any], payload: dict[str, Any] | None) -> float | None:
    """The time cost the payload quoted for the action that was chosen."""
    costs = (payload or {}).get("action_costs") or {}
    action, target = action_dict.get("action"), action_dict.get("target_id")
    if action == "move":
        return (costs.get("move_eta_s") or {}).get(target)
    if action == "place_trap":
        return ((costs.get("place_trap") or {}).get(target) or {}).get(action_dict.get("trap_type"))
    if action in ("inspect", "interact", "disarm"):
        return (costs.get(action) or {}).get(target)
    if action == "send_message":
        return costs.get("send_message")
    return None


# -- decision summary (the model's own short `reason`) -----------------------------

SUMMARY_MAX_CHARS = 180
# numbers are whole tokens (never the "5" of "1.5 s"), directly tied to the word
_NUM = r"(?<![\d.])(-?\d{1,4}(?:\.\d+)?)"
_MMSS = re.compile(r"(?<![\d.:])(\d{1,2}):(\d{2})\b")
_SECONDS_LEFT = re.compile(_NUM + r"\s*(?:s|sec|secs|seconds)?\s+(?:left|remain|remaining)\b", re.I)
_SLACK = re.compile(r"slack(?:\s+(?:of|is|=|:))?\s*\+?" + _NUM + r"|" + _NUM + r"\s*(?:s|sec|seconds)?\s+(?:of\s+)?slack",
                    re.I)


def rationale_fact_check(summary: str | None, facts: dict[str, Any] | None) -> list[str]:
    """RATIONALE_FACT_MISMATCH details when the summary's time claims contradict
    the structured facts (tolerance: max(5 s, 10 %)). Empty = no contradiction
    found (or nothing checkable). The summary itself is never rewritten."""
    if not summary or not facts:
        return []
    problems = []

    def off(claimed: float, actual: float) -> bool:
        return abs(claimed - actual) > max(5.0, 0.1 * abs(actual))

    remaining = facts.get("time_remaining_s")
    claims = [int(m) * 60 + int(s) for m, s in _MMSS.findall(summary)]
    claims += [float(n) for n in _SECONDS_LEFT.findall(summary)]
    if remaining is not None:
        for claimed in claims:
            if off(claimed, remaining):
                problems.append(f"time_remaining claimed {claimed:g}s, actual {remaining}s")
    slack = facts.get("slack_s")
    if slack is not None:
        for a, b in _SLACK.findall(summary):
            claimed = float(a or b)
            if off(claimed, slack):
                problems.append(f"slack claimed {claimed:g}s, actual {slack}s")
    return problems


def facts_caption(facts: dict[str, Any] | None, action_cost: float | None = None) -> str:
    """Neutral system caption from structured facts - never presented as the agent's thought."""
    if not facts:
        return ""
    parts = [f"{facts.get('time_remaining_s')} s left"]
    if facts.get("slack_s") is not None:
        parts.append(f"slack {facts['slack_s']:+.0f} s")
    parts.append(str(facts.get("urgency", "")).lower())
    if action_cost is not None:
        parts.append(f"action {action_cost:g} s")
    return " · ".join(p for p in parts if p)
