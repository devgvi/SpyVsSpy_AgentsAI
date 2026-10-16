"""Encounter intents: APPROACH_OPPONENT / AVOID_OPPONENT (bugfix live flow V2, §51-56).

The LLM picks an intent; this module turns it into a reachable walking
goal, deterministically, from the actor's own Observation only:

* APPROACH_OPPONENT -> the walkable cell nearest to the opponent's current,
  legitimately visible position, reachable from the actor (walls, closed or
  locked doors and furniture respected - the same NavGrid every MOVE uses).
  The executor re-plans toward the opponent as it moves, while visible.
* AVOID_OPPONENT -> an open exit of the current room whose far side is
  farther from the opponent than the actor is now, with a path that does
  not run through the opponent; otherwise the reachable cell (within
  AVOID_SEARCH_RADIUS) that is farthest from the opponent, if that gains at
  least AVOID_MIN_GAIN. Only geometry the actor can see (the walkable grid,
  the doors of its own room) - no hidden room, no opponent memory.
* IGNORE_OPPONENT is not an action: it means choosing any ordinary mission
  action (it never forces WAIT).

The Engine resolves any resulting contact (src/encounters.py). The model
never supplies coordinates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src import constants

APPROACH_OPPONENT = "approach_opponent"
AVOID_OPPONENT = "avoid_opponent"
ENCOUNTER_ACTIONS = (APPROACH_OPPONENT, AVOID_OPPONENT)
# accepted spellings, only when encounters are a rule of the match
ENCOUNTER_ALIASES = {
    "approach": APPROACH_OPPONENT, "approach_opponent": APPROACH_OPPONENT, "intercept": APPROACH_OPPONENT,
    "chase": APPROACH_OPPONENT, "confront": APPROACH_OPPONENT, "fight": APPROACH_OPPONENT,
    "avoid": AVOID_OPPONENT, "avoid_opponent": AVOID_OPPONENT, "flee": AVOID_OPPONENT,
    "evade": AVOID_OPPONENT, "retreat": AVOID_OPPONENT,
}
AVOID_SEARCH_RADIUS = 360.0
AVOID_MIN_GAIN = 48.0
AVOID_PATH_CLEARANCE = 1.5 * constants.ENCOUNTER_DISTANCE  # a flight path must not brush past the opponent
# Encounter alert (live, GameState.encounter_alerts): the opponent is close
# enough that contact - a fight - is a matter of a second or two. It is a
# decision point for the agent, never an automatic reaction: the executor
# stops the current walk (MOVE / avoid) so the model is asked again, and the
# payload says so (`encounter.alert`). The model then chooses to fight
# (approach_opponent), run (avoid_opponent) or carry on with its mission.
ENCOUNTER_ALERT_DISTANCE = 140.0  # px feet to feet: about 0.4 s when both walk toward each other
ENCOUNTER_ALERT_REARM_DISTANCE = 220.0  # a new alert needs the spies to separate this far first
ENCOUNTER_ALERT_CLOSING_PX = 40.0  # while running away: closer by this much than at the start = alert again


@dataclass(frozen=True)
class Goal:
    point: tuple[float, float]
    kind: str  # "opponent" | "exit" | "cell"
    door_id: str | None = None
    destination_room: str | None = None


def normalize_encounter_action(name: str) -> str:
    return ENCOUNTER_ALIASES.get(name, name)


def encounter_available(observation: Any) -> bool:
    """Encounter intents are offered only when encounters are a rule of
    this match and the opponent is legitimately visible right now."""
    return bool(getattr(observation, "encounters_enabled", False)) and observation.opponent is not None \
        and not observation.stunned


def _reachable(observation: Any):
    return observation.nav_grid.reachable_cells(observation.position)


def approach_goal(observation: Any) -> Goal | None:
    if not encounter_available(observation):
        return None
    grid = observation.nav_grid
    reachable = _reachable(observation)
    ox, oy = observation.opponent.position
    cell = grid.nearest_walkable(ox, oy)
    if cell is None or cell not in reachable:
        # the opponent stands somewhere this actor cannot walk to exactly:
        # the nearest reachable cell within a short radius
        best, best_d = None, math.inf
        for c in reachable:
            cx, cy = grid.center_of(c)
            d = math.hypot(cx - ox, cy - oy)
            if d < best_d:
                best, best_d = c, d
        if best is None or best_d > constants.INTERACTION_DISTANCE:
            return None
        cell = best
    return Goal(point=grid.center_of(cell), kind="opponent")


def _path_clear_of(path: list[tuple[float, float]], point: tuple[float, float], clearance: float) -> bool:
    return all(math.hypot(x - point[0], y - point[1]) >= clearance for x, y in path)


def avoid_goal(observation: Any, *, check_paths: bool = True) -> Goal | None:
    """`check_paths=False` is the cheap legality check (no A*); the executor
    resolves with full path checks at start."""
    if not encounter_available(observation):
        return None
    grid = observation.nav_grid
    reachable = _reachable(observation)
    here = observation.position
    opp = observation.opponent.position
    d_now = math.hypot(here[0] - opp[0], here[1] - opp[1])
    exits = []
    for obj in observation.objects:
        wp = obj.crossing_waypoint
        if obj.type != "door" or wp is None or obj.connects is None:
            continue
        cell = grid.nearest_walkable(*wp)
        if cell is None or cell not in reachable:
            continue
        gain = math.hypot(wp[0] - opp[0], wp[1] - opp[1]) - d_now
        if gain <= 0:
            continue
        destination = obj.connects[0] if obj.connects[1] == observation.current_room else obj.connects[1]
        exits.append((gain, obj.id, wp, destination))
    # a flight path must not bring the actor closer to the opponent than the
    # clearance - or than it already is, when it starts closer than that
    clearance = min(AVOID_PATH_CLEARANCE, d_now)
    for gain, door_id, wp, destination in sorted(exits, key=lambda e: (-e[0], e[1])):
        if not check_paths:
            return Goal(point=wp, kind="exit", door_id=door_id, destination_room=destination)
        path = grid.find_path(here, wp)
        if path and _path_clear_of(path, opp, clearance):
            return Goal(point=wp, kind="exit", door_id=door_id, destination_room=destination)
    # no good exit: the farthest reachable cell nearby
    candidates = []
    for c in reachable:
        cx, cy = grid.center_of(c)
        if math.hypot(cx - here[0], cy - here[1]) > AVOID_SEARCH_RADIUS:
            continue
        gain = math.hypot(cx - opp[0], cy - opp[1]) - d_now
        if gain >= AVOID_MIN_GAIN:
            candidates.append((gain, c))
    for gain, c in sorted(candidates, key=lambda e: (-e[0], e[1]))[:12]:
        point = grid.center_of(c)
        if not check_paths:
            return Goal(point=point, kind="cell")
        path = grid.find_path(here, point)
        if path and _path_clear_of(path, opp, clearance):
            return Goal(point=point, kind="cell")
    if check_paths:
        # nothing clears the opponent: the reachable option whose path keeps
        # the most distance from it (legality only promised a reachable one)
        options = [(Goal(point=wp, kind="exit", door_id=door_id, destination_room=destination), wp)
                   for _, door_id, wp, destination in exits]
        options += [(Goal(point=grid.center_of(c), kind="cell"), grid.center_of(c))
                    for _, c in sorted(candidates, key=lambda e: (-e[0], e[1]))[:12]]
        best, best_margin = None, -1.0
        for goal, point in options:
            path = grid.find_path(here, point)
            if not path:
                continue
            margin = min(math.hypot(x - opp[0], y - opp[1]) for x, y in path)
            if margin > best_margin:
                best, best_margin = goal, margin
        return best
    return None


def encounter_alert(observation: Any) -> dict[str, Any] | None:
    """The alert facts when the opponent is close and a fight is possible on
    contact (live only), else None. Only the actor's own observation."""
    if not getattr(observation, "encounter_alerts", False) or not encounter_available(observation):
        return None
    distance = observation.opponent_distance
    if distance is None or distance > ENCOUNTER_ALERT_DISTANCE or not fight_possible(observation):
        return None
    gap = max(0.0, distance - constants.ENCOUNTER_DISTANCE)
    return {
        "opponent_close": True,
        "opponent_distance": int(round(distance)),
        "contact_in_s_if_both_walk_toward_each_other": round(gap / (2 * constants.PLAYER_SPEED), 1),
        "your_lives": getattr(observation, "lives", None),
        "opponent_lives": getattr(observation, "opponent_lives", None),
        "decide": "approach_opponent to fight now, avoid_opponent to get away, or carry on with your mission "
                  "knowing that touching the opponent starts a fight",
    }


def encounter_legal_actions(observation: Any) -> list[dict[str, Any]]:
    """The encounter entries for `enumerate_legal_actions` (live surface)."""
    if not encounter_available(observation):
        return []
    entries = []
    target = observation.other_actor_id
    if fight_possible(observation) and approach_goal(observation) is not None:
        entries.append({"action": APPROACH_OPPONENT, "target_id": target})
    if avoid_goal(observation, check_paths=False) is not None:
        entries.append({"action": AVOID_OPPONENT, "target_id": target})
    return entries


def fight_possible(observation: Any) -> bool:
    """A fight can happen on contact right now: no cooldown, and the spies
    have separated since the last fight. Approaching is offered only then -
    walking into a spy you just fought does nothing."""
    return observation.encounter_cooldown <= 0 and getattr(observation, "encounter_armed", True) \
        and getattr(observation, "opponent_out_of_match", None) is None


def encounter_facts(observation: Any) -> dict[str, Any] | None:
    """Compact game-rule facts for the model payload (§49-50), or None when
    encounters are not a rule of this match."""
    if not getattr(observation, "encounters_enabled", False):
        return None
    distance = observation.opponent_distance
    return {
        "rule": "if you and the opponent touch (feet within encounter_distance, same room) a fight happens at once; "
                "either spy wins with equal chance; " + (
                    f"the loser loses one life and is knocked out for {constants.FIGHT_KO_SECONDS:g} s, then respawns elsewhere"
                    if getattr(observation, "fight_ko_respawn", False) and getattr(observation, "lives", None) is not None
                    else "the loser is stunned"),
        "encounter_distance": int(constants.ENCOUNTER_DISTANCE),
        "win_chance": constants.FIGHT_BLACK_WIN_PROBABILITY,
        # the penalty that applies in this match (the live KO rule, or the stun)
        **({"loser_loses_life": getattr(observation, "lives", None) is not None,
            "loser_knocked_out_seconds": constants.FIGHT_KO_SECONDS}
           if getattr(observation, "fight_ko_respawn", False)
           else {"loser_stunned_seconds": constants.FIGHT_LOSER_STUN_SECONDS}),
        "fight_possible_now": fight_possible(observation),
        "next_fight_possible_in": int(math.ceil(float(observation.encounter_cooldown))),  # whole seconds
        "after_a_fight_separate_by": int(constants.ENCOUNTER_REARM_DISTANCE),
        "opponent_visible": observation.opponent is not None,
        "opponent_room": observation.opponent.room if observation.opponent is not None else None,
        "opponent_distance": None if distance is None else int(round(distance)),
        "options": "approach_opponent / avoid_opponent (target_id = the opponent's id), or ignore and keep doing your mission",
        **({"opponent_lives": observation.opponent_lives}
           if getattr(observation, "opponent_lives", None) is not None else {}),
        # live: the opponent is out of the match (eliminated or escaped) - no fight can happen any more
        **({"opponent_out_of_match": out, "fights_over": True}
           if (out := getattr(observation, "opponent_out_of_match", None)) is not None else {}),
        **({"alert": alert} if (alert := encounter_alert(observation)) is not None else {}),
    }
