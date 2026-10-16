"""What each action costs in mission time (bugfix mission time budget V2, §16-23).

One definition, next to the engine's own `_charge_time` branches
(src/state.py), from the same constants: the agent-facing cost of an action
is exactly what `GameState` will charge (tests C1-C7 check it against the
engine). A cost is time the actor is busy - it cannot move or act - while
its mission time keeps running.

* INSPECT: TIME_COST_INSPECT;
* INTERACT on furniture (search): TIME_COST_SEARCH, or ENTER_CODE / UNLOCK
  for a locked object; plus TIME_COST_PICKUP if the search finds an item;
* INTERACT on a door: OPEN, or UNLOCK / ENTER_CODE when it was locked; the
  exit: OPEN;
* PLACE_TRAP: TRAP_PLACEMENT_COST[trap_type];
* DISARM: TIME_COST_DISARM;
* SEND_MESSAGE: TIME_COST_SEND_MESSAGE;
* MOVE: walking time, path length / PLAYER_SPEED (see mission_time.py);
* a fight has no duration; the loser is stunned FIGHT_LOSER_STUN_SECONDS.
"""
from __future__ import annotations

from typing import Any

from src import constants


def interact_cost(obj: Any) -> float:
    """Cost of INTERACT on `obj` (an ObservedObject or GameObject), before any item pickup."""
    if obj.id == "exit_door":
        return constants.TIME_COST_OPEN
    locked = bool(getattr(obj, "locked", False))
    if obj.type == "door":
        if locked:
            return constants.TIME_COST_ENTER_CODE if obj.lock_type == "code" else constants.TIME_COST_UNLOCK
        return constants.TIME_COST_OPEN
    # furniture: GameState._search bundles unlock + search for a lock-typed object
    if obj.lock_type is None:
        return constants.TIME_COST_SEARCH
    return constants.TIME_COST_ENTER_CODE if obj.lock_type == "code" else constants.TIME_COST_UNLOCK


def inspect_cost(observation: Any = None) -> float:
    """The engine's INSPECT cost for this actor's game (live rules: LIVE_TIME_COST_INSPECT)."""
    cost = getattr(observation, "inspect_cost", None)
    return constants.TIME_COST_INSPECT if cost is None else cost


def place_trap_cost(trap_type: str) -> float:
    return constants.TRAP_PLACEMENT_COST[trap_type]


def disarm_cost() -> float:
    return constants.TIME_COST_DISARM


def send_message_cost() -> float:
    return constants.TIME_COST_SEND_MESSAGE


PICKUP_EXTRA = constants.TIME_COST_PICKUP
