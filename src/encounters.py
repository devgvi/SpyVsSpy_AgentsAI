"""Spy encounters and fights (bugfix live flow V2, Part D).

Two separate things:

* the ENCOUNTER DECISION - approach, avoid or ignore the other spy - belongs
  to the agent (the LLM, through `approach_opponent` / `avoid_opponent` or
  any ordinary mission action; see src/encounter_intents.py);
* the FIGHT OUTCOME belongs to the engine: once the spies physically meet,
  this module resolves the fight at once, with no model call.

Rules (constants.py):

* trigger: both spies active (not escaped, not stunned), in the same room
  (engine room memory, so a doorway does not break it) and their feet
  points within ENCOUNTER_DISTANCE (28 px: the 28x28 collision boxes touch);
* outcome: Black wins with FIGHT_BLACK_WIN_PROBABILITY (0.5), drawn from
  `GameState.fight_rng` - an engine-owned RNG of its own, seeded from the
  game seed, so the shared `rng` (INSPECT detection, dependency placement)
  sees exactly the draws it saw before this feature existed;
* consequence: the loser is stunned for FIGHT_LOSER_STUN_SECONDS through
  the existing stun mechanic (same timer, same ACTOR_STUNNED /
  ACTOR_RECOVERED events a trap uses). No item loss, no death, no reset;
* no loop: no new fight for FIGHT_COOLDOWN_SECONDS of game time *and* until
  the spies have been at least ENCOUNTER_REARM_DISTANCE apart again;
* replay: `GameState.recorded_fight_winners` (a list of actor ids) makes the
  engine use the recorded winners in order instead of drawing - a replay
  never re-rolls.

Off by default (`GameState.encounters_enabled = False`): every evaluation
run, stored replay and workshop lab keeps its exact event log and state.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src import constants

if TYPE_CHECKING:  # pragma: no cover
    from src.state import GameState

FIGHT_EVENT_TYPES = ("SPY_ENCOUNTER", "SPY_FIGHT_STARTED", "SPY_FIGHT_RESOLVED")


def feet_distance(state: "GameState") -> float:
    black = state.actors[constants.BLACK_SPY_ID]
    white = state.actors[constants.WHITE_SPY_ID]
    return math.hypot(black.x - white.x, black.y - white.y)


def update_encounters(state: "GameState", dt: float, current_rooms: dict[str, str | None]) -> None:
    """Called from GameState.update when encounters are enabled."""
    if state.encounter_cooldown > 0:
        state.encounter_cooldown = max(0.0, state.encounter_cooldown - dt)
    black_id, white_id = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID
    distance = feet_distance(state)
    if distance >= constants.ENCOUNTER_REARM_DISTANCE:
        state.encounter_armed = True
    if not state.encounter_armed or state.encounter_cooldown > 0:
        return
    if any(state.mission_complete.get(a) or state.out_of_play(a) for a in (black_id, white_id)):
        return
    if any(state.actors[a].is_stunned for a in (black_id, white_id)):
        return
    room_b, room_w = current_rooms.get(black_id), current_rooms.get(white_id)
    if room_b is None or room_b != room_w:
        return
    if distance > constants.ENCOUNTER_DISTANCE:
        return
    resolve_fight(state, room_b, distance)


def resolve_fight(state: "GameState", room: str, distance: float) -> dict:
    black_id, white_id = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID
    p_black = constants.FIGHT_BLACK_WIN_PROBABILITY
    state._log_event("SPY_ENCOUNTER", black_id=black_id, white_id=white_id, room=room,
                     distance=round(distance, 1), game_time=round(state.elapsed_time, 3))
    state._log_event("SPY_FIGHT_STARTED", black_id=black_id, white_id=white_id, room=room,
                     game_time=round(state.elapsed_time, 3))
    if state.recorded_fight_winners:
        winner_id = state.recorded_fight_winners.pop(0)
        roll, source = None, "recorded"
    else:
        roll = state.fight_rng.random()
        winner_id = black_id if roll < p_black else white_id
        source = "rng"
    loser_id = white_id if winner_id == black_id else black_id
    loser = state.actors[loser_id]
    ko = getattr(state, "fight_ko_respawn", False)
    if ko:
        # live: KO for FIGHT_KO_SECONDS of match time, then a safe respawn
        # (GameState.knock_out / _advance_knockouts). Costs no trap life.
        seconds = constants.FIGHT_KO_SECONDS
        penalty = {"type": "ko_respawn", "actor_id": loser_id, "seconds": seconds}
    else:
        seconds = constants.FIGHT_LOSER_STUN_SECONDS
        loser.stunned_timer = max(loser.stunned_timer, seconds)
        loser.moving = False
        penalty = {"type": "stun", "actor_id": loser_id, "seconds": seconds}
    state.encounter_cooldown = constants.FIGHT_COOLDOWN_SECONDS
    state.encounter_armed = False
    record = {
        "black_id": black_id, "white_id": white_id, "winner_id": winner_id, "loser_id": loser_id,
        "rng_roll": None if roll is None else round(roll, 6), "fight_probability": p_black,
        "outcome_source": source, "room": room, "game_time": round(state.elapsed_time, 3), "penalty": penalty,
    }
    if ko and state.lives_enabled:
        # live: losing a fight costs a heart too (traps and fights share the 3 lives)
        lives_before = state.lives.get(loser_id)
        record["penalty"] = {**penalty, "life_lost": True, "lives_before": lives_before,
                             "lives_after": max(0, (lives_before or 0) - 1)}
    state._log_event("SPY_FIGHT_RESOLVED", **record)
    if ko:
        eliminated = state.lose_life(loser_id, "fight") if state.lives_enabled else False
        # live one-copy rule: the winner takes a required item from the loser
        record["items_taken"] = state.steal_item(winner_id, loser_id, everything=eliminated)
        if not eliminated:
            state.knock_out(loser_id, seconds)
    else:
        state._log_event("ACTOR_STUNNED", actor_id=loser_id, seconds=seconds)
    state.fights.append(record)
    return record


def recorded_winners(events: list[dict]) -> list[str]:
    """The fight winners, in order, from a recorded run: the engine's own
    `SPY_FIGHT_RESOLVED` events (event_log) or the live telemetry's
    `spy_fight` events. Assign the result to
    `GameState.recorded_fight_winners` to replay those fights exactly."""
    return [e["winner_id"] for e in events
            if e.get("type") == "SPY_FIGHT_RESOLVED" or e.get("event_type") == "spy_fight"]
