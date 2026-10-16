"""Final match ranking (bugfix live match flow V3, Part I).

Reuses the M4E score (`src.scoring.compute_score`, constants SCORE_*) as is -
no new scoring model. Success and ranking are separate: MISSION SUCCESS is
"every required item + escaped before the deadline", RANKING compares the
two spies, so both can succeed while one ranks first.

Deterministic tie-break (no randomness), in order:
  1. higher score
  2. earlier escape (escape order 1 before 2; not escaped last)
  3. more time remaining (at escape; else what is left now)
  4. more lives remaining
  5. fewer trap hits
  6. otherwise TIE (same rank)
Traps give no points; lives only ever break a tie.
"""
from __future__ import annotations

from typing import Any

from src import constants
from src.scoring import compute_score


def actor_result(state, actor_id: str) -> dict[str, Any]:
    score = compute_score(state, actor_id)
    escaped = score.escaped
    order = state.finish_order.index(actor_id) + 1 if actor_id in state.finish_order else None
    remaining = state.escape_snapshot.get(actor_id) if escaped else max(0.0, state.time_remaining_for(actor_id))
    lives = state.lives.get(actor_id) if getattr(state, "lives_enabled", False) else None
    return {
        "actor_id": actor_id,
        "lifecycle": state.lifecycle(actor_id),
        "mission_success": bool(score.mission_success and escaped),
        "objectives_complete": score.mission_success,
        "escaped": escaped,
        "failed": score.failed,
        "score_total": round(score.total, 2),
        "score_mission": constants.SCORE_COMPLETE_OWN_MISSION if score.mission_success else 0.0,
        "score_escape": constants.SCORE_ESCAPE_BEFORE_DEADLINE if escaped else 0.0,
        "score_first_escape": constants.SCORE_ESCAPE_FIRST if score.escaped_first else 0.0,
        "score_other_survives": constants.SCORE_OTHER_ALSO_ESCAPES if escaped and score.other_escaped else 0.0,
        "score_time_bonus": round(score.remaining_time_bonus, 2),
        "score_failure": constants.SCORE_FAILURE_TO_ESCAPE if score.failed else 0.0,
        "escape_order": order,
        "escape_game_time": round(state.escape_elapsed[actor_id], 2) if actor_id in state.escape_elapsed else None,
        "time_remaining_s": round(remaining or 0.0, 2),
        "lives": lives,
        "trap_hits": state.trap_hits.get(actor_id, 0),
    }


def _key(result: dict[str, Any]) -> tuple:
    return (
        -result["score_total"],
        result["escape_order"] if result["escape_order"] is not None else float("inf"),
        -result["time_remaining_s"],
        -(result["lives"] if result["lives"] is not None else 0),
        result["trap_hits"],
    )


def rank_match(state) -> list[dict[str, Any]]:
    """Every actor's result with `rank` (1 = first) and `tie`, best first."""
    results = sorted((actor_result(state, a) for a in state.actors), key=lambda r: (_key(r), r["actor_id"]))
    rank = 0
    previous = None
    for i, result in enumerate(results):
        key = _key(result)
        if key != previous:
            rank = i + 1
            previous = key
        result["rank"] = rank
    for result in results:
        result["tie"] = sum(1 for r in results if r["rank"] == result["rank"]) > 1
    return results
