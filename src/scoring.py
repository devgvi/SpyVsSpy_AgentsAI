"""Milestone 4E: competitive/cooperative victory scoring.

`Milestone_4.md` is explicit this is not zero-sum: each spy wants to (1)
complete their own mission, (2) escape before the deadline, (3) preferably
escape first - while the other spy also succeeding has some *positive*
value, not none and not a penalty. Both spies may score well in the same
episode.

Every line below is a pure function of state that already exists -
nothing here is itself new persisted state beyond the two facts that
genuinely can't be reconstructed any other way (`GameState.finish_order`/
`escape_snapshot` - who got out in what order, and how much of the shared
clock was left when they did). Own mission success, whether the other
actor also escaped, and whether the deadline has passed are all derived
live every time `compute_score` runs. That's what makes "both successes
persist" (Milestone_4.md's acceptance line) fall out for free: if actor A
escapes first and actor B escapes twenty seconds later, querying A's
score *after* B's escape correctly includes the "other spy also escaped"
bonus - nothing about A's own result needed to be found and edited
retroactively, because it was never frozen in the first place.

Trapping is deliberately absent from every line here - Milestone_4.md:
"Do NOT directly reward trapping. Sabotage matters only when it improves
the real objective." A trap can only ever affect a score indirectly, by
changing whether/when/how completely an actor's own mission and escape
happen.
"""

from dataclasses import dataclass

from src import constants
from src.state import GameState


@dataclass(frozen=True)
class ScoreBreakdown:
    actor_id: str
    mission_success: bool  # collected every required item - independent of ever escaping
    escaped: bool  # physically got out while mission_time_remaining was still > 0
    escaped_first: bool  # only meaningful when escaped
    other_escaped: bool  # the other actor has (also) escaped, as of right now
    failed: bool  # the deadline passed and this actor never got out
    remaining_time_bonus: float  # only nonzero when escaped
    total: float


def compute_score(state: GameState, actor_id: str) -> ScoreBreakdown:
    other_id = next(aid for aid in state.actors if aid != actor_id)
    actor = state.actors[actor_id]

    mission_success = actor.inventory.has_all(state.items.required)
    escaped = state.mission_complete[actor_id]
    other_escaped = state.mission_complete[other_id]
    # Mutually exclusive with `escaped` by construction: GameState._attempt_exit
    # refuses to succeed at all once the deadline has passed, so an actor
    # can never be both `escaped` and `failed`.
    # MISSION_IMPOSSIBLE (live, one copy of each item) is a failure like the
    # other two: the same SCORE_FAILURE_TO_ESCAPE, nothing else changes.
    failed = not escaped and (state.time_remaining_for(actor_id) <= 0 or bool(getattr(state, "eliminated", {}).get(actor_id))
                              or actor_id in getattr(state, "mission_impossible", {}))
    escaped_first = escaped and bool(state.finish_order) and state.finish_order[0] == actor_id

    total = 0.0
    remaining_time_bonus = 0.0

    if mission_success:
        total += constants.SCORE_COMPLETE_OWN_MISSION
    if escaped:
        total += constants.SCORE_ESCAPE_BEFORE_DEADLINE
        if escaped_first:
            total += constants.SCORE_ESCAPE_FIRST
        if other_escaped:
            total += constants.SCORE_OTHER_ALSO_ESCAPES
        remaining_time_bonus = state.escape_snapshot.get(actor_id, 0.0) * constants.SCORE_REMAINING_TIME_BONUS_PER_SECOND
        total += remaining_time_bonus
    if failed:
        total += constants.SCORE_FAILURE_TO_ESCAPE

    return ScoreBreakdown(
        actor_id=actor_id,
        mission_success=mission_success,
        escaped=escaped,
        escaped_first=escaped_first,
        other_escaped=other_escaped,
        failed=failed,
        remaining_time_bonus=remaining_time_bonus,
        total=total,
    )
