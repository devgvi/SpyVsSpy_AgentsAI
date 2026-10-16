"""Milestone 4H: Episode / Replay Foundation.

`Milestone_4.md`'s exact requirement: "Every episode records at least:
episode_id, seed, agent_profiles, mission_duration, winner/first_escape,
black_success, white_success, black_escape_time, white_escape_time,
event_log." `EpisodeRecord` is that summary, and `build_episode_record`
produces it the same way `src/scoring.py`'s `compute_score` produces a
score: a pure query over `GameState`, not a second copy of the same facts
kept in sync by hand. It can be called at any point in an episode - mid-
game, right after the first escape, or once both actors have resolved -
and always reflects the truth as of that call.

`agent_profiles` is the one input this module can't derive from
`GameState` itself: which profile (or "HUMAN") is driving each actor is a
controller-level fact, not engine state (`Game` owns it, per the
architectural invariant that controllers - and what picks them - stay
outside the engine's own truth). Callers pass it in explicitly.
"""

from dataclasses import dataclass

from src import constants
from src.scoring import compute_score
from src.state import GameState


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    seed: int
    agent_profiles: dict[str, str]  # actor_id -> profile name (or "HUMAN")
    mission_duration: float  # the configured deadline for this episode (constants.MISSION_TIME by default)
    winner: str | None  # actor_id with the strictly higher total score, or None if tied (including 0-0)
    first_escape: str | None  # actor_id of whoever escaped first, or None if nobody has yet
    black_success: bool
    white_success: bool
    black_escape_time: float | None  # elapsed game-time at escape, or None if not yet escaped
    white_escape_time: float | None
    event_log: tuple[dict, ...]  # an immutable snapshot, not a live view of GameState.event_log


def build_episode_record(state: GameState, agent_profiles: dict[str, str]) -> EpisodeRecord:
    black_id, white_id = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID
    black_score = compute_score(state, black_id)
    white_score = compute_score(state, white_id)

    winner = None
    if black_score.total != white_score.total:
        winner = black_id if black_score.total > white_score.total else white_id

    return EpisodeRecord(
        episode_id=state.episode_id,
        seed=state.seed,
        agent_profiles=dict(agent_profiles),
        mission_duration=state.mission_time_total,
        winner=winner,
        first_escape=state.finish_order[0] if state.finish_order else None,
        black_success=state.mission_complete[black_id],
        white_success=state.mission_complete[white_id],
        black_escape_time=state.escape_elapsed.get(black_id),
        white_escape_time=state.escape_elapsed.get(white_id),
        event_log=tuple(state.event_log),
    )
