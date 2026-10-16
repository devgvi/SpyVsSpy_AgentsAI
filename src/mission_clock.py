"""Mission clock modes (bugfix mission time budget V2, §5-11).

SIMULATION_TIME (the default - every evaluation run, stored replay and lab):
the one shared `GameState.mission_time_remaining` depletes by every frame's
dt. In the live_async timing a spy stands still while its model thinks, so
there provider latency spends mission time: a slower model gets less done
before the deadline.

ACTION_BUDGET (the live workshop default): each spy has its own mission
budget (`GameState.mission_budget[actor]`) that depletes while it plays -
walking, paying an action's time cost, being stunned, waiting after
choosing WAIT - but NOT while it is only waiting for its own model's answer
(`GameState.budget_hold`, set each frame by the game loop). Provider latency
is then a responsiveness metric, not a mission skill. The world keeps
moving on its own clock (`elapsed_time`); a hard world-time cap
(WORLD_TIME_CAP_FACTOR = 6 x the limit) ends a match whose models stop
answering.

Both modes: when a spy's own remaining time reaches 0 and it has not
escaped, it has failed (the existing timer-loss outcome) - enforced by the
engine (`_attempt_exit`, `scoring.compute_score`), not by prompt wording.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from src.state import GameState

SIMULATION_TIME = "simulation_time"
ACTION_BUDGET = "action_budget"
MISSION_CLOCK_MODES = (SIMULATION_TIME, ACTION_BUDGET)
WORLD_TIME_CAP_FACTOR = 6.0  # models ~4 s/answer spend ~75 % of world time waiting (measured)


def time_remaining_for(state: "GameState", actor_id: str) -> float:
    if state.mission_clock_mode == ACTION_BUDGET:
        return state.mission_budget.get(actor_id, 0.0)
    return state.mission_time_remaining


def time_used_for(state: "GameState", actor_id: str) -> float:
    return max(0.0, state.mission_time_total - time_remaining_for(state, actor_id))


def enable_action_budget(state: "GameState") -> None:
    state.mission_clock_mode = ACTION_BUDGET
    state.mission_budget = {a: state.mission_time_total for a in state.actors}
    state.budget_hold = set()


def update_budgets(state: "GameState", dt: float) -> None:
    """ACTION_BUDGET only; called from GameState.update instead of the shared countdown."""
    for actor_id in state.actors:
        if state.mission_complete.get(actor_id) or actor_id in state.budget_hold:
            continue
        before = state.mission_budget[actor_id]
        state.mission_budget[actor_id] = max(0.0, before - dt)
        if before > 0 and state.mission_budget[actor_id] == 0:
            state._log_event("MISSION_TIME_EXPIRED", actor_id=actor_id, clock=ACTION_BUDGET)
    if state.elapsed_time >= state.mission_time_total * WORLD_TIME_CAP_FACTOR:
        for actor_id in state.actors:
            if not state.mission_complete.get(actor_id) and state.mission_budget[actor_id] > 0:
                state.mission_budget[actor_id] = 0.0
                state._log_event("MISSION_TIME_EXPIRED", actor_id=actor_id, clock="world_time_cap")
    active = [state.mission_budget[a] for a in state.actors if not state.mission_complete.get(a)]
    # the shared value the HUD and older readers show: the longest budget still in play
    state.mission_time_remaining = max(active) if active else state.mission_time_remaining


# -- V3: live wall-clock mission timer ----------------------------------------
# Live play: the shared countdown is real time. `remaining = limit - active
# wall time elapsed`: it keeps running while the models think (inference is
# not free), a manual pause stops it (skip()), and a slow or stalled frame
# does not slow it down (the world's own dt is capped at FRAME_DT_CAP; this
# clock is not). Evaluation / replay never use it (GameState.external_mission_clock).
WALL_CLOCK = "wall_clock"
GAME_TIME = "game_time"
MISSION_TIMERS = (WALL_CLOCK, GAME_TIME)
WALL_CLOCK_MAX_STEP = 5.0  # a longer gap (laptop asleep, debugger) is not match time


class LiveWallClock:
    """Measures unpaused wall time between frames (monotonic)."""

    def __init__(self, time_fn=None) -> None:
        import time as _time

        self._time_fn = time_fn or _time.monotonic
        self._last: float | None = None
        self.active_seconds = 0.0

    def tick(self, speed: float = 1.0) -> float:
        """Seconds of match time since the previous tick (0 on the first)."""
        now = self._time_fn()
        last, self._last = self._last, now
        if last is None:
            return 0.0
        step = min(max(0.0, now - last), WALL_CLOCK_MAX_STEP) * speed
        self.active_seconds += step
        return step

    def skip(self) -> None:
        """A paused frame: the time since the previous tick does not count."""
        self._last = self._time_fn()
