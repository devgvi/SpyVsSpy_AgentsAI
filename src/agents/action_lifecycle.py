"""M5D.2: shared action-execution lifecycle for every agent runtime.

Fixes the root cause M5D.1 exposed (see CHANGELOG.md's M5D.2 entry for the
full diagnostic trace): an accepted MOVE decision was re-decided as a
brand-new strategic request on almost every tick. `LLMController.get_action`
used a one-shot straight-line direction computed once at decision time,
then just kept re-returning `self.last_action` until the observation
digest changed - but the actor's own natural position drift *while
executing that same accepted move* changes the digest just as much as any
genuinely new situation would, and evaluation mode's real-time throttle
(0.5s) is far shorter than real model latency (20-45s), so by the time a
decision resolves and movement starts, the very next tick's position
change immediately re-triggers a whole new (expensive, frozen-time)
decision cycle. Net effect: each "decision" bought only a fraction of a
second of real movement before being abandoned.

The fix: once MOVE is accepted, execute it across as many engine ticks as
it takes via the same NavGrid/A* pathfinding `ScriptedBotController`
already uses (src/navigation.py) - not a new movement system - and don't
let the owning controller request a new strategic decision until a
genuine lifecycle boundary is reached (target reached, path unreachable,
or an interruption). INTERACT/INSPECT/PLACE_TRAP/DISARM resolve within a
single engine tick already (state.py's apply_action) and don't need
multi-tick tracking here - the observation legitimately changes once they
resolve, which correctly (and, pre-M5D.2, already correctly) warrants a
new decision on the very next tick.
"""

from __future__ import annotations

import dataclasses
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from src import constants
from src.actions import ActionType, GameAction
from src.interaction_range import distance_to_rect
from src.encounter_intents import (
    APPROACH_OPPONENT,
    AVOID_OPPONENT,
    ENCOUNTER_ACTIONS,
    ENCOUNTER_ALERT_CLOSING_PX,
    ENCOUNTER_ALERT_REARM_DISTANCE,
    approach_goal,
    avoid_goal,
    encounter_alert,
)
from src.observation import Observation

MOVE_ACTION = "move"
INSTANT_TARGET_ACTIONS = frozenset({"interact", "inspect", "place_trap", "disarm"})

# Spec section 5's boundary vocabulary.
TARGET_REACHED = "target_reached"
PATH_UNREACHABLE = "path_unreachable"
INTERRUPTED = "interrupted"
RESOLVED = "resolved"
EXPLICIT_WAIT = "explicit_wait"
# M5E.4.3: a room-crossing MOVE's path was fully walked (geometric
# completion) but the actor still isn't in the intended destination room
# (semantic completion) - reported distinctly rather than silently folded
# into TARGET_REACHED (spec section 40: "do not silently report success").
DOORWAY_CROSSING_INCOMPLETE = "doorway_crossing_incomplete"
# live: the opponent came close (encounter_alert) - the walk stops so the agent decides again
OPPONENT_NEARBY = "opponent_nearby"
# Bugfix live flow V2 (Part D): encounter intents. APPROACH ends at contact
# (TARGET_REACHED - the engine resolves the fight), when the opponent is no
# longer visible (OPPONENT_LOST) or after APPROACH_MAX_SECONDS of chasing
# (CHASE_TIME_LIMIT); AVOID ends at its exit / far cell (TARGET_REACHED).
OPPONENT_LOST = "opponent_lost"
CHASE_TIME_LIMIT = "chase_time_limit"
APPROACH_MAX_SECONDS = 8.0
APPROACH_REPLAN_SECONDS = 0.5
# Bugfix (navigation/exploration). PATH_UNREACHABLE keeps its name (memory,
# telemetry and the liveness fix already use it) and now means exactly
# "no reachable approach point": the approach is chosen among cells
# connected to the actor, so a failure is never a picked-but-unreachable
# cell. Two new failure boundaries replace silent success / endless moves:
APPROACH_INCOMPLETE = "approach_incomplete"  # path walked, target still not the engine's near object
MOVE_STALLED = "move_stalled"  # no movement for STALL_GAME_SECONDS of game time, or over the time budget
FAILED_MOVE_BOUNDARIES = frozenset({PATH_UNREACHABLE, APPROACH_INCOMPLETE, MOVE_STALLED})
# Game time, not ticks: while the other spy's model is thinking the shared
# clock is frozen (evaluation mode) and nothing moves, which is not a stall.
STALL_GAME_SECONDS = 1.0
MOVE_TIME_BUDGET_FACTOR = 3.0  # budget = factor x (path length / speed) + slack
MOVE_TIME_BUDGET_SLACK = 2.0
# Bugfix live flow (Part B): "close approach" - live play only (set by the
# workshop LiveGame). The engine range stays 70 px from the feet, but a MOVE_TO
# an object no longer stops on the first frame the object enters that range
# (median 60 px in the M7.1 ui-check run, a visible gap of up to 119 screen
# px): it walks on to its approach point - the reachable cell closest to the
# object - and ends there, or earlier once the feet are within this distance
# of the object's collision rect.
CLOSE_APPROACH_PX = 24.0
# Bugfix spawn/combat/visual polish (Part C, live close approach only): an
# INSPECT / INTERACT / PLACE_TRAP / DISARM the model chose on an object that
# is in engine range but still visibly apart (feet > CLOSE_APPROACH_PX from
# its collision rect) first walks the last pixels to the approach point, then
# performs that same action - one decision, no second model call.
CLOSE_RANGE_ACTIONS = frozenset({"inspect", "interact", "place_trap", "disarm", "enter_safe_code"})


@dataclass
class ActiveAction:
    """An accepted action currently executing. Only MOVE actually spans
    more than one call (`path`/`path_index` track its progress); other
    action types are recorded here only for correlation/telemetry - they
    never re-enter `continue_`."""

    action_id: str
    decision_id: str
    action: str  # canonical normalized name
    target_id: str | None
    trap_type: str | None
    started_game_time: float
    simulation_ticks: int = 0
    path: list[tuple[float, float]] = field(default_factory=list)
    path_index: int = 0
    # M5E.4.3: set only for a MOVE routed to a door's `crossing_waypoint`
    # (see `ActionExecutor.start`) - the room this move must actually end
    # up in for TARGET_REACHED to be honest. `None` for every other MOVE
    # (unchanged pre-M5E.4.3 proximity-based completion).
    destination_room: str | None = None
    # M5F.2: the SEND_MESSAGE text payload - only ever set when
    # `action == "send_message"`, carried here purely for `finish()`'s own
    # sender-side memory recording (spec section 7), never re-validated.
    message: str | None = None
    # M5F.2.2: an optional message riding along with ANY action type (the
    # "zero decision cost" envelope, spec section 19) - `{"recipient_id",
    # "text"}` or `None`. Carried here purely for `finish()`'s own
    # sender-side memory recording, same discipline as `message` above -
    # the GameAction itself is where actual engine delivery happens (see
    # `GameState.apply_action`'s pre-dispatch delivery step).
    accompanying_message: dict | None = None
    # Bugfix (navigation/exploration): MOVE instrumentation, surfaced in the
    # action_lifecycle telemetry so a decision error (wrong target) can be told
    # apart from an execution error (right target, wrong place).
    approach_point: tuple[float, float] | None = None
    start_position: tuple[float, float] | None = None
    final_position: tuple[float, float] | None = None
    distance_to_target: float | None = None
    near_object_id: str | None = None
    failure_detail: str | None = None
    budget_game_seconds: float | None = None
    _progress_position: tuple[float, float] | None = None
    _progress_game_time: float | None = None
    # the mission clock (seconds remaining) when the MOVE started - the time
    # budget is measured on this clock. `started_game_time` is the caller's
    # decision time (Native/Strands pass a synthetic decision_seq x interval),
    # a different clock: comparing the two (bugfix live flow) made every MOVE
    # late in a match "exceed" its budget at once.
    _start_clock: float | None = None
    # encounter intents only
    encounter_goal_kind: str | None = None
    _replan_clock: float | None = None
    # close-range interaction: the chosen action, performed once in close range
    pending_game_action: GameAction | None = None
    fired: bool = False
    # live encounter alerts: how far the opponent was when this action started
    start_opponent_distance: float | None = None


def choose_approach_point(observation: Observation, target_obj) -> tuple[float, float] | None:
    """Production approach point for MOVE_TO(target_obj) (bugfix §4-5).

    Uses `NavGrid.reachable_approach_point`: a cell connected to the actor,
    within interaction range of the target's collision rect, where the
    engine's own nearest-object rule picks this target. Test doubles that
    only implement the older `nearest_approach_point` keep working.
    """
    grid = observation.nav_grid
    if hasattr(grid, "reachable_approach_point"):
        interactables = [(o.id, o.collision_rect) for o in observation.objects if o.interactable]
        return grid.reachable_approach_point(target_obj.id, target_obj.collision_rect, observation.position, interactables)
    return grid.nearest_approach_point(target_obj.collision_rect, constants.INTERACTION_DISTANCE, observation.position)


def _move_budget(start: tuple[float, float], path: list[tuple[float, float]]) -> float:
    length, (px, py) = 0.0, start
    for wx, wy in path:
        length += math.hypot(wx - px, wy - py)
        px, py = wx, wy
    return MOVE_TIME_BUDGET_FACTOR * length / constants.PLAYER_SPEED + MOVE_TIME_BUDGET_SLACK


def _instant_or_wait_game_action(
    normalized_action: str, target_id: str | None, trap_type: str | None, message: str | None = None,
    code: str | None = None,
) -> GameAction:
    if normalized_action == "enter_safe_code":  # live riddle (src/puzzles.py)
        return GameAction(ActionType.INTERACT, target_id=target_id, code=code)
    type_map = {
        "interact": ActionType.INTERACT,
        "inspect": ActionType.INSPECT,
        "place_trap": ActionType.PLACE_TRAP,
        "disarm": ActionType.DISARM,
        "send_message": ActionType.SEND_MESSAGE,
    }
    if normalized_action in type_map:
        return GameAction(type_map[normalized_action], target_id=target_id, trap_type=trap_type, message=message)
    return GameAction(ActionType.WAIT)


class ActionExecutor:
    """Owned by one controller/adapter instance (composition, not a
    thread-shared singleton) - the owning controller's existing single-
    background-thread discipline already serializes access: `start()` is
    only ever called from the decision-request background thread, right
    before it signals completion; `continue_()` is only ever called from
    the main-thread `get_action()` loop while no decision is in flight.
    Those two never overlap in time (see the owning controller's docstring
    for the exact handoff), so no extra locking is needed here.
    """

    def __init__(self) -> None:
        self.active: ActiveAction | None = None
        # The ActiveAction that just reached a boundary, so a caller that
        # only gets a boundary_reason string back can still recover
        # action_id/decision_id/timing for telemetry and memory
        # correlation. Overwritten on the next start()/continue_() call
        # that finishes something - a caller must read it immediately
        # after seeing a non-None boundary_reason, not defer it.
        self.last_finished: ActiveAction | None = None
        # M5D.2 spec section 11: strategic decisions vs. accepted-action
        # execution, tracked separately. Both LLMController and
        # StrandsAgentAdapter compose one ActionExecutor each and read
        # these directly (or via a thin property) - one definition shared
        # by every runtime instead of duplicated bookkeeping.
        self.actions_accepted = 0
        self.actions_started = 0
        self.actions_completed = 0
        self.actions_failed = 0
        self.actions_interrupted = 0
        self.actions_cancelled = 0
        # live play only (see CLOSE_APPROACH_PX); evaluation keeps the
        # first-frame-in-range completion every stored run was recorded with
        self.close_approach = False
        # live play only: an encounter alert stops a walk so the agent decides
        # (fight / run / carry on); one alert per close encounter
        self.encounter_alerts = False
        self._alert_armed = True

    @property
    def in_progress(self) -> bool:
        return self.active is not None

    def fired_at_start(self) -> bool:
        """The active close-range action fired inside start() and no frame has
        passed since: start() returned the action to perform. In evaluation
        timing start() runs on the request thread, so the frame that performs
        it is the hand-over frame - the controller hands that action over
        instead of calling continue_(), which would end it unperformed."""
        return self.active is not None and self.active.fired and self.active.simulation_ticks == 0

    def start(
        self,
        observation: Observation,
        decision_id: str,
        normalized_action: str,
        target_id: str | None,
        trap_type: str | None,
        game_time: float,
        message: str | None = None,
        accompanying_message: dict | None = None,
        code: str | None = None,
    ) -> tuple[GameAction, str | None, str]:
        """Begin executing a newly-accepted action this tick.

        Returns (game_action_for_this_tick, boundary_reason, action_id).
        boundary_reason is None only when a MOVE has genuinely started and
        is still in progress (self.active is set); otherwise the action
        resolved (or failed) within this single call and self.active stays
        None - e.g. PATH_UNREACHABLE (MOVE couldn't even start - a
        legitimate *execution* failure, distinct from model-output
        validity, per spec section 13), TARGET_REACHED (already in range),
        RESOLVED (an instant action was applied this tick; M5B's own
        pending-decision mechanism records its actual outcome one tick
        later), or EXPLICIT_WAIT.

        `accompanying_message` (M5F.2.2 spec sections 19-20): attached to
        whichever GameAction this call returns - regardless of which
        branch below fires, including a PATH_UNREACHABLE MOVE - since the
        decision boundary (and therefore the message) was already reached
        the instant `start()` was called, independent of whether the
        physical action goes on to succeed engine-side. Never attached on
        a later `continue_()` call, so a multi-tick MOVE delivers its
        message exactly once, at acceptance time.
        """
        self.actions_accepted += 1
        action_id = str(uuid.uuid4())
        if self.encounter_alerts:
            self._note_opponent(observation, deciding=True)
        record = ActiveAction(
            action_id=action_id,
            decision_id=decision_id,
            action=normalized_action,
            target_id=target_id,
            trap_type=trap_type,
            started_game_time=game_time,
            message=message,
            accompanying_message=accompanying_message,
        )

        if normalized_action == MOVE_ACTION:
            record.start_position = tuple(observation.position)
            target_obj = observation.get(target_id) if target_id else None
            if target_obj is None:
                record.failure_detail = "unknown_target"
                self._record_move_end(record, observation)
                self.last_finished = record
                return self._with_accompanying_message(GameAction(ActionType.WAIT), accompanying_message), PATH_UNREACHABLE, action_id

            # M5E.4.3: a door with a real destination-side waypoint (open,
            # known connects, actor currently on one of its two sides)
            # routes the *entire* path there instead of merely hugging the
            # door's own rect - see `find_room_crossing_waypoint`. Every
            # other target (including a closed door, where crossing isn't
            # possible anyway) keeps the exact pre-M5E.4.3 behavior.
            destination_room: str | None = None
            crossing = target_obj.type == "door" and target_obj.crossing_waypoint is not None and target_obj.connects is not None
            if crossing:
                approach = target_obj.crossing_waypoint
                current_room = observation.current_room
                if current_room in target_obj.connects:
                    destination_room = target_obj.connects[0] if target_obj.connects[1] == current_room else target_obj.connects[1]
            elif observation.nearby_object_id == target_id and not self._should_close_in(observation, target_obj):
                # Bugfix: already in interaction range - nothing to walk. Before,
                # the approach/path were computed first, so an actor standing in
                # range but in a pocket the grid cannot path out of failed with
                # PATH_UNREACHABLE.
                self.actions_started += 1
                self._record_move_end(record, observation)
                self.last_finished = record
                return self._with_accompanying_message(GameAction(ActionType.WAIT), accompanying_message), TARGET_REACHED, action_id
            else:
                approach = choose_approach_point(observation, target_obj)

            path = observation.nav_grid.find_path(observation.position, approach) if approach is not None else None
            if not path and self.close_approach and observation.nearby_object_id == target_id:
                # in range but no walkable way closer: in range is enough
                self.actions_started += 1
                self._record_move_end(record, observation)
                self.last_finished = record
                return self._with_accompanying_message(GameAction(ActionType.WAIT), accompanying_message), TARGET_REACHED, action_id
            record.approach_point = approach
            if not path:
                record.failure_detail = "no_reachable_approach_point"
                self._record_move_end(record, observation)
                self.last_finished = record
                return self._with_accompanying_message(GameAction(ActionType.WAIT), accompanying_message), PATH_UNREACHABLE, action_id
            record.path = path
            record.destination_room = destination_room
            record.budget_game_seconds = _move_budget(observation.position, path)
            record._progress_position = tuple(observation.position)
            record._progress_game_time = observation.mission_time_remaining
            record._start_clock = observation.mission_time_remaining
            record.start_opponent_distance = observation.opponent_distance
            self.active = record
            self.actions_started += 1
            action, boundary = self._advance_move(observation)
            return self._with_accompanying_message(action, accompanying_message), boundary, action_id

        if normalized_action in ENCOUNTER_ACTIONS:
            action, boundary = self._start_encounter(record, observation)
            return self._with_accompanying_message(action, accompanying_message), boundary, action_id

        game_action = _instant_or_wait_game_action(normalized_action, target_id, trap_type, message, code=code)
        if self._needs_close_in_for(normalized_action, target_id, observation):
            started = self._start_interaction_approach(record, observation, game_action)
            if started is not None:
                action, boundary = started
                return self._with_accompanying_message(action, accompanying_message), boundary, action_id
        boundary = EXPLICIT_WAIT if normalized_action == "wait" else RESOLVED
        self.actions_started += 1
        self.last_finished = record
        return self._with_accompanying_message(game_action, accompanying_message), boundary, action_id

    def _note_opponent(self, observation: Observation, *, deciding: bool = False) -> None:
        """Re-arm once the spies have separated; a decision made while the
        alert is on uses it up (the agent already saw it)."""
        distance = observation.opponent_distance
        if distance is None or distance > ENCOUNTER_ALERT_REARM_DISTANCE:
            self._alert_armed = True
        elif deciding and encounter_alert(observation) is not None:
            self._alert_armed = False

    def _alert_due(self, active: ActiveAction, observation: Observation) -> bool:
        """A perception event, not a behaviour: the walk stops and the agent is
        asked. Never during a deliberate approach (the agent chose contact)."""
        self._note_opponent(observation)
        if active.action == APPROACH_OPPONENT or encounter_alert(observation) is None:
            return False
        if self._alert_armed:
            self._alert_armed = False
            return True
        start = active.start_opponent_distance
        if active.action == AVOID_OPPONENT and start is not None \
                and observation.opponent_distance < start - ENCOUNTER_ALERT_CLOSING_PX:
            active.start_opponent_distance = observation.opponent_distance
            return True  # running away, yet the opponent is closing in: decide again
        return False

    def _needs_close_in_for(self, action: str, target_id: str | None, observation: Observation) -> bool:
        if not self.close_approach or action not in CLOSE_RANGE_ACTIONS or not target_id:
            return False
        if observation.nearby_object_id != target_id:
            return False  # not in engine range: the engine refuses it anyway (legal surface never offers it)
        target_obj = observation.get(target_id)
        if target_obj is None or getattr(observation, "nav_grid", None) is None:
            return False
        return distance_to_rect(*observation.position, target_obj.collision_rect) > CLOSE_APPROACH_PX

    def _start_interaction_approach(self, record: ActiveAction, observation: Observation,
                                    game_action: GameAction) -> tuple[GameAction, str | None] | None:
        """Walk the last pixels, then act (see CLOSE_RANGE_ACTIONS). None: no
        walkable way closer - act from where the actor stands (it is in range)."""
        target_obj = observation.get(record.target_id)
        approach = choose_approach_point(observation, target_obj)
        path = observation.nav_grid.find_path(observation.position, approach) if approach is not None else None
        if not path:
            return None
        record.start_position = tuple(observation.position)
        record.approach_point = approach
        record.path = path
        record.pending_game_action = game_action
        record.budget_game_seconds = _move_budget(observation.position, path)
        record._progress_position = tuple(observation.position)
        record._progress_game_time = observation.mission_time_remaining
        record._start_clock = observation.mission_time_remaining
        self.active = record
        self.actions_started += 1
        return self._advance_interaction_approach(observation)

    def _advance_interaction_approach(self, observation: Observation) -> tuple[GameAction, str | None]:
        active = self.active
        assert active is not None
        target = observation.get(active.target_id)
        # close enough AND the engine's nearest object is the target (on the
        # way in another object can be nearest for a moment - keep walking)
        close = (target is not None and observation.nearby_object_id == active.target_id
                 and distance_to_rect(*observation.position, target.collision_rect) <= CLOSE_APPROACH_PX)
        action = GameAction(ActionType.WAIT) if close else self._follow_path(observation, active)
        if close or active.path_index >= len(active.path):
            if observation.nearby_object_id != active.target_id:
                active.failure_detail = "target_not_in_interaction_range"
                return self._end(active, observation, APPROACH_INCOMPLETE)
            # perform the chosen action this frame; the lifecycle ends next frame
            active.fired = True
            self._record_move_end(active, observation)
            return active.pending_game_action, None
        return action, None

    def _should_close_in(self, observation: Observation, target_obj) -> bool:
        """Live close approach: in range, but still visibly apart."""
        # inter-room doors keep their crossing logic; the exit (a door with no
        # far room) is approached like furniture - live feedback: spies stopped
        # at 69 px from the exit, at the very edge of range, and a nudge took
        # the exit out of range again, so they never escaped
        inter_room_door = target_obj.type == "door" and target_obj.connects is not None
        return (self.close_approach and not inter_room_door
                and distance_to_rect(*observation.position, target_obj.collision_rect) > CLOSE_APPROACH_PX)

    def _in_range_of_target(self, active: ActiveAction, observation: Observation) -> bool:
        return active.destination_room is None and active.target_id is not None and observation.nearby_object_id == active.target_id

    @staticmethod
    def _with_accompanying_message(action: GameAction, accompanying_message: dict | None) -> GameAction:
        if accompanying_message is None:
            return action
        return dataclasses.replace(action, accompanying_message=accompanying_message)

    def continue_(self, observation: Observation) -> tuple[GameAction, str | None]:
        """Advance the in-progress MOVE by one tick. Returns
        (game_action_for_this_tick, boundary_reason). boundary_reason is
        None while still genuinely in progress - the caller must not
        request a new strategic decision this tick in that case."""
        active = self.active
        if active is None:  # pragma: no cover - defensive; callers check in_progress first
            return GameAction(ActionType.WAIT), RESOLVED

        active.simulation_ticks += 1
        if active.fired:  # the close-range interaction was applied last frame
            self.last_finished = active
            self.active = None
            return GameAction(ActionType.WAIT), RESOLVED
        if observation.mission_complete or observation.stunned or observation.busy or observation.time_expired:
            return self._end(active, observation, INTERRUPTED)
        if self.encounter_alerts and self._alert_due(active, observation):
            return self._end(active, observation, OPPONENT_NEARBY)
        if active.action == APPROACH_OPPONENT:
            ended = self._approach_check(active, observation)
            if ended is not None:
                return ended

        # Bugfix: no-progress detection, in game time (see STALL_GAME_SECONDS).
        position = tuple(observation.position)
        now = observation.mission_time_remaining
        if active._progress_position is None or math.dist(position, active._progress_position) > 0.5:
            active._progress_position, active._progress_game_time = position, now
        elif active._progress_game_time is not None and active._progress_game_time - now >= STALL_GAME_SECONDS:
            if self.close_approach and self._in_range_of_target(active, observation):
                return self._end(active, observation, TARGET_REACHED)  # blocked on the way in, but in range
            active.failure_detail = "no_progress"
            return self._end(active, observation, MOVE_STALLED)
        if (active.budget_game_seconds is not None and active._start_clock is not None
                and active._start_clock - now > active.budget_game_seconds):
            if self.close_approach and self._in_range_of_target(active, observation):
                return self._end(active, observation, TARGET_REACHED)
            active.failure_detail = "time_budget_exceeded"
            return self._end(active, observation, MOVE_STALLED)

        if active.action in ENCOUNTER_ACTIONS:
            return self._advance_encounter(observation)
        if active.pending_game_action is not None:
            return self._advance_interaction_approach(observation)
        return self._advance_move(observation)

    # -- encounter intents (bugfix live flow V2, Part D) ---------------------

    def _start_encounter(self, record: ActiveAction, observation: Observation) -> tuple[GameAction, str | None]:
        record.start_position = tuple(observation.position)
        goal = approach_goal(observation) if record.action == APPROACH_OPPONENT else avoid_goal(observation)
        path = observation.nav_grid.find_path(observation.position, goal.point) if goal is not None else None
        if not path:
            record.failure_detail = "no_reachable_encounter_goal"
            self._record_move_end(record, observation)
            self.last_finished = record
            return GameAction(ActionType.WAIT), PATH_UNREACHABLE
        record.path = path
        record.approach_point = goal.point
        record.destination_room = goal.destination_room
        record.encounter_goal_kind = goal.kind
        now = observation.mission_time_remaining
        record.budget_game_seconds = (APPROACH_MAX_SECONDS if record.action == APPROACH_OPPONENT
                                      else _move_budget(observation.position, path))
        record._progress_position = tuple(observation.position)
        record._progress_game_time = now
        record._start_clock = now
        record._replan_clock = now
        record.start_opponent_distance = observation.opponent_distance
        self.active = record
        self.actions_started += 1
        return self._advance_encounter(observation)

    def _approach_check(self, active: ActiveAction, observation: Observation) -> tuple[GameAction, str] | None:
        """Checked before stall/budget: contact or losing sight ends a chase."""
        distance = observation.opponent_distance
        if distance is None:
            active.failure_detail = "opponent_not_visible"
            return self._end(active, observation, OPPONENT_LOST)
        if distance <= constants.ENCOUNTER_DISTANCE:
            return self._end(active, observation, TARGET_REACHED)
        if (active._start_clock is not None and active.budget_game_seconds is not None
                and active._start_clock - observation.mission_time_remaining > active.budget_game_seconds):
            return self._end(active, observation, CHASE_TIME_LIMIT)
        return None

    def _advance_encounter(self, observation: Observation) -> tuple[GameAction, str | None]:
        active = self.active
        assert active is not None
        now = observation.mission_time_remaining
        if active.action == APPROACH_OPPONENT:
            # the opponent moves: re-plan toward where it is now, a few times a second
            due = active._replan_clock is None or active._replan_clock - now >= APPROACH_REPLAN_SECONDS
            if due or active.path_index >= len(active.path):
                goal = approach_goal(observation)
                path = observation.nav_grid.find_path(observation.position, goal.point) if goal is not None else None
                active._replan_clock = now
                if path:
                    active.path, active.path_index, active.approach_point = path, 0, goal.point
                elif active.path_index >= len(active.path):
                    active.failure_detail = "no_reachable_encounter_goal"
                    return self._end(active, observation, PATH_UNREACHABLE)
            action = self._follow_path(observation, active)
            if action.type is ActionType.WAIT:  # on the goal cell, the opponent is right here: step onto it
                ox, oy = observation.opponent.position
                dx, dy = ox - observation.position[0], oy - observation.position[1]
                d = math.hypot(dx, dy)
                if d > 0:
                    action = GameAction(ActionType.MOVE, direction=(dx / d, dy / d))
            return action, None
        action = self._follow_path(observation, active)
        if active.path_index >= len(active.path):
            if active.destination_room is not None and observation.current_room != active.destination_room:
                return self._end(active, observation, DOORWAY_CROSSING_INCOMPLETE)
            return self._end(active, observation, TARGET_REACHED)
        return action, None

    def _end(self, active: ActiveAction, observation: Observation, boundary: str) -> tuple[GameAction, str]:
        self._record_move_end(active, observation)
        self.last_finished = active
        self.active = None
        return GameAction(ActionType.WAIT), boundary

    @staticmethod
    def _record_move_end(record: ActiveAction, observation: Observation) -> None:
        record.final_position = tuple(observation.position)
        record.near_object_id = observation.nearby_object_id
        target = observation.get(record.target_id) if record.target_id else None
        if target is not None:
            record.distance_to_target = round(distance_to_rect(*observation.position, target.collision_rect), 1)

    def _advance_move(self, observation: Observation) -> tuple[GameAction, str | None]:
        active = self.active
        assert active is not None
        # M5E.4.3: a room-crossing move must not exit early just because
        # the actor is within interaction range of the door's own rect -
        # that range comfortably covers the dead zone this milestone
        # exists to fix. Every other MOVE keeps this exact pre-M5E.4.3
        # proximity shortcut.
        if self._in_range_of_target(active, observation):
            target = observation.get(active.target_id)
            if target is None or not self._should_close_in(observation, target):
                return self._end(active, observation, TARGET_REACHED)

        action = self._follow_path(observation, active)
        if active.path_index >= len(active.path):
            if active.destination_room is not None and observation.current_room != active.destination_room:
                # Geometric completion without semantic completion (spec
                # section 40) - never silently reported as success.
                return self._end(active, observation, DOORWAY_CROSSING_INCOMPLETE)
            if active.destination_room is None and active.target_id is not None and observation.nearby_object_id != active.target_id:
                # Bugfix: the path is walked but the engine does not consider
                # the target in range (or another object is nearer) - the
                # intended interaction is not legal, so this is not success.
                active.failure_detail = "target_not_in_interaction_range"
                return self._end(active, observation, APPROACH_INCOMPLETE)
            return self._end(active, observation, TARGET_REACHED)
        return action, None

    def finish(
        self,
        memory: Any,
        telemetry: Any,
        episode_id: str,
        actor_id: str,
        agent_runtime: str,
        agent_runtime_version: str,
        observation: Observation,
        boundary: str,
    ) -> None:
        """Call once `start()`/`continue_()` returns a non-None boundary
        reason: updates the actions_*/counters, folds the outcome back
        into AgentMemory (a path-unreachable MOVE becomes a legitimate
        recorded failure, the same "unreachable" vocabulary M5B already
        uses for interact-type failures - spec section 13), and emits
        `action_lifecycle` telemetry. Interact-type actions' actual
        success/failure is tracked separately by AgentMemoryTracker's
        existing pending-decision mechanism (unchanged) - this only adds
        the lifecycle-telemetry/action-counter layer for them.
        """
        finished = self.last_finished
        if finished is None:  # pragma: no cover - defensive
            return

        game_time = observation.mission_time_remaining

        # M5F.2.2 spec section 7 (reused): the sender's own memory of an
        # accompanying zero-decision-cost message - recorded regardless of
        # which boundary fired below (the message was already delivered
        # engine-side at acceptance time, independent of whether the
        # physical action itself went on to succeed - see `start()`'s
        # docstring), unlike the FULL_COST `send_message` branch further
        # down which only applies to that one specific standalone action.
        if finished.accompanying_message is not None:
            memory.record_message_sent(
                game_time=game_time, message_id=finished.decision_id,
                recipient_actor_id=finished.accompanying_message.get("recipient_id"),
                text=finished.accompanying_message.get("text"),
            )
            memory.record_event(
                game_time, "MESSAGE_SENT",
                finished.accompanying_message.get("recipient_id"), finished.accompanying_message.get("text"),
            )

        if boundary == PATH_UNREACHABLE:
            self.actions_failed += 1
            memory.record_failure(
                game_time=game_time,
                action=finished.action,
                target_id=finished.target_id,
                reason="unreachable",
                detail="no path found",
            )
            memory.update_decision_result(finished.decision_id, "blocked_unreachable")
            memory.record_event(game_time, "ACTION_FAILED", finished.target_id, "unreachable")
        elif boundary in (APPROACH_INCOMPLETE, MOVE_STALLED):
            # Bugfix: execution failures, reported as such - never success.
            self.actions_failed += 1
            reason = "out_of_range" if boundary == APPROACH_INCOMPLETE else "stalled"
            memory.record_failure(
                game_time=game_time,
                action=finished.action,
                target_id=finished.target_id,
                reason=reason,
                detail=finished.failure_detail or boundary,
            )
            memory.update_decision_result(finished.decision_id, f"move_{reason}")
            memory.record_event(game_time, "ACTION_FAILED", finished.target_id, reason)
        elif boundary == INTERRUPTED:
            self.actions_interrupted += 1
        elif boundary == OPPONENT_NEARBY:
            self.actions_interrupted += 1
            memory.update_decision_result(finished.decision_id, "stopped_opponent_nearby")
            memory.record_event(game_time, "OPPONENT_NEARBY", finished.target_id, "opponent close: fight, run or carry on")
        elif boundary == DOORWAY_CROSSING_INCOMPLETE:
            # The MOVE itself genuinely executed (real path, real
            # movement) - only the semantic crossing goal wasn't met, so
            # this counts as completed *execution*, distinctly flagged via
            # the existing action-outcome vocabulary (spec section 44:
            # reuse normal action-outcome recording, not a new memory
            # feature) rather than a fabricated new "you failed" note.
            self.actions_completed += 1
            memory.update_decision_result(finished.decision_id, "doorway_crossing_incomplete")
            memory.record_event(game_time, "ACTION_FAILED", finished.target_id, "doorway_crossing_incomplete")
        else:  # TARGET_REACHED, RESOLVED, EXPLICIT_WAIT
            self.actions_completed += 1
            if boundary == TARGET_REACHED:
                memory.update_decision_result(finished.decision_id, "success")
                memory.record_event(game_time, "MOVE_COMPLETED", finished.target_id)
            elif boundary in (OPPONENT_LOST, CHASE_TIME_LIMIT):
                memory.update_decision_result(finished.decision_id, boundary)
                memory.record_event(game_time, "ENCOUNTER_ENDED", finished.target_id, boundary)
            elif finished.action == "send_message":
                # M5F.2 spec section 7: the sender's own memory of what it
                # sent - recorded here (the one place an accepted action's
                # outcome is already known synchronously, same as every
                # other instant action) rather than waiting a tick for
                # AgentMemoryTracker's authoritative-outcome diff, which
                # only watches Observation *object* state and would never
                # see this at all.
                memory.update_decision_result(finished.decision_id, "success")
                memory.record_message_sent(
                    game_time=game_time, message_id=finished.decision_id,
                    recipient_actor_id=finished.target_id, text=finished.message,
                )
                memory.record_event(game_time, "MESSAGE_SENT", finished.target_id, finished.message)

        if telemetry:
            telemetry.emit_action_lifecycle(
                episode_id=episode_id,
                actor_id=actor_id,
                decision_id=finished.decision_id,
                action_id=finished.action_id,
                action=finished.action,
                target_id=finished.target_id,
                action_status=("failed" if boundary in FAILED_MOVE_BOUNDARIES
                               else "interrupted" if boundary in (INTERRUPTED, OPPONENT_NEARBY) else "completed"),
                action_terminal_reason=boundary,
                action_started_game_time=finished.started_game_time,
                action_completed_game_time=game_time,
                action_simulation_ticks=finished.simulation_ticks,
                agent_runtime=agent_runtime,
                agent_runtime_version=agent_runtime_version,
                navigation=(self.navigation_telemetry(finished)
                            if finished.action == MOVE_ACTION or finished.action in ENCOUNTER_ACTIONS
                            or finished.pending_game_action is not None else None),
            )

    @staticmethod
    def navigation_telemetry(record: ActiveAction) -> dict[str, Any]:
        """Compact MOVE instrumentation (bugfix §29): where the actor aimed,
        where it ended, and whether the intended interaction became legal."""
        def pt(p):
            return None if p is None else [round(float(p[0]), 1), round(float(p[1]), 1)]
        return {
            "move_target_id": record.target_id,
            "start_position": pt(record.start_position),
            "approach_point": pt(record.approach_point),
            "final_position": pt(record.final_position),
            "distance_to_target_after_move": record.distance_to_target,
            "interaction_range": constants.INTERACTION_DISTANCE,
            "near_object_id": record.near_object_id,
            "interaction_legal_after_move": record.target_id is not None and record.near_object_id == record.target_id,
            "failure_detail": record.failure_detail,
            **({"encounter_goal_kind": record.encounter_goal_kind} if record.encounter_goal_kind else {}),
            **({"close_range_approach": True} if record.pending_game_action is not None else {}),
        }

    @staticmethod
    def _follow_path(observation: Observation, active: ActiveAction) -> GameAction:
        ax, ay = observation.position
        while active.path_index < len(active.path):
            wx, wy = active.path[active.path_index]
            dist = math.hypot(wx - ax, wy - ay)
            if dist > constants.WAYPOINT_ARRIVAL_DISTANCE and not _passed_waypoint(active, (ax, ay)):
                return GameAction(ActionType.MOVE, direction=((wx - ax) / dist, (wy - ay) / dist))
            active.path_index += 1
        return GameAction(ActionType.WAIT)


def _passed_waypoint(active: ActiveAction, position: tuple[float, float]) -> bool:
    """Bugfix: with a large frame step (a slow frame, the live loop's 0.1 s cap,
    speed x2) the actor can overshoot a waypoint by more than the 6 px arrival
    tolerance and then oscillate around it forever. A waypoint counts as reached
    once the actor has moved past it along the segment from the previous waypoint
    (projection >= segment length)."""
    i = active.path_index
    # The first waypoint is the snapped start cell centre; its segment starts
    # where the MOVE started.
    prev = active.path[i - 1] if i > 0 else active.start_position
    if prev is None:
        return False
    px, py = prev
    wx, wy = active.path[i]
    sx, sy = wx - px, wy - py
    seg2 = sx * sx + sy * sy
    if seg2 == 0:
        return True
    ax, ay = position
    return ((ax - px) * sx + (ay - py) * sy) >= seg2
