"""A deterministic, non-LLM controller for White Spy.

Produces GameAction objects exactly like HumanController does - the game
engine (state.py, game.py) cannot tell the two apart, and never will need
to change when a future LLMController replaces this one (see Milestone 3's
architecture diagram). Unlike HumanController, this one has real "world
knowledge" to reason about, so as of Milestone 3D (Milestone_32.md's
numbering; MILESTONE_3.md calls the same content M3C, "Partial
Information") it never touches GameState at all - it only ever sees an
Observation (src/observation.py), built fresh each tick and scoped to
exactly what this actor legitimately knows. It still never sets position,
inventory, or search state itself; every effect goes through GameAction ->
state.apply_action(), the same as the human. The one write channel it does
use is `Observation.log_event` - purely observational bookkeeping (target
selection, path computation), never gameplay state (see that method's
docstring on GameState).

M3A gave White Spy plain search/collect/escape (see the original version of
this docstring in git history). M3C (Milestone_32.md's numbering) upgraded
that into a strategy-aware policy, evaluated in this order every time
something is within interaction range (see `_decide_interaction`):

    IF mission complete (already escaped):        WAIT
    ELIF stunned:                                  WAIT (nothing to send -
                                                    apply_action() would
                                                    drop it anyway, but
                                                    skipping pathing too
                                                    avoids wasted work)
    ELIF holding all required items:               pursue the exit door
    ELIF nearby object is a *detected* enemy trap: DISARM
    ELIF nearby useful object not yet inspected:   INSPECT
    ELIF nearby useful object inspected (hit or
         miss) and not searched yet:               SEARCH (INTERACT) -
                                                     a miss doesn't make the
                                                     bot omniscient; it
                                                     risks the same
                                                     undetected trap a
                                                     human could
    ELIF nearby object just searched, believed
         clear, not yet attempted, and
         strategically worth trapping:              PLACE_TRAP
    ELSE:                                           pick the nearest object
                                                     this actor hasn't
                                                     searched yet, path to it

This stays deliberately simple (MILESTONE_3.md explicitly says "avoid
complex planning" for this baseline): trap placement is opportunistic -
only considered on an object the bot is already standing next to because
it just searched it, never a special trip.

Milestone 4F ("Strategic Trap Economy") replaced the old "trap if X or Y"
check with an actual value estimate, `_trap_value` - `Milestone_4.md`'s
own framing: `trap value ~= expected opponent delay - placement cost -
strategic risk`. PLACE_TRAP's cost (`constants.TRAP_PLACEMENT_COST[trap_type]`
- a flat 3.0s before Milestone 4I gave each of the four trap types its own
number) means an object only clears the bar when its relevance signals
(searching it once handed *this actor* an item, it sits near the exit, the
opponent is currently visible, last-seen, or - M4I - just revealed by
their own alarm) combine to outweigh that cost plus this actor's own
strategic risk (low remaining time, already mission-ready, down to its
last trap of that type) - see `_best_trap_choice`/`_trap_value`/
`_strategic_risk`. Every signal is something this actor is legitimately
allowed to know; `MILESTONE_3.md`'s suggested trap-placement signal
"opponent hasn't searched it" - the other actor's private search history
- is still intentionally absent, same reason as before M4F.

Because trap placement now goes off *belief* ("clear_after_inspection",
which explicitly does not mean "actually safe" - see observation.py)
rather than ground truth, a placement attempt can silently fail (the
object already secretly has an enemy trap). `_trap_attempted` remembers
which objects this actor has already tried, so a failed attempt doesn't
turn into an infinite retry loop that would otherwise softlock exploration
- the bot simply gives up on that one spot and moves on, same as it would
after any other single failed environment interaction.

Known limitation of the "opportunistic, no revisits" design: once an
object is marked searched by this actor, the bot never inspects or
targets it again, so it won't detect (and can't disarm) a trap the
opponent places there afterward - it can only walk into it.

Milestone 4B added one more rung, ahead of "pick the nearest unsearched
object": a locked non-exit door this actor already holds the key/keycard
for (`ObservedObject.can_unlock`, derived only from this actor's own
inventory/knowledge - see observation.py) is unlocked and opened on
sight, and becomes a valid pathing target in `_select_target` so the bot
actually walks to it rather than only reacting if it happens to wander
past. A locked door the bot *can't* yet open is simply invisible to
target selection, same as before M4B - it'll naturally become reachable
once the right container is searched.

Milestone 4G ("Agent Profiles"): the same ladder, the same
Observation/GameAction/rules for every profile (`Milestone_4.md` is
explicit about this) - `AgentProfile` is just two parameterized weights on
the decisions this controller already makes, not a different controller
per profile ("prefer parameterized weights over unrelated controllers").
`RACER` sets `inspect_before_search=False`, so the existing "not inspected
and not searched -> INSPECT" rung above is simply skipped and the very
next rung ("not searched -> SEARCH") fires instead - faster, but blind to
undetected enemy traps, and (as a direct consequence of inspecting less)
it also detects and disarms fewer traps, without a separate "disarm less"
parameter needed. `CAUTIOUS` keeps `inspect_before_search=True` (today's
baseline) but adds a `trap_value_bias` that makes `_trap_value` (see
below) clear its bar far less often - risk reduction for *itself*, not
sabotage of the opponent. `OPPORTUNIST` is the M4F baseline exactly:
inspects first, and traps whenever the unbiased value formula says it's
worth it. See `RACER`/`CAUTIOUS`/`OPPORTUNIST` below for the exact values.
"""

import hashlib
import json
import math
import uuid
from dataclasses import dataclass

from src import constants
from src.actions import ActionType, GameAction
from src.llm.telemetry import TelemetryLogger
from src.observation import Observation, ObservedObject

_EXIT_STALL_TICKS = 120  # ~2.0s at constants.FPS - Observation carries no dt, so ticks stand in for elapsed time
_EXIT_STALL_EPSILON = 2.0  # world units of movement below which a tick doesn't count as "progress"

# M5E spec section 28: a small, closed vocabulary naming which policy
# branch produced a given decision - stable strings, never derived from
# hidden state, so opponent behavior stays auditable from telemetry alone.
# The base eight are the spec's own suggested list; the three marked "extra"
# below fill gaps its own policy ladder has that the suggested list didn't
# name (a locked door it can open, an exit path with no items yet, or
# genuinely nothing left to do) - added rather than overloading an existing
# code with two different meanings.
REASON_MISSION_READY_EXIT = "MISSION_READY_EXIT"
REASON_DISARM_DETECTED_TRAP = "DISARM_DETECTED_TRAP"
REASON_INSPECT_NEAR_TARGET = "INSPECT_NEAR_TARGET"
REASON_SEARCH_NEAR_TARGET = "SEARCH_NEAR_TARGET"
REASON_PLACE_TRAP_OPPORTUNITY = "PLACE_TRAP_OPPORTUNITY"
REASON_MOVE_TO_MISSION_TARGET = "MOVE_TO_MISSION_TARGET"
REASON_MOVE_TO_EXPLORATION_TARGET = "MOVE_TO_EXPLORATION_TARGET"
REASON_WAIT_NO_ACTION = "WAIT_NO_ACTION"
REASON_WAIT_STUNNED_OR_COMPLETE = "WAIT_STUNNED_OR_COMPLETE"  # extra
REASON_INTERACT_UNLOCK_DOOR = "INTERACT_UNLOCK_DOOR"  # extra
REASON_WAIT_PATH_UNREACHABLE = "WAIT_PATH_UNREACHABLE"  # extra


def infer_reason_code(observation: Observation, action: GameAction) -> str:
    """Classifies an already-decided GameAction into one of the reason
    codes above, purely from the (observation, action) pair - deliberately
    kept separate from (and never changing) `ScriptedBotController`'s own
    decision logic, so instrumentation can never introduce a behavioral
    side effect in code that M3/M4 already validated. Every fact used here
    (stunned/busy/complete, has_all_required_items, the nearby object's own
    trap_status/locked/can_unlock) is the same Observation-only knowledge
    the controller itself was already limited to.
    """
    if observation.mission_complete or observation.stunned or observation.busy:
        return REASON_WAIT_STUNNED_OR_COMPLETE

    nearby = observation.get(observation.nearby_object_id) if observation.nearby_object_id else None

    if action.type == ActionType.DISARM:
        return REASON_DISARM_DETECTED_TRAP
    if action.type == ActionType.INSPECT:
        return REASON_INSPECT_NEAR_TARGET
    if action.type == ActionType.PLACE_TRAP:
        return REASON_PLACE_TRAP_OPPORTUNITY
    if action.type == ActionType.INTERACT:
        if nearby is not None and nearby.id == "exit_door":
            return REASON_MISSION_READY_EXIT
        if nearby is not None and nearby.type == "door":
            return REASON_INTERACT_UNLOCK_DOOR
        return REASON_SEARCH_NEAR_TARGET
    if action.type == ActionType.MOVE:
        return REASON_MOVE_TO_MISSION_TARGET if observation.has_all_required_items() else REASON_MOVE_TO_EXPLORATION_TARGET
    # WAIT: distinguish "nothing legal to do" from "computed a path but it
    # turned out unreachable" only by the one signal available here -
    # whether a target was actually selected (bot_controller._current_target_id
    # isn't visible to a pure function, so this stays a coarser, honest
    # default rather than reaching into controller internals).
    return REASON_WAIT_NO_ACTION


@dataclass(frozen=True)
class AgentProfile:
    """Milestone 4G: a named bundle of the two weights `ScriptedBotController`
    already exposes as decision knobs. Every profile plays by the exact
    same rules, Observation, and GameAction set - only these numbers
    differ, which is the whole point ("prefer parameterized weights over
    unrelated controllers").
    """

    name: str
    inspect_before_search: bool  # False: skip straight to SEARCH, accepting the undetected-trap risk, for speed
    trap_value_bias: float  # added to _trap_value's strategic risk - higher means "trap less often"


# "Start with" (Milestone_4.md's exact three). RACER's trap_value_bias is
# moot, not just low: PLACE_TRAP only ever fires from the "clear_after_
# inspection" branch below, which RACER can never reach at all since it
# skips inspection entirely - "fewer traps" falls out of "less inspection"
# on its own, with nothing left for a bias to bite on (0.0 here is honest
# about that, not a missed tuning opportunity). CAUTIOUS keeps inspecting
# (today's baseline) but raises the bar for actually trapping what it
# finds - risk reduction for itself, not sabotage. OPPORTUNIST is the
# unmodified Milestone 4F baseline: inspects first, trap value unbiased.
RACER = AgentProfile(name="RACER", inspect_before_search=False, trap_value_bias=0.0)
CAUTIOUS = AgentProfile(name="CAUTIOUS", inspect_before_search=True, trap_value_bias=3.0)
OPPORTUNIST = AgentProfile(name="OPPORTUNIST", inspect_before_search=True, trap_value_bias=0.0)
AGENT_PROFILES = {p.name: p for p in (RACER, CAUTIOUS, OPPORTUNIST)}


class ScriptedBotController:
    def __init__(
        self,
        actor_id: str,
        profile: AgentProfile = OPPORTUNIST,
        telemetry: TelemetryLogger | None = None,
        episode_id: str | None = None,
    ) -> None:
        self.actor_id = actor_id
        self.profile = profile
        self._current_target_id: str | None = None
        self._trap_attempted: set[str] = set()
        self._path: list[tuple[float, float]] = []
        self._path_index = 0
        self._path_target_id: str | None = None
        # M4A.3 diagnostics only - none of this feeds back into any
        # decision; see _watch_exit_stall.
        self._mission_ready_logged = False
        self._exit_stall_ticks = 0
        self._exit_stall_logged = False
        self._last_watched_pos: tuple[float, float] | None = None
        # M5E: optional, additive-only opponent-decision telemetry (spec
        # section 27) - every existing White-scripted caller passes
        # neither of these and gets byte-identical behavior to before.
        self.telemetry = telemetry
        self.episode_id = episode_id or "ep_unknown"
        self.decisions_made = 0
        # M5E bug fix (found during the full 3-seed benchmark, not the
        # pilot - see CHANGELOG): `get_action` is called once per render
        # frame (60Hz), same as White's bot_controller always has been -
        # that part is pre-existing and fine, since a stable MOVE simply
        # keeps returning the same action every physics tick. What is new
        # in M5E is telemetry: emitting `opponent_decision` unconditionally
        # on every one of those frame calls turned a handful of real
        # high-level decisions into tens of thousands of byte-identical
        # rows per episode (up to 54k in the live benchmark, ballooning
        # decisions.jsonl to 112MB for 9 episodes) - most of it logged
        # while the evaluated agent's LLM call was pending and effective_dt
        # was frozen at 0.0, so the *same* decision got re-emitted every
        # frame for as long as inference took. This tracks the last
        # emitted decision's identity so telemetry (and decisions_made)
        # only advances on an actual decision-boundary change, matching
        # spec section 11's "should not ... constantly replan" intent and
        # section 27's one-record-per-decision contract, without touching
        # `_compute_action` or the GameAction returned to the engine.
        self._last_decision_key: tuple[str, str | None, str] | None = None

    def get_action(self, observation: Observation) -> GameAction:
        """Public entry point: computes the action via the exact same,
        unmodified decision ladder as before M5E (`_compute_action`), then
        - only if a telemetry sink was supplied - separately classifies and
        records it (`infer_reason_code`/`emit_opponent_decision`). This
        split means instrumentation can never change what action gets
        chosen, only whether it also gets reported.
        """
        action = self._compute_action(observation)
        if self.telemetry is not None:
            reason_code = infer_reason_code(observation, action)
            # PLACE_TRAP/INSPECT/INTERACT/DISARM never carry an explicit
            # target_id (see _decide_interaction above) - they resolve
            # against current_interaction_target at execution time, which
            # this observation's own nearby_object_id already mirrors
            # (actor-legitimate, not hidden). Recording that here instead
            # of the GameAction's always-None field is what actually makes
            # target_id auditable in telemetry (spec section 27); MOVE
            # already carries its own real target_id explicitly.
            effective_target_id = action.target_id or observation.nearby_object_id
            decision_key = (action.type.value, effective_target_id, reason_code)
            if decision_key != self._last_decision_key:
                self._last_decision_key = decision_key
                self.decisions_made += 1
                observation_hash = hashlib.sha256(
                    json.dumps(
                        {"position": observation.position, "nearby": observation.nearby_object_id, "room": observation.current_room},
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                self.telemetry.emit_opponent_decision(
                    episode_id=self.episode_id,
                    actor_id=self.actor_id,
                    decision_id=str(uuid.uuid4()),
                    controller="scripted_active_v1",
                    observation_hash=observation_hash,
                    selected_action=action.type.value,
                    target_id=effective_target_id,
                    reason_code=reason_code,
                    game_time=observation.mission_time_remaining,
                )
        else:
            self.decisions_made += 1
        return action

    def _compute_action(self, observation: Observation) -> GameAction:
        if observation.has_all_required_items() and not self._mission_ready_logged:
            self._mission_ready_logged = True
            observation.log_event("MISSION_READY")

        if observation.mission_complete or observation.stunned or observation.busy:
            # M4D: `busy` (mid-cooldown from a timed action) drops every
            # action exactly like `stunned` already did (see
            # GameState.apply_action) - WAIT here is purely to skip the
            # wasted work of picking a target/computing a path this tick,
            # not something the engine actually needs to see.
            return GameAction(ActionType.WAIT)

        # "near target": act on whatever's usefully in range right now,
        # even if it isn't the object a path was aimed at - two objects can
        # share overlapping interaction ranges (e.g. the sofa and the side
        # table), and insisting on an exact match would leave the bot stuck
        # waiting a step early forever.
        if observation.nearby_object_id is not None:
            action = self._decide_interaction(observation, observation.get(observation.nearby_object_id))
            if action is not None:
                return action

        if observation.has_all_required_items():
            target_obj = observation.get("exit_door")
        else:
            target_obj = self._pick_target(observation)

        if target_obj is None:
            return GameAction(ActionType.WAIT)  # nothing left this actor hasn't already searched

        if self._current_target_id != target_obj.id:
            self._current_target_id = target_obj.id
            observation.log_event("TARGET_SELECTED", object_id=target_obj.id)
            self._path = []

        if not self._path or self._path_target_id != target_obj.id:
            self._compute_path(observation, target_obj)

        if not self._path:
            # Computed but empty: nearest_approach_point/find_path couldn't
            # actually reach this target. Drop it so the next tick re-picks
            # rather than sitting on a target _pick_target would otherwise
            # keep returning forever (see _pick_target's stickiness).
            self._current_target_id = None
            return GameAction(ActionType.WAIT)

        if target_obj.id == "exit_door":
            self._watch_exit_stall(observation)

        return self._follow_path(observation)

    def _watch_exit_stall(self, observation: Observation) -> None:
        """Diagnostic only - never changes behavior
        (M4A_3_LIVE_PLAY_DIAGNOSTICS_DOOR_POLISH.md section 5 and 27 are
        explicit: instrumentation is allowed, behavioral hacks are not).
        Logs one detailed `EXIT_STALL_DIAGNOSTIC` event if White appears to
        make no real progress toward the exit for ~2 seconds while it's
        actually pursuing it - exactly the symptom a live-play session
        reported, without assuming it's a bug. This method is only ever
        reached while un-stunned and un-completed (get_action returns
        before this point otherwise), so a legitimate stun never counts
        against it - it simply doesn't tick during one.
        """
        pos = observation.position
        moved = self._last_watched_pos is None or math.dist(pos, self._last_watched_pos) >= _EXIT_STALL_EPSILON
        self._last_watched_pos = pos

        if moved:
            self._exit_stall_ticks = 0
            self._exit_stall_logged = False
            return

        self._exit_stall_ticks += 1
        if self._exit_stall_ticks >= _EXIT_STALL_TICKS and not self._exit_stall_logged:
            self._exit_stall_logged = True
            exit_obj = observation.get("exit_door")
            observation.log_event(
                "EXIT_STALL_DIAGNOSTIC",
                seed=observation.seed,
                position=pos,
                target_id=self._current_target_id,
                distance_to_target=math.dist(pos, exit_obj.center) if exit_obj else None,
                mission_ready=observation.has_all_required_items(),
                door_open=exit_obj.door_open if exit_obj else None,
                path_remaining=max(0, len(self._path) - self._path_index),
            )

    def _pick_target(self, observation: Observation) -> ObservedObject | None:
        """Keep pursuing whatever object we're already committed to,
        rather than re-running "nearest unsearched" from scratch every
        frame. `_select_target` ranks purely by straight-line distance
        from the actor's *current* position - close enough to a wobble
        near a room boundary, two candidates in different rooms can be
        near-tied, and re-picking every frame flip-flops between them
        forever with zero net progress (a real bug found in M4A.2's
        tighter layout, not merely a hypothetical one). Only re-evaluate
        once the current target is actually resolved.
        """
        current = observation.get(self._current_target_id) if self._current_target_id else None
        if current is not None and not self._target_resolved(current):
            return current
        return self._select_target(observation)

    @staticmethod
    def _target_resolved(obj: ObservedObject) -> bool:
        """M4B: a locked door the bot is pursuing (see _select_target)
        counts as resolved once it's actually open, not once it's
        "searched" - doors were never searchable to begin with."""
        if obj.type == "door":
            return bool(obj.door_open)
        return obj.searched

    # -- interaction policy -------------------------------------------------

    def _decide_interaction(self, observation: Observation, obj: ObservedObject | None) -> GameAction | None:
        """What to do about the nearby object `obj` right now, or None if
        there's nothing useful here (the caller should go find a new
        target instead). See the module docstring for the priority order.
        """
        if obj is None:
            return None

        if obj.type == "door" and obj.id != "exit_door":
            # M4B: a locked inter-room door is worth a deliberate INTERACT
            # if this actor already holds/knows what unlocks it (see
            # ObservedObject.can_unlock - derived from this actor's own
            # inventory/knowledge only). An already-open door, or one still
            # locked with nothing to open it yet, has nothing for the bot
            # to do here - same as before M4B.
            if obj.locked and obj.can_unlock:
                return GameAction(ActionType.INTERACT)
            return None

        if obj.id == "exit_door":
            return GameAction(ActionType.INTERACT) if observation.has_all_required_items() else None

        if obj.trap_status == "detected" and not obj.trap_is_mine:
            return GameAction(ActionType.DISARM)

        if not obj.inspected and not obj.searched and self.profile.inspect_before_search:
            return GameAction(ActionType.INSPECT)

        # Either already inspected (hit or miss) and not a *detected* enemy
        # trap (handled above), or this profile skips inspection entirely
        # (RACER - `obj.inspected` stays False here on purpose): search
        # anyway. A miss must not make the bot omniscient - it accepts the
        # same residual risk of triggering an undetected trap that a human
        # choosing [E] Search over [I] Inspect would (see
        # state.get_prompt_lines: an undetected enemy trap still offers
        # Search, never silently blocks it). For RACER this is the whole
        # point, not a missed detection - it never tries to detect at all.
        if not obj.searched:
            return GameAction(ActionType.INTERACT)

        # obj.searched implies obj.inspected for every profile that
        # inspects first (the object can only reach "searched" after this
        # actor inspected it) - except RACER, which never inspects at all,
        # so `trap_status` for a RACER-searched object is always either
        # "unknown" (no trap) or "detected" via the trigger path (never
        # "clear_after_inspection", since it skipped that step) - meaning
        # RACER never reaches this branch on an object it searched blind,
        # consistent with "less inspection, fewer traps."
        if obj.trap_status == "clear_after_inspection" and obj.id not in self._trap_attempted:
            trap_type = self._best_trap_choice(observation, obj)
            if trap_type is not None:
                self._trap_attempted.add(obj.id)
                return GameAction(ActionType.PLACE_TRAP, trap_type=trap_type)

        return None

    def _best_trap_choice(self, observation: Observation, obj: ObservedObject) -> str | None:
        """Milestone 4F ("Strategic Trap Economy"): the highest-value trap
        type worth placing on `obj` right now, or None if nothing clears
        the bar - `Milestone_4.md`'s "since PLACE_TRAP costs about 3s,
        agents should not trap every eligible object" is the whole point
        of this method existing instead of the old "always trap if X or Y"
        check. Evaluates every trap type this actor still has (not just
        the first one in `constants.TRAP_TYPES`), so the stronger,
        currently-more-valuable trap wins when both are viable - a
        genuinely *strategic* choice, not just "whatever's first."
        """
        best_type = None
        best_value = 0.0
        for trap_type in constants.TRAP_TYPES:
            if observation.trap_inventory.get(trap_type, 0) <= 0:
                continue
            value = self._trap_value(observation, obj, trap_type)
            if value > best_value:
                best_value = value
                best_type = trap_type
        return best_type

    def _trap_value(self, observation: Observation, obj: ObservedObject, trap_type: str) -> float:
        """trap value ~= expected opponent delay - placement cost -
        strategic risk (`Milestone_4.md`'s exact framing). Every input is
        something this actor is legitimately allowed to know - its own
        search history (`items_found_here`), the object's distance to the
        exit, the opponent's position only while visible, last-seen, or
        (M4I) just revealed by their own alarm trap, and this actor's own
        remaining time/trap inventory/mission progress -
        never the opponent's inventory, target, private search/inspect
        history, or raw trap/content ground truth (`MILESTONE_3.md`'s
        third suggested trap signal, "opponent hasn't searched it," is
        still intentionally absent for exactly that reason).

        `relevance` is a deliberately uncapped sum of independent signals,
        not a 0-1 probability - two weak signals should be able to add up
        to "worth it" without needing real probabilistic modeling (out of
        scope: "sophisticated beliefs").
        """
        relevance = 0.0
        if obj.id in observation.items_found_here:
            relevance += constants.TRAP_VALUE_ITEM_SIGNAL
        exit_obj = observation.get("exit_door")
        if exit_obj is not None and math.dist(obj.center, exit_obj.center) <= constants.BOT_TRAP_NEAR_EXIT_DISTANCE:
            relevance += constants.TRAP_VALUE_NEAR_EXIT_SIGNAL
        opponent_pos = self._known_opponent_position(observation)
        if opponent_pos is not None and math.dist(obj.center, opponent_pos) <= constants.TRAP_VALUE_OPPONENT_PROXIMITY_RANGE:
            relevance += constants.TRAP_VALUE_OPPONENT_PROXIMITY_SIGNAL

        expected_delay = constants.TRAP_STUN_SECONDS[trap_type] * relevance
        # M4I: placement cost now varies per trap type (a bomb costs more
        # of the shared clock to set than an alarm), so it has to weigh
        # against *this* trap type's own cost, not one flat number.
        return expected_delay - constants.TRAP_PLACEMENT_COST[trap_type] - self._strategic_risk(observation, trap_type)

    @staticmethod
    def _known_opponent_position(observation: Observation) -> tuple[float, float] | None:
        """The opponent's position, only while legitimately known, freshest
        source first: M4I's `revealed_opponent` (a live feed while this
        actor's own alarm trap has just gone off under them) beats
        `opponent` (currently visible - same room) beats the frozen
        snapshot from the last time they were seen (`last_seen_opponent`),
        or `None` if this actor has never known at all. Never the live
        ground-truth position of an opponent this actor can't currently
        see - that's exactly the boundary `visible_other_actor` (and now
        `revealed_opponent`'s own timer) exists to enforce.
        """
        if observation.revealed_opponent is not None:
            return observation.revealed_opponent.position
        if observation.opponent is not None:
            return observation.opponent.position
        if observation.last_seen_opponent is not None:
            return observation.last_seen_opponent["position"]
        return None

    def _strategic_risk(self, observation: Observation, trap_type: str) -> float:
        """Reasons this actor's own situation makes spending a trap less
        attractive right now, independent of which object it's on - all
        self-knowledge (remaining time, own mission progress, own
        remaining inventory of this trap type, and - Milestone 4G - this
        actor's own agent profile)."""
        risk = self.profile.trap_value_bias
        if observation.mission_time_remaining < constants.TRAP_STRATEGIC_RISK_LOW_TIME_THRESHOLD:
            risk += constants.TRAP_STRATEGIC_RISK_LOW_TIME_PENALTY
        if observation.has_all_required_items():
            risk += constants.TRAP_STRATEGIC_RISK_MISSION_READY_PENALTY
        if observation.trap_inventory.get(trap_type, 0) <= 1:
            risk += constants.TRAP_STRATEGIC_RISK_LAST_ONE_PENALTY
        return risk

    # -- target selection -------------------------------------------------

    def _select_target(self, observation: Observation) -> ObservedObject | None:
        """Nearest unsearched object, plus (M4B) any locked non-exit door
        this actor already holds/knows what unlocks - a door is otherwise
        never a target (see _target_resolved/_decide_interaction), but it
        must become one here or the bot would stand next to a room it can
        now open without ever pathing to the door itself.

        A locked *non-door* object (the safe) this actor can't currently
        open is excluded outright - not merely deprioritized. `_search`
        never marks a locked object searched (see state.py), so if
        `_pick_target`'s stickiness ever committed to one, it would stay
        "unresolved" forever and permanently starve every other candidate
        (a real bug found while headless-testing M4B: the bot walked
        straight to the safe at tick 97 of a 60s run and never moved
        again). Once this actor gains the code, `can_unlock` flips true and
        the safe becomes selectable normally - locks never re-lock, so a
        target that passed this filter can't fail it later.
        """
        candidates = [
            obj
            for obj in observation.objects
            if obj.interactable
            and obj.type != "door"
            and not obj.searched
            and not (obj.locked and not obj.can_unlock)
        ]
        candidates += [
            obj
            for obj in observation.objects
            if obj.type == "door" and obj.id != "exit_door" and obj.locked and obj.can_unlock
        ]
        # M4B: a whole room can now be sealed off by a locked door, not
        # just an individual locked object - straight-line distance alone
        # can no longer tell "closest" from "actually reachable" (an object
        # on the far side of a wall from a locked room can measure closer
        # than one two rooms away by the corridor). Filter to candidates
        # this actor's own nav_grid can currently path to; a real bug found
        # headless-testing M4B: the bot picked the (unreachable) nearest
        # object behind a locked archives door, failed to path to it,
        # dropped it, and picked the exact same nearest-but-unreachable
        # object again next tick - forever, never touching anything it
        # could actually reach.
        candidates = [obj for obj in candidates if self._reachable(observation, obj)]
        if not candidates:
            return None
        return min(candidates, key=lambda o: math.dist(observation.position, o.center))

    @staticmethod
    def _reachable(observation: Observation, obj: ObservedObject) -> bool:
        approach = observation.nav_grid.nearest_approach_point(
            obj.collision_rect, constants.INTERACTION_DISTANCE, observation.position
        )
        if approach is None:
            return False
        return observation.nav_grid.find_path(observation.position, approach) is not None

    # -- navigation ---------------------------------------------------------

    def _compute_path(self, observation: Observation, target_obj: ObservedObject) -> None:
        approach = observation.nav_grid.nearest_approach_point(
            target_obj.collision_rect, constants.INTERACTION_DISTANCE, observation.position
        )
        if approach is None:
            # Fallback: no walkable cell found in range (shouldn't happen on
            # this level's layout) - aim at the object's own cell anyway.
            approach = target_obj.center

        path = observation.nav_grid.find_path(observation.position, approach)
        self._path = path or []
        self._path_index = 0
        self._path_target_id = target_obj.id
        observation.log_event("PATH_COMPUTED", object_id=target_obj.id, waypoints=len(self._path))

    def _follow_path(self, observation: Observation) -> GameAction:
        ax, ay = observation.position
        while self._path_index < len(self._path):
            wx, wy = self._path[self._path_index]
            dist = math.dist((ax, ay), (wx, wy))
            if dist > constants.WAYPOINT_ARRIVAL_DISTANCE:
                return GameAction(ActionType.MOVE, direction=((wx - ax) / dist, (wy - ay) / dist))
            self._path_index += 1

        return GameAction(ActionType.WAIT)
