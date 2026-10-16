"""Central game state and the rules that mutate it.

Every GameAction, regardless of which actor or controller produced it, is
applied here through `apply_action(actor_id, action, dt)`. This is the
only place world state changes. Interaction rules return an
InteractionResult - a structured record of what happened - which is what
drives both the (human-facing) status message and the event log.

Milestone 3A: the game now has two actors (black_spy, white_spy) sharing
the same movement/collision/search/inventory rules. Search ownership and
inventory are tracked per actor; door state and mission-item *contents*
remain shared world truth (see Actor.inventory and object_states).

Milestone 3B: adds INSPECT / PLACE_TRAP / DISARM. Trap *truth* (whether an
object has an armed trap, its type and owner) is shared world state, same
as door state - but trap *knowledge* (has this actor inspected the object?
did their inspection detect the trap?) is tracked per actor, same pattern
as searched_by. See `_object_entry` for the per-object shape and
`_inspect`/`_search`/`_place_trap`/`_disarm` for the rules themselves.

Milestone 4D ("Time Economy"): every timed action now costs the acting
actor real time, via `_charge_time` setting `Actor.action_cooldown` -
mechanically the exact same non-blocking timer `stunned_timer` already
was (blocks every action, including movement, until it expires; never
`time.sleep()`, never blocks the game loop itself). A single shared clock,
`mission_time_remaining`, depletes by real elapsed `dt` every `update()`
call regardless of what either actor is doing - a busy/costly action
doesn't pause that clock, which is what actually makes the cost "real":
three seconds spent placing a trap are three seconds gone from the shared
180s budget, not three seconds paid from some separate pool. See
`constants.TIME_COST_*` for the exact per-action costs.

Milestone 4E ("Competitive / Cooperative Victory"): not zero-sum - both
actors may succeed in the same episode. `_attempt_exit` now also enforces
the deadline it merely displayed under M4D: once `mission_time_remaining`
hits zero, escaping is permanently blocked for whoever hasn't already
gotten out, regardless of how complete their own mission is (that gap is
deliberate - "collected every required item" and "got out before the
clock ran out" are scored as two separate facts, see src/scoring.py, not
one implying the other). `finish_order`/`escape_snapshot` record the only
two facts about an escape nothing else could reconstruct later (who got
out in what order, how much time was left); every other scoring input
(own mission success, whether the other actor also escaped, whether the
deadline has passed) is derived live from state that already exists.

Milestone 4H ("Episode / Replay Foundation"): `episode_id` and
`mission_time_total` are set once at `new_game()` and never change -
identifying facts about *this run*, as opposed to `mission_time_remaining`
which does. `escape_elapsed` records each actor's `elapsed_time` at the
moment they escaped (alongside `escape_snapshot`'s *remaining* time,
which M4E's scoring already needed) - together with everything already
on `GameState`, this is what `src/episode.py`'s `build_episode_record`
reads to produce `Milestone_4.md`'s exact required episode summary.
RNG was already fully centralized before this milestone (M3B's
`self.rng`/M4B's `dependency_generation.generate_dependencies(rng=...)`
are the only two `random.Random` consumers anywhere in `src/`, both fed
from the same seeded stream `new_game()` creates) - confirmed, not
rebuilt, by a repo-wide grep finding no other call site.

Milestone 4I ("Trap Effects & Gameplay Feedback"): the two M3B placeholder
trap types (mechanical_trap/alarm_trap) became four distinct ones (bomb,
electric, snare, alarm), each with its own tuning
(`constants.TRAP_STUN_SECONDS`/`TRAP_PLACEMENT_COST`) and its own
short thematic trigger message (`constants.TRAP_TRIGGER_MESSAGES`) instead
of a generic "You're stunned." `_trigger_trap` is otherwise unchanged
logic - same single-use consumption, same per-actor stun timer - except
for alarm's one genuinely new mechanic: triggering it also starts a
timer on the *owner's* own `Actor.alarm_reveal_timer`, which
`observation.py` surfaces as that owner's `revealed_opponent` - never a
second, uncontrolled way for a controller to read the victim's position.
Everything visual (the explosion/arc/snare/beacon animations themselves)
is the renderer's job (`src/effects.py`), reacting to the same
`TRAP_TRIGGERED` event this method already logged before M4I existed -
no new gameplay-truth channel was needed for presentation.
"""

import math
import random
import uuid
from dataclasses import dataclass, field

from src import constants
from src.actions import ActionType, GameAction, InteractionResult
from src.actor import Actor
from src.dependency_generation import DependencyPlacement, apply_placement, generate_dependencies
from src.interactable import GameObject
from src.interaction_range import nearest_interactable_id
from src.items import ItemCatalog
from src.level import Level
from src.navigation import NavGrid, build_nav_grid
from src.encounters import update_encounters
from src.mission_clock import ACTION_BUDGET, SIMULATION_TIME, time_remaining_for, update_budgets


@dataclass
class GameState:
    level: Level
    items: ItemCatalog
    actors: dict[str, Actor]
    nav_grid: NavGrid
    object_states: dict[str, dict] = field(default_factory=dict)
    door_states: dict[str, str] = field(default_factory=dict)
    current_interaction_target: dict[str, str | None] = field(default_factory=dict)
    status_message: str = ""
    status_message_success: bool = True
    status_message_timer: float = 0.0
    mission_complete: dict[str, bool] = field(default_factory=dict)
    event_log: list[dict] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)
    seed: int = 0
    dependency_placement: DependencyPlacement | None = None  # M4B - see src/dependency_generation.py
    elapsed_time: float = 0.0  # M4C - game-time, for last_seen_opponent's timestamp
    _opponent_visible: dict[str, bool] = field(default_factory=dict)  # M4C bookkeeping only - not in Observation
    mission_time_remaining: float = constants.MISSION_TIME  # M4D - one shared clock, not per-actor
    # M4E: the only two facts about an escape that genuinely can't be
    # recovered from other state later - who got out in what order, and
    # how much of the shared clock was left when they did. Everything else
    # scoring needs (own mission success, whether the other actor also
    # escaped, whether the deadline has passed) is derived live from
    # state that already exists - see src/scoring.py.
    finish_order: list[str] = field(default_factory=list)
    escape_snapshot: dict[str, float] = field(default_factory=dict)  # actor_id -> mission_time_remaining at escape
    # M4H: identifying facts about *this run*, set once and never changed -
    # as opposed to mission_time_remaining/elapsed_time, which do.
    episode_id: str = ""
    mission_time_total: float = constants.MISSION_TIME  # the configured deadline, unlike mission_time_remaining
    escape_elapsed: dict[str, float] = field(default_factory=dict)  # actor_id -> elapsed_time at escape
    # M5F.2: whether SEND_MESSAGE is a legal action this episode at all - a
    # fixed, per-episode condition (spec section 13's OFF/ON), never
    # toggled mid-episode. `messages` is engine-truth communication state
    # (spec section 5), append-only, in send order - every message ever
    # sent this episode, to either actor; `build_observation` filters this
    # down to each actor's own bounded recent-received window (spec
    # section 21). Plain dicts (not a dataclass), matching `event_log`'s
    # own shape, with keys message_id/sender_actor_id/recipient_actor_id/
    # game_time/text (spec section 5's required per-message fields).
    communication_enabled: bool = False
    messages: list[dict] = field(default_factory=list)
    # M5F.2.2 spec section 1: a second, mutually-exclusive representation
    # of the same underlying communication capability - `False` (every
    # pre-M5F.2.2 caller's implicit value) is FULL_COST, byte-identical to
    # M5F.2/M5F.2.1 (SEND_MESSAGE stays a standalone legal action - spec
    # section 3). `True` is ZERO_DECISION_COST: SEND_MESSAGE stops being
    # offered as a standalone action (see `enumerate_legal_actions`) and
    # an optional message may instead accompany any physical action's own
    # GameAction (`GameAction.accompanying_message`), delivered via this
    # same `_do_send_message` engine path but never charged its own time
    # cost (see `_do_send_message`'s `charge_time` param).
    zero_decision_cost_messaging: bool = False
    # Bugfix live flow V2, Part D: spy encounters (src/encounters.py). Off by
    # default - only the workshop live game turns it on - so every recorded
    # run keeps its exact events and state. `fight_rng` is the engine's own
    # fight RNG (seeded from `seed` in new_game), never the shared `rng`.
    # `recorded_fight_winners`: a replay feeds recorded winners here so a
    # fight is never re-rolled.
    encounters_enabled: bool = False
    fight_rng: random.Random = field(default_factory=random.Random)
    encounter_cooldown: float = 0.0
    encounter_armed: bool = True
    recorded_fight_winners: list[str] | None = None
    fights: list[dict] = field(default_factory=list)
    # Bugfix mission time budget V2: which clock the deadline runs on
    # (src/mission_clock.py). SIMULATION_TIME keeps every recorded run exact;
    # ACTION_BUDGET (live workshop) gives each spy its own budget that does
    # not deplete while it only waits for its own model.
    # Live workshop feedback (off by default - recorded runs unchanged):
    # `lives_enabled`: each trap triggered under a spy costs one of its
    # SPY_LIVES; at 0 the spy is eliminated (failed, out of the match).
    # `escaped_leave_world`: a spy that escaped has left the building - the
    # other spy no longer sees it (it is not an opponent to approach).
    lives_enabled: bool = False
    lives: dict[str, int] = field(default_factory=dict)
    trap_hits: dict[str, int] = field(default_factory=dict)
    eliminated: dict[str, bool] = field(default_factory=dict)
    escaped_leave_world: bool = False
    mission_clock_mode: str = SIMULATION_TIME
    mission_budget: dict[str, float] = field(default_factory=dict)
    budget_hold: set[str] = field(default_factory=set)
    # Bugfix live match flow V3 (all off by default - recorded runs unchanged):
    # `own_traps_dangerous`: a spy's own armed trap triggers on it too (no
    # owner immunity); each spy remembers the traps it placed
    # (`self_placed_traps[actor][target]`), trap events carry ids/owner/
    # before-after facts, and an owner disarming a trap it just placed
    # without leaving the object is logged as SELF_REVERSAL_NO_STATE_CHANGE.
    # `live_lifecycle`: a spy whose time ran out on the shared clock is out of
    # play (TIMED_OUT) - no more decisions - exactly like ACTION_BUDGET.
    # `external_mission_clock`: the shared countdown is driven by the live
    # runner's wall clock (`advance_mission_clock`), not by frame dt.
    own_traps_dangerous: bool = False
    self_placed_traps: dict[str, dict[str, dict]] = field(default_factory=dict)
    trap_seq: int = 0
    live_lifecycle: bool = False
    external_mission_clock: bool = False
    # V3 Part D (live only, src/exit_geometry.py): after a successful exit the
    # spy walks through the open doorway to the exterior zone on its own -
    # no further model decision - then it has left the building.
    exit_walkout_path: list[tuple[float, float]] | None = None
    exit_walking: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    left_building: dict[str, bool] = field(default_factory=dict)
    # Bugfix spawn/combat/visual polish (live only, src/spawns.py): the spawn
    # points in use, and the fight rule "loser KO'd for FIGHT_KO_SECONDS of
    # match time, then respawns at a safe point" (`fight_ko_respawn`) instead
    # of the V2 3 s stun. `knocked_out[actor]` = the match-clock reading at
    # which it respawns; `ko_count[actor]` lets a live agent drop an answer
    # to a request made before its KO.
    spawn_record: dict | None = None
    spawn_points: dict[str, tuple[float, float]] = field(default_factory=dict)
    fight_ko_respawn: bool = False
    # live trap rules (see constants.LIVE_TIME_COST_INSPECT / DISARM_SUCCESS_PROBABILITY);
    # disarm outcomes come from their own seeded RNG, never the shared `rng`
    live_trap_rules: bool = False
    # live: the opponent coming close is a decision point (src/encounter_intents.encounter_alert)
    encounter_alerts: bool = False
    # live "one copy" rule: there is ONE of each required item. The first spy
    # to search its object takes it; later the object is empty. The fight
    # winner takes one required item from the loser; a spy eliminated by a
    # fight hands everything to the winner, one eliminated by a trap leaves
    # its required items on that object. `known_opponent_items[actor]` = the
    # required items this actor knows the opponent holds (it found their place
    # empty, or lost them in a fight) - actor knowledge, never hidden truth.
    single_copy_items: bool = False
    known_opponent_items: dict[str, set] = field(default_factory=dict)
    # live, one copy rule + live lifecycle: actor -> the MISSION_IMPOSSIBLE
    # record once the opponent escaped with a required item this actor lacks
    # (that copy has left the building for good). Terminal, scored as a failure.
    mission_impossible: dict[str, dict] = field(default_factory=dict)
    # live safe puzzles (src/puzzles.py): "none" (recorded rule), "split", "riddle"
    puzzle_mode: str = "none"
    puzzle: dict | None = None
    inscriptions: dict[str, str] = field(default_factory=dict)  # object_id -> text (riddle)
    seen_inscriptions: dict[str, dict[str, str]] = field(default_factory=dict)  # actor -> {object_id: text}
    code_attempts: dict[str, int] = field(default_factory=dict)
    disarm_rng: random.Random = field(default_factory=random.Random)
    knocked_out: dict[str, float] = field(default_factory=dict)
    ko_count: dict[str, int] = field(default_factory=dict)

    def enable_lives(self) -> None:
        self.lives_enabled = True
        self.lives = {a: constants.SPY_LIVES for a in self.actors}
        self.eliminated = {a: False for a in self.actors}

    def out_of_play(self, actor_id: str) -> bool:
        """Failed and no longer acting: out of time (action budget, or the
        shared clock in live play) or out of lives."""
        return bool(self.eliminated.get(actor_id)) or actor_id in self.mission_impossible or (
            (self.mission_clock_mode == ACTION_BUDGET or self.live_lifecycle)
            and not self.mission_complete.get(actor_id) and self.time_remaining_for(actor_id) <= 0)

    def lifecycle(self, actor_id: str) -> str:
        """V3 explicit actor lifecycle (a pure query): ESCAPED, ELIMINATED and
        TIMED_OUT are terminal; ACTIVE and OBJECTIVES_COMPLETE (every required
        item held, still inside the building) may still decide."""
        if self.mission_complete.get(actor_id):
            return constants.LIFECYCLE_ESCAPED
        if self.eliminated.get(actor_id):
            return constants.LIFECYCLE_ELIMINATED
        if actor_id in self.mission_impossible:
            return constants.LIFECYCLE_MISSION_IMPOSSIBLE
        if self.time_remaining_for(actor_id) <= 0:
            return constants.LIFECYCLE_TIMED_OUT
        if self.actors[actor_id].inventory.has_all(self.items.required):
            return constants.LIFECYCLE_OBJECTIVES_COMPLETE
        return constants.LIFECYCLE_ACTIVE

    def advance_mission_clock(self, seconds: float) -> None:
        """Live wall clock (external_mission_clock): the runner passes the real
        unpaused wall time since the previous frame (x speed)."""
        if seconds <= 0:
            return
        before = self.mission_time_remaining
        self.mission_time_remaining = max(0.0, before - seconds)
        if before > 0 and self.mission_time_remaining == 0:
            self._log_event("MISSION_TIME_EXPIRED", actor_id=None, clock="live_wall_clock")
            for actor_id in self.actors:
                if not self.mission_complete.get(actor_id) and not self.eliminated.get(actor_id) \
                        and actor_id not in self.mission_impossible:
                    self._log_event("ACTOR_TIMED_OUT", actor_id=actor_id)

    def inspect_cost(self) -> float:
        return constants.LIVE_TIME_COST_INSPECT if self.live_trap_rules else constants.TIME_COST_INSPECT

    def disarm_success_probability(self, trap_type: str) -> float | None:
        """None: disarming always works (recorded rules)."""
        return constants.DISARM_SUCCESS_PROBABILITY[trap_type] if self.live_trap_rules else None

    def match_clock(self) -> float:
        """Match seconds elapsed on the clock the deadline runs on (the shared
        countdown - wall clock in live play - or world time with per-spy budgets)."""
        if self.mission_clock_mode == ACTION_BUDGET:
            return self.elapsed_time
        return self.mission_time_total - self.mission_time_remaining

    # -- live "one copy" rule ----------------------------------------------------------
    @property
    def contested_items(self) -> tuple[str, ...]:
        return tuple(self.items.required) if self.single_copy_items else ()

    def _take_single_copy(self, actor: Actor, obj: GameObject, entry: dict) -> tuple[list[str], list[str]]:
        """Items this search picks up, and required items found already taken."""
        taken = entry.setdefault("taken", set())
        dropped = entry.setdefault("dropped", [])
        found, gone = [], []
        for item_id in list(obj.contains) + list(dropped):
            if item_id in self.contested_items:
                if item_id in dropped:
                    dropped.remove(item_id)
                elif item_id in taken:
                    if not actor.inventory.has(item_id):
                        gone.append(item_id)
                    continue
                taken.add(item_id)
            if actor.inventory.add(item_id):
                found.append(item_id)
        if gone:
            other = self._other_id(actor.id)
            for item_id in gone:
                if self.actors[other].inventory.has(item_id):
                    self.known_opponent_items.setdefault(actor.id, set()).add(item_id)
            self._log_event("ITEM_ALREADY_TAKEN", actor_id=actor.id, object_id=obj.id, items=sorted(gone))
        for item_id in found:
            self.known_opponent_items.get(actor.id, set()).discard(item_id)
        return found, gone

    def _other_id(self, actor_id: str) -> str:
        return next(a for a in self.actors if a != actor_id)

    def steal_item(self, winner_id: str, loser_id: str, everything: bool = False) -> list[str]:
        """Fight result under the one-copy rule: the winner takes one required
        item from the loser (all of them when the loser is eliminated)."""
        if not self.single_copy_items:
            return []
        winner, loser = self.actors[winner_id], self.actors[loser_id]
        held = [i for i in self.contested_items if loser.inventory.has(i) and not winner.inventory.has(i)]
        moved = held if everything else held[:1]
        for item_id in moved:
            loser.inventory.items.discard(item_id)
            winner.inventory.add(item_id)
            self.known_opponent_items.setdefault(loser_id, set()).add(item_id)
            self.known_opponent_items.setdefault(winner_id, set()).discard(item_id)
            self._log_event("ITEM_STOLEN", actor_id=winner_id, victim_id=loser_id, item_id=item_id)
        return moved

    def drop_items_at(self, actor_id: str, object_id: str) -> list[str]:
        """A spy eliminated by a trap leaves its required items on that object."""
        if not self.single_copy_items:
            return []
        actor = self.actors[actor_id]
        held = [i for i in self.contested_items if actor.inventory.has(i)]
        entry = self._object_entry(object_id)
        for item_id in held:
            actor.inventory.items.discard(item_id)
            entry.setdefault("dropped", []).append(item_id)
        if held:
            self._log_event("ITEMS_DROPPED", actor_id=actor_id, object_id=object_id, items=held)
        return held

    def lose_life(self, actor_id: str, cause: str) -> bool:
        """One life lost (lives rule); True if that was the last one (eliminated)."""
        if not self.lives_enabled or self.eliminated.get(actor_id):
            return False
        self.lives[actor_id] = max(0, self.lives.get(actor_id, constants.SPY_LIVES) - 1)
        self._log_event("LIFE_LOST", actor_id=actor_id, lives_left=self.lives[actor_id], cause=cause)
        if self.lives[actor_id] == 0:
            self.eliminated[actor_id] = True
            self.actors[actor_id].moving = False
            self._log_event("ACTOR_ELIMINATED", actor_id=actor_id, cause="no_lives_left")
            return True
        return False

    def knock_out(self, actor_id: str, seconds: float) -> None:
        """Fight loser (fight_ko_respawn): inactive for `seconds` of match time."""
        actor = self.actors[actor_id]
        self.knocked_out[actor_id] = self.match_clock() + seconds
        self.ko_count[actor_id] = self.ko_count.get(actor_id, 0) + 1
        actor.stunned_timer = seconds + 1.0  # held above 0 until the respawn (see _advance_knockouts)
        actor.moving = False
        self._log_event("ACTOR_KNOCKED_OUT", actor_id=actor_id, seconds=seconds,
                        position=[round(actor.x, 1), round(actor.y, 1)], ko_count=self.ko_count[actor_id])

    def ko_seconds_left(self, actor_id: str) -> float | None:
        release = self.knocked_out.get(actor_id)
        return None if release is None else max(0.0, release - self.match_clock())

    def _advance_knockouts(self) -> None:
        from src.spawns import safe_respawn_point

        for actor_id in list(self.knocked_out):
            actor = self.actors[actor_id]
            left = self.ko_seconds_left(actor_id)
            if left > 1e-6 and not self.mission_complete.get(actor_id):
                actor.stunned_timer = left + 1.0
                actor.moving = False
                continue
            del self.knocked_out[actor_id]
            before = (round(actor.x, 1), round(actor.y, 1))
            actor.x, actor.y = safe_respawn_point(self, actor_id)
            actor.stunned_timer = 0.0
            actor.moving = False
            self._log_event("ACTOR_RESPAWNED", actor_id=actor_id, from_position=list(before),
                            position=[round(actor.x, 1), round(actor.y, 1)], room=self.level.room_id_at(actor.x, actor.y))

    def own_trap_record(self, actor_id: str, object_id: str) -> dict | None:
        """This actor's memory of the trap it placed on `object_id`, with its
        current status as the owner can know it (own traps are owner-visible)."""
        record = self.self_placed_traps.get(actor_id, {}).get(object_id)
        if record is None:
            return None
        trap = self.trap_on(object_id)
        armed = trap is not None and trap.get("id") == record["trap_id"]
        status = constants.OWN_TRAP_ARMED if armed else record.get("status_if_gone", "GONE")
        return {**record, "armed": armed, "status": status}

    def time_remaining_for(self, actor_id: str) -> float:
        """This actor's own remaining mission time under the active clock mode."""
        return time_remaining_for(self, actor_id)

    @classmethod
    def new_game(
        cls,
        level: Level,
        items: ItemCatalog,
        seed: int | None = None,
        mission_time: float | None = None,
        communication_enabled: bool = False,
    ) -> "GameState":
        """M4A.3 (`M4A_3_LIVE_PLAY_DIAGNOSTICS_DOOR_POLISH.md` section 6): a
        live-play failure report is useless without the seed that produced
        it, so an unspecified seed is no longer silently `random.Random()`
        system entropy that's gone the moment the process exits - it's
        resolved to a concrete int up front and stored on the state itself
        (`Game.__init__` prints it at startup; see also `[LAYOUT]`), so any
        run can be named and, if needed, replayed exactly.

        `mission_time` (M4D) is the "configurable" half of
        `Milestone_4.md`'s "Initial configurable mission deadline" -
        defaults to `constants.MISSION_TIME` (180s) when unspecified, same
        optional-with-a-default pattern as `seed`.
        """
        resolved_seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
        rng = random.Random(resolved_seed)
        resolved_mission_time = mission_time if mission_time is not None else constants.MISSION_TIME
        # M4H: identifies *this run* for src/episode.py's EpisodeRecord -
        # a plain uuid4, not derived from the seed (two runs of the same
        # seed are still two different episodes).
        episode_id = str(uuid.uuid4())

        # M4B: which specific object holds the archive key / keycard /
        # safe code / passport is decided fresh per seed here, mutating
        # this freshly-loaded Level in place before anything else reads
        # it - see dependency_generation.py. `level`'s lock topology itself
        # (which doors are gated, by what) is static level_01.json data;
        # only the placement is seeded.
        placement = generate_dependencies(level, items, rng)
        apply_placement(level, placement)

        door_states = {obj.id: obj.default_state for obj in level.objects if obj.type == "door"}
        nav_grid = build_nav_grid(level, door_states)

        bx, by = level.player_spawn
        wx, wy = level.white_spy_spawn
        actors = {
            constants.BLACK_SPY_ID: Actor(id=constants.BLACK_SPY_ID, x=bx, y=by),
            constants.WHITE_SPY_ID: Actor(id=constants.WHITE_SPY_ID, x=wx, y=wy),
        }

        state = cls(
            level=level,
            items=items,
            actors=actors,
            nav_grid=nav_grid,
            door_states=door_states,
            mission_complete={actor_id: False for actor_id in actors},
            rng=rng,
            seed=resolved_seed,
            dependency_placement=placement,
            mission_time_remaining=resolved_mission_time,
            episode_id=episode_id,
            mission_time_total=resolved_mission_time,
            communication_enabled=communication_enabled,
            fight_rng=random.Random(f"spy-fight:{resolved_seed}"),
            disarm_rng=random.Random(f"disarm:{resolved_seed}"),
        )

        # M5F.3A (spec sections 1-7, 32-33, 37): traps present at
        # initialization, independent of either spy - reuses the exact
        # trap dict shape `_place_trap` writes (`{"type", "owner_id",
        # "armed"}`), just written directly into object_states instead of
        # going through a PLACE_TRAP action, so no TRAP_PLACED event fires
        # and neither spy gets placement credit (spec section 4/11). Level
        # data (not Python) decides which objects get one - level_01.json
        # itself declares none, so normal gameplay and every regression
        # test outside this milestone's own fixtures is unaffected; only
        # a caller that explicitly sets `level.initial_traps` (M5F.3A's
        # own scenario builders) ever seeds one.
        for spec in level.initial_traps:
            target_id = spec["target_id"]
            trap_type = spec["trap_type"].lower()
            entry = state._object_entry(target_id)
            entry["trap"] = {"type": trap_type, "owner_id": constants.ENVIRONMENT_OWNER_ID, "armed": True}
            state._log_event(
                "ENVIRONMENTAL_TRAP_INITIALIZED",
                trap_id=spec.get("id", f"{target_id}_trap"),
                trap_owner=constants.ENVIRONMENT_OWNER_ID,
                trap_type=trap_type,
                target_id=target_id,
                initially_present=True,
            )

        return state

    def get_prompt_lines(self, obj: GameObject, actor_id: str = constants.BLACK_SPY_ID) -> list[str]:
        """HUD prompt strings for `obj`, from `actor_id`'s point of view.
        Milestone 3.2's exact table (see Milestone_32.md section 1):

            no trap             -> Search, then Inspect / Place trap
            own trap            -> Search, Disarm (removes it - a known
                                    quantity, no need to re-detect it)
            enemy trap, unknown -> Search, Inspect (never Disarm - that
                                    would leak the trap's existence)
            enemy trap, detected-> Search, Disarm

        Must never expose [D] Disarm for a trap this actor hasn't detected
        and doesn't own - that would leak hidden engine truth through the UI.
        """
        actor = self.actors[actor_id]

        if obj.id == "exit_door":
            return ["[E] Escape"] if actor.inventory.has_all(self.items.required) else [obj.prompt_text]

        if not obj.trappable:
            return [obj.prompt_text]

        verb = (obj.interaction or "search").capitalize()
        search_line = f"[E] {verb} {obj.label}"

        trap = self.trap_on(obj.id)
        if trap is None:
            lines = [search_line]
            if not self.is_inspected_by(obj.id, actor_id):
                lines.append("[I] Inspect")
            elif any(count > 0 for count in actor.trap_inventory.values()):
                lines.append("[T] Place trap")
            return lines

        knows_trap = trap["owner_id"] == actor_id or self.detected_trap_type(obj.id, actor_id) == trap["type"]
        return [search_line, "[D] Disarm"] if knows_trap else [search_line, "[I] Inspect"]

    def _check_mission_impossible(self) -> None:
        """Live, one copy of each item: a spy whose opponent escaped holding a
        required item it lacks can never finish - that copy left with the
        opponent. It becomes MISSION_IMPOSSIBLE at once (terminal, no more
        decisions, scored as a failure). Runs at the start of `update`, after
        the tick's actions, so every action decided before the escape was
        applied, and before the clock moves: the event carries the escape
        tick's mission time."""
        if not (self.single_copy_items and self.live_lifecycle) or not any(self.mission_complete.values()):
            return
        required = list(self.items.required)
        for actor_id, actor in self.actors.items():
            if self.mission_complete.get(actor_id) or self.eliminated.get(actor_id) \
                    or actor_id in self.mission_impossible or self.out_of_play(actor_id):
                continue
            for other_id, other in self.actors.items():
                if other_id == actor_id or not self.mission_complete.get(other_id):
                    continue
                gone = [i for i in required if not actor.inventory.has(i) and other.inventory.has(i)]
                if not gone:
                    continue
                record = {"opponent_id": other_id, "items": gone, "reason": constants.MISSION_IMPOSSIBLE_REASON,
                          "mission_time_remaining": round(self.time_remaining_for(actor_id), 3),
                          "game_time": round(self.elapsed_time, 3)}
                self.mission_impossible[actor_id] = record
                actor.moving = False
                self._log_event("MISSION_IMPOSSIBLE", actor_id=actor_id, **record)
                break

    def update(self, dt: float) -> None:
        self._check_mission_impossible()
        self.elapsed_time += dt
        # M4D: the shared mission clock only ever depletes by real elapsed
        # dt - never sleep()'d, never directly decremented by an action's
        # cost. It keeps ticking through a busy/stunned actor's cooldown
        # too, which is exactly what makes a costly action "expensive": the
        # clock doesn't wait for you to finish. Clamped, not allowed to run
        # negative (Milestone_4.md leaves what happens at 0 to M4E's
        # deadline/scoring rules - this milestone only owns the countdown).
        if self.mission_clock_mode == ACTION_BUDGET:
            update_budgets(self, dt)
        elif not self.external_mission_clock:
            self.mission_time_remaining = max(0.0, self.mission_time_remaining - dt)
        if self.knocked_out:
            self._advance_knockouts()
        current_rooms: dict[str, str | None] = {}

        for actor_id, actor in self.actors.items():
            if actor.stunned_timer > 0:
                actor.stunned_timer = max(0.0, actor.stunned_timer - dt)
                if actor.stunned_timer == 0:
                    self._log_event("ACTOR_RECOVERED", actor_id=actor_id)

            if actor.action_cooldown > 0:
                actor.action_cooldown = max(0.0, actor.action_cooldown - dt)

            if actor.alarm_reveal_timer > 0:
                actor.alarm_reveal_timer = max(0.0, actor.alarm_reveal_timer - dt)
                if actor.alarm_reveal_timer == 0:
                    self._log_event("ALARM_REVEAL_ENDED", actor_id=actor_id)

            if self.mission_complete[actor_id]:
                continue
            self.current_interaction_target[actor_id] = self._find_nearest_interactable(actor)
            current_rooms[actor_id] = self._update_room_memory(actor_id, actor)

        if self.own_traps_dangerous:
            # a trap stays "just placed" only while its owner is still at it
            for actor_id, records in self.self_placed_traps.items():
                for target_id, record in records.items():
                    if record["fresh"] and self.current_interaction_target.get(actor_id) != target_id:
                        record["fresh"] = False
        self._update_opponent_visibility(current_rooms)
        if self.encounters_enabled:
            update_encounters(self, dt, current_rooms)

        if self.exit_walking:
            self._advance_exit_walks(dt)

        if self.status_message_timer > 0:
            self.status_message_timer -= dt
            if self.status_message_timer <= 0:
                self.status_message_timer = 0
                self.status_message = ""

    def _advance_exit_walks(self, dt: float) -> None:
        """V3 exit transition: an escaped spy crosses the open doorway to the
        exterior zone at walking speed (scripted - the door is open and the
        path is the doorway itself), then it has left the building."""
        for actor_id, path in list(self.exit_walking.items()):
            actor = self.actors[actor_id]
            step = constants.PLAYER_SPEED * dt
            while path and step > 0:
                tx, ty = path[0]
                dx, dy = tx - actor.x, ty - actor.y
                dist = math.hypot(dx, dy)
                if dist <= step:
                    actor.x, actor.y = tx, ty
                    step -= dist
                    path.pop(0)
                else:
                    actor.x += dx / dist * step
                    actor.y += dy / dist * step
                    actor.update_facing((dx, dy))
                    step = 0
            actor.moving = bool(path)
            if not path:
                del self.exit_walking[actor_id]
                self.left_building[actor_id] = True
                self._log_event("ACTOR_LEFT_BUILDING", actor_id=actor_id, position=[round(actor.x, 1), round(actor.y, 1)],
                                room=self.level.room_id_at(actor.x, actor.y))

    def apply_action(self, actor_id: str, action: GameAction, dt: float) -> None:
        if self.mission_complete[actor_id]:
            return
        if self.mission_clock_mode == ACTION_BUDGET and self.mission_budget.get(actor_id, 0.0) <= 0:
            self.actors[actor_id].moving = False
            return  # out of mission time: this spy has failed and no longer acts
        if self.eliminated.get(actor_id):
            self.actors[actor_id].moving = False
            return  # out of lives: this spy has failed and no longer acts
        if actor_id in self.mission_impossible:
            self.actors[actor_id].moving = False
            return  # its mission can never be completed: failed, no longer acts

        actor = self.actors[actor_id]

        if actor.is_stunned or actor.is_busy:
            # M4D: `is_busy` (a timed action's cost) blocks everything the
            # same way `is_stunned` already did - the exact "reuse
            # non-blocking timed actions" pattern Milestone_4.md asks for,
            # not a new mechanism. Movement is included deliberately: an
            # actor mid-search/mid-placing-a-trap isn't free to also walk
            # away this frame.
            actor.moving = False
            return

        # M5F.2.2 spec section 20's required deterministic order: any
        # accompanying message is delivered BEFORE the physical action
        # begins/applies, at this same decision boundary - regardless of
        # `action.type` (a MOVE's accompanying message is delivered here
        # even though the move itself may still take many further ticks
        # to resolve; a message is never re-delivered on those later
        # ticks since `ActionExecutor` only ever attaches it to the one
        # GameAction returned from `start()`, never from `continue_()`).
        # Never charged its own time cost (`charge_time=False`) - the
        # physical action's own cost below is the only cost paid, keeping
        # "zero DECISION cost" from silently becoming a fractional TIME
        # cost instead (spec section 35 forbids introducing new
        # communication economics here).
        if action.accompanying_message is not None:
            self._do_send_message(
                actor_id, action.accompanying_message.get("recipient_id"), action.accompanying_message.get("text"),
                charge_time=False,
            )

        if action.type is ActionType.MOVE and action.direction is not None:
            actor.move(action.direction, dt, self.level.solid_rects(self.door_states))
        elif action.type is ActionType.INTERACT:
            self._interact(actor_id, action.target_id, code=getattr(action, "code", None))
        elif action.type is ActionType.INSPECT:
            self._do_inspect(actor_id, action.target_id)
        elif action.type is ActionType.PLACE_TRAP:
            self._do_place_trap(actor_id, action.trap_type, action.target_id)
        elif action.type is ActionType.DISARM:
            self._do_disarm(actor_id, action.target_id)
        elif action.type is ActionType.SEND_MESSAGE:
            self._do_send_message(actor_id, action.target_id, action.message)
        elif action.type is ActionType.WAIT:
            actor.moving = False

    # -- interaction rules --------------------------------------------------

    def _target_object(self, actor_id: str, requested_target_id: str | None = None) -> GameObject | None:
        """Every interact-type action ultimately resolves against
        `current_interaction_target` - the one object (if any) currently
        within `INTERACTION_DISTANCE` (see `_find_nearest_interactable`) -
        never against an arbitrary caller-supplied id, since the actor can
        only ever physically be acting on what it's actually standing next
        to. `requested_target_id=None` (HumanController/ScriptedBotController,
        M3/M4 behavior, unchanged) means "whatever is nearest"; a caller
        that *does* name a target (M5D.2+'s LLM-driven controllers, which
        route the decided target_id all the way through) must have it match
        what's actually nearest right now, or the request is rejected here
        rather than silently redirected to a different object than the one
        asked for (M5D.4 spec section 13: engine authority is final, and a
        mismatch here is exactly the "stale by execution time" case
        AgentMemoryTracker's pending-decision check already classifies as a
        failure - this is what makes that classification actually true,
        rather than the engine having ignored target_id all along).
        """
        target_id = self.current_interaction_target.get(actor_id)
        if target_id is None:
            return None
        if requested_target_id is not None and requested_target_id != target_id:
            return None
        return self.level.get_object(target_id)

    def _interact(self, actor_id: str, requested_target_id: str | None = None, code: str | None = None) -> None:
        obj = self._target_object(actor_id, requested_target_id)
        if obj is None:
            return

        actor = self.actors[actor_id]
        from src import puzzles
        if code is not None and self.puzzle_mode == puzzles.RIDDLE and obj.id == puzzles.SAFE_ID \
                and puzzles.SAFE_CODE not in actor.knowledge:
            self._apply_interaction_result(actor_id, puzzles.enter_code(self, actor, obj, code))
            return
        if obj.id == "exit_door":
            result = self._attempt_exit(actor, obj)
        elif obj.type == "door":
            result = self._open_door(actor, obj)
        else:
            result = self._search(actor, obj)

        self._apply_interaction_result(actor_id, result)

    def _do_inspect(self, actor_id: str, requested_target_id: str | None = None) -> None:
        obj = self._target_object(actor_id, requested_target_id)
        if obj is None or not obj.trappable:
            return
        result = self._inspect(self.actors[actor_id], obj)
        self._apply_interaction_result(actor_id, result)

    def _do_place_trap(self, actor_id: str, trap_type: str | None, requested_target_id: str | None = None) -> None:
        obj = self._target_object(actor_id, requested_target_id)
        if obj is None:
            return
        result = self._place_trap(self.actors[actor_id], obj, trap_type)
        self._apply_interaction_result(actor_id, result)

    def _do_disarm(self, actor_id: str, requested_target_id: str | None = None) -> None:
        obj = self._target_object(actor_id, requested_target_id)
        if obj is None:
            return
        result = self._disarm(self.actors[actor_id], obj)
        self._apply_interaction_result(actor_id, result)

    def _do_send_message(self, actor_id: str, recipient_id: str | None, text: str | None, charge_time: bool = True) -> None:
        """M5F.2: unlike every other action above, this never resolves
        against `current_interaction_target` - communication has no
        proximity requirement (spec sections 3, 5 describe a channel, not
        a physical interaction). Defensive, not merely trusting - every
        check here (recipient real/other/message non-empty/length) is
        already enforced upstream by `validate_model_output`/
        `convert_response_to_action` (spec section 39), but the engine
        never assumes a caller validated correctly (same discipline as
        every other `_do_*` rule silently no-oping on a bad request rather
        than raising).

        M5F.2.2 spec sections 1, 20: `charge_time=False` for a
        ZERO_DECISION_COST accompanying message (called from
        `apply_action`'s own pre-dispatch delivery step) - the physical
        action already carries its own real cost, so a message riding
        along with it must not ALSO independently consume game-time (that
        would be a new fractional-cost mechanic, explicitly out of scope -
        spec section 35). The standalone FULL_COST SEND_MESSAGE action
        (dispatched from the normal `action.type` branch below) keeps
        `charge_time=True`, its own pre-existing behavior, unchanged.
        """
        if not self.communication_enabled:
            return
        if recipient_id is None or recipient_id == actor_id or recipient_id not in self.actors:
            return
        if not text or not text.strip():
            return

        clean_text = text.strip()[: constants.MESSAGE_MAX_LENGTH]
        message_id = str(uuid.uuid4())
        message = {
            "message_id": message_id,
            "sender_actor_id": actor_id,
            "recipient_actor_id": recipient_id,
            "game_time": self.elapsed_time,
            "text": clean_text,
        }
        # Spec section 20: delivered exactly once - appended to engine
        # state exactly once per `_do_send_message` call, itself only ever
        # invoked once per accepted SEND_MESSAGE decision (ActionExecutor
        # treats it as an instant, single-tick action - see
        # action_lifecycle.py - never replayed across ticks the way a
        # multi-tick MOVE is).
        self.messages.append(message)
        # Spec section 23: both canonical events recorded at the one engine
        # mutation point where the message becomes real, engine-truth,
        # reliably-delivered state (spec section 5) - not deferred to
        # whenever some later Observation happens to include it. Fairness
        # (spec section 17) instead comes from build_observation/apply_action
        # ordering: every Observation this tick was already built from
        # last tick's `self.messages` before this call ever runs.
        self._log_event("MESSAGE_SENT", actor_id=actor_id, recipient_id=recipient_id, message_id=message_id, text=clean_text)
        self._log_event("MESSAGE_RECEIVED", actor_id=recipient_id, sender_id=actor_id, message_id=message_id, text=clean_text)
        if charge_time:
            self._charge_time(self.actors[actor_id], constants.TIME_COST_SEND_MESSAGE)

    def _search(self, actor: Actor, obj: GameObject) -> InteractionResult:
        # M4B: a lock gates access to this object's contents specifically -
        # not whether it can be inspected/trapped (see _inspect/_place_trap,
        # neither of which checks this). Checked before the trap roll too:
        # you can't spring a trap on something you can't even open yet.
        if obj.lock_type is not None and not self.requirement_met(actor, obj.lock_type, obj.requires):
            return InteractionResult(False, self._locked_message(obj))

        entry = self._object_entry(obj.id)
        verb = obj.interaction or "search"

        trap = entry["trap"]
        if trap is not None and trap["owner_id"] == actor.id and self.own_traps_dangerous:
            # V3: no owner immunity - searching your own armed trap sets it off
            return self._trigger_trap(actor, obj, entry, trap)
        if trap is not None and trap["owner_id"] != actor.id:
            if entry["detected_by"].get(actor.id) != trap["type"]:
                return self._trigger_trap(actor, obj, entry, trap)

        dropped = entry.get("dropped", []) if self.single_copy_items else []
        if actor.id in entry["searched_by"] and not dropped:
            return InteractionResult(True, f"You already {verb}ed the {obj.label}.")

        entry["searched_by"].add(actor.id)
        self._log_event("OBJECT_SEARCHED", actor_id=actor.id, object_id=obj.id)

        if self.single_copy_items:
            newly_found, already_taken = self._take_single_copy(actor, obj, entry)
        else:
            newly_found, already_taken = [item_id for item_id in obj.contains if actor.inventory.add(item_id)], []
        # M4D: an ordinary search costs SEARCH; a locked object (the safe)
        # bundles unlocking and searching into this one call (there's no
        # separate UNLOCK step for furniture the way there is for a door -
        # see _open_door), so it costs ENTER_CODE/UNLOCK instead. Either
        # way, finding something adds PICKUP's small extra cost on top -
        # actually grabbing what you found isn't free.
        if obj.lock_type is None:
            base_cost = constants.TIME_COST_SEARCH
        elif obj.lock_type == "code":
            base_cost = constants.TIME_COST_ENTER_CODE
        else:
            base_cost = constants.TIME_COST_UNLOCK
        self._charge_time(actor, base_cost + (constants.TIME_COST_PICKUP if newly_found else 0.0))
        messages = []
        if newly_found:
            for item_id in newly_found:
                self._log_event("ITEM_FOUND", actor_id=actor.id, object_id=obj.id, item_id=item_id)
            names = ", ".join(self.items.items[item_id].name for item_id in newly_found)
            messages.append(f"Found: {names}")

        # M4B: knowledge (a code/password), never inventory - added to the
        # actor's own Actor.knowledge, same "who learned what" shape as
        # searched_by/inspected_by, but this one persists on the actor
        # itself since it must survive independent of any object state.
        if obj.reveals_knowledge and obj.reveals_knowledge not in actor.knowledge:
            actor.knowledge.add(obj.reveals_knowledge)
            self._log_event("KNOWLEDGE_LEARNED", actor_id=actor.id, object_id=obj.id, knowledge_id=obj.reveals_knowledge)
            messages.append(f"Learned: {self.items.knowledge_name(obj.reveals_knowledge)}")
            if self.puzzle_mode != "none":
                from src import puzzles
                puzzles.after_knowledge(self, actor)
        if self.puzzle_mode != "none":
            from src import puzzles
            inscription = puzzles.read_inscription(self, actor, obj)
            if inscription:
                messages.append(f"Inscription on the {obj.label}: {inscription}")

        if already_taken:
            names = ", ".join(self.items.items[i].name for i in already_taken)
            messages.append(f"Empty - someone already took: {names}")
        if messages:
            return InteractionResult(True, "\n".join(messages), item_found=newly_found[0] if newly_found else None)

        return InteractionResult(True, obj.message or f"You {verb} the {obj.label}.")

    @staticmethod
    def requirement_met(actor: Actor, lock_type: str, requires: str) -> bool:
        """Whether `actor` currently satisfies a lock of `lock_type`
        needing `requires` - a code is knowledge, a key/keycard is a
        physical item (Milestone_4.md's central distinction). Static
        because it's pure self-knowledge about `actor`, reusable from
        observation.py to answer "can I open this" without duplicating
        the rule.
        """
        if lock_type == "code":
            return requires in actor.knowledge
        return actor.inventory.has(requires)

    def _locked_message(self, obj: GameObject) -> str:
        phrase = constants.LOCK_PHRASES.get(obj.lock_type, "something you don't have")
        return f"The {obj.label} is locked. You need {phrase}."

    def _charge_time(self, actor: Actor, cost: float) -> None:
        """M4D: called only from the branch of a rule that actually did
        something - a failed/no-op attempt (locked, already searched, no
        trap to disarm, missing mission items at the exit, ...) costs
        nothing, since nothing happened. A plain assignment, not additive:
        `apply_action` already refuses every action while `is_busy`, so
        this can never be called on an actor who's already mid-cooldown.
        """
        actor.action_cooldown = cost

    def _trigger_trap(self, actor: Actor, obj: GameObject, entry: dict, trap: dict) -> InteractionResult:
        """An armed enemy trap on `obj` goes off under `actor` - it's spent
        (single-use) and immobilizes them; the search itself does not
        happen this time (see MILESTONE_3.md section 12).

        M4I: the message is now trap-specific
        (`Milestone_4I_TRAP_EFFECTS.md` section 10 - "replace generic
        long-lived STUNNED text"); the *visual* reaction is entirely the
        renderer's job (`src/effects.py`, reacting to this same
        TRAP_TRIGGERED event), not this method's. Alarm's extra effect -
        revealing this victim's position to the trap's owner - is handled
        separately below, since it's the one trap type that does
        something to an actor *other* than the victim.
        """
        trap_type = trap["type"]
        owner_id = trap["owner_id"]
        entry["trap"] = None
        actor.stunned_timer = constants.TRAP_STUN_SECONDS[trap_type]
        actor.moving = False
        hits_before = self.trap_hits.get(actor.id, 0)
        lives_before = self.lives.get(actor.id) if self.lives_enabled else None
        self.trap_hits[actor.id] = hits_before + 1
        if self.lives_enabled:
            self.lives[actor.id] = max(0, self.lives.get(actor.id, constants.SPY_LIVES) - 1)
        eliminated_now = self.lives_enabled and self.lives[actor.id] == 0 and not self.eliminated.get(actor.id)
        extra = {}
        if self.own_traps_dangerous:  # V3 trap telemetry (live only)
            extra = {"trap_id": trap.get("id"), "target_id": obj.id, "own_trap": owner_id == actor.id,
                     "armed_before": True, "armed_after": False,
                     "trap_hits_before": hits_before, "trap_hits_after": self.trap_hits[actor.id],
                     "lives_before": lives_before, "lives_after": self.lives.get(actor.id) if self.lives_enabled else None,
                     "eliminated": bool(eliminated_now)}
            if owner_id == actor.id:
                record = self.self_placed_traps.get(actor.id, {}).get(obj.id)
                if record is not None:
                    record["status_if_gone"] = "TRIGGERED_ON_YOU"
        self._log_event("TRAP_TRIGGERED", actor_id=actor.id, object_id=obj.id, trap_type=trap_type, trap_owner_id=owner_id, **extra)
        self._log_event("ACTOR_STUNNED", actor_id=actor.id, seconds=actor.stunned_timer)
        if self.lives_enabled:
            self._log_event("LIFE_LOST", actor_id=actor.id, lives_left=self.lives[actor.id], cause=f"{trap_type}_trap")
            if eliminated_now:
                self.eliminated[actor.id] = True
                actor.moving = False
                self._log_event("ACTOR_ELIMINATED", actor_id=actor.id, cause="no_lives_left")
                self.drop_items_at(actor.id, obj.id)

        if trap_type == "alarm" and owner_id in self.actors:
            # M4I section 5: "reveal victim position to opponent" - the
            # trap's *owner* (not the engine, not the renderer) is who
            # gains this knowledge, exactly the way M4C's opponent
            # visibility already flows: a timer on the owner's own Actor,
            # surfaced through that owner's Observation.revealed_opponent
            # (never read directly off GameState by any controller).
            #
            # M5F.3A bug fix: an environmental trap's owner_id
            # ("environment") is not a key in self.actors - the alarm
            # reveal is a reward for a *spy* who set a trap, and an
            # environmental alarm trap has no such spy to reward, so this
            # branch is simply skipped for it (the trap still triggers and
            # stuns normally above; only the reveal-to-owner step doesn't
            # apply). Before this guard, an environmental alarm trap would
            # have raised KeyError here the first time it fired.
            self.actors[owner_id].alarm_reveal_timer = constants.ALARM_REVEAL_DURATION
            self._log_event(
                "ALARM_REVEALED", actor_id=owner_id, victim_id=actor.id, seconds=constants.ALARM_REVEAL_DURATION
            )

        return InteractionResult(False, constants.TRAP_TRIGGER_MESSAGES[trap_type], trap_triggered=trap_type)

    def _inspect(self, actor: Actor, obj: GameObject) -> InteractionResult:
        entry = self._object_entry(obj.id)
        entry["inspected_by"].add(actor.id)
        self._log_event("OBJECT_INSPECTED", actor_id=actor.id, object_id=obj.id)
        # M4D: every real inspect attempt costs the same, hit or miss -
        # INSPECT being the slower/safer alternative to blind SEARCH is the
        # whole point (Milestone_4.md's "purpose" note), so the cost can't
        # depend on whether it happened to find anything.
        self._charge_time(actor, self.inspect_cost())
        if self.puzzle_mode != "none":
            from src import puzzles
            puzzles.read_inscription(self, actor, obj)  # looking closely reads any inscription

        trap = entry["trap"]
        if trap is not None and trap["owner_id"] == actor.id and self.own_traps_dangerous:
            label = constants.TRAP_LABELS[trap["type"]]
            return InteractionResult(True, f"Your own {label} is armed on the {obj.label}.")
        if trap is None or trap["owner_id"] == actor.id:
            return InteractionResult(True, f"You inspect the {obj.label}. It looks clear.")

        if entry["detected_by"].get(actor.id) == trap["type"]:
            label = constants.TRAP_LABELS[trap["type"]]
            return InteractionResult(True, f"{label} detected on the {obj.label}!", trap_detected=trap["type"])

        probability = constants.TRAP_DETECTION_PROBABILITY[trap["type"]]
        if self.rng.random() < probability:
            entry["detected_by"][actor.id] = trap["type"]
            self._log_event("TRAP_DETECTED", actor_id=actor.id, object_id=obj.id, trap_type=trap["type"])
            label = constants.TRAP_LABELS[trap["type"]]
            return InteractionResult(True, f"{label} detected on the {obj.label}!", trap_detected=trap["type"])

        self._log_event("TRAP_NOT_DETECTED", actor_id=actor.id, object_id=obj.id, trap_type=trap["type"])
        return InteractionResult(True, f"You inspect the {obj.label}. It looks clear.")

    def _place_trap(self, actor: Actor, obj: GameObject, trap_type: str | None) -> InteractionResult:
        if not obj.trappable:
            return InteractionResult(False, f"You can't trap the {obj.label}.")

        entry = self._object_entry(obj.id)
        # live trap rules: searching an object is knowing it too (a search
        # sets off any enemy trap on it), so a searched object can be trapped
        # without a 4 s inspection first. Recorded rules: inspect first.
        knows_object = actor.id in entry["inspected_by"] or (self.live_trap_rules and actor.id in entry["searched_by"])
        if not knows_object:
            return InteractionResult(False, f"Inspect the {obj.label} before trapping it.")

        if entry["trap"] is not None:
            return InteractionResult(False, f"The {obj.label} already has a trap on it.")

        if trap_type is None or actor.trap_inventory.get(trap_type, 0) <= 0:
            return InteractionResult(False, "You have no traps of that type left.")

        actor.trap_inventory[trap_type] -= 1
        entry["trap"] = {"type": trap_type, "owner_id": actor.id, "armed": True}
        if self.own_traps_dangerous:
            # V3: engine trap id + the owner's own memory of the trap
            self.trap_seq += 1
            trap_id = f"trap_{self.trap_seq:02d}"
            entry["trap"]["id"] = trap_id
            self.self_placed_traps.setdefault(actor.id, {})[obj.id] = {
                "trap_id": trap_id, "target_id": obj.id, "trap_type": trap_type,
                "room": self.level.object_room(obj), "placed_at": round(self.elapsed_time, 3), "fresh": True,
            }
            self._log_event("TRAP_PLACED", actor_id=actor.id, object_id=obj.id, trap_type=trap_type, trap_id=trap_id,
                            target_id=obj.id, trap_owner_id=actor.id, armed_before=False, armed_after=True)
        else:
            self._log_event("TRAP_PLACED", actor_id=actor.id, object_id=obj.id, trap_type=trap_type)
        # M4D: PLACE_TRAP costs real self time now, for a possible
        # opponent delay later (Milestone_4.md's "purpose" note) - M4I
        # made that cost vary per trap type (constants.TRAP_PLACEMENT_COST)
        # instead of one flat number, the pricier traps costing more.
        self._charge_time(actor, constants.TRAP_PLACEMENT_COST[trap_type])
        label = constants.TRAP_LABELS[trap_type]
        return InteractionResult(True, f"You place a {label} on the {obj.label}.")

    def _disarm(self, actor: Actor, obj: GameObject) -> InteractionResult:
        """can_disarm = trap.active and (trap.owner_id == actor_id or
        actor_knows_trap_exists) - Milestone_32.md section 1. Removing your
        own trap is a known quantity, not a "disarm" in the detection
        sense; DISARM handles it too rather than adding a separate
        REMOVE_TRAP action, per that doc's explicit allowance to defer that
        split until it's actually needed.
        """
        entry = self._object_entry(obj.id)
        trap = entry["trap"]

        if trap is None:
            return InteractionResult(False, "There's no trap here to disarm.")

        is_owner = trap["owner_id"] == actor.id
        if not is_owner and entry["detected_by"].get(actor.id) != trap["type"]:
            return InteractionResult(False, "You haven't detected a trap here yet.")

        trap_type = trap["type"]
        p_success = None if is_owner else self.disarm_success_probability(trap_type)
        if p_success is not None:
            roll = self.disarm_rng.random()
            self._log_event("DISARM_ATTEMPTED", actor_id=actor.id, object_id=obj.id, trap_type=trap_type,
                            trap_owner_id=trap["owner_id"], success_probability=p_success,
                            roll=round(roll, 6), success=roll < p_success)
            if roll >= p_success:
                # a failed disarm sets the trap off under the spy who tried
                self._log_event("TRAP_DISARM_FAILED", actor_id=actor.id, object_id=obj.id, trap_type=trap_type,
                                trap_owner_id=trap["owner_id"])
                return self._trigger_trap(actor, obj, entry, trap)
        entry["trap"] = None
        label = constants.TRAP_LABELS[trap_type]
        # M4D: charged once here, above the owner/enemy fork below, since
        # both branches are a genuine DISARM that actually removed a trap -
        # only the two early failure returns above (no trap / undetected)
        # skip it, because nothing happened there.
        self._charge_time(actor, constants.TIME_COST_DISARM)
        extra = {}
        if self.own_traps_dangerous:  # V3 trap telemetry (live only)
            extra = {"trap_id": trap.get("id"), "target_id": obj.id, "trap_owner_id": trap["owner_id"],
                     "armed_before": True, "armed_after": False}

        if is_owner:
            record = self.self_placed_traps.get(actor.id, {}).get(obj.id) if self.own_traps_dangerous else None
            if record is not None:
                record["status_if_gone"] = "REMOVED_BY_YOU"
                if record["fresh"]:
                    # placed, then removed without ever leaving the object
                    self._log_event("SELF_REVERSAL_NO_STATE_CHANGE", actor_id=actor.id, object_id=obj.id,
                                    trap_id=record["trap_id"], trap_type=trap_type)
            self._log_event("TRAP_REMOVED", actor_id=actor.id, object_id=obj.id, trap_type=trap_type, **extra)
            return InteractionResult(True, f"You remove your {label} from the {obj.label}.")

        self._log_event("TRAP_DISARMED", actor_id=actor.id, object_id=obj.id, trap_type=trap_type, **extra)
        return InteractionResult(True, f"You disarm the {label} on the {obj.label}.")

    def _open_door(self, actor: Actor, obj: GameObject) -> InteractionResult:
        # M4B: a "locked" door is a third door_states value alongside
        # "open"/"closed" - solid in both physics and nav_grid exactly like
        # "closed" (see Level.solid_rects / navigation.build_nav_grid,
        # neither needed a change - both already only special-case "open").
        # One E press unlocks *and* opens in the same motion if the actor
        # holds what it takes; otherwise the door stays locked and the
        # actor learns only the lock's shape, not which specific item
        # would satisfy it (see _locked_message).
        was_locked = self.door_states[obj.id] == "locked"
        if was_locked:
            if not self.requirement_met(actor, obj.lock_type, obj.requires):
                return InteractionResult(False, self._locked_message(obj))
            self.door_states[obj.id] = "closed"
            self._log_event("DOOR_UNLOCKED", actor_id=actor.id, object_id=obj.id, lock_type=obj.lock_type)

        self.door_states[obj.id] = "open"
        self._rebuild_nav_grid()
        self._log_event("DOOR_OPENED", actor_id=actor.id, object_id=obj.id)
        # M4D: unlocking-and-opening a code lock (ENTER_CODE) costs more
        # than a key/keycard (UNLOCK), which costs more than a door that
        # was merely closed, not locked (OPEN) - Milestone_4.md's table,
        # picked by what this door actually needed, not a flat door cost.
        if was_locked:
            cost = constants.TIME_COST_ENTER_CODE if obj.lock_type == "code" else constants.TIME_COST_UNLOCK
        else:
            cost = constants.TIME_COST_OPEN
        self._charge_time(actor, cost)
        return InteractionResult(True, obj.message or "The door opens.")

    def _attempt_exit(self, actor: Actor, obj: GameObject) -> InteractionResult:
        # M4E: "deadline may make one/both fail" - once the shared clock
        # hits zero, the extraction window is permanently closed for
        # whoever hasn't already gotten out, however complete their own
        # mission is. Checked first, and unconditionally: a late attempt
        # costs no time either (nothing happened), same as every other
        # failed/no-op interaction (see _charge_time's docstring).
        if self.time_remaining_for(actor.id) <= 0:
            self._log_event("EXIT_ATTEMPTED", actor_id=actor.id, missing=[], deadline_passed=True)
            return InteractionResult(False, "Time's up. The extraction window has closed.")

        required = self.items.required
        missing = actor.inventory.missing(required)

        if missing:
            self._log_event("EXIT_ATTEMPTED", actor_id=actor.id, missing=list(missing))
            names = [self.items.items[item_id].name for item_id in missing]
            lines = "You still need:\n" + "\n".join(f"- {name}" for name in names)
            return InteractionResult(False, lines)

        self.door_states[obj.id] = "open"
        self._rebuild_nav_grid()
        self.mission_complete[actor.id] = True
        # M4E/M4H: the escape facts that can't be recovered later - see
        # the GameState field comments, src/scoring.py, and
        # src/episode.py. Recorded here, at the single call site an
        # escape can ever succeed from.
        self.finish_order.append(actor.id)
        self.escape_snapshot[actor.id] = self.time_remaining_for(actor.id)
        self.escape_elapsed[actor.id] = self.elapsed_time
        actor.moving = False
        self._log_event("DOOR_OPENED", actor_id=actor.id, object_id=obj.id)
        self._log_event(
            "MISSION_COMPLETED", actor_id=actor.id, finish_place=len(self.finish_order), remaining_time=self.time_remaining_for(actor.id)
        )
        if self.exit_walkout_path:
            self.exit_walking[actor.id] = [tuple(p) for p in self.exit_walkout_path]
            self._log_event("EXIT_TRANSITION_STARTED", actor_id=actor.id, to=list(self.exit_walkout_path[-1]))
        # M4D: the exit is never lock_type-gated (it's gated by required
        # inventory, checked above), so it's always a plain OPEN, never an
        # UNLOCK/ENTER_CODE - and charging it at all is harmless anyway
        # since mission_complete now short-circuits every future
        # apply_action for this actor regardless.
        self._charge_time(actor, constants.TIME_COST_OPEN)
        return InteractionResult(True, "MISSION COMPLETE\nYou escaped the embassy.")

    def _rebuild_nav_grid(self) -> None:
        """A door's open/closed state feeds directly into which cells the
        walkability grid blocks (see navigation.build_nav_grid) - but that
        grid is otherwise only built once, at new_game(). Milestone 3's
        level had no through-passage doors (Room A's door and the exit
        were both dead ends physics never needed to path across), so this
        staleness never mattered. Milestone 4A's inter-room doors are
        genuine through-passages: leaving the grid stale after a door
        opens would permanently strand either actor's pathfinding on
        whichever side of it they started, and reopening/closing a shared
        door must affect both actors' paths immediately (see
        MILESTONE_4.md's M4A acceptance: "shared door changes affect
        both").
        """
        self.nav_grid = build_nav_grid(self.level, self.door_states)

    def _update_room_memory(self, actor_id: str, actor: Actor) -> str | None:
        """M4C: accumulate this actor's persistent exploration memory.
        `visited_rooms` and `known_locked_objects` only ever grow - nothing
        here is per-tick-recomputed-and-discarded, since "knowledge
        persists" is M4C's whole point (Milestone_4.md's acceptance line).
        Room/object geometry was never hidden in the first place (no
        fog-of-war yet - a separate, later milestone), so in this
        implementation these end up mirroring what's already globally
        visible; the scaffolding is real per-actor memory regardless, ready
        for a future milestone that actually restricts visibility by room.
        Returns the room this actor is in right now, for
        `_update_opponent_visibility` to compare between actors.
        """
        room_id = self.level.room_id_at(actor.x, actor.y)
        if room_id is None:
            return None

        # Bugfix (exploration state): a change of room is a traversal of the
        # door connecting the two rooms. No event is logged, so event_log
        # contents and hashes of every existing run are unchanged.
        if actor.last_room is not None and actor.last_room != room_id:
            for door_id in self.level.doors_between(actor.last_room, room_id):
                actor.traversed_doors.add(door_id)
        actor.last_room = room_id
        actor.visited_rooms.add(room_id)
        for obj in self.level.objects:
            if obj.lock_type is None:
                continue
            # A door's own collision rect straddles the wall gap between
            # two rooms - its center often falls in neither room's rect
            # (room rects stop at the inner wall face), so `room_id_at`
            # would never match it. Use `connects` instead for doors;
            # ordinary furniture (the safe) still uses its actual position.
            in_this_room = room_id in obj.connects if obj.type == "door" and obj.connects else self.level.room_id_at(*obj.center) == room_id
            if not in_this_room:
                continue
            if obj.id not in actor.known_locked_objects:
                actor.known_locked_objects[obj.id] = obj.lock_type
                self._log_event("OBJECT_LOCK_DISCOVERED", actor_id=actor_id, object_id=obj.id, lock_type=obj.lock_type)
        return room_id

    def _update_opponent_visibility(self, current_rooms: dict[str, str | None]) -> None:
        """M4C's one genuinely new hide/reveal boundary: room-based
        visibility of the *other actor* (Milestone_4.md: "same room =
        visible; different room = hidden... preserve last-seen room/
        position/game-time"). Symmetric - each actor's own
        `last_seen_opponent` is updated independently, so a stunned or
        already-escaped actor (absent from `current_rooms`) simply can't
        currently see or be seen, without needing a special case here.
        """
        actor_ids = list(current_rooms.keys())
        if len(actor_ids) != 2:
            return

        a_id, b_id = actor_ids
        room_a, room_b = current_rooms[a_id], current_rooms[b_id]
        visible = room_a is not None and room_a == room_b

        for observer_id, other_id, room in ((a_id, b_id, room_a), (b_id, a_id, room_b)):
            was_visible = self._opponent_visible.get(observer_id, False)
            self._opponent_visible[observer_id] = visible
            if not visible:
                continue
            other = self.actors[other_id]
            self.actors[observer_id].last_seen_opponent = {
                "actor_id": other_id,
                "room": room,
                "position": (other.x, other.y),
                "facing": other.facing,
                "game_time": self.elapsed_time,
            }
            if not was_visible:
                self._log_event("OPPONENT_SPOTTED", actor_id=observer_id, other_actor_id=other_id, room=room)

    def _apply_interaction_result(self, actor_id: str, result: InteractionResult) -> None:
        if actor_id != constants.BLACK_SPY_ID:
            return  # only the human sees HUD text; every actor's actions still hit the event log
        self.status_message = result.message
        self.status_message_success = result.success
        self.status_message_timer = constants.STATUS_MESSAGE_SECONDS

    def log_event(self, event_type: str, **fields) -> None:
        """Append a structured event. Public: controllers may log purely
        observational bookkeeping (e.g. a bot's target selection) here
        without that counting as mutating gameplay state - nothing a
        controller logs can affect position, inventory, or search state.
        """
        self.event_log.append({"type": event_type, **fields})

    _log_event = log_event  # internal call sites read more naturally with the underscore

    # -- queries --------------------------------------------------------

    def _object_entry(self, object_id: str) -> dict:
        """Per-object state: searched_by/inspected_by are per-actor
        knowledge (who has done what); trap is shared world truth (at most
        one trap per object - MILESTONE_3.md section 5); detected_by is
        per-actor knowledge of that trap (section 11 - detecting it does
        not reveal it to the other actor).
        """
        return self.object_states.setdefault(
            object_id,
            {"searched_by": set(), "inspected_by": set(), "trap": None, "detected_by": {}},
        )

    def is_searched_by(self, object_id: str, actor_id: str) -> bool:
        return actor_id in self.object_states.get(object_id, {}).get("searched_by", set())

    def is_inspected_by(self, object_id: str, actor_id: str) -> bool:
        return actor_id in self.object_states.get(object_id, {}).get("inspected_by", set())

    def trap_on(self, object_id: str) -> dict | None:
        return self.object_states.get(object_id, {}).get("trap")

    def detected_trap_type(self, object_id: str, actor_id: str) -> str | None:
        return self.object_states.get(object_id, {}).get("detected_by", {}).get(actor_id)

    def _find_nearest_interactable(self, actor: Actor) -> str | None:
        # The one interaction-range rule, shared with navigation's approach
        # choice (see src/interaction_range.py) - unchanged semantics.
        return nearest_interactable_id(actor.x, actor.y, ((obj.id, obj.collision_rect) for obj in self.level.interactables()))
