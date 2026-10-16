"""Builds a per-actor Observation from GameState - the one door a
controller (human or bot, and eventually an LLM) may read the world
through, instead of reaching into GameState's ground truth directly.

Milestone 3D (Milestone_32.md's numbering; MILESTONE_3.md calls the same
content M3C, "Partial Information"): the engine knows everything (object
contents, trap truth and owner, both actors' full search/inspect/detect
history, both actors' positions) but a controller must not read that
ground truth directly - see the "Architecture requirement for Milestone 3"
diagram in MILESTONE_3.md (GameState -> build_observation() -> a
controller -> GameAction).

What an actor is allowed to see, and why:

- its own position/facing/inventory/trap loadout/stun state - obviously.
- the room's physical layout (every object's type/label/position/
  footprint) - both spies can see the room; there's no fog-of-war yet
  (that's a separate, later milestone - MILESTONE_3.md's Milestone 5,
  "Incomplete information: visibility, fog of war"). Only per-object
  *state* is hidden, not the room's geometry.
- for each object: whether *this actor* has searched/inspected it (that's
  self-knowledge), and this actor's own trap_status for it - never the
  other actor's search/inspect history, never an object's `contains`
  before this actor has actually searched it, and never raw trap truth
  for a trap this actor hasn't personally placed or detected.

`trap_status` follows MILESTONE_3.md section 3's recommended values:
"unknown" / "clear_after_inspection" / "detected" ("suspected" is reserved
for a future probabilistic-belief pass, per that section's own note, and
isn't produced yet). Critically, "clear_after_inspection" does NOT mean an
object is actually free of an enemy trap - it means this actor inspected
it and didn't detect one (section 3's explicit distinction). A controller
that only ever looks at trap_status, never at GameState.trap_on(), cannot
tell those two situations apart - which is the whole point.

Milestone 4C ("Exploration, Memory & Backtracking") adds persistent,
per-actor memory (`current_room`/`known_rooms`/`known_doors`/
`known_locked_objects`/`known_item_locations`, accumulated in
`GameState._update_room_memory` and never reset mid-episode) plus this
module's one genuinely new hide/reveal boundary: `visible_other_actor`/
`opponent`/`last_seen_opponent`. Room/object geometry was never hidden to
begin with (see above - that's still Milestone 5's job), so the memory
fields mostly mirror what's already globally visible; the opponent's
*position*, by contrast, was never exposed at all before M4C and is now
genuinely room-gated - visible only while both actors share a room, with a
stale `last_seen_opponent` (room/position/facing/game_time) persisting
once they don't.

Milestone 4D ("Time Economy") adds `busy`/`busy_seconds` (this actor's own
`Actor.action_cooldown` - self-knowledge, same category as `stunned`) and
`mission_time_remaining`, the one shared countdown clock both actors race
- not hidden per-actor state at all, just world truth every controller is
allowed to see, same as `nav_grid`.

Milestone 4I ("Trap Effects & Gameplay Feedback") adds `revealed_opponent`/
`revealed_opponent_seconds` - alarm's one mechanic that acts on someone
other than the trap's victim. Triggering an alarm starts a timer on the
*owner's* `Actor.alarm_reveal_timer`; while it runs, that owner's own
`revealed_opponent` carries the victim's live position, exactly the "the
opponent should receive this information through legitimate observation/
knowledge state, not by directly reading hidden GameState" requirement
that doc states explicitly. It auto-expires (`GameState.update`) and is
never merged into `opponent`/`last_seen_opponent` - a controller can tell
"I currently share a room with them," "I last saw them here a while
ago," and "their alarm just told me exactly where they are" apart.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import pygame

from src import constants
from src.navigation import NavGrid, find_room_crossing_waypoint
from src.state import GameState

TrapStatus = Literal["unknown", "clear_after_inspection", "detected"]


@dataclass(frozen=True)
class ObservedObject:
    id: str
    type: str
    label: str
    position: tuple[float, float]
    collision_size: tuple[float, float]
    interactable: bool
    trappable: bool
    searched: bool  # by the observing actor only
    inspected: bool  # by the observing actor only
    trap_status: TrapStatus
    trap_type: str | None = None  # only meaningful when trap_status == "detected"
    trap_is_mine: bool = False  # only meaningful when trap_status == "detected"
    door_open: bool | None = None  # doors only - shared world truth (like door_states itself), not hidden knowledge
    # M5E.4.2: which two rooms this door connects (doors only) - the exact
    # same `GameObject.connects` the level/engine already uses for its own
    # room-membership checks (see `GameState._update_room_memory`), now
    # also surfaced to the actor. Fixes the M5E.4.1-discovered defect where
    # nothing in the agent-facing representation told the model what was
    # on the other side of a door it had just opened.
    connects: tuple[str, str] | None = None
    # M5E.4.3: a real point *inside* the room on the other side of this
    # door from the observing actor's current room - only set when the
    # door is open and a destination-side point was actually found (see
    # `src.navigation.find_room_crossing_waypoint`). `None` for a closed
    # door, an unknown-connection door, or the actor's own current room
    # (nothing to cross) - callers fall back to the pre-M5E.4.3 generic
    # approach-point behavior in every `None` case, never a guess.
    crossing_waypoint: tuple[float, float] | None = None
    # M4B: the shape of the lock (level geometry, visible to anyone looking
    # at the door/safe - like seeing a keyhole) plus two facts derived from
    # *this actor's own* inventory/knowledge, never the other actor's:
    # `locked` (still needs unlocking) and `can_unlock` (this actor already
    # holds/knows what that takes). Never exposes which specific item id
    # satisfies it beyond lock_type itself - see GameState._locked_message.
    lock_type: str | None = None
    locked: bool = False
    can_unlock: bool = False
    # mission time budget V2: the room the object stands in (level layout);
    # used only to decide whether a route is actor-known (mission_time.py)
    room: str | None = None

    @property
    def collision_rect(self) -> pygame.Rect:
        x, y = self.position
        w, h = self.collision_size
        return pygame.Rect(int(x), int(y), int(w), int(h))

    @property
    def center(self) -> tuple[float, float]:
        return self.collision_rect.center


@dataclass(frozen=True)
class OpponentView:
    """M4C: the other actor, as seen *right now* - only ever populated
    when `Observation.visible_other_actor` is True (same room). A stale
    sighting belongs on `Observation.last_seen_opponent` instead, never
    here - this type exists specifically so a controller can't accidentally
    treat an old position as a live one.
    """

    actor_id: str
    position: tuple[float, float]
    facing: str
    room: str | None
    stunned: bool


@dataclass(frozen=True)
class Observation:
    actor_id: str
    seed: int  # M4A.3: so a live-play failure report can name the exact run it happened in
    position: tuple[float, float]
    facing: str
    stunned: bool
    stunned_seconds: float
    busy: bool  # M4D - mid-cooldown from a timed action; every action is dropped, same as stunned
    busy_seconds: float
    mission_time_remaining: float  # M4D - the one shared clock both actors race, not per-actor
    mission_complete: bool
    inventory: frozenset[str]
    knowledge: frozenset[str]  # M4B - codes/passwords learned; never inventory
    required_items: tuple[str, ...]
    trap_inventory: dict[str, int]
    nearby_object_id: str | None  # this actor's own current_interaction_target
    items_found_here: frozenset[str]  # object ids that have handed *this actor* a mission item
    objects: tuple[ObservedObject, ...]
    nav_grid: NavGrid  # the room's static walkable geometry - physical, not hidden per-actor state
    log_event: Callable[..., None]  # appends to the shared event log with this actor_id already bound

    # M4C ("Exploration, Memory & Backtracking") - see GameState._update_room_memory
    # /_update_opponent_visibility. Room/object geometry was never hidden
    # to begin with (no fog-of-war yet), so current_room/known_rooms/
    # known_doors/known_locked_objects/known_item_locations mostly mirror
    # what's already globally visible above; `visible_other_actor` /
    # `opponent` / `last_seen_opponent` are the one genuinely new boundary
    # - the other actor's position is now hidden outside this actor's
    # current room, with only a stale last-known sighting persisting.
    current_room: str | None
    known_rooms: frozenset[str]
    known_doors: frozenset[str]
    known_locked_objects: dict[str, str]  # object_id -> lock_type, ever encountered
    known_item_locations: dict[str, str]  # item_id -> object_id, this actor's own finds
    visible_other_actor: bool
    opponent: OpponentView | None  # only set when visible_other_actor is True
    last_seen_opponent: dict | None  # {actor_id, room, position, facing, game_time}; None if never seen
    # M4I: populated only while this actor's own alarm trap has *just*
    # gone off under the opponent (Actor.alarm_reveal_timer > 0) - a live
    # position, refreshed every tick for the reveal's duration, distinct
    # from `opponent` (which requires actually sharing a room) and from
    # `last_seen_opponent` (a frozen snapshot). Auto-expires - see
    # GameState.update - never a permanent leak.
    revealed_opponent: OpponentView | None
    revealed_opponent_seconds: float  # 0.0 when revealed_opponent is None

    # M5F.2: `other_actor_id` is legitimate self-knowledge (a spy always
    # knows who its one opponent is - not hidden state), always populated
    # regardless of `communication_enabled`, since it's also the only
    # possible SEND_MESSAGE recipient (this is a two-actor game).
    # `communication_enabled` is a fixed, per-episode condition (spec
    # section 13's OFF/ON), read by `enumerate_legal_actions` to decide
    # whether SEND_MESSAGE is offered at all - never toggled mid-episode.
    # `messages_received_recently` is this actor's own bounded window
    # (spec section 21) into `GameState.messages`, already filtered to
    # `recipient_actor_id == actor_id` and capped to the last
    # `constants.RECENT_MESSAGES_LIMIT` - oldest-of-the-window first, most
    # recent last (chronological, matching `GameState.messages`' own
    # append order). Always empty when communication is disabled.
    # Bugfix (exploration state): see `build_exploration` - this actor's own
    # per-room exploration facts. `None` only for Observations constructed
    # directly in tests; `build_observation` always fills it.
    # Defaulted (unlike every other field on this dataclass) so every
    # pre-M5F.2 test that constructs `Observation(...)` directly, without
    # these three new fields, keeps working unchanged - `build_observation`
    # itself always supplies real values explicitly (see below), never
    # relies on these defaults.
    other_actor_id: str = ""
    communication_enabled: bool = False
    messages_received_recently: tuple[dict, ...] = ()
    # M5F.2.2: which communication cost representation this episode uses -
    # `False` (FULL_COST, every pre-M5F.2.2 caller's implicit value) keeps
    # `enumerate_legal_actions` offering the standalone SEND_MESSAGE action
    # exactly as before; `True` (ZERO_DECISION_COST) suppresses it (spec
    # section 2's envelope mechanism replaces it for that condition).
    # Meaningless when `communication_enabled` is False.
    zero_decision_cost_messaging: bool = False
    exploration: dict | None = None
    # Bugfix live flow V2, Part D: whether spy encounters are a rule of this
    # match (a fixed per-match condition, like communication_enabled) and
    # the time left before the next fight can happen. The opponent's
    # distance is derived from `opponent` (only set when legitimately
    # visible) - see `opponent_distance`.
    encounters_enabled: bool = False
    encounter_cooldown: float = 0.0
    encounter_armed: bool = True  # False after a fight until the spies have separated (ENCOUNTER_REARM_DISTANCE)
    # Bugfix mission time budget V2 (src/mission_clock.py): the deadline this
    # actor plays against. `mission_time_remaining` above is this actor's own
    # remaining time under the active clock mode (the shared clock in
    # SIMULATION_TIME, its own budget in ACTION_BUDGET). `time_expired`: it
    # ran out without escaping - it has failed and no longer decides.
    mission_time_limit: float = 180.0
    mission_clock_mode: str = "simulation_time"
    time_expired: bool = False  # out of play: out of time (action budget) or out of lives
    lives: int | None = None  # live workshop: lives left (None = no lives rule)
    trap_hits: int = 0  # traps that went off under this actor
    # V3 (live only, GameState.own_traps_dangerous): own traps trigger on
    # their owner, and the owner remembers each trap it placed - see
    # GameState.own_trap_record. Empty/False everywhere else.
    own_traps_dangerous: bool = field(default=False, compare=False)
    self_placed_traps: tuple = field(default=(), compare=False)
    lifecycle: str = field(default="ACTIVE", compare=False)
    # live fight rule: KO'd (inactive) until it respawns; ko_count changes at each KO
    knocked_out: bool = field(default=False, compare=False)
    # live trap rules: what INSPECT costs and how likely an enemy disarm is to work
    inspect_cost: float | None = field(default=None, compare=False)
    # live trap rules: a searched object can be trapped without inspecting it first
    trap_after_search: bool = field(default=False, compare=False)
    fight_ko_respawn: bool = field(default=False, compare=False)
    encounter_alerts: bool = field(default=False, compare=False)
    # live one-copy rule (GameState.single_copy_items)
    single_copy_items: bool = field(default=False, compare=False)
    opponent_holds: tuple = field(default=(), compare=False)  # required items this actor KNOWS the opponent holds
    # live safe puzzle (src/puzzles.py): what this actor knows about it, None when off
    puzzle: dict | None = field(default=None, compare=False)
    opponent_lives: int | None = field(default=None, compare=False)  # only while the opponent is visible (its hearts are on screen)
    # live: "ELIMINATED" / "ESCAPED" once the opponent is out of the match (public: the screen shows it)
    opponent_out_of_match: str | None = field(default=None, compare=False)
    disarm_success_probability: dict | None = field(default=None, compare=False)
    ko_seconds_left: float | None = field(default=None, compare=False)
    ko_count: int = field(default=0, compare=False)
    room_at: Callable[[float, float], str | None] | None = field(default=None, compare=False, repr=False)
    # live one-copy rule: the opponent escaped with a required item this actor
    # lacks (GameState.mission_impossible). `time_expired` is True too (out of
    # play); never serialized into a model payload - no request follows it.
    mission_impossible: bool = field(default=False, compare=False)

    @property
    def opponent_distance(self) -> float | None:
        """Feet-to-feet distance to the opponent, only while it is visible."""
        if self.opponent is None:
            return None
        ox, oy = self.opponent.position
        return math.hypot(ox - self.position[0], oy - self.position[1])

    def get(self, object_id: str) -> ObservedObject | None:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def has_all_required_items(self) -> bool:
        return all(item_id in self.inventory for item_id in self.required_items)


def _puzzle_view(state: GameState, actor_id: str) -> dict | None:
    if getattr(state, "puzzle_mode", "none") == "none":
        return None
    from src import puzzles
    return puzzles.view(state, actor_id)


def build_observation(state: GameState, actor_id: str) -> Observation:
    actor = state.actors[actor_id]

    items_found_here = frozenset(
        e["object_id"] for e in state.event_log if e["type"] == "ITEM_FOUND" and e["actor_id"] == actor_id
    )
    known_item_locations = {
        e["item_id"]: e["object_id"]
        for e in state.event_log
        if e["type"] == "ITEM_FOUND" and e["actor_id"] == actor_id
    }

    current_room = state.level.room_id_at(actor.x, actor.y)
    known_rooms = frozenset(actor.visited_rooms)
    known_doors = frozenset(
        obj.id
        for obj in state.level.objects
        if obj.type == "door" and obj.connects is not None and (obj.connects[0] in known_rooms or obj.connects[1] in known_rooms)
    )

    other_id = next(aid for aid in state.actors if aid != actor_id)
    other = state.actors[other_id]
    other_room = state.level.room_id_at(other.x, other.y)
    visible_other_actor = current_room is not None and current_room == other_room
    if state.escaped_leave_world and state.mission_complete.get(other_id):
        visible_other_actor = False  # it has left the building
    # live: an eliminated spy is out of the match - no fight, no alert, no
    # approach/avoid against it (src/encounters.py already never fights it)
    opponent_out = None
    if state.live_lifecycle and state.lifecycle(other_id) in (constants.LIFECYCLE_ELIMINATED,
                                                              constants.LIFECYCLE_ESCAPED):
        opponent_out = state.lifecycle(other_id)
        if opponent_out == constants.LIFECYCLE_ELIMINATED:
            visible_other_actor = False
    opponent = (
        OpponentView(actor_id=other_id, position=(other.x, other.y), facing=other.facing, room=other_room, stunned=other.is_stunned)
        if visible_other_actor
        else None
    )
    # M4I: alarm's reveal - a live position feed, independent of room
    # sharing, active only while this actor's own alarm_reveal_timer is
    # running (see GameState._trigger_trap/update).
    revealed_opponent = (
        OpponentView(actor_id=other_id, position=(other.x, other.y), facing=other.facing, room=other_room, stunned=other.is_stunned)
        if actor.alarm_reveal_timer > 0 and opponent_out != constants.LIFECYCLE_ELIMINATED
        else None
    )

    objects = tuple(
        ObservedObject(
            id=obj.id,
            type=obj.type,
            label=obj.label,
            position=obj.position,
            collision_size=obj.collision_size,
            interactable=obj.interactable,
            trappable=obj.trappable,
            searched=state.is_searched_by(obj.id, actor_id),
            inspected=state.is_inspected_by(obj.id, actor_id),
            door_open=(state.door_states.get(obj.id) == "open") if obj.type == "door" else None,
            connects=obj.connects if obj.type == "door" else None,
            crossing_waypoint=_crossing_waypoint(state, obj, current_room),
            **_trap_knowledge(state, obj.id, actor_id),
            **_lock_knowledge(state, obj, actor),
            room=state.level.object_room(obj),
        )
        for obj in state.level.objects
    )

    return Observation(
        actor_id=actor_id,
        seed=state.seed,
        position=(actor.x, actor.y),
        facing=actor.facing,
        stunned=actor.is_stunned,
        stunned_seconds=actor.stunned_timer,
        busy=actor.is_busy,
        busy_seconds=actor.action_cooldown,
        mission_time_remaining=state.time_remaining_for(actor_id),
        mission_complete=state.mission_complete[actor_id],
        inventory=frozenset(actor.inventory.items),
        knowledge=frozenset(actor.knowledge),
        required_items=tuple(state.items.required),
        trap_inventory=dict(actor.trap_inventory),
        nearby_object_id=state.current_interaction_target.get(actor_id),
        items_found_here=items_found_here,
        objects=objects,
        nav_grid=state.nav_grid,
        log_event=lambda event_type, **fields: state.log_event(event_type, actor_id=actor_id, **fields),
        current_room=current_room,
        known_rooms=known_rooms,
        known_doors=known_doors,
        known_locked_objects=dict(actor.known_locked_objects),
        known_item_locations=known_item_locations,
        visible_other_actor=visible_other_actor,
        opponent=opponent,
        last_seen_opponent=dict(actor.last_seen_opponent) if actor.last_seen_opponent else None,
        revealed_opponent=revealed_opponent,
        revealed_opponent_seconds=actor.alarm_reveal_timer,
        other_actor_id=other_id,
        communication_enabled=state.communication_enabled,
        messages_received_recently=_recent_messages_for(state, actor_id),
        zero_decision_cost_messaging=state.zero_decision_cost_messaging,
        exploration=build_exploration(state, actor_id, current_room),
        encounters_enabled=state.encounters_enabled,
        encounter_cooldown=state.encounter_cooldown if state.encounters_enabled else 0.0,
        encounter_armed=state.encounter_armed if state.encounters_enabled else True,
        mission_time_limit=state.mission_time_total,
        mission_clock_mode=state.mission_clock_mode,
        # ACTION_BUDGET only: recorded SIMULATION_TIME runs keep deciding past the deadline exactly as before
        time_expired=state.out_of_play(actor_id),
        lives=state.lives.get(actor_id) if state.lives_enabled else None,
        trap_hits=state.trap_hits.get(actor_id, 0),
        own_traps_dangerous=state.own_traps_dangerous,
        self_placed_traps=tuple(
            state.own_trap_record(actor_id, target_id)
            for target_id in sorted(state.self_placed_traps.get(actor_id, {}))
        ) if state.own_traps_dangerous else (),
        lifecycle=state.lifecycle(actor_id),
        knocked_out=actor_id in state.knocked_out,
        inspect_cost=state.inspect_cost(),
        trap_after_search=state.live_trap_rules,
        fight_ko_respawn=state.fight_ko_respawn,
        encounter_alerts=state.encounter_alerts,
        single_copy_items=state.single_copy_items,
        puzzle=_puzzle_view(state, actor_id),
        opponent_holds=tuple(sorted(i for i in state.known_opponent_items.get(actor_id, set())
                                    if state.actors[other_id].inventory.has(i))) if state.single_copy_items else (),
        opponent_lives=(state.lives.get(other_id) if state.lives_enabled and other_id in state.actors
                        and visible_other_actor else None),
        opponent_out_of_match=opponent_out,
        disarm_success_probability=dict(constants.DISARM_SUCCESS_PROBABILITY) if state.live_trap_rules else None,
        ko_seconds_left=state.ko_seconds_left(actor_id),
        ko_count=state.ko_count.get(actor_id, 0),
        room_at=state.level.room_id_at,
        mission_impossible=actor_id in state.mission_impossible,
    )


# -- Bugfix: structured exploration state -------------------------------------

UNVISITED, PARTIAL, COMPLETE = "UNVISITED", "PARTIAL", "COMPLETE"


def build_exploration(state: GameState, actor_id: str, current_room: str | None = None) -> dict:
    """This actor's own exploration facts, per visited room (bugfix §11-24).

    Only rooms the actor has stood in are listed (nothing about unvisited
    rooms, their objects or contents). For each visited room:

    - `unsearched`: interactable objects in the room this actor has not
      INTERACTed with yet (the engine's `searched_by`; for "inspect"-style
      objects such as a painting the INTERACT is the look, recorded the same
      way). Locked objects this actor cannot open are listed under `blocked`
      instead.
    - `uninspected`: objects this actor has not trap-INSPECTed. Informative
      only: inspecting checks for traps, it never reveals contents, so it
      does not count towards completion.
    - `untraversed_exits` / `traversed_exits`: doors between this room and
      another room, split by whether *this actor* has walked through them
      (`traversed_exits` maps each to the room it led to - the destination
      is only stated once traversed). Doors locked for this actor are listed
      under `blocked` instead of `untraversed_exits`.
    - `blocked`: {id: reason} for objects/exits this actor cannot currently
      open (e.g. "locked:keycard"); they do not keep a room PARTIAL.
    - `status`: COMPLETE when nothing is unsearched and no exit is
      untraversed (blocked ones excepted); PARTIAL otherwise. Entering a room
      never makes it COMPLETE by itself.

    Lists are sorted and dict keys are emitted in sorted order, so the
    representation is deterministic. `exit_door` (the mission exit) belongs
    to no room and is not an exploration exit.
    """
    actor = state.actors[actor_id]
    level = state.level
    visited = set(actor.visited_rooms)
    if current_room is not None:
        visited.add(current_room)
    rooms: dict[str, dict] = {}
    for room_id in sorted(visited):
        unsearched, uninspected, blocked = [], [], {}
        for obj in level.objects:
            if not obj.interactable or obj.type == "door" or level.object_room(obj) != room_id:
                continue
            if not state.is_inspected_by(obj.id, actor_id):
                uninspected.append(obj.id)
            if state.is_searched_by(obj.id, actor_id):
                continue
            if obj.lock_type is not None and not GameState.requirement_met(actor, obj.lock_type, obj.requires):
                blocked[obj.id] = f"locked:{obj.lock_type}"
            else:
                unsearched.append(obj.id)
        untraversed, traversed = [], {}
        for door_id in level.room_exits(room_id):
            door = level.get_object(door_id)
            other = door.connects[0] if door.connects[1] == room_id else door.connects[1]
            if door_id in actor.traversed_doors:
                traversed[door_id] = other
            elif state.door_states.get(door_id) == "locked" and not GameState.requirement_met(actor, door.lock_type, door.requires):
                blocked[door_id] = f"locked:{door.lock_type}"
            else:
                untraversed.append(door_id)
        rooms[room_id] = {
            "status": COMPLETE if not unsearched and not untraversed else PARTIAL,
            "unsearched": sorted(unsearched),
            "uninspected": sorted(uninspected),
            "untraversed_exits": sorted(untraversed),
            "traversed_exits": dict(sorted(traversed.items())),
            "blocked": dict(sorted(blocked.items())),
        }
    return {"current_room": current_room, "rooms": rooms}


def room_exploration_status(exploration: dict | None, room_id: str | None) -> str:
    if not exploration or room_id is None:
        return UNVISITED
    room = exploration.get("rooms", {}).get(room_id)
    return room["status"] if room else UNVISITED


def _recent_messages_for(state: GameState, actor_id: str) -> tuple[dict, ...]:
    """M5F.2 spec section 21: this actor's own last
    `constants.RECENT_MESSAGES_LIMIT` received messages, oldest-of-the-
    window first (matches `GameState.messages`' own append/chronological
    order) - never the sender's own sent messages, never the other actor's
    inbox. Always empty when communication is off (spec section 13:
    disabled means genuinely unavailable, not merely empty by chance)."""
    if not state.communication_enabled:
        return ()
    received = [m for m in state.messages if m["recipient_actor_id"] == actor_id]
    return tuple(received[-constants.RECENT_MESSAGES_LIMIT :])


def _lock_knowledge(state: GameState, obj, actor) -> dict:
    if obj.lock_type is None:
        return {"lock_type": None, "locked": False, "can_unlock": False}

    can_unlock = GameState.requirement_met(actor, obj.lock_type, obj.requires)
    if obj.type == "door":
        locked = state.door_states.get(obj.id) == "locked"
    else:
        locked = not can_unlock
    return {"lock_type": obj.lock_type, "locked": locked, "can_unlock": can_unlock}


def _crossing_waypoint(state: GameState, obj, current_room: str | None) -> tuple[float, float] | None:
    """M5E.4.3: `None` for every non-door object, a closed door, a door
    whose `connects` isn't known, or when the actor's own current room
    isn't one side of this specific door (nothing to cross) - real
    geometry only in every other case (see `find_room_crossing_waypoint`).
    """
    if obj.type != "door" or obj.connects is None or current_room is None:
        return None
    if state.door_states.get(obj.id) != "open":
        return None
    if current_room not in obj.connects:
        return None
    destination_room = obj.connects[0] if obj.connects[1] == current_room else obj.connects[1]
    return find_room_crossing_waypoint(state.level, state.nav_grid, obj.collision_rect, destination_room)


def _trap_knowledge(state: GameState, object_id: str, actor_id: str) -> dict:
    trap = state.trap_on(object_id)

    if trap is not None and trap["owner_id"] == actor_id:
        return {"trap_status": "detected", "trap_type": trap["type"], "trap_is_mine": True}

    if trap is not None and state.detected_trap_type(object_id, actor_id) == trap["type"]:
        return {"trap_status": "detected", "trap_type": trap["type"], "trap_is_mine": False}

    if state.is_inspected_by(object_id, actor_id):
        return {"trap_status": "clear_after_inspection"}

    return {"trap_status": "unknown"}
