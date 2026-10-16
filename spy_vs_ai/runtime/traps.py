"""Enhanced gameplay: door traps and trap variants (FINAL-2 gameplay polish, spec 7-31).

A playable-runtime rule, not a research mechanic: none of this was part of Stages 3-5. The frozen policies keep their
one high-level trap action, PLACE_TRAP (and DISARM); the TrapResolver sits downstream of that decision:

    policy -> PLACE_TRAP(trap_type) -> TrapResolver -> target (an object, as before, or a doorway) -> variant -> trap

It is installed on one GameState instance (the class is never patched), around `apply_action` and `update`, before
the replay recorder, so a recorded call includes whatever the resolver did and an offline replay - which re-installs
it - reproduces the target, the variant, the trigger order and the effect exactly. Every random draw comes from
`random.Random` instances keyed by the game seed, the actor and a per-actor counter (never `state.rng`, the fight or
the disarm generators), so the engine's own random streams are untouched.

Rules (documented choices):
  * PLACE_TRAP is resolved only when the engine would accept it as an object trap (the near object is trappable,
    known, untrapped and the spy holds that trap type) - a door trap never makes an otherwise refused action succeed.
    The spy's personality sets how often it prefers a doorway (playable flavour, not a validated behaviour).
  * Eligible doorways: a door of the room the spy is in, not the exit, not locked, within DOOR_RANGE, with no armed
    trap (at most one trap per doorway). The highest-traffic doorway (the number of exits of the two rooms it joins)
    is chosen. A door trap never blocks the doorway: it is risk, not a wall.
  * The consumed inventory unit and the time cost are those of the trap type the policy chose; the variant is chosen
    by personality. Object traps keep the engine's own effects; their variant name is the legacy mapping
    (bomb -> explosive). Historical replays carry no variant: the legacy default is explosive.
  * A door trap goes off when an opposing spy's feet cross the doorway threshold (a side change of the threshold line
    between two consecutive frames, within the door's span) - not by standing near it or moving along it.
  * Owner immunity: a spy never sets off its own door trap. A spy that detected a door trap steps over it (as the
    engine's rule that a detected enemy trap does not go off on a search).
  * Detection: an INSPECT within DETECT_RANGE of a trapped doorway also examines that doorway (per-variant chance).
    Door traps are never revealed to the opponent otherwise.
  * DISARM: when the near object has no engine trap, a DISARM resolves to a detected (or own) door trap within
    DISARM_RANGE: the owner removes it; anyone else succeeds with the variant's probability, and a failure sets it
    off. Frozen policies issue DISARM only when their legal actions offer it.
  * Effects (short, mechanically distinct): explosive - one life and a 2 s stun; electric - a 3 s stun, no life;
    gas - a 4 s cloud on the doorway and a 4 s poisoning of the victim (half speed for whoever is poisoned or inside
    the cloud, the owner excepted); snare - rooted for 2.5 s (no movement; other actions stay possible);
    alarm - no damage, the owner sees the victim's position for ALARM_REVEAL_DURATION (the engine's alarm reveal).
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from src import constants
from src.actions import ActionType, GameAction
from src.mission_clock import ACTION_BUDGET

VARIANTS = ("explosive", "electric", "gas", "snare", "alarm")
LEGACY_VARIANT = {"bomb": "explosive", "electric": "electric", "snare": "snare", "alarm": "alarm"}
LEGACY_DEFAULT_VARIANT = "explosive"  # historical replays have no variant metadata
CUE = {"explosive": "BOOM", "electric": "ZAP", "gas": "GAS", "snare": "SNARED", "alarm": "ALARM"}
EFFECT_LABEL = {"explosive": "life lost + 2 s stun", "electric": "3 s stun", "gas": "4 s poison cloud (half speed)",
                "snare": "rooted 2.5 s", "alarm": "position revealed to the owner"}

# playable flavour (spec 16): how often a spy prefers a doorway, and which variant it sets there
DOOR_PREFERENCE = {"saboteur": 0.7, "cautious": 0.6, "opportunist": 0.5, "racer": 0.4}
FLAVOR = {
    "saboteur": (("explosive", 0.45), ("gas", 0.45), ("electric", 0.10)),
    "cautious": (("snare", 0.45), ("alarm", 0.35), ("electric", 0.20)),
    "racer": (("electric", 0.60), ("snare", 0.20), ("explosive", 0.20)),
    "opportunist": tuple((v, 0.2) for v in VARIANTS),
}
DETECTION_PROBABILITY = {"explosive": 0.90, "electric": 0.85, "gas": 0.80, "snare": 0.70, "alarm": 0.75}
DISARM_PROBABILITY = {"explosive": 0.55, "electric": 0.65, "gas": 0.60, "snare": 0.75, "alarm": 0.85}

DOOR_RANGE = 260.0     # px from the spy's feet to the doorway centre, to trap it
DETECT_RANGE = 160.0   # an INSPECT this close to a trapped doorway examines it too
DISARM_RANGE = 160.0
TELEPORT_STEP = 64.0   # a longer jump between two frames is a respawn, never a crossing
SPAN_MARGIN = 4.0      # px of tolerance at the door edges

EXPLOSIVE_STUN = 2.0
ELECTRIC_STUN = 3.0
GAS_SECONDS = 4.0
GAS_RADIUS = 72.0
GAS_SLOW = 0.5
SNARE_SECONDS = 2.5


@dataclass
class DoorTrap:
    trap_id: str
    door_id: str
    owner_id: str
    variant: str
    source_trap_type: str
    placed_at: float
    mission_time_remaining: float
    decision: int
    position: tuple[float, float]
    armed: bool = True
    status: str = "ARMED"
    detected_by: dict[str, float] = field(default_factory=dict)
    triggered_at: float | None = None
    victim_id: str | None = None
    disarmed_at: float | None = None

    def public(self) -> dict[str, Any]:
        out = asdict(self)
        out["position"] = [round(self.position[0], 1), round(self.position[1], 1)]
        return out


@dataclass
class Effect:
    kind: str                     # gas_cloud | poison | root
    trap_id: str
    owner_id: str
    remaining: float
    actor_id: str | None = None   # poison / root: who
    position: tuple[float, float] | None = None  # gas cloud: where


def _weighted(rng: random.Random, table: tuple[tuple[str, float], ...]) -> str:
    total = sum(w for _, w in table)
    roll = rng.random() * total
    for name, weight in table:
        roll -= weight
        if roll < 0:
            return name
    return table[-1][0]


class TrapResolver:
    """Door traps and trap variants on one GameState instance (see the module docstring for the rules)."""

    def __init__(self, personalities: dict[str, str]) -> None:
        self.personalities = dict(personalities)
        self.state = None
        self.traps: dict[str, DoorTrap] = {}
        self.effects: list[Effect] = []
        self.records: list[dict[str, Any]] = []  # runtime telemetry (enhanced_trap events), in order
        self.placements: dict[str, int] = {}
        self._draws: dict[str, int] = {}
        self._prev: dict[str, tuple[float, float]] = {}
        self._base_speed: dict[str, float] = {}
        self._slowed: set[str] = set()
        self._seq = 0

    # -- installation ----------------------------------------------------------------------------------
    def install(self, state) -> "TrapResolver":
        self.state = state
        apply, update = state.apply_action, state.update
        self._prev = {a: (actor.x, actor.y) for a, actor in state.actors.items()}
        self._base_speed = {a: actor.speed for a, actor in state.actors.items()}

        def apply_action(actor_id, action, dt):
            return self.apply_action(apply, actor_id, action, dt)

        def update_(dt):
            result = update(dt)
            self.after_update(dt)
            return result

        state.apply_action, state.update = apply_action, update_
        state.enhanced_traps = self  # read by the runtime renderer (markers, effects)
        return self

    # -- deterministic draws -----------------------------------------------------------------------------
    def _rng(self, purpose: str, *keys: Any) -> random.Random:
        key = ":".join(str(k) for k in (purpose, self.state.seed, *keys))
        n = self._draws.get(key, 0)
        self._draws[key] = n + 1
        return random.Random(f"{key}:{n}")

    # -- queries -----------------------------------------------------------------------------------------
    def armed_on(self, door_id: str) -> DoorTrap | None:
        return next((t for t in self.traps.values() if t.armed and t.door_id == door_id), None)

    def armed(self) -> list[DoorTrap]:
        return [t for t in self.traps.values() if t.armed]

    def visible_to(self, actor_id: str) -> list[DoorTrap]:
        """Armed door traps this spy knows about: its own and the ones it detected (spec 25)."""
        return [t for t in self.armed() if t.owner_id == actor_id or actor_id in t.detected_by]

    def active_effects(self) -> list[Effect]:
        return list(self.effects)

    def _room_of(self, actor) -> str | None:
        return self.state.level.room_id_at(actor.x, actor.y) or actor.last_room

    def eligible_doors(self, actor_id: str) -> list[Any]:
        """Doorways this spy could trap now, best first (traffic, then distance, then id)."""
        state, actor = self.state, self.state.actors[actor_id]
        room = self._room_of(actor)
        if room is None:
            return []
        level = state.level
        out = []
        for door_id in level.room_exits(room):
            door = level.get_object(door_id)
            if door is None or door.id == "exit_door" or state.door_states.get(door_id) == "locked":
                continue
            if self.armed_on(door_id) is not None:
                continue
            cx, cy = door.collision_rect.center
            distance = math.hypot(cx - actor.x, cy - actor.y)
            if distance > DOOR_RANGE:
                continue
            traffic = sum(len(level.room_exits(r)) for r in (door.connects or ()))
            out.append((-traffic, round(distance, 3), door.id, door))
        return [d for *_, d in sorted(out, key=lambda e: e[:3])]

    def _object_trap_accepted(self, actor_id: str, action: GameAction) -> bool:
        """Whether the engine would place this PLACE_TRAP on the near object (the same checks, read-only)."""
        state = self.state
        obj = state._target_object(actor_id, action.target_id)
        if obj is None or not obj.trappable:
            return False
        entry = state.object_states.get(obj.id) or {"inspected_by": set(), "searched_by": set(), "trap": None}
        knows = actor_id in entry["inspected_by"] or (state.live_trap_rules and actor_id in entry["searched_by"])
        if not knows or entry["trap"] is not None:
            return False
        trap_type = action.trap_type
        return trap_type is not None and state.actors[actor_id].trap_inventory.get(trap_type, 0) > 0

    def _gated(self, actor_id: str) -> bool:
        """The engine's own refusals at the top of apply_action (it then does nothing at all)."""
        state = self.state
        if state.mission_complete.get(actor_id) or state.eliminated.get(actor_id) or actor_id in state.mission_impossible:
            return True
        if state.mission_clock_mode == ACTION_BUDGET and state.mission_budget.get(actor_id, 0.0) <= 0:
            return True
        actor = state.actors[actor_id]
        return actor.is_stunned or actor.is_busy

    # -- the action hook -----------------------------------------------------------------------------------
    def apply_action(self, apply, actor_id: str, action: GameAction, dt: float):
        if action.type is ActionType.PLACE_TRAP and not self._gated(actor_id):
            return self._place(apply, actor_id, action, dt)
        if action.type is ActionType.DISARM and not self._gated(actor_id):
            if self._disarm(apply, actor_id, action, dt):
                return None
        before = len(self.state.event_log)
        result = apply(actor_id, action, dt)
        if action.type is ActionType.INSPECT:
            inspected = any(e.get("type") == "OBJECT_INSPECTED" and e.get("actor_id") == actor_id
                            for e in self.state.event_log[before:])
            if inspected:
                self._examine_doorways(actor_id)
        return result

    def _record(self, kind: str, trap: DoorTrap | None = None, **fields: Any) -> dict[str, Any]:
        state = self.state
        record = {"kind": kind, "game_time": round(state.elapsed_time, 3),
                  "mission_time_remaining": round(state.mission_time_remaining, 3)}
        if trap is not None:
            record.update({"trap_id": trap.trap_id, "owner": trap.owner_id, "target_type": "doorway",
                           "target_id": trap.door_id, "variant": trap.variant})
        record.update(fields)
        self.records.append(record)
        state.log_event("ENHANCED_TRAP_" + kind.upper(), **{k: v for k, v in record.items() if k != "kind"})
        return record

    def _place(self, apply, actor_id: str, action: GameAction, dt: float):
        state = self.state
        if not self._object_trap_accepted(actor_id, action):
            return apply(actor_id, action, dt)  # the engine refuses it exactly as before
        n = self.placements.get(actor_id, 0) + 1
        self.placements[actor_id] = n
        personality = self.personalities.get(actor_id, "opportunist")
        rng = self._rng("traps", actor_id, n)
        doors = self.eligible_doors(actor_id)
        wants_door = rng.random() < DOOR_PREFERENCE.get(personality, 0.5)
        if not doors or not wants_door:
            before = len(state.event_log)
            result = apply(actor_id, action, dt)  # an object trap: the engine path, unchanged
            placed = next((e for e in state.event_log[before:] if e.get("type") == "TRAP_PLACED"
                           and e.get("actor_id") == actor_id), None)
            if placed is not None:
                self._record("resolved", owner=actor_id, target_type="object", target_id=placed.get("object_id"),
                             variant=LEGACY_VARIANT.get(placed.get("trap_type"), LEGACY_DEFAULT_VARIANT),
                             source_trap_type=placed.get("trap_type"), decision=n,
                             reason="no eligible doorway" if not doors else "personality chose the object")
            return result
        door = doors[0]
        variant = _weighted(rng, FLAVOR.get(personality, FLAVOR["opportunist"]))
        actor = state.actors[actor_id]
        if action.accompanying_message is not None:
            # the engine delivers an accompanying message before the physical action; a WAIT carries it
            apply(actor_id, GameAction(ActionType.WAIT, accompanying_message=action.accompanying_message), dt)
        trap_type = action.trap_type
        actor.trap_inventory[trap_type] -= 1
        actor.moving = False
        actor.action_cooldown = constants.TRAP_PLACEMENT_COST[trap_type]  # the engine's own placement cost
        self.instantiate(actor_id, door.id, variant, source_trap_type=trap_type, decision=n, personality=personality,
                         candidates=[d.id for d in doors], cost_s=constants.TRAP_PLACEMENT_COST[trap_type])
        return None

    def instantiate(self, owner_id: str, door_id: str, variant: str, *, source_trap_type: str, decision: int = 0,
                    **fields: Any) -> DoorTrap:
        """A door trap, armed on `door_id` for `owner_id` (the PLACE_TRAP path above; also the demo / test entry)."""
        state = self.state
        if variant not in VARIANTS:
            raise ValueError(f"unknown trap variant {variant!r}")
        if self.armed_on(door_id) is not None:
            raise ValueError(f"{door_id} already has an armed trap (at most one per doorway)")
        door = state.level.get_object(door_id)
        self._seq += 1
        trap = DoorTrap(trap_id=f"door_trap_{self._seq:02d}", door_id=door_id, owner_id=owner_id, variant=variant,
                        source_trap_type=source_trap_type, placed_at=round(state.elapsed_time, 3),
                        mission_time_remaining=round(state.mission_time_remaining, 3), decision=decision,
                        position=tuple(float(v) for v in door.collision_rect.center))
        self.traps[trap.trap_id] = trap
        self._record("placed", trap, source_trap_type=source_trap_type, decision=decision, **fields)
        return trap

    def _examine_doorways(self, actor_id: str) -> None:
        actor = self.state.actors[actor_id]
        for trap in self.armed():
            if trap.owner_id == actor_id or actor_id in trap.detected_by:
                continue
            if math.hypot(trap.position[0] - actor.x, trap.position[1] - actor.y) > DETECT_RANGE:
                continue
            roll = self._rng("trapdetect", trap.trap_id, actor_id).random()
            if roll < DETECTION_PROBABILITY[trap.variant]:
                trap.detected_by[actor_id] = round(self.state.elapsed_time, 3)
                self._record("detected", trap, actor_id=actor_id, roll=round(roll, 6))
            else:
                self._record("not_detected", trap, actor_id=actor_id, roll=round(roll, 6))

    def _disarm(self, apply, actor_id: str, action: GameAction, dt: float) -> bool:
        """True when this DISARM resolved to a door trap (the engine call is then not made)."""
        state = self.state
        obj = state._target_object(actor_id, action.target_id)
        if obj is not None and state.trap_on(obj.id) is not None:
            return False  # an object trap: the engine's DISARM, unchanged
        actor = state.actors[actor_id]
        known = [t for t in self.visible_to(actor_id)
                 if math.hypot(t.position[0] - actor.x, t.position[1] - actor.y) <= DISARM_RANGE]
        if not known:
            return False
        trap = min(known, key=lambda t: (math.hypot(t.position[0] - actor.x, t.position[1] - actor.y), t.trap_id))
        if action.accompanying_message is not None:
            apply(actor_id, GameAction(ActionType.WAIT, accompanying_message=action.accompanying_message), dt)
        actor.moving = False
        if trap.owner_id == actor_id:
            trap.armed, trap.status, trap.disarmed_at = False, "REMOVED", round(state.elapsed_time, 3)
            actor.action_cooldown = constants.TIME_COST_DISARM
            self._record("removed", trap, actor_id=actor_id)
            return True
        p = DISARM_PROBABILITY[trap.variant]
        roll = self._rng("trapdisarm", trap.trap_id, actor_id).random()
        if roll >= p:
            self._record("disarm_failed", trap, actor_id=actor_id, success_probability=p, roll=round(roll, 6))
            self._trigger(trap, actor_id, cause="failed_disarm")
            return True
        trap.armed, trap.status, trap.disarmed_at = False, "DISARMED", round(state.elapsed_time, 3)
        actor.action_cooldown = constants.TIME_COST_DISARM
        self._record("disarmed", trap, actor_id=actor_id, success_probability=p, roll=round(roll, 6))
        return True

    # -- the frame hook --------------------------------------------------------------------------------
    def after_update(self, dt: float) -> None:
        state = self.state
        for actor_id, actor in state.actors.items():
            now = (actor.x, actor.y)
            prev = self._prev.get(actor_id, now)
            self._prev[actor_id] = now
            if state.mission_complete.get(actor_id) or state.out_of_play(actor_id) or actor_id in state.exit_walking:
                continue
            if math.hypot(now[0] - prev[0], now[1] - prev[1]) > TELEPORT_STEP:
                continue
            for trap in self.armed():
                if trap.owner_id == actor_id or actor_id in trap.detected_by:
                    continue  # owner immunity; a detected trap is stepped over
                door = state.level.get_object(trap.door_id)
                if door is not None and self.crosses(door.collision_rect, prev, now):
                    self._trigger(trap, actor_id, cause="crossed_doorway")
        self._advance_effects(dt)

    @staticmethod
    def crosses(rect, prev: tuple[float, float], now: tuple[float, float]) -> bool:
        """The feet changed side of the doorway's threshold line, inside the door's span. A wide door rect is in a
        horizontal wall (crossed vertically); a tall one is in a vertical wall (crossed horizontally)."""
        (px, py), (nx, ny) = prev, now
        if rect.width >= rect.height:
            line, lo, hi = rect.centery, rect.left - SPAN_MARGIN, rect.right + SPAN_MARGIN
            a, b, ua, ub = py, ny, px, nx
        else:
            line, lo, hi = rect.centerx, rect.top - SPAN_MARGIN, rect.bottom + SPAN_MARGIN
            a, b, ua, ub = px, nx, py, ny
        if (a < line) == (b < line):
            return False
        t = (line - a) / (b - a) if b != a else 0.0
        u = ua + (ub - ua) * t
        return lo <= u <= hi

    def _trigger(self, trap: DoorTrap, victim_id: str, *, cause: str) -> None:
        state = self.state
        victim = state.actors[victim_id]
        trap.armed, trap.status = False, "TRIGGERED"
        trap.triggered_at, trap.victim_id = round(state.elapsed_time, 3), victim_id
        state.trap_hits[victim_id] = state.trap_hits.get(victim_id, 0) + 1  # traps that went off under this spy
        effect: dict[str, Any] = {"effect": EFFECT_LABEL[trap.variant], "cue": CUE[trap.variant]}
        v = trap.variant
        if v == "explosive":
            victim.stunned_timer = max(victim.stunned_timer, EXPLOSIVE_STUN)
            victim.moving = False
            lives_before = state.lives.get(victim_id) if state.lives_enabled else None
            eliminated = state.lose_life(victim_id, "door_trap_explosive")
            effect.update({"stun_s": EXPLOSIVE_STUN, "lives_before": lives_before,
                           "lives_after": state.lives.get(victim_id) if state.lives_enabled else None,
                           "eliminated": bool(eliminated)})
            if eliminated:
                drop = self._nearest_searchable(victim)
                if drop is not None:
                    effect["items_dropped"] = state.drop_items_at(victim_id, drop.id)
                    effect["dropped_on"] = drop.id
        elif v == "electric":
            victim.stunned_timer = max(victim.stunned_timer, ELECTRIC_STUN)
            victim.moving = False
            effect["stun_s"] = ELECTRIC_STUN
        elif v == "gas":
            self.effects.append(Effect("gas_cloud", trap.trap_id, trap.owner_id, GAS_SECONDS, position=trap.position))
            self.effects.append(Effect("poison", trap.trap_id, trap.owner_id, GAS_SECONDS, actor_id=victim_id))
            effect.update({"cloud_s": GAS_SECONDS, "poison_s": GAS_SECONDS, "speed_factor": GAS_SLOW})
        elif v == "snare":
            victim.moving = False
            self.effects.append(Effect("root", trap.trap_id, trap.owner_id, SNARE_SECONDS, actor_id=victim_id))
            effect.update({"rooted_s": SNARE_SECONDS})
        elif v == "alarm" and trap.owner_id in state.actors:
            state.actors[trap.owner_id].alarm_reveal_timer = constants.ALARM_REVEAL_DURATION
            state.log_event("ALARM_REVEALED", actor_id=trap.owner_id, victim_id=victim_id,
                            seconds=constants.ALARM_REVEAL_DURATION)
            effect["reveal_s"] = constants.ALARM_REVEAL_DURATION
        self._record("triggered", trap, victim=victim_id, cause=cause,
                     position=[round(victim.x, 1), round(victim.y, 1)], **effect)
        self._apply_speeds()

    def _nearest_searchable(self, actor):
        room = self._room_of(actor)
        objects = [o for o in self.state.level.objects if o.trappable and self.state.level.object_room(o) == room]
        if not objects:
            return None
        return min(objects, key=lambda o: (math.hypot(o.center[0] - actor.x, o.center[1] - actor.y), o.id))

    def _advance_effects(self, dt: float) -> None:
        if dt > 0:
            for effect in self.effects:
                effect.remaining = max(0.0, effect.remaining - dt)
        ended = [e for e in self.effects if e.remaining <= 0]
        if ended:
            self.effects = [e for e in self.effects if e.remaining > 0]
            for e in ended:
                trap = self.traps.get(e.trap_id)
                self._record("effect_ended", trap, effect=e.kind, actor_id=e.actor_id)
        self._apply_speeds()

    def speed_factor(self, actor_id: str) -> float:
        actor = self.state.actors[actor_id]
        factor = 1.0
        for e in self.effects:
            if e.kind == "root" and e.actor_id == actor_id:
                return 0.0
            if e.kind == "poison" and e.actor_id == actor_id:
                factor = min(factor, GAS_SLOW)
            elif e.kind == "gas_cloud" and e.owner_id != actor_id and e.position is not None \
                    and math.hypot(e.position[0] - actor.x, e.position[1] - actor.y) <= GAS_RADIUS:
                factor = min(factor, GAS_SLOW)
        return factor

    def _apply_speeds(self) -> None:
        for actor_id, actor in self.state.actors.items():
            factor = self.speed_factor(actor_id)
            base = self._base_speed.get(actor_id, constants.PLAYER_SPEED)
            if factor < 1.0:
                actor.speed = base * factor
                self._slowed.add(actor_id)
            elif actor_id in self._slowed:
                actor.speed = base
                self._slowed.discard(actor_id)

    # -- summaries ---------------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {"door_traps": [t.public() for t in self.traps.values()],
                "events": len(self.records),
                "triggered": sum(1 for t in self.traps.values() if t.status == "TRIGGERED")}


def variant_of_engine_trap(trap_type: str | None) -> str:
    """The variant name of an engine (object) trap - the legacy default for anything unknown."""
    return LEGACY_VARIANT.get(trap_type or "", LEGACY_DEFAULT_VARIANT)


__all__ = ["TrapResolver", "DoorTrap", "Effect", "VARIANTS", "LEGACY_VARIANT", "LEGACY_DEFAULT_VARIANT", "CUE",
           "FLAVOR", "DOOR_PREFERENCE", "variant_of_engine_trap"]
