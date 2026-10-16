"""Actor: position, facing, inventory, and collision-aware movement.

Shared by Black Spy (human-controlled) and White Spy (bot-controlled) -
per Milestone 3A, both use the exact same movement, collision, and
inventory model. Nothing here knows or cares which controller is driving
it.
"""

import math
from dataclasses import dataclass, field

import pygame

from src import constants
from src.inventory import Inventory

Facing = str  # one of "up", "down", "left", "right"


@dataclass
class Actor:
    id: str
    x: float
    y: float
    facing: Facing = "down"
    speed: float = constants.PLAYER_SPEED
    moving: bool = False  # physical fact (button held/bot-driven), read by the renderer for animation
    inventory: Inventory = field(default_factory=Inventory)
    trap_inventory: dict[str, int] = field(default_factory=lambda: dict(constants.TRAP_LOADOUT))
    stunned_timer: float = 0.0  # seconds remaining immobilized after triggering an enemy trap
    # M4D: seconds remaining "busy" after a timed action (search/inspect/
    # place trap/disarm/unlock/open) - mechanically identical to
    # stunned_timer (blocks every action until it expires - see
    # GameState.apply_action), but a distinct field/name because it means
    # something different (an actor doing something on purpose, not being
    # punished for a trap) and the two must read separately in the HUD/F3.
    action_cooldown: float = 0.0
    # M4B: codes/passwords this actor has learned (e.g. "safe_code") -
    # deliberately separate from `inventory`. Milestone_4.md is explicit
    # that a code is knowledge, never a physical item: it can't be dropped,
    # stolen, or found on a body, and both spies can independently learn
    # the same code without either one running out of it.
    knowledge: set[str] = field(default_factory=set)
    # M4C ("Exploration, Memory & Backtracking"): persistent per-actor
    # memory, updated from GameState.update() every tick and never reset
    # mid-episode - "knowledge persists" is the whole point. Room/object
    # *geometry* was never hidden to begin with (no fog-of-war - that's a
    # separate, later milestone), so these mostly mirror what's already
    # globally visible; the one genuinely new hide/reveal boundary this
    # milestone adds is `last_seen_opponent` (see GameState._update_room_memory).
    visited_rooms: set[str] = field(default_factory=set)  # every room id this actor has ever stood in
    # Bugfix (exploration state): which doors this actor has actually walked
    # through (room A -> room B across the door connecting them), and the last
    # room it stood in (door gaps belong to no room, so this bridges them).
    # Self-knowledge only - updated in GameState._update_room_memory.
    traversed_doors: set[str] = field(default_factory=set)
    last_room: str | None = None
    known_locked_objects: dict[str, str] = field(default_factory=dict)  # object_id -> lock_type, first encountered
    last_seen_opponent: dict | None = None  # {actor_id, room, position, facing, game_time} or None if never seen
    # M4H ("Episode / Replay Foundation"): Milestone_4.md names "distance
    # travelled" as a metric event logs should eventually support, but no
    # event currently records movement (it happens continuously via MOVE,
    # not as a discrete per-step event) - cheap to track directly instead
    # of retrofitting one. Only the actually-applied step counts (see
    # _move_axis) - a MOVE blocked entirely by collision adds nothing.
    distance_travelled: float = 0.0
    # M4I ("Trap Effects & Gameplay Feedback"): seconds remaining that this
    # actor can see the *opponent's* live position, thanks to their own
    # alarm trap having just gone off - the one trap effect that acts on
    # someone other than the victim. Lives on the *owner*, not the victim;
    # surfaced read-only via Observation.revealed_opponent (build_observation),
    # never read directly off GameState by any controller.
    alarm_reveal_timer: float = 0.0

    @property
    def is_stunned(self) -> bool:
        return self.stunned_timer > 0

    @property
    def is_busy(self) -> bool:
        return self.action_cooldown > 0

    @property
    def has_alarm_reveal(self) -> bool:
        return self.alarm_reveal_timer > 0

    @property
    def collision_rect(self) -> pygame.Rect:
        w, h = constants.PLAYER_COLLISION_SIZE
        return pygame.Rect(int(self.x - w / 2), int(self.y - h / 2), int(w), int(h))

    @property
    def sprite_rect(self) -> pygame.Rect:
        """Max sprite bounding box, bottom-anchored at (x, y) - the feet.

        Collision stays independently centered on (x, y) (see
        collision_rect); this only changes where the visual is drawn.
        """
        w, h = constants.PLAYER_SIZE
        return pygame.Rect(int(self.x - w / 2), int(self.y - h), int(w), int(h))

    def update_facing(self, direction: tuple[float, float]) -> None:
        dx, dy = direction
        if dx == 0 and dy == 0:
            return
        # Dominant axis decides facing; ties favor horizontal.
        if abs(dx) >= abs(dy):
            self.facing = "right" if dx > 0 else "left"
        else:
            self.facing = "down" if dy > 0 else "up"

    def move(self, direction: tuple[float, float], dt: float, solids: list[pygame.Rect]) -> None:
        """Move by direction (already normalized), resolving collisions per axis."""
        dx, dy = direction
        if dx == 0 and dy == 0:
            self.moving = False
            return

        self.moving = True
        self.update_facing(direction)

        step_x = dx * self.speed * dt
        step_y = dy * self.speed * dt

        # Bugfix (navigation): a long frame (slow frame, the live loop's 0.1 s
        # cap, speed x2) used to be one all-or-nothing collision test per axis,
        # so near a wall or a piece of furniture the whole step was rejected
        # and the actor stalled short of its path. Long steps are split into
        # sub-steps of at most MAX_COLLISION_SUBSTEP px, each collision-checked
        # the same way. Steps up to that size (every existing test and
        # evaluation dt: 3-6 px) take exactly one sub-step, as before.
        substeps = max(1, math.ceil(max(abs(step_x), abs(step_y)) / constants.MAX_COLLISION_SUBSTEP))
        for _ in range(substeps):
            self._move_axis(step_x / substeps, 0, solids)
            self._move_axis(0, step_y / substeps, solids)

    def _move_axis(self, dx: float, dy: float, solids: list[pygame.Rect]) -> None:
        new_x = self.x + dx
        new_y = self.y + dy

        w, h = constants.PLAYER_COLLISION_SIZE
        test_rect = pygame.Rect(int(new_x - w / 2), int(new_y - h / 2), int(w), int(h))

        for solid in solids:
            if test_rect.colliderect(solid):
                return  # blocked: leave position unchanged along this axis

        self.distance_travelled += abs(dx) + abs(dy)  # exactly one of dx/dy is nonzero per call - see move()
        self.x = new_x
        self.y = new_y
