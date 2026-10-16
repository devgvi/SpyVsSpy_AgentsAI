"""Visual-only sprite-pose selection for the player sprite.

Reads `player.facing`/`player.moving` (physical facts already on
GameState) and the GameAction each controller just produced, but owns its
own pose-hold timer - that's presentation detail, not gameplay state, so
it lives here in the rendering layer rather than on GameState. See
`Renderer._draw_actor` for how it's driven each frame.
"""

from src.actions import ActionType

ACTION_POSE_HOLD_SECONDS = 0.35  # how long a search/inspect/trap/disarm pose is shown before reverting

# Which GameAction types get a dedicated pose (the asset pack's search,
# inspect, place_trap, disarm sprites). MOVE/WAIT aren't here - moving vs
# idle already comes from `Actor.moving`, not from the action type.
_ACTION_POSES = {
    ActionType.INTERACT: "search",  # search AND open-door/escape all use this one pose - no distinct sprite for those
    ActionType.INSPECT: "inspect",
    ActionType.PLACE_TRAP: "place_trap",
    ActionType.DISARM: "disarm",
}


class PlayerAnimator:
    def __init__(self) -> None:
        self._pose: str | None = None
        self._pose_timer = 0.0

    def trigger_action(self, action_type: ActionType) -> None:
        """Called once per frame with whatever action this actor's
        controller just produced. A no-op for MOVE/WAIT or an action that
        was dropped (e.g. the actor was stunned) - see the call site.
        """
        pose = _ACTION_POSES.get(action_type)
        if pose is not None:
            self._pose = pose
            self._pose_timer = ACTION_POSE_HOLD_SECONDS

    def update(self, dt: float) -> None:
        if self._pose_timer > 0:
            self._pose_timer = max(0.0, self._pose_timer - dt)
            if self._pose_timer == 0:
                self._pose = None

    def sprite_key(self, facing: str, moving: bool) -> str:
        if self._pose is not None:
            return f"{self._pose}_{facing}"
        return f"walk_{facing}" if moving else f"idle_{facing}"


# -- bugfix spawn/combat/visual polish (Part D): multi-frame animation -----------

WALK_FPS = 12.0  # the V4 walk cycles are 13-15 frames
ACTION_FPS = 4.0
LOOP_FPS = 5.0
FIGHT_SECONDS = 0.7
RESPAWN_SECONDS = 0.8
CELEBRATE_SECONDS = 1.2
HIT_SECONDS = 0.5
DISAPPEAR_SECONDS = 1.0  # the last second of a KO: the spy fades before it respawns elsewhere


class SpyAnimator:
    """Chooses (state, frame index) for one spy each frame, from physical
    facts only - the engine state and the action just applied. Priority:
    KO (hit -> ko -> disappear), respawn, celebrate, fight, trap stun (hit),
    an action pose (held while the action's time cost runs), walk, idle."""

    def __init__(self) -> None:
        self._pose: str | None = None
        self._pose_timer = 0.0
        self._clock = 0.0
        self._timers: dict[str, float] = {}
        self._state_started: dict[str, float] = {}
        self._last_state: str | None = None

    def trigger_action(self, action_type: ActionType) -> None:
        pose = _ACTION_POSES.get(action_type)
        if pose is not None:
            self._pose = pose
            self._pose_timer = ACTION_POSE_HOLD_SECONDS

    def start(self, state: str, seconds: float) -> None:
        """An event-driven state (fight, respawn, celebrate, hit) for `seconds`."""
        self._timers[state] = seconds
        self._state_started[state] = self._clock

    def active(self, state: str) -> bool:
        return self._timers.get(state, 0.0) > 0

    def update(self, dt: float) -> None:
        self._clock += dt
        if self._pose_timer > 0:
            self._pose_timer = max(0.0, self._pose_timer - dt)
        for key in list(self._timers):
            self._timers[key] = max(0.0, self._timers[key] - dt)

    def choose(self, actor, *, ko_left: float | None, ko_total: float, frames=None) -> tuple[str, int]:
        """`frames(state) -> frame count` for this spy and facing (default: the v2 counts)."""
        if frames is None:
            from src.character_art import V2_FRAMES
            frames = V2_FRAMES.__getitem__

        class _Counts:
            def __getitem__(self, state):
                return frames(state)
        STATE_FRAMES = _Counts()

        if ko_left is not None:
            elapsed = ko_total - ko_left
            if elapsed < HIT_SECONDS:
                return "hit", int(elapsed * LOOP_FPS * 2) % STATE_FRAMES["hit"]
            if ko_left <= DISAPPEAR_SECONDS:
                progress = 1.0 - ko_left / DISAPPEAR_SECONDS
                return "disappear", min(STATE_FRAMES["disappear"] - 1, int(progress * STATE_FRAMES["disappear"]))
            return "ko", int(self._clock * LOOP_FPS / 2) % STATE_FRAMES["ko"]
        for state, seconds in (("respawn", RESPAWN_SECONDS), ("celebrate", CELEBRATE_SECONDS)):
            if self.active(state):
                progress = 1.0 - self._timers[state] / seconds
                return state, min(STATE_FRAMES[state] - 1, int(progress * STATE_FRAMES[state]))
        if self.active("fight"):
            return "fight", int(self._clock * LOOP_FPS * 2) % STATE_FRAMES["fight"]
        if actor.is_stunned or self.active("hit"):
            return "hit", int(self._clock * LOOP_FPS) % STATE_FRAMES["hit"]
        if self._pose is not None and (self._pose_timer > 0 or actor.is_busy):
            return self._pose, int(self._clock * ACTION_FPS) % STATE_FRAMES[self._pose]
        self._pose = None
        if actor.moving:
            return "walk", int(self._clock * WALK_FPS) % STATE_FRAMES["walk"]
        return "idle", 0
