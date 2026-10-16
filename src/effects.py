"""Milestone 4I ("Trap Effects & Gameplay Feedback"): presentation-only
trap-trigger animations.

`EffectAnimation` reacts to `TRAP_TRIGGERED` events already sitting on
`GameState.event_log` - it is never a second source of gameplay truth,
and never read by any controller (that milestone's section 7: "Effects
are presentation state, not hidden game truth exposed to controllers").
`GameState`/`Observation` are completely unaware this module exists;
`src/renderer.py` is the only consumer. Frame *content* and duration are
config-driven from `constants.TRAP_EFFECT_FRAMES`/`TRAP_STUN_SECONDS`, not
hard-coded per trap type here - the same "one generic mechanism, keyed by
trap_type" pattern M4F's trap-value formula and M4G's agent profiles both
already established, so a fifth trap type would need new data, not new
branches.
"""

from dataclasses import dataclass


@dataclass
class EffectAnimation:
    effect_type: str
    world_position: tuple[float, float]  # used verbatim when follows_actor_id is None
    frames: tuple  # tuple[pygame.Surface, ...], pre-scaled - see Renderer._effect_frames
    frame_duration: float  # seconds per frame
    elapsed: float = 0.0
    loop: bool = False
    follows_actor_id: str | None = None  # if set, the renderer reads that actor's *live* position instead

    @property
    def total_duration(self) -> float:
        return self.frame_duration * len(self.frames)

    @property
    def finished(self) -> bool:
        """Looping effects never finish on their own - nothing in this
        milestone actually uses loop=True (every trap's frames are
        stretched to match its own gameplay penalty exactly, so the
        animation and the penalty always start and end together), but the
        field exists for a future effect that should persist until some
        external condition clears it instead.
        """
        return (not self.loop) and self.elapsed >= self.total_duration

    def current_frame(self):
        if not self.frames:
            return None
        index = int(self.elapsed / self.frame_duration)
        index = index % len(self.frames) if self.loop else min(index, len(self.frames) - 1)
        return self.frames[index]

    def advance(self, dt: float) -> None:
        self.elapsed += dt
