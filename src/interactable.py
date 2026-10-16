"""Data-driven world objects (furniture, doors, decoration)."""

from dataclasses import dataclass, field

import pygame


@dataclass
class GameObject:
    """A single object placed in a level, as loaded from JSON.

    `position` anchors the object's logical footprint - `collision_size`
    from there is the rect used for movement collision and interaction
    range (also serves as `interaction_rect`; nothing currently needs a
    third, different box for that).

    `render_width`/`render_height`/`render_offset` are purely cosmetic and
    independent of collision_size:

    - both None -> render is capped to collision_size (the old default)
    - only one given -> the other is derived from the sprite's real aspect
      ratio at draw time (see assets.get_fitted) - a desk can say
      `render_width: 220` and get a correctly-proportioned height without
      anyone hand-computing it
    - both given -> treated as a bounding box the sprite is fit inside,
      aspect preserved (never independently stretched)

    A sprite is never stretched to fill collision_size. render_offset
    defaults to bottom-center-anchoring the (possibly sprite-dependent)
    render size on the footprint, computed in renderer.py once the actual
    size is known - see Renderer._placement_rect.
    """

    id: str
    type: str
    label: str
    position: tuple[float, float]
    collision_size: tuple[float, float]
    solid: bool
    interactable: bool
    interaction: str | None
    message: str | None
    sprite: str | None = None
    sprite_open: str | None = None  # used by doors when their state is "open"
    sprite_locked: str | None = None  # M4B - used by doors when their state is "locked"; falls back to `sprite`
    contains: list[str] = field(default_factory=list)  # mission item ids found on search
    default_state: str = "closed"  # doors only - the door_states value GameState.new_game seeds
    connects: tuple[str, str] | None = None  # doors only - the two room ids either side (Milestone 4A)
    # M4B: static "shape of the lock" - level geometry, like where a door
    # is, not hidden per-actor knowledge. lock_type is one of
    # "key"/"keycard"/"code" (or None - unlocked); `requires` names the
    # item id (key/keycard) or knowledge id (code) that satisfies it. Which
    # *specific* object holds that key/code is decided per-seed by
    # src/dependency_generation.py, not authored here.
    lock_type: str | None = None
    requires: str | None = None
    # M4B: knowledge id this object teaches on a successful search (e.g.
    # "safe_code") - added to the searching actor's Actor.knowledge, never
    # their inventory (Milestone_4.md: "codes are knowledge, NEVER
    # inventory"). Which object reveals which knowledge is also decided
    # per-seed - see dependency_generation.apply_placement.
    reveals_knowledge: str | None = None
    render_width: float | None = None
    render_height: float | None = None
    render_offset: tuple[float, float] | None = None  # None -> auto bottom-center anchor

    @property
    def render_max_size(self) -> tuple[float | None, float | None]:
        """Bounding box passed to assets.get_fitted(). If neither render
        dimension is set, that box is the footprint itself (contain-fit);
        otherwise either component may be None, meaning "derive from the
        sprite's aspect ratio" rather than "cap at this footprint size".
        """
        if self.render_width is None and self.render_height is None:
            return self.collision_size
        return (self.render_width, self.render_height)

    @property
    def collision_rect(self) -> pygame.Rect:
        x, y = self.position
        w, h = self.collision_size
        return pygame.Rect(int(x), int(y), int(w), int(h))

    @property
    def rect(self) -> pygame.Rect:
        """Alias for collision_rect - used wherever "footprint" is meant."""
        return self.collision_rect

    @property
    def center(self) -> tuple[float, float]:
        return self.collision_rect.center

    @property
    def prompt_text(self) -> str:
        verb = (self.interaction or "use").capitalize()
        return f"[E] {verb} {self.label}"

    @property
    def trappable(self) -> bool:
        """Whether INSPECT/PLACE_TRAP/DISARM apply to this object at all -
        doors (Room A, the exit) are excluded; every other interactable is
        fair game. A property of the object's type, not of game state.
        """
        return self.interactable and self.type != "door"
