"""Draws GameState to the screen. Owns no gameplay rules.

Everything here only reads from GameState/Level; it never mutates them.
Every object may declare a PNG sprite in level data - if that file is
missing, we fall back to the original colored placeholder rect (spec
section 1's "this fallback is important").

Standing objects (furniture, doors, the player) are drawn in a single
Y-sorted pass keyed by each one's collision-rect bottom ("feet"), so the
spy appears in front of furniture it's below and behind furniture it's
above (Milestone 2.6). Floor and rug are always beneath that; walls
always on top of it.

Milestone 4A: the embassy is bigger than the window, so every world-space
draw goes through a scrolling camera (see `_compute_camera` / `_screen_rect`
/ `_screen_pos`) that follows Black Spy, clamped to the map's edges.

Milestone 4A.1: the camera also zooms (`constants.CAMERA_ZOOM`) - world
layers are drawn onto `self.world` (a Surface sized WINDOW/zoom, so the
existing camera/screen-space math above is unchanged) which is then
scaled up onto the real screen. World space itself (collision, level
data, pathfinding) never changes - only what fraction of it a screen
pixel represents. The HUD is drawn straight onto `self.screen` afterward,
in real screen space, untouched by either the camera or the zoom.
"""

import pygame

from src import assets, constants
from src.actions import GameAction
from src import character_art
from src.animation import CELEBRATE_SECONDS, FIGHT_SECONDS, HIT_SECONDS, RESPAWN_SECONDS, SpyAnimator
from src.interactable import GameObject
from src.effects import EffectAnimation
from src.layout_validation import door_clearance_rects, door_visual_exclusion_rects
from src.scoring import ScoreBreakdown, compute_score
from src.state import GameState

FACING_OFFSETS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


DOOR_ART_DIR = "assets/doors_v2"


def _door_sprite_path(obj: GameObject, door_state: str | None) -> str | None:
    """M4B added a third door_states value ("locked") alongside open/
    closed - falls back to the plain `sprite` (closed look) if a door has
    no dedicated `sprite_locked`, same fallback pattern `sprite_open`
    already used.

    Bugfix visual polish (Part F): a door that has art in the level uses the
    one opaque door set in assets/doors_v2 (tools/build_live_sprites.py) -
    same geometry for every door, 96x48 in a horizontal wall and 48x96 in a
    vertical one: closed, locked (padlock), keycard (card reader), open (an
    empty doorway with the leaf swung against a post - never a see-through
    closed door), and a distinct green exit."""
    if obj.type != "door":
        return obj.sprite
    if obj.sprite:
        if obj.id == "exit_door":
            return f"{DOOR_ART_DIR}/exit_{'open' if door_state == 'open' else 'closed'}.png"
        orientation = "horizontal" if obj.collision_rect.width >= obj.collision_rect.height else "vertical"
        if door_state == "open":
            look = "open"
        elif door_state == "locked":
            look = "keycard" if obj.lock_type == "keycard" else "locked"
        else:
            look = "closed"
        return f"{DOOR_ART_DIR}/door_{orientation}_{look}.png"
    if door_state == "open" and obj.sprite_open:
        return obj.sprite_open
    if door_state == "locked" and obj.sprite_locked:
        return obj.sprite_locked
    return obj.sprite


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.zoom = constants.CAMERA_ZOOM
        self.view_width = constants.WINDOW_WIDTH / self.zoom
        self.view_height = constants.WINDOW_HEIGHT / self.zoom
        self.world = pygame.Surface((round(self.view_width), round(self.view_height)))

        try:
            if not pygame.font.get_init():
                pygame.font.init()
        except Exception:
            # Some unit tests and headless setups stub pygame.init() without
            # initializing the font subsystem; fall back to the default
            # system font only when the backend is still usable.
            pass

        self.font = pygame.font.SysFont("couriernew", 16, bold=True)
        self.label_font = pygame.font.SysFont("couriernew", 12)
        self.title_font = pygame.font.SysFont("couriernew", 20, bold=True)
        self.debug_font = pygame.font.SysFont("couriernew", 11, bold=True)
        self.animators = {constants.BLACK_SPY_ID: SpyAnimator(), constants.WHITE_SPY_ID: SpyAnimator()}
        self.debug_enabled = False
        self.camera = (0, 0)

        # M4I: each trap type's 8-frame trigger effect, loaded once and
        # fit to a shared max bounding box (constants.EFFECT_RENDER_SIZE) -
        # same get_fitted/never-independently-stretch discipline every
        # other sprite in this project already follows.
        self._effect_frames: dict[str, tuple[pygame.Surface, ...]] = {
            trap_type: tuple(
                frame
                for frame in (
                    assets.get_fitted(path, (constants.EFFECT_RENDER_SIZE, constants.EFFECT_RENDER_SIZE))
                    for path in paths
                )
                if frame is not None
            )
            for trap_type, paths in constants.TRAP_EFFECT_FRAMES.items()
        }
        self._active_effects: list[EffectAnimation] = []
        # How many of state.event_log's entries this renderer has already
        # scanned for TRAP_TRIGGERED events - never re-processed, so an
        # effect is spawned exactly once per real trigger, no matter how
        # many frames render() gets called for while it plays out.
        self._processed_event_count = 0
        self.last_animation: dict[str, tuple[str, int]] = {}  # actor -> (state, frame) drawn last (tests / debug)

    def toggle_debug(self) -> None:
        self.debug_enabled = not self.debug_enabled

    def render(
        self,
        state: GameState,
        dt: float,
        selected_trap_type: str | None = None,
        actions: dict[str, GameAction] | None = None,
        bot_controller=None,
    ) -> None:
        self.camera = self._compute_camera(state)

        self._draw_floor(state)
        self._draw_decorations(state)
        self._draw_y_sorted(state, dt, actions)
        self._update_effects(state, dt)
        self._draw_effects(state)
        if self.debug_enabled:
            self._draw_debug(state)

        pygame.transform.scale(self.world, self.screen.get_size(), self.screen)

        self._draw_hud(state, selected_trap_type, bot_controller)
        if self.debug_enabled:
            self._draw_bot_diagnostics(state, bot_controller)
        pygame.display.flip()

    # -- camera -----------------------------------------------------------

    def _compute_camera(self, state: GameState) -> tuple[int, int]:
        """Top-left world coordinate of the viewport, following Black Spy
        (the human player - White Spy has no on-screen presence of its
        own to center on either, same as the HUD being Black Spy's POV),
        clamped so the view never scrolls past the map's edges. The
        viewport is `self.view_width x self.view_height` - the *zoomed*
        visible area, smaller than the window itself.
        """
        focus = state.actors[constants.BLACK_SPY_ID]
        map_w, map_h = state.level.size

        cam_x = focus.x - self.view_width / 2
        cam_y = focus.y - self.view_height / 2

        max_cam_x = max(0, map_w - self.view_width)
        max_cam_y = max(0, map_h - self.view_height)

        return int(min(max(0, cam_x), max_cam_x)), int(min(max(0, cam_y), max_cam_y))

    def _screen_rect(self, world_rect: pygame.Rect) -> pygame.Rect:
        cam_x, cam_y = self.camera
        return pygame.Rect(world_rect.x - cam_x, world_rect.y - cam_y, world_rect.width, world_rect.height)

    def _screen_pos(self, x: float, y: float) -> tuple[int, int]:
        cam_x, cam_y = self.camera
        return int(x - cam_x), int(y - cam_y)

    # -- world layers (drawn onto self.world, then scaled onto self.screen) --

    def _draw_floor(self, state: GameState) -> None:
        tile = assets.get_tile(constants.FLOOR_TILE_SPRITE, constants.FLOOR_TILE_TARGET_SIZE)
        if tile is None:
            self.world.fill(constants.COLOR_FLOOR)
            return

        cam_x, cam_y = self.camera
        tw, th = tile.get_size()
        start_x = cam_x - (cam_x % tw)
        start_y = cam_y - (cam_y % th)
        for y in range(start_y, cam_y + self.world.get_height() + th, th):
            for x in range(start_x, cam_x + self.world.get_width() + tw, tw):
                self.world.blit(tile, (x - cam_x, y - cam_y))

    def _draw_decorations(self, state: GameState) -> None:
        """Rugs: always drawn first, and never as a stretched object
        sprite (spec: rugs "should NOT be scaled using a generic 64x64
        object sprite") - a simple bordered rectangle at the size set in
        level data reads more clearly than a warped icon.
        """
        for obj in state.level.objects:
            if obj.type != "rug":
                continue
            rect = self._screen_rect(obj.collision_rect)
            pygame.draw.rect(self.world, constants.COLOR_RUG, rect)
            border = rect.inflate(-28, -28)
            if border.width > 0 and border.height > 0:
                pygame.draw.rect(self.world, constants.RUG_COLOR_BORDER, border, width=6)

    def _draw_y_sorted(self, state: GameState, dt: float, actions: dict[str, GameAction] | None) -> None:
        """Everything with physical depth - furniture, actors, and (M4A.1)
        walls - drawn in one pass ordered by each thing's own bottom edge,
        so an actor standing in front of (south of) a wall renders over
        it and one standing behind renders under it, the same rule
        furniture already used. Before this, walls were a flat pass drawn
        unconditionally after every actor, so anyone merely *near* a
        horizontal wall - not actually behind it - got its texture painted
        over their sprite; invisible at the old thin 32px walls and boxy
        placeholders, obvious once walls doubled in thickness and actors
        got real art.
        """
        entries = []  # list of (feet_y, draw_fn)

        for obj in state.level.objects:
            if obj.type == "rug":
                continue
            # Part F: an open doorway is floor-level - a spy walking through it
            # is drawn over it, never hidden behind the swung-open door
            is_open_door = obj.type == "door" and state.door_states.get(obj.id) == "open"
            key = float("-inf") if is_open_door else obj.collision_rect.bottom
            entries.append((key, lambda o=obj: self._draw_object(state, o)))

        for wall in state.level.walls:
            entries.append((wall.bottom, lambda w=wall: self._draw_wall(w)))

        for actor_id in state.actors:
            entries.append(
                (
                    state.actors[actor_id].collision_rect.bottom,
                    lambda a=actor_id: self._draw_actor(state, a, dt, actions),
                )
            )

        for _, draw in sorted(entries, key=lambda e: e[0]):
            draw()

    # -- trap effects (Milestone 4I) -----------------------------------------

    def _update_effects(self, state: GameState, dt: float) -> None:
        """Reacts to every new `TRAP_TRIGGERED` event since the last call
        (never re-scanning the whole log - `_processed_event_count` only
        moves forward) by spawning one `EffectAnimation`, then advances
        and prunes whatever's currently playing. Purely presentational -
        see effects.py's docstring; nothing here reads back into
        GameState or Observation.
        """
        new_events = state.event_log[self._processed_event_count :]
        self._processed_event_count = len(state.event_log)

        for event in new_events:
            if event["type"] == "TRAP_TRIGGERED":
                self._spawn_trap_effect(state, event)
            self._animate_event(event)

        for effect in self._active_effects:
            effect.advance(dt)
        self._active_effects = [e for e in self._active_effects if not e.finished]

    def _animate_event(self, event: dict) -> None:
        """Event-driven animation states (presentation only)."""
        kind = event.get("type")
        if kind == "SPY_FIGHT_RESOLVED":
            winner = self.animators.get(event.get("winner_id"))
            if winner is not None:
                winner.start("fight", FIGHT_SECONDS)
            loser = self.animators.get(event.get("loser_id"))
            if loser is not None and (event.get("penalty") or {}).get("type") != "ko_respawn":
                loser.start("hit", HIT_SECONDS)
        elif kind == "ACTOR_RESPAWNED" and event.get("actor_id") in self.animators:
            self.animators[event["actor_id"]].start("respawn", RESPAWN_SECONDS)
        elif kind == "ACTOR_LEFT_BUILDING" and event.get("actor_id") in self.animators:
            self.animators[event["actor_id"]].start("celebrate", CELEBRATE_SECONDS)
        elif kind == "TRAP_TRIGGERED" and event.get("actor_id") in self.animators:
            self.animators[event["actor_id"]].start("hit", HIT_SECONDS)

    def _spawn_trap_effect(self, state: GameState, event: dict) -> None:
        trap_type = event["trap_type"]
        frames = self._effect_frames.get(trap_type)
        if not frames:
            return  # missing/unloaded art - degrade to no visual rather than crash (spec section 1's fallback rule)

        victim_id = event["actor_id"]
        obj = state.level.get_object(event["object_id"])
        victim = state.actors.get(victim_id)
        anchor = obj.center if obj is not None else (victim.x, victim.y) if victim is not None else (0.0, 0.0)
        follows = constants.TRAP_EFFECT_FOLLOWS_ACTOR.get(trap_type, False)
        duration = constants.TRAP_STUN_SECONDS.get(trap_type, 1.0)  # the effect always spans the real penalty exactly

        self._active_effects.append(
            EffectAnimation(
                effect_type=trap_type,
                world_position=anchor,
                frames=frames,
                frame_duration=duration / len(frames),
                follows_actor_id=victim_id if follows else None,
            )
        )

    def _draw_effects(self, state: GameState) -> None:
        for effect in self._active_effects:
            frame = effect.current_frame()
            if frame is None:
                continue
            if effect.follows_actor_id is not None:
                actor = state.actors.get(effect.follows_actor_id)
                position = (actor.x, actor.y) if actor is not None else effect.world_position
            else:
                position = effect.world_position
            rect = frame.get_rect(center=self._screen_pos(*position))
            self.world.blit(frame, rect.topleft)

    def _draw_object(self, state: GameState, obj: GameObject) -> None:
        is_open = obj.type == "door" and state.door_states.get(obj.id) == "open"
        sprite_path = _door_sprite_path(obj, state.door_states.get(obj.id))
        if sprite_path and sprite_path.startswith(DOOR_ART_DIR):
            # Part F: a door fills exactly its own doorway (its collision rect
            # in the wall) - no sliver of floor beside it, same scale for all
            image = assets.get_scaled_to(sprite_path, obj.collision_rect.size)
            render_rect = self._screen_rect(obj.collision_rect)
        else:
            image = assets.get_fitted(sprite_path, obj.render_max_size)
            render_rect = self._screen_rect(self._placement_rect(obj, image))

        if image is not None:
            self.world.blit(image, render_rect.topleft)
        else:
            color = constants.OBJECT_COLORS.get(obj.type, (120, 120, 120))
            if is_open:
                color = tuple(min(255, c + 40) for c in color)
            pygame.draw.rect(self.world, color, render_rect)
            self._draw_centered_label(obj.label, render_rect)

        if obj.id == state.current_interaction_target.get(constants.BLACK_SPY_ID):
            pygame.draw.rect(self.world, constants.COLOR_TEXT, self._screen_rect(obj.collision_rect), width=2)

        self._draw_known_trap_icon(state, obj, render_rect)

    def _draw_known_trap_icon(self, state: GameState, obj: GameObject, render_rect: pygame.Rect) -> None:
        """M4I: a small icon (assets/traps/) in the corner of a trapped
        object - shown only for Black Spy's own trap or one Black Spy has
        actually detected, the exact same visibility rule
        `state.get_prompt_lines`/the HUD prompt already use. An enemy trap
        neither owned nor detected stays invisible here too - this must
        never become a second way to leak hidden trap truth onto the
        screen.
        """
        trap = state.trap_on(obj.id)
        if trap is None:
            return
        black_id = constants.BLACK_SPY_ID
        known = trap["owner_id"] == black_id or state.detected_trap_type(obj.id, black_id) == trap["type"]
        if not known:
            return

        icon = assets.get_fitted(constants.TRAP_ICON_SPRITES.get(trap["type"]), (24, 24))
        if icon is None:
            return
        self.world.blit(icon, (render_rect.left - 4, render_rect.top - 4))

    def _placement_rect(self, obj: GameObject, image: pygame.Surface | None) -> pygame.Rect:
        """Where to draw `image` (or the fallback placeholder box if None)
        for `obj`, in WORLD coordinates (camera-agnostic - callers apply
        `_screen_rect` themselves). Honors an explicit render_offset
        override; otherwise bottom-center-anchors the actual resolved size
        on the footprint - computed here, once the real size is known,
        rather than assumed upfront (a sprite fit to only one dimension
        doesn't know its other dimension until assets.get_fitted has
        measured the real image).
        """
        collision = obj.collision_rect

        if image is not None:
            w, h = image.get_size()
        else:
            max_w, max_h = obj.render_max_size
            w = max_w if max_w is not None else collision.width
            h = max_h if max_h is not None else collision.height

        if obj.render_offset is not None:
            ox, oy = obj.render_offset
            x, y = obj.position[0] + ox, obj.position[1] + oy
        else:
            x = collision.centerx - w / 2
            y = collision.bottom - h

        return pygame.Rect(int(x), int(y), int(w), int(h))

    def _draw_wall(self, wall: pygame.Rect) -> None:
        cam_x, cam_y = self.camera
        viewport = pygame.Rect(cam_x, cam_y, self.world.get_width(), self.world.get_height())
        if not wall.colliderect(viewport):
            return  # off-screen - skip the (potentially large) tiling loop entirely

        tile = assets.get_tile(constants.WALL_TILE_SPRITE)
        if tile is None:
            pygame.draw.rect(self.world, constants.COLOR_WALL, self._screen_rect(wall))
            return

        tw, th = tile.get_size()
        visible = wall.clip(viewport)
        previous_clip = self.world.get_clip()
        self.world.set_clip(self._screen_rect(visible))
        for y in range(wall.top - (wall.top % th), wall.bottom, th):
            for x in range(wall.left - (wall.left % tw), wall.right, tw):
                self.world.blit(tile, self._screen_pos(x, y))
        self.world.set_clip(previous_clip)

    def _draw_actor(self, state: GameState, actor_id: str, dt: float, actions: dict[str, GameAction] | None) -> None:
        actor = state.actors[actor_id]
        animator = self.animators[actor_id]

        # A stunned actor's action was dropped by apply_action (see
        # GameState.apply_action) - don't show a search/inspect/trap/disarm
        # pose for something that didn't actually happen. Opening a door is
        # not a search: no pose for an INTERACT aimed at a door.
        action = actions.get(actor_id) if actions else None
        if action is not None and not actor.is_stunned:
            target = state.level.get_object(action.target_id) if action.target_id else None
            if not (target is not None and target.type == "door"):
                animator.trigger_action(action.type)
        animator.update(dt)

        # Bugfix visual polish (Part D): one canvas, one scale, one foot
        # anchor for every frame (src/character_art.py) - no per-pose crop.
        ko_left = state.ko_seconds_left(actor_id) if hasattr(state, "ko_seconds_left") else None
        anim_state, index = animator.choose(
            actor, ko_left=ko_left, ko_total=constants.FIGHT_KO_SECONDS,
            frames=lambda st: character_art.frame_count(actor_id, st, actor.facing))
        self.last_animation[actor_id] = (anim_state, index)
        image = character_art.scaled_frame(character_art.frame_path(actor_id, anim_state, actor.facing, index))
        if image is not None:
            placed = character_art.placed_rect(actor.x, actor.y)
            self.world.blit(image, self._screen_pos(placed.x, placed.y))
            return

        world_box = actor.sprite_rect  # max bounding box, bottom-anchored at the feet
        box = self._screen_rect(world_box)

        color, outline = constants.ACTOR_COLORS[actor_id]
        pygame.draw.rect(self.world, color, box)
        pygame.draw.rect(self.world, outline, box, width=2)

        fx, fy = FACING_OFFSETS[actor.facing]
        marker_center = (box.centerx + fx * box.width * 0.35, box.centery + fy * box.height * 0.35)
        pygame.draw.circle(self.world, constants.COLOR_FACING_MARKER, marker_center, 4)

    # -- debug overlay (F3) ------------------------------------------------

    def _draw_debug(self, state: GameState) -> None:
        for room in state.level.rooms:
            rect = self._screen_rect(room.rect)
            pygame.draw.rect(self.world, constants.DEBUG_ROOM_COLOR, rect, width=1)
            label_surf = self.debug_font.render(room.id, True, constants.DEBUG_ROOM_COLOR)
            self.world.blit(label_surf, (rect.x + 4, rect.y + 4))

        for wall in state.level.walls:
            pygame.draw.rect(self.world, constants.DEBUG_WALL_COLOR, self._screen_rect(wall), width=1)

        # Mileston_4A2.md section 30 / M4A_3_LIVE_PLAY_DIAGNOSTICS_DOOR_
        # POLISH.md section 18: a door's authoritative opening_rect (blue),
        # its doorway clearance zone (magenta, collision/navigation), and
        # its larger visual exclusion zone (yellow, composition only) - the
        # same rects layout_validation.py actually checked, not a redrawn
        # approximation.
        for obj in state.level.objects:
            if obj.type != "door":
                continue
            pygame.draw.rect(self.world, constants.DEBUG_OPENING_COLOR, self._screen_rect(obj.collision_rect), width=2)
            for exclusion in door_visual_exclusion_rects(obj):
                pygame.draw.rect(self.world, constants.DEBUG_VISUAL_EXCLUSION_COLOR, self._screen_rect(exclusion), width=1)
            for clearance in door_clearance_rects(obj):
                pygame.draw.rect(self.world, constants.DEBUG_CLEARANCE_COLOR, self._screen_rect(clearance), width=1)

        for obj in state.level.objects:
            sprite_path = _door_sprite_path(obj, state.door_states.get(obj.id))
            image = assets.get_fitted(sprite_path, obj.render_max_size)
            render_rect = self._screen_rect(self._placement_rect(obj, image))

            pygame.draw.rect(self.world, constants.DEBUG_COLLISION_COLOR, self._screen_rect(obj.collision_rect), width=1)
            pygame.draw.rect(self.world, constants.DEBUG_RENDER_COLOR, render_rect, width=1)
            anchor = self._screen_pos(*obj.position)
            pygame.draw.circle(self.world, constants.DEBUG_ANCHOR_COLOR, anchor, 3)

            label_bits = [obj.id]
            if obj.type == "door":
                label_bits.append(f"[{state.door_states.get(obj.id)}]")
            entry = state.object_states.get(obj.id, {})
            trap = entry.get("trap")
            if trap:
                label_bits.append(f"TRAP:{trap['type']}/{trap['owner_id']}")
            if entry.get("inspected_by"):
                label_bits.append("insp:" + ",".join(sorted(entry["inspected_by"])))
            if entry.get("detected_by"):
                label_bits.append("det:" + ",".join(f"{a}={t}" for a, t in entry["detected_by"].items()))

            id_surf = self.debug_font.render(" ".join(label_bits), True, constants.DEBUG_TEXT_COLOR)
            screen_rect = self._screen_rect(obj.collision_rect)
            self.world.blit(id_surf, (screen_rect.x, screen_rect.y - 13))

        for actor_id, actor in state.actors.items():
            screen_rect = self._screen_rect(actor.collision_rect)
            pygame.draw.rect(self.world, constants.DEBUG_COLLISION_COLOR, screen_rect, width=1)
            pygame.draw.circle(
                self.world, constants.DEBUG_RADIUS_COLOR, self._screen_pos(actor.x, actor.y), constants.INTERACTION_DISTANCE, width=1
            )
            room_id = state.level.room_id_at(actor.x, actor.y)
            label = f"{actor_id} [{room_id}]" if room_id else actor_id
            if actor.is_stunned:
                label += f" STUNNED {actor.stunned_timer:.1f}s"
            elif actor.is_busy:
                label += f" BUSY {actor.action_cooldown:.1f}s"
            if actor.has_alarm_reveal:
                # M4I section 10: F3 may still show raw states like
                # "REVEALED 4.1s" even though the HUD/status message never
                # names a specific trap type generically like this.
                label += f" REVEALED {actor.alarm_reveal_timer:.1f}s"
            score = compute_score(state, actor_id)
            if state.mission_complete[actor_id]:
                label += " MISSION COMPLETE"
            elif score.failed:
                label += " MISSION FAILED (deadline)"
            else:
                required = state.items.required
                have = sum(1 for item_id in required if actor.inventory.has(item_id))
                label += f" items:{have}/{len(required)}"
            if score.escaped or score.failed:
                label += f" score={score.total:.0f}"
            # M4C: room-based opponent visibility, read straight off
            # GameState (the renderer isn't bound by the Observation
            # boundary a controller is) - shows exactly what this actor's
            # own Observation would report, for debugging that boundary.
            other_id = next(aid for aid in state.actors if aid != actor_id)
            other_room = state.level.room_id_at(*state.actors[other_id].collision_rect.center)
            if room_id is not None and room_id == other_room:
                label += f" sees:{other_id}"
            elif actor.last_seen_opponent:
                seen = actor.last_seen_opponent
                label += f" last_saw:{seen['actor_id']}@{seen['room']}(t={seen['game_time']:.0f}s)"
            id_surf = self.debug_font.render(label, True, constants.DEBUG_TEXT_COLOR)
            self.world.blit(id_surf, (screen_rect.x, screen_rect.y - 13))

    def _draw_bot_diagnostics(self, state: GameState, bot_controller) -> None:
        """M4A.3 (`M4A_3_LIVE_PLAY_DIAGNOSTICS_DOOR_POLISH.md` section 25):
        a compact, screen-space panel making a live-play stall
        diagnosable without needing to read the console - the active
        seed, White's current objective/target, path progress, and its
        mission/stun/escape state, all read-only off the bot controller
        and GameState. `bot_controller` is duck-typed (whatever object
        `Game` passes as White's controller) so this has no import-time
        dependency on `ScriptedBotController` specifically; any attribute
        it doesn't have just falls back to `"?"` rather than erroring.
        """
        white = state.actors[constants.WHITE_SPY_ID]
        target_id = getattr(bot_controller, "_current_target_id", None) or "?"
        path = getattr(bot_controller, "_path", [])
        path_index = getattr(bot_controller, "_path_index", 0)
        objective = "EXIT" if white.inventory.has_all(state.items.required) else "EXPLORE"

        score = compute_score(state, constants.WHITE_SPY_ID)
        lines = [
            f"seed={state.seed}",
            f"time={state.mission_time_remaining:.0f}s",
            f"BOT objective={objective} target={target_id}",
            f"path {path_index}/{len(path)}",
            f"mission_ready={white.inventory.has_all(state.items.required)} stunned={white.is_stunned} busy={white.is_busy}",
            f"escaped={state.mission_complete[constants.WHITE_SPY_ID]} failed={score.failed}",
        ]
        if score.escaped or score.failed:
            lines.append(f"score={score.total:.0f}")

        panel = pygame.Rect(constants.WINDOW_WIDTH - 300, 84, 290, 16 * len(lines) + 8)
        self._draw_hud_backing(panel)
        y = panel.y + 4
        for line in lines:
            surf = self.debug_font.render(line, True, constants.DEBUG_TEXT_COLOR)
            self.screen.blit(surf, (panel.x + 6, y))
            y += 16

    # -- HUD (drawn straight onto self.screen - real screen space, no zoom) --

    def _draw_hud(self, state: GameState, selected_trap_type: str | None, bot_controller=None) -> None:
        self._draw_hud_backing(pygame.Rect(0, 0, 300, 84))

        title_surf = self.title_font.render("SPY vs AI", True, constants.COLOR_HUD_TITLE)
        self.screen.blit(title_surf, (12, 8))
        self._draw_mission_timer(state)

        self._draw_mission_objective()
        self._draw_trap_panel(state, selected_trap_type)
        self._draw_inventory_panel(state)
        self._draw_white_spy_panel(state, bot_controller)

        if state.mission_complete[constants.BLACK_SPY_ID]:
            self._draw_mission_complete_overlay(state)
            return

        if compute_score(state, constants.BLACK_SPY_ID).failed:
            self._draw_mission_failed_overlay(state)
            return

        target_id = state.current_interaction_target.get(constants.BLACK_SPY_ID)
        if target_id is not None:
            obj = state.level.get_object(target_id)
            if obj is not None:
                lines = state.get_prompt_lines(obj, constants.BLACK_SPY_ID)
                self._draw_prompt("  ·  ".join(line.replace("[", "").replace("]", "") for line in lines))

        if state.status_message:
            self._draw_status_message(state.status_message, state.status_message_success)

    def _draw_white_spy_panel(self, state: GameState, bot_controller) -> None:
        if bot_controller is None:
            return

        status = getattr(bot_controller, "status", "UNKNOWN")
        last_action = getattr(bot_controller, "last_action", None)
        last_reason = getattr(bot_controller, "last_reason", "") or ""
        action_name = "WAIT"
        target_name = None
        if last_action is not None:
            if hasattr(last_action, "type") and last_action.type is not None:
                action_name = getattr(last_action.type, "value", str(last_action.type))
            target_name = getattr(last_action, "target_id", None)

        provider = getattr(bot_controller, "provider_name", None)
        model = getattr(bot_controller, "model_name", None)
        lines = [
            "WHITE SPY",
            f"status={status}",
            f"does={action_name}" + (f" target={target_name}" if target_name else ""),
            f"thinks={last_reason[:90]}" + ("..." if len(last_reason) > 90 else ""),
        ]
        if provider and model:
            lines.append(f"model={provider}/{model}")

        panel = pygame.Rect(constants.WINDOW_WIDTH - 330, 84, 320, 18 * len(lines) + 10)
        self._draw_hud_backing(panel)
        y = panel.y + 6
        for line in lines:
            surf = self.debug_font.render(line, True, constants.COLOR_TEXT)
            self.screen.blit(surf, (panel.x + 6, y))
            y += 18

    def _draw_hud_backing(self, rect: pygame.Rect) -> None:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((*constants.COLOR_OVERLAY, 120))
        self.screen.blit(panel, rect.topleft)

    def _draw_mission_timer(self, state: GameState) -> None:
        """M4D: "Display remaining mission time" - the one shared clock,
        right-aligned against the title on the same line so the compact
        HUD panel (Milestone 4A.1) doesn't need to grow. Reads a bright
        warning color once under 30s, same "getting worse" visual language
        the trap-stun/status-message colors already use elsewhere.
        """
        remaining = state.mission_time_remaining
        minutes, seconds = divmod(int(remaining), 60)
        color = constants.COLOR_TEXT if remaining > 30 else constants.COLOR_MESSAGE_FAIL
        surf = self.title_font.render(f"{minutes:01d}:{seconds:02d}", True, color)
        self.screen.blit(surf, (300 - 12 - surf.get_width(), 8))

    def _draw_mission_objective(self) -> None:
        y = 34
        for line in (constants.MISSION_TITLE, constants.MISSION_OBJECTIVE):
            surf = self.label_font.render(line, True, constants.COLOR_TEXT)
            self.screen.blit(surf, (12, y))
            y += surf.get_height() + 1

    def _draw_trap_panel(self, state: GameState, selected_trap_type: str | None) -> None:
        actor = state.actors[constants.BLACK_SPY_ID]
        parts = []
        for i, trap_type in enumerate(constants.TRAP_TYPES, start=1):
            count = actor.trap_inventory.get(trap_type, 0)
            short_label = constants.TRAP_LABELS[trap_type].split()[0]
            marker = "*" if trap_type == selected_trap_type else " "
            parts.append(f"{marker}{i}:{short_label[:4]} x{count}")

        surf = self.label_font.render(" ".join(parts), True, constants.COLOR_TEXT)
        self.screen.blit(surf, (12, 64))

    def _draw_inventory_panel(self, state: GameState) -> None:
        required = state.items.required
        slot = constants.INVENTORY_SLOT_SIZE
        gap = constants.INVENTORY_SLOT_GAP
        margin = constants.INVENTORY_SLOT_MARGIN
        inventory = state.actors[constants.BLACK_SPY_ID].inventory

        total_w = len(required) * slot + (len(required) - 1) * gap
        x = constants.WINDOW_WIDTH - margin - total_w
        y = margin

        for item_id in required:
            item = state.items.items[item_id]
            rect = pygame.Rect(x, y, slot, slot)
            collected = inventory.has(item_id)

            pygame.draw.rect(
                self.screen,
                constants.COLOR_SLOT_FILLED if collected else constants.COLOR_SLOT_EMPTY,
                rect,
                border_radius=4,
            )
            pygame.draw.rect(self.screen, constants.COLOR_SLOT_BORDER, rect, width=2, border_radius=4)

            if collected:
                icon = assets.get_fitted(item.sprite, (slot - 8, slot - 8))
                if icon is not None:
                    self.screen.blit(icon, icon.get_rect(center=rect.center))
                else:
                    letter_surf = self.font.render(item.name[0], True, constants.COLOR_TEXT)
                    self.screen.blit(letter_surf, letter_surf.get_rect(center=rect.center))

            x += slot + gap

    def _draw_prompt(self, text: str) -> None:
        surf = self.font.render(text, True, constants.COLOR_TEXT)
        pad_x, pad_y = 8, 4
        box = pygame.Surface((surf.get_width() + pad_x * 2, surf.get_height() + pad_y * 2))
        box.fill(constants.COLOR_PROMPT_BG)
        box.blit(surf, (pad_x, pad_y))

        x = constants.WINDOW_WIDTH // 2 - box.get_width() // 2
        y = constants.WINDOW_HEIGHT - 56
        self.screen.blit(box, (x, y))

    def _draw_status_message(self, text: str, success: bool) -> None:
        color = constants.COLOR_MESSAGE_OK if success else constants.COLOR_MESSAGE_FAIL
        lines = text.split("\n")
        surfaces = [self.font.render(line, True, color) for line in lines]
        total_h = sum(s.get_height() for s in surfaces) + 4 * (len(surfaces) - 1)
        max_w = max(s.get_width() for s in surfaces)

        top = constants.WINDOW_HEIGHT - 100 - total_h
        backing = pygame.Rect(0, 0, max_w + 24, total_h + 12)
        backing.center = (constants.WINDOW_WIDTH // 2, top + total_h // 2)
        self._draw_hud_backing(backing)

        y = top
        for surf in surfaces:
            x = constants.WINDOW_WIDTH // 2 - surf.get_width() // 2
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 4

    def _draw_mission_complete_overlay(self, state: GameState) -> None:
        score = compute_score(state, constants.BLACK_SPY_ID)
        lines = ["MISSION COMPLETE", "You escaped the embassy.", ""] + self._score_lines(score)
        self._draw_outcome_overlay(lines, constants.COLOR_MISSION_COMPLETE)

    def _draw_mission_failed_overlay(self, state: GameState) -> None:
        """M4E: the deadline consequence M4D deliberately left unbuilt -
        `_attempt_exit` now refuses to succeed at all once
        `mission_time_remaining` hits zero, so an actor who never made it
        out sees this instead of merely watching the HUD clock read 0:00.
        """
        score = compute_score(state, constants.BLACK_SPY_ID)
        lines = ["MISSION FAILED", "The extraction window has closed.", ""] + self._score_lines(score)
        self._draw_outcome_overlay(lines, constants.COLOR_MESSAGE_FAIL)

    def _score_lines(self, score: ScoreBreakdown) -> list[str]:
        """Itemized score breakdown (Milestone_4.md's scoring table) -
        only the lines that actually applied, so a failed run doesn't show
        a wall of "+0" bonuses that never fired."""
        lines = []
        if score.mission_success:
            lines.append(f"+{constants.SCORE_COMPLETE_OWN_MISSION:.0f}  mission complete")
        if score.escaped:
            lines.append(f"+{constants.SCORE_ESCAPE_BEFORE_DEADLINE:.0f}  escaped before deadline")
            if score.escaped_first:
                lines.append(f"+{constants.SCORE_ESCAPE_FIRST:.0f}  escaped first")
            if score.other_escaped:
                lines.append(f"+{constants.SCORE_OTHER_ALSO_ESCAPES:.0f}  other spy also escaped")
            lines.append(f"+{score.remaining_time_bonus:.1f}  remaining time bonus")
        if score.failed:
            lines.append(f"{constants.SCORE_FAILURE_TO_ESCAPE:.0f}  failed to escape")
        lines.append(f"TOTAL: {score.total:.0f}")
        return lines

    def _draw_outcome_overlay(self, lines: list[str], title_color) -> None:
        overlay = pygame.Surface((constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((*constants.COLOR_OVERLAY, constants.OVERLAY_ALPHA))
        self.screen.blit(overlay, (0, 0))

        surfaces = []
        for i, line in enumerate(lines):
            if not line:
                surfaces.append(None)
                continue
            font = self.title_font if i < 2 else self.label_font
            color = title_color if i < 2 else constants.COLOR_TEXT
            surfaces.append(font.render(line, True, color))

        total_h = sum((s.get_height() if s else 8) + 4 for s in surfaces)
        y = constants.WINDOW_HEIGHT // 2 - total_h // 2
        for surf in surfaces:
            if surf is None:
                y += 12
                continue
            x = constants.WINDOW_WIDTH // 2 - surf.get_width() // 2
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 4

    def _draw_centered_label(self, label: str, rect: pygame.Rect) -> None:
        if rect.width < 30 or rect.height < 20:
            return  # too small to caption legibly
        surf = self.label_font.render(label, True, constants.OBJECT_LABEL_COLOR)
        if surf.get_width() > rect.width - 4:
            return
        pos = (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2)
        self.world.blit(surf, pos)
