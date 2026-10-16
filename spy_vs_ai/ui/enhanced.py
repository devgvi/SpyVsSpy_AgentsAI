"""Enhanced gameplay visuals (FINAL-2 gameplay polish, spec 3, 25, 26): room dressing, door-trap markers, trigger
effects and cues. A mixin for the runtime renderer; active only when the enhanced gameplay rules run.

Presentation only: it reads the level, the engine event log and the trap resolver's state, and changes nothing.

* Props (spy_vs_ai.runtime.dressing) are drawn like furniture: rugs on the floor, everything else y-sorted.
* Door-trap markers (spec 25): a trap is marked only for a spy that knows it - its owner, or a spy that detected it.
  Split view: each pane shows what its own spy knows; single view: the Black Spy's knowledge (the camera's spy);
  overview (the spectator's map): what either spy knows.
* Trigger feedback (spec 26): a short text cue naming the trap, the victim and the effect (BOOM / ZAP / GAS /
  SNARED / ALARM - never colour alone) and a visual: explosion at the doorway, arcs on the victim, a lingering gas
  cloud on the doorway and a small cloud over a poisoned spy, a snare on a rooted spy, the alarm flash.
  Lingering effects (gas, snare) are drawn from the resolver's own timers, so they last exactly as long as the
  gameplay effect, live and in a replay.
"""
from __future__ import annotations

import time
from typing import Any

import pygame

from src import assets, constants
from src.effects import EffectAnimation
from spy_vs_ai.runtime import dressing, traps as tr
from workshop.live.config import ACTOR_LABELS
from workshop.live.overlay import CUE_SECONDS, SPY_TEXT_COLORS, THEME

TRAP_ICONS = {"explosive": "assets/traps/trap_bomb.png", "electric": "assets/traps/trap_electric.png",
              "snare": "assets/traps/trap_snare.png", "alarm": "assets/traps/trap_alarm.png",
              "gas": "assets/traps/trap_gas.png"}
GAS_FRAMES = tuple(f"assets/effects/gas/gas_{i:02d}.png" for i in range(1, 9))
GAS_CLOUD_PX = int(tr.GAS_RADIUS * 2)
POISON_PX = 46
DETECTED_COLOR = THEME["warning_text"]
TRIGGER_CUE_SECONDS = 2.6
EXPLOSION_SECONDS = 1.2
ALARM_SECONDS = 1.6


def _name(actor_id: str | None) -> str:
    return ACTOR_LABELS.get(actor_id, str(actor_id)).split()[0].upper()


class EnhancedLayer:
    """Mixin (before LiveRenderer in the bases): overrides a few world-layer hooks when `self.enhanced` is set."""

    enhanced = False

    def _enhanced_init(self) -> None:
        self._pane_actor: str | None = constants.BLACK_SPY_ID
        self._props: list[dressing.Prop] | None = None
        self._props_level: Any = None
        self._enh_fx_seen = 0
        self._enh_cue_seen = 0
        self._gas_frames: tuple[pygame.Surface, ...] | None = None
        self._poison_frames: tuple[pygame.Surface, ...] | None = None

    # -- helpers -------------------------------------------------------------------------------------------
    def props(self, state) -> list[dressing.Prop]:
        if self._props is None or self._props_level is not state.level:
            self._props = dressing.props_for(state.level)
            self._props_level = state.level
        return self._props

    @staticmethod
    def resolver(state):
        return getattr(state, "enhanced_traps", None)

    def _frames(self, size: int) -> tuple[pygame.Surface, ...]:
        return tuple(f for f in (assets.get_fitted(p, (size, size)) for p in GAS_FRAMES) if f is not None)

    def gas_frames(self) -> tuple[pygame.Surface, ...]:
        if self._gas_frames is None:
            self._gas_frames = self._frames(GAS_CLOUD_PX)
        return self._gas_frames

    def poison_frames(self) -> tuple[pygame.Surface, ...]:
        if self._poison_frames is None:
            self._poison_frames = self._frames(POISON_PX)
        return self._poison_frames

    # -- camera bookkeeping: whose knowledge a view shows ---------------------------------------------------
    def _camera_for(self, state, actor_id: str, view_w: float, view_h: float):
        self._pane_actor = actor_id
        return super()._camera_for(state, actor_id, view_w, view_h)

    def _render_overview(self, state, dt, actions, game):
        self._pane_actor = None  # the spectator's map: what either spy knows
        try:
            return super()._render_overview(state, dt, actions, game)
        finally:
            self._pane_actor = constants.BLACK_SPY_ID

    # -- props ---------------------------------------------------------------------------------------------
    def _draw_decorations(self, state) -> None:
        super()._draw_decorations(state)
        if not self.enhanced:
            return
        for prop in self.props(state):
            if prop.layer == "floor":
                self._draw_prop(prop)

    def _draw_prop(self, prop: dressing.Prop) -> None:
        rect = prop.rect
        if prop.layer == "floor":
            image = assets.get_fitted(prop.sprite, (rect.width, rect.height))
            if image is not None:
                self.world.blit(image, self._screen_rect(image.get_rect(center=rect.center)).topleft)
            return
        image = assets.get_fitted(prop.sprite, (None, prop.height))
        if image is None:
            return  # missing art: nothing (a prop is never gameplay)
        place = image.get_rect(midbottom=(rect.centerx, rect.bottom))
        self.world.blit(image, self._screen_rect(place).topleft)

    def _draw_y_sorted(self, state, dt, actions) -> None:
        """src.renderer.Renderer._draw_y_sorted (same order and keys) plus the props, then the door-trap markers."""
        if not self.enhanced:
            return super()._draw_y_sorted(state, dt, actions)
        entries = []
        for obj in state.level.objects:
            if obj.type == "rug":
                continue
            is_open_door = obj.type == "door" and state.door_states.get(obj.id) == "open"
            key = float("-inf") if is_open_door else obj.collision_rect.bottom
            entries.append((key, lambda o=obj: self._draw_object(state, o)))
        for wall in state.level.walls:
            entries.append((wall.bottom, lambda w=wall: self._draw_wall(w)))
        for actor_id in state.actors:
            entries.append((state.actors[actor_id].collision_rect.bottom,
                            lambda a=actor_id: self._draw_actor(state, a, dt, actions)))
        for prop in self.props(state):
            if prop.layer != "floor":
                entries.append((prop.rect.bottom, lambda p=prop: self._draw_prop(p)))
        for _, draw in sorted(entries, key=lambda e: e[0]):
            draw()
        self._draw_door_trap_markers(state)

    # -- door-trap markers (spec 25) -----------------------------------------------------------------------
    def known_door_traps(self, state, viewer: str | None) -> list[tuple[tr.DoorTrap, bool]]:
        """(trap, owned_by_a_viewer) for the armed door traps the viewer knows (None: either spy)."""
        resolver = self.resolver(state)
        if resolver is None:
            return []
        viewers = [viewer] if viewer is not None else list(state.actors)
        out = []
        for trap in resolver.armed():
            if trap.owner_id in viewers:
                out.append((trap, True))
            elif any(v in trap.detected_by for v in viewers):
                out.append((trap, False))
        return out

    def _draw_door_trap_markers(self, state) -> None:
        for trap, owned in self.known_door_traps(state, self._pane_actor):
            door = state.level.get_object(trap.door_id)
            if door is None:
                continue
            rect = self._screen_rect(door.collision_rect).inflate(6, 6)
            color = SPY_TEXT_COLORS.get(trap.owner_id, THEME["primary_text"]) if owned else DETECTED_COLOR
            # a dashed outline: subtle, readable without colour (the icon names the trap)
            for i, (a, b) in enumerate(((rect.topleft, rect.topright), (rect.topright, rect.bottomright),
                                        (rect.bottomright, rect.bottomleft), (rect.bottomleft, rect.topleft))):
                length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
                for s in range(0, length, 10):
                    t0, t1 = s / max(1, length), min(1.0, (s + 5) / max(1, length))
                    p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                    p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                    pygame.draw.line(self.world, color, p0, p1, 2)
            icon = assets.get_fitted(TRAP_ICONS.get(trap.variant), (22, 22))
            if icon is not None:
                spot = icon.get_rect(center=(rect.right - 2, rect.top + 2))
                pygame.draw.circle(self.world, THEME["panel_background"], spot.center, 13)
                pygame.draw.circle(self.world, color, spot.center, 13, width=2)
                self.world.blit(icon, spot.topleft)
            if not owned:
                mark = self.debug_font.render("!", True, color)
                self.world.blit(mark, mark.get_rect(center=(rect.left + 2, rect.top + 2)))

    # -- trigger effects (spec 26) ---------------------------------------------------------------------------
    def _update_effects(self, state, dt: float) -> None:
        super()._update_effects(state, dt)
        if not self.enhanced:
            return
        log = state.event_log
        if len(log) < self._enh_fx_seen:
            self._enh_fx_seen = 0
        for event in log[self._enh_fx_seen:]:
            if event.get("type") == "ENHANCED_TRAP_TRIGGERED":
                self._spawn_door_effect(state, event)
        self._enh_fx_seen = len(log)

    def _spawn_door_effect(self, state, event: dict) -> None:
        variant, victim = event.get("variant"), event.get("victim")
        door = state.level.get_object(event.get("target_id"))
        anchor = door.collision_rect.center if door is not None else tuple(event.get("position") or (0, 0))
        if victim in self.animators:
            self.animators[victim].start("hit", 0.6)
        spec = {"explosive": ("bomb", EXPLOSION_SECONDS, None), "electric": ("electric", tr.ELECTRIC_STUN, victim),
                "alarm": ("alarm", ALARM_SECONDS, victim)}.get(variant)
        if spec is None:
            return  # gas and snare linger: drawn from the resolver's own timers (_draw_effects)
        kind, seconds, follows = spec
        frames = self._effect_frames.get(kind)
        if frames:
            self._active_effects.append(EffectAnimation(effect_type=f"door_{variant}", world_position=anchor,
                                                        frames=frames, frame_duration=seconds / len(frames),
                                                        follows_actor_id=follows))

    def _draw_effects(self, state) -> None:
        if self.enhanced:
            self._draw_lingering(state)
        super()._draw_effects(state)

    def _draw_lingering(self, state) -> None:
        resolver = self.resolver(state)
        if resolver is None:
            return
        for effect in resolver.active_effects():
            total = tr.GAS_SECONDS if effect.kind in ("gas_cloud", "poison") else tr.SNARE_SECONDS
            progress = 1.0 - effect.remaining / total
            if effect.kind == "gas_cloud" and effect.position is not None:
                frames = self.gas_frames()
                position = effect.position
            elif effect.kind == "poison":
                frames = self.poison_frames()
                actor = state.actors.get(effect.actor_id)
                position = (actor.x, actor.y - 70) if actor is not None else None
            elif effect.kind == "root":
                frames = self._effect_frames.get("snare")
                actor = state.actors.get(effect.actor_id)
                position = (actor.x, actor.y - 20) if actor is not None else None
            else:
                continue
            if not frames or position is None:
                continue
            frame = frames[min(len(frames) - 1, int(progress * len(frames)))]
            self.world.blit(frame, frame.get_rect(center=self._screen_pos(*position)).topleft)

    # -- cues (spec 26) ----------------------------------------------------------------------------------------
    def _collect_cues(self, state) -> None:
        super()._collect_cues(state)
        if not self.enhanced:
            return
        now = time.monotonic()
        log = state.event_log
        if len(log) < self._enh_cue_seen:
            self._enh_cue_seen = 0
        for event in log[self._enh_cue_seen:]:
            cue = self.trap_cue(state, event)
            if cue is not None:
                seconds, text, pos, color = cue
                self._cues.append((now + seconds, text, pos, color))
        self._enh_cue_seen = len(log)

    @staticmethod
    def trap_cue(state, event: dict):
        """(seconds, text, world position, colour) for an enhanced-trap event, or None."""
        kind = event.get("type", "")
        if not kind.startswith("ENHANCED_TRAP_"):
            return None
        door = state.level.get_object(event.get("target_id")) if event.get("target_type") == "doorway" else None
        at = door.collision_rect.midtop if door is not None else None
        variant = event.get("variant")
        if kind == "ENHANCED_TRAP_TRIGGERED":
            victim = state.actors.get(event.get("victim"))
            pos = (victim.x, victim.y - 96) if victim is not None else at
            text = f"{tr.CUE.get(variant, 'TRAP')}! {_name(event.get('victim'))} - {tr.EFFECT_LABEL.get(variant, '')}"
            return TRIGGER_CUE_SECONDS, text, pos, THEME["critical_text"]
        if at is None:
            return None
        owner_color = SPY_TEXT_COLORS.get(event.get("owner"), THEME["primary_text"])
        actor_color = SPY_TEXT_COLORS.get(event.get("actor_id"), THEME["primary_text"])
        if kind == "ENHANCED_TRAP_PLACED":
            return CUE_SECONDS, f"{variant} door trap set", at, owner_color
        if kind == "ENHANCED_TRAP_DETECTED":
            return CUE_SECONDS, "door trap spotted", at, actor_color
        if kind == "ENHANCED_TRAP_DISARMED":
            return CUE_SECONDS, f"{variant} door trap disarmed", at, actor_color
        if kind == "ENHANCED_TRAP_DISARM_FAILED":
            return CUE_SECONDS, "door disarm failed!", at, actor_color
        if kind == "ENHANCED_TRAP_REMOVED":
            return CUE_SECONDS, "door trap removed", at, actor_color
        return None


__all__ = ["EnhancedLayer", "TRAP_ICONS", "GAS_FRAMES"]
