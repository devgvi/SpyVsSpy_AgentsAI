"""The runtime renderer: play view and research view (FINAL-2 spec 16-25, 38-40).

`RuntimeRenderer` is the live game's renderer (workshop.live.overlay.LiveRenderer: the production world view, split
cameras, the sidebar) with:

- per spy, in the sidebar: policy, surrogate identity, provider / model, the scientific label, the compatibility
  label, THINKING / current action, and (Y) the telemetry overlay - provider calls, escalations, tokens, average
  latency, fallbacks, the last normalized error and the mission progress;
- a banner with the execution mode (PLAYABLE MODE - execution semantics differ from the frozen evaluation protocol);
- research view (M): the decision inspector (I) and the decision history (H) over the game view - only fields the
  telemetry carries, never hidden reasoning;
- the match summary screen at the end (winner / outcome per spy and in total, the replay path);
- a pause note: a provider call already in flight is never cancelled;
- with the enhanced gameplay rules: room dressing, door-trap markers, trigger cues and effects (spy_vs_ai.ui.enhanced).

It only reads: the decision log is fed from the match's own telemetry events and the observation the agent was
given. Nothing here can change a decision.
"""
from __future__ import annotations

import textwrap
from typing import Any

import pygame

from src import constants
from spy_vs_ai.runtime import config as rc
from spy_vs_ai.runtime import policy_registry as pr
from spy_vs_ai.runtime.decisions import ACTORS, DECISION_EVENTS, LABEL, history_line, inspector_lines, summary_lines
from spy_vs_ai.providers.base import FALLBACK_USED, user_error
from spy_vs_ai.ui.enhanced import EnhancedLayer
from workshop.live.overlay import (COLOR_COMMIT, COLOR_DECIDING, COLOR_DIM, COLOR_WARN, LIFECYCLE_LABELS, LINE_HEIGHT,
                                   THEME, LiveRenderer)

SIDE_OF = {"black_spy": "black", "white_spy": "white"}
COMPAT_COLORS = {pr.EVALUATED: COLOR_COMMIT, pr.COMPATIBLE: COLOR_WARN, pr.EXPERIMENTAL: COLOR_WARN,
                 pr.FAKE: THEME["muted_text"], pr.BASELINE: THEME["secondary_text"]}
KEYS_HELP = "V show/hide panels  M play/research  I inspector  H history  Y AI telemetry  Tab spy  Up/Down pick"
PAUSE_NOTE = "PAUSED - a provider call already in flight is not cancelled; no new decision is requested"
BASE_NAMES = {"b1": "B1 (table-v1)", "b2_v1": "B2-v1 (logistic-v1)", "b2_v2": "B2-v2 (candidate C1)"}
CHAR_PX = 7  # debug_font (Courier New 11 bold) advance, about 7 px


def wrap(text: str, chars: int) -> list[str]:
    return textwrap.wrap(str(text), max(10, chars)) or [""]


class RuntimeRenderer(EnhancedLayer, LiveRenderer):
    """LiveRenderer plus the runtime's play / research views (see the module docstring)."""

    def __init__(self, screen: pygame.Surface, runtime_config: rc.GameRuntimeConfig, *,
                 replay: dict[str, Any] | None = None) -> None:
        super().__init__(screen)
        self._enhanced_init()
        self.runtime_config = runtime_config
        self.runtime_view = runtime_config.view
        research = runtime_config.view == "research"
        self.show_inspector = research
        self.show_history = research
        # spec 24: optional in play mode, visible by default in research mode
        self.show_runtime_telemetry = runtime_config.show_telemetry or research
        self.show_telemetry = runtime_config.show_telemetry  # the live game's own agent panels (T)
        self.show_detail = research  # V shows / hides the panels over the game area and the sidebar detail together
        self.decision_log = None
        self.runner = None
        self.summary: dict[str, Any] | None = None
        self.replay = replay  # replay mode: {"label", "frame", "frames", "path", "verdict"}
        self.inspect_actor: str | None = None  # None = both spies
        self.selected = 0  # 0 = the latest decision, n = n decisions back
        ex = rc.execution(runtime_config)
        self.mode_label = ex["label"]
        self.mode_notes = list(ex["notes"])
        self.evaluation_mode = runtime_config.mode == "evaluation"
        self.enhanced = rc.enhanced(runtime_config)
        self.spy_info: dict[str, dict[str, Any]] = {}
        for actor in ACTORS:
            a = runtime_config.agent(SIDE_OF[actor])
            d = a.descriptor
            comp = pr.compatibility(a.policy, provider=a.provider, model=a.model, identity=a.surrogate_identity,
                                    level=runtime_config.level, mode=runtime_config.mode,
                                    time_limit=runtime_config.time_limit, enhanced=self.enhanced)
            self.spy_info[actor] = {
                "policy": a.policy, "label": d.label, "kind": d.kind, "status": d.scientific_label,
                "summary": list(d.summary), "compat": comp.label, "compat_reasons": list(comp.reasons),
                "provider": f"{a.provider} / {a.model}" if a.calls_provider else None, "personality": a.personality,
                "key": a.policy_key if d.kind in ("surrogate", "hybrid") else None,
                "base": BASE_NAMES.get(d.base or "")}

    # -- the decision log (read-only feed from the match telemetry) ---------------------------------------
    def feed(self) -> None:
        log, game = self.decision_log, self.game
        if log is None or game is None:
            return
        events = getattr(getattr(game, "telemetry", None), "events", None)
        if not events or log.seen >= len(events):
            return
        if not getattr(game, "replay_mode", False) and hasattr(game, "observation_payload"):
            actors = {e.get("actor_id") for e in events[log.seen:] if e.get("event_type") in DECISION_EVENTS}
            for actor in actors & set(ACTORS):
                log.note_observation(actor, game.observation_payload(actor))
        log.update(events)

    def records(self) -> list[dict[str, Any]]:
        if self.decision_log is None:
            return []
        recs = list(self.decision_log.records)
        return [r for r in recs if r["spy"] == self.inspect_actor] if self.inspect_actor else recs

    def selected_record(self) -> dict[str, Any] | None:
        recs = self.records()
        if not recs:
            return None
        return recs[-1 - min(self.selected, len(recs) - 1)]

    def last_record(self, actor: str) -> dict[str, Any] | None:
        if self.decision_log is None:
            return None
        for rec in reversed(self.decision_log.records):
            if rec["spy"] == actor:
                return rec
        return None

    # -- keys (forwarded by the match runner / the replay loop) ------------------------------------------
    def panels_visible(self) -> bool:
        return self.runtime_view == "research" and (self.show_inspector or self.show_history)

    def toggle_panels(self) -> None:
        """V: the panels drawn over the game area (inspector + history) on / off, with the sidebar detail."""
        show = not self.panels_visible()
        if show:
            self.runtime_view = "research"
        self.show_inspector = self.show_history = self.show_detail = show

    def handle_key(self, key: int) -> None:
        if key == pygame.K_v:
            self.toggle_panels()
        elif key == pygame.K_m:
            self.runtime_view = "play" if self.runtime_view == "research" else "research"
            research = self.runtime_view == "research"
            self.show_inspector = self.show_history = self.show_detail = research
            self.show_runtime_telemetry = research or self.runtime_config.show_telemetry
        elif key == pygame.K_i:
            self.show_inspector = not self.show_inspector
        elif key == pygame.K_h:
            self.show_history = not self.show_history
        elif key == pygame.K_y:
            self.show_runtime_telemetry = not self.show_runtime_telemetry
        elif key == pygame.K_TAB:
            order = [None, *ACTORS]
            self.inspect_actor = order[(order.index(self.inspect_actor) + 1) % len(order)]
            self.selected = 0
        elif key == pygame.K_UP:
            self.selected = min(self.selected + 1, max(0, len(self.records()) - 1))
        elif key == pygame.K_DOWN:
            self.selected = max(0, self.selected - 1)
        elif key == pygame.K_HOME:
            self.selected = max(0, len(self.records()) - 1)
        elif key == pygame.K_END:
            self.selected = 0

    def ui_state(self) -> dict[str, Any]:
        rec = self.selected_record()
        return {"view": self.runtime_view, "panels": self.panels_visible(), "inspector": self.show_inspector,
                "history": self.show_history, "detail": self.show_detail,
                "ai_telemetry": self.show_runtime_telemetry, "inspect": self.inspect_actor or "both",
                "selected": self.selected, "selected_decision": rec["seq"] if rec else None}

    # -- sidebar -------------------------------------------------------------------------------------------
    def spy_lines(self, actor: str, chars: int) -> list[tuple[str, tuple[int, int, int]]]:
        """Play view: policy, provider / model, labels and the current action; Y adds the telemetry overlay."""
        info = self.spy_info[actor]
        game = self.game
        state = game.state
        snap = None if self.replay else self._snapshot(actor)
        lines: list[tuple[str, tuple[int, int, int]]] = []
        title = info["label"] + (f"  [{info['key']}]" if info["key"] else "")
        lines += [(t, THEME["primary_text"]) for t in wrap(title, chars)]
        if info["provider"]:
            lines += [(t, THEME["secondary_text"]) for t in wrap(info["provider"], chars)]
        lines += [(t, THEME["secondary_text"]) for t in wrap(info["status"], chars)]
        if info["policy"] == "hybrid_v2":
            lines.append((info["summary"][0], THEME["muted_text"]))
        lines += [(t, COMPAT_COLORS.get(info["compat"], COLOR_WARN)) for t in wrap(info["compat"], chars)]
        # what it is doing now
        lifecycle = state.lifecycle(actor) if hasattr(state, "lifecycle") else "ACTIVE"
        rec = self.last_record(actor)
        if self.replay:
            word = LIFECYCLE_LABELS.get(lifecycle, lifecycle)
            color = THEME["secondary_text"]
        else:
            word, color = self._phase_word(state, actor, snap)
            in_flight = actor in (game.in_flight() if hasattr(game, "in_flight") else [])
            if in_flight and lifecycle not in constants.TERMINAL_LIFECYCLES:
                word, color = "THINKING... (provider call in flight)", COLOR_DECIDING
        action = (snap.action if snap is not None and snap.action else None) or (rec or {}).get("executed") or "-"
        lines += [(t, color) for t in wrap(f"{word}  {action}", chars)]
        if rec and rec.get("fallback"):
            lines.append((FALLBACK_USED, COLOR_WARN))
        if self.show_runtime_telemetry and self.decision_log is not None:
            t = self.decision_log.telemetry()[actor]
            lat = f"{t['avg_latency_ms'] / 1000:.2f} s" if t.get("avg_latency_ms") else "-"
            lines.append((f"decisions {t['decisions']}  provider calls {t['provider_calls']}  escalations "
                          f"{t['escalations']}", THEME["primary_text"]))
            lines.append((f"tokens {t['input_tokens']} in / {t['output_tokens']} out  avg latency {lat}",
                          THEME["secondary_text"]))
            lines.append((f"fallbacks {t['fallbacks']}  repairs {t['repairs']}  provider errors "
                          f"{t['provider_call_failures']}", COLOR_WARN if t["fallbacks"] or t["provider_call_failures"]
                          else THEME["secondary_text"]))
            last_error = ((rec or {}).get("llm") or {}).get("error")
            if not last_error and t["errors"]:
                last_error = user_error(next(reversed(t["errors"])))
            if last_error:
                lines.append((last_error, COLOR_WARN))
            required = state.items.required
            have = sum(1 for i in required if state.actors[actor].inventory.has(i))
            lines.append((f"mission: items {have}/{len(required)}  {LIFECYCLE_LABELS.get(lifecycle, lifecycle)}",
                          COLOR_COMMIT if lifecycle in ("ESCAPED", "OBJECTIVES_COMPLETE") else THEME["primary_text"]))
        return lines

    def _agent_panel_lines(self, actor: str) -> list[tuple[str, tuple[int, int, int]]]:
        """The live game's own agent panel (T), without its first line (shown above) and with the surrogate base
        named by the registry (never a different version's name)."""
        if self.view is None or self.replay:
            return []
        try:
            lines = self.view.actor_lines(self.view.actor(self.game, actor), self.show_detail)[1:]
        except Exception:  # noqa: BLE001 - the overlay never breaks the frame
            return []
        base = self.spy_info[actor]["base"]
        out = []
        for text, color in lines:
            if text.startswith("base: ") and base:
                text = f"base: {base}"
            elif text.startswith("emulator: ") and base:
                text = f"emulator: {base}"
            out.append((text, color))
        return out

    def draw_overlay(self) -> None:
        self.feed()
        bar = self.sidebar_rect()
        self.screen.fill(THEME["panel_background"], bar)
        pygame.draw.line(self.screen, THEME["border"], bar.topleft, bar.bottomleft, 2)
        if self.game is None:
            return
        state = self.game.state
        if self.view is not None and not self.replay:
            self.view.update(self.game)
        chars = max(20, (bar.width - 20) // CHAR_PX)
        y = bar.y + self._draw_match_timer(bar, state)
        status = self._status_lines(bar.width) if not self.replay else self._replay_status(chars)
        status += [(t, COLOR_DIM) for t in wrap(KEYS_HELP, chars)]
        bottom = bar.bottom - LINE_HEIGHT * len(status) - 8
        section_h = (bottom - y) // 2
        for actor in ACTORS:
            section = pygame.Rect(bar.x + 6, y, bar.width - 12, section_h - 6)
            pygame.draw.rect(self.screen, THEME["border"], section, width=1)
            used = self._draw_spy_header(state, actor, section)
            self.header_rects[actor] = pygame.Rect(section.x, section.y, section.width, used)
            panel = pygame.Rect(section.x, section.y + used, section.width, section.height - used)
            self.panel_rects[actor] = panel
            lines = self.spy_lines(actor, chars - 1)
            if self.show_telemetry:
                lines += self._agent_panel_lines(actor)
            ly = panel.y + 2
            for text, color in lines:
                if ly + LINE_HEIGHT > panel.bottom:
                    break  # what does not fit is cut, never drawn over the next spy
                self.screen.blit(self.debug_font.render(text, True, color), (panel.x + 6, ly))
                ly += LINE_HEIGHT
            y += section_h
        ly = bottom + 4
        for text, color in status:
            self.screen.blit(self.debug_font.render(text, True, color), (bar.x + 8, ly))
            ly += LINE_HEIGHT
        self._draw_game_area_overlays()

    def _replay_status(self, chars: int) -> list[tuple[str, tuple[int, int, int]]]:
        r = self.replay or {}
        text = (f"REPLAY frame {r.get('frame', 0)}/{r.get('frames', 0)}  speed x{self.speed:g}"
                + ("  PAUSED" if self.paused else "") + "   Space pause  +/- speed  Esc quit")
        return [(t, COLOR_DECIDING) for t in wrap(text, chars)]

    # -- game-area overlays ----------------------------------------------------------------------------
    def _text_box(self, rect: pygame.Rect, lines: list[tuple[str, tuple[int, int, int]]], *, title: str | None = None,
                  highlight: int | None = None) -> None:
        self._draw_panel_backing(rect)
        pygame.draw.rect(self.screen, THEME["border"], rect, width=1)
        y = rect.y + 6
        if title:
            surf = self.label_font.render(title, True, THEME["primary_text"])
            self.screen.blit(surf, (rect.x + 8, y))
            y += surf.get_height() + 4
        for i, (text, color) in enumerate(lines):
            if y + LINE_HEIGHT > rect.bottom - 4:
                break
            if highlight is not None and i == highlight:
                pygame.draw.rect(self.screen, THEME["border"], pygame.Rect(rect.x + 3, y - 1, rect.width - 6, LINE_HEIGHT))
            self.screen.blit(self.debug_font.render(text, True, color), (rect.x + 8, y),
                             area=pygame.Rect(0, 0, rect.width - 16, LINE_HEIGHT))
            y += LINE_HEIGHT

    def _draw_game_area_overlays(self) -> None:
        game = self.game_rect()
        # the execution-mode banner (spec 20), always visible
        banner = [self.mode_label]
        if self.enhanced:
            banner.append(rc.ENHANCED_LABEL)  # spec 2, 36: not part of Stages 3-5
        if self.replay:
            banner.insert(0, f"REPLAY - {self.replay.get('label', 'recorded match')} - 0 provider calls")
        y = 4
        for i, text in enumerate(banner):
            color = COLOR_DECIDING if (self.replay and i == 0) else (COLOR_COMMIT if self.evaluation_mode else COLOR_WARN)
            surf = self.debug_font.render(text, True, color)
            box = surf.get_rect(midtop=(game.centerx, y)).inflate(12, 4)
            self._draw_panel_backing(box)
            self.screen.blit(surf, (box.x + 6, box.y + 2))
            y = box.bottom + 2
        if self.paused and not self.match_over and not self.replay:
            surf = self.label_font.render(PAUSE_NOTE, True, COLOR_DECIDING)
            box = surf.get_rect(midtop=(game.centerx, y + 4)).inflate(16, 8)
            self._draw_panel_backing(box)
            self.screen.blit(surf, (box.x + 8, box.y + 4))
        if self.runtime_view != "research" or self.match_result is not None:
            return
        height = min(330, game.height // 2)
        top = game.bottom - height - 8
        chars_l = (game.width * 3 // 5 - 30) // CHAR_PX
        if self.show_inspector:
            rect = pygame.Rect(game.x + 8, top, game.width * 3 // 5 - 12, height)
            rec = self.selected_record()
            lines = []
            for i, text in enumerate(inspector_lines(rec)):
                for part in wrap(text, chars_l):
                    color = THEME["primary_text"] if i == 0 else THEME["secondary_text"]
                    if FALLBACK_USED in part or "failed" in part or "error" in part.lower():
                        color = COLOR_WARN
                    lines.append((part, color))
            which = "both spies" if self.inspect_actor is None else LABEL[self.inspect_actor]
            where = "latest" if self.selected == 0 else f"{self.selected} back"
            self._text_box(rect, lines, title=f"DECISION INSPECTOR - {which} - {where}")
        if self.show_history:
            x = game.x + game.width * 3 // 5 + 2 if self.show_inspector else game.x + 8
            rect = pygame.Rect(x, top, game.right - 8 - x, height)
            recs = self.records()
            rows = max(1, (height - 30) // LINE_HEIGHT)
            pick = len(recs) - 1 - min(self.selected, max(0, len(recs) - 1))
            start = max(0, min(pick - rows + 1, len(recs) - rows)) if recs else 0
            start = max(0, start)
            window = recs[start:start + rows]
            lines = [(history_line(r), COLOR_WARN if r.get("fallback") else (
                THEME["black_spy_text"] if r["spy"] == "black_spy" else THEME["white_spy_text"])) for r in window]
            highlight = pick - start if recs else None
            self._text_box(rect, lines or [("no decision yet", COLOR_DIM)], title="DECISION HISTORY (Up/Down)",
                           highlight=highlight)

    # -- the match summary (spec 25) -----------------------------------------------------------------------
    def summary_text(self, match_result: dict) -> list[tuple[str, tuple[int, int, int]]]:
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if self.summary:
            for i, text in enumerate(summary_lines(self.summary)):
                lines.append((text, THEME["primary_text"] if i == 0 else THEME["secondary_text"]))
            lines.append(("", THEME["muted_text"]))
        ranking = self.ranking_lines(match_result)
        lines += ranking[:-1]
        lines.append((self.mode_label, COLOR_COMMIT if self.evaluation_mode else COLOR_WARN))
        if self.enhanced:
            lines.append((rc.ENHANCED_LABEL, COLOR_WARN))
        replay = (self.summary or {}).get("replay")
        if replay:
            lines.append((f"Replay saved: {replay}", THEME["secondary_text"]))
            lines.append(("Watch it again offline (0 provider calls): python -m spy_vs_ai.replay <path>", COLOR_DIM))
        if self.replay:
            verdict = self.replay.get("verdict")
            lines.append((f"Replay verification: {verdict}" if verdict else "Replay finished", COLOR_DECIDING))
            lines.append(("One match - not a measure of model quality.   Esc quit", THEME["muted_text"]))
        else:
            lines.append(ranking[-1])
        return lines

    def _draw_final_ranking(self, match_result: dict) -> None:
        lines = self.summary_text(match_result)
        rendered = [(self.title_font if i == 0 else self.label_font).render(t or " ", True, c)
                    for i, (t, c) in enumerate(lines)]
        game = self.game_rect()
        width = min(game.width - 40, max(s.get_width() for s in rendered) + 40)
        height = min(game.height - 20, sum(s.get_height() + 4 for s in rendered) + 24)
        box = pygame.Rect(0, 0, width, height)
        box.center = game.center
        self._draw_panel_backing(box)
        self._draw_panel_backing(box)  # twice: the summary is read, not glanced at
        pygame.draw.rect(self.screen, THEME["border"], box, width=2)
        y = box.y + 12
        for surf in rendered:
            if y + surf.get_height() > box.bottom - 6:
                break
            self.screen.blit(surf, (box.x + 20, y), area=pygame.Rect(0, 0, box.width - 40, surf.get_height()))
            y += surf.get_height() + 4


__all__ = ["RuntimeRenderer", "KEYS_HELP", "PAUSE_NOTE"]
