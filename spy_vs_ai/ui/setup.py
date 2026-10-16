"""The visual pre-match setup screen (FINAL-2 spec 15, 21, 30, 35).

Per spy: policy, provider, model, personality (and the surrogate identity of B1 / B2 / hybrid spies); for the match:
level, seed, mission time, game speed, mode (evaluation / playable), view (play / research), telemetry. Each spy shows
its scientific status and compatibility label; provider status (configured / unavailable, Ollama reachable /
unreachable) is checked once when the screen opens - by presence only, no credential value is read and no model is
called. START MATCH is enabled only when the configuration validates.

Mouse: click a field to open its list and click an option; the model list offers the provider's known models (the
installed ones for Ollama) and "other model id", which lets you type any id; click the seed to type it. Keyboard:
Up / Down or Tab move between fields, Left / Right change a choice, Enter opens / confirms, typing edits a text
field, Esc closes a list or quits. Nothing internal to the research (router thresholds, datasets) is exposed here.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

import pygame

from spy_vs_ai.runtime import config as rc
from spy_vs_ai.runtime import policy_registry as pr
from spy_vs_ai.runtime import provider_registry as prov
from workshop.live.overlay import COLOR_COMMIT, COLOR_DECIDING, COLOR_WARN, THEME

WIDTH, HEIGHT = 1280, 760
AUTO = "auto"
OTHER_MODEL = "__other_model__"  # the model list's "type another id" entry (never a model value)
MODEL_LABELS = {"us.anthropic.claude-sonnet-5-5": "Claude Sonnet 5.5", "us.openai.gpt-6.1-sol": "GPT-6.1 Sol"}
TIME_LIMITS = (300.0, 180.0, 120.0, 60.0, 600.0)
COMPAT_COLORS = {pr.EVALUATED: COLOR_COMMIT, pr.COMPATIBLE: COLOR_WARN, pr.EXPERIMENTAL: COLOR_WARN,
                 pr.FAKE: THEME["muted_text"], pr.BASELINE: THEME["secondary_text"]}
STATE_COLORS = {"configured": COLOR_COMMIT, "reachable": COLOR_COMMIT, "available": THEME["secondary_text"],
                "unavailable": COLOR_WARN, "unreachable": COLOR_WARN}
ENHANCED_WORD = {None: "auto", True: "on", False: "off"}
DEFAULT_MODEL = {"bedrock": "us.anthropic.claude-sonnet-5-5", "openai": "", "anthropic": "", "ollama": "qwen",
                 "fake": "fake-legal"}


@dataclass
class Field:
    key: str
    label: str
    kind: str                       # choice | text | toggle | button
    side: str | None = None
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))


class SetupScreen:
    """State + drawing + event handling of the setup screen; `run()` returns the chosen config (or None on quit)."""

    def __init__(self, config: rc.GameRuntimeConfig | None = None, *, screen: pygame.Surface | None = None,
                 event_source: Callable[[], list] | None = None, provider_status: dict[str, dict[str, Any]] | None = None,
                 ollama_models: list[str] | None = None) -> None:
        self.screen = screen
        self.event_source = event_source
        self.status = provider_status if provider_status is not None else {}
        self.ollama_models = list(ollama_models if ollama_models is not None else
                                  (self.status.get("ollama") or {}).get("models") or [])
        self.values: dict[str, Any] = {}
        self.extra: dict[str, dict[str, Any]] = {"black": {}, "white": {}}  # provider options / budgets kept from a file
        self.fields: list[Field] = []
        for side in rc.ACTORS:
            name = side.upper()
            self.fields += [Field(f"{side}.policy", f"{name} policy", "choice", side),
                            Field(f"{side}.provider", "provider", "choice", side),
                            Field(f"{side}.model", "model", "text", side),
                            Field(f"{side}.personality", "personality", "choice", side),
                            Field(f"{side}.identity", "surrogate identity", "choice", side)]
        self.fields += [Field("level", "Level", "choice"), Field("seed", "Seed", "text"),
                        Field("time_limit", "Mission time", "choice"), Field("speed", "Game speed", "choice"),
                        Field("mode", "Mode", "choice"), Field("view", "View", "choice"),
                        Field("show_telemetry", "Show telemetry", "toggle"),
                        Field("enhanced_gameplay", "Enhanced gameplay", "choice"), Field("start", "START MATCH", "button")]
        self.focus = 0
        self.open: str | None = None    # the key of the open dropdown
        self.open_index = 0
        self.editing: str | None = None  # the key of the text field being edited
        self.result: rc.GameRuntimeConfig | None = None
        self.quit = False
        self.message = ""
        self.base = config or rc.GameRuntimeConfig(
            black=rc.AgentRuntimeConfig("hybrid_v1", "cautious", provider="bedrock", model=DEFAULT_MODEL["bedrock"]),
            white=rc.AgentRuntimeConfig("b2_v1", "opportunist"))
        self.load(self.base)
        self._fonts()

    # -- state <-> config -----------------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: rc.GameRuntimeConfig, **kwargs: Any) -> "SetupScreen":
        return cls(config, **kwargs)

    def load(self, config: rc.GameRuntimeConfig) -> None:
        for side in rc.ACTORS:
            a = config.agent(side)
            self.values.update({f"{side}.policy": a.policy, f"{side}.provider": a.provider or "bedrock",
                                f"{side}.model": a.model if a.model is not None else DEFAULT_MODEL.get(a.provider or "bedrock", ""),
                                f"{side}.personality": a.personality, f"{side}.identity": a.identity or AUTO})
            self.extra[side] = {"provider_options": dict(a.provider_options), "max_provider_calls": a.max_provider_calls}
        self.values.update({"level": config.level, "seed": str(config.seed), "time_limit": float(config.time_limit),
                            "speed": float(config.speed), "mode": config.mode, "view": config.view,
                            "show_telemetry": bool(config.show_telemetry),
                            "enhanced_gameplay": ENHANCED_WORD[config.enhanced_gameplay]})

    def agent(self, side: str) -> rc.AgentRuntimeConfig:
        v = self.values
        policy = v[f"{side}.policy"]
        d = pr.POLICIES.get(policy)
        provider = model = None
        options: dict[str, Any] = {}
        calls = None
        if d is not None and d.requires_provider:
            provider = v[f"{side}.provider"]
            model = (v[f"{side}.model"] or "").strip() or None
            if provider == self._loaded_provider(side):
                options = dict(self.extra[side]["provider_options"])
            if d.kind == "hybrid":
                calls = self.extra[side]["max_provider_calls"]
        identity = v[f"{side}.identity"]
        identity = None if identity == AUTO or d is None or d.kind not in ("surrogate", "hybrid") else identity
        return rc.AgentRuntimeConfig(policy, v[f"{side}.personality"], provider=provider, model=model, identity=identity,
                                     provider_options=options, max_provider_calls=calls)

    def _loaded_provider(self, side: str) -> str | None:
        return self.base.agent(side).provider

    def to_config(self) -> rc.GameRuntimeConfig:
        """The configuration as currently shown (not validated: see problems())."""
        v = self.values
        try:
            seed: Any = int(str(v["seed"]).strip())
        except ValueError:
            seed = str(v["seed"])
        return dataclasses.replace(self.base, black=self.agent("black"), white=self.agent("white"), level=v["level"],
                                   seed=seed, time_limit=float(v["time_limit"]), speed=float(v["speed"]), mode=v["mode"],
                                   view=v["view"], show_telemetry=bool(v["show_telemetry"]),
                                   enhanced_gameplay=rc.ENHANCED_CHOICES[v["enhanced_gameplay"]])

    def problems(self) -> list[str]:
        cfg = self.to_config()
        if not isinstance(cfg.seed, int):
            return ["seed must be an integer"] + [p for p in rc.problems(dataclasses.replace(cfg, seed=0)) if "seed" not in p]
        return rc.problems(cfg)

    # -- choices --------------------------------------------------------------------------------------------
    def choices(self, key: str) -> list[tuple[Any, str]]:
        """(value, label) per option of a choice field."""
        side, _, name = key.partition(".") if "." in key else (None, "", key)
        if name == "policy":
            return [(pid, f"{d.label}  -  {d.scientific_label}") for pid, d in pr.POLICIES.items()]
        if name == "provider":
            return [(pid, f"{p.label}  -  {self.status_word(pid)}") for pid, p in prov.PROVIDERS.items()]
        if name == "personality":
            return [(p, p) for p in pr.PERSONALITIES]
        if name == "identity":
            return [(AUTO, "auto (from the model)")] + [(i, f"{i} ({m})") for i, m in pr.IDENTITIES.items()]
        if name == "model":
            known = self.model_suggestions(side)
            current = (self.values.get(key) or "").strip()
            options = ([current] if current and current not in known else []) + known
            out = [(m, f"{m}  -  {MODEL_LABELS[m]}" if m in MODEL_LABELS else m) for m in options]
            return out + [(OTHER_MODEL, "other model id... (type it)")]
        if key == "level":
            return [(lv, lv.replace("_", " ")) for lv in rc.LEVELS]
        if key == "time_limit":
            return [(t, f"{t:g} s" + ("  (research protocol)" if t == 300.0 else "")) for t in TIME_LIMITS]
        if key == "speed":
            return [(s, f"{s:g}x") for s in rc.SPEEDS]
        if key == "mode":
            return [("playable", "playable - smoother live play (not the frozen protocol)"),
                    ("evaluation", "evaluation - frozen research execution semantics")]
        if key == "view":
            return [("play", "play - the game first"), ("research", "research - decision inspector, history, telemetry")]
        if key == "enhanced_gameplay":
            return [("auto", "auto - on in playable, off in evaluation"),
                    ("on", "on - door traps, trap variants, room dressing (runtime rules)"),
                    ("off", "off - the frozen research game rules")]
        return []

    def model_suggestions(self, side: str) -> list[str]:
        provider = self.values[f"{side}.provider"]
        out = list(prov.PROVIDERS[provider].model_examples) if provider in prov.PROVIDERS else []
        if provider == "ollama":
            out = self.ollama_models + [m for m in out if m not in self.ollama_models]
        return out

    def status_word(self, provider: str) -> str:
        s = self.status.get(provider)
        return s.get("state", "?") if s else "not checked"

    def field_enabled(self, f: Field) -> bool:
        if f.side is None:
            return True
        d = pr.POLICIES.get(self.values[f"{f.side}.policy"])
        name = f.key.split(".", 1)[1]
        if name in ("provider", "model"):
            return d is not None and d.requires_provider
        if name == "identity":
            return d is not None and d.kind in ("surrogate", "hybrid")
        return True

    def set_value(self, key: str, value: Any) -> None:
        old = self.values.get(key)
        self.values[key] = value
        side, _, name = key.partition(".")
        if name == "provider" and value != old:
            suggestions = self.model_suggestions(side)
            self.values[f"{side}.model"] = DEFAULT_MODEL.get(value) or (suggestions[0] if suggestions else "")
            if value == "ollama" and self.ollama_models:
                self.values[f"{side}.model"] = self.ollama_models[0]
        self.message = ""

    def cycle(self, key: str, step: int) -> None:
        f = self._field(key)
        if f.kind == "toggle":
            self.set_value(key, not self.values[key])
            return
        if f.kind == "text" and key.endswith(".model"):
            options = self.model_suggestions(f.side)
            if options:
                cur = self.values[key]
                i = options.index(cur) if cur in options else -1
                self.set_value(key, options[(i + step) % len(options)])
            return
        options = [v for v, _ in self.choices(key)]
        if not options:
            return
        cur = self.values[key]
        i = options.index(cur) if cur in options else 0
        self.set_value(key, options[(i + step) % len(options)])

    def _field(self, key: str) -> Field:
        return next(f for f in self.fields if f.key == key)

    def enabled_fields(self) -> list[Field]:
        return [f for f in self.fields if self.field_enabled(f)]

    # -- events ---------------------------------------------------------------------------------------------
    def start(self) -> bool:
        found = self.problems()
        if found:
            self.message = "Fix first: " + found[0]
            return False
        self.result = rc.validate(self.to_config())
        return True

    def handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.quit = True
        elif event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 1) == 1:
            self._click(event.pos)
        elif event.type == pygame.KEYDOWN:
            self._key(event)

    def _choose(self, key: str, value: Any) -> None:
        """An option picked in an open list. "other model id" turns the model field into a text box (empty)."""
        self.open = None
        if value == OTHER_MODEL:
            self.values[key] = ""
            self.editing = key
            self.message = "Type the provider's model id, then Enter"
            return
        self.set_value(key, value)

    def _click(self, pos) -> None:
        if self.open is not None:
            key = self.open
            self.open = None
            for i, rect in enumerate(self._option_rects_for(key)):
                if rect.collidepoint(pos):
                    self._choose(key, self.choices(key)[i][0])
                    break
            return
        self.editing = None
        for f in self.enabled_fields():
            if f.rect.collidepoint(pos):
                self.focus = self.fields.index(f)
                self._activate(f)
                return

    def _activate(self, f: Field) -> None:
        if f.kind == "button":
            self.start()
        elif f.kind == "toggle":
            self.cycle(f.key, 1)
        elif f.kind == "text" and not f.key.endswith(".model"):
            self.editing = f.key
        elif f.kind == "choice" or f.key.endswith(".model"):  # the model field is a list (+ "other model id")
            options = [v for v, _ in self.choices(f.key)]
            self.open = f.key
            self.open_index = options.index(self.values[f.key]) if self.values[f.key] in options else 0

    def _move_focus(self, step: int) -> None:
        enabled = self.enabled_fields()
        cur = self.fields[self.focus]
        i = enabled.index(cur) if cur in enabled else 0
        self.focus = self.fields.index(enabled[(i + step) % len(enabled)])
        self.editing = None

    def _key(self, event) -> None:
        key = event.key
        if self.open is not None:
            options = self.choices(self.open)
            if key == pygame.K_ESCAPE:
                self.open = None
            elif key in (pygame.K_UP, pygame.K_DOWN):
                self.open_index = (self.open_index + (1 if key == pygame.K_DOWN else -1)) % len(options)
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._choose(self.open, options[self.open_index][0])
            return
        if self.editing is not None:
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE, pygame.K_TAB):
                self.editing = None
                if key == pygame.K_TAB:
                    self._move_focus(1)
            elif key == pygame.K_BACKSPACE:
                self.values[self.editing] = str(self.values[self.editing])[:-1]
            elif key in (pygame.K_UP, pygame.K_DOWN) and self.editing.endswith(".model"):
                self.cycle(self.editing, 1 if key == pygame.K_DOWN else -1)
            else:
                text = getattr(event, "unicode", "") or ""
                if text and text.isprintable() and len(str(self.values[self.editing])) < 120:
                    self.values[self.editing] = str(self.values[self.editing]) + text
            return
        f = self.fields[self.focus]
        if key == pygame.K_ESCAPE:
            self.quit = True
        elif key in (pygame.K_DOWN, pygame.K_TAB):
            self._move_focus(1)
        elif key == pygame.K_UP:
            self._move_focus(-1)
        elif key in (pygame.K_LEFT, pygame.K_RIGHT) and f.kind in ("choice", "toggle", "text"):
            if f.kind != "text" or f.key.endswith(".model"):
                self.cycle(f.key, 1 if key == pygame.K_RIGHT else -1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._activate(f)

    # -- drawing --------------------------------------------------------------------------------------------
    def _fonts(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.SysFont("couriernew", 14, bold=True)
        self.small = pygame.font.SysFont("couriernew", 12)
        self.title = pygame.font.SysFont("couriernew", 24, bold=True)

    def value_text(self, f: Field) -> str:
        v = self.values.get(f.key)
        if f.kind == "toggle":
            return "on" if v else "off"
        if f.kind == "button":
            return ""
        if f.kind == "text":
            return str(v) + ("_" if self.editing == f.key else "")
        for value, label in self.choices(f.key):
            if value == v:
                return label
        return str(v)

    def _text(self, text: str, pos, color=None, font=None) -> pygame.Rect:
        surf = (font or self.font).render(text, True, color or THEME["primary_text"])
        return self.screen.blit(surf, pos)

    def layout(self) -> None:
        col_w = (WIDTH - 60) // 2
        for i, side in enumerate(rc.ACTORS):
            x, y = 20 + i * (col_w + 20), 96
            for name in ("policy", "provider", "model", "personality", "identity"):
                f = self._field(f"{side}.{name}")
                f.rect = pygame.Rect(x + 170, y, col_w - 180, 24)
                y += 30
        x, y = 20, 400
        for n, key in enumerate(("level", "seed", "time_limit", "speed", "mode", "view", "show_telemetry",
                                 "enhanced_gameplay")):
            f = self._field(key)
            col, row = n % 2, n // 2
            f.rect = pygame.Rect(x + col * (col_w + 20) + 170, y + row * 30, col_w - 180, 24)
        self._field("start").rect = pygame.Rect(WIDTH // 2 - 120, HEIGHT - 70, 240, 44)

    def _option_rects(self) -> list[pygame.Rect]:
        return [] if self.open is None else self._option_rects_for(self.open)

    def _option_rects_for(self, key: str) -> list[pygame.Rect]:
        f = self._field(key)
        n = len(self.choices(key))
        width = max(f.rect.width, 520)
        x = min(f.rect.x, WIDTH - width - 10)
        top = f.rect.bottom + 2
        if top + n * 24 > HEIGHT - 10:
            top = f.rect.y - 2 - n * 24
        return [pygame.Rect(x, top + i * 24, width, 24) for i in range(n)]

    def draw(self) -> None:
        if self.screen is None:
            return
        s = self.screen
        s.fill(THEME["panel_background"])
        self.layout()
        self._text("SPY VS AI - MATCH SETUP", (20, 18), font=self.title)
        self._text("Two spies, each with its own policy, provider, model and personality. Runtime compatibility is not "
                   "scientific validation.", (20, 48), THEME["muted_text"], self.small)
        problems = self.problems()
        cfg = self.to_config()
        col_w = (WIDTH - 60) // 2
        for i, side in enumerate(rc.ACTORS):
            x = 20 + i * (col_w + 20)
            pygame.draw.rect(s, THEME["border"], pygame.Rect(x - 6, 64, col_w + 8, 316), width=1)
            color = THEME["black_spy_text"] if side == "black" else THEME["white_spy_text"]
            self._text(f"{side.upper()} SPY", (x, 72), color)
            a = cfg.agent(side)
            d = pr.POLICIES.get(a.policy)
            y = 250
            chars = (col_w - 12) // 7  # small font: about 7 px per character, kept inside the column
            if d is not None:
                self._text(d.scientific_label[:chars], (x, y), THEME["secondary_text"], self.small)
                y += 18
                for line in d.summary:
                    self._text(line[:chars], (x, y), THEME["muted_text"], self.small)
                    y += 16
                try:
                    comp = pr.compatibility(a.policy, provider=a.provider, model=a.model, identity=a.surrogate_identity,
                                            level=cfg.level, mode=cfg.mode, time_limit=float(cfg.time_limit),
                                            enhanced=rc.enhanced(cfg))
                    self._text(comp.label, (x, y + 2), COMPAT_COLORS.get(comp.label, COLOR_WARN))
                    y += 20
                    for reason in comp.reasons[:3]:
                        self._text(reason[:chars], (x, y), THEME["muted_text"], self.small)
                        y += 16
                except Exception:  # noqa: BLE001 - shown as a problem below
                    pass
        for f in self.fields:
            enabled = self.field_enabled(f)
            focused = self.fields.index(f) == self.focus
            if f.kind == "button":
                ok = not problems
                pygame.draw.rect(s, COLOR_COMMIT if ok else THEME["border"], f.rect, width=0 if ok and focused else 2)
                label = self.title.render(f.label, True, THEME["panel_background"] if ok and focused else
                                          (COLOR_COMMIT if ok else THEME["muted_text"]))
                s.blit(label, label.get_rect(center=f.rect.center))
                continue
            label_x = f.rect.x - 170
            self._text(f.label if f.side is None or not f.key.endswith(".policy") else "policy", (label_x, f.rect.y + 4),
                       THEME["secondary_text"] if enabled else THEME["muted_text"], self.small)
            if not enabled:
                self._text("- (no provider call)" if f.key.endswith((".provider", ".model")) else "-", (f.rect.x + 6, f.rect.y + 4),
                           THEME["muted_text"], self.small)
                continue
            pygame.draw.rect(s, (30, 30, 36), f.rect)
            pygame.draw.rect(s, COLOR_DECIDING if focused else THEME["border"], f.rect, width=2 if focused else 1)
            arrow = " v" if (f.kind == "choice" or f.key.endswith(".model")) and self.editing != f.key else ""
            text = self.value_text(f)
            max_chars = max(8, (f.rect.width - 24) // 8)
            self._text((text[:max_chars - 1] + "~" if len(text) > max_chars else text) + arrow, (f.rect.x + 6, f.rect.y + 4))
        # provider status (spec 21)
        x, y = 20, 520
        self._text("PROVIDER STATUS (presence only - no credential value is read, no model is called)", (x, y),
                   THEME["secondary_text"], self.small)
        y += 18
        for n, (pid, p) in enumerate(prov.PROVIDERS.items()):
            st = self.status.get(pid) or {"state": "not checked", "detail": ""}
            self._text(f"{p.label:<30} {st.get('state', '?'):<12} {str(st.get('detail', ''))[:80]}",
                       (x, y + n * 16), STATE_COLORS.get(st.get("state"), THEME["muted_text"]), self.small)
        y += 16 * len(prov.PROVIDERS) + 4
        ex = rc.execution(cfg) if cfg.mode in rc.MODES else {"label": ""}
        self._text(ex["label"], (x, y), COLOR_COMMIT if cfg.mode == "evaluation" else COLOR_WARN, self.small)
        if ex.get("enhanced_gameplay"):
            self._text(rc.ENHANCED_LABEL, (x, y + 16), COLOR_WARN, self.small)
        if problems or self.message:
            self._text((self.message or problems[0])[:150], (x, HEIGHT - 96), COLOR_WARN, self.small)
        self._text("Enter/click: open or edit   Left/Right: change   Up/Down/Tab: move   Esc: quit", (20, HEIGHT - 20),
                   THEME["muted_text"], self.small)
        if self.open is not None:
            for i, (rect, (value, label)) in enumerate(zip(self._option_rects(), self.choices(self.open))):
                pygame.draw.rect(s, (44, 44, 52) if i == self.open_index else (24, 24, 30), rect)
                pygame.draw.rect(s, THEME["border"], rect, width=1)
                self._text(label[:70], (rect.x + 6, rect.y + 4),
                           COLOR_DECIDING if value == self.values[self.open] else THEME["primary_text"], self.small)

    def run(self, *, max_frames: int | None = None) -> rc.GameRuntimeConfig | None:
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Spy vs AI - match setup")
            from spy_vs_ai.ui import bring_to_front
            bring_to_front()
        events = self.event_source or pygame.event.get
        clock = pygame.time.Clock()
        frames = 0
        pygame.key.set_repeat(350, 40)
        while not self.quit and self.result is None:
            self.layout()
            for event in events():
                self.handle_event(event)
            self.draw()
            if pygame.display.get_surface() is not None:
                pygame.display.flip()
            clock.tick(30)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        pygame.key.set_repeat()
        return self.result


def provider_status(options: dict[str, dict[str, Any]] | None = None, *, ollama_opener=None) -> dict[str, dict[str, Any]]:
    """The lightweight pre-match status of every provider (spec 21)."""
    return prov.status_all(options, ollama_opener=ollama_opener)


__all__ = ["SetupScreen", "provider_status", "WIDTH", "HEIGHT"]
