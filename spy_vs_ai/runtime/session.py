"""The runtime match session (FINAL-2 spec 2, 5, 12, 20, 26, 37-41).

`RuntimeGame` is the existing live game (workshop.live.session.LiveGame) with only its per-spy construction hooks
overridden: each spy's controller comes from the policy registry, wrapping the frozen research implementation
(SurrogateController with B1 / B2-v1 / B2-v2, HybridController with R2.1, the LLMController) loaded read-only from
the frozen artifacts; each provider client comes from the provider registry. The engine path is unchanged:

    GameState -> build_observation(actor) -> Observation -> controller.get_action -> GameAction -> engine -> renderer

`RuntimeMatchRunner` is the existing live MatchRunner (same loop, pause, speed, headless) with the runtime renderer,
the research-mode keys, a credential-scoped telemetry logger, the runtime manifest and the replay bundle.
Outputs go to runs/<run-id>/, never into workshop/policies, workshop/policy_data or workshop/policy_reports.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pygame

from spy_vs_ai import RUNTIME_SCHEMA, __version__
from spy_vs_ai.providers.secrets import Scrubber, leaks
from spy_vs_ai.runtime import config as rc
from spy_vs_ai.runtime import policy_registry as pr
from spy_vs_ai.runtime import provider_registry as prov
from spy_vs_ai.runtime.decisions import DecisionLog, match_summary
from spy_vs_ai.runtime.recorder import EngineRecorder, bundle as replay_bundle
from src import constants
from src.llm.telemetry import TelemetryLogger
from workshop.live.config import ActorSpec, LiveConfig, SCRIPTED_MODEL
from workshop.live.monitor import LiveMonitor
from workshop.live.overlay import TelemetryView
from workshop.live.session import LiveGame, MatchRunner, _agents, verify_live_agents

SIDES = {constants.BLACK_SPY_ID: "black", constants.WHITE_SPY_ID: "white"}
ACTOR_OF = {v: k for k, v in SIDES.items()}
DEFAULT_HYBRID_CALLS = 300  # per hybrid spy and match, unless max_provider_calls says otherwise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RuntimeTelemetryLogger(TelemetryLogger):
    """The live telemetry logger, scrubbing only the selected providers' secret values (spec 11)."""

    def __init__(self, path: str | None, scrub: Scrubber) -> None:
        self.scrub = scrub
        super().__init__(path)

    def _sanitize(self, event: dict[str, Any]) -> dict[str, Any]:
        return super()._sanitize(self.scrub.value(event))


# -- configuration -> the live game's configuration ------------------------------------------------------------

def to_live_config(config: rc.GameRuntimeConfig) -> LiveConfig:
    """The LiveConfig the existing LiveGame runs on. Per-spy construction is the runtime's (RuntimeGame), so the
    actor specs carry the runtime policy; the shared settings are the mode's execution settings."""
    ex = rc.execution(config)

    def spec(side: str) -> ActorSpec:
        a = config.agent(side)
        actor = ACTOR_OF[side]
        kind = a.descriptor.kind
        if kind == "scripted":
            return ActorSpec(actor, "scripted", SCRIPTED_MODEL, a.personality, controller="scripted")
        if kind == "surrogate":
            return ActorSpec(actor, "surrogate", a.policy_key, a.personality, controller="surrogate", policy=a.policy_key)
        options = a.provider_options or {}
        request_fields = options.get("request_fields")
        fields = json.dumps(request_fields, sort_keys=True) if isinstance(request_fields, dict) else request_fields
        if kind == "hybrid":
            return ActorSpec(actor, a.provider, a.model, a.personality, request_fields=fields, controller="hybrid",
                             policy=a.policy_key)
        return ActorSpec(actor, a.provider, a.model, a.personality, request_fields=fields)

    region = next((a.provider_options.get("region") for a in (config.black, config.white)
                   if a.provider == "bedrock" and a.provider_options.get("region")), None)
    return LiveConfig(
        black=spec("black"), white=spec("white"), runtime=ex["runtime"], region=region or "us-east-1",
        timeout_seconds=120.0 if config.mode == "evaluation" else 60.0, bedrock_temperature="provider-default",
        seed=config.seed, decision_timing=ex["decision_timing"], personality_preset=None, encounters=ex["encounters"],
        time_limit=float(config.time_limit), mission_preset=None, mission_clock=ex["mission_clock"],
        mission_timer=ex["mission_timer"], commit_policy=ex["commit_policy"], lives=ex["lives"], spawn=ex["spawn"],
        items=ex["items"], puzzle=ex["puzzle"], policy_dataset=pr.DATASET, policy_emulator="logistic-v1",
        policy_target=pr.TARGET, surrogate_seed=None, hybrid_router=pr.ROUTER if config.has_hybrid else None,
        sources={"runtime": "spy_vs_ai.runtime.config"})


def load_bundles(config: rc.GameRuntimeConfig) -> dict[str, pr.FrozenBundle]:
    return {ACTOR_OF[s]: pr.load_frozen(config.agent(s).policy, config.agent(s).policy_key) for s in rc.ACTORS}


def client_factory_for(config: rc.GameRuntimeConfig, scrub: Scrubber, *, injections: dict[str, dict[str, Any]] | None = None,
                       clients: dict[str, Any] | None = None) -> Callable:
    """LiveGame's client hook: the provider registry's ModelClient for that spy's own provider / model / options."""
    injections = injections or {}

    def factory(spec: ActorSpec, live_config: LiveConfig):
        a = config.agent(SIDES[spec.actor_id])
        client = prov.build_client(a.provider, a.model, a.provider_options, scrub=scrub,
                                   **injections.get(a.provider, {}))
        if clients is not None:
            clients[spec.actor_id] = client
        return client
    return factory


class ReplayAgent:
    """Replay mode stands in for every controller: it never decides, never builds a client, never calls anything."""
    controller_kind = "replay"
    decisions_requested = 0
    execution_mode = "replay"

    def __init__(self, actor_id: str, spec: ActorSpec) -> None:
        self.actor_id = actor_id
        self.provider_name = spec.provider
        self.model_name = spec.model

    def get_action(self, observation):  # pragma: no cover - the replay never polls agents
        raise RuntimeError("a replay never asks an agent for a decision")


class RuntimeGame(LiveGame):
    """LiveGame with per-spy construction from the policy / provider registries (and a replay mode with no agents)."""

    def __init__(self, live_config: LiveConfig, telemetry, *, runtime_config: rc.GameRuntimeConfig,
                 bundles: dict[str, pr.FrozenBundle] | None = None, replay: bool = False, record: bool = True,
                 client_factory: Callable | None = None, on_finish: Callable | None = None, **kwargs: Any) -> None:
        self.runtime_config = runtime_config
        self.frozen = bundles or {}
        self.replay_mode = replay
        self.on_finish = on_finish
        self._finished_notified = False
        self.recorder = EngineRecorder() if (record and not replay) else None
        self.hybrid_budgets: dict[str, Any] = {}
        self.captured: dict[str, Any] = {}
        self._trap_records_emitted = 0
        self.traps = None
        super().__init__(live_config, telemetry, client_factory=client_factory, **kwargs)
        # enhanced gameplay (door traps, trap variants): installed before the recorder, so a recorded engine call
        # includes what the resolver did, and a replay (which installs it again) reproduces it
        if rc.enhanced(runtime_config):
            from spy_vs_ai.runtime.traps import TrapResolver
            self.traps = TrapResolver({ACTOR_OF[s]: runtime_config.agent(s).personality for s in rc.ACTORS})
            self.traps.install(self.state)
        if self.recorder is not None:
            self.recorder.install(self.state)
        if not replay:
            self._capture_observations()
        from spy_vs_ai.ui.research import RuntimeRenderer
        self.renderer = RuntimeRenderer(self.screen, runtime_config)
        pygame.display.set_caption("Spy vs AI - replay" if replay else "Spy vs AI")
        from spy_vs_ai.ui import bring_to_front
        bring_to_front()

    # -- per-spy construction (the only overridden LiveGame hooks) --------------------------------------------
    def _build_llm_agent(self, actor_id: str, agent_runtime: str, provider: str, llm_model: str | None, *args, **kwargs):
        if self.replay_mode:
            return ReplayAgent(actor_id, self.live_config.actors[actor_id])
        return super()._build_llm_agent(actor_id, agent_runtime, provider, llm_model, *args, **kwargs)

    def _build_hybrid_agent(self, spec, delegate, episode_id: str):
        from workshop.policy_emulator.hybrid import HybridBudget, HybridController, lab_pricing
        bundle = self.frozen[spec.actor_id]
        a = self.runtime_config.agent(SIDES[spec.actor_id])
        cap = DEFAULT_HYBRID_CALLS if a.max_provider_calls is None else a.max_provider_calls
        budget = HybridBudget(max_llm_calls_per_spy=cap)
        budget.new_match()
        self.hybrid_budgets[spec.actor_id] = budget
        return HybridController(spec.actor_id, bundle.base, bundle.b1, delegate, router_id=pr.ROUTER,
                                protocol=bundle.protocol, budget=budget, telemetry=self.telemetry, episode_id=episode_id,
                                seed=self.state.seed, dataset=pr.DATASET, environment_seed=self.state.seed,
                                provider_name=spec.provider, model_name=spec.model, pricing=lab_pricing())

    def _build_non_llm_agent(self, spec, episode_id: str):
        if spec.controller == "surrogate":
            from workshop.policy_emulator.surrogate import SurrogateController
            return SurrogateController(spec.actor_id, self.frozen[spec.actor_id].base, telemetry=self.telemetry,
                                       episode_id=episode_id, seed=self.state.seed, dataset=pr.DATASET,
                                       environment_seed=self.state.seed)
        return super()._build_non_llm_agent(spec, episode_id)  # scripted: the live game's own construction

    # -- observation capture for the inspector (the agent's own observation, read-only) ---------------------
    def _capture_observations(self) -> None:
        for actor_id, agent in _agents(self).items():
            if agent is None or not hasattr(agent, "get_action"):
                continue
            original = agent.get_action

            def wrapped(observation, _original=original, _actor=actor_id):
                self.captured[_actor] = observation
                return _original(observation)
            agent.get_action = wrapped  # instance attribute: the class (frozen code) is untouched

    def observation_payload(self, actor_id: str) -> dict[str, Any] | None:
        obs = self.captured.get(actor_id)
        if obs is None:
            return None
        from src.llm_controller import serialize_observation
        try:
            return serialize_observation(obs, live_state=True, live_workshop=True)
        except Exception:  # noqa: BLE001 - the inspector shows what it can
            return None

    # -- frame / end -------------------------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        if self.recorder is not None:
            self.recorder.begin()
        super()._step(dt)
        self._emit_trap_records()
        if self.recorder is not None:
            self.recorder.end(len(self.telemetry.events) if self.telemetry is not None else 0)

    def _emit_trap_records(self) -> None:
        """Runtime telemetry of the enhanced traps (spec 27): one `enhanced_trap` event per resolver record."""
        if self.traps is None:
            return
        records = self.traps.records
        while self._trap_records_emitted < len(records):
            record = records[self._trap_records_emitted]
            self._trap_records_emitted += 1
            if self.telemetry is not None:
                self.telemetry.emit_event({"event_type": "enhanced_trap", "episode_id": self.state.episode_id,
                                           "runtime_rule": "enhanced gameplay (not part of the frozen research)",
                                           **record})

    def finish_match(self) -> dict:
        result = super().finish_match()
        if not self._finished_notified and self.on_finish is not None:
            self._finished_notified = True
            self.on_finish(self)
        return result

    def in_flight(self) -> list[str]:
        return sorted(a for a, ag in _agents(self).items() if ag is not None and (
            getattr(ag, "_thinking_in_progress", False) or getattr(getattr(ag, "_ctl", None), "_thinking_in_progress", False)))


# -- manifest -------------------------------------------------------------------------------------------------

def spies_identity(config: rc.GameRuntimeConfig, bundles: dict[str, pr.FrozenBundle]) -> dict[str, Any]:
    out = {}
    for side in rc.ACTORS:
        a = config.agent(side)
        d = a.descriptor
        comp = pr.compatibility(a.policy, provider=a.provider, model=a.model, identity=a.surrogate_identity,
                                level=config.level, mode=config.mode, time_limit=config.time_limit,
                                enhanced=rc.enhanced(config))
        entry: dict[str, Any] = {"policy": a.policy, "label": d.label, "kind": d.kind, "personality": a.personality,
                                 "scientific_status": d.scientific_status, "status_summary": list(d.summary),
                                 "compatibility": comp.label, "compatibility_reasons": list(comp.reasons),
                                 "frozen_artifact": d.frozen_artifact,
                                 "artifact_identity": bundles[ACTOR_OF[side]].identity if bundles else {}}
        if d.kind in ("surrogate", "hybrid"):
            entry.update({"surrogate_identity": a.surrogate_identity, "policy_key": a.policy_key})
        if a.calls_provider:
            p = prov.PROVIDERS[a.provider]
            entry["provider"] = {"id": a.provider, "label": p.label, "model": a.model, "adapter": p.module,
                                 "credential_env_names": list(p.credential_env), "options": dict(a.provider_options)}
            if d.kind == "hybrid":
                entry["max_provider_calls"] = DEFAULT_HYBRID_CALLS if a.max_provider_calls is None else a.max_provider_calls
        out[side] = entry
    return out


def engine_identity() -> dict[str, Any]:
    from workshop.scripts.tournament import code_fingerprint
    final = Path(__file__).resolve().parents[2] / "workshop/policies/final-release/freeze_manifest.json"
    final_hash = json.loads(final.read_text()).get("manifest_hash") if final.is_file() else None
    return {"game_code_fingerprint": code_fingerprint(), "runtime_version": __version__, "runtime_schema": RUNTIME_SCHEMA,
            "final1_freeze_manifest_hash": final_hash, "python": platform.python_version(), "platform": platform.platform()}


def start_manifest(config: rc.GameRuntimeConfig, live: LiveConfig, bundles, run_dir: Path) -> dict[str, Any]:
    from workshop.policy_emulator import stage4_level_select as ls
    return {"kind": "runtime match manifest (not a scientific experiment manifest)", "schema": RUNTIME_SCHEMA,
            "run_dir": str(run_dir), "started_utc": _now(), "config": config.public(),
            "effective": {s: config.agent(s).effective() for s in rc.ACTORS}, "execution": rc.execution(config),
            "live_config": live.to_public_dict(), "spies": spies_identity(config, bundles), "engine": engine_identity(),
            "level_file": {"level": config.level, "path": ls.level_path(config.level), "sha256": ls.file_sha256(config.level)},
            "secrets_recorded": False}


class RuntimeMatchRunner(MatchRunner):
    """The live MatchRunner (loop, pause, speed, restart, headless) on RuntimeGame, plus the runtime outputs."""

    # V is the live game's detail key; here it also shows / hides the panels over the game area
    RUNTIME_KEYS = {pygame.K_v: "V", pygame.K_m: "M", pygame.K_i: "I", pygame.K_h: "H", pygame.K_y: "Y", pygame.K_UP: "Up",
                    pygame.K_DOWN: "Down", pygame.K_HOME: "Home", pygame.K_END: "End", pygame.K_TAB: "Tab"}

    def __init__(self, config: rc.GameRuntimeConfig, run_dir: Path, *, headless: bool = False, fast: bool | None = None,
                 max_decisions: int | None = None, max_wall_seconds: float | None = None,
                 event_source: Callable[[], list] | None = None, provider_injections: dict[str, dict] | None = None,
                 frame_sleep: float = 0.0, verbose: bool = True) -> None:
        self.runtime_config = rc.validate(config)
        self.live_config = to_live_config(config)
        self.scrub = prov.scrubber(config.providers_used)
        self.bundles = load_bundles(config)
        self.clients: dict[str, Any] = {}
        self.verbose = verbose
        factory = client_factory_for(config, self.scrub, injections=provider_injections, clients=self.clients)
        super().__init__(self.live_config, run_dir, headless=headless, max_decisions=max_decisions,
                         max_wall_seconds=max_wall_seconds, client_factory=factory, frame_sleep=frame_sleep,
                         event_source=event_source, split_view=True)
        self._raw_events = self.event_source
        # headless: a fixed simulated step (as fast as possible) when the mission timer is game time
        self.fast = (headless and self.live_config.mission_timer == "game_time") if fast is None else fast
        self.speed = config.speed
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = start_manifest(config, self.live_config, self.bundles, self.run_dir)
        self._write("runtime_manifest.json", self.manifest)
        self.replays: list[str] = []
        self.summaries: list[dict[str, Any]] = []
        self.logs: dict[int, DecisionLog] = {}
        self.started_utc = _now()

    def _write(self, name: str, body: Any) -> Path:
        text = self.scrub.json(body)
        found = leaks(text, self.scrub.env_names)
        if found:
            raise RuntimeError(f"refusing to write {name}: it would contain credential-like values ({len(found)})")
        path = self.run_dir / name
        path.write_text(text + "\n", encoding="utf-8")
        return path

    def _agents_info(self) -> dict[str, dict[str, Any]]:
        out = {}
        for side in rc.ACTORS:
            a = self.runtime_config.agent(side)
            out[ACTOR_OF[side]] = {"policy": a.policy, "label": a.descriptor.label, "kind": a.descriptor.kind,
                                   "base": a.descriptor.base, "personality": a.personality, "provider": a.provider,
                                   "model": a.model}
        return out

    # -- the live MatchRunner's new_match, on RuntimeGame -------------------------------------------------
    def new_match(self) -> None:
        self._close_debug("restart")
        self._close_match()
        self._over_frame_logged = False
        prefs = None
        if self.game is not None:
            old = self.game.renderer
            prefs = (old.show_telemetry, old.show_detail, old.debug_enabled, old.split, old.overview, old.zoom_level,
                     old.runtime_view, old.show_inspector, old.show_history, old.show_runtime_telemetry)
        self.matches += 1
        logger = RuntimeTelemetryLogger(str(self.run_dir / f"telemetry_match{self.matches:02d}.jsonl"), self.scrub)
        self.loggers.append(logger)
        self.game = RuntimeGame(self.live_config, logger, runtime_config=self.runtime_config, bundles=self.bundles,
                                client_factory=self.client_factory, on_finish=self._on_finish)
        problems = verify_live_agents(self.game)
        if problems:
            raise RuntimeError("runtime agents could not be verified: " + "; ".join(problems))
        self.view = TelemetryView(self.config)
        self.monitor = LiveMonitor()
        self.log = DecisionLog(self._agents_info())
        self.logs[self.matches] = self.log
        self.match_started = time.monotonic()
        renderer = self.game.renderer
        if prefs is not None:
            (renderer.show_telemetry, renderer.show_detail, renderer.debug_enabled, renderer.split, renderer.overview,
             renderer.zoom_level, renderer.runtime_view, renderer.show_inspector, renderer.show_history,
             renderer.show_runtime_telemetry) = prefs
        renderer.view, renderer.game, renderer.decision_log = self.view, self.game, self.log
        renderer.paused, renderer.speed = self.paused, self.speed
        renderer.runner = self
        if self.fast:
            from workshop.scripts.closed_loop_policy_eval import FixedStepClock
            game, view = self.game, self.view
            self.game.clock = FixedStepClock(constants.FPS)
            self._real_render = type(renderer).render
            renderer.render = lambda *args, **kwargs: (view.update(game), renderer.feed())
            self.monitor.check_overlay = lambda *args, **kwargs: None
            step, agents = self.monitor.step, [a for a in _agents(game).values() if a is not None]

            def stepped(g, dt, _step=step):
                # as the research fixed-step runner: a few ms per frame while a model thinks (the world is frozen)
                _step(g, dt)
                if any(getattr(a, "_thinking_in_progress", False) for a in agents):
                    time.sleep(0.005)
            self.monitor.step = stepped
        if self.verbose:
            print(f"[PLAY] match {self.matches}: " + "; ".join(
                f"{SIDES[a]}={i['label']}" + (f" on {i['provider']}/{i['model']}" if i['provider'] else "")
                for a, i in self._agents_info().items()) + f" | {self.runtime_config.level} seed {self.runtime_config.seed}"
                f" | {rc.execution(self.runtime_config)['label']}", flush=True)

    def screenshot(self, name: str) -> None:
        if self.fast and hasattr(self, "_real_render"):
            self._real_render(self.game.renderer, self.game.state, 0.0, None, None, self.game.bot_controller)
        super().screenshot(name)

    def _handle_events(self) -> bool:
        events = list(self._raw_events())
        rest = []
        for event in events:
            if event.type == pygame.KEYDOWN and event.key in self.RUNTIME_KEYS and self.game is not None:
                self.game.renderer.handle_key(event.key)
                self.controls_log.append({"t": round(time.monotonic() - self._started, 2),
                                          "key": self.RUNTIME_KEYS[event.key], "state": self.game.renderer.ui_state()})
            else:
                rest.append(event)
        self.event_source = lambda: rest
        try:
            return super()._handle_events()
        finally:
            self.event_source = self._raw_events

    # -- outputs ------------------------------------------------------------------------------------------
    def _on_finish(self, game: RuntimeGame) -> None:
        """Once per finished match: the summary and the replay bundle (shown on the summary screen)."""
        self.log.update(game.telemetry.events)
        summary = match_summary(game, self.log, wall_seconds=time.monotonic() - self.match_started)
        if game.traps is not None:
            summary["enhanced_traps"] = game.traps.summary()  # playable-runtime rule (not a research measure)
        self.summaries.append(summary)
        if self.runtime_config.save_replay:
            path = self.save_replay(game, complete=True)
            summary["replay"] = str(path)
        game.renderer.summary = summary

    def save_replay(self, game: RuntimeGame, *, complete: bool) -> Path:
        self.log.update(game.telemetry.events)
        body = replay_bundle(runtime_config=self.runtime_config.public(), live_config=self.live_config.to_public_dict(),
                             manifest=self.manifest, recorder=game.recorder, events=list(game.telemetry.events),
                             decisions=list(self.log.records), summary=self.summaries[-1] if self.summaries else {},
                             game=game, complete=complete, started_utc=self.started_utc, finished_utc=_now())
        path = self._write(f"replay_match{self.matches:02d}.json", body)
        self.replays.append(str(path))
        return path

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        summary = super().run()
        game = self.game
        if game is not None and game.match_result is None and self.runtime_config.save_replay:
            self.log.update(game.telemetry.events)
            self.summaries.append(match_summary(game, self.log, wall_seconds=time.monotonic() - self.match_started))
            self.save_replay(game, complete=False)
        live_summary_path = self._write("live_summary.json", summary)
        totals = {k: sum(s["total"][k] for s in self.summaries) for k in ("decisions", "provider_calls", "escalations",
                                                                           "input_tokens", "output_tokens", "fallbacks")} \
            if self.summaries else {}
        latency = [s["spies"][a]["avg_latency_ms"] for s in self.summaries for a in s["spies"]
                   if s["spies"][a].get("avg_latency_ms")]
        self.manifest.update({"finished_utc": _now(), "wall_seconds": round(time.monotonic() - started, 1),
                              "stop_reason": summary.get("stop_reason"), "matches": self.matches,
                              "outcomes": [{"winner": s["winner"], "complete": s["complete"],
                                            "spies": {a: v["outcome"] for a, v in s["spies"].items()}} for s in self.summaries],
                              "totals": totals, "average_latency_ms": round(sum(latency) / len(latency), 1) if latency else None,
                              "replays": self.replays, "summary": str(live_summary_path),
                              "provider_calls_by_spy": {a: getattr(c, "calls", None) for a, c in self.clients.items()}})
        self._write("runtime_manifest.json", self.manifest)
        summary["runtime"] = {"summaries": self.summaries, "replays": self.replays, "manifest": str(self.run_dir / "runtime_manifest.json")}
        return summary


def new_run_dir(config: rc.GameRuntimeConfig, root: Path | None = None) -> Path:
    base = Path(root or config.output_dir)
    if not base.is_absolute():
        base = Path(__file__).resolve().parents[2] / base
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path, n = base / stamp, 1
    while path.exists():
        n += 1
        path = base / f"{stamp}-{n}"
    path.mkdir(parents=True)
    return path


__all__ = ["RuntimeGame", "RuntimeMatchRunner", "to_live_config", "load_bundles", "client_factory_for", "new_run_dir",
           "RuntimeTelemetryLogger", "ReplayAgent", "start_manifest"]
