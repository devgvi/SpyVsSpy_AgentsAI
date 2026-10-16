"""A live Spy vs AI match: two real LLM agents in the production game.

`LiveGame` is the production `Game` with exactly two things changed, both
at construction time:

* each spy's provider client comes from `workshop.live.providers.build_client`
  for that spy's own (provider, model) - which is what makes per-spy
  overrides possible without touching `src/game.py`, and which raises for an
  unsupported provider instead of `Game`'s silent mock fallback;
* telemetry goes to a `SafeTelemetryLogger` under the workshop output root.

Everything else is production: the same agent runtimes, prompts,
validation/repair/fallback, the engine and the renderer (plus the read-only
overlay). Two live-only choices are passed through as ordinary `Game`
arguments:

* decision timing - `live_async` by default (the clock runs while a model
  thinks, each spy commits on its own after a stale check; see
  src/agents/live_async.py), or `evaluation` (the frozen-clock M6G.3
  response barrier the experiments use);
* each spy's personality profile (existing profiles only).
"""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pygame

from src import constants
from src.agents.live_async import LIVE_ASYNC
from src.game import DEFAULT_COMMIT_POLICY, Game
from src.llm.telemetry import TelemetryLogger
from src.mission_clock import SIMULATION_TIME, WALL_CLOCK, LiveWallClock
from src.scoring import compute_score
from workshop.live.config import ACTOR_LABELS, ACTORS, LiveConfig
from workshop.live.debug import DebugRecorder
from workshop.live.monitor import LiveMonitor
from workshop.live.overlay import LiveRenderer, TelemetryView
from workshop.live.providers import build_client, redact

SPEEDS = (0.5, 1.0, 2.0)
FRAME_DT_CAP = 0.1  # a stalled frame never advances the world by more than this
# V3: engine events copied into the telemetry file (trap ownership / lives /
# lifecycle), in engine order
TELEMETRY_ENGINE_EVENTS = ("TRAP_PLACED", "TRAP_TRIGGERED", "TRAP_DETECTED", "TRAP_DISARMED", "TRAP_REMOVED",
                           "SELF_REVERSAL_NO_STATE_CHANGE", "LIFE_LOST", "ACTOR_ELIMINATED", "ACTOR_TIMED_OUT",
                           "MISSION_TIME_EXPIRED", "MISSION_COMPLETED", "EXIT_TRANSITION_STARTED",
                           "ACTOR_LEFT_BUILDING", "EXIT_ATTEMPTED", "ITEM_FOUND", "OBJECT_SEARCHED",
                           "OBJECT_INSPECTED", "SPAWNS_ASSIGNED", "ACTOR_KNOCKED_OUT", "ACTOR_RESPAWNED",
                           "DISARM_ATTEMPTED", "TRAP_DISARM_FAILED", "ITEM_STOLEN", "ITEM_ALREADY_TAKEN",
                           "ITEMS_DROPPED", "PUZZLE_SET_UP", "KNOWLEDGE_LEARNED", "KNOWLEDGE_COMBINED",
                           "INSCRIPTION_READ", "SAFE_CODE_SOLVED", "SAFE_CODE_WRONG", "ALARM_REVEALED",
                           "MISSION_IMPOSSIBLE")
KEY_NAMES = {pygame.K_SPACE: "Space", pygame.K_r: "R", pygame.K_PLUS: "+", pygame.K_EQUALS: "+", pygame.K_KP_PLUS: "+",
             pygame.K_MINUS: "-", pygame.K_KP_MINUS: "-", pygame.K_t: "T", pygame.K_v: "V", pygame.K_F3: "F3",
             pygame.K_s: "S", pygame.K_p: "P", pygame.K_o: "O", pygame.K_z: "Z", pygame.K_ESCAPE: "Esc"}


def _deep_redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _deep_redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_deep_redact(v) for v in value]
    return value


class SafeTelemetryLogger(TelemetryLogger):
    """TelemetryLogger that also removes live secret values anywhere in an event."""

    def _sanitize(self, event: dict[str, Any]) -> dict[str, Any]:
        return super()._sanitize(_deep_redact(event))


class LiveGame(Game):
    """Production Game, with each spy's client built from its own LiveConfig spec."""

    def __init__(self, live_config: LiveConfig, telemetry: TelemetryLogger | None, *,
                 client_factory: Callable | None = None, wall_time_fn: Callable[[], float] | None = None,
                 hybrid_budget: Any = None, **kwargs: Any) -> None:
        self.live_config = live_config
        # Stage 3 milestone 4: hybrid spies share one provider-call budget (a MatchRunner's spans its matches)
        if live_config.hybrid_actors:
            from workshop.policy_emulator.hybrid import HybridBudget
            self.hybrid_budget = hybrid_budget or HybridBudget(
                live_config.max_llm_calls_per_spy, live_config.max_llm_calls_per_match, live_config.max_llm_calls_total)
            self.hybrid_budget.new_match()
        else:
            self.hybrid_budget = None
        self._live_telemetry = telemetry
        self._client_factory = client_factory or build_client
        self.live_clients: dict[str, Any] = {}
        self._building_actor: str | None = None
        # Game would otherwise open its own, unredacted telemetry file here.
        os.environ.pop("SPY_VS_AI_TELEMETRY_PATH", None)
        black = live_config.black
        super().__init__(
            seed=live_config.seed,
            white_controller="llm",
            black_controller="llm",
            # Game-level provider/model only feed its startup print for
            # Black; each spy's real client is built in _build_llm_agent.
            llm_provider=black.provider,
            llm_model=black.model,
            llm_region=live_config.region,
            llm_timeout=live_config.timeout_seconds,
            llm_execution_mode=live_config.decision_timing,
            agent_runtime=live_config.registry_runtime,
            commit_policy=live_config.commit_policy or DEFAULT_COMMIT_POLICY,
            black_personality=live_config.black.personality,
            white_personality=live_config.white.personality,
            mission_time=live_config.time_limit,
            mission_clock_mode=live_config.mission_clock,
            telemetry_path=None,
            **kwargs,
        )
        # Live play presentation (bugfix live flow, Part B): a MOVE_TO an
        # object ends next to it, not at the edge of the 70 px range, and an
        # interacting spy faces its target. Engine range rules are unchanged.
        self.face_interaction_targets = True
        # V2 Part D: spy encounters are a rule of live matches (engine-owned,
        # src/encounters.py); off everywhere else.
        self.state.encounters_enabled = live_config.encounters
        # live feedback: 3 lives (each trap triggered under a spy costs one),
        # and an escaped spy leaves the building
        if live_config.lives:
            self.state.enable_lives()
        self.state.escaped_leave_world = True
        # V3: own traps are dangerous to their owner (with own-trap memory);
        # explicit terminal lifecycle (a timed-out spy stops deciding); the
        # exit is a real doorway in the building perimeter with an automatic
        # walk-out; the shared countdown runs on the wall clock.
        from src import exit_geometry
        self.state.own_traps_dangerous = True
        self.state.live_lifecycle = True
        self.exit_geometry = exit_geometry.attach_exit_to_perimeter(self.state.level)
        self.state._rebuild_nav_grid()
        exterior = exit_geometry.exterior_point(self.state.level)
        exit_obj = self.state.level.get_object("exit_door")
        if exterior is not None and exit_obj is not None:
            door = exit_obj.collision_rect
            inside_x = door.left - 20 if exterior[0] > door.centerx else door.right + 20
            self.state.exit_walkout_path = [(float(inside_x), float(door.centery)), exterior]
        # spawn/combat polish: different start rooms (seeded), fight loser KO'd
        # for 5 s then respawned at a safe point
        from src.spawns import DIFFERENT_ROOMS, SAME_ROOM, apply_spawns
        self.spawn_record = apply_spawns(self.state, DIFFERENT_ROOMS if live_config.spawn == "different_rooms" else SAME_ROOM)
        self.state.fight_ko_respawn = True
        # live balance: INSPECT costs real time, disarming an enemy trap can fail
        self.state.live_trap_rules = True
        # the opponent coming close is a decision point for each agent
        self.state.encounter_alerts = live_config.encounters
        # one copy of each required item (--items contested, the default)
        self.state.single_copy_items = live_config.items == "contested"
        # the safe puzzle (--puzzle none|split|riddle) - src/puzzles.py
        from src import puzzles
        self.puzzle_record = puzzles.setup(self.state, live_config.puzzle)
        for actor_id, agent in _agents(self).items():
            if agent is not None and hasattr(agent, "_make_tool"):  # Strands: the tool with `code`
                spec = live_config.actors[actor_id]
                agent.request_fields = json.loads(spec.request_fields) if spec.request_fields else None
                agent.riddle_tool = live_config.puzzle == puzzles.RIDDLE
                # same temperature choice as the Native BedrockClient (--bedrock-temperature)
                agent.bedrock_temperature = live_config.bedrock_temperature_value()
                agent.prompt_cache = True  # cache the system prompt (Native: BedrockClient)
        self.state.external_mission_clock = (live_config.mission_clock == SIMULATION_TIME
                                             and live_config.mission_timer == WALL_CLOCK)
        self.wall_clock = LiveWallClock(wall_time_fn)
        self.clock_speed = 1.0
        self._events_emitted = 0
        self.match_result: dict | None = None
        # Mission time budget V2: time budget + action costs in every request,
        # and the shared mission-first directive + decision-summary contract
        # appended to each spy's system prompt (personality texts untouched).
        from src.llm.personality import render_live_mission_block
        for actor_id, agent in _agents(self).items():
            if agent is None:
                continue
            block = render_live_mission_block(live_config.actors[actor_id].personality,
                                              lives=constants.SPY_LIVES if live_config.lives else None, scoring=True,
                                              trap_risk=True,
                                              fights_cost_a_life=live_config.lives and live_config.encounters,
                                              encounters=live_config.encounters,
                                              single_copy_items=live_config.items == "contested",
                                              puzzle=live_config.puzzle)
            # token diet (src/llm/live_payload.py): a smaller payload with the same facts,
            # compact JSON, and the legend that replaces the per-call notes
            from src.llm.live_payload import LIVE_PAYLOAD_LEGEND, compose, slim_live_payload
            block = f"{block}\n{LIVE_PAYLOAD_LEGEND}"
            config = getattr(agent, "config", None)
            if config is not None and hasattr(config, "live_time_facts"):
                config.live_time_facts = True
                config.live_prompt_addendum = block
                config.model_payload_transform = compose(config.model_payload_transform, slim_live_payload)
                config.compact_payload_json = True
                # simulated clock (evaluation timing): idle re-decide and request throttle on game time
                config.game_clock_pacing = live_config.decision_timing == "evaluation"
                if config.model_response_transform is None:
                    from src.llm_controller import unwrap_function_call_envelope
                    config.model_response_transform = unwrap_function_call_envelope
            elif hasattr(agent, "live_time_facts"):  # Strands
                agent.live_time_facts = True
                agent._effective_system_prompt = f"{agent._effective_system_prompt}\n\n{block}"
                agent._model_payload_transform = compose(agent._model_payload_transform, slim_live_payload)
                agent.compact_payload_json = True
                agent.game_clock_pacing = live_config.decision_timing == "evaluation"
        self._fights_emitted = 0
        for agent in _agents(self).values():
            executor = getattr(agent, "_executor", None) if agent is not None else None
            if executor is not None:
                executor.close_approach = True
                executor.encounter_alerts = live_config.encounters
        # the info sidebar sits to the right of the game view: a wider window
        from workshop.live.overlay import SIDEBAR_WIDTH
        self.screen = pygame.display.set_mode((constants.WINDOW_WIDTH + SIDEBAR_WIDTH, constants.WINDOW_HEIGHT))
        pygame.display.set_caption("Spy vs AI - live")
        self.renderer = LiveRenderer(self.screen)

    def _build_llm_agent(self, actor_id: str, agent_runtime: str, provider: str, llm_model: str | None, *args, **kwargs):
        if self.telemetry is None:
            self.telemetry = self._live_telemetry
        spec = self.live_config.actors[actor_id]
        if spec.controller == "hybrid":  # Stage 3 milestone 4: the real LLM agent, wrapped by the hybrid router
            self._building_actor = actor_id
            try:
                delegate = super()._build_llm_agent(actor_id, agent_runtime, spec.provider, spec.model, *args, **kwargs)
            finally:
                self._building_actor = None
            return self._build_hybrid_agent(spec, delegate, kwargs.get("episode_id") or self.state.episode_id)
        if not spec.is_llm:  # Stage 3: an explicitly configured surrogate or scripted spy (no provider client)
            return self._build_non_llm_agent(spec, kwargs.get("episode_id") or self.state.episode_id)
        self._building_actor = actor_id
        try:
            return super()._build_llm_agent(actor_id, agent_runtime, spec.provider, spec.model, *args, **kwargs)
        finally:
            self._building_actor = None

    def _build_hybrid_agent(self, spec, delegate, episode_id: str):
        from workshop.policy_emulator import router
        from workshop.policy_emulator.hybrid import HybridController, lab_pricing
        base = load_surrogate_policy(self.live_config, spec, emulator=router.BASE_EMULATOR, target=router.TARGET_POLICY)
        b1 = load_surrogate_policy(self.live_config, spec, emulator=router.DIAGNOSTIC_EMULATOR, target=router.TARGET_POLICY)
        protocol = router.protocol_for(self.live_config.hybrid_router, self.live_config.policy_dataset)
        seed = self.live_config.surrogate_seed if self.live_config.surrogate_seed is not None else self.state.seed
        return HybridController(spec.actor_id, base, b1, delegate, router_id=self.live_config.hybrid_router,
                                protocol=protocol, budget=self.hybrid_budget, telemetry=self.telemetry,
                                episode_id=episode_id, seed=seed, dataset=self.live_config.policy_dataset,
                                environment_seed=self.state.seed, provider_name=spec.provider, model_name=spec.model,
                                pricing=lab_pricing())

    def _build_non_llm_agent(self, spec, episode_id: str):
        """Stage 3: the same controller contract (get_action(observation) -> GameAction), no model call."""
        if spec.controller == "surrogate":
            from workshop.policy_emulator.surrogate import SurrogateController
            policy = load_surrogate_policy(self.live_config, spec)
            seed = self.live_config.surrogate_seed if self.live_config.surrogate_seed is not None else self.state.seed
            return SurrogateController(spec.actor_id, policy, telemetry=self.telemetry, episode_id=episode_id, seed=seed,
                                       dataset=self.live_config.policy_dataset, environment_seed=self.state.seed)
        from src.bot_controller import CAUTIOUS, OPPORTUNIST, RACER, ScriptedBotController
        profile = {"racer": RACER, "cautious": CAUTIOUS}.get(spec.personality, OPPORTUNIST)
        return ScriptedBotController(spec.actor_id, profile=profile, telemetry=self.telemetry, episode_id=episode_id)

    def _build_llm_client(self, provider: str, llm_model: str | None, llm_base_url: str | None, llm_api_key: str | None):
        spec = self.live_config.actors[self._building_actor]
        client = self._client_factory(spec, self.live_config)
        self.live_clients[spec.actor_id] = client
        return client

    def _step(self, dt: float) -> None:
        if self.state.external_mission_clock:
            # V3: real unpaused wall time, applied before the agents are
            # polled - a spy whose time ran out is terminal this very frame
            self.state.advance_mission_clock(self.wall_clock.tick(self.clock_speed))
        super()._step(dt)
        self._note_terminals()
        self._emit_barrier_commits()
        # record every fight in the telemetry file too: a replay feeds the
        # recorded winners back (GameState.recorded_fight_winners), never re-rolls
        fights = self.state.fights
        while self._fights_emitted < len(fights):
            record = fights[self._fights_emitted]
            self._fights_emitted += 1
            if self.telemetry is not None:
                self.telemetry.emit_event({"event_type": "spy_fight", "episode_id": self.state.episode_id, **record})
        self._emit_engine_events()

    def _note_terminals(self) -> None:
        """The frame each spy became terminal (escaped, eliminated, timed out,
        mission impossible), with the requests it had launched by then. This
        frame's requests were launched before its actions were applied, so they
        count as before; any later request is a request after the end."""
        records = self.__dict__.setdefault("_terminal_records", {})
        agents = _agents(self)
        for actor_id in ACTORS:
            if actor_id in records:
                continue
            lifecycle = self.state.lifecycle(actor_id)
            if lifecycle in constants.TERMINAL_LIFECYCLES or self.state.out_of_play(actor_id):
                records[actor_id] = {"lifecycle": lifecycle,
                                     "mission_time_remaining": round(self.state.mission_time_remaining, 3),
                                     "world_time": round(self.state.elapsed_time, 3),
                                     "requests_at_terminal": getattr(agents.get(actor_id), "decisions_requested", 0)}

    def terminal_record(self, actor_id: str) -> dict | None:
        record = self.__dict__.get("_terminal_records", {}).get(actor_id)
        if record is None:
            return None
        requested = getattr(_agents(self).get(actor_id), "decisions_requested", 0)
        return {**record, "requests_after_terminal": requested - record["requests_at_terminal"]}

    def _emit_barrier_commits(self) -> None:
        """Evaluation timing: one `barrier_commit` record per decision a barrier
        release applies - the equivalent of live_async_commit, which that mode
        never writes. Times follow the live_async convention (commit_game_time =
        mission time remaining) so one analysis reads both modes."""
        release = self.last_barrier_release
        if release is None or release is getattr(self, "_last_release_emitted", None):
            return
        self._last_release_emitted = release
        if self.telemetry is None:
            return
        agents = _agents(self)
        world, remaining = self.release_clock or (self.state.elapsed_time, self.state.mission_time_remaining)
        for index, actor_id in enumerate(release.get("commit_order", [])):
            agent = agents.get(actor_id)
            req_world, req_remaining = self.request_clock.get(actor_id, (None, None))
            self.telemetry.emit_event({
                "event_type": "barrier_commit", "episode_id": self.state.episode_id, "actor_id": actor_id,
                "decision_id": getattr(agent, "_last_decision_id", None),
                "decision_boundary_id": release.get("decision_boundary_id"), "commit_index": index,
                "commit_policy": self.commit_policy, "game_time_convention": "mission_time_remaining",
                "commit_game_time": round(remaining, 3), "world_time": round(world, 3),
                "request_game_time": None if req_remaining is None else round(req_remaining, 3),
                "request_world_time": None if req_world is None else round(req_world, 3),
                # world / mission seconds that passed between this request and its commit
                # (0 under the simulated clock: the world is frozen while a model thinks)
                "world_seconds_waiting": None if req_world is None else round(world - req_world, 3),
                "mission_seconds_waiting": None if req_remaining is None else round(req_remaining - remaining, 3),
                "discarded_due_to_stale": False})

    def pause_clock(self) -> None:
        """A paused (or finished) frame: that wall time is not match time."""
        self.wall_clock.skip()

    def _emit_engine_events(self) -> None:
        """V3 §16/§48: trap, life and lifecycle engine events, in engine order."""
        log = self.state.event_log
        while self._events_emitted < len(log):
            event = log[self._events_emitted]
            self._events_emitted += 1
            if self.telemetry is not None and event.get("type") in TELEMETRY_ENGINE_EVENTS:
                self.telemetry.emit_event({"event_type": "engine_event", "episode_id": self.state.episode_id,
                                           "engine_seq": self._events_emitted - 1,
                                           "game_time": round(self.state.elapsed_time, 3),
                                           "mission_time_remaining": round(self.state.mission_time_remaining, 3),
                                           "engine_event_type": event["type"],
                                           **{k: v for k, v in event.items() if k != "type"}})

    def match_over(self) -> bool:
        state = self.state
        # an escaping spy finishes its walk out of the building first
        return not state.exit_walking and all(
            state.lifecycle(a) in constants.TERMINAL_LIFECYCLES or compute_score(state, a).failed for a in ACTORS)

    def discard_terminal_answers(self) -> None:
        """Match over: an answer that arrives for a finished spy is recorded as
        DISCARDED_TERMINAL and never committed (V3 §31)."""
        from src.agents import live_async
        from src.observation import build_observation
        for actor_id, agent in _agents(self).items():
            if agent is None or not getattr(agent, "_live_async", False):
                continue
            if getattr(agent, "_live_pending", None) is not None and agent._response_ready_uncommitted \
                    and not agent._thinking_in_progress:
                live_async.cancel_ready_decision(agent, build_observation(self.state, actor_id), live_async.DISCARDED_TERMINAL)
                agent._cancel_ready_response(live_async.DISCARDED_TERMINAL)

    def finish_match(self) -> dict:
        """Once, when every spy is terminal: the final ranking (no model call)."""
        self.discard_terminal_answers()
        if self.match_result is None:
            from src.ranking import rank_match
            self._emit_engine_events()
            ranking = rank_match(self.state)
            in_flight = sorted(a for a, ag in _agents(self).items() if ag is not None and getattr(ag, "_thinking_in_progress", False))
            self.match_result = {"event_type": "match_result", "episode_id": self.state.episode_id,
                                 "game_time": round(self.state.elapsed_time, 3),
                                 "escape_order": list(self.state.finish_order), "ranking": ranking,
                                 # launched before the end, answered after: never committed
                                 "requests_in_flight_at_match_end": in_flight}
            if self.telemetry is not None:
                self.telemetry.emit_event(dict(self.match_result))
        return self.match_result


def _agents(game: Game) -> dict[str, Any]:
    return {constants.BLACK_SPY_ID: game.black_llm_controller, constants.WHITE_SPY_ID: game.bot_controller}


def load_surrogate_policy(config: LiveConfig, spec, *, emulator: str | None = None, target: str | None = None):
    """The surrogate spy's trained artifact, provenance-checked against the current dataset."""
    from workshop.policy_emulator import dataset as policy_dataset
    from workshop.policy_emulator.artifacts import load_policy

    try:
        expected = policy_dataset.load_provenance(config.policy_dataset)["dataset_hash"]
    except (FileNotFoundError, KeyError):
        raise RuntimeError(f"policy dataset '{config.policy_dataset}' is not built: "
                           f"python -m workshop.scripts.build_policy_dataset --dataset {config.policy_dataset}") from None
    return load_policy(config.policy_dataset, spec.policy, emulator=emulator or config.policy_emulator,
                       target=target or config.policy_target, expected_dataset_hash=expected)


def _verify_non_llm(actor_id: str, spec, agent) -> list[str]:
    from src.bot_controller import ScriptedBotController
    from workshop.policy_emulator.surrogate import SurrogateController

    if spec.controller == "hybrid":
        from src.llm.clients.mock_client import MockModelClient
        from workshop.policy_emulator.hybrid import HybridController
        if not isinstance(agent, HybridController):
            return [f"{ACTOR_LABELS[actor_id]} is not the configured hybrid controller"]
        problems = []
        if agent.policy.policy_key != spec.policy:
            problems.append(f"{ACTOR_LABELS[actor_id]} plays {agent.policy.policy_key}, expected {spec.policy}")
        delegate = agent._ctl
        if getattr(delegate, "provider_name", None) != spec.provider or getattr(delegate, "model_name", None) != spec.model:
            problems.append(f"{ACTOR_LABELS[actor_id]} escalates to {getattr(delegate, 'provider_name', '?')}/"
                            f"{getattr(delegate, 'model_name', '?')}, expected {spec.provider}/{spec.model}")
        if isinstance(getattr(delegate, "model_client", None), MockModelClient):
            problems.append(f"{ACTOR_LABELS[actor_id]} has a mock model client")
        return problems
    if spec.controller == "surrogate":
        if not isinstance(agent, SurrogateController):
            return [f"{ACTOR_LABELS[actor_id]} is not the configured surrogate"]
        if agent.policy.policy_key != spec.policy:
            return [f"{ACTOR_LABELS[actor_id]} plays {agent.policy.policy_key}, expected {spec.policy}"]
        return []
    if not isinstance(agent, ScriptedBotController):
        return [f"{ACTOR_LABELS[actor_id]} is not the configured scripted bot"]
    return []


def verify_live_agents(game: LiveGame) -> list[str]:
    """Proves the match really runs the configured agents: LLM agents on the configured
    providers (no mock, no script), and a surrogate or scripted spy only where one was
    explicitly configured (Stage 3)."""
    from src.bot_controller import ScriptedBotController
    from src.llm.clients.mock_client import MockModelClient

    problems = []
    for actor_id, agent in _agents(game).items():
        spec = game.live_config.actors[actor_id]
        if not spec.is_llm:
            problems.extend(_verify_non_llm(actor_id, spec, agent))
            continue
        if agent is None or isinstance(agent, ScriptedBotController):
            problems.append(f"{ACTOR_LABELS[actor_id]} is not an LLM agent")
            continue
        if getattr(agent, "provider_name", None) != spec.provider or getattr(agent, "model_name", None) != spec.model:
            problems.append(f"{ACTOR_LABELS[actor_id]} is on {getattr(agent, 'provider_name', '?')}/{getattr(agent, 'model_name', '?')}, "
                            f"expected {spec.provider}/{spec.model}")
        controller = getattr(agent, "controller", None)
        client = getattr(controller, "model_client", None) if controller is not None else None
        if isinstance(client, MockModelClient):
            problems.append(f"{ACTOR_LABELS[actor_id]} has a mock model client")
    timing = game.live_config.decision_timing
    llm_agents = {a: ag for a, ag in _agents(game).items() if game.live_config.actors[a].is_llm}
    if timing == LIVE_ASYNC:
        for actor_id, agent in llm_agents.items():
            if agent is None:
                continue
            if getattr(agent, "execution_mode", None) != LIVE_ASYNC:
                problems.append(f"{ACTOR_LABELS[actor_id]} is not in live_async timing")
            if not _ready_guard(agent):
                problems.append(f"{ACTOR_LABELS[actor_id]} has no HP09 ready-response guard")
    elif len(llm_agents) == len(ACTORS) and not game._barrier_active():
        # the barrier is between two thinking models; a surrogate or scripted spy never thinks
        problems.append("the production response barrier is not active")
    return problems


def _ready_guard(agent) -> bool:
    config = getattr(agent, "config", None)
    if config is not None and hasattr(config, "ready_response_guard"):
        return bool(config.ready_response_guard)
    return bool(getattr(agent, "ready_response_guard", False))


def summarize(game: LiveGame, view: TelemetryView, started: float, *, matches: int, stop_reason: str) -> dict[str, Any]:
    events = game.telemetry.events if game.telemetry is not None else []
    per_actor = {}
    for actor_id, agent in _agents(game).items():
        spec = game.live_config.actors[actor_id]
        kind = {"surrogate": "surrogate_decision", "scripted": "opponent_decision",
                "hybrid": "hybrid_decision"}.get(spec.controller, "llm_decision")
        decisions = [e for e in events if e.get("event_type") == kind and e.get("actor_id") == actor_id]
        calls = [e for e in events if e.get("event_type") == "provider_call" and e.get("actor_id") == actor_id]
        latencies = sorted(e["latency_ms"] for e in decisions if isinstance(e.get("latency_ms"), int | float))
        per_actor[actor_id] = {
            "provider": spec.provider, "model": spec.model,
            "agent_runtime": getattr(agent, "runtime_name", None),
            "telemetry_providers": sorted({str(e.get("provider")) for e in decisions}),
            "telemetry_models": sorted({str(e.get("model")) for e in decisions}),
            "decisions": len(decisions),
            "provider_calls": len(calls),
            "action_source_llm": sum(1 for e in decisions if e.get("action_source") == "llm"),
            "repairs_attempted": sum(1 for e in decisions if e.get("repair_attempted")),
            "fallbacks": sum(1 for e in decisions if e.get("fallback_used")),
            "provider_call_failures": sum(1 for e in calls if e.get("success") is False),
            "provider_error_types": sorted({str(e.get("provider_error_type")) for e in calls if e.get("success") is False}),
            "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
            "unique_decision_ids": len({e.get("decision_id") for e in decisions}),
            "personality": spec.personality,
            "personality_in_telemetry": sorted({str(e.get("personality_profile")) for e in decisions}),
            # provenance (cross-model lab, section 19): what this spy actually ran with
            "temperature": (game.live_config.bedrock_temperature if spec.provider == "bedrock"
                            else game.live_config.openai_temperature),
            "actual_model_ids": _actual_models(game.live_clients.get(actor_id)),
            "versions": {key: sorted({str(e.get(key)) for e in decisions if e.get(key) is not None})
                         for key in ("prompt_version", "observation_schema_version", "memory_schema_version",
                                     "action_schema_version", "agent_runtime_version", "representation_variant")},
            "prompt_cache": bool(getattr(game.live_clients.get(actor_id), "cache_system_prompt", False)),
            "request_fields": json.loads(spec.request_fields) if spec.request_fields else None,
            "live_async": (agent.live_async_stats.as_dict()
                           if getattr(agent, "execution_mode", None) == LIVE_ASYNC and getattr(agent, "live_async_stats", None) else None),
            "actions": _count(e.get("executed_action") or e.get("validated_action") for e in decisions),
            "outcome": "escaped" if game.state.mission_complete.get(actor_id)
            else ("failed (eliminated)" if game.state.eliminated.get(actor_id)
                  else "failed (mission impossible)" if actor_id in game.state.mission_impossible
                  else "failed (time expired)" if compute_score(game.state, actor_id).failed else "in progress"),
            # when this spy became terminal, and the requests it launched after that (must be 0)
            "terminal": game.terminal_record(actor_id),
            "mission_impossible": game.state.mission_impossible.get(actor_id),
            "lifecycle": game.state.lifecycle(actor_id),
            "lives": game.state.lives.get(actor_id) if game.state.lives_enabled else None,
            "trap_hits": game.state.trap_hits.get(actor_id, 0),
            "time_used_s": round(game.state.mission_time_total - game.state.time_remaining_for(actor_id), 1),
            "time_remaining_s": round(game.state.time_remaining_for(actor_id), 1),
        }
        if spec.controller == "hybrid":  # Stage 3 milestone 4: B2 decisions + escalated real-LLM decisions
            llm = [e for e in events if e.get("event_type") == "llm_decision" and e.get("actor_id") == actor_id]
            per_actor[actor_id].update({
                "controller": spec.controller, "policy": spec.policy,
                "decision_sources": _count({"action": e.get("decision_source")} for e in decisions),
                "llm_decisions": len(llm), "llm_repairs": sum(1 for e in llm if e.get("repair_attempted")),
                "llm_fallbacks": sum(1 for e in llm if e.get("fallback_used")),
                "router_budget_exhausted": sum(1 for e in decisions if e.get("router_budget_exhausted")),
                "surrogate": agent.stats() if hasattr(agent, "stats") else None,
                "actions": _count(e.get("executed_action") or {"action": e.get("sampled_action")} for e in decisions)})
        elif not spec.is_llm:  # Stage 3: what played this spy instead of a model (an LLM spy's entry is unchanged)
            per_actor[actor_id].update({
                "controller": spec.controller, "policy": spec.policy, "temperature": None, "actual_model_ids": None,
                "surrogate": agent.stats() if hasattr(agent, "stats") else None,
                "actions": _count(e.get("executed_action") or {"action": e.get("selected_action")} for e in decisions)})
    return {
        "episode_id": game.state.episode_id,
        "seed": game.state.seed,
        "matches_started": matches,
        "stop_reason": stop_reason,
        "wall_seconds": round(time.monotonic() - started, 1),
        "mission_time_remaining": round(game.state.mission_time_remaining, 1),
        # world (simulation) time; with the action_budget clock each spy's own
        # mission time is in actors.<id>.time_used_s / time_remaining_s
        "game_time_elapsed": round(game.state.elapsed_time, 1),
        "mission": {"time_limit_s": game.state.mission_time_total, "clock": game.state.mission_clock_mode,
                    "preset": game.live_config.mission_preset},
        "commit_policy": game.commit_policy,
        "decision_timing": game.live_config.decision_timing,
        "barrier_active": game._barrier_active(),
        "barrier_releases_seen": view.commits,
        "live_async_commits_seen": view.live_commits,
        "last_barrier_release": game.last_barrier_release,
        "actors": per_actor,
        # engine events of the last match (spectator facts; traps / inspection)
        "engine_events": {a: dict(sorted(_count_events(game.state.event_log, a).items())) for a in ACTORS},
        "encounters_enabled": game.state.encounters_enabled,
        "fights": list(game.state.fights),
        "mission_timer": "wall_clock" if game.state.external_mission_clock else "game_time",
        "exit_geometry": getattr(game, "exit_geometry", None),
        "spawn": getattr(game, "spawn_record", None),
        "match_result": getattr(game, "match_result", None),
        "telemetry_events": len(events),
    }


def _actual_models(client) -> list[str] | str:
    """The model ids the provider reported back; Bedrock Converse reports none."""
    seen = sorted(getattr(client, "actual_models_seen", None) or [])
    if seen:
        return seen
    return "not reported by the provider (Bedrock Converse returns no model id)" \
        if getattr(client, "provider", None) == "bedrock" else "not reported"


ENGINE_EVENT_TYPES = ("OBJECT_INSPECTED", "OBJECT_SEARCHED", "TRAP_PLACED", "TRAP_TRIGGERED", "TRAP_DETECTED",
                      "TRAP_DISARMED", "TRAP_REMOVED", "ITEM_FOUND")


def _count_events(log, actor_id) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in log:
        if e.get("type") in ENGINE_EVENT_TYPES and e.get("actor_id") == actor_id:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
    return counts


def _count(actions) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in actions:
        if isinstance(a, dict):
            key = str(a.get("action") or a.get("type") or "?").lower()
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, default=str)
    path.write_text(redact(text) + "\n", encoding="utf-8")


def _code_fingerprint() -> str:
    """No VCS in this project: a hash of the code and data a match runs on."""
    from workshop.scripts.tournament import code_fingerprint
    return code_fingerprint()


def run_metadata(config: LiveConfig, *, session: str, run_id: str, headless: bool, preflight: dict,
                 debug: bool = False) -> dict[str, Any]:
    return {
        "kind": "live_llm_game",
        "session_id": session,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m workshop.play",
        "mode": "headless" if headless else "window",
        "debug": debug,
        "configuration": config.to_public_dict(),
        "game_version": _code_fingerprint(),
        "commit_policy": config.commit_policy,
        "llm_execution_mode": config.decision_timing,
        "scripted_or_replayed_agents": config.non_llm,
        **({"kind": "live_game_with_non_llm_spy",
            "controllers": {a: {"controller": s.controller, "policy": s.policy} for a, s in config.actors.items()}}
           if config.non_llm else {}),
        **({"hybrid_router_hash": _hybrid_router_hash(config)} if config.hybrid_actors else {}),
        "preflight": preflight,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "secrets_recorded": False,
    }


def _hybrid_router_hash(config: LiveConfig) -> str | None:
    from workshop.policy_emulator import router
    try:
        return router.protocol_for(config.hybrid_router, config.policy_dataset)["router_hash"]
    except Exception:  # reported by the agent build itself
        return None


class MatchRunner:
    """The interactive (or headless) loop around LiveGame.

    `event_source` defaults to `pygame.event.get`; tests inject scripted
    key presses through it. Every control key is recorded in
    `controls_log` with the state it produced, so a manual QA session
    leaves evidence in summary.json."""

    def __init__(self, config: LiveConfig, run_dir: Path, *, headless: bool = False, max_decisions: int | None = None,
                 max_wall_seconds: float | None = None, client_factory: Callable | None = None,
                 frame_sleep: float = 0.0, event_source: Callable[[], list] | None = None, debug: bool = False,
                 split_view: bool = True, view: str | None = None, zoom: float | None = None) -> None:
        self.config = config
        self.run_dir = run_dir
        self.headless = headless
        self.max_decisions = max_decisions
        self.max_wall_seconds = max_wall_seconds
        self.client_factory = client_factory
        self.frame_sleep = frame_sleep
        self.event_source = event_source or pygame.event.get
        self.matches = 0
        self.loggers: list[TelemetryLogger] = []
        self.game: LiveGame | None = None
        self.view: TelemetryView | None = None
        self.monitor: LiveMonitor | None = None
        self.match_reports: list[dict] = []
        self.paused = False
        self.speed = 1.0
        self.screenshots: list[str] = []
        self.controls_log: list[dict] = []
        self.paused_frames = 0
        self.paused_clock_drift = 0.0
        self.paused_requests = 0
        self._started = time.monotonic()
        # --debug: one debug_matchNN.jsonl per match (workshop/live/debug.py)
        self.debug = debug
        self.split_view = split_view
        # --view split|single|overview (wins over split_view) and --zoom
        self.initial_view = view or ("split" if split_view else "single")
        self.initial_zoom = zoom
        self.debug_recorder: DebugRecorder | None = None
        self.debug_files: list[str] = []

    def _close_match(self) -> None:
        if self.game is not None and self.monitor is not None:
            report = self.monitor.report(self.game)
            report["match"] = self.matches
            report["episode_id"] = self.game.state.episode_id
            report["decisions"] = dict(self.view.decision_counts)
            report["commits"] = self.view.commits
            self.match_reports.append(report)

    def _close_debug(self, reason: str) -> None:
        if self.debug_recorder is not None:
            self.debug_recorder.close(reason, self.game)
            self.debug_recorder = None

    def _debug_frame(self, dt: float) -> None:
        if self.debug_recorder is not None:
            self.debug_recorder.frame(self.game, dt=dt, speed=self.speed, paused=self.paused,
                                      actions=getattr(self.game.renderer, "last_actions", None))

    def new_match(self) -> None:
        self._close_debug("restart")
        self._close_match()
        self._over_frame_logged = False
        self._view_prefs = None
        if self.game is not None:
            old = self.game.renderer
            self._view_prefs = (old.show_telemetry, old.show_detail, old.debug_enabled, old.split, old.overview,
                                old.zoom_level)
        self.matches += 1
        logger = SafeTelemetryLogger(str(self.run_dir / f"telemetry_match{self.matches:02d}.jsonl"))
        self.loggers.append(logger)
        # Stage 3 milestone 4: one provider-call budget for every match of this run (max_llm_calls_total)
        if self.config.hybrid_actors and getattr(self, "hybrid_budget", None) is None:
            from workshop.policy_emulator.hybrid import HybridBudget
            self.hybrid_budget = HybridBudget(self.config.max_llm_calls_per_spy, self.config.max_llm_calls_per_match,
                                              self.config.max_llm_calls_total)
        self.game = LiveGame(self.config, logger, client_factory=self.client_factory,
                             **({"hybrid_budget": self.hybrid_budget} if self.config.hybrid_actors else {}))
        problems = verify_live_agents(self.game)
        if problems:
            raise RuntimeError("live agents could not be verified: " + "; ".join(problems))
        self.view = TelemetryView(self.config)
        self.monitor = LiveMonitor()
        if self.debug:
            name = f"debug_match{self.matches:02d}.jsonl"
            self.debug_recorder = DebugRecorder(self.run_dir / name, redact)
            self.debug_recorder.start(self.game, self.config)
            self.debug_files.append(name)
        renderer: LiveRenderer = self.game.renderer
        if self._view_prefs is not None:  # restart keeps the viewer's panel choices
            (renderer.show_telemetry, renderer.show_detail, renderer.debug_enabled, renderer.split,
             renderer.overview, renderer.zoom_level) = self._view_prefs
        else:
            renderer.split = self.initial_view != "single"
            renderer.overview = self.initial_view == "overview"
            if self.initial_zoom is not None:
                renderer.zoom_level = self.initial_zoom
        renderer.view, renderer.game = self.view, self.game
        renderer.paused, renderer.speed = self.paused, self.speed
        print(f"[LIVE] match {self.matches}: " + ", ".join(
            f"{ACTOR_LABELS[a]}={s.provider}/{s.model}" for a, s in self.config.actors.items())
              + f" runtime={self.config.runtime} timing={self.config.decision_timing}"
              + f" personalities=black:{self.config.black.personality},white:{self.config.white.personality}")

    def screenshot(self, name: str) -> None:
        path = self.run_dir / name
        pygame.image.save(self.game.screen, str(path))
        self.screenshots.append(name)

    def control_state(self) -> dict[str, Any]:
        renderer: LiveRenderer = self.game.renderer
        agents = _agents(self.game)
        return {
            "paused": self.paused, "speed": self.speed, "telemetry_panels": renderer.show_telemetry,
            "detail": renderer.show_detail, "debug": renderer.debug_enabled, "split": renderer.split, "match": self.matches,
            "view": renderer.view_mode, "zoom": renderer.zoom_level,
            "episode_id": self.game.state.episode_id,
            "agents": {a: f"{getattr(ag, 'provider_name', '?')}/{getattr(ag, 'model_name', '?')}/{getattr(ag, 'runtime_name', '?')}"
                       for a, ag in agents.items()},
            "game_time_remaining": round(self.game.state.mission_time_remaining, 2),
            "screenshots": len(self.screenshots),
        }

    def _handle_events(self) -> bool:
        """Workshop controls; returns False to quit. Game._step then sees no events."""
        keep_running = True
        for event in self.event_source():
            if event.type == pygame.QUIT:
                self.controls_log.append({"t": round(time.monotonic() - self._started, 2), "key": "window close"})
                return False
            if event.type != pygame.KEYDOWN or event.key not in KEY_NAMES:
                continue
            renderer: LiveRenderer = self.game.renderer
            key = event.key
            if key == pygame.K_ESCAPE:
                keep_running = False
            elif key == pygame.K_SPACE:
                self.paused = not self.paused
            elif key == pygame.K_r:
                self.new_match()
            elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.speed = SPEEDS[min(SPEEDS.index(self.speed) + 1, len(SPEEDS) - 1)]
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = SPEEDS[max(SPEEDS.index(self.speed) - 1, 0)]
            elif key == pygame.K_t:
                renderer.show_telemetry = not renderer.show_telemetry
            elif key == pygame.K_v:
                renderer.show_detail = not renderer.show_detail
            elif key == pygame.K_F3:
                renderer.toggle_debug()
            elif key == pygame.K_s:
                self.screenshot(f"screenshot_{len(self.screenshots) + 1:02d}.png")
            elif key == pygame.K_p:
                renderer.split = not renderer.split
                renderer.overview = False
            elif key == pygame.K_o:
                renderer.overview = not renderer.overview
            elif key == pygame.K_z:
                renderer.cycle_zoom()
            self.controls_log.append({"t": round(time.monotonic() - self._started, 2), "key": KEY_NAMES[key],
                                      "state": self.control_state()})
            if self.debug_recorder is not None:
                self.debug_recorder.key(KEY_NAMES[key], self.control_state())
            if not keep_running:
                return False
        return True

    def _decisions(self) -> int:
        return min(self.view.decision_counts.get(a, 0) for a in ACTORS) if self.view else 0

    def _paused_frame(self, game) -> None:
        agents = _agents(game)
        before = game.state.elapsed_time
        requested = sum(getattr(ag, "decisions_requested", 0) for ag in agents.values())
        game.renderer.render(game.state, 0.0, None, None, game.bot_controller)
        self.paused_frames += 1
        self.paused_clock_drift += abs(game.state.elapsed_time - before)
        self.paused_requests += sum(getattr(ag, "decisions_requested", 0) for ag in agents.values()) - requested

    def run(self) -> dict[str, Any]:
        started = self._started = time.monotonic()
        self.new_match()
        stop_reason = "quit"
        first_commit_shot = False
        try:
            while True:
                dt = min(self.game.clock.tick(constants.FPS) / 1000.0, FRAME_DT_CAP)
                if not self.headless and not self._handle_events():
                    stop_reason = "quit"
                    break
                game, renderer = self.game, self.game.renderer
                over = game.match_over()
                renderer.paused, renderer.speed, renderer.match_over = self.paused, self.speed, over
                if over:
                    renderer.match_result = game.finish_match()
                if self.paused or over:
                    game.pause_clock()
                    self._paused_frame(game)
                    # once the match is over nothing changes: one debug frame,
                    # not one per frame while the ranking screen stays open
                    # (an open window filled a debug file to 560 MB)
                    if not (over and getattr(self, "_over_frame_logged", False)):
                        self._debug_frame(0.0)
                    self._over_frame_logged = over
                else:
                    game.clock_speed = self.speed
                    self.monitor.step(game, dt * self.speed)
                    self.monitor.check_overlay(game, self.view)
                    self.monitor.record_memory(self.view)
                    self._debug_frame(dt * self.speed)
                self.monitor.note_progress(self.view, running=not (self.paused or over), game=game)
                first_decided = self.view.commits + self.view.live_commits >= 1 or (
                    self.config.non_llm and sum(self.view.decision_counts.values()) >= 1)
                if self.headless and not first_commit_shot and first_decided:
                    self.screenshot("screenshot_first_commit.png")
                    first_commit_shot = True
                if self.headless and over:
                    stop_reason = "match_over"
                    break
                if self.max_decisions is not None and self._decisions() >= self.max_decisions:
                    stop_reason = f"max_decisions={self.max_decisions}"
                    break
                if self.max_wall_seconds is not None and time.monotonic() - started >= self.max_wall_seconds:
                    stop_reason = f"max_wall_seconds={self.max_wall_seconds:g}"
                    break
                if self.frame_sleep:
                    time.sleep(self.frame_sleep)
            if self.headless:
                self.game.renderer.render(self.game.state, 0.0, None, None, self.game.bot_controller)
                self.screenshot("screenshot_final.png")
            self._close_debug(stop_reason)
            self._close_match()
            summary = summarize(self.game, self.view, started, matches=self.matches, stop_reason=stop_reason)
            summary["stability"] = self.match_reports
            summary["pause"] = {"paused_frames": self.paused_frames, "game_time_drift_while_paused": round(self.paused_clock_drift, 4),
                                "requests_launched_while_paused": self.paused_requests}
            summary["controls_log"] = self.controls_log
            summary["screenshots"] = list(self.screenshots)
            summary["telemetry_files"] = [Path(lg.path).name for lg in self.loggers if lg.path]
            summary["debug_files"] = list(self.debug_files)
            return summary
        except BaseException as exc:
            if self.debug_recorder is not None:
                self.debug_recorder.crash(exc)
                self._close_debug("crash")
            raise
        finally:
            pygame.quit()
            # Late answers from in-flight daemon threads may still append;
            # give them a moment, then close the files.
            time.sleep(0.2)
            for logger in self.loggers:
                try:
                    logger.close()
                except Exception:
                    pass
