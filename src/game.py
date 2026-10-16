"""Owns the main loop: input -> action -> state -> render.

This is the only module that knows about all four layers at once. The
simulation itself never asks whether an action came from a human or a bot
(or, later, an LLM) - see controllers.py / bot_controller.py. Both actors'
actions are applied through the exact same GameState.apply_action() call.
"""

import os

import pygame

from src import constants
from src.actions import ActionType, GameAction
from src.agents.live_async import LIVE_ASYNC
from src.mission_clock import ACTION_BUDGET, enable_action_budget
from src.agents.registry import DEFAULT_AGENT_RUNTIME, create_agent_runtime
from src.bot_controller import OPPORTUNIST, AgentProfile, ScriptedBotController
from src.controllers import HumanController
from src.episode import EpisodeRecord, build_episode_record
from src.items import load_items
from src.layout_validation import log_validation
from src.level import load_level
from src.llm.clients.anthropic_client import AnthropicClient
from src.llm.clients.bedrock_client import BedrockClient
from src.llm.clients.mock_client import MockModelClient
from src.llm.clients.ollama_client import OllamaClient
from src.llm.clients.openai_compatible_client import OpenAICompatibleClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig
from src.observation import build_observation
from src.renderer import Renderer
from src.state import GameState

# M6G.3: how the live loop commits two LLM decisions of one decision
# boundary. "barrier_answer_order" (P1, the default) holds a ready decision
# until the other actor's decision of the same boundary is ready too, then
# commits both on one tick in the order the answers became ready - the
# previous loop's world semantics for the same ready order, with no
# replacement request while an answer is ready but uncommitted (HP09).
# Answers that reach the loop on the same tick commit in publication order
# (response_ready_at), not Black first; only an exact timestamp tie falls
# back to Black then White. "answer_order" (P0) is the pre-M6G.3 loop
# exactly: each decision commits on the tick it is handed over, same-tick
# hand-overs Black first. See
# evaluation_runs/m6g3-response-barrier-promotion-v1/promotion_design.md.
COMMIT_POLICY_BARRIER = "barrier_answer_order"
COMMIT_POLICY_ANSWER_ORDER = "answer_order"
# Cross-model lab (workshop/labs/cross_model_full_match): the same barrier,
# but decisions released together commit in an order drawn from an RNG of
# their own (commit:<seed>) instead of first-ready first - with first-ready,
# the faster model would win every same-tick race (a contested item, a door,
# a trap) because it answered sooner in wall time.
COMMIT_POLICY_BARRIER_SEEDED = "barrier_seeded_order"
COMMIT_POLICIES = (COMMIT_POLICY_BARRIER, COMMIT_POLICY_ANSWER_ORDER, COMMIT_POLICY_BARRIER_SEEDED)
BARRIER_POLICIES = (COMMIT_POLICY_BARRIER, COMMIT_POLICY_BARRIER_SEEDED)
DEFAULT_COMMIT_POLICY = COMMIT_POLICY_BARRIER
ACTOR_ORDER = (constants.BLACK_SPY_ID, constants.WHITE_SPY_ID)  # the existing apply order, and the exact-tie fallback


class Game:
    def __init__(
        self,
        seed: int | None = None,
        white_spy_profile: AgentProfile = OPPORTUNIST,
        white_controller: str = "scripted",
        llm_provider: str = "mock",
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        # M5E.3: Bedrock's own "where" concept is a region, not a URL
        # (spec section 5) - resolved by BedrockClient itself (BEDROCK_REGION
        # -> AWS_REGION -> AWS SDK default) when this stays None, which
        # every pre-M5E.3 caller (every other provider) does.
        llm_region: str | None = None,
        llm_timeout: float | None = None,
        telemetry_path: str | None = None,
        llm_execution_mode: str = "evaluation",
        agent_runtime: str = DEFAULT_AGENT_RUNTIME,
        controller_episode_id: str | None = None,
        # M5E: an active, deterministic Black Spy opponent - "human" (the
        # existing, unchanged default: a real player, or nothing at all in
        # evaluation mode) vs "scripted" (ScriptedBotController driving
        # Black Spy through the identical Observation/GameAction boundary
        # White's scripted mode already uses - spec section 2). Every
        # pre-M5E caller passes neither of the next two args and gets
        # exactly the old behavior.
        black_controller: str = "human",
        black_spy_profile: AgentProfile = OPPORTUNIST,
        # M5E.1: a purely diagnostic tag ("passive"/"active"), threaded
        # through to the evaluated agent's telemetry only (spec section 5)
        # - never read by any decision/validation logic. None (the default
        # every pre-M5E.1 caller gets) is identical to not having this
        # field at all.
        condition: str | None = None,
        # M5E.1.1: optional context-window/residency controls (spec
        # sections 3-4, 16), threaded through to whichever client/adapter
        # the selected agent_runtime actually builds. Both None by default
        # - every pre-M5E.1.1 caller gets byte-identical behavior.
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        # M5F: alternates which actor's Observation/decision is built
        # first within one tick (spec section 26) - "black_first" (the
        # default) is byte-identical to every pre-M5F caller's own
        # unmodified ordering. Only meaningful when `black_controller ==
        # "llm"`; with a non-blocking background-thread decision call on
        # both sides (see LLMController.get_action), this never lets one
        # actor's Bedrock call start before the other's within the same
        # tick - both threads are spawned within the same `_step` call
        # regardless of order - but is still recorded/balanced across
        # episodes per spec section 26's own instruction.
        decision_call_order: str = "black_first",
        # M6G.3: commitment policy - see COMMIT_POLICY_* above. Only changes
        # anything when both actors are LLM agents in evaluation mode (the
        # barrier) and, for the HP09 guard, in every LLM controller.
        commit_policy: str = DEFAULT_COMMIT_POLICY,
        # M5F.1: per-actor strategic-priority profile (spec section 13) -
        # "default" (every pre-M5F.1 caller's implicit value) is
        # byte-identical to no profile at all. Each actor's own
        # LLMControllerConfig.personality_profile carries only its own
        # value - see `_build_llm_agent` - never the other actor's (spec
        # section 14's privacy requirement).
        black_personality: str | None = None,
        white_personality: str | None = None,
        # Bugfix mission time budget V2: the deadline (None = MISSION_TIME,
        # 180 s, every recorded run) and which clock it runs on
        # (src/mission_clock.py; SIMULATION_TIME = historical behaviour).
        mission_time: float | None = None,
        mission_clock_mode: str = "simulation_time",
        # M5F.2 spec section 13: the fixed, per-episode OFF/ON communication
        # condition - `False` (every pre-M5F.2 caller's implicit value) is
        # byte-identical to no communication channel existing at all (see
        # GameState.new_game/LLMControllerConfig.communication_enabled).
        # Symmetric for both actors - there is no such thing as "White can
        # message but Black can't" (spec section 11 keeps the whole channel
        # one on/off condition, not a per-actor toggle).
        communication_enabled: bool = False,
        # M5F.2.2/M5F.2.3: the communication cost/schema/cooldown
        # representation - all `False`/`0` (every pre-M5F.2.2 caller's
        # implicit value) is byte-identical FULL_COST behavior. Symmetric
        # for both actors, same reasoning as `communication_enabled` above
        # (spec M5F.2.3 section 11's per-actor cooldown STATE is still
        # independent per controller - only the representation choice
        # itself is episode-wide, not the runtime availability).
        zero_decision_cost_messaging: bool = False,
        flat_message_schema: bool = False,
        message_cooldown_decisions: int = 0,
        # M5F.3B spec section 11: lets a caller override the canonical
        # level's own `initial_traps` (normally `[]` for `data/level_01.json`
        # - see `Level.initial_traps`) so a real two-LLM duel can include a
        # small fixed deterministic set of environmental traps, the same way
        # `social_influence_scenarios.build_world()` already does for the
        # single-recipient probe harness. `None` (every pre-M5F.3B caller's
        # implicit value) leaves the loaded level's own traps untouched.
        initial_traps: list[dict] | None = None,
        # M5F.3C spec sections 7, 14: an optional, per-actor evaluation-only
        # hook (`(dict) -> dict`, same shape as `LLMControllerConfig.
        # model_payload_transform` - M5F.3A.4/M5F.3B.1's own hook, reused
        # unchanged) letting a caller inject that ONE actor's own bounded
        # cross-episode social memory into its model-facing payload -
        # deliberately per-actor (Black's memory of White's past behavior
        # is never the same content as White's memory of Black's), never a
        # shared/global transform. `None` (every pre-M5F.3C caller's
        # implicit value) is byte-identical to no transform at all.
        black_model_payload_transform=None,
        white_model_payload_transform=None,
    ) -> None:
        self.decision_call_order = decision_call_order.lower()
        self.commit_policy = (commit_policy or DEFAULT_COMMIT_POLICY).lower()
        if self.commit_policy not in COMMIT_POLICIES:
            raise ValueError(f"Unknown commit_policy {commit_policy!r}. Available: {', '.join(COMMIT_POLICIES)}.")
        # M6G.3 response-barrier state (see _barrier_commit): decisions handed
        # over but held, the logical decision boundary, the last release.
        self._held_decisions: dict[str, tuple[GameAction, float | None]] = {}
        self.decision_boundary_id = 0
        self._boundary_open = False
        self.request_boundary_ids: dict[str, int] = {}
        self.last_barrier_release: dict | None = None
        # telemetry only: (world time, mission time remaining) when each actor's
        # last request was launched, and when the last release committed
        self.request_clock: dict[str, tuple[float, float]] = {}
        self.release_clock: tuple[float, float] | None = None
        self._black_personality = black_personality
        self._white_personality = white_personality
        self.communication_enabled = communication_enabled
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        pygame.init()
        pygame.display.set_caption("Spy vs AI")
        self.screen = pygame.display.set_mode((constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        level = load_level(constants.LEVEL_PATH)
        if initial_traps is not None:
            level.initial_traps = initial_traps
        log_validation(level)  # fail loudly at startup on a broken level (Mileston_4A2.md section 17)
        items = load_items(constants.ITEMS_PATH)
        self.state = GameState.new_game(level, items, seed=seed, communication_enabled=communication_enabled,
                                        mission_time=mission_time)
        if mission_clock_mode == ACTION_BUDGET:
            enable_action_budget(self.state)
        self.state.zero_decision_cost_messaging = zero_decision_cost_messaging
        # M4A.3: a live-play bug report is only reproducible if the seed
        # that produced it was ever written down - print it once, always,
        # not just in a debug build (see GameState.new_game's docstring).
        print(f"[SEED] {self.state.seed}")
        # M4G: which of the "start with" three profiles White Spy plays as
        # this run - same reasoning as printing the seed, so a run can be
        # named and (with the same seed) reproduced exactly.
        print(f"[PROFILE] {white_spy_profile.name}")
        # M4H: identifies this run for src/episode.py's EpisodeRecord.
        print(f"[EPISODE] {self.state.episode_id}")
        self._white_spy_profile = white_spy_profile
        self.white_controller = white_controller.lower()
        self.llm_execution_mode = (llm_execution_mode or "evaluation").lower()
        # Bugfix live flow (Part B §25): live play turns an interacting actor
        # toward its target (set by the workshop LiveGame; off everywhere else).
        self.face_interaction_targets = False
        provider = llm_provider.lower() if llm_provider else "mock"
        telemetry_path = telemetry_path or os.getenv("SPY_VS_AI_TELEMETRY_PATH")
        self.telemetry = TelemetryLogger(telemetry_path) if telemetry_path else None
        self.llm_timeout = llm_timeout
        self._llm_region = llm_region
        if self.white_controller == "llm":
            # M5C: agent runtime is resolved through the registry rather than
            # constructing LLMController directly, so a future runtime name
            # (e.g. a Strands adapter) plugs in here without game.py changing
            # again. Unknown names raise clearly - see registry.py - never a
            # silent fallback to native.
            self.bot_controller = self._build_llm_agent(
                actor_id=constants.WHITE_SPY_ID, agent_runtime=agent_runtime, provider=provider,
                llm_model=llm_model, llm_base_url=llm_base_url, llm_api_key=llm_api_key, condition=condition,
                num_ctx=num_ctx, keep_alive=keep_alive, llm_region=llm_region,
                episode_id=controller_episode_id or self.state.episode_id,
                personality_profile=white_personality,
                communication_enabled=communication_enabled,
                zero_decision_cost_messaging=zero_decision_cost_messaging,
                flat_message_schema=flat_message_schema,
                message_cooldown_decisions=message_cooldown_decisions,
                model_payload_transform=white_model_payload_transform,
            )
        else:
            self.bot_controller = ScriptedBotController(constants.WHITE_SPY_ID, profile=white_spy_profile)

        # M5E: Black Spy's own controller - "human" preserves the exact
        # pre-M5E HumanController path; "scripted" gives Black Spy the
        # identical deterministic policy White's scripted mode already
        # uses (spec section 3), through the same actor-scoped Observation
        # boundary (build_observation(state, BLACK_SPY_ID), never GameState
        # directly - spec section 2). self.human_controller is always
        # constructed (cheap, stateless) since the renderer's HUD still
        # reads its selected_trap_type either way.
        self._black_spy_profile = black_spy_profile
        self.human_controller = HumanController()
        self.black_controller_name = (black_controller or "human").lower()
        self.black_scripted_controller = (
            ScriptedBotController(
                constants.BLACK_SPY_ID,
                profile=black_spy_profile,
                telemetry=self.telemetry,
                episode_id=controller_episode_id or self.state.episode_id,
            )
            if self.black_controller_name == "scripted"
            else None
        )
        if self.black_controller_name == "scripted":
            print(f"[OPPONENT] scripted_active_v1 profile={black_spy_profile.name}")

        # M5F: Black Spy driven by its own, fully independent LLM agent
        # instance - same provider/model/runtime as White (spec sections
        # 3-5, 38: this milestone's baseline never mixes models/providers/
        # runtimes between spies), but a SEPARATE `create_agent_runtime(...)`
        # call, meaning a separate AgentAdapter/LLMController instance with
        # its own AgentMemory, decision history, background thinking
        # thread, and (a deliberate, simplicity-over-micro-optimization
        # choice - spec section 8 permits sharing a stateless client but
        # does not require it) its own provider client instance. No
        # mutable state of any kind is shared with White's agent - see
        # `_build_llm_agent`, the exact same helper White's own path above
        # just used, called again here with BLACK_SPY_ID.
        self.black_llm_controller = (
            self._build_llm_agent(
                actor_id=constants.BLACK_SPY_ID, agent_runtime=agent_runtime, provider=provider,
                llm_model=llm_model, llm_base_url=llm_base_url, llm_api_key=llm_api_key, condition=condition,
                num_ctx=num_ctx, keep_alive=keep_alive, llm_region=llm_region,
                episode_id=controller_episode_id or self.state.episode_id,
                personality_profile=black_personality,
                communication_enabled=communication_enabled,
                zero_decision_cost_messaging=zero_decision_cost_messaging,
                flat_message_schema=flat_message_schema,
                message_cooldown_decisions=message_cooldown_decisions,
                model_payload_transform=black_model_payload_transform,
            )
            if self.black_controller_name == "llm"
            else None
        )
        if self.black_controller_name == "llm":
            print(f"[OPPONENT] llm provider={provider} model={llm_model or 'model'} runtime={agent_runtime} personality={black_personality or 'default'}")

        self.renderer = Renderer(self.screen)

        self.running = True

    def _build_llm_client(self, provider: str, llm_model: str | None, llm_base_url: str | None, llm_api_key: str | None):
        model = llm_model or "mock-model"
        if provider == "mock":
            return MockModelClient(provider="mock", model=model)
        if provider == "openai":
            return OpenAICompatibleClient(provider_name="openai", model_name=model, base_url=llm_base_url or "https://api.openai.com/v1", api_key=llm_api_key)
        if provider == "anthropic":
            return AnthropicClient(model_name=model, api_key=llm_api_key, base_url=llm_base_url or "https://api.anthropic.com")
        if provider == "bedrock":
            timeout = self.llm_timeout if getattr(self, "llm_timeout", None) is not None else None
            return BedrockClient(model_name=model, region=getattr(self, "_llm_region", None), timeout=timeout or 30.0)
        if provider == "ollama":
            timeout = self.llm_timeout if getattr(self, 'llm_timeout', None) is not None else None
            return OllamaClient(
                model_name=model, base_url=llm_base_url or "http://localhost:11434", timeout=timeout or 8.0,
                num_ctx=getattr(self, "_num_ctx", None), keep_alive=getattr(self, "_keep_alive", None),
            )
        if provider in {"openai_compatible", "local"}:
            return OpenAICompatibleClient(
                provider_name="openai_compatible",
                model_name=model,
                base_url=llm_base_url or "http://localhost:11434/v1",
                api_key=llm_api_key,
            )
        return MockModelClient(provider="mock", model=model)

    def _build_llm_agent(
        self,
        actor_id: str,
        agent_runtime: str,
        provider: str,
        llm_model: str | None,
        llm_base_url: str | None,
        llm_api_key: str | None,
        condition: str | None,
        num_ctx: int | None,
        keep_alive: str | None,
        llm_region: str | None,
        episode_id: str,
        personality_profile: str | None = None,
        communication_enabled: bool = False,
        zero_decision_cost_messaging: bool = False,
        flat_message_schema: bool = False,
        message_cooldown_decisions: int = 0,
        model_payload_transform=None,
    ):
        """Builds one fully independent LLM-driven agent for `actor_id` -
        its own provider client instance (never shared - spec section 7/8)
        and its own `create_agent_runtime(...)` adapter (own AgentMemory,
        decision history, background thinking thread). White's own
        construction and M5F's new Black construction both call this
        exact function, so the two paths cannot silently diverge.

        M6A.2 spec section 6/33 ("same personality is used" as a STOP
        CONDITION item): a real, live bug found and fixed here - this
        docstring used to document `agent_runtime="strands"` silently
        getting no personality block as an accepted, out-of-scope gap
        (true when written, before `StrandsAgentAdapter` itself gained a
        `personality_profile` parameter in M6A). That adapter-level fix
        never propagated up to this call site, so `create_agent_runtime`
        was still never given `personality_profile` for the Strands
        branch - meaning any Strands episode run through `Game` would
        silently run with no personality at all, while a paired Native
        episode with the same `white_personality` would not, breaking
        exactly the "same personality" controlled comparison this
        function exists to guarantee. Fixed by adding it to the Strands
        kwargs below too.
        """
        provider_client = self._build_llm_client(provider, llm_model, llm_base_url, llm_api_key)
        return create_agent_runtime(
            agent_runtime,
            actor_id=actor_id,
            model_client=provider_client,
            telemetry=self.telemetry,
            episode_id=episode_id,
            provider_name=provider,
            model_name=llm_model or getattr(provider_client, "model_name", "model"),
            execution_mode=self.llm_execution_mode,
            **(
                {
                    "llm_base_url": llm_base_url,
                    "llm_timeout": self.llm_timeout,
                    "condition": condition,
                    "num_ctx": num_ctx,
                    "keep_alive": keep_alive,
                    "llm_region": llm_region,
                    "personality_profile": personality_profile,
                    # M6B.2 spec sections 15, 17 (RP10 fix): the real,
                    # previously-undocumented gap M6B.1's own runtime
                    # parity manifest found - this branch never threaded
                    # `model_payload_transform` through at all, so a
                    # Strands episode had no way to receive the same
                    # cross-episode-memory-injection hook Native's own
                    # LLMControllerConfig branch below already gets. Same
                    # fix pattern as this docstring's own M6A.2 story for
                    # `personality_profile` just above.
                    "model_payload_transform": model_payload_transform,
                    # M6G.3: the HP09 guard follows the commit policy.
                    "ready_response_guard": self.commit_policy in BARRIER_POLICIES,
                }
                if agent_runtime == "strands"
                else {
                    "config": LLMControllerConfig(
                        condition=condition, personality_profile=personality_profile,
                        communication_enabled=communication_enabled,
                        zero_decision_cost_messaging=zero_decision_cost_messaging,
                        flat_message_schema=flat_message_schema,
                        message_cooldown_decisions=message_cooldown_decisions,
                        model_payload_transform=model_payload_transform,
                        ready_response_guard=self.commit_policy in BARRIER_POLICIES,
                    )
                }
            ),
        )

    def agent_profiles(self) -> dict[str, str]:
        """actor_id -> profile name, for `build_episode_record` -
        controller-level bookkeeping `GameState` deliberately doesn't own
        (see episode.py's docstring). Black Spy is "HUMAN" unless M5E's
        active opponent is enabled, in which case it's that profile's name
        (matching how White's own profile is already reported).
        """
        black_profile = "HUMAN"
        if self.black_controller_name == "scripted":
            black_profile = self._black_spy_profile.name
        elif self.black_controller_name == "llm":
            black_profile = "LLM"
        return {constants.BLACK_SPY_ID: black_profile, constants.WHITE_SPY_ID: self._white_spy_profile.name}

    def episode_record(self) -> EpisodeRecord:
        """The current episode's summary (Milestone_4.md's exact required
        fields) - callable at any point, mid-game or after either/both
        actors have resolved; see build_episode_record's docstring."""
        return build_episode_record(self.state, self.agent_profiles())

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(constants.FPS) / 1000.0
            self._step(dt)
        pygame.quit()

    def _step(self, dt: float) -> None:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                self.renderer.toggle_debug()

        keys = pygame.key.get_pressed()

        def _decide_black():
            # M5E/M5F: Black Spy's own actor-scoped Observation, built and
            # decided from *every* tick unconditionally - the opponent
            # must keep progressing on the shared simulation clock exactly
            # like White does, never frozen just because White's action
            # happens to be mid-execution. Fairness itself (spec sections
            # 21-22) comes from effective_dt below being one shared value
            # applied to both actors, not from anything special here.
            if self.black_controller_name == "scripted" and self.black_scripted_controller is not None:
                black_observation = build_observation(self.state, constants.BLACK_SPY_ID)
                return self.black_scripted_controller.get_action(black_observation)
            if self.black_controller_name == "llm" and self.black_llm_controller is not None:
                black_observation = build_observation(self.state, constants.BLACK_SPY_ID)
                return self.black_llm_controller.get_action(black_observation)
            return self.human_controller.get_action(events, keys)

        def _decide_white():
            bot_observation = build_observation(self.state, constants.WHITE_SPY_ID)
            return self.bot_controller.get_action(bot_observation)

        # M5F spec section 26: which actor's Observation/decision-request
        # happens first within a tick is a deterministic, recorded choice,
        # not always Black - `decision_call_order` defaults to the exact
        # pre-M5F "Black always first" behavior for every other caller.
        # M6G.3: under the response barrier an actor whose decision is being
        # held is not polled at all (it cannot re-request while its decision
        # is uncommitted - HP09); with nothing held this is the old loop.
        barrier = self._barrier_active()
        requests_before = self._requests_launched() if barrier else None
        deciders = {constants.BLACK_SPY_ID: _decide_black, constants.WHITE_SPY_ID: _decide_white}
        call_order = ((constants.WHITE_SPY_ID, constants.BLACK_SPY_ID) if self.decision_call_order == "white_first"
                      else ACTOR_ORDER)
        actions = {}
        for actor_id in call_order:
            actions[actor_id] = GameAction(ActionType.WAIT) if actor_id in self._held_decisions else deciders[actor_id]()

        # M5F spec sections 20-22: an LLM-controlled Black must freeze the
        # *same* shared clock White's own inference already does - checked
        # for both actors, independently, after both have already made
        # this tick's decision call (regardless of which was called first
        # above), so neither's in-flight status is ever missed.
        white_mode = getattr(self.bot_controller, "execution_mode", self.llm_execution_mode)
        white_thinking = (
            self.white_controller == "llm"
            and white_mode == "evaluation"
            and getattr(self.bot_controller, "_thinking_in_progress", False)
        )
        black_mode = getattr(self.black_llm_controller, "execution_mode", self.llm_execution_mode)
        black_thinking = (
            self.black_controller_name == "llm"
            and black_mode == "evaluation"
            and getattr(self.black_llm_controller, "_thinking_in_progress", False)
        )
        should_pause_time = white_thinking or black_thinking
        # Black then White, unless the barrier releases a boundary this tick
        # (then first-ready first) - see _barrier_commit.
        apply_order = ACTOR_ORDER
        if barrier:
            should_pause_time, apply_order = self._barrier_commit(actions, should_pause_time, requests_before)
        effective_dt = 0.0 if should_pause_time else dt

        live_positions = self._actor_positions() if getattr(self, "llm_execution_mode", None) == LIVE_ASYNC else None
        if getattr(self, "face_interaction_targets", False):
            self._face_interaction_targets(actions)
        if self.state.mission_clock_mode == ACTION_BUDGET:
            self.state.budget_hold = self._actors_waiting_for_their_model()
        for actor_id in apply_order:
            self.state.apply_action(actor_id, actions[actor_id], effective_dt)
        self.state.update(effective_dt)
        if live_positions is not None:
            self._note_live_tick(live_positions)

        self.renderer.render(self.state, dt, self.human_controller.selected_trap_type, actions, self.bot_controller)

    # -- Bugfix live flow: LIVE_ASYNC per-frame flow telemetry -----------------

    def _actors_waiting_for_their_model(self) -> set[str]:
        """ACTION_BUDGET: a spy whose only activity this frame is waiting for
        its own model's answer (request in flight or answer not yet taken,
        nothing executing, not busy with an action's cost, not stunned)."""
        held = set()
        for actor_id, agent in self._llm_agents().items():
            if agent is None or not hasattr(agent, "_executor"):
                continue
            waiting = getattr(agent, "_thinking_in_progress", False) or getattr(agent, "response_ready_uncommitted", False) \
                or actor_id in getattr(self, "_held_decisions", {})
            actor = self.state.actors[actor_id]
            if waiting and agent._executor.active is None and not actor.is_busy and not actor.is_stunned:
                held.add(actor_id)
        return held

    _FACING_ACTIONS = (ActionType.INSPECT, ActionType.INTERACT, ActionType.PLACE_TRAP, ActionType.DISARM)

    def _face_interaction_targets(self, actions: dict) -> None:
        """Live play only (Part B §25): an actor that starts an interaction
        turns toward the object it acts on. Only when the engine will really
        act on that object this frame (it is the actor's current interaction
        target and the actor is free) - no fake movement, no teleport."""
        for actor_id, action in actions.items():
            if action.type not in self._FACING_ACTIONS or not action.target_id:
                continue
            actor = self.state.actors.get(actor_id)
            if actor is None or actor.is_stunned or actor.is_busy:
                continue
            if self.state.current_interaction_target.get(actor_id) != action.target_id:
                continue
            obj = self.state.level.get_object(action.target_id)
            if obj is not None:
                center = obj.collision_rect.center
                actor.update_facing((center[0] - actor.x, center[1] - actor.y))

    def _actor_positions(self) -> dict[str, tuple[float, float]]:
        return {a: (actor.x, actor.y) for a, actor in self.state.actors.items()}

    def _note_live_tick(self, before: dict[str, tuple[float, float]]) -> None:
        """Live async only: tells each LLM agent whether it / its opponent
        moved this frame while a request was in flight (telemetry only -
        never read by a decision)."""
        after = self._actor_positions()
        agents = {a: ag for a, ag in self._llm_agents().items() if ag is not None and hasattr(ag, "note_live_tick")}
        moved = {a: before.get(a) != after.get(a) for a in after}
        for actor_id, agent in agents.items():
            other = constants.WHITE_SPY_ID if actor_id == constants.BLACK_SPY_ID else constants.BLACK_SPY_ID
            other_agent = self._llm_agents().get(other)
            agent.note_live_tick(
                self_moved=moved.get(actor_id, False), opponent_moved=moved.get(other, False),
                opponent_thinking=bool(getattr(other_agent, "_thinking_in_progress", False)),
            )

    # -- M6G.3: P1 BARRIER_ANSWER_ORDER (see promotion_design.md) --------------

    def _llm_agents(self) -> dict:
        return {constants.BLACK_SPY_ID: self.black_llm_controller, constants.WHITE_SPY_ID: self.bot_controller}

    def _barrier_active(self) -> bool:
        """The barrier only applies between two LLM agents that both freeze the
        shared clock while thinking (evaluation mode). Against a scripted or
        human opponent there is no second model decision to wait for, and
        playable mode never freezes time - those keep the per-decision commit
        (the controllers' HP09 guard still applies). "answer_order" is P0."""
        if self.commit_policy not in BARRIER_POLICIES:
            return False
        if self.white_controller != "llm" or self.black_controller_name != "llm" or self.black_llm_controller is None:
            return False
        return all(getattr(agent, "execution_mode", self.llm_execution_mode) == "evaluation" for agent in self._llm_agents().values())

    def _requests_launched(self) -> dict[str, int]:
        return {a: getattr(agent, "decisions_requested", 0) for a, agent in self._llm_agents().items()}

    def _barrier_commit(self, actions: dict, pause: bool, requests_before: dict[str, int]) -> tuple[bool, tuple[str, ...]]:
        """Runs after both get_action calls of the tick; returns (pause, apply order).

        A decision handed over this tick is held (WAIT now, zero game time)
        while the other actor's decision is outstanding - in flight, or ready
        but not handed over yet. When nothing is outstanding every held
        decision commits on this tick, first-ready first (an exact tie keeps
        Black then White). A lone decision is "held" and released within the
        same tick, i.e. commits exactly as before."""
        agents = self._llm_agents()
        thinking = {a: bool(getattr(agents[a], "_thinking_in_progress", False)) for a in ACTOR_ORDER}
        ready = {a: bool(getattr(agents[a], "response_ready_uncommitted", False)) for a in ACTOR_ORDER}
        # decision_boundary_id: boundary N opens with the first request launched
        # while nothing is outstanding or held; requests launched while it is
        # open belong to N (nothing commits meanwhile - they all see one world)
        for a in ACTOR_ORDER:
            if getattr(agents[a], "decisions_requested", 0) > requests_before[a]:
                if not self._boundary_open:
                    self.decision_boundary_id += 1
                    self._boundary_open = True
                self.request_boundary_ids[a] = self.decision_boundary_id
                self.request_clock[a] = (self.state.elapsed_time, self.state.mission_time_remaining)
        for a in ACTOR_ORDER:
            # held actors were not polled, so only fresh hand-overs are read here
            if a not in self._held_decisions and getattr(agents[a], "last_call_committed_ready_response", False):
                self._held_decisions[a] = (actions[a], getattr(agents[a], "response_ready_at", None))
        outstanding = [a for a in ACTOR_ORDER if a not in self._held_decisions and (thinking[a] or ready[a])]
        apply_order: tuple[str, ...] = ACTOR_ORDER
        if self._held_decisions and not outstanding:
            held = self._held_decisions
            if self.commit_policy == COMMIT_POLICY_BARRIER_SEEDED and len(held) > 1:
                if getattr(self, "_commit_rng", None) is None:
                    import random as _random
                    self._commit_rng = _random.Random(f"commit:{self.state.seed}")
                order = self._commit_rng.sample(sorted(held, key=ACTOR_ORDER.index), len(held))
            else:
                order = sorted(held, key=lambda a: (held[a][1] is None, held[a][1] or 0.0, ACTOR_ORDER.index(a)))
            for a in order:
                actions[a] = held[a][0]
            apply_order = tuple(order) + tuple(a for a in ACTOR_ORDER if a not in held)
            self.last_barrier_release = {"decision_boundary_id": self.decision_boundary_id, "commit_order": list(order),
                                         "response_ready_at": {a: held[a][1] for a in order}}
            self.release_clock = (self.state.elapsed_time, self.state.mission_time_remaining)
            self._held_decisions = {}
        elif self._held_decisions:
            for a in self._held_decisions:
                actions[a] = GameAction(ActionType.WAIT)
            pause = True
        # an answer that landed after its own poll freezes time until it is
        # handed over next tick (no live tick between readiness and commit)
        if any(ready[a] for a in ACTOR_ORDER):
            pause = True
        if not self._held_decisions and not any(thinking.values()) and not any(ready.values()):
            self._boundary_open = False
        return pause, apply_order
