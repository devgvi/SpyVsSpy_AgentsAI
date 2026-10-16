"""StrandsAgentAdapter: M5C.1's first external agent runtime, using the
real Strands Agents SDK (https://pypi.org/project/strands-agents/).

Architecture (see M5C.1_STRANDS_AGENT_ADAPTER.md):

    Observation + M5B memory
            |
      StrandsAgentAdapter
            |
      strands.Agent (fresh per decision - no persisted conversation,
                      context_manager=False, bounded turns)
            |
      choose_action tool  <-- proposes an action, never mutates state
            |
      validate_model_output (same M5B.1 strict gate Native uses)
            |
      canonical GameAction -> Validator -> Engine

Deliberate choices, per the spec:

- Tool design: ONE canonical `choose_action(action, target_id, trap_type,
  reason)` tool (spec section 10, "Option B") rather than nine
  action-specific tools. It maps 1:1 onto the exact same canonical
  action/target_id/trap_type/reason contract Native already uses, so the
  identical `validate_model_output` strict gate (M5B.1) applies unchanged -
  no second validation implementation to keep in sync.
- Conversation state: a brand-new `strands.Agent` is constructed for every
  `decide()` call (spec section 15). There is no reused message history
  between decisions; all persistent context flows through the explicit M5B
  `AgentMemoryTracker` payload, identically to Native. `context_manager`
  is explicitly disabled (spec section 16) - the simplest stable context
  behavior for one controlled decision at a time.
- Bounded decisions: `agent(prompt, limits={"turns": 1})` bounds the SDK's
  own internal loop (spec section 13). Exploration showed that without a
  turn limit, or even at turns=2, a local model can call the tool more than
  once with *different* proposed actions in a single invocation - the tool
  only ever executes as a proposal capture, never mutates state, and this
  adapter uses only the *first* captured proposal, recording how many were
  seen (spec section 29).
- Repair: exactly one additional Strands invocation (a second, again fully
  fresh Agent) with a short repair_instruction appended to the same
  payload, mirroring M5B.1's `LLMController._attempt_repair` - never an
  unbounded retry loop (spec section 13).
- Provider: goes through Strands' own model backends (`OllamaModel`, or
  (M5E.4) `BedrockModel` for `provider_name == "bedrock"`), not the
  project's ModelClient (spec section 18) - the point is to exercise the
  real Strands runtime end to end, not just relay a prompt through it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from src.actions import ActionType, GameAction
from src.agents import live_async
from src.mission_time import annotate_live_payload
from src.agents.action_lifecycle import ActionExecutor
from src.agents.base import AgentAdapter, AgentDecision
from src.llm.agent_memory import AgentMemoryTracker
from src.llm.personality import render_personality_block
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import (
    DEFAULT_SYSTEM_PROMPT,
    OBSERVATION_SCHEMA_VERSION,
    PROMPT_VERSION,
    REPAIR_PROMPT_VERSION,
    STRANDS_TOOL_REPRESENTATION,
    ModelOutputValidation,
    _recovery_outcome,
    _repair_instruction,
    classify_provider_call_error,
    compact_observation_for_model,
    invalid_action_signature,
    normalize_provider_error_message,
    place_trap_repair_hints,
    serialize_observation,
    validate_model_output,
)
from src.observation import Observation

STRANDS_RUNTIME_NAME = "strands"
STRANDS_RUNTIME_VERSION = "strands-adapter-v3"  # M5E.4: Bedrock model backend support, see _build_default_agent
AGENT_SDK_NAME = "strands-agents"
DECISION_TURN_LIMIT = 1  # spec section 13: bounded, appropriate for one game action

STRANDS_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + "\n\nCall the choose_action tool exactly once with your selected action. "
    "The tool only records your proposal for validation - it does not act in the game by itself."
)

TOOL_DESCRIPTION = (
    "Select exactly one action for this decision. Choose only from "
    "legal_actions_compact. Calling this tool does not execute anything by "
    "itself - it only proposes your choice for validation.\n\n"
    "Args:\n"
    "    action: one canonical action name (wait, move, interact, inspect, place_trap, disarm).\n"
    "    target_id: the object id to act on - required for every action except wait.\n"
    "    trap_type: which trap to place - only used by place_trap.\n"
    "    reason: a short one-sentence rationale (optional)."
)


class StrandsUnavailableError(RuntimeError):
    """agent_runtime='strands' was selected but the SDK (or the requested
    provider integration) isn't usable. Spec section 25: fail clearly,
    never silently fall back to Native."""


def _import_strands():
    try:
        import strands
    except ImportError as exc:
        raise StrandsUnavailableError(
            "agent_runtime='strands' was selected but the 'strands-agents' package "
            "is not installed. Install it with: pip install 'strands-agents[ollama]'"
        ) from exc
    return strands


def _agent_sdk_version() -> str | None:
    try:
        return importlib.metadata.version(AGENT_SDK_NAME)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - only if uninstalled
        return None


@dataclass
class _StrandsInvocation:
    """One real Strands Agent() call - initial or repair. Never more than
    one proposal is trusted even if the SDK's own loop produced several."""

    proposal: dict[str, Any]
    proposal_count: int
    stop_reason: str | None
    cycle_count: int
    tool_call_count: int
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    error_type: str | None
    error_message: str | None


class StrandsAgentAdapter(AgentAdapter):
    runtime_name = STRANDS_RUNTIME_NAME
    runtime_version = STRANDS_RUNTIME_VERSION

    def __init__(
        self,
        actor_id: str,
        model_client: Any = None,  # unused for Strands (spec section 18); accepted so the
        # registry call shape matches NativeAgentAdapter's.
        config: Any = None,
        telemetry: TelemetryLogger | None = None,
        episode_id: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        execution_mode: str | None = None,
        llm_base_url: str | None = None,
        llm_timeout: float | None = None,
        agent_factory: Callable[[str, Any], Any] | None = None,
        condition: str | None = None,
        # M5E.1.1: optional context-window/residency controls, mirroring
        # OllamaClient's (spec sections 3-4, 16) - both None by default,
        # meaning every pre-M5E.1.1 caller gets byte-identical OllamaModel
        # construction (Ollama's own default num_ctx/keep_alive).
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        # M5E.4: Bedrock's own region concept (mirrors BedrockClient/
        # Game's `llm_region`) - only read when `provider_name == "bedrock"`
        # (see `_build_default_agent`); None/unused for every Ollama call,
        # preserving byte-identical pre-M5E.4 behavior there.
        llm_region: str | None = None,
        # M6A spec sections 7, 10: closes a real, previously-documented gap
        # (the `config` param above has always existed but was never read
        # for anything, including personality) - Strands could not receive
        # the same personality text Native's `LLMControllerConfig.
        # personality_profile` already injects, making a "same personality"
        # controlled comparison structurally impossible before this.
        # Reuses `render_personality_block` unmodified (the exact same
        # pure function/text Native uses - spec section 7's "do not modify
        # personality text"), computed once here exactly like Native's own
        # `self._personality_block` (never re-read from a mutated config
        # at call time). `None` (every pre-M6A caller's implicit value) is
        # byte-identical to the old, fixed `STRANDS_SYSTEM_PROMPT`.
        personality_profile: str | None = None,
        # M6B.2 spec sections 15-21 (RP10 fix): the Strands-side equivalent
        # of `LLMControllerConfig.model_payload_transform` (M5F.3A.4/
        # M5F.3B.1) - previously real, undiscovered-until-M6B.1 asymmetry:
        # this parameter simply did not exist here at all, so
        # `Game._build_llm_agent` had nothing to thread it into for the
        # Strands branch (see game.py). Same input/output/opportunity
        # semantics as Native's hook (spec section 17's "same transform
        # input semantics, same transform opportunity, same transformed
        # logical context" - never byte-identical internal payloads,
        # never a second, diverging memory mechanism - spec section 18).
        # `None` (every pre-M6B.2 caller's implicit value) is
        # byte-identical to no transform at all (spec section 19).
        model_payload_transform: Any = None,
        # M6G.3 (HP09): mirrors LLMControllerConfig.ready_response_guard -
        # never launch a replacement request while a completed decision
        # has not been handed to the game loop. `False` is the pre-M6G.3
        # lifecycle exactly (Game(commit_policy="answer_order")).
        ready_response_guard: bool = True,
        # Bugfix (navigation/exploration): mirrors
        # LLMControllerConfig.live_state_representation - same one
        # serialize_observation call, so both runtimes send identical facts.
        live_state_representation: bool = True,
    ) -> None:
        self._strands = _import_strands()
        self.ready_response_guard = bool(ready_response_guard)
        self.live_state_representation = bool(live_state_representation)
        self.live_time_facts = False  # mission time budget V2 - see LLMControllerConfig.live_time_facts
        self.personality_profile = personality_profile
        self._personality_block = render_personality_block(personality_profile)
        self._effective_system_prompt = (
            f"{STRANDS_SYSTEM_PROMPT}\n\n{self._personality_block}" if self._personality_block is not None else STRANDS_SYSTEM_PROMPT
        )
        self.actor_id = actor_id
        self.telemetry = telemetry
        self.episode_id = episode_id or "ep_unknown"
        self.provider_name = provider_name or "ollama"
        self.model_name = model_name or "qwen:latest"
        # M5C.1 spec section 20: evaluation mode only - no provisional
        # strategic action while Strands thinks. Playable mode is out of
        # scope for the first external runtime (see module docstring).
        self.execution_mode = str(execution_mode or "evaluation").lower()
        # Bugfix live flow (Part A): see src/agents/live_async.py and
        # LLMController.__init__ - identical state, identical helpers.
        self._live_async = self.execution_mode == live_async.LIVE_ASYNC
        self._live_pending: live_async.PendingLiveDecision | None = None
        self._live_start_with_messages = False
        self.live_async_stats = live_async.LiveAsyncStats()
        self.status = "IDLE"
        self.last_action = GameAction(ActionType.WAIT)
        self.last_reason = "No model decision yet."
        self.max_retries = 1  # bounded repair, mirrors M5B.1

        self._llm_base_url = llm_base_url or "http://localhost:11434"
        self._llm_timeout = llm_timeout or 60.0
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._llm_region = llm_region
        self._agent_factory = agent_factory or self._build_default_agent
        self._model_payload_transform = model_payload_transform
        # M5E.1: a purely diagnostic tag ("passive"/"active"), mirroring
        # LLMControllerConfig.condition - never read by any decision logic.
        self.condition = condition

        self._memory_tracker = AgentMemoryTracker(actor_id)
        self._last_decision_id: str | None = None
        self._last_action_source: str = "llm"

        self.calls = 0
        self.invalid_calls = 0
        self.fallback_count = 0
        self.repair_count = 0
        # M6A.2: a real, previously-undiscovered gap found live - Strands
        # never got M5F.3B.1's own counter-separation fix at all (only
        # `duel_runner.py` was updated; `episode_runner.py`'s stop-rule
        # still reads the combined `fallback_count`, which both runtimes
        # increment identically). A live M6A.2 pilot episode hit 21/39
        # fallbacks, almost all `invalid_trap_type` (a clean, harmlessly-
        # repaired validation failure, never a real provider/infra
        # problem), and the harness aborted the episode at
        # `max_runtime_failures=20` before it ever finished - discarding
        # a real, valuable behavioral finding as a technical failure.
        # Mirrors `LLMController`'s own split exactly (see
        # `_request_action`): only a genuine `provider_error` counts here.
        self.provider_failure_count = 0
        self.validation_failure_count = 0
        self.repair_failure_count = 0
        # M6B.2 spec section 11: mirrors LLMController's identical counter -
        # see `invalid_action_signature`.
        self._invalid_signature_counts: dict[str, int] = {}
        # M5D.2: strategic decisions vs. accepted-action execution, kept
        # explicitly distinct (spec section 11) - the same ActionExecutor
        # LLMController uses, so both runtimes get the identical MOVE
        # multi-tick pathfinding fix and lifecycle bookkeeping from one
        # shared definition (see action_lifecycle.py).
        self._executor = ActionExecutor()
        self.decisions_requested = 0
        self.decisions_completed = 0
        self._decision_seq = 0
        self._decision_interval_seconds = 0.5
        self._next_decision_allowed_at = 0.0
        self._last_observation_digest: str | None = None
        self._thinking_thread: threading.Thread | None = None
        self._thinking_lock = threading.Lock()
        self._thinking_in_progress = False
        # M6G.3: RESPONSE_READY_UNCOMMITTED bookkeeping - identical to
        # LLMController's (see its __init__), read by Game._step's barrier.
        self._response_ready_uncommitted = False
        self.response_ready_at: float | None = None
        self.last_call_committed_ready_response = False
        self.ready_responses_cancelled = 0
        self.last_ready_response_cancel_reason: str | None = None

    @property
    def memory(self):
        return self._memory_tracker.memory

    @property
    def response_ready_uncommitted(self) -> bool:
        """M6G.3: a completed decision is waiting to be handed to the game loop."""
        return self._response_ready_uncommitted

    def _hand_over(self, action: GameAction) -> GameAction:
        """M6G.3: see LLMController._hand_over."""
        if self._response_ready_uncommitted and not self._thinking_in_progress:
            self._response_ready_uncommitted = False
            self.last_call_committed_ready_response = True
        return action

    def _cancel_ready_response(self, reason: str) -> None:
        """M6G.3: see LLMController._cancel_ready_response."""
        if self._response_ready_uncommitted and not self._thinking_in_progress:
            self._response_ready_uncommitted = False
            self.ready_responses_cancelled += 1
            self.last_ready_response_cancel_reason = reason

    def note_live_tick(self, *, self_moved: bool, opponent_moved: bool, opponent_thinking: bool) -> None:
        """Live async only: see LLMController.note_live_tick."""
        live_async.note_world_tick(self, self_moved=self_moved, opponent_moved=opponent_moved,
                                   opponent_thinking=opponent_thinking)

    @property
    def actions_accepted(self) -> int:
        return self._executor.actions_accepted

    @property
    def actions_started(self) -> int:
        return self._executor.actions_started

    @property
    def actions_completed(self) -> int:
        return self._executor.actions_completed

    @property
    def actions_failed(self) -> int:
        return self._executor.actions_failed

    @property
    def actions_interrupted(self) -> int:
        return self._executor.actions_interrupted

    @property
    def actions_cancelled(self) -> int:
        return self._executor.actions_cancelled

    def _finish_action_lifecycle(self, observation: Observation, boundary: str) -> None:
        self._executor.finish(
            memory=self.memory,
            telemetry=self.telemetry,
            episode_id=self.episode_id,
            actor_id=self.actor_id,
            agent_runtime=self.runtime_name,
            agent_runtime_version=self.runtime_version,
            observation=observation,
            boundary=boundary,
        )
        from src.llm_controller import forget_digest_after_unreachable_move
        forget_digest_after_unreachable_move(self, boundary)

    def _record_invalid_signature(self, response: dict[str, Any] | None) -> int:
        """M6B.2 spec section 11: mirrors LLMController._record_invalid_signature exactly."""
        response = response or {}
        signature = invalid_action_signature(response.get("action"), response.get("target_id"), response.get("trap_type"))
        self._invalid_signature_counts[signature] = self._invalid_signature_counts.get(signature, 0) + 1
        return self._invalid_signature_counts[signature]

    def _build_default_agent(self, system_prompt: str, tool: Any):
        # M5E.4: Bedrock uses Strands' own native `BedrockModel` (a real,
        # first-party backend - Strands is an AWS project) rather than
        # OllamaModel; region resolution/auth are boto3's, identical in
        # spirit to `BedrockClient` (AWS_BEARER_TOKEN_BEDROCK picked up
        # from the environment with no custom credential code here either -
        # `BedrockModel` builds its own boto3 client the same way).
        # `num_ctx`/`keep_alive`/`llm_base_url` are Ollama-specific and
        # meaningless for Bedrock, so they're simply not passed.
        if self.provider_name == "bedrock":
            from strands.models.bedrock import BedrockModel

            # 0.0 as in every recorded run; None (live --bedrock-temperature
            # provider-default, set by LiveGame) omits it for models that reject it
            temperature = getattr(self, "bedrock_temperature", 0.0)
            extra: dict[str, Any] = {} if temperature is None else {"temperature": temperature}
            if getattr(self, "request_fields", None):  # live: the same extra fields as the Native client
                extra["additional_request_fields"] = self.request_fields
            if getattr(self, "prompt_cache", False):  # live: cache the system prompt, as BedrockClient does
                from strands.models.model import CacheConfig

                extra["cache_config"] = CacheConfig(strategy="auto")
            model = BedrockModel(region_name=self._llm_region, model_id=self.model_name, **extra)
        else:
            from strands.models.ollama import OllamaModel

            model_kwargs: dict[str, Any] = {}
            if self._num_ctx is not None:
                model_kwargs["options"] = {"num_ctx": self._num_ctx}
            if self._keep_alive is not None:
                model_kwargs["keep_alive"] = self._keep_alive
            model = OllamaModel(
                host=self._llm_base_url,
                model_id=self.model_name,
                temperature=0.0,
                ollama_client_args={"timeout": self._llm_timeout},
                **model_kwargs,
            )
        # context_manager=False: spec section 16, the simplest stable
        # context behavior - no automatic summarization/offloading for this
        # first baseline. A fresh Agent every call also means no persisted
        # conversation history between decisions (spec section 15).
        return self._strands.Agent(
            model=model,
            tools=[tool],
            system_prompt=system_prompt,
            callback_handler=None,
            context_manager=False,
        )

    def _make_tool(self, capture: list[dict[str, Any]]):
        strands = self._strands
        if getattr(self, "riddle_tool", False):
            return self._make_riddle_tool(capture)

        @strands.tool(description=TOOL_DESCRIPTION)
        def choose_action(
            action: str,
            target_id: str | None = None,
            trap_type: str | None = None,
            reason: str | None = None,
        ) -> str:
            # M5C.1 spec section 9: a proposal capture only - no GameState
            # reference exists anywhere in this closure, so calling this
            # tool cannot mutate actor position, inventory, doors, traps,
            # or mission/timer state by construction, not by convention.
            capture.append({"action": action, "target_id": target_id, "trap_type": trap_type, "reason": reason})
            return "action recorded"

        return choose_action

    def _make_riddle_tool(self, capture: list[dict[str, Any]]):
        """Live riddle puzzle only (set by LiveGame): the same tool plus `code`.
        Every other Strands run keeps the exact tool above."""
        strands = self._strands

        @strands.tool(description=TOOL_DESCRIPTION + "\n    code: the safe code you worked out - only used by enter_safe_code.")
        def choose_action(
            action: str,
            target_id: str | None = None,
            trap_type: str | None = None,
            reason: str | None = None,
            code: str | None = None,
        ) -> str:
            entry = {"action": action, "target_id": target_id, "trap_type": trap_type, "reason": reason}
            if code is not None:
                entry["code"] = code
            capture.append(entry)
            return "action recorded"

        return choose_action

    # -- non-blocking decision lifecycle (mirrors LLMController) --------

    def get_action(self, observation: Observation) -> GameAction:
        self.last_call_committed_ready_response = False
        if observation.mission_complete or observation.stunned or observation.busy or observation.time_expired:
            if self._live_async and self._response_ready_uncommitted and not self._thinking_in_progress:
                live_async.cancel_ready_decision(self, observation, live_async.cancel_reason(observation))
            self._cancel_ready_response("busy_stunned_or_mission_complete_at_hand_over")
            self.status = "IDLE"
            self.last_action = GameAction(ActionType.WAIT)
            self.last_reason = "Mission complete, stunned, or busy."
            if self._executor.in_progress:
                _, boundary = self._executor.continue_(observation)
                if boundary is not None:
                    self._finish_action_lifecycle(observation, boundary)
            return self.last_action

        self._memory_tracker.update_from_observation(observation)

        # M5D.2: an accepted action already in progress (currently only
        # MOVE spans multiple ticks) keeps executing without requesting a
        # new strategic decision - see LLMController.get_action's identical
        # logic and action_lifecycle.py for the full rationale.
        if self._executor.in_progress:
            if (not self._live_async and self._response_ready_uncommitted and not self._thinking_in_progress
                    and self._executor.fired_at_start()):
                return self._hand_over(self.last_action)  # see LLMController.get_action
            action, boundary = self._executor.continue_(observation)
            self.last_action = action
            if boundary is None:
                return self._hand_over(action)
            self._finish_action_lifecycle(observation, boundary)
            # Boundary just reached - fall through so a new decision can be
            # considered this same tick instead of wasting one doing nothing.

        observation_payload = serialize_observation(observation, live_state=self.live_state_representation,
                                                    live_workshop=self.live_time_facts)
        payload_for_digest = dict(observation_payload)
        payload_for_digest.pop("time", None)
        observation_digest = hashlib.sha256(json.dumps(payload_for_digest, sort_keys=True).encode("utf-8")).hexdigest()

        # Live async: see LLMController.get_action.
        if self._live_async and self._live_pending is not None and self._response_ready_uncommitted and not self._thinking_in_progress:
            return live_async.commit_ready_decision(self, observation, observation_payload["legal_actions"], observation_digest)

        # simulated-clock lab: pacing on the game clock, as Native (LLMControllerConfig.game_clock_pacing)
        game_now = -observation.mission_time_remaining if getattr(self, "game_clock_pacing", False) else None
        if self._live_async:
            live_async.idle_redecide(self)
        elif game_now is not None:
            live_async.idle_redecide(self, now=game_now)
        if self._last_observation_digest == observation_digest:
            if self._thinking_in_progress:
                return self._inflight_action()
            return self._hand_over(self.last_action)

        # the game-clock throttle keeps its own deadline (game seconds, from -inf)
        now = time.monotonic() if game_now is None else game_now
        if now < (self._next_decision_allowed_at if game_now is None
                  else getattr(self, "_next_game_decision_allowed_at", float("-inf"))):
            if self._thinking_in_progress:
                return self._inflight_action()
            return self._hand_over(self.last_action)

        # M6G.3 (HP09): see LLMController.get_action - hand the ready
        # decision over instead of replacing it; digest deliberately not stored.
        if self.ready_response_guard and self._response_ready_uncommitted and not self._thinking_in_progress:
            return self._hand_over(self.last_action)

        self._last_observation_digest = observation_digest
        if game_now is None:
            self._next_decision_allowed_at = now + self._decision_interval_seconds
        else:
            self._next_game_decision_allowed_at = now + self._decision_interval_seconds
        self._decision_seq += 1

        if self._thinking_in_progress:
            return self._inflight_action()

        if self._response_ready_uncommitted:
            # guard off only (pre-M6G.3 lifecycle): see LLMController.get_action
            self._hand_over(self.last_action)

        self._thinking_in_progress = True
        self.status = "THINKING"
        # M5D.2: incremented exactly once per genuinely-spawned strategic
        # request (spec section 11) - see LLMController.get_action.
        self.decisions_requested += 1
        if self._live_async:
            live_async.on_request_launched(self, observation, observation_digest)
        thread = threading.Thread(target=self._request_action_background, args=(observation, observation_payload))
        thread.daemon = True
        thread.start()
        self._thinking_thread = thread
        thread.join(timeout=0.05)
        if self._response_ready_uncommitted and not self._thinking_in_progress:
            if self._live_async and self._live_pending is not None:
                return live_async.commit_ready_decision(self, observation, observation_payload["legal_actions"], observation_digest)
            # answered within the join: this very call hands the decision over
            return self._hand_over(self._inflight_action(handing_over=True))
        return self._inflight_action()

    def _inflight_action(self, *, handing_over: bool = False) -> GameAction:
        # Evaluation mode only: never a provisional strategic action while
        # thinking (spec section 20) - just whatever the last accepted
        # decision was, same rule as Native's evaluation-mode branch.
        if self._live_async:
            # live async: a thinking actor with no active action stands still
            return GameAction(ActionType.WAIT)
        action = self.last_action
        # M6G.3 (HP09): see LLMController._inflight_action - a decision
        # published a moment ago is handed over by the next call, never
        # returned here as the in-flight action.
        if not handing_over and self.ready_response_guard and self._response_ready_uncommitted:
            return GameAction(ActionType.WAIT)
        return action

    def _request_action_background(self, observation: Observation, observation_payload: dict[str, Any]) -> None:
        try:
            decision = self._request_action(observation, observation_payload)
        except Exception:
            decision = GameAction(ActionType.WAIT)
        with self._thinking_lock:
            if decision is None:
                decision = GameAction(ActionType.WAIT)
            if self._live_async and self._live_pending is None:
                self._live_pending = live_async.make_pending(
                    self, decision_id=self._last_decision_id, action_dict={"action": "wait"}, valid=False,
                    decision_game_time=float(self._decision_seq) * self._decision_interval_seconds,
                )
            # M6G.3: flag published before the decision and before the
            # in-flight flag clears - see LLMController._request_action_background.
            self.response_ready_at = time.monotonic()
            self._response_ready_uncommitted = True
            self.last_action = decision
            self.status = "EXECUTING" if decision.type != ActionType.WAIT else "IDLE"
            self._thinking_in_progress = False

    def decide(self, observation: Observation) -> AgentDecision:
        action = self.get_action(observation)
        return AgentDecision(
            action=action,
            runtime=self.runtime_name,
            decision_id=self._last_decision_id or "none",
            action_source=self._last_action_source,
            metadata={"provider": self.provider_name, "model": self.model_name, "runtime_version": self.runtime_version},
        )

    # -- the actual decision: at most one Strands call, one repair call --

    def _request_action(self, observation: Observation, observation_payload: dict[str, Any]) -> GameAction:
        # Bugfix mission time budget V2: time budget + action costs, live
        # workshop payloads only (set by LiveGame); built here, on the request
        # thread, because the MOVE ETAs need path searches.
        self._last_request_facts = None
        if self.live_time_facts:
            observation_payload = annotate_live_payload(observation, observation_payload)
            self._last_request_facts = {"time_budget": observation_payload["time_budget"],
                                        "action_costs": observation_payload["action_costs"]}
        legal_actions = observation_payload["legal_actions"]
        decision_tick = self._decision_seq
        decision_game_time = float(self._decision_seq) * self._decision_interval_seconds
        observation_hash = hashlib.sha256(json.dumps(observation_payload, sort_keys=True).encode("utf-8")).hexdigest()

        model_payload = compact_observation_for_model(observation_payload)
        model_payload["strategy_memory"] = self.memory.to_compact_dict()
        # M6B.2 spec sections 15-21 (RP10 fix): applied once here, exactly
        # where LLMController._request_action applies its own
        # `config.model_payload_transform` (same input shape - the compact,
        # already-strategy_memory-augmented model-facing payload; same
        # opportunity - once per decision, before either the initial or
        # the repair call, since `_invoke_strands` below reuses this same
        # `model_payload` object for both). Default `None` leaves every
        # existing caller byte-identical (spec section 19).
        if self._model_payload_transform is not None:
            model_payload = self._model_payload_transform(model_payload)
        observation_size = len(json.dumps(observation_payload, sort_keys=True))
        model_payload_size = len(json.dumps(model_payload, sort_keys=True, **self._json_kwargs()))
        serialized_legal_actions_chars = len(json.dumps(legal_actions, sort_keys=True))

        decision_id = str(uuid.uuid4())
        self._last_decision_id = decision_id

        initial = self._invoke_strands(model_payload, repair_instruction=None)
        self.calls += 1
        initial_validation = validate_model_output(observation, initial.proposal, legal_actions)
        initial_error_type = initial.error_type or initial_validation.error_type
        self._emit_provider_call(
            decision_id=decision_id,
            decision_number=decision_tick,
            call_index=1,
            call_type="initial",
            latency_ms=initial.latency_ms,
            error_type=initial_error_type,
            error_message=initial.error_message or initial_validation.error_message,
            input_tokens=initial.input_tokens,
            output_tokens=initial.output_tokens,
        )

        repair_attempted = False
        repair_invocation: _StrandsInvocation | None = None
        repair_validation: ModelOutputValidation | None = None
        repair_context_chars: int | None = None
        final_validation = initial_validation

        if not initial_validation.valid and self.max_retries > 0:
            repair_attempted = True
            self.repair_count += 1
            # M5D.5: identical repair contract to LLMController - same
            # shared function, same echoed previous proposal/near_object_id/
            # legal_action_count - so both runtimes get semantically
            # equivalent recovery inputs (spec sections 15-17), not two
            # independently-drifting implementations.
            #
            # M6B real bug found and fixed live: this call never passed
            # `legal_trap_types` at all, unlike `LLMController`'s own call
            # (see its M5F.3B.1 comment - found live there for the exact
            # same reason: a place_trap repair with no legal-values hint
            # lands on the same hallucinated "stun" every time). That
            # fix was never ported here, so despite the comment above
            # claiming an "identical repair contract", Strands' repair
            # instruction never actually told the model which trap types
            # it legitimately holds - a retrospective analysis of M6A.2's
            # own telemetry found Strands attempted place_trap 99 times
            # across that milestone's 4 episodes and succeeded ZERO of
            # those times (Native: 11/11), with "stun"/"stun_trap" as the
            # hallucinated trap_type in effectively every failure. This is
            # the single largest concrete mechanism identified for that
            # gap.
            legal_trap_types = None
            legal_place_trap_targets = None
            if (initial.proposal or {}).get("action") == "place_trap":
                # M6B.2 spec sections 9, 13-14, 24-25 (RP12_LEGAL_TARGET_HINTS):
                # the exact same shared helper LLMController._attempt_repair
                # calls - this is the parity fix RP12 exists to keep honest
                # (one definition, never two independently-drifting copies).
                legal_trap_types, legal_place_trap_targets = place_trap_repair_hints(observation, legal_actions)
            reason = _repair_instruction(
                initial_validation.error_type,
                initial_validation.error_message,
                proposed_action=(initial.proposal or {}).get("action"),
                proposed_target_id=(initial.proposal or {}).get("target_id"),
                near_object_id=observation.nearby_object_id,
                legal_action_count=len(legal_actions),
                legal_trap_types=legal_trap_types,
                legal_place_trap_targets=legal_place_trap_targets,
            )
            repair_context_chars = len(reason)
            repair_invocation = self._invoke_strands(model_payload, repair_instruction=reason)
            self.calls += 1
            repair_validation = validate_model_output(observation, repair_invocation.proposal, legal_actions)
            self._emit_provider_call(
                decision_id=decision_id,
                decision_number=decision_tick,
                call_index=2,
                call_type="repair",
                latency_ms=repair_invocation.latency_ms,
                error_type=(repair_invocation.error_type or (None if repair_validation.valid else repair_validation.error_type)),
                error_message=(repair_invocation.error_message or (None if repair_validation.valid else repair_validation.error_message)),
                input_tokens=repair_invocation.input_tokens,
                output_tokens=repair_invocation.output_tokens,
            )
            if repair_validation.valid:
                final_validation = repair_validation

        final_model_action_valid = final_validation.valid
        fallback_used = not final_model_action_valid
        action_source = "llm" if final_model_action_valid else "fallback"
        executed_action_dict = final_validation.validated_action if final_model_action_valid else {"action": "wait"}

        # M5D.2: hand the accepted decision to the shared ActionExecutor
        # rather than using ModelOutputValidation's own one-shot GameAction
        # directly - same fix as LLMController (see action_lifecycle.py).
        if final_model_action_valid and self._live_async:
            # live async: started at commit on the game thread - see LLMController
            self._live_pending = live_async.make_pending(
                self, decision_id=decision_id, action_dict=executed_action_dict, valid=True,
                decision_game_time=decision_game_time,
            )
            executed_action = GameAction(ActionType.WAIT)
        elif final_model_action_valid:
            normalized_action = executed_action_dict.get("action", "wait")
            target_id = executed_action_dict.get("target_id")
            trap_type = executed_action_dict.get("trap_type")
            executed_action, start_boundary, _action_id = self._executor.start(
                observation, decision_id, normalized_action, target_id, trap_type, decision_game_time,
                code=executed_action_dict.get("code"),  # live riddle, as Native
            )
            if start_boundary is not None:
                self._finish_action_lifecycle(observation, start_boundary)
        else:
            executed_action = GameAction(ActionType.WAIT)

        if final_model_action_valid:
            self.last_reason = executed_action_dict.get("reason") or ("repaired model action" if repair_attempted else "validated model action")
            final_error_type = None
            final_error_message = None
            same_invalid_action_signature_count = None
        else:
            self.invalid_calls += 1
            self.fallback_count += 1
            self.status = "ERROR"
            self.last_reason = "invalid model output rejected"
            if repair_attempted and repair_validation is not None:
                final_error_type = repair_validation.error_type
                final_error_message = repair_validation.error_message
                same_invalid_action_signature_count = self._record_invalid_signature(repair_invocation.proposal if repair_invocation else None)
            else:
                final_error_type = initial_error_type
                final_error_message = initial_validation.error_message
                same_invalid_action_signature_count = self._record_invalid_signature(initial.proposal)
            # M6A.2 spec section 12 of the original M5F.3B.1 fix, mirrored
            # here: a repair attempt can itself fail via a genuine
            # provider exception - that's still a provider/infrastructure
            # failure, not ordinary model-behavior validation noise.
            #
            # Deliberately classifies from the INVOCATION's own error_type
            # (`initial.error_type`/`repair_invocation.error_type`), never
            # from `final_error_type` above - a real bug found and fixed
            # while first writing this milestone's own test: unlike
            # Native (which short-circuits with an early, separate
            # `provider_failure_count` increment the moment the raw call
            # itself raises - see `LLMController._request_action`'s outer
            # `except Exception` - and only reaches this branch at all for
            # a content-level rejection), Strands' `_invoke_strands` never
            # raises up to here - a call-level failure and a content-level
            # rejection both flow through the exact same
            # validate-then-repair pipeline. `final_error_type` prioritizes
            # `repair_validation.error_type` - the VALIDATION classification
            # of whatever (possibly empty) proposal the repair produced -
            # which is the right choice for telemetry's own "what got
            # rejected and why" field, but would misclassify a genuine
            # repair-call-level exception as validation noise here, since
            # `validate_model_output` never emits "strands_runtime_error"
            # itself. Strands' own error taxonomy also differs from
            # Native's (it never produces the literal string
            # "provider_error" - a real SDK/network exception comes back
            # as "strands_runtime_error", and the adapter's own watchdog
            # reports "watchdog_timeout"): both are genuine call-level
            # failures, not model-output issues like "invalid_trap_type"/
            # "missing_target"/"no_tool_selected".
            final_invocation_error_type = (
                repair_invocation.error_type if repair_attempted and repair_invocation is not None else initial.error_type
            )
            if final_invocation_error_type in ("provider_error", "strands_runtime_error", "watchdog_timeout"):
                self.provider_failure_count += 1
            else:
                self.validation_failure_count += 1
            if repair_attempted:
                self.repair_failure_count += 1

        def _sum_optional(*values: int | float | None) -> int | float | None:
            present = [v for v in values if v is not None]
            return sum(present) if present else None

        total_latency_ms = _sum_optional(initial.latency_ms, repair_invocation.latency_ms if repair_invocation else None) or 0.0
        total_input_tokens = _sum_optional(initial.input_tokens, repair_invocation.input_tokens if repair_invocation else None)
        total_output_tokens = _sum_optional(initial.output_tokens, repair_invocation.output_tokens if repair_invocation else None)
        total_tool_calls = initial.tool_call_count + (repair_invocation.tool_call_count if repair_invocation else 0)

        self._emit_telemetry(
            decision_tick=decision_tick,
            decision_game_time=decision_game_time,
            observation_hash=observation_hash,
            observation_payload=observation_payload,
            legal_actions=legal_actions,
            raw_response=initial.proposal,
            executed_action_dict=executed_action_dict,
            final_model_action_valid=final_model_action_valid,
            initial_validation=initial_validation,
            initial_error_type=initial_error_type,
            repair_attempted=repair_attempted,
            repair_invocation=repair_invocation,
            repair_validation=repair_validation,
            fallback_used=fallback_used,
            action_source=action_source,
            latency_ms=total_latency_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            error_type=final_error_type,
            error_message=final_error_message,
            observation_size=observation_size,
            model_payload_size=model_payload_size,
            decision_id=decision_id,
            initial_invocation=initial,
            total_tool_calls=total_tool_calls,
            serialized_legal_actions_chars=serialized_legal_actions_chars,
            repair_context_chars=repair_context_chars,
            same_invalid_action_signature_count=same_invalid_action_signature_count,
        )

        selected_action = executed_action_dict.get("action", "wait") if final_model_action_valid else "wait"
        target_for_memory = executed_action_dict.get("target_id") if final_model_action_valid else None
        self.memory.record_decision(
            game_time=decision_game_time,
            decision_id=decision_id,
            selected_action=selected_action,
            target_id=target_for_memory,
            valid=final_model_action_valid,
            action_source=action_source,
            observation_hash=observation_hash,
        )
        self._last_action_source = action_source
        if final_model_action_valid:
            self._memory_tracker.set_pending_decision(decision_id, selected_action, target_for_memory, observation.nearby_object_id)
        else:
            self._memory_tracker.clear_pending_decision()

        self.decisions_completed += 1
        return executed_action

    def _json_kwargs(self) -> dict[str, Any]:
        """Live play (set by LiveGame, src/llm/live_payload.py): no spaces in the payload JSON."""
        return {"separators": (",", ":")} if getattr(self, "compact_payload_json", False) else {}

    def _invoke_strands(self, model_payload: dict[str, Any], repair_instruction: str | None) -> _StrandsInvocation:
        capture: list[dict[str, Any]] = []
        tool = self._make_tool(capture)
        prompt_payload = dict(model_payload)
        if repair_instruction:
            prompt_payload["repair_instruction"] = repair_instruction
            prompt_payload["repair_prompt_version"] = REPAIR_PROMPT_VERSION
        prompt_json = json.dumps(prompt_payload, sort_keys=True, **self._json_kwargs())

        start = time.monotonic()
        agent = self._agent_factory(self._effective_system_prompt, tool)

        # M5E.1.1 fix (strands-adapter-v2): M5E.1 found - and M5E.1.1's
        # controlled localhost test/CHANGELOG entry proves at the source
        # level - that Strands' own `llm_timeout` does not bound total
        # call duration the way Native's `OllamaClient` does. Strands'
        # real model call streams (`"stream": True` in
        # strands.models.ollama.OllamaModel.format_request), and a plain
        # float `timeout` passed to `ollama.AsyncClient`/httpx bounds each
        # individual read operation, not the whole request - a response
        # that keeps trickling small chunks can run far past the
        # configured timeout without ever erroring. This watchdog closes
        # that gap at the adapter level: run the real (unmodified) call on
        # a background thread and stop *waiting* on it once
        # `self._llm_timeout` elapses, exactly mirroring Native's ceiling.
        # The orphaned thread is never forcibly killed (Python cannot do
        # that safely) - it simply stops blocking this decision, matching
        # spec section 26's "do not hide recovery behind automatic
        # retries": the timeout is classified and reported, not absorbed.
        call_result: dict[str, Any] = {}

        def _run_agent_call() -> None:
            try:
                call_result["result"] = agent(prompt_json, limits={"turns": DECISION_TURN_LIMIT})
            except Exception as exc:  # pragma: no cover - defensive safety boundary
                call_result["exception"] = exc

        thread = threading.Thread(target=_run_agent_call, daemon=True)
        thread.start()
        thread.join(timeout=self._llm_timeout)
        latency_ms = (time.monotonic() - start) * 1000.0

        if thread.is_alive():
            return _StrandsInvocation(
                proposal={},
                proposal_count=0,
                stop_reason=None,
                cycle_count=0,
                tool_call_count=0,
                input_tokens=None,
                output_tokens=None,
                latency_ms=latency_ms,
                # M5E.2 spec section 15: `watchdog_terminations` must be
                # countable separately from a real, provider-reported
                # `provider_timeouts` (Native's own caught-exception path) -
                # a distinct error_type carrying "timed out" in the message
                # still classifies as the same provider_timeout *category*
                # (see classify_provider_call_error) for existing
                # aggregate reporting, while remaining distinguishable by
                # error_type for this adapter-level watchdog specifically.
                error_type="watchdog_timeout",
                error_message="timed out (adapter watchdog - underlying call abandoned, not provider-reported)",
            )

        if "exception" in call_result:
            exc = call_result["exception"]
            return _StrandsInvocation(
                proposal={},
                proposal_count=0,
                stop_reason=None,
                cycle_count=0,
                tool_call_count=0,
                input_tokens=None,
                output_tokens=None,
                latency_ms=latency_ms,
                error_type="strands_runtime_error",
                error_message=str(exc),
            )

        result = call_result["result"]
        usage = result.metrics.accumulated_usage or {}
        tool_call_count = sum(m.call_count for m in result.metrics.tool_metrics.values())
        # spec section 29: the SDK's own loop can call the tool more than
        # once with different proposals in one invocation. Only the first
        # is ever trusted; the rest are recorded, never executed.
        proposal = capture[0] if capture else {}
        error_type = None if capture else "no_tool_selected"
        error_message = None if capture else "agent completed without calling choose_action"
        return _StrandsInvocation(
            proposal=proposal,
            proposal_count=len(capture),
            stop_reason=result.stop_reason,
            cycle_count=result.metrics.cycle_count,
            tool_call_count=tool_call_count,
            # a cache write is billed like input (live prompt caching); cache reads
            # have no field in this record and are not counted here
            input_tokens=None if usage.get("inputTokens") is None
            else usage.get("inputTokens") + (usage.get("cacheWriteInputTokens") or 0),
            output_tokens=usage.get("outputTokens"),
            latency_ms=latency_ms,
            error_type=error_type,
            error_message=error_message,
        )

    def _emit_provider_call(
        self,
        *,
        decision_id: str,
        decision_number: int,
        call_index: int,
        call_type: str,
        latency_ms: float,
        error_type: str | None,
        error_message: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        """M5E.1 spec section 9: one record per real Strands `Agent()`
        invocation - initial or repair - mirroring
        `LLMController._emit_provider_call`. `_invoke_strands` only
        measures total latency around the call, not separate start/end
        wall-clock instants, so both timestamps are derived from `now`
        minus that latency - close enough for diagnostic sequencing,
        never used as an authoritative duration (latency_ms already is)."""
        if not self.telemetry:
            return
        from datetime import datetime, timedelta, timezone

        end = datetime.now(timezone.utc)
        start = end - timedelta(milliseconds=max(latency_ms, 0.0))
        call_error_type = classify_provider_call_error(error_type, error_message)
        self.telemetry.emit_provider_call(
            episode_id=self.episode_id,
            actor_id=self.actor_id,
            decision_id=decision_id,
            decision_number=decision_number,
            call_index=call_index,
            call_type=call_type,
            runtime=self.runtime_name,
            provider=self.provider_name,
            model=self.model_name,
            condition=self.condition,
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            latency_ms=round(latency_ms, 2),
            success=(call_error_type is None),
            provider_error_type=call_error_type,
            provider_error_message_normalized=normalize_provider_error_message(error_message),
            timeout=(call_error_type == "provider_timeout"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _emit_telemetry(self, **kw: Any) -> None:
        if not self.telemetry:
            return
        memory_kw = self.memory.to_telemetry_dict()
        initial: _StrandsInvocation = kw["initial_invocation"]
        repair: _StrandsInvocation | None = kw["repair_invocation"]
        repair_validation: ModelOutputValidation | None = kw["repair_validation"]

        runtime_metadata = {
            "runtime_invocation_count": 2 if kw["repair_attempted"] else 1,
            "model_call_count": 2 if kw["repair_attempted"] else 1,
            "tool_call_count": kw["total_tool_calls"],
            "tool_name": "choose_action",
            "initial_stop_reason": initial.stop_reason,
            "initial_cycle_count": initial.cycle_count,
            "initial_proposal_count": initial.proposal_count,
            "initial_runtime_error_type": initial.error_type,
            "decision_turn_limit": DECISION_TURN_LIMIT,
        }
        if repair is not None:
            runtime_metadata.update(
                {
                    "repair_stop_reason": repair.stop_reason,
                    "repair_cycle_count": repair.cycle_count,
                    "repair_proposal_count": repair.proposal_count,
                    "repair_runtime_error_type": repair.error_type,
                }
            )

        self.telemetry.emit_llm_decision(
            episode_id=self.episode_id,
            actor_id=self.actor_id,
            tick=kw["decision_tick"],
            game_time=kw["decision_game_time"],
            provider=self.provider_name,
            model=self.model_name,
            prompt_version=PROMPT_VERSION,
            observation_schema_version=OBSERVATION_SCHEMA_VERSION,
            action_schema_version="action-v1",
            observation_hash=kw["observation_hash"],
            observation=kw["observation_payload"],
            legal_actions=kw["legal_actions"],
            raw_response=kw["raw_response"],
            validated_action=(kw["executed_action_dict"] if kw["final_model_action_valid"] else None),
            valid=kw["final_model_action_valid"],
            repair_attempted=kw["repair_attempted"],
            repair_raw_response=(repair.proposal if repair is not None else None),
            repair_valid=(repair_validation.valid if repair_validation is not None else None),
            repair_error_type=(repair_validation.error_type if repair_validation is not None else None),
            initial_model_output_valid=kw["initial_validation"].valid,
            initial_error_type=kw["initial_error_type"],
            final_model_action_valid=kw["final_model_action_valid"],
            fallback_used=kw["fallback_used"],
            fallback_action=({"action": "wait"} if kw["fallback_used"] else None),
            executed_action=kw["executed_action_dict"],
            executed_action_valid=True,
            action_source=kw["action_source"],
            latency_ms=kw["latency_ms"],
            input_tokens=kw["input_tokens"],
            output_tokens=kw["output_tokens"],
            error_type=kw["error_type"],
            error_message=kw["error_message"],
            observation_size_chars=kw["observation_size"],
            model_payload_size_chars=kw["model_payload_size"],
            legal_actions_count=len(kw["legal_actions"]),
            decision_id=kw["decision_id"],
            # M5D.5: recovery telemetry (spec section 18) - identical
            # shape/semantics to LLMController's, so the two runtimes are
            # directly comparable rather than each inventing their own.
            initial_proposal=kw["raw_response"],
            repair_context_version=(REPAIR_PROMPT_VERSION if kw["repair_attempted"] else None),
            repair_legal_action_count=(len(kw["legal_actions"]) if kw["repair_attempted"] else None),
            repair_near_object_id=(
                (kw["observation_payload"].get("self", {}).get("near_object_id")) if kw["repair_attempted"] else None
            ),
            recovery_outcome=_recovery_outcome(
                kw["initial_validation"].valid,
                kw["repair_attempted"],
                repair_validation.valid if repair_validation is not None else None,
            ),
            representation_variant=STRANDS_TOOL_REPRESENTATION,
            agent_runtime=self.runtime_name,
            agent_runtime_version=self.runtime_version,
            agent_sdk=AGENT_SDK_NAME,
            agent_sdk_version=_agent_sdk_version(),
            runtime_metadata=runtime_metadata,
            decision_number=kw["decision_tick"],
            condition=self.condition,
            personality_profile=self.personality_profile,
            serialized_legal_actions_chars=kw.get("serialized_legal_actions_chars"),
            repair_context_chars=kw.get("repair_context_chars"),
            same_invalid_action_signature_count=kw.get("same_invalid_action_signature_count"),
            tool_schema_chars=len(TOOL_DESCRIPTION),
            total_tokens=(
                (kw["input_tokens"] + kw["output_tokens"])
                if (kw["input_tokens"] or kw["output_tokens"])
                else None
            ),
            **memory_kw,
        )
