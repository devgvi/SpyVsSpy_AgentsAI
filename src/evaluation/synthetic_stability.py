"""M5E.1 spec sections 14-19: synthetic sequential-call harness.

"Game-light" (spec section 14): builds one real `GameState`/`Observation`
using the exact same primitives `Game.__init__` does (`load_level`,
`load_items`, `GameState.new_game`, `build_observation`), but never runs
`Game`'s pygame loop, `ActionExecutor` multi-tick movement, or any
full-episode simulation. The point is to isolate the *provider/runtime*
layer from the game engine (never the other way around) - so this module
drives the real, unmodified `native_tool_like`/`strands` adapters (spec
section 15: no provider shortcut, no bypass) through their normal async
decision lifecycle with a small polling loop, the same contract
`Game._step`/`EpisodeRunner` already rely on
(`get_action()` returns immediately; a background thread resolves the
real model call; poll until `decisions_completed` advances).

One deliberate simplification, stated here rather than left implicit:
between every synthetic decision this harness resets the adapter's
`ActionExecutor` to a fresh, not-in-progress one. Without that, a MOVE
decision would leave the *real* multi-tick action lifecycle "in progress"
against a game engine this harness never advances, silently starving
every later decision instead of exercising the runtime again - and the
executor's continue/boundary logic is exactly the game-engine-side
concern spec section 14 says to isolate away from, not something this
module is trying to test.

Three patterns (spec section 14):
  S1 constant     - the observation changes by the smallest possible
                    amount (a tiny position nudge) between decisions, so
                    each is a genuinely new decision to make (the
                    controller's own digest/throttle logic would
                    otherwise just replay the last decision) without any
                    real growth in prompt/memory size. Answers: "does
                    repeated sequential use alone destabilize the stack?"
  S2 growing      - the same tiny nudge, plus synthetic `AgentMemory`
                    records (interactions/failures/events) injected
                    directly, in controlled, increasing volume before
                    each decision - the "decision 1 small memory ->
                    decision 15 larger" curve spec section 18 asks for.
  S3 active-like  - both actors share a room from spawn (true by default
                    on this level - confirmed in M5E), so
                    visible_other_actor/opponent are populated for free;
                    this pattern additionally seeds one object's engine
                    state directly (`state.object_states[...]["trap"]`/
                    `["detected_by"]`, `current_interaction_target`) so a
                    detected enemy trap and its DISARM legal action exist,
                    reproducing M5E's active-opponent context shape
                    without requiring live simulation (spec section 19
                    explicitly does not require one).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src import constants
from src.agents.action_lifecycle import ActionExecutor
from src.agents.native_agent import NativeToolLikeAgentAdapter
from src.agents.strands_agent import StrandsAgentAdapter
from src.items import load_items
from src.level import load_level
from src.llm.clients.ollama_client import OllamaClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig
from src.observation import Observation, build_observation
from src.state import GameState

PATTERN_S1_CONSTANT = "S1_constant"
PATTERN_S2_GROWING = "S2_growing"
PATTERN_S3_ACTIVE_LIKE = "S3_active_like"
PATTERNS = (PATTERN_S1_CONSTANT, PATTERN_S2_GROWING, PATTERN_S3_ACTIVE_LIKE)


def _build_base_state(seed: int, communication_enabled: bool = False) -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=seed, communication_enabled=communication_enabled)


def _other_actor_id(state: GameState, actor_id: str) -> str:
    return next(aid for aid in state.actors if aid != actor_id)


def _seed_detected_enemy_trap(state: GameState, actor_id: str, other_id: str) -> str | None:
    """S3 only: directly seeds one trappable object's engine state as an
    enemy-owned, already-detected trap, and points `actor_id`'s
    `current_interaction_target` at it - the minimum engine state needed
    for `_trap_knowledge`/`enumerate_legal_actions` to legitimately offer
    DISARM, without running the real place/inspect action sequence that
    would normally produce it. Returns the object id, or None if this
    level has no trappable object (would make S3 silently equivalent to
    S1/S2, so callers should treat None as a setup failure)."""
    target = next((obj for obj in state.level.objects if obj.trappable), None)
    if target is None:
        return None
    state._object_entry(target.id)["trap"] = {"type": "bomb", "owner_id": other_id}
    state._object_entry(target.id)["detected_by"][actor_id] = "bomb"
    state.current_interaction_target[actor_id] = target.id
    return target.id


def _nudged_observation(state: GameState, actor_id: str, decision_index: int) -> Observation:
    """A position nudge tiny enough to add zero real complexity, but
    large enough to change `serialize_observation`'s JSON (and therefore
    the controller's own decision digest) so every call is treated as a
    genuinely new decision rather than replaying the last one."""
    actor = state.actors[actor_id]
    actor.x += 0.01 if decision_index % 2 == 0 else -0.01
    return build_observation(state, actor_id)


def _grow_memory(memory: Any, decision_index: int) -> None:
    """S2 only: linear, controlled memory growth - one additional
    synthetic interaction/failure/event per decision, independent of
    whatever the model actually decided (so the growth curve is exactly
    reproducible regardless of live model behavior)."""
    game_time = float(decision_index)
    memory.record_interaction(
        game_time=game_time,
        action="search",
        target_id=f"synthetic_object_{decision_index}",
        result="found_item",
        detail=f"synthetic search result #{decision_index} for growth control",
    )
    memory.record_failure(
        game_time=game_time,
        action="move",
        target_id=f"synthetic_unreachable_{decision_index}",
        reason="path_unreachable",
        detail="synthetic failure injected for S2 growth control",
    )
    memory.record_event(
        game_time=game_time,
        event_type="SYNTHETIC_EVENT",
        target_id=f"synthetic_object_{decision_index}",
        detail="synthetic event injected for S2 growth control",
    )
    memory.record_visited_room(f"synthetic_room_{decision_index % 6}")


@dataclass
class SyntheticRunResult:
    pattern: str
    runtime: str
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "runtime": self.runtime, "decisions": self.decisions}


def _make_adapter(
    runtime: str,
    actor_id: str,
    provider: str,
    model: str,
    base_url: str,
    timeout: float,
    telemetry: TelemetryLogger,
    episode_id: str,
    condition: str,
    model_client: Any = None,
    agent_factory: Callable[[str, Any], Any] | None = None,
    num_ctx: int | None = None,
    keep_alive: str | None = None,
    llm_region: str | None = None,
    personality_profile: str | None = None,
    communication_enabled: bool = False,
):
    if runtime == "native_tool_like":
        client = model_client or OllamaClient(model_name=model, base_url=base_url, timeout=timeout, num_ctx=num_ctx, keep_alive=keep_alive)
        return NativeToolLikeAgentAdapter(
            actor_id,
            model_client=client,
            config=LLMControllerConfig(
                condition=condition, personality_profile=personality_profile, communication_enabled=communication_enabled
            ),
            telemetry=telemetry,
            episode_id=episode_id,
            provider_name=provider,
            model_name=model,
            execution_mode="evaluation",
        )
    if runtime == "strands":
        return StrandsAgentAdapter(
            actor_id,
            telemetry=telemetry,
            episode_id=episode_id,
            provider_name=provider,
            model_name=model,
            execution_mode="evaluation",
            llm_base_url=base_url,
            llm_timeout=timeout,
            condition=condition,
            agent_factory=agent_factory,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            llm_region=llm_region,
        )
    raise ValueError(f"synthetic harness supports native_tool_like/strands only, got {runtime!r}")


def _reset_executor(adapter: Any) -> None:
    fresh = ActionExecutor()
    if hasattr(adapter, "controller"):
        adapter.controller._executor = fresh
    else:
        adapter._executor = fresh


def run_synthetic_pattern(
    *,
    pattern: str,
    runtime: str,
    n_decisions: int,
    provider: str = "ollama",
    model: str = "gemma4:latest",
    base_url: str = "http://localhost:11434",
    timeout: float = 60.0,
    telemetry: TelemetryLogger | None = None,
    episode_id: str | None = None,
    seed: int = 901,
    actor_id: str = constants.WHITE_SPY_ID,
    per_decision_wall_timeout: float = 180.0,
    poll_interval: float = 0.05,
    on_decision: Callable[[int, dict[str, Any]], None] | None = None,
    model_client: Any = None,
    agent_factory: Callable[[str, Any], Any] | None = None,
    num_ctx: int | None = None,
    keep_alive: str | None = None,
    llm_region: str | None = None,
    personality_profile: str | None = None,
    communication_enabled: bool = False,
    # M5F.2: orthogonal to `pattern` (never a fourth pattern name - this
    # only ever seeds one extra fact, the way S3's detected-trap seeding
    # does) - when set (and communication is enabled), one message from
    # the opponent is placed in `state.messages` before the first decision,
    # giving the model a real, concrete in-context reason to consider
    # SEND_MESSAGE (a genuine received message, never a change to the
    # neutral communication policy's own wording - spec section 51 only
    # forbids the latter).
    seed_incoming_message: str | None = None,
) -> SyntheticRunResult:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown synthetic pattern {pattern!r}; expected one of {PATTERNS}")
    episode_id = episode_id or f"synthetic-{pattern}-{runtime}-{seed}"
    telemetry = telemetry or TelemetryLogger()

    state = _build_base_state(seed, communication_enabled=communication_enabled)
    other_id = _other_actor_id(state, actor_id)
    detected_trap_object_id: str | None = None
    if pattern == PATTERN_S3_ACTIVE_LIKE:
        detected_trap_object_id = _seed_detected_enemy_trap(state, actor_id, other_id)
    if seed_incoming_message and communication_enabled:
        state.messages.append({
            "message_id": "seed-incoming-0",
            "sender_actor_id": other_id,
            "recipient_actor_id": actor_id,
            "game_time": state.elapsed_time,
            "text": seed_incoming_message,
        })

    adapter = _make_adapter(
        runtime, actor_id, provider, model, base_url, timeout, telemetry, episode_id,
        condition="synthetic_" + pattern, model_client=model_client, agent_factory=agent_factory,
        num_ctx=num_ctx, keep_alive=keep_alive, llm_region=llm_region, personality_profile=personality_profile,
        communication_enabled=communication_enabled,
    )

    result = SyntheticRunResult(pattern=pattern, runtime=runtime)
    for i in range(1, n_decisions + 1):
        _reset_executor(adapter)
        observation = _nudged_observation(state, actor_id, i)
        if pattern == PATTERN_S2_GROWING:
            _grow_memory(adapter.memory, i)

        target_completed = adapter.decisions_completed + 1
        start = time.monotonic()
        events_before = len(telemetry.events)
        while adapter.decisions_completed < target_completed:
            adapter.get_action(observation)
            if time.monotonic() - start > per_decision_wall_timeout:
                raise TimeoutError(
                    f"{runtime}/{pattern} decision {i} did not complete within {per_decision_wall_timeout}s"
                )
            time.sleep(poll_interval)

        decision_events = [e for e in telemetry.events[events_before:] if e.get("event_type") == "llm_decision"]
        record = decision_events[-1] if decision_events else {"decision_number": i, "final_model_action_valid": None}
        record = dict(record)
        record["synthetic_pattern"] = pattern
        record["synthetic_decision_index"] = i
        record["detected_trap_object_id"] = detected_trap_object_id
        result.decisions.append(record)
        if on_decision is not None:
            on_decision(i, record)

    return result
