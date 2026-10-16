"""M5D: EpisodeRunner - runs exactly one episode for one EpisodeSpec,
headless, and returns a structured EpisodeResult.

Wraps the existing `Game` loop (the same one the interactive game uses)
rather than reimplementing it: the M5D evaluation invariant is "same
controlled game conditions," which means going through the identical
engine/adapter/telemetry path a human session would - `Game` ->
`create_agent_runtime` -> `AgentAdapter` -> engine, unchanged.

Episode status (spec section 9):
    CREATED -> PREFLIGHT -> READY -> RUNNING -> COMPLETED
    or terminal: INCOMPATIBLE, FAILED, TIMED_OUT, ABORTED, DECISION_LIMIT_REACHED
A legitimate mission loss (the deadline passes without escaping) is still
COMPLETED - episode status and mission outcome are different axes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src import constants
from src.evaluation.compatibility import CompatibilityResult, CompatibilityStatus, check_compatibility
from src.evaluation.metrics import compute_metrics
from src.evaluation.spec import EpisodeSpec
from src.llm.telemetry import TelemetryLogger


class EpisodeStatus(str, Enum):
    COMPLETED = "COMPLETED"
    INCOMPATIBLE = "INCOMPATIBLE"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    ABORTED = "ABORTED"
    DECISION_LIMIT_REACHED = "DECISION_LIMIT_REACHED"


TRAP_EVENT_TYPES = {"TRAP_PLACED", "TRAP_TRIGGERED", "TRAP_DETECTED", "TRAP_NOT_DETECTED", "TRAP_DISARMED", "TRAP_REMOVED"}


def _extract_game_facts(state: Any, actor_id: str) -> dict[str, Any]:
    """Objective, engine-truth-derived facts for one actor.

    The *evaluator* (not the actor) is explicitly allowed to inspect world
    truth to verify reproducibility and derive metrics (spec section 11) -
    this reads GameState only after the episode has ended, and never feeds
    anything back into that actor's Observation. Hidden truth about the
    *other* actor or unfound items is deliberately not extracted here.
    """
    actor = state.actors[actor_id]
    required = list(state.items.required)
    have = [item for item in required if actor.inventory.has(item)]
    own_events = [e for e in state.event_log if e.get("actor_id") == actor_id]

    def count(event_type: str) -> int:
        return sum(1 for e in own_events if e["type"] == event_type)

    return {
        "mission_items_required": len(required),
        "mission_items_acquired": len(have),
        "mission_items_missing": [item for item in required if item not in have],
        "mission_success": bool(state.mission_complete.get(actor_id, False)),
        "escape_time": state.escape_elapsed.get(actor_id),
        "rooms_visited": sorted(actor.visited_rooms),
        "unique_rooms_visited": len(actor.visited_rooms),
        # M5D.2 spec section 19: diagnostic only, not a quality score - just
        # whether the actor ever stood somewhere other than its spawn room,
        # the exact question M5D.1 found no benchmark episode could answer
        # "yes" to.
        "left_starting_room": len(actor.visited_rooms) > 1,
        "object_searched_count": count("OBJECT_SEARCHED"),
        "object_inspected_count": count("OBJECT_INSPECTED"),
        "item_found_count": count("ITEM_FOUND"),
        "knowledge_learned_count": count("KNOWLEDGE_LEARNED"),
        "door_unlocked_count": count("DOOR_UNLOCKED"),
        "door_opened_count": count("DOOR_OPENED"),
        # M5D.2 spec section 19: authoritative signals of forward mission
        "mission_progress_events": (
            count("ITEM_FOUND") + count("DOOR_OPENED") + count("DOOR_UNLOCKED") + count("KNOWLEDGE_LEARNED")
        ),
        "traps_placed": count("TRAP_PLACED"),
        "traps_triggered": count("TRAP_TRIGGERED"),
        "enemy_traps_detected": count("TRAP_DETECTED"),
        "traps_disarmed": count("TRAP_DISARMED"),
        "stuns_suffered": count("ACTOR_STUNNED"),
        "opponent_encounters": count("OPPONENT_SPOTTED"),
        "deadline_remaining": state.mission_time_remaining,
        "mission_time_total": state.mission_time_total,
    }


def _extract_competitive_facts(state: Any, agent_id: str, opponent_id: str) -> dict[str, Any]:
    """M5E spec sections 18, 30-31: objective, authoritative race/
    competition facts that only make sense with *two* facts about the
    same event_log entry (who triggered which owner's trap, who escaped
    first) - never a derived "score," never speculative causal cost (spec
    section 33 - only what an authoritative event actually recorded).

    In this game's model, "mission complete" and "escape" are the same
    moment (`GameState._attempt_exit` sets `mission_complete` only at a
    successful exit - there is no separate "acquired all items but hasn't
    left yet" completion event) - `*_mission_completion_time` and
    `*_escape_time` are therefore identical here, reported under both
    names for spec section 30's vocabulary rather than inventing a second,
    different timestamp that doesn't exist. `time_in_same_room` and
    `last_seen_updates` (also listed in spec section 30) are not backed by
    any existing discrete event and are deliberately omitted rather than
    approximated (spec section 30's own "only implement metrics supported
    by authoritative events").
    """
    from src import constants as _constants

    agent_events = [e for e in state.event_log if e.get("actor_id") == agent_id]
    opponent_events = [e for e in state.event_log if e.get("actor_id") == opponent_id]

    def triggered_by_owner(events: list[dict[str, Any]], owner_id: str) -> int:
        return sum(1 for e in events if e["type"] == "TRAP_TRIGGERED" and e.get("trap_owner_id") == owner_id)

    def time_lost_to_traps(events: list[dict[str, Any]]) -> float:
        return sum(_constants.TRAP_STUN_SECONDS.get(e.get("trap_type"), 0.0) for e in events if e["type"] == "TRAP_TRIGGERED")

    agent_escape = state.escape_elapsed.get(agent_id)
    opponent_escape = state.escape_elapsed.get(opponent_id)
    both_escaped = bool(state.mission_complete.get(agent_id)) and bool(state.mission_complete.get(opponent_id))

    return {
        "agent_traps_placed": sum(1 for e in agent_events if e["type"] == "TRAP_PLACED"),
        "opponent_traps_placed": sum(1 for e in opponent_events if e["type"] == "TRAP_PLACED"),
        # "agent triggered an opponent-owned trap" = a TRAP_TRIGGERED event
        # where *agent* is the victim (actor_id) and *opponent* is the owner.
        "agent_triggered_opponent_traps": triggered_by_owner(agent_events, opponent_id),
        "opponent_triggered_agent_traps": triggered_by_owner(opponent_events, agent_id),
        "agent_stuns": sum(1 for e in agent_events if e["type"] == "ACTOR_STUNNED"),
        "opponent_stuns": sum(1 for e in opponent_events if e["type"] == "ACTOR_STUNNED"),
        "agent_time_lost_to_traps": time_lost_to_traps(agent_events),
        "opponent_time_lost_to_traps": time_lost_to_traps(opponent_events),
        "agent_mission_completion_time": agent_escape,
        "opponent_mission_completion_time": opponent_escape,
        "agent_escape_time": agent_escape,
        "opponent_escape_time": opponent_escape,
        "first_mission_complete_actor": state.finish_order[0] if state.finish_order else None,
        "first_escape_actor": state.finish_order[0] if state.finish_order else None,
        "both_escaped": both_escaped,
        "mission_completion_time_delta": (
            abs(agent_escape - opponent_escape) if agent_escape is not None and opponent_escape is not None else None
        ),
        "escape_time_delta": (
            abs(agent_escape - opponent_escape) if agent_escape is not None and opponent_escape is not None else None
        ),
    }


@dataclass
class EpisodeResult:
    experiment_id: str
    variant_id: str
    episode_id: str
    seed: int
    status: EpisodeStatus
    compatibility_status: CompatibilityStatus
    compatibility_reason: str | None = None
    mission_success: bool | None = None
    escape_success: bool | None = None
    winner: str | None = None
    game_time: float | None = None
    wall_time: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    decision_count: int = 0
    game_facts: dict[str, Any] | None = None
    # M5E: only populated when the episode ran with an active opponent
    # (VariantSpec.opponent_type == "scripted_active") - None for every
    # pre-M5E passive-opponent episode, preserving that shape exactly.
    opponent_game_facts: dict[str, Any] | None = None
    competitive_facts: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "status": self.status.value,
            "compatibility_status": self.compatibility_status.value,
            "compatibility_reason": self.compatibility_reason,
            "mission_success": self.mission_success,
            "escape_success": self.escape_success,
            "winner": self.winner,
            "game_time": self.game_time,
            "wall_time": self.wall_time,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "decision_count": self.decision_count,
            "metrics": self.metrics,
            "game_facts": self.game_facts,
            "opponent_game_facts": self.opponent_game_facts,
            "competitive_facts": self.competitive_facts,
        }


class EpisodeRunner:
    """Runs one episode headlessly. Preflight always runs first, every
    time - compatibility is verified per episode, never assumed from a
    prior run of the same variant."""

    def __init__(self, frame_dt: float = 1.0 / 30.0, poll_sleep: float = 0.01):
        self.frame_dt = frame_dt
        self.poll_sleep = poll_sleep

    def run(
        self,
        episode_spec: EpisodeSpec,
        telemetry_path: str,
        on_decision: Callable[[int, Any], None] | None = None,
    ) -> EpisodeResult:
        """`on_decision`, when given, is called `(decisions_made, game.state)`
        once right before the loop starts (decisions_made=0, the episode's
        initial state) and again every time a new decision completes -
        M5E.4.1's diagnostic hook for capturing a per-decision mission-
        progress snapshot (spec section 13). The evaluator - not the actor
        - is explicitly allowed to read `GameState` directly (see
        `_extract_game_facts`'s own docstring); this hook only ever hands
        that same already-permitted access to an external callback, never
        anything the actor's own Observation didn't already legally see
        being *used* for feedback. `None` (the default) preserves every
        pre-M5E.4.1 caller's behavior exactly - zero overhead, zero calls."""
        variant = episode_spec.variant
        compatibility = check_compatibility(variant)

        if not compatibility.supported:
            # Spec section 8: no gameplay episode starts, and gameplay
            # metrics are None/N/A - never a fake zero that would look like
            # a real, played mission loss.
            return EpisodeResult(
                experiment_id=episode_spec.experiment_id,
                variant_id=variant.variant_id,
                episode_id=episode_spec.episode_id,
                seed=episode_spec.seed,
                status=EpisodeStatus.INCOMPATIBLE,
                compatibility_status=compatibility.status,
                compatibility_reason=compatibility.reason,
            )

        try:
            return self._run_compatible_episode(episode_spec, telemetry_path, compatibility, on_decision)
        except Exception as exc:  # pragma: no cover - defensive: a failure this deep still
            # produces a structured result rather than killing the experiment
            # (spec section 33: failure isolation).
            return EpisodeResult(
                experiment_id=episode_spec.experiment_id,
                variant_id=variant.variant_id,
                episode_id=episode_spec.episode_id,
                seed=episode_spec.seed,
                status=EpisodeStatus.FAILED,
                compatibility_status=compatibility.status,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def _run_compatible_episode(
        self,
        episode_spec: EpisodeSpec,
        telemetry_path: str,
        compatibility: CompatibilityResult,
        on_decision: Callable[[int, Any], None] | None = None,
    ) -> EpisodeResult:
        from src.game import Game  # deferred: only import/construct once we know we're really running

        variant = episode_spec.variant
        wall_start = time.monotonic()
        status = EpisodeStatus.COMPLETED
        error_type: str | None = None
        error_message: str | None = None

        from src.bot_controller import AGENT_PROFILES, OPPORTUNIST

        # M5E: "scripted" is a deterministic-reference agent_runtime (spec
        # section 21) - White Spy driven by the same ScriptedBotController
        # the active opponent uses, bypassing the LLM registry entirely
        # (no provider/model call at all). Every other agent_runtime keeps
        # the pre-M5E LLM path unchanged.
        is_scripted_reference = variant.agent_runtime == "scripted"
        game = Game(
            seed=episode_spec.seed,
            white_controller=("scripted" if is_scripted_reference else "llm"),
            white_spy_profile=AGENT_PROFILES.get(variant.agent_profile, OPPORTUNIST),
            llm_provider=variant.provider,
            llm_model=variant.model,
            llm_base_url=variant.llm_base_url,
            llm_timeout=variant.llm_timeout,
            llm_execution_mode=episode_spec.execution_mode,
            agent_runtime=variant.agent_runtime,
            telemetry_path=telemetry_path,
            # So decisions.jsonl's episode_id matches episodes.jsonl's -
            # otherwise telemetry correlates only via GameState's own
            # internal UUID, which nothing else in the experiment's output
            # ever mentions (caught by integrity.py's decision-episode
            # correlation check).
            controller_episode_id=episode_spec.episode_id,
            # M5E spec section 20-22: an active, deterministic opponent for
            # Black Spy - "passive" (default, pre-M5E behavior) or
            # "scripted_active" (ScriptedBotController, spec section 4's
            # scripted_active_v1 policy - identical config/profile applied
            # regardless of which variant is being evaluated, spec section 15).
            black_controller=("scripted" if variant.opponent_type == "scripted_active" else "human"),
            black_spy_profile=AGENT_PROFILES.get(variant.opponent_profile, OPPORTUNIST),
            # M5E.1 spec section 5: tags every decision this episode's
            # evaluated agent makes with which opponent condition produced
            # it - purely descriptive telemetry, derived here from the same
            # `variant.opponent_type` value that already decides
            # `black_controller` above, never a second source of truth.
            condition=("active" if variant.opponent_type == "scripted_active" else "passive"),
            num_ctx=variant.num_ctx,
            keep_alive=variant.keep_alive,
            llm_region=variant.llm_region,
            white_personality=variant.personality_profile,
        )
        controller = game.bot_controller
        last_decisions_made = 0
        if on_decision is not None:
            on_decision(0, game.state)

        try:
            while True:
                wall_elapsed = time.monotonic() - wall_start
                if wall_elapsed >= episode_spec.max_episode_wall_time:
                    status = EpisodeStatus.TIMED_OUT
                    break

                state = game.state
                game_elapsed = state.mission_time_total - state.mission_time_remaining
                if state.mission_time_remaining > 0 and game_elapsed >= episode_spec.max_episode_game_time:
                    # The harness's own cap, cutting the episode short before
                    # the engine's own deadline would have - an explicit
                    # limit, not a disguised loss (spec section 13).
                    status = EpisodeStatus.ABORTED
                    error_type = "game_time_budget_exceeded"
                    break

                # Not controller._decision_seq: that counts every tick a new
                # enough observation *asked* for a decision, including ticks
                # where a previous one was still in flight (a slow model
                # leaves the loop spinning through many ticks before the
                # observation digest settles) - it can run well ahead of how
                # many decisions actually completed. The telemetry event
                # count is the real, one-per-completed-decision number.
                decisions_made = sum(1 for e in game.telemetry.events if e.get("event_type") == "llm_decision") if game.telemetry else 0
                if on_decision is not None and decisions_made > last_decisions_made:
                    on_decision(decisions_made, state)
                    last_decisions_made = decisions_made
                if decisions_made >= episode_spec.max_decisions:
                    status = EpisodeStatus.DECISION_LIMIT_REACHED
                    break

                # M5E: a scripted-reference White controller (ScriptedBotController)
                # has no model calls and therefore no fallback_count at all -
                # 0 is the correct, honest default (there is nothing to fail).
                #
                # M6A.2: a real, previously-undiscovered bug found live -
                # this used to read the combined `fallback_count`, which
                # both `LLMController` and `StrandsAgentAdapter` increment
                # identically for EVERY fallback, genuine provider failure
                # or ordinary already-repaired MODEL VALIDATION failure
                # alike (e.g. a hallucinated `trap_type`). `duel_runner.py`
                # was fixed for exactly this conflation back in M5F.3B.1,
                # but that fix never reached this harness - a live M6A.2
                # pilot episode hit 21/39 fallbacks (almost all a clean,
                # harmless `invalid_trap_type` repair case, never a real
                # infra problem) and this check aborted the episode at
                # `max_runtime_failures=20` before it ever finished,
                # discarding a real behavioral result as a technical
                # failure. Now reads `provider_failure_count` (added to
                # `StrandsAgentAdapter` this same milestone, mirroring
                # `LLMController`'s own M5F.3B.1 split) when the
                # controller has it, falling back to the old combined
                # counter only for a controller with no split at all
                # (ScriptedBotController) - byte-identical behavior there.
                provider_failures = getattr(controller, "provider_failure_count", None)
                if provider_failures is None:
                    provider_failures = getattr(controller, "fallback_count", 0)
                if provider_failures > episode_spec.max_runtime_failures:
                    status = EpisodeStatus.ABORTED
                    error_type = "max_runtime_failures_exceeded"
                    break

                # M6A.2: a separate, more generous cap for MODEL VALIDATION
                # failures specifically (spec section 45) - `None` (every
                # pre-M6A.2 `EpisodeSpec`'s implicit value) means this
                # check never fires, preserving every existing caller's
                # exact prior behavior.
                if episode_spec.max_validation_failures is not None:
                    validation_failures = getattr(controller, "validation_failure_count", 0)
                    if validation_failures > episode_spec.max_validation_failures:
                        status = EpisodeStatus.ABORTED
                        error_type = "max_validation_failures_exceeded"
                        break

                if state.mission_complete.get(constants.WHITE_SPY_ID, False):
                    status = EpisodeStatus.COMPLETED
                    break
                if state.mission_time_remaining <= 0:
                    # The engine's own deadline passed without escaping -
                    # a legitimate mission loss, not a harness abort. The
                    # episode still completed normally (spec section 9).
                    status = EpisodeStatus.COMPLETED
                    break

                game._step(self.frame_dt)
                time.sleep(self.poll_sleep)
        except Exception as exc:
            status = EpisodeStatus.FAILED
            error_type = type(exc).__name__
            error_message = str(exc)

        wall_time = time.monotonic() - wall_start
        if game.telemetry is not None:
            decision_events = [
                e for e in game.telemetry.events
                if e.get("event_type") == "llm_decision" and e.get("episode_id") == episode_spec.episode_id
            ]
            # M5D.2/M5D.3: the coarser accepted-action-boundary trace -
            # one row per MOVE/INSPECT/etc. reaching a terminal lifecycle
            # state, distinct from (and usually fewer than) decision_events.
            action_lifecycle_events = [
                e for e in game.telemetry.events
                if e.get("event_type") == "action_lifecycle" and e.get("episode_id") == episode_spec.episode_id
            ]
        else:  # pragma: no cover - telemetry_path is always passed by callers
            decision_events = []
            action_lifecycle_events = []
        # M5E: ScriptedBotController (the "scripted" reference agent_runtime)
        # has no AgentMemory at all - {} is the correct, honest snapshot,
        # matching every memory-derived metric's own None/0 default.
        memory_snapshot = controller.memory.to_compact_dict() if hasattr(controller, "memory") else {}
        game_facts = _extract_game_facts(game.state, constants.WHITE_SPY_ID)
        opponent_game_facts = (
            _extract_game_facts(game.state, constants.BLACK_SPY_ID) if variant.opponent_type == "scripted_active" else None
        )
        competitive_facts = (
            _extract_competitive_facts(game.state, constants.WHITE_SPY_ID, constants.BLACK_SPY_ID)
            if variant.opponent_type == "scripted_active"
            else None
        )
        metrics = compute_metrics(decision_events, memory_snapshot, game_facts, action_lifecycle_events)

        return EpisodeResult(
            experiment_id=episode_spec.experiment_id,
            variant_id=variant.variant_id,
            episode_id=episode_spec.episode_id,
            seed=episode_spec.seed,
            status=status,
            compatibility_status=compatibility.status,
            mission_success=game_facts["mission_success"],
            escape_success=game_facts["mission_success"],  # this game's only success mode is escaping with required items
            winner=game.episode_record().winner,
            game_time=game.state.mission_time_total - game.state.mission_time_remaining,
            wall_time=wall_time,
            error_type=error_type,
            error_message=error_message,
            decision_count=len(decision_events),
            metrics=metrics,
            game_facts=game_facts,
            opponent_game_facts=opponent_game_facts,
            competitive_facts=competitive_facts,
        )
