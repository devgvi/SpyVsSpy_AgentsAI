"""M5F: drives one real LLM-vs-LLM duel episode - both Black and White
independently LLM-controlled, through the same `Game`/`GameState` loop
every other milestone uses.

This is a purpose-built driver, not an extension of the shared M5D
`EpisodeRunner`/`VariantSpec` harness: that harness was built around
evaluating exactly one LLM-controlled actor against a scripted/passive
opponent (its decision counting isn't `actor_id`-filtered, its
`fallback_count`/mission-complete checks read only `controller` = White -
see the M5F investigation this module's own CHANGELOG entry documents).
Retrofitting genuine two-actor symmetry into that harness would touch it
about as invasively as a rewrite, risking every other milestone's already-
validated behavior for a fundamentally different episode shape. A
dedicated driver, matching this project's established per-milestone-driver
precedent (`run_m5e3.py`, `run_m5e4.py`, ...), is the safer choice.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time
from dataclasses import dataclass
from typing import Any, Callable

from src import constants
from src.evaluation.episode_runner import _extract_competitive_facts, _extract_game_facts
from src.game import Game

MAX_LLM_DECISIONS_PER_ACTOR = 40


@dataclass
class DuelResult:
    episode_id: str
    seed: int
    status: str
    stop_reason: str | None
    game_time: float
    wall_time: float
    decision_call_order: str
    black_decision_count: int
    white_decision_count: int
    black_fallback_count: int
    white_fallback_count: int
    black_game_facts: dict[str, Any]
    white_game_facts: dict[str, Any]
    competitive_facts: dict[str, Any]
    winner: str | None
    black_personality: str | None = None
    white_personality: str | None = None
    communication_enabled: bool = False
    # M5F.3B.1 spec section 12: `fallback_count` above stays the historical
    # combined total; these two split out which kind of failure actually
    # happened, since only genuine provider/infrastructure failures should
    # end an otherwise-healthy episode (see `max_runtime_failures` below).
    black_provider_failure_count: int = 0
    white_provider_failure_count: int = 0
    black_validation_failure_count: int = 0
    white_validation_failure_count: int = 0


def _count_actor_decisions(telemetry_events: list[dict], episode_id: str, actor_id: str) -> int:
    """Spec section 5 of the M5F investigation: `llm_decision` events
    already carry `actor_id` - the shared `EpisodeRunner`'s own decision
    count never filters by it (single-actor-only harness), so this
    driver does its own filtering rather than reusing that count."""
    return sum(
        1
        for e in telemetry_events
        if e.get("event_type") == "llm_decision" and e.get("episode_id") == episode_id and e.get("actor_id") == actor_id
    )


def run_duel_episode(
    seed: int,
    provider: str,
    model: str,
    region: str | None,
    agent_runtime: str,
    telemetry_path: str,
    episode_id: str,
    black_spy_profile=None,
    white_spy_profile=None,
    decision_call_order: str = "black_first",
    max_decisions_per_actor: int = MAX_LLM_DECISIONS_PER_ACTOR,
    max_episode_game_time: float = 180.0,
    max_episode_wall_time: float = 3600.0,
    max_runtime_failures: int = 20,
    # M5F.3B.1 spec sections 12-13: a separate, more generous cap for
    # cleanly-rejected-and-repaired-or-fallen-back MODEL validation
    # failures (e.g. a hallucinated trap type) - these are not
    # infrastructure instability and must not trip `max_runtime_failures`
    # (see the split `provider_failure_count`/`validation_failure_count`
    # below), but an unbounded stream of them would otherwise silently
    # burn an entire episode's decision budget on repeated WAIT fallbacks.
    # Explicit and conservative per spec section 13 - well below the
    # 40-decision-per-actor cap, but well above ordinary occasional
    # repair-path noise.
    max_validation_failures: int = 30,
    llm_timeout: float = 60.0,
    frame_dt: float = 1.0 / 30.0,
    poll_sleep: float = 0.0,
    on_decision: Callable[[int, int, Any], None] | None = None,
    black_personality: str | None = None,
    white_personality: str | None = None,
    communication_enabled: bool = False,
    zero_decision_cost_messaging: bool = False,
    flat_message_schema: bool = False,
    message_cooldown_decisions: int = 0,
    # M5F.3B spec section 11: threads straight into `Game.__init__`'s own
    # new `initial_traps` override - `None` (every pre-M5F.3B caller's
    # implicit value) is byte-identical to not having this parameter at all.
    initial_traps: list[dict] | None = None,
    # M5F.3C spec sections 7, 14: per-actor evaluation-only hooks, threaded
    # straight into `Game.__init__`'s own new per-actor `model_payload_transform`
    # params - `None` (every pre-M5F.3C caller's implicit value) is
    # byte-identical to not having this parameter at all.
    black_model_payload_transform=None,
    white_model_payload_transform=None,
) -> DuelResult:
    """Runs exactly one duel to a stop condition (spec section 31: the
    decision cap is a ceiling, not a target - mission deadline/completion/
    technical failure can all end it earlier). `on_decision(black_n,
    white_n, state)` fires once at the start (0, 0) and again every time
    either actor's own decision count increases - the evaluator-side hook
    M5E.4.1 introduced, reused here for both actors."""
    from src.bot_controller import AGENT_PROFILES, OPPORTUNIST

    kwargs: dict[str, Any] = dict(
        seed=seed, white_controller="llm", black_controller="llm",
        llm_provider=provider, llm_model=model, llm_region=region, llm_timeout=llm_timeout,
        llm_execution_mode="evaluation", agent_runtime=agent_runtime,
        telemetry_path=telemetry_path, controller_episode_id=episode_id,
        decision_call_order=decision_call_order,
        black_personality=black_personality, white_personality=white_personality,
        communication_enabled=communication_enabled,
        zero_decision_cost_messaging=zero_decision_cost_messaging,
        flat_message_schema=flat_message_schema,
        message_cooldown_decisions=message_cooldown_decisions,
    )
    if initial_traps is not None:
        kwargs["initial_traps"] = initial_traps
    if black_model_payload_transform is not None:
        kwargs["black_model_payload_transform"] = black_model_payload_transform
    if white_model_payload_transform is not None:
        kwargs["white_model_payload_transform"] = white_model_payload_transform
    if black_spy_profile is not None:
        kwargs["black_spy_profile"] = AGENT_PROFILES.get(black_spy_profile, OPPORTUNIST)
    if white_spy_profile is not None:
        kwargs["white_spy_profile"] = AGENT_PROFILES.get(white_spy_profile, OPPORTUNIST)
    game = Game(**kwargs)

    wall_start = time.monotonic()
    status = "COMPLETED"
    stop_reason: str | None = None
    last_black_n = 0
    last_white_n = 0
    if on_decision is not None:
        on_decision(0, 0, game.state)

    try:
        while True:
            wall_elapsed = time.monotonic() - wall_start
            if wall_elapsed >= max_episode_wall_time:
                status, stop_reason = "TIMED_OUT", "wall_time_budget_exceeded"
                break

            state = game.state
            game_elapsed = state.mission_time_total - state.mission_time_remaining
            if state.mission_time_remaining > 0 and game_elapsed >= max_episode_game_time:
                status, stop_reason = "ABORTED", "game_time_budget_exceeded"
                break

            events = game.telemetry.events if game.telemetry else []
            black_n = _count_actor_decisions(events, episode_id, constants.BLACK_SPY_ID)
            white_n = _count_actor_decisions(events, episode_id, constants.WHITE_SPY_ID)
            if on_decision is not None and (black_n > last_black_n or white_n > last_white_n):
                on_decision(black_n, white_n, state)
                last_black_n, last_white_n = black_n, white_n

            # Spec section 31: a ceiling per actor, not a shared budget -
            # the first actor to reach it ends the duel (spec doesn't
            # mandate letting the other continue alone, and doing so would
            # need a "freeze one actor's decisions while the other keeps
            # going" mechanism this milestone doesn't otherwise need).
            if black_n >= max_decisions_per_actor or white_n >= max_decisions_per_actor:
                status = "DECISION_LIMIT_REACHED"
                stop_reason = "black_decision_cap" if black_n >= max_decisions_per_actor else "white_decision_cap"
                break

            # M5F.3B.1 spec section 12: only genuine provider/infrastructure
            # failures trip the runtime-failure cap now - a model
            # repeatedly proposing an invalid action (e.g. a hallucinated
            # trap type) is cleanly rejected/repaired/fallen-back by
            # `LLMController` itself and must not end an otherwise-healthy
            # episode on its own (spec section 13). Checked separately,
            # with its own conservative cap and stop_reason, below.
            black_provider_failures = getattr(game.black_llm_controller, "provider_failure_count", 0)
            white_provider_failures = getattr(game.bot_controller, "provider_failure_count", 0)
            if black_provider_failures > max_runtime_failures or white_provider_failures > max_runtime_failures:
                status, stop_reason = "ABORTED", "max_runtime_failures_exceeded"
                break

            black_validation_failures = getattr(game.black_llm_controller, "validation_failure_count", 0)
            white_validation_failures = getattr(game.bot_controller, "validation_failure_count", 0)
            if black_validation_failures > max_validation_failures or white_validation_failures > max_validation_failures:
                status, stop_reason = "ABORTED", "max_validation_failures_exceeded"
                break

            if state.mission_time_remaining <= 0:
                status, stop_reason = "COMPLETED", "mission_deadline"
                break

            # Spec section 39: both actors may succeed - only stop once
            # *both* have completed, so one escaping doesn't end the
            # other's race early. `LLMController.get_action` already
            # short-circuits on `observation.mission_complete` (returns
            # WAIT, requests no further decisions), so a completed actor
            # simply stops consuming its own decision budget from here on.
            both_done = (
                state.mission_complete.get(constants.BLACK_SPY_ID, False)
                and state.mission_complete.get(constants.WHITE_SPY_ID, False)
            )
            if both_done:
                status, stop_reason = "COMPLETED", "both_missions_complete"
                break

            game._step(frame_dt)
            if poll_sleep:
                time.sleep(poll_sleep)
    except Exception as exc:
        status, stop_reason = "FAILED", f"{type(exc).__name__}: {exc}"

    wall_time = time.monotonic() - wall_start
    events = game.telemetry.events if game.telemetry else []
    black_n = _count_actor_decisions(events, episode_id, constants.BLACK_SPY_ID)
    white_n = _count_actor_decisions(events, episode_id, constants.WHITE_SPY_ID)
    black_facts = _extract_game_facts(game.state, constants.BLACK_SPY_ID)
    white_facts = _extract_game_facts(game.state, constants.WHITE_SPY_ID)
    competitive_facts = _extract_competitive_facts(game.state, constants.WHITE_SPY_ID, constants.BLACK_SPY_ID)

    return DuelResult(
        episode_id=episode_id, seed=seed, status=status, stop_reason=stop_reason,
        game_time=game.state.mission_time_total - game.state.mission_time_remaining, wall_time=wall_time,
        decision_call_order=decision_call_order,
        black_decision_count=black_n, white_decision_count=white_n,
        black_fallback_count=getattr(game.black_llm_controller, "fallback_count", 0),
        white_fallback_count=getattr(game.bot_controller, "fallback_count", 0),
        black_provider_failure_count=getattr(game.black_llm_controller, "provider_failure_count", 0),
        white_provider_failure_count=getattr(game.bot_controller, "provider_failure_count", 0),
        black_validation_failure_count=getattr(game.black_llm_controller, "validation_failure_count", 0),
        white_validation_failure_count=getattr(game.bot_controller, "validation_failure_count", 0),
        black_game_facts=black_facts, white_game_facts=white_facts, competitive_facts=competitive_facts,
        winner=game.episode_record().winner,
        black_personality=black_personality, white_personality=white_personality,
        communication_enabled=communication_enabled,
    )
