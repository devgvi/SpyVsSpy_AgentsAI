"""M5E.1 spec sections 11-12: failure-streak and stop-reason computation.

Pure, read-only analysis over an episode's already-recorded `llm_decision`
telemetry events (or an `EpisodeResult`) - never touches gameplay, never
changes what LLMController/StrandsAgentAdapter record. Exists because M5E
found real, sustained failure streaks (see CHANGELOG's M5E.1 entry) and
the raw telemetry alone doesn't answer "when did it start" / "did it ever
recover" / "why did the episode actually stop" without re-deriving it by
hand each time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm_controller import PROVIDER_TIMEOUT, RUNTIME_EXCEPTION, classify_provider_call_error

# Spec section 11's own vocabulary, plus two pragmatic additions this
# codebase's actual EpisodeStatus/error_type values need to represent
# honestly (TIMED_OUT is a wall-clock harness limit distinct from
# mission_deadline/decision_limit/runtime_failure_budget, and FAILED is a
# genuine harness exception, not a diagnostic choice) - see
# `classify_stop_reason`'s docstring.
STOP_NORMAL_COMPLETION = "normal_completion"
STOP_MISSION_DEADLINE = "mission_deadline"
STOP_DECISION_LIMIT = "decision_limit"
STOP_RUNTIME_FAILURE_BUDGET = "runtime_failure_budget"
STOP_PROVIDER_TIMEOUT_BUDGET = "provider_timeout_budget"
STOP_MANUAL_DIAGNOSTIC_STOP = "manual_diagnostic_stop"
STOP_WALL_TIME_BUDGET_EXCEEDED = "wall_time_budget_exceeded"
STOP_HARNESS_EXCEPTION = "harness_exception"


def classify_stop_reason(
    status: str,
    error_type: str | None,
    mission_success: bool | None,
    provider_timeout_fraction: float | None = None,
) -> str:
    """`status`/`error_type` are `EpisodeResult.status.value`/`.error_type`
    exactly as `episode_runner.py` sets them (see its `_run_compatible_episode`
    loop) - never re-derived from raw events, since the harness already
    knows definitively why it stopped. `provider_timeout_fraction` (0-1),
    when given, is the share of that episode's failed provider calls that
    were specifically `provider_timeout` (see `classify_provider_call_error`
    in `src.llm_controller`) - used only to decide whether a
    `runtime_failure_budget` stop should be reported more specifically as
    `provider_timeout_budget` (spec section 11 lists both as distinct
    reasons); a majority-timeout episode gets the more specific label.
    """
    if status == "COMPLETED":
        return STOP_NORMAL_COMPLETION if mission_success else STOP_MISSION_DEADLINE
    if status == "DECISION_LIMIT_REACHED":
        return STOP_DECISION_LIMIT
    if status == "TIMED_OUT":
        return STOP_WALL_TIME_BUDGET_EXCEEDED
    if status == "ABORTED":
        if error_type == "game_time_budget_exceeded":
            return STOP_MISSION_DEADLINE
        if error_type == "max_runtime_failures_exceeded":
            if provider_timeout_fraction is not None and provider_timeout_fraction > 0.5:
                return STOP_PROVIDER_TIMEOUT_BUDGET
            return STOP_RUNTIME_FAILURE_BUDGET
        return STOP_RUNTIME_FAILURE_BUDGET
    if status == "FAILED":
        return STOP_HARNESS_EXCEPTION
    return STOP_MANUAL_DIAGNOSTIC_STOP


@dataclass
class FailureStreakReport:
    first_failure_decision: int | None
    consecutive_failure_streak_length: int
    max_consecutive_failure_streak: int
    recovered_after_failure: bool
    decision_count_before_abort: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_failure_decision": self.first_failure_decision,
            "consecutive_failure_streak_length": self.consecutive_failure_streak_length,
            "max_consecutive_failure_streak": self.max_consecutive_failure_streak,
            "recovered_after_failure": self.recovered_after_failure,
            "decision_count_before_abort": self.decision_count_before_abort,
        }


def compute_failure_streak(decisions: list[dict[str, Any]]) -> FailureStreakReport:
    """`decisions` is one episode's `llm_decision` events, in the order
    they were emitted (already the natural order they're written to
    decisions.jsonl in). A "failure" is `final_model_action_valid is
    False` - the same field `_recovery_outcome`/every prior milestone's
    reporting already treats as the authoritative pass/fail signal, not a
    new definition invented here.
    """
    if not decisions:
        return FailureStreakReport(None, 0, 0, False, 0)

    first_failure_decision: int | None = None
    current_streak = 0
    max_streak = 0
    recovered = False
    seen_failure = False

    for idx, decision in enumerate(decisions, start=1):
        decision_number = decision.get("decision_number") if decision.get("decision_number") is not None else idx
        failed = decision.get("final_model_action_valid") is False
        if failed:
            if first_failure_decision is None:
                first_failure_decision = decision_number
            seen_failure = True
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            if seen_failure and current_streak > 0:
                recovered = True
            current_streak = 0

    return FailureStreakReport(
        first_failure_decision=first_failure_decision,
        consecutive_failure_streak_length=current_streak,
        max_consecutive_failure_streak=max_streak,
        recovered_after_failure=recovered,
        decision_count_before_abort=len(decisions),
    )


@dataclass
class LongHorizonStabilityReport:
    """M5E.2 spec section 15: per-episode long-horizon stability facts,
    combining `compute_failure_streak`'s decision-level view with
    disjoint provider-call-error-category counts (spec section 8's
    `classify_provider_call_error`, already computed per `provider_call`
    event - never recomputed from raw error strings here)."""

    evaluated_agent_decisions: int
    valid_decisions: int
    # M5E.4 addition: `valid_decisions` above is *final* validity (after
    # any repair) - the right signal for the majority-invalid technical-
    # failure ratio, but a runtime comparison also needs the narrower
    # "needed no repair at all" count (spec section 31's actual "first-
    # pass validity" metric) kept genuinely distinct from it, not the
    # same number under two names.
    first_pass_valid_decisions: int
    first_failure_decision: int | None
    max_consecutive_technical_failures: int
    provider_timeouts: int
    runtime_exceptions: int
    watchdog_terminations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_agent_decisions": self.evaluated_agent_decisions,
            "valid_decisions": self.valid_decisions,
            "first_pass_valid_decisions": self.first_pass_valid_decisions,
            "first_failure_decision": self.first_failure_decision,
            "max_consecutive_technical_failures": self.max_consecutive_technical_failures,
            "provider_timeouts": self.provider_timeouts,
            "runtime_exceptions": self.runtime_exceptions,
            "watchdog_terminations": self.watchdog_terminations,
        }


def compute_long_horizon_stability(
    decisions: list[dict[str, Any]],
    provider_calls: list[dict[str, Any]],
) -> LongHorizonStabilityReport:
    """`decisions` and `provider_calls` are one episode's own
    `llm_decision`/`provider_call` telemetry events. `max_consecutive_technical_failures`
    reuses `compute_failure_streak`'s streak logic - a "failure" there
    already means "this decision fell back," the same authoritative
    signal M5E/M5E.1 reporting has always used, not a new definition.
    """
    streak = compute_failure_streak(decisions)
    valid_decisions = sum(1 for d in decisions if d.get("final_model_action_valid") is True)
    first_pass_valid_decisions = sum(1 for d in decisions if d.get("recovery_outcome") == "FIRST_PASS_VALID")

    provider_timeouts = 0
    runtime_exceptions = 0
    watchdog_terminations = 0
    for call in provider_calls:
        if call.get("provider_error_type") is None:
            continue
        # watchdog_terminations is a strict subset of provider_timeout's
        # category (spec section 15 wants it counted separately, not as
        # an additional/overlapping bucket) - checked first so it isn't
        # double-counted into provider_timeouts below.
        if call.get("provider_error_type") == PROVIDER_TIMEOUT and _is_watchdog_termination(call):
            watchdog_terminations += 1
            continue
        if call.get("provider_error_type") == PROVIDER_TIMEOUT:
            provider_timeouts += 1
        elif call.get("provider_error_type") == RUNTIME_EXCEPTION:
            runtime_exceptions += 1

    return LongHorizonStabilityReport(
        evaluated_agent_decisions=len(decisions),
        valid_decisions=valid_decisions,
        first_pass_valid_decisions=first_pass_valid_decisions,
        first_failure_decision=streak.first_failure_decision,
        max_consecutive_technical_failures=streak.max_consecutive_failure_streak,
        provider_timeouts=provider_timeouts,
        runtime_exceptions=runtime_exceptions,
        watchdog_terminations=watchdog_terminations,
    )


def _is_watchdog_termination(provider_call_event: dict[str, Any]) -> bool:
    """A `provider_call` event doesn't carry the adapter's internal
    `error_type` string directly (only the already-classified
    `provider_error_type` category) - `provider_error_message_normalized`
    is what actually distinguishes StrandsAgentAdapter's v2 watchdog
    ("... adapter watchdog ...") from a real provider-reported timeout."""
    message = provider_call_event.get("provider_error_message_normalized") or ""
    return "adapter watchdog" in message
