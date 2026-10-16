"""M5E.3.1: pure, testable building blocks for the Bedrock multi-seed
stability confirmation - mid-run auth-expiry / repeated-failure stop
rules, the spec's five-way technical-status vocabulary (a thin relabeling
of M5E.2's `classify_episode_technical_status`, not a new classifier),
per-seed/aggregate reporting tables, and host-sleep-log contamination
detection (spec section 23). Never touches gameplay or the LLM client -
everything here operates on already-recorded `EpisodeResult`s and
telemetry events.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from src.evaluation.competitive_aggregation import (
    DEADLINE_COMPLETION,
    DECISION_CAP_COMPLETION,
    GAMEPLAY_COMPLETION,
    TECHNICAL_FAILURE,
    classify_episode_technical_status,
)
from src.evaluation.failure_streak import LongHorizonStabilityReport
from src.llm_controller import PROVIDER_AUTH_ERROR

# Spec section 10's five-way vocabulary is M5E.2's own four categories
# (imported above, already fully tested there) plus one derived boolean
# label used for the "N/5 technically clean" success criterion (spec
# sections 16-17) - not a second, parallel classifier.
TECHNICALLY_CLEAN = "technically_clean"
GAMEPLAY_COMPLETE = "gameplay_complete"
DECISION_LIMIT_REACHED = "decision_limit_reached"
MISSION_DEADLINE_REACHED = "mission_deadline_reached"

_M5E2_TO_M5E3_1_LABEL = {
    GAMEPLAY_COMPLETION: GAMEPLAY_COMPLETE,
    DECISION_CAP_COMPLETION: DECISION_LIMIT_REACHED,
    DEADLINE_COMPLETION: MISSION_DEADLINE_REACHED,
    TECHNICAL_FAILURE: TECHNICAL_FAILURE,
}


def classify_episode_for_m5e3_1(episode: Any, long_horizon_stability: LongHorizonStabilityReport | None) -> str:
    """Spec section 10's specific-reason label (`gameplay_complete` /
    `decision_limit_reached` / `mission_deadline_reached` / `technical_failure`),
    reusing M5E.2's already-tested `classify_episode_technical_status`
    unchanged - only the label strings are M5E.3.1's own vocabulary."""
    m5e2_status = classify_episode_technical_status(episode, long_horizon_stability)
    return _M5E2_TO_M5E3_1_LABEL[m5e2_status]


def is_technically_clean(specific_status: str) -> bool:
    """Spec section 16-17's success criterion: any classification other
    than `technical_failure` counts as clean for the "N/5 technically
    clean" headline number."""
    return specific_status != TECHNICAL_FAILURE


# -- provider-failure category tallying --------------------------------------


def count_provider_failures_by_category(provider_calls: list[dict[str, Any]]) -> dict[str, int]:
    """Spec section 11: every distinct `provider_error_type` category seen
    on this episode's own `provider_call` telemetry events, never merged
    into one bucket."""
    counts: dict[str, int] = {}
    for call in provider_calls:
        category = call.get("provider_error_type")
        if category:
            counts[category] = counts.get(category, 0) + 1
    return counts


def episode_has_auth_expiry(provider_calls: list[dict[str, Any]]) -> bool:
    """Spec section 3: did this episode's own provider calls include a
    `provider_auth_error` - the specific category a genuinely expired
    bearer token now classifies to (see `bedrock_client.classify_bedrock_exception`'s
    M5E.3.1 reclassification of `AccessDeniedException: ... expired`)."""
    return any(call.get("provider_error_type") == PROVIDER_AUTH_ERROR for call in provider_calls)


def failure_signature(episode: Any, provider_calls: list[dict[str, Any]], specific_status: str) -> str | None:
    """Spec section 19's "same technical failure signature" - the
    dominant provider-error category for a technically-failed episode, or
    the harness-level `error_type` (e.g. `max_runtime_failures_exceeded`)
    when the failure never produced a classified provider call. `None`
    for any non-failed episode (no signature to compare)."""
    if specific_status != TECHNICAL_FAILURE:
        return None
    counts = count_provider_failures_by_category(provider_calls)
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    return getattr(episode, "error_type", None) or "unknown_technical_failure"


# -- sequential multi-seed orchestration with explicit stop rules -----------


@dataclass
class MultiSeedRunOutcome:
    episodes: list[Any]
    seeds_run: list[int]
    seeds_not_run: list[int]
    stop_reason: str | None  # None | "auth_token_expired_mid_run" | "repeated_failure_signature"
    stop_detail: str | None


def run_seeds_with_stop_rules(
    episode_specs: list[Any],
    run_one_episode: Callable[[Any], Any],
    load_provider_calls: Callable[[str], list[dict[str, Any]]],
    classify: Callable[[Any, list[dict[str, Any]]], str] = None,
) -> MultiSeedRunOutcome:
    """Runs `episode_specs` in order via `run_one_episode(episode_spec) ->
    EpisodeResult`, stopping before any further seed the moment either
    spec section 3 (auth expired mid-run) or section 19 (>= 2 episodes
    share the same technical-failure signature) fires. `load_provider_calls(episode_id)`
    reads that episode's own telemetry (injected so this stays testable
    without a real telemetry file). Never silently retries or reorders -
    `episodes`/`seeds_run` are exactly the seeds actually attempted, in
    the order attempted (spec sections 20-21)."""
    episodes: list[Any] = []
    seeds_run: list[int] = []
    signature_counts: dict[str, int] = {}
    stop_reason: str | None = None
    stop_detail: str | None = None

    for i, episode_spec in enumerate(episode_specs):
        result = run_one_episode(episode_spec)
        episodes.append(result)
        seeds_run.append(episode_spec.seed)
        provider_calls = load_provider_calls(episode_spec.episode_id)

        if episode_has_auth_expiry(provider_calls):
            stop_reason = "auth_token_expired_mid_run"
            stop_detail = f"provider_auth_error observed during seed {episode_spec.seed} (episode {episode_spec.episode_id})"
            break

        specific_status = classify(result, provider_calls) if classify else None
        if specific_status is not None:
            signature = failure_signature(result, provider_calls, specific_status)
            if signature is not None:
                signature_counts[signature] = signature_counts.get(signature, 0) + 1
                if signature_counts[signature] >= 2:
                    stop_reason = "repeated_failure_signature"
                    stop_detail = f"signature {signature!r} occurred on {signature_counts[signature]} episodes (latest: seed {episode_spec.seed})"
                    break

    seeds_not_run = [es.seed for es in episode_specs[len(seeds_run):]]
    return MultiSeedRunOutcome(
        episodes=episodes, seeds_run=seeds_run, seeds_not_run=seeds_not_run,
        stop_reason=stop_reason, stop_detail=stop_detail,
    )


# -- per-seed / aggregate reporting tables (spec sections 39-40) -----------


def build_per_seed_row(
    episode: Any,
    specific_status: str,
    long_horizon: LongHorizonStabilityReport,
    provider_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = episode.metrics or {}
    latency = metrics.get("latency_ms") or {}
    failure_counts = count_provider_failures_by_category(provider_calls)
    return {
        "seed": episode.seed,
        "technical_status": specific_status,
        "decisions": long_horizon.evaluated_agent_decisions,
        "provider_failures": sum(failure_counts.values()),
        "runtime_failures": long_horizon.runtime_exceptions,
        # M5E.4 fix: this is genuine FIRST-pass validity (no repair
        # needed) - previously this field used `valid_decisions` (final,
        # post-repair validity), which is only correct when repairs
        # happen to be zero; a real repair now makes those two numbers
        # provably different (see failure_streak.py's own distinction).
        "first_pass_valid": long_horizon.first_pass_valid_decisions,
        "repairs": metrics.get("repair_attempts", 0),
        "fallbacks": metrics.get("fallbacks", 0),
        "median_latency_ms": latency.get("median"),
        "max_latency_ms": latency.get("max"),
        "input_tokens": metrics.get("input_tokens_total"),
        "output_tokens": metrics.get("output_tokens_total"),
        "mission_complete": episode.mission_success,
        "escaped": episode.escape_success,
        "opponent_escaped": (episode.opponent_game_facts or {}).get("mission_success") if episode.opponent_game_facts else None,
    }


def build_aggregate_table(per_seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_seed_rows)
    clean = sum(1 for r in per_seed_rows if r["technical_status"] != TECHNICAL_FAILURE)
    all_latencies_median = [r["median_latency_ms"] for r in per_seed_rows if r["median_latency_ms"] is not None]
    all_latencies_max = [r["max_latency_ms"] for r in per_seed_rows if r["max_latency_ms"] is not None]
    return {
        "episodes_requested": n,
        "episodes_technically_clean": clean,
        "technical_failure_episodes": n - clean,
        "total_decisions": sum(r["decisions"] for r in per_seed_rows),
        "provider_failures": sum(r["provider_failures"] for r in per_seed_rows),
        "runtime_failures": sum(r["runtime_failures"] for r in per_seed_rows),
        "first_pass_valid": sum(r["first_pass_valid"] for r in per_seed_rows),
        "repair_attempts": sum(r["repairs"] for r in per_seed_rows),
        "fallbacks": sum(r["fallbacks"] for r in per_seed_rows),
        "median_latency_ms": (sorted(all_latencies_median)[len(all_latencies_median) // 2] if all_latencies_median else None),
        "max_latency_ms": (max(all_latencies_max) if all_latencies_max else None),
        "total_input_tokens": sum(r["input_tokens"] or 0 for r in per_seed_rows),
        "total_output_tokens": sum(r["output_tokens"] or 0 for r in per_seed_rows),
    }


def cumulative_bedrock_decisions(prior_clean_decisions: int, prior_total_decisions: int, new_clean_decisions: int, new_total_decisions: int) -> dict[str, Any]:
    """Spec section 41's cumulative evidence line - plain arithmetic over
    the M5E.3 CHANGELOG-documented historical figure and this run's own
    aggregate, never a re-derivation from raw files that could silently
    drift from what was actually reported."""
    return {
        "m5e3_prior_live_decisions_clean": prior_clean_decisions,
        "m5e3_prior_live_decisions_total": prior_total_decisions,
        "m5e3_1_new_live_decisions_clean": new_clean_decisions,
        "m5e3_1_new_live_decisions_total": new_total_decisions,
        "cumulative_clean": prior_clean_decisions + new_clean_decisions,
        "cumulative_total": prior_total_decisions + new_total_decisions,
    }


# -- output-directory rerun preservation (spec sections 20-21, 38) ---------


def resolve_output_dir(base_dir: str, now: datetime | None = None) -> str:
    """Never silently overwrite a prior run's outputs. If `base_dir`
    already holds a completed run's `experiment.json`, this invocation is
    a rerun - it gets its own distinct, clearly-labeled directory instead
    (`<base_dir>-rerun-<UTC timestamp>`), and the original is left
    untouched."""
    if not os.path.exists(os.path.join(base_dir, "experiment.json")):
        return base_dir
    now = now or datetime.now(timezone.utc)
    return f"{base_dir}-rerun-{now.strftime('%Y%m%dT%H%M%SZ')}"


# -- host sleep-log contamination check (spec sections 22-23) ---------------

_SLEEP_DOMAINS = {"Sleep", "Wake", "DarkWake"}
_PMSET_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4})\s+(\S+)")


@dataclass
class SleepContaminationReport:
    contaminated: bool
    matching_events: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"sleep_contamination_detected": self.contaminated, "matching_events": self.matching_events}


def detect_sleep_contamination(pmset_log_text: str, window_start: datetime, window_end: datetime) -> SleepContaminationReport:
    """Spec section 23: cross-reference `pmset -g log` output against a
    run's own wall-clock window, exactly the methodology M5E.1.3 used
    manually (its finding: real sleep/wake events exactly explained an
    apparent "instability" that had nothing to do with request pacing).
    `window_start`/`window_end` must be timezone-aware (the log's own
    `+HHMM` offset is parsed per line, so naive datetimes would compare
    incorrectly across a DST boundary or a non-UTC caller)."""
    matches: list[str] = []
    for line in pmset_log_text.splitlines():
        match = _PMSET_LINE_RE.match(line)
        if not match:
            continue
        timestamp_str, domain = match.group(1), match.group(2)
        if domain not in _SLEEP_DOMAINS:
            continue
        try:
            event_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S %z")
        except ValueError:  # pragma: no cover - defensive: a malformed line shouldn't crash the check
            continue
        if window_start <= event_time <= window_end:
            matches.append(line.strip())
    return SleepContaminationReport(contaminated=bool(matches), matching_events=matches)
