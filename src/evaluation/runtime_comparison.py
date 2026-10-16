"""M5E.4: pure, testable building blocks for the controlled Bedrock
runtime comparison (Native tool-like vs Strands) - balanced A/B run
order, paired per-seed rows/narrative, and reuse (never reinvention) of
M5E.3.1's technical-status classification, mid-run stop rules, and
aggregate table for each variant's own episode list.

Honesty note (spec sections 26-29): `decisions_to_mission_completion`/
`decisions_to_escape` are derivable exactly from an already-COMPLETED
episode's own `decision_count` (this game's mission-complete and escape
events are the same moment - `EpisodeRunner`'s loop stops the instant
`mission_complete` becomes true, so the final decision count *is* the
completion decision). `decisions_to_first_room_exit`/
`decisions_to_first_mission_item`/`decisions_to_first_successful_interaction`
and the "strategy-relevant observable behaviors" spec section 29
describes (route changes after opponent sighting, trap placement after
encounter, etc.) would need per-decision room/inventory/encounter state
snapshots that no current telemetry event records - they are left `None`
/ unimplemented here rather than approximated from the final snapshot,
and this module says so explicitly instead of silently omitting them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.evaluation.multiseed_stability import count_provider_failures_by_category, episode_has_auth_expiry, failure_signature

VARIANT_A_NATIVE = "A_native_tool_like_bedrock"
VARIANT_B_STRANDS = "B_strands_bedrock"


def balanced_run_order(seeds: list[int]) -> dict[int, tuple[str, str]]:
    """Spec section 15: alternate which variant runs first per seed
    (never always Native-first) - seed[0]: A then B, seed[1]: B then A,
    seed[2]: A then B, ... Returns `{seed: (first_variant_id, second_variant_id)}`."""
    order: dict[int, tuple[str, str]] = {}
    for i, seed in enumerate(seeds):
        if i % 2 == 0:
            order[seed] = (VARIANT_A_NATIVE, VARIANT_B_STRANDS)
        else:
            order[seed] = (VARIANT_B_STRANDS, VARIANT_A_NATIVE)
    return order


# -- action-efficiency (spec section 26) - only the two truthfully derivable ones

NOT_INSTRUMENTED = None  # explicit marker: this event's decision-index isn't tracked by any current telemetry


def action_efficiency_metrics(episode: Any) -> dict[str, int | None]:
    completion_decision = episode.decision_count if episode.mission_success else None
    return {
        "decisions_to_first_room_exit": NOT_INSTRUMENTED,
        "decisions_to_first_mission_item": NOT_INSTRUMENTED,
        "decisions_to_first_successful_interaction": NOT_INSTRUMENTED,
        "decisions_to_mission_completion": completion_decision,
        # This game's mission-complete and escape events are the same
        # moment (see episode_runner.py's own docstring) - identical value,
        # reported under both names for spec section 26's vocabulary.
        "decisions_to_escape": completion_decision,
    }


# -- looping/repetition + invalid-action-category metrics (spec sections 27-28)
# already computed by compute_metrics() - surfaced here, not recomputed.

LOOP_REPETITION_METRIC_KEYS = (
    "repeated_action_count",
    "recent_repetitions_count",
    "revisited_failed_target_count",
    "same_illegal_action_repeated",
)

INVALID_ACTION_CATEGORY_KEYS = (
    "illegal_actions",
    "out_of_range_model_actions",
    "unknown_target_actions",
    "engine_rejected_actions",
    "stale_actions",
)


def loop_and_invalid_action_metrics(episode: Any) -> dict[str, int]:
    metrics = episode.metrics or {}
    return {key: metrics.get(key, 0) for key in (LOOP_REPETITION_METRIC_KEYS + INVALID_ACTION_CATEGORY_KEYS)}


# -- per-seed row (spec section 47) -----------------------------------------


def build_variant_seed_row(
    episode: Any,
    variant_id: str,
    specific_status: str,
    long_horizon: Any,
    provider_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = episode.metrics or {}
    latency = metrics.get("latency_ms") or {}
    row = {
        "seed": episode.seed,
        "variant_id": variant_id,
        "technical_status": specific_status,
        "mission_completed": episode.mission_success,
        "escaped": episode.escape_success,
        "decision_count": long_horizon.evaluated_agent_decisions,
        # Genuine first-pass validity (needed no repair) - see
        # failure_streak.py's `first_pass_valid_decisions` vs
        # `valid_decisions` distinction (spec section 31 wants this
        # reported separately from repairs, not the post-repair figure).
        "first_pass_valid": long_horizon.first_pass_valid_decisions,
        "repairs": metrics.get("repair_attempts", 0),
        "repair_successes": metrics.get("repair_successes", 0),
        "fallbacks": metrics.get("fallbacks", 0),
        "provider_failures": sum(count_provider_failures_by_category(provider_calls).values()),
        "rooms_visited": metrics.get("unique_rooms_visited"),
        "mission_items_acquired": metrics.get("mission_items_acquired"),
        "successful_interactions": metrics.get("successful_interactions"),
        "traps_placed": metrics.get("traps_placed"),
        "disarms": metrics.get("traps_disarmed"),
        "median_latency_ms": latency.get("median"),
        "max_latency_ms": latency.get("max"),
        "input_tokens": metrics.get("input_tokens_total"),
        "output_tokens": metrics.get("output_tokens_total"),
        # Strands-only (spec section 37) - `compute_metrics` sums each
        # decision's `runtime_metadata.tool_call_count` into `tool_calls`;
        # always 0 for Native (which never sets that field, having no
        # "tool call" concept of its own - `first_pass_valid` doubles as
        # its "structured-action-proposal validity" count instead).
        "tool_call_count": metrics.get("tool_calls"),
    }
    row.update(action_efficiency_metrics(episode))
    row.update(loop_and_invalid_action_metrics(episode))
    return row


@dataclass
class PairedRunOutcome:
    episodes: list[Any]  # (episode_spec, EpisodeResult) actually run, in run order
    stop_reason: str | None
    stop_detail: str | None
    tasks_not_run: list[tuple[int, str]]  # [(seed, variant_id), ...] never attempted


def run_paired_seeds_with_stop_rules(
    seeds: list[int],
    run_order: dict[int, tuple[str, str]],
    episode_spec_for: Callable[[int, str], Any],
    run_one_episode: Callable[[Any], Any],
    load_provider_calls: Callable[[str], list[dict[str, Any]]],
    classify: Callable[[Any, list[dict[str, Any]]], str],
) -> PairedRunOutcome:
    """Spec section 58's stop conditions, tracked *per runtime* (a
    repeated-failure-signature stop on Native must never be triggered by
    Strands episodes, and vice versa - the two variants' failure
    histories are independent pools). `episode_spec_for(seed, variant_id)`
    builds that one EpisodeSpec; `run_order[seed]` decides which variant
    goes first for that seed (spec section 15's balanced ordering)."""
    tasks: list[tuple[int, str]] = []
    for seed in seeds:
        first, second = run_order[seed]
        tasks.append((seed, first))
        tasks.append((seed, second))

    episodes: list[tuple[Any, Any]] = []
    signature_counts: dict[tuple[str, str], int] = {}
    stop_reason: str | None = None
    stop_detail: str | None = None

    for i, (seed, variant_id) in enumerate(tasks):
        episode_spec = episode_spec_for(seed, variant_id)
        result = run_one_episode(episode_spec)
        episodes.append((episode_spec, result))
        provider_calls = load_provider_calls(episode_spec.episode_id)

        if episode_has_auth_expiry(provider_calls):
            stop_reason = "auth_token_expired_mid_run"
            stop_detail = f"provider_auth_error observed on seed {seed} / {variant_id} (episode {episode_spec.episode_id})"
            break

        specific_status = classify(result, provider_calls)
        signature = failure_signature(result, provider_calls, specific_status)
        if signature is not None:
            key = (variant_id, signature)
            signature_counts[key] = signature_counts.get(key, 0) + 1
            if signature_counts[key] >= 2:
                stop_reason = "repeated_failure_signature"
                stop_detail = f"runtime={variant_id} signature={signature!r} occurred on {signature_counts[key]} episodes (latest: seed {seed})"
                break

    tasks_not_run = tasks[len(episodes):]
    return PairedRunOutcome(episodes=episodes, stop_reason=stop_reason, stop_detail=stop_detail, tasks_not_run=tasks_not_run)


def paired_seed_narrative(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, Any]:
    """Spec section 49: a short, purely factual (never speculative)
    account of where the two variants' recorded trajectories diverged on
    the same seed - built entirely from the objective counters already in
    each row, no inference about hidden reasoning."""
    facts: list[str] = []
    if row_a["mission_completed"] != row_b["mission_completed"]:
        winner = "Native" if row_a["mission_completed"] else "Strands"
        facts.append(f"{winner} completed the mission; the other did not.")
    if row_a["technical_status"] != row_b["technical_status"]:
        facts.append(f"technical status differed: Native={row_a['technical_status']!r}, Strands={row_b['technical_status']!r}")
    if row_a["repairs"] != row_b["repairs"]:
        facts.append(f"repair attempts: Native={row_a['repairs']}, Strands={row_b['repairs']}")
    if row_a["fallbacks"] != row_b["fallbacks"]:
        facts.append(f"fallbacks: Native={row_a['fallbacks']}, Strands={row_b['fallbacks']}")
    if row_a["rooms_visited"] != row_b["rooms_visited"]:
        facts.append(f"rooms visited: Native={row_a['rooms_visited']}, Strands={row_b['rooms_visited']}")
    if row_a["mission_items_acquired"] != row_b["mission_items_acquired"]:
        facts.append(f"mission items acquired: Native={row_a['mission_items_acquired']}, Strands={row_b['mission_items_acquired']}")
    return {"seed": row_a["seed"], "facts": facts or ["no notable divergence in the recorded counters"]}
