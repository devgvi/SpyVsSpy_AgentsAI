"""M5D.1: integrity checks over a completed ExperimentResult.

Before any benchmark result is reported, verify the raw data actually
supports it (spec section 25: "Before accepting results verify..."). This
never touches gameplay - it's a pure, read-only audit over an already
finished ExperimentResult plus its written decisions.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.results import ExperimentResult


@dataclass
class IntegrityReport:
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "problems": self.problems}


def verify_integrity(
    result: ExperimentResult,
    variant_ids_expected_same_seeds: list[str] | None = None,
    decisions_jsonl_path: str | None = None,
) -> IntegrityReport:
    """Runs the checks spec section 25 asks for. `variant_ids_expected_same_seeds`
    restricts the "same seed set" check to the variants that are actually
    supposed to share seeds (e.g. A/B/C, not a compatibility-only control
    variant that never runs gameplay at all).
    """
    checks: dict[str, bool] = {}
    problems: list[str] = []

    # -- no missing result: one EpisodeResult per expanded EpisodeSpec ----
    expected_episode_ids = [e.episode_id for e in result.spec.expand()]
    actual_episode_ids = [e.episode_id for e in result.episodes]
    no_missing = sorted(expected_episode_ids) == sorted(actual_episode_ids)
    checks["no_missing_results"] = no_missing
    if not no_missing:
        missing = sorted(set(expected_episode_ids) - set(actual_episode_ids))
        extra = sorted(set(actual_episode_ids) - set(expected_episode_ids))
        problems.append(f"missing episode results: {missing}; unexpected extra: {extra}")

    # -- no duplicate episode IDs -----------------------------------------
    no_duplicates = len(actual_episode_ids) == len(set(actual_episode_ids))
    checks["no_duplicate_episode_ids"] = no_duplicates
    if not no_duplicates:
        seen = set()
        dupes = sorted({eid for eid in actual_episode_ids if eid in seen or seen.add(eid)})
        problems.append(f"duplicate episode_ids: {dupes}")

    # -- same seed set across the variants that are supposed to share one --
    variant_ids = variant_ids_expected_same_seeds or [v.variant_id for v in result.spec.variants]
    seed_sets = {
        vid: sorted({e.seed for e in result.episodes if e.variant_id == vid})
        for vid in variant_ids
    }
    distinct_seed_sets = {tuple(s) for s in seed_sets.values()}
    same_seeds = len(distinct_seed_sets) <= 1
    checks["same_seed_set_across_variants"] = same_seeds
    if not same_seeds:
        problems.append(f"seed sets differ across variants: {seed_sets}")

    # -- decision -> episode correlation: every decision in decisions.jsonl
    # references a real episode_id, and every SUPPORTED/gameplay episode's
    # own decision_count matches how many decisions actually reference it --
    if decisions_jsonl_path:
        try:
            with open(decisions_jsonl_path, "r", encoding="utf-8") as f:
                decision_events = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            decision_events = []
        decision_episode_ids = {e.get("episode_id") for e in decision_events if e.get("event_type") == "llm_decision"}
        known_episode_ids = set(actual_episode_ids)
        orphaned = sorted(decision_episode_ids - known_episode_ids)
        correlation_ok = not orphaned
        checks["decision_episode_correlation"] = correlation_ok
        if not correlation_ok:
            problems.append(f"decisions.jsonl has decisions for unknown episode_ids: {orphaned}")

        per_episode_decision_count = {
            eid: sum(1 for e in decision_events if e.get("event_type") == "llm_decision" and e.get("episode_id") == eid)
            for eid in known_episode_ids
        }
        mismatches = [
            e.episode_id
            for e in result.episodes
            if e.compatibility_status.value == "SUPPORTED" and per_episode_decision_count.get(e.episode_id, 0) != e.decision_count
        ]
        decision_counts_match = not mismatches
        checks["decision_counts_reconcile"] = decision_counts_match
        if not decision_counts_match:
            problems.append(f"episode.decision_count disagrees with decisions.jsonl for: {mismatches}")

        # -- M5D.2/M5D.3 spec section 30: "M5D.2 action lifecycle present" -
        # every episode whose own metrics record at least one accepted
        # action must actually have matching action_lifecycle telemetry on
        # disk, not just a metrics field asserting it happened.
        per_episode_lifecycle_count = {
            eid: sum(1 for e in decision_events if e.get("event_type") == "action_lifecycle" and e.get("episode_id") == eid)
            for eid in known_episode_ids
        }
        lifecycle_mismatches = [
            e.episode_id
            for e in result.episodes
            if e.compatibility_status.value == "SUPPORTED"
            and (e.metrics or {}).get("actions_accepted", 0) > 0
            and per_episode_lifecycle_count.get(e.episode_id, 0) == 0
        ]
        lifecycle_present = not lifecycle_mismatches
        checks["action_lifecycle_present"] = lifecycle_present
        if not lifecycle_present:
            problems.append(
                f"episode metrics report accepted actions but decisions.jsonl has no action_lifecycle "
                f"events for: {lifecycle_mismatches}"
            )

        # -- M5D.6 spec section 28: "recovery telemetry present" - every
        # llm_decision event should carry a non-null recovery_outcome
        # (M5D.5), for every SUPPORTED episode that actually made decisions.
        supported_with_decisions = {
            e.episode_id for e in result.episodes if e.compatibility_status.value == "SUPPORTED" and e.decision_count > 0
        }
        missing_recovery_outcome = sorted(
            {
                e.get("episode_id")
                for e in decision_events
                if e.get("event_type") == "llm_decision"
                and e.get("episode_id") in supported_with_decisions
                and e.get("recovery_outcome") is None
            }
        )
        recovery_telemetry_present = not missing_recovery_outcome
        checks["recovery_telemetry_present"] = recovery_telemetry_present
        if not recovery_telemetry_present:
            problems.append(f"llm_decision events missing recovery_outcome for episodes: {missing_recovery_outcome}")

        # -- M5E spec section 41: "actor-tagged telemetry complete" - every
        # SUPPORTED episode run with the active opponent
        # (competitive_facts is not None) must have real, actor-tagged
        # opponent_decision events on disk, not just a metrics field
        # asserting the opponent acted.
        active_opponent_episodes = {
            e.episode_id for e in result.episodes if e.compatibility_status.value == "SUPPORTED" and e.competitive_facts is not None
        }
        per_episode_opponent_decision_count = {
            eid: sum(
                1
                for e in decision_events
                if e.get("event_type") == "opponent_decision" and e.get("episode_id") == eid and e.get("actor_id")
            )
            for eid in active_opponent_episodes
        }
        missing_opponent_telemetry = sorted(eid for eid in active_opponent_episodes if per_episode_opponent_decision_count.get(eid, 0) == 0)
        opponent_telemetry_present = not missing_opponent_telemetry
        checks["opponent_decision_telemetry_present"] = opponent_telemetry_present
        if not opponent_telemetry_present:
            problems.append(f"active-opponent episodes with no opponent_decision telemetry: {missing_opponent_telemetry}")

    # -- aggregate counts reconcile with raw episodes ----------------------
    summary = result.to_summary_dict()
    recomputed_requested = len(result.episodes)
    recomputed_supported = sum(1 for e in result.episodes if e.compatibility_status.value == "SUPPORTED")
    recomputed_completed = sum(1 for e in result.episodes if e.status.value == "COMPLETED")
    recomputed_successful = sum(1 for e in result.episodes if e.status.value == "COMPLETED" and e.mission_success)
    aggregate_ok = (
        summary["requested"] == recomputed_requested
        and summary["supported"] == recomputed_supported
        and summary["completed"] == recomputed_completed
        and summary["successful"] == recomputed_successful
    )
    checks["aggregate_reconciles_with_raw_episodes"] = aggregate_ok
    if not aggregate_ok:
        problems.append(
            "summary aggregate counts do not match a fresh recomputation from result.episodes: "
            f"summary={{'requested': {summary['requested']}, 'supported': {summary['supported']}, "
            f"'completed': {summary['completed']}, 'successful': {summary['successful']}}} "
            f"recomputed={{'requested': {recomputed_requested}, 'supported': {recomputed_supported}, "
            f"'completed': {recomputed_completed}, 'successful': {recomputed_successful}}}"
        )

    ok = all(checks.values())
    return IntegrityReport(ok=ok, checks=checks, problems=problems)
