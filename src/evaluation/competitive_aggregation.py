"""M5E.2: pure, read-only aggregation over an already-completed
ExperimentResult's episodes - three-way (A/B/C) per-seed pairing,
technical-failure exclusion from competitive denominators (spec section
28), and first-escape/both-escaped/trap-total rollups (spec sections 9,
12, 24). Never touches gameplay; never invents a composite score (spec
section 25).
"""

from __future__ import annotations

from typing import Any

# Episode statuses spec section 8/28 call a "technical failure" - anything
# that stopped the episode for a harness/provider/runtime reason rather
# than a legitimate gameplay outcome (a natural mission-deadline/decision-
# cap COMPLETED/DECISION_LIMIT_REACHED is not a technical failure by
# itself; only these three EpisodeStatus values are).
TECHNICAL_FAILURE_STATUSES = {"ABORTED", "TIMED_OUT", "FAILED"}


def is_technical_failure(episode: Any) -> bool:
    status = episode.status.value if hasattr(episode.status, "value") else episode.status
    return status in TECHNICAL_FAILURE_STATUSES


def filter_technically_valid_episodes(
    episodes: list[Any], technical_status_by_episode_id: dict[str, str] | None = None
) -> list[Any]:
    """Spec section 28: a technical failure is excluded from direct
    competitive-outcome denominators, never silently converted into an
    opponent win. This is the one function every competitive-outcome
    aggregator below should filter through first.

    Without `technical_status_by_episode_id`, "technical failure" means
    only `is_technical_failure` (an ABORTED/TIMED_OUT/FAILED
    `EpisodeStatus`) - the narrow, harness-status-only definition. Pass a
    `{episode_id: classify_episode_technical_status(...)}` map (spec
    section 8's richer four-way classification, which also catches a
    DECISION_LIMIT_REACHED episode whose budget was actually consumed by
    persistent provider/runtime failures rather than real gameplay) to use
    that instead - the real M5E.2 live analysis does this; unit tests that
    only care about the harness-status axis can omit it.
    """
    if technical_status_by_episode_id is not None:
        return [e for e in episodes if technical_status_by_episode_id.get(e.episode_id) != TECHNICAL_FAILURE]
    return [e for e in episodes if not is_technical_failure(e)]


def three_way_seed_pairing(episodes: list[Any], variant_ids: tuple[str, str, str]) -> list[dict[str, Any]]:
    """One row per seed present (as a SUPPORTED episode) for *every* one
    of the three variants - spec section 23's per-seed A/B/C view. Each
    variant's own cell reports technical_status, competitive_facts, and
    game_facts untouched (never re-derived), so a reader can see exactly
    what each variant did on that shared seed."""
    by_variant_seed: dict[str, dict[int, Any]] = {
        vid: {e.seed: e for e in episodes if e.variant_id == vid and e.compatibility_status.value == "SUPPORTED"}
        for vid in variant_ids
    }
    shared_seeds = set(by_variant_seed[variant_ids[0]])
    for vid in variant_ids[1:]:
        shared_seeds &= set(by_variant_seed[vid])

    rows = []
    for seed in sorted(shared_seeds):
        row: dict[str, Any] = {"seed": seed}
        for vid in variant_ids:
            ep = by_variant_seed[vid][seed]
            row[vid] = {
                "technical_status": ep.status.value,
                "is_technical_failure": is_technical_failure(ep),
                "decision_count": ep.decision_count,
                "game_facts": ep.game_facts,
                "competitive_facts": ep.competitive_facts,
            }
        rows.append(row)
    return rows


def aggregate_first_escape_counts(episodes: list[Any], technical_status_by_episode_id: dict[str, str] | None = None) -> dict[str, int]:
    """Counts of which side's `first_escape_actor` won the race, over the
    technically-valid episodes only (spec section 28) - `None` (neither
    escaped) counted separately, never silently folded into either side."""
    valid = filter_technically_valid_episodes(episodes, technical_status_by_episode_id)
    counts = {"agent_first": 0, "opponent_first": 0, "neither": 0}
    for ep in valid:
        facts = ep.competitive_facts or {}
        first = facts.get("first_escape_actor")
        if first is None:
            counts["neither"] += 1
        elif first == "white_spy":
            counts["agent_first"] += 1
        else:
            counts["opponent_first"] += 1
    return counts


def aggregate_both_escaped_count(episodes: list[Any], technical_status_by_episode_id: dict[str, str] | None = None) -> int:
    valid = filter_technically_valid_episodes(episodes, technical_status_by_episode_id)
    return sum(1 for ep in valid if (ep.competitive_facts or {}).get("both_escaped") is True)


GAMEPLAY_COMPLETION = "gameplay_completion"
DEADLINE_COMPLETION = "deadline_completion"
DECISION_CAP_COMPLETION = "decision_cap_completion"
TECHNICAL_FAILURE = "technical_failure"

# An episode that never technically aborted can still have spent most of
# its decision budget on fallback WAITs from real provider/runtime
# failures rather than genuine gameplay (M5E.2 spec section 8's four
# categories are mutually exclusive - a decision-cap completion whose
# *cause* was persistent technical failure is reported as a technical
# failure, not as ordinary decision-cap gameplay). Majority-invalid is the
# line: below it, more decisions failed technically than the model ever
# got to actually play.
MIN_VALID_DECISION_RATIO_FOR_GAMEPLAY = 0.5


def classify_episode_technical_status(episode: Any, long_horizon_stability: Any | None = None) -> str:
    """Spec section 8's four-way classification. `long_horizon_stability`
    is that episode's own `compute_long_horizon_stability(...)` result
    (pass None for a variant with no LLM decisions at all, e.g. the
    scripted reference - it can never be technically failed this way)."""
    if is_technical_failure(episode):
        return TECHNICAL_FAILURE
    status = episode.status.value if hasattr(episode.status, "value") else episode.status
    if long_horizon_stability is not None and long_horizon_stability.evaluated_agent_decisions > 0:
        ratio = long_horizon_stability.valid_decisions / long_horizon_stability.evaluated_agent_decisions
        if ratio < MIN_VALID_DECISION_RATIO_FOR_GAMEPLAY:
            return TECHNICAL_FAILURE
    if status == "DECISION_LIMIT_REACHED":
        return DECISION_CAP_COMPLETION
    if status == "COMPLETED" and episode.mission_success:
        return GAMEPLAY_COMPLETION
    if status == "COMPLETED":
        return DEADLINE_COMPLETION
    return TECHNICAL_FAILURE


TRAP_TOTAL_FIELDS = (
    "agent_traps_placed",
    "opponent_traps_placed",
    "agent_triggered_opponent_traps",
    "opponent_triggered_agent_traps",
    "agent_stuns",
    "opponent_stuns",
)


def aggregate_trap_totals(episodes: list[Any], technical_status_by_episode_id: dict[str, str] | None = None) -> dict[str, int]:
    """Sums each of `TRAP_TOTAL_FIELDS` across every technically-valid
    episode's own `competitive_facts` - a pure rollup, never a re-derivation
    from raw events (those were already authoritative when
    `_extract_competitive_facts` computed them per episode)."""
    valid = filter_technically_valid_episodes(episodes, technical_status_by_episode_id)
    totals = {field_name: 0 for field_name in TRAP_TOTAL_FIELDS}
    for ep in valid:
        facts = ep.competitive_facts or {}
        for field_name in TRAP_TOTAL_FIELDS:
            totals[field_name] += facts.get(field_name) or 0
    return totals
