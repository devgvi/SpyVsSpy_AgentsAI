"""M5F.1: rule-based behavioral classification and per-profile aggregation
- spec sections 16-24. Never a subjective/LLM-judged score (spec sections
26, 54); rationale text is never the sole classifier (spec section 22),
only ever corroborating evidence a human reading the trace can check.
"""

from __future__ import annotations

from typing import Any

MISSION_DIRECTED = "MISSION_DIRECTED"
OPPONENT_DIRECTED = "OPPONENT_DIRECTED"
SAFETY_DIRECTED = "SAFETY_DIRECTED"
OTHER = "OTHER"

_MISSION_PROGRESS_CATEGORIES = frozenset(
    {"NAVIGATION_PROGRESS", "DISCOVERY_PROGRESS", "REQUIREMENT_PROGRESS", "ITEM_PROGRESS", "MISSION_PROGRESS"}
)
_SAFETY_ACTIONS = frozenset({"inspect", "disarm"})


def classify_decision(record: dict[str, Any], contested_targets: frozenset[str] | None = None) -> list[str]:
    """Spec section 22's four rule-based categories, from objective
    fields only:
    - MISSION_DIRECTED: the decision's own `progress_delta` (already
      computed by `decision_trace.classify_progress_delta`) shows real
      mission-relevant progress.
    - OPPONENT_DIRECTED: a trap placed while the opponent is/was visible,
      or a target that is independently known to be `contested` (the
      other actor targeted the same object within the same duel's
      contested-objective window - see `duel_trace.contested_objective_events`).
    - SAFETY_DIRECTED: INSPECT or DISARM (spec section 22's own examples).
    A decision may carry more than one tag; `OTHER` only when none apply.
    """
    tags: list[str] = []
    progress = record.get("progress_delta") or []
    if any(p in _MISSION_PROGRESS_CATEGORIES for p in progress):
        tags.append(MISSION_DIRECTED)

    action = (record.get("final_action") or "").lower()
    target = record.get("final_target")
    opponent_relevant = bool(record.get("opponent_visible_before") or record.get("opponent_visible_after"))
    contested = bool(contested_targets and target in contested_targets)
    if (action == "place_trap" and opponent_relevant) or contested:
        tags.append(OPPONENT_DIRECTED)

    if action in _SAFETY_ACTIONS:
        tags.append(SAFETY_DIRECTED)

    return tags or [OTHER]


def annotate_behavior_tags(records: list[dict[str, Any]], contested_targets: frozenset[str] | None = None) -> None:
    for r in records:
        r["behavior_tags"] = classify_decision(r, contested_targets)


def return_to_mission_latencies(records: list[dict[str, Any]]) -> list[int | None]:
    """Spec section 24: for every OPPONENT_DIRECTED decision, how many
    decisions until the next MISSION_DIRECTED one - `None` if the actor
    never returns to mission progress before the trace ends."""
    ordered = sorted(records, key=lambda r: r["decision_number"])
    latencies: list[int | None] = []
    for i, r in enumerate(ordered):
        if OPPONENT_DIRECTED not in r.get("behavior_tags", []):
            continue
        latency = None
        for later in ordered[i + 1:]:
            if MISSION_DIRECTED in later.get("behavior_tags", []):
                latency = later["decision_number"] - r["decision_number"]
                break
        latencies.append(latency)
    return latencies


# -- mirror-pair generation (spec sections 31-33) -----------------------------


def generate_mirror_pairs(matchups: list[tuple[str, str]], seeds: list[int]) -> list[dict[str, Any]]:
    """Spec section 32-33: for every (profile_a, profile_b) matchup and
    every seed, both actor-assignment mirrors - controls for actor/color/
    spawn asymmetry, never assumed away."""
    pairs = []
    for profile_a, profile_b in matchups:
        for seed in seeds:
            pairs.append({"seed": seed, "black_personality": profile_a, "white_personality": profile_b, "matchup": (profile_a, profile_b), "mirror": "A"})
            pairs.append({"seed": seed, "black_personality": profile_b, "white_personality": profile_a, "matchup": (profile_a, profile_b), "mirror": "B"})
    return pairs


# -- per-profile aggregation (spec sections 20-21, 49) -------------------------


def aggregate_profile_metrics(records_by_profile: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """One descriptive-only row per profile (spec section 49) - action-
    distribution shares, trap/inspect/disarm rates, repeat rate, and
    return-to-mission latency, pooled across every episode that profile
    appeared in (mirrored or not). No overall score, no ranking."""
    result: dict[str, dict[str, Any]] = {}
    for profile, records in records_by_profile.items():
        n = len(records)
        if n == 0:
            result[profile] = {"decision_count": 0}
            continue
        mission_n = sum(1 for r in records if MISSION_DIRECTED in r.get("behavior_tags", []))
        opponent_n = sum(1 for r in records if OPPONENT_DIRECTED in r.get("behavior_tags", []))
        safety_n = sum(1 for r in records if SAFETY_DIRECTED in r.get("behavior_tags", []))
        trap_n = sum(1 for r in records if (r.get("final_action") or "").lower() == "place_trap")
        inspect_n = sum(1 for r in records if (r.get("final_action") or "").lower() == "inspect")
        disarm_n = sum(1 for r in records if (r.get("final_action") or "").lower() == "disarm")
        repeat_n = sum(1 for r in records if (r.get("repeat_count") or 0) > 0)
        raw_latencies = return_to_mission_latencies(records)
        latencies = [l for l in raw_latencies if l is not None]
        never_returned = sum(1 for l in raw_latencies if l is None)

        result[profile] = {
            "decision_count": n,
            "mission_directed_share": round(mission_n / n, 3),
            "opponent_directed_share": round(opponent_n / n, 3),
            "safety_directed_share": round(safety_n / n, 3),
            "trap_placement_rate": round(trap_n / n, 3),
            "inspect_rate": round(inspect_n / n, 3),
            "disarm_rate": round(disarm_n / n, 3),
            "repeat_rate": round(repeat_n / n, 3),
            "median_return_to_mission_latency": (sorted(latencies)[len(latencies) // 2] if latencies else None),
            "opponent_directed_decisions_with_no_return": never_returned,
        }
    return result
