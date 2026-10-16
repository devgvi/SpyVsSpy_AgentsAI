"""M6A.2: pure, network-free full-episode trajectory-analysis functions
(spec sections 14-23, 41, 53-54) - first-divergence detection between a
paired Native/Strands episode, pre/post-divergence classification,
trajectory trace generation, and pair/overall classification.

Core principle (spec section 1): after the first divergent decision,
later state differences cannot be treated as if both runtimes are still
seeing the same world - `find_first_divergence` exists specifically to
draw that line, and nothing downstream of it compares later decisions as
same-state choices.

All functions here operate on already-emitted `llm_decision` telemetry
events (both Native and Strands already emit `observation_hash`,
`legal_actions`, and the full `observation`/`executed_action` dicts
identically - reused directly rather than recomputed by hand, since a
hand-reconstruction risks a subtle convention mismatch, e.g. M6A.1's own
`prompt_hash` bug)."""

from __future__ import annotations

from typing import Any

from src.evaluation.runtime_variance_analysis import legal_actions_hash, stable_hash, trial_signature

# -- "actor-visible GameState" (spec section 15) -----------------------------
#
# Deliberate scope decision, documented rather than silently narrowed: this
# project's own `Observation` (see `src/observation.py`'s module docstring)
# already *is* the authoritative actor-visible projection of `GameState` -
# there is no separate, pre-existing "visible GameState but not Observation"
# concept anywhere else in the codebase. Rather than hand-roll a second,
# parallel state serializer (a real risk of silently omitting a field
# `Observation` already gets right), `world_state_hash` below reuses the
# same `observation_hash`-equivalent computation. `legal_actions_hash` and
# `memory_summary_hash` cover the other two same-state preconditions.


def event_observation_hash(event: dict[str, Any]) -> str | None:
    """Both runtimes already compute and emit this identically
    (`hashlib.sha256(json.dumps(observation_payload, sort_keys=True))` on
    the full serialized observation, time field included) - see
    `src/llm_controller.py`'s and `src/agents/strands_agent.py`'s own
    `_request_action`/`_invoke_strands`. Reused verbatim rather than
    recomputed, so this can never itself introduce a convention mismatch."""
    return event.get("observation_hash")


def event_legal_actions_hash(event: dict[str, Any]) -> str:
    return legal_actions_hash(event.get("legal_actions") or [])


_MEMORY_SUMMARY_FIELDS = (
    "visited_rooms_count", "known_requirements_count", "known_empty_objects_count",
    "recent_events_count", "recent_failures_count", "recent_decisions_count", "recent_repetitions_count",
)


def memory_summary_hash(event: dict[str, Any]) -> str:
    """No raw `AgentMemory.to_compact_dict()` is emitted to telemetry
    (only these count summaries - see `emit_llm_decision`'s own field
    list), so this is a documented proxy for "memory state summary
    equality", not literal full-content equality. Sufficient for this
    milestone's purpose: detecting whether the two runtimes' own within-
    episode memory has already drifted apart by this decision."""
    return stable_hash({f: event.get(f) for f in _MEMORY_SUMMARY_FIELDS})


def decision_state_key(event: dict[str, Any]) -> tuple[str | None, str, str]:
    return (event_observation_hash(event), event_legal_actions_hash(event), memory_summary_hash(event))


# -- decision-context categorization (spec section 27) -----------------------
#
# A live-episode approximation of M6A.1's hand-constructed frozen-snapshot
# categories - built from whatever a real `llm_decision` event's own
# `legal_actions`/`observation` fields show, since live episodes present
# naturally-arising contexts rather than deliberately engineered ones.

SIMPLE_NAVIGATION = "SIMPLE_NAVIGATION"
INTERACTION_AVAILABLE = "INTERACTION_AVAILABLE"
MULTIPLE_LEGAL_TARGETS = "MULTIPLE_LEGAL_TARGETS"
LOCKED_OBJECT = "LOCKED_OBJECT"
TRAP_PLACEMENT_OPPORTUNITY = "TRAP_PLACEMENT_OPPORTUNITY"
DETECTED_TRAP = "DETECTED_TRAP"

# M6A.1's own live Phase B findings (spec section 28) - fixed prior
# evidence, not re-derived from this milestone's own data.
_RUNTIME_EFFECT_SUPPORTED_CONTEXTS = frozenset({MULTIPLE_LEGAL_TARGETS})
_INTRINSIC_VARIANCE_POSSIBLE_CONTEXTS = frozenset({TRAP_PLACEMENT_OPPORTUNITY})

RUNTIME_EFFECT_SUPPORTED_CONTEXT = "RUNTIME_EFFECT_SUPPORTED_CONTEXT"
INTRINSIC_VARIANCE_POSSIBLE = "INTRINSIC_VARIANCE_POSSIBLE"


def categorize_decision_context(event: dict[str, Any]) -> str:
    legal = event.get("legal_actions") or []
    actions_present = {a.get("action") for a in legal}
    if "place_trap" in actions_present:
        return TRAP_PLACEMENT_OPPORTUNITY
    if "disarm" in actions_present:
        return DETECTED_TRAP

    interact_targets = {a.get("target_id") for a in legal if a.get("action") in ("interact", "inspect")}
    if interact_targets:
        obs = event.get("observation") or {}
        room = obs.get("room") or {}
        objs = room.get("objects") if isinstance(room, dict) else None
        nearby_id = (obs.get("self") or {}).get("near_object_id")
        nearby_obj = next((o for o in (objs or []) if o.get("id") == nearby_id), None) if objs else None
        if nearby_obj and nearby_obj.get("locked") and not nearby_obj.get("can_unlock"):
            return LOCKED_OBJECT
        move_targets = {a.get("target_id") for a in legal if a.get("action") == "move"}
        if len(move_targets) > 1:
            return MULTIPLE_LEGAL_TARGETS
        return INTERACTION_AVAILABLE
    return SIMPLE_NAVIGATION


def tag_provider_variance_context(category: str) -> str | None:
    if category in _RUNTIME_EFFECT_SUPPORTED_CONTEXTS:
        return RUNTIME_EFFECT_SUPPORTED_CONTEXT
    if category in _INTRINSIC_VARIANCE_POSSIBLE_CONTEXTS:
        return INTRINSIC_VARIANCE_POSSIBLE
    return None


# -- first divergence (spec sections 14-17) ----------------------------------

PAIR_INTEGRITY_BROKEN = "PAIR_INTEGRITY_BROKEN"
DIVERGENCE_FOUND = "DIVERGENCE_FOUND"
NO_DIVERGENCE = "NO_DIVERGENCE"


def find_first_divergence(native_events: list[dict[str, Any]], strands_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Walks paired decisions index-by-index (both start from the same
    seed/world/personality/opponent, so decision 0 must be same-state by
    construction). At each index: same-state precondition first (spec
    section 15) - if it fails at index 0, the pair itself is invalid
    (never a "divergence"); if it fails at index > 0, a divergence the
    action-signature check below should already have caught was missed,
    recorded as `state_drift` rather than silently treated as ordinary
    same-state divergence. Otherwise compare canonical decision
    signatures - the first mismatch is the first divergence."""
    n = min(len(native_events), len(strands_events))
    for i in range(n):
        native_key = decision_state_key(native_events[i])
        strands_key = decision_state_key(strands_events[i])
        if native_key != strands_key:
            if i == 0:
                return {"status": PAIR_INTEGRITY_BROKEN, "decision_index": 0, "reason": "state mismatch at decision 0 - seed/world/config pairing itself is broken"}
            return {
                "status": DIVERGENCE_FOUND, "decision_index": i - 1, "method": "state_drift_after_matching_action",
                "context_category": categorize_decision_context(native_events[i - 1]),
                "native_action": trial_signature({"action": native_events[i - 1].get("executed_action", {}).get("action"), "target": native_events[i - 1].get("executed_action", {}).get("target_id"), "trap_type": native_events[i - 1].get("executed_action", {}).get("trap_type")}),
                "strands_action": trial_signature({"action": strands_events[i - 1].get("executed_action", {}).get("action"), "target": strands_events[i - 1].get("executed_action", {}).get("target_id"), "trap_type": strands_events[i - 1].get("executed_action", {}).get("trap_type")}),
            }

        native_action = native_events[i].get("executed_action") or {}
        strands_action = strands_events[i].get("executed_action") or {}
        native_sig = trial_signature({"action": native_action.get("action"), "target": native_action.get("target_id"), "trap_type": native_action.get("trap_type")})
        strands_sig = trial_signature({"action": strands_action.get("action"), "target": strands_action.get("target_id"), "trap_type": strands_action.get("trap_type")})
        if native_sig != strands_sig:
            category = categorize_decision_context(native_events[i])
            return {
                "status": DIVERGENCE_FOUND, "decision_index": i, "method": "action_diff",
                "context_category": category, "provider_variance_tag": tag_provider_variance_context(category),
                "native_action": native_sig, "strands_action": strands_sig,
            }

    return {"status": NO_DIVERGENCE, "decision_index": None, "native_decisions": len(native_events), "strands_decisions": len(strands_events)}


# -- trajectory trace (spec section 50) --------------------------------------


def build_trajectory_trace(events: list[dict[str, Any]], runtime_label: str, first_divergence_index: int | None) -> list[dict[str, Any]]:
    trace = []
    for i, e in enumerate(events):
        action = e.get("executed_action") or {}
        obs = e.get("observation") or {}
        room = obs.get("room") or {}
        trace.append({
            "decision_index": i,
            "room": room.get("id") if isinstance(room, dict) else None,
            "target": action.get("target_id"),
            "action": action.get("action"),
            "runtime": runtime_label,
            "is_first_divergence": (first_divergence_index is not None and i == first_divergence_index),
        })
    return trace


# -- pair classification (spec section 53) -----------------------------------

NO_DIVERGENCE_CLASS = "NO_DIVERGENCE"
EARLY_DIVERGENCE_HIGH_IMPACT = "EARLY_DIVERGENCE_HIGH_IMPACT"
EARLY_DIVERGENCE_LOW_IMPACT = "EARLY_DIVERGENCE_LOW_IMPACT"
LATE_DIVERGENCE = "LATE_DIVERGENCE"
RECONVERGED = "RECONVERGED"
TECHNICALLY_INVALID = "TECHNICALLY_INVALID"


def _outcome_impact(native_outcome: dict[str, Any], strands_outcome: dict[str, Any]) -> str:
    """`'high'` (completion or escape itself differs - the two outcomes
    that actually decide the episode), `'low'` (only item count or room
    coverage differs, completion/escape the same), or `'none'` (every
    tracked outcome matches). Real bug found and fixed while writing this
    milestone's own tests: an earlier single-boolean version made
    `EARLY_DIVERGENCE_LOW_IMPACT` unreachable dead code in `classify_pair`
    below (its `not high_impact` early-return for RECONVERGED meant the
    "else" branch offering LOW_IMPACT could never run) - a 3-way tier
    fixes that."""
    major_diff = (
        native_outcome.get("mission_success") != strands_outcome.get("mission_success")
        or native_outcome.get("escape_success") != strands_outcome.get("escape_success")
    )
    if major_diff:
        return "high"
    minor_diff = (
        native_outcome.get("mission_items_acquired") != strands_outcome.get("mission_items_acquired")
        or set(native_outcome.get("rooms_visited") or []) != set(strands_outcome.get("rooms_visited") or [])
    )
    return "low" if minor_diff else "none"


def classify_pair(
    divergence: dict[str, Any],
    native_outcome: dict[str, Any],
    strands_outcome: dict[str, Any],
    native_decision_count: int,
    strands_decision_count: int,
    technically_valid: bool = True,
) -> str:
    """Documented, fixed thresholds (spec section 53) - decided before
    looking at any live pair's own data: "early" = at or before the
    midpoint of the shorter paired episode's own decision count."""
    if not technically_valid or divergence.get("status") == PAIR_INTEGRITY_BROKEN:
        return TECHNICALLY_INVALID
    if divergence.get("status") == NO_DIVERGENCE:
        return NO_DIVERGENCE_CLASS

    shorter_len = min(native_decision_count, strands_decision_count) or 1
    idx = divergence.get("decision_index") or 0
    is_early = idx <= (shorter_len // 2)
    impact = _outcome_impact(native_outcome, strands_outcome)

    if impact == "none":
        return RECONVERGED
    if is_early:
        return EARLY_DIVERGENCE_HIGH_IMPACT if impact == "high" else EARLY_DIVERGENCE_LOW_IMPACT
    return LATE_DIVERGENCE


# -- overall classification (spec section 54) --------------------------------

RUNTIME_TRAJECTORY_EFFECT_SUPPORTED = "RUNTIME_TRAJECTORY_EFFECT_SUPPORTED"
RUNTIME_EFFECT_LOCAL_BUT_TRAJECTORY_SMALL = "RUNTIME_EFFECT_LOCAL_BUT_TRAJECTORY_SMALL"
INTRINSIC_VARIANCE_DOMINATES = "INTRINSIC_VARIANCE_DOMINATES"
RUNTIME_EFFECT_CONTEXT_DEPENDENT = "RUNTIME_EFFECT_CONTEXT_DEPENDENT"
TECHNICAL_COST_DIFFERENCE_ONLY = "TECHNICAL_COST_DIFFERENCE_ONLY"
OVERALL_MIXED = "MIXED"


def classify_overall_trajectory(pair_classifications: list[str], divergence_context_tags: list[str | None], cost_differs: bool) -> str:
    """Priority-ordered, documented before looking at live pair data
    (spec section 29: a single divergent episode never establishes a
    general effect - this looks at recurrence across the whole pair set)."""
    n = len(pair_classifications)
    if n == 0:
        return OVERALL_MIXED

    high_impact = pair_classifications.count(EARLY_DIVERGENCE_HIGH_IMPACT)
    local_only = sum(1 for c in pair_classifications if c in (EARLY_DIVERGENCE_LOW_IMPACT, RECONVERGED, LATE_DIVERGENCE))
    no_divergence = pair_classifications.count(NO_DIVERGENCE_CLASS)
    supported_context_hits = divergence_context_tags.count(RUNTIME_EFFECT_SUPPORTED_CONTEXT)
    intrinsic_context_hits = divergence_context_tags.count(INTRINSIC_VARIANCE_POSSIBLE)

    if high_impact >= 2 or (high_impact >= 1 and supported_context_hits >= 1 and (high_impact / n) >= 0.4):
        return RUNTIME_TRAJECTORY_EFFECT_SUPPORTED
    if high_impact == 0 and local_only > 0:
        return RUNTIME_EFFECT_LOCAL_BUT_TRAJECTORY_SMALL
    if no_divergence == n:
        return TECHNICAL_COST_DIFFERENCE_ONLY if cost_differs else INTRINSIC_VARIANCE_DOMINATES
    if high_impact == 0 and local_only == 0 and intrinsic_context_hits > supported_context_hits:
        return INTRINSIC_VARIANCE_DOMINATES
    return RUNTIME_EFFECT_CONTEXT_DEPENDENT
