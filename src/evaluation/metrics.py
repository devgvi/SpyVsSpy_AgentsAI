"""M5D: pure metric computation over raw traces.

`compute_metrics` never touches the engine, GameState, Observation, or
gameplay - it only reads what EpisodeRunner already extracted:

- `decision_events`: this episode's own raw `llm_decision` telemetry
  events (M5A/M5B.1 fields - initial/repair/final validity, fallback,
  action_source, latency, tokens). Unbounded and complete for the episode.
- `memory_snapshot`: the actor's own `AgentMemory.to_compact_dict()` at
  episode end - legitimate, authoritative-derived context. Some fields
  (recent_interactions/recent_failures/recent_repetitions) are bounded to
  memory's own recent-history window (see agent_memory.py), so counts
  derived from them undercount on episodes with more interactions than
  that window holds - documented, not hidden.
- `game_facts`: engine-truth counts EpisodeRunner already extracted from
  GameState/event_log after the episode ended (mission outcome, room/trap/
  interaction counts unbounded across the whole episode).
- `action_lifecycle_events` (M5D.2/M5D.3): this episode's own raw
  `action_lifecycle` telemetry events - one per accepted action reaching a
  terminal boundary (a different, coarser granularity than `decision_events`
  - one MOVE decision may span many ticks before producing one of these).
  Optional/defaulted for backward compatibility with pre-M5D.2 callers.

Every metric here must be traceable back to one of these inputs - never
invented, never affecting gameplay, and safe to recompute later from the
same raw trace (spec section 21-22).
"""

from __future__ import annotations

import json
from typing import Any

# M5D.5: "out_of_range" (src/llm_controller.py's validate_model_output) is
# a more specific split of what used to be a generic "illegal_target" -
# still counted here so the pre-existing broad `illegal_actions` metric
# doesn't silently under-count now that the common out-of-range case has
# its own category (see out_of_range_model_actions for the specific count).
ILLEGAL_ERROR_CATEGORIES = {"illegal_action", "illegal_target", "out_of_range"}


def _latency_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "count": n,
        "mean": sum(ordered) / n,
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_metrics(
    decision_events: list[dict[str, Any]] | None,
    memory_snapshot: dict[str, Any] | None,
    game_facts: dict[str, Any] | None,
    action_lifecycle_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    decision_events = decision_events or []
    memory_snapshot = memory_snapshot or {}
    game_facts = game_facts or {}
    action_lifecycle_events = action_lifecycle_events or []

    # -- section 15: decision/validity metrics (M5B.1 semantics, preserved) --
    decisions_requested = len(decision_events)
    first_pass_valid = sum(1 for e in decision_events if e.get("initial_model_output_valid"))
    first_pass_invalid = decisions_requested - first_pass_valid
    repair_attempts = sum(1 for e in decision_events if e.get("repair_attempted"))
    repair_successes = sum(1 for e in decision_events if e.get("repair_attempted") and e.get("repair_valid"))
    final_valid_llm_actions = sum(1 for e in decision_events if e.get("final_model_action_valid"))
    fallbacks = sum(1 for e in decision_events if e.get("fallback_used"))
    explicit_waits = sum(
        1
        for e in decision_events
        if e.get("final_model_action_valid") and (e.get("validated_action") or {}).get("action") == "wait"
    )
    invalid_actions = sum(1 for e in decision_events if not e.get("final_model_action_valid"))
    illegal_actions = sum(
        1
        for e in decision_events
        if e.get("initial_error_type") in ILLEGAL_ERROR_CATEGORIES or e.get("repair_error_type") in ILLEGAL_ERROR_CATEGORIES
    )
    stale_actions = sum(
        1 for e in decision_events if e.get("initial_error_type") == "stale_target" or e.get("repair_error_type") == "stale_target"
    )
    # Each decision made exactly one real model/runtime call, plus one more
    # if a repair was attempted - true for both Native (LLMController) and
    # Strands (StrandsAgentAdapter), which both cap repair at one attempt.
    runtime_invocations = decisions_requested + repair_attempts

    # -- section 16: runtime/model metrics -------------------------------
    latencies = [e["latency_ms"] for e in decision_events if e.get("latency_ms") is not None]
    input_tokens = [e["input_tokens"] for e in decision_events if e.get("input_tokens") is not None]
    output_tokens = [e["output_tokens"] for e in decision_events if e.get("output_tokens") is not None]
    runtime_errors = sum(1 for e in decision_events if (e.get("error_type") or "") == "strands_runtime_error")
    provider_errors = sum(1 for e in decision_events if e.get("error_type") == "provider_error")
    tool_calls = sum((e.get("runtime_metadata") or {}).get("tool_call_count", 0) for e in decision_events)

    # M5D.6 spec sections 25-26: a provider timeout is not a model
    # reasoning failure - classified separately from other provider/
    # runtime errors and from invalid-model-output fallbacks, so an
    # episode's `max_runtime_failures_exceeded` abort can be attributed to
    # its actual contributing causes rather than one undifferentiated
    # fallback count.
    def _is_timeout(e: dict[str, Any]) -> bool:
        text = f"{e.get('error_message') or ''} {(e.get('raw_response') or {}).get('error') or ''}".lower()
        return "timeout" in text or "timed out" in text

    provider_timeouts = sum(1 for e in decision_events if e.get("error_type") == "provider_error" and _is_timeout(e))
    runtime_exceptions = sum(
        1
        for e in decision_events
        if e.get("fallback_used") and e.get("error_type") in {"provider_error", "strands_runtime_error"} and not _is_timeout(e)
    )
    invalid_model_output_fallbacks = sum(
        1
        for e in decision_events
        if e.get("fallback_used") and e.get("error_type") not in {"provider_error", "strands_runtime_error"}
    )

    # -- section 17: navigation facts (unique_rooms_visited is unbounded -
    # AgentMemory.exploration.visited_rooms only ever grows, never trimmed) --
    unique_rooms_visited = game_facts.get("unique_rooms_visited")
    move_actions = sum(
        1
        for e in decision_events
        if e.get("final_model_action_valid") and (e.get("validated_action") or {}).get("action") == "move"
    )

    # -- section 18: interaction facts, from the bounded memory snapshot --
    recent_interactions = memory_snapshot.get("recent_interactions", [])
    recent_failures = memory_snapshot.get("recent_failures", [])
    inspect_actions = sum(1 for r in recent_interactions if r.get("action") == "inspect")
    search_actions = sum(1 for r in recent_interactions if r.get("action") == "search")
    successful_searches = sum(1 for r in recent_interactions if r.get("action") == "search" and r.get("result") == "found_item")
    empty_searches = sum(1 for r in recent_interactions if r.get("action") == "search" and r.get("result") == "empty")
    failed_interactions = len(recent_failures)
    locked_interaction_attempts = sum(1 for r in recent_failures if r.get("reason") == "locked")
    unlock_successes = game_facts.get("door_unlocked_count")

    # -- section 19: memory/repetition facts -----------------------------
    recent_repetitions = memory_snapshot.get("recent_repetitions", [])
    repeated_action_count = sum(max(0, rep.get("count", 0) - 1) for rep in recent_repetitions)
    revisited_failed_target_count = sum(
        1 for rep in recent_repetitions if rep.get("last_result") and str(rep.get("last_result")).startswith("blocked")
    )

    # -- M5D.2/M5D.3 spec section 19: action-execution lifecycle facts,
    # from this episode's own action_lifecycle telemetry (one row per
    # accepted action reaching a terminal boundary - a coarser granularity
    # than decision_events). actions_accepted counts every boundary (each
    # implies an accept happened); actions_started excludes only
    # PATH_UNREACHABLE, the one boundary that fires before a MOVE ever
    # actually starts executing (spec section 13). actions_cancelled is
    # always 0 here - no code path in ActionExecutor currently produces a
    # CANCELLED boundary (spec listed it as a terminal alternative, never
    # wired up), recorded honestly rather than hidden.
    actions_accepted = len(action_lifecycle_events)
    actions_started = sum(1 for e in action_lifecycle_events if e.get("action_terminal_reason") != "path_unreachable")
    actions_completed = sum(1 for e in action_lifecycle_events if e.get("action_status") == "completed")
    actions_failed = sum(1 for e in action_lifecycle_events if e.get("action_status") == "failed")
    actions_interrupted = sum(1 for e in action_lifecycle_events if e.get("action_status") == "interrupted")
    actions_cancelled = sum(1 for e in action_lifecycle_events if e.get("action_status") == "cancelled")
    completed_ticks = [
        e["action_simulation_ticks"] for e in action_lifecycle_events if e.get("action_status") == "completed" and e.get("action_simulation_ticks") is not None
    ]
    completed_durations = [
        e["action_duration_game_time"]
        for e in action_lifecycle_events
        if e.get("action_status") == "completed" and e.get("action_duration_game_time") is not None
    ]
    actions_by_type: dict[str, dict[str, int]] = {}
    for e in action_lifecycle_events:
        bucket = actions_by_type.setdefault(e.get("action", "unknown"), {"completed": 0, "failed": 0, "interrupted": 0})
        status = e.get("action_status")
        if status in bucket:
            bucket[status] += 1
    completed_move_actions = actions_by_type.get("move", {}).get("completed", 0)
    failed_move_actions = actions_by_type.get("move", {}).get("failed", 0)

    # -- M5D.4 spec sections 12, 32-33: interaction affordance/no-op
    # classification, kept distinct from move (that's section 17's
    # navigation block above). "Interaction" here means the four
    # proximity/precondition-gated action types (inspect/interact/
    # place_trap/disarm) - move is reachability-gated, a different
    # legality question entirely (spec section 7).
    INTERACTION_TYPES = {"inspect", "interact", "place_trap", "disarm"}

    def _proposed_action_names(e: dict[str, Any]) -> set[str]:
        names = set()
        raw = e.get("raw_response") or {}
        if isinstance(raw.get("action"), str):
            names.add(raw["action"].strip().lower())
        repair_raw = e.get("repair_raw_response") or {}
        if isinstance(repair_raw.get("action"), str):
            names.add(repair_raw["action"].strip().lower())
        return names

    interaction_proposals = sum(1 for e in decision_events if _proposed_action_names(e) & INTERACTION_TYPES)
    illegal_model_actions = sum(
        1
        for e in decision_events
        if (e.get("initial_error_type") in {"illegal_action", "illegal_target"} or e.get("repair_error_type") in {"illegal_action", "illegal_target"})
    )
    # M5D.5 spec section 5: "out_of_range" is now its own validator category
    # (src/llm_controller.py's validate_model_output), split out of the
    # more generic "illegal_target" - by far the most common real failure
    # (M5D.4/M5D.5 telemetry audits), so it gets its own explicit count
    # rather than being folded into illegal_model_actions above.
    out_of_range_model_actions = sum(
        1
        for e in decision_events
        if (e.get("initial_error_type") == "out_of_range" or e.get("repair_error_type") == "out_of_range")
    )
    unknown_target_actions = sum(
        1
        for e in decision_events
        if (e.get("initial_error_type") == "unknown_target" or e.get("repair_error_type") == "unknown_target")
    )
    legal_interactions = sum(
        1
        for e in decision_events
        if e.get("final_model_action_valid") and (e.get("validated_action") or {}).get("action") in INTERACTION_TYPES
    )
    accepted_interactions = sum(
        actions_by_type.get(a, {}).get("completed", 0)
        + actions_by_type.get(a, {}).get("failed", 0)
        + actions_by_type.get(a, {}).get("interrupted", 0)
        for a in INTERACTION_TYPES
    )
    executed_interactions = sum(actions_by_type.get(a, {}).get("completed", 0) for a in INTERACTION_TYPES)
    # recent_decisions is memory's own bounded (last-10) history - same
    # caveat as recent_interactions/recent_failures elsewhere in this file.
    recent_decisions = memory_snapshot.get("recent_decisions", [])
    successful_interactions = sum(
        1 for d in recent_decisions if d.get("selected_action") in INTERACTION_TYPES and d.get("engine_result") == "success"
    )
    valid_empty_interactions = sum(
        1 for r in recent_interactions if r.get("action") in {"search", "inspect"} and r.get("result") in {"empty", "clear"}
    )
    engine_rejected_actions = sum(
        1
        for d in recent_decisions
        if d.get("selected_action") in INTERACTION_TYPES and (d.get("engine_result") or "").startswith("blocked")
    )
    interaction_success_rate = (successful_interactions / accepted_interactions) if accepted_interactions else None

    # -- M5D.5 spec section 19: recovery metrics, explicit denominators --
    LEGALITY_ERROR_CATEGORIES = {"illegal_action", "illegal_target", "out_of_range", "unknown_target"}
    illegal_first_pass_actions = sum(1 for e in decision_events if e.get("initial_error_type") in LEGALITY_ERROR_CATEGORIES)
    repair_success_rate = (repair_successes / repair_attempts) if repair_attempts else None
    fallback_after_illegal = sum(
        1 for e in decision_events if e.get("fallback_used") and e.get("initial_error_type") in LEGALITY_ERROR_CATEGORIES
    )

    def _action_target(d: dict[str, Any] | None) -> tuple[Any, Any]:
        d = d or {}
        return (d.get("action"), d.get("target_id"))

    same_illegal_action_repeated = sum(
        1
        for e in decision_events
        if e.get("repair_attempted")
        and not e.get("repair_valid")
        and _action_target(e.get("initial_proposal") or e.get("raw_response")) == _action_target(e.get("repair_raw_response"))
    )
    recovered_to_move = sum(
        1
        for e in decision_events
        if e.get("repair_attempted") and e.get("repair_valid") and (e.get("repair_raw_response") or {}).get("action") == "move"
    )
    recovered_to_near_interaction = sum(
        1
        for e in decision_events
        if e.get("repair_attempted")
        and e.get("repair_valid")
        and (e.get("repair_raw_response") or {}).get("action") in INTERACTION_TYPES
        and (e.get("repair_raw_response") or {}).get("target_id") == e.get("repair_near_object_id")
    )

    return {
        # mission (section 14) - from game_facts, engine-truth
        "mission_success": game_facts.get("mission_success"),
        "escape_success": game_facts.get("mission_success"),
        "mission_items_required": game_facts.get("mission_items_required"),
        "mission_items_acquired": game_facts.get("mission_items_acquired"),
        "mission_progress_ratio": (
            game_facts["mission_items_acquired"] / game_facts["mission_items_required"]
            if game_facts.get("mission_items_required")
            else None
        ),
        "deadline_remaining": game_facts.get("deadline_remaining"),
        # decision/validity (section 15)
        "decisions_requested": decisions_requested,
        "runtime_invocations": runtime_invocations,
        "first_pass_valid": first_pass_valid,
        "first_pass_invalid": first_pass_invalid,
        "repair_attempts": repair_attempts,
        "repair_successes": repair_successes,
        "final_valid_llm_actions": final_valid_llm_actions,
        "fallbacks": fallbacks,
        "explicit_waits": explicit_waits,
        "invalid_actions": invalid_actions,
        "illegal_actions": illegal_actions,
        "stale_actions": stale_actions,
        # runtime/model (section 16)
        "model_calls": runtime_invocations,
        "tool_calls": tool_calls,
        "runtime_errors": runtime_errors,
        "provider_errors": provider_errors,
        # M5D.6: episode failure-budget attribution (spec section 26) -
        # these three are disjoint and sum to the episode's total
        # fallback_used count, so `max_runtime_failures_exceeded` can be
        # explained rather than left as one opaque counter.
        "provider_timeouts": provider_timeouts,
        "runtime_exceptions": runtime_exceptions,
        "invalid_model_output_fallbacks": invalid_model_output_fallbacks,
        "latency_ms": _latency_stats(latencies),
        "input_tokens_total": sum(input_tokens) if input_tokens else None,
        "output_tokens_total": sum(output_tokens) if output_tokens else None,
        # navigation (section 17)
        "unique_rooms_visited": unique_rooms_visited,
        "move_actions": move_actions,
        "completed_move_actions": completed_move_actions,
        "failed_move_actions": failed_move_actions,
        # action-execution lifecycle (M5D.2/M5D.3 spec section 19) - from
        # action_lifecycle telemetry, unbounded/complete for the episode
        "actions_accepted": actions_accepted,
        "actions_started": actions_started,
        "actions_completed": actions_completed,
        "actions_failed": actions_failed,
        "actions_interrupted": actions_interrupted,
        "actions_cancelled": actions_cancelled,
        "mean_simulation_ticks_per_completed_action": _mean(completed_ticks),
        "mean_game_time_per_completed_action": _mean(completed_durations),
        "actions_by_type": actions_by_type,
        # M5D.4: interaction affordance/no-op classification and funnel
        # (spec sections 12, 32-33). Denominators are explicit per spec
        # section 32's own example - interaction_success_rate divides by
        # accepted_interactions, never by decisions_requested/proposed.
        "illegal_model_actions": illegal_model_actions,
        "out_of_range_model_actions": out_of_range_model_actions,
        "unknown_target_actions": unknown_target_actions,
        "engine_rejected_actions": engine_rejected_actions,
        "valid_empty_interactions": valid_empty_interactions,
        "successful_interactions": successful_interactions,
        "interaction_success_rate": interaction_success_rate,
        "interaction_funnel": {
            "proposed": interaction_proposals,
            "legal": legal_interactions,
            "accepted": accepted_interactions,
            "executed": executed_interactions,
            "successful": successful_interactions,
        },
        # M5D.5 spec section 19: recovery metrics - explicit denominators
        # (repair_success_rate divides by repair_attempts, never
        # decisions_requested).
        "illegal_first_pass_actions": illegal_first_pass_actions,
        "repair_success_rate": repair_success_rate,
        "same_illegal_action_repeated": same_illegal_action_repeated,
        "fallback_after_illegal": fallback_after_illegal,
        "recovered_to_move": recovered_to_move,
        "recovered_to_near_interaction": recovered_to_near_interaction,
        # interaction (section 18) - bounded to memory's recent window, see module docstring
        "inspect_actions": inspect_actions,
        "search_actions": search_actions,
        "successful_searches": successful_searches,
        "empty_searches": empty_searches,
        "failed_interactions": failed_interactions,
        "locked_interaction_attempts": locked_interaction_attempts,
        "unlock_successes": unlock_successes,
        # memory/repetition (section 19) - memory_snapshot is
        # AgentMemory.to_compact_dict(), which has no size field of its own
        # (that lives on to_telemetry_dict()), so measure the same
        # structure directly rather than requiring a second snapshot shape.
        "memory_size_chars": (len(json.dumps(memory_snapshot, separators=(",", ":"))) if memory_snapshot else None),
        "recent_repetitions_count": len(recent_repetitions),
        "repeated_action_count": repeated_action_count,
        "revisited_failed_target_count": revisited_failed_target_count,
        # trap/competition (section 20) - from game_facts, engine-truth
        "traps_placed": game_facts.get("traps_placed"),
        "traps_triggered": game_facts.get("traps_triggered"),
        "enemy_traps_detected": game_facts.get("enemy_traps_detected"),
        "traps_disarmed": game_facts.get("traps_disarmed"),
        "stuns_suffered": game_facts.get("stuns_suffered"),
        "opponent_encounters": game_facts.get("opponent_encounters"),
        # M5D.2 spec section 19: diagnostic-only facts answering the
        # specific question M5D.1 could not - did the actor ever do
        # anything beyond drift in its spawn room? Never a quality score,
        # and never used by mission_success/escape_success above - purely
        # descriptive, all engine-truth (unbounded, unlike the memory-
        # derived interaction counts in section 18).
        "meaningful_gameplay": {
            "left_starting_room": game_facts.get("left_starting_room"),
            "successful_inspections": game_facts.get("object_inspected_count"),
            "successful_searches": game_facts.get("item_found_count"),
            "door_traversals": game_facts.get("door_opened_count"),
            "mission_progress_events": game_facts.get("mission_progress_events"),
            "meaningful_gameplay_observed": bool(
                game_facts.get("left_starting_room")
                or (game_facts.get("object_inspected_count") or 0) > 0
                or (game_facts.get("item_found_count") or 0) > 0
            ),
        },
    }
