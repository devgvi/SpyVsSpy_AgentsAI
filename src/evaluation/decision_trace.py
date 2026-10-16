"""M5E.4.1: per-decision trace and mission-progress instrumentation.

Never requests or stores hidden chain-of-thought (spec section 1) - the
only "rationale" field used anywhere here is the model's own existing
`reason` string, exactly as already captured by `llm_decision` telemetry
(see `src.llm_controller._request_action` / `StrandsAgentAdapter._emit_telemetry`
- both runtimes populate the identical `reason` key, confirmed by reading
both call sites before writing this module).

Scope honesty (matches M5E.4's own precedent of saying what wasn't
built rather than silently omitting it): `agent_position_summary`,
`legal_actions_summary`, `near_object_id`, `visible_opponent`, and
`detected_traps` at decision time are NOT reconstructed here - no
existing telemetry event records the Observation an LLM call actually
saw (only its response and validation outcome), and rebuilding that
digest from `GameState` after the fact would risk silently disagreeing
with what the model was really shown. `current_room` IS included - it's
cheaply and exactly derivable from the evaluator-side `GameState` snapshot
the new `EpisodeRunner.on_decision` hook already provides (`state.level.room_id_at`),
the same privileged evaluator access `_extract_game_facts` already uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.episode_runner import _extract_game_facts
from src.llm.clients.bedrock_client import redact_bedrock_token

CHECKPOINT_DECISIONS = (10, 20, 30, 40)
REPEAT_WINDOW = 5
NAVIGATION_ACTIONS = {"move_to", "move"}
INTERACTION_ACTIONS = {"inspect", "search", "interact", "disarm", "place_trap"}

# -- progress snapshot (spec section 13) -------------------------------------


def build_progress_snapshot(state: Any, actor_id: str) -> dict[str, Any]:
    """Evaluator-side only (spec section 13) - reuses `_extract_game_facts`
    (already correct/tested since M5D) rather than re-deriving rooms/items/
    traps/opponent-encounter counts from scratch, plus two genuinely new
    fields it doesn't already compute."""
    facts = dict(_extract_game_facts(state, actor_id))
    actor = state.actors[actor_id]
    facts["requirements_known"] = sorted(actor.known_locked_objects.keys())
    facts["exit_unlocked"] = state.door_states.get("exit_door") != "locked"
    try:
        facts["current_room"] = state.level.room_id_at(actor.x, actor.y)
    except Exception:  # pragma: no cover - defensive: never let a display field crash the trace
        facts["current_room"] = None
    # M5E.4.2: every door's real state/connection at this instant -
    # evaluator-side only (never fed to the actor), needed to classify
    # door sequences (open/traverse/re-close) across decisions.
    facts["door_states"] = {obj.id: state.door_states.get(obj.id) for obj in state.level.objects if obj.type == "door"}
    facts["door_connects"] = {obj.id: (list(obj.connects) if obj.connects else None) for obj in state.level.objects if obj.type == "door"}
    return facts


PROGRESS_CATEGORIES = (
    "NAVIGATION_PROGRESS",
    "DISCOVERY_PROGRESS",
    "REQUIREMENT_PROGRESS",
    "ITEM_PROGRESS",
    "MISSION_PROGRESS",
    "OPPONENT_PROGRESS",
    "TRAP_PROGRESS",
)


def classify_progress_delta(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Spec section 14: objective, rule-based, never subjectively weighted -
    a decision may land in more than one category, or none (`NO_PROGRESS`)."""
    categories: list[str] = []
    if after.get("unique_rooms_visited", 0) > before.get("unique_rooms_visited", 0):
        categories.append("NAVIGATION_PROGRESS")
    if len(after.get("requirements_known") or []) > len(before.get("requirements_known") or []) or after.get(
        "knowledge_learned_count", 0
    ) > before.get("knowledge_learned_count", 0):
        categories.append("DISCOVERY_PROGRESS")
    if after.get("door_unlocked_count", 0) > before.get("door_unlocked_count", 0) or after.get(
        "mission_items_acquired", 0
    ) > before.get("mission_items_acquired", 0):
        # `mission_items_acquired` is a count (see `_extract_game_facts`),
        # not a list - compared as an int, not `len()`'d.
        categories.append("REQUIREMENT_PROGRESS")
    if after.get("item_found_count", 0) > before.get("item_found_count", 0):
        categories.append("ITEM_PROGRESS")
    if after.get("mission_progress_events", 0) > before.get("mission_progress_events", 0) or (
        after.get("mission_success") and not before.get("mission_success")
    ):
        categories.append("MISSION_PROGRESS")
    if after.get("opponent_encounters", 0) > before.get("opponent_encounters", 0):
        categories.append("OPPONENT_PROGRESS")
    if (
        after.get("traps_placed", 0) > before.get("traps_placed", 0)
        or after.get("traps_triggered", 0) > before.get("traps_triggered", 0)
        or after.get("traps_disarmed", 0) > before.get("traps_disarmed", 0)
        or after.get("enemy_traps_detected", 0) > before.get("enemy_traps_detected", 0)
    ):
        categories.append("TRAP_PROGRESS")
    return categories or ["NO_PROGRESS"]


# -- repeat/loop detection (spec section 16) ---------------------------------


def annotate_repeats(records: list[dict[str, Any]], window: int = REPEAT_WINDOW) -> None:
    """Mutates each record in place, adding `repeat_count`/`repeat_signature`
    (spec section 16): how many times the same `(action, target)` pair
    appears among the `window` decisions immediately before this one."""
    for i, record in enumerate(records):
        signature = f"{record.get('proposed_action')}:{record.get('proposed_target')}"
        window_start = max(0, i - window)
        prior = records[window_start:i]
        count = sum(1 for p in prior if f"{p.get('proposed_action')}:{p.get('proposed_target')}" == signature)
        record["repeat_signature"] = signature
        record["repeat_count"] = count


# -- action effectiveness (spec section 15) ----------------------------------


def classify_action_effectiveness(record: dict[str, Any]) -> str:
    """Rule-based only (spec section 15) - never a subjective judgment.
    `invalid`: the model's proposal never became a real executed action
    (fell back). `ineffective`: legal and executed, but either a repeat of
    a recent identical proposal or the engine reports it didn't complete.
    `effective`: everything else - a genuine, non-repeated, engine-accepted
    attempt, even one that found nothing (spec section 15's own INSPECT
    example)."""
    if record.get("action_source") == "fallback" or not record.get("final_action_valid"):
        return "invalid"
    if record.get("repeat_count", 0) > 0:
        return "ineffective"
    if record.get("engine_action_status") in ("failed", "interrupted"):
        return "ineffective"
    return "effective"


# -- joining llm_decision + action_lifecycle + progress snapshots -----------


def _index_action_lifecycle_by_decision_id(action_lifecycle_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["decision_id"]: e for e in action_lifecycle_events if e.get("decision_id")}


def build_decision_trace_records(
    decisions: list[dict[str, Any]],
    action_lifecycle_events: list[dict[str, Any]],
    progress_snapshots: dict[int, dict[str, Any]],
    episode_id: str,
    seed: int,
    runtime: str,
) -> list[dict[str, Any]]:
    """Spec section 9's per-decision record, joined from three already-
    recorded sources - never re-derived model output, never a new provider
    call. `progress_snapshots` maps decision_number -> `build_progress_snapshot`
    result (0 = the episode's initial state, captured by `EpisodeRunner.on_decision`)."""
    lifecycle_by_decision = _index_action_lifecycle_by_decision_id(action_lifecycle_events)
    ordered = sorted(decisions, key=lambda d: d.get("decision_number") or 0)
    records: list[dict[str, Any]] = []

    for d in ordered:
        n = d.get("decision_number") or 0
        before = progress_snapshots.get(n - 1, {})
        after = progress_snapshots.get(n, {})
        lifecycle = lifecycle_by_decision.get(d.get("decision_id"))
        initial = d.get("raw_response") or {}
        repair = d.get("repair_raw_response") or {}
        executed = d.get("executed_action") or {}

        record = {
            "episode_id": episode_id,
            "seed": seed,
            "runtime": runtime,
            "decision_id": d.get("decision_id"),
            "decision_number": n,
            "game_time": d.get("game_time"),
            "current_room": after.get("current_room") or before.get("current_room"),
            "mission_progress_before": before,
            "mission_progress_after": after,
            "agent_stated_reason": redact_bedrock_token(initial.get("reason")),
            "proposed_action": initial.get("action"),
            "proposed_target": initial.get("target_id"),
            "proposed_trap_type": initial.get("trap_type"),
            "first_pass_valid": d.get("recovery_outcome") == "FIRST_PASS_VALID",
            "validation_error": d.get("initial_error_type"),
            "repair_attempted": bool(d.get("repair_attempted")),
            "repair_reason": redact_bedrock_token(repair.get("reason")),
            "repair_action": repair.get("action"),
            "repair_target": repair.get("target_id"),
            "repair_valid": d.get("repair_valid"),
            "final_action": executed.get("action"),
            "final_target": executed.get("target_id"),
            # M5F.2: only ever non-None for a valid send_message decision -
            # every other action's `executed_action` dict has no "message"
            # key at all (see llm_controller.py's validated_action
            # construction), so `.get` here is a safe, additive no-op for
            # every pre-M5F.2 record shape.
            "final_message": redact_bedrock_token(executed.get("message")),
            "final_action_valid": d.get("final_model_action_valid"),
            "action_source": d.get("action_source"),
            "engine_action_status": lifecycle.get("action_status") if lifecycle else None,
            "engine_terminal_reason": lifecycle.get("action_terminal_reason") if lifecycle else None,
            "progress_delta": classify_progress_delta(before, after) if before and after else ["NO_PROGRESS"],
            "decision_latency_ms": d.get("latency_ms"),
            "input_tokens": d.get("input_tokens"),
            "output_tokens": d.get("output_tokens"),
        }
        records.append(record)

    annotate_repeats(records)
    for record in records:
        record["action_effectiveness"] = classify_action_effectiveness(record)
    annotate_door_sequences(records)
    annotate_navigation_completion(records)
    return records


# -- navigation completion: geometric vs semantic (M5E.4.3 spec section 27-28) --

GEOMETRIC_ONLY_REASONS = frozenset({"target_reached", "doorway_crossing_incomplete"})


def annotate_navigation_completion(records: list[dict[str, Any]]) -> None:
    """Mutates each record in place with `geometric_target_reached`/
    `semantic_target_reached`/`movement_completion_reason` (spec section
    27) - derived only from the engine's own already-recorded
    `action_lifecycle` boundary (`engine_terminal_reason`, including the
    new `doorway_crossing_incomplete` M5E.4.3 introduced) plus M5E.4.2's
    own `door_sequence_state`, never a new engine call. `None`/`None` for
    a decision with no `action_lifecycle` correlation at all (e.g. a
    fallback)."""
    for record in records:
        reason = record.get("engine_terminal_reason")
        record["movement_completion_reason"] = reason
        if reason is None:
            record["geometric_target_reached"] = None
            record["semantic_target_reached"] = None
            continue
        record["geometric_target_reached"] = reason in GEOMETRIC_ONLY_REASONS
        if record.get("door_sequence_state") is not None:
            # A door-targeting move: semantic success means it actually
            # crossed (TRAVERSED), not merely that the engine reported
            # target_reached (which, pre-M5E.4.3, could itself be a false
            # positive for a crossing move - it no longer is, but the
            # trace still verifies via the real door_sequence_state
            # rather than trusting the label alone).
            record["semantic_target_reached"] = record["door_sequence_state"] == DOOR_TRAVERSED
        else:
            record["semantic_target_reached"] = reason == "target_reached"


def navigation_completion_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec sections 28-30's before/after navigation metrics - crossing
    attempts/successes, the exact geometric-but-not-semantic defect this
    milestone targets, and the M5E.4.2 replacement-loop metric
    (repeated MOVE_TO the same door after it already geometrically
    completed)."""
    crossing_records = [r for r in records if r.get("door_sequence_state") is not None]
    attempts = len(crossing_records)
    successful = sum(1 for r in crossing_records if r.get("semantic_target_reached"))
    failed = sum(1 for r in crossing_records if r.get("geometric_target_reached") and not r.get("semantic_target_reached"))
    geometric_but_not_semantic = sum(
        1 for r in records if r.get("geometric_target_reached") and r.get("semantic_target_reached") is False
    )
    repeated_after_geometric_completion = sum(
        1 for r in crossing_records if r.get("repeat_count", 0) > 0 and r.get("geometric_target_reached")
    )
    return {
        "doorway_crossing_attempts": attempts,
        "successful_room_crossings": successful,
        "failed_room_crossings": failed,
        "crossing_success_rate": round(successful / attempts, 3) if attempts else None,
        "geometric_reached_but_semantic_not_reached": geometric_but_not_semantic,
        "repeated_move_to_same_target_after_geometric_completion": repeated_after_geometric_completion,
    }


# -- door sequences (M5E.4.2 spec sections 24-25, 28-29) ---------------------

DOOR_OPENED = "OPENED"
DOOR_RE_CLOSED = "RE-CLOSED"
DOOR_TRAVERSED = "TRAVERSED"
DOOR_REPEATED_OPEN_INTERACT = "REPEATED_OPEN_INTERACT"
DOOR_NO_CHANGE = "NO_CHANGE"


def _door_target(record: dict[str, Any]) -> str | None:
    """The record's own target, whichever field actually names a door -
    prefers the final/executed target (what really happened) over the
    raw proposal, falling back to the proposal only when no lifecycle/
    execution target is present (e.g. a decision that fell back)."""
    return record.get("final_target") or record.get("proposed_target")


def annotate_door_sequences(records: list[dict[str, Any]]) -> None:
    """Mutates each record in place, adding `door_state_before`/
    `door_state_after`/`door_traversable`/`connected_room`/
    `door_sequence_state` (spec section 24) whenever that decision's own
    target is a door with recorded state - `None` for every non-door
    decision, never a fabricated value."""
    for record in records:
        target = _door_target(record)
        before = record.get("mission_progress_before") or {}
        after = record.get("mission_progress_after") or {}
        door_states_before = before.get("door_states") or {}
        door_states_after = after.get("door_states") or {}

        if target is None or (target not in door_states_before and target not in door_states_after):
            record["door_state_before"] = None
            record["door_state_after"] = None
            record["door_traversable"] = None
            record["connected_room"] = None
            record["door_sequence_state"] = None
            continue

        state_before = door_states_before.get(target)
        state_after = door_states_after.get(target)
        connects = (after.get("door_connects") or {}).get(target) or (before.get("door_connects") or {}).get(target)
        room_before = before.get("current_room")
        room_after = after.get("current_room")
        action = (record.get("final_action") or record.get("proposed_action") or "").lower()

        record["door_state_before"] = state_before
        record["door_state_after"] = state_after
        record["door_traversable"] = state_after == "open"
        record["connected_room"] = connects

        # M5E.4.2 real bug, caught by the pilot run: requiring only
        # `room_after in connects` false-positived on a move that changed
        # room via a completely different door (room_before wasn't even
        # one of *this* door's two connected rooms - the actor was just
        # passing through en route to this door's position, not crossing
        # it). Both sides of the transition must belong to this specific
        # door's own `connects` pair.
        if (
            action in NAVIGATION_ACTIONS and connects and room_before != room_after
            and room_before in connects and room_after in connects
        ):
            record["door_sequence_state"] = DOOR_TRAVERSED
        elif action == "interact" and state_before != "open" and state_after == "open":
            record["door_sequence_state"] = DOOR_OPENED
        elif action == "interact" and state_before == "open" and state_after != "open":
            record["door_sequence_state"] = DOOR_RE_CLOSED
        elif action == "interact" and state_before == "open" and state_after == "open":
            record["door_sequence_state"] = DOOR_REPEATED_OPEN_INTERACT
        else:
            record["door_sequence_state"] = DOOR_NO_CHANGE


def door_sequence_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec section 25/28: per-door open->traverse timing and repeated-
    open-interact counts, plus the aggregate traversal-success rollup."""
    by_door: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        target = _door_target(r)
        if target is not None and r.get("door_sequence_state") is not None:
            by_door.setdefault(target, []).append(r)

    doors_opened = 0
    doors_traversed = 0
    closed_again_before_traverse = 0
    open_to_traverse: dict[str, int | None] = {}
    repeated_before_traverse: dict[str, int] = {}

    for door_id, door_records in by_door.items():
        opened_at: int | None = None
        traversed_at: int | None = None
        repeated_count = 0
        closed_again = False
        for r in door_records:
            seq = r["door_sequence_state"]
            if seq == DOOR_OPENED and opened_at is None:
                opened_at = r["decision_number"]
                doors_opened += 1
            elif seq == DOOR_REPEATED_OPEN_INTERACT and traversed_at is None:
                repeated_count += 1
            elif seq == DOOR_RE_CLOSED and traversed_at is None and opened_at is not None:
                closed_again = True
            elif seq == DOOR_TRAVERSED and traversed_at is None:
                traversed_at = r["decision_number"]
                doors_traversed += 1
        if closed_again and traversed_at is None:
            closed_again_before_traverse += 1
        open_to_traverse[door_id] = (traversed_at - opened_at) if (opened_at is not None and traversed_at is not None) else None
        repeated_before_traverse[door_id] = repeated_count

    return {
        "doors_opened": doors_opened,
        "doors_traversed": doors_traversed,
        "closed_again_before_traverse": closed_again_before_traverse,
        "open_to_traverse_success_rate": round(doors_traversed / doors_opened, 3) if doors_opened else None,
        "door_open_to_traverse_decisions": open_to_traverse,
        "repeated_open_door_interacts_before_traverse": repeated_before_traverse,
    }


def door_action_sequence_narrative(records: list[dict[str, Any]]) -> list[str]:
    """Spec section 41: a compact factual per-door sequence line for
    every door the episode actually touched, e.g. `D10: opened door;
    D11: MOVE_TO -> traversed` or `D10: opened; D11-D13: interact -> open
    (repeated 3x, never traversed)`."""
    by_door: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        target = _door_target(r)
        if target is not None and r.get("door_sequence_state") is not None:
            by_door.setdefault(target, []).append(r)

    lines = []
    for door_id, door_records in by_door.items():
        steps = [f"D{r['decision_number']:02d}:{r['door_sequence_state']}" for r in door_records]
        lines.append(f"{door_id}: " + " -> ".join(steps))
    return lines


# -- checkpoints (spec section 25) -------------------------------------------


def build_checkpoints(records: list[dict[str, Any]], points: tuple[int, ...] = CHECKPOINT_DECISIONS) -> list[dict[str, Any]]:
    checkpoints = []
    for point in points:
        at_or_before = [r for r in records if r["decision_number"] <= point]
        if not at_or_before:
            continue
        latest = at_or_before[-1]
        progress = latest.get("mission_progress_after") or {}
        checkpoints.append({
            "decision": point,
            "rooms_visited": progress.get("unique_rooms_visited"),
            "mission_items_acquired": progress.get("mission_items_acquired"),
            "requirements_satisfied": progress.get("mission_items_acquired"),
            "successful_interactions": sum(1 for r in at_or_before if r["action_effectiveness"] == "effective"),
            "repeated_or_no_progress_decisions": sum(
                1 for r in at_or_before if r["action_effectiveness"] == "ineffective" or r["progress_delta"] == ["NO_PROGRESS"]
            ),
            "mission_complete": bool(progress.get("mission_success")),
        })
    return checkpoints


# -- decision budget breakdown (spec section 26) -----------------------------


def decision_budget_breakdown(records: list[dict[str, Any]]) -> dict[str, int]:
    total = len(records)
    navigation = sum(1 for r in records if (r.get("proposed_action") or "").lower() in NAVIGATION_ACTIONS)
    interaction = sum(1 for r in records if (r.get("proposed_action") or "").lower() in INTERACTION_ACTIONS)
    mission_progress = sum(1 for r in records if r["progress_delta"] != ["NO_PROGRESS"])
    trap_opponent = sum(1 for r in records if "TRAP_PROGRESS" in r["progress_delta"] or "OPPONENT_PROGRESS" in r["progress_delta"])
    # Kept genuinely distinct (spec section 46 needs to tell "looping" (C)
    # apart from "stuck without looping, i.e. game friction" (D)): a
    # decision can be NO_PROGRESS without ever repeating a prior proposal,
    # and a repeated proposal could in principle still be legal/novel-ish -
    # in practice here `repeated` implies no-progress too, but the two
    # fractions below are computed independently, never conflated into one.
    repeated = sum(1 for r in records if r["repeat_count"] > 0)
    no_progress = sum(1 for r in records if r["progress_delta"] == ["NO_PROGRESS"])
    repeated_or_no_progress = sum(1 for r in records if r["repeat_count"] > 0 or r["progress_delta"] == ["NO_PROGRESS"])
    invalid_repaired_fallback = sum(1 for r in records if not r["first_pass_valid"])
    return {
        "total_decisions": total,
        "navigation": navigation,
        "inspection_search_interaction": interaction,
        "mission_progress_actions": mission_progress,
        "trap_or_opponent_actions": trap_opponent,
        "repeated_actions": repeated,
        "no_progress_actions": no_progress,
        "repeated_or_no_progress_actions": repeated_or_no_progress,
        "invalid_repaired_fallback_actions": invalid_repaired_fallback,
        "navigation_fraction": round(navigation / total, 3) if total else None,
        "repeated_fraction": round(repeated / total, 3) if total else None,
        "no_progress_fraction": round(no_progress / total, 3) if total else None,
    }


# -- decision-to-progress milestones (spec section 27) -----------------------


def decision_to_progress_metrics(records: list[dict[str, Any]]) -> dict[str, int | None]:
    def first_decision_where(predicate) -> int | None:
        for r in records:
            if predicate(r):
                return r["decision_number"]
        return None

    return {
        "decision_to_first_room_exit": first_decision_where(lambda r: "NAVIGATION_PROGRESS" in r["progress_delta"]),
        "decision_to_first_successful_interaction": first_decision_where(lambda r: r["action_effectiveness"] == "effective"),
        "decision_to_first_requirement_discovery": first_decision_where(lambda r: "DISCOVERY_PROGRESS" in r["progress_delta"]),
        "decision_to_first_mission_item": first_decision_where(lambda r: "ITEM_PROGRESS" in r["progress_delta"]),
        "decision_to_exit_unlock": first_decision_where(lambda r: (r.get("mission_progress_after") or {}).get("exit_unlocked")),
        "decision_to_mission_completion": first_decision_where(lambda r: (r.get("mission_progress_after") or {}).get("mission_success")),
    }


# -- budget conclusion (spec section 46) --------------------------------------

BUDGET_CLEARLY_TOO_SHORT = "A_clearly_too_short"
BUDGET_PROBABLY_TOO_SHORT = "B_probably_too_short"
BUDGET_SUFFICIENT_INEFFICIENT_PLANNING = "C_sufficient_but_inefficient_planning"
BUDGET_SUFFICIENT_GAME_FRICTION = "D_sufficient_but_game_friction"
BUDGET_MIXED_UNCLEAR = "E_mixed_unclear"


def classify_budget_conclusion(records: list[dict[str, Any]], breakdown: dict[str, Any]) -> str:
    """Spec section 46 - a rule-based classification from trace evidence,
    never a subjective score. Not a substitute for reading the actual
    trace narrative; this is the milestone's required headline label."""
    if not records:
        return BUDGET_MIXED_UNCLEAR
    total = len(records)
    repeated_fraction = breakdown["repeated_fraction"] or 0.0
    no_progress_fraction = breakdown["no_progress_fraction"] or 0.0
    last_quarter = records[int(total * 0.75):]
    progressing_late = any(r["progress_delta"] != ["NO_PROGRESS"] for r in last_quarter)

    # Genuine repetition (same proposal seen again) dominating is spec
    # section 48's explicit "do not raise the budget" case - a planning/
    # loop-avoidance problem, regardless of whether some late progress
    # still eked out around the loops.
    if repeated_fraction >= 0.3:
        return BUDGET_SUFFICIENT_INEFFICIENT_PLANNING
    if progressing_late and no_progress_fraction < 0.2:
        return BUDGET_CLEARLY_TOO_SHORT
    # Stalled, but NOT because the agent kept repeating itself - it tried
    # varied, non-repeated, legal actions that still produced nothing, and
    # never recovered late. That's friction in the game/action
    # representation, not agent looping (spec sections 46, 48).
    if no_progress_fraction >= 0.3 and repeated_fraction < 0.3 and not progressing_late:
        return BUDGET_SUFFICIENT_GAME_FRICTION
    if no_progress_fraction >= 0.15 and progressing_late:
        return BUDGET_PROBABLY_TOO_SHORT
    return BUDGET_MIXED_UNCLEAR


# -- stall analysis (spec section 45) -----------------------------------------


def stall_analysis(records: list[dict[str, Any]]) -> str:
    """Factual, rule-based description of where progress slowed - never
    speculative. Looks for the longest run of consecutive NO_PROGRESS/
    ineffective decisions and describes its dominant repeated signature."""
    if not records:
        return "no decisions recorded"

    longest_run: list[dict[str, Any]] = []
    current_run: list[dict[str, Any]] = []
    for r in records:
        stalled = r["progress_delta"] == ["NO_PROGRESS"] or r["action_effectiveness"] == "ineffective"
        if stalled:
            current_run.append(r)
            if len(current_run) > len(longest_run):
                longest_run = list(current_run)
        else:
            current_run = []

    if not longest_run:
        return "no obvious stall - steady progress until the decision cap"

    signatures = [r["repeat_signature"] for r in longest_run]
    dominant = max(set(signatures), key=signatures.count)
    start, end = longest_run[0]["decision_number"], longest_run[-1]["decision_number"]
    return f"stalled D{start}-D{end} ({len(longest_run)} decisions), dominant repeated pattern: {dominant}"


# -- human-readable markdown trace (spec section 10) --------------------------


def render_markdown_trace(records: list[dict[str, Any]], episode_id: str, seed: int, runtime: str) -> str:
    lines = [f"# Decision trace: {episode_id} (seed={seed}, runtime={runtime})", ""]
    for r in records:
        after = r.get("mission_progress_after") or {}
        lines.append(f"D{r['decision_number']:02d} | {r.get('current_room') or 'unknown room'}")
        lines.append(f"Proposed: {r.get('proposed_action')} {r.get('proposed_target') or ''}".rstrip())
        rationale = r.get("agent_stated_reason") or "(no reason given)"
        lines.append(f'Rationale: "{rationale}"')
        validity = "valid first pass" if r["first_pass_valid"] else (
            f"repaired ({r.get('repair_action')} {r.get('repair_target') or ''})".rstrip() if r.get("repair_attempted") else "fallback"
        )
        lines.append(f"Validation: {validity}")
        lines.append(f"Engine: {r.get('engine_action_status') or 'n/a'} ({r.get('engine_terminal_reason') or 'n/a'})")
        if r.get("door_sequence_state"):
            lines.append(f"Door: {r.get('door_state_before')}->{r.get('door_state_after')} [{r['door_sequence_state']}]")
        lines.append(f"Progress: {', '.join(r['progress_delta'])}")
        if r["repeat_count"] > 0:
            lines.append(f"Repeat: seen {r['repeat_count']}x in the last {REPEAT_WINDOW} decisions")
        lines.append(f"Effectiveness: {r['action_effectiveness']}")
        lines.append("")
    return "\n".join(lines)
