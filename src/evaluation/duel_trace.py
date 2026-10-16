"""M5F: multi-agent duel trace - merges each actor's own
`build_decision_trace_records` (M5E.4.1) into one chronological, actor-
attributed duel trace ordered by real game simulation time (spec section
58, never provider wall-clock completion order), plus opponent-encounter
and contested-objective detection built entirely from already-recorded
telemetry (`llm_decision`'s own stored `observation` payload already
carries `opponent.visible`/`last_seen` per decision - see
`src/llm/telemetry.py:186` - no new engine instrumentation needed).
"""

from __future__ import annotations

from typing import Any

from src import constants
from src.evaluation.decision_trace import build_decision_trace_records


def build_actor_trace_records(
    decisions: list[dict[str, Any]],
    action_lifecycle_events: list[dict[str, Any]],
    progress_snapshots: dict[int, dict[str, Any]],
    episode_id: str,
    seed: int,
    runtime: str,
    actor_id: str,
) -> list[dict[str, Any]]:
    """Thin wrapper around `build_decision_trace_records` - filters the
    raw telemetry to `actor_id` first (spec section 19: every row is
    actor-attributed) and tags each resulting record with `actor_id`."""
    actor_decisions = [d for d in decisions if d.get("actor_id") == actor_id]
    actor_lifecycle = [e for e in action_lifecycle_events if e.get("actor_id") == actor_id]
    records = build_decision_trace_records(actor_decisions, actor_lifecycle, progress_snapshots, episode_id, seed, runtime)
    for r in records:
        r["actor_id"] = actor_id
    return records


def merge_duel_trace(black_records: list[dict[str, Any]], white_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spec section 58: ordered by real game simulation time, not
    provider completion order - `game_time` is already recorded per
    decision (`LLMController`'s own decision_game_time, not wall-clock)."""
    merged = list(black_records) + list(white_records)
    merged.sort(key=lambda r: (r.get("game_time") if r.get("game_time") is not None else float("inf"), r.get("decision_number") or 0))
    return merged


def render_duel_markdown(merged_records: list[dict[str, Any]], episode_id: str, seed: int) -> str:
    lines = [f"# Duel trace: {episode_id} (seed={seed})", ""]
    for r in merged_records:
        actor_label = "BLACK" if r.get("actor_id") == constants.BLACK_SPY_ID else "WHITE"
        lines.append(f"T={r.get('game_time'):.1f}" if r.get("game_time") is not None else "T=?")
        lines.append(f"{actor_label} D{r['decision_number']:02d}")
        lines.append(f"room = {r.get('current_room') or 'unknown'}")
        after = r.get("mission_progress_after") or {}
        opponent_visible = (r.get("opponent_visible_before"), r.get("opponent_visible_after"))
        lines.append(f"opponent visible = {opponent_visible[1]}")
        rationale = r.get("agent_stated_reason") or "(no reason given)"
        lines.append(f'rationale = "{rationale}"')
        lines.append(f"action = {r.get('final_action')} {r.get('final_target') or ''}".rstrip())
        lines.append(f"result = {r.get('engine_action_status') or 'n/a'} ({r.get('engine_terminal_reason') or 'n/a'})")
        lines.append("")
    return "\n".join(lines)


# -- opponent encounters (spec sections 41-43) --------------------------------


def annotate_opponent_visibility(records: list[dict[str, Any]], decisions_by_id: dict[str, dict[str, Any]]) -> None:
    """Mutates each record with `opponent_visible_before`/`opponent_visible_after`,
    read directly from the real, already-stored `observation.opponent.visible`
    field of this decision and the immediately preceding one for the same
    actor - never inferred, never a new telemetry event."""
    by_actor: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_actor.setdefault(r["actor_id"], []).append(r)

    for actor_records in by_actor.values():
        actor_records.sort(key=lambda r: r["decision_number"])
        previous_visible: bool | None = None
        for r in actor_records:
            raw = decisions_by_id.get(r["decision_id"], {})
            observation = raw.get("observation") or {}
            visible = (observation.get("opponent") or {}).get("visible")
            r["opponent_visible_before"] = previous_visible
            r["opponent_visible_after"] = visible
            previous_visible = visible


def extract_opponent_encounters(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spec section 41: every `not visible -> visible` and `visible ->
    not visible` transition, per actor, with `game_time`/`room`/`decision_id`."""
    encounters = []
    for r in records:
        before, after = r.get("opponent_visible_before"), r.get("opponent_visible_after")
        if before is False and after is True:
            encounters.append({
                "actor_id": r["actor_id"], "transition": "opponent_not_visible_to_visible",
                "game_time": r.get("game_time"), "room": r.get("current_room"),
                "decision_id": r.get("decision_id"), "decision_number": r.get("decision_number"),
            })
        elif before is True and after is False:
            encounters.append({
                "actor_id": r["actor_id"], "transition": "opponent_visible_to_not_visible",
                "game_time": r.get("game_time"), "room": r.get("current_room"),
                "decision_id": r.get("decision_id"), "decision_number": r.get("decision_number"),
            })
    return encounters


def reaction_to_sighting(records: list[dict[str, Any]], actor_id: str, sighting_decision_number: int, window: int = 3) -> list[dict[str, Any]]:
    """Spec section 42: the next 1-3 high-level decisions for `actor_id`
    after a sighting, with only factual changes recorded - never
    smart/dumb labels."""
    actor_records = sorted(
        [r for r in records if r["actor_id"] == actor_id and r["decision_number"] > sighting_decision_number],
        key=lambda r: r["decision_number"],
    )[:window]
    # Seed with the pre-sighting target (the sighting decision's own
    # record, if present) so the *first* post-sighting reaction can be
    # correctly compared against what the actor was doing right before -
    # not treated as "no change" merely for being first in this window.
    pre_sighting = [r for r in records if r["actor_id"] == actor_id and r["decision_number"] == sighting_decision_number]
    reactions = []
    prior_target = pre_sighting[0].get("final_target") if pre_sighting else None
    for r in actor_records:
        target = r.get("final_target")
        reactions.append({
            "decision_number": r["decision_number"],
            "action": r.get("final_action"),
            "target": target,
            "target_changed": prior_target is not None and target != prior_target,
            "trap_placed": r.get("final_action") == "place_trap",
            "trap_disarmed": r.get("final_action") == "disarm",
            "is_wait": r.get("final_action") == "wait",
        })
        prior_target = target
    return reactions


def mutual_same_room_encounters(black_records: list[dict[str, Any]], white_records: list[dict[str, Any]]) -> int:
    """Spec section 72: a real mutual encounter - both actors' own
    records show the opponent visible with an overlapping current_room at
    a comparable game_time (nearest-record match, not just "both had
    *a* sighting somewhere")."""
    count = 0
    for b in black_records:
        if not b.get("opponent_visible_after"):
            continue
        for w in white_records:
            if not w.get("opponent_visible_after"):
                continue
            if b.get("current_room") and b.get("current_room") == w.get("current_room"):
                bt, wt = b.get("game_time"), w.get("game_time")
                if bt is not None and wt is not None and abs(bt - wt) <= 2.0:
                    count += 1
                    break
    return count


# -- contested objectives (spec section 47) -----------------------------------


def contested_objective_events(black_records: list[dict[str, Any]], white_records: list[dict[str, Any]], window_seconds: float = 5.0) -> list[dict[str, Any]]:
    """Both actors proposing the same target within `window_seconds` of
    each other - a purely event-based fact, no interpretation of intent."""
    events = []
    for b in black_records:
        b_target, b_time = b.get("proposed_target"), b.get("game_time")
        if not b_target or b_time is None:
            continue
        for w in white_records:
            w_target, w_time = w.get("proposed_target"), w.get("game_time")
            if w_target != b_target or w_time is None:
                continue
            if abs(b_time - w_time) <= window_seconds:
                events.append({
                    "target": b_target, "black_decision_number": b["decision_number"], "white_decision_number": w["decision_number"],
                    "black_game_time": b_time, "white_game_time": w_time,
                })
    return events
