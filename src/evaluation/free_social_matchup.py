"""M5F.3B: paired SOCIAL_OFF/SOCIAL_ON full-duel matchup driver. Returns
to real multi-room gameplay (`data/level_01.json`, via `Game`/`duel_runner`
- not the M5F.3A.x symmetric micro-harness, per spec section 2), running
each (seed, matchup, mirror) pair through both communication conditions
with everything else held identical (spec section 3), then extracting
real messages/social events from telemetry - never scripting what either
agent says (spec section 1).
"""

from __future__ import annotations

import json
import os
from typing import Any

from src import constants
from src.evaluation.duel_runner import run_duel_episode
from src.evaluation.duel_trace import build_actor_trace_records, contested_objective_events, merge_duel_trace, mutual_same_room_encounters
from src.evaluation.free_social import (
    COORDINATION_PROPOSAL,
    FACTUAL_CLAIM,
    SAFETY_WARNING,
    THREAT_LIKE_CONDITIONAL,
    COUNTER_THREAT_LIKE,
    claim_conflicts_with_sender_plan,
    classify_claim_consistency_real,
    classify_message_category,
    cooperation_follow_through,
    misdirection_pattern,
    recipient_response_tags,
    threat_capability_snapshot,
    threat_follow_through,
)
from src.llm.personality import CAUTIOUS, OPPORTUNIST, RACER, SABOTEUR

# Spec section 14's frozen primary matchups (profile_a, profile_b) - each
# expanded to both actor-assignment mirrors by `generate_mirror_pairs`.
MATCHUPS: list[tuple[str, str]] = [
    (OPPORTUNIST, OPPORTUNIST),
    (RACER, SABOTEUR),
    (CAUTIOUS, SABOTEUR),
]

# Spec section 11: a small, fixed, deterministic environmental trap -
# reusing M5F.3A's own `filing_cabinet_01`/`electric` precedent
# (`social_influence_scenarios.build_world`) rather than inventing a new
# target, so a legitimate truthful-warning opportunity exists independent
# of either agent's own hostility.
ENV_TRAP_TARGET_ID = "filing_cabinet_01"
ENV_TRAP_TYPE = "electric"


def build_initial_traps() -> list[dict]:
    return [{"id": "m5f3b_env_trap", "target_id": ENV_TRAP_TARGET_ID, "trap_type": ENV_TRAP_TYPE, "owner": "environment", "hidden": True}]


def _load_decisions_and_lifecycle(telemetry_path: str, episode_id: str) -> tuple[list[dict], list[dict]]:
    decisions, lifecycle = [], []
    if not os.path.exists(telemetry_path):
        return decisions, lifecycle
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("episode_id") != episode_id:
                continue
            if e.get("event_type") == "llm_decision":
                decisions.append(e)
            elif e.get("event_type") == "action_lifecycle":
                lifecycle.append(e)
    return decisions, lifecycle


def _referenced_target(text: str | None, room_objects: list[dict]) -> str | None:
    """Best-effort, conservative target-reference resolution for FREE-FORM
    spontaneous text (spec sections 33-34 need "message references a
    target" as a factual precondition, but nothing here scripts the
    message itself, so there's no known-in-advance id to match against -
    unlike the M5F.3A.x controlled families). Matches only a whole,
    case-insensitive token against the object's own `id` or `label` (from
    that decision's real, already-serialized `room.objects` - never a
    fuzzy/substring match), returning `None` (no reference) rather than
    guessing when nothing matches, per spec section 30's "prefer
    UNCLASSIFIED over fragile guessing."""
    if not text:
        return None
    from src.evaluation.opportunity_metrics import _tokenize

    tokens = set(_tokenize(text))
    for obj in room_objects or []:
        obj_id = (obj.get("id") or "").lower()
        label_tokens = set(_tokenize(obj.get("label") or ""))
        if obj_id and obj_id in tokens:
            return obj.get("id")
        if label_tokens and label_tokens <= tokens:
            return obj.get("id")
    return None


def _trap_status_for(room_objects: list[dict], target_id: str | None) -> str | None:
    if not target_id:
        return None
    for obj in room_objects or []:
        if obj.get("id") == target_id:
            return obj.get("trap_status")
    return None


def _trap_inventory_at(snapshots: dict[str, dict[int, dict]], actor_id: str, decision_number: int | None) -> dict[str, int]:
    """`snapshots[actor_id]` is a `{decision_number: trap_inventory_dict}`
    map captured live via `run_duel_episode`'s own `on_decision` hook
    (`GameState.actors[...].trap_inventory` isn't part of any serialized
    Observation - trap counts are private, self-only state - so this is
    genuinely new evaluator-side-only instrumentation, never fed to
    either agent). Falls back to the nearest earlier decision number if
    the exact one wasn't captured."""
    actor_snapshots = snapshots.get(actor_id) or {}
    if decision_number in actor_snapshots:
        return actor_snapshots[decision_number]
    earlier = [n for n in actor_snapshots if decision_number is not None and n <= decision_number]
    if earlier:
        return actor_snapshots[max(earlier)]
    return {}


def _window_after(records: list[dict[str, Any]], actor_id: str, game_time: float | None, window: int = 3) -> list[dict[str, Any]]:
    """Spec sections 28, 41: the recipient's/sender's next 1-3 decisions
    - filtered by real game_time, never by decision_number, since each
    actor keeps its own independent per-actor decision_number sequence
    (see `duel_runner.run_duel_episode`'s per-actor counting) - comparing
    one actor's decision_number against another actor's would silently
    compare unrelated counters."""
    if game_time is None:
        return []
    ordered = sorted(
        [r for r in records if r["actor_id"] == actor_id and r.get("game_time") is not None and r["game_time"] > game_time],
        key=lambda r: r["game_time"],
    )
    return ordered[:window]


def _pre_message_target(records: list[dict[str, Any]], actor_id: str, game_time: float | None) -> str | None:
    if game_time is None:
        return None
    prior = [r for r in records if r["actor_id"] == actor_id and r.get("game_time") is not None and r["game_time"] < game_time]
    if not prior:
        return None
    return sorted(prior, key=lambda r: r["game_time"])[-1].get("final_target")


def _recipient_observation_before(decisions: list[dict], recipient_id: str, game_time: float | None) -> dict | None:
    """The recipient's most recently *stored* real Observation payload
    strictly before this message was sent (by real game_time - each
    actor's own `decision_number` is a separate per-actor counter, never
    comparable across actors, see `_window_after`) - used to determine
    what the recipient already legitimately knew (spec section 33's third
    clause), never the final end-of-episode state."""
    if game_time is None:
        return None
    candidates = [d for d in decisions if d.get("actor_id") == recipient_id and d.get("game_time") is not None and d["game_time"] < game_time]
    if not candidates:
        return None
    return sorted(candidates, key=lambda d: d["game_time"])[-1].get("observation")


def analyze_social_on_episode(
    decisions: list[dict], lifecycle: list[dict], episode_id: str, seed: int,
    trap_inventory_snapshots: dict[str, dict[int, dict]] | None = None,
    progress_snapshots: dict[str, dict[int, dict]] | None = None,
) -> dict[str, Any]:
    """Extracts real messages plus every M5F.3B classifier output for one
    SOCIAL_ON episode's telemetry (spec sections 26, 29-42). Purely
    post-hoc analysis of already-recorded facts - never influences the
    episode itself. `progress_snapshots[actor_id]` (from the same live
    `on_decision` hook that captures trap inventory) supplies
    `current_room`/mission-progress context per decision - without it,
    `mutual_same_room_encounters`/`contested_objective_events` below would
    silently always read zero (every record's `current_room` would be
    `None`), not because nothing happened but because nothing was
    recorded to check against."""
    progress = progress_snapshots or {}
    black_records = build_actor_trace_records(decisions, lifecycle, progress.get(constants.BLACK_SPY_ID, {}), episode_id, seed, "native_tool_like", constants.BLACK_SPY_ID)
    white_records = build_actor_trace_records(decisions, lifecycle, progress.get(constants.WHITE_SPY_ID, {}), episode_id, seed, "native_tool_like", constants.WHITE_SPY_ID)
    decisions_by_id = {d["decision_id"]: d for d in decisions}
    from src.evaluation.duel_trace import annotate_opponent_visibility

    annotate_opponent_visibility(black_records, decisions_by_id)
    annotate_opponent_visibility(white_records, decisions_by_id)
    records_by_actor = {constants.BLACK_SPY_ID: black_records, constants.WHITE_SPY_ID: white_records}

    messages_out = []
    social_events = []
    threat_events = []
    cooperation_events = []
    warning_events = []
    claim_consistency_out = []

    trap_snapshots = trap_inventory_snapshots or {}
    ordered_decisions = sorted(decisions, key=lambda d: (d.get("decision_number") or 0))
    last_threat_by_recipient: dict[str, str] = {}  # recipient_id -> message_class, for COUNTER_THREAT_LIKE detection

    for d in ordered_decisions:
        executed = d.get("executed_action") or {}
        accompanying = executed.get("accompanying_message")
        if accompanying is None:
            continue
        sender_id = d.get("actor_id")
        recipient_id = accompanying.get("recipient_id")
        text = accompanying.get("text")
        decision_number = d.get("decision_number")
        game_time = d.get("decision_game_time") or d.get("game_time")
        observation = d.get("observation") or {}
        room_objects = (observation.get("room") or {}).get("objects") or []

        referenced_target = _referenced_target(text, room_objects)
        sender_trap_status = _trap_status_for(room_objects, referenced_target)
        sender_detected = sender_trap_status == "detected"
        sender_knows_trapped = sender_detected

        recipient_obs_before = _recipient_observation_before(decisions, recipient_id, game_time) or {}
        recipient_room_objects = (recipient_obs_before.get("room") or {}).get("objects") or []
        recipient_trap_status = _trap_status_for(recipient_room_objects, referenced_target)
        recipient_already_detected = recipient_trap_status == "detected"

        replying_to_threat = last_threat_by_recipient.get(sender_id) is not None

        category = classify_message_category(
            text,
            references_target=referenced_target is not None,
            sender_legitimately_detected_trap=sender_knows_trapped,
            recipient_independently_detected_trap=recipient_already_detected,
            replying_to_opponent_threat=replying_to_threat,
        )

        if category in (THREAT_LIKE_CONDITIONAL, COUNTER_THREAT_LIKE):
            last_threat_by_recipient[recipient_id] = category
        elif category == COORDINATION_PROPOSAL:
            last_threat_by_recipient.pop(sender_id, None)

        recipient_records = records_by_actor.get(recipient_id, [])
        pre_target = _pre_message_target(recipient_records, recipient_id, game_time)
        window = _window_after(recipient_records, recipient_id, game_time)
        response_tags = recipient_response_tags(
            message_category=category, message_text=text, referenced_target_id=referenced_target,
            pre_message_target=pre_target, window_records=window,
        )

        message_row = {
            # `accompanying_message` (the executed-action envelope) never
            # itself carries a message_id - that's stamped later, only on
            # the engine's own `state.messages` entry (see
            # `GameState._do_send_message`), which isn't threaded back
            # into telemetry. The fallback id must include `sender_id`,
            # not just `episode_id`/`decision_number` - each actor keeps
            # an independent per-actor decision_number counter, so two
            # different senders' own decision 1 would otherwise collide
            # on the same fallback id (a real bug found and fixed live:
            # 3 of 19 Stage 1 messages initially collided this way).
            "message_id": accompanying.get("message_id") or f"{episode_id}-{sender_id}-d{decision_number}",
            "episode_id": episode_id, "sender_actor_id": sender_id, "sender_profile": d.get("personality_profile"),
            "recipient_actor_id": recipient_id, "game_time": game_time, "sender_decision_id": d.get("decision_id"),
            "message_text": text, "sender_visible_reason": d.get("agent_stated_reason") or (d.get("raw_response") or {}).get("reason"),
            "sender_current_target": executed.get("target_id"), "sender_current_room": (observation.get("self") or {}).get("room"),
            "referenced_target_id": referenced_target, "category": category, "recipient_response_tags": response_tags,
        }
        messages_out.append(message_row)
        social_events.append({"episode_id": episode_id, "decision_number": decision_number, "category": category, "sender_actor_id": sender_id, "recipient_actor_id": recipient_id})

        if category == COORDINATION_PROPOSAL:
            sender_records = records_by_actor.get(sender_id, [])
            sender_window = _window_after(sender_records, sender_id, game_time)
            follow = cooperation_follow_through(
                sender_claimed_target=referenced_target, recipient_expected_complementary_target=None,
                sender_window_records=sender_window, recipient_window_records=window,
            )
            cooperation_events.append({**message_row, **follow})

        if category in (THREAT_LIKE_CONDITIONAL, COUNTER_THREAT_LIKE):
            sender_records = records_by_actor.get(sender_id, [])
            sender_window = _window_after(sender_records, sender_id, game_time)
            snapshot = threat_capability_snapshot(
                sender_trap_inventory=_trap_inventory_at(trap_snapshots, sender_id, decision_number),
                target_trappable=bool(next((o.get("trappable") for o in room_objects if o.get("id") == referenced_target), False)),
                target_already_trapped=sender_trap_status == "detected" and bool(next((o.get("trap_is_mine") for o in room_objects if o.get("id") == referenced_target), False)),
                # Known simplification, documented rather than silently
                # assumed: full reachability would need real pathfinding
                # against the level's own lock/key topology at message
                # time. Every object in `data/level_01.json` is reachable
                # given enough time/keys in this level's own design, so a
                # fixed True never fabricates a false negative here; it
                # just can't yet distinguish "reachable eventually" from
                # "reachable right now" - see summary.md's honest caveat.
                sender_can_plausibly_access_target=True,
            )
            follow_through = threat_follow_through(referenced_target, sender_window)
            threat_events.append({**message_row, "capability_snapshot": snapshot, "follow_through": follow_through})

        if category == SAFETY_WARNING or (category == FACTUAL_CLAIM and referenced_target is not None):
            consistency = classify_claim_consistency_real(text, sender_knows_trapped)
            conflict = claim_conflicts_with_sender_plan(text, executed.get("action"), executed.get("target_id"), (referenced_target,) if referenced_target else ())
            claim_row = {**message_row, "consistency": consistency, "conflicts_with_sender_plan": conflict}
            claim_consistency_out.append(claim_row)
            if category == SAFETY_WARNING:
                warning_events.append(claim_row)

        misdir = misdirection_pattern(
            message_claims_safe=category == FACTUAL_CLAIM and text is not None and ("clear" in text.lower() or "safe" in text.lower()),
            sender_knows_target_trapped=sender_knows_trapped, sender_warned_recipient_away=category == SAFETY_WARNING,
            sender_window_records=_window_after(records_by_actor.get(sender_id, []), sender_id, game_time),
            referenced_target_id=referenced_target,
        )
        if any(misdir.values()):
            social_events[-1]["misdirection_pattern"] = misdir

    encounters = mutual_same_room_encounters(black_records, white_records)
    contested = contested_objective_events(black_records, white_records)

    return {
        "messages": messages_out, "social_events": social_events, "threat_events": threat_events,
        "cooperation_events": cooperation_events, "warning_events": warning_events, "claim_consistency": claim_consistency_out,
        "mutual_same_room_encounters": encounters, "contested_objective_events": contested,
        "black_records": black_records, "white_records": white_records,
    }


def run_paired_duel(
    *, seed: int, black_personality: str, white_personality: str, mirror: str, provider: str, model: str, region: str | None,
    agent_runtime: str, telemetry_path: str, episode_prefix: str = "m5f3b", max_decisions_per_actor: int = 40,
    message_cooldown_decisions: int = 3, llm_timeout: float = 60.0,
) -> dict[str, Any]:
    """Runs SOCIAL_OFF then SOCIAL_ON for one (seed, matchup, mirror) pair
    (spec sections 6, 16) - everything else (seed, personalities, actor
    assignment, environmental traps) held identical (spec section 3)."""
    initial_traps = build_initial_traps()
    pair_label = f"seed{seed}-{black_personality}_vs_{white_personality}-mirror{mirror}"
    results: dict[str, Any] = {"pair_label": pair_label, "seed": seed, "black_personality": black_personality, "white_personality": white_personality, "mirror": mirror}

    off_episode_id = f"{episode_prefix}-{pair_label}-SOCIAL_OFF"
    off_result = run_duel_episode(
        seed=seed, provider=provider, model=model, region=region, agent_runtime=agent_runtime,
        telemetry_path=telemetry_path, episode_id=off_episode_id, max_decisions_per_actor=max_decisions_per_actor,
        llm_timeout=llm_timeout, black_personality=black_personality, white_personality=white_personality,
        communication_enabled=False, initial_traps=initial_traps,
    )
    results["social_off"] = {
        "episode_id": off_episode_id, "status": off_result.status, "stop_reason": off_result.stop_reason,
        "black_decisions": off_result.black_decision_count, "white_decisions": off_result.white_decision_count,
        "black_fallbacks": off_result.black_fallback_count, "white_fallbacks": off_result.white_fallback_count,
        "black_game_facts": off_result.black_game_facts, "white_game_facts": off_result.white_game_facts,
        "competitive_facts": off_result.competitive_facts, "winner": off_result.winner,
    }

    on_episode_id = f"{episode_prefix}-{pair_label}-SOCIAL_ON"
    # Evaluator-side-only instrumentation (spec section 26/35): trap
    # inventory is private, self-only state never carried by any
    # Observation (see `threat_capability_snapshot`'s own docstring), so
    # it has to be captured live via this hook, keyed by decision number,
    # rather than reconstructed after the fact from telemetry.
    trap_snapshots: dict[str, dict[int, dict]] = {constants.BLACK_SPY_ID: {}, constants.WHITE_SPY_ID: {}}
    # Reuses `run_m5f2_3.duel_sanity_check`'s own established snapshot
    # pattern (`build_progress_snapshot`) so `current_room` - needed by
    # `mutual_same_room_encounters`/`contested_objective_events` - is
    # genuinely populated rather than silently always None.
    from src.evaluation.decision_trace import build_progress_snapshot

    progress_snapshots: dict[str, dict[int, dict]] = {constants.BLACK_SPY_ID: {}, constants.WHITE_SPY_ID: {}}
    last_black_n = last_white_n = 0

    def _on_decision(black_n: int, white_n: int, state) -> None:
        nonlocal last_black_n, last_white_n
        if black_n > last_black_n or (black_n == 0 and white_n == 0):
            trap_snapshots[constants.BLACK_SPY_ID][black_n] = dict(state.actors[constants.BLACK_SPY_ID].trap_inventory)
            progress_snapshots[constants.BLACK_SPY_ID][black_n] = build_progress_snapshot(state, constants.BLACK_SPY_ID)
            last_black_n = black_n
        if white_n > last_white_n or (black_n == 0 and white_n == 0):
            trap_snapshots[constants.WHITE_SPY_ID][white_n] = dict(state.actors[constants.WHITE_SPY_ID].trap_inventory)
            progress_snapshots[constants.WHITE_SPY_ID][white_n] = build_progress_snapshot(state, constants.WHITE_SPY_ID)
            last_white_n = white_n

    on_result = run_duel_episode(
        seed=seed, provider=provider, model=model, region=region, agent_runtime=agent_runtime,
        telemetry_path=telemetry_path, episode_id=on_episode_id, max_decisions_per_actor=max_decisions_per_actor,
        llm_timeout=llm_timeout, black_personality=black_personality, white_personality=white_personality,
        communication_enabled=True, zero_decision_cost_messaging=True, flat_message_schema=True,
        message_cooldown_decisions=message_cooldown_decisions, initial_traps=initial_traps,
        on_decision=_on_decision,
    )
    decisions, lifecycle = _load_decisions_and_lifecycle(telemetry_path, on_episode_id)
    analysis = analyze_social_on_episode(decisions, lifecycle, on_episode_id, seed, trap_inventory_snapshots=trap_snapshots, progress_snapshots=progress_snapshots)
    results["social_on"] = {
        "episode_id": on_episode_id, "status": on_result.status, "stop_reason": on_result.stop_reason,
        "black_decisions": on_result.black_decision_count, "white_decisions": on_result.white_decision_count,
        "black_fallbacks": on_result.black_fallback_count, "white_fallbacks": on_result.white_fallback_count,
        "black_game_facts": on_result.black_game_facts, "white_game_facts": on_result.white_game_facts,
        "competitive_facts": on_result.competitive_facts, "winner": on_result.winner,
        "messages_sent": len(analysis["messages"]), "analysis": analysis,
    }
    return results
