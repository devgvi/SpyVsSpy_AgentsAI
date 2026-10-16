"""M6C: memory architecture as a harness variable (spec sections 8-22).

Three primary conditions, all applied through the now-parity-tested
`model_payload_transform` hook (RP10, closed in M6B.2) so both Native and
Strands receive equivalent logical memory content (spec section 20):

- M0_MEMORY_OFF: strips the AgentMemory-sourced `strategy_memory` block
  entirely, replacing it with a minimal disabled marker. Everything already
  part of the current Observation (`model_payload["memory"]`/`["mission"]`/
  `["opponent"]` - see `compact_observation_for_model`) is untouched (spec
  section 9: "Do not remove information that is already part of the
  current Observation").
- M1_CURRENT_MEMORY: no transform at all - the existing production
  `AgentMemory.to_compact_dict()` output, byte-identical to every
  pre-M6C caller (spec section 10: "No wording cleanup. No ordering
  changes. No compaction changes.").
- M2_STRUCTURED_FACTS: the exact same facts M1 carries, reshaped into a
  canonical structured form (spec section 11) - never a separate,
  independently-computed source of truth. `extract_memory_facts()` below
  is the executable fact-equivalence check (spec section 12) - it reduces
  either shape to the same canonical, order-independent tuple set.

Duplicate-fact note (spec sections 18-19, real and pre-existing, not
introduced by M6C): `AgentMemory.to_compact_dict()`'s own `mission`/
`exploration.visited_rooms`/`opponent_memory` fields substantially
duplicate facts `compact_observation_for_model` already surfaces
independently from the raw Observation (`model_payload["mission"]`/
`["memory"]["visited_rooms"]`/`["opponent"]`) - both ultimately sourced
from the same GameState-level per-actor memory (`Actor.visited_rooms`/
`known_locked_objects`), just accumulated through two separate code paths
(`GameState._update_room_memory` vs. `AgentMemoryTracker.
update_from_observation`). Recorded here, not silently removed - see
`memory_representation_audit.md`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

M0_MEMORY_OFF = "M0_MEMORY_OFF"
M1_CURRENT_MEMORY = "M1_CURRENT_MEMORY"
M2_STRUCTURED_FACTS = "M2_STRUCTURED_FACTS"
MEMORY_CONDITIONS = (M0_MEMORY_OFF, M1_CURRENT_MEMORY, M2_STRUCTURED_FACTS)

STRUCTURED_MEMORY_SCHEMA_VERSION = "memory-structured-v1"


def m0_memory_off_transform(model_payload: dict[str, Any]) -> dict[str, Any]:
    """Spec section 9: persistent AgentMemory content removed; current
    Observation content (`memory`/`mission`/`opponent` keys, built
    independently by `compact_observation_for_model`) is a sibling key in
    `model_payload`, never touched here."""
    payload = dict(model_payload)
    payload["strategy_memory"] = {
        "schema_version": STRUCTURED_MEMORY_SCHEMA_VERSION,
        "memory_condition": M0_MEMORY_OFF,
    }
    return payload


def _structured_facts_from_compact(m1_dict: dict[str, Any]) -> dict[str, Any]:
    """Pure reshape of AgentMemory.to_compact_dict()'s own output (spec
    section 11) - every field below reads directly from `m1_dict`, never
    from a second data source, so no fact can be silently added or lost by
    construction. Unordered collections get canonical sorted order (spec
    section 16); chronological ones (`recent_decisions`/`recent_failures`/
    `recent_events`) keep M1's own order unchanged."""
    opponent = m1_dict.get("opponent_memory") or {}
    last_seen = None
    if opponent.get("last_seen_room") is not None:
        last_seen = {"room": opponent["last_seen_room"], "last_seen_time": opponent.get("last_seen_time")}
    return {
        "schema_version": STRUCTURED_MEMORY_SCHEMA_VERSION,
        "memory_condition": M2_STRUCTURED_FACTS,
        "mission": m1_dict.get("mission"),
        "visited_rooms": sorted(m1_dict.get("exploration", {}).get("visited_rooms", [])),
        "known_locks": sorted(m1_dict.get("known_locks", []) or [], key=lambda entry: entry.get("target_id", "")),
        "known_empty_objects": sorted(m1_dict.get("known_empty_objects", [])),
        "known_searched_objects": sorted(m1_dict.get("known_searched_objects", [])),
        "opponent_currently_visible": bool(opponent.get("currently_visible", False)),
        "opponent_last_seen": last_seen,
        "recent_interactions": [
            {"action": r.get("action"), "target_id": r.get("target_id"), "result": r.get("result"), "detail": r.get("detail"), "game_time": r.get("game_time")}
            for r in (m1_dict.get("recent_interactions") or [])
        ],
        "recent_failures": [
            {"action": r.get("action"), "target_id": r.get("target_id"), "reason": r.get("reason"), "detail": r.get("detail"), "game_time": r.get("game_time")}
            for r in (m1_dict.get("recent_failures") or [])
        ],
        "recent_events": [
            {"event_type": e.get("event_type"), "target_id": e.get("target_id"), "detail": e.get("detail"), "game_time": e.get("game_time")}
            for e in (m1_dict.get("recent_events") or [])
        ],
        "recent_decisions": [
            {"action": d.get("selected_action"), "target_id": d.get("target_id"), "valid": d.get("valid"), "game_time": d.get("game_time")}
            for d in (m1_dict.get("recent_decisions") or [])
        ],
        "recent_repetitions": sorted(
            (m1_dict.get("recent_repetitions") or []), key=lambda r: (r.get("action") or "", r.get("target_id") or "")
        ),
        "messages_sent_count": m1_dict.get("messages_sent_count", 0),
        "messages_received_count": m1_dict.get("messages_received_count", 0),
    }


def m2_structured_facts_transform(model_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(model_payload)
    m1_dict = payload.get("strategy_memory") or {}
    payload["strategy_memory"] = _structured_facts_from_compact(m1_dict)
    return payload


MEMORY_TRANSFORMS = {
    M0_MEMORY_OFF: m0_memory_off_transform,
    M1_CURRENT_MEMORY: None,
    M2_STRUCTURED_FACTS: m2_structured_facts_transform,
}


def frozen_condition_transform(condition: str, synthetic_m1_dict: dict[str, Any]):
    """For frozen-replay snapshots (spec sections 24-27): the adapter's own
    real AgentMemory starts empty for a single isolated decision (no
    organic history), so - unlike a live episode, where M1 needs no
    transform at all (spec section 10) - reproducing a specific memory
    CONTENT for a frozen snapshot requires injecting a synthetic,
    pre-built `AgentMemory.to_compact_dict()` result (`synthetic_m1_dict`)
    via this same `model_payload_transform` path, for every condition
    including M1 itself. M0/M2 still derive from `synthetic_m1_dict`
    exactly as `m0_memory_off_transform`/`m2_structured_facts_transform`
    do for a live episode - no separate logic, no separate fact source.
    """
    if condition == M0_MEMORY_OFF:
        return m0_memory_off_transform
    if condition == M2_STRUCTURED_FACTS:
        def _inject_m2(payload: dict[str, Any]) -> dict[str, Any]:
            payload = dict(payload)
            payload["strategy_memory"] = _structured_facts_from_compact(synthetic_m1_dict)
            return payload
        return _inject_m2
    if condition == M1_CURRENT_MEMORY:
        def _inject_m1(payload: dict[str, Any]) -> dict[str, Any]:
            payload = dict(payload)
            payload["strategy_memory"] = synthetic_m1_dict
            return payload
        return _inject_m1
    raise ValueError(f"unknown memory condition {condition!r}")


def extract_memory_facts(strategy_memory_dict: dict[str, Any]) -> frozenset:
    """Spec sections 12, 62: reduces EITHER M1's native shape or M2's
    structured shape to the same canonical, order-independent set of
    (fact_type, value) tuples - the executable fact-equivalence check.
    Dispatches on `memory_condition`/`schema_version` to know which shape
    it was given; both branches must read every field the other reads, or
    a genuine fact difference would go undetected."""
    d = strategy_memory_dict or {}
    condition = d.get("memory_condition")

    if condition == M0_MEMORY_OFF:
        return frozenset()

    if condition == M2_STRUCTURED_FACTS:
        facts = set()
        mission = d.get("mission") or {}
        facts.add(("mission_have", tuple(sorted(mission.get("have", []) or []))))
        facts.add(("mission_missing", tuple(sorted(mission.get("missing", []) or []))))
        facts.add(("mission_ready_to_exit", bool(mission.get("ready_to_exit", False))))
        facts.add(("visited_rooms", tuple(sorted(d.get("visited_rooms", []) or []))))
        facts.add(("known_locks", tuple(sorted(
            (l.get("target_id"), l.get("lock_type"), l.get("requires"), l.get("satisfied")) for l in (d.get("known_locks") or [])
        ))))
        facts.add(("known_empty_objects", tuple(sorted(d.get("known_empty_objects", []) or []))))
        facts.add(("known_searched_objects", tuple(sorted(d.get("known_searched_objects", []) or []))))
        facts.add(("opponent_currently_visible", bool(d.get("opponent_currently_visible", False))))
        last_seen = d.get("opponent_last_seen")
        facts.add(("opponent_last_seen", (last_seen.get("room"), last_seen.get("last_seen_time")) if last_seen else None))
        facts.add(("recent_interactions", tuple(
            (r.get("action"), r.get("target_id"), r.get("result"), r.get("detail"), r.get("game_time")) for r in (d.get("recent_interactions") or [])
        )))
        facts.add(("recent_failures", tuple(
            (r.get("action"), r.get("target_id"), r.get("reason"), r.get("detail"), r.get("game_time")) for r in (d.get("recent_failures") or [])
        )))
        facts.add(("recent_events", tuple(
            (e.get("event_type"), e.get("target_id"), e.get("detail"), e.get("game_time")) for e in (d.get("recent_events") or [])
        )))
        facts.add(("recent_decisions", tuple(
            (r.get("action"), r.get("target_id"), r.get("valid"), r.get("game_time")) for r in (d.get("recent_decisions") or [])
        )))
        facts.add(("recent_repetitions_count", len(d.get("recent_repetitions") or [])))
        facts.add(("messages_sent_count", d.get("messages_sent_count", 0)))
        facts.add(("messages_received_count", d.get("messages_received_count", 0)))
        return frozenset(facts)

    # M1 native shape (memory_condition unset - the byte-identical
    # production AgentMemory.to_compact_dict() output).
    facts = set()
    mission = d.get("mission") or {}
    facts.add(("mission_have", tuple(sorted(mission.get("have", []) or []))))
    facts.add(("mission_missing", tuple(sorted(mission.get("missing", []) or []))))
    facts.add(("mission_ready_to_exit", bool(mission.get("ready_to_exit", False))))
    facts.add(("visited_rooms", tuple(sorted(d.get("exploration", {}).get("visited_rooms", []) or []))))
    facts.add(("known_locks", tuple(sorted(
        (l.get("target_id"), l.get("lock_type"), l.get("requires"), l.get("satisfied")) for l in (d.get("known_locks") or [])
    ))))
    facts.add(("known_empty_objects", tuple(sorted(d.get("known_empty_objects", []) or []))))
    facts.add(("known_searched_objects", tuple(sorted(d.get("known_searched_objects", []) or []))))
    opponent = d.get("opponent_memory") or {}
    facts.add(("opponent_currently_visible", bool(opponent.get("currently_visible", False))))
    facts.add((
        "opponent_last_seen",
        (opponent.get("last_seen_room"), opponent.get("last_seen_time")) if opponent.get("last_seen_room") is not None else None,
    ))
    facts.add(("recent_interactions", tuple(
        (r.get("action"), r.get("target_id"), r.get("result"), r.get("detail"), r.get("game_time")) for r in (d.get("recent_interactions") or [])
    )))
    facts.add(("recent_failures", tuple(
        (r.get("action"), r.get("target_id"), r.get("reason"), r.get("detail"), r.get("game_time")) for r in (d.get("recent_failures") or [])
    )))
    facts.add(("recent_events", tuple(
        (e.get("event_type"), e.get("target_id"), e.get("detail"), e.get("game_time")) for e in (d.get("recent_events") or [])
    )))
    facts.add(("recent_decisions", tuple(
        (r.get("selected_action"), r.get("target_id"), r.get("valid"), r.get("game_time")) for r in (d.get("recent_decisions") or [])
    )))
    facts.add(("recent_repetitions_count", len(d.get("recent_repetitions") or [])))
    facts.add(("messages_sent_count", d.get("messages_sent_count", 0)))
    facts.add(("messages_received_count", d.get("messages_received_count", 0)))
    return frozenset(facts)


def memory_fact_hash(strategy_memory_dict: dict[str, Any]) -> str:
    """Spec section 22: a stable hash of the CANONICAL fact set - equal
    across M1/M2 for the same underlying facts, distinct whenever a real
    fact differs."""
    facts = extract_memory_facts(strategy_memory_dict)
    canonical = json.dumps(sorted(repr(f) for f in facts), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def memory_representation_hash(strategy_memory_dict: dict[str, Any]) -> str:
    """Spec section 22: a hash of the exact serialized representation
    (shape included) - differs between M1 and M2 even for identical facts,
    by design (spec section 62's expected `representation hash equal =
    false`)."""
    canonical = json.dumps(strategy_memory_dict or {}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def memory_payload_length(strategy_memory_dict: dict[str, Any]) -> int:
    return len(json.dumps(strategy_memory_dict or {}, sort_keys=True, default=str))


def memory_token_estimate(strategy_memory_dict: dict[str, Any]) -> int:
    # Same rough estimator convention used elsewhere in this project
    # (~4 chars/token) - a comparative estimate, never billed truth.
    return max(1, memory_payload_length(strategy_memory_dict) // 4)
