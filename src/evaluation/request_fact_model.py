"""M6E spec sections 18-20: a canonical, deterministic fact extractor over
the real production context (`compact_observation_for_model()` output +
`AgentMemory.to_compact_dict()`), used to verify fact-equivalence between
compaction conditions (spec sections 10, 13) without any LLM summarizer
or relevance heuristic (spec sections 16-17).

One `Fact` per semantic identity within a category - e.g. one VISITED_ROOMS
fact per room id, not one fact per section. `duplicate_locations` records
every section that independently carries that same semantic fact (spec
section 19), so `request_fact_hash()` can hash the *set* of facts,
independent of how many sections repeat them (spec section 13's
requirement that `request_fact_set(C0) == request_fact_set(C2)` even
though section-level duplication differs).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

CURRENT_OBSERVATION = "CURRENT_OBSERVATION"
ROOM_STATE = "ROOM_STATE"
VISIBLE_OBJECTS = "VISIBLE_OBJECTS"
LEGAL_ACTIONS = "LEGAL_ACTIONS"
MISSION_STATE = "MISSION_STATE"
VISITED_ROOMS = "VISITED_ROOMS"
KNOWN_LOCKS = "KNOWN_LOCKS"
LAST_SEEN_OPPONENT = "LAST_SEEN_OPPONENT"
RECENT_DECISIONS = "RECENT_DECISIONS"
RECENT_FAILURES = "RECENT_FAILURES"
RECENT_INTERACTIONS = "RECENT_INTERACTIONS"
INVENTORY = "INVENTORY"

FACT_CATEGORIES = (
    CURRENT_OBSERVATION, ROOM_STATE, VISIBLE_OBJECTS, LEGAL_ACTIONS, MISSION_STATE, VISITED_ROOMS,
    KNOWN_LOCKS, LAST_SEEN_OPPONENT, RECENT_DECISIONS, RECENT_FAILURES, RECENT_INTERACTIONS, INVENTORY,
)

# Sections that carry no history (chronological=False) vs. ones that do -
# predeclared, generic, never per-snapshot (spec section 17).
_CHRONOLOGICAL_CATEGORIES = {RECENT_DECISIONS, RECENT_FAILURES, RECENT_INTERACTIONS}


@dataclass(frozen=True)
class Fact:
    fact_id: str  # f"{category}:{identity}"
    category: str
    identity: str
    value: Any
    source_sections: tuple[str, ...]  # every section this same semantic fact was found in
    chronological: bool
    actor_visible: bool = True  # everything reaching a model-facing payload is actor-visible by construction

    @property
    def duplicate_locations(self) -> tuple[str, ...]:
        return self.source_sections[1:]

    def semantic_signature(self) -> tuple[str, str, str]:
        return (self.category, self.identity, json.dumps(self.value, sort_keys=True, default=str))


def _merge(facts: dict[tuple[str, str], Fact], category: str, identity: str, value: Any, section: str, chronological: bool) -> None:
    key = (category, identity)
    existing = facts.get(key)
    if existing is None:
        facts[key] = Fact(f"{category}:{identity}", category, identity, value, (section,), chronological)
    else:
        # Prefer the richer value (more keys) when sections disagree on detail
        # (e.g. observation's id-only known-lock entry vs memory's full record)
        # - never drop information already captured by an earlier section.
        richer = value if isinstance(value, dict) and isinstance(existing.value, dict) and len(value) > len(existing.value) else existing.value
        facts[key] = Fact(existing.fact_id, category, identity, richer, existing.source_sections + (section,), chronological)


def extract_request_facts(observation_payload: dict[str, Any], memory_payload: dict[str, Any]) -> list[Fact]:
    """Deterministic, field-based extraction only - no LLM, no relevance
    heuristic (spec sections 16-17). `observation_payload` is the compact,
    tool-like observation dict (`compact_observation_for_model()` output);
    `memory_payload` is `AgentMemory.to_compact_dict()`."""
    facts: dict[tuple[str, str], Fact] = {}

    self_block = observation_payload.get("self", {})
    _merge(facts, CURRENT_OBSERVATION, "self", {
        "position": self_block.get("position"), "facing": self_block.get("facing"),
        "status": self_block.get("status"), "near_object_id": self_block.get("near_object_id"),
        "time_remaining": observation_payload.get("time", {}).get("remaining"),
    }, "observation.self", chronological=False)

    room = observation_payload.get("room", {})
    _merge(facts, ROOM_STATE, "room", {"id": room.get("id")}, "observation.room.id", chronological=False)

    for obj in room.get("objects", []):
        obj_id = obj.get("id")
        if obj_id is not None:
            _merge(facts, VISIBLE_OBJECTS, obj_id, obj, "observation.room.objects", chronological=False)

    legal_compact = observation_payload.get("legal_actions_compact")
    if legal_compact is not None:
        _merge(facts, LEGAL_ACTIONS, "legal_actions", legal_compact, "observation.legal_actions_compact", chronological=False)

    obs_mission = observation_payload.get("mission")
    if obs_mission:
        _merge(facts, MISSION_STATE, "mission", obs_mission, "observation.mission", chronological=False)
    mem_mission = memory_payload.get("mission")
    if mem_mission:
        _merge(facts, MISSION_STATE, "mission", mem_mission, "memory.mission", chronological=False)

    for room_id in observation_payload.get("memory", {}).get("visited_rooms", []):
        _merge(facts, VISITED_ROOMS, room_id, {"room_id": room_id}, "observation.memory.visited_rooms", chronological=False)
    for room_id in memory_payload.get("exploration", {}).get("visited_rooms", []):
        _merge(facts, VISITED_ROOMS, room_id, {"room_id": room_id}, "memory.exploration.visited_rooms", chronological=False)

    for target_id in observation_payload.get("memory", {}).get("known_locked_objects", []):
        _merge(facts, KNOWN_LOCKS, target_id, {"target_id": target_id}, "observation.memory.known_locked_objects", chronological=False)
    for lock in memory_payload.get("known_locks", []):
        target_id = lock.get("target_id")
        if target_id is not None:
            _merge(facts, KNOWN_LOCKS, target_id, lock, "memory.known_locks", chronological=False)

    obs_opponent = observation_payload.get("opponent") or {}
    if obs_opponent:
        _merge(facts, LAST_SEEN_OPPONENT, "opponent", obs_opponent, "observation.opponent", chronological=False)
    mem_opponent = memory_payload.get("opponent_memory") or {}
    if mem_opponent.get("last_seen_room") is not None:
        _merge(facts, LAST_SEEN_OPPONENT, "opponent", mem_opponent, "memory.opponent_memory", chronological=True)

    for i, d in enumerate(memory_payload.get("recent_decisions", [])):
        _merge(facts, RECENT_DECISIONS, str(i), d, "memory.recent_decisions", chronological=True)
    for i, f in enumerate(memory_payload.get("recent_failures", [])):
        _merge(facts, RECENT_FAILURES, str(i), f, "memory.recent_failures", chronological=True)
    for i, r in enumerate(memory_payload.get("recent_interactions", [])):
        _merge(facts, RECENT_INTERACTIONS, str(i), r, "memory.recent_interactions", chronological=True)

    inventory = self_block.get("inventory")
    if inventory is not None:
        _merge(facts, INVENTORY, "inventory", {"items": inventory}, "observation.self.inventory", chronological=False)

    return list(facts.values())


def request_fact_hash(facts: list[Fact]) -> str:
    """The `request_fact_set` hash (spec section 13): identical for two
    fact lists that carry the same semantic facts, regardless of which
    sections (or how many) contributed each one."""
    canonical = sorted(f.semantic_signature() for f in facts)
    return hashlib.sha256(json.dumps(canonical, default=str).encode("utf-8")).hexdigest()


def facts_by_category(facts: list[Fact]) -> dict[str, list[Fact]]:
    out: dict[str, list[Fact]] = {c: [] for c in FACT_CATEGORIES}
    for f in facts:
        out.setdefault(f.category, []).append(f)
    return out


def duplicate_fact_table(facts: list[Fact]) -> list[dict[str, Any]]:
    """Spec section 64's required dedup table rows - one per fact that
    appears in more than one section."""
    return [
        {
            "fact_id": f.fact_id, "category": f.category, "identity": f.identity,
            "original_locations": list(f.source_sections), "surviving_location": f.source_sections[0],
            "duplicate_locations": list(f.duplicate_locations),
        }
        for f in facts if len(f.source_sections) > 1
    ]
