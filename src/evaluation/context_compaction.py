"""M6E spec sections 6-17: the compaction taxonomy, kept as four distinct
operations (never mixed into one generic "compact" condition - spec
section 6). Operates on the real production shapes: the compact
observation dict (`compact_observation_for_model()`) and
`AgentMemory.to_compact_dict()`. All transforms are deterministic,
field-based, and predeclared - no LLM summarizer (spec section 16), no
relevance heuristic (spec section 17).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

C0_FULL_CONTEXT = "C0_FULL_CONTEXT"
C1_STRUCTURAL_COMPACT = "C1_STRUCTURAL_COMPACT"
C2_DEDUPLICATED = "C2_DEDUPLICATED"
C3A_LAST_3 = "C3A_LAST_3"
C3B_LAST_1 = "C3B_LAST_1"

CONDITIONS = (C0_FULL_CONTEXT, C1_STRUCTURAL_COMPACT, C2_DEDUPLICATED, C3A_LAST_3, C3B_LAST_1)
HISTORY_TRUNCATION_CONDITIONS = (C3A_LAST_3, C3B_LAST_1)
LOSSLESS_CONDITIONS = (C0_FULL_CONTEXT, C1_STRUCTURAL_COMPACT, C2_DEDUPLICATED)

# spec section 17: generic, predeclared, field-based - identical rule set
# applied regardless of snapshot content. Removed only when the same
# semantic fact is confirmed still available at `keep_in` (spec section 12).
DEDUPLICATION_RULES: tuple[dict[str, str], ...] = (
    {"category": "VISITED_ROOMS", "remove_from": "observation.memory.visited_rooms", "keep_in": "memory.exploration.visited_rooms"},
    {"category": "KNOWN_LOCKS", "remove_from": "observation.memory.known_locked_objects", "keep_in": "memory.known_locks"},
    {"category": "MISSION_STATE", "remove_from": "memory.mission", "keep_in": "observation.mission"},
)

# spec section 9: pure representation - carries zero facts beyond
# `legal_actions_compact` (format_legal_actions_tool_like's own docstring:
# "never adds, removes, or reorders which actions/targets are legal").
_C1_REDUNDANT_KEYS = ("available_actions_text",)

_HISTORY_LIST_FIELDS = ("recent_interactions", "recent_failures", "recent_events", "recent_decisions")


@dataclass
class CompactionResult:
    condition: str
    observation_payload: dict[str, Any]
    memory_payload: dict[str, Any]
    removed_report: dict[str, Any]

    def to_json(self) -> str:
        """Only C0 uses production's current serialization (sort_keys,
        default separators - unchanged, spec section 8). Every other
        condition - C1, C2, C3A and C3B - is serialized with compact
        separators (",", ":"); only C1 also drops `available_actions_text`
        (see apply_condition). M6E.3 later showed that this bundling
        confounded M6E's history-truncation arms: C3A/C3B were never
        history-only manipulations. The code is left unchanged so M6E's
        stored results stay reproducible."""
        payload = {**self.observation_payload, "strategy_memory": self.memory_payload}
        if self.condition == C0_FULL_CONTEXT:
            return json.dumps(payload, sort_keys=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def apply_condition(condition: str, observation_payload: dict[str, Any], memory_payload: dict[str, Any]) -> CompactionResult:
    obs = copy.deepcopy(observation_payload)
    mem = copy.deepcopy(memory_payload)
    removed: dict[str, Any] = {"condition": condition}

    if condition == C0_FULL_CONTEXT:
        removed["removed_facts"] = []
        removed["removed_representation_keys"] = []
        return CompactionResult(condition, obs, mem, removed)

    if condition == C1_STRUCTURAL_COMPACT:
        removed_keys = [k for k in _C1_REDUNDANT_KEYS if k in obs]
        for k in _C1_REDUNDANT_KEYS:
            obs.pop(k, None)
        removed["removed_facts"] = []  # fact-preserving by construction
        removed["removed_representation_keys"] = removed_keys
        return CompactionResult(condition, obs, mem, removed)

    if condition == C2_DEDUPLICATED:
        # Safety invariant (spec section 12): remove an entry ONLY when the
        # identical semantic fact is CONFIRMED present at `keep_in` - never
        # the whole section wholesale. A naive "clear the list" approach
        # would silently destroy facts unique to `remove_from` (e.g. a
        # GameState-tracked known lock that never made it into the
        # synthetic AgentMemory seeded for a snapshot) - caught by this
        # module's own network-free equivalence tests before any live call.
        dedup_log = []

        obs_visited = obs.get("memory", {}).get("visited_rooms", [])
        mem_visited = set(mem.get("exploration", {}).get("visited_rooms", []))
        overlap = [r for r in obs_visited if r in mem_visited]
        if overlap:
            dedup_log.append({"category": "VISITED_ROOMS", "removed_from": "observation.memory.visited_rooms", "kept_in": "memory.exploration.visited_rooms", "removed_value": overlap})
            obs["memory"]["visited_rooms"] = [r for r in obs_visited if r not in mem_visited]

        obs_locks = obs.get("memory", {}).get("known_locked_objects", [])
        mem_lock_ids = {lock.get("target_id") for lock in mem.get("known_locks", [])}
        overlap = [t for t in obs_locks if t in mem_lock_ids]
        if overlap:
            dedup_log.append({"category": "KNOWN_LOCKS", "removed_from": "observation.memory.known_locked_objects", "kept_in": "memory.known_locks", "removed_value": overlap})
            obs["memory"]["known_locked_objects"] = [t for t in obs_locks if t not in mem_lock_ids]

        obs_mission = obs.get("mission")
        mem_mission = mem.get("mission")
        if obs_mission and mem_mission and obs_mission == mem_mission:
            dedup_log.append({"category": "MISSION_STATE", "removed_from": "memory.mission", "kept_in": "observation.mission", "removed_value": mem_mission})
            mem.pop("mission", None)

        removed["removed_facts"] = []  # facts survive in `keep_in`, never actually lost
        removed["deduplication_log"] = dedup_log
        return CompactionResult(condition, obs, mem, removed)

    if condition in HISTORY_TRUNCATION_CONDITIONS:
        keep_n = 3 if condition == C3A_LAST_3 else 1
        history_loss: dict[str, Any] = {}
        for field_name in _HISTORY_LIST_FIELDS:
            items = mem.get(field_name, [])
            if len(items) > keep_n:
                # keep the LAST keep_n items (most recent), in original
                # chronological order (spec section 15: never reorder).
                kept = items[-keep_n:]
                removed_items = items[:-keep_n]
                mem[field_name] = kept
                history_loss[field_name] = {"removed_count": len(removed_items), "removed_items": removed_items, "retained_count": len(kept)}
            else:
                history_loss[field_name] = {"removed_count": 0, "removed_items": [], "retained_count": len(items)}
        removed["history_loss"] = history_loss
        removed["removed_facts"] = "see history_loss - genuinely lossy, not fact-preserving (spec section 34: do not call C3 lossless)"
        return CompactionResult(condition, obs, mem, removed)

    raise ValueError(f"unknown compaction condition {condition!r}")


def token_estimate(text: str) -> int:
    """A simple, deterministic characters/4 heuristic - used only for the
    static (pre-live) audit; real provider `input_tokens` from telemetry
    is used wherever available for live-call rows (spec section 21)."""
    return max(1, len(text) // 4)


def representation_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def section_hashes(observation_payload: dict[str, Any], memory_payload: dict[str, Any]) -> dict[str, str]:
    def h(v: Any) -> str:
        return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    return {
        "self": h(observation_payload.get("self")),
        "room": h(observation_payload.get("room")),
        "mission": h(observation_payload.get("mission")),
        "legal_actions_compact": h(observation_payload.get("legal_actions_compact")),
        "memory_exploration": h(memory_payload.get("exploration")),
        "memory_known_locks": h(memory_payload.get("known_locks")),
        "memory_recent_decisions": h(memory_payload.get("recent_decisions")),
        "memory_recent_failures": h(memory_payload.get("recent_failures")),
        "memory_opponent": h(memory_payload.get("opponent_memory")),
    }


def history_hash(memory_payload: dict[str, Any]) -> str:
    history = {k: memory_payload.get(k) for k in _HISTORY_LIST_FIELDS}
    return hashlib.sha256(json.dumps(history, sort_keys=True, default=str).encode("utf-8")).hexdigest()
