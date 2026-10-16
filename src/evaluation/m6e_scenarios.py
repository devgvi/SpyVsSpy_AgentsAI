"""M6E spec sections 23-26: the seven scenario families, reusing M6C's own
`memory_snapshot_set.py` snapshots for S0-S5 (spec section 23: "Reuse
M6C/M6C.1 states where valid") and adding exactly one new snapshot,
S6 LONG_HISTORY (spec section 25 - no equivalent state existed before,
since M6C's own snapshots deliberately kept memory small/focused).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src import constants
from src.evaluation.memory_snapshot_set import (
    BACKTRACK_REQUIRED,
    KNOWN_LOCKED_OBJECT,
    LAST_SEEN_OPPONENT,
    MULTIPLE_LEGAL_TARGETS,
    NO_MEMORY_RELEVANT,
    RECENT_FAILED_ACTION,
    build_memory_snapshot_set,
)
from src.evaluation.runtime_snapshot_replay import _fresh_state, _place_near_object, _snapshot_from_state
from src.llm.agent_memory import AgentMemory
from src.llm_controller import NATIVE_TOOL_LIKE_REPRESENTATION, compact_observation_for_model, serialize_observation

S0_NO_HISTORY_RELEVANT = "S0_NO_HISTORY_RELEVANT"
S1_RECENT_FAILED_ACTION = "S1_RECENT_FAILED_ACTION"
S2_KNOWN_LOCKED_OBJECT = "S2_KNOWN_LOCKED_OBJECT"
S3_BACKTRACK_REQUIRED = "S3_BACKTRACK_REQUIRED"
S4_LAST_SEEN_OPPONENT = "S4_LAST_SEEN_OPPONENT"
S5_MULTIPLE_LEGAL_TARGETS = "S5_MULTIPLE_LEGAL_TARGETS"
S6_LONG_HISTORY = "S6_LONG_HISTORY"

SCENARIO_FAMILIES = (
    S0_NO_HISTORY_RELEVANT, S1_RECENT_FAILED_ACTION, S2_KNOWN_LOCKED_OBJECT, S3_BACKTRACK_REQUIRED,
    S4_LAST_SEEN_OPPONENT, S5_MULTIPLE_LEGAL_TARGETS, S6_LONG_HISTORY,
)

_REUSED_CATEGORY_MAP = {
    S0_NO_HISTORY_RELEVANT: NO_MEMORY_RELEVANT,
    S1_RECENT_FAILED_ACTION: RECENT_FAILED_ACTION,
    S2_KNOWN_LOCKED_OBJECT: KNOWN_LOCKED_OBJECT,
    S3_BACKTRACK_REQUIRED: BACKTRACK_REQUIRED,
    S4_LAST_SEEN_OPPONENT: LAST_SEEN_OPPONENT,
    S5_MULTIPLE_LEGAL_TARGETS: MULTIPLE_LEGAL_TARGETS,
}

MEMORY_RELEVANT_FAMILIES = (S1_RECENT_FAILED_ACTION, S2_KNOWN_LOCKED_OBJECT, S3_BACKTRACK_REQUIRED, S6_LONG_HISTORY)


@dataclass(frozen=True)
class M6EScenario:
    scenario_id: str
    family: str
    observation: Any
    legal_actions: list
    memory_payload: dict[str, Any]


def _build_long_history_snapshot() -> M6EScenario:
    """Real, distinct events across ~70s of accumulated play - not
    meaningless filler (spec section 25). Deliberately structured so
    `filing_cabinet_01`'s already-searched-empty fact sits 3rd-from-most-
    recent in `recent_failures`: retained under C0/C3A (LAST_3), lost
    under C3B (LAST_1) - the concrete information-loss boundary spec
    section 70 asks this milestone to be able to identify, now embedded
    in genuinely long history rather than being the only fact present."""
    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "filing_cabinet_01")
    memory = AgentMemory(constants.WHITE_SPY_ID)

    for room_id in ("corridor", "reception", "archives", "security_room", "main_office"):
        memory.record_visited_room(room_id)
    memory.record_known_lock("ambassador_safe", "code", "requires archive_code", satisfied=False)
    memory.record_known_lock("archives_cabinet", "key", "requires archives_key", satisfied=False)

    memory.record_failure(10.0, "search", "filing_cabinet_01", "already_searched", detail="found nothing of value")
    memory.record_failure(40.0, "search", "drawer_01", "already_searched", detail="found nothing of value")
    memory.record_failure(70.0, "search", "locker_02", "already_searched", detail="found nothing of value")

    memory.record_interaction(15.0, "search", "desk_01", "found_item", detail="found a note")
    memory.record_interaction(30.0, "inspect", "ambassador_safe", "no_change")
    memory.record_interaction(55.0, "search", "locker_01", "empty")
    memory.record_interaction(65.0, "inspect", "drawer_01", "no_change")

    memory.record_decision(12.0, "d1", "move", "door_corridor_main_office", True, "llm")
    memory.record_decision(28.0, "d2", "search", "desk_01", True, "llm")
    memory.record_decision(45.0, "d3", "inspect", "ambassador_safe", True, "llm")
    memory.record_decision(60.0, "d4", "search", "locker_01", True, "llm")
    memory.record_decision(72.0, "d5", "search", "locker_02", True, "llm")

    memory.record_event(30.0, "DOOR_OPENED", "door_corridor_main_office")
    memory.record_event(50.0, "ITEM_FOUND", "desk_01")

    ds = _snapshot_from_state("m6e_s6_long_history", S6_LONG_HISTORY, state)
    return M6EScenario(ds.snapshot_id, S6_LONG_HISTORY, ds.observation, ds.legal_actions, memory.to_compact_dict())


def build_compact_observation_payload(scenario: M6EScenario) -> dict[str, Any]:
    """The C0 production observation shape (spec section 8) - the same
    `compact_observation_for_model(..., NATIVE_TOOL_LIKE_REPRESENTATION)`
    call every M6-family Native driver has used since M6D."""
    observation_payload = serialize_observation(scenario.observation)
    observation_payload["legal_actions"] = scenario.legal_actions
    return compact_observation_for_model(observation_payload, representation_variant=NATIVE_TOOL_LIKE_REPRESENTATION)


def build_all_scenarios() -> dict[str, M6EScenario]:
    reused = {s.category: s for s in build_memory_snapshot_set()}
    out: dict[str, M6EScenario] = {}
    for family, category in _REUSED_CATEGORY_MAP.items():
        snap = reused[category]
        out[family] = M6EScenario(snap.snapshot_id, family, snap.observation, snap.legal_actions, snap.synthetic_memory)
    out[S6_LONG_HISTORY] = _build_long_history_snapshot()
    return out
