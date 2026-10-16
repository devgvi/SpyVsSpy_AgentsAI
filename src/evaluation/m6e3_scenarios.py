"""M6E.3 spec sections 3-5, 9-10: the three scenarios (LONG_HISTORY
primary, S1/S0 mandatory controls), with representation completely frozen
to M6E's own C0_FULL_CONTEXT (natural duplicate locations, full-format
serialization - spec section 5 explicitly forbids reopening the
auxiliary-location/compact-format axes M6E.1/M6E.2 already localized).
Only the history window varies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.context_compaction import C0_FULL_CONTEXT, apply_condition
from src.evaluation.history_window import WINDOW_ORDER, WindowResult, apply_window
from src.evaluation.m6e_scenarios import (
    S0_NO_HISTORY_RELEVANT,
    S1_RECENT_FAILED_ACTION,
    S6_LONG_HISTORY,
    build_all_scenarios,
    build_compact_observation_payload,
)
from src.llm.agent_memory import AgentMemory

PRIMARY_FAMILIES = (S6_LONG_HISTORY, S1_RECENT_FAILED_ACTION, S0_NO_HISTORY_RELEVANT)

# spec section 17: the exact known-failure event for LONG_HISTORY, and
# spec section 19: the frozen primary outcome definition - identified and
# recorded BEFORE any live call.
KNOWN_FAILURE_EVENT = {"field": "recent_failures", "target_id": "filing_cabinet_01", "game_time": 10.0}
KNOWN_FAILURE_ACTION_SIGNATURE = {"action": "interact", "target_id": "filing_cabinet_01"}  # normalized


@dataclass(frozen=True)
class M6E3Scenario:
    scenario_id: str
    family: str
    observation: Any
    legal_actions: list
    observation_payload: dict[str, Any]  # frozen, natural, full-format (C0)
    memory_payload_full: dict[str, Any]  # H_FULL memory, before any window truncation


def _seed_s0_history() -> dict[str, Any]:
    """S0_NO_HISTORY_RELEVANT's own real frozen state has ZERO history
    entries by M6C's original design (a genuinely empty negative control -
    see memory_snapshot_set.py's own docstring). Spec section 37 needs
    S0 to have SOME history to shorten (to test generic salience under
    truncation, not just under total absence) - so this seeds a modest,
    plausible-but-IRRELEVANT history (objects/rooms with no bearing on
    S0's current decision, which is an open floor with nothing
    interactable nearby) via the same real `AgentMemory.record_*` API
    every other M6-family synthetic memory uses. This is an M6E.3-specific
    augmentation only - M6C's and M6E's own frozen S0 snapshots elsewhere
    are never touched."""
    memory = AgentMemory("white_spy")
    memory.record_interaction(5.0, "search", "locker_01", "empty")
    memory.record_interaction(20.0, "search", "drawer_01", "found_item", detail="found a memo")
    memory.record_failure(15.0, "interact", "archives_cabinet", "locked", detail="requires archives_key")
    memory.record_failure(35.0, "interact", "ambassador_safe", "locked", detail="requires a code")
    memory.record_event(10.0, "DOOR_OPENED", "door_corridor_main_office")
    memory.record_decision(3.0, "d1", "move", "door_main_office_ambassador", True, "llm")
    memory.record_decision(25.0, "d2", "search", "locker_01", True, "llm")
    memory.record_decision(40.0, "d3", "interact", "ambassador_safe", True, "llm")
    return memory.to_compact_dict()


def build_primary_scenarios() -> dict[str, M6E3Scenario]:
    base = build_all_scenarios()
    out = {}
    for family in PRIMARY_FAMILIES:
        s = base[family]
        obs = build_compact_observation_payload(s)
        memory_payload = _seed_s0_history() if family == S0_NO_HISTORY_RELEVANT else s.memory_payload
        c0 = apply_condition(C0_FULL_CONTEXT, obs, memory_payload)
        out[family] = M6E3Scenario(s.scenario_id, family, s.observation, s.legal_actions, c0.observation_payload, c0.memory_payload)
    return out


def build_window(scenario: M6E3Scenario, window: str) -> WindowResult:
    return apply_window(scenario.memory_payload_full, window)


def repeats_known_failure(action: str, target_id: str) -> bool:
    from src.llm_controller import _normalize_action_name

    return _normalize_action_name(action) == KNOWN_FAILURE_ACTION_SIGNATURE["action"] and target_id == KNOWN_FAILURE_ACTION_SIGNATURE["target_id"]
