"""M6D spec sections 18-19, 38: a frozen invalid-action corpus, one entry
per supported failure family, each built from a REAL `GameState`/
`Observation`/legal-action set (never a synthetic dict detached from
gameplay semantics) - reusing existing, already-validated snapshot
builders from M6A/M6B/M6B.2 rather than inventing new game states.

Scope decision, documented rather than silently applied (spec section 39
offers historical-log mining as an option): this corpus is built fresh
against real frozen states, not mined from prior milestones' raw
telemetry files - consistent with this session's established cost/scope
discipline (e.g. M6C's own full-episode scope decision). Each entry's
`source` field still names which milestone/mechanism it evidences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.evaluation.runtime_snapshot_replay import build_snapshot_set
from src.evaluation.target_legality_snapshot_set import build_target_legality_snapshot_set
from src.evaluation.trap_snapshot_set import build_trap_snapshot_set
from src.llm_controller import validate_model_output

INVALID_ACTION_TYPE = "INVALID_ACTION_TYPE"
INVALID_TARGET = "INVALID_TARGET"
TARGET_NOT_INSPECTED = "TARGET_NOT_INSPECTED"
INVALID_TRAP_TYPE = "INVALID_TRAP_TYPE"
DEPLETED_TRAP_TYPE = "DEPLETED_TRAP_TYPE"
MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
WRONG_ARGUMENT_TYPE = "WRONG_ARGUMENT_TYPE"
MALFORMED_JSON_OR_TOOL_CALL = "MALFORMED_JSON_OR_TOOL_CALL"
ACTION_CURRENTLY_ILLEGAL = "ACTION_CURRENTLY_ILLEGAL"

CORPUS_FAILURE_FAMILIES = (
    INVALID_ACTION_TYPE, INVALID_TARGET, TARGET_NOT_INSPECTED, INVALID_TRAP_TYPE, DEPLETED_TRAP_TYPE,
    MISSING_REQUIRED_FIELD, WRONG_ARGUMENT_TYPE, MALFORMED_JSON_OR_TOOL_CALL, ACTION_CURRENTLY_ILLEGAL,
)


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    failure_family: str
    source: str
    observation: Any
    legal_actions: list[dict[str, Any]] = field(repr=False)
    invalid_raw_response: dict[str, Any] = field(repr=False)
    validation_error_type: str
    validation_error_message: str


def build_invalid_action_corpus() -> list[CorpusCase]:
    trap_snapshots = {s.category: s for s in build_trap_snapshot_set()}
    target_legality_snapshots = {s.category: s for s in build_target_legality_snapshot_set()}
    base_snapshots = {s.category: s for s in build_snapshot_set()}

    cases: list[CorpusCase] = []

    def add(case_id, family, source, snap, raw_response):
        v = validate_model_output(snap.observation, raw_response, snap.legal_actions)
        assert not v.valid, f"{case_id} was expected to be invalid but validated OK"
        cases.append(CorpusCase(
            case_id=case_id, failure_family=family, source=source, observation=snap.observation,
            legal_actions=snap.legal_actions, invalid_raw_response=raw_response,
            validation_error_type=v.error_type, validation_error_message=v.error_message,
        ))

    # 1. INVALID_ACTION_TYPE - an unrecognized action name entirely.
    add(
        "case01_invalid_action_type", INVALID_ACTION_TYPE, "M5B.1 (unknown_action category)",
        trap_snapshots["MULTIPLE_TYPES_REMAINING"],
        {"action": "attack", "target_id": "ambassador_safe", "reason": "x"},
    )

    # 2. INVALID_TARGET - a target_id naming no real object in this
    # actor's own Observation.
    add(
        "case02_invalid_target", INVALID_TARGET, "M5D.4/M5D.5 (unknown_target category)",
        base_snapshots["MULTIPLE_LEGAL_TARGETS"],
        {"action": "inspect", "target_id": "not_a_real_object_xyz", "reason": "x"},
    )

    # 3. TARGET_NOT_INSPECTED - the exact M6B.1/M6B.2 mechanism.
    add(
        "case03_target_not_inspected", TARGET_NOT_INSPECTED, "M6B.2 (target_not_inspected category)",
        target_legality_snapshots["UNINSPECTED_TRAPPABLE_NEARBY"],
        {"action": "place_trap", "target_id": target_legality_snapshots["UNINSPECTED_TRAPPABLE_NEARBY"].observation.nearby_object_id, "trap_type": "snare", "reason": "x"},
    )

    # 4. INVALID_TRAP_TYPE - the M6B "stun" hallucination.
    add(
        "case04_invalid_trap_type", INVALID_TRAP_TYPE, "M6B (invalid_trap_type category)",
        trap_snapshots["MULTIPLE_TYPES_REMAINING"],
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"},
    )

    # 5. DEPLETED_TRAP_TYPE - a real trap type this actor no longer holds.
    depleted_snap = trap_snapshots["ONE_TYPE_DEPLETED"]
    from src import constants

    depleted_type = next(t for t, c in depleted_snap.observation.trap_inventory.items() if c == 0)
    add(
        "case05_depleted_trap_type", DEPLETED_TRAP_TYPE, "M5F.3B.1 (trap_type_not_in_inventory category)",
        depleted_snap,
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": depleted_type, "reason": "x"},
    )

    # 6. MISSING_REQUIRED_FIELD - target_id omitted for an action that requires it.
    add(
        "case06_missing_required_field", MISSING_REQUIRED_FIELD, "M5B.1 (missing_target category)",
        base_snapshots["MULTIPLE_LEGAL_TARGETS"],
        {"action": "inspect", "reason": "x"},
    )

    # 7. WRONG_ARGUMENT_TYPE - target_id present but the wrong JSON type.
    add(
        "case07_wrong_argument_type", WRONG_ARGUMENT_TYPE, "M5B.1 (invalid_target: not a string, category)",
        base_snapshots["MULTIPLE_LEGAL_TARGETS"],
        {"action": "inspect", "target_id": 12345, "reason": "x"},
    )

    # 8. MALFORMED_JSON_OR_TOOL_CALL - schema_error: action not even a string.
    add(
        "case08_malformed_tool_call", MALFORMED_JSON_OR_TOOL_CALL, "M5C.1 (schema_error category)",
        base_snapshots["MULTIPLE_LEGAL_TARGETS"],
        {"action": 123, "reason": "x"},
    )

    # 9. ACTION_CURRENTLY_ILLEGAL - interact on a real, in-range, locked
    # object the actor cannot currently unlock - INTERACT is not offered
    # at all for it (enumerate_legal_actions's own lock gate).
    add(
        "case09_action_currently_illegal", ACTION_CURRENTLY_ILLEGAL, "M5D.4 (illegal_target category)",
        base_snapshots["LOCKED_OBJECT"],
        {"action": "interact", "target_id": base_snapshots["LOCKED_OBJECT"].observation.nearby_object_id, "reason": "x"},
    )

    return cases
