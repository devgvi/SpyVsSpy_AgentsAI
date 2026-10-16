"""M6B: pure, network-free PLACE_TRAP reliability classification (spec
section 9's taxonomy) - retrospective analysis of already-collected
M6A.2 telemetry (Phase A, spec section 27) and the shared classifier
Phase C's mock tool-call tests reuse.

Core principle (spec section 1): a tool-call failure is not "the model is
bad at tools" until classified against the actual harness contract. This
module never asserts a mechanism - it only counts and categorizes what
the telemetry already recorded, using the game's own real validator
error_types (see `src/llm_controller.py`'s `validate_model_output`),
never a parallel/approximate reimplementation of that logic.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.llm_controller import _normalize_action_name

# -- M6B failure taxonomy (spec section 9) -----------------------------------

SUCCESS = "SUCCESS"
MISSING_TRAP_TYPE = "MISSING_TRAP_TYPE"
INVALID_TRAP_TYPE = "INVALID_TRAP_TYPE"
DEPLETED_TRAP_TYPE = "DEPLETED_TRAP_TYPE"
MISSING_TARGET = "MISSING_TARGET"
INVALID_TARGET = "INVALID_TARGET"
WRONG_ARGUMENT_NAME = "WRONG_ARGUMENT_NAME"
WRONG_ARGUMENT_TYPE = "WRONG_ARGUMENT_TYPE"
MALFORMED_TOOL_CALL = "MALFORMED_TOOL_CALL"
TOOL_NOT_SELECTED = "TOOL_NOT_SELECTED"
VALIDATION_REJECTED_OTHER = "VALIDATION_REJECTED_OTHER"
RUNTIME_PARSE_FAILURE = "RUNTIME_PARSE_FAILURE"

ALL_FAILURE_CATEGORIES = (
    MISSING_TRAP_TYPE, INVALID_TRAP_TYPE, DEPLETED_TRAP_TYPE, MISSING_TARGET, INVALID_TARGET,
    WRONG_ARGUMENT_NAME, WRONG_ARGUMENT_TYPE, MALFORMED_TOOL_CALL, VALIDATION_REJECTED_OTHER, RUNTIME_PARSE_FAILURE,
)

# Real, provider/runtime-level error_types (never a validate_model_output
# game-rule rejection) that mean "the call itself never produced a usable
# proposal" - classified as RUNTIME_PARSE_FAILURE when they occur on an
# attempt this module has independent evidence was aimed at place_trap
# (e.g. the initial proposal, before the failure, already named it).
_RUNTIME_LEVEL_ERROR_TYPES = frozenset({
    "provider_error", "strands_runtime_error", "watchdog_timeout", "no_tool_selected",
    "empty_response", "json_parse_error", "schema_error",
})


def _is_place_trap_proposal(proposal: dict[str, Any] | None) -> bool:
    if not proposal:
        return False
    action = proposal.get("action")
    if not isinstance(action, str) or not action.strip():
        return False
    try:
        return _normalize_action_name(action) == "place_trap"
    except Exception:  # pragma: no cover - defensive, an unrecognized raw string
        return False


def _classify_error(error_type: str | None, proposal: dict[str, Any] | None) -> str:
    """Maps the game's own real validator `error_type` (never a
    reimplementation of `validate_model_output`'s own logic) to the M6B
    taxonomy. `missing_target` is deliberately overloaded by the engine
    for place_trap (see `validate_model_output`'s own comment: "a
    place_trap's very first rejection is often missing_target (trap_type
    omitted entirely)") - disambiguated here by inspecting the actual
    proposal dict's own `trap_type`/`target_id` fields, never guessed."""
    proposal = proposal or {}
    if error_type == "missing_target":
        trap_type = proposal.get("trap_type")
        if not trap_type or not isinstance(trap_type, str):
            return MISSING_TRAP_TYPE
        return MISSING_TARGET
    if error_type == "invalid_target":
        return INVALID_TARGET
    if error_type == "invalid_trap_type":
        return INVALID_TRAP_TYPE
    if error_type == "trap_type_not_in_inventory":
        return DEPLETED_TRAP_TYPE
    if error_type in _RUNTIME_LEVEL_ERROR_TYPES:
        return RUNTIME_PARSE_FAILURE
    if error_type in ("missing_action", "unknown_action"):
        return MALFORMED_TOOL_CALL
    return VALIDATION_REJECTED_OTHER


def classify_place_trap_event(event: dict[str, Any]) -> str | None:
    """Returns one of `SUCCESS`/the failure taxonomy, or `None` if this
    decision was never a PLACE_TRAP attempt at all (spec section 32's
    "tool selection vs argument reliability" distinction - a `None`
    result means "not counted in the PLACE_TRAP denominator", not a
    failure).

    Real bug found and fixed while first validating this function against
    M6A.2's actual telemetry: a repair does not have to retry the SAME
    action - the model can abandon `place_trap` entirely and propose a
    completely different action on repair (observed live: an initial
    `place_trap` with a hallucinated `trap_type` like "stun", followed by
    a repair that proposes `move` instead, which the repair mechanism
    accepts as valid on its own terms). In that case the telemetry's
    top-level `error_type` (post-repair) is `None` - it describes the
    *accepted* `move`, not the abandoned `place_trap` - so classifying
    the place_trap failure from it produced nonsense (`VALIDATION_REJECTED_OTHER`
    for what was really `INVALID_TRAP_TYPE`). The correct source per case:
    the repair's own error_type/proposal only when the repair *itself*
    still targeted place_trap; the initial attempt's own error_type/
    proposal whenever place_trap was abandoned on repair (or no repair
    was attempted at all)."""
    executed = event.get("executed_action") or {}
    final_valid = bool(event.get("final_model_action_valid") or event.get("valid"))
    if final_valid and _normalize_action_name(executed.get("action") or "wait") == "place_trap":
        return SUCCESS

    repair_attempted = bool(event.get("repair_attempted"))
    initial_proposal = event.get("raw_response") or event.get("initial_proposal")
    repair_proposal = event.get("repair_raw_response")
    repair_targeted_place_trap = repair_attempted and _is_place_trap_proposal(repair_proposal)

    if not repair_targeted_place_trap and not _is_place_trap_proposal(initial_proposal):
        return None  # never attempted place_trap at all this decision

    if repair_targeted_place_trap:
        return _classify_error(event.get("repair_error_type"), repair_proposal)
    return _classify_error(event.get("initial_error_type"), initial_proposal)


# -- M6B.1 PROCESS taxonomy (spec section 7) ---------------------------------
#
# A second, orthogonal dimension to the REASON taxonomy above: WHAT
# HAPPENED during the decision (first-pass valid, repaired, abandoned,
# fell back, or a genuine runtime/provider-level call failure), as
# opposed to WHY a validation rejection occurred. Reported separately
# (spec sections 7, 41-42), never blended into one combined label.

VALID_FIRST_PASS = "VALID_FIRST_PASS"
VALID_AFTER_REPAIR = "VALID_AFTER_REPAIR"
ABANDONED_AFTER_REPAIR = "ABANDONED_AFTER_REPAIR"
FALLBACK_AFTER_FAILED_REPAIR = "FALLBACK_AFTER_FAILED_REPAIR"
RUNTIME_FAILURE = "RUNTIME_FAILURE"
PROVIDER_FAILURE = "PROVIDER_FAILURE"

ALL_PROCESS_OUTCOMES = (
    VALID_FIRST_PASS, VALID_AFTER_REPAIR, ABANDONED_AFTER_REPAIR,
    FALLBACK_AFTER_FAILED_REPAIR, RUNTIME_FAILURE, PROVIDER_FAILURE,
)

# error_types that mean "the call itself failed", split into the two
# process-level categories spec section 7 asks be kept distinct (never
# conflated the way the pre-M6A.2 `fallback_count` conflation did).
_PROVIDER_FAILURE_ERROR_TYPES = frozenset({"provider_error"})
_RUNTIME_FAILURE_ERROR_TYPES = frozenset({"strands_runtime_error", "watchdog_timeout"})


def classify_place_trap_process(event: dict[str, Any]) -> str | None:
    """Returns one of `ALL_PROCESS_OUTCOMES`, or `None` if PLACE_TRAP was
    never the model's INITIAL proposal this decision (the process
    dimension is only meaningful starting from an actual place_trap
    attempt - spec section 8's own framing: "if initial output is
    PLACE_TRAP but repair changes it... classify ABANDONED_AFTER_REPAIR,
    not successful trap placement" presupposes place_trap was the
    starting point)."""
    initial_proposal = event.get("raw_response") or event.get("initial_proposal")
    if not _is_place_trap_proposal(initial_proposal):
        return None

    executed = event.get("executed_action") or {}
    final_valid = bool(event.get("final_model_action_valid") or event.get("valid"))
    repair_attempted = bool(event.get("repair_attempted"))

    if not repair_attempted:
        if final_valid and _normalize_action_name(executed.get("action") or "wait") == "place_trap":
            return VALID_FIRST_PASS
        initial_error_type = event.get("initial_error_type")
        if initial_error_type in _PROVIDER_FAILURE_ERROR_TYPES:
            return PROVIDER_FAILURE
        if initial_error_type in _RUNTIME_FAILURE_ERROR_TYPES:
            return RUNTIME_FAILURE
        # No repair budget was spent (max_retries==0) and the initial
        # attempt failed - the engine's own fallback applies with no
        # second chance; still a fallback, not an abandonment (nothing
        # was ever proposed to abandon TO).
        return FALLBACK_AFTER_FAILED_REPAIR

    repair_proposal = event.get("repair_raw_response")
    if not _is_place_trap_proposal(repair_proposal):
        return ABANDONED_AFTER_REPAIR

    if final_valid and _normalize_action_name(executed.get("action") or "wait") == "place_trap":
        return VALID_AFTER_REPAIR
    repair_error_type = event.get("repair_error_type")
    if repair_error_type in _PROVIDER_FAILURE_ERROR_TYPES:
        return PROVIDER_FAILURE
    if repair_error_type in _RUNTIME_FAILURE_ERROR_TYPES:
        return RUNTIME_FAILURE
    return FALLBACK_AFTER_FAILED_REPAIR


def summarize_place_trap_process(events: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = [classify_place_trap_process(e) for e in events]
    attempts = [c for c in classifications if c is not None]
    counts = Counter(attempts)
    total = len(attempts)
    return {
        "initial_place_trap_selections": total,
        "counts_by_outcome": {k: counts.get(k, 0) for k in ALL_PROCESS_OUTCOMES if counts.get(k, 0) > 0},
        "conditional_valid_rate": (
            (counts.get(VALID_FIRST_PASS, 0) + counts.get(VALID_AFTER_REPAIR, 0)) / total if total else None
        ),
    }


# M6B.1 spec sections 9, 42's own reduced 5-category REASON table - a
# rollup view of M6B's own finer taxonomy above, never a replacement for
# it (the full breakdown is still reported - see `summarize_place_trap_attempts`).
_M6B1_REASON_ROLLUP = {
    MISSING_TRAP_TYPE: MISSING_TRAP_TYPE,
    INVALID_TRAP_TYPE: INVALID_TRAP_TYPE,
    DEPLETED_TRAP_TYPE: DEPLETED_TRAP_TYPE,
    INVALID_TARGET: INVALID_TARGET,
}
OTHER_VALIDATION_FAILURE = "OTHER_VALIDATION_FAILURE"


def rollup_reason_to_m6b1_table(category: str) -> str:
    return _M6B1_REASON_ROLLUP.get(category, OTHER_VALIDATION_FAILURE)


def summarize_place_trap_attempts(events: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = [classify_place_trap_event(e) for e in events]
    attempts = [c for c in classifications if c is not None]
    counts = Counter(attempts)
    total = len(attempts)
    successes = counts.get(SUCCESS, 0)
    failures = total - successes
    reason_rollup = Counter(rollup_reason_to_m6b1_table(c) for c in attempts if c != SUCCESS)
    return {
        "attempts": total,
        "successful": successes,
        "failed": failures,
        "failure_rate": (failures / total) if total else None,
        "counts_by_category": {k: counts.get(k, 0) for k in (SUCCESS,) + ALL_FAILURE_CATEGORIES if counts.get(k, 0) > 0},
        "m6b1_reason_table": {k: reason_rollup.get(k, 0) for k in (MISSING_TRAP_TYPE, INVALID_TRAP_TYPE, DEPLETED_TRAP_TYPE, INVALID_TARGET, OTHER_VALIDATION_FAILURE) if reason_rollup.get(k, 0) > 0},
    }
