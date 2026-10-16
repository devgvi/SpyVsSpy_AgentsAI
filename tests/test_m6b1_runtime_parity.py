"""M6B.1 Phase A (network-free parity tests, spec section 22) + Phase B
(mock repair matrix, spec sections 17, 23) - the executable enforcement
behind `src/evaluation/runtime_parity.py`'s manifest (RP01-RP11). Also
covers the PROCESS-outcome taxonomy (spec section 7) unit tests. No
network calls.

Personality/model/region parity (RP01/RP02) already live in
`tests/test_m6b_harness_invariants.py` - not duplicated here, referenced
by the manifest instead (spec section 21: prefer one test, not two
independently-drifting copies).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import inspect

import pytest

from src import constants
from src.agents.native_agent import NativeAgentAdapter
from src.agents.strands_agent import StrandsAgentAdapter
from src.evaluation.place_trap_reliability_analysis import (
    ABANDONED_AFTER_REPAIR,
    DEPLETED_TRAP_TYPE,
    FALLBACK_AFTER_FAILED_REPAIR,
    INVALID_TARGET,
    INVALID_TRAP_TYPE,
    MISSING_TARGET,
    MISSING_TRAP_TYPE,
    SUCCESS,
    VALID_AFTER_REPAIR,
    VALID_FIRST_PASS,
    classify_place_trap_event,
    classify_place_trap_process,
)
from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST, RUNTIME_SPECIFIC, manifest_as_dicts, parity_summary
from src.evaluation.runtime_snapshot_replay import build_snapshot_set, run_one_trial
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig, VALID_ACTION_NAMES, _normalize_action_name, serialize_observation
from tests.test_strands_agent import QueuedStrandsFactory

TRAP_SNAPSHOT = next(s for s in build_snapshot_set() if s.category == "TRAP_PLACEMENT_OPPORTUNITY")


def _native_trial(response, obs=TRAP_SNAPSHOT.observation, episode_id="m6b1-native"):
    telemetry = TelemetryLogger()
    adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=MockModelClient(response=response),
        config=LLMControllerConfig(), telemetry=telemetry, episode_id=episode_id,
    )
    run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)
    return next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {}), adapter


def _strands_trial(queued_calls, obs=TRAP_SNAPSHOT.observation, episode_id="m6b1-strands"):
    telemetry = TelemetryLogger()
    factory = QueuedStrandsFactory(queued_calls)
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory, telemetry=telemetry, episode_id=episode_id)
    run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)
    return next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {}), adapter


def _held_trap_type():
    return next(t for t, c in TRAP_SNAPSHOT.observation.trap_inventory.items() if c > 0)


def _depleted_trap_type():
    import dataclasses

    depleted_type = constants.TRAP_TYPES[0]
    thin_inventory = dict(TRAP_SNAPSHOT.observation.trap_inventory)
    thin_inventory[depleted_type] = 0
    return depleted_type, dataclasses.replace(TRAP_SNAPSHOT.observation, trap_inventory=thin_inventory)


# -- the manifest itself is internally consistent ----------------------------


def test_manifest_has_unique_ids():
    # M6B.2 added RP12_LEGAL_TARGET_HINTS (spec section 14) - exact count
    # asserted once, in tests/test_m6b2_target_legality_and_payload_parity.py
    # ::test_manifest_now_has_twelve_entries_with_unique_ids, not duplicated
    # here (spec section 21: prefer one test, not two independently-drifting
    # copies of the same fact).
    ids = [e.entry_id for e in RUNTIME_PARITY_MANIFEST]
    assert len(ids) == len(set(ids))


def test_manifest_every_required_entry_has_a_real_test_id():
    for entry in RUNTIME_PARITY_MANIFEST:
        if entry.parity_required and entry.status not in (RUNTIME_SPECIFIC,):
            assert entry.test_id and entry.test_id != "n/a", entry.entry_id


def test_manifest_as_dicts_and_summary_are_consistent():
    dicts = manifest_as_dicts()
    assert len(dicts) == len(RUNTIME_PARITY_MANIFEST)
    summary = parity_summary()
    assert summary["total_entries"] == len(RUNTIME_PARITY_MANIFEST)
    assert sum(summary["counts_by_status"].values()) == len(RUNTIME_PARITY_MANIFEST)


def test_no_required_parity_entry_is_silently_passed_without_evidence():
    # Spec section 29: a parity violation must STOP behavioral
    # interpretation, never be silently marked PASS.
    #
    # M6B.2 update: RP10 was this manifest's one real, documented FAIL at
    # M6B.1 time - fixed in M6B.2 (StrandsAgentAdapter now accepts
    # model_payload_transform too, see tests/
    # test_m6b2_target_legality_and_payload_parity.py's own RP10 tests),
    # so no FAIL should remain unexplained in the manifest at all now
    # (spec section 40's parity test gate).
    for entry in RUNTIME_PARITY_MANIFEST:
        assert entry.status != FAIL, f"unexplained FAIL: {entry.entry_id}"
    rp10 = next(e for e in RUNTIME_PARITY_MANIFEST if e.entry_id == "RP10_MEMORY_FLAGS")
    assert "model_payload_transform" in rp10.notes


# -- RP03: action contract parity --------------------------------------------


def test_rp03_action_contract_parity():
    # Both runtimes' own action vocabulary funnels through the identical
    # normalizer/enum - never two independently-maintained lists.
    for name in VALID_ACTION_NAMES:
        assert _normalize_action_name(name) == name
    # Strands' own choose_action signature covers exactly the same action
    # set via its `action: str` argument, validated identically (RP04).
    from src.agents.strands_agent import TOOL_DESCRIPTION

    for name in ("wait", "move", "interact", "inspect", "place_trap", "disarm"):
        assert name in TOOL_DESCRIPTION


# -- RP04: validation parity matrix (spec section 17) ------------------------


@pytest.mark.parametrize("case", [
    "missing_trap_type", "invalid_stun", "depleted_valid_trap_type", "invalid_target", "missing_target", "valid_place_trap",
])
def test_rp04_validation_parity_matrix(case):
    held = _held_trap_type()
    obs = TRAP_SNAPSHOT.observation
    expected = None

    if case == "missing_trap_type":
        native_resp = [{"action": "place_trap", "target_id": "ambassador_safe", "reason": "x"}, {"action": "wait", "reason": "y"}]
        strands_calls = [[{"action": "place_trap", "target_id": "ambassador_safe", "reason": "x"}], [{"action": "wait", "reason": "y"}]]
        expected = MISSING_TRAP_TYPE
    elif case == "invalid_stun":
        native_resp = [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"}, {"action": "wait", "reason": "y"}]
        strands_calls = [[{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"}], [{"action": "wait", "reason": "y"}]]
        expected = INVALID_TRAP_TYPE
    elif case == "depleted_valid_trap_type":
        depleted_type, obs = _depleted_trap_type()
        native_resp = [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": depleted_type, "reason": "x"}, {"action": "wait", "reason": "y"}]
        strands_calls = [[{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": depleted_type, "reason": "x"}], [{"action": "wait", "reason": "y"}]]
        expected = DEPLETED_TRAP_TYPE
    elif case == "invalid_target":
        native_resp = [{"action": "place_trap", "target_id": 12345, "trap_type": held, "reason": "x"}, {"action": "wait", "reason": "y"}]
        strands_calls = [[{"action": "place_trap", "target_id": 12345, "trap_type": held, "reason": "x"}], [{"action": "wait", "reason": "y"}]]
        expected = INVALID_TARGET
    elif case == "missing_target":
        native_resp = [{"action": "place_trap", "trap_type": held, "reason": "x"}, {"action": "wait", "reason": "y"}]
        strands_calls = [[{"action": "place_trap", "trap_type": held, "reason": "x"}], [{"action": "wait", "reason": "y"}]]
        expected = MISSING_TARGET
    else:  # valid_place_trap
        native_resp = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held, "reason": "x"}
        strands_calls = [[{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held, "reason": "x"}]]
        expected = SUCCESS

    native_event, _ = _native_trial(native_resp, obs=obs)
    strands_event, _ = _strands_trial(strands_calls, obs=obs)
    native_result = classify_place_trap_event(native_event)
    strands_result = classify_place_trap_event(strands_event)
    assert native_result == expected, f"native: {native_result}"
    assert strands_result == expected, f"strands: {strands_result}"


# -- RP05: repair budget parity -----------------------------------------------


def test_rp05_repair_budget_parity():
    native_event, native_adapter = _native_trial([
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"},
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun_trap", "reason": "still wrong"},
    ])
    strands_event, strands_adapter = _strands_trial([
        [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"}],
        [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun_trap", "reason": "still wrong"}],
    ])
    assert native_adapter.controller.config.max_retries == strands_adapter.max_retries == 1
    # Exactly one repair attempted, never a second - both fall back.
    assert native_event.get("repair_attempted") is True
    assert strands_event.get("repair_attempted") is True
    assert native_event.get("fallback_used") is True
    assert strands_event.get("fallback_used") is True


# -- RP07: fallback shape parity ----------------------------------------------


def test_rp07_fallback_shape_parity():
    native_event, _ = _native_trial([
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"},
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun_trap", "reason": "y"},
    ])
    strands_event, _ = _strands_trial([
        [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"}],
        [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun_trap", "reason": "y"}],
    ])
    assert native_event.get("executed_action", {}).get("action") == "wait"
    assert strands_event.get("executed_action", {}).get("action") == "wait"


# -- RP10: within-episode memory parity + documented cross-episode gap ------


def test_rp10_within_episode_memory_parity():
    from src.llm.agent_memory import AgentMemory

    native_event, native_adapter = _native_trial({"action": "wait", "reason": "x"})
    strands_event, strands_adapter = _strands_trial([[{"action": "wait", "reason": "x"}]])
    # Both adapters end up with the same underlying AgentMemory shape -
    # AgentMemoryTracker.memory, the same shared class either way (never
    # two independently-drifting memory representations).
    assert isinstance(native_adapter.controller.memory, AgentMemory)
    assert isinstance(strands_adapter._memory_tracker.memory, AgentMemory)


def test_rp10_cross_episode_hook_now_exists_on_both_runtimes():
    # M6B.1 documented this as Native-only (a real gap, never silently
    # assumed equivalent - spec section 29). M6B.2 closes it (spec
    # sections 15-21) - both now accept the same-shaped hook. See
    # tests/test_m6b2_target_legality_and_payload_parity.py for the fuller
    # RP10 parity suite (application-once semantics, no-transform
    # identity, hash parity).
    native_params = inspect.signature(LLMControllerConfig).parameters
    assert "model_payload_transform" in native_params
    strands_params = inspect.signature(StrandsAgentAdapter.__init__).parameters
    assert "model_payload_transform" in strands_params


# -- RP11: telemetry field parity ---------------------------------------------


def test_rp11_telemetry_field_parity():
    native_event, _ = _native_trial({"action": "move", "target_id": "desk_01", "reason": "x"})
    strands_event, _ = _strands_trial([[{"action": "move", "target_id": "desk_01", "reason": "x"}]])
    shared_fields = (
        "observation_hash", "legal_actions", "executed_action", "raw_response", "repair_raw_response",
        "valid", "final_model_action_valid", "repair_attempted", "fallback_used",
        "provider", "model", "agent_runtime", "representation_variant",
    )
    for field in shared_fields:
        assert field in native_event, f"native missing {field}"
        assert field in strands_event, f"strands missing {field}"


# -- process taxonomy unit tests (spec section 7) ----------------------------


def _process_event(action="place_trap", target="safe", trap_type="bomb", executed_action="place_trap",
                    executed_target="safe", executed_trap_type="bomb", valid=True, repair_attempted=False,
                    initial_error_type=None, repair_error_type=None, repair_action=None, repair_target=None, repair_trap_type=None):
    raw_response = {"action": action, "target_id": target, "trap_type": trap_type} if action else None
    repair_raw_response = {"action": repair_action, "target_id": repair_target, "trap_type": repair_trap_type} if repair_action else None
    return {
        "raw_response": raw_response, "repair_raw_response": repair_raw_response,
        "executed_action": {"action": executed_action, "target_id": executed_target, "trap_type": executed_trap_type},
        "valid": valid, "final_model_action_valid": valid, "repair_attempted": repair_attempted,
        "initial_error_type": initial_error_type, "repair_error_type": repair_error_type,
    }


def test_process_valid_first_pass():
    assert classify_place_trap_process(_process_event(valid=True)) == VALID_FIRST_PASS


def test_process_valid_after_repair():
    e = _process_event(trap_type="stun", valid=True, repair_attempted=True, initial_error_type="invalid_trap_type",
                        repair_action="place_trap", repair_target="safe", repair_trap_type="bomb")
    assert classify_place_trap_process(e) == VALID_AFTER_REPAIR


def test_process_abandoned_after_repair():
    e = _process_event(trap_type="stun", valid=True, repair_attempted=True, initial_error_type="invalid_trap_type",
                        repair_action="move", repair_target="door", executed_action="move", executed_target="door")
    assert classify_place_trap_process(e) == ABANDONED_AFTER_REPAIR


def test_process_fallback_after_failed_repair():
    e = _process_event(trap_type="stun", valid=False, repair_attempted=True, initial_error_type="invalid_trap_type",
                        repair_action="place_trap", repair_target="safe", repair_trap_type="stun_trap",
                        repair_error_type="invalid_trap_type", executed_action="wait", executed_target=None)
    assert classify_place_trap_process(e) == FALLBACK_AFTER_FAILED_REPAIR


def test_process_not_a_place_trap_attempt_returns_none():
    e = _process_event(action="move", target="desk", trap_type=None, executed_action="move", executed_target="desk")
    assert classify_place_trap_process(e) is None


def test_process_and_reason_taxonomies_agree_on_which_decisions_count():
    # Every decision the REASON taxonomy counts as an attempt whose
    # initial proposal was place_trap must also be counted by the
    # PROCESS taxonomy, and vice versa (spec sections 7-9 are two views
    # of the same underlying attempts, never two different denominators).
    events = [
        _process_event(valid=True),
        _process_event(action="move", target="desk", trap_type=None, executed_action="move", executed_target="desk"),
        _process_event(trap_type="stun", valid=True, repair_attempted=True, initial_error_type="invalid_trap_type",
                        repair_action="move", repair_target="door", executed_action="move", executed_target="door"),
    ]
    for e in events:
        reason_counted = classify_place_trap_event(e) is not None
        process_counted = classify_place_trap_process(e) is not None
        assert reason_counted == process_counted, e
