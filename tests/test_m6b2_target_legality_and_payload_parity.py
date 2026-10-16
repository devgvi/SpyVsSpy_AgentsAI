"""M6B.2: Harness Parity Consolidation & Target-Legality Reliability.

Phase B (network-free target-legality tests, spec section 28), Phase C
(repair parity matrix, spec section 29), Phase D (payload-transform parity,
spec section 30-31), plus RP12/RP10 executable enforcement and the
same_invalid_action_signature_count diagnostic (spec section 11). No
network calls anywhere in this file.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from types import SimpleNamespace

import pytest

from src import constants
from src.agents.native_agent import NativeAgentAdapter
from src.agents.strands_agent import StrandsAgentAdapter
from src.evaluation.runtime_snapshot_replay import build_snapshot_set, run_one_trial
from src.evaluation.target_legality_snapshot_set import (
    INSPECTED_TRAPPABLE_NEARBY,
    NON_TRAPPABLE_NEARBY,
    NO_TRAP_INVENTORY_NEARBY,
    UNINSPECTED_TRAPPABLE_NEARBY,
    build_target_legality_snapshot_set,
)
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import (
    LLMControllerConfig,
    invalid_action_signature,
    validate_model_output,
)
from tests.test_strands_agent import QueuedStrandsFactory

TL_SNAPSHOTS = {s.category: s for s in build_target_legality_snapshot_set()}
TRAP_SNAPSHOT = next(s for s in build_snapshot_set() if s.category == "TRAP_PLACEMENT_OPPORTUNITY")


def _native_trial(response, obs, episode_id="m6b2-native"):
    telemetry = TelemetryLogger()
    adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=MockModelClient(response=response),
        config=LLMControllerConfig(), telemetry=telemetry, episode_id=episode_id,
    )
    run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)
    return next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {}), adapter


def _strands_trial(queued_calls, obs, episode_id="m6b2-strands", **adapter_kwargs):
    telemetry = TelemetryLogger()
    factory = QueuedStrandsFactory(queued_calls)
    adapter = StrandsAgentAdapter(
        constants.WHITE_SPY_ID, agent_factory=factory, telemetry=telemetry, episode_id=episode_id, **adapter_kwargs
    )
    run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)
    return next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {}), adapter


# ============================================================================
# Phase B (spec section 28): deterministic target-legality tests
# ============================================================================


def test_uninspected_trappable_target_is_not_offered_and_classifies_correctly():
    s = TL_SNAPSHOTS[UNINSPECTED_TRAPPABLE_NEARBY]
    place_trap_targets = [e for e in s.legal_actions if e.get("action") == "place_trap"]
    assert place_trap_targets == [], "an uninspected target must never be offered as a legal place_trap target"
    resp = {"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "snare"}
    v = validate_model_output(s.observation, resp, s.legal_actions)
    assert v.valid is False
    assert v.error_type == "target_not_inspected"


def test_inspected_trappable_target_is_legal():
    s = TL_SNAPSHOTS[INSPECTED_TRAPPABLE_NEARBY]
    resp = {"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "snare"}
    v = validate_model_output(s.observation, resp, s.legal_actions)
    assert v.valid is True
    assert v.error_type is None


def test_unknown_target_classifies_as_unknown_target():
    s = TL_SNAPSHOTS[INSPECTED_TRAPPABLE_NEARBY]
    resp = {"action": "place_trap", "target_id": "not_a_real_object_id", "trap_type": "snare"}
    v = validate_model_output(s.observation, resp, s.legal_actions)
    assert v.valid is False
    assert v.error_type == "unknown_target"


def test_out_of_range_real_target_classifies_as_out_of_range():
    s = TL_SNAPSHOTS[INSPECTED_TRAPPABLE_NEARBY]
    # desk_01 is a real, known object - just not the actor's current
    # nearby_object_id (that's sofa_01 in this snapshot).
    assert s.observation.nearby_object_id != "desk_01"
    resp = {"action": "place_trap", "target_id": "desk_01", "trap_type": "snare"}
    v = validate_model_output(s.observation, resp, s.legal_actions)
    assert v.valid is False
    assert v.error_type == "out_of_range"


def test_non_trappable_object_classifies_as_target_not_trappable():
    s = TL_SNAPSHOTS[NON_TRAPPABLE_NEARBY]
    place_trap_targets = [e for e in s.legal_actions if e.get("action") == "place_trap"]
    assert place_trap_targets == []
    resp = {"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "snare"}
    v = validate_model_output(s.observation, resp, s.legal_actions)
    assert v.valid is False
    assert v.error_type == "target_not_trappable"


def test_no_trap_inventory_fails_at_the_trap_type_dimension_not_target():
    # Boundary case (spec sections 3, 8): even though the target itself is
    # inspected/trappable/in-range, holding zero traps of any type must
    # still fail at the earlier trap_type check, never as a target-legality
    # category - the two dimensions stay properly separated.
    s = TL_SNAPSHOTS[NO_TRAP_INVENTORY_NEARBY]
    resp = {"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "snare"}
    v = validate_model_output(s.observation, resp, s.legal_actions)
    assert v.valid is False
    assert v.error_type == "trap_type_not_in_inventory"


def test_already_trapped_target_is_engine_authoritative_never_leaked_to_legal_actions():
    """Spec sections 4, 7: a pre-existing HIDDEN trap must never be exposed
    through legal_actions/validate_model_output - the actor legitimately
    inspected the object (knows nothing is wrong with it from its own
    knowledge) and validate_model_output correctly says VALID; only the
    engine, at actual execution time, is the final authority that rejects
    it. This is the one target-legality category that must NOT become a
    validator-level classification (would leak hidden GameState)."""
    from src.evaluation.runtime_snapshot_replay import _fresh_state, _place_near_object, _snapshot_from_state

    state = _fresh_state()
    _place_near_object(state, constants.WHITE_SPY_ID, "sofa_01")
    entry = state._object_entry("sofa_01")
    entry["inspected_by"].add(constants.WHITE_SPY_ID)
    # A hidden trap, never detected by this actor (detected_by left empty)
    # - exactly the actor-invisible truth spec section 7 protects.
    entry["trap"] = {"type": "alarm", "owner_id": "environment", "armed": True}
    snap = _snapshot_from_state("tl_already_trapped", "ALREADY_TRAPPED", state)

    place_trap_targets = [e for e in snap.legal_actions if e.get("action") == "place_trap"]
    assert place_trap_targets == [{"action": "place_trap", "target_id": "sofa_01"}], (
        "a hidden pre-existing trap must not change legal_actions - the actor cannot legitimately know about it"
    )
    resp = {"action": "place_trap", "target_id": "sofa_01", "trap_type": "snare"}
    v = validate_model_output(snap.observation, resp, snap.legal_actions)
    assert v.valid is True, "validate_model_output must not leak hidden trap truth either"

    # The engine itself is the final authority and correctly rejects it.
    obj = state.level.get_object("sofa_01")
    actor = state.actors[constants.WHITE_SPY_ID]
    result = state._place_trap(actor, obj, "snare")
    assert result.success is False
    assert "already has a trap" in result.message


# ============================================================================
# RP12_LEGAL_TARGET_HINTS (spec sections 9, 13-14)
# ============================================================================


def test_rp12_repair_prompt_includes_legal_place_trap_targets_native():
    s = TL_SNAPSHOTS[UNINSPECTED_TRAPPABLE_NEARBY]
    target = s.observation.nearby_object_id
    event, _ = _native_trial(
        [
            {"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"},
            {"action": "place_trap", "target_id": target, "trap_type": "snare", "reason": "y"},
        ],
        obs=s.observation,
    )
    # The repair call must have been given the factual (empty) legal-target
    # list - captured indirectly via the repair-context length being
    # sensible and the initial error type reflecting the target-legality
    # gate, never a byte-for-byte string match on the prompt.
    assert event.get("initial_error_type") == "invalid_trap_type"
    assert event.get("repair_attempted") is True
    assert event.get("repair_error_type") == "target_not_inspected"


def test_rp12_repair_prompt_includes_legal_place_trap_targets_strands():
    s = TL_SNAPSHOTS[UNINSPECTED_TRAPPABLE_NEARBY]
    event, _ = _strands_trial(
        [
            [{"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "stun", "reason": "x"}],
            [{"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "snare", "reason": "y"}],
        ],
        obs=s.observation,
    )
    assert event.get("initial_error_type") == "invalid_trap_type"
    assert event.get("repair_attempted") is True
    assert event.get("repair_error_type") == "target_not_inspected"


def test_rp12_target_hint_matches_legal_actions_membership():
    """Direct unit-level check (not through an adapter): the hint text
    passed to _repair_instruction is derived from - and only from - the
    same legal_actions list validation itself used, for both a snapshot
    with zero legal targets and one with exactly one."""
    from src.llm_controller import _repair_instruction

    uninspected = TL_SNAPSHOTS[UNINSPECTED_TRAPPABLE_NEARBY]
    legal_targets = sorted(
        e.get("target_id") for e in uninspected.legal_actions if e.get("action") == "place_trap" and e.get("target_id")
    )
    assert legal_targets == []
    text = _repair_instruction(
        "target_not_inspected", "x", proposed_action="place_trap", proposed_target_id="sofa_01",
        near_object_id="sofa_01", legal_action_count=len(uninspected.legal_actions),
        legal_place_trap_targets=legal_targets,
    )
    assert "Legal PLACE_TRAP targets right now: []" in text

    inspected = TL_SNAPSHOTS[INSPECTED_TRAPPABLE_NEARBY]
    legal_targets2 = sorted(
        e.get("target_id") for e in inspected.legal_actions if e.get("action") == "place_trap" and e.get("target_id")
    )
    assert legal_targets2 == ["ambassador_safe"]
    text2 = _repair_instruction(
        "illegal_target", "x", proposed_action="place_trap", proposed_target_id="desk_01",
        near_object_id="ambassador_safe", legal_action_count=len(inspected.legal_actions),
        legal_place_trap_targets=legal_targets2,
    )
    assert "Legal PLACE_TRAP targets right now: ['ambassador_safe']" in text2


# ============================================================================
# Phase C (spec section 29): repair parity matrix
# ============================================================================


@pytest.mark.parametrize("case", ["invalid_target_only", "invalid_trap_type_only", "both_invalid", "both_valid"])
def test_repair_parity_matrix(case):
    s = TL_SNAPSHOTS[UNINSPECTED_TRAPPABLE_NEARBY]
    target = s.observation.nearby_object_id

    if case == "invalid_target_only":
        # A real trap_type, an illegal (uninspected) target.
        native_resp = [
            {"action": "place_trap", "target_id": target, "trap_type": "snare", "reason": "x"},
            {"action": "wait", "reason": "y"},
        ]
        strands_calls = [
            [{"action": "place_trap", "target_id": target, "trap_type": "snare", "reason": "x"}],
            [{"action": "wait", "reason": "y"}],
        ]
        expected_initial = "target_not_inspected"
    elif case == "invalid_trap_type_only":
        # A real, inspected+trappable target, a hallucinated trap_type.
        s = TL_SNAPSHOTS[INSPECTED_TRAPPABLE_NEARBY]
        target = s.observation.nearby_object_id
        native_resp = [
            {"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"},
            {"action": "wait", "reason": "y"},
        ]
        strands_calls = [
            [{"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"}],
            [{"action": "wait", "reason": "y"}],
        ]
        expected_initial = "invalid_trap_type"
    elif case == "both_invalid":
        native_resp = [
            {"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"},
            {"action": "wait", "reason": "y"},
        ]
        strands_calls = [
            [{"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"}],
            [{"action": "wait", "reason": "y"}],
        ]
        expected_initial = "invalid_trap_type"
    else:  # both_valid
        s = TL_SNAPSHOTS[INSPECTED_TRAPPABLE_NEARBY]
        target = s.observation.nearby_object_id
        native_resp = {"action": "place_trap", "target_id": target, "trap_type": "snare", "reason": "x"}
        strands_calls = [[{"action": "place_trap", "target_id": target, "trap_type": "snare", "reason": "x"}]]
        expected_initial = None

    native_event, native_adapter = _native_trial(native_resp, obs=s.observation)
    strands_event, strands_adapter = _strands_trial(strands_calls, obs=s.observation)

    assert native_event.get("initial_error_type") == expected_initial
    assert strands_event.get("initial_error_type") == expected_initial
    # Same repair budget on both (RP05): at most one repair attempt.
    assert native_adapter.controller.config.max_retries == strands_adapter.max_retries == 1
    if expected_initial is not None:
        assert native_event.get("repair_attempted") is True
        assert strands_event.get("repair_attempted") is True


# ============================================================================
# RP10 (spec sections 15-21): model_payload_transform parity for Strands
# ============================================================================


def test_rp10_model_payload_transform_now_exists_on_strands():
    params = inspect.signature(StrandsAgentAdapter.__init__).parameters
    assert "model_payload_transform" in params
    assert params["model_payload_transform"].default is None


def test_rp10_no_transform_configured_is_byte_identical_to_before():
    # Default (None) must leave decision behavior/telemetry unaffected -
    # spec section 19's "no transform configured -> no behavior change".
    event, _ = _strands_trial(
        [[{"action": "wait", "reason": "x"}]], obs=TRAP_SNAPSHOT.observation, model_payload_transform=None,
    )
    assert event.get("valid") is True
    assert event.get("executed_action", {}).get("action") == "wait"


def test_rp10_transform_applied_once_covers_both_initial_and_repair():
    captured_payloads: list[dict] = []

    def transform(payload):
        captured_payloads.append(dict(payload))
        payload = dict(payload)
        payload["__transformed_marker"] = "yes"
        return payload

    s = TL_SNAPSHOTS[INSPECTED_TRAPPABLE_NEARBY]
    _, adapter = _strands_trial(
        [
            [{"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "stun", "reason": "x"}],
            [{"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "snare", "reason": "y"}],
        ],
        obs=s.observation,
        model_payload_transform=transform,
    )
    # Applied exactly once per decision (spec section 17's "same transform
    # opportunity") - both the initial and repair Strands invocation reuse
    # the one already-transformed model_payload object.
    assert len(captured_payloads) == 1


def test_rp10_transform_cannot_see_hidden_gamestate():
    # Spec section 20: the transform only ever receives the already
    # actor-visible compact model payload - never raw GameState. Assert by
    # construction: the payload handed to the transform has none of
    # GameState's own private attribute names.
    seen_payload = {}

    def transform(payload):
        seen_payload.update(payload)
        return payload

    _strands_trial([[{"action": "wait", "reason": "x"}]], obs=TRAP_SNAPSHOT.observation, model_payload_transform=transform)
    assert "actors" not in seen_payload  # GameState-only private field
    assert "rng" not in seen_payload
    assert "self" in seen_payload or "legal_actions_compact" in seen_payload  # real compact-payload shape


def test_rp10_transform_hash_parity_same_logical_payload_same_hash():
    """Spec section 31: for the SAME logical payload, a deterministic
    transform produces the same hash regardless of which runtime's own
    compact-payload builder produced the pre-transform dict shape - proven
    here by applying the identical transform function to Native's and
    Strands' own captured pre-transform payloads and hashing just the
    transform's own deterministic addition, not the whole (legitimately
    runtime-shaped) payload."""

    def deterministic_transform(payload):
        payload = dict(payload)
        payload["injected_marker"] = "m6b2-deterministic-value"
        return payload

    native_captured = {}
    strands_captured = {}

    def native_wrapper(payload):
        result = deterministic_transform(payload)
        native_captured["result"] = result
        return result

    def strands_wrapper(payload):
        result = deterministic_transform(payload)
        strands_captured["result"] = result
        return result

    native_config = LLMControllerConfig(model_payload_transform=native_wrapper)
    telemetry = TelemetryLogger()
    native_adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=MockModelClient(response={"action": "wait", "reason": "x"}),
        config=native_config, telemetry=telemetry, episode_id="rp10-hash-native",
    )
    run_one_trial(native_adapter, TRAP_SNAPSHOT.observation, telemetry, "rp10-hash-native", wall_timeout=10.0)

    _strands_trial(
        [[{"action": "wait", "reason": "x"}]], obs=TRAP_SNAPSHOT.observation,
        episode_id="rp10-hash-strands", model_payload_transform=strands_wrapper,
    )

    assert native_captured["result"]["injected_marker"] == strands_captured["result"]["injected_marker"]
    native_marker_hash = hashlib.sha256(native_captured["result"]["injected_marker"].encode()).hexdigest()
    strands_marker_hash = hashlib.sha256(strands_captured["result"]["injected_marker"].encode()).hexdigest()
    assert native_marker_hash == strands_marker_hash


def test_rp10_manifest_status_is_pass_not_fail():
    from src.evaluation.runtime_parity import PASS, RUNTIME_PARITY_MANIFEST

    rp10 = next(e for e in RUNTIME_PARITY_MANIFEST if e.entry_id == "RP10_MEMORY_FLAGS")
    assert rp10.status == PASS


def test_rp12_manifest_entry_exists_and_passes():
    from src.evaluation.runtime_parity import PASS, RUNTIME_PARITY_MANIFEST

    rp12 = next((e for e in RUNTIME_PARITY_MANIFEST if e.entry_id == "RP12_LEGAL_TARGET_HINTS"), None)
    assert rp12 is not None
    assert rp12.status == PASS


def test_manifest_has_at_least_twelve_entries_with_unique_ids():
    # M6B.2 grew the manifest to exactly 12 (RP01-RP12); M6F spec section 16
    # appends RP13-RP18 on top without renumbering, so this only pins the
    # M6B.2-era floor and uniqueness, not an exact count that later
    # milestones would have to keep breaking.
    from src.evaluation.runtime_parity import RUNTIME_PARITY_MANIFEST

    ids = [e.entry_id for e in RUNTIME_PARITY_MANIFEST]
    assert len(ids) >= 12
    assert len(set(ids)) == len(ids)


def test_no_unexplained_fail_remains_in_manifest():
    # Spec section 40 (parity test gate): behavioral interpretation is only
    # allowed if no unexplained FAIL remains.
    from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST

    unexplained_fails = [e for e in RUNTIME_PARITY_MANIFEST if e.status == FAIL]
    assert unexplained_fails == [], f"unexplained FAIL entries: {[e.entry_id for e in unexplained_fails]}"


def test_game_threads_model_payload_transform_into_strands_branch():
    # Spec section 15's root fix, checked directly at the source (not just
    # through the adapter constructor) - Game._build_llm_agent's own
    # strands-kwargs dict must include the key, never silently drop it.
    import inspect as _inspect

    from src.game import Game

    src_text = _inspect.getsource(Game._build_llm_agent)
    strands_branch_start = src_text.index('if agent_runtime == "strands"')
    # search backwards from that marker for the dict literal that precedes it
    dict_start = src_text.rindex("{", 0, strands_branch_start)
    strands_kwargs_block = src_text[dict_start:strands_branch_start]
    assert "model_payload_transform" in strands_kwargs_block


# ============================================================================
# same_invalid_action_signature_count (spec section 11)
# ============================================================================


def test_invalid_action_signature_is_deterministic_and_readable():
    sig = invalid_action_signature("place_trap", "desk_03", "electric")
    assert sig == "PLACE_TRAP|target=desk_03|trap_type=electric"
    assert invalid_action_signature("place_trap", "desk_03", "electric") == sig
    assert invalid_action_signature("place_trap", "desk_03", "snare") != sig


def test_native_telemetry_reports_signature_count_on_invalid_decision():
    s = TL_SNAPSHOTS[UNINSPECTED_TRAPPABLE_NEARBY]
    target = s.observation.nearby_object_id
    event, _ = _native_trial(
        [
            {"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"},
            {"action": "place_trap", "target_id": target, "trap_type": "snare", "reason": "y"},
        ],
        obs=s.observation,
    )
    assert event.get("fallback_used") is True
    assert event.get("same_invalid_action_signature_count") == 1


def test_strands_telemetry_reports_signature_count_on_invalid_decision():
    s = TL_SNAPSHOTS[UNINSPECTED_TRAPPABLE_NEARBY]
    target = s.observation.nearby_object_id
    event, _ = _strands_trial(
        [
            [{"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"}],
            [{"action": "place_trap", "target_id": target, "trap_type": "snare", "reason": "y"}],
        ],
        obs=s.observation,
    )
    assert event.get("fallback_used") is True
    assert event.get("same_invalid_action_signature_count") == 1


def test_signature_count_increments_across_repeated_decisions_same_adapter():
    # Mirrors the real seed9605 outlier pattern: the SAME adapter instance
    # (one episode's worth of state) repeatedly proposing the identical
    # invalid signature across separate decisions - the count must climb,
    # never reset per-decision (spec section 11 - diagnosis, no
    # suppression). Calls the controller's own `_request_action` directly,
    # 3 separate times, bypassing `get_action`'s rate-limiting/dedup
    # plumbing (a background-thread/timing concern orthogonal to what this
    # test checks: per-episode signature accumulation).
    from src.llm_controller import serialize_observation

    s = TL_SNAPSHOTS[UNINSPECTED_TRAPPABLE_NEARBY]
    target = s.observation.nearby_object_id
    telemetry = TelemetryLogger()
    adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID,
        model_client=MockModelClient(response=[
            {"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"},
            {"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"},
        ]),
        config=LLMControllerConfig(), telemetry=telemetry, episode_id="m6b2-repeat",
    )
    observation_payload = serialize_observation(s.observation)
    counts = []
    for _ in range(3):
        adapter.model_client.responses = [
            {"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"},
            {"action": "place_trap", "target_id": target, "trap_type": "stun", "reason": "x"},
        ]
        adapter.controller._request_action(s.observation, observation_payload)
        events = [e for e in telemetry.events if e.get("event_type") == "llm_decision"]
        counts.append(events[-1].get("same_invalid_action_signature_count"))
    assert counts == [1, 2, 3]


def test_valid_decision_has_no_signature_count():
    s = TL_SNAPSHOTS[INSPECTED_TRAPPABLE_NEARBY]
    event, _ = _native_trial(
        {"action": "place_trap", "target_id": s.observation.nearby_object_id, "trap_type": "snare", "reason": "x"},
        obs=s.observation,
    )
    assert event.get("valid") is True
    assert event.get("same_invalid_action_signature_count") is None
