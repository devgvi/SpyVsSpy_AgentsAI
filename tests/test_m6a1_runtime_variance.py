"""M6A.1 Phase A (spec section 26): network-free tests for snapshot
integrity, decision-signature normalization, distribution/modal/overlap
computation, classification logic, and hash stability. No network calls -
Strands exercised via the same fake `agent_factory` boundary
`tests/test_strands_agent.py` already established, exactly like M6A's own
`tests/test_m6a_runtime_comparison.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from collections import Counter

from src import constants
from src.agents.native_agent import NativeAgentAdapter
from src.evaluation.runtime_snapshot_replay import build_snapshot_set, run_one_trial
from src.evaluation.runtime_variance_analysis import (
    BETWEEN_RUNTIME_SHIFT,
    INTRINSIC_VARIANCE_DOMINATES,
    MIXED,
    NO_DETECTABLE_RUNTIME_EFFECT,
    RUNTIME_EFFECT_CONTEXT_DEPENDENT,
    RUNTIME_EFFECT_SUPPORTED,
    RUNTIME_SPECIFIC_MODE,
    TECHNICAL_COST_DIFFERENCE_ONLY,
    WITHIN_RUNTIME_VARIANCE_DOMINATES,
    action_only_distribution,
    assert_snapshot_integrity,
    build_distribution,
    canonical_decision_signature,
    classify_overall,
    classify_snapshot_runtime_effect,
    content_hash,
    distribution_overlap,
    legal_actions_hash,
    memory_hash,
    modal_decision,
    observation_hash,
    payload_shape_hash,
    trial_signature,
    unique_decision_count,
)
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig, serialize_observation

SNAPSHOTS = {s.category: s for s in build_snapshot_set()}


def _native_trial(response, obs, episode_id="m6a1-native"):
    telemetry = TelemetryLogger()
    adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=MockModelClient(response=response),
        config=LLMControllerConfig(), telemetry=telemetry, episode_id=episode_id,
    )
    return run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)


# -- snapshot integrity (spec sections 4-5, 26) ------------------------------


def test_every_m6a_snapshot_passes_integrity_check():
    # Spec section 4: reuse the exact valid M6A snapshot set; if any
    # failed integrity, it would need explicit exclusion+documentation
    # rather than silent replacement. All 8 categories are expected clean
    # since M6A's own live run already used the post-`state.update(0.0)`-fix
    # builder.
    for category, snapshot in SNAPSHOTS.items():
        violations = assert_snapshot_integrity(snapshot)
        assert violations == [], f"{category}: {violations}"


def test_integrity_check_catches_the_real_m6a_bug():
    # Regression test for the actual bug M6A found: a nearby_object_id
    # that doesn't match any gated legal-action target (simulating the
    # pre-`state.update(0.0)`-fix state, where INSPECT/INTERACT were
    # offered without a correspondingly-updated nearby_object_id).
    snapshot = SNAPSHOTS["LOCKED_OBJECT"]
    import dataclasses

    broken_legal_actions = list(snapshot.legal_actions) + [{"action": "inspect", "target_id": "some_other_object"}]
    broken = dataclasses.replace(snapshot, legal_actions=broken_legal_actions)
    violations = assert_snapshot_integrity(broken)
    assert any("nearby_object_id" in v for v in violations)


# -- canonical decision signature (spec section 8) ---------------------------


def test_canonical_signature_move():
    assert canonical_decision_signature("move", "door_security", None) == "MOVE|door_security"


def test_canonical_signature_place_trap_with_parameter():
    assert canonical_decision_signature("place_trap", "safe_office", "electric") == "PLACE_TRAP|safe_office|electric"


def test_canonical_signature_wait_has_no_target():
    assert canonical_decision_signature("wait", None, None) == "WAIT"


def test_canonical_signature_ignores_reason_text():
    # trial_signature only reads action/target/trap_type - never `reason`.
    trial = {"action": "inspect", "target": "desk_03", "trap_type": None, "reason": "checking for documents"}
    assert trial_signature(trial) == "INSPECT|desk_03"


def test_canonical_signature_no_action_is_its_own_value():
    assert canonical_decision_signature(None, None, None) == "NO_ACTION"


# -- distribution / modal / overlap (spec sections 9-13) ---------------------


def test_build_distribution_counts_signatures():
    trials = [
        {"action": "move", "target": "a", "trap_type": None},
        {"action": "move", "target": "a", "trap_type": None},
        {"action": "move", "target": "b", "trap_type": None},
    ]
    dist = build_distribution(trials)
    assert dist == Counter({"MOVE|a": 2, "MOVE|b": 1})


def test_modal_decision_rate():
    dist = Counter({"MOVE|a": 7, "MOVE|b": 3})
    signature, rate = modal_decision(dist)
    assert signature == "MOVE|a"
    assert rate == 0.7


def test_modal_decision_empty_distribution():
    assert modal_decision(Counter()) == (None, 0.0)


def test_unique_decision_count():
    assert unique_decision_count(Counter({"WAIT": 5, "MOVE|a": 5})) == 2


def test_distribution_overlap_identical_is_one():
    dist = Counter({"MOVE|a": 6, "MOVE|b": 4})
    assert distribution_overlap(dist, dist) == 1.0


def test_distribution_overlap_disjoint_is_zero():
    a = Counter({"MOVE|a": 10})
    b = Counter({"MOVE|b": 10})
    assert distribution_overlap(a, b) == 0.0


def test_distribution_overlap_partial():
    # Spec section 49's own "strong runtime effect" example.
    a = Counter({"MOVE_TO|A": 9, "MOVE_TO|B": 1})
    b = Counter({"MOVE_TO|A": 1, "MOVE_TO|B": 9})
    assert distribution_overlap(a, b) == 0.2  # min(0.9,0.1) + min(0.1,0.9)


def test_distribution_overlap_empty_is_zero_not_error():
    assert distribution_overlap(Counter(), Counter({"WAIT": 1})) == 0.0


def test_action_only_distribution_collapses_targets():
    trials = [{"action": "move", "target": "a"}, {"action": "move", "target": "b"}, {"action": "wait", "target": None}]
    assert action_only_distribution(trials) == Counter({"move": 2, "wait": 1})


# -- per-snapshot classification (spec section 14) ---------------------------


def test_classify_no_detectable_runtime_effect():
    same = Counter({"MOVE|a": 10})
    assert classify_snapshot_runtime_effect(same, same) == NO_DETECTABLE_RUNTIME_EFFECT


def test_classify_runtime_specific_mode():
    # Spec section 49's strong-effect example: both internally stable,
    # different modal decision.
    native = Counter({"MOVE_TO|A": 9, "MOVE_TO|B": 1})
    strands = Counter({"MOVE_TO|B": 9, "MOVE_TO|A": 1})
    assert classify_snapshot_runtime_effect(native, strands) == RUNTIME_SPECIFIC_MODE


def test_classify_within_runtime_variance_dominates():
    # Spec section 50's weak-effect example: neither is highly stable,
    # substantial overlap remains.
    native = Counter({"A": 5, "B": 5})
    strands = Counter({"A": 4, "B": 6})
    assert classify_snapshot_runtime_effect(native, strands) == WITHIN_RUNTIME_VARIANCE_DOMINATES


def test_classify_between_runtime_shift_low_overlap_unstable():
    native = Counter({"A": 6, "B": 4})
    strands = Counter({"C": 6, "D": 4})
    assert classify_snapshot_runtime_effect(native, strands) == BETWEEN_RUNTIME_SHIFT


def test_classify_empty_is_mixed():
    assert classify_snapshot_runtime_effect(Counter(), Counter({"WAIT": 1})) == MIXED


# -- overall interpretation (spec section 48) --------------------------------


def test_overall_intrinsic_variance_dominates_when_no_snapshot_evidence_and_no_cost_diff():
    assert classify_overall([NO_DETECTABLE_RUNTIME_EFFECT, WITHIN_RUNTIME_VARIANCE_DOMINATES], cost_differs=False) == INTRINSIC_VARIANCE_DOMINATES


def test_overall_technical_cost_difference_only():
    assert classify_overall([NO_DETECTABLE_RUNTIME_EFFECT, WITHIN_RUNTIME_VARIANCE_DOMINATES], cost_differs=True) == TECHNICAL_COST_DIFFERENCE_ONLY


def test_overall_runtime_effect_supported_when_majority_show_evidence():
    labels = [RUNTIME_SPECIFIC_MODE, BETWEEN_RUNTIME_SHIFT, RUNTIME_SPECIFIC_MODE, NO_DETECTABLE_RUNTIME_EFFECT]
    assert classify_overall(labels, cost_differs=True) == RUNTIME_EFFECT_SUPPORTED


def test_overall_context_dependent_when_minority_show_evidence():
    labels = [RUNTIME_SPECIFIC_MODE, NO_DETECTABLE_RUNTIME_EFFECT, NO_DETECTABLE_RUNTIME_EFFECT, WITHIN_RUNTIME_VARIANCE_DOMINATES]
    assert classify_overall(labels, cost_differs=True) == RUNTIME_EFFECT_CONTEXT_DEPENDENT


def test_overall_empty_is_mixed():
    assert classify_overall([], cost_differs=False) == MIXED


# -- hash stability (spec section 25) ----------------------------------------


def test_observation_hash_stable_for_same_snapshot():
    obs = SNAPSHOTS["SIMPLE_NAVIGATION"].observation
    payload = serialize_observation(obs)
    assert observation_hash(payload) == observation_hash(serialize_observation(obs))


def test_observation_hash_differs_across_snapshots():
    a = serialize_observation(SNAPSHOTS["SIMPLE_NAVIGATION"].observation)
    b = serialize_observation(SNAPSHOTS["LOCKED_OBJECT"].observation)
    assert observation_hash(a) != observation_hash(b)


def test_legal_actions_hash_stable_and_order_sensitive_input_is_order_insensitive_output():
    legal = SNAPSHOTS["MULTIPLE_LEGAL_TARGETS"].legal_actions
    assert legal_actions_hash(legal) == legal_actions_hash(list(legal))


def test_memory_hash_stable_for_identical_dict():
    d = {"visited_rooms": ["reception"], "known_requirements": []}
    assert memory_hash(d) == memory_hash(dict(d))


def test_content_hash_distinguishes_different_prompt_text():
    # Regression test for a real bug found live: two different observation
    # payloads, both already-serialized to JSON strings, hashed through
    # `payload_shape_hash` collapse to the same "str, str" shape
    # regardless of content - `content_hash` must actually distinguish them.
    a = content_hash("system prompt", '{"observation": "A"}')
    b = content_hash("system prompt", '{"observation": "B"}')
    assert a != b


def test_content_hash_stable_for_identical_text():
    assert content_hash("sys", "user") == content_hash("sys", "user")


def test_payload_shape_hash_collapses_different_strings_to_the_same_shape():
    # Documents the exact bug above rather than hiding it: this is why
    # `payload_shape_hash` must never be used for already-serialized
    # prompt strings, only for real nested dicts like `output_schema`.
    from src.evaluation.runtime_variance_analysis import payload_shape_hash as shape_hash

    a = shape_hash({"system_prompt": "x", "user_prompt": '{"observation": "A"}'})
    b = shape_hash({"system_prompt": "x", "user_prompt": '{"observation": "totally different B"}'})
    assert a == b  # same shape (two strings) despite very different content


def test_payload_shape_hash_ignores_content_but_not_structure():
    a = {"action": "move", "target_id": "desk_01", "reason": "x"}
    b = {"action": "wait", "target_id": "door_01", "reason": "y"}
    c = {"action": "move", "target_id": "desk_01"}  # missing a key -> different shape
    assert payload_shape_hash(a) == payload_shape_hash(b)
    assert payload_shape_hash(a) != payload_shape_hash(c)


# -- telemetry completeness (spec section 26) --------------------------------


def test_trial_carries_fields_needed_for_variance_analysis():
    obs = SNAPSHOTS["MULTIPLE_LEGAL_TARGETS"].observation
    trial = _native_trial({"action": "move", "target_id": "desk_01", "reason": "x"}, obs)
    for field in ("action", "target", "trap_type", "valid", "repair_attempted", "fallback_used", "runtime_error", "latency_ms", "input_tokens", "output_tokens"):
        assert field in trial
    assert trial_signature(trial) == "MOVE|desk_01"
