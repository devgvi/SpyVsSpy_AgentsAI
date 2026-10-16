"""M6D Phase A (spec section 22): network-free recovery-semantics tests.

Verifies: P0 makes no second model call, P1 uses current production
repair, P2 contains minimal factual repair only, P3 includes legal
alternatives only when permitted, all policies use the same validator,
all policies use the same fallback, failure counters remain distinct.
Also covers the invalid-action corpus (9 families, real game states) and
fallback-policy mechanics (F0/F1/F2, candidate-order dependence). No
network calls.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.agents.strands_agent import StrandsAgentAdapter
from src.evaluation.recovery_corpus import CORPUS_FAILURE_FAMILIES, build_invalid_action_corpus
from src.evaluation.recovery_policy import (
    DIFFERENT_ACTION,
    F0_CURRENT_SAFE_FALLBACK,
    F1_WAIT_FALLBACK,
    F2_DETERMINISTIC_FIRST_LEGAL,
    FALLBACK,
    P0_NO_REPAIR_SAFE_FALLBACK,
    P1_CURRENT_ONE_REPAIR,
    P2_MINIMAL_REPAIR,
    P3_FULL_LEGAL_REPAIR,
    REPAIR_POLICIES,
    SAME_ACTION_SAME_TARGET,
    apply_fallback_policy,
    build_repair_prompt,
    classify_intent,
    full_legal_repair_instruction,
    minimal_repair_instruction,
    run_recovery_policy,
    run_recovery_policy_strands,
)
from src.llm.clients.mock_client import MockModelClient
from src.llm_controller import _repair_instruction, validate_model_output
from tests.test_strands_agent import QueuedStrandsFactory

CORPUS = build_invalid_action_corpus()
CORPUS_BY_ID = {c.case_id: c for c in CORPUS}


# -- corpus integrity (spec sections 18-19, 63) ------------------------------


def test_corpus_covers_all_nine_failure_families():
    families = {c.failure_family for c in CORPUS}
    assert families == set(CORPUS_FAILURE_FAMILIES)
    assert len(CORPUS) == 9


def test_every_corpus_case_is_genuinely_invalid_against_real_game_state():
    for case in CORPUS:
        v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
        assert v.valid is False, case.case_id
        assert v.error_type == case.validation_error_type, case.case_id


# -- P0: no repair call (spec sections 9, 18) --------------------------------


def test_p0_makes_no_repair_call():
    case = CORPUS_BY_ID["case04_invalid_trap_type"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    outcome = run_recovery_policy(
        P0_NO_REPAIR_SAFE_FALLBACK, case.observation, case.legal_actions, {}, case.invalid_raw_response, v,
        model_client=None, system_prompt="sys", output_schema={},
    )
    assert outcome.repair_attempted is False
    assert outcome.repair_prompt is None
    assert outcome.fallback_used is True
    assert outcome.final_action == "wait"


# -- P1: current production repair, byte-identical (spec section 8) ---------


def test_p1_uses_the_real_unmodified_repair_instruction():
    case = CORPUS_BY_ID["case04_invalid_trap_type"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    client = MockModelClient(response={"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "snare", "reason": "y"})
    outcome = run_recovery_policy(
        P1_CURRENT_ONE_REPAIR, case.observation, case.legal_actions, {"strategy_memory": {}}, case.invalid_raw_response, v,
        client, "sys", {},
    )
    from src.llm_controller import place_trap_repair_hints

    legal_trap_types, legal_place_trap_targets = place_trap_repair_hints(case.observation, case.legal_actions)
    expected = _repair_instruction(
        v.error_type, v.error_message, proposed_action="place_trap", proposed_target_id="ambassador_safe",
        near_object_id=case.observation.nearby_object_id, legal_action_count=len(case.legal_actions),
        legal_trap_types=legal_trap_types, legal_place_trap_targets=legal_place_trap_targets,
    )
    assert outcome.repair_prompt == expected


# -- P2: minimal only (spec sections 10, 25 not-applicable to P2) -----------


def test_p2_minimal_repair_excludes_near_object_and_legal_hints():
    text = minimal_repair_instruction("invalid_trap_type", "'stun' is not a trap type this game supports", "place_trap", "ambassador_safe")
    assert "'stun' is not a trap type this game supports" in text
    assert "near" not in text.lower()
    assert "legal_actions" not in text
    assert "Allowed trap types" not in text
    assert "Legal PLACE_TRAP targets" not in text


def test_p2_repair_call_is_genuinely_minimal_end_to_end():
    case = CORPUS_BY_ID["case04_invalid_trap_type"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    client = MockModelClient(response={"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "snare", "reason": "y"})
    outcome = run_recovery_policy(P2_MINIMAL_REPAIR, case.observation, case.legal_actions, {}, case.invalid_raw_response, v, client, "sys", {})
    assert "Allowed trap types" not in outcome.repair_prompt
    assert "near" not in outcome.repair_prompt.lower()


# -- P3: full legal alternatives, actor-visible only (spec sections 11, 294) --


def test_p3_full_legal_repair_includes_the_complete_legal_action_set():
    case = CORPUS_BY_ID["case04_invalid_trap_type"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    from src.llm_controller import place_trap_repair_hints

    legal_trap_types, legal_place_trap_targets = place_trap_repair_hints(case.observation, case.legal_actions)
    text = full_legal_repair_instruction(
        v.error_type, v.error_message, "place_trap", "ambassador_safe", case.observation.nearby_object_id,
        case.legal_actions, legal_trap_types, legal_place_trap_targets,
    )
    assert "Every currently legal action right now" in text
    # every legal target actually offered must appear somewhere in the text
    for entry in case.legal_actions:
        tgt = entry.get("target_id")
        if tgt:
            assert tgt in text, tgt
    assert "Allowed trap types" in text


def test_p3_never_ranks_or_recommends():
    case = CORPUS_BY_ID["case04_invalid_trap_type"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    from src.llm_controller import place_trap_repair_hints

    legal_trap_types, legal_place_trap_targets = place_trap_repair_hints(case.observation, case.legal_actions)
    text = full_legal_repair_instruction(
        v.error_type, v.error_message, "place_trap", "ambassador_safe", case.observation.nearby_object_id,
        case.legal_actions, legal_trap_types, legal_place_trap_targets,
    )
    for banned in ("best", "recommend", "should choose", "optimal", "preferable"):
        assert banned not in text.lower()


@pytest.mark.parametrize("case_id", [c.case_id for c in CORPUS])
def test_all_three_repair_policies_use_the_same_validator_for_every_corpus_case(case_id):
    case = CORPUS_BY_ID[case_id]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    assert v.error_type == case.validation_error_type

    for policy, response in (
        (P1_CURRENT_ONE_REPAIR, {"action": "wait", "reason": "y"}),
        (P2_MINIMAL_REPAIR, {"action": "wait", "reason": "y"}),
        (P3_FULL_LEGAL_REPAIR, {"action": "wait", "reason": "y"}),
    ):
        client = MockModelClient(response=response)
        outcome = run_recovery_policy(policy, case.observation, case.legal_actions, {}, case.invalid_raw_response, v, client, "sys", {})
        # A valid `wait` response must be accepted identically under
        # every policy - same validator, same acceptance criteria.
        assert outcome.repair_valid is True
        assert outcome.final_action == "wait"
        assert outcome.fallback_used is False


# -- intent-preservation classification (spec sections 26-27) ---------------


def test_repair_that_switches_action_is_not_mislabeled_as_repair_success():
    outcome_category = classify_intent("place_trap", "ambassador_safe", "move", "door_1", fallback_used=False)
    assert outcome_category == DIFFERENT_ACTION
    # Validity and continuity are reported as two SEPARATE facts elsewhere
    # (repair_valid=True alongside intent_category=DIFFERENT_ACTION) -
    # this test only confirms the categorization function itself never
    # conflates them into "PLACE_TRAP repaired successfully".


def test_same_action_same_target_classified_correctly():
    assert classify_intent("place_trap", "safe", "place_trap", "safe", False) == SAME_ACTION_SAME_TARGET


def test_fallback_always_classified_as_fallback_regardless_of_final_action():
    assert classify_intent("place_trap", "safe", "wait", None, True) == FALLBACK


# -- fallback policies (spec sections 14-17, 43) -----------------------------


def test_f0_and_f1_are_both_wait_documented_as_behaviorally_identical():
    case = CORPUS_BY_ID["case09_action_currently_illegal"]
    f0 = apply_fallback_policy(F0_CURRENT_SAFE_FALLBACK, case.legal_actions)
    f1 = apply_fallback_policy(F1_WAIT_FALLBACK, case.legal_actions)
    assert f0 == f1 == {"action": "wait"}


def test_f2_picks_first_non_wait_legal_action_candidate_order_dependent():
    case = CORPUS_BY_ID["case09_action_currently_illegal"]
    f2 = apply_fallback_policy(F2_DETERMINISTIC_FIRST_LEGAL, case.legal_actions)
    assert f2["action"] != "wait"
    first_non_wait = next(e for e in case.legal_actions if e.get("action") != "wait")
    assert f2["action"] == first_non_wait["action"]
    assert f2.get("target_id") == first_non_wait.get("target_id")


def test_f2_order_reversal_control_confirms_mechanical_dependence():
    # Spec section 44: one small order-reversal control.
    case = CORPUS_BY_ID["case09_action_currently_illegal"]
    reversed_actions = list(reversed(case.legal_actions))
    f2_normal = apply_fallback_policy(F2_DETERMINISTIC_FIRST_LEGAL, case.legal_actions)
    f2_reversed = apply_fallback_policy(F2_DETERMINISTIC_FIRST_LEGAL, reversed_actions)
    assert f2_normal != f2_reversed, "F2 must mechanically depend on candidate order, proven by reversing it"


def test_f2_falls_back_to_wait_if_no_non_wait_legal_action_exists():
    assert apply_fallback_policy(F2_DETERMINISTIC_FIRST_LEGAL, [{"action": "wait"}]) == {"action": "wait"}


# -- failure-counter separation (spec section 45, structural check) ---------


def test_recovery_outcome_never_conflates_validation_with_provider_failure():
    # A repair call that succeeds validly must never be flagged as any
    # kind of failure at all.
    case = CORPUS_BY_ID["case06_missing_required_field"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    client = MockModelClient(response={"action": "wait", "reason": "y"})
    outcome = run_recovery_policy(P1_CURRENT_ONE_REPAIR, case.observation, case.legal_actions, {}, case.invalid_raw_response, v, client, "sys", {})
    assert outcome.repair_valid is True
    assert outcome.fallback_used is False


def test_all_four_policies_are_registered():
    assert set(REPAIR_POLICIES) == {P0_NO_REPAIR_SAFE_FALLBACK, P1_CURRENT_ONE_REPAIR, P2_MINIMAL_REPAIR, P3_FULL_LEGAL_REPAIR}


# -- Strands runtime parity (spec sections 47-49, HP02) ----------------------


@pytest.mark.parametrize("policy", [P1_CURRENT_ONE_REPAIR, P2_MINIMAL_REPAIR, P3_FULL_LEGAL_REPAIR])
def test_strands_repair_uses_the_identical_shared_prompt_builder(policy):
    case = CORPUS_BY_ID["case04_invalid_trap_type"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    expected_prompt = build_repair_prompt(policy, case.observation, case.legal_actions, case.invalid_raw_response, v)

    factory = QueuedStrandsFactory([[{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "snare", "reason": "y"}]])
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory)
    outcome = run_recovery_policy_strands(policy, case.observation, case.legal_actions, {"strategy_memory": {}}, case.invalid_raw_response, v, adapter)
    assert outcome.repair_prompt == expected_prompt


def test_strands_and_native_produce_the_same_repair_prompt_text_for_every_policy():
    # HP02: never two independently-drifting repair-text implementations -
    # the exact text handed to Native's generate_action and Strands'
    # _invoke_strands must be byte-identical for the same policy/case.
    case = CORPUS_BY_ID["case06_missing_required_field"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    for policy in (P1_CURRENT_ONE_REPAIR, P2_MINIMAL_REPAIR, P3_FULL_LEGAL_REPAIR):
        native_client = MockModelClient(response={"action": "wait", "reason": "y"})
        native_outcome = run_recovery_policy(policy, case.observation, case.legal_actions, {}, case.invalid_raw_response, v, native_client, "sys", {})

        factory = QueuedStrandsFactory([[{"action": "wait", "reason": "y"}]])
        adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory)
        strands_outcome = run_recovery_policy_strands(policy, case.observation, case.legal_actions, {"strategy_memory": {}}, case.invalid_raw_response, v, adapter)

        assert native_outcome.repair_prompt == strands_outcome.repair_prompt, policy


def test_strands_p0_matches_native_p0_exactly():
    case = CORPUS_BY_ID["case09_action_currently_illegal"]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    native_p0 = run_recovery_policy(P0_NO_REPAIR_SAFE_FALLBACK, case.observation, case.legal_actions, {}, case.invalid_raw_response, v, None, "sys", {})
    strands_p0 = run_recovery_policy_strands(P0_NO_REPAIR_SAFE_FALLBACK, case.observation, case.legal_actions, {}, case.invalid_raw_response, v, None)
    assert native_p0.final_action == strands_p0.final_action == "wait"
    assert native_p0.intent_category == strands_p0.intent_category == FALLBACK
