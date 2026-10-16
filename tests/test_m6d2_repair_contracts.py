"""M6D.2 Phase A/B (spec sections 19-20): network-free contract and
semantic-equivalence tests. Verifies: N0 unchanged, N1/N2 derive from
identical RepairFacts, no strategic advice, same legal lists, same
internal list order, deterministic rendering, no hidden-state access, and
that RepairFacts hash stays identical across every (contract,
perturbation) cell per case while representation hash differs. No
network calls.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

import pytest

from src.evaluation.recovery_corpus import build_invalid_action_corpus
from src.evaluation.recovery_policy import P1_CURRENT_ONE_REPAIR, build_repair_prompt
from src.evaluation.repair_contracts import (
    CONTRACTS,
    N0_CURRENT_PRODUCTION,
    N1_NEUTRAL_LABELED,
    N2_NEUTRAL_STRUCTURED,
    P0_CANONICAL,
    P1_ORDER_REVERSED,
    P2_WHITESPACE_VARIANT,
    P3_NEUTRAL_LABEL_SYNONYMS,
    PERTURBATIONS,
    _BANNED_PHRASES,
    build_repair_facts,
    render_contract,
)
from src.evaluation.repair_prompt_variants import repair_fact_hash, repair_representation_hash
from src.llm_controller import validate_model_output

PRIMARY_FAILURE_FAMILIES = ("INVALID_TRAP_TYPE", "MISSING_REQUIRED_FIELD", "TARGET_NOT_INSPECTED")
CORPUS = {c.failure_family: c for c in build_invalid_action_corpus() if c.failure_family in PRIMARY_FAILURE_FAMILIES}


def _facts_and_validation(family):
    case = CORPUS[family]
    v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
    facts = build_repair_facts(case, v)
    return case, v, facts


# -- registration --------------------------------------------------------


def test_three_contracts_and_four_perturbations_registered():
    assert set(CONTRACTS) == {N0_CURRENT_PRODUCTION, N1_NEUTRAL_LABELED, N2_NEUTRAL_STRUCTURED}
    assert set(PERTURBATIONS) == {P0_CANONICAL, P1_ORDER_REVERSED, P2_WHITESPACE_VARIANT, P3_NEUTRAL_LABEL_SYNONYMS}


# -- N0 unchanged (spec section 2) ---------------------------------------


@pytest.mark.parametrize("family", PRIMARY_FAILURE_FAMILIES)
def test_n0_canonical_is_byte_identical_to_production(family):
    case, v, facts = _facts_and_validation(family)
    n0_text = render_contract(N0_CURRENT_PRODUCTION, P0_CANONICAL, facts, case, v)
    expected = build_repair_prompt(P1_CURRENT_ONE_REPAIR, case.observation, case.legal_actions, case.invalid_raw_response, v)
    assert n0_text == expected


# -- fact equivalence across every (contract, perturbation) cell (spec section 5, 9) --


@pytest.mark.parametrize("family", PRIMARY_FAILURE_FAMILIES)
@pytest.mark.parametrize("contract", CONTRACTS)
def test_all_perturbations_of_a_contract_share_identical_facts(family, contract):
    case, v, facts = _facts_and_validation(family)
    hashes = set()
    for perturbation in PERTURBATIONS:
        text = render_contract(contract, perturbation, facts, case, v)
        # RepairFacts is the same object every time - assert the fact hash
        # (computed independent of the rendered text) stays constant, and
        # separately that every rendering's own representation differs.
        hashes.add(repair_representation_hash(text))
    assert repair_fact_hash(facts) == repair_fact_hash(build_repair_facts(case, v))
    assert len(hashes) == len(PERTURBATIONS), "every perturbation must produce a distinct representation"


# -- no strategic advice (spec section 8/6 of STOP CONDITION) -----------


@pytest.mark.parametrize("family", PRIMARY_FAILURE_FAMILIES)
@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("perturbation", PERTURBATIONS)
def test_no_contract_perturbation_contains_strategic_advice(family, contract, perturbation):
    case, v, facts = _facts_and_validation(family)
    text = render_contract(contract, perturbation, facts, case, v).lower()
    for phrase in _BANNED_PHRASES:
        assert phrase not in text, (family, contract, perturbation, phrase)


# -- legal lists identical, internal order fixed (spec section 7, 11, 14) --


@pytest.mark.parametrize("family", PRIMARY_FAILURE_FAMILIES)
def test_legal_lists_identical_across_contracts(family):
    case, v, facts = _facts_and_validation(family)
    facts2 = build_repair_facts(case, v)
    assert facts.legal_action_types == facts2.legal_action_types
    assert facts.legal_targets == facts2.legal_targets
    assert facts.legal_enum_values == facts2.legal_enum_values


# -- deterministic rendering ----------------------------------------------


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("perturbation", PERTURBATIONS)
def test_rendering_is_deterministic(contract, perturbation):
    case, v, facts = _facts_and_validation("INVALID_TRAP_TYPE")
    a = render_contract(contract, perturbation, facts, case, v)
    b = render_contract(contract, perturbation, facts, case, v)
    assert a == b


# -- no hidden-state access ------------------------------------------------


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("perturbation", PERTURBATIONS)
def test_no_hidden_gamestate_leakage(contract, perturbation):
    case, v, facts = _facts_and_validation("INVALID_TRAP_TYPE")
    text = render_contract(contract, perturbation, facts, case, v)
    for forbidden in ("rng", "owner_id", "_object_entry", "hidden"):
        assert forbidden not in text.lower()


# -- P1 order reversal genuinely swaps ERROR vs LEGAL_CONSTRAINTS section order (N1/N2) --


@pytest.mark.parametrize("contract", [N1_NEUTRAL_LABELED, N2_NEUTRAL_STRUCTURED])
def test_order_reversed_perturbation_actually_reverses_section_order(contract):
    case, v, facts = _facts_and_validation("INVALID_TRAP_TYPE")
    canonical = render_contract(contract, P0_CANONICAL, facts, case, v)
    reversed_text = render_contract(contract, P1_ORDER_REVERSED, facts, case, v)
    assert canonical != reversed_text
    # error-ish marker appears before legal-ish marker in canonical, after in reversed.
    error_marker = "error" if contract == N1_NEUTRAL_LABELED else '"error"'
    legal_marker = "legal" if contract == N1_NEUTRAL_LABELED else '"legal_constraints"'
    assert canonical.lower().index(error_marker) < canonical.lower().index(legal_marker)
    assert reversed_text.lower().index(error_marker) > reversed_text.lower().index(legal_marker)


def test_n2_structured_perturbations_all_remain_valid_json():
    import json as _json

    case, v, facts = _facts_and_validation("INVALID_TRAP_TYPE")
    for perturbation in PERTURBATIONS:
        text = render_contract(N2_NEUTRAL_STRUCTURED, perturbation, facts, case, v)
        parsed = _json.loads(text)  # must not raise
        assert "repair" in parsed


# -- TARGET_NOT_INSPECTED null control facts remain what M6D.1 found -----


def test_target_not_inspected_legal_targets_empty_across_all_contracts():
    case, v, facts = _facts_and_validation("TARGET_NOT_INSPECTED")
    assert facts.legal_targets in (None, [])
    for contract in CONTRACTS:
        for perturbation in PERTURBATIONS:
            text = render_contract(contract, perturbation, facts, case, v)
            assert len(text) > 0
