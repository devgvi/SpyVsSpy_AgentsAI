"""M6E.1 spec section 20 (Phase A): network-free tests verifying the
duplicate-count/location/format variant construction before any live
call. No model calls anywhere in this file."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import copy

import pytest

from src.evaluation.duplicate_fact_variants import (
    AUXILIARY,
    CONDITIONS,
    D1_SINGLE_COPY,
    D2_DOUBLE_COPY,
    D3_TRIPLE_COPY,
    L_MEM_AUX,
    L_OBS_AUX,
    MEMORY,
    OBSERVATION,
    R1_COMPACT_FORMAT,
    background_fact_payload,
    build_variant,
    multiset_fact_hash,
    representation_hash,
    unique_fact_hash,
)
from src.evaluation.m6e1_scenarios import FULL_TREATMENT_FAMILIES, build_full_treatment_scenarios
from src.evaluation.request_fact_model import extract_request_facts, request_fact_hash

SCENARIOS = build_full_treatment_scenarios()


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_unique_fact_hash_identical_across_every_condition(family):
    s = SCENARIOS[family]
    hashes = {cond: unique_fact_hash(build_variant(cond, s.observation_payload, s.memory_payload, s.fact).observation_payload,
                                       build_variant(cond, s.observation_payload, s.memory_payload, s.fact).memory_payload, s.fact)
              for cond in CONDITIONS}
    assert len(set(hashes.values())) == 1


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_d1_vs_d3_multiset_hash_differs_but_unique_hash_matches(family):
    s = SCENARIOS[family]
    d1 = build_variant(D1_SINGLE_COPY, s.observation_payload, s.memory_payload, s.fact)
    d3 = build_variant(D3_TRIPLE_COPY, s.observation_payload, s.memory_payload, s.fact)
    assert unique_fact_hash(d1.observation_payload, d1.memory_payload, s.fact) == unique_fact_hash(d3.observation_payload, d3.memory_payload, s.fact)
    assert multiset_fact_hash(d1.copy_count, d1.locations, s.fact) != multiset_fact_hash(d3.copy_count, d3.locations, s.fact)


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_copy_count_is_exactly_controlled(family):
    s = SCENARIOS[family]
    expected_counts = {D1_SINGLE_COPY: 1, D2_DOUBLE_COPY: 2, D3_TRIPLE_COPY: 2, L_OBS_AUX: 2, L_MEM_AUX: 2, R1_COMPACT_FORMAT: 2}
    for cond, expected in expected_counts.items():
        v = build_variant(cond, s.observation_payload, s.memory_payload, s.fact)
        if cond == D3_TRIPLE_COPY:
            assert v.copy_count == 3
        else:
            assert v.copy_count == expected


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_location_axis_holds_copy_count_fixed_at_two(family):
    s = SCENARIOS[family]
    for cond in (D2_DOUBLE_COPY, L_OBS_AUX, L_MEM_AUX):
        v = build_variant(cond, s.observation_payload, s.memory_payload, s.fact)
        assert v.copy_count == 2
    locs = {cond: set(build_variant(cond, s.observation_payload, s.memory_payload, s.fact).locations) for cond in (D2_DOUBLE_COPY, L_OBS_AUX, L_MEM_AUX)}
    assert locs[D2_DOUBLE_COPY] == {MEMORY, OBSERVATION}
    assert locs[L_OBS_AUX] == {OBSERVATION, AUXILIARY}
    assert locs[L_MEM_AUX] == {MEMORY, AUXILIARY}
    assert len({frozenset(v) for v in locs.values()}) == 3  # all three location pairs genuinely distinct


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_format_axis_holds_count_and_location_fixed(family):
    s = SCENARIOS[family]
    d2 = build_variant(D2_DOUBLE_COPY, s.observation_payload, s.memory_payload, s.fact)
    r1 = build_variant(R1_COMPACT_FORMAT, s.observation_payload, s.memory_payload, s.fact)
    assert d2.copy_count == r1.copy_count
    assert set(d2.locations) == set(r1.locations)
    assert multiset_fact_hash(d2.copy_count, d2.locations, s.fact) == multiset_fact_hash(r1.copy_count, r1.locations, s.fact)
    assert d2.format_variant != r1.format_variant
    assert len(r1.to_json()) < len(d2.to_json())


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_every_other_fact_stays_identical_across_all_conditions(family):
    """Spec section 6: the unique_request_fact_set (excluding the one
    manipulated fact) must be identical no matter which condition is
    applied."""
    s = SCENARIOS[family]
    hashes = set()
    for cond in CONDITIONS:
        v = build_variant(cond, s.observation_payload, s.memory_payload, s.fact)
        bg_obs, bg_mem = background_fact_payload(v.observation_payload, v.memory_payload, s.fact)
        hashes.add(request_fact_hash(extract_request_facts(bg_obs, bg_mem)))
    assert len(hashes) == 1


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_legal_actions_and_observation_never_change(family):
    s = SCENARIOS[family]
    baseline_legal = s.observation_payload["legal_actions_compact"]
    observation_before = copy.deepcopy(s.observation)
    for cond in CONDITIONS:
        v = build_variant(cond, s.observation_payload, s.memory_payload, s.fact)
        assert v.observation_payload["legal_actions_compact"] == baseline_legal
    assert s.observation == observation_before


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_no_hidden_state_original_payloads_never_mutated(family):
    s = SCENARIOS[family]
    obs_before, mem_before = copy.deepcopy(s.observation_payload), copy.deepcopy(s.memory_payload)
    for cond in CONDITIONS:
        build_variant(cond, s.observation_payload, s.memory_payload, s.fact)
    assert s.observation_payload == obs_before
    assert s.memory_payload == mem_before


@pytest.mark.parametrize("family", FULL_TREATMENT_FAMILIES)
def test_representation_hashes_differ_only_as_intended(family):
    s = SCENARIOS[family]
    d1 = build_variant(D1_SINGLE_COPY, s.observation_payload, s.memory_payload, s.fact)
    d2 = build_variant(D2_DOUBLE_COPY, s.observation_payload, s.memory_payload, s.fact)
    d3 = build_variant(D3_TRIPLE_COPY, s.observation_payload, s.memory_payload, s.fact)
    assert representation_hash(d1.to_json()) != representation_hash(d2.to_json()) != representation_hash(d3.to_json())
    # deterministic
    again = build_variant(D2_DOUBLE_COPY, s.observation_payload, s.memory_payload, s.fact)
    assert representation_hash(d2.to_json()) == representation_hash(again.to_json())


def test_duplicate_copies_are_byte_identical_not_paraphrased():
    """Spec section 14: no wording drift between copies. `memory` and
    `auxiliary` hold the full canonical record byte-identically; the real
    production `observation` echo for KNOWN_LOCKS is genuinely id-only
    (matching its actual production shape, per spec section 7's
    'byte-identical where possible') - checked here by identity, not
    literal dict equality, since that's the real shape asymmetry M6E
    itself already documented, not a manufactured wording difference."""
    s = SCENARIOS["S2_KNOWN_LOCKED_OBJECT"]
    d3 = build_variant(D3_TRIPLE_COPY, s.observation_payload, s.memory_payload, s.fact)
    mem_copy = d3.memory_payload["known_locks"][-1]
    obs_copy_id = d3.observation_payload["memory"]["known_locked_objects"][-1]
    aux_copy = d3.observation_payload["auxiliary_context"]["known_locks"][0]
    assert mem_copy == aux_copy == s.fact.canonical_value
    assert obs_copy_id == s.fact.identity == s.fact.canonical_value["target_id"]


def test_observation_echo_never_disturbs_unrelated_ids_already_present():
    """Regression guard for the real bug found and fixed while building
    this module: a first version wrote to a fabricated parallel key
    (`known_locks`) instead of the real production echo key
    (`known_locked_objects`), leaving that real key - and any OTHER real
    id already in it - completely untouched by every condition. Fixed to
    operate on the real key, adding/removing only this fact's own id."""
    s = SCENARIOS["S6_LONG_HISTORY"]
    # S6's real snapshot already has an unrelated real lock echo
    # (door_security_corridor) that must survive every condition.
    baseline = s.observation_payload["memory"]["known_locked_objects"]
    assert "door_security_corridor" in baseline
    assert s.fact.identity not in baseline  # archives_cabinet is not naturally echoed

    for cond in (D1_SINGLE_COPY, D2_DOUBLE_COPY, D3_TRIPLE_COPY, L_OBS_AUX):
        v = build_variant(cond, s.observation_payload, s.memory_payload, s.fact)
        echoed = v.observation_payload["memory"]["known_locked_objects"]
        assert "door_security_corridor" in echoed
        locations = CONDITIONS[cond][0]
        if OBSERVATION in locations:
            assert s.fact.identity in echoed
        else:
            assert s.fact.identity not in echoed


def test_no_emphasis_language_anywhere_in_variant_payloads():
    banned = ("important", "remember", "warning", "critical")
    for family in FULL_TREATMENT_FAMILIES:
        s = SCENARIOS[family]
        for cond in CONDITIONS:
            text = build_variant(cond, s.observation_payload, s.memory_payload, s.fact).to_json().lower()
            for word in banned:
                assert word not in text, f"{family}/{cond} contains banned emphasis word {word!r}"
