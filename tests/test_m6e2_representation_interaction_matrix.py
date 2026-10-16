"""M6E.2 spec section 17 (Phase A): network-free factorial construction
tests. No model calls anywhere in this file."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import copy

import pytest

from src.evaluation.duplicate_fact_variants import AUXILIARY, MEMORY, OBSERVATION, unique_fact_hash
from src.evaluation.m6e2_factorial import (
    A0_SINGLE_COPY,
    A1_DOUBLE_COPY,
    B0_NATURAL_LOCATION,
    B1_AUXILIARY_LOCATION,
    C0_FULL_FORMAT,
    C1_COMPACT_FORMAT,
    FACTORIAL_CELLS,
    PRIMARY_FAMILIES,
    build_cell,
    build_primary_scenarios,
    duplicate_location_hash,
    format_hash,
)
from src.evaluation.request_fact_model import extract_request_facts, request_fact_hash
from src.evaluation.duplicate_fact_variants import background_fact_payload, multiset_fact_hash

SCENARIOS = build_primary_scenarios()
ALL_CELLS = list(FACTORIAL_CELLS.keys())


def test_all_three_primary_families_exist():
    assert set(SCENARIOS.keys()) == set(PRIMARY_FAMILIES)


def test_factorial_cube_has_exactly_eight_cells():
    assert len(FACTORIAL_CELLS) == 8
    axes_a = {c[0] for c in ALL_CELLS}
    axes_b = {c[1] for c in ALL_CELLS}
    axes_c = {c[2] for c in ALL_CELLS}
    assert axes_a == {A0_SINGLE_COPY, A1_DOUBLE_COPY}
    assert axes_b == {B0_NATURAL_LOCATION, B1_AUXILIARY_LOCATION}
    assert axes_c == {C0_FULL_FORMAT, C1_COMPACT_FORMAT}


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_unique_fact_hash_identical_across_all_eight_cells(family):
    s = SCENARIOS[family]
    hashes = {unique_fact_hash(build_cell(cell, s).observation_payload, build_cell(cell, s).memory_payload, s.fact) for cell in ALL_CELLS}
    assert len(hashes) == 1


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_copy_count_matches_axis_a(family):
    s = SCENARIOS[family]
    for cell in ALL_CELLS:
        v = build_cell(cell, s)
        expected = 1 if cell[0] == A0_SINGLE_COPY else 2
        assert v.copy_count == expected


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_memory_present_in_every_cell_location_varies_only_the_second_section(family):
    """Axis B holds `memory` constant and varies only whether the second
    slot is `observation` or `auxiliary` - never both presence-of-memory
    and the other section at once (spec section 5's intended surgical
    reading, kept consistent with M6E.1's own one-axis-at-a-time
    discipline one level up)."""
    s = SCENARIOS[family]
    for cell in ALL_CELLS:
        v = build_cell(cell, s)
        if cell[0] == A1_DOUBLE_COPY:
            assert MEMORY in v.locations
            other = set(v.locations) - {MEMORY}
            assert other == ({OBSERVATION} if cell[1] == B0_NATURAL_LOCATION else {AUXILIARY})
        else:
            assert v.locations == ((MEMORY,) if cell[1] == B0_NATURAL_LOCATION else (AUXILIARY,))


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_format_matches_axis_c(family):
    s = SCENARIOS[family]
    for cell in ALL_CELLS:
        v = build_cell(cell, s)
        expected = "FULL_FORMAT" if cell[2] == C0_FULL_FORMAT else "COMPACT_FORMAT"
        assert v.format_variant == expected


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_multiset_hash_tracks_copy_count_duplicate_location_hash_tracks_placement_format_hash_tracks_serialization(family):
    s = SCENARIOS[family]
    variants = {cell: build_cell(cell, s) for cell in ALL_CELLS}
    multiset_hashes = {cell: multiset_fact_hash(v.copy_count, v.locations, s.fact) for cell, v in variants.items()}
    loc_hashes = {cell: duplicate_location_hash(v.locations) for cell, v in variants.items()}
    fmt_hashes = {cell: format_hash(v.format_variant) for cell, v in variants.items()}

    # Cells sharing the same (A,B) but differing C must share multiset+location hash, differ in format hash.
    for a in (A0_SINGLE_COPY, A1_DOUBLE_COPY):
        for b in (B0_NATURAL_LOCATION, B1_AUXILIARY_LOCATION):
            c0, c1 = (a, b, C0_FULL_FORMAT), (a, b, C1_COMPACT_FORMAT)
            assert multiset_hashes[c0] == multiset_hashes[c1]
            assert loc_hashes[c0] == loc_hashes[c1]
            assert fmt_hashes[c0] != fmt_hashes[c1]

    # Cells sharing the same A but differing B must differ in location hash.
    for a in (A0_SINGLE_COPY, A1_DOUBLE_COPY):
        for c in (C0_FULL_FORMAT, C1_COMPACT_FORMAT):
            c_nat, c_aux = (a, B0_NATURAL_LOCATION, c), (a, B1_AUXILIARY_LOCATION, c)
            assert loc_hashes[c_nat] != loc_hashes[c_aux]

    # Cells sharing the same B but differing A must differ in multiset hash.
    for b in (B0_NATURAL_LOCATION, B1_AUXILIARY_LOCATION):
        for c in (C0_FULL_FORMAT, C1_COMPACT_FORMAT):
            c_single, c_double = (A0_SINGLE_COPY, b, c), (A1_DOUBLE_COPY, b, c)
            assert multiset_hashes[c_single] != multiset_hashes[c_double]


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_legal_actions_and_candidate_order_identical_across_all_cells(family):
    s = SCENARIOS[family]
    baseline = s.observation_payload["legal_actions_compact"]
    for cell in ALL_CELLS:
        v = build_cell(cell, s)
        assert v.observation_payload["legal_actions_compact"] == baseline


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_observation_semantics_never_change(family):
    s = SCENARIOS[family]
    observation_before = copy.deepcopy(s.observation)
    for cell in ALL_CELLS:
        build_cell(cell, s)
    assert s.observation == observation_before


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_background_facts_identical_across_all_eight_cells(family):
    s = SCENARIOS[family]
    hashes = set()
    for cell in ALL_CELLS:
        v = build_cell(cell, s)
        bg_obs, bg_mem = background_fact_payload(v.observation_payload, v.memory_payload, s.fact)
        hashes.add(request_fact_hash(extract_request_facts(bg_obs, bg_mem)))
    assert len(hashes) == 1


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_deterministic_rendering(family):
    s = SCENARIOS[family]
    for cell in ALL_CELLS:
        a = build_cell(cell, s)
        b = build_cell(cell, s)
        assert a.to_json() == b.to_json()


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_no_hidden_state_leakage_original_payloads_never_mutated(family):
    s = SCENARIOS[family]
    obs_before, mem_before = copy.deepcopy(s.observation_payload), copy.deepcopy(s.memory_payload)
    for cell in ALL_CELLS:
        build_cell(cell, s)
    assert s.observation_payload == obs_before
    assert s.memory_payload == mem_before


def test_fabricated_key_permanent_regression_guard_still_green():
    """Spec section 9: the M6E.1 bug (observation echo written to a
    fabricated key while the real production echo stayed untouched) must
    remain caught. Re-verified here at the factorial level: the real
    `known_locked_objects` key only ever gets populated at the one cell
    that actually includes `observation` (A1/double-copy, B0/natural) -
    never a synthetic shadow key substituting for it, and never populated
    by cells that don't include `observation` in their locations."""
    s = SCENARIOS["S2_KNOWN_LOCKED_OBJECT"]
    double_natural = build_cell((A1_DOUBLE_COPY, B0_NATURAL_LOCATION, C0_FULL_FORMAT), s)  # locations = memory+observation
    double_aux = build_cell((A1_DOUBLE_COPY, B1_AUXILIARY_LOCATION, C0_FULL_FORMAT), s)  # locations = memory+auxiliary
    single_natural = build_cell((A0_SINGLE_COPY, B0_NATURAL_LOCATION, C0_FULL_FORMAT), s)  # locations = memory only

    assert s.fact.identity in double_natural.observation_payload["memory"]["known_locked_objects"]
    assert s.fact.identity not in double_aux.observation_payload["memory"]["known_locked_objects"]
    assert s.fact.identity not in single_natural.observation_payload["memory"]["known_locked_objects"]
    for v in (double_natural, double_aux, single_natural):
        assert "known_locks" not in v.observation_payload["memory"]  # no fabricated shadow key


def test_no_emphasis_language_anywhere_in_any_cell():
    banned = ("important", "remember", "critical", "warning", "priority")
    for family in PRIMARY_FAMILIES:
        s = SCENARIOS[family]
        for cell in ALL_CELLS:
            text = build_cell(cell, s).to_json().lower()
            for word in banned:
                assert word not in text, f"{family}/{cell} contains banned emphasis word {word!r}"
