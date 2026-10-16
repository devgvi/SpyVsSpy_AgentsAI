"""M6E.3 spec section 20 (Phase A): network-free history-window tests. No
model calls anywhere in this file."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import copy

import pytest

from src.evaluation.history_window import (
    H0,
    H1,
    H2,
    H3,
    H4,
    H_FULL,
    HISTORY_FIELDS,
    WINDOW_ORDER,
    WINDOWS,
    apply_window,
    event_manifest,
    history_fact_hash,
    history_full_hash,
    history_sequence_hash,
    history_window_hash,
)
from src.evaluation.m6e3_scenarios import (
    KNOWN_FAILURE_ACTION_SIGNATURE,
    PRIMARY_FAMILIES,
    build_primary_scenarios,
    build_window,
    repeats_known_failure,
)
from src.evaluation.m6e_scenarios import S6_LONG_HISTORY

SCENARIOS = build_primary_scenarios()


def test_window_order_is_largest_to_smallest():
    assert WINDOW_ORDER == (H_FULL, H4, H3, H2, H1, H0)
    caps = [WINDOWS[w] for w in WINDOW_ORDER[1:]]  # exclude H_FULL (None)
    assert caps == sorted(caps, reverse=True)


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_windows_retain_exactly_the_most_recent_n_entries(family):
    s = SCENARIOS[family]
    for window, cap in WINDOWS.items():
        r = build_window(s, window)
        for field in HISTORY_FIELDS:
            full = s.memory_payload_full.get(field, [])
            expected = full if cap is None else (full[-cap:] if cap > 0 else [])
            assert r.memory_payload.get(field, []) == expected


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_chronology_preserved_within_every_window(family):
    s = SCENARIOS[family]
    for window in WINDOW_ORDER:
        r = build_window(s, window)
        for field in HISTORY_FIELDS:
            times = [e.get("game_time") for e in r.memory_payload.get(field, [])]
            assert times == sorted(times)


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_current_observation_legal_actions_unchanged_across_windows(family):
    s = SCENARIOS[family]
    baseline_legal = s.observation_payload["legal_actions_compact"]
    observation_before = copy.deepcopy(s.observation)
    for window in WINDOW_ORDER:
        build_window(s, window)
        assert s.observation_payload["legal_actions_compact"] == baseline_legal
    assert s.observation == observation_before


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_nonhistory_memory_fields_unchanged_across_windows(family):
    s = SCENARIOS[family]
    nonhistory_keys = [k for k in s.memory_payload_full if k not in HISTORY_FIELDS]
    for window in WINDOW_ORDER:
        r = build_window(s, window)
        for k in nonhistory_keys:
            assert r.memory_payload.get(k) == s.memory_payload_full.get(k)


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_no_hidden_state_original_memory_never_mutated(family):
    s = SCENARIOS[family]
    before = copy.deepcopy(s.memory_payload_full)
    for window in WINDOW_ORDER:
        apply_window(s.memory_payload_full, window)
    assert s.memory_payload_full == before


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_no_fabricated_events_every_retained_entry_traces_to_full_history(family):
    s = SCENARIOS[family]
    full_entries = {field: [str(e) for e in s.memory_payload_full.get(field, [])] for field in HISTORY_FIELDS}
    for window in WINDOW_ORDER:
        r = build_window(s, window)
        for field in HISTORY_FIELDS:
            for entry in r.memory_payload.get(field, []):
                assert str(entry) in full_entries[field]


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_hashes_deterministic(family):
    s = SCENARIOS[family]
    for window in WINDOW_ORDER:
        a = build_window(s, window)
        b = build_window(s, window)
        assert history_window_hash(a.memory_payload) == history_window_hash(b.memory_payload)
        assert history_fact_hash(a.memory_payload) == history_fact_hash(b.memory_payload)
        assert history_sequence_hash(a.memory_payload) == history_sequence_hash(b.memory_payload)


def test_h_full_hash_matches_history_full_hash():
    s = SCENARIOS[S6_LONG_HISTORY]
    full = build_window(s, H_FULL)
    assert history_window_hash(full.memory_payload) == history_full_hash(s.memory_payload_full)


def test_known_failure_event_present_h_full_h4_h3_absent_h2_h1_h0():
    """Spec section 17: the exact known-failure event and its window
    boundary, recorded before live execution."""
    s = SCENARIOS[S6_LONG_HISTORY]
    for window in (H_FULL, H4, H3):
        r = build_window(s, window)
        targets = [e.get("target_id") for e in r.memory_payload.get("recent_failures", [])]
        assert "filing_cabinet_01" in targets
    for window in (H2, H1, H0):
        r = build_window(s, window)
        targets = [e.get("target_id") for e in r.memory_payload.get("recent_failures", [])]
        assert "filing_cabinet_01" not in targets


def test_repeats_known_failure_matches_frozen_signature():
    assert repeats_known_failure("INTERACT", "filing_cabinet_01") is True
    assert repeats_known_failure("search", "filing_cabinet_01") is True  # aliases to interact
    assert repeats_known_failure("inspect", "filing_cabinet_01") is False
    assert repeats_known_failure("interact", "drawer_01") is False
    assert KNOWN_FAILURE_ACTION_SIGNATURE == {"action": "interact", "target_id": "filing_cabinet_01"}


def test_event_manifest_presence_flags_match_window_semantics():
    s = SCENARIOS[S6_LONG_HISTORY]
    manifest = event_manifest(s.memory_payload_full)
    failure_entries = [e for e in manifest if e["field"] == "recent_failures"]
    assert len(failure_entries) == 3
    known = next(e for e in failure_entries if e["event"].get("target_id") == "filing_cabinet_01")
    assert known["present_in_window"][H_FULL] is True
    assert known["present_in_window"][H3] is True
    assert known["present_in_window"][H2] is False


@pytest.mark.parametrize("family", PRIMARY_FAMILIES)
def test_s1_and_s0_redundant_windows_are_byte_identical_by_construction(family):
    """S1's history lists top out at 1 entry and S0's seeded history tops
    out at 3 - several windows in WINDOW_ORDER therefore coincide exactly
    (spec section 6's own accommodation for scenarios with fewer entries
    than the illustrative ladder). This test documents which windows are
    genuinely redundant so live sampling can skip them without silently
    under-testing S6."""
    s = SCENARIOS[family]
    hashes = {w: history_window_hash(build_window(s, w).memory_payload) for w in WINDOW_ORDER}
    distinct = len(set(hashes.values()))
    if family == S6_LONG_HISTORY:
        assert distinct == 6  # every window genuinely distinct
    else:
        assert distinct < 6  # at least one redundant pair exists for S1/S0
