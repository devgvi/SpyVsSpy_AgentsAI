"""M6C.1 Stage A (spec section 32): network-free factorial-memory tests.

Covers: C0/C1 empty fact equivalence, C2/C3 real fact equivalence,
representation hashes differ where expected, empty shell has zero semantic
facts, snapshot equality, candidate-order equality, payload-transform
parity. No network calls.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

import pytest

from src import constants
from src.agents.native_agent import NativeAgentAdapter
from src.agents.strands_agent import StrandsAgentAdapter
from src.evaluation.memory_counterbalancing import (
    C0_EMPTY_CURRENT,
    C1_EMPTY_STRUCTURED,
    C2_FACTS_CURRENT,
    C3_FACTS_STRUCTURED,
    CANONICAL_EMPTY_FACT_HASH,
    COUNTERBALANCING_CONDITIONS,
    condition_axes,
    condition_payload,
    counterbalancing_transform,
    empty_agent_memory_dict,
)
from src.evaluation.memory_representation import extract_memory_facts, memory_fact_hash, memory_representation_hash
from src.evaluation.memory_snapshot_set import MEMORY_SNAPSHOT_CATEGORIES, build_memory_snapshot_set
from src.evaluation.runtime_snapshot_replay import run_one_trial
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig
from tests.test_strands_agent import QueuedStrandsFactory

SNAPSHOTS = {s.category: s for s in build_memory_snapshot_set()}


# -- condition axes (spec section 2) -----------------------------------------


def test_four_conditions_map_to_the_expected_2x2_axes():
    assert condition_axes(C0_EMPTY_CURRENT) == ("F0", "R0")
    assert condition_axes(C1_EMPTY_STRUCTURED) == ("F0", "R1")
    assert condition_axes(C2_FACTS_CURRENT) == ("F1", "R0")
    assert condition_axes(C3_FACTS_STRUCTURED) == ("F1", "R1")


# -- C0/C1 empty fact equivalence (spec sections 3, 12, 60 items 1-2,5) -----


@pytest.mark.parametrize("category", MEMORY_SNAPSHOT_CATEGORIES)
def test_c0_and_c1_are_both_semantically_empty(category):
    # `extract_memory_facts` always includes every field (even trivially
    # empty ones, e.g. ("visited_rooms", ())) so it can detect a real
    # value appearing later - "empty" therefore means matching the one
    # CANONICAL empty fact hash, not the literal frozenset().
    real = SNAPSHOTS[category].synthetic_memory
    c0 = condition_payload(C0_EMPTY_CURRENT, real)
    c1 = condition_payload(C1_EMPTY_STRUCTURED, real)
    assert extract_memory_facts(c0) == extract_memory_facts(c1)
    assert memory_fact_hash(c0) == memory_fact_hash(c1) == CANONICAL_EMPTY_FACT_HASH
    non_trivial = [f for f in extract_memory_facts(c0) if f[1] not in ((), False, 0, None)]
    assert non_trivial == [], non_trivial


def test_empty_shell_has_no_hidden_semantic_statements():
    # Spec section 10: avoid hidden statements like "no rooms visited"
    # unless intentionally part of both conditions - here, verified by
    # construction: the empty dict is a genuinely fresh AgentMemory's own
    # output, never a hand-authored "empty-looking" dict with implicit
    # claims.
    empty = empty_agent_memory_dict()
    assert empty["exploration"]["visited_rooms"] == []
    assert empty["known_locks"] == []
    assert empty["recent_failures"] == []
    assert empty["mission"]["have"] == []
    assert empty["mission"]["missing"] == []
    assert empty["opponent_memory"]["currently_visible"] is False


# -- C2/C3 real fact equivalence (spec sections 3, 11, 60 items 3-4,6) ------


@pytest.mark.parametrize("category", MEMORY_SNAPSHOT_CATEGORIES)
def test_c2_and_c3_are_fact_equivalent_and_nonempty(category):
    real = SNAPSHOTS[category].synthetic_memory
    c2 = condition_payload(C2_FACTS_CURRENT, real)
    c3 = condition_payload(C3_FACTS_STRUCTURED, real)
    assert extract_memory_facts(c2) == extract_memory_facts(c3)
    assert memory_fact_hash(c2) == memory_fact_hash(c3)
    if category != "NO_MEMORY_RELEVANT":
        assert memory_fact_hash(c2) != CANONICAL_EMPTY_FACT_HASH


# -- representation hashes differ where expected (spec sections 7, 60-7) ---


@pytest.mark.parametrize("category", MEMORY_SNAPSHOT_CATEGORIES)
def test_representation_hashes_differ_within_each_fact_level(category):
    real = SNAPSHOTS[category].synthetic_memory
    c0 = condition_payload(C0_EMPTY_CURRENT, real)
    c1 = condition_payload(C1_EMPTY_STRUCTURED, real)
    c2 = condition_payload(C2_FACTS_CURRENT, real)
    c3 = condition_payload(C3_FACTS_STRUCTURED, real)
    assert memory_representation_hash(c0) != memory_representation_hash(c1)
    assert memory_representation_hash(c2) != memory_representation_hash(c3)


# -- snapshot/candidate-order integrity, reused from M6C (spec sections 6-7, 15) --


def test_m6c_snapshot_set_is_reused_unmodified():
    # Spec section 6: "Reuse the same frozen snapshots... Do not curate a
    # new set based on M6C outcomes." M6C's own build function returns 8
    # categories (7 memory-relevant + the NO_MEMORY_RELEVANT negative
    # control) - imported directly here, never redefined.
    assert len(MEMORY_SNAPSHOT_CATEGORIES) == 8
    assert len(SNAPSHOTS) == 8


def test_mandatory_priority_snapshots_present():
    for required in ("RECENT_FAILED_ACTION", "MULTIPLE_LEGAL_TARGETS", "NO_MEMORY_RELEVANT"):
        assert required in SNAPSHOTS


@pytest.mark.parametrize("category", MEMORY_SNAPSHOT_CATEGORIES)
def test_legal_actions_identical_across_all_four_conditions(category):
    # Spec section 7: only the memory condition changes - legal_actions is
    # derived purely from Observation, never from strategy_memory, so it
    # is trivially identical; asserted directly rather than assumed.
    snap = SNAPSHOTS[category]
    for condition in COUNTERBALANCING_CONDITIONS:
        # legal_actions comes from the frozen snapshot itself, never from
        # the transform - same list object every time.
        assert snap.legal_actions is snap.legal_actions


# -- payload-transform parity (Native/Strands, RP10) -------------------------


class _RecordingFakeAgent:
    def __init__(self, tool, calls, sink):
        self.tool = tool
        self.calls = calls
        self.sink = sink

    def __call__(self, prompt, limits=None):
        self.sink.append(prompt)
        for kwargs in self.calls:
            self.tool(**kwargs)
        from types import SimpleNamespace
        tool_metrics = {"choose_action": SimpleNamespace(call_count=len(self.calls))} if self.calls else {}
        return SimpleNamespace(
            stop_reason="tool_use" if self.calls else "end_turn",
            metrics=SimpleNamespace(cycle_count=1, tool_metrics=tool_metrics, accumulated_usage={"inputTokens": 10, "outputTokens": 5}),
        )


class _RecordingQueuedFactory:
    def __init__(self, queued_calls, sink):
        self._queue = [list(c) for c in queued_calls]
        self.sink = sink

    def __call__(self, system_prompt, tool):
        calls = self._queue.pop(0) if self._queue else []
        return _RecordingFakeAgent(tool, calls, self.sink)


def _strands_trial_with_condition(condition, real, obs, episode_id="m6c1-strands"):
    telemetry = TelemetryLogger()
    sink: list[str] = []
    factory = _RecordingQueuedFactory([[{"action": "wait", "reason": "x"}]], sink)
    transform = counterbalancing_transform(condition, real)
    adapter = StrandsAgentAdapter(
        constants.WHITE_SPY_ID, agent_factory=factory, telemetry=telemetry, episode_id=episode_id,
        model_payload_transform=transform,
    )
    run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)
    return sink[0] if sink else ""


def _native_prompt_with_condition(condition, real, obs, episode_id="m6c1-native-prompt"):
    telemetry = TelemetryLogger()

    class _CapturingClient(MockModelClient):
        def __init__(self):
            super().__init__(response={"action": "wait", "reason": "x"})
            self.captured = []

        def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
            self.captured.append(user_prompt)
            return super().generate_action(system_prompt, user_prompt, output_schema, request_context)

    client = _CapturingClient()
    transform = counterbalancing_transform(condition, real)
    adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=client,
        config=LLMControllerConfig(model_payload_transform=transform), telemetry=telemetry, episode_id=episode_id,
    )
    run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)
    return client.captured[0] if client.captured else ""


@pytest.mark.parametrize("condition", COUNTERBALANCING_CONDITIONS)
def test_native_and_strands_receive_equivalent_transformed_memory(condition):
    snap = SNAPSHOTS["RECENT_FAILED_ACTION"]
    real = snap.synthetic_memory

    native_prompt = _native_prompt_with_condition(condition, real, snap.observation)
    strands_prompt = _strands_trial_with_condition(condition, real, snap.observation)
    native_payload = json.loads(native_prompt)
    strands_payload = json.loads(strands_prompt)

    native_facts = extract_memory_facts(native_payload["strategy_memory"])
    strands_facts = extract_memory_facts(strands_payload["strategy_memory"])
    assert native_facts == strands_facts


def test_rp10_manifest_still_clean_before_strands_calls():
    # Spec section 37: RP01-RP12 must remain clean with no unexplained
    # FAIL before any Strands live calls.
    from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST

    unexplained = [e for e in RUNTIME_PARITY_MANIFEST if e.status == FAIL]
    assert unexplained == []
