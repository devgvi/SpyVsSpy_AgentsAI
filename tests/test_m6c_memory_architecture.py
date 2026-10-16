"""M6C Phase A (spec section 23): network-free memory-architecture tests.

Covers: M0 contains no persistent memory, M1 unchanged, M2 fact-equivalent
to M1, M2 deterministic serialization, no hidden-state leakage, chronology
preserved, unordered sets canonicalized, Native/Strands transform parity
(RP10), memory hashes stable, snapshot integrity. No network calls.
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
from src.evaluation.memory_representation import (
    M0_MEMORY_OFF,
    M1_CURRENT_MEMORY,
    M2_STRUCTURED_FACTS,
    MEMORY_CONDITIONS,
    extract_memory_facts,
    frozen_condition_transform,
    m0_memory_off_transform,
    m2_structured_facts_transform,
    memory_fact_hash,
    memory_payload_length,
    memory_representation_hash,
    memory_token_estimate,
)
from src.evaluation.memory_snapshot_set import (
    MEMORY_SNAPSHOT_CATEGORIES,
    NO_MEMORY_RELEVANT,
    build_memory_snapshot_set,
)
from src.evaluation.runtime_snapshot_replay import run_one_trial
from src.llm.agent_memory import AgentMemory
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig
from tests.test_strands_agent import QueuedStrandsFactory

SNAPSHOTS = {s.category: s for s in build_memory_snapshot_set()}


def _sample_memory() -> AgentMemory:
    mem = AgentMemory(constants.WHITE_SPY_ID)
    mem.record_visited_room("corridor")
    mem.record_visited_room("reception")
    mem.record_known_lock("ambassador_safe", "code", "unknown", satisfied=False)
    mem.record_interaction(10.0, "search", "desk_01", "empty")
    mem.record_failure(12.0, "interact", "ambassador_safe", "locked")
    mem.record_decision(13.0, "d1", "move", "door_1", True, "llm")
    mem.record_decision(14.0, "d2", "inspect", "sofa_01", True, "llm")
    mem.record_opponent_sighting(14.0, "corridor")
    return mem


# -- snapshot integrity (spec section 26) ------------------------------------


def test_eight_memory_snapshot_categories_present_and_unique():
    assert len(MEMORY_SNAPSHOT_CATEGORIES) == 8
    assert len(set(MEMORY_SNAPSHOT_CATEGORIES)) == 8
    assert set(SNAPSHOTS.keys()) == set(MEMORY_SNAPSHOT_CATEGORIES)


def test_negative_control_has_no_nearby_interactable_and_empty_memory():
    s = SNAPSHOTS[NO_MEMORY_RELEVANT]
    assert s.observation.nearby_object_id is None
    facts = extract_memory_facts(s.synthetic_memory)
    # Every fact present should be the trivial/empty default - no seeded
    # content at all (spec section 25).
    non_trivial = [f for f in facts if f[0] not in (
        "mission_have", "mission_missing", "mission_ready_to_exit", "visited_rooms", "known_locks",
        "known_empty_objects", "known_searched_objects", "opponent_currently_visible", "opponent_last_seen",
        "recent_interactions", "recent_failures", "recent_events", "recent_decisions",
        "recent_repetitions_count", "messages_sent_count", "messages_received_count",
    ) or (f[1] not in ((), False, 0, None))]
    assert non_trivial == [], non_trivial


def test_every_memory_snapshot_has_a_distinct_synthetic_memory_from_no_memory_relevant():
    empty_facts = extract_memory_facts(SNAPSHOTS[NO_MEMORY_RELEVANT].synthetic_memory)
    for category, snap in SNAPSHOTS.items():
        if category == NO_MEMORY_RELEVANT:
            continue
        assert extract_memory_facts(snap.synthetic_memory) != empty_facts, category


# -- M0/M1/M2 core semantics (spec sections 9-12, 62) ------------------------


def test_m0_contains_no_persistent_memory():
    payload = m0_memory_off_transform({"strategy_memory": _sample_memory().to_compact_dict(), "other_key": 1})
    assert payload["other_key"] == 1  # non-memory payload keys untouched
    assert extract_memory_facts(payload["strategy_memory"]) == frozenset()
    assert payload["strategy_memory"]["memory_condition"] == M0_MEMORY_OFF


def test_m1_is_byte_identical_to_agentmemory_to_compact_dict():
    mem = _sample_memory()
    m1_dict = mem.to_compact_dict()
    # M1 requires no transform at all in live episodes (spec section 10) -
    # the dict handed to _repair_instruction/json.dumps is exactly what
    # AgentMemory itself produces, nothing added or removed.
    assert m1_dict == mem.to_compact_dict()
    assert "memory_condition" not in m1_dict  # never stamped - genuinely untouched


def test_m2_is_fact_equivalent_to_m1_but_different_representation():
    mem = _sample_memory()
    m1_dict = mem.to_compact_dict()
    m2_payload = m2_structured_facts_transform({"strategy_memory": m1_dict})
    m2_dict = m2_payload["strategy_memory"]

    assert extract_memory_facts(m1_dict) == extract_memory_facts(m2_dict)
    assert memory_fact_hash(m1_dict) == memory_fact_hash(m2_dict)
    # Expected per spec section 62: facts equal, representation hash NOT equal.
    assert memory_representation_hash(m1_dict) != memory_representation_hash(m2_dict)
    assert m2_dict["memory_condition"] == M2_STRUCTURED_FACTS


def test_m2_serialization_is_deterministic():
    mem = _sample_memory()
    m1_dict = mem.to_compact_dict()
    m2_a = m2_structured_facts_transform({"strategy_memory": m1_dict})["strategy_memory"]
    m2_b = m2_structured_facts_transform({"strategy_memory": m1_dict})["strategy_memory"]
    assert m2_a == m2_b
    assert memory_representation_hash(m2_a) == memory_representation_hash(m2_b)


@pytest.mark.parametrize("category", MEMORY_SNAPSHOT_CATEGORIES)
def test_fact_equivalence_holds_for_every_real_snapshot(category):
    snap = SNAPSHOTS[category]
    m1_dict = snap.synthetic_memory
    m2_dict = m2_structured_facts_transform({"strategy_memory": m1_dict})["strategy_memory"]
    assert extract_memory_facts(m1_dict) == extract_memory_facts(m2_dict), category


# -- no hidden-state leakage (spec section 13) -------------------------------


def test_m2_never_contains_hidden_gamestate_fields():
    # AgentMemory itself only ever records actor-legitimate facts (its own
    # class docstring: "Memory never reconstructs hidden GameState truth"),
    # and M2 only reshapes what M1 already contains - assert by construction
    # that no private/engine-only key names leak through the reshape.
    forbidden_keys = {"rng", "actors", "level", "_object_entry", "trap", "owner_id"}
    for category, snap in SNAPSHOTS.items():
        m2_dict = m2_structured_facts_transform({"strategy_memory": snap.synthetic_memory})["strategy_memory"]
        serialized = json.dumps(m2_dict, default=str)
        for key in forbidden_keys:
            assert f'"{key}"' not in serialized, (category, key)


# -- chronology / canonical ordering (spec section 16) -----------------------


def test_chronological_facts_preserve_order_in_m2():
    mem = AgentMemory(constants.WHITE_SPY_ID)
    mem.record_decision(1.0, "d1", "move", "a", True, "llm")
    mem.record_decision(2.0, "d2", "move", "b", True, "llm")
    mem.record_decision(3.0, "d3", "inspect", "c", True, "llm")
    m1_dict = mem.to_compact_dict()
    m2_dict = m2_structured_facts_transform({"strategy_memory": m1_dict})["strategy_memory"]

    m1_order = [(d["selected_action"], d["target_id"]) for d in m1_dict["recent_decisions"]]
    m2_order = [(d["action"], d["target_id"]) for d in m2_dict["recent_decisions"]]
    assert m1_order == m2_order == [("move", "a"), ("move", "b"), ("inspect", "c")]


def test_unordered_sets_are_canonicalized_in_m2():
    mem = AgentMemory(constants.WHITE_SPY_ID)
    mem.record_visited_room("zeta_room")
    mem.record_visited_room("alpha_room")
    mem.record_visited_room("mu_room")
    m1_dict = mem.to_compact_dict()
    m2_dict = m2_structured_facts_transform({"strategy_memory": m1_dict})["strategy_memory"]
    assert m2_dict["visited_rooms"] == sorted(m2_dict["visited_rooms"])
    assert set(m2_dict["visited_rooms"]) == set(m1_dict["exploration"]["visited_rooms"])


# -- memory hashes (spec section 22) -----------------------------------------


def test_memory_hashes_are_stable_and_sensitive_to_real_changes():
    mem_a = _sample_memory()
    mem_b = _sample_memory()
    dict_a = mem_a.to_compact_dict()
    dict_b = mem_b.to_compact_dict()
    assert memory_fact_hash(dict_a) == memory_fact_hash(dict_b)
    assert memory_representation_hash(dict_a) == memory_representation_hash(dict_b)

    mem_b.record_visited_room("a_genuinely_new_room")
    dict_b2 = mem_b.to_compact_dict()
    assert memory_fact_hash(dict_a) != memory_fact_hash(dict_b2)


def test_token_and_length_estimates_are_positive_and_monotonic_with_content():
    empty_dict = AgentMemory(constants.WHITE_SPY_ID).to_compact_dict()
    full_dict = _sample_memory().to_compact_dict()
    assert memory_payload_length(full_dict) > memory_payload_length(empty_dict) > 0
    assert memory_token_estimate(full_dict) >= memory_token_estimate(empty_dict) >= 1


# -- Native/Strands transform parity (RP10) ----------------------------------


class _CapturingMockModelClient(MockModelClient):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured_prompts: list[str] = []

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        self.captured_prompts.append(user_prompt)
        return super().generate_action(system_prompt, user_prompt, output_schema, request_context)


def _native_trial_with_transform(transform, obs, episode_id="m6c-native"):
    telemetry = TelemetryLogger()
    client = _CapturingMockModelClient(response={"action": "wait", "reason": "x"})
    adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=client,
        config=LLMControllerConfig(model_payload_transform=transform), telemetry=telemetry, episode_id=episode_id,
    )
    run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)
    return client.captured_prompts[0] if client.captured_prompts else ""


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


def _strands_trial_with_transform(transform, obs, episode_id="m6c-strands"):
    telemetry = TelemetryLogger()
    sink: list[str] = []
    factory = _RecordingQueuedFactory([[{"action": "wait", "reason": "x"}]], sink)
    adapter = StrandsAgentAdapter(
        constants.WHITE_SPY_ID, agent_factory=factory, telemetry=telemetry, episode_id=episode_id,
        model_payload_transform=transform,
    )
    run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)
    return sink[0] if sink else ""


@pytest.mark.parametrize("condition", MEMORY_CONDITIONS)
def test_native_and_strands_receive_equivalent_transformed_memory(condition):
    snap = SNAPSHOTS[NO_MEMORY_RELEVANT]  # deterministic, no live-model-choice dependence
    # A richer synthetic memory than the negative control's own empty one,
    # injected identically for both runtimes.
    synthetic = _sample_memory().to_compact_dict()
    transform = frozen_condition_transform(condition, synthetic)

    native_prompt = _native_trial_with_transform(transform, snap.observation)
    strands_prompt = _strands_trial_with_transform(transform, snap.observation)

    native_payload = json.loads(native_prompt)
    strands_payload = json.loads(strands_prompt)
    native_facts = extract_memory_facts(native_payload["strategy_memory"])
    strands_facts = extract_memory_facts(strands_payload["strategy_memory"])
    assert native_facts == strands_facts
    assert native_payload["strategy_memory"].get("memory_condition") == strands_payload["strategy_memory"].get("memory_condition")


def test_rp10_memory_transform_still_default_off_is_unchanged():
    # Spec section 19 (carried over from M6B.2): omitting the transform
    # entirely must be byte-identical to every pre-M6C caller.
    snap = SNAPSHOTS[NO_MEMORY_RELEVANT]
    native_prompt = _native_trial_with_transform(None, snap.observation)
    payload = json.loads(native_prompt)
    assert "strategy_memory" in payload
    assert "memory_condition" not in payload["strategy_memory"]
