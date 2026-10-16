"""M6B: PLACE_TRAP tool-schema/argument-reliability investigation.

Phase A (retrospective classification, network-free, spec section 27) +
regression tests for the real repair-hint bug found while starting this
milestone + Phase B (network-free schema-variant contract tests) + Phase C
(mock tool-call tests across both adapters, spec section 29). No network
calls anywhere in this file.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from types import SimpleNamespace

from src import constants
from src.agents.native_agent import NativeAgentAdapter
from src.agents.strands_agent import StrandsAgentAdapter
from src.evaluation.place_trap_reliability_analysis import (
    DEPLETED_TRAP_TYPE,
    INVALID_TARGET,
    INVALID_TRAP_TYPE,
    MISSING_TARGET,
    MISSING_TRAP_TYPE,
    RUNTIME_PARSE_FAILURE,
    SUCCESS,
    classify_place_trap_event,
    summarize_place_trap_attempts,
)
from src.evaluation.runtime_snapshot_replay import build_snapshot_set, run_one_trial
from src.evaluation.runtime_variance_analysis import assert_snapshot_integrity
from src.evaluation.trap_snapshot_set import (
    MULTIPLE_TYPES_REMAINING,
    ONE_TYPE_DEPLETED,
    OPPONENT_NEARBY,
    SINGLE_TYPE_REMAINING,
    build_trap_snapshot_set,
)
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig, serialize_observation
from tests.test_strands_agent import QueuedStrandsFactory

TRAP_SNAPSHOT = next(s for s in build_snapshot_set() if s.category == "TRAP_PLACEMENT_OPPORTUNITY")


class _RecordingFakeAgent:
    """Like `test_strands_agent.py`'s own `_FakeAgent`, but also captures
    the exact prompt text sent - needed to verify the repair-hint bug fix
    (spec section 21's error-feedback audit)."""

    def __init__(self, tool, calls, sink: list[str]):
        self.tool = tool
        self.calls = calls
        self.sink = sink

    def __call__(self, prompt, limits=None):
        self.sink.append(prompt)
        for kwargs in self.calls:
            self.tool(**kwargs)
        tool_metrics = {"choose_action": SimpleNamespace(call_count=len(self.calls))} if self.calls else {}
        return SimpleNamespace(
            stop_reason="tool_use" if self.calls else "end_turn",
            metrics=SimpleNamespace(cycle_count=1, tool_metrics=tool_metrics, accumulated_usage={"inputTokens": 50, "outputTokens": 20}),
        )


class _RecordingQueuedFactory:
    def __init__(self, queued_calls: list[list[dict]], sink: list[str]):
        self._queue = [list(c) for c in queued_calls]
        self.sink = sink

    def __call__(self, system_prompt, tool):
        calls = self._queue.pop(0) if self._queue else []
        return _RecordingFakeAgent(tool, calls, self.sink)


# -- regression test for the real bug found this milestone -------------------


def test_strands_repair_prompt_includes_legal_trap_values():
    """Real bug found live at the start of M6B: `StrandsAgentAdapter`'s
    own repair call never passed `legal_trap_types` to `_repair_instruction`
    at all (unlike `LLMController`'s own M5F.3B.1 fix for the identical
    "hallucinated 'stun' every time" problem) - retrospective analysis of
    M6A.2's own telemetry found Strands attempted PLACE_TRAP 99 times
    across that milestone and succeeded 0 of those times, "stun"/
    "stun_trap" the hallucinated trap_type in effectively every failure.
    Fixed to mirror Native's own call exactly."""
    prompts: list[str] = []
    factory = _RecordingQueuedFactory(
        [
            [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"}],
            [{"action": "wait", "reason": "repaired"}],
        ],
        prompts,
    )
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory)
    adapter._request_action(TRAP_SNAPSHOT.observation, serialize_observation(TRAP_SNAPSHOT.observation))
    assert len(prompts) == 2
    repair_prompt = prompts[1]
    held_types = [t for t, c in TRAP_SNAPSHOT.observation.trap_inventory.items() if c > 0]
    assert held_types, "fixture must hold at least one real trap type to make this test meaningful"
    for trap_type in held_types:
        assert trap_type in repair_prompt


def test_strands_repair_prompt_fires_even_when_initial_error_is_missing_target():
    # Mirrors LLMController's own M5F.3B.1 regression test: the model's
    # INITIAL proposal often omits trap_type entirely (missing_target, not
    # yet invalid_trap_type) - the hint must still fire on that repair.
    prompts: list[str] = []
    factory = _RecordingQueuedFactory(
        [
            [{"action": "place_trap", "target_id": "ambassador_safe", "reason": "x"}],  # trap_type omitted
            [{"action": "wait", "reason": "repaired"}],
        ],
        prompts,
    )
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory)
    adapter._request_action(TRAP_SNAPSHOT.observation, serialize_observation(TRAP_SNAPSHOT.observation))
    repair_prompt = prompts[1]
    for trap_type, count in TRAP_SNAPSHOT.observation.trap_inventory.items():
        if count > 0:
            assert trap_type in repair_prompt


def test_strands_repair_hint_does_not_fire_for_non_place_trap_repairs():
    # The hint is place_trap-specific (spec section 22: factual, never a
    # strategic nudge) - a repair for some other invalid action must not
    # spuriously mention trap types.
    prompts: list[str] = []
    factory = _RecordingQueuedFactory(
        [
            [{"action": "move", "reason": "x"}],  # missing target_id -> triggers repair
            [{"action": "wait", "reason": "repaired"}],
        ],
        prompts,
    )
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory)
    adapter._request_action(TRAP_SNAPSHOT.observation, serialize_observation(TRAP_SNAPSHOT.observation))
    repair_prompt = prompts[1]
    assert "Allowed trap types" not in repair_prompt


# -- Phase A: retrospective classification (network-free) -------------------


def _event(action=None, target=None, trap_type=None, executed_action="wait", executed_target=None, executed_trap_type=None,
           valid=True, repair_attempted=False, initial_error_type=None, repair_error_type=None, repair_action=None,
           repair_target=None, repair_trap_type=None):
    raw_response = {"action": action, "target_id": target, "trap_type": trap_type} if action else None
    repair_raw_response = {"action": repair_action, "target_id": repair_target, "trap_type": repair_trap_type} if repair_action else None
    return {
        "raw_response": raw_response,
        "repair_raw_response": repair_raw_response,
        "executed_action": {"action": executed_action, "target_id": executed_target, "trap_type": executed_trap_type},
        "valid": valid,
        "final_model_action_valid": valid,
        "repair_attempted": repair_attempted,
        "initial_error_type": initial_error_type,
        "repair_error_type": repair_error_type,
        "error_type": (repair_error_type if repair_attempted else initial_error_type),
    }


def test_classify_not_a_place_trap_attempt_returns_none():
    event = _event(action="move", target="desk_01", executed_action="move", executed_target="desk_01")
    assert classify_place_trap_event(event) is None


def test_classify_success():
    event = _event(action="place_trap", target="safe", trap_type="bomb", executed_action="place_trap", executed_target="safe", executed_trap_type="bomb", valid=True)
    assert classify_place_trap_event(event) == SUCCESS


def test_classify_missing_trap_type():
    event = _event(action="place_trap", target="safe", trap_type=None, valid=False, initial_error_type="missing_target")
    assert classify_place_trap_event(event) == MISSING_TRAP_TYPE


def test_classify_missing_target():
    event = _event(action="place_trap", target=None, trap_type="bomb", valid=False, initial_error_type="missing_target")
    assert classify_place_trap_event(event) == MISSING_TARGET


def test_classify_invalid_trap_type():
    event = _event(action="place_trap", target="safe", trap_type="stun", valid=False, initial_error_type="invalid_trap_type")
    assert classify_place_trap_event(event) == INVALID_TRAP_TYPE


def test_classify_depleted_trap_type():
    event = _event(action="place_trap", target="safe", trap_type="bomb", valid=False, initial_error_type="trap_type_not_in_inventory")
    assert classify_place_trap_event(event) == DEPLETED_TRAP_TYPE


def test_classify_invalid_target():
    event = _event(action="place_trap", target=123, trap_type="bomb", valid=False, initial_error_type="invalid_target")
    assert classify_place_trap_event(event) == INVALID_TARGET


def test_classify_malformed_tool_call():
    event = _event(action="totally_unknown_action", valid=False, initial_error_type="unknown_action")
    # not itself a place_trap attempt (action isn't place_trap) -> None
    assert classify_place_trap_event(event) is None


def test_classify_runtime_parse_failure():
    event = _event(action="place_trap", target="safe", trap_type="bomb", valid=False, initial_error_type="strands_runtime_error")
    assert classify_place_trap_event(event) == RUNTIME_PARSE_FAILURE


def test_classify_uses_repair_when_repair_still_targets_place_trap():
    event = _event(
        action="place_trap", target="safe", trap_type="stun", valid=False, repair_attempted=True,
        initial_error_type="invalid_trap_type", repair_action="place_trap", repair_target="safe", repair_trap_type="stun_trap",
        repair_error_type="invalid_trap_type",
    )
    assert classify_place_trap_event(event) == INVALID_TRAP_TYPE


def test_classify_uses_initial_when_repair_abandons_place_trap_for_a_different_action():
    # Regression test for the real bug found live while first validating
    # this classifier against M6A.2's own telemetry: a repair that
    # abandons place_trap for `move` (which then validates fine) must
    # still classify the ABANDONED initial place_trap attempt using its
    # own error_type/proposal, not the top-level (post-repair) error_type
    # (which is None here, since the accepted `move` was valid).
    event = _event(
        action="place_trap", target="safe", trap_type="stun", valid=True, repair_attempted=True,
        initial_error_type="invalid_trap_type", repair_action="move", repair_target="door_01",
        executed_action="move", executed_target="door_01",
    )
    assert classify_place_trap_event(event) == INVALID_TRAP_TYPE


def test_summarize_place_trap_attempts_matches_m6a2_corrected_totals():
    # Reproduces (in miniature) the exact real pattern found in M6A.2's
    # own telemetry: Strands attempted place_trap far more than Native but
    # succeeded far less - the actual, corrected numbers (not M6A.2's own
    # informal "57.5% of all decisions" framing, a different denominator)
    # are 99 attempts / 0 successes for Strands vs 11/11 for Native across
    # that milestone's 4 real episodes.
    native_events = [_event(action="place_trap", target="safe", trap_type="bomb", executed_action="place_trap", executed_target="safe", executed_trap_type="bomb", valid=True) for _ in range(3)]
    strands_events = [_event(action="place_trap", target="safe", trap_type="stun", valid=False, initial_error_type="invalid_trap_type") for _ in range(3)]
    native_summary = summarize_place_trap_attempts(native_events)
    strands_summary = summarize_place_trap_attempts(strands_events)
    assert native_summary["attempts"] == 3 and native_summary["successful"] == 3 and native_summary["failure_rate"] == 0.0
    assert strands_summary["attempts"] == 3 and strands_summary["successful"] == 0 and strands_summary["failure_rate"] == 1.0
    assert strands_summary["counts_by_category"] == {INVALID_TRAP_TYPE: 3}


# -- Phase C: mock tool-call tests across both adapters (spec section 29) ---


def _native_place_trap_trial(response):
    telemetry = TelemetryLogger()
    adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=MockModelClient(response=response),
        config=LLMControllerConfig(), telemetry=telemetry, episode_id="m6b-native",
    )
    trial = run_one_trial(adapter, TRAP_SNAPSHOT.observation, telemetry, "m6b-native", wall_timeout=10.0)
    decision_event = next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {})
    return trial, decision_event


def _strands_place_trap_trial(queued_calls):
    telemetry = TelemetryLogger()
    factory = QueuedStrandsFactory(queued_calls)
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory, telemetry=telemetry, episode_id="m6b-strands")
    trial = run_one_trial(adapter, TRAP_SNAPSHOT.observation, telemetry, "m6b-strands", wall_timeout=10.0)
    decision_event = next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {})
    return trial, decision_event


def _held_trap_type():
    return next(t for t, c in TRAP_SNAPSHOT.observation.trap_inventory.items() if c > 0)


def test_phase_c_correct_place_trap_succeeds_on_both_runtimes():
    held = _held_trap_type()
    good = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held, "reason": "x"}
    _, native_event = _native_place_trap_trial(good)
    _, strands_event = _strands_place_trap_trial([[good]])
    assert classify_place_trap_event(native_event) == SUCCESS
    assert classify_place_trap_event(strands_event) == SUCCESS


def test_phase_c_missing_trap_type_then_successful_repair_on_both_runtimes():
    held = _held_trap_type()
    missing_type = {"action": "place_trap", "target_id": "ambassador_safe", "reason": "x"}
    repaired = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held, "reason": "fixed"}
    _, native_event = _native_place_trap_trial([missing_type, repaired])
    _, strands_event = _strands_place_trap_trial([[missing_type], [repaired]])
    assert classify_place_trap_event(native_event) == SUCCESS
    assert classify_place_trap_event(strands_event) == SUCCESS  # only true post-fix - this is the whole point


def test_phase_c_wrong_trap_type_then_repair_gives_up():
    bad = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"}
    gave_up = {"action": "wait", "reason": "giving up"}
    _, native_event = _native_place_trap_trial([bad, gave_up])
    _, strands_event = _strands_place_trap_trial([[bad], [gave_up]])
    assert classify_place_trap_event(native_event) == INVALID_TRAP_TYPE
    assert classify_place_trap_event(strands_event) == INVALID_TRAP_TYPE


def test_phase_c_depleted_trap_type():
    import dataclasses

    # The shared TRAP_SNAPSHOT fixture happens to hold all 4 real trap
    # types (spec section 24 recommends "one type depleted" as a distinct
    # variation) - override just the inventory here rather than adding a
    # whole second snapshot for one field.
    depleted_type = constants.TRAP_TYPES[0]
    thin_inventory = dict(TRAP_SNAPSHOT.observation.trap_inventory)
    thin_inventory[depleted_type] = 0
    thin_observation = dataclasses.replace(TRAP_SNAPSHOT.observation, trap_inventory=thin_inventory)

    bad = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": depleted_type, "reason": "x"}
    gave_up = {"action": "wait", "reason": "giving up"}

    telemetry = TelemetryLogger()
    native_adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=MockModelClient(response=[bad, gave_up]),
        config=LLMControllerConfig(), telemetry=telemetry, episode_id="m6b-native-depleted",
    )
    run_one_trial(native_adapter, thin_observation, telemetry, "m6b-native-depleted", wall_timeout=10.0)
    native_event = next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {})

    telemetry2 = TelemetryLogger()
    strands_adapter = StrandsAgentAdapter(
        constants.WHITE_SPY_ID, agent_factory=QueuedStrandsFactory([[bad], [gave_up]]),
        telemetry=telemetry2, episode_id="m6b-strands-depleted",
    )
    run_one_trial(strands_adapter, thin_observation, telemetry2, "m6b-strands-depleted", wall_timeout=10.0)
    strands_event = next((e for e in telemetry2.events if e.get("event_type") == "llm_decision"), {})

    assert classify_place_trap_event(native_event) == DEPLETED_TRAP_TYPE
    assert classify_place_trap_event(strands_event) == DEPLETED_TRAP_TYPE


def test_phase_c_missing_target():
    held = _held_trap_type()
    bad = {"action": "place_trap", "trap_type": held, "reason": "x"}
    gave_up = {"action": "wait", "reason": "giving up"}
    _, native_event = _native_place_trap_trial([bad, gave_up])
    _, strands_event = _strands_place_trap_trial([[bad], [gave_up]])
    assert classify_place_trap_event(native_event) == MISSING_TARGET
    assert classify_place_trap_event(strands_event) == MISSING_TARGET


# -- trap-opportunity snapshot integrity (spec sections 24-26) --------------


def test_trap_snapshot_set_has_all_four_categories():
    snaps = build_trap_snapshot_set()
    categories = {s.category for s in snaps}
    assert categories == {MULTIPLE_TYPES_REMAINING, SINGLE_TYPE_REMAINING, ONE_TYPE_DEPLETED, OPPONENT_NEARBY}


def test_every_trap_snapshot_passes_integrity_check():
    for snapshot in build_trap_snapshot_set():
        violations = assert_snapshot_integrity(snapshot)
        assert violations == [], f"{snapshot.category}: {violations}"


def test_every_trap_snapshot_actually_has_place_trap_legal():
    for snapshot in build_trap_snapshot_set():
        assert any(a.get("action") == "place_trap" for a in snapshot.legal_actions), snapshot.category


def test_phase_c_extra_field_does_not_break_validation():
    # WRONG_ARGUMENT_NAME/extra-field robustness: an extra, unexpected
    # field must not itself cause a different failure category.
    held = _held_trap_type()
    extra = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held, "reason": "x", "confidence": 0.9}
    _, native_event = _native_place_trap_trial(extra)
    _, strands_event = _strands_place_trap_trial([[{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held, "reason": "x"}]])
    assert classify_place_trap_event(native_event) == SUCCESS
    assert classify_place_trap_event(strands_event) == SUCCESS
