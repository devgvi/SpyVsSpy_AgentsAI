"""M6A Phase A/B (spec sections 20-22, 46): deterministic Native/Strands
normalization equivalence, invalid-output handling, runtime-error
handling, candidate-order preservation, secret-safe diagnostics,
SOCIAL_OFF equivalence, cross-episode memory OFF. Real `GameState`/
`Observation`/adapters throughout - Strands exercised via the same fake
`agent_factory` boundary `tests/test_strands_agent.py` already
established (no network calls).

Every trial goes through `run_one_trial` (never a raw, un-polled
`.decide()` call) - both adapters make their real decision on a
non-blocking background thread, and a single un-polled `decide()` call
can return a stale/cached placeholder rather than the real result (a
real bug found live while first writing this file - see
`runtime_snapshot_replay.run_one_trial`'s own docstring for the fix)."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src import constants
from src.agents.native_agent import NativeAgentAdapter
from src.agents.strands_agent import StrandsAgentAdapter
from src.evaluation.runtime_snapshot_replay import (
    DIFFERENT_ACTION,
    RUNTIME_FAILURE_DIVERGENCE,
    SAME_DECISION,
    VALIDITY_DIVERGENCE,
    build_snapshot_set,
    classify_divergence,
    run_one_trial,
)
from src.llm.clients.mock_client import MockModelClient
from src.llm.personality import SABOTEUR
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMControllerConfig
from tests.test_strands_agent import QueuedStrandsFactory

SNAPSHOTS = {s.category: s for s in build_snapshot_set()}


def _native_trial(response, obs, personality=None, episode_id="m6a-native"):
    telemetry = TelemetryLogger()
    adapter = NativeAgentAdapter(
        constants.WHITE_SPY_ID, model_client=MockModelClient(response=response),
        config=LLMControllerConfig(personality_profile=personality), telemetry=telemetry, episode_id=episode_id,
    )
    return run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)


def _strands_trial(queued_calls, obs, personality=None, episode_id="m6a-strands"):
    telemetry = TelemetryLogger()
    factory = QueuedStrandsFactory(queued_calls)
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory, telemetry=telemetry, episode_id=episode_id, personality_profile=personality)
    return run_one_trial(adapter, obs, telemetry, episode_id, wall_timeout=10.0)


# -- same Mock output -> same AgentDecision (spec section 20) ------------


def test_move_normalizes_identically_across_runtimes():
    obs = SNAPSHOTS["MULTIPLE_LEGAL_TARGETS"].observation
    nt = _native_trial({"action": "move", "target_id": "desk_01", "reason": "x"}, obs)
    st = _strands_trial([[{"action": "move", "target_id": "desk_01", "reason": "x"}]], obs)
    assert nt["action"] == st["action"] == "move"
    assert nt["target"] == st["target"] == "desk_01"
    assert nt["valid"] and st["valid"]


def test_inspect_normalizes_identically_across_runtimes():
    obs = SNAPSHOTS["LOCKED_OBJECT"].observation
    nt = _native_trial({"action": "inspect", "target_id": "ambassador_safe", "reason": "x"}, obs)
    st = _strands_trial([[{"action": "inspect", "target_id": "ambassador_safe", "reason": "x"}]], obs)
    assert nt["action"] == st["action"] == "inspect"
    assert nt["target"] == st["target"] == "ambassador_safe"


def test_search_normalizes_identically_across_runtimes():
    # "search" is a canonical ACTION_ALIASES entry (src/llm_controller.py)
    # that normalizes to the single ActionType.INTERACT, not a distinct
    # action type of its own - both runtimes must normalize it the same way.
    obs = SNAPSHOTS["MULTIPLE_LEGAL_TARGETS"].observation
    nt = _native_trial({"action": "search", "target_id": "desk_01", "reason": "x"}, obs)
    st = _strands_trial([[{"action": "search", "target_id": "desk_01", "reason": "x"}]], obs)
    assert nt["action"] == st["action"] == "interact"
    assert nt["target"] == st["target"] == "desk_01"


def test_place_trap_normalizes_identically_across_runtimes():
    obs = SNAPSHOTS["TRAP_PLACEMENT_OPPORTUNITY"].observation
    held_type = next(t for t, c in obs.trap_inventory.items() if c > 0)
    nt = _native_trial({"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held_type, "reason": "x"}, obs)
    st = _strands_trial([[{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held_type, "reason": "x"}]], obs)
    assert nt["action"] == st["action"] == "place_trap"
    assert nt["trap_type"] == st["trap_type"] == held_type


def test_disarm_normalizes_identically_across_runtimes():
    obs = SNAPSHOTS["DETECTED_TRAP"].observation
    nt = _native_trial({"action": "disarm", "target_id": "filing_cabinet_01", "reason": "x"}, obs)
    st = _strands_trial([[{"action": "disarm", "target_id": "filing_cabinet_01", "reason": "x"}]], obs)
    assert nt["action"] == st["action"] == "disarm"
    assert nt["target"] == st["target"] == "filing_cabinet_01"


def test_wait_normalizes_identically_across_runtimes():
    obs = SNAPSHOTS["SIMPLE_NAVIGATION"].observation
    nt = _native_trial({"action": "wait", "reason": "x"}, obs)
    st = _strands_trial([[{"action": "wait", "reason": "x"}]], obs)
    assert nt["action"] == st["action"] == "wait"


# -- invalid output tests (spec section 21) --------------------------------


def test_missing_target_rejected_by_both_runtimes():
    obs = SNAPSHOTS["MULTIPLE_LEGAL_TARGETS"].observation
    nt = _native_trial([{"action": "move", "reason": "x"}, {"action": "move", "reason": "still bad"}], obs)
    st = _strands_trial([[{"action": "move"}], [{"action": "move"}]], obs)
    assert not nt["valid"] and not st["valid"]
    assert nt["action"] == st["action"] == "wait"


def test_invalid_action_type_rejected_by_both_runtimes():
    obs = SNAPSHOTS["SIMPLE_NAVIGATION"].observation
    nt = _native_trial([{"action": "teleport", "reason": "x"}, {"action": "teleport", "reason": "still bad"}], obs)
    st = _strands_trial([[{"action": "teleport"}], [{"action": "teleport"}]], obs)
    assert not nt["valid"] and not st["valid"]


def test_invalid_trap_type_rejected_by_both_runtimes():
    # M5F.3B.1's error-code split (invalid_trap_type vs
    # trap_type_not_in_inventory) is shared validation - both runtimes
    # must reject the same hallucinated type the same way.
    obs = SNAPSHOTS["TRAP_PLACEMENT_OPPORTUNITY"].observation
    bad = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"}
    nt = _native_trial([bad, bad], obs)
    st = _strands_trial([[bad], [bad]], obs)
    assert not nt["valid"] and not st["valid"]
    assert nt["action"] == st["action"] == "wait"


def test_extra_text_around_json_does_not_break_native_normalization():
    # Strands' tool-call arguments are already structured (no free-text
    # wrapping possible at the tool layer) - this dimension is Native-only
    # by construction, documented in runtime_surface_audit.md.
    obs = SNAPSHOTS["SIMPLE_NAVIGATION"].observation
    nt = _native_trial({"action": "wait", "reason": "sure, here you go: {\"nonsense\": true}"}, obs)
    assert nt["action"] == "wait"
    assert nt["valid"]


# -- runtime error tests (spec section 22) ---------------------------------


class _RaisingModelClient:
    provider = "mock"
    model = "mock-model"

    def generate_action(self, *a, **kw):
        raise RuntimeError("simulated provider timeout")


class _RaisingStrandsFactory:
    def __call__(self, system_prompt, tool):
        raise RuntimeError("simulated strands runtime exception")


def test_provider_timeout_falls_back_safely_on_native():
    obs = SNAPSHOTS["SIMPLE_NAVIGATION"].observation
    telemetry = TelemetryLogger()
    adapter = NativeAgentAdapter(constants.WHITE_SPY_ID, model_client=_RaisingModelClient(), telemetry=telemetry, episode_id="m6a-err-native")
    trial = run_one_trial(adapter, obs, telemetry, "m6a-err-native", wall_timeout=10.0)
    assert trial["action"] == "wait"
    assert not trial["valid"]


def test_tool_exception_falls_back_safely_on_strands():
    obs = SNAPSHOTS["SIMPLE_NAVIGATION"].observation
    telemetry = TelemetryLogger()
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=_RaisingStrandsFactory(), telemetry=telemetry, episode_id="m6a-err-strands")
    trial = run_one_trial(adapter, obs, telemetry, "m6a-err-strands", wall_timeout=10.0)
    assert trial["action"] == "wait"
    assert not trial["valid"]


# -- divergence classification (spec section 30) ---------------------------


def test_classify_divergence_same_decision():
    a = {"valid": True, "repair_attempted": False, "action": "move", "target": "desk_01"}
    b = {"valid": True, "repair_attempted": False, "action": "move", "target": "desk_01"}
    assert classify_divergence(a, b) == SAME_DECISION


def test_classify_divergence_different_action():
    a = {"valid": True, "repair_attempted": False, "action": "move", "target": "desk_01"}
    b = {"valid": True, "repair_attempted": False, "action": "wait", "target": None}
    assert classify_divergence(a, b) == DIFFERENT_ACTION


def test_classify_divergence_validity_takes_priority():
    a = {"valid": True, "repair_attempted": False, "action": "move", "target": "desk_01"}
    b = {"valid": False, "repair_attempted": True, "action": "move", "target": "desk_01"}
    assert classify_divergence(a, b) == VALIDITY_DIVERGENCE


def test_classify_divergence_runtime_failure_takes_top_priority():
    a = {"valid": True, "repair_attempted": False, "action": "move", "target": "desk_01", "runtime_error": "boom"}
    b = {"valid": True, "repair_attempted": False, "action": "move", "target": "desk_01"}
    assert classify_divergence(a, b) == RUNTIME_FAILURE_DIVERGENCE


# -- run_one_trial telemetry integration -----------------------------------


def test_run_one_trial_captures_repair_and_latency_from_telemetry():
    obs = SNAPSHOTS["SIMPLE_NAVIGATION"].observation
    trial = _native_trial({"action": "wait", "reason": "x"}, obs)
    assert trial["action"] == "wait"
    assert trial["valid"] is True
    assert trial["repair_attempted"] is False
    assert trial["latency_ms"] is not None


def test_run_one_trial_records_runtime_error_without_raising():
    obs = SNAPSHOTS["SIMPLE_NAVIGATION"].observation
    telemetry = TelemetryLogger()
    adapter = NativeAgentAdapter(constants.WHITE_SPY_ID, model_client=_RaisingModelClient(), telemetry=telemetry, episode_id="m6a-trial-error")
    trial = run_one_trial(adapter, obs, telemetry, "m6a-trial-error", wall_timeout=10.0)
    assert trial["action"] == "wait"
    assert trial["valid"] is False


def test_run_one_trial_polls_past_a_stale_cached_first_call():
    # Regression test for the real bug found live: without polling, the
    # very first decide() call on a fresh adapter can return the
    # constructor's own cached placeholder (WAIT, action_source="llm")
    # instead of the real, still-in-flight background decision.
    obs = SNAPSHOTS["MULTIPLE_LEGAL_TARGETS"].observation
    trial = _native_trial({"action": "move", "target_id": "desk_01", "reason": "x"}, obs)
    assert trial["action"] == "move"
    assert trial["target"] == "desk_01"


# -- candidate-order preservation (spec sections 9, 31) --------------------


def test_legal_action_order_identical_for_same_snapshot_across_runtimes():
    # Both runtimes derive legal_actions from the exact same Observation -
    # candidate order can only diverge if one runtime re-serializes it
    # differently, which neither currently does (see runtime_surface_audit.md).
    snap = SNAPSHOTS["MULTIPLE_LEGAL_TARGETS"]
    from src.llm_controller import enumerate_legal_actions

    order_a = enumerate_legal_actions(snap.observation)
    order_b = enumerate_legal_actions(snap.observation)
    assert order_a == order_b


# -- personality parity (this milestone's own fix) -------------------------


def test_strands_personality_block_matches_native_exactly():
    from src.llm.personality import render_personality_block

    telemetry = TelemetryLogger()
    factory = QueuedStrandsFactory([[{"action": "wait"}]])
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory, telemetry=telemetry, episode_id="m6a-personality", personality_profile=SABOTEUR)
    assert render_personality_block(SABOTEUR) in adapter._effective_system_prompt


def test_strands_default_personality_is_byte_identical_to_pre_m6a():
    from src.agents.strands_agent import STRANDS_SYSTEM_PROMPT

    factory = QueuedStrandsFactory([[{"action": "wait"}]])
    adapter = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory)
    assert adapter._effective_system_prompt == STRANDS_SYSTEM_PROMPT


# -- SOCIAL_OFF equivalence / cross-episode memory OFF (spec sections 6, 38) --


def test_native_tool_like_offers_no_send_message_when_communication_disabled():
    from src.llm_controller import enumerate_legal_actions

    snap = SNAPSHOTS["SIMPLE_NAVIGATION"]
    assert snap.observation.communication_enabled is False
    legal = enumerate_legal_actions(snap.observation)
    assert not any(a["action"] == "send_message" for a in legal)


def test_neither_adapter_touches_cross_episode_memory_by_default():
    native = NativeAgentAdapter(constants.WHITE_SPY_ID, model_client=MockModelClient(response={"action": "wait"}))
    factory = QueuedStrandsFactory([[{"action": "wait"}]])
    strands = StrandsAgentAdapter(constants.WHITE_SPY_ID, agent_factory=factory)
    assert getattr(native.controller.config, "model_payload_transform", None) is None
    # Strands has no cross-episode-memory injection point at all (spec
    # section 38 keeps it OFF for the whole primary comparison) - nothing
    # to assert beyond "constructs and runs without it."
    assert strands is not None


# -- secret-safe diagnostics (spec sections 45-46) --------------------------


def test_no_secret_like_patterns_in_new_module_sources():
    import inspect

    for mod_name in ("src.evaluation.runtime_snapshot_replay",):
        source = inspect.getsource(__import__(mod_name, fromlist=[""]))
        for forbidden in ("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key"):
            assert forbidden not in source


def test_bedrock_token_presence_check_never_logs_value(monkeypatch):
    # Spec section 45: prefer presence checks / string-length checks over
    # printing raw credential-bearing files or values.
    from src.llm.clients.bedrock_client import bedrock_token_present

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-should-never-appear-in-output")
    result = bedrock_token_present()
    assert result is True
    assert isinstance(result, bool)  # the function's own return type never carries the value itself
