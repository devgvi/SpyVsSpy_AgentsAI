"""M5E.1: network-free tests for the long-horizon stability diagnostics -
error taxonomy, failure-streak/stop-reason computation, the new
decision/provider-call telemetry fields, and the synthetic S1/S2/S3
harness mechanics (driven by MockModelClient/a fake Strands agent
factory - never real Ollama; live results are reported separately in
CHANGELOG.md per spec section 48)."""

from __future__ import annotations

from src.evaluation.failure_streak import (
    STOP_DECISION_LIMIT,
    STOP_MISSION_DEADLINE,
    STOP_NORMAL_COMPLETION,
    STOP_PROVIDER_TIMEOUT_BUDGET,
    STOP_RUNTIME_FAILURE_BUDGET,
    STOP_WALL_TIME_BUDGET_EXCEEDED,
    classify_stop_reason,
    compute_failure_streak,
)
from src.evaluation.synthetic_stability import (
    PATTERN_S1_CONSTANT,
    PATTERN_S2_GROWING,
    PATTERN_S3_ACTIVE_LIKE,
    run_synthetic_pattern,
)
from src.llm.agent_memory import AgentMemory
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import (
    LLMController,
    LLMControllerConfig,
    classify_provider_call_error,
    normalize_provider_error_message,
)
from src.observation import build_observation
from src import constants
from src.items import load_items
from src.level import load_level
from src.state import GameState


# -- error taxonomy ----------------------------------------------------

def test_classify_provider_call_error_categories():
    assert classify_provider_call_error("provider_error", "Read timed out") == "provider_timeout"
    assert classify_provider_call_error("provider_error", "Connection refused") == "provider_transport_error"
    assert classify_provider_call_error("provider_error", "HTTP 500 status") == "provider_http_error"
    assert classify_provider_call_error("empty_response", "response contained no fields") == "provider_invalid_response"
    assert classify_provider_call_error("strands_runtime_error", "boom") == "runtime_exception"


def test_classify_provider_call_error_never_flags_game_rule_validation_as_a_call_failure():
    # A well-formed, successfully-returned response that just picked an
    # illegal target is not a provider/runtime failure (spec section 10
    # keeps validation_failure disjoint) - the call itself succeeded.
    for error_type in ("missing_target", "illegal_target", "unknown_target", "out_of_range"):
        assert classify_provider_call_error(error_type, "some message") is None


def test_classify_provider_call_error_none_when_no_error():
    assert classify_provider_call_error(None, None) is None


def test_normalize_provider_error_message_redacts_secret_like_content():
    normalized = normalize_provider_error_message("failed: api_key=sk-abc123 rejected")
    assert "sk-abc123" not in normalized
    assert "[redacted]" in normalized


def test_normalize_provider_error_message_truncates():
    long_message = "x" * 1000
    assert len(normalize_provider_error_message(long_message)) == 300


# -- stop-reason classification -----------------------------------------

def test_classify_stop_reason_normal_completion():
    assert classify_stop_reason("COMPLETED", None, True) == STOP_NORMAL_COMPLETION


def test_classify_stop_reason_mission_deadline_on_unsuccessful_completion():
    assert classify_stop_reason("COMPLETED", None, False) == STOP_MISSION_DEADLINE


def test_classify_stop_reason_decision_limit():
    assert classify_stop_reason("DECISION_LIMIT_REACHED", None, False) == STOP_DECISION_LIMIT


def test_classify_stop_reason_wall_time_budget():
    assert classify_stop_reason("TIMED_OUT", None, False) == STOP_WALL_TIME_BUDGET_EXCEEDED


def test_classify_stop_reason_runtime_failure_budget_vs_provider_timeout_budget():
    assert classify_stop_reason("ABORTED", "max_runtime_failures_exceeded", False) == STOP_RUNTIME_FAILURE_BUDGET
    assert (
        classify_stop_reason("ABORTED", "max_runtime_failures_exceeded", False, provider_timeout_fraction=0.9)
        == STOP_PROVIDER_TIMEOUT_BUDGET
    )
    assert (
        classify_stop_reason("ABORTED", "max_runtime_failures_exceeded", False, provider_timeout_fraction=0.2)
        == STOP_RUNTIME_FAILURE_BUDGET
    )


# -- failure-streak computation -----------------------------------------

def _decision(number: int, valid: bool) -> dict:
    return {"decision_number": number, "final_model_action_valid": valid}


def test_compute_failure_streak_no_failures():
    report = compute_failure_streak([_decision(i, True) for i in range(1, 6)])
    assert report.first_failure_decision is None
    assert report.max_consecutive_failure_streak == 0
    assert report.recovered_after_failure is False
    assert report.decision_count_before_abort == 5


def test_compute_failure_streak_persistent_failure_from_middecision_matches_m5e_pattern():
    # The real M5E pattern: clean start, then a persistent streak to the
    # end - exactly what CHANGELOG.md's M5E entry described for A/B.
    decisions = [_decision(i, True) for i in range(1, 7)] + [_decision(i, False) for i in range(7, 16)]
    report = compute_failure_streak(decisions)
    assert report.first_failure_decision == 7
    assert report.max_consecutive_failure_streak == 9
    assert report.consecutive_failure_streak_length == 9  # still failing at the end
    assert report.recovered_after_failure is False
    assert report.decision_count_before_abort == 15


def test_compute_failure_streak_recovers_after_a_transient_failure():
    decisions = [_decision(1, True), _decision(2, False), _decision(3, False), _decision(4, True), _decision(5, True)]
    report = compute_failure_streak(decisions)
    assert report.first_failure_decision == 2
    assert report.max_consecutive_failure_streak == 2
    assert report.consecutive_failure_streak_length == 0  # not failing at the end
    assert report.recovered_after_failure is True


def test_compute_failure_streak_empty():
    report = compute_failure_streak([])
    assert report.decision_count_before_abort == 0
    assert report.first_failure_decision is None


# -- decision/provider-call telemetry fields -----------------------------

def _real_observation(actor_id: str = constants.WHITE_SPY_ID):
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=777)
    return build_observation(state, actor_id)


def test_llm_decision_telemetry_carries_new_m5e1_fields():
    telemetry = TelemetryLogger()
    client = MockModelClient(response={"action": "wait"})
    controller = LLMController(
        constants.WHITE_SPY_ID,
        model_client=client,
        config=LLMControllerConfig(condition="passive"),
        telemetry=telemetry,
        episode_id="ep-stability-1",
        provider_name="mock",
        model_name="mock-model",
    )
    observation = _real_observation()
    controller.get_action(observation)

    decisions = [e for e in telemetry.events if e["event_type"] == "llm_decision"]
    assert len(decisions) == 1
    event = decisions[0]
    assert event["condition"] == "passive"
    assert event["decision_number"] == 1
    assert isinstance(event["serialized_legal_actions_chars"], int) and event["serialized_legal_actions_chars"] > 0
    assert event["total_tokens"] == 64 + 8  # MockModelClient's fixed input/output tokens


def test_provider_call_telemetry_emitted_once_per_real_call_initial_and_repair():
    telemetry = TelemetryLogger()
    # First response is an illegal action name -> triggers exactly one
    # repair call; second response is a legitimate wait.
    client = MockModelClient(response=[{"action": "not_a_real_action"}, {"action": "wait"}])
    controller = LLMController(
        constants.WHITE_SPY_ID,
        model_client=client,
        config=LLMControllerConfig(condition="active"),
        telemetry=telemetry,
        episode_id="ep-stability-2",
        provider_name="mock",
        model_name="mock-model",
    )
    observation = _real_observation()
    controller.get_action(observation)

    calls = [e for e in telemetry.events if e["event_type"] == "provider_call"]
    assert [c["call_type"] for c in calls] == ["initial", "repair"]
    assert [c["call_index"] for c in calls] == [1, 2]
    assert all(c["condition"] == "active" for c in calls)
    assert all(c["decision_id"] for c in calls)


def test_provider_call_telemetry_marks_timeout_on_provider_exception():
    telemetry = TelemetryLogger()

    class _ExplodingClient:
        provider = "mock"
        model = "mock-model"

        def generate_action(self, *args, **kwargs):
            raise TimeoutError("Read timed out")

    controller = LLMController(
        constants.WHITE_SPY_ID,
        model_client=_ExplodingClient(),
        telemetry=telemetry,
        episode_id="ep-stability-3",
        provider_name="mock",
        model_name="mock-model",
    )
    controller.get_action(_real_observation())

    calls = [e for e in telemetry.events if e["event_type"] == "provider_call"]
    assert len(calls) == 1
    assert calls[0]["success"] is False
    assert calls[0]["timeout"] is True
    assert calls[0]["provider_error_type"] == "provider_timeout"


# -- memory growth telemetry ---------------------------------------------

def test_agent_memory_telemetry_includes_entry_count_and_opponent_presence():
    memory = AgentMemory(constants.WHITE_SPY_ID)
    telemetry_dict = memory.to_telemetry_dict()
    assert telemetry_dict["memory_entries"] == 0
    assert telemetry_dict["last_seen_opponent_present"] is False

    memory.record_interaction(game_time=1.0, action="search", target_id="desk", result="empty")
    memory.record_event(game_time=1.0, event_type="X")
    memory.record_opponent_sighting(game_time=1.0, room_id="corridor")
    telemetry_dict = memory.to_telemetry_dict()
    assert telemetry_dict["memory_entries"] == 2
    assert telemetry_dict["last_seen_opponent_present"] is True


# -- synthetic harness mechanics (network-free) --------------------------

def _mock_client(responses):
    return MockModelClient(response=list(responses))


def test_synthetic_s1_constant_runs_all_decisions_with_mock_client():
    result = run_synthetic_pattern(
        pattern=PATTERN_S1_CONSTANT,
        runtime="native_tool_like",
        n_decisions=5,
        seed=901,
        model_client=_mock_client([{"action": "wait"}] * 10),
    )
    assert len(result.decisions) == 5
    assert all(d["synthetic_pattern"] == PATTERN_S1_CONSTANT for d in result.decisions)


def test_synthetic_s1_survives_repeated_move_decisions_without_stalling():
    # Regression for the ActionExecutor-reset fix this module documents:
    # without resetting the executor between synthetic decisions, a MOVE
    # decision would leave the multi-tick lifecycle "in progress" against
    # an engine this harness never advances, and every later call would
    # silently stall instead of requesting a new decision.
    result = run_synthetic_pattern(
        pattern=PATTERN_S1_CONSTANT,
        runtime="native_tool_like",
        n_decisions=4,
        seed=901,
        model_client=_mock_client([{"action": "move", "target_id": "exit_door"}] * 10),
        per_decision_wall_timeout=10.0,
    )
    assert len(result.decisions) == 4


def test_synthetic_s2_growing_memory_increases_monotonically():
    result = run_synthetic_pattern(
        pattern=PATTERN_S2_GROWING,
        runtime="native_tool_like",
        n_decisions=6,
        seed=902,
        model_client=_mock_client([{"action": "wait"}] * 10),
    )
    memory_sizes = [d["memory_size_chars"] for d in result.decisions]
    assert memory_sizes == sorted(memory_sizes)
    assert memory_sizes[-1] > memory_sizes[0]


def test_synthetic_s1_memory_grows_far_less_than_s2():
    # S1 still records ordinary per-decision bookkeeping (recent_decisions
    # etc - real, legitimate LLMController behavior, unrelated to this
    # harness), so it isn't perfectly flat - but S2's *deliberately*
    # injected interactions/failures/events/rooms must grow its memory
    # payload much faster over the same number of decisions.
    s1 = run_synthetic_pattern(
        pattern=PATTERN_S1_CONSTANT,
        runtime="native_tool_like",
        n_decisions=5,
        seed=903,
        model_client=_mock_client([{"action": "wait"}] * 10),
    )
    s2 = run_synthetic_pattern(
        pattern=PATTERN_S2_GROWING,
        runtime="native_tool_like",
        n_decisions=5,
        seed=905,
        model_client=_mock_client([{"action": "wait"}] * 10),
    )
    s1_growth = s1.decisions[-1]["memory_size_chars"] - s1.decisions[0]["memory_size_chars"]
    s2_growth = s2.decisions[-1]["memory_size_chars"] - s2.decisions[0]["memory_size_chars"]
    assert s2_growth > s1_growth * 2


def test_synthetic_s3_active_like_exposes_detected_trap_and_opponent():
    result = run_synthetic_pattern(
        pattern=PATTERN_S3_ACTIVE_LIKE,
        runtime="native_tool_like",
        n_decisions=1,
        seed=904,
        model_client=_mock_client([{"action": "wait"}] * 5),
    )
    assert result.decisions[0]["detected_trap_object_id"] is not None
    observation = result.decisions[0]["observation"]
    assert observation["opponent"]["visible"] is True
    legal_actions = result.decisions[0]["legal_actions"]
    assert any(a["action"] == "disarm" for a in legal_actions)


def test_synthetic_harness_rejects_unknown_pattern():
    import pytest

    with pytest.raises(ValueError):
        run_synthetic_pattern(pattern="not_a_pattern", runtime="native_tool_like", n_decisions=1)


# -- synthetic harness mechanics, Strands runtime (network-free fake) ----


class _FakeStrandsAgent:
    """Same test-double shape test_strands_agent.py's `_FakeAgent` uses:
    invokes the real `@strands.tool` object it's handed (genuine
    schema/capture behavior), no network."""

    def __init__(self, tool):
        self.tool = tool

    def __call__(self, prompt, limits=None):
        from types import SimpleNamespace

        self.tool(action="wait")
        return SimpleNamespace(
            stop_reason="tool_use",
            metrics=SimpleNamespace(
                cycle_count=1,
                tool_metrics={"choose_action": SimpleNamespace(call_count=1)},
                accumulated_usage={"inputTokens": 50, "outputTokens": 20},
            ),
        )


def _fake_strands_factory(system_prompt, tool):
    return _FakeStrandsAgent(tool)


def test_synthetic_s1_runs_with_strands_runtime_via_fake_agent_factory():
    result = run_synthetic_pattern(
        pattern=PATTERN_S1_CONSTANT,
        runtime="strands",
        n_decisions=3,
        seed=906,
        agent_factory=_fake_strands_factory,
    )
    assert len(result.decisions) == 3
    assert all(d["final_model_action_valid"] for d in result.decisions)


# -- Ollama health snapshot (network-free, mocked urlopen) ---------------


def test_snapshot_ollama_health_handles_unreachable_provider(monkeypatch):
    import urllib.error

    from src.evaluation.ollama_health import snapshot_ollama_health

    def _raise(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    snapshot = snapshot_ollama_health()
    assert snapshot["provider_process_reachable"] is False
    assert snapshot["ollama_version"] is None
    assert snapshot["model_metadata"] is None


def test_snapshot_ollama_health_parses_reachable_provider(monkeypatch):
    import io
    import json

    def _fake_urlopen(req, timeout=5.0):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/api/version"):
            body = json.dumps({"version": "0.9.9"}).encode()
        elif url.endswith("/api/show"):
            body = json.dumps(
                {
                    "details": {"family": "gemma4"},
                    "model_info": {"gemma4.context_length": 131072},
                    "parameters": "temperature 1",
                    "template": "a template",
                }
            ).encode()
        else:
            body = json.dumps({"models": [{"name": "gemma4:latest"}]}).encode()
        return io.BytesIO(body)

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _wrapped(req, timeout=5.0):
        return _Resp(_fake_urlopen(req, timeout).read())

    monkeypatch.setattr("urllib.request.urlopen", _wrapped)

    from src.evaluation.ollama_health import snapshot_ollama_health

    snapshot = snapshot_ollama_health(model_name="gemma4:latest")
    assert snapshot["provider_process_reachable"] is True
    assert snapshot["ollama_version"] == "0.9.9"
    assert snapshot["model_present_in_tags"] is True
    assert snapshot["model_metadata"]["context_length"] == 131072


# -- runtime lifecycle audit (spec sections 24-25, 38) --------------------
#
# Documents/verifies, rather than assumes, the actual current behavior:
# Native's OllamaClient sends exactly one fresh (system, user) message
# pair per HTTP call - no growing message history - and Strands builds a
# brand-new strands.Agent object for every single invocation. Neither
# runtime's own request/agent construction accumulates conversation state
# across decisions; whatever context growth is later observed in a live
# run comes only from the *content* (AgentMemory/legal actions), not from
# provider-side conversation accumulation - see CHANGELOG.md's M5E.1 entry
# for the full audit writeup this pair of tests backs.


def test_native_ollama_client_never_grows_its_message_list_across_calls():
    from src.llm.clients.ollama_client import OllamaClient

    seen_payloads = []

    def fake_request(payload):
        seen_payloads.append(payload)
        return {"message": {"content": '{"action": "wait"}'}, "done": True, "prompt_eval_count": 10, "eval_count": 2}

    client = OllamaClient(model_name="gemma4:latest", request_fn=fake_request)
    for i in range(5):
        client.generate_action("system prompt", f"user prompt turn {i}", {"type": "object"})

    assert len(seen_payloads) == 5
    for payload in seen_payloads:
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"


def test_strands_agent_factory_invoked_fresh_for_every_decision_never_reused():
    import time

    from src.agents.strands_agent import StrandsAgentAdapter

    factory_invocations = []

    def counting_factory(system_prompt, tool):
        factory_invocations.append(id(tool))
        return _FakeStrandsAgent(tool)

    adapter = StrandsAgentAdapter("white_spy", agent_factory=counting_factory)
    state = _build_base_state_for_lifecycle_test()

    for i in range(3):
        state.actors["white_spy"].x += 0.01
        observation = build_observation(state, "white_spy")
        target_completed = adapter.decisions_completed + 1
        while adapter.decisions_completed < target_completed:
            adapter.get_action(observation)
            time.sleep(0.01)

    # One real Strands invocation per decision (no repair needed - the
    # fake agent always returns a valid "wait") - each got its own fresh
    # agent_factory call, never a shared/reused Agent object.
    assert len(factory_invocations) == 3


def _build_base_state_for_lifecycle_test():
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=811)


# -- M5E.2: long-horizon stability aggregation, incl. watchdog_terminations --


def test_compute_long_horizon_stability_separates_watchdog_from_provider_timeouts():
    from src.evaluation.failure_streak import compute_long_horizon_stability

    decisions = [
        {"decision_number": 1, "final_model_action_valid": True},
        {"decision_number": 2, "final_model_action_valid": False},
        {"decision_number": 3, "final_model_action_valid": False},
    ]
    provider_calls = [
        {"provider_error_type": None},
        {"provider_error_type": "provider_timeout", "provider_error_message_normalized": "timed out"},
        {
            "provider_error_type": "provider_timeout",
            "provider_error_message_normalized": "timed out (adapter watchdog - underlying call abandoned, not provider-reported)",
        },
        {"provider_error_type": "runtime_exception", "provider_error_message_normalized": "boom"},
    ]
    report = compute_long_horizon_stability(decisions, provider_calls)
    assert report.evaluated_agent_decisions == 3
    assert report.valid_decisions == 1
    assert report.first_failure_decision == 2
    assert report.max_consecutive_technical_failures == 2
    assert report.provider_timeouts == 1
    assert report.watchdog_terminations == 1
    assert report.runtime_exceptions == 1


def test_compute_long_horizon_stability_healthy_baseline_is_all_zero_failures():
    from src.evaluation.failure_streak import compute_long_horizon_stability

    decisions = [{"decision_number": i, "final_model_action_valid": True} for i in range(1, 16)]
    provider_calls = [{"provider_error_type": None} for _ in range(15)]
    report = compute_long_horizon_stability(decisions, provider_calls)
    assert report.valid_decisions == 15
    assert report.first_failure_decision is None
    assert report.provider_timeouts == 0
    assert report.runtime_exceptions == 0
    assert report.watchdog_terminations == 0


# -- M5E.4: first-pass validity distinct from final (post-repair) validity --


def test_first_pass_valid_decisions_excludes_repaired_ones():
    """`valid_decisions` (final, post-repair) and `first_pass_valid_decisions`
    (needed no repair at all) must be genuinely different numbers when a
    real repair happens - a runtime comparison that reports repairs
    separately from first-pass validity (spec M5E.4 section 31) needs
    both, not the same count under two names."""
    from src.evaluation.failure_streak import compute_long_horizon_stability

    decisions = [
        {"final_model_action_valid": True, "recovery_outcome": "FIRST_PASS_VALID"},
        {"final_model_action_valid": True, "recovery_outcome": "FIRST_PASS_VALID"},
        {"final_model_action_valid": True, "recovery_outcome": "REPAIRED_VALID"},
        {"final_model_action_valid": False, "recovery_outcome": "FALLBACK"},
    ]
    report = compute_long_horizon_stability(decisions, [])
    assert report.valid_decisions == 3  # 2 first-pass + 1 repaired
    assert report.first_pass_valid_decisions == 2
    assert report.to_dict()["first_pass_valid_decisions"] == 2


def test_first_pass_valid_decisions_equals_valid_decisions_when_no_repairs_occur():
    from src.evaluation.failure_streak import compute_long_horizon_stability

    decisions = [{"final_model_action_valid": True, "recovery_outcome": "FIRST_PASS_VALID"} for _ in range(5)]
    report = compute_long_horizon_stability(decisions, [])
    assert report.valid_decisions == report.first_pass_valid_decisions == 5
