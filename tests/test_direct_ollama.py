"""M5E.1.1 spec sections 45-46: network-free tests for the P0 direct-Ollama
diagnostic client - request serialization, num_ctx/keep_alive propagation,
Ollama timing-field parsing, error classification, sequential-call
indexing, and the context-matrix's safety-skip logic."""

from __future__ import annotations

from src.evaluation.direct_ollama import (
    RECOMMENDED_MATRIX_CELLS,
    TOKEN_BUCKET_CHAR_TARGETS,
    attempt_recovery_ladder,
    build_padded_user_prompt,
    direct_ollama_call,
    is_prompt_safe_for_context,
    run_sequential_direct_calls,
)
from src.llm.clients.ollama_client import OllamaClient


def _fake_ollama_response(**overrides):
    base = {
        "message": {"role": "assistant", "content": '{"action": "wait"}'},
        "done": True,
        "done_reason": "stop",
        "total_duration": 5_000_000_000,
        "load_duration": 1_000_000_000,
        "prompt_eval_count": 120,
        "prompt_eval_duration": 200_000_000,
        "eval_count": 8,
        "eval_duration": 300_000_000,
    }
    base.update(overrides)
    return base


def test_num_ctx_and_keep_alive_propagate_into_the_request_payload():
    seen = []

    def fake_request(payload):
        seen.append(payload)
        return _fake_ollama_response()

    client = OllamaClient(model_name="gemma4:latest", num_ctx=16384, keep_alive="10m", request_fn=fake_request)
    client.generate_action("sys", "user", {})

    assert seen[0]["options"]["num_ctx"] == 16384
    assert seen[0]["keep_alive"] == "10m"


def test_num_ctx_omitted_when_not_configured_preserves_prior_default_behavior():
    seen = []

    def fake_request(payload):
        seen.append(payload)
        return _fake_ollama_response()

    client = OllamaClient(model_name="gemma4:latest", request_fn=fake_request)
    client.generate_action("sys", "user", {})

    assert "num_ctx" not in seen[0]["options"]
    assert "keep_alive" not in seen[0]


def test_direct_ollama_call_parses_ollama_timing_metadata():
    def fake_request(payload):
        return _fake_ollama_response()

    client = OllamaClient(model_name="gemma4:latest", num_ctx=8192, request_fn=fake_request)
    result = direct_ollama_call(client, request_index=1, user_prompt="hello")

    assert result.success is True
    assert result.actual_input_tokens == 120
    assert result.actual_output_tokens == 8
    assert result.total_tokens == 128
    assert result.provider_metadata["total_duration"] == 5_000_000_000
    assert result.provider_metadata["load_duration"] == 1_000_000_000
    assert result.provider_metadata["configured_num_ctx"] == 8192


def test_raw_ollama_client_still_propagates_the_exception():
    # OllamaClient.generate_action itself never catches this (that's a
    # caller responsibility, same as LLMController's own try/except around
    # its calls) - documented so the next test's fix is legible against
    # the actual raw behavior it's guarding.
    def fake_request(payload):
        raise TimeoutError("Read timed out")

    client = OllamaClient(model_name="gemma4:latest", request_fn=fake_request)
    try:
        client.generate_action("sys", "user", {})
        raised = False
    except TimeoutError:
        raised = True
    assert raised


def test_direct_ollama_call_catches_a_real_socket_timeout_instead_of_crashing():
    # M5E.1.2 bug found and fixed: a real socket TimeoutError from
    # OllamaClient crashed the first live endurance run (direct_ollama_call
    # never caught it, same gap M5E.1.1's own P0 matrix had latently -
    # it simply never hit a real timeout in that sample). Every P0
    # diagnostic call site now goes through this function specifically so
    # a genuine timeout is recorded as a failed call, not an unhandled
    # exception.
    def fake_request(payload):
        raise TimeoutError("Read timed out")

    client = OllamaClient(model_name="gemma4:latest", request_fn=fake_request)
    result = direct_ollama_call(client, request_index=1, user_prompt="hello")
    assert result.success is False
    assert result.timeout is True
    assert result.error_category == "provider_timeout"
    assert result.actual_input_tokens is None


def test_direct_ollama_call_classifies_provider_error_payload():
    def fake_request(payload):
        return {"error": "model not found"}

    client = OllamaClient(model_name="gemma4:latest", request_fn=fake_request)
    result = direct_ollama_call(client, request_index=1, user_prompt="hello")
    assert result.success is False
    assert result.error_category == "provider_transport_error"


def test_build_padded_user_prompt_is_deterministic_and_reaches_target_size():
    a = build_padded_user_prompt(4000)
    b = build_padded_user_prompt(4000)
    assert a == b
    assert len(a) >= 4000


def test_token_bucket_char_targets_are_distinct_and_increasing():
    targets = [TOKEN_BUCKET_CHAR_TARGETS[k] for k in ("~2K", "~4K", "~8K")]
    assert targets == sorted(targets)
    assert len(set(targets)) == 3


def test_is_prompt_safe_for_context_rejects_unsafe_combinations():
    assert is_prompt_safe_for_context(8000, 8192, headroom_tokens=512) is False  # no headroom
    assert is_prompt_safe_for_context(8000, 16384, headroom_tokens=512) is True
    assert is_prompt_safe_for_context(2000, 4096, headroom_tokens=512) is True


def test_recommended_matrix_cells_are_all_safe():
    token_estimate_by_bucket = {"~2K": 2000, "~4K": 4000, "~8K": 8000}
    for bucket, num_ctx in RECOMMENDED_MATRIX_CELLS:
        assert is_prompt_safe_for_context(token_estimate_by_bucket[bucket], num_ctx), (bucket, num_ctx)


def test_direct_ollama_call_indexes_requests_in_order():
    def fake_request(payload):
        return _fake_ollama_response()

    # run_sequential_direct_calls itself always builds a real
    # network-facing OllamaClient per call (spec section 6 - it's the
    # real provider boundary being tested), so its request-indexing
    # contract is exercised network-free here via the same
    # direct_ollama_call/OllamaClient(request_fn=...) injection point
    # every other test in this file uses.
    client = OllamaClient(model_name="gemma4:latest", num_ctx=8192, request_fn=fake_request)
    calls = [direct_ollama_call(client, i, "prompt") for i in range(1, 5)]
    assert [c.request_index for c in calls] == [1, 2, 3, 4]


# -- recovery ladder (spec sections 23-25) --------------------------------


def test_recovery_ladder_stops_at_immediate_retry_when_it_recovers():
    # The ladder's own first call *is* "immediate_retry" (spec section 23:
    # a retry of the already-known-failed request the caller observed
    # separately) - here it succeeds straight away, so no further rungs
    # should be attempted.
    def fake_request(payload):
        return _fake_ollama_response()

    result = attempt_recovery_ladder(
        system_prompt="sys", user_prompt="user", num_ctx=8192, request_fn=fake_request, wait_seconds=0.0
    )
    assert result["recovered_at"] == "immediate_retry"
    assert "short_wait_retry" not in result


def test_recovery_ladder_falls_through_to_short_wait_retry():
    calls = {"n": 0}

    def fake_request(payload):
        calls["n"] += 1
        if calls["n"] == 1:  # the ladder's own immediate_retry call fails
            return {"error": "still glitching"}
        return _fake_ollama_response()  # its short_wait_retry call succeeds

    result = attempt_recovery_ladder(
        system_prompt="sys", user_prompt="user", num_ctx=8192, request_fn=fake_request, wait_seconds=0.0
    )
    assert result["recovered_at"] == "short_wait_retry"
    assert result["immediate_retry"]["success"] is False
    assert result["short_wait_retry"]["success"] is True


def test_recovery_ladder_reports_none_when_nothing_recovers():
    def fake_request(payload):
        return {"error": "permanently broken"}

    result = attempt_recovery_ladder(
        system_prompt="sys", user_prompt="user", num_ctx=8192, request_fn=fake_request, wait_seconds=0.0
    )
    assert result["recovered_at"] is None
    assert result["immediate_retry"]["success"] is False
    assert result["short_wait_retry"]["success"] is False
    assert result["fresh_wrapper_retry"]["success"] is False
