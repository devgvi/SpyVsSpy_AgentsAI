"""M5E.1.2 spec sections 52-53: network-free / localhost-only tests for
the endurance harness - background resource sampler lifecycle, post-
timeout probe scheduling, recovery-ladder stage labeling (no uncontrolled
immediate retry), first-failure-window extraction, and latency-trend
aggregation. All provider calls are injected via `request_fn` - no real
network."""

from __future__ import annotations

import time

from src.evaluation.endurance import (
    BackgroundResourceSampler,
    compute_latency_degradation,
    extract_first_failure_window,
    run_post_timeout_probe_sequence,
    run_recovery_ladder_v2,
)


# -- background resource sampler -------------------------------------------


def test_background_resource_sampler_collects_multiple_samples_and_stops_cleanly():
    # M7.1: one real sample shells out to ps/vm_stat/... and took 0.06-0.24 s
    # idle and up to 0.25 s under load, so a fixed 0.25 s sleep could end
    # before the second sample (failed 2/3 under 10 busy loops, 0/10 idle).
    # Wait for the condition instead, bounded at 5 s; assertions unchanged.
    sampler = BackgroundResourceSampler("test", ollama_pid=None, interval_seconds=0.05)
    sampler.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with sampler._lock:
            if len(sampler.samples) >= 2:
                break
        time.sleep(0.02)
    samples = sampler.stop()
    assert len(samples) >= 2
    assert all(s["label"] == "test_continuous" for s in samples)
    assert all("system" in s and "model" in s for s in samples)


def test_background_resource_sampler_stop_is_idempotent_shaped():
    sampler = BackgroundResourceSampler("test", ollama_pid=None, interval_seconds=0.05)
    sampler.start()
    time.sleep(0.1)
    samples = sampler.stop()
    assert isinstance(samples, list)


# -- post-timeout probe sequence --------------------------------------------


def _ok_response():
    return {"message": {"content": '{"pong": true}'}, "done": True, "prompt_eval_count": 5, "eval_count": 2}


def test_post_timeout_probe_sequence_records_t0_snapshot_and_all_probes():
    def fake_request(payload):
        return _ok_response()

    results = run_post_timeout_probe_sequence(
        schedule_seconds=(0.0, 0.05, 0.1, 0.15), request_fn=fake_request
    )
    offsets = [r["t_offset_s"] for r in results]
    assert offsets == [0.0, 0.05, 0.1, 0.15]
    assert results[0]["probe"] is None  # T+0 is resource-only, no probe
    assert all(r["probe"]["success"] for r in results[1:])


def test_post_timeout_probe_sequence_stops_early_once_healthy_at_t30_equivalent():
    def fake_request(payload):
        return _ok_response()

    # Using the real (0.0, 5.0, 15.0, 30.0) schedule is too slow for a
    # unit test; the "stop once healthy" behavior is schedule-shape
    # agnostic, so a fast equivalent schedule exercises the same branch.
    results = run_post_timeout_probe_sequence(schedule_seconds=(0.0, 0.02, 0.04, 0.06), request_fn=fake_request)
    assert len(results) == 4  # never extends to the optional T+60-equivalent step


def test_post_timeout_probe_sequence_extends_when_still_unhealthy_at_final_scheduled_probe():
    def fake_request(payload):
        raise TimeoutError("Read timed out")

    results = run_post_timeout_probe_sequence(schedule_seconds=(0.0, 0.02, 0.04, 0.06), request_fn=fake_request)
    # Still unhealthy at the last scheduled (0.06s, standing in for T+30) ->
    # one more optional probe is appended (standing in for T+60).
    assert len(results) == 5
    assert all(r["probe"] is None or r["probe"]["success"] is False for r in results)


def test_post_timeout_probe_never_uses_the_normal_production_workload():
    # spec section 31's "immediate-retry trap": the probe must always be
    # the tiny fixed request, never a full production-shaped payload.
    seen_prompts = []

    def fake_request(payload):
        seen_prompts.append(payload["messages"][1]["content"])
        return _ok_response()

    run_post_timeout_probe_sequence(schedule_seconds=(0.0, 0.01), request_fn=fake_request)
    assert len(seen_prompts) == 1
    assert "pong" in seen_prompts[0]


# -- recovery ladder v2 ------------------------------------------------------


def test_recovery_ladder_stops_at_fresh_wrapper_stage_when_it_recovers():
    ladder = run_recovery_ladder_v2(runtime="P1_native", make_request_fn=lambda: True)
    assert ladder["recovered_at"] == "fresh_wrapper_retry"
    assert "model_reload" not in ladder


def test_recovery_ladder_falls_through_to_model_reload_and_labels_ollama_restart_as_skipped():
    def fake_request(payload):
        return _ok_response()

    calls = {"n": 0}

    def make_request_fn():
        calls["n"] += 1
        return calls["n"] >= 2  # fails the fresh-wrapper stage, succeeds after reload

    ladder = run_recovery_ladder_v2(
        runtime="P0_direct_ollama",
        make_request_fn=make_request_fn,
        request_fn=fake_request,
        unload_wait_seconds=0.0,
    )
    assert ladder["fresh_wrapper_retry"]["success"] is False
    assert ladder["model_reload"]["retry_success"] is True
    assert ladder["recovered_at"] == "model_reload"
    assert "not attempted" in ladder["ollama_restart"]


def test_recovery_ladder_reports_none_when_nothing_recovers():
    def fake_request(payload):
        return _ok_response()

    ladder = run_recovery_ladder_v2(
        runtime="P0_direct_ollama", make_request_fn=lambda: False, request_fn=fake_request, unload_wait_seconds=0.0
    )
    assert ladder["recovered_at"] is None
    assert ladder["model_reload"]["retry_success"] is False


# -- first-failure window / latency degradation -----------------------------


def _rec(success, latency_ms):
    return {"success": success, "wall_latency_ms": latency_ms}


def test_extract_first_failure_window_slices_correctly():
    records = [_rec(True, 100) for _ in range(8)] + [_rec(False, 60000)] + [_rec(True, 90) for _ in range(6)]
    window = extract_first_failure_window(records, before=5, after=5)
    assert window["first_failure_index"] == 8
    assert len(window["window"]) == 11  # 5 before + failure + 5 after
    assert window["window"][5]["success"] is False


def test_extract_first_failure_window_handles_no_failures():
    records = [_rec(True, 100) for _ in range(10)]
    window = extract_first_failure_window(records)
    assert window["first_failure_index"] is None
    assert window["window"] == []


def test_extract_first_failure_window_clamps_at_sequence_boundaries():
    records = [_rec(False, 60000)] + [_rec(True, 100) for _ in range(2)]
    window = extract_first_failure_window(records, before=5, after=5)
    assert window["first_failure_index"] == 0
    assert len(window["window"]) == 3  # nothing before index 0, only 2 after


def test_compute_latency_degradation_detects_gradual_trend():
    records = [_rec(True, 10000 + i * 5000) for i in range(10)] + [_rec(False, 60000)]
    trend = compute_latency_degradation(records)
    assert trend["median_latency_last_5_before_failure_ms"] > trend["median_latency_first_5_ms"]
    assert trend["sudden_onset"] is False


def test_compute_latency_degradation_detects_sudden_onset():
    records = [_rec(True, 15000) for _ in range(10)] + [_rec(False, 60000)]
    trend = compute_latency_degradation(records)
    assert trend["sudden_onset"] is True


def test_compute_latency_degradation_handles_no_successful_calls():
    records = [_rec(False, 60000)]
    trend = compute_latency_degradation(records)
    assert trend["median_latency_first_5_ms"] is None
    assert trend["sudden_onset"] is None


# -- M5E.1.2 bug fix regression: run_p0_endurance survives a real exception --


def test_run_p0_endurance_survives_a_socket_timeout_without_crashing(monkeypatch):
    """The exact crash M5E.1.2's first live run hit: OllamaClient raising
    a real TimeoutError must not propagate out of run_p0_endurance - it
    must be recorded as a failed EnduranceCallRecord and the sequence
    must continue."""
    import src.evaluation.endurance as endurance_module

    call_count = {"n": 0}

    class _FakeOllamaClient:
        def __init__(self, *a, num_ctx=None, keep_alive=None, **kw):
            self.num_ctx = num_ctx
            self.keep_alive = keep_alive

        def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise TimeoutError("Read timed out")
            from src.llm.clients.base import ModelResponse

            return ModelResponse(
                payload={"action": "wait"}, raw_text='{"action": "wait"}', provider="ollama", model="gemma4:latest",
                latency_ms=10.0, input_tokens=100, output_tokens=5,
            )

    monkeypatch.setattr(endurance_module, "OllamaClient", _FakeOllamaClient)

    records = endurance_module.run_p0_endurance(n_requests=4, run_recovery_on_first_failure=False)

    assert len(records) == 4
    assert [r.success for r in records] == [True, False, True, True]
    assert records[1].failure_category == "provider_timeout"
