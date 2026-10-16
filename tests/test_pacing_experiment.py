"""M5E.1.3 spec sections 40-42: network-free tests for the pacing/
connection-reuse experiment - gap scheduling, monotonic gap measurement,
fresh-vs-reused client lifecycle, condition matrix generation and balanced
ordering, no-automatic-retry, and the success-rate/failure-horizon/tail-
latency/provider-vs-wall-timing/production-gap-extraction analysis
functions. All provider calls are injected via `request_fn` - no network.
"""

from __future__ import annotations

import json
import time

from src.evaluation.pacing_experiment import (
    CLIENT_FRESH,
    CLIENT_REUSED,
    CorpusItem,
    PacingCondition,
    balanced_condition_order,
    build_condition_matrix,
    build_corpus,
    compare_connection_reuse,
    compare_provider_vs_wall_timing,
    load_provider_calls_by_episode,
    measure_production_gap_stats,
    run_pacing_condition,
    summarize_condition,
)


def _ok_response(input_tokens=1000, output_tokens=50, total_duration_ns=5_000_000_000):
    return {
        "message": {"content": '{"action": "wait"}'},
        "done": True,
        "done_reason": "stop",
        "total_duration": total_duration_ns,
        "prompt_eval_count": input_tokens,
        "eval_count": output_tokens,
    }


# -- production-gap extraction (spec section 8) -----------------------------


def test_measure_production_gap_stats_computes_real_wall_clock_gaps():
    episode_calls = {
        "ep1": [
            {"start_time": "2026-01-01T00:00:00.000000+00:00", "end_time": "2026-01-01T00:00:05.000000+00:00"},
            {"start_time": "2026-01-01T00:00:05.100000+00:00", "end_time": "2026-01-01T00:00:10.000000+00:00"},
            {"start_time": "2026-01-01T00:00:10.300000+00:00", "end_time": "2026-01-01T00:00:15.000000+00:00"},
        ]
    }
    stats = measure_production_gap_stats(episode_calls)
    assert stats["n"] == 2
    assert abs(stats["median_ms"] - 200.0) < 1.0  # median of 100ms and 300ms


def test_measure_production_gap_stats_handles_empty_input():
    stats = measure_production_gap_stats({})
    assert stats["n"] == 0
    assert stats["median_ms"] is None


def test_load_provider_calls_by_episode_excludes_fresh_wrapper_check(tmp_path):
    path = tmp_path / "decisions.jsonl"
    events = [
        {"event_type": "provider_call", "episode_id": "ep__native", "start_time": "x", "end_time": "y"},
        {"event_type": "provider_call", "episode_id": "ep__fresh_wrapper_check", "start_time": "x", "end_time": "y"},
        {"event_type": "llm_decision", "episode_id": "ep__native"},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events))
    grouped = load_provider_calls_by_episode(str(path))
    assert list(grouped.keys()) == ["ep__native"]
    assert len(grouped["ep__native"]) == 1


# -- deterministic corpus (spec section 5) -----------------------------------


def test_build_corpus_is_deterministic_and_grows_in_size():
    corpus_a = build_corpus(10, seed=971)
    corpus_b = build_corpus(10, seed=971)
    assert [c.user_prompt for c in corpus_a] == [c.user_prompt for c in corpus_b]
    sizes = [c.prompt_chars for c in corpus_a]
    assert sizes == sorted(sizes)  # growth curve, monotonically non-decreasing... allow ties
    assert sizes[-1] > sizes[0]


# -- condition matrix + balanced ordering (spec sections 9-10, 20) ----------


def test_build_condition_matrix_covers_all_required_cells():
    matrix = build_condition_matrix(production_gap_ms=101.2)
    ids = {c.condition_id for c in matrix}
    assert ids == {
        "gap_0ms_reused", "gap_0ms_fresh",
        "gap_250ms_reused", "gap_500ms_reused",
        "gap_1000ms_reused", "gap_1000ms_fresh",
        "gap_2000ms_reused",
        "gap_production_reused", "gap_production_fresh",
    }
    prod = next(c for c in matrix if c.condition_id == "gap_production_reused")
    assert prod.gap_target_ms == 101.2


def test_build_condition_matrix_has_both_client_modes_at_0ms_and_1000ms_and_production():
    matrix = build_condition_matrix(production_gap_ms=100.0)
    by_id = {c.condition_id: c for c in matrix}
    assert by_id["gap_0ms_reused"].client_mode == CLIENT_REUSED
    assert by_id["gap_0ms_fresh"].client_mode == CLIENT_FRESH
    assert by_id["gap_1000ms_reused"].client_mode == CLIENT_REUSED
    assert by_id["gap_1000ms_fresh"].client_mode == CLIENT_FRESH


def test_balanced_condition_order_is_not_increasing_gap_order():
    matrix = build_condition_matrix(production_gap_ms=100.0)
    ordered = balanced_condition_order(matrix)
    gaps_in_order = [c.gap_target_ms for c in ordered]
    assert gaps_in_order != sorted(gaps_in_order)


def test_balanced_condition_order_does_not_front_load_one_client_mode():
    matrix = build_condition_matrix(production_gap_ms=100.0)
    ordered = balanced_condition_order(matrix)
    first_three_modes = {c.client_mode for c in ordered[:3]}
    # both modes should appear within the first few conditions, not all
    # reused-then-all-fresh or vice versa
    assert len(first_three_modes) == 2


def test_balanced_condition_order_is_deterministic_and_lossless():
    matrix = build_condition_matrix(production_gap_ms=100.0)
    ordered_a = balanced_condition_order(matrix)
    ordered_b = balanced_condition_order(matrix)
    assert [c.condition_id for c in ordered_a] == [c.condition_id for c in ordered_b]
    assert {c.condition_id for c in ordered_a} == {c.condition_id for c in matrix}


# -- gap scheduling + client lifecycle (spec sections 6, 9, 14, 40) --------


def test_zero_gap_condition_has_no_intentional_delay():
    def fake_request(payload):
        return _ok_response()

    corpus = [CorpusItem(prompt_id=i, system_prompt="sys", user_prompt=f"prompt {i}", prompt_chars=10) for i in range(1, 6)]
    condition = PacingCondition("gap_0ms_reused", 0.0, CLIENT_REUSED)

    start = time.monotonic()
    records = run_pacing_condition(
        experiment_id="test", condition=condition, corpus=corpus, sample_resources=False, request_fn=fake_request
    )
    elapsed = time.monotonic() - start
    assert elapsed < 1.0  # 5 near-instant fake calls, no artificial gap
    assert records[1].actual_gap_ms is not None
    assert records[1].actual_gap_ms < 50.0  # no intentional delay at 0ms


def test_nonzero_gap_condition_enforces_the_target_gap():
    def fake_request(payload):
        return _ok_response()

    corpus = [CorpusItem(prompt_id=i, system_prompt="sys", user_prompt=f"prompt {i}", prompt_chars=10) for i in range(1, 4)]
    condition = PacingCondition("gap_test", 150.0, CLIENT_REUSED)

    records = run_pacing_condition(
        experiment_id="test", condition=condition, corpus=corpus, sample_resources=False, request_fn=fake_request
    )
    assert records[1].actual_gap_ms >= 140.0  # enforced, allowing small scheduling slack
    assert records[2].actual_gap_ms >= 140.0


def test_reused_client_mode_shares_one_client_object_across_requests():
    seen_client_ids = []
    original_init = None

    from src.evaluation import pacing_experiment as pe

    real_ollama_client = pe.OllamaClient

    class _TrackingClient(real_ollama_client):
        def generate_action(self, *a, **kw):
            seen_client_ids.append(id(self))
            return real_ollama_client.generate_action(self, *a, **kw)

    def fake_request(payload):
        return _ok_response()

    pe_client_backup = pe.OllamaClient
    pe.OllamaClient = _TrackingClient
    try:
        corpus = [CorpusItem(prompt_id=i, system_prompt="sys", user_prompt=f"p{i}", prompt_chars=5) for i in range(1, 4)]
        condition = PacingCondition("reused_test", 0.0, CLIENT_REUSED)
        run_pacing_condition(experiment_id="test", condition=condition, corpus=corpus, sample_resources=False, request_fn=fake_request)
    finally:
        pe.OllamaClient = pe_client_backup

    assert len(set(seen_client_ids)) == 1  # same object every call


def test_fresh_client_mode_constructs_a_new_client_every_request():
    seen_client_ids = []
    from src.evaluation import pacing_experiment as pe

    real_ollama_client = pe.OllamaClient

    class _TrackingClient(real_ollama_client):
        def generate_action(self, *a, **kw):
            seen_client_ids.append(id(self))
            return real_ollama_client.generate_action(self, *a, **kw)

    def fake_request(payload):
        return _ok_response()

    pe_client_backup = pe.OllamaClient
    pe.OllamaClient = _TrackingClient
    try:
        corpus = [CorpusItem(prompt_id=i, system_prompt="sys", user_prompt=f"p{i}", prompt_chars=5) for i in range(1, 4)]
        condition = PacingCondition("fresh_test", 0.0, CLIENT_FRESH)
        run_pacing_condition(experiment_id="test", condition=condition, corpus=corpus, sample_resources=False, request_fn=fake_request)
    finally:
        pe.OllamaClient = pe_client_backup

    assert len(set(seen_client_ids)) == 3  # a distinct object every call


def test_no_automatic_retry_one_request_one_outcome():
    calls = {"n": 0}

    def fake_request(payload):
        calls["n"] += 1
        return {"error": "always fails"}

    corpus = [CorpusItem(prompt_id=i, system_prompt="sys", user_prompt=f"p{i}", prompt_chars=5) for i in range(1, 6)]
    condition = PacingCondition("no_retry_test", 0.0, CLIENT_REUSED)
    records = run_pacing_condition(experiment_id="test", condition=condition, corpus=corpus, sample_resources=False, request_fn=fake_request)

    assert calls["n"] == 5  # exactly one call per corpus item, never retried
    assert all(not r.success for r in records)


# -- analysis: success rate / failure horizon / tail latency ----------------


def _rec(idx, success, latency_ms, timeout=False, gap_ms=100.0):
    return {"request_index": idx, "success": success, "wall_latency_ms": latency_ms, "timeout": timeout, "actual_gap_ms": gap_ms}


def test_summarize_condition_success_rate_and_failure_horizon():
    records = [_rec(1, True, 10000), _rec(2, True, 12000), _rec(3, False, 60000, timeout=True), _rec(4, False, 60000, timeout=True), _rec(5, True, 11000)]
    summary = summarize_condition(records)
    assert summary["attempts"] == 5
    assert summary["successes"] == 3
    assert summary["timeouts"] == 2
    assert summary["success_rate"] == 0.6
    assert summary["first_failure_request"] == 3
    assert summary["max_consecutive_failures"] == 2


def test_summarize_condition_tail_latency():
    records = [_rec(i, True, 10000 + i * 100) for i in range(1, 21)]
    summary = summarize_condition(records)
    assert summary["median_latency_ms"] < summary["p95_latency_ms"] <= summary["max_latency_ms"]


def test_summarize_condition_handles_all_failures():
    records = [_rec(1, False, 60000, timeout=True)]
    summary = summarize_condition(records)
    assert summary["median_latency_ms"] is None
    assert summary["success_rate"] == 0.0


def test_compare_connection_reuse_reports_both_sides():
    reused = [_rec(1, True, 20000), _rec(2, False, 60000, timeout=True)]
    fresh = [_rec(1, True, 15000), _rec(2, True, 16000)]
    comparison = compare_connection_reuse(reused, fresh)
    assert comparison["reused_success_rate"] == 0.5
    assert comparison["fresh_success_rate"] == 1.0


# -- provider vs wall timing (spec section 28) -------------------------------


def test_compare_provider_vs_wall_timing_detects_transport_overhead():
    records = [
        {"success": True, "wall_latency_ms": 30000, "provider_metadata": {"total_duration": 10_000_000_000}},
        {"success": True, "wall_latency_ms": 32000, "provider_metadata": {"total_duration": 11_000_000_000}},
    ]
    result = compare_provider_vs_wall_timing(records)
    assert result["n"] == 2
    assert result["wall_much_greater_than_provider"] is True  # ~20s overhead vs ~10.5s provider time


def test_compare_provider_vs_wall_timing_detects_matched_timing():
    records = [
        {"success": True, "wall_latency_ms": 10200, "provider_metadata": {"total_duration": 10_000_000_000}},
    ]
    result = compare_provider_vs_wall_timing(records)
    assert result["wall_much_greater_than_provider"] is False


def test_compare_provider_vs_wall_timing_handles_no_data():
    result = compare_provider_vs_wall_timing([{"success": False, "wall_latency_ms": 60000, "provider_metadata": None}])
    assert result["n"] == 0
    assert result["wall_much_greater_than_provider"] is None
