"""M5E.2 spec sections 17, 31: context/headroom metric tests."""

from __future__ import annotations

from src.evaluation.context_utilization import compute_context_utilization


def test_compute_context_utilization_basic_stats():
    report = compute_context_utilization([1000, 2000, 3000, 4000], configured_num_ctx=16384)
    assert report.max_input_tokens == 4000
    assert report.mean_input_tokens == 2500
    assert report.max_context_utilization_ratio == 4000 / 16384


def test_compute_context_utilization_none_num_ctx_reports_none_ratio():
    report = compute_context_utilization([1000, 2000], configured_num_ctx=None)
    assert report.max_input_tokens == 2000
    assert report.max_context_utilization_ratio is None


def test_compute_context_utilization_ignores_missing_token_counts():
    report = compute_context_utilization([1000, None, 3000, None], configured_num_ctx=16384)
    assert report.max_input_tokens == 3000
    assert report.mean_input_tokens == 2000


def test_compute_context_utilization_empty_input_is_all_none():
    report = compute_context_utilization([], configured_num_ctx=16384)
    assert report.max_input_tokens is None
    assert report.mean_input_tokens is None
    assert report.max_context_utilization_ratio is None


def test_compute_context_utilization_p95_within_range():
    values = list(range(1, 101))  # 1..100
    report = compute_context_utilization(values, configured_num_ctx=16384)
    assert 94 <= report.p95_input_tokens <= 100
