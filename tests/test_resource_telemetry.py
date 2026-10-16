"""M5E.1.1 spec section 28: resource telemetry sanity checks. Uses real
`ps`/`lsof` against the current test process (no network, no external
service) rather than mocking the OS - cheap and genuinely exercises the
parsing logic against real `ps` output formatting."""

from __future__ import annotations

import os

from src.evaluation.resource_telemetry import sample_process, sample_resources


def test_sample_process_reports_real_values_for_current_process():
    sample = sample_process(os.getpid())
    assert sample is not None
    assert sample["rss_bytes"] > 0
    assert sample["cpu_percent"] is not None


def test_sample_process_returns_none_for_a_nonexistent_pid():
    # PID 999999 essentially never exists on a real system.
    assert sample_process(999999) is None


def test_sample_resources_labels_and_never_fabricates_missing_ollama_pid():
    sample = sample_resources("before", ollama_pid=None)
    assert sample["label"] == "before"
    assert sample["ollama"] is None
    assert sample["diagnostic_process"] is not None


# -- M5E.1.2: system memory metrics (real vm_stat/sysctl, no network) -----


def test_sample_vm_stat_returns_real_nonzero_fields():
    from src.evaluation.resource_telemetry import sample_vm_stat

    sample = sample_vm_stat()
    assert sample is not None
    assert sample["page_size_bytes"] > 0
    assert sample["free_bytes"] > 0
    assert sample["compressed_bytes"] is not None


def test_sample_swap_usage_returns_real_fields():
    from src.evaluation.resource_telemetry import sample_swap_usage

    sample = sample_swap_usage()
    assert sample is not None
    assert sample["total_mb"] >= 0
    assert sample["used_mb"] >= 0


def test_sample_memory_pressure_derives_a_state_label_not_kernel_reported():
    from src.evaluation.resource_telemetry import sample_memory_pressure

    sample = sample_memory_pressure()
    assert sample is not None
    assert 0 <= sample["free_percent"] <= 100
    assert "derived" in sample["pressure_state"]


def test_sample_physical_memory_matches_real_machine():
    from src.evaluation.resource_telemetry import sample_physical_memory_bytes

    assert sample_physical_memory_bytes() > 0


def test_sample_system_memory_never_fabricates_missing_fields(monkeypatch):
    from src.evaluation import resource_telemetry as rt

    monkeypatch.setattr(rt, "_run", lambda *a, **k: None)
    sample = rt.sample_system_memory()
    assert all(v is None for v in sample.values())


def test_sample_model_residency_handles_unreachable_ollama(monkeypatch):
    from src.evaluation import resource_telemetry as rt

    monkeypatch.setattr(rt, "_get_json", lambda *a, **k: None, raising=False)
    import src.evaluation.ollama_health as health

    monkeypatch.setattr(health, "_get_json", lambda *a, **k: None)
    result = rt.sample_model_residency()
    assert result == {"resident": None, "details": None}


def test_sample_full_resource_state_combines_all_four_sources():
    from src.evaluation.resource_telemetry import sample_full_resource_state

    sample = sample_full_resource_state("baseline", ollama_pid=None)
    assert sample["label"] == "baseline"
    assert "system" in sample and "model" in sample
    assert sample["ollama"] is None  # no pid given
