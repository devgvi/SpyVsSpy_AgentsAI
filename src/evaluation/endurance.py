"""M5E.1.2: long-horizon endurance tests across P0 (direct Ollama), P1
(native_tool_like production path), P2 (Strands v2 production path), with
continuous host-resource sampling and a controlled post-timeout probe/
recovery sequence - never gameplay, never a competitive comparison (spec
section 46).

Content parity across all three paths (spec section 10's "same machine,
same sequential execution policy") comes from reusing M5E.1's synthetic
S2_growing machinery for the *shape* of the growth curve: P1/P2 run it
directly (`run_synthetic_pattern`, unmodified production adapters); P0
replicates the identical Observation/memory growth sequence and sends the
exact same `compact_observation_for_model(...) + strategy_memory` payload
`LLMController._request_action` would build, through the same `OllamaClient`
Native uses - never a hand-simplified prompt (spec section 8).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.evaluation.direct_ollama import direct_ollama_call, unload_model
from src.evaluation.resource_telemetry import find_ollama_pid, sample_full_resource_state
from src.evaluation.synthetic_stability import (
    PATTERN_S2_GROWING,
    _build_base_state,
    _grow_memory,
    _nudged_observation,
    run_synthetic_pattern,
)
from src.llm.agent_memory import AgentMemory
from src.llm.clients.ollama_client import OllamaClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import (
    NATIVE_TOOL_LIKE_REPRESENTATION,
    NATIVE_TOOL_LIKE_SYSTEM_PROMPT,
    classify_provider_call_error,
    compact_observation_for_model,
    normalize_provider_error_message,
    serialize_observation,
)
from src import constants

TINY_PROBE_SYSTEM_PROMPT = "You output JSON only."
TINY_PROBE_USER_PROMPT = '{"ping": true} - respond with {"pong": true} and nothing else.'


# -- background continuous resource sampler --------------------------------


class BackgroundResourceSampler:
    """Samples `sample_full_resource_state` every `interval_seconds` on a
    daemon thread until `stop()` - spec section 18's "collect host metrics
    throughout each sequence... every 2-5 seconds." Kept deliberately
    simple (a list append under a lock) so it cannot materially perturb
    the benchmark it's observing."""

    def __init__(self, label_prefix: str, ollama_pid: int | None, interval_seconds: float = 3.0):
        self.label_prefix = label_prefix
        self.ollama_pid = ollama_pid
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = sample_full_resource_state(f"{self.label_prefix}_continuous", self.ollama_pid)
            with self._lock:
                self.samples.append(sample)
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2.0)
        with self._lock:
            return list(self.samples)


# -- post-timeout probe sequence (spec sections 28-30) ----------------------


def run_post_timeout_probe_sequence(
    *,
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma4:latest",
    timeout: float = 60.0,
    ollama_pid: int | None = None,
    schedule_seconds: tuple[float, ...] = (0.0, 5.0, 15.0, 30.0),
    extension_offset_seconds: float | None = None,
    request_fn: Any = None,
) -> list[dict[str, Any]]:
    """Spec section 29: T+0 resource snapshot, then a tiny fixed probe at
    T+5/15/30 (an optional final probe only if still unhealthy at the last
    scheduled offset - `extension_offset_seconds`, defaulting to double
    the gap between the last two scheduled offsets so a caller using a
    faster test schedule doesn't accidentally wait a real 60s here).
    Never the normal production workload (spec section 31's
    "immediate-retry trap") - a minimal, cheap request whose only purpose
    is detecting whether the provider has become responsive again."""
    if extension_offset_seconds is None:
        if len(schedule_seconds) >= 2:
            extension_offset_seconds = schedule_seconds[-1] + (schedule_seconds[-1] - schedule_seconds[-2])
        else:
            extension_offset_seconds = schedule_seconds[-1] * 2 if schedule_seconds else 60.0
    if ollama_pid is None:
        ollama_pid = find_ollama_pid()
    results = []
    start = time.monotonic()
    last_unhealthy = False
    for target_offset in schedule_seconds:
        wait_for = target_offset - (time.monotonic() - start)
        if wait_for > 0:
            time.sleep(wait_for)
        resource_snapshot = sample_full_resource_state("post_timeout_probe", ollama_pid, base_url, model_name)
        if target_offset == 0.0:
            results.append({"t_offset_s": 0.0, "probe": None, "resource_snapshot": resource_snapshot})
            continue
        client = OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, request_fn=request_fn)
        probe_start = time.monotonic()
        try:
            response = client.generate_action(TINY_PROBE_SYSTEM_PROMPT, TINY_PROBE_USER_PROMPT, {})
            probe_latency_ms = (time.monotonic() - probe_start) * 1000.0
            error_category = classify_provider_call_error(response.error_type, response.error_message)
            success = error_category is None
        except Exception as exc:  # pragma: no cover - defensive: a probe itself can time out
            probe_latency_ms = (time.monotonic() - probe_start) * 1000.0
            success = False
            error_category = "provider_timeout"
        last_unhealthy = not success
        results.append(
            {
                "t_offset_s": target_offset,
                "probe": {"success": success, "latency_ms": probe_latency_ms, "error_category": error_category},
                "resource_snapshot": resource_snapshot,
            }
        )
        if target_offset == schedule_seconds[-1] and not last_unhealthy:
            break  # spec section 29: the extension probe is optional, only "if still unhealthy"
    if last_unhealthy and extension_offset_seconds not in schedule_seconds:
        wait_for = extension_offset_seconds - (time.monotonic() - start)
        if wait_for > 0:
            time.sleep(wait_for)
        resource_snapshot = sample_full_resource_state("post_timeout_probe", ollama_pid, base_url, model_name)
        client = OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, request_fn=request_fn)
        probe_start = time.monotonic()
        try:
            response = client.generate_action(TINY_PROBE_SYSTEM_PROMPT, TINY_PROBE_USER_PROMPT, {})
            probe_latency_ms = (time.monotonic() - probe_start) * 1000.0
            error_category = classify_provider_call_error(response.error_type, response.error_message)
            success = error_category is None
        except Exception:
            probe_latency_ms = (time.monotonic() - probe_start) * 1000.0
            success, error_category = False, "provider_timeout"
        results.append(
            {
                "t_offset_s": extension_offset_seconds,
                "probe": {"success": success, "latency_ms": probe_latency_ms, "error_category": error_category},
                "resource_snapshot": resource_snapshot,
            }
        )
    return results


# -- recovery ladder v2 (spec section 32): wait/probe -> fresh wrapper -> reload --


def run_recovery_ladder_v2(
    *,
    runtime: str,
    make_request_fn: Callable[[], bool],
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma4:latest",
    timeout: float = 60.0,
    request_fn: Any = None,
    unload_wait_seconds: float = 2.0,
) -> dict[str, Any]:
    """`make_request_fn` repeats the exact semantic request that failed
    (fresh wrapper each call - spec section 33) and returns whether it
    succeeded. Stage order: wait/probe only (the post-timeout probe
    sequence itself is stage 1 and is expected to already have run before
    this is called) -> fresh wrapper retry -> explicit model unload/reload.
    Ollama restart (spec section 35) is deliberately not implemented here:
    restarting the actual local Ollama process is outside this project's
    safe automated scope (same judgment call M5E.1.1 made), documented as
    a skipped step rather than attempted.
    """
    stages: dict[str, Any] = {}

    stage2_success = make_request_fn()
    stages["fresh_wrapper_retry"] = {"runtime": runtime, "success": stage2_success}
    if stage2_success:
        stages["recovered_at"] = "fresh_wrapper_retry"
        return stages

    resource_before_unload = sample_full_resource_state("before_unload", find_ollama_pid(), base_url, model_name)
    unloaded = unload_model(base_url=base_url, model_name=model_name, request_fn=request_fn)
    time.sleep(unload_wait_seconds)
    resource_after_unload = sample_full_resource_state("after_unload", find_ollama_pid(), base_url, model_name)
    # A trivial warm call reloads the model (spec section 34: "reload
    # Gemma4"); it also doubles as this stage's probe. Uses
    # direct_ollama_call (not a bare generate_action) so a still-unhealthy
    # provider timing out on the reload probe itself is recorded, not a
    # crash.
    warm_client = OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, request_fn=request_fn)
    warm_call = direct_ollama_call(warm_client, request_index=0, user_prompt=TINY_PROBE_USER_PROMPT, system_prompt=TINY_PROBE_SYSTEM_PROMPT)
    resource_after_reload = sample_full_resource_state("after_reload", find_ollama_pid(), base_url, model_name)
    stage3_success = make_request_fn()
    stages["model_reload"] = {
        "unload_probe_succeeded": unloaded,
        "resource_before_unload": resource_before_unload,
        "resource_after_unload": resource_after_unload,
        "resource_after_reload": resource_after_reload,
        "reload_warm_call_error": warm_call.error_category,
        "retry_success": stage3_success,
    }
    stages["recovered_at"] = "model_reload" if stage3_success else None
    stages["ollama_restart"] = "not attempted - outside this project's safe automated scope (spec section 35)"
    return stages


# -- P0: direct-Ollama endurance, content-matched to the P1/P2 growth curve --


@dataclass
class EnduranceCallRecord:
    sequence_id: int
    path: str
    request_index: int
    timestamp_start: str
    timestamp_end: str
    configured_num_ctx: int | None
    actual_input_tokens: int | None
    actual_output_tokens: int | None
    prompt_size_bucket: str
    prompt_chars: int
    success: bool
    failure_category: str | None
    failure_message_normalized: str | None
    wall_latency_ms: float
    provider_metadata: dict[str, Any] | None
    pre_request_resource_snapshot: dict[str, Any]
    post_request_resource_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "endurance_call",
            "sequence_id": self.sequence_id,
            "path": self.path,
            "request_index": self.request_index,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "configured_num_ctx": self.configured_num_ctx,
            "actual_input_tokens": self.actual_input_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "prompt_size_bucket": self.prompt_size_bucket,
            "prompt_chars": self.prompt_chars,
            "success": self.success,
            "failure_category": self.failure_category,
            "failure_message_normalized": self.failure_message_normalized,
            "wall_latency_ms": self.wall_latency_ms,
            "provider_metadata": self.provider_metadata,
            "pre_request_resource_snapshot": self.pre_request_resource_snapshot,
            "post_request_resource_snapshot": self.post_request_resource_snapshot,
        }


def extract_first_failure_window(records: list[dict[str, Any]], before: int = 5, after: int = 5) -> dict[str, Any]:
    """Spec section 24: 5 requests before the first failure, the failure
    itself, and 5 after - the slice a human would actually want to stare
    at, computed once instead of by hand every time."""
    first_failure_idx = next((i for i, r in enumerate(records) if not r.get("success")), None)
    if first_failure_idx is None:
        return {"first_failure_index": None, "window": []}
    start = max(0, first_failure_idx - before)
    end = min(len(records), first_failure_idx + after + 1)
    return {"first_failure_index": first_failure_idx, "window": records[start:end]}


def compute_latency_degradation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec section 25: does latency degrade gradually before the first
    failure, or does it appear suddenly? Uses only the *successful* calls
    before the first failure - a timed-out call's own latency is the
    timeout ceiling, not a meaningful data point for a "getting slower"
    trend."""
    first_failure_idx = next((i for i, r in enumerate(records) if not r.get("success")), None)
    before = records[:first_failure_idx] if first_failure_idx is not None else records
    successful_before = [r["wall_latency_ms"] for r in before if r.get("success")]
    if not successful_before:
        return {
            "median_latency_first_5_ms": None,
            "median_latency_last_5_before_failure_ms": None,
            "max_successful_latency_ms": None,
            "sudden_onset": None,
        }
    import statistics

    first_5 = successful_before[:5]
    last_5 = successful_before[-5:]
    median_first = statistics.median(first_5)
    median_last = statistics.median(last_5)
    # "Sudden" = the last successful calls weren't already trending much
    # slower than the first ones - a real gradual-degradation signature
    # would show median_last well above median_first instead.
    sudden_onset = median_last < median_first * 1.5
    return {
        "median_latency_first_5_ms": median_first,
        "median_latency_last_5_before_failure_ms": median_last,
        "max_successful_latency_ms": max(successful_before),
        "sudden_onset": sudden_onset,
    }


def _bucket_for_tokens(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    if tokens < 3500:
        return "small"
    if tokens < 5500:
        return "medium"
    return "large"


def run_p0_endurance(
    *,
    n_requests: int,
    num_ctx: int = 16384,
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma4:latest",
    timeout: float = 60.0,
    seed: int = 931,
    actor_id: str = constants.WHITE_SPY_ID,
    run_recovery_on_first_failure: bool = True,
    on_call: Callable[[int, EnduranceCallRecord], None] | None = None,
) -> list[EnduranceCallRecord]:
    """P0 direct-Ollama endurance (spec sections 6, 15): same growth curve
    as `run_synthetic_pattern(PATTERN_S2_GROWING, ...)` would drive Native/
    Strands through, sent as the exact production `compact_observation_for_model
    (...) + strategy_memory` JSON payload, via the same `OllamaClient`
    Native uses - never a hand-simplified prompt."""
    state = _build_base_state(seed)
    memory = AgentMemory(actor_id)
    records: list[EnduranceCallRecord] = []
    ollama_pid = find_ollama_pid()
    first_failure_handled = False

    for i in range(1, n_requests + 1):
        observation = _nudged_observation(state, actor_id, i)
        _grow_memory(memory, i)
        observation_payload = serialize_observation(observation)
        model_payload = compact_observation_for_model(observation_payload, NATIVE_TOOL_LIKE_REPRESENTATION)
        model_payload["strategy_memory"] = memory.to_compact_dict()
        prompt_json = json.dumps(model_payload, sort_keys=True)

        pre_snapshot = sample_full_resource_state("p0_pre", ollama_pid, base_url, model_name)
        client = OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, num_ctx=num_ctx)
        # M5E.1.2 bug fix: use direct_ollama_call, not a bare
        # client.generate_action(...) - the latter never catches the raw
        # socket TimeoutError a real timeout raises (OllamaClient itself
        # doesn't either), which crashed the first live endurance run the
        # moment a genuine timeout occurred instead of recording it as a
        # failed call. See direct_ollama.py's direct_ollama_call docstring.
        call = direct_ollama_call(client, request_index=i, user_prompt=prompt_json, system_prompt=NATIVE_TOOL_LIKE_SYSTEM_PROMPT)
        post_snapshot = sample_full_resource_state("p0_post", ollama_pid, base_url, model_name)

        record = EnduranceCallRecord(
            sequence_id=i,
            path="P0_direct_ollama",
            request_index=i,
            timestamp_start=call.start_time,
            timestamp_end=call.end_time,
            configured_num_ctx=num_ctx,
            actual_input_tokens=call.actual_input_tokens,
            actual_output_tokens=call.actual_output_tokens,
            prompt_size_bucket=_bucket_for_tokens(call.actual_input_tokens),
            prompt_chars=len(prompt_json),
            success=call.success,
            failure_category=call.error_category,
            failure_message_normalized=call.error_message_normalized,
            wall_latency_ms=call.latency_ms,
            provider_metadata=call.provider_metadata,
            pre_request_resource_snapshot=pre_snapshot,
            post_request_resource_snapshot=post_snapshot,
        )
        records.append(record)
        if on_call is not None:
            on_call(i, record)

        if not record.success and run_recovery_on_first_failure and not first_failure_handled:
            first_failure_handled = True
            probe_sequence = run_post_timeout_probe_sequence(base_url=base_url, model_name=model_name, timeout=timeout, ollama_pid=ollama_pid)
            record_dict = record.to_dict()
            record_dict["post_timeout_probe_sequence"] = probe_sequence

            def _retry() -> bool:
                fresh_client = OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, num_ctx=num_ctx)
                retry_call = direct_ollama_call(fresh_client, request_index=i, user_prompt=prompt_json, system_prompt=NATIVE_TOOL_LIKE_SYSTEM_PROMPT)
                return retry_call.success

            recovery = run_recovery_ladder_v2(runtime="P0_direct_ollama", make_request_fn=_retry, base_url=base_url, model_name=model_name, timeout=timeout)
            record_dict["recovery_ladder"] = recovery
            if on_call is not None:
                on_call(-i, record_dict)  # negative index signals "supplementary probe/recovery data for request i"

    return records


# -- P1/P2: production-wrapper endurance (spec sections 16-17) -------------


def run_wrapper_endurance(
    *,
    runtime: str,
    n_requests: int,
    num_ctx: int = 16384,
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma4:latest",
    timeout: float = 60.0,
    seed: int = 931,
    telemetry: TelemetryLogger | None = None,
    episode_id: str | None = None,
    on_call: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """P1 (`native_tool_like`) / P2 (`strands`) endurance via the real,
    unmodified production adapters (spec section 16: "use production
    code... do not implement a special simplified Native path") - reuses
    `run_synthetic_pattern`'s already-tested S2_growing sequence (the same
    growth curve P0 replicates above) rather than re-implementing the
    async decision polling loop a second time.

    Recovery-ladder note (spec section 33): exactly replaying the *same*
    Observation/memory state a mid-sequence wrapper failure saw would
    require intercepting `run_synthetic_pattern`'s internal loop, which
    this function deliberately does not duplicate. Instead, after the run
    completes, one fresh adapter instance is built and given one more
    decision at a comparable (not byte-identical) growth position - a
    real "does a brand-new wrapper succeed where a long-running one
    didn't" check, honestly documented as an approximation rather than an
    exact request replay.
    """
    ollama_pid = find_ollama_pid()
    telemetry = telemetry or TelemetryLogger()
    first_failure_handled = {"done": False}
    supplementary: dict[str, Any] = {}

    def _on_decision(i: int, record: dict[str, Any]) -> None:
        record["resource_snapshot_after"] = sample_full_resource_state(f"{runtime}_post_decision", ollama_pid, base_url, model_name)
        if on_call is not None:
            on_call(i, record)
        if record.get("final_model_action_valid") is False and not first_failure_handled["done"]:
            first_failure_handled["done"] = True
            supplementary["first_failure_decision_index"] = i
            supplementary["post_timeout_probe_sequence"] = run_post_timeout_probe_sequence(
                base_url=base_url, model_name=model_name, timeout=timeout, ollama_pid=ollama_pid
            )

    result = run_synthetic_pattern(
        pattern=PATTERN_S2_GROWING,
        runtime=runtime,
        n_decisions=n_requests,
        provider="ollama",
        model=model_name,
        base_url=base_url,
        timeout=timeout,
        telemetry=telemetry,
        episode_id=episode_id or f"m5e1_2_endurance__{runtime}",
        seed=seed,
        per_decision_wall_timeout=180.0,
        num_ctx=num_ctx,
        on_decision=_on_decision,
    )

    fresh_wrapper_check: dict[str, Any] | None = None
    if first_failure_handled["done"]:
        fresh_result = run_synthetic_pattern(
            pattern=PATTERN_S2_GROWING,
            runtime=runtime,
            n_decisions=1,
            provider="ollama",
            model=model_name,
            base_url=base_url,
            timeout=timeout,
            telemetry=telemetry,
            episode_id=(episode_id or f"m5e1_2_endurance__{runtime}") + "__fresh_wrapper_check",
            seed=seed + 1,  # a different but comparable-size growth position, not a mid-sequence replay
            per_decision_wall_timeout=180.0,
            num_ctx=num_ctx,
        )
        fresh_wrapper_check = {
            "success": bool(fresh_result.decisions and fresh_result.decisions[0].get("final_model_action_valid")),
            "decision": fresh_result.decisions[0] if fresh_result.decisions else None,
        }
        supplementary["fresh_wrapper_check"] = fresh_wrapper_check

    return {"result": result, "supplementary": supplementary}
