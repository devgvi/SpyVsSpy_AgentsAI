"""M5E.1.3: isolates request pacing (inter-request gap) from HTTP
connection reuse as separate variables in direct-Ollama (P0) long-horizon
stability - M5E.1.2 confounded the two (P0 used a fresh `OllamaClient` per
call at a 0ms gap; the production wrappers use one reused client at a
~100ms real gap). Diagnostic only - no gameplay/model/`num_ctx` change.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.evaluation.direct_ollama import direct_ollama_call
from src.evaluation.endurance import _bucket_for_tokens
from src.evaluation.resource_telemetry import find_ollama_pid, sample_full_resource_state
from src.evaluation.synthetic_stability import _build_base_state, _grow_memory, _nudged_observation
from src.llm.agent_memory import AgentMemory
from src.llm.clients.ollama_client import OllamaClient
from src.llm_controller import NATIVE_TOOL_LIKE_REPRESENTATION, NATIVE_TOOL_LIKE_SYSTEM_PROMPT, compact_observation_for_model, serialize_observation
from src import constants

CLIENT_REUSED = "reused"
CLIENT_FRESH = "fresh"


# -- production-cadence measurement (spec sections 7-8) ---------------------


def measure_production_gap_stats(provider_call_events_by_episode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """`provider_call_events_by_episode` groups real M5E.1.2 `provider_call`
    telemetry events by episode_id (any grouping key works - callers
    typically pass one entry per endurance episode). Measures the real,
    observed wall-clock idle time between one call's `end_time` and the
    next call's `start_time` within each group, in chronological order -
    never inferred from decision-interval config or frame timing (spec
    section 7)."""
    all_gaps: list[float] = []
    for calls in provider_call_events_by_episode.values():
        ordered = sorted(calls, key=lambda c: c["start_time"])
        for i in range(1, len(ordered)):
            prev_end = datetime.fromisoformat(ordered[i - 1]["end_time"])
            cur_start = datetime.fromisoformat(ordered[i]["start_time"])
            gap_ms = (cur_start - prev_end).total_seconds() * 1000.0
            all_gaps.append(max(gap_ms, 0.0))
    if not all_gaps:
        return {"n": 0, "median_ms": None, "p25_ms": None, "p75_ms": None, "min_ms": None, "max_ms": None}
    ordered_gaps = sorted(all_gaps)
    n = len(ordered_gaps)
    return {
        "n": n,
        "median_ms": statistics.median(ordered_gaps),
        "p25_ms": ordered_gaps[int(n * 0.25)],
        "p75_ms": ordered_gaps[int(n * 0.75)],
        "min_ms": ordered_gaps[0],
        "max_ms": ordered_gaps[-1],
    }


def load_provider_calls_by_episode(*paths: str) -> dict[str, list[dict[str, Any]]]:
    """Reads real M5E.1.2 `wrapper_decisions.jsonl`-shaped files and groups
    `provider_call` events by `episode_id`, excluding the one-off
    `fresh_wrapper_check` supplementary episode (a single isolated call,
    not a real cadence sample)."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event_type") != "provider_call":
                    continue
                episode_id = event.get("episode_id", "")
                if "fresh_wrapper_check" in episode_id:
                    continue
                grouped.setdefault(episode_id, []).append(event)
    return grouped


# -- deterministic production-like corpus (spec section 5) ------------------


@dataclass
class CorpusItem:
    prompt_id: int
    system_prompt: str
    user_prompt: str
    prompt_chars: int


def build_corpus(n_items: int, seed: int = 971, actor_id: str = constants.WHITE_SPY_ID) -> list[CorpusItem]:
    """The exact same growth-curve generator M5E.1.2's `run_p0_endurance`
    used (same underlying `_nudged_observation`/`_grow_memory`/
    `compact_observation_for_model` production call), built once and
    reused byte-for-byte across every pacing/client-mode condition (spec
    section 5's "use the same deterministic ordering across all pacing
    conditions") - never regenerated per condition."""
    state = _build_base_state(seed)
    memory = AgentMemory(actor_id)
    items = []
    for i in range(1, n_items + 1):
        observation = _nudged_observation(state, actor_id, i)
        _grow_memory(memory, i)
        observation_payload = serialize_observation(observation)
        model_payload = compact_observation_for_model(observation_payload, NATIVE_TOOL_LIKE_REPRESENTATION)
        model_payload["strategy_memory"] = memory.to_compact_dict()
        prompt_json = json.dumps(model_payload, sort_keys=True)
        items.append(CorpusItem(prompt_id=i, system_prompt=NATIVE_TOOL_LIKE_SYSTEM_PROMPT, user_prompt=prompt_json, prompt_chars=len(prompt_json)))
    return items


# -- condition matrix (spec sections 6, 9-10, 20) ----------------------------


@dataclass
class PacingCondition:
    condition_id: str
    gap_target_ms: float
    client_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {"condition_id": self.condition_id, "gap_target_ms": self.gap_target_ms, "client_mode": self.client_mode}


def build_condition_matrix(production_gap_ms: float) -> list[PacingCondition]:
    """Spec section 10's required cells only (the three optional
    intermediate-gap fresh-client cells are explicitly allowed to be
    skipped "if runtime cost is high" - documented, not silent, given
    this session's cumulative live-diagnostic time today)."""
    return [
        PacingCondition("gap_0ms_reused", 0.0, CLIENT_REUSED),
        PacingCondition("gap_0ms_fresh", 0.0, CLIENT_FRESH),
        PacingCondition("gap_250ms_reused", 250.0, CLIENT_REUSED),
        PacingCondition("gap_500ms_reused", 500.0, CLIENT_REUSED),
        PacingCondition("gap_1000ms_reused", 1000.0, CLIENT_REUSED),
        PacingCondition("gap_1000ms_fresh", 1000.0, CLIENT_FRESH),
        PacingCondition("gap_2000ms_reused", 2000.0, CLIENT_REUSED),
        PacingCondition("gap_production_reused", production_gap_ms, CLIENT_REUSED),
        PacingCondition("gap_production_fresh", production_gap_ms, CLIENT_FRESH),
    ]


def balanced_condition_order(conditions: list[PacingCondition]) -> list[PacingCondition]:
    """Spec section 20: not increasing gap order, and reused/fresh must
    not be systematically front- or back-loaded. A fixed, deterministic
    interleaving (not randomized - a random order would not be
    reproducible) that alternates gap magnitude and client mode."""
    by_id = {c.condition_id: c for c in conditions}
    preferred_order = [
        "gap_1000ms_reused",
        "gap_0ms_fresh",
        "gap_2000ms_reused",
        "gap_500ms_reused",
        "gap_production_fresh",
        "gap_250ms_reused",
        "gap_0ms_reused",
        "gap_production_reused",
        "gap_1000ms_fresh",
    ]
    ordered = [by_id[cid] for cid in preferred_order if cid in by_id]
    remaining = [c for c in conditions if c.condition_id not in preferred_order]
    return ordered + remaining


# -- per-request record + condition runner (spec sections 14-17) ------------


@dataclass
class PacedRequestRecord:
    experiment_id: str
    condition_id: str
    gap_target_ms: float
    actual_gap_ms: float | None
    client_mode: str
    request_index: int
    prompt_id: int
    prompt_size_bucket: str
    actual_input_tokens: int | None
    configured_num_ctx: int | None
    request_start: str
    response_complete: str
    success: bool
    timeout: bool
    failure_category: str | None
    failure_message_normalized: str | None
    wall_latency_ms: float
    provider_metadata: dict[str, Any] | None
    resource_snapshot: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "paced_request",
            "experiment_id": self.experiment_id,
            "condition_id": self.condition_id,
            "gap_target_ms": self.gap_target_ms,
            "actual_gap_ms": self.actual_gap_ms,
            "client_mode": self.client_mode,
            "request_index": self.request_index,
            "prompt_id": self.prompt_id,
            "prompt_size_bucket": self.prompt_size_bucket,
            "actual_input_tokens": self.actual_input_tokens,
            "configured_num_ctx": self.configured_num_ctx,
            "request_start": self.request_start,
            "response_complete": self.response_complete,
            "success": self.success,
            "timeout": self.timeout,
            "failure_category": self.failure_category,
            "failure_message_normalized": self.failure_message_normalized,
            "wall_latency_ms": self.wall_latency_ms,
            "provider_metadata": self.provider_metadata,
            "resource_snapshot": self.resource_snapshot,
        }


def run_pacing_condition(
    *,
    experiment_id: str,
    condition: PacingCondition,
    corpus: list[CorpusItem],
    num_ctx: int = 16384,
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma4:latest",
    timeout: float = 60.0,
    sample_resources: bool = True,
    on_request: Any = None,
    request_fn: Any = None,
) -> list[PacedRequestRecord]:
    """Spec sections 12-13: strictly sequential, single in-flight request,
    no automatic retry - one request produces exactly one recorded
    outcome. `client_mode=CLIENT_REUSED` constructs one `OllamaClient`
    before the loop and reuses that same object for every request in this
    condition; `CLIENT_FRESH` constructs a brand-new one immediately
    before every single request."""
    ollama_pid = find_ollama_pid() if sample_resources else None
    reused_client = (
        OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, num_ctx=num_ctx, request_fn=request_fn)
        if condition.client_mode == CLIENT_REUSED
        else None
    )

    records: list[PacedRequestRecord] = []
    previous_response_complete: float | None = None

    for item in corpus:
        if previous_response_complete is not None and condition.gap_target_ms > 0:
            wait_for = (condition.gap_target_ms / 1000.0) - (time.monotonic() - previous_response_complete)
            if wait_for > 0:
                time.sleep(wait_for)

        gap_start = time.monotonic()
        actual_gap_ms = (gap_start - previous_response_complete) * 1000.0 if previous_response_complete is not None else None

        client = (
            reused_client
            if condition.client_mode == CLIENT_REUSED
            else OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, num_ctx=num_ctx, request_fn=request_fn)
        )
        resource_snapshot = sample_full_resource_state(f"{condition.condition_id}_pre", ollama_pid, base_url, model_name) if sample_resources else None
        call = direct_ollama_call(client, request_index=item.prompt_id, user_prompt=item.user_prompt, system_prompt=item.system_prompt)
        previous_response_complete = time.monotonic()

        record = PacedRequestRecord(
            experiment_id=experiment_id,
            condition_id=condition.condition_id,
            gap_target_ms=condition.gap_target_ms,
            actual_gap_ms=actual_gap_ms,
            client_mode=condition.client_mode,
            request_index=item.prompt_id,
            prompt_id=item.prompt_id,
            prompt_size_bucket=_bucket_for_tokens(call.actual_input_tokens),
            actual_input_tokens=call.actual_input_tokens,
            configured_num_ctx=num_ctx,
            request_start=call.start_time,
            response_complete=call.end_time,
            success=call.success,
            timeout=call.timeout,
            failure_category=call.error_category,
            failure_message_normalized=call.error_message_normalized,
            wall_latency_ms=call.latency_ms,
            provider_metadata=call.provider_metadata,
            resource_snapshot=resource_snapshot,
        )
        records.append(record)
        if on_request is not None:
            on_request(item.prompt_id, record)

    return records


# -- analysis (spec sections 24-29) ------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def summarize_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec section 24's success-rate table plus section 25's failure-
    horizon and section 26's latency stats, computed together since every
    field comes from the same one pass over `records`."""
    n = len(records)
    successes = [r for r in records if r["success"]]
    timeouts = sum(1 for r in records if r.get("timeout"))
    other_failures = sum(1 for r in records if not r["success"] and not r.get("timeout"))
    first_failure_idx = next((i for i, r in enumerate(records) if not r["success"]), None)
    latencies = [r["wall_latency_ms"] for r in successes]
    actual_gaps = [r["actual_gap_ms"] for r in records if r.get("actual_gap_ms") is not None]

    max_consecutive_failures = 0
    current_streak = 0
    for r in records:
        if not r["success"]:
            current_streak += 1
            max_consecutive_failures = max(max_consecutive_failures, current_streak)
        else:
            current_streak = 0

    return {
        "attempts": n,
        "successes": len(successes),
        "timeouts": timeouts,
        "other_failures": other_failures,
        "success_rate": (len(successes) / n) if n else None,
        "first_failure_request": (records[first_failure_idx]["request_index"] if first_failure_idx is not None else None),
        "max_consecutive_failures": max_consecutive_failures,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 95) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "actual_median_gap_ms": statistics.median(actual_gaps) if actual_gaps else None,
    }


def compare_connection_reuse(records_reused: list[dict[str, Any]], records_fresh: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec sections 29, 45: matched-gap reused-vs-fresh comparison."""
    reused_summary = summarize_condition(records_reused)
    fresh_summary = summarize_condition(records_fresh)
    return {
        "reused_success_rate": reused_summary["success_rate"],
        "fresh_success_rate": fresh_summary["success_rate"],
        "reused_median_latency_ms": reused_summary["median_latency_ms"],
        "fresh_median_latency_ms": fresh_summary["median_latency_ms"],
        "reused_max_latency_ms": reused_summary["max_latency_ms"],
        "fresh_max_latency_ms": fresh_summary["max_latency_ms"],
    }


def compare_provider_vs_wall_timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec section 28: is wall latency much larger than Ollama's own
    reported `total_duration` (client/transport-side delay), or do both
    move together (provider-side compute)? Only over successful calls
    with real provider_metadata - a failed call has no total_duration to
    compare against."""
    pairs = [
        (r["wall_latency_ms"], r["provider_metadata"]["total_duration"] / 1e6)
        for r in records
        if r["success"] and r.get("provider_metadata") and r["provider_metadata"].get("total_duration") is not None
    ]
    if not pairs:
        return {"n": 0, "mean_wall_minus_provider_ms": None, "wall_much_greater_than_provider": None}
    diffs = [wall - provider for wall, provider in pairs]
    mean_diff = statistics.mean(diffs)
    mean_provider = statistics.mean(p for _, p in pairs)
    # "much greater" = client/transport overhead is at least 20% of the
    # provider's own reported duration, on average - an explicit,
    # documented threshold, not an arbitrary eyeball call.
    wall_much_greater = mean_provider > 0 and (mean_diff / mean_provider) > 0.2
    return {"n": len(pairs), "mean_wall_minus_provider_ms": mean_diff, "wall_much_greater_than_provider": wall_much_greater}
