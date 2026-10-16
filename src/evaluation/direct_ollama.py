"""M5E.1.1 spec section 6: a minimal diagnostic client for Ollama's
`POST /api/chat`, isolating the provider/model layer itself from every
higher layer (`Game`, `EpisodeRunner`, `AgentMemoryTracker`,
`NativeAgentAdapter`, `StrandsAgentAdapter` - none imported here). Built
directly on `OllamaClient` (the same low-level HTTP client
`LLMController`/Native already uses) rather than reinventing the HTTP
call, so any difference later found between "P0 direct" and "P1 native"
can only be attributed to `LLMController`'s own orchestration, not to two
subtly-different request implementations. Diagnostic only (spec section
6's own instruction) - never a new gameplay provider abstraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.llm.clients.ollama_client import OllamaClient
from src.llm_controller import classify_provider_call_error, normalize_provider_error_message

FIXED_WORKLOAD_SYSTEM_PROMPT = (
    "You control one spy in a competitive embassy escape game. "
    "Choose exactly one action from legal_actions. "
    "Respond with JSON only: {\"action\": ..., \"target_id\": ..., \"reason\": ...}."
)

# spec section 8's example shape, held byte-for-byte identical across
# every S1 call - this is the "actual prompt size" that stays fixed while
# only `num_ctx` (configured capacity) varies across the matrix.
FIXED_WORKLOAD_USER_PROMPT = (
    "Observation:\n"
    "- room = office\n"
    "- near_object_id = statue_01\n\n"
    "Legal actions:\n"
    "- MOVE_TO(desk_01)\n"
    "- INSPECT(statue_01)\n\n"
    "Return one canonical action as JSON."
)

_FILLER_LINE = (
    "recent_decision: tick={i} action=search target=synthetic_object_{i} "
    "result=empty detail='deterministic filler content for prompt-size control, not chain-of-thought' | "
)


def build_padded_user_prompt(target_chars: int) -> str:
    """Deterministic, semantically-neutral filler (spec section 10: "Do
    NOT use hidden chain-of-thought") appended to the fixed workload
    prompt until it reaches approximately `target_chars` characters.
    Reproducible for a given `target_chars` - same content every call at
    that size, so repeated calls at one bucket are a genuine "constant
    actual prompt size" test (spec section 9), not incidentally-varying
    padding."""
    base = FIXED_WORKLOAD_USER_PROMPT + "\n\nContext log:\n"
    if len(base) >= target_chars:
        return base
    filler_parts = []
    i = 0
    total_len = len(base)
    while total_len < target_chars:
        line = _FILLER_LINE.format(i=i)
        filler_parts.append(line)
        total_len += len(line)
        i += 1
    return base + "".join(filler_parts)


# Calibrated once against a real call on this exact filler text/system
# prompt (8,066 chars -> Ollama-reported prompt_eval_count=2,062), not a
# generic assumption - ~3.91 chars/token for this specific content. Only
# used to pick a starting `target_chars` per bucket; every result still
# reports the real, Ollama-measured `actual_input_tokens` (spec section 4:
# never present an estimate as measured).
CHARS_PER_TOKEN_CALIBRATED = 3.91

# Char targets for spec section 10/12's ~2K/~4K/~8K actual-prompt-size
# buckets, derived from the calibration above.
TOKEN_BUCKET_CHAR_TARGETS = {
    "~2K": round(2000 * CHARS_PER_TOKEN_CALIBRATED),
    "~4K": round(4000 * CHARS_PER_TOKEN_CALIBRATED),
    "~8K": round(8000 * CHARS_PER_TOKEN_CALIBRATED),
}


@dataclass
class DirectCallResult:
    request_index: int
    configured_num_ctx: int | None
    configured_keep_alive: str | None
    prompt_chars: int
    prompt_bytes: int
    actual_input_tokens: int | None
    actual_output_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    success: bool
    timeout: bool
    error_category: str | None
    error_message_normalized: str | None
    provider_metadata: dict[str, Any] | None
    start_time: str
    end_time: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "provider_call",
            "path": "P0_direct_ollama",
            "request_index": self.request_index,
            "configured_num_ctx": self.configured_num_ctx,
            "configured_keep_alive": self.configured_keep_alive,
            "prompt_chars": self.prompt_chars,
            "prompt_bytes": self.prompt_bytes,
            "actual_input_tokens": self.actual_input_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "timeout": self.timeout,
            "error_category": self.error_category,
            "error_message_normalized": self.error_message_normalized,
            "provider_metadata": self.provider_metadata,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


def direct_ollama_call(
    client: OllamaClient,
    request_index: int,
    user_prompt: str,
    system_prompt: str = FIXED_WORKLOAD_SYSTEM_PROMPT,
) -> DirectCallResult:
    """M5E.1.2 bug fix: unlike `LLMController`/the agent adapters,
    `OllamaClient.generate_action` itself never catches the raw socket
    exception `urllib.request.urlopen(timeout=...)` raises on a real
    timeout (`TimeoutError`) - it propagates straight out. Every prior P0
    diagnostic call site (M5E.1.1's context matrix, this function) called
    it unguarded; M5E.1.1's 54/54-success matrix simply never happened to
    hit a real timeout, so the gap stayed latent until a genuine timeout
    during M5E.1.2's endurance run crashed the whole script instead of
    being recorded as a failed call. Now caught here, at the one shared
    call site every P0 diagnostic path goes through."""
    start_iso = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        response = client.generate_action(system_prompt, user_prompt, output_schema={})
    except Exception as exc:
        end_iso = datetime.now(timezone.utc).isoformat()
        latency_ms = (time.monotonic() - started) * 1000.0
        error_category = classify_provider_call_error("provider_error", str(exc))
        return DirectCallResult(
            request_index=request_index,
            configured_num_ctx=client.num_ctx,
            configured_keep_alive=client.keep_alive,
            prompt_chars=len(user_prompt),
            prompt_bytes=len(user_prompt.encode("utf-8")),
            actual_input_tokens=None,
            actual_output_tokens=None,
            total_tokens=None,
            latency_ms=latency_ms,
            success=False,
            timeout=(error_category == "provider_timeout"),
            error_category=error_category,
            error_message_normalized=normalize_provider_error_message(str(exc)),
            provider_metadata=None,
            start_time=start_iso,
            end_time=end_iso,
        )
    end_iso = datetime.now(timezone.utc).isoformat()
    error_category = classify_provider_call_error(response.error_type, response.error_message)
    return DirectCallResult(
        request_index=request_index,
        configured_num_ctx=client.num_ctx,
        configured_keep_alive=client.keep_alive,
        prompt_chars=len(user_prompt),
        prompt_bytes=len(user_prompt.encode("utf-8")),
        actual_input_tokens=response.input_tokens,
        actual_output_tokens=response.output_tokens,
        total_tokens=(
            (response.input_tokens + response.output_tokens)
            if (response.input_tokens is not None and response.output_tokens is not None)
            else None
        ),
        latency_ms=response.latency_ms or 0.0,
        success=(error_category is None),
        timeout=(error_category == "provider_timeout"),
        error_category=error_category,
        error_message_normalized=normalize_provider_error_message(response.error_message),
        provider_metadata=response.provider_metadata,
        start_time=start_iso,
        end_time=end_iso,
    )


@dataclass
class SequentialRunResult:
    label: str
    calls: list[DirectCallResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "calls": [c.to_dict() for c in self.calls]}


def is_prompt_safe_for_context(prompt_token_estimate: int, num_ctx: int, headroom_tokens: int = 512) -> bool:
    """Spec section 11: never test an actual prompt larger than the
    configured `num_ctx`, and leave headroom for output tokens/formatting
    overhead - a prompt within `headroom_tokens` of the ceiling is not
    "safely below" it."""
    return prompt_token_estimate + headroom_tokens <= num_ctx


# Spec section 12's recommended minimum matrix, expressed as (bucket
# label, num_ctx) pairs - built once here so the live driver and tests
# share one definition instead of two independently-drifting lists.
RECOMMENDED_MATRIX_CELLS: list[tuple[str, int]] = [
    ("~2K", 4096),
    ("~2K", 8192),
    ("~2K", 16384),
    ("~2K", 32768),
    ("~4K", 8192),
    ("~4K", 16384),
    ("~4K", 32768),
    ("~8K", 16384),
    ("~8K", 32768),
]


SHORT_WAIT_RETRY_SECONDS = 5.0  # spec section 24: "use a documented interval" - fixed, not tuned per-run.


def attempt_recovery_ladder(
    *,
    system_prompt: str,
    user_prompt: str,
    num_ctx: int | None,
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma4:latest",
    timeout: float = 60.0,
    keep_alive: str | None = None,
    request_fn: Any = None,
    wait_seconds: float | None = None,
) -> dict[str, Any]:
    """M5E.1.1 spec sections 23-25: after a real provider failure, try -
    in order, stopping at the first that recovers - an immediate retry
    (same client), a short documented wait then retry (same client), and
    a fresh-wrapper retry (brand-new OllamaClient, same request). This is
    diagnostic only (never an automatic multi-retry policy baked into
    gameplay, spec section 23) - callers invoke it once, opportunistically,
    when the matrix runner above already observed a real failure.
    """
    steps: dict[str, Any] = {}
    wait_seconds = SHORT_WAIT_RETRY_SECONDS if wait_seconds is None else wait_seconds

    def _call() -> DirectCallResult:
        client = OllamaClient(
            model_name=model_name, base_url=base_url, timeout=timeout, num_ctx=num_ctx, keep_alive=keep_alive, request_fn=request_fn
        )
        return direct_ollama_call(client, request_index=0, user_prompt=user_prompt, system_prompt=system_prompt)

    immediate = _call()
    steps["immediate_retry"] = immediate.to_dict()
    if immediate.success:
        steps["recovered_at"] = "immediate_retry"
        return steps

    time.sleep(wait_seconds)
    waited = _call()
    steps["short_wait_retry"] = waited.to_dict()
    steps["short_wait_retry_seconds"] = wait_seconds
    if waited.success:
        steps["recovered_at"] = "short_wait_retry"
        return steps

    fresh = _call()  # a brand-new OllamaClient every _call() already - "fresh wrapper" is the same call again
    steps["fresh_wrapper_retry"] = fresh.to_dict()
    if fresh.success:
        steps["recovered_at"] = "fresh_wrapper_retry"
        return steps

    steps["recovered_at"] = None
    return steps


def unload_model(
    base_url: str = "http://localhost:11434", model_name: str = "gemma4:latest", timeout: float = 30.0, request_fn: Any = None
) -> bool:
    """M5E.1.1 spec section 26: a safe, officially-supported Ollama
    mechanism (`keep_alive: 0` on a trivial request unloads the model
    immediately afterward) - not a process restart (spec section 27,
    which this module does not implement: killing/restarting the Ollama
    process is outside this project's safe automated scope, documented as
    a skipped/optional step)."""
    client = OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, keep_alive="0s", request_fn=request_fn)
    call = direct_ollama_call(client, request_index=0, user_prompt='{"ping": true}', system_prompt="You output JSON only.")
    return call.success


def run_sequential_direct_calls(
    *,
    label: str,
    n_requests: int,
    num_ctx: int | None,
    prompt_chars_target: int,
    base_url: str = "http://localhost:11434",
    model_name: str = "gemma4:latest",
    timeout: float = 60.0,
    keep_alive: str | None = None,
    on_call: Any = None,
) -> SequentialRunResult:
    """Runs `n_requests` sequential P0 calls against a fixed-size prompt
    (spec section 8-9). A fresh `OllamaClient` per call (matching how
    every other M5E.1/M5E.1.1 diagnostic isolates call-to-call state -
    the client itself is stateless/connection-per-call already, so this
    is purely about not sharing Python-side bookkeeping across requests)."""
    prompt = build_padded_user_prompt(prompt_chars_target)
    result = SequentialRunResult(label=label)
    for i in range(1, n_requests + 1):
        client = OllamaClient(model_name=model_name, base_url=base_url, timeout=timeout, num_ctx=num_ctx, keep_alive=keep_alive)
        call = direct_ollama_call(client, i, prompt)
        result.calls.append(call)
        if on_call is not None:
            on_call(i, call)
    return result
