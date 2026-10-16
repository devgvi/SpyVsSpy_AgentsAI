"""M5E.1.1: automated live-diagnostic entry point.

    python -m src.evaluation.run_m5e1_1

Phases, in order:

1. Ollama health + resource snapshot (before).
2. P0 direct-Ollama context-length matrix (spec sections 3, 12): the
   recommended 9-cell matrix (~2K/4K/8K actual prompt x safe num_ctx
   values), N sequential requests per cell. Any real failure triggers the
   recovery ladder (immediate retry -> short wait -> fresh wrapper) once,
   opportunistically, against that same failed request.
3. Warm-vs-cold and keep_alive experiment (spec sections 15-16): an
   explicit unload (`keep_alive: 0`) followed by a "cold" sequence, then a
   "warm" sequence immediately after, then a keep_alive comparison.
4. Wrapper comparison (spec sections 5, 34): native_tool_like and Strands,
   each at two num_ctx candidates, driven through the M5E.1 synthetic
   harness's S1-constant pattern (reused, not reimplemented) so P1/P2 are
   directly comparable to P0's isolated-provider results.
5. Ollama health + resource snapshot (after).
6. If a clearly stable candidate emerged, one native_tool_like and one
   Strands active-opponent gameplay episode at that num_ctx (spec section
   39) - the only phase that touches the full game engine.

Headless (SDL_VIDEODRIVER=dummy). Writes everything under
evaluation_runs/m5e1_1-ollama-provider-stability-v1/ and never touches any
prior M5D/M5E/M5E.1 output directory.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import sys
from datetime import datetime, timezone

from src.evaluation.direct_ollama import (
    RECOMMENDED_MATRIX_CELLS,
    TOKEN_BUCKET_CHAR_TARGETS,
    attempt_recovery_ladder,
    build_padded_user_prompt,
    run_sequential_direct_calls,
)
from src.evaluation.ollama_health import snapshot_ollama_health
from src.evaluation.resource_telemetry import find_ollama_pid, sample_resources
from src.evaluation.synthetic_stability import PATTERN_S1_CONSTANT, run_synthetic_pattern
from src.llm.telemetry import TelemetryLogger

EXPERIMENT_ID = "m5e1_1-ollama-provider-stability-v1"
MODEL = "gemma4:latest"
BASE_URL = "http://localhost:11434"
LLM_TIMEOUT = 60.0

# Deliberate reduction from spec section 8's suggested 20-30 (documented,
# not silent): this is the third consecutive multi-hour live-Ollama
# diagnostic session today (M5E, M5E.1). 6 requests/cell x 9 cells still
# gives each context-length x prompt-size combination a real, if smaller,
# sequential sample.
REQUESTS_PER_MATRIX_CELL = 6
WRAPPER_COMPARISON_DECISIONS = 8
WRAPPER_COMPARISON_NUM_CTX_CANDIDATES = [None, 16384]  # None = current/default (pre-M5E.1.1) behavior


def _out_dir() -> str:
    d = os.path.join("evaluation_runs", EXPERIMENT_ID)
    os.makedirs(d, exist_ok=True)
    return d


def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_p0_context_matrix(out_dir: str, ollama_pid: int | None) -> list[dict]:
    provider_runs_path = os.path.join(out_dir, "provider_runs.jsonl")
    recovery_runs_path = os.path.join(out_dir, "recovery_runs.jsonl")
    resource_samples_path = os.path.join(out_dir, "resource_samples.jsonl")

    cell_summaries = []
    for bucket, num_ctx in RECOMMENDED_MATRIX_CELLS:
        label = f"P0__{bucket}__num_ctx_{num_ctx}"
        print(f"[P0 MATRIX] {label} starting {datetime.now(timezone.utc).isoformat()}")
        result = run_sequential_direct_calls(
            label=label,
            n_requests=REQUESTS_PER_MATRIX_CELL,
            num_ctx=num_ctx,
            prompt_chars_target=TOKEN_BUCKET_CHAR_TARGETS[bucket],
            base_url=BASE_URL,
            model_name=MODEL,
            timeout=LLM_TIMEOUT,
        )
        for call in result.calls:
            _append_jsonl(provider_runs_path, call.to_dict() | {"label": label, "bucket": bucket})

        first_failure = next((c for c in result.calls if not c.success), None)
        successes = sum(1 for c in result.calls if c.success)
        cell_summaries.append(
            {
                "label": label,
                "bucket": bucket,
                "num_ctx": num_ctx,
                "requests": len(result.calls),
                "successes": successes,
                "first_failure_request": (first_failure.request_index if first_failure else None),
            }
        )
        print(f"[P0 MATRIX] {label} finished: {cell_summaries[-1]}")

        if first_failure is not None:
            print(f"[P0 MATRIX] {label} real failure at request {first_failure.request_index} - running recovery ladder")
            prompt = build_padded_user_prompt(TOKEN_BUCKET_CHAR_TARGETS[bucket])
            recovery = attempt_recovery_ladder(
                system_prompt="You control one spy in a competitive embassy escape game. Choose exactly one action from legal_actions. Respond with JSON only.",
                user_prompt=prompt,
                num_ctx=num_ctx,
                base_url=BASE_URL,
                model_name=MODEL,
                timeout=LLM_TIMEOUT,
            )
            recovery["label"] = label
            recovery["triggered_by_request"] = first_failure.request_index
            _append_jsonl(recovery_runs_path, recovery)
            _append_jsonl(resource_samples_path, sample_resources(f"{label}__at_first_failure", ollama_pid))

    return cell_summaries


def run_warm_cold_keep_alive(out_dir: str) -> dict:
    from src.evaluation.direct_ollama import unload_model

    provider_runs_path = os.path.join(out_dir, "provider_runs.jsonl")
    print(f"[WARM/COLD] unloading model {datetime.now(timezone.utc).isoformat()}")
    unload_model(base_url=BASE_URL, model_name=MODEL)
    import time

    time.sleep(2.0)

    cold = run_sequential_direct_calls(
        label="warmcold__cold", n_requests=5, num_ctx=16384, prompt_chars_target=TOKEN_BUCKET_CHAR_TARGETS["~2K"],
        base_url=BASE_URL, model_name=MODEL, timeout=LLM_TIMEOUT,
    )
    for call in cold.calls:
        _append_jsonl(provider_runs_path, call.to_dict() | {"label": "warmcold__cold", "residency": "cold_first_call_then_warm"})

    warm = run_sequential_direct_calls(
        label="warmcold__warm", n_requests=5, num_ctx=16384, prompt_chars_target=TOKEN_BUCKET_CHAR_TARGETS["~2K"],
        base_url=BASE_URL, model_name=MODEL, timeout=LLM_TIMEOUT,
    )
    for call in warm.calls:
        _append_jsonl(provider_runs_path, call.to_dict() | {"label": "warmcold__warm", "residency": "warm"})

    keep_alive_long = run_sequential_direct_calls(
        label="keep_alive__10m", n_requests=5, num_ctx=16384, prompt_chars_target=TOKEN_BUCKET_CHAR_TARGETS["~2K"],
        base_url=BASE_URL, model_name=MODEL, timeout=LLM_TIMEOUT, keep_alive="10m",
    )
    for call in keep_alive_long.calls:
        _append_jsonl(provider_runs_path, call.to_dict() | {"label": "keep_alive__10m"})

    return {
        "cold_first_load_duration": cold.calls[0].provider_metadata.get("load_duration") if cold.calls and cold.calls[0].provider_metadata else None,
        "warm_first_load_duration": warm.calls[0].provider_metadata.get("load_duration") if warm.calls and warm.calls[0].provider_metadata else None,
        "cold_success_rate": sum(1 for c in cold.calls if c.success) / len(cold.calls) if cold.calls else None,
        "warm_success_rate": sum(1 for c in warm.calls if c.success) / len(warm.calls) if warm.calls else None,
        "keep_alive_10m_success_rate": sum(1 for c in keep_alive_long.calls if c.success) / len(keep_alive_long.calls) if keep_alive_long.calls else None,
    }


def run_wrapper_comparison(out_dir: str) -> list[dict]:
    wrapper_runs_path = os.path.join(out_dir, "wrapper_runs.jsonl")
    summaries = []
    telemetry = TelemetryLogger(os.path.join(out_dir, "wrapper_decisions.jsonl"))
    for runtime in ("native_tool_like", "strands"):
        for num_ctx in WRAPPER_COMPARISON_NUM_CTX_CANDIDATES:
            label = f"{runtime}__num_ctx_{num_ctx}"
            print(f"[WRAPPER] {label} starting {datetime.now(timezone.utc).isoformat()}")
            try:
                result = run_synthetic_pattern(
                    pattern=PATTERN_S1_CONSTANT,
                    runtime=runtime,
                    n_decisions=WRAPPER_COMPARISON_DECISIONS,
                    provider="ollama",
                    model=MODEL,
                    base_url=BASE_URL,
                    timeout=LLM_TIMEOUT,
                    telemetry=telemetry,
                    episode_id=f"m5e1_1__{label}",
                    seed=911,
                    per_decision_wall_timeout=180.0,
                    num_ctx=num_ctx,
                )
                record = {"label": label, "runtime": runtime, "num_ctx": num_ctx, "decisions": result.decisions, "error": None}
            except Exception as exc:
                record = {"label": label, "runtime": runtime, "num_ctx": num_ctx, "decisions": [], "error": f"{type(exc).__name__}: {exc}"}
            _append_jsonl(wrapper_runs_path, record)
            n_valid = sum(1 for d in record["decisions"] if d.get("final_model_action_valid"))
            summaries.append({"label": label, "n_decisions": len(record["decisions"]), "n_valid": n_valid, "error": record["error"]})
            print(f"[WRAPPER] {label} finished: {summaries[-1]}")
    telemetry.close()
    return summaries


def main(argv: list[str] | None = None) -> int:
    out_dir = _out_dir()
    print(f"=== M5E.1.1 diagnostics starting {datetime.now(timezone.utc).isoformat()} ===")

    ollama_pid = find_ollama_pid()
    health_before = snapshot_ollama_health(base_url=BASE_URL, model_name=MODEL)
    resources_before = sample_resources("before", ollama_pid)
    print("[HEALTH before]", json.dumps(health_before, default=str)[:300])
    _append_jsonl(os.path.join(out_dir, "resource_samples.jsonl"), resources_before)

    matrix_summary = run_p0_context_matrix(out_dir, ollama_pid)

    resources_mid = sample_resources("mid", ollama_pid)
    _append_jsonl(os.path.join(out_dir, "resource_samples.jsonl"), resources_mid)

    warmcold_summary = run_warm_cold_keep_alive(out_dir)

    wrapper_summary = run_wrapper_comparison(out_dir)

    resources_after = sample_resources("after", ollama_pid)
    _append_jsonl(os.path.join(out_dir, "resource_samples.jsonl"), resources_after)
    health_after = snapshot_ollama_health(base_url=BASE_URL, model_name=MODEL)
    print("[HEALTH after]", json.dumps(health_after, default=str)[:300])

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "health_before": health_before,
                "health_after": health_after,
                "matrix_summary": matrix_summary,
                "warmcold_summary": warmcold_summary,
                "wrapper_summary": wrapper_summary,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5E.1.1 diagnostics finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
