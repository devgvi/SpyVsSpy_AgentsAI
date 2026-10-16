"""M5E.1.2: automated live-diagnostic entry point.

    python -m src.evaluation.run_m5e1_2

Stages, in order (spec sections 13-17):

A. baseline resource snapshot, before touching Ollama.
B. warm the model with one unmeasured call, snapshot again.
C. P0 direct-Ollama endurance (30 sequential production-shaped calls).
D. P1 native_tool_like endurance (30 sequential decisions, real adapter).
E. P2 Strands v2 endurance (30 sequential decisions, real adapter).

A continuous background resource sampler runs throughout C/D/E. On each
path's first failure, a post-timeout probe sequence and (if the provider
remains unresponsive at the probe's end) the recovery ladder run
immediately - never queued behind more normal-workload calls (spec
section 31).

Headless (SDL_VIDEODRIVER=dummy). Writes everything under
evaluation_runs/m5e1_2-resource-saturation-v1/ and never touches any prior
M5D/M5E/M5E.1/M5E.1.1/M5E.2 output directory. Staged escalation only
(spec section 12) - this script runs the initial 30-call target and stops;
extending to 50 calls or a quiet-machine comparison is a deliberate,
separate follow-up decision made after reading these results, not
automatic.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import sys
from datetime import datetime, timezone

from src.evaluation.endurance import (
    BackgroundResourceSampler,
    compute_latency_degradation,
    extract_first_failure_window,
    run_wrapper_endurance,
    run_p0_endurance,
)
from src.evaluation.ollama_health import snapshot_ollama_health
from src.evaluation.resource_telemetry import find_ollama_pid, sample_full_resource_state
from src.llm.clients.ollama_client import OllamaClient
from src.llm.telemetry import TelemetryLogger

MODEL = "gemma4:latest"
BASE_URL = "http://localhost:11434"
NUM_CTX = 16384
LLM_TIMEOUT = 60.0
N_REQUESTS_PER_PATH = 30
OUT_DIR = os.path.join("evaluation_runs", "m5e1_2-resource-saturation-v1")


def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    requests_path = os.path.join(OUT_DIR, "requests.jsonl")
    resource_samples_path = os.path.join(OUT_DIR, "resource_samples.jsonl")
    timeouts_path = os.path.join(OUT_DIR, "timeouts.jsonl")
    recovery_probes_path = os.path.join(OUT_DIR, "recovery_probes.jsonl")

    print(f"=== M5E.1.2 diagnostics starting {datetime.now(timezone.utc).isoformat()} ===")
    print("[QUIET MACHINE] not controlled - normal desktop state, documented per spec section 38/40, not silently redefined as the production environment")

    # -- Stage A: baseline, before touching Ollama --------------------------
    ollama_pid_before = find_ollama_pid()
    stage_a = sample_full_resource_state("stage_a_baseline", ollama_pid_before, BASE_URL, MODEL)
    _append_jsonl(resource_samples_path, stage_a)
    print("[STAGE A] baseline:", json.dumps(stage_a["system"]))

    # -- Stage B: warm the model, snapshot again -----------------------------
    warm_client = OllamaClient(model_name=MODEL, base_url=BASE_URL, timeout=LLM_TIMEOUT, num_ctx=NUM_CTX)
    warm_client.generate_action("You output JSON only.", '{"ping": true}', {})
    ollama_pid = find_ollama_pid()
    stage_b = sample_full_resource_state("stage_b_warm", ollama_pid, BASE_URL, MODEL)
    _append_jsonl(resource_samples_path, stage_b)
    print("[STAGE B] warm:", json.dumps(stage_b["system"]), json.dumps(stage_b["model"]))

    path_summaries = {}

    # -- Stage C: P0 direct-Ollama endurance ---------------------------------
    print(f"[STAGE C] P0 direct Ollama endurance starting {datetime.now(timezone.utc).isoformat()}")
    sampler = BackgroundResourceSampler("stage_c_p0", ollama_pid, interval_seconds=3.0)
    sampler.start()
    p0_records = []

    def _on_p0_call(i, record_or_dict):
        if i < 0:
            # supplementary probe/recovery payload for the |i|-th request
            _append_jsonl(timeouts_path, record_or_dict)
            print(f"[STAGE C] request {-i} FAILED - probe/recovery sequence recorded")
            return
        d = record_or_dict.to_dict()
        p0_records.append(d)
        _append_jsonl(requests_path, d)
        if i % 5 == 0 or not d["success"]:
            print(f"[STAGE C] #{i} success={d['success']} latency={d['wall_latency_ms']:.0f}ms tokens={d['actual_input_tokens']}")

    run_p0_endurance(n_requests=N_REQUESTS_PER_PATH, num_ctx=NUM_CTX, base_url=BASE_URL, model_name=MODEL, timeout=LLM_TIMEOUT, seed=951, on_call=_on_p0_call)
    continuous_samples = sampler.stop()
    for s in continuous_samples:
        _append_jsonl(resource_samples_path, s)
    path_summaries["P0_direct_ollama"] = {
        "requests": len(p0_records),
        "successes": sum(1 for r in p0_records if r["success"]),
        "first_failure_window": extract_first_failure_window(p0_records),
        "latency_degradation": compute_latency_degradation(p0_records),
    }
    print(f"[STAGE C] finished: {path_summaries['P0_direct_ollama']['successes']}/{len(p0_records)} successful")

    # -- Stage D: P1 native_tool_like endurance ------------------------------
    print(f"[STAGE D] P1 native_tool_like endurance starting {datetime.now(timezone.utc).isoformat()}")
    sampler = BackgroundResourceSampler("stage_d_p1", ollama_pid, interval_seconds=3.0)
    sampler.start()
    p1_decisions = []
    wrapper_telemetry = TelemetryLogger(os.path.join(OUT_DIR, "wrapper_decisions.jsonl"))

    def _on_p1_call(i, record):
        p1_decisions.append(record)
        if i % 5 == 0 or not record.get("final_model_action_valid"):
            print(f"[STAGE D] #{i} valid={record.get('final_model_action_valid')} input_tokens={record.get('input_tokens')}")

    p1_out = run_wrapper_endurance(
        runtime="native_tool_like", n_requests=N_REQUESTS_PER_PATH, num_ctx=NUM_CTX, base_url=BASE_URL, model_name=MODEL,
        timeout=LLM_TIMEOUT, seed=952, telemetry=wrapper_telemetry, on_call=_on_p1_call,
    )
    continuous_samples = sampler.stop()
    for s in continuous_samples:
        _append_jsonl(resource_samples_path, s)
    if p1_out["supplementary"]:
        _append_jsonl(recovery_probes_path, {"path": "P1_native_tool_like", **p1_out["supplementary"]})
    p1_success = sum(1 for d in p1_decisions if d.get("final_model_action_valid"))
    path_summaries["P1_native_tool_like"] = {"requests": len(p1_decisions), "successes": p1_success, "supplementary": bool(p1_out["supplementary"])}
    print(f"[STAGE D] finished: {p1_success}/{len(p1_decisions)} valid")

    # -- Stage E: P2 Strands v2 endurance ------------------------------------
    print(f"[STAGE E] P2 strands endurance starting {datetime.now(timezone.utc).isoformat()}")
    sampler = BackgroundResourceSampler("stage_e_p2", ollama_pid, interval_seconds=3.0)
    sampler.start()
    p2_decisions = []

    def _on_p2_call(i, record):
        p2_decisions.append(record)
        if i % 5 == 0 or not record.get("final_model_action_valid"):
            print(f"[STAGE E] #{i} valid={record.get('final_model_action_valid')} input_tokens={record.get('input_tokens')}")

    p2_out = run_wrapper_endurance(
        runtime="strands", n_requests=N_REQUESTS_PER_PATH, num_ctx=NUM_CTX, base_url=BASE_URL, model_name=MODEL,
        timeout=LLM_TIMEOUT, seed=953, telemetry=wrapper_telemetry, on_call=_on_p2_call,
    )
    continuous_samples = sampler.stop()
    for s in continuous_samples:
        _append_jsonl(resource_samples_path, s)
    if p2_out["supplementary"]:
        _append_jsonl(recovery_probes_path, {"path": "P2_strands", **p2_out["supplementary"]})
    p2_success = sum(1 for d in p2_decisions if d.get("final_model_action_valid"))
    path_summaries["P2_strands"] = {"requests": len(p2_decisions), "successes": p2_success, "supplementary": bool(p2_out["supplementary"])}
    print(f"[STAGE E] finished: {p2_success}/{len(p2_decisions)} valid")
    wrapper_telemetry.close()

    health_after = snapshot_ollama_health(base_url=BASE_URL, model_name=MODEL)
    stage_final = sample_full_resource_state("final", find_ollama_pid(), BASE_URL, MODEL)
    _append_jsonl(resource_samples_path, stage_final)

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"path_summaries": path_summaries, "health_after": health_after, "stage_a": stage_a, "stage_b": stage_b, "stage_final": stage_final},
            f, indent=2, default=str,
        )

    print(json.dumps(path_summaries, indent=2, default=str))
    print(f"\nOutputs written under: {OUT_DIR}/")
    print(f"=== M5E.1.2 diagnostics finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
