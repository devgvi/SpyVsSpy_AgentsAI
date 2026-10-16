"""M5E.3: automated live-diagnostic entry point for the Amazon Bedrock
provider baseline.

    python -m src.evaluation.run_m5e3

Phases (spec sections 29-34), run only while `AWS_BEARER_TOKEN_BEDROCK` is
present - this script refuses to start (Phase B fails immediately) rather
than silently skipping ahead if it is not:

B. minimal live preflight (one tiny Converse call).
C. canonical action smoke test (5 real decisions against a fixed-shape
   synthetic Observation, through the real validator).
D. 15 sequential decision boundaries through the real production
   controller stack (native_tool_like).
E. one real active-opponent episode (native_tool_like + Bedrock vs
   scripted_active_v1), via the same EpisodeRunner/VariantSpec harness
   every other provider uses - no Bedrock-specific gameplay path.
F. 30 sequential production-like decisions (the long-horizon stability
   check, same growth-curve corpus shape M5E.1.2 used for Ollama).

Never logs/writes AWS_BEARER_TOKEN_BEDROCK anywhere (spec section 3) -
only `bedrock_auth_mode`/`token_present`. Headless (SDL_VIDEODRIVER=dummy).
Writes under evaluation_runs/m5e3-bedrock-provider-baseline-v1/ and never
touches any prior experiment output.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import sys
from datetime import datetime, timezone

from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.reporting import render_terminal
from src.evaluation.spec import ExperimentSpec, VariantSpec
from src.evaluation.synthetic_stability import PATTERN_S1_CONSTANT, PATTERN_S2_GROWING, run_synthetic_pattern
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.telemetry import TelemetryLogger

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
TIMEOUT = 60.0
OUT_DIR = os.path.join("evaluation_runs", "m5e3-bedrock-provider-baseline-v1")


def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _summarize_decisions(decisions: list[dict]) -> dict:
    n = len(decisions)
    valid = sum(1 for d in decisions if d.get("final_model_action_valid"))
    repaired = sum(1 for d in decisions if d.get("recovery_outcome") == "REPAIRED_VALID")
    fallback = sum(1 for d in decisions if d.get("fallback_used"))
    latencies = [d["latency_ms"] for d in decisions if d.get("latency_ms")]
    tokens_in = [d["input_tokens"] for d in decisions if d.get("input_tokens")]
    tokens_out = [d["output_tokens"] for d in decisions if d.get("output_tokens")]
    return {
        "calls_attempted": n,
        "first_pass_valid": sum(1 for d in decisions if d.get("recovery_outcome") == "FIRST_PASS_VALID"),
        "repairs": repaired,
        "fallbacks": fallback,
        "provider_failures": sum(1 for d in decisions if (d.get("initial_error_type") or "") not in ("", "missing_target", "illegal_target", "unknown_target", "out_of_range")),
        "median_latency_ms": (sorted(latencies)[len(latencies) // 2] if latencies else None),
        "max_latency_ms": (max(latencies) if latencies else None),
        "input_tokens_total": sum(tokens_in) if tokens_in else None,
        "output_tokens_total": sum(tokens_out) if tokens_out else None,
    }


def main(argv: list[str] | None = None) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    decisions_path = os.path.join(OUT_DIR, "decisions.jsonl")
    provider_events_path = os.path.join(OUT_DIR, "provider_events.jsonl")

    print(f"=== M5E.3 Bedrock baseline starting {datetime.now(timezone.utc).isoformat()} ===")

    if not bedrock_token_present():
        print("[PHASE B] FAILED PREFLIGHT: AWS_BEARER_TOKEN_BEDROCK is not set. Refusing to start any episode.")
        return 1

    try:
        region = resolve_bedrock_region()
    except Exception as exc:
        print(f"[PHASE B] FAILED PREFLIGHT: {exc}")
        return 1

    print(f"[CONFIG] provider=bedrock model_id={MODEL_ID} region={region} bedrock_auth_mode=bearer_token_env token_present=True")

    # -- Phase B: minimal live preflight -------------------------------------
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=TIMEOUT)
    _append_jsonl(provider_events_path, {"phase": "B_preflight", **preflight.to_dict()})
    print(f"[PHASE B] preflight: {preflight.to_dict()}")
    if not preflight.ok:
        print("[PHASE B] FAILED - stopping before any further phase.")
        return 1

    # -- Phase C: canonical action smoke test --------------------------------
    print(f"[PHASE C] canonical action smoke test starting {datetime.now(timezone.utc).isoformat()}")
    telemetry = TelemetryLogger(decisions_path)
    bedrock_client = BedrockClient(model_name=MODEL_ID, region=region, timeout=TIMEOUT)
    smoke_result = run_synthetic_pattern(
        pattern=PATTERN_S1_CONSTANT, runtime="native_tool_like", n_decisions=5, provider="bedrock", model=MODEL_ID,
        telemetry=telemetry, model_client=bedrock_client, episode_id="m5e3-phase-c-smoke", seed=1001,
        per_decision_wall_timeout=180.0,
    )
    smoke_summary = _summarize_decisions(smoke_result.decisions)
    print(f"[PHASE C] {smoke_summary}")
    if smoke_summary["first_pass_valid"] + smoke_summary["repairs"] < 5:
        print("[PHASE C] WARNING: fewer than 5/5 valid decisions - inspect representation/schema before proceeding, per spec section 30.")

    # -- Phase D: 15 sequential decision boundaries --------------------------
    print(f"[PHASE D] 15-decision native+Bedrock run starting {datetime.now(timezone.utc).isoformat()}")
    bedrock_client_d = BedrockClient(model_name=MODEL_ID, region=region, timeout=TIMEOUT)
    phase_d_result = run_synthetic_pattern(
        pattern=PATTERN_S2_GROWING, runtime="native_tool_like", n_decisions=15, provider="bedrock", model=MODEL_ID,
        telemetry=telemetry, model_client=bedrock_client_d, episode_id="m5e3-phase-d-15decisions", seed=1002,
        per_decision_wall_timeout=180.0,
    )
    phase_d_summary = _summarize_decisions(phase_d_result.decisions)
    print(f"[PHASE D] {phase_d_summary}")

    # -- Phase E: one active-opponent episode --------------------------------
    print(f"[PHASE E] active-opponent episode starting {datetime.now(timezone.utc).isoformat()}")
    spec = ExperimentSpec(
        experiment_id="m5e3-bedrock-provider-baseline-v1",
        variants=[
            VariantSpec(
                variant_id="A_native_tool_like_bedrock", agent_runtime="native_tool_like", provider="bedrock",
                model=MODEL_ID, llm_timeout=TIMEOUT, llm_region=region,
                opponent_type="scripted_active", opponent_profile="OPPORTUNIST",
            )
        ],
        seeds=[431],
        execution_mode="evaluation",
        max_episode_game_time=180.0,
        max_episode_wall_time=1800.0,
        max_decisions=15,
        max_runtime_failures=20,
    )
    runner = ExperimentRunner(episode_runner=EpisodeRunner(poll_sleep=0.0), output_root="evaluation_runs")
    phase_e_result = runner.run(spec)
    print(render_terminal(phase_e_result))

    # -- Phase F: 30-call long-horizon stability check -----------------------
    print(f"[PHASE F] 30-call long-horizon stability check starting {datetime.now(timezone.utc).isoformat()}")
    bedrock_client_f = BedrockClient(model_name=MODEL_ID, region=region, timeout=TIMEOUT)
    phase_f_result = run_synthetic_pattern(
        pattern=PATTERN_S2_GROWING, runtime="native_tool_like", n_decisions=30, provider="bedrock", model=MODEL_ID,
        telemetry=telemetry, model_client=bedrock_client_f, episode_id="m5e3-phase-f-30call", seed=1003,
        per_decision_wall_timeout=180.0,
    )
    phase_f_summary = _summarize_decisions(phase_f_result.decisions)
    print(f"[PHASE F] {phase_f_summary}")
    telemetry.close()

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "provider": "bedrock", "region": region, "model_id": MODEL_ID,
                "preflight": preflight.to_dict(),
                "phase_c_smoke": smoke_summary,
                "phase_d_15decisions": phase_d_summary,
                "phase_e_active_opponent": phase_e_result.to_summary_dict(),
                "phase_f_30call": phase_f_summary,
            },
            f, indent=2, default=str,
        )

    print(f"\nOutputs written under: {OUT_DIR}/")
    print(f"=== M5E.3 Bedrock baseline finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
