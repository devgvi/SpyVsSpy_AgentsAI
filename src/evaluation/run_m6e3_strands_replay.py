"""M6E.3 Phase D (spec sections 39-42): Strands replication, run only
after Native analysis. Native found a clean NO_HISTORY_WINDOW_EFFECT for
all three scenarios (no reproducible boundary, no surgical add-back
needed) - so per spec section 39's own adapted guidance ("strongest
LONG_HISTORY boundary" / "key surgical add-back if one exists" / "S1
positive-control boundary" / "one S0 negative-control comparison"), this
replicates the H_FULL vs H0 endpoints for all three scenarios: the
starkest possible comparison (all retained history vs none), which is
exactly what would show the effect most clearly on Strands if the Native
null were somehow runtime-specific.

    python -m src.evaluation.run_m6e3_strands_replay --region us-east-1
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src.agents.registry import create_agent_runtime
from src.evaluation.history_window import history_fact_hash, history_sequence_hash, history_window_hash
from src.evaluation.m6e3_scenarios import PRIMARY_FAMILIES, build_primary_scenarios, build_window, repeats_known_failure
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6e3-memory-horizon-v1")
SAMPLES_PER_CELL = 5
WINDOWS_TO_REPLICATE = ("H_FULL", "H0")


def _verify_sleep_assertion() -> bool:
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
        return "PreventUserIdleSystemSleep" in output and "caffeinate" in output
    except Exception:  # pragma: no cover
        return False


def _rp_manifest_clean() -> bool:
    from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST

    return all(e.status != FAIL for e in RUNTIME_PARITY_MANIFEST)


def _preflight(region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight_strands_replay", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _run_one(region: str, family: str, scenario, window: str, sample: int) -> dict:
    from src.llm.telemetry import TelemetryLogger

    r = build_window(scenario, window)
    payload = {**scenario.observation_payload, "strategy_memory": r.memory_payload}
    adapter = create_agent_runtime(
        "strands", actor_id="white", telemetry=TelemetryLogger(),
        episode_id=f"m6e3-strands-{family}-{window}-{sample}", provider_name="bedrock", model_name=MODEL_ID,
        execution_mode="evaluation", llm_region=region, llm_timeout=60.0, personality_profile=None,
    )
    invocation = adapter._invoke_strands(payload, repair_instruction=None)
    raw = invocation.proposal if isinstance(invocation.proposal, dict) else {}
    v = validate_model_output(scenario.observation, raw, scenario.legal_actions)
    repeat_failure = repeats_known_failure(raw.get("action"), raw.get("target_id")) if family == "S6_LONG_HISTORY" else None

    return {
        "runtime": "strands", "scenario_family": family, "scenario_id": scenario.scenario_id, "window": window, "sample": sample,
        "history_window_hash": history_window_hash(r.memory_payload), "history_sequence_hash": history_sequence_hash(r.memory_payload),
        "history_fact_hash": history_fact_hash(r.memory_payload),
        "action": raw.get("action"), "target_id": raw.get("target_id"), "reason": raw.get("reason"),
        "repeats_known_failure": repeat_failure,
        "validation_status": "valid" if v.valid else f"invalid:{v.error_type}",
        "latency_ms": invocation.latency_ms, "input_tokens": invocation.input_tokens, "output_tokens": invocation.output_tokens,
    }


def run(region: str) -> dict:
    scenarios = build_primary_scenarios()
    rows: list[dict] = []
    for family in PRIMARY_FAMILIES:
        scenario = scenarios[family]
        for window in WINDOWS_TO_REPLICATE:
            for i in range(SAMPLES_PER_CELL):
                row = _run_one(region, family, scenario, window, i)
                rows.append(row)
                print(f"[STRANDS {family}/{window}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']}) repeat_failure={row['repeats_known_failure']}")
    with open(os.path.join(OUT_DIR, "strands_trials.jsonl"), "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return {"rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not _rp_manifest_clean():
        print("[FATAL] RP01-RP18 manifest has an unexplained FAIL - refusing to start Strands live calls.", file=sys.stderr)
        return 1

    from src.llm.clients.bedrock_client import resolve_bedrock_region

    region = resolve_bedrock_region(args.region)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1
    if not _preflight(region):
        print("[FATAL] Bedrock preflight failed.", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[OK] Preflight passed.")
        return 0

    start = datetime.now(timezone.utc)
    result = run(region)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] M6E.3 Strands replication complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
