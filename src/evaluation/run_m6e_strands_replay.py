"""M6E Stage 4 (spec sections 40-43): focused Strands replication, run
only after Native analysis (Stages 1-3) is complete. Covers exactly the
subset spec section 40 asks for: RECENT_FAILED_ACTION (the mandatory
memory-positive control), one negative control (NO_HISTORY_RELEVANT), and
whichever lossless conditions Native analysis identified as the clearest
stable/behavior-changing cases - pass them via `--scenarios`/`--conditions`
after reviewing `native_trials.jsonl`, rather than hardcoding a guess here.
Gated on RP01-RP18 remaining clean (spec section 41) and stays entirely in
ordinary/memory paths (spec section 42 - never repair/tool-action, which
M6F already showed are runtime-sensitive).

    python -m src.evaluation.run_m6e_strands_replay --region us-east-1 \\
        --scenarios S1_RECENT_FAILED_ACTION S0_NO_HISTORY_RELEVANT \\
        --conditions C0_FULL_CONTEXT C1_STRUCTURAL_COMPACT
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
from src.evaluation.context_compaction import apply_condition, representation_hash
from src.evaluation.m6e_scenarios import build_all_scenarios, build_compact_observation_payload
from src.evaluation.request_fact_model import extract_request_facts, request_fact_hash
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6e-context-compaction-v1")
SAMPLES_PER_CELL = 5


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


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _run_one(region: str, scenario_family: str, scenario, condition: str, sample: int) -> dict:
    from src.llm.telemetry import TelemetryLogger

    obs = build_compact_observation_payload(scenario)
    result = apply_condition(condition, obs, scenario.memory_payload)
    facts = extract_request_facts(result.observation_payload, result.memory_payload)
    payload = dict(result.observation_payload)
    payload["strategy_memory"] = result.memory_payload

    adapter = create_agent_runtime(
        "strands", actor_id="white", telemetry=TelemetryLogger(),
        episode_id=f"m6e-strands-{scenario_family}-{condition}-{sample}", provider_name="bedrock",
        model_name=MODEL_ID, execution_mode="evaluation", llm_region=region, llm_timeout=60.0, personality_profile=None,
    )
    invocation = adapter._invoke_strands(payload, repair_instruction=None)
    raw = invocation.proposal if isinstance(invocation.proposal, dict) else {}
    v = validate_model_output(scenario.observation, raw, scenario.legal_actions)

    return {
        "runtime": "strands", "scenario_family": scenario_family, "scenario_id": scenario.scenario_id,
        "condition": condition, "sample": sample,
        "request_fact_hash": request_fact_hash(facts), "representation_hash": representation_hash(json.dumps(payload, sort_keys=True)),
        "action": raw.get("action"), "target_id": raw.get("target_id"), "reason": raw.get("reason"),
        "validation_status": "valid" if v.valid else f"invalid:{v.error_type}",
        "latency_ms": invocation.latency_ms, "input_tokens": invocation.input_tokens, "output_tokens": invocation.output_tokens,
    }


def run(region: str, scenario_families: list[str], conditions: list[str]) -> dict:
    scenarios = build_all_scenarios()
    rows: list[dict] = []
    for family in scenario_families:
        scenario = scenarios[family]
        for condition in conditions:
            for i in range(SAMPLES_PER_CELL):
                row = _run_one(region, family, scenario, condition, i)
                rows.append(row)
                print(f"[STRANDS {family}/{condition}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']})")
    _write_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"), rows)
    return {"rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--scenarios", nargs="+", required=False, default=["S1_RECENT_FAILED_ACTION", "S0_NO_HISTORY_RELEVANT"])
    parser.add_argument("--conditions", nargs="+", required=False, default=["C0_FULL_CONTEXT", "C1_STRUCTURAL_COMPACT"])
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
    result = run(region, args.scenarios, args.conditions)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] M6E Strands replication complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
