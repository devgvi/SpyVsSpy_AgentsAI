"""M6E Stages 1-3 (spec sections 29, 31, 33): live Native replay.

Stage 1 (C0 vs C1) and Stage 2 (C0 vs C2) share the same C0 samples - C0
is run exactly once per scenario, never twice, since re-running it a
second time for Stage 2 would be the identical experiment at double cost.
Stage 3 (C3A/C3B history truncation) runs only on the memory-relevant
snapshots (spec section 33), after C1/C2 are already understood (this
script is meant to run after `run_m6e_phase_b_audit.py`'s static findings
are reviewed - spec section 24: "Only after C1/C2 are understood").

    python -m src.evaluation.run_m6e_native_replay --region us-east-1
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.evaluation.context_compaction import (
    C0_FULL_CONTEXT,
    C1_STRUCTURAL_COMPACT,
    C2_DEDUPLICATED,
    C3A_LAST_3,
    C3B_LAST_1,
    apply_condition,
    representation_hash,
)
from src.evaluation.m6e_scenarios import MEMORY_RELEVANT_FAMILIES, SCENARIO_FAMILIES, build_all_scenarios, build_compact_observation_payload
from src.evaluation.request_fact_model import extract_request_facts, request_fact_hash
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import NATIVE_TOOL_LIKE_SYSTEM_PROMPT, VALID_ACTION_NAMES, validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6e-context-compaction-v1")
SAMPLES_PER_CELL = 5
STAGE_1_2_CONDITIONS = (C0_FULL_CONTEXT, C1_STRUCTURAL_COMPACT, C2_DEDUPLICATED)
STAGE_3_CONDITIONS = (C3A_LAST_3, C3B_LAST_1)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(VALID_ACTION_NAMES)},
        "target_id": {"type": ["string", "null"]},
        "trap_type": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["action"],
}


def _verify_sleep_assertion() -> bool:
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
        return "PreventUserIdleSystemSleep" in output and "caffeinate" in output
    except Exception:  # pragma: no cover
        return False


def _preflight(region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight_native_replay", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _run_one(region: str, scenario_family: str, scenario, condition: str, sample: int) -> dict:
    from src.llm.clients.bedrock_client import BedrockClient

    obs = build_compact_observation_payload(scenario)
    result = apply_condition(condition, obs, scenario.memory_payload)
    facts = extract_request_facts(result.observation_payload, result.memory_payload)
    prompt_json = result.to_json()

    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
    start = time.monotonic()
    response = client.generate_action(NATIVE_TOOL_LIKE_SYSTEM_PROMPT, prompt_json, _OUTPUT_SCHEMA)
    latency_ms = response.latency_ms if getattr(response, "latency_ms", None) is not None else (time.monotonic() - start) * 1000.0
    raw = response.payload if isinstance(response.payload, dict) else {}
    v = validate_model_output(scenario.observation, raw, scenario.legal_actions)

    return {
        "scenario_family": scenario_family, "scenario_id": scenario.scenario_id, "condition": condition, "sample": sample,
        "request_fact_hash": request_fact_hash(facts), "representation_hash": representation_hash(prompt_json),
        "characters": len(prompt_json),
        "action": raw.get("action"), "target_id": raw.get("target_id"), "reason": raw.get("reason"),
        "validation_status": "valid" if v.valid else f"invalid:{v.error_type}",
        "latency_ms": latency_ms, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
    }


def run(region: str, stage3: bool) -> dict:
    scenarios = build_all_scenarios()
    rows: list[dict] = []

    for family in SCENARIO_FAMILIES:
        scenario = scenarios[family]
        for condition in STAGE_1_2_CONDITIONS:
            for i in range(SAMPLES_PER_CELL):
                row = _run_one(region, family, scenario, condition, i)
                rows.append(row)
                print(f"[NATIVE {family}/{condition}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']})")

    if stage3:
        for family in MEMORY_RELEVANT_FAMILIES:
            scenario = scenarios[family]
            for condition in STAGE_3_CONDITIONS:
                for i in range(SAMPLES_PER_CELL):
                    row = _run_one(region, family, scenario, condition, i)
                    rows.append(row)
                    print(f"[NATIVE {family}/{condition}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']})")

    _write_jsonl(os.path.join(OUT_DIR, "native_trials.jsonl"), rows)
    return {"rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-stage-3", action="store_true")
    args = parser.parse_args()

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
    result = run(region, stage3=not args.skip_stage_3)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] M6E Native replay complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
