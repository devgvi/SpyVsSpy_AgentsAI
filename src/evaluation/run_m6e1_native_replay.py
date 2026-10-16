"""M6E.1 Phases B-D (spec sections 21, 24, 26): live Native replay across
all 6 duplicate-count/location/format conditions, for each of the 3
full-treatment scenarios (S1, S2, S6). One driver since all three phases
share the identical call pattern - only which (scenario, condition)
cells are compared differs at analysis time, matching M6E's own driver
precedent.

    python -m src.evaluation.run_m6e1_native_replay --region us-east-1
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

from src.evaluation.duplicate_fact_variants import CONDITIONS, build_variant, multiset_fact_hash, representation_hash, unique_fact_hash
from src.evaluation.m6e1_scenarios import FULL_TREATMENT_FAMILIES, build_full_treatment_scenarios
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import NATIVE_TOOL_LIKE_SYSTEM_PROMPT, VALID_ACTION_NAMES, validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6e1-lossless-compaction-counterbalancing-v1")
SAMPLES_PER_CELL = 5

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


def run_one(region: str, family: str, scenario, condition: str, sample: int) -> dict:
    from src.llm.clients.bedrock_client import BedrockClient

    variant = build_variant(condition, scenario.observation_payload, scenario.memory_payload, scenario.fact)
    prompt_json = variant.to_json()

    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
    start = time.monotonic()
    response = client.generate_action(NATIVE_TOOL_LIKE_SYSTEM_PROMPT, prompt_json, _OUTPUT_SCHEMA)
    latency_ms = response.latency_ms if getattr(response, "latency_ms", None) is not None else (time.monotonic() - start) * 1000.0
    raw = response.payload if isinstance(response.payload, dict) else {}
    v = validate_model_output(scenario.observation, raw, scenario.legal_actions)

    return {
        "runtime": "native", "scenario_family": family, "scenario_id": scenario.scenario_id, "condition": condition, "sample": sample,
        "copy_count": variant.copy_count, "locations": list(variant.locations), "format_variant": variant.format_variant,
        "unique_fact_hash": unique_fact_hash(variant.observation_payload, variant.memory_payload, scenario.fact),
        "multiset_fact_hash": multiset_fact_hash(variant.copy_count, variant.locations, scenario.fact),
        "representation_hash": representation_hash(prompt_json), "characters": len(prompt_json),
        "action": raw.get("action"), "target_id": raw.get("target_id"), "reason": raw.get("reason"),
        "validation_status": "valid" if v.valid else f"invalid:{v.error_type}",
        "latency_ms": latency_ms, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
    }


def run(region: str) -> dict:
    scenarios = build_full_treatment_scenarios()
    rows: list[dict] = []
    for family in FULL_TREATMENT_FAMILIES:
        scenario = scenarios[family]
        for condition in CONDITIONS:
            for i in range(SAMPLES_PER_CELL):
                row = run_one(region, family, scenario, condition, i)
                rows.append(row)
                print(f"[NATIVE {family}/{condition}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']})")
    _write_jsonl(os.path.join(OUT_DIR, "native_duplicate_count_trials.jsonl"), rows)
    return {"rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
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
    result = run(region)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] M6E.1 Native replay complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
