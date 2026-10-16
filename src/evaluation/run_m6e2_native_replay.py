"""M6E.2 Phase B (spec section 33): live Native factorial replay for the
cells NOT already reused from M6E.1 (`run_m6e2_reuse.py`) - S2's and S1's
3-4 genuinely new cells, plus S0's full 8-cell cube (S0 never received
fact-injection treatment in M6E.1).

    python -m src.evaluation.run_m6e2_native_replay --region us-east-1
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

from src.evaluation.duplicate_fact_variants import multiset_fact_hash, representation_hash, unique_fact_hash
from src.evaluation.m6e2_factorial import FACTORIAL_CELLS, build_cell, build_primary_scenarios, duplicate_location_hash, format_hash
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import NATIVE_TOOL_LIKE_SYSTEM_PROMPT, VALID_ACTION_NAMES, validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6e2-representation-interaction-matrix-v1")
SAMPLES_PER_CELL = 5

# Cells still needed live per scenario (everything else was reused from M6E.1).
NEW_CELLS_BY_FAMILY = {
    "S2_KNOWN_LOCKED_OBJECT": [c for c, cond in FACTORIAL_CELLS.items() if cond in ("A0B1C0_SINGLE_AUX_FULL", "A0B1C1_SINGLE_AUX_COMPACT", "A1B1C1_DOUBLE_AUX_COMPACT")],
    "S1_RECENT_FAILED_ACTION": [c for c, cond in FACTORIAL_CELLS.items() if cond in ("I1_SINGLE_COPY_COMPACT", "A0B1C0_SINGLE_AUX_FULL", "A0B1C1_SINGLE_AUX_COMPACT", "A1B1C1_DOUBLE_AUX_COMPACT")],
    "S0_NO_HISTORY_RELEVANT": list(FACTORIAL_CELLS.keys()),
}

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


def run_one(region: str, family: str, scenario, cell: tuple, sample: int) -> dict:
    from src.llm.clients.bedrock_client import BedrockClient

    variant = build_cell(cell, scenario)
    prompt_json = variant.to_json()

    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
    start = time.monotonic()
    response = client.generate_action(NATIVE_TOOL_LIKE_SYSTEM_PROMPT, prompt_json, _OUTPUT_SCHEMA)
    latency_ms = response.latency_ms if getattr(response, "latency_ms", None) is not None else (time.monotonic() - start) * 1000.0
    raw = response.payload if isinstance(response.payload, dict) else {}
    v = validate_model_output(scenario.observation, raw, scenario.legal_actions)

    return {
        "runtime": "native", "scenario_family": family, "scenario_id": scenario.scenario_id,
        "condition": FACTORIAL_CELLS[cell], "cell": list(cell), "sample": sample,
        "copy_count": variant.copy_count, "locations": list(variant.locations), "format_variant": variant.format_variant,
        "unique_fact_hash": unique_fact_hash(variant.observation_payload, variant.memory_payload, scenario.fact),
        "multiset_fact_hash": multiset_fact_hash(variant.copy_count, variant.locations, scenario.fact),
        "duplicate_location_hash": duplicate_location_hash(variant.locations), "format_hash": format_hash(variant.format_variant),
        "representation_hash": representation_hash(prompt_json), "characters": len(prompt_json),
        "action": raw.get("action"), "target_id": raw.get("target_id"), "reason": raw.get("reason"),
        "validation_status": "valid" if v.valid else f"invalid:{v.error_type}",
        "latency_ms": latency_ms, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
    }


def run(region: str) -> dict:
    scenarios = build_primary_scenarios()
    rows: list[dict] = []
    for family, cells in NEW_CELLS_BY_FAMILY.items():
        scenario = scenarios[family]
        for cell in cells:
            for i in range(SAMPLES_PER_CELL):
                row = run_one(region, family, scenario, cell, i)
                rows.append(row)
                print(f"[NATIVE {family}/{FACTORIAL_CELLS[cell]}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']})")
    _write_jsonl(os.path.join(OUT_DIR, "native_trials.jsonl"), rows)
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
    print(f"\n[DONE] M6E.2 Native replay complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
