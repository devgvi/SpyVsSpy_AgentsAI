"""M6E.3 Phase B (spec section 24): live Native window sweep. Only
genuinely distinct (scenario, window) payloads are sampled - S1's and
S0's redundant windows (byte-identical to H_FULL, per the static audit)
are not re-run, since that would be the same experiment at extra cost.

    python -m src.evaluation.run_m6e3_native_replay --region us-east-1
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

from src.evaluation.history_window import WINDOW_ORDER, history_fact_hash, history_sequence_hash, history_window_hash
from src.evaluation.m6e3_scenarios import PRIMARY_FAMILIES, build_primary_scenarios, build_window, repeats_known_failure
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import NATIVE_TOOL_LIKE_SYSTEM_PROMPT, VALID_ACTION_NAMES, validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6e3-memory-horizon-v1")
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


def _distinct_windows_by_family() -> dict[str, list[str]]:
    scenarios = build_primary_scenarios()
    out: dict[str, list[str]] = {}
    for family in PRIMARY_FAMILIES:
        s = scenarios[family]
        seen: dict[str, str] = {}
        distinct = []
        for w in WINDOW_ORDER:
            wh = history_window_hash(build_window(s, w).memory_payload)
            if wh not in seen:
                seen[wh] = w
                distinct.append(w)
        out[family] = distinct
    return out


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


def run_one(region: str, family: str, scenario, window: str, sample: int) -> dict:
    from src.llm.clients.bedrock_client import BedrockClient

    r = build_window(scenario, window)
    payload = {**scenario.observation_payload, "strategy_memory": r.memory_payload}
    prompt_json = json.dumps(payload, sort_keys=True)

    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
    start = time.monotonic()
    response = client.generate_action(NATIVE_TOOL_LIKE_SYSTEM_PROMPT, prompt_json, _OUTPUT_SCHEMA)
    latency_ms = response.latency_ms if getattr(response, "latency_ms", None) is not None else (time.monotonic() - start) * 1000.0
    raw = response.payload if isinstance(response.payload, dict) else {}
    v = validate_model_output(scenario.observation, raw, scenario.legal_actions)
    repeat_failure = repeats_known_failure(raw.get("action"), raw.get("target_id")) if family == "S6_LONG_HISTORY" else None

    return {
        "runtime": "native", "scenario_family": family, "scenario_id": scenario.scenario_id, "window": window, "sample": sample,
        "history_window_hash": history_window_hash(r.memory_payload), "history_sequence_hash": history_sequence_hash(r.memory_payload),
        "history_fact_hash": history_fact_hash(r.memory_payload), "characters": len(prompt_json),
        "action": raw.get("action"), "target_id": raw.get("target_id"), "reason": raw.get("reason"),
        "repeats_known_failure": repeat_failure,
        "validation_status": "valid" if v.valid else f"invalid:{v.error_type}",
        "latency_ms": latency_ms, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
    }


def run(region: str) -> dict:
    scenarios = build_primary_scenarios()
    distinct = _distinct_windows_by_family()
    rows: list[dict] = []
    for family, windows in distinct.items():
        scenario = scenarios[family]
        for window in windows:
            for i in range(SAMPLES_PER_CELL):
                row = run_one(region, family, scenario, window, i)
                rows.append(row)
                print(f"[NATIVE {family}/{window}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']}) repeat_failure={row['repeats_known_failure']}")
    _write_jsonl(os.path.join(OUT_DIR, "native_window_trials.jsonl"), rows)
    return {"rows": len(rows), "distinct_windows_by_family": distinct}


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
    print(f"\n[DONE] M6E.3 Native replay complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
