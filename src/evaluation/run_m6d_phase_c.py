"""M6D Phase C (spec sections 33-37): organic invalid outputs. Runs a REAL
initial live call (never injected) against snapshots historically prone to
a genuine first-pass validation miss (trap-placement opportunities, where
`trap_type` is frequently omitted on the first try - confirmed at a
~80-100% initial-miss rate in M6B.2's own Phase E data). Whichever raw
invalid outputs are organically captured are then replayed - the EXACT
same captured raw response, unmodified - through P0/P1/P2/P3 (spec
section 37's "preferred causal method"), so the first call is held
identical across every policy comparison (spec section 35).

    python -m src.evaluation.run_m6d_phase_c --region us-east-1
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

from src.evaluation.recovery_policy import (
    F0_CURRENT_SAFE_FALLBACK,
    P0_NO_REPAIR_SAFE_FALLBACK,
    P1_CURRENT_ONE_REPAIR,
    P2_MINIMAL_REPAIR,
    P3_FULL_LEGAL_REPAIR,
    run_recovery_policy,
)
from src.evaluation.trap_snapshot_set import build_trap_snapshot_set
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import (
    NATIVE_TOOL_LIKE_REPRESENTATION,
    NATIVE_TOOL_LIKE_SYSTEM_PROMPT,
    VALID_ACTION_NAMES,
    compact_observation_for_model,
    serialize_observation,
    validate_model_output,
)
from src import constants

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6d-retry-repair-fallback-policy-v1")
HARVEST_SAMPLES_PER_SNAPSHOT = 15
MAX_CAPTURED_CASES = 8
LIVE_POLICIES = (P1_CURRENT_ONE_REPAIR, P2_MINIMAL_REPAIR, P3_FULL_LEGAL_REPAIR)

# Spec section 4: the real production system prompt, matching the
# "native_tool_like" baseline every other M6-family live driver has used
# throughout this project.
_SYSTEM_PROMPT = NATIVE_TOOL_LIKE_SYSTEM_PROMPT
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


def _preflight(out_dir: str, region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight_phase_c", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def run(region: str) -> dict:
    from src.llm.clients.bedrock_client import BedrockClient

    snapshots = build_trap_snapshot_set()
    harvest_rows: list[dict] = []
    captured = []  # list of (snapshot, invalid_raw_response, validation)

    for snap in snapshots:
        observation_payload = serialize_observation(snap.observation)
        model_payload = compact_observation_for_model(observation_payload, representation_variant=NATIVE_TOOL_LIKE_REPRESENTATION)
        model_payload["strategy_memory"] = AgentMemory(constants.WHITE_SPY_ID).to_compact_dict()
        prompt_json = json.dumps(model_payload, sort_keys=True)
        initial_prompt_hash = hashlib.sha256(prompt_json.encode()).hexdigest()
        for i in range(HARVEST_SAMPLES_PER_SNAPSHOT):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            response = client.generate_action(_SYSTEM_PROMPT, prompt_json, _OUTPUT_SCHEMA)
            raw = response.payload if isinstance(response.payload, dict) else {}
            v = validate_model_output(snap.observation, raw, snap.legal_actions)
            harvest_rows.append({
                "snapshot_id": snap.snapshot_id, "sample": i, "raw_response": raw, "valid": v.valid,
                "error_type": v.error_type, "initial_prompt_hash": initial_prompt_hash,
            })
            print(f"[HARVEST] {snap.snapshot_id} #{i}: valid={v.valid} error={v.error_type} raw={raw}")
            if not v.valid and len(captured) < MAX_CAPTURED_CASES:
                captured.append((snap, dict(raw), v, initial_prompt_hash))

    _write_jsonl(os.path.join(OUT_DIR, "organic_invalid_trials.jsonl"), harvest_rows)

    replay_rows: list[dict] = []
    for idx, (snap, raw, v, prompt_hash) in enumerate(captured):
        p0 = run_recovery_policy(
            P0_NO_REPAIR_SAFE_FALLBACK, snap.observation, snap.legal_actions, {}, raw, v,
            model_client=None, system_prompt=_SYSTEM_PROMPT, output_schema=_OUTPUT_SCHEMA, fallback_policy=F0_CURRENT_SAFE_FALLBACK,
        )
        replay_rows.append({
            "captured_case": idx, "snapshot_id": snap.snapshot_id, "initial_prompt_hash": prompt_hash, "policy": P0_NO_REPAIR_SAFE_FALLBACK,
            "original_action": p0.original_action, "original_target": p0.original_target,
            "final_action": p0.final_action, "fallback_used": p0.fallback_used, "intent_category": p0.intent_category,
        })
        for policy in LIVE_POLICIES:
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            outcome = run_recovery_policy(
                policy, snap.observation, snap.legal_actions, {"strategy_memory": {}}, raw, v,
                client, _SYSTEM_PROMPT, _OUTPUT_SCHEMA, fallback_policy=F0_CURRENT_SAFE_FALLBACK,
            )
            replay_rows.append({
                "captured_case": idx, "snapshot_id": snap.snapshot_id, "initial_prompt_hash": prompt_hash, "policy": policy,
                "original_action": outcome.original_action, "original_target": outcome.original_target,
                "repair_valid": outcome.repair_valid, "final_action": outcome.final_action, "final_target": outcome.final_target,
                "fallback_used": outcome.fallback_used, "intent_category": outcome.intent_category,
                "latency_ms": outcome.latency_ms, "input_tokens": outcome.input_tokens, "output_tokens": outcome.output_tokens,
            })
            print(f"[REPLAY {policy}] captured#{idx} ({snap.snapshot_id}): valid={outcome.repair_valid} final={outcome.final_action} intent={outcome.intent_category}")

    _write_jsonl(os.path.join(OUT_DIR, "organic_invalid_replay.jsonl"), replay_rows)

    n_harvest = len(harvest_rows) or 1
    result = {
        "region": region, "model_id": MODEL_ID,
        "harvested": len(harvest_rows), "raw_invalid_rate": sum(1 for r in harvest_rows if not r["valid"]) / n_harvest,
        "captured_cases": len(captured), "replayed_rows": len(replay_rows),
    }
    with open(os.path.join(OUT_DIR, "organic_invalid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


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

    if not _preflight(OUT_DIR, region):
        print("[FATAL] Bedrock preflight failed.", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[OK] Preflight passed.")
        return 0

    start = datetime.now(timezone.utc)
    result = run(region)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] Phase C complete ({start.isoformat()} - {end.isoformat()}). Output: {OUT_DIR}")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
