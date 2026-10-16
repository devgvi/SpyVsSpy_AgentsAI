"""M6F Phase C (spec sections 28-31, 39): live Native vs. Strands replay
for the three non-repair scenario families (S0 ordinary, S1 memory, S3
tool-action) - 5 samples per runtime per scenario (spec section 28's
stated minimum). S2 (repair) is deliberately NOT re-run live here: M6D.2
already collected 231 live calls across exactly these two failure
families (INVALID_TRAP_TYPE, MISSING_REQUIRED_FIELD) and exactly these
two diagnostic contracts (CURRENT_PRODUCTION, NEUTRAL_STRUCTURED) - spec
section 26's "reuse proven snapshots... do not create a totally new world
unless necessary" applies just as much to re-spending live budget as to
building new game states. `run_m6f_phase_c_repair_reuse.py` re-analyzes
that existing data under M6F's own framing instead.

    python -m src.evaluation.run_m6f_phase_c --region us-east-1
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
import time
from datetime import datetime, timezone

from src import constants
from src.agents.registry import create_agent_runtime
from src.evaluation.agent_contract import (
    build_agent_response,
    native_model_payload,
    native_system_prompt,
    strands_model_payload,
)
from src.evaluation.m6f_scenarios import (
    S0_ORDINARY_DECISION,
    S1_MEMORY_DECISION,
    S3_TOOL_ACTION_DECISION,
    build_s0_ordinary_decision,
    build_s1_memory_decision,
    build_s3_tool_action_decision,
)
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import VALID_ACTION_NAMES

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
PROVIDER_NAME = "bedrock"
OUT_DIR = os.path.join("evaluation_runs", "m6f-runtime-interoperability-v1")
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


def _rp_manifest_clean() -> bool:
    from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST

    return all(e.status != FAIL for e in RUNTIME_PARITY_MANIFEST)


def _preflight(region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight_phase_c", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _hash_repr(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _run_native_sample(region: str, scenario, sample: int) -> dict:
    from src.llm.clients.bedrock_client import BedrockClient

    request = scenario.request
    system_prompt = native_system_prompt(request)
    payload = native_model_payload(request)
    prompt_json = json.dumps(payload, sort_keys=True)

    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
    start = time.monotonic()
    response = client.generate_action(system_prompt, prompt_json, _OUTPUT_SCHEMA)
    latency_ms = response.latency_ms if getattr(response, "latency_ms", None) is not None else (time.monotonic() - start) * 1000.0
    raw = response.payload if isinstance(response.payload, dict) else {}

    agent_response = build_agent_response(
        raw_model_output=raw, observation=scenario.observation, legal_actions=scenario.legal_actions, tool_calls=1,
        runtime_metadata={"runtime": "native"},
        provider_metadata={
            "latency_ms": latency_ms, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
            "model_id": MODEL_ID, "provider": PROVIDER_NAME,
        },
    )
    return {
        "scenario_id": scenario.scenario_id, "scenario_family": scenario.request.metadata["scenario_family"],
        "runtime": "native", "sample": sample,
        "canonical_request_hash": request.request_semantic_hash(),
        "runtime_representation_hash": _hash_repr({"system_prompt": system_prompt, "payload": payload}),
        "raw_model_output": agent_response.raw_model_output,
        "normalized_action": agent_response.normalized_action,
        "validation_status": agent_response.validation_status,
        "provider_metadata": agent_response.provider_metadata,
        "runtime_metadata": agent_response.runtime_metadata,
    }


def _run_strands_sample(region: str, scenario, sample: int) -> dict:
    from src.llm.telemetry import TelemetryLogger

    request = scenario.request
    payload = strands_model_payload(request)
    adapter = create_agent_runtime(
        "strands", actor_id=request.actor_id, telemetry=TelemetryLogger(),
        episode_id=f"{request.episode_id}-{sample}", provider_name=PROVIDER_NAME, model_name=MODEL_ID,
        execution_mode="evaluation", llm_region=region, llm_timeout=60.0, personality_profile=request.personality_policy,
    )
    invocation = adapter._invoke_strands(payload, repair_instruction=None)
    raw = invocation.proposal if isinstance(invocation.proposal, dict) else {}

    agent_response = build_agent_response(
        raw_model_output=raw, observation=scenario.observation, legal_actions=scenario.legal_actions,
        tool_calls=invocation.tool_call_count,
        runtime_metadata={"runtime": "strands", "stop_reason": invocation.stop_reason, "cycle_count": invocation.cycle_count, "proposal_count": invocation.proposal_count},
        provider_metadata={
            "latency_ms": invocation.latency_ms, "input_tokens": invocation.input_tokens, "output_tokens": invocation.output_tokens,
            "model_id": MODEL_ID, "provider": PROVIDER_NAME,
        },
    )
    return {
        "scenario_id": scenario.scenario_id, "scenario_family": scenario.request.metadata["scenario_family"],
        "runtime": "strands", "sample": sample,
        "canonical_request_hash": request.request_semantic_hash(),
        "runtime_representation_hash": _hash_repr({"system_prompt": adapter._effective_system_prompt, "payload": payload}),
        "raw_model_output": agent_response.raw_model_output,
        "normalized_action": agent_response.normalized_action,
        "validation_status": agent_response.validation_status,
        "provider_metadata": agent_response.provider_metadata,
        "runtime_metadata": agent_response.runtime_metadata,
        "error_type": invocation.error_type,
    }


def run(region: str) -> dict:
    scenarios = {
        S0_ORDINARY_DECISION: build_s0_ordinary_decision(MODEL_ID, PROVIDER_NAME),
        S1_MEMORY_DECISION: build_s1_memory_decision(MODEL_ID, PROVIDER_NAME),
        S3_TOOL_ACTION_DECISION: build_s3_tool_action_decision(MODEL_ID, PROVIDER_NAME),
    }
    native_rows, strands_rows = [], []
    for family, scenario in scenarios.items():
        for i in range(SAMPLES_PER_CELL):
            row = _run_native_sample(region, scenario, i)
            native_rows.append(row)
            print(f"[NATIVE {family}] #{i}: valid={row['validation_status']} action={row['normalized_action']}")
        for i in range(SAMPLES_PER_CELL):
            row = _run_strands_sample(region, scenario, i)
            strands_rows.append(row)
            print(f"[STRANDS {family}] #{i}: valid={row['validation_status']} action={row['normalized_action']}")

    _write_jsonl(os.path.join(OUT_DIR, "native_trials.jsonl"), native_rows)
    _write_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"), strands_rows)
    return {"native_count": len(native_rows), "strands_count": len(strands_rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not _rp_manifest_clean():
        print("[FATAL] RP01-RP18 manifest has an unexplained FAIL - refusing to start live calls.", file=sys.stderr)
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
    print(f"\n[DONE] M6F Phase C complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
