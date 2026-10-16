"""M6D Strands replication (spec sections 47-49): after Native results are
clean, replicate a selected subset under Strands - one invalid trap-type
case, one illegal-target case, one missing-field case, one fallback case
(P0 doubles as the fallback case for all of them, so no separate case is
needed there). Gated on RP01-RP12 remaining clean (spec section 48).

    python -m src.evaluation.run_m6d_stage_strands --region us-east-1
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

from src import constants
from src.agents.registry import create_agent_runtime
from src.evaluation.recovery_corpus import build_invalid_action_corpus
from src.evaluation.recovery_policy import (
    F0_CURRENT_SAFE_FALLBACK,
    P0_NO_REPAIR_SAFE_FALLBACK,
    P1_CURRENT_ONE_REPAIR,
    P2_MINIMAL_REPAIR,
    P3_FULL_LEGAL_REPAIR,
    run_recovery_policy_strands,
)
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import VALID_ACTION_NAMES, compact_observation_for_model, serialize_observation, validate_model_output
from src.llm.telemetry import TelemetryLogger

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6d-retry-repair-fallback-policy-v1")
SAMPLES_PER_POLICY = 3
LIVE_POLICIES = (P1_CURRENT_ONE_REPAIR, P2_MINIMAL_REPAIR, P3_FULL_LEGAL_REPAIR)
SELECTED_CASE_IDS = ("case04_invalid_trap_type", "case09_action_currently_illegal", "case06_missing_required_field")

_SYSTEM_PROMPT = "You are a covert operative in a two-agent stealth game. Respond with one JSON action."
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


def _preflight(out_dir: str, region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight_stage_strands", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def run(region: str) -> dict:
    """Uses the real `StrandsAgentAdapter._invoke_strands` SDK call path
    for the repair step (via `run_recovery_policy_strands`) - genuinely
    exercises Strands' own tool-calling loop with each policy's repair
    text, never a Native call mislabeled as Strands."""
    corpus = {c.case_id: c for c in build_invalid_action_corpus()}
    rows: list[dict] = []

    for case_id in SELECTED_CASE_IDS:
        case = corpus[case_id]
        v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
        observation_payload = serialize_observation(case.observation)
        model_payload = compact_observation_for_model(observation_payload)
        from src.llm.agent_memory import AgentMemory

        model_payload["strategy_memory"] = AgentMemory(constants.WHITE_SPY_ID).to_compact_dict()

        p0 = run_recovery_policy_strands(
            P0_NO_REPAIR_SAFE_FALLBACK, case.observation, case.legal_actions, model_payload, case.invalid_raw_response, v,
            strands_adapter=None, fallback_policy=F0_CURRENT_SAFE_FALLBACK,
        )
        rows.append({
            "case_id": case.case_id, "runtime": "strands", "policy": P0_NO_REPAIR_SAFE_FALLBACK, "sample": 0,
            "final_action": p0.final_action, "fallback_used": p0.fallback_used, "intent_category": p0.intent_category,
        })

        for policy in LIVE_POLICIES:
            for i in range(SAMPLES_PER_POLICY):
                telemetry = TelemetryLogger()
                adapter = create_agent_runtime(
                    "strands", actor_id=constants.WHITE_SPY_ID, telemetry=telemetry, episode_id=f"m6d-strands-{case_id}-{policy}-{i}",
                    provider_name="bedrock", model_name=MODEL_ID, execution_mode="evaluation",
                    llm_region=region, llm_timeout=60.0, personality_profile=None,
                )
                outcome = run_recovery_policy_strands(
                    policy, case.observation, case.legal_actions, model_payload, case.invalid_raw_response, v,
                    strands_adapter=adapter, fallback_policy=F0_CURRENT_SAFE_FALLBACK,
                )
                rows.append({
                    "case_id": case.case_id, "runtime": "strands", "policy": policy, "sample": i,
                    "original_action": outcome.original_action, "original_target": outcome.original_target,
                    "repair_valid": outcome.repair_valid, "final_action": outcome.final_action, "final_target": outcome.final_target,
                    "fallback_used": outcome.fallback_used, "intent_category": outcome.intent_category,
                    "latency_ms": outcome.latency_ms, "input_tokens": outcome.input_tokens, "output_tokens": outcome.output_tokens,
                })
                print(f"[STRANDS {policy}] {case_id} #{i}: valid={outcome.repair_valid} final={outcome.final_action} intent={outcome.intent_category}")

    _write_jsonl(os.path.join(OUT_DIR, "strands_recovery_trials.jsonl"), rows)
    result = {"region": region, "model_id": MODEL_ID, "cases": SELECTED_CASE_IDS, "samples_per_policy": SAMPLES_PER_POLICY, "rows": len(rows)}
    with open(os.path.join(OUT_DIR, "strands_recovery_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not _rp_manifest_clean():
        print("[FATAL] RP01-RP12 manifest has an unexplained FAIL - refusing to start Strands live calls.", file=sys.stderr)
        return 1

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
    print(f"\n[DONE] Strands replication complete ({start.isoformat()} - {end.isoformat()}). Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
