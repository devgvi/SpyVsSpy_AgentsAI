"""M6D Phase B (spec section 23): live injected-invalid recovery. For each
of the 9 corpus failure families, the SAME fixed invalid first output is
replayed through P1/P2/P3 (P0 applied directly with no live call), 5
samples each (spec section 46: primary M6D uses 0 or 1 repairs, matching
M6C/M6C.1's own minimum-sample precedent). Native only (spec section 5).

    python -m src.evaluation.run_m6d_phase_b --region us-east-1
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

from src import constants
from src.evaluation.recovery_corpus import build_invalid_action_corpus
from src.evaluation.recovery_policy import (
    F0_CURRENT_SAFE_FALLBACK,
    P0_NO_REPAIR_SAFE_FALLBACK,
    P1_CURRENT_ONE_REPAIR,
    P2_MINIMAL_REPAIR,
    P3_FULL_LEGAL_REPAIR,
    run_recovery_policy,
)
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

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6d-retry-repair-fallback-policy-v1")
SAMPLES_PER_POLICY = 5
LIVE_POLICIES = (P1_CURRENT_ONE_REPAIR, P2_MINIMAL_REPAIR, P3_FULL_LEGAL_REPAIR)

# Spec section 4 ("keep fixed: ... Observation ... memory ... personality"):
# the REAL production system prompt and REAL compact observation payload,
# never a stripped-down placeholder - a repair call given no game-state
# context at all cannot be judged as representative of production repair
# behavior.
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


def _model_payload_for(observation) -> dict:
    observation_payload = serialize_observation(observation)
    model_payload = compact_observation_for_model(observation_payload, representation_variant=NATIVE_TOOL_LIKE_REPRESENTATION)
    model_payload["strategy_memory"] = AgentMemory(constants.WHITE_SPY_ID).to_compact_dict()
    return model_payload


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
        f.write(json.dumps({"phase": "preflight_phase_b", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def run(region: str) -> dict:
    from src.llm.clients.bedrock_client import BedrockClient

    corpus = build_invalid_action_corpus()
    rows: list[dict] = []

    for case in corpus:
        v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
        assert not v.valid
        model_payload = _model_payload_for(case.observation)

        # P0 - deterministic, no live call.
        p0 = run_recovery_policy(
            P0_NO_REPAIR_SAFE_FALLBACK, case.observation, case.legal_actions, model_payload, case.invalid_raw_response, v,
            model_client=None, system_prompt=_SYSTEM_PROMPT, output_schema=_OUTPUT_SCHEMA, fallback_policy=F0_CURRENT_SAFE_FALLBACK,
        )
        rows.append({
            "case_id": case.case_id, "failure_family": case.failure_family, "policy": P0_NO_REPAIR_SAFE_FALLBACK, "sample": 0,
            "original_action": p0.original_action, "original_target": p0.original_target,
            "repair_attempted": p0.repair_attempted, "repair_valid": p0.repair_valid,
            "final_action": p0.final_action, "final_target": p0.final_target, "fallback_used": p0.fallback_used,
            "intent_category": p0.intent_category, "latency_ms": p0.latency_ms,
            "input_tokens": p0.input_tokens, "output_tokens": p0.output_tokens,
        })
        print(f"[P0] {case.case_id}: fallback={p0.final_action}")

        for policy in LIVE_POLICIES:
            for i in range(SAMPLES_PER_POLICY):
                client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                outcome = run_recovery_policy(
                    policy, case.observation, case.legal_actions, model_payload, case.invalid_raw_response, v,
                    client, _SYSTEM_PROMPT, _OUTPUT_SCHEMA, fallback_policy=F0_CURRENT_SAFE_FALLBACK,
                )
                rows.append({
                    "case_id": case.case_id, "failure_family": case.failure_family, "policy": policy, "sample": i,
                    "original_action": outcome.original_action, "original_target": outcome.original_target,
                    "repair_attempted": outcome.repair_attempted, "repair_valid": outcome.repair_valid,
                    "repair_error_type": outcome.repair_error_type, "provider_error_type": outcome.provider_error_type,
                    "final_action": outcome.final_action, "final_target": outcome.final_target, "fallback_used": outcome.fallback_used,
                    "intent_category": outcome.intent_category, "latency_ms": outcome.latency_ms,
                    "input_tokens": outcome.input_tokens, "output_tokens": outcome.output_tokens,
                })
                print(f"[{policy}] {case.case_id} #{i}: valid={outcome.repair_valid} final={outcome.final_action}({outcome.final_target}) intent={outcome.intent_category}")

    _write_jsonl(os.path.join(OUT_DIR, "injected_invalid_trials.jsonl"), rows)

    # Repair-outcome metrics per policy/failure-class (spec sections 28, 64).
    metrics: dict[str, dict] = {}
    for case in corpus:
        for policy in (P0_NO_REPAIR_SAFE_FALLBACK,) + LIVE_POLICIES:
            key = f"{case.failure_family}__{policy}"
            subset = [r for r in rows if r["case_id"] == case.case_id and r["policy"] == policy]
            n = len(subset) or 1
            metrics[key] = {
                "attempts": len(subset),
                "valid_after_repair": sum(1 for r in subset if r["repair_valid"]) / n if policy != P0_NO_REPAIR_SAFE_FALLBACK else None,
                "same_action_retained": sum(1 for r in subset if r["intent_category"] in ("SAME_ACTION_SAME_TARGET", "SAME_ACTION_DIFFERENT_TARGET")) / n,
                "same_target_retained": sum(1 for r in subset if r["intent_category"] == "SAME_ACTION_SAME_TARGET") / n,
                "action_changed": sum(1 for r in subset if r["intent_category"] == "DIFFERENT_ACTION") / n,
                "fallback_rate": sum(1 for r in subset if r["fallback_used"]) / n,
                "mean_latency_ms": sum(r["latency_ms"] or 0.0 for r in subset) / n,
                "mean_input_tokens": sum(r["input_tokens"] or 0 for r in subset) / n,
                "mean_output_tokens": sum(r["output_tokens"] or 0 for r in subset) / n,
            }

    result = {"region": region, "model_id": MODEL_ID, "samples_per_policy": SAMPLES_PER_POLICY, "metrics": metrics}
    with open(os.path.join(OUT_DIR, "repair_policy_metrics.json"), "w", encoding="utf-8") as f:
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
    print(f"\n[DONE] Phase B complete ({start.isoformat()} - {end.isoformat()}). Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
