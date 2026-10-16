"""M6D.2 Phase C (spec section 21): live Native robustness replay. Same
fixed invalid raw output per case (from M6D's own corpus), replayed
through 3 contracts x 4 perturbations x 5 repeats = 180 live calls.

    python -m src.evaluation.run_m6d2_phase_c --region us-east-1
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

from src import constants
from src.evaluation.recovery_corpus import build_invalid_action_corpus
from src.evaluation.recovery_policy import F0_CURRENT_SAFE_FALLBACK, apply_fallback_policy, classify_intent
from src.evaluation.repair_contracts import CONTRACTS, PERTURBATIONS, build_repair_facts, render_contract
from src.evaluation.repair_prompt_variants import repair_fact_hash, repair_representation_hash
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import (
    NATIVE_TOOL_LIKE_REPRESENTATION,
    NATIVE_TOOL_LIKE_SYSTEM_PROMPT,
    REPAIR_PROMPT_VERSION,
    VALID_ACTION_NAMES,
    compact_observation_for_model,
    serialize_observation,
    validate_model_output,
)

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6d2-neutral-repair-contract-v1")
SAMPLES_PER_CELL = 5
PRIMARY_FAILURE_FAMILIES = ("INVALID_TRAP_TYPE", "MISSING_REQUIRED_FIELD", "TARGET_NOT_INSPECTED")

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


def _model_payload_for(observation) -> dict:
    observation_payload = serialize_observation(observation)
    model_payload = compact_observation_for_model(observation_payload, representation_variant=NATIVE_TOOL_LIKE_REPRESENTATION)
    model_payload["strategy_memory"] = AgentMemory(constants.WHITE_SPY_ID).to_compact_dict()
    return model_payload


def run_one_repair(client, case, facts, contract, perturbation, initial_validation, fallback_policy=F0_CURRENT_SAFE_FALLBACK):
    repair_text = render_contract(contract, perturbation, facts, case, initial_validation)
    payload = dict(_model_payload_for(case.observation))
    payload["repair_instruction"] = repair_text
    payload["repair_prompt_version"] = REPAIR_PROMPT_VERSION
    prompt_json = json.dumps(payload, sort_keys=True)

    start = time.monotonic()
    response = client.generate_action(_SYSTEM_PROMPT, prompt_json, _OUTPUT_SCHEMA)
    latency_ms = response.latency_ms if getattr(response, "latency_ms", None) is not None else (time.monotonic() - start) * 1000.0
    raw = response.payload if isinstance(response.payload, dict) else {}
    v = validate_model_output(case.observation, raw, case.legal_actions)

    original_action = case.invalid_raw_response.get("action")
    original_target = case.invalid_raw_response.get("target_id")
    if v.valid:
        final_action, final_target, fallback_used = raw.get("action"), raw.get("target_id"), False
    else:
        fb = apply_fallback_policy(fallback_policy, case.legal_actions)
        final_action, final_target, fallback_used = fb.get("action"), fb.get("target_id"), True

    intent = classify_intent(original_action, original_target, final_action, final_target, fallback_used)
    return {
        "case_id": case.case_id, "failure_family": case.failure_family, "contract": contract, "perturbation": perturbation,
        "repair_fact_hash": repair_fact_hash(facts), "repair_representation_hash": repair_representation_hash(repair_text),
        "repair_prompt_chars": len(repair_text), "repair_valid": v.valid, "final_action": final_action, "final_target": final_target,
        "fallback_used": fallback_used, "intent_category": intent, "latency_ms": latency_ms,
        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
    }


def run(region: str) -> dict:
    from src.llm.clients.bedrock_client import BedrockClient

    corpus = {c.failure_family: c for c in build_invalid_action_corpus() if c.failure_family in PRIMARY_FAILURE_FAMILIES}
    rows: list[dict] = []

    for family, case in corpus.items():
        v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
        facts = build_repair_facts(case, v)
        for contract in CONTRACTS:
            for perturbation in PERTURBATIONS:
                for i in range(SAMPLES_PER_CELL):
                    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                    row = run_one_repair(client, case, facts, contract, perturbation, v)
                    row["sample"] = i
                    rows.append(row)
                    print(f"[{contract}/{perturbation}] {family} #{i}: valid={row['repair_valid']} final={row['final_action']}({row['final_target']}) intent={row['intent_category']}")

    _write_jsonl(os.path.join(OUT_DIR, "native_trials.jsonl"), rows)

    metrics: dict[str, dict] = {}
    for family in corpus:
        for contract in CONTRACTS:
            for perturbation in PERTURBATIONS:
                key = f"{family}__{contract}__{perturbation}"
                subset = [r for r in rows if r["failure_family"] == family and r["contract"] == contract and r["perturbation"] == perturbation]
                n = len(subset) or 1
                metrics[key] = {
                    "attempts": len(subset),
                    "valid_after_repair": sum(1 for r in subset if r["repair_valid"]) / n,
                    "same_action_retained": sum(1 for r in subset if r["intent_category"] in ("SAME_ACTION_SAME_TARGET", "SAME_ACTION_DIFFERENT_TARGET")) / n,
                    "same_target_retained": sum(1 for r in subset if r["intent_category"] == "SAME_ACTION_SAME_TARGET") / n,
                    "action_changed": sum(1 for r in subset if r["intent_category"] == "DIFFERENT_ACTION") / n,
                    "fallback_rate": sum(1 for r in subset if r["fallback_used"]) / n,
                    "mean_latency_ms": sum(r["latency_ms"] or 0.0 for r in subset) / n,
                    "mean_input_tokens": sum(r["input_tokens"] or 0 for r in subset) / n,
                    "mean_output_tokens": sum(r["output_tokens"] or 0 for r in subset) / n,
                }

    result = {"region": region, "model_id": MODEL_ID, "samples_per_cell": SAMPLES_PER_CELL, "metrics": metrics}
    with open(os.path.join(OUT_DIR, "robustness_metrics.json"), "w", encoding="utf-8") as f:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
