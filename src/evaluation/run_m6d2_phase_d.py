"""M6D.2 Phase D (spec sections 29-32): Strands replication, run only
after Native analysis (Phase C + provider-variance control) is complete.
Replicates: N0_CURRENT_PRODUCTION (baseline) + N2_NEUTRAL_STRUCTURED (the
Native candidate with the lowest retention range for INVALID_TRAP_TYPE,
0pp vs N0's 80pp and N1's 40pp) across their 4 perturbations, on the two
failure classes that showed real representation sensitivity
(INVALID_TRAP_TYPE, MISSING_REQUIRED_FIELD) - plus M6D.1's own
R2_LEGAL_FIRST as the explicit "intentionally sensitive control" for
INVALID_TRAP_TYPE. Gated on RP01-RP12 remaining clean.

    python -m src.evaluation.run_m6d2_phase_d --region us-east-1
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
from src.evaluation.recovery_policy import F0_CURRENT_SAFE_FALLBACK, apply_fallback_policy, classify_intent
from src.evaluation.repair_contracts import (
    N0_CURRENT_PRODUCTION,
    N2_NEUTRAL_STRUCTURED,
    PERTURBATIONS,
    build_repair_facts,
    render_contract,
)
from src.evaluation.repair_prompt_variants import R2_LEGAL_FIRST, render_variant, repair_fact_hash, repair_representation_hash
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import compact_observation_for_model, serialize_observation, validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6d2-neutral-repair-contract-v1")
SAMPLES_PER_CELL = 3
SELECTED_CONTRACTS = (N0_CURRENT_PRODUCTION, N2_NEUTRAL_STRUCTURED)
SELECTED_FAILURE_FAMILIES = ("INVALID_TRAP_TYPE", "MISSING_REQUIRED_FIELD")


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
        f.write(json.dumps({"phase": "preflight_phase_d", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _model_payload_for(observation) -> dict:
    observation_payload = serialize_observation(observation)
    model_payload = compact_observation_for_model(observation_payload)
    model_payload["strategy_memory"] = AgentMemory(constants.WHITE_SPY_ID).to_compact_dict()
    return model_payload


def _one_strands_repair(region, case, repair_text, v, label_contract, label_perturbation):
    telemetry = None
    from src.llm.telemetry import TelemetryLogger

    telemetry = TelemetryLogger()
    adapter = create_agent_runtime(
        "strands", actor_id=constants.WHITE_SPY_ID, telemetry=telemetry, episode_id=f"m6d2-strands-{case.case_id}-{label_contract}-{label_perturbation}",
        provider_name="bedrock", model_name=MODEL_ID, execution_mode="evaluation",
        llm_region=region, llm_timeout=60.0, personality_profile=None,
    )
    model_payload = _model_payload_for(case.observation)
    invocation = adapter._invoke_strands(model_payload, repair_instruction=repair_text)
    raw = invocation.proposal if isinstance(invocation.proposal, dict) else {}
    rv = validate_model_output(case.observation, raw, case.legal_actions)

    original_action = case.invalid_raw_response.get("action")
    original_target = case.invalid_raw_response.get("target_id")
    if rv.valid:
        final_action, final_target, fallback_used = raw.get("action"), raw.get("target_id"), False
    else:
        fb = apply_fallback_policy(F0_CURRENT_SAFE_FALLBACK, case.legal_actions)
        final_action, final_target, fallback_used = fb.get("action"), fb.get("target_id"), True
    intent = classify_intent(original_action, original_target, final_action, final_target, fallback_used)
    return {
        "repair_valid": rv.valid, "final_action": final_action, "final_target": final_target,
        "fallback_used": fallback_used, "intent_category": intent,
        "latency_ms": invocation.latency_ms, "input_tokens": invocation.input_tokens, "output_tokens": invocation.output_tokens,
    }


def run(region: str) -> dict:
    corpus = {c.failure_family: c for c in build_invalid_action_corpus() if c.failure_family in SELECTED_FAILURE_FAMILIES}
    rows: list[dict] = []

    for family, case in corpus.items():
        v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
        facts = build_repair_facts(case, v)

        for contract in SELECTED_CONTRACTS:
            for perturbation in PERTURBATIONS:
                repair_text = render_contract(contract, perturbation, facts, case, v)
                for i in range(SAMPLES_PER_CELL):
                    outcome = _one_strands_repair(region, case, repair_text, v, contract, perturbation)
                    row = {
                        "case_id": case.case_id, "failure_family": family, "contract": contract, "perturbation": perturbation,
                        "sample": i, "runtime": "strands",
                        "repair_fact_hash": repair_fact_hash(facts), "repair_representation_hash": repair_representation_hash(repair_text),
                        **outcome,
                    }
                    rows.append(row)
                    print(f"[STRANDS {contract}/{perturbation}] {family} #{i}: valid={row['repair_valid']} final={row['final_action']}({row['final_target']}) intent={row['intent_category']}")

        # M6D.1's own R2_LEGAL_FIRST as the explicit "intentionally sensitive control" - INVALID_TRAP_TYPE only.
        if family == "INVALID_TRAP_TYPE":
            from src.evaluation.repair_prompt_variants import build_repair_facts as build_m6d1_facts

            m6d1_facts = build_m6d1_facts(case, v)
            repair_text = render_variant(R2_LEGAL_FIRST, m6d1_facts, case, v)
            for i in range(SAMPLES_PER_CELL):
                outcome = _one_strands_repair(region, case, repair_text, v, "M6D1_R2_LEGAL_FIRST_CONTROL", "N_A")
                row = {
                    "case_id": case.case_id, "failure_family": family, "contract": "M6D1_R2_LEGAL_FIRST_CONTROL", "perturbation": "N/A",
                    "sample": i, "runtime": "strands", **outcome,
                }
                rows.append(row)
                print(f"[STRANDS M6D1_CONTROL] {family} #{i}: valid={row['repair_valid']} final={row['final_action']}({row['final_target']}) intent={row['intent_category']}")

    _write_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"), rows)

    metrics: dict[str, dict] = {}
    for row_key in {(r["failure_family"], r["contract"], r["perturbation"]) for r in rows}:
        family, contract, perturbation = row_key
        subset = [r for r in rows if r["failure_family"] == family and r["contract"] == contract and r["perturbation"] == perturbation]
        n = len(subset) or 1
        metrics[f"{family}__{contract}__{perturbation}"] = {
            "attempts": len(subset),
            "valid_after_repair": sum(1 for r in subset if r["repair_valid"]) / n,
            "same_action_retained": sum(1 for r in subset if r["intent_category"] in ("SAME_ACTION_SAME_TARGET", "SAME_ACTION_DIFFERENT_TARGET")) / n,
            "action_changed": sum(1 for r in subset if r["intent_category"] == "DIFFERENT_ACTION") / n,
        }

    result = {"region": region, "model_id": MODEL_ID, "samples_per_cell": SAMPLES_PER_CELL, "metrics": metrics}
    with open(os.path.join(OUT_DIR, "strands_robustness_metrics.json"), "w", encoding="utf-8") as f:
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
    print(f"\n[DONE] Phase D complete ({start.isoformat()} - {end.isoformat()}). Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
