"""M6D.1 Phase D (spec sections 25, 33-34): Strands replication, run only
after Native analysis (Phase B + provider-variance control) is complete.
3 selected cases: INVALID_TRAP_TYPE (strongest section-order AND
structure/compactness effect), MISSING_REQUIRED_FIELD (independent
section-order-effect replication, second failure class), TARGET_NOT_INSPECTED
(cleanest null control - 0% retention across every variant in Native).
All 6 variants x 3 samples per case. Gated on RP01-RP12 remaining clean.

    python -m src.evaluation.run_m6d1_phase_d --region us-east-1
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
from src.evaluation.repair_prompt_variants import REPAIR_VARIANTS, build_repair_facts, render_variant, repair_fact_hash, repair_representation_hash
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import compact_observation_for_model, serialize_observation, validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6d1-repair-prompt-counterbalancing-v1")
SAMPLES_PER_VARIANT = 3
SELECTED_FAILURE_FAMILIES = ("INVALID_TRAP_TYPE", "MISSING_REQUIRED_FIELD", "TARGET_NOT_INSPECTED")


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


def run(region: str) -> dict:
    corpus = {c.failure_family: c for c in build_invalid_action_corpus() if c.failure_family in SELECTED_FAILURE_FAMILIES}
    rows: list[dict] = []

    for family, case in corpus.items():
        v = validate_model_output(case.observation, case.invalid_raw_response, case.legal_actions)
        facts = build_repair_facts(case, v)
        model_payload = _model_payload_for(case.observation)

        for variant in REPAIR_VARIANTS:
            repair_text = render_variant(variant, facts, case, v)
            for i in range(SAMPLES_PER_VARIANT):
                telemetry = TelemetryLogger()
                adapter = create_agent_runtime(
                    "strands", actor_id=constants.WHITE_SPY_ID, telemetry=telemetry, episode_id=f"m6d1-strands-{family}-{variant}-{i}",
                    provider_name="bedrock", model_name=MODEL_ID, execution_mode="evaluation",
                    llm_region=region, llm_timeout=60.0, personality_profile=None,
                )
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

                row = {
                    "case_id": case.case_id, "failure_family": family, "variant": variant, "sample": i, "runtime": "strands",
                    "repair_fact_hash": repair_fact_hash(facts), "repair_representation_hash": repair_representation_hash(repair_text),
                    "repair_valid": rv.valid, "final_action": final_action, "final_target": final_target,
                    "fallback_used": fallback_used, "intent_category": intent,
                    "latency_ms": invocation.latency_ms, "input_tokens": invocation.input_tokens, "output_tokens": invocation.output_tokens,
                }
                rows.append(row)
                print(f"[STRANDS {variant}] {family} #{i}: valid={rv.valid} final={final_action}({final_target}) intent={intent}")

    _write_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"), rows)

    metrics: dict[str, dict] = {}
    for family in corpus:
        for variant in REPAIR_VARIANTS:
            key = f"{family}__{variant}"
            subset = [r for r in rows if r["failure_family"] == family and r["variant"] == variant]
            n = len(subset) or 1
            metrics[key] = {
                "attempts": len(subset),
                "valid_after_repair": sum(1 for r in subset if r["repair_valid"]) / n,
                "same_action_retained": sum(1 for r in subset if r["intent_category"] in ("SAME_ACTION_SAME_TARGET", "SAME_ACTION_DIFFERENT_TARGET")) / n,
                "action_changed": sum(1 for r in subset if r["intent_category"] == "DIFFERENT_ACTION") / n,
                "fallback_rate": sum(1 for r in subset if r["fallback_used"]) / n,
            }

    result = {"region": region, "model_id": MODEL_ID, "cases": list(corpus), "samples_per_variant": SAMPLES_PER_VARIANT, "metrics": metrics}
    with open(os.path.join(OUT_DIR, "strands_repair_representation_metrics.json"), "w", encoding="utf-8") as f:
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
