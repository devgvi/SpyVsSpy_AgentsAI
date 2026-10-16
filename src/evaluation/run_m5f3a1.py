"""M5F.3A.1: automated live-diagnostic entry point for the deconfounded
social-influence replication.

    python -m src.evaluation.run_m5f3a1 --calibration  # Phase C: 5 NO_MESSAGE_CONTROL trials/profile, dominance check
    python -m src.evaluation.run_m5f3a1 --pilot         # Phase D: one trial per message class, sanity only
    python -m src.evaluation.run_m5f3a1 --matrix        # Phase E: primary replication (5 trials/condition/profile)

Phase A (dominance preflight) and Phase B (fixture audit) are covered by
tests/test_m5f3a1_dominance_preflight.py and
evaluation_runs/m5f3a1-deconfounded-social-influence-v1/fixture_audit.md -
not this driver. Reuses M5F.3A's infrastructure unchanged wherever
possible (spec section 2): same message texts, same response taggers,
same Bedrock provider/native_tool_like runtime, same personality mapping -
only `world_builder=build_deconfounded_world` and the `m5f3a1` episode
prefix differ from `run_m5f3a.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src import constants
from src.evaluation.dominance_preflight import BALANCED, classify_control_calibration, classify_dominance
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.opportunity_metrics import classify_claim_consistency, classify_message_category, classify_recipient_influence
from src.evaluation.social_influence_metrics import (
    classify_cooperation_response,
    classify_threat_capability,
    classify_threat_response,
    classify_warning_response,
    recipient_capability_knowledge,
)
from src.evaluation.social_influence_scenarios import (
    COOPERATION,
    DECONFOUNDED_CANDIDATES,
    ENV_TRAP_TARGET_ID,
    MESSAGE_CLASSES,
    MISDIRECTION,
    NON_CREDIBLE_THREAT,
    NO_MESSAGE_CONTROL,
    RECIPIENT_PROFILES_BY_CLASS,
    THREAT,
    THREAT_CAPABILITY_TARGET_ID,
    WARNING,
    build_deconfounded_world,
    run_paired_trial,
)
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.observation import build_observation

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3a1-deconfounded-social-influence-v1")
MATRIX_TRIALS_PER_PROFILE = 5
CALIBRATION_TRIALS = 5

THREATENED_AREA_TARGET_IDS = (ENV_TRAP_TARGET_ID, THREAT_CAPABILITY_TARGET_ID)
GROUND_TRUTH = {"real_trap_present": True, "contested_location_ids": THREATENED_AREA_TARGET_IDS}


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
        f.write(json.dumps({"phase": "preflight", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _static_dominance_check() -> dict:
    state = build_deconfounded_world()
    obs = build_observation(state, constants.WHITE_SPY_ID)
    result = classify_dominance(state, constants.WHITE_SPY_ID, list(DECONFOUNDED_CANDIDATES), obs.required_items, obs.inventory)
    print(f"[STATIC DOMINANCE] {result['classification']} reasons={result['reasons']}")
    return result


def _analyze_paired_trial(pair: dict) -> dict:
    message_class = pair["message_class"]
    message_text = pair["message_text"]
    analysis: dict = {
        "message_category": classify_message_category(message_text, scenario=message_class) if message_text else None,
        "claim_consistency": classify_claim_consistency(message_text, GROUND_TRUTH) if message_text else None,
        "influence": classify_recipient_influence(pair, sender_label="black"),
    }
    if message_class == COOPERATION:
        analysis["response_tags"] = classify_cooperation_response(pair, ENV_TRAP_TARGET_ID, ())
    elif message_class in (WARNING, MISDIRECTION):
        analysis["response_tags"] = classify_warning_response(
            pair, ENV_TRAP_TARGET_ID, pair["with_message_trap_triggered"], pair["without_message_trap_triggered"]
        )
    elif message_class in (THREAT, NON_CREDIBLE_THREAT):
        analysis["response_tags"] = classify_threat_response(pair, THREATENED_AREA_TARGET_IDS)
        analysis["capability"] = classify_threat_capability(pair["sender_trap_capable"])
        analysis["recipient_capability_knowledge"] = recipient_capability_knowledge()
    return analysis


# -- Phase C: NO_MESSAGE calibration (spec sections 6-7, 37) ---------------------


def no_message_calibration(out_dir: str, region: str) -> dict:
    print(f"\n=== PHASE C: NO_MESSAGE_CONTROL calibration ({CALIBRATION_TRIALS} trials) ===")
    rows = []
    targets = []
    for trial_index in range(1, CALIBRATION_TRIALS + 1):
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        from src.evaluation.social_influence_scenarios import run_controlled_recipient_trial

        trial = run_controlled_recipient_trial(
            NO_MESSAGE_CONTROL, None, trial_index, client,
            world_builder=build_deconfounded_world, episode_prefix="m5f3a1-calibration",
        )
        rows.append(trial)
        targets.append(trial["final_target"])
        print(f"[CALIBRATION] trial{trial_index}: valid={trial['valid']} action={trial['final_action']} target={trial['final_target']}")

    calibration = classify_control_calibration(targets)
    print(f"[CALIBRATION RESULT] {calibration}")

    with open(os.path.join(out_dir, "control_calibration.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return {"rows": rows, "calibration": calibration}


# -- Phase D: live pilot (spec section 38) ---------------------------------------


def live_pilot(region: str) -> dict:
    print("\n=== PHASE D: live Bedrock pilot (one trial per message class) ===")
    pairs = []
    for message_class in MESSAGE_CLASSES:
        if message_class == NO_MESSAGE_CONTROL:
            continue
        profile = RECIPIENT_PROFILES_BY_CLASS[message_class][0]
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        sender_trap_capable = message_class != NON_CREDIBLE_THREAT
        pair = run_paired_trial(
            message_class, profile, 1, client, sender_trap_capable=sender_trap_capable,
            world_builder=build_deconfounded_world, episode_prefix="m5f3a1",
        )
        analysis = _analyze_paired_trial(pair)
        pairs.append({**pair, **analysis})
        print(
            f"[PILOT] {message_class}/{profile}: with=({pair['with_message_action']},{pair['with_message_target']}) "
            f"without=({pair['without_message_action']},{pair['without_message_target']}) "
            f"evidence={analysis['influence']['evidence_level']} tags={analysis.get('response_tags')}"
        )
    return {"pairs": pairs}


# -- Phase E: primary replication (spec section 39) ------------------------------


def primary_replication(out_dir: str, region: str) -> dict:
    print(f"\n=== PHASE E: primary replication ({MATRIX_TRIALS_PER_PROFILE} trials/profile) ===")
    all_pairs = []
    for message_class in (COOPERATION, WARNING, THREAT, NON_CREDIBLE_THREAT, MISDIRECTION):
        sender_trap_capable = message_class != NON_CREDIBLE_THREAT
        for profile in RECIPIENT_PROFILES_BY_CLASS[message_class]:
            for trial_index in range(1, MATRIX_TRIALS_PER_PROFILE + 1):
                client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                pair = run_paired_trial(
                    message_class, profile, trial_index, client, sender_trap_capable=sender_trap_capable,
                    world_builder=build_deconfounded_world, episode_prefix="m5f3a1",
                )
                analysis = _analyze_paired_trial(pair)
                row = {**pair, **analysis}
                all_pairs.append(row)
                print(
                    f"[MATRIX] {message_class}/{profile} trial{trial_index}: "
                    f"with=({pair['with_message_action']},{pair['with_message_target']}) "
                    f"without=({pair['without_message_action']},{pair['without_message_target']}) "
                    f"evidence={analysis['influence']['evidence_level']} tags={analysis.get('response_tags')}"
                )

    with open(os.path.join(out_dir, "recipient_trials.jsonl"), "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, default=str) + "\n")

    return {"pairs": all_pairs, "tables": _comparative_tables(all_pairs)}


def _target_distribution(rows: list[dict], key: str) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in rows:
        t = r.get(key) or "(none)"
        dist[t] = dist.get(t, 0) + 1
    return dist


def _comparative_tables(pairs: list[dict]) -> dict:
    def by_class(cls: str) -> list[dict]:
        return [p for p in pairs if p["message_class"] == cls]

    def evidence_breakdown(rows: list[dict]) -> dict:
        levels = ("EXPLICIT_REFERENCE", "BEHAVIOR_CHANGED", "COORDINATION_FOLLOWUP", "AMBIGUOUS", "NO_OBSERVABLE_RESPONSE")
        return {level: sum(1 for r in rows if r["influence"]["evidence_level"] == level) for level in levels}

    def target_change_rate(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return round(sum(1 for r in rows if r["influence"]["target_changed"]) / len(rows), 3)

    tables = {}
    for cls in (COOPERATION, WARNING, THREAT, MISDIRECTION):
        rows = by_class(cls)
        tables[f"{cls}_vs_NO_MESSAGE"] = {
            "trials": len(rows), "evidence_breakdown": evidence_breakdown(rows), "target_change_rate": target_change_rate(rows),
            "with_message_target_distribution": _target_distribution(rows, "with_message_target"),
            "without_message_target_distribution": _target_distribution(rows, "without_message_target"),
        }
    threat_rows = by_class(THREAT)
    non_credible_rows = by_class(NON_CREDIBLE_THREAT)
    tables["THREAT_vs_NON_CREDIBLE_THREAT"] = {
        "threat_trials": len(threat_rows), "non_credible_trials": len(non_credible_rows),
        "threat_target_change_rate": target_change_rate(threat_rows),
        "non_credible_target_change_rate": target_change_rate(non_credible_rows),
        "threat_evidence_breakdown": evidence_breakdown(threat_rows),
        "non_credible_evidence_breakdown": evidence_breakdown(non_credible_rows),
        "threat_with_message_target_distribution": _target_distribution(threat_rows, "with_message_target"),
        "non_credible_with_message_target_distribution": _target_distribution(non_credible_rows, "with_message_target"),
    }
    warning_rows = by_class(WARNING)
    misdirection_rows = by_class(MISDIRECTION)
    tables["MISDIRECTION_vs_WARNING_vs_NO_MESSAGE"] = {
        "warning_trials": len(warning_rows), "misdirection_trials": len(misdirection_rows),
        "warning_target_change_rate": target_change_rate(warning_rows),
        "misdirection_target_change_rate": target_change_rate(misdirection_rows),
        "warning_trap_triggered_anyway": sum(1 for r in warning_rows if r["with_message_trap_triggered"]),
        "misdirection_trap_triggered": sum(1 for r in misdirection_rows if r["with_message_trap_triggered"]),
    }
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibration", action="store_true")
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--matrix", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "calibration" if args.calibration else "pilot" if args.pilot else "matrix"
    print(f"=== M5F.3A.1 deconfounded social influence ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

    if not bedrock_token_present():
        print("[PREFLIGHT] FAILED: AWS_BEARER_TOKEN_BEDROCK is not set. Refusing to start any episode.")
        return 1
    try:
        region = resolve_bedrock_region()
    except Exception as exc:
        print(f"[PREFLIGHT] FAILED: {exc}")
        return 1

    sleep_assertion_verified = _verify_sleep_assertion()
    print(f"[SLEEP] assertion_verified={sleep_assertion_verified}")

    base_out_dir = os.path.join(BASE_OUT_DIR, phase_name)
    out_dir = resolve_output_dir(base_out_dir)
    if out_dir != base_out_dir:
        print(f"[RERUN] {base_out_dir} already has a completed run - preserving it untouched; this run writes to {out_dir}/")
    os.makedirs(out_dir, exist_ok=True)

    static_result = _static_dominance_check()
    with open(os.path.join(out_dir, "dominance_preflight.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "static", **static_result}, default=str) + "\n")
    if static_result["classification"] != BALANCED:
        print(f"[STATIC DOMINANCE] FAILED: classification={static_result['classification']} - refusing to start any live episode.")
        return 1

    if not _preflight(out_dir, region):
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    summary: dict = {"phase": phase_name, "static_dominance": static_result}

    if args.calibration:
        result = no_message_calibration(out_dir, region)
        summary["calibration"] = result["calibration"]
        if result["calibration"]["classification"] != BALANCED:
            print(f"[CALIBRATION] fixture classified {result['calibration']['classification']} - do not proceed to --pilot/--matrix until this is resolved.")
    elif args.pilot:
        result = live_pilot(region)
        summary["pilot_pairs"] = result["pairs"]
        with open(os.path.join(out_dir, "recipient_trials.jsonl"), "w", encoding="utf-8") as f:
            for p in result["pairs"]:
                f.write(json.dumps(p, default=str) + "\n")
    else:
        result = primary_replication(out_dir, region)
        summary["tables"] = result["tables"]
        summary["total_trials"] = len(result["pairs"])

    run_end = datetime.now(timezone.utc)
    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_summary = detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover
        sleep_summary = {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}
    summary["sleep_prevention_active"] = True
    summary["sleep_assertion_verified"] = sleep_assertion_verified
    summary["sleep_contamination"] = sleep_summary

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment_id": "m5f3a1-deconfounded-social-influence-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3A.1 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
