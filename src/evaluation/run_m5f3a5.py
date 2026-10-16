"""M5F.3A.5: automated live-diagnostic entry point for the strategic goal
selection contract.

    python -m src.evaluation.run_m5f3a5 --calibration  # Phase C: 40 NO_MESSAGE decisions, both goal orders
    python -m src.evaluation.run_m5f3a5 --social-pilot   # Phase F: tiny COOPERATION/THREAT pilot (only if calibration gate passes)

Phase A/B (contract design + network-free validation) is `contract_audit.md`
+ tests/test_m5f3a5_strategic_goal.py - not this driver. Phase D (CURRENT
vs STRATEGIC_GOAL comparison) reuses M5F.3A.4's own CURRENT data (same
symmetric world/seed - spec section 37's "do not compare different
seeds/world states") rather than re-spending live budget re-measuring an
already-established baseline; comparison numbers are assembled in
`summary.md` directly from `../m5f3a4-candidate-representation-decoupling-v1/representation_metrics.json`.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.opportunity_metrics import classify_claim_consistency, classify_message_category
from src.evaluation.order_control import OrderPolicy
from src.evaluation.strategic_goal import count_door_move_candidates, strategic_goal_model_payload_transform
from src.evaluation.symmetric_harness import COOPERATION, NO_MESSAGE_CONTROL, THREAT, alpha_beta_object_ids, run_symmetric_trial, static_symmetry_audit
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.personality import OPPORTUNIST

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3a5-strategic-goal-selection-v1")
CALIBRATION_TRIALS_PER_ORDER = 20
SOCIAL_PILOT_TRIALS_PER_CONDITION = 3

CANDIDATES = alpha_beta_object_ids(True, neutral_ids=True)
GROUND_TRUTH = {"real_trap_present": True, "contested_location_ids": CANDIDATES}


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


# -- Phase C: NO_MESSAGE calibration ----------------------------------------------


def calibration(out_dir: str, region: str) -> dict:
    print(f"\n=== PHASE C: strategic-goal NO_MESSAGE calibration ({CALIBRATION_TRIALS_PER_ORDER} trials/order) ===")
    rows = []
    door_selection_count = 0
    for policy, label in ((OrderPolicy.CANONICAL, "GOAL1_FIRST"), (OrderPolicy.REVERSED, "GOAL2_FIRST")):
        for trial_index in range(1, CALIBRATION_TRIALS_PER_ORDER + 1):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            sink: dict = {}
            transform = strategic_goal_model_payload_transform(
                candidate_ids=CANDIDATES, order_policy=policy, trial_index=trial_index, metadata_sink=sink,
            )
            captured: dict = {}

            def snapshot(payload, transform=transform, captured=captured):
                result = transform(payload)
                captured["payload"] = result
                return result

            trial = run_symmetric_trial(
                NO_MESSAGE_CONTROL, None, OPPORTUNIST, trial_index, client, neutral_ids=True,
                model_payload_transform=snapshot, episode_prefix=f"m5f3a5-goal-{label}",
            )
            door_count = count_door_move_candidates(captured.get("payload", {}))
            door_selection_count += door_count
            row = {**trial, "decision_contract": "strategic_goal", "order_label": label, "order_metadata": sink.get("metadata", {}), "door_move_candidates_shown": door_count}
            rows.append(row)
            print(f"[GOAL-CALIBRATION] {label} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']} door_candidates={door_count}")

            if trial_index == 1:
                os.makedirs(os.path.join(out_dir, "prompt_snapshots"), exist_ok=True)
                with open(os.path.join(out_dir, "prompt_snapshots", f"{label}_trial{trial_index}.json"), "w", encoding="utf-8") as f:
                    json.dump(captured.get("payload", {}), f, indent=2, default=str)

    with open(os.path.join(out_dir, "strategic_goal_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    valid_rows = [r for r in rows if r["valid"]]
    goal1_first = [r for r in valid_rows if r["order_label"] == "GOAL1_FIRST"]
    goal2_first = [r for r in valid_rows if r["order_label"] == "GOAL2_FIRST"]
    metrics = {
        "trials": len(rows), "valid_trials": len(valid_rows),
        "door_move_candidates_total": door_selection_count,
        "GOAL1_FIRST": {"n": len(goal1_first), "goal1": sum(1 for r in goal1_first if r["resolved_branch"] == "alpha"), "goal2": sum(1 for r in goal1_first if r["resolved_branch"] == "beta"), "other": sum(1 for r in goal1_first if r["resolved_branch"] == "OTHER")},
        "GOAL2_FIRST": {"n": len(goal2_first), "goal1": sum(1 for r in goal2_first if r["resolved_branch"] == "alpha"), "goal2": sum(1 for r in goal2_first if r["resolved_branch"] == "beta"), "other": sum(1 for r in goal2_first if r["resolved_branch"] == "OTHER")},
    }
    print(f"[GOAL-CALIBRATION RESULT] {metrics}")
    return {"rows": rows, "metrics": metrics}


# -- Phase F: tiny social pilot (only if calibration gate passes) ----------------


def _analyze(message_class: str, text: str | None) -> dict:
    return {
        "message_category": classify_message_category(text, scenario=message_class) if text else None,
        "claim_consistency": classify_claim_consistency(text, GROUND_TRUTH) if text else None,
    }


def social_pilot(out_dir: str, region: str) -> dict:
    print(f"\n=== PHASE F: tiny social pilot (COOPERATION/THREAT, {SOCIAL_PILOT_TRIALS_PER_CONDITION} trials/condition) ===")
    rows = []
    for message_class in (COOPERATION, THREAT):
        for target_branch in ("alpha", "beta"):
            for policy, order_label in ((OrderPolicy.CANONICAL, "GOAL1_FIRST"), (OrderPolicy.REVERSED, "GOAL2_FIRST")):
                for trial_index in range(1, SOCIAL_PILOT_TRIALS_PER_CONDITION + 1):
                    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                    sink: dict = {}
                    transform = strategic_goal_model_payload_transform(
                        candidate_ids=CANDIDATES, order_policy=policy, trial_index=trial_index, metadata_sink=sink,
                    )
                    trial = run_symmetric_trial(
                        message_class, target_branch, OPPORTUNIST, trial_index, client, neutral_ids=True,
                        model_payload_transform=transform, episode_prefix=f"m5f3a5-social-{message_class}-{target_branch}-{order_label}",
                    )
                    row = {**trial, "decision_contract": "strategic_goal", "order_label": order_label, "order_metadata": sink.get("metadata", {}), **_analyze(message_class, trial["message_text"])}
                    rows.append(row)
                    print(f"[SOCIAL PILOT] {message_class}/{target_branch}/{order_label} trial{trial_index}: valid={trial['valid']} branch={trial['resolved_branch']}")

    with open(os.path.join(out_dir, "social_pilot.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibration", action="store_true")
    group.add_argument("--social-pilot", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "calibration" if args.calibration else "social-pilot"
    print(f"=== M5F.3A.5 strategic goal selection ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    static_result = static_symmetry_audit()
    print(f"[STATIC SYMMETRY] {static_result['result']}")
    if static_result["result"] != "SYMMETRIC":
        print("[STATIC SYMMETRY] FAILED - refusing to start any live episode.")
        return 1

    if not _preflight(out_dir, region):
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    summary: dict = {"phase": phase_name, "static_symmetry": static_result["result"]}

    if args.calibration:
        result = calibration(out_dir, region)
        summary["metrics"] = result["metrics"]
    else:
        result = social_pilot(out_dir, region)
        summary["total_trials"] = len(result["rows"])

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
        json.dump({"experiment_id": "m5f3a5-strategic-goal-selection-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3A.5 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
