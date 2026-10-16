"""M5F.3A.2: automated live-diagnostic entry point for the symmetric
strategic-choice harness.

    python -m src.evaluation.run_m5f3a2 --calibration  # Stage 1: NO_MESSAGE, both layouts, hard-stop checks
    python -m src.evaluation.run_m5f3a2 --treatments    # Stage 2: one profile, all 5 social treatment classes

Stage 0 (static symmetry audit) is covered by
tests/test_m5f3a2_symmetric_harness.py and this run's own
`symmetry_audit.md` - not this driver. `--treatments` should only be run
after manually inspecting `--calibration`'s own hard-stop checks (spec
sections 22-24, 63-65) - this driver does not auto-gate on a prior run's
saved result, the same operating discipline M5F.3A.1's driver used.

Note on spec sections 14/15 ("geometry mirror" vs "label swap"): this
harness's `alpha_on_left` toggle is the SAME single manipulation read two
ways (there is only one bit of freedom in a two-branch structure: which
physical side gets which label) - not two independent axes. Calibration
below runs `alpha_on_left` at True and False (10 trials each, 20 total,
matching spec section 20's recommended total), and a stuck preference is
read as side bias if it follows the physical side across the flip, or
label bias if it follows the id.
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
from src.evaluation.social_influence_metrics import classify_threat_capability, recipient_capability_knowledge
from src.evaluation.symmetric_harness import (
    COOPERATION,
    MISDIRECTION,
    NON_CREDIBLE_THREAT,
    NO_MESSAGE_CONTROL,
    THREAT,
    WARNING,
    run_symmetric_trial,
    static_symmetry_audit,
)
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.personality import OPPORTUNIST

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3a2-symmetric-strategic-choice-v1")
CALIBRATION_TRIALS_PER_LAYOUT = 10
STAGE2_TRIALS_PER_CONDITION = 5
OTHER_RATE_STOP_THRESHOLD = 0.20

GROUND_TRUTH = {"real_trap_present": True, "contested_location_ids": ("objective_alpha", "objective_beta")}


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


def _distribution(rows: list[dict]) -> dict[str, int]:
    dist = {"alpha": 0, "beta": 0, "OTHER": 0}
    for r in rows:
        dist[r["resolved_branch"]] = dist.get(r["resolved_branch"], 0) + 1
    return dist


# -- Stage 1: NO_MESSAGE calibration (spec sections 19-24, 63-65) ----------------


def calibration(out_dir: str, region: str, *, neutral_ids: bool = False) -> dict:
    print(f"\n=== STAGE 1: NO_MESSAGE calibration ({CALIBRATION_TRIALS_PER_LAYOUT} trials x 2 layouts, neutral_ids={neutral_ids}) ===")
    rows = []
    for alpha_on_left in (True, False):
        for trial_index in range(1, CALIBRATION_TRIALS_PER_LAYOUT + 1):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            trial = run_symmetric_trial(
                NO_MESSAGE_CONTROL, None, OPPORTUNIST, trial_index, client, alpha_on_left=alpha_on_left, neutral_ids=neutral_ids,
            )
            rows.append(trial)
            print(f"[CALIBRATION] alpha_on_left={alpha_on_left} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']}")

    with open(os.path.join(out_dir, "baseline_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    left_rows = [r for r in rows if r["alpha_on_left"]]
    right_rows = [r for r in rows if not r["alpha_on_left"]]
    left_dist = _distribution(left_rows)
    right_dist = _distribution(right_rows)
    overall_dist = _distribution(rows)
    other_rate = overall_dist["OTHER"] / len(rows) if rows else None

    # Side bias (spec section 23): does the SAME PHYSICAL SIDE win
    # regardless of which label is on it? alpha_on_left=True means alpha
    # is physically left; alpha_on_left=False means beta is physically
    # left. "always picks the physically-left object" would show as
    # left_dist favoring alpha AND right_dist favoring beta.
    physically_left_wins_left_layout = left_dist["alpha"]
    physically_left_wins_right_layout = right_dist["beta"]
    side_bias_suspected = (
        physically_left_wins_left_layout >= 0.8 * len(left_rows) and physically_left_wins_right_layout >= 0.8 * len(right_rows)
        if left_rows and right_rows else False
    )

    # Label bias (spec section 24): does the SAME LABEL win regardless of
    # physical side?
    label_bias_suspected = (
        left_dist["alpha"] >= 0.8 * len(left_rows) and right_dist["alpha"] >= 0.8 * len(right_rows)
        if left_rows and right_rows else False
    )

    hard_stop_20_of_20 = len(rows) == 20 and (overall_dist["alpha"] == 20 or overall_dist["beta"] == 20)

    result = {
        "trials": len(rows),
        "left_layout_distribution": left_dist,
        "right_layout_distribution": right_dist,
        "overall_distribution": overall_dist,
        "other_rate": other_rate,
        "side_bias_suspected": side_bias_suspected,
        "label_bias_suspected": label_bias_suspected,
        "hard_stop_20_of_20_same_objective": hard_stop_20_of_20,
        "hard_stop_other_rate_exceeds_20pct": bool(other_rate is not None and other_rate > OTHER_RATE_STOP_THRESHOLD),
    }
    print(f"[CALIBRATION RESULT] {result}")
    return {"rows": rows, "result": result}


# -- Stage 2: treatment classes (spec sections 25-38, 43-46, 48-50) --------------


def _analyze_trial(trial: dict) -> dict:
    message_class = trial["message_class"]
    text = trial["message_text"]
    analysis: dict = {
        "message_category": classify_message_category(text, scenario=message_class) if text else None,
        "claim_consistency": classify_claim_consistency(text, GROUND_TRUTH) if text else None,
    }
    if message_class in (THREAT, NON_CREDIBLE_THREAT):
        analysis["capability"] = classify_threat_capability(trial["sender_trap_capable"])
        analysis["recipient_capability_knowledge"] = recipient_capability_knowledge()
    return analysis


def treatments(out_dir: str, region: str, *, neutral_ids: bool = True) -> dict:
    # Defaults to neutral_ids=True: Stage 1 calibration with plain
    # "alpha"/"beta" ids found a clean label bias (see summary.md), so
    # treatments only ever run against the re-audited neutral-id world -
    # never against the plain-id world calibration already rejected.
    print(f"\n=== STAGE 2: social treatments ({STAGE2_TRIALS_PER_CONDITION} trials/condition, one profile: {OPPORTUNIST}, neutral_ids={neutral_ids}) ===")
    all_rows = []
    for message_class in (COOPERATION, WARNING, THREAT, NON_CREDIBLE_THREAT, MISDIRECTION):
        for target_branch in ("alpha", "beta"):
            sender_trap_capable = message_class != NON_CREDIBLE_THREAT
            for trial_index in range(1, STAGE2_TRIALS_PER_CONDITION + 1):
                client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                trial = run_symmetric_trial(
                    message_class, target_branch, OPPORTUNIST, trial_index, client,
                    sender_trap_capable=sender_trap_capable, neutral_ids=neutral_ids,
                )
                row = {**trial, **_analyze_trial(trial)}
                all_rows.append(row)
                print(f"[TREATMENT] {message_class}/{target_branch} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']}")

    with open(os.path.join(out_dir, "treatment_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, default=str) + "\n")

    tables = {}
    for message_class in (COOPERATION, WARNING, THREAT, NON_CREDIBLE_THREAT, MISDIRECTION):
        for target_branch in ("alpha", "beta"):
            key = f"{message_class}_{target_branch}"
            rows = [r for r in all_rows if r["message_class"] == message_class and r["target_branch"] == target_branch]
            tables[key] = _distribution(rows)
    return {"rows": all_rows, "tables": tables}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibration", action="store_true")
    group.add_argument("--neutral-calibration", action="store_true", help="re-audit calibration with hex-token ids (spec section 24's label-bias remedy)")
    group.add_argument("--treatments", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "calibration" if args.calibration else "neutral-calibration" if args.neutral_calibration else "treatments"
    print(f"=== M5F.3A.2 symmetric strategic choice ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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
    with open(os.path.join(out_dir, "static_symmetry_audit.json"), "w", encoding="utf-8") as f:
        json.dump(static_result, f, indent=2, default=str)
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
        result = calibration(out_dir, region, neutral_ids=False)
        summary["calibration"] = result["result"]
    elif args.neutral_calibration:
        result = calibration(out_dir, region, neutral_ids=True)
        summary["calibration"] = result["result"]
    else:
        result = treatments(out_dir, region)
        summary["tables"] = result["tables"]
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
        json.dump({"experiment_id": "m5f3a2-symmetric-strategic-choice-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3A.2 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
