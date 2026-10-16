"""M5F.3A.4: automated live-diagnostic entry point for candidate
representation decoupling.

    python -m src.evaluation.run_m5f3a4 --current            # Stage C: fresh CURRENT replication
    python -m src.evaluation.run_m5f3a4 --decoupled           # Stage C: B1-B4 attribution
    python -m src.evaluation.run_m5f3a4 --single-candidate    # Stage E: SINGLE_CANDIDATE_VIEW calibration

Stage A (serialization audit) is `serialization_audit.md`. Stage B/D
(representation infrastructure) is `src/evaluation/representation_control.py`
+ tests/test_m5f3a4_representation_control.py - not this driver.
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
from src.evaluation.order_control import OrderPolicy
from src.evaluation.representation_control import (
    count_candidate_occurrences,
    decoupled_model_payload_transform,
    single_candidate_model_payload_transform,
)
from src.evaluation.symmetric_harness import NO_MESSAGE_CONTROL, alpha_beta_object_ids, run_symmetric_trial
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.personality import OPPORTUNIST

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3a4-candidate-representation-decoupling-v1")
CURRENT_TRIALS_PER_ORDER = 12
DECOUPLED_TRIALS_PER_CONDITION = 8
SINGLE_CANDIDATE_TRIALS_PER_ORDER = 20

CANDIDATES = alpha_beta_object_ids(True, neutral_ids=True)


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


def _rate(numer: int, denom: int) -> float | None:
    return round(numer / denom, 3) if denom else None


# -- Stage C: fresh CURRENT replication -------------------------------------------


def current_replication(out_dir: str, region: str) -> dict:
    print(f"\n=== STAGE C: fresh CURRENT replication ({CURRENT_TRIALS_PER_ORDER} trials/order) ===")
    rows = []
    for policy, label in ((OrderPolicy.CANONICAL, "ORDER_A"), (OrderPolicy.REVERSED, "ORDER_B")):
        for trial_index in range(1, CURRENT_TRIALS_PER_ORDER + 1):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            from src.evaluation.order_control import permute_observation_objects

            sink = {}

            def transform(obs, sink=sink, policy=policy, trial_index=trial_index):
                permuted, meta = permute_observation_objects(obs, policy, trial_index)
                sink["meta"] = meta
                return permuted

            trial = run_symmetric_trial(
                NO_MESSAGE_CONTROL, None, OPPORTUNIST, trial_index, client, neutral_ids=True,
                observation_transform=transform, episode_prefix=f"m5f3a4-current-{label}",
            )
            row = {**trial, "representation_mode": "CURRENT", "order_label": label, "order_metadata": sink.get("meta", {})}
            rows.append(row)
            print(f"[CURRENT] {label} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']}")

    with open(os.path.join(out_dir, "current_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    valid_rows = [r for r in rows if r["valid"]]
    order_a = [r for r in valid_rows if r["order_label"] == "ORDER_A"]
    order_b = [r for r in valid_rows if r["order_label"] == "ORDER_B"]
    metrics = {
        "trials": len(rows), "valid_trials": len(valid_rows),
        "ORDER_A": {"n": len(order_a), "alpha": sum(1 for r in order_a if r["resolved_branch"] == "alpha"), "beta": sum(1 for r in order_a if r["resolved_branch"] == "beta")},
        "ORDER_B": {"n": len(order_b), "alpha": sum(1 for r in order_b if r["resolved_branch"] == "alpha"), "beta": sum(1 for r in order_b if r["resolved_branch"] == "beta")},
    }
    print(f"[CURRENT RESULT] {metrics}")
    return {"rows": rows, "metrics": metrics}


# -- Stage C: DECOUPLED B1-B4 attribution -----------------------------------------


def decoupled_attribution(out_dir: str, region: str) -> dict:
    print(f"\n=== STAGE C: DECOUPLED B1/B3/B4 attribution ({DECOUPLED_TRIALS_PER_CONDITION} trials/condition) ===")
    # B2 ("objects reversed, legal_actions canonical, room.objects
    # canonical") is deliberately NOT run: given "objects order" has no
    # independent serialized manifestation separate from what
    # legal_actions/room.objects themselves display (serialization_audit.md),
    # forcing both downstream lists back to true canonical while "objects"
    # is reversed produces a payload BYTE-IDENTICAL to B1 - a degenerate,
    # not merely redundant, condition for this harness's architecture.
    # Running it would just re-measure B1 a second time under a different
    # label. M5F.3A.3's own Stage B (40 trials, ORDER_A/ORDER_B) already
    # covers the "objects order changes and both downstream lists inherit
    # that change" condition this milestone's CURRENT replication repeats
    # fresh above.
    conditions = {
        "B1": (OrderPolicy.CANONICAL, OrderPolicy.CANONICAL),
        "B3_legal_actions_reversed": (OrderPolicy.REVERSED, OrderPolicy.CANONICAL),
        "B4_room_objects_reversed": (OrderPolicy.CANONICAL, OrderPolicy.REVERSED),
    }
    rows = []
    for label, (legal_actions_policy, room_objects_policy) in conditions.items():
        for trial_index in range(1, DECOUPLED_TRIALS_PER_CONDITION + 1):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            sink: dict = {}
            transform = decoupled_model_payload_transform(
                legal_actions_policy=legal_actions_policy, room_objects_policy=room_objects_policy,
                trial_index=trial_index, candidate_ids=CANDIDATES, metadata_sink=sink,
            )
            trial = run_symmetric_trial(
                NO_MESSAGE_CONTROL, None, OPPORTUNIST, trial_index, client, neutral_ids=True,
                model_payload_transform=transform, episode_prefix=f"m5f3a4-decoupled-{label}",
            )
            row = {**trial, "representation_mode": "DECOUPLED", "condition": label, "order_metadata": sink.get("metadata", {})}
            rows.append(row)
            print(f"[DECOUPLED] {label} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']}")

    with open(os.path.join(out_dir, "decoupled_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    metrics = {}
    for label in conditions:
        condition_rows = [r for r in rows if r["condition"] == label and r["valid"]]
        metrics[label] = {
            "n": len(condition_rows),
            "alpha": sum(1 for r in condition_rows if r["resolved_branch"] == "alpha"),
            "beta": sum(1 for r in condition_rows if r["resolved_branch"] == "beta"),
            "other": sum(1 for r in condition_rows if r["resolved_branch"] == "OTHER"),
        }
    print(f"[DECOUPLED RESULT] {metrics}")
    return {"rows": rows, "metrics": metrics}


# -- Stage E: SINGLE_CANDIDATE_VIEW calibration -----------------------------------


def single_candidate_calibration(out_dir: str, region: str) -> dict:
    print(f"\n=== STAGE E: SINGLE_CANDIDATE_VIEW calibration ({SINGLE_CANDIDATE_TRIALS_PER_ORDER} trials/order) ===")
    rows = []
    prompt_snapshot_saved = False
    for policy, label in ((OrderPolicy.CANONICAL, "CAND1_FIRST"), (OrderPolicy.REVERSED, "CAND2_FIRST")):
        for trial_index in range(1, SINGLE_CANDIDATE_TRIALS_PER_ORDER + 1):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            sink: dict = {}
            transform = single_candidate_model_payload_transform(
                candidate_ids=CANDIDATES, order_policy=policy, trial_index=trial_index, metadata_sink=sink,
            )
            captured_payload = {}

            def snapshot_transform(payload, transform=transform, captured_payload=captured_payload):
                result = transform(payload)
                captured_payload["payload"] = result
                return result

            trial = run_symmetric_trial(
                NO_MESSAGE_CONTROL, None, OPPORTUNIST, trial_index, client, neutral_ids=True,
                model_payload_transform=snapshot_transform, episode_prefix=f"m5f3a4-single-{label}",
            )
            occurrences = {cid: count_candidate_occurrences(captured_payload.get("payload", {}), cid) for cid in CANDIDATES}
            row = {**trial, "representation_mode": "SINGLE_CANDIDATE_VIEW", "order_label": label, "order_metadata": sink.get("metadata", {}), "candidate_occurrences": occurrences}
            rows.append(row)
            print(f"[SINGLE] {label} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']}")

            if not prompt_snapshot_saved:
                os.makedirs(os.path.join(out_dir, "prompt_snapshots"), exist_ok=True)
                with open(os.path.join(out_dir, "prompt_snapshots", f"{label}_trial{trial_index}.json"), "w", encoding="utf-8") as f:
                    json.dump(captured_payload.get("payload", {}), f, indent=2, default=str)
                prompt_snapshot_saved = True

    with open(os.path.join(out_dir, "single_candidate_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    valid_rows = [r for r in rows if r["valid"]]
    cand1_first = [r for r in valid_rows if r["order_label"] == "CAND1_FIRST"]
    cand2_first = [r for r in valid_rows if r["order_label"] == "CAND2_FIRST"]
    metrics = {
        "trials": len(rows), "valid_trials": len(valid_rows),
        "CAND1_FIRST": {"n": len(cand1_first), "cand1": sum(1 for r in cand1_first if r["resolved_branch"] == "alpha"), "cand2": sum(1 for r in cand1_first if r["resolved_branch"] == "beta"), "other": sum(1 for r in cand1_first if r["resolved_branch"] == "OTHER")},
        "CAND2_FIRST": {"n": len(cand2_first), "cand1": sum(1 for r in cand2_first if r["resolved_branch"] == "alpha"), "cand2": sum(1 for r in cand2_first if r["resolved_branch"] == "beta"), "other": sum(1 for r in cand2_first if r["resolved_branch"] == "OTHER")},
    }
    print(f"[SINGLE RESULT] {metrics}")
    return {"rows": rows, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--current", action="store_true")
    group.add_argument("--decoupled", action="store_true")
    group.add_argument("--single-candidate", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "current" if args.current else "decoupled" if args.decoupled else "single-candidate"
    print(f"=== M5F.3A.4 candidate representation decoupling ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    if not _preflight(out_dir, region):
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    summary: dict = {"phase": phase_name}

    if args.current:
        result = current_replication(out_dir, region)
    elif args.decoupled:
        result = decoupled_attribution(out_dir, region)
    else:
        result = single_candidate_calibration(out_dir, region)
    summary["metrics"] = result["metrics"]

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
        json.dump({"experiment_id": "m5f3a4-candidate-representation-decoupling-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3A.4 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
