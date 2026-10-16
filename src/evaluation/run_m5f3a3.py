"""M5F.3A.3: automated live-diagnostic entry point for observation
order-bias quantification and counterbalancing.

    python -m src.evaluation.run_m5f3a3 --object-order       # Stage B: 40 trials, ORDER_A vs ORDER_B
    python -m src.evaluation.run_m5f3a3 --message-order       # Stage C: message-order probe (independent structure)
    python -m src.evaluation.run_m5f3a3 --counterbalanced-baseline  # Stage E: 20 trials, COUNTERBALANCED objects
    python -m src.evaluation.run_m5f3a3 --treatment-pilot     # Stage F: small pilot, 5 classes x 2 branches x 2 orders

Stage A (root-cause audit) is `order_audit.md`. Stage D (counterbalancing
infrastructure) is `src/evaluation/order_control.py` +
tests/test_m5f3a3_order_control.py - not this driver.
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
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.order_control import OrderPolicy, permute_observation_messages, permute_observation_objects
from src.evaluation.scenario_harness import _last_decision_event, _run_one_decision
from src.evaluation.social_influence_metrics import classify_threat_capability, recipient_capability_knowledge
from src.evaluation.symmetric_harness import (
    COOPERATION,
    MISDIRECTION,
    NON_CREDIBLE_THREAT,
    NO_MESSAGE_CONTROL,
    THREAT,
    WARNING,
    alpha_beta_object_ids,
    build_symmetric_state,
    message_text,
    resolve_branch,
    run_symmetric_trial,
)
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.personality import OPPORTUNIST
from src.llm.telemetry import TelemetryLogger
from src.observation import build_observation

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3a3-observation-order-bias-v1")
OBJECT_ORDER_TRIALS_PER_CONDITION = 20
MESSAGE_ORDER_TRIALS_PER_CONDITION = 8
COUNTERBALANCED_BASELINE_TRIALS = 20


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


def _objects_transform_capturing(policy: OrderPolicy, trial_index: int, sink: dict):
    def transform(observation):
        permuted, meta = permute_observation_objects(observation, policy, trial_index)
        sink["meta"] = meta
        return permuted

    return transform


def _classify_position_bias(first_rate: float, n: int) -> str:
    if n == 0:
        return "MIXED"
    if first_rate >= 0.85:
        return "STRONG_FIRST_POSITION_BIAS"
    if first_rate >= 0.65:
        return "MODERATE_FIRST_POSITION_BIAS"
    if first_rate <= 0.5:
        return "WEAK_OR_NO_POSITION_BIAS"
    return "MIXED"


# -- Stage B: object-order primacy quantification (spec sections 8-12) ----------


def object_order_probe(out_dir: str, region: str) -> dict:
    print(f"\n=== STAGE B: object-order primacy quantification ({OBJECT_ORDER_TRIALS_PER_CONDITION} trials/order) ===")
    rows = []
    for policy, label in ((OrderPolicy.CANONICAL, "ORDER_A"), (OrderPolicy.REVERSED, "ORDER_B")):
        for trial_index in range(1, OBJECT_ORDER_TRIALS_PER_CONDITION + 1):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            sink: dict = {}
            transform = _objects_transform_capturing(policy, trial_index, sink)
            trial = run_symmetric_trial(
                NO_MESSAGE_CONTROL, None, OPPORTUNIST, trial_index, client, neutral_ids=True,
                observation_transform=transform, episode_prefix=f"m5f3a3-objorder-{label}",
            )
            meta = sink.get("meta", {})
            row = {**trial, "order_label": label, "order_metadata": meta}
            rows.append(row)
            print(f"[OBJECT-ORDER] {label} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']}")

    with open(os.path.join(out_dir, "object_order_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    # Position resolution: for each trial, "first-position object" is the
    # FIRST objective console (door entries never selected as final
    # targets meaningfully here - both doors always precede both
    # objectives in canonical order, and permutation reverses the WHOLE
    # objects list including doors, so the first OBJECTIVE in presented
    # order is whichever ends up last after a full-list reversal of
    # [door_alpha, door_beta, obj_alpha, obj_beta]).
    valid_rows = [r for r in rows if r["valid"]]
    first_position_selections = 0
    second_position_selections = 0
    other_selections = 0
    for r in valid_rows:
        presented_objectives = [oid for oid in r["order_metadata"]["presented_order"] if oid.startswith("objective_")]
        final_target = r["final_target"]
        if not presented_objectives or final_target not in presented_objectives:
            other_selections += 1
        elif final_target == presented_objectives[0]:
            first_position_selections += 1
        else:
            second_position_selections += 1

    n = len(valid_rows)
    first_rate = round(first_position_selections / n, 3) if n else None
    alpha_count = sum(1 for r in valid_rows if r["resolved_branch"] == "alpha")
    beta_count = sum(1 for r in valid_rows if r["resolved_branch"] == "beta")
    classification = _classify_position_bias(first_rate or 0.0, n)

    metrics = {
        "trials": len(rows), "valid_trials": n,
        "first_position_selections": first_position_selections,
        "second_position_selections": second_position_selections,
        "other_selections": other_selections,
        "alpha_selections": alpha_count, "beta_selections": beta_count,
        "first_position_selection_rate": first_rate,
        "classification": classification,
    }
    print(f"[OBJECT-ORDER RESULT] {metrics}")
    return {"rows": rows, "metrics": metrics}


# -- Stage C: message-order probe (spec sections 17-18) --------------------------


def message_order_probe(out_dir: str, region: str) -> dict:
    print(f"\n=== STAGE C: message-order probe ({MESSAGE_ORDER_TRIALS_PER_CONDITION} trials/order) ===")
    rows = []
    for order_label, first_branch in (("A_FIRST", "alpha"), ("B_FIRST", "beta")):
        second_branch = "beta" if first_branch == "alpha" else "alpha"
        for trial_index in range(1, MESSAGE_ORDER_TRIALS_PER_CONDITION + 1):
            state = build_symmetric_state(neutral_ids=True)
            alpha_id, beta_id = alpha_beta_object_ids(True, neutral_ids=True)
            text_first = message_text(MISDIRECTION, first_branch, neutral_ids=True)
            text_second = message_text(MISDIRECTION, second_branch, neutral_ids=True)
            state.messages.append({
                "message_id": "m1", "sender_actor_id": constants.BLACK_SPY_ID, "recipient_actor_id": constants.WHITE_SPY_ID,
                "game_time": 0.0, "text": text_first,
            })
            state.messages.append({
                "message_id": "m2", "sender_actor_id": constants.BLACK_SPY_ID, "recipient_actor_id": constants.WHITE_SPY_ID,
                "game_time": 1.0, "text": text_second,
            })
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            telemetry = TelemetryLogger()
            episode_id = f"m5f3a3-msgorder-{order_label}-trial{trial_index}"
            _run_one_decision(state, constants.WHITE_SPY_ID, OPPORTUNIST, client, episode_id, telemetry, wall_timeout=180.0)
            decision = _last_decision_event(telemetry, episode_id)
            executed = decision.get("executed_action") or {}
            final_target = executed.get("target_id")
            reason = (decision.get("raw_response") or {}).get("reason") or ""
            row = {
                "order_label": order_label, "trial_index": trial_index,
                "message_first": text_first, "message_second": text_second,
                "valid": bool(decision.get("final_model_action_valid")),
                "final_action": executed.get("action"), "final_target": final_target,
                "resolved_branch": resolve_branch(final_target, neutral_ids=True),
                "reason": reason,
                "explicit_reference_to_first": (first_branch in reason.lower()) or (text_first.split()[1] in reason if len(text_first.split()) > 1 else False),
            }
            rows.append(row)
            print(f"[MESSAGE-ORDER] {order_label} trial{trial_index}: valid={row['valid']} target={final_target} branch={row['resolved_branch']}")

    with open(os.path.join(out_dir, "message_order_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    valid_rows = [r for r in rows if r["valid"]]
    a_first_rows = [r for r in valid_rows if r["order_label"] == "A_FIRST"]
    b_first_rows = [r for r in valid_rows if r["order_label"] == "B_FIRST"]
    metrics = {
        "trials": len(rows), "valid_trials": len(valid_rows),
        "a_first_alpha_selected": sum(1 for r in a_first_rows if r["resolved_branch"] == "alpha"),
        "a_first_beta_selected": sum(1 for r in a_first_rows if r["resolved_branch"] == "beta"),
        "b_first_alpha_selected": sum(1 for r in b_first_rows if r["resolved_branch"] == "alpha"),
        "b_first_beta_selected": sum(1 for r in b_first_rows if r["resolved_branch"] == "beta"),
    }
    print(f"[MESSAGE-ORDER RESULT] {metrics}")
    return {"rows": rows, "metrics": metrics}


# -- Stage E: counterbalanced NO_MESSAGE baseline (spec section 33) -------------


def counterbalanced_baseline(out_dir: str, region: str) -> dict:
    print(f"\n=== STAGE E: counterbalanced NO_MESSAGE baseline ({COUNTERBALANCED_BASELINE_TRIALS} trials) ===")
    rows = []
    for trial_index in range(1, COUNTERBALANCED_BASELINE_TRIALS + 1):
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        sink: dict = {}
        transform = _objects_transform_capturing(OrderPolicy.COUNTERBALANCED, trial_index, sink)
        trial = run_symmetric_trial(
            NO_MESSAGE_CONTROL, None, OPPORTUNIST, trial_index, client, neutral_ids=True,
            observation_transform=transform, episode_prefix="m5f3a3-counterbalanced",
        )
        row = {**trial, "order_metadata": sink.get("meta", {})}
        rows.append(row)
        print(f"[COUNTERBALANCED] trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']} first={sink.get('meta',{}).get('presented_order',[None])[-1]}")

    with open(os.path.join(out_dir, "counterbalanced_baseline.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    valid_rows = [r for r in rows if r["valid"]]
    alpha_first_trials = [r for r in valid_rows if r["order_metadata"]["trial_index"] % 2 == 0]
    beta_first_trials = [r for r in valid_rows if r["order_metadata"]["trial_index"] % 2 == 1]
    metrics = {
        "trials": len(rows), "valid_trials": len(valid_rows),
        "alpha_first_count": len(alpha_first_trials), "beta_first_count": len(beta_first_trials),
        "alpha_selections": sum(1 for r in valid_rows if r["resolved_branch"] == "alpha"),
        "beta_selections": sum(1 for r in valid_rows if r["resolved_branch"] == "beta"),
        "other_selections": sum(1 for r in valid_rows if r["resolved_branch"] == "OTHER"),
    }
    print(f"[COUNTERBALANCED RESULT] {metrics}")
    return {"rows": rows, "metrics": metrics}


# -- Stage F: social treatment pilot, crossed with order (spec section 54) ------


def treatment_pilot(out_dir: str, region: str) -> dict:
    print("\n=== STAGE F: social treatment pilot (crossed with object order) ===")
    rows = []
    for message_class in (COOPERATION, WARNING, THREAT, NON_CREDIBLE_THREAT, MISDIRECTION):
        sender_trap_capable = message_class != NON_CREDIBLE_THREAT
        for target_branch in ("alpha", "beta"):
            for trial_index, policy in ((1, OrderPolicy.CANONICAL), (2, OrderPolicy.REVERSED)):
                client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                sink: dict = {}
                transform = _objects_transform_capturing(policy, trial_index, sink)
                trial = run_symmetric_trial(
                    message_class, target_branch, OPPORTUNIST, trial_index, client, neutral_ids=True,
                    sender_trap_capable=sender_trap_capable, observation_transform=transform,
                    episode_prefix="m5f3a3-pilot",
                )
                meta = sink.get("meta", {})
                presented_objectives = [oid for oid in meta.get("presented_order", []) if oid.startswith("objective_")]
                target_id = alpha_beta_object_ids(True, neutral_ids=True)[0 if target_branch == "alpha" else 1]
                target_first = bool(presented_objectives and presented_objectives[0] == target_id)
                row = {**trial, "order_metadata": meta, "target_serialized_first": target_first}
                rows.append(row)
                print(f"[PILOT] {message_class}/{target_branch} order={policy.value} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} branch={trial['resolved_branch']} target_first={target_first}")

    with open(os.path.join(out_dir, "social_treatment_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--object-order", action="store_true")
    group.add_argument("--message-order", action="store_true")
    group.add_argument("--counterbalanced-baseline", action="store_true")
    group.add_argument("--treatment-pilot", action="store_true")
    args = parser.parse_args(argv)

    phase_name = (
        "object-order" if args.object_order else "message-order" if args.message_order
        else "counterbalanced-baseline" if args.counterbalanced_baseline else "treatment-pilot"
    )
    print(f"=== M5F.3A.3 observation order bias ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    if args.object_order:
        result = object_order_probe(out_dir, region)
        summary["metrics"] = result["metrics"]
    elif args.message_order:
        result = message_order_probe(out_dir, region)
        summary["metrics"] = result["metrics"]
    elif args.counterbalanced_baseline:
        result = counterbalanced_baseline(out_dir, region)
        summary["metrics"] = result["metrics"]
    else:
        result = treatment_pilot(out_dir, region)
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
        json.dump({"experiment_id": "m5f3a3-observation-order-bias-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3A.3 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
