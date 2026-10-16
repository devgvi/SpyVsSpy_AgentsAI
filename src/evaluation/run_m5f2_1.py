"""M5F.2.1: automated live-diagnostic entry point for communication
opportunity/incentive calibration.

    python -m src.evaluation.run_m5f2_1 --dry-audit     # serialize all 5 scenarios, no live calls
    python -m src.evaluation.run_m5f2_1 --stop-check    # 5 trials (1 per scenario, first-planned profile)
    python -m src.evaluation.run_m5f2_1 --matrix        # full 10 conditions x 10 trials (only if stop-check clears)

Phase A (network-free) is covered by tests/test_m5f2_1_scenario_harness.py
and tests/test_m5f2_1_opportunity_metrics.py - not this driver. Freezes
provider=Bedrock, native_tool_like runtime, same model, frozen M5F.1
personalities and M5F.2 communication policy (spec sections 1-3).
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
from src.evaluation.opportunity_metrics import (
    claim_conflicts_with_sender_plan,
    classify_claim_consistency,
    classify_message_category,
    classify_recipient_influence,
    sender_condition_metrics,
)
from src.evaluation.scenario_harness import (
    SCENARIO_BUILDERS,
    SCENARIO_GROUND_TRUTH,
    SCENARIO_NAMES,
    SCENARIO_PROFILES,
    run_recipient_probe,
    run_sender_trial,
)
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm_controller import enumerate_legal_actions, serialize_observation
from src.observation import build_observation

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f2_1-communication-opportunity-v1")
TRIALS_PER_CONDITION = 10
CONTROL_EXCESSIVE_RATE = 0.70


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


def dry_scenario_audit() -> dict:
    """Spec section 30: before any Bedrock call, serialize each scenario
    and verify the intended opportunity exists without leaking hidden
    state - sender Observation, legal actions, known facts, and (for
    SAFETY_INFORMATION specifically) that the recipient's own view of the
    seeded fact is genuinely still hidden."""
    print("\n=== DRY SCENARIO AUDIT (no live calls) ===")
    audit = {}
    for name in SCENARIO_NAMES:
        state = SCENARIO_BUILDERS[name]()
        sender_obs = build_observation(state, constants.BLACK_SPY_ID)
        legal = enumerate_legal_actions(sender_obs)
        actions = sorted({a["action"] for a in legal})
        non_message = [a for a in actions if a != "send_message"]
        payload = serialize_observation(sender_obs)
        entry = {
            "send_message_legal": "send_message" in actions,
            "non_message_actions": non_message,
            "non_message_action_count": len(non_message),
            "sender_current_room": sender_obs.current_room,
            "sender_nearby_object_id": sender_obs.nearby_object_id,
            "opponent_visible_to_sender": sender_obs.visible_other_actor,
            "ground_truth": SCENARIO_GROUND_TRUTH[name],
        }
        if name == "SAFETY_INFORMATION":
            recipient_obs = build_observation(state, constants.WHITE_SPY_ID)
            entry["sender_trap_status_filing_cabinet_01"] = sender_obs.get("filing_cabinet_01").trap_status
            entry["recipient_trap_status_filing_cabinet_01"] = recipient_obs.get("filing_cabinet_01").trap_status
            entry["hidden_fact_preserved"] = entry["recipient_trap_status_filing_cabinet_01"] == "unknown"
        ok = entry["send_message_legal"] and entry["non_message_action_count"] >= 2
        entry["scenario_valid"] = ok
        print(f"[AUDIT] {name}: send_message={entry['send_message_legal']} non_message={non_message} valid={ok}")
        audit[name] = entry
        assert ok, f"{name}: scenario does not satisfy spec section 4 (send_message + >=2 physical alternatives)"
        if name == "SAFETY_INFORMATION":
            assert entry["hidden_fact_preserved"], "SAFETY_INFORMATION leaked the trap fact to the recipient"
    return audit


def _run_sender_condition(scenario: str, profile: str, n_trials: int, region: str) -> list[dict]:
    trials = []
    for i in range(1, n_trials + 1):
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        result = run_sender_trial(scenario, profile, i, client)
        trials.append(result)
        print(f"[SENDER] {scenario}/{profile} trial {i}: action={result['final_action']} target={result['final_target']} valid={result['valid']}")
    return trials


def _analyze_message(scenario: str, trial: dict) -> dict:
    text = trial.get("final_message")
    ground_truth = SCENARIO_GROUND_TRUTH[scenario]
    return {
        "scenario": scenario, "profile": trial["profile"], "trial_index": trial["trial_index"],
        "episode_id": trial["episode_id"], "recipient_actor_id": trial.get("final_target"),
        "message_text": text, "sender_reason": trial.get("reason"),
        "message_category": classify_message_category(text, scenario=scenario),
        "claim_consistency": classify_claim_consistency(text, ground_truth),
        "claim_conflicts_with_sender_plan": claim_conflicts_with_sender_plan(
            text, trial.get("final_action"), trial.get("final_target"), ground_truth["contested_location_ids"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-audit", action="store_true")
    group.add_argument("--stop-check", action="store_true")
    group.add_argument("--matrix", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "dry-audit" if args.dry_audit else "stop-check" if args.stop_check else "matrix"
    print(f"=== M5F.2.1 communication opportunity ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

    audit = dry_scenario_audit()
    if args.dry_audit:
        base_out_dir = os.path.join(BASE_OUT_DIR, "dry-audit")
        out_dir = resolve_output_dir(base_out_dir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "scenarios.json"), "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, default=str)
        with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
            json.dump({"experiment_id": "m5f2_1-communication-opportunity-v1", "phase": phase_name, "finished_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        print(f"\nOutputs written under: {out_dir}/")
        print(f"=== M5F.2.1 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
        return 0

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

    if args.stop_check:
        # Spec section 26: exactly one trial per scenario, using that
        # scenario's own first-planned profile - never the full matrix.
        conditions = [(name, SCENARIO_PROFILES[name][0]) for name in SCENARIO_NAMES]
        n_trials = 1
    else:
        conditions = [(name, profile) for name in SCENARIO_NAMES for profile in SCENARIO_PROFILES[name]]
        n_trials = TRIALS_PER_CONDITION

    all_trials: list[dict] = []
    all_messages: list[dict] = []
    condition_metrics: dict[str, dict] = {}
    recipient_probes: list[dict] = []

    for scenario, profile in conditions:
        trials = _run_sender_condition(scenario, profile, n_trials, region)
        for t in trials:
            if t["final_action"] == "send_message" and t["valid"]:
                msg = _analyze_message(scenario, t)
                all_messages.append(msg)
                probe_client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                probe = run_recipient_probe(scenario, t["final_target"], t["final_message"], probe_client, t["trial_index"])
                probe["sender_profile"] = profile
                sender_label = "black"  # sender is always Black Spy in this harness
                probe.update(classify_recipient_influence(probe, sender_label))
                recipient_probes.append(probe)
                print(f"[RECIPIENT PROBE] {scenario}/{profile} trial {t['trial_index']}: with={probe['with_message_action']}/{probe['with_message_target']} without={probe['without_message_action']}/{probe['without_message_target']} evidence={probe['evidence_level']}")
        condition_key = f"{scenario}__{profile}"
        metrics = sender_condition_metrics(trials)
        condition_metrics[condition_key] = {"scenario": scenario, "profile": profile, **metrics}
        print(f"[CONDITION] {condition_key}: {metrics}")
        all_trials.extend({k: v for k, v in t.items() if k != "state"} for t in trials)

        # Spec section 27: CONTROL excessive-baseline stop rule.
        if scenario == "CONTROL" and metrics["selection_rate"] is not None and metrics["selection_rate"] >= CONTROL_EXCESSIVE_RATE:
            print(f"[STOP RULE] CONTROL/{profile} selection_rate={metrics['selection_rate']} >= {CONTROL_EXCESSIVE_RATE} - classifying communication policy as potentially over-incentivized.")
            summary["control_excessive_messaging_stop_rule_triggered"] = True

    total_sends = sum(m["send_message_selections"] for m in condition_metrics.values())
    summary["total_send_message_selections"] = total_sends
    summary["condition_metrics"] = condition_metrics

    if args.stop_check:
        summary["zero_adoption_stop_rule_triggered"] = total_sends == 0
        if total_sends == 0:
            print("\n[STOP RULE] 0 SEND_MESSAGE selections across all five scenario classes - per spec section 26, STOP before the full 100-trial matrix.")
        else:
            print(f"\n[STOP RULE] {total_sends} SEND_MESSAGE selection(s) observed across the 5 stop-check trials - proceeding to the full matrix is warranted.")

    with open(os.path.join(out_dir, "sender_trials.jsonl"), "w", encoding="utf-8") as f:
        for t in all_trials:
            f.write(json.dumps(t, default=str) + "\n")
    with open(os.path.join(out_dir, "messages.jsonl"), "w", encoding="utf-8") as f:
        for m in all_messages:
            f.write(json.dumps(m, default=str) + "\n")
    with open(os.path.join(out_dir, "recipient_probes.jsonl"), "w", encoding="utf-8") as f:
        for p in recipient_probes:
            f.write(json.dumps(p, default=str) + "\n")

    claim_analysis = {
        "consistent": sum(1 for m in all_messages if m["claim_consistency"] == "CONSISTENT_WITH_SENDER_OBSERVATION"),
        "unsupported": sum(1 for m in all_messages if m["claim_consistency"] == "UNSUPPORTED_BY_SENDER_OBSERVATION"),
        "contradicted": sum(1 for m in all_messages if m["claim_consistency"] == "CONTRADICTED_BY_SENDER_OBSERVATION"),
        "not_fact_checkable": sum(1 for m in all_messages if m["claim_consistency"] == "NOT_FACT_CHECKABLE"),
        "conflicts_with_sender_plan": sum(1 for m in all_messages if m["claim_conflicts_with_sender_plan"]),
    }
    with open(os.path.join(out_dir, "claim_analysis.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(claim_analysis, default=str) + "\n")
    summary["claim_analysis"] = claim_analysis
    summary["recipient_probes_summary"] = {
        "total_probes": len(recipient_probes),
        "by_evidence_level": {
            level: sum(1 for p in recipient_probes if p["evidence_level"] == level)
            for level in ("EXPLICIT_REFERENCE", "BEHAVIOR_CHANGED", "COORDINATION_FOLLOWUP", "AMBIGUOUS", "NO_OBSERVABLE_RESPONSE")
        },
    }

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
        json.dump({"experiment_id": "m5f2_1-communication-opportunity-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\n[SUMMARY] total_send_message_selections={total_sends}")
    print(f"Outputs written under: {out_dir}/")
    print(f"=== M5F.2.1 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
