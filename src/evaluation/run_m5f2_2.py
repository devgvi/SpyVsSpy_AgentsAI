"""M5F.2.2: automated live-diagnostic entry point for communication cost
isolation (FULL_COST vs ZERO_DECISION_COST).

    python -m src.evaluation.run_m5f2_2 --static-audit   # Phase B: parity check, no live calls
    python -m src.evaluation.run_m5f2_2 --smoke          # Phase C: one real message+action, ZERO_DECISION_COST
    python -m src.evaluation.run_m5f2_2 --stop-check     # Phase D: 5 scenarios x first profile x 2 cost modes
    python -m src.evaluation.run_m5f2_2 --matrix         # Phase E: full 10 conditions x 10 trials x 2 cost modes (only if warranted)

Phase A (network-free) is covered by tests/test_m5f2_2_cost_isolation.py -
not this driver. Freezes provider=Bedrock, native_tool_like runtime, same
model, M5F.2.1's frozen scenarios/profiles, M5F.2's frozen communication
policy wording (spec sections 1, 23, 33).
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
    message_adoption_delta,
    physical_action_displacement,
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
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f2_2-communication-cost-isolation-v1")
TRIALS_PER_CONDITION = 10
CONTROL_EXCESSIVE_RATE = 0.80


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


# -- Phase B: static parity audit (spec section 37) -----------------------------


def static_parity_audit() -> dict:
    print("\n=== PHASE B: static parity audit (no live calls) ===")
    audit = {}
    for name in SCENARIO_NAMES:
        full_state = SCENARIO_BUILDERS[name]()
        zero_state = SCENARIO_BUILDERS[name]()
        zero_state.zero_decision_cost_messaging = True

        full_obs = build_observation(full_state, constants.BLACK_SPY_ID)
        zero_obs = build_observation(zero_state, constants.BLACK_SPY_ID)
        full_payload = serialize_observation(full_obs)
        zero_payload = serialize_observation(zero_obs)

        identical_position = full_payload["self"]["position"] == zero_payload["self"]["position"]
        identical_room = full_payload["room"] == zero_payload["room"]
        identical_mission = full_payload["mission"] == zero_payload["mission"]
        full_actions = {a["action"] for a in enumerate_legal_actions(full_obs)}
        zero_actions = {a["action"] for a in enumerate_legal_actions(zero_obs)}
        only_send_message_differs = (full_actions - {"send_message"}) == zero_actions and "send_message" in full_actions and "send_message" not in zero_actions

        ok = identical_position and identical_room and identical_mission and only_send_message_differs
        entry = {
            "identical_position": identical_position, "identical_room": identical_room,
            "identical_mission": identical_mission, "only_send_message_action_differs": only_send_message_differs,
            "full_cost_actions": sorted(full_actions), "zero_cost_actions": sorted(zero_actions),
            "parity_ok": ok,
        }
        print(f"[PARITY] {name}: {entry}")
        audit[name] = entry
        assert ok, f"{name}: FULL_COST/ZERO_DECISION_COST fixtures are not otherwise identical"
    return audit


# -- Phase C: live mechanism smoke (spec section 38) -----------------------------


def live_mechanism_smoke(out_dir: str, region: str) -> dict:
    print("\n=== PHASE C: live mechanism smoke (ZERO_DECISION_COST, real Bedrock) ===")
    profile = SCENARIO_PROFILES["TACTICAL_MISDIRECTION"][0]  # SABOTEUR - real M5F.2 precedent for spontaneous use
    attempts = []
    found = None
    for i in range(1, 6):
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        trial = run_sender_trial("TACTICAL_MISDIRECTION", profile, i, client, zero_decision_cost=True)
        attempts.append({k: v for k, v in trial.items() if k != "state"})
        print(f"[SMOKE] attempt {i}: action={trial['final_action']} message_included={trial['message_included']} valid={trial['valid']}")
        if trial["valid"] and trial["message_included"]:
            found = trial
            break
    result = {
        "attempts": attempts,
        "found_message_plus_action": found is not None,
        "action_with_message": found.get("final_action") if found else None,
        "message_text": found.get("accompanying_message_text") if found else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "phase_c_smoke.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# -- shared trial/analysis machinery ---------------------------------------------


def _analyze_message(scenario: str, trial: dict, cost_mode: str) -> dict:
    text = trial.get("final_message") or trial.get("accompanying_message_text")
    ground_truth = SCENARIO_GROUND_TRUTH[scenario]
    recipient_actor_id = trial.get("final_target") if cost_mode == "FULL_COST" else trial.get("accompanying_message_recipient")
    return {
        "scenario": scenario, "profile": trial["profile"], "trial_index": trial["trial_index"], "cost_mode": cost_mode,
        "episode_id": trial["episode_id"], "recipient_actor_id": recipient_actor_id,
        "message_text": text, "sender_reason": trial.get("reason"),
        "sender_physical_action": trial.get("final_action"),
        "message_category": classify_message_category(text, scenario=scenario),
        "claim_consistency": classify_claim_consistency(text, ground_truth),
        "claim_conflicts_with_sender_plan": claim_conflicts_with_sender_plan(
            text, trial.get("final_action"), trial.get("final_target"), ground_truth["contested_location_ids"]
        ),
    }


def _run_condition(scenario: str, profile: str, n_trials: int, cost_mode: str, region: str, cost_tag: str) -> tuple[list[dict], list[dict], list[dict]]:
    zero_decision_cost = cost_mode == "ZERO_DECISION_COST"
    trials = []
    messages = []
    probes = []
    for i in range(1, n_trials + 1):
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        trial = run_sender_trial(scenario, profile, i, client, zero_decision_cost=zero_decision_cost)
        print(f"[SENDER {cost_mode}] {scenario}/{profile} trial {i}: action={trial['final_action']} message_included={trial['message_included']} valid={trial['valid']}")
        if trial["valid"] and trial["message_included"]:
            msg = _analyze_message(scenario, trial, cost_mode)
            messages.append(msg)
            recipient_target = trial.get("final_target") if cost_mode == "FULL_COST" else trial.get("accompanying_message_recipient")
            message_text = trial.get("final_message") if cost_mode == "FULL_COST" else trial.get("accompanying_message_text")
            probe_client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            probe = run_recipient_probe(scenario, recipient_target, message_text, probe_client, i, cost_tag=cost_tag)
            probe["sender_profile"] = profile
            probe["cost_mode"] = cost_mode
            probe.update(classify_recipient_influence(probe, sender_label="black"))
            probes.append(probe)
            print(f"[RECIPIENT PROBE {cost_mode}] {scenario}/{profile} trial {i}: evidence={probe['evidence_level']}")
        trials.append({k: v for k, v in trial.items() if k != "state"})
    return trials, messages, probes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--static-audit", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--stop-check", action="store_true")
    group.add_argument("--matrix", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "static-audit" if args.static_audit else "smoke" if args.smoke else "stop-check" if args.stop_check else "matrix"
    print(f"=== M5F.2.2 communication cost isolation ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

    audit = static_parity_audit()

    if args.static_audit:
        base_out_dir = os.path.join(BASE_OUT_DIR, "static-audit")
        out_dir = resolve_output_dir(base_out_dir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "scenarios.json"), "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, default=str)
        with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
            json.dump({"experiment_id": "m5f2_2-communication-cost-isolation-v1", "phase": phase_name, "finished_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        print(f"\nOutputs written under: {out_dir}/")
        print(f"=== M5F.2.2 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
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

    if args.smoke:
        summary["phase_c"] = live_mechanism_smoke(out_dir, region)
        print(f"\n[SUMMARY] found_message_plus_action={summary['phase_c']['found_message_plus_action']}")
    else:
        if args.stop_check:
            conditions = [(name, SCENARIO_PROFILES[name][0]) for name in SCENARIO_NAMES]
            n_trials = 1
        else:
            conditions = [(name, profile) for name in SCENARIO_NAMES for profile in SCENARIO_PROFILES[name]]
            n_trials = TRIALS_PER_CONDITION

        all_trials: list[dict] = []
        all_messages: list[dict] = []
        all_probes: list[dict] = []
        condition_metrics: dict[str, dict] = {}
        cost_deltas: dict[str, dict] = {}
        displacement: dict[str, dict] = {}

        for scenario, profile in conditions:
            full_trials, full_messages, full_probes = _run_condition(scenario, profile, n_trials, "FULL_COST", region, "full")
            zero_trials, zero_messages, zero_probes = _run_condition(scenario, profile, n_trials, "ZERO_DECISION_COST", region, "zero")

            all_trials.extend(full_trials + zero_trials)
            all_messages.extend(full_messages + zero_messages)
            all_probes.extend(full_probes + zero_probes)

            full_metrics = sender_condition_metrics(full_trials)
            zero_metrics = sender_condition_metrics(zero_trials)
            key = f"{scenario}__{profile}"
            condition_metrics[key] = {"scenario": scenario, "profile": profile, "FULL_COST": full_metrics, "ZERO_DECISION_COST": zero_metrics}
            cost_deltas[key] = message_adoption_delta(full_metrics, zero_metrics)
            displacement[key] = physical_action_displacement(full_trials, zero_trials)
            print(f"[CONDITION] {key}: delta={cost_deltas[key]} displacement={displacement[key]}")

            if scenario == "CONTROL" and zero_metrics["selection_rate"] is not None and zero_metrics["selection_rate"] >= CONTROL_EXCESSIVE_RATE:
                print(f"[STOP RULE] CONTROL/{profile} ZERO_DECISION_COST selection_rate={zero_metrics['selection_rate']} >= {CONTROL_EXCESSIVE_RATE} - over-adoption.")
                summary["control_over_adoption_stop_rule_triggered"] = True

        total_zero_sends = sum(m["ZERO_DECISION_COST"]["send_message_selections"] for m in condition_metrics.values())
        total_full_sends = sum(m["FULL_COST"]["send_message_selections"] for m in condition_metrics.values())
        summary["total_full_cost_selections"] = total_full_sends
        summary["total_zero_cost_selections"] = total_zero_sends
        summary["condition_metrics"] = condition_metrics
        summary["cost_deltas"] = cost_deltas
        summary["physical_action_displacement"] = displacement

        if args.stop_check:
            summary["no_cost_effect_stop_rule_triggered"] = total_zero_sends == 0
            if total_zero_sends == 0:
                print("\n[STOP RULE] 0 ZERO_DECISION_COST message selections across all five scenarios - per spec section 30, decision cost alone does not explain non-adoption.")
            else:
                print(f"\n[STOP RULE] {total_zero_sends} ZERO_DECISION_COST selection(s) observed - continuing full profile coverage may be warranted.")

        with open(os.path.join(out_dir, "sender_trials.jsonl"), "w", encoding="utf-8") as f:
            for t in all_trials:
                f.write(json.dumps(t, default=str) + "\n")
        with open(os.path.join(out_dir, "messages.jsonl"), "w", encoding="utf-8") as f:
            for m in all_messages:
                f.write(json.dumps(m, default=str) + "\n")
        with open(os.path.join(out_dir, "recipient_probes.jsonl"), "w", encoding="utf-8") as f:
            for p in all_probes:
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
            "total_probes": len(all_probes),
            "by_evidence_level": {
                level: sum(1 for p in all_probes if p["evidence_level"] == level)
                for level in ("EXPLICIT_REFERENCE", "BEHAVIOR_CHANGED", "COORDINATION_FOLLOWUP", "AMBIGUOUS", "NO_OBSERVABLE_RESPONSE")
            },
        }
        with open(os.path.join(out_dir, "cost_comparison.json"), "w", encoding="utf-8") as f:
            json.dump({"cost_deltas": cost_deltas, "physical_action_displacement": displacement}, f, indent=2, default=str)

        print(f"\n[SUMMARY] total_full_cost_selections={total_full_sends} total_zero_cost_selections={total_zero_sends}")

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
        json.dump({"experiment_id": "m5f2_2-communication-cost-isolation-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.2.2 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
