"""M5F.2.3: automated live-diagnostic entry point for bounded free
communication (flat schema + cooldown).

    python -m src.evaluation.run_m5f2_3 --flat-pilot     # Phase B: flat schema, uncapped, technical-only
    python -m src.evaluation.run_m5f2_3 --bounded-check  # Phase C: bounded-mechanic determinism (network-free-ish, quick live confirm)
    python -m src.evaluation.run_m5f2_3 --calibration    # Phase D: 5 scenarios x 2 profiles x 1 bounded sequence
    python -m src.evaluation.run_m5f2_3 --duel-sanity    # Phase E: 2 mirrored communication-enabled duels

Phase A (network-free) is covered by
tests/test_m5f2_3_bounded_free_communication.py - not this driver. Freezes
provider=Bedrock, native_tool_like runtime, M5F.2.1's frozen scenarios/
profiles, M5F.2's frozen communication policy (spec sections 1, 21, 34).
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
from src.evaluation.decision_trace import build_progress_snapshot
from src.evaluation.duel_runner import run_duel_episode
from src.evaluation.duel_trace import annotate_opponent_visibility, build_actor_trace_records, merge_duel_trace, render_duel_markdown
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.opportunity_metrics import (
    claim_conflicts_with_sender_plan,
    classify_claim_consistency,
    classify_message_category,
    classify_recipient_influence,
)
from src.evaluation.scenario_harness import (
    SCENARIO_GROUND_TRUTH,
    SCENARIO_NAMES,
    SCENARIO_PROFILES,
    run_bounded_sequence,
    run_recipient_probe,
    run_sender_trial,
)
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f2_3-bounded-free-communication-v1")
BOUNDED_COOLDOWN = 3
SEQUENCE_LENGTH = 12  # >= 2 * (cooldown + 1) so at least 2 full availability cycles can be observed


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


def _technical_metrics(trials: list[dict]) -> dict:
    total = len(trials)
    valid = sum(1 for t in trials if t["valid"])
    first_pass = sum(1 for t in trials if t["first_pass_valid"])
    fallback = sum(1 for t in trials if t.get("action_source") == "fallback")
    return {
        "trials": total, "valid": valid, "valid_rate": round(valid / total, 3) if total else None,
        "first_pass_valid": first_pass, "first_pass_rate": round(first_pass / total, 3) if total else None,
        "fallback": fallback, "fallback_rate": round(fallback / total, 3) if total else None,
    }


# -- Phase B: flat schema pilot (spec sections 17-19) ---------------------------


def flat_schema_pilot(region: str) -> dict:
    print("\n=== PHASE B: live flat-schema technical pilot (uncapped, technical-only) ===")
    trials = []
    for scenario in SCENARIO_NAMES:
        profiles = SCENARIO_PROFILES[scenario][:2]
        for profile in profiles:
            for i in range(1, 4):
                client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                trial = run_sender_trial(scenario, profile, i, client, zero_decision_cost=True, flat_message_schema=True, message_cooldown_decisions=0)
                trials.append({k: v for k, v in trial.items() if k != "state"})
                print(f"[FLAT PILOT] {scenario}/{profile} trial {i}: valid={trial['valid']} first_pass={trial['first_pass_valid']} message_included={trial['message_included']}")
    metrics = _technical_metrics(trials)
    print(f"\n[FLAT PILOT METRICS] {metrics}")
    return {"trials": trials, "metrics": metrics}


# -- Phase C: bounded mechanic live confirmation (spec section 37) --------------


def bounded_mechanic_check(region: str) -> dict:
    print("\n=== PHASE C: bounded mechanic live confirmation (deterministic counter semantics) ===")
    scenario = "TACTICAL_MISDIRECTION"
    profile = SCENARIO_PROFILES[scenario][0]  # SABOTEUR - real precedent for organic messaging
    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
    decisions, state = run_bounded_sequence(scenario, profile, SEQUENCE_LENGTH, client, message_cooldown_decisions=BOUNDED_COOLDOWN, flat_message_schema=True)
    rows = []
    decisions_since_send = None
    for d in decisions:
        executed = d.get("executed_action") or {}
        sent = executed.get("accompanying_message") is not None
        available_shown = d.get("communication_available")
        next_in_shown = d.get("communication_next_available_in_decisions")
        rows.append({
            "decision_number": d.get("decision_number"), "sent": sent,
            "available_now_shown": available_shown, "next_available_in_decisions_shown": next_in_shown,
            "error_type": d.get("error_type"),
        })
        print(f"[BOUNDED] decision {d.get('decision_number')}: sent={sent} available_shown={available_shown} next_in={next_in_shown}")
        if sent:
            decisions_since_send = 0
        elif decisions_since_send is not None:
            decisions_since_send += 1
    return {"scenario": scenario, "profile": profile, "rows": rows}


# -- Phase D: controlled behavioral calibration (spec sections 20, 38) ----------


def _analyze_message(scenario: str, sender_reason: str | None, message_text: str | None, physical_action: str | None, physical_target: str | None) -> dict:
    ground_truth = SCENARIO_GROUND_TRUTH[scenario]
    return {
        "message_category": classify_message_category(message_text, scenario=scenario),
        "claim_consistency": classify_claim_consistency(message_text, ground_truth),
        "claim_conflicts_with_sender_plan": claim_conflicts_with_sender_plan(
            message_text, physical_action, physical_target, ground_truth["contested_location_ids"]
        ),
    }


def behavioral_calibration(out_dir: str, region: str) -> dict:
    print(f"\n=== PHASE D: controlled behavioral calibration (bounded, cooldown={BOUNDED_COOLDOWN}) ===")
    sequences = []
    all_messages = []
    all_probes = []
    for scenario in SCENARIO_NAMES:
        for profile in SCENARIO_PROFILES[scenario][:2]:
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            decisions, state = run_bounded_sequence(scenario, profile, SEQUENCE_LENGTH, client, message_cooldown_decisions=BOUNDED_COOLDOWN, flat_message_schema=True)
            eligible = 0
            sent = 0
            action_counts: dict[str, int] = {}
            seq_rows = []
            for d in decisions:
                executed = d.get("executed_action") or {}
                is_eligible = bool(d.get("communication_available"))
                eligible += int(is_eligible)
                physical_action = executed.get("action")
                if physical_action and physical_action != "send_message":
                    action_counts[physical_action] = action_counts.get(physical_action, 0) + 1
                accompanying = executed.get("accompanying_message")
                row = {
                    "decision_number": d.get("decision_number"), "valid": bool(d.get("final_model_action_valid")),
                    "eligible": is_eligible, "physical_action": physical_action, "physical_target": executed.get("target_id"),
                    "message_included": accompanying is not None,
                }
                if accompanying is not None:
                    sent += 1
                    reason = (d.get("raw_response") or {}).get("reason")
                    analysis = _analyze_message(scenario, reason, accompanying.get("text"), physical_action, executed.get("target_id"))
                    msg = {
                        "scenario": scenario, "profile": profile, "decision_number": d.get("decision_number"),
                        "recipient_actor_id": accompanying.get("recipient_id"), "message_text": accompanying.get("text"),
                        "sender_reason": reason, "sender_physical_action": physical_action, **analysis,
                    }
                    all_messages.append(msg)
                    probe_client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                    probe = run_recipient_probe(scenario, accompanying.get("recipient_id"), accompanying.get("text"), probe_client, d.get("decision_number"), cost_tag="bounded")
                    probe["sender_profile"] = profile
                    probe.update(classify_recipient_influence(probe, sender_label="black"))
                    all_probes.append(probe)
                    print(f"[RECIPIENT PROBE bounded] {scenario}/{profile} d{d.get('decision_number')}: evidence={probe['evidence_level']}")
                seq_rows.append(row)
            metrics = {
                "decisions": len(decisions), "eligible_decisions": eligible, "messages_sent": sent,
                "messages_per_10_decisions": round(sent / len(decisions) * 10, 3) if decisions else None,
                "eligibility_utilization": round(sent / eligible, 3) if eligible else None,
                "physical_action_distribution": action_counts,
            }
            print(f"[SEQUENCE] {scenario}/{profile}: {metrics}")
            sequences.append({"scenario": scenario, "profile": profile, "rows": seq_rows, "metrics": metrics})

    with open(os.path.join(out_dir, "bounded_sequences.jsonl"), "w", encoding="utf-8") as f:
        for s in sequences:
            f.write(json.dumps(s, default=str) + "\n")
    with open(os.path.join(out_dir, "messages.jsonl"), "w", encoding="utf-8") as f:
        for m in all_messages:
            f.write(json.dumps(m, default=str) + "\n")
    with open(os.path.join(out_dir, "recipient_probes.jsonl"), "w", encoding="utf-8") as f:
        for p in all_probes:
            f.write(json.dumps(p, default=str) + "\n")

    control_sequences = [s for s in sequences if s["scenario"] == "CONTROL"]
    control_rate = (
        sum(s["metrics"]["messages_sent"] for s in control_sequences) / sum(s["metrics"]["eligible_decisions"] for s in control_sequences)
        if control_sequences and sum(s["metrics"]["eligible_decisions"] for s in control_sequences) else None
    )
    return {"sequences": sequences, "messages": all_messages, "probes": all_probes, "control_eligibility_utilization": control_rate}


# -- Phase E: short real-duel sanity check (spec section 39) --------------------


def duel_sanity_check(out_dir: str, region: str) -> dict:
    print("\n=== PHASE E: short real-duel sanity check (2 mirrored communication-enabled duels) ===")
    results = []
    pairs = [
        {"seed": 950, "black": "saboteur", "white": "opportunist", "mirror": "A"},
        {"seed": 950, "black": "opportunist", "white": "saboteur", "mirror": "B"},
    ]
    for pair in pairs:
        episode_id = f"m5f2_3-duel-sanity-{pair['black']}_vs_{pair['white']}-mirror{pair['mirror']}"
        telemetry_path = os.path.join(out_dir, "duel_decisions.jsonl")
        black_snapshots: dict[int, dict] = {}
        white_snapshots: dict[int, dict] = {}
        last_black_n = last_white_n = 0

        def on_decision(black_n, white_n, state):
            nonlocal last_black_n, last_white_n
            if black_n == 0 and white_n == 0:
                black_snapshots[0] = build_progress_snapshot(state, constants.BLACK_SPY_ID)
                white_snapshots[0] = build_progress_snapshot(state, constants.WHITE_SPY_ID)
            if black_n > last_black_n:
                black_snapshots[black_n] = build_progress_snapshot(state, constants.BLACK_SPY_ID)
                last_black_n = black_n
            if white_n > last_white_n:
                white_snapshots[white_n] = build_progress_snapshot(state, constants.WHITE_SPY_ID)
                last_white_n = white_n

        print(f"[DUEL] seed={pair['seed']} black={pair['black']} white={pair['white']} mirror={pair['mirror']} starting {datetime.now(timezone.utc).isoformat()}")
        result = run_duel_episode(
            seed=pair["seed"], provider="bedrock", model=MODEL_ID, region=region, agent_runtime="native_tool_like",
            telemetry_path=telemetry_path, episode_id=episode_id, max_decisions_per_actor=40,
            max_episode_game_time=180.0, max_episode_wall_time=3600.0, llm_timeout=60.0,
            on_decision=on_decision, black_personality=pair["black"], white_personality=pair["white"],
            communication_enabled=True, zero_decision_cost_messaging=True, flat_message_schema=True,
            message_cooldown_decisions=BOUNDED_COOLDOWN,
        )
        print(f"[DUEL] status={result.status} stop_reason={result.stop_reason} black_decisions={result.black_decision_count} white_decisions={result.white_decision_count} black_fallbacks={result.black_fallback_count} white_fallbacks={result.white_fallback_count}")

        all_decisions = []
        if os.path.exists(telemetry_path):
            with open(telemetry_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if e.get("event_type") == "llm_decision" and e.get("episode_id") == episode_id:
                        all_decisions.append(e)
        all_lifecycle = []
        if os.path.exists(telemetry_path):
            with open(telemetry_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if e.get("event_type") == "action_lifecycle" and e.get("episode_id") == episode_id:
                        all_lifecycle.append(e)

        black_records = build_actor_trace_records(all_decisions, all_lifecycle, black_snapshots, episode_id, pair["seed"], "native_tool_like", constants.BLACK_SPY_ID)
        white_records = build_actor_trace_records(all_decisions, all_lifecycle, white_snapshots, episode_id, pair["seed"], "native_tool_like", constants.WHITE_SPY_ID)
        decisions_by_id = {d["decision_id"]: d for d in all_decisions}
        annotate_opponent_visibility(black_records, decisions_by_id)
        annotate_opponent_visibility(white_records, decisions_by_id)
        merged = merge_duel_trace(black_records, white_records)
        duel_markdown = render_duel_markdown(merged, episode_id, pair["seed"])
        with open(os.path.join(out_dir, "duel_trace.md"), "a", encoding="utf-8") as f:
            f.write(duel_markdown)
            f.write("\n---\n\n")

        messages_sent = sum(
            1 for d in all_decisions if (d.get("executed_action") or {}).get("accompanying_message") is not None
        )
        results.append({
            "episode_id": episode_id, "seed": pair["seed"], "black": pair["black"], "white": pair["white"], "mirror": pair["mirror"],
            "status": result.status, "stop_reason": result.stop_reason,
            "black_decisions": result.black_decision_count, "white_decisions": result.white_decision_count,
            "black_fallbacks": result.black_fallback_count, "white_fallbacks": result.white_fallback_count,
            "total_messages_sent": messages_sent,
        })
    return {"duels": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--flat-pilot", action="store_true")
    group.add_argument("--bounded-check", action="store_true")
    group.add_argument("--calibration", action="store_true")
    group.add_argument("--duel-sanity", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "flat-pilot" if args.flat_pilot else "bounded-check" if args.bounded_check else "calibration" if args.calibration else "duel-sanity"
    print(f"=== M5F.2.3 bounded free communication ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    if args.flat_pilot:
        result = flat_schema_pilot(region)
        summary["flat_pilot"] = result["metrics"]
        with open(os.path.join(out_dir, "flat_schema_trials.jsonl"), "w", encoding="utf-8") as f:
            for t in result["trials"]:
                f.write(json.dumps(t, default=str) + "\n")
    elif args.bounded_check:
        summary["bounded_check"] = bounded_mechanic_check(region)
    elif args.calibration:
        result = behavioral_calibration(out_dir, region)
        summary["sequence_metrics"] = [{"scenario": s["scenario"], "profile": s["profile"], **s["metrics"]} for s in result["sequences"]]
        summary["control_eligibility_utilization"] = result["control_eligibility_utilization"]
        summary["total_messages"] = len(result["messages"])
        summary["claim_analysis"] = {
            "consistent": sum(1 for m in result["messages"] if m["claim_consistency"] == "CONSISTENT_WITH_SENDER_OBSERVATION"),
            "unsupported": sum(1 for m in result["messages"] if m["claim_consistency"] == "UNSUPPORTED_BY_SENDER_OBSERVATION"),
            "contradicted": sum(1 for m in result["messages"] if m["claim_consistency"] == "CONTRADICTED_BY_SENDER_OBSERVATION"),
            "not_fact_checkable": sum(1 for m in result["messages"] if m["claim_consistency"] == "NOT_FACT_CHECKABLE"),
            "conflicts_with_sender_plan": sum(1 for m in result["messages"] if m["claim_conflicts_with_sender_plan"]),
        }
        summary["recipient_probes_summary"] = {
            "total_probes": len(result["probes"]),
            "by_evidence_level": {
                level: sum(1 for p in result["probes"] if p["evidence_level"] == level)
                for level in ("EXPLICIT_REFERENCE", "BEHAVIOR_CHANGED", "COORDINATION_FOLLOWUP", "AMBIGUOUS", "NO_OBSERVABLE_RESPONSE")
            },
        }
    else:
        summary["duel_sanity"] = duel_sanity_check(out_dir, region)

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
        json.dump({"experiment_id": "m5f2_3-bounded-free-communication-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.2.3 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
