"""M5F.2: automated live-diagnostic entry point for the communication
baseline (SEND_MESSAGE).

    python -m src.evaluation.run_m5f2 --phase-b    # real Bedrock smoke: >=3 successful SEND_MESSAGE actions
    python -m src.evaluation.run_m5f2 --phase-c    # one full communication-enabled pilot duel
    python -m src.evaluation.run_m5f2 --stage1     # 3 matchups x 2 mirrors x OFF/ON = 12 paired duels (one seed)
    python -m src.evaluation.run_m5f2 --stage2     # +12 more (a second seed) - only run if Stage 1 warrants it

Phase A (network-free) is covered by tests/test_m5f2_send_message.py and
tests/test_m5f2_memory_fairness_injection.py - not this driver. Freezes
provider=Bedrock, native_tool_like runtime, same model/decision-cap/
deadline as M5F/M5F.1 (spec sections 11-12).
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
from src.evaluation.communication_metrics import (
    build_message_records,
    communication_frequency_metrics,
    cooperation_events,
    message_cost_metrics,
    render_communication_narrative,
    response_metrics,
)
from src.evaluation.decision_trace import build_progress_snapshot
from src.evaluation.duel_runner import run_duel_episode
from src.evaluation.duel_trace import (
    annotate_opponent_visibility,
    build_actor_trace_records,
    contested_objective_events,
    merge_duel_trace,
    render_duel_markdown,
)
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.personality_metrics import annotate_behavior_tags
from src.evaluation.synthetic_stability import PATTERN_S3_ACTIVE_LIKE, run_synthetic_pattern
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.personality import CAUTIOUS, OPPORTUNIST, RACER, SABOTEUR

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f2-communication-negotiation-baseline-v1")
AGENT_RUNTIME = "native_tool_like"
MAX_DECISIONS_PER_ACTOR = 40
MAX_EPISODE_GAME_TIME = 180.0

# Spec section 14's recommended focused matchups - reused unchanged from
# M5F.1's own profile vocabulary (M5F.2 freezes personality definitions,
# spec section 10/17).
MATCHUPS = [(RACER, OPPORTUNIST), (CAUTIOUS, OPPORTUNIST), (RACER, SABOTEUR)]
STAGE1_SEED = 903
# Only used if Stage 1 warrants confirmation (spec section 15) - a second,
# independent seed, never a re-run of Stage 1's own seed.
STAGE2_SEED = 904


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


def _generate_mirror_communication_pairs(matchups, seed: int) -> list[dict]:
    """Spec section 15: OFF/ON share the same seed within each mirrored
    matchup - built directly (not via `generate_mirror_pairs`, which has
    no communication axis) so every pair below carries an explicit
    `communication` tag alongside the existing seed/personality/mirror
    fields."""
    pairs = []
    for a, b in matchups:
        for mirror, (black_p, white_p) in (("A", (a, b)), ("B", (b, a))):
            for communication in ("OFF", "ON"):
                pairs.append({
                    "seed": seed, "matchup": f"{a}_vs_{b}", "mirror": mirror,
                    "black_personality": black_p, "white_personality": white_p,
                    "communication": communication,
                })
    return pairs


# -- Phase B: real Bedrock smoke (spec section 43) -----------------------------


_PHASE_B_INCOMING_MESSAGE = "Are you close to the ambassador's office yet? Let me know if it's safe."


def phase_b_bedrock_smoke(out_dir: str, region: str) -> dict:
    """Spec section 43: at least 3 successful SEND_MESSAGE actions total.
    Runs the S3 (opponent-visible) synthetic pattern first, purely
    organically; if that alone doesn't reach the bar, a second batch adds
    one real seeded incoming message (spec section 61's "communication
    affordance" inspection, not a policy-wording change - the neutral
    guidance text itself is never touched) to give the model a concrete,
    realistic reason to reply."""
    print("\n=== PHASE B: real Bedrock communication smoke (target: >=3 successful SEND_MESSAGE actions) ===")
    attempts = []
    total_send_message = 0
    attempt = 0
    while total_send_message < 3 and attempt < 3:
        attempt += 1
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        episode_id = f"m5f2-phase-b-attempt{attempt}"
        result = run_synthetic_pattern(
            pattern=PATTERN_S3_ACTIVE_LIKE, runtime=AGENT_RUNTIME, n_decisions=8, provider="bedrock", model=MODEL_ID,
            model_client=client, episode_id=episode_id, seed=2000 + attempt, actor_id=constants.BLACK_SPY_ID,
            communication_enabled=True, per_decision_wall_timeout=180.0,
        )
        sent = [
            d for d in result.decisions
            if (d.get("executed_action") or {}).get("action") == "send_message" and d.get("final_model_action_valid")
        ]
        total_send_message += len(sent)
        canonical_ok = all(
            isinstance((d.get("executed_action") or {}).get("message"), str) for d in sent
        )
        attempts.append({
            "attempt": attempt, "episode_id": episode_id, "decisions": len(result.decisions),
            "send_message_count": len(sent), "canonical_output_ok": canonical_ok,
            "technical_failures": sum(1 for d in result.decisions if not d.get("final_model_action_valid") and d.get("action_source") == "fallback"),
        })
        print(f"[PHASE B] attempt {attempt}: {len(sent)}/{len(result.decisions)} send_message decisions (running total {total_send_message})")

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"phase_b_decisions_attempt{attempt}.jsonl"), "w", encoding="utf-8") as f:
            for d in result.decisions:
                f.write(json.dumps(d, default=str) + "\n")

    reply_attempts = []
    reply_attempt = 0
    while total_send_message < 3 and reply_attempt < 3:
        reply_attempt += 1
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        episode_id = f"m5f2-phase-b-reply-attempt{reply_attempt}"
        result = run_synthetic_pattern(
            pattern=PATTERN_S3_ACTIVE_LIKE, runtime=AGENT_RUNTIME, n_decisions=4, provider="bedrock", model=MODEL_ID,
            model_client=client, episode_id=episode_id, seed=2100 + reply_attempt, actor_id=constants.BLACK_SPY_ID,
            communication_enabled=True, per_decision_wall_timeout=180.0, seed_incoming_message=_PHASE_B_INCOMING_MESSAGE,
        )
        sent = [
            d for d in result.decisions
            if (d.get("executed_action") or {}).get("action") == "send_message" and d.get("final_model_action_valid")
        ]
        total_send_message += len(sent)
        reply_attempts.append({
            "attempt": reply_attempt, "episode_id": episode_id, "decisions": len(result.decisions),
            "send_message_count": len(sent),
        })
        print(f"[PHASE B] reply-elicitation attempt {reply_attempt}: {len(sent)}/{len(result.decisions)} send_message decisions (running total {total_send_message})")
        with open(os.path.join(out_dir, f"phase_b_reply_decisions_attempt{reply_attempt}.jsonl"), "w", encoding="utf-8") as f:
            for d in result.decisions:
                f.write(json.dumps(d, default=str) + "\n")

    return {
        "organic_attempts": attempts, "reply_elicitation_attempts": reply_attempts,
        "total_send_message_actions": total_send_message,
        "requirement_met": total_send_message >= 3,
    }


# -- Phase C: pilot duel (spec section 44) -------------------------------------


def phase_c_pilot_duel(out_dir: str, region: str) -> dict:
    print("\n=== PHASE C: communication-enabled pilot duel ===")
    return _run_one_communication_duel(
        out_dir, {"seed": 902, "matchup": "pilot", "mirror": "-", "black_personality": None, "white_personality": None, "communication": "ON"}, region,
    )


# -- shared duel runner + analysis ---------------------------------------------


def _load_events(telemetry_path: str, episode_id: str, event_type: str) -> list[dict]:
    events = []
    if not os.path.exists(telemetry_path):
        return events
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("event_type") == event_type and e.get("episode_id") == episode_id:
                events.append(e)
    return events


def _run_one_communication_duel(out_dir: str, pair: dict, region: str) -> dict:
    seed, black_p, white_p, comm = pair["seed"], pair["black_personality"], pair["white_personality"], pair["communication"]
    experiment_id = "m5f2-communication-negotiation-baseline-v1"
    episode_id = f"{experiment_id}__{black_p}_vs_{white_p}__seed{seed}__mirror{pair['mirror']}__{comm}"
    telemetry_path = os.path.join(out_dir, "decisions.jsonl")
    communication_enabled = comm == "ON"

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

    print(f"[DUEL] seed={seed} black={black_p} white={white_p} mirror={pair['mirror']} communication={comm} starting {datetime.now(timezone.utc).isoformat()}")
    result = run_duel_episode(
        seed=seed, provider="bedrock", model=MODEL_ID, region=region, agent_runtime=AGENT_RUNTIME,
        telemetry_path=telemetry_path, episode_id=episode_id, max_decisions_per_actor=MAX_DECISIONS_PER_ACTOR,
        max_episode_game_time=MAX_EPISODE_GAME_TIME, max_episode_wall_time=3600.0, llm_timeout=60.0,
        on_decision=on_decision, black_personality=black_p, white_personality=white_p,
        communication_enabled=communication_enabled,
    )
    print(f"[DUEL] status={result.status} stop_reason={result.stop_reason} black_decisions={result.black_decision_count} white_decisions={result.white_decision_count}")

    all_decisions = _load_events(telemetry_path, episode_id, "llm_decision")
    all_lifecycle = _load_events(telemetry_path, episode_id, "action_lifecycle")

    black_records = build_actor_trace_records(all_decisions, all_lifecycle, black_snapshots, episode_id, seed, AGENT_RUNTIME, constants.BLACK_SPY_ID)
    white_records = build_actor_trace_records(all_decisions, all_lifecycle, white_snapshots, episode_id, seed, AGENT_RUNTIME, constants.WHITE_SPY_ID)
    decisions_by_id = {d["decision_id"]: d for d in all_decisions}
    annotate_opponent_visibility(black_records, decisions_by_id)
    annotate_opponent_visibility(white_records, decisions_by_id)

    contested = contested_objective_events(black_records, white_records)
    contested_targets = frozenset(e["target"] for e in contested)
    annotate_behavior_tags(black_records, contested_targets)
    annotate_behavior_tags(white_records, contested_targets)

    message_records = build_message_records(black_records, white_records, black_p, white_p)
    narrative = render_communication_narrative(message_records)

    merged = merge_duel_trace(black_records, white_records)
    duel_markdown = render_duel_markdown(merged, episode_id, seed)

    return {
        "seed": seed, "episode_id": episode_id, "matchup": pair["matchup"], "mirror": pair["mirror"], "communication": comm,
        "black_personality": black_p, "white_personality": white_p,
        "status": result.status, "stop_reason": result.stop_reason,
        "black_decisions": result.black_decision_count, "white_decisions": result.white_decision_count,
        "black_fallbacks": result.black_fallback_count, "white_fallbacks": result.white_fallback_count,
        "black_mission_success": result.black_game_facts.get("mission_success"),
        "white_mission_success": result.white_game_facts.get("mission_success"),
        "contested_objective_events": len(contested),
        "black_repeat_rate": round(sum(1 for r in black_records if r["repeat_count"] > 0) / len(black_records), 3) if black_records else None,
        "white_repeat_rate": round(sum(1 for r in white_records if r["repeat_count"] > 0) / len(white_records), 3) if white_records else None,
        "black_no_progress_rate": round(sum(1 for r in black_records if r["progress_delta"] == ["NO_PROGRESS"]) / len(black_records), 3) if black_records else None,
        "white_no_progress_rate": round(sum(1 for r in white_records if r["progress_delta"] == ["NO_PROGRESS"]) / len(white_records), 3) if white_records else None,
        "black_communication_frequency": communication_frequency_metrics(black_records, message_records, constants.BLACK_SPY_ID),
        "white_communication_frequency": communication_frequency_metrics(white_records, message_records, constants.WHITE_SPY_ID),
        "black_response_metrics": response_metrics(message_records, constants.BLACK_SPY_ID),
        "white_response_metrics": response_metrics(message_records, constants.WHITE_SPY_ID),
        "black_message_cost": message_cost_metrics(black_records, constants.BLACK_SPY_ID),
        "white_message_cost": message_cost_metrics(white_records, constants.WHITE_SPY_ID),
        "cooperation_events": cooperation_events(message_records),
        "message_records": message_records,
        "black_records": black_records, "white_records": white_records,
        "duel_markdown": duel_markdown, "communication_narrative": narrative,
    }


def _off_vs_on_summary(duel_results: list[dict]) -> dict:
    """Spec section 57: descriptive OFF vs ON comparison - no single
    overall score, every number a plain count/average of already-computed
    per-duel facts."""
    by_condition: dict[str, list[dict]] = {"OFF": [], "ON": []}
    for d in duel_results:
        by_condition[d["communication"]].append(d)

    summary = {}
    for cond, duels in by_condition.items():
        if not duels:
            summary[cond] = None
            continue
        total_messages = sum(len(d["message_records"]) for d in duels)
        total_contested = sum(d["contested_objective_events"] for d in duels)
        completed = sum(1 for d in duels if d["status"] in ("COMPLETED", "DECISION_LIMIT_REACHED"))
        summary[cond] = {
            "duels": len(duels),
            "total_messages_sent": total_messages,
            "avg_messages_per_duel": round(total_messages / len(duels), 2),
            "total_contested_objective_events": total_contested,
            "avg_contested_objective_events_per_duel": round(total_contested / len(duels), 2),
            "avg_black_repeat_rate": round(sum(d["black_repeat_rate"] or 0 for d in duels) / len(duels), 3),
            "avg_white_repeat_rate": round(sum(d["white_repeat_rate"] or 0 for d in duels) / len(duels), 3),
            "avg_black_no_progress_rate": round(sum(d["black_no_progress_rate"] or 0 for d in duels) / len(duels), 3),
            "avg_white_no_progress_rate": round(sum(d["white_no_progress_rate"] or 0 for d in duels) / len(duels), 3),
            "mission_success_count": sum(1 for d in duels if d["black_mission_success"] or d["white_mission_success"]),
            "completed_or_capped_count": completed,
            "total_fallbacks": sum(d["black_fallbacks"] + d["white_fallbacks"] for d in duels),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase-b", action="store_true")
    group.add_argument("--phase-c", action="store_true")
    group.add_argument("--stage1", action="store_true")
    group.add_argument("--stage2", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "phase-b" if args.phase_b else "phase-c" if args.phase_c else "stage1" if args.stage1 else "stage2"
    print(f"=== M5F.2 communication baseline ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    if args.phase_b:
        summary["phase_b"] = phase_b_bedrock_smoke(out_dir, region)
    elif args.phase_c:
        d = phase_c_pilot_duel(out_dir, region)
        with open(os.path.join(out_dir, "duel_trace.md"), "w", encoding="utf-8") as f:
            f.write(d["duel_markdown"])
        with open(os.path.join(out_dir, "communication_narrative.md"), "w", encoding="utf-8") as f:
            f.write(d["communication_narrative"] or "(no messages sent this duel)\n")
        with open(os.path.join(out_dir, "messages.jsonl"), "w", encoding="utf-8") as f:
            for m in d["message_records"]:
                f.write(json.dumps(m, default=str) + "\n")
        summary["phase_c"] = {k: v for k, v in d.items() if k not in ("black_records", "white_records", "duel_markdown", "communication_narrative")}
        print(f"[PHASE C] status={d['status']} messages_sent={len(d['message_records'])}")
    else:
        seed = STAGE1_SEED if args.stage1 else STAGE2_SEED
        pairs = _generate_mirror_communication_pairs(MATCHUPS, seed)
        duel_results = [_run_one_communication_duel(out_dir, pair, region) for pair in pairs]

        with open(os.path.join(out_dir, "duel_trace.md"), "a", encoding="utf-8") as f:
            for d in duel_results:
                f.write(d["duel_markdown"]); f.write("\n---\n\n")
        with open(os.path.join(out_dir, "duel_trace.jsonl"), "a", encoding="utf-8") as f:
            for d in duel_results:
                for r in d["black_records"] + d["white_records"]:
                    f.write(json.dumps(r, default=str) + "\n")
        with open(os.path.join(out_dir, "messages.jsonl"), "a", encoding="utf-8") as f:
            for d in duel_results:
                for m in d["message_records"]:
                    f.write(json.dumps({**m, "episode_id": d["episode_id"]}, default=str) + "\n")
        with open(os.path.join(out_dir, "communication_narrative.md"), "a", encoding="utf-8") as f:
            for d in duel_results:
                f.write(f"## {d['episode_id']}\n\n")
                f.write(d["communication_narrative"] or "(no messages sent this duel)\n")
                f.write("\n---\n\n")

        off_vs_on = _off_vs_on_summary(duel_results)
        with open(os.path.join(out_dir, "communication_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(off_vs_on, f, indent=2, default=str)

        summary["duels"] = [
            {k: v for k, v in d.items() if k not in ("black_records", "white_records", "duel_markdown", "communication_narrative", "message_records")}
            for d in duel_results
        ]
        summary["off_vs_on"] = off_vs_on
        print(f"\n[OFF vs ON] {json.dumps(off_vs_on, indent=2, default=str)}")

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
        json.dump({"experiment_id": "m5f2-communication-negotiation-baseline-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.2 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
