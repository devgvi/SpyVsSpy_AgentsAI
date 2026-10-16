"""M5F.1: automated live-diagnostic entry point for personality-profile
behavioral differentiation.

    python -m src.evaluation.run_m5f1 --phase-c    # 12 synthetic decisions (3 x 4 profiles)
    python -m src.evaluation.run_m5f1 --phase-d    # 4 solo episodes vs scripted_active_v1
    python -m src.evaluation.run_m5f1 --stage1     # 8 mirrored LLM-vs-LLM duels
    python -m src.evaluation.run_m5f1 --stage2     # +8 more (seed 902) - only run if Stage 1 warrants it

Phase B (static context inspection) needs no live calls - already covered
by `tests/test_m5f1_personality_profiles.py`'s unit tests and printed
here for the record. Freezes provider=Bedrock, native_tool_like runtime,
same model/opponent/decision-cap/deadline as M5F (spec sections 34-36).
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
from src.evaluation.duel_trace import (
    annotate_opponent_visibility,
    build_actor_trace_records,
    contested_objective_events,
    merge_duel_trace,
    render_duel_markdown,
)
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.personality_metrics import aggregate_profile_metrics, annotate_behavior_tags, generate_mirror_pairs
from src.evaluation.spec import ExperimentSpec, VariantSpec
from src.evaluation.synthetic_stability import PATTERN_S1_CONSTANT, run_synthetic_pattern
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.personality import CAUTIOUS, OPPORTUNIST, RACER, SABOTEUR, render_personality_block

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f1-personality-profiles-v1")
AGENT_RUNTIME = "native_tool_like"
NAMED_PROFILES = [RACER, CAUTIOUS, SABOTEUR, OPPORTUNIST]
MATCHUPS = [(RACER, SABOTEUR), (CAUTIOUS, SABOTEUR), (RACER, OPPORTUNIST), (CAUTIOUS, OPPORTUNIST)]
STAGE1_SEEDS = [901]
# Stage 2 adds only the NEW seed (902) - Stage 1's 8 duels already cover
# 901 and are preserved on disk; re-running them here would be wasteful,
# redundant live API cost for zero new information (spec section 33's
# own staged-cost-control intent).
STAGE2_SEEDS = [902]
MAX_DECISIONS_PER_ACTOR = 40
MAX_EPISODE_GAME_TIME = 180.0


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


def phase_b_static_context_inspection() -> None:
    print("\n=== PHASE B: static context inspection (no live calls) ===")
    for profile in NAMED_PROFILES:
        block = render_personality_block(profile)
        others = [p.upper() for p in NAMED_PROFILES if p != profile]
        leaked = [o for o in others if o in block]
        print(f"[PHASE B] {profile}: block_len={len(block)} contains_only_own_priorities={not leaked}")
        assert not leaked, f"{profile}'s block unexpectedly contains another profile's name: {leaked}"


def phase_c_synthetic_smoke(region: str) -> dict:
    print("\n=== PHASE C: synthetic decision smoke (3 decisions x 4 profiles) ===")
    results = {}
    for profile in NAMED_PROFILES:
        client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
        result = run_synthetic_pattern(
            pattern=PATTERN_S1_CONSTANT, runtime=AGENT_RUNTIME, n_decisions=3, provider="bedrock", model=MODEL_ID,
            model_client=client, episode_id=f"m5f1-phase-c-{profile}", seed=1001, personality_profile=profile,
            per_decision_wall_timeout=180.0,
        )
        valid = sum(1 for d in result.decisions if d.get("final_model_action_valid"))
        print(f"[PHASE C] {profile}: {valid}/{len(result.decisions)} valid decisions")
        results[profile] = {"valid": valid, "total": len(result.decisions)}
    return results


def phase_d_solo_sanity(region: str) -> dict:
    print("\n=== PHASE D: solo behavioral sanity (1 episode/profile vs scripted_active_v1) ===")
    variants = tuple(
        VariantSpec(
            variant_id=f"solo_{profile}", agent_runtime=AGENT_RUNTIME, provider="bedrock", model=MODEL_ID,
            llm_timeout=60.0, llm_region=region, opponent_type="scripted_active", opponent_profile="OPPORTUNIST",
            personality_profile=profile,
        )
        for profile in NAMED_PROFILES
    )
    spec = ExperimentSpec(
        experiment_id="m5f1-phase-d-solo-sanity", variants=variants, seeds=(1002,), execution_mode="evaluation",
        max_episode_game_time=MAX_EPISODE_GAME_TIME, max_episode_wall_time=1800.0, max_decisions=15, max_runtime_failures=20,
    )
    runner = ExperimentRunner(episode_runner=EpisodeRunner(poll_sleep=0.0), output_root="evaluation_runs")
    result = runner.run(spec)
    summary = {}
    for ep in result.episodes:
        print(f"[PHASE D] {ep.variant_id}: status={ep.status.value} decisions={ep.decision_count} mission_success={ep.mission_success}")
        summary[ep.variant_id] = {"status": ep.status.value, "decisions": ep.decision_count, "mission_success": ep.mission_success}
    return summary


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


def _run_one_personality_duel(out_dir: str, pair: dict, region: str) -> dict:
    seed, black_p, white_p = pair["seed"], pair["black_personality"], pair["white_personality"]
    experiment_id = "m5f1-personality-profiles-v1"
    episode_id = f"{experiment_id}__{black_p}_vs_{white_p}__seed{seed}__mirror{pair['mirror']}"
    telemetry_path = os.path.join(out_dir, "decisions.jsonl")

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

    print(f"[DUEL] seed={seed} black={black_p} white={white_p} mirror={pair['mirror']} starting {datetime.now(timezone.utc).isoformat()}")
    result = run_duel_episode(
        seed=seed, provider="bedrock", model=MODEL_ID, region=region, agent_runtime=AGENT_RUNTIME,
        telemetry_path=telemetry_path, episode_id=episode_id, max_decisions_per_actor=MAX_DECISIONS_PER_ACTOR,
        max_episode_game_time=MAX_EPISODE_GAME_TIME, max_episode_wall_time=3600.0, llm_timeout=60.0,
        on_decision=on_decision, black_personality=black_p, white_personality=white_p,
    )
    print(f"[DUEL] status={result.status} stop_reason={result.stop_reason} black_decisions={result.black_decision_count} white_decisions={result.white_decision_count}")

    all_decisions = _load_events(telemetry_path, episode_id, "llm_decision")
    all_lifecycle = _load_events(telemetry_path, episode_id, "action_lifecycle")
    decisions_by_id = {d["decision_id"]: d for d in all_decisions}

    black_records = build_actor_trace_records(all_decisions, all_lifecycle, black_snapshots, episode_id, seed, AGENT_RUNTIME, constants.BLACK_SPY_ID)
    white_records = build_actor_trace_records(all_decisions, all_lifecycle, white_snapshots, episode_id, seed, AGENT_RUNTIME, constants.WHITE_SPY_ID)
    annotate_opponent_visibility(black_records, decisions_by_id)
    annotate_opponent_visibility(white_records, decisions_by_id)

    contested = contested_objective_events(black_records, white_records)
    contested_targets = frozenset(e["target"] for e in contested)
    annotate_behavior_tags(black_records, contested_targets)
    annotate_behavior_tags(white_records, contested_targets)

    merged = merge_duel_trace(black_records, white_records)
    duel_markdown = render_duel_markdown(merged, episode_id, seed)

    return {
        "seed": seed, "episode_id": episode_id, "matchup": pair["matchup"], "mirror": pair["mirror"],
        "black_personality": black_p, "white_personality": white_p,
        "status": result.status, "stop_reason": result.stop_reason,
        "black_decisions": result.black_decision_count, "white_decisions": result.white_decision_count,
        "black_fallbacks": result.black_fallback_count, "white_fallbacks": result.white_fallback_count,
        "black_mission_success": result.black_game_facts.get("mission_success"),
        "white_mission_success": result.white_game_facts.get("mission_success"),
        "contested_objective_events": len(contested),
        "black_records": black_records, "white_records": white_records, "duel_markdown": duel_markdown,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase-c", action="store_true")
    group.add_argument("--phase-d", action="store_true")
    group.add_argument("--stage1", action="store_true")
    group.add_argument("--stage2", action="store_true")
    args = parser.parse_args(argv)

    phase_name = "phase-c" if args.phase_c else "phase-d" if args.phase_d else "stage1" if args.stage1 else "stage2"
    print(f"=== M5F.1 personality profiles ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

    phase_b_static_context_inspection()

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

    if args.phase_c:
        summary["phase_c"] = phase_c_synthetic_smoke(region)
    elif args.phase_d:
        summary["phase_d"] = phase_d_solo_sanity(region)
    else:
        seeds = STAGE1_SEEDS if args.stage1 else STAGE2_SEEDS
        pairs = generate_mirror_pairs(MATCHUPS, seeds)
        duel_results = [_run_one_personality_duel(out_dir, pair, region) for pair in pairs]

        with open(os.path.join(out_dir, "duel_trace.md"), "a", encoding="utf-8") as f:
            for d in duel_results:
                f.write(d["duel_markdown"]); f.write("\n---\n\n")
        with open(os.path.join(out_dir, "duel_trace.jsonl"), "a", encoding="utf-8") as f:
            for d in duel_results:
                for r in d["black_records"] + d["white_records"]:
                    f.write(json.dumps(r, default=str) + "\n")

        records_by_profile: dict[str, list[dict]] = {}
        for d in duel_results:
            records_by_profile.setdefault(d["black_personality"], []).extend(d["black_records"])
            records_by_profile.setdefault(d["white_personality"], []).extend(d["white_records"])
        profile_metrics = aggregate_profile_metrics(records_by_profile)
        with open(os.path.join(out_dir, "profile_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(profile_metrics, f, indent=2, default=str)

        summary["duels"] = [{k: v for k, v in d.items() if k not in ("black_records", "white_records", "duel_markdown")} for d in duel_results]
        summary["profile_metrics"] = profile_metrics

        for profile, metrics in profile_metrics.items():
            print(f"\n[PROFILE {profile}] {metrics}")

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
        json.dump({"experiment_id": "m5f1-personality-profiles-v1", "phase": phase_name, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.1 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
