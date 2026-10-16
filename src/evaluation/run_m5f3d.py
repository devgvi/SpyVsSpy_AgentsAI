"""M5F.3D: automated live-diagnostic entry point for repeated-game
strategy & relationship dynamics - MEMORY_OFF vs MEMORY_ON vs
MEMORY_RESET, 5-episode series.

    python -m src.evaluation.run_m5f3d --pilot                            # Phase B: CAUTIOUS vs SABOTEUR, MEMORY_RESET, 5 episodes
    python -m src.evaluation.run_m5f3d --benchmark --matchup 0 --condition off    # Phase C
    python -m src.evaluation.run_m5f3d --benchmark --matchup 0 --condition on
    python -m src.evaluation.run_m5f3d --benchmark --matchup 0 --condition reset
    python -m src.evaluation.run_m5f3d --benchmark --matchup 1 --condition off    # matchup 1 = OPPORTUNIST vs OPPORTUNIST
    ...

Phase A (network-free) is covered by tests/test_m5f3d_repeated_game_dynamics.py -
not this driver. Freezes provider=Bedrock, native_tool_like runtime, the
M5F.3C/M5F.3D spec's own frozen matchups/personalities/environmental trap
fixture, communication policy, and memory representation (spec section 31:
do not optimize M5F.3C memory wording during this milestone).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.repeated_series import run_series
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3d-repeated-game-relationship-dynamics-v1")

# Spec section 5: CAUTIOUS-v-SABOTEUR (strongest M5F.3C memory effect) and
# OPPORTUNIST-v-OPPORTUNIST (quiet-control matchup) - both required.
MATCHUPS = [("cautious", "saboteur"), ("opportunist", "opportunist")]
# Spec section 9: one fixed 5-seed schedule, used identically across all
# three memory conditions - never chosen after inspecting outcomes.
SERIES_SEEDS = [5001, 5002, 5003, 5004, 5005]
# Spec section 7: reset immediately before episode 4 (after episode 3).
RESET_AFTER_EPISODE = 3


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


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _run_and_record(out_dir: str, region: str, black_personality: str, white_personality: str, condition: str) -> dict:
    memory_on = condition != "off"
    reset_after_episode = RESET_AFTER_EPISODE if condition == "reset" else None
    telemetry_path = os.path.join(out_dir, "duel_decisions.jsonl")
    label = f"{black_personality}_vs_{white_personality}-{condition.upper()}"
    print(f"\n=== SERIES: {label} (seeds={SERIES_SEEDS}) ===")
    result = run_series(
        black_personality=black_personality, white_personality=white_personality, seeds=SERIES_SEEDS, memory_on=memory_on,
        reset_after_episode=reset_after_episode,
        provider="bedrock", model=MODEL_ID, region=region, agent_runtime="native_tool_like",
        telemetry_path=telemetry_path, llm_timeout=60.0,
        # Real bug found and fixed live: `run_series`'s own default
        # `episode_prefix` is "m5f3c" - since M5F.3D's pilot (Phase B) and
        # Phase C benchmark can run the identical (matchup, condition,
        # seed) combination (CAUTIOUS-v-SABOTEUR/MEMORY_RESET did exactly
        # this), both produced byte-identical episode_ids and their
        # telemetry silently merged in the shared duel_decisions.jsonl -
        # not a live-execution bug (both episodes ran correctly and
        # independently), but an analysis-corrupting id collision. Fixed
        # by using a milestone-specific prefix so M5F.3D never collides
        # with M5F.3C's own runs (or with itself across phases) again.
        episode_prefix="m5f3d",
    )
    for e in result["episodes"]:
        print(f"[{label}] ep{e['episode_number']} seed={e['seed']}: status={e['status']} stop={e['stop_reason']} "
              f"black_d={e['black_decisions']} white_d={e['white_decisions']} messages={e['messages_sent']} "
              f"reset={e['is_reset_episode']} black_mem_chars={e['black_memory_size_chars']}")
    _write_jsonl(os.path.join(out_dir, "episodes.jsonl"), [
        {"series_label": label, "black_personality": black_personality, "white_personality": white_personality, "condition": condition, **e}
        for e in result["episodes"]
    ])
    if memory_on:
        _write_jsonl(os.path.join(out_dir, "memory_snapshots.jsonl"), [
            {"series_label": label, "episode_number": e["episode_number"], "is_reset_episode": e["is_reset_episode"],
             "included_prior_episode_ids_black": e["included_prior_episode_ids_black"], "included_prior_episode_ids_white": e["included_prior_episode_ids_white"],
             "black_memory_snapshot_text": e["black_memory_snapshot_text"], "white_memory_snapshot_text": e["white_memory_snapshot_text"],
             "black_memory_size_chars": e["black_memory_size_chars"], "white_memory_size_chars": e["white_memory_size_chars"]}
            for e in result["episodes"]
        ])
    return result


def pilot(out_dir: str, region: str) -> dict:
    print("\n=== PHASE B: repeated-game pilot (CAUTIOUS vs SABOTEUR, MEMORY_RESET, 5 episodes) ===")
    result = _run_and_record(out_dir, region, "cautious", "saboteur", condition="reset")
    return {"series": [result]}


def benchmark(out_dir: str, region: str, matchup_index: int, condition: str) -> dict:
    black_personality, white_personality = MATCHUPS[matchup_index]
    print(f"\n=== PHASE C: matchup {matchup_index} ({black_personality} vs {white_personality}), condition={condition} ===")
    result = _run_and_record(out_dir, region, black_personality, white_personality, condition=condition)
    return {"series": [result]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    parser.add_argument("--matchup", type=int, default=None, choices=(0, 1), help="Benchmark only: 0=CAUTIOUS-v-SABOTEUR, 1=OPPORTUNIST-v-OPPORTUNIST")
    parser.add_argument("--condition", type=str, default=None, choices=("off", "on", "reset"), help="Benchmark only: memory condition")
    args = parser.parse_args(argv)

    if args.benchmark and (args.matchup is None or args.condition is None):
        print("--benchmark requires --matchup and --condition")
        return 1

    phase_name = "pilot" if args.pilot else f"benchmark-matchup{args.matchup}-{args.condition}"
    print(f"=== M5F.3D repeated-game relationship dynamics ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    if args.benchmark:
        out_dir = BASE_OUT_DIR
        print(f"[STAGED] matchup={args.matchup} condition={args.condition} - writing into the shared {out_dir}/ (not treated as a rerun)")
    else:
        out_dir = resolve_output_dir(BASE_OUT_DIR)
        if out_dir != BASE_OUT_DIR:
            print(f"[RERUN] {BASE_OUT_DIR} already has a completed run - preserving it untouched; this run writes to {out_dir}/")
    os.makedirs(out_dir, exist_ok=True)

    if not _preflight(out_dir, region):
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    result = pilot(out_dir, region) if args.pilot else benchmark(out_dir, region, args.matchup, args.condition)
    run_end = datetime.now(timezone.utc)

    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_summary = detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover
        sleep_summary = {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}

    summary = {
        "phase": phase_name, "seeds": SERIES_SEEDS, "series_run": len(result["series"]),
        "sleep_prevention_active": True, "sleep_assertion_verified": sleep_assertion_verified, "sleep_contamination": sleep_summary,
    }
    with open(os.path.join(out_dir, f"run_summary_{phase_name}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3D {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
