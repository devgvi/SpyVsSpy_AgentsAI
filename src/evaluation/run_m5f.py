"""M5F: automated live-diagnostic entry point for the first controlled
LLM-vs-LLM duel.

    python -m src.evaluation.run_m5f --pilot     # 1 duel, seed 900, 10 decisions/actor
    python -m src.evaluation.run_m5f --primary   # 3 duels, seeds 901/902/903, 40 decisions/actor

Both spies use the same validated Bedrock model and `native_tool_like`
runtime (spec sections 3-5, 37-38 - no runtime/model comparison in this
milestone). Each duel is driven by `run_duel_episode`
(`src/evaluation/duel_runner.py`), a purpose-built driver rather than the
shared single-actor `EpisodeRunner` (see that module's own docstring for
why). Produces per-actor and merged chronological duel traces
(`src/evaluation/duel_trace.py`), opponent-encounter/contested-objective
analysis, and a before/after-free (M5F has no prior baseline to compare
against - this establishes one) summary.

Never logs/writes AWS_BEARER_TOKEN_BEDROCK anywhere. Headless
(SDL_VIDEODRIVER=dummy). Historical M5E.x outputs are untouched (separate
output directory, separate telemetry file).
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
    extract_opponent_encounters,
    merge_duel_trace,
    mutual_same_room_encounters,
    reaction_to_sighting,
    render_duel_markdown,
)
from src.evaluation.decision_trace import decision_budget_breakdown, render_markdown_trace
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f-llm-vs-llm-baseline-v1")
AGENT_RUNTIME = "native_tool_like"  # spec sections 3-4: fixed for M5F baseline

PILOT_SEED = 900
PILOT_MAX_DECISIONS = 10
PRIMARY_SEEDS = [901, 902, 903]
PRIMARY_MAX_DECISIONS = 40  # spec section 31: a ceiling, not a target
MAX_EPISODE_GAME_TIME = 180.0
# Spec section 26: deterministic alternating call order across primary
# episodes - never always Black first.
PRIMARY_CALL_ORDER = {901: "black_first", 902: "white_first", 903: "black_first"}


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


def _verify_sleep_assertion() -> bool:
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
        return "PreventUserIdleSystemSleep" in output and "caffeinate" in output
    except Exception:  # pragma: no cover - diagnostic only
        return False


def _run_one_traced_duel(out_dir: str, seed: int, region: str, max_decisions: int, call_order: str) -> dict:
    experiment_id = "m5f-llm-vs-llm-baseline-v1"
    episode_id = f"{experiment_id}__duel__seed{seed}"
    telemetry_path = os.path.join(out_dir, "decisions.jsonl")

    black_snapshots: dict[int, dict] = {}
    white_snapshots: dict[int, dict] = {}
    last_black_n = 0
    last_white_n = 0

    def on_decision(black_n, white_n, state):
        nonlocal last_black_n, last_white_n
        if black_n > last_black_n:
            black_snapshots[black_n] = build_progress_snapshot(state, constants.BLACK_SPY_ID)
            last_black_n = black_n
        if white_n > last_white_n:
            white_snapshots[white_n] = build_progress_snapshot(state, constants.WHITE_SPY_ID)
            last_white_n = white_n
        if black_n == 0 and white_n == 0:
            black_snapshots[0] = build_progress_snapshot(state, constants.BLACK_SPY_ID)
            white_snapshots[0] = build_progress_snapshot(state, constants.WHITE_SPY_ID)

    print(f"[DUEL] seed={seed} call_order={call_order} max_decisions_per_actor={max_decisions} starting {datetime.now(timezone.utc).isoformat()}")
    result = run_duel_episode(
        seed=seed, provider="bedrock", model=MODEL_ID, region=region, agent_runtime=AGENT_RUNTIME,
        telemetry_path=telemetry_path, episode_id=episode_id, decision_call_order=call_order,
        max_decisions_per_actor=max_decisions, max_episode_game_time=MAX_EPISODE_GAME_TIME,
        max_episode_wall_time=3600.0, llm_timeout=60.0, on_decision=on_decision,
    )
    print(f"[DUEL] seed={seed} status={result.status} stop_reason={result.stop_reason} black_decisions={result.black_decision_count} white_decisions={result.white_decision_count}")

    all_decisions = _load_events(telemetry_path, episode_id, "llm_decision")
    all_lifecycle = _load_events(telemetry_path, episode_id, "action_lifecycle")
    decisions_by_id = {d["decision_id"]: d for d in all_decisions}

    black_records = build_actor_trace_records(all_decisions, all_lifecycle, black_snapshots, episode_id, seed, AGENT_RUNTIME, constants.BLACK_SPY_ID)
    white_records = build_actor_trace_records(all_decisions, all_lifecycle, white_snapshots, episode_id, seed, AGENT_RUNTIME, constants.WHITE_SPY_ID)
    annotate_opponent_visibility(black_records, decisions_by_id)
    annotate_opponent_visibility(white_records, decisions_by_id)

    merged = merge_duel_trace(black_records, white_records)
    duel_markdown = render_duel_markdown(merged, episode_id, seed)
    black_markdown = render_markdown_trace(black_records, episode_id, seed, "native_tool_like_black")
    white_markdown = render_markdown_trace(white_records, episode_id, seed, "native_tool_like_white")

    encounters = extract_opponent_encounters(black_records) + extract_opponent_encounters(white_records)
    contested = contested_objective_events(black_records, white_records)
    mutual = mutual_same_room_encounters(black_records, white_records)

    reactions = []
    for e in extract_opponent_encounters(black_records):
        if e["transition"] == "opponent_not_visible_to_visible":
            reactions.append({"actor_id": constants.BLACK_SPY_ID, "sighting_decision": e["decision_number"], "reactions": reaction_to_sighting(black_records, constants.BLACK_SPY_ID, e["decision_number"])})
    for e in extract_opponent_encounters(white_records):
        if e["transition"] == "opponent_not_visible_to_visible":
            reactions.append({"actor_id": constants.WHITE_SPY_ID, "sighting_decision": e["decision_number"], "reactions": reaction_to_sighting(white_records, constants.WHITE_SPY_ID, e["decision_number"])})

    black_breakdown = decision_budget_breakdown(black_records)
    white_breakdown = decision_budget_breakdown(white_records)

    return {
        "seed": seed, "episode_id": episode_id, "call_order": call_order,
        "status": result.status, "stop_reason": result.stop_reason,
        "game_time": result.game_time, "wall_time": result.wall_time, "winner": result.winner,
        "black": {
            "decisions": result.black_decision_count, "fallbacks": result.black_fallback_count,
            "game_facts": result.black_game_facts, "decision_budget_breakdown": black_breakdown,
        },
        "white": {
            "decisions": result.white_decision_count, "fallbacks": result.white_fallback_count,
            "game_facts": result.white_game_facts, "decision_budget_breakdown": white_breakdown,
        },
        "competitive_facts": result.competitive_facts,
        "encounters": encounters, "mutual_same_room_encounters": mutual,
        "contested_objective_events": contested, "reactions_to_sighting": reactions,
        "duel_markdown": duel_markdown, "black_markdown": black_markdown, "white_markdown": white_markdown,
        "black_records": black_records, "white_records": white_records,
    }


def _write_trace_files(out_dir: str, duel_results: list[dict]) -> None:
    with open(os.path.join(out_dir, "duel_trace.md"), "a", encoding="utf-8") as df, \
         open(os.path.join(out_dir, "black_trace.md"), "a", encoding="utf-8") as bf, \
         open(os.path.join(out_dir, "white_trace.md"), "a", encoding="utf-8") as wf, \
         open(os.path.join(out_dir, "duel_trace.jsonl"), "a", encoding="utf-8") as jf:
        for d in duel_results:
            df.write(d["duel_markdown"]); df.write("\n---\n\n")
            bf.write(d["black_markdown"]); bf.write("\n---\n\n")
            wf.write(d["white_markdown"]); wf.write("\n---\n\n")
            for r in d["black_records"] + d["white_records"]:
                jf.write(json.dumps(r, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--primary", action="store_true")
    args = parser.parse_args(argv)

    print(f"=== M5F LLM-vs-LLM duel ({'pilot' if args.pilot else 'primary'}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    base_out_dir = os.path.join(BASE_OUT_DIR, "pilot" if args.pilot else "primary")
    out_dir = resolve_output_dir(base_out_dir)
    if out_dir != base_out_dir:
        print(f"[RERUN] {base_out_dir} already has a completed run - preserving it untouched; this run writes to {out_dir}/")
    os.makedirs(out_dir, exist_ok=True)

    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    with open(os.path.join(out_dir, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    if not preflight.ok:
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    if args.pilot:
        duel_results = [_run_one_traced_duel(out_dir, PILOT_SEED, region, PILOT_MAX_DECISIONS, "black_first")]
    else:
        duel_results = [
            _run_one_traced_duel(out_dir, seed, region, PRIMARY_MAX_DECISIONS, PRIMARY_CALL_ORDER[seed])
            for seed in PRIMARY_SEEDS
        ]
    run_end = datetime.now(timezone.utc)

    _write_trace_files(out_dir, duel_results)

    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_summary = detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover - diagnostic only
        sleep_summary = {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}

    summary_duels = [{k: v for k, v in d.items() if k not in ("black_records", "white_records", "duel_markdown", "black_markdown", "white_markdown")} for d in duel_results]
    summary = {
        "experiment_id": "m5f-llm-vs-llm-baseline-v1", "provider": "bedrock", "region": region, "model_id": MODEL_ID,
        "agent_runtime": AGENT_RUNTIME, "max_decisions_per_actor": PILOT_MAX_DECISIONS if args.pilot else PRIMARY_MAX_DECISIONS,
        "sleep_prevention_active": True, "sleep_assertion_verified": sleep_assertion_verified,
        "sleep_contamination": sleep_summary,
        "duels": summary_duels,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment_id": "m5f-llm-vs-llm-baseline-v1", "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    for d in duel_results:
        print(f"\n[SUMMARY seed={d['seed']}] status={d['status']} stop_reason={d['stop_reason']} winner={d['winner']}")
        print(f"  black: decisions={d['black']['decisions']} fallbacks={d['black']['fallbacks']} mission_success={d['black']['game_facts']['mission_success']}")
        print(f"  white: decisions={d['white']['decisions']} fallbacks={d['white']['fallbacks']} mission_success={d['white']['game_facts']['mission_success']}")
        print(f"  encounters={len(d['encounters'])} mutual_same_room={d['mutual_same_room_encounters']} contested_objectives={len(d['contested_objective_events'])}")

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
