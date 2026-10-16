"""M5E.4.2: automated live-diagnostic entry point for the action-semantics
/ interaction-sequencing clarity fix.

    python -m src.evaluation.run_m5e4_2 --pilot     # 1 short episode
    python -m src.evaluation.run_m5e4_2 --primary   # same 2x2 as M5E.4.1

Reruns the exact same seeds/runtimes/budget as M5E.4.1 (801/802 x
native_tool_like/strands, 40 decisions, provider/model/opponent
unchanged) after the door-affordance representation fix in
`src/llm_controller.py`/`src/observation.py` - purely a representation
change, no engine/gameplay/budget change (spec section 5). Produces the
same trace instrumentation as M5E.4.1 plus the new door-sequence fields/
metrics, and a before/after comparison against M5E.4.1's own recorded
summary.json (spec section 40) - read from disk, never re-derived or
guessed.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src.evaluation.decision_trace import (
    build_checkpoints,
    build_decision_trace_records,
    build_progress_snapshot,
    classify_budget_conclusion,
    decision_budget_breakdown,
    decision_to_progress_metrics,
    door_action_sequence_narrative,
    door_sequence_metrics,
    render_markdown_trace,
    stall_analysis,
)
from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.multiseed_stability import classify_episode_for_m5e3_1, detect_sleep_contamination, resolve_output_dir
from src.evaluation.pacing_experiment import load_provider_calls_by_episode
from src.evaluation.failure_streak import compute_long_horizon_stability
from src.evaluation.run_m5e4 import MAX_EPISODE_WALL_TIME, MAX_RUNTIME_FAILURES, TIMEOUT, _variant
from src.evaluation.spec import MAP_VERSION, EpisodeSpec
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm_controller import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, PROMPT_VERSION
from src import constants

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5e4_2-action-semantics-v1")
M5E4_1_PRIMARY_SUMMARY = os.path.join("evaluation_runs", "m5e4_1-mission-progress-trace-v1", "primary", "summary.json")

PILOT_SEED = 800
PILOT_MAX_DECISIONS = 10
PRIMARY_SEEDS = [801, 802]  # spec section 6: same diagnostic seeds as M5E.4.1
PRIMARY_MAX_DECISIONS = 40  # spec section 7: NOT raised
MAX_EPISODE_GAME_TIME = 180.0


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


def _episode_spec(experiment_id: str, seed: int, variant, max_decisions: int) -> EpisodeSpec:
    return EpisodeSpec(
        experiment_id=experiment_id, episode_id=f"{experiment_id}__{variant.variant_id}__seed{seed}",
        variant=variant, seed=seed, execution_mode="evaluation",
        max_episode_game_time=MAX_EPISODE_GAME_TIME, max_episode_wall_time=MAX_EPISODE_WALL_TIME,
        max_decisions=max_decisions, max_runtime_failures=MAX_RUNTIME_FAILURES,
        prompt_version=PROMPT_VERSION, observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        memory_schema_version=AgentMemory.MEMORY_SCHEMA_VERSION, action_schema_version=ACTION_SCHEMA_VERSION,
        map_version=MAP_VERSION,
    )


def _run_one_traced_episode(experiment_id: str, out_dir: str, seed: int, runtime: str, region: str, max_decisions: int) -> dict:
    variant = _variant(f"{runtime}_bedrock", runtime, region)
    episode_spec = _episode_spec(experiment_id, seed, variant, max_decisions)
    telemetry_path = os.path.join(out_dir, "decisions.jsonl")

    progress_snapshots: dict[int, dict] = {}

    def on_decision(n, state):
        progress_snapshots[n] = build_progress_snapshot(state, constants.WHITE_SPY_ID)

    print(f"[RUN] {runtime} seed={seed} max_decisions={max_decisions} starting {datetime.now(timezone.utc).isoformat()}")
    runner = EpisodeRunner()
    result = runner.run(episode_spec, telemetry_path=telemetry_path, on_decision=on_decision)
    print(f"[RUN] {runtime} seed={seed} status={result.status.value} mission_success={result.mission_success} decisions={result.decision_count}")

    decisions = _load_events(telemetry_path, result.episode_id, "llm_decision")
    lifecycle = _load_events(telemetry_path, result.episode_id, "action_lifecycle")
    provider_calls = load_provider_calls_by_episode(telemetry_path).get(result.episode_id, [])
    long_horizon = compute_long_horizon_stability(decisions, provider_calls)
    technical_status = classify_episode_for_m5e3_1(result, long_horizon)

    records = build_decision_trace_records(decisions, lifecycle, progress_snapshots, result.episode_id, seed, runtime)
    markdown = render_markdown_trace(records, result.episode_id, seed, runtime)
    checkpoints = build_checkpoints(records)
    breakdown = decision_budget_breakdown(records)
    progress_metrics = decision_to_progress_metrics(records)
    budget_conclusion = classify_budget_conclusion(records, breakdown)
    stall = stall_analysis(records)
    door_metrics = door_sequence_metrics(records)
    door_narrative = door_action_sequence_narrative(records)

    open_door_interact_loop_present = any(v and v > 0 for v in door_metrics["repeated_open_door_interacts_before_traverse"].values())

    return {
        "episode_id": result.episode_id, "seed": seed, "runtime": runtime, "technical_status": technical_status,
        "decisions_used": result.decision_count, "mission_success": result.mission_success,
        "first_pass_valid": long_horizon.first_pass_valid_decisions, "final_valid": long_horizon.valid_decisions,
        "repairs": sum(1 for r in records if r["repair_attempted"]),
        "provider_failures": sum(1 for c in provider_calls if c.get("provider_error_type")),
        "records": records, "markdown": markdown, "checkpoints": checkpoints,
        "decision_budget_breakdown": breakdown, "decision_to_progress_metrics": progress_metrics,
        "budget_conclusion": budget_conclusion, "stall_analysis": stall,
        "door_sequence_metrics": door_metrics, "door_action_sequence_narrative": door_narrative,
        "open_door_interact_loop_present": open_door_interact_loop_present,
    }


def _write_trace_files(out_dir: str, episode_results: list[dict]) -> None:
    with open(os.path.join(out_dir, "decision_trace.jsonl"), "a", encoding="utf-8") as jf, \
         open(os.path.join(out_dir, "decision_trace.md"), "a", encoding="utf-8") as mf, \
         open(os.path.join(out_dir, "door_sequences.jsonl"), "a", encoding="utf-8") as df:
        for ep in episode_results:
            for record in ep["records"]:
                jf.write(json.dumps(record, default=str) + "\n")
            mf.write(ep["markdown"])
            mf.write("\n---\n\n")
            df.write(json.dumps({
                "episode_id": ep["episode_id"], "seed": ep["seed"], "runtime": ep["runtime"],
                "door_sequence_metrics": ep["door_sequence_metrics"],
                "door_action_sequence_narrative": ep["door_action_sequence_narrative"],
            }, default=str) + "\n")


def _load_m5e4_1_baseline() -> dict | None:
    if not os.path.exists(M5E4_1_PRIMARY_SUMMARY):
        return None
    with open(M5E4_1_PRIMARY_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)


def _before_after_table(baseline: dict | None, episode_results: list[dict]) -> dict:
    """Spec section 40 - built only from the two runs' own already-
    recorded numbers, never re-derived or estimated."""
    if baseline is None:
        return {"note": "M5E.4.1 primary summary.json not found - before/after comparison unavailable"}

    def agg(episodes: list[dict], key_path: list) -> float | None:
        values = []
        for ep in episodes:
            v = ep
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else getattr(v, k, None)
                if v is None:
                    break
            if isinstance(v, (int, float)):
                values.append(v)
        return round(sum(values) / len(values), 3) if values else None

    before_episodes = baseline.get("episodes", [])
    after_episodes = episode_results

    def loop_count(episodes, repeated_key_getter):
        return sum(1 for ep in episodes if repeated_key_getter(ep))

    before_loops = sum(
        1 for ep in before_episodes
        if "door_reception_corridor" in (ep.get("stall_analysis") or "") or "door_main_office_ambassador" in (ep.get("stall_analysis") or "")
    )
    after_loops = sum(1 for ep in after_episodes if ep["open_door_interact_loop_present"])

    return {
        "repeat_rate_before": agg(before_episodes, ["decision_budget_breakdown", "repeated_fraction"]),
        "repeat_rate_after": agg(after_episodes, ["decision_budget_breakdown", "repeated_fraction"]),
        "open_door_interact_loop_episodes_before": before_loops,
        "open_door_interact_loop_episodes_after": after_loops,
        "episodes_total": len(after_episodes),
        "mission_items_before": agg(before_episodes, ["decision_to_progress_metrics", "decision_to_first_mission_item"]),
        "note_mission_items": "average decision-of-first-item where it occurred; None entries excluded, not treated as 0",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--primary", action="store_true")
    args = parser.parse_args(argv)

    print(f"=== M5E.4.2 action-semantics fix ({'pilot' if args.pilot else 'primary'}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=TIMEOUT)
    with open(os.path.join(out_dir, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    if not preflight.ok:
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    if args.pilot:
        experiment_id = "m5e4_2-action-semantics-v1-pilot"
        episode_results = [_run_one_traced_episode(experiment_id, out_dir, PILOT_SEED, "native_tool_like", region, PILOT_MAX_DECISIONS)]
    else:
        experiment_id = "m5e4_2-action-semantics-v1-primary"
        episode_results = []
        for seed in PRIMARY_SEEDS:
            for runtime in ("native_tool_like", "strands"):
                episode_results.append(_run_one_traced_episode(experiment_id, out_dir, seed, runtime, region, PRIMARY_MAX_DECISIONS))
    run_end = datetime.now(timezone.utc)

    _write_trace_files(out_dir, episode_results)

    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_summary = detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover - diagnostic only
        sleep_summary = {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}

    before_after = _before_after_table(_load_m5e4_1_baseline(), episode_results) if not args.pilot else None

    summary_episodes = [
        {k: v for k, v in ep.items() if k not in ("records", "markdown")} for ep in episode_results
    ]
    summary = {
        "experiment_id": experiment_id, "provider": "bedrock", "region": region, "model_id": MODEL_ID,
        "max_decisions": PILOT_MAX_DECISIONS if args.pilot else PRIMARY_MAX_DECISIONS,
        "sleep_prevention_active": True, "sleep_assertion_verified": sleep_assertion_verified,
        "sleep_contamination": sleep_summary,
        "episodes": summary_episodes,
        "before_after_vs_m5e4_1": before_after,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    # Real bug found during this milestone's own pilot rerun: without an
    # `experiment.json` marker, `resolve_output_dir`'s rerun-detection
    # never fires (it only checks for that one file), so a second
    # invocation with the same experiment_id silently appended into the
    # *same* decisions.jsonl as the first - first_pass_valid=20 on a
    # 10-decision episode was the tell. Write it, matching the exact
    # marker every other M5E.3.1/M5E.4/M5E.4.1 driver already relies on.
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment_id": experiment_id, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    for ep in episode_results:
        print(f"\n[EPISODE] seed={ep['seed']} runtime={ep['runtime']} status={ep['technical_status']}")
        print(f"  decisions_used={ep['decisions_used']} first_pass_valid={ep['first_pass_valid']} repairs={ep['repairs']}")
        print(f"  budget_conclusion={ep['budget_conclusion']}")
        print(f"  stall_analysis={ep['stall_analysis']}")
        print(f"  open_door_interact_loop_present={ep['open_door_interact_loop_present']}")
        print(f"  door_sequence_metrics={ep['door_sequence_metrics']}")
        for line in ep["door_action_sequence_narrative"]:
            print(f"    {line}")

    if before_after:
        print(f"\n[BEFORE/AFTER] {before_after}")

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5E.4.2 finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
