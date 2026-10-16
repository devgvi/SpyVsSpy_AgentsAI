"""M5E.4: automated live-diagnostic entry point for the controlled
Bedrock runtime comparison (Native tool-like vs Strands).

    python -m src.evaluation.run_m5e4 --pilot     # seed 700 only, both variants
    python -m src.evaluation.run_m5e4 --primary   # 5 paired seeds, 701-705

Run the pilot first and inspect its result before ever invoking
--primary (spec section 39: "pilot results must not be merged into the
primary 5-seed comparison" - they are written to a separate `pilot/`
subdirectory and never touch `primary/`).

Freezes provider=bedrock, the same model for both runtimes, and the same
scripted_active_v1/OPPORTUNIST opponent (spec sections 4, 16). Both
variants use `max_decisions=40` (spec section 17, well beyond M5E.3.1's
15-decision smoke cap) and the real `MISSION_TIME=180.0` canonical
deadline (spec section 18, unchanged from `src/constants.py`).

Never logs/writes AWS_BEARER_TOKEN_BEDROCK anywhere - only
`bedrock_auth_mode`/`token_present`. Headless (SDL_VIDEODRIVER=dummy).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.experiment_runner import build_config_snapshot
from src.evaluation.failure_streak import compute_long_horizon_stability
from src.evaluation.multiseed_stability import classify_episode_for_m5e3_1, detect_sleep_contamination, resolve_output_dir
from src.evaluation.pacing_experiment import load_provider_calls_by_episode
from src.evaluation.reporting import write_report
from src.evaluation.results import ExperimentResult
from src.evaluation.runtime_comparison import (
    VARIANT_A_NATIVE,
    VARIANT_B_STRANDS,
    balanced_run_order,
    build_variant_seed_row,
    paired_seed_narrative,
    run_paired_seeds_with_stop_rules,
)
from src.evaluation.spec import MAP_VERSION, EpisodeSpec, ExperimentSpec, VariantSpec
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm_controller import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, PROMPT_VERSION

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
TIMEOUT = 60.0
PILOT_SEED = 700
PRIMARY_SEEDS = [701, 702, 703, 704, 705]
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5e4-bedrock-runtime-comparison-v1")

# Spec sections 17-18: a genuinely longer decision budget than M5E.3.1's
# 15-decision smoke cap, and the real unmodified mission deadline.
MAX_DECISIONS = 40
MAX_EPISODE_GAME_TIME = 180.0  # src.constants.MISSION_TIME - unchanged canonical value
MAX_EPISODE_WALL_TIME = 3600.0
MAX_RUNTIME_FAILURES = 20


def _variant(variant_id: str, agent_runtime: str, region: str) -> VariantSpec:
    return VariantSpec(
        variant_id=variant_id, agent_runtime=agent_runtime, provider="bedrock", model=MODEL_ID,
        llm_timeout=TIMEOUT, llm_region=region, opponent_type="scripted_active", opponent_profile="OPPORTUNIST",
    )


def _episode_spec(experiment_id: str, seed: int, variant: VariantSpec) -> EpisodeSpec:
    return EpisodeSpec(
        experiment_id=experiment_id, episode_id=f"{experiment_id}__{variant.variant_id}__seed{seed}",
        variant=variant, seed=seed, execution_mode="evaluation",
        max_episode_game_time=MAX_EPISODE_GAME_TIME, max_episode_wall_time=MAX_EPISODE_WALL_TIME,
        max_decisions=MAX_DECISIONS, max_runtime_failures=MAX_RUNTIME_FAILURES,
        prompt_version=PROMPT_VERSION, observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        memory_schema_version=AgentMemory.MEMORY_SCHEMA_VERSION, action_schema_version=ACTION_SCHEMA_VERSION,
        map_version=MAP_VERSION,
    )


def _load_decision_events(telemetry_path: str, episode_id: str) -> list[dict]:
    events: list[dict] = []
    if not os.path.exists(telemetry_path):
        return events
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") == "llm_decision" and event.get("episode_id") == episode_id:
                events.append(event)
    return events


def _preflight(out_dir: str, region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=TIMEOUT)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _sleep_check(run_start: datetime, run_end: datetime) -> dict:
    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        return detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover - diagnostic only, never fatal
        return {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}


def _run_phase(experiment_id: str, out_dir: str, seeds: list[int], region: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    telemetry_path = os.path.join(out_dir, "decisions.jsonl")
    variant_a = _variant(VARIANT_A_NATIVE, "native_tool_like", region)
    variant_b = _variant(VARIANT_B_STRANDS, "strands", region)
    variants_by_id = {VARIANT_A_NATIVE: variant_a, VARIANT_B_STRANDS: variant_b}
    order = balanced_run_order(seeds)
    print(f"[ORDER] {order}")

    episode_runner = EpisodeRunner()

    def episode_spec_for(seed: int, variant_id: str):
        return _episode_spec(experiment_id, seed, variants_by_id[variant_id])

    def run_one(episode_spec):
        print(f"[RUN] {episode_spec.variant_id} seed={episode_spec.seed} starting {datetime.now(timezone.utc).isoformat()}")
        result = episode_runner.run(episode_spec, telemetry_path=telemetry_path)
        print(f"[RUN] {episode_spec.variant_id} seed={episode_spec.seed} status={result.status.value} mission_success={result.mission_success} decisions={result.decision_count}")
        return result

    def load_provider_calls(episode_id: str) -> list[dict]:
        return load_provider_calls_by_episode(telemetry_path).get(episode_id, [])

    def classify(result, provider_calls):
        decisions = _load_decision_events(telemetry_path, result.episode_id)
        long_horizon = compute_long_horizon_stability(decisions, provider_calls)
        return classify_episode_for_m5e3_1(result, long_horizon)

    run_start = datetime.now(timezone.utc)
    outcome = run_paired_seeds_with_stop_rules(seeds, order, episode_spec_for, run_one, load_provider_calls, classify)
    run_end = datetime.now(timezone.utc)

    if outcome.stop_reason:
        print(f"[STOP] {outcome.stop_reason}: {outcome.stop_detail}")
        print(f"[STOP] tasks not run: {outcome.tasks_not_run}")

    all_episode_specs = [spec for spec, _ in outcome.episodes]
    all_results = [result for _, result in outcome.episodes]
    combined_spec = ExperimentSpec(
        experiment_id=experiment_id, variants=(variant_a, variant_b), seeds=tuple(seeds),
        execution_mode="evaluation", max_episode_game_time=MAX_EPISODE_GAME_TIME,
        max_episode_wall_time=MAX_EPISODE_WALL_TIME, max_decisions=MAX_DECISIONS, max_runtime_failures=MAX_RUNTIME_FAILURES,
    )
    experiment_result = ExperimentResult(
        experiment_id=experiment_id, spec=combined_spec, episodes=all_results,
        config_snapshot=build_config_snapshot(combined_spec),
        started_at=run_start.isoformat(), finished_at=run_end.isoformat(),
    )
    experiment_result.write(out_dir)
    write_report(experiment_result, out_dir)

    rows_by_variant: dict[str, list[dict]] = {VARIANT_A_NATIVE: [], VARIANT_B_STRANDS: []}
    rows_by_seed: dict[int, dict[str, dict]] = {}
    for episode_spec, result in outcome.episodes:
        provider_calls = load_provider_calls(result.episode_id)
        decisions = _load_decision_events(telemetry_path, result.episode_id)
        long_horizon = compute_long_horizon_stability(decisions, provider_calls)
        status = classify_episode_for_m5e3_1(result, long_horizon)
        row = build_variant_seed_row(result, episode_spec.variant_id, status, long_horizon, provider_calls)
        rows_by_variant[episode_spec.variant_id].append(row)
        rows_by_seed.setdefault(episode_spec.seed, {})[episode_spec.variant_id] = row

    narratives = []
    for seed in seeds:
        pair = rows_by_seed.get(seed, {})
        if VARIANT_A_NATIVE in pair and VARIANT_B_STRANDS in pair:
            narratives.append(paired_seed_narrative(pair[VARIANT_A_NATIVE], pair[VARIANT_B_STRANDS]))

    sleep_summary = _sleep_check(run_start, run_end)

    summary = {
        "experiment_id": experiment_id, "provider": "bedrock", "region": region, "model_id": MODEL_ID,
        "seeds_requested": seeds, "run_order": {str(k): v for k, v in order.items()},
        "tasks_run": [(spec.seed, spec.variant_id) for spec in all_episode_specs],
        "tasks_not_run": outcome.tasks_not_run, "stop_reason": outcome.stop_reason, "stop_detail": outcome.stop_detail,
        "episode_limits": {
            "max_episode_game_time": MAX_EPISODE_GAME_TIME, "max_episode_wall_time": MAX_EPISODE_WALL_TIME,
            "max_decisions": MAX_DECISIONS, "max_runtime_failures": MAX_RUNTIME_FAILURES,
        },
        "per_seed_rows": {"A_native_tool_like_bedrock": rows_by_variant[VARIANT_A_NATIVE], "B_strands_bedrock": rows_by_variant[VARIANT_B_STRANDS]},
        "paired_seed_narratives": narratives,
        "sleep_contamination": sleep_summary,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nOutputs written under: {out_dir}/")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--primary", action="store_true")
    args = parser.parse_args(argv)

    print(f"=== M5E.4 Bedrock runtime comparison ({'pilot' if args.pilot else 'primary'}) starting {datetime.now(timezone.utc).isoformat()} ===")

    if not bedrock_token_present():
        print("[PREFLIGHT] FAILED: AWS_BEARER_TOKEN_BEDROCK is not set. Refusing to start any episode.")
        return 1
    try:
        region = resolve_bedrock_region()
    except Exception as exc:
        print(f"[PREFLIGHT] FAILED: {exc}")
        return 1
    print(f"[CONFIG] provider=bedrock model_id={MODEL_ID} region={region} bedrock_auth_mode=bearer_token_env token_present=True")

    if args.pilot:
        experiment_id = "m5e4-bedrock-runtime-comparison-v1-pilot"
        base_out_dir = os.path.join(BASE_OUT_DIR, "pilot")
        seeds = [PILOT_SEED]
    else:
        experiment_id = "m5e4-bedrock-runtime-comparison-v1-primary"
        base_out_dir = os.path.join(BASE_OUT_DIR, "primary")
        seeds = PRIMARY_SEEDS

    out_dir = resolve_output_dir(base_out_dir)
    if out_dir != base_out_dir:
        print(f"[RERUN] {base_out_dir} already has a completed run - preserving it untouched; this run writes to {out_dir}/")

    if not _preflight(out_dir, region):
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1
    summary = _run_phase(experiment_id, out_dir, seeds, region)

    print(f"\n[SUMMARY] stop_reason={summary['stop_reason']} tasks_run={len(summary['tasks_run'])} tasks_not_run={len(summary['tasks_not_run'])}")
    print(f"=== M5E.4 finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0 if summary["stop_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
