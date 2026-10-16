"""M5E.3.1: automated live-diagnostic entry point for the Bedrock
multi-seed stability confirmation.

    python -m src.evaluation.run_m5e3_1

Freezes the exact M5E.3-validated config (native_tool_like + Bedrock vs
scripted_active_v1/OPPORTUNIST, same episode limits M5E.2's own
controlled-competitive-benchmark and M5E.3's Phase E already used - these
are the project's validated real active-opponent limits, not smoke-only
ones, so spec section 9 does not require new tuning here) and runs one
real gameplay episode per seed across 5 fixed seeds (601-605), stopping
early (spec sections 3, 19) only if the bearer token expires mid-run or
two episodes show the same technical-failure signature.

Never logs/writes AWS_BEARER_TOKEN_BEDROCK anywhere (spec section 3) -
only `bedrock_auth_mode`/`token_present`. Headless (SDL_VIDEODRIVER=dummy).
Writes under evaluation_runs/m5e3_1-bedrock-multiseed-stability-v1/ (or a
clearly separate `-rerun-<timestamp>` directory if that already has a
completed run - spec sections 20-21, 38) and never touches M5E.3's own
evaluation_runs/m5e3-bedrock-provider-baseline-v1/ output.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import json
import subprocess
import sys
from datetime import datetime, timezone

from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.experiment_runner import build_config_snapshot
from src.evaluation.failure_streak import compute_long_horizon_stability
from src.evaluation.multiseed_stability import (
    build_aggregate_table,
    build_per_seed_row,
    classify_episode_for_m5e3_1,
    cumulative_bedrock_decisions,
    detect_sleep_contamination,
    resolve_output_dir,
    run_seeds_with_stop_rules,
)
from src.evaluation.pacing_experiment import load_provider_calls_by_episode
from src.evaluation.reporting import write_report
from src.evaluation.results import ExperimentResult
from src.evaluation.spec import ExperimentSpec, VariantSpec
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
TIMEOUT = 60.0
SEEDS = [601, 602, 603, 604, 605]
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5e3_1-bedrock-multiseed-stability-v1")

# Spec section 9: reuse of the exact same validated active-opponent
# limits M5E.2's real 5-seed competitive benchmark and M5E.3's Phase E
# episode both already used - a documented decision, not new tuning.
MAX_EPISODE_GAME_TIME = 180.0
MAX_EPISODE_WALL_TIME = 1800.0
MAX_DECISIONS = 15
MAX_RUNTIME_FAILURES = 20

# Spec section 41: M5E.3's own two-day live result, as documented in
# CHANGELOG.md's M5E.3 entry - a fixed historical constant, not
# re-derived from files that could silently drift from what was reported.
M5E3_PRIOR_LIVE_DECISIONS_CLEAN = 95
M5E3_PRIOR_LIVE_DECISIONS_TOTAL = 95


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


def main(argv: list[str] | None = None) -> int:
    print(f"=== M5E.3.1 Bedrock multi-seed stability confirmation starting {datetime.now(timezone.utc).isoformat()} ===")

    if not bedrock_token_present():
        print("[PREFLIGHT] FAILED: AWS_BEARER_TOKEN_BEDROCK is not set. Refusing to start any episode.")
        return 1
    try:
        region = resolve_bedrock_region()
    except Exception as exc:
        print(f"[PREFLIGHT] FAILED: {exc}")
        return 1

    print(f"[CONFIG] provider=bedrock model_id={MODEL_ID} region={region} bedrock_auth_mode=bearer_token_env token_present=True seeds={SEEDS}")

    out_dir = resolve_output_dir(BASE_OUT_DIR)
    if out_dir != BASE_OUT_DIR:
        print(f"[RERUN] {BASE_OUT_DIR} already has a completed run - preserving it untouched; this run writes to {out_dir}/")
    os.makedirs(out_dir, exist_ok=True)
    provider_events_path = os.path.join(out_dir, "provider_events.jsonl")

    # -- preflight (spec section 2) ------------------------------------------
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=TIMEOUT)
    with open(provider_events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    if not preflight.ok:
        # Spec section 2: a stale preflight token is never counted as a
        # gameplay/provider-stability failure - the run stops here, before
        # any episode, with nothing added to the per-seed/aggregate tables.
        print("[PREFLIGHT] FAILED - stopping before any benchmark episode (not counted as a stability failure).")
        return 1

    variant = VariantSpec(
        variant_id="A_native_tool_like_bedrock", agent_runtime="native_tool_like", provider="bedrock",
        model=MODEL_ID, llm_timeout=TIMEOUT, llm_region=region,
        opponent_type="scripted_active", opponent_profile="OPPORTUNIST",
    )
    spec = ExperimentSpec(
        experiment_id="m5e3_1-bedrock-multiseed-stability-v1", variants=(variant,), seeds=tuple(SEEDS),
        execution_mode="evaluation", max_episode_game_time=MAX_EPISODE_GAME_TIME,
        max_episode_wall_time=MAX_EPISODE_WALL_TIME, max_decisions=MAX_DECISIONS,
        max_runtime_failures=MAX_RUNTIME_FAILURES,
    )
    episode_specs = spec.expand()
    telemetry_path = os.path.join(out_dir, "decisions.jsonl")
    episode_runner = EpisodeRunner()

    def run_one(episode_spec):
        print(f"[SEED {episode_spec.seed}] starting {datetime.now(timezone.utc).isoformat()}")
        result = episode_runner.run(episode_spec, telemetry_path=telemetry_path)
        print(f"[SEED {episode_spec.seed}] status={result.status.value} mission_success={result.mission_success} decisions={result.decision_count}")
        return result

    def load_provider_calls(episode_id: str) -> list[dict]:
        return load_provider_calls_by_episode(telemetry_path).get(episode_id, [])

    def classify(result, provider_calls):
        decisions = _load_decision_events(telemetry_path, result.episode_id)
        long_horizon = compute_long_horizon_stability(decisions, provider_calls)
        return classify_episode_for_m5e3_1(result, long_horizon)

    run_start = datetime.now(timezone.utc)
    outcome = run_seeds_with_stop_rules(episode_specs, run_one, load_provider_calls, classify)
    run_end = datetime.now(timezone.utc)

    if outcome.stop_reason:
        print(f"[STOP] {outcome.stop_reason}: {outcome.stop_detail}")
        print(f"[STOP] seeds not run (preserved as absent, never silently substituted): {outcome.seeds_not_run}")

    experiment_result = ExperimentResult(
        experiment_id=spec.experiment_id, spec=spec, episodes=outcome.episodes,
        config_snapshot=build_config_snapshot(spec),
        started_at=run_start.isoformat(), finished_at=run_end.isoformat(),
    )
    experiment_result.write(out_dir)
    write_report(experiment_result, out_dir)

    per_seed_rows = []
    for episode_spec, result in zip(episode_specs[: len(outcome.episodes)], outcome.episodes):
        provider_calls = load_provider_calls(result.episode_id)
        decisions = _load_decision_events(telemetry_path, result.episode_id)
        long_horizon = compute_long_horizon_stability(decisions, provider_calls)
        status = classify_episode_for_m5e3_1(result, long_horizon)
        per_seed_rows.append(build_per_seed_row(result, status, long_horizon, provider_calls))

    aggregate = build_aggregate_table(per_seed_rows)
    cumulative = cumulative_bedrock_decisions(
        M5E3_PRIOR_LIVE_DECISIONS_CLEAN, M5E3_PRIOR_LIVE_DECISIONS_TOTAL,
        aggregate["first_pass_valid"], aggregate["total_decisions"],
    )

    # -- host sleep-log contamination check (spec sections 22-23) -----------
    sleep_prevention_active = os.environ.get("M5E3_1_SLEEP_PREVENTION_ACTIVE") == "1"
    sleep_summary: dict = {"sleep_contamination_detected": None, "matching_events": [], "note": "pmset check unavailable"}
    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_report = detect_sleep_contamination(pmset_output, run_start, run_end)
        sleep_summary = sleep_report.to_dict()
    except Exception as exc:  # pragma: no cover - defensive: sleep-log check is diagnostic, never fatal to the run
        sleep_summary["note"] = f"pmset check failed: {type(exc).__name__}"

    summary = {
        "provider": "bedrock", "region": region, "model_id": MODEL_ID, "runtime": "native_tool_like",
        "opponent": "scripted_active_v1", "opponent_profile": "OPPORTUNIST",
        "seeds_requested": SEEDS, "seeds_run": outcome.seeds_run, "seeds_not_run": outcome.seeds_not_run,
        "stop_reason": outcome.stop_reason, "stop_detail": outcome.stop_detail,
        "episode_limits": {
            "max_episode_game_time": MAX_EPISODE_GAME_TIME, "max_episode_wall_time": MAX_EPISODE_WALL_TIME,
            "max_decisions": MAX_DECISIONS, "max_runtime_failures": MAX_RUNTIME_FAILURES,
        },
        "preflight": preflight.to_dict(),
        "per_seed": per_seed_rows,
        "aggregate": aggregate,
        "cumulative_bedrock_evidence": cumulative,
        "sleep_prevention_active": sleep_prevention_active,
        "sleep_contamination": sleep_summary,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[AGGREGATE] {aggregate}")
    print(f"[CUMULATIVE] {cumulative}")
    print(f"[SLEEP] prevention_active={sleep_prevention_active} contamination_detected={sleep_summary.get('sleep_contamination_detected')}")
    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5E.3.1 finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0 if outcome.stop_reason is None else 1


if __name__ == "__main__":
    sys.exit(main())
