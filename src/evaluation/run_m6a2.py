"""M6A.2: live driver - full-episode Native-vs-Strands trajectory
comparison. One LLM-controlled White Spy (OPPORTUNIST) against a
deterministic scripted_active_v1/OPPORTUNIST Black Spy, SOCIAL_OFF,
cross-episode memory OFF, same canonical 40-decision/180s budget M5E.4
established. Reuses `EpisodeRunner`/`ExperimentSpec`/`VariantSpec` and
`runtime_comparison.py`'s already-tested `balanced_run_order`/
`run_paired_seeds_with_stop_rules` (spec section 45's ">=2 identical
technical-failure-signature episodes -> STOP" is exactly what that
function's `repeated_failure_signature` stop already implements) rather
than re-deriving paired-episode machinery `run_m5e4.py` already built
and validated.

    python -m src.evaluation.run_m6a2 --pilot     # 1 seed, both runtimes
    python -m src.evaluation.run_m6a2 --primary   # 3 paired seeds (spec section 10's documented minimum)
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.episode_trajectory_analysis import (
    build_trajectory_trace,
    classify_overall_trajectory,
    classify_pair,
    find_first_divergence,
)
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.runtime_comparison import (
    VARIANT_A_NATIVE,
    VARIANT_B_STRANDS,
    balanced_run_order,
    run_paired_seeds_with_stop_rules,
)
from src.evaluation.spec import MAP_VERSION, EpisodeSpec, VariantSpec
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm_controller import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, PROMPT_VERSION

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
TIMEOUT = 60.0
BASE_OUT_DIR = os.path.join("evaluation_runs", "m6a2-runtime-trajectory-effects-v1")

PILOT_SEED = 9500
# Spec section 10: recommended 5, documented minimum 3 - the same
# scope-reduction pattern M6A/M6A.1 both used, given this is a MUCH more
# expensive unit (a full up-to-40-decision episode, not one frozen
# decision) on top of an already-large cumulative session live budget
# (M6A: 48 decisions, M6A.1: 110 decisions). Fresh seeds, never reused
# from any prior milestone's own seed list, chosen before any live run.
PRIMARY_SEEDS = [9501, 9502, 9503]

# Spec sections 6, 12-13: identical to M6A.1's own Bedrock model/settings
# and M5E.4's own canonical episode budget - neither raised nor lowered.
MAX_DECISIONS = 40
MAX_EPISODE_GAME_TIME = 180.0
MAX_EPISODE_WALL_TIME = 3600.0
MAX_RUNTIME_FAILURES = 20
PRIMARY_PERSONALITY = "OPPORTUNIST"


def _variant(variant_id: str, agent_runtime: str, region: str) -> VariantSpec:
    return VariantSpec(
        variant_id=variant_id, agent_runtime=agent_runtime, provider="bedrock", model=MODEL_ID,
        llm_timeout=TIMEOUT, llm_region=region, opponent_type="scripted_active", opponent_profile="OPPORTUNIST",
        personality_profile=PRIMARY_PERSONALITY,
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


def _verify_sleep_assertion() -> bool:
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
        return "PreventUserIdleSystemSleep" in output and "caffeinate" in output
    except Exception:  # pragma: no cover
        return False


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
    except Exception as exc:  # pragma: no cover
        return {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}


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


def _outcome_from_result(result) -> dict:
    facts = result.game_facts or {}
    return {
        "mission_success": result.mission_success,
        "escape_success": result.escape_success,
        "mission_items_acquired": facts.get("mission_items_acquired"),
        "rooms_visited": facts.get("rooms_visited"),
    }


def _cost_from_events(events: list[dict]) -> dict:
    n = len(events) or 1
    return {
        "n": len(events),
        "mean_latency_ms": sum(e.get("latency_ms") or 0.0 for e in events) / n,
        "mean_input_tokens": sum(e.get("input_tokens") or 0 for e in events) / n,
        "mean_output_tokens": sum(e.get("output_tokens") or 0 for e in events) / n,
        "total_input_tokens": sum(e.get("input_tokens") or 0 for e in events),
        "total_output_tokens": sum(e.get("output_tokens") or 0 for e in events),
        "model_calls": sum(1 + (1 if e.get("repair_attempted") else 0) for e in events),
        "tool_calls": sum((e.get("runtime_metadata") or {}).get("tool_call_count") or 0 for e in events),
    }


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
        return []  # M6A.2 doesn't need M5E.3.1's raw-provider-call classification; pairing stop rules use technical status below.

    def classify(result, provider_calls):
        if result.status.value == "FAILED":
            return f"failed:{result.error_type}"
        if result.status.value == "ABORTED":
            return f"aborted:{result.error_type}"
        return "ok"

    run_start = datetime.now(timezone.utc)
    outcome = run_paired_seeds_with_stop_rules(seeds, order, episode_spec_for, run_one, load_provider_calls, classify)
    run_end = datetime.now(timezone.utc)

    if outcome.stop_reason:
        print(f"[STOP] {outcome.stop_reason}: {outcome.stop_detail}")
        print(f"[STOP] tasks not run: {outcome.tasks_not_run}")

    results_by_seed: dict[int, dict[str, object]] = {}
    for episode_spec, result in outcome.episodes:
        results_by_seed.setdefault(episode_spec.seed, {})[episode_spec.variant_id] = (episode_spec, result)

    pair_rows: list[dict] = []
    divergence_rows: list[dict] = []
    trace_rows: list[dict] = []
    pair_classifications: list[str] = []
    divergence_context_tags: list[str | None] = []
    native_all_events: list[dict] = []
    strands_all_events: list[dict] = []

    for seed in seeds:
        pair = results_by_seed.get(seed, {})
        if VARIANT_A_NATIVE not in pair or VARIANT_B_STRANDS not in pair:
            continue  # a stop rule cut this pair short - not technically comparable
        native_spec, native_result = pair[VARIANT_A_NATIVE]
        strands_spec, strands_result = pair[VARIANT_B_STRANDS]
        native_events = _load_decision_events(telemetry_path, native_result.episode_id)
        strands_events = _load_decision_events(telemetry_path, strands_result.episode_id)
        native_all_events.extend(native_events)
        strands_all_events.extend(strands_events)

        native_technically_valid = native_result.status.value not in ("FAILED", "ABORTED")
        strands_technically_valid = strands_result.status.value not in ("FAILED", "ABORTED")
        technically_valid = native_technically_valid and strands_technically_valid

        divergence = find_first_divergence(native_events, strands_events) if technically_valid else {"status": "PAIR_INTEGRITY_BROKEN", "decision_index": None, "reason": "technical failure on at least one side"}
        native_outcome = _outcome_from_result(native_result)
        strands_outcome = _outcome_from_result(strands_result)
        pair_class = classify_pair(divergence, native_outcome, strands_outcome, native_result.decision_count, strands_result.decision_count, technically_valid=technically_valid)
        pair_classifications.append(pair_class)
        divergence_context_tags.append(divergence.get("provider_variance_tag"))

        pair_id = f"pair_seed{seed}"
        pair_rows.append({
            "pair_id": pair_id, "seed": seed,
            "native": {"episode_result": native_result.status.value, "mission_progress": native_outcome, "escape": native_outcome["escape_success"],
                       "decision_count": native_result.decision_count, "cost": _cost_from_events(native_events)},
            "strands": {"episode_result": strands_result.status.value, "mission_progress": strands_outcome, "escape": strands_outcome["escape_success"],
                        "decision_count": strands_result.decision_count, "cost": _cost_from_events(strands_events)},
            "first_divergence_index": divergence.get("decision_index"), "first_divergence_context": divergence.get("context_category"),
            "technical_status": "valid" if technically_valid else "invalid",
            "pair_classification": pair_class,
        })
        divergence_rows.append({
            "pair_id": pair_id, "first_shared_state_status": divergence.get("status"),
            "first_divergence_decision": divergence.get("decision_index"),
            "native_action": divergence.get("native_action"), "strands_action": divergence.get("strands_action"),
            "context_category": divergence.get("context_category"), "provider_variance_tag": divergence.get("provider_variance_tag"),
            "trajectory_consequence": pair_class,
        })
        div_idx = divergence.get("decision_index")
        trace_rows.append({"pair_id": pair_id, "native_trace": build_trajectory_trace(native_events, "native", div_idx), "strands_trace": build_trajectory_trace(strands_events, "strands", div_idx)})

    native_cost = _cost_from_events(native_all_events)
    strands_cost = _cost_from_events(strands_all_events)
    cost_differs = native_cost["mean_latency_ms"] > 0 and abs(strands_cost["mean_latency_ms"] - native_cost["mean_latency_ms"]) / native_cost["mean_latency_ms"] >= 0.10
    overall = classify_overall_trajectory(pair_classifications, divergence_context_tags, cost_differs)

    with open(os.path.join(out_dir, "paired_episodes.jsonl"), "w", encoding="utf-8") as f:
        for row in pair_rows:
            f.write(json.dumps(row, default=str) + "\n")
    with open(os.path.join(out_dir, "first_divergence.jsonl"), "w", encoding="utf-8") as f:
        for row in divergence_rows:
            f.write(json.dumps(row, default=str) + "\n")
    with open(os.path.join(out_dir, "decision_trace.jsonl"), "w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, default=str) + "\n")
    with open(os.path.join(out_dir, "trajectory_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pair_classification_counts": dict(Counter(pair_classifications)),
            "divergence_context_tag_counts": dict(Counter(t for t in divergence_context_tags if t)),
            "overall_classification": overall, "cost_differs": cost_differs,
        }, f, indent=2, default=str)
    with open(os.path.join(out_dir, "runtime_cost_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"native_tool_like": native_cost, "strands": strands_cost}, f, indent=2, default=str)

    sleep_summary = _sleep_check(run_start, run_end)
    experiment = {
        "experiment_id": experiment_id, "provider": "bedrock", "region": region, "model_id": MODEL_ID,
        "personality_profile": PRIMARY_PERSONALITY, "seeds": seeds,
        "started_utc": run_start.isoformat(), "ended_utc": run_end.isoformat(),
        "episode_limits": {"max_episode_game_time": MAX_EPISODE_GAME_TIME, "max_episode_wall_time": MAX_EPISODE_WALL_TIME, "max_decisions": MAX_DECISIONS, "max_runtime_failures": MAX_RUNTIME_FAILURES},
    }
    # Written BEFORE summary.json so resolve_output_dir's own rerun-
    # detection (it checks for this file's existence) correctly redirects
    # a later invocation targeting the same base_out_dir to a fresh
    # `-rerun-<timestamp>` directory instead of silently appending
    # duplicate/colliding episode_id telemetry into this one - a real
    # gap found live (this file didn't exist at all until now, so a
    # rerun after an earlier failed/aborted attempt would have collided).
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, default=str)

    summary = {
        "experiment_id": experiment_id, "provider": "bedrock", "region": region, "model_id": MODEL_ID,
        "seeds_requested": seeds, "run_order": {str(k): v for k, v in order.items()},
        "tasks_run": [(spec.seed, spec.variant_id) for spec, _ in outcome.episodes],
        "tasks_not_run": outcome.tasks_not_run, "stop_reason": outcome.stop_reason, "stop_detail": outcome.stop_detail,
        "episode_limits": {"max_episode_game_time": MAX_EPISODE_GAME_TIME, "max_episode_wall_time": MAX_EPISODE_WALL_TIME, "max_decisions": MAX_DECISIONS, "max_runtime_failures": MAX_RUNTIME_FAILURES},
        "personality_profile": PRIMARY_PERSONALITY,
        "pair_classification_counts": dict(Counter(pair_classifications)),
        "overall_classification": overall, "cost_differs": cost_differs,
        "native_cost": native_cost, "strands_cost": strands_cost,
        "sleep_contamination": sleep_summary,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nOutputs written under: {out_dir}/")
    print(f"[OVERALL] {overall}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--primary", action="store_true")
    parser.add_argument("--region", type=str, default=None)
    args = parser.parse_args(argv)

    print(f"=== M6A.2 full-episode runtime trajectory comparison ({'pilot' if args.pilot else 'primary'}) starting {datetime.now(timezone.utc).isoformat()} ===")

    if not bedrock_token_present():
        print("[PREFLIGHT] FAILED: AWS_BEARER_TOKEN_BEDROCK is not set. Refusing to start any episode.")
        return 1
    try:
        region = resolve_bedrock_region(args.region)
    except Exception as exc:
        print(f"[PREFLIGHT] FAILED: {exc}")
        return 1
    print(f"[CONFIG] provider=bedrock model_id={MODEL_ID} region={region} bedrock_auth_mode=bearer_token_env token_present=True")

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1

    if args.pilot:
        experiment_id = "m6a2-runtime-trajectory-effects-v1-pilot"
        base_out_dir = os.path.join(BASE_OUT_DIR, "pilot")
        seeds = [PILOT_SEED]
    else:
        experiment_id = "m6a2-runtime-trajectory-effects-v1-primary"
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
    print(f"=== M6A.2 finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0 if summary["stop_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
