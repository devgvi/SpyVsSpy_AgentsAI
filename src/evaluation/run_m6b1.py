"""M6B.1: live driver - episode-scale confirmation of the M6B repair-hint
fix. Same paired-episode infrastructure M6A.2 already established
(`EpisodeRunner`/`VariantSpec`/`runtime_comparison.py`'s balanced order +
stop-on-repeated-failure), but the analysis is PLACE_TRAP-outcome-focused
(spec sections 7-11) rather than trajectory-divergence-focused. A fresh
seed set (never reused from M6A.2's own 9500-9503).

    python -m src.evaluation.run_m6b1 --region us-east-1
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
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.place_trap_reliability_analysis import (
    classify_place_trap_event,
    classify_place_trap_process,
    summarize_place_trap_attempts,
    summarize_place_trap_process,
)
from src.evaluation.runtime_comparison import VARIANT_A_NATIVE, VARIANT_B_STRANDS, balanced_run_order, run_paired_seeds_with_stop_rules
from src.evaluation.spec import MAP_VERSION, EpisodeSpec, VariantSpec
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm_controller import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, PROMPT_VERSION

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
TIMEOUT = 60.0
BASE_OUT_DIR = os.path.join("evaluation_runs", "m6b1-tool-contract-confirmation-v1")

# Spec section 6: a fresh, frozen seed list - never reused from M6A.2's
# own 9500-9503.
SEEDS = [9601, 9602, 9603, 9604, 9605]
MAX_DECISIONS = 40
MAX_EPISODE_GAME_TIME = 180.0
MAX_EPISODE_WALL_TIME = 3600.0
MAX_RUNTIME_FAILURES = 20
PERSONALITY = "OPPORTUNIST"

# Spec section 27/28 stop rules - thresholds decided before looking at
# this milestone's own live data.
MIN_VALID_RATE_STOP_THRESHOLD = 0.50
NEW_FAILURE_CLASS_STOP_COUNT = 3


def _variant(variant_id: str, agent_runtime: str, region: str) -> VariantSpec:
    return VariantSpec(
        variant_id=variant_id, agent_runtime=agent_runtime, provider="bedrock", model=MODEL_ID,
        llm_timeout=TIMEOUT, llm_region=region, opponent_type="scripted_active", opponent_profile="OPPORTUNIST",
        personality_profile=PERSONALITY,
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


def _load_decision_events(telemetry_path: str, episode_id: str) -> list[dict]:
    events: list[dict] = []
    if not os.path.exists(telemetry_path):
        return events
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("event_type") == "llm_decision" and e.get("episode_id") == episode_id:
                events.append(e)
    return events


def _episode_metrics(result, events: list[dict]) -> dict:
    n = len(events) or 1
    facts = result.game_facts or {}
    return {
        "episode_id": result.episode_id, "status": result.status.value, "decision_count": result.decision_count,
        "mission_success": result.mission_success, "escape_success": result.escape_success,
        "mission_items_acquired": facts.get("mission_items_acquired"), "rooms_visited": facts.get("rooms_visited"),
        "traps_placed": facts.get("traps_placed"),
        "repair_count": sum(1 for e in events if e.get("repair_attempted")),
        "fallback_count": sum(1 for e in events if e.get("fallback_used")),
        "mean_latency_ms": sum(e.get("latency_ms") or 0.0 for e in events) / n,
        "mean_output_tokens": sum(e.get("output_tokens") or 0 for e in events) / n,
        "total_input_tokens": sum(e.get("input_tokens") or 0 for e in events),
        "total_output_tokens": sum(e.get("output_tokens") or 0 for e in events),
        "model_calls": sum(1 + (1 if e.get("repair_attempted") else 0) for e in events),
        "tool_calls": sum((e.get("runtime_metadata") or {}).get("tool_call_count") or 0 for e in events),
    }


def _run_phase(experiment_id: str, out_dir: str, region: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    telemetry_path = os.path.join(out_dir, "decisions.jsonl")
    variant_a = _variant(VARIANT_A_NATIVE, "native_tool_like", region)
    variant_b = _variant(VARIANT_B_STRANDS, "strands", region)
    variants_by_id = {VARIANT_A_NATIVE: variant_a, VARIANT_B_STRANDS: variant_b}
    order = balanced_run_order(SEEDS)
    print(f"[ORDER] {order}")

    episode_runner = EpisodeRunner()

    def episode_spec_for(seed: int, variant_id: str):
        return _episode_spec(experiment_id, seed, variants_by_id[variant_id])

    def run_one(episode_spec):
        print(f"[RUN] {episode_spec.variant_id} seed={episode_spec.seed} starting {datetime.now(timezone.utc).isoformat()}")
        result = episode_runner.run(episode_spec, telemetry_path=telemetry_path)
        print(f"[RUN] {episode_spec.variant_id} seed={episode_spec.seed} status={result.status.value} decisions={result.decision_count}")
        return result

    def load_provider_calls(episode_id: str) -> list[dict]:
        return []

    def classify(result, provider_calls):
        if result.status.value in ("FAILED", "ABORTED"):
            return f"{result.status.value.lower()}:{result.error_type}"
        return "ok"

    run_start = datetime.now(timezone.utc)
    outcome = run_paired_seeds_with_stop_rules(SEEDS, order, episode_spec_for, run_one, load_provider_calls, classify)
    run_end = datetime.now(timezone.utc)

    if outcome.stop_reason:
        print(f"[STOP] {outcome.stop_reason}: {outcome.stop_detail}")

    native_events: list[dict] = []
    strands_events: list[dict] = []
    episode_rows: list[dict] = []
    for episode_spec, result in outcome.episodes:
        events = _load_decision_events(telemetry_path, result.episode_id)
        metrics = _episode_metrics(result, events)
        metrics["variant_id"] = episode_spec.variant_id
        metrics["seed"] = episode_spec.seed
        episode_rows.append(metrics)
        if episode_spec.variant_id == VARIANT_A_NATIVE:
            native_events.extend(events)
        else:
            strands_events.extend(events)

    with open(os.path.join(out_dir, "episode_metrics.jsonl"), "w", encoding="utf-8") as f:
        for row in episode_rows:
            f.write(json.dumps(row, default=str) + "\n")

    place_trap_events = [e for e in (native_events + strands_events) if e.get("raw_response") and e["raw_response"].get("action") == "place_trap"]
    with open(os.path.join(out_dir, "place_trap_events.jsonl"), "w", encoding="utf-8") as f:
        for e in place_trap_events:
            row = {
                "episode_id": e.get("episode_id"), "runtime": "native_tool_like" if VARIANT_A_NATIVE in (e.get("episode_id") or "") else "strands",
                "raw_response": e.get("raw_response"), "repair_raw_response": e.get("repair_raw_response"),
                "executed_action": e.get("executed_action"), "reason_classification": classify_place_trap_event(e),
                "process_classification": classify_place_trap_process(e),
            }
            f.write(json.dumps(row, default=str) + "\n")

    native_reason = summarize_place_trap_attempts(native_events)
    strands_reason = summarize_place_trap_attempts(strands_events)
    native_process = summarize_place_trap_process(native_events)
    strands_process = summarize_place_trap_process(strands_events)

    native_cost = _cost_summary(native_events)
    strands_cost = _cost_summary(strands_events)

    # Stop rules (spec sections 27-28), evaluated on the collected data.
    strands_conditional_valid_rate = strands_process.get("conditional_valid_rate")
    stop_rule_low_validity = (
        strands_conditional_valid_rate is not None and strands_conditional_valid_rate < MIN_VALID_RATE_STOP_THRESHOLD
    )
    new_failure_counts = Counter(c for e in strands_events for c in [classify_place_trap_event(e)] if c and c != "SUCCESS")
    stop_rule_new_failure_class = {k: v for k, v in new_failure_counts.items() if v >= NEW_FAILURE_CLASS_STOP_COUNT}

    result = {
        "experiment_id": experiment_id, "region": region, "model_id": MODEL_ID, "seeds": SEEDS,
        "stop_reason": outcome.stop_reason, "stop_detail": outcome.stop_detail,
        "tasks_run": [(s.seed, s.variant_id) for s, _ in outcome.episodes], "tasks_not_run": outcome.tasks_not_run,
        "reason_taxonomy": {"native_tool_like": native_reason, "strands": strands_reason},
        "process_taxonomy": {"native_tool_like": native_process, "strands": strands_process},
        "cost": {"native_tool_like": native_cost, "strands": strands_cost},
        "stop_rule_low_validity_triggered": stop_rule_low_validity,
        "stop_rule_new_failure_class_triggered": stop_rule_new_failure_class,
        "sleep_contamination": _sleep_check(run_start, run_end),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def _cost_summary(events: list[dict]) -> dict:
    n = len(events) or 1
    return {
        "n_decisions": len(events),
        "mean_latency_ms": sum(e.get("latency_ms") or 0.0 for e in events) / n,
        "mean_output_tokens": sum(e.get("output_tokens") or 0 for e in events) / n,
        "mean_input_tokens": sum(e.get("input_tokens") or 0 for e in events) / n,
        "model_calls": sum(1 + (1 if e.get("repair_attempted") else 0) for e in events),
        "tool_calls": sum((e.get("runtime_metadata") or {}).get("tool_call_count") or 0 for e in events),
        "repair_attempts": sum(1 for e in events if e.get("repair_attempted")),
    }


def _sleep_check(run_start, run_end) -> dict:
    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        return detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover
        return {"sleep_contamination_detected": None, "note": f"pmset check failed: {type(exc).__name__}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    args = parser.parse_args()

    print(f"=== M6B.1 episode-scale tool contract confirmation starting {datetime.now(timezone.utc).isoformat()} ===")
    if not bedrock_token_present():
        print("[PREFLIGHT] FAILED: AWS_BEARER_TOKEN_BEDROCK is not set.")
        return 1
    try:
        region = resolve_bedrock_region(args.region)
    except Exception as exc:
        print(f"[PREFLIGHT] FAILED: {exc}")
        return 1

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start.", file=sys.stderr)
        return 1

    experiment_id = "m6b1-tool-contract-confirmation-v1"
    out_dir = resolve_output_dir(BASE_OUT_DIR)
    if out_dir != BASE_OUT_DIR:
        print(f"[RERUN] {BASE_OUT_DIR} already has a completed run - writing to {out_dir}/")

    if not _preflight(out_dir, region):
        print("[PREFLIGHT] FAILED.")
        return 1

    result = _run_phase(experiment_id, out_dir, region)
    print(f"\n[SUMMARY] stop_reason={result['stop_reason']}")
    print(json.dumps({"reason_taxonomy": result["reason_taxonomy"], "process_taxonomy": result["process_taxonomy"]}, indent=2, default=str))
    print(f"=== M6B.1 finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
