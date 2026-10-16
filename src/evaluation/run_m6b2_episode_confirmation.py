"""M6B.2 spec section 35 (optional episode confirmation, run only because
Phase E's focused single-decision tests came back clean - spec section
35's own gate condition): a SMALL live episode-scale check specifically of
the never-inspected-target repeated-failure pattern (M6B.1's seed 9605
outlier), now that RP12's repair hint exists. Phase E (single, isolated
decisions) structurally cannot exercise a MULTI-decision fixation loop -
only a real multi-decision episode can show whether the new repair hint
actually breaks one once it starts. Deliberately small (spec section 35:
"Do not rerun a large benchmark") - 2 fresh seeds (never reused from
M6A.2's 9500-9503 or M6B.1's 9601-9605), paired Native/Strands, giving
exactly 2 Native + 2 Strands episodes (spec section 35's own number).
Same harness settings as M6B.1 throughout (spec section 37): OPPORTUNIST
personality, scripted_active_v1/OPPORTUNIST opponent, 40-decision/180s
budget, SOCIAL_OFF, cross-episode memory OFF.

    python -m src.evaluation.run_m6b2_episode_confirmation --region us-east-1
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

from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.multiseed_stability import detect_sleep_contamination
from src.evaluation.place_trap_reliability_analysis import classify_place_trap_event
from src.evaluation.runtime_comparison import VARIANT_A_NATIVE, VARIANT_B_STRANDS, balanced_run_order, run_paired_seeds_with_stop_rules
from src.evaluation.spec import MAP_VERSION, EpisodeSpec, VariantSpec
from src.llm.agent_memory import AgentMemory
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm_controller import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, PROMPT_VERSION

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
TIMEOUT = 60.0
OUT_DIR = os.path.join("evaluation_runs", "m6b2-harness-parity-target-legality-v1")

# Spec section 35: fresh, frozen, never reused from M6A.2 (9500-9503) or
# M6B.1 (9601-9605).
SEEDS = [9801, 9802]
MAX_DECISIONS = 40
MAX_EPISODE_GAME_TIME = 180.0
MAX_EPISODE_WALL_TIME = 3600.0
MAX_RUNTIME_FAILURES = 20
PERSONALITY = "OPPORTUNIST"


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
        f.write(json.dumps({"phase": "preflight_episode_confirmation", **preflight.to_dict()}, default=str) + "\n")
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


def _same_target_repeat_rate(events: list[dict]) -> dict:
    """Spec section 36's secondary metric: within one episode, how often a
    place_trap attempt repeats the exact same (target, trap_type) pair a
    PRIOR attempt in this same episode already had rejected - the seed9605
    signature. Uses the same invalid_action_signature the harness itself
    now tracks (spec section 11), read straight from telemetry rather than
    recomputed here."""
    max_signature_count = 0
    invalid_place_trap_attempts = 0
    for e in events:
        ip = e.get("initial_proposal") or {}
        if ip.get("action") != "place_trap":
            continue
        invalid_place_trap_attempts += 1 if not e.get("final_model_action_valid") else 0
        count = e.get("same_invalid_action_signature_count")
        if count is not None:
            max_signature_count = max(max_signature_count, count)
    return {"invalid_place_trap_attempts": invalid_place_trap_attempts, "max_same_invalid_action_signature_count": max_signature_count}


def _episode_metrics(result, events: list[dict]) -> dict:
    place_trap_events = [e for e in events if (e.get("initial_proposal") or {}).get("action") == "place_trap"]
    target_legality_failures = [
        e for e in place_trap_events
        if e.get("error_type") in ("target_not_inspected", "target_not_trappable", "illegal_target", "out_of_range")
    ]
    repeat = _same_target_repeat_rate(events)
    return {
        "episode_id": result.episode_id, "status": result.status.value, "decision_count": result.decision_count,
        "mission_success": result.mission_success, "escape_success": result.escape_success,
        "place_trap_attempts": len(place_trap_events),
        "place_trap_target_legality_failures": len(target_legality_failures),
        "place_trap_repair_success": sum(1 for e in place_trap_events if e.get("repair_attempted") and e.get("final_model_action_valid")),
        "place_trap_fallback": sum(1 for e in place_trap_events if e.get("fallback_used")),
        **repeat,
    }


def _sleep_check(run_start, run_end) -> dict:
    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        return detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover
        return {"sleep_contamination_detected": None, "note": f"pmset check failed: {type(exc).__name__}"}


def run(region: str) -> dict:
    experiment_id = "m6b2-episode-confirmation-v1"
    telemetry_path = os.path.join(OUT_DIR, "episode_confirmation_decisions.jsonl")
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

    episode_rows: list[dict] = []
    for episode_spec, result in outcome.episodes:
        events = _load_decision_events(telemetry_path, result.episode_id)
        metrics = _episode_metrics(result, events)
        metrics["variant_id"] = episode_spec.variant_id
        metrics["seed"] = episode_spec.seed
        episode_rows.append(metrics)

    with open(os.path.join(OUT_DIR, "episode_confirmation_metrics.jsonl"), "w", encoding="utf-8") as f:
        for row in episode_rows:
            f.write(json.dumps(row, default=str) + "\n")

    result = {
        "experiment_id": experiment_id, "region": region, "model_id": MODEL_ID, "seeds": SEEDS,
        "stop_reason": outcome.stop_reason, "stop_detail": outcome.stop_detail,
        "tasks_run": [(s.seed, s.variant_id) for s, _ in outcome.episodes], "tasks_not_run": outcome.tasks_not_run,
        "episode_metrics": episode_rows,
        "sleep_contamination": _sleep_check(run_start, run_end),
    }
    with open(os.path.join(OUT_DIR, "episode_confirmation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    args = parser.parse_args()

    print(f"=== M6B.2 episode confirmation starting {datetime.now(timezone.utc).isoformat()} ===")
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

    if not _preflight(OUT_DIR, region):
        print("[PREFLIGHT] FAILED.")
        return 1

    result = run(region)
    print(f"\n[SUMMARY] stop_reason={result['stop_reason']}")
    print(json.dumps(result["episode_metrics"], indent=2, default=str))
    print(f"=== M6B.2 episode confirmation finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
