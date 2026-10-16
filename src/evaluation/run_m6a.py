"""M6A: live Stage 1 (frozen decision replay) driver - Native (native_tool_like
representation, spec section 8) vs Strands, same Bedrock model/generation
settings/prompt/action schema/observation serializer/memory version/frozen
snapshot set (spec section 47's live preflight checklist).

    python -m src.evaluation.run_m6a --preflight-only
    python -m src.evaluation.run_m6a --samples 3

Phase A (runtime_surface_audit.md) and Phase B (tests/test_m6a_runtime_comparison.py)
are both network-free and already complete - this script is Stage 1 only
(spec sections 23-31). Stage 2 (full episode comparison) is a separate,
later decision gated on this stage being clean (spec section 48).
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

from src import constants
from src.evaluation.multiseed_stability import resolve_output_dir
from src.evaluation.runtime_snapshot_replay import (
    build_native_adapter,
    build_snapshot_set,
    build_strands_adapter,
    classify_divergence,
    run_one_trial,
)
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm.telemetry import TelemetryLogger

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m6a-native-vs-strands-runtime-v1")
PROMPT_VERSION = "m5a-agent-v1"
OBSERVATION_SCHEMA_VERSION = "obs-v1"
ACTION_SCHEMA_VERSION = "action-v1"
# Spec section 27: recommended 5, minimum 3. Scoped to the documented
# minimum given this session's already-large cumulative live Bedrock spend
# across M5F.3A-D (a explicit, session-established scope-reduction pattern,
# not a silent cut) - 8 categories x 3 samples x 2 runtimes = 48 live calls.
DEFAULT_SAMPLES_PER_SNAPSHOT = 3
WALL_TIMEOUT = 120.0


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


def run_stage1(out_dir: str, region: str, samples_per_snapshot: int) -> dict:
    snapshots = build_snapshot_set()
    manifest = [
        {"snapshot_id": s.snapshot_id, "category": s.category, "legal_actions": s.legal_actions}
        for s in snapshots
    ]
    _write_jsonl(os.path.join(out_dir, "snapshot_manifest.jsonl"), manifest)

    trials: list[dict] = []
    for snapshot in snapshots:
        for sample_index in range(samples_per_snapshot):
            episode_id = f"m6a-native-{snapshot.snapshot_id}-{sample_index}"
            telemetry = TelemetryLogger(path=os.path.join(out_dir, "runtime_steps.jsonl"))
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            adapter = build_native_adapter(
                actor_id=constants.WHITE_SPY_ID, model_client=client, telemetry=telemetry, episode_id=episode_id,
                provider_name="bedrock", model_name=MODEL_ID, personality_profile=None,
            )
            trial = run_one_trial(adapter, snapshot.observation, telemetry, episode_id, wall_timeout=WALL_TIMEOUT)
            trial.update(snapshot_id=snapshot.snapshot_id, category=snapshot.category, sample_index=sample_index, runtime_label="native_tool_like")
            trials.append(trial)
            print(f"[NATIVE] {snapshot.category} #{sample_index}: action={trial['action']} target={trial['target']} valid={trial['valid']} error={trial['runtime_error']}")

        for sample_index in range(samples_per_snapshot):
            episode_id = f"m6a-strands-{snapshot.snapshot_id}-{sample_index}"
            telemetry = TelemetryLogger(path=os.path.join(out_dir, "runtime_steps.jsonl"))
            adapter = build_strands_adapter(
                actor_id=constants.WHITE_SPY_ID, telemetry=telemetry, episode_id=episode_id,
                provider_name="bedrock", model_name=MODEL_ID, region=region, timeout=60.0, personality_profile=None,
            )
            trial = run_one_trial(adapter, snapshot.observation, telemetry, episode_id, wall_timeout=WALL_TIMEOUT)
            trial.update(snapshot_id=snapshot.snapshot_id, category=snapshot.category, sample_index=sample_index, runtime_label="strands")
            trials.append(trial)
            print(f"[STRANDS] {snapshot.category} #{sample_index}: action={trial['action']} target={trial['target']} valid={trial['valid']} error={trial['runtime_error']}")

    _write_jsonl(os.path.join(out_dir, "frozen_replay_trials.jsonl"), trials)
    return {"snapshots": manifest, "trials": trials}


def summarize(trials: list[dict], samples_per_snapshot: int) -> dict:
    by_snapshot: dict[str, dict[str, list[dict]]] = {}
    for t in trials:
        by_snapshot.setdefault(t["snapshot_id"], {}).setdefault(t["runtime_label"], []).append(t)

    per_snapshot_metrics = {}
    divergence_counts: Counter = Counter()
    for snapshot_id, by_runtime in by_snapshot.items():
        native_trials = by_runtime.get("native_tool_like", [])
        strands_trials = by_runtime.get("strands", [])
        metrics = {}
        for label, group in (("native_tool_like", native_trials), ("strands", strands_trials)):
            n = len(group) or 1
            metrics[label] = {
                "action_distribution": dict(Counter(t["action"] for t in group)),
                "target_distribution": dict(Counter(t["target"] for t in group)),
                "first_pass_validity_rate": sum(1 for t in group if t["valid"] and not t["repair_attempted"]) / n,
                "repair_rate": sum(1 for t in group if t["repair_attempted"]) / n,
                "fallback_rate": sum(1 for t in group if t["fallback_used"]) / n,
                "runtime_error_rate": sum(1 for t in group if t["runtime_error"]) / n,
                "mean_latency_ms": (sum(t["latency_ms"] or 0.0 for t in group) / n),
                "mean_input_tokens": (sum(t["input_tokens"] or 0 for t in group) / n),
                "mean_output_tokens": (sum(t["output_tokens"] or 0 for t in group) / n),
            }
        per_snapshot_metrics[snapshot_id] = metrics

        for i in range(min(len(native_trials), len(strands_trials))):
            div = classify_divergence(native_trials[i], strands_trials[i])
            divergence_counts[div] += 1

    return {
        "per_snapshot_metrics": per_snapshot_metrics,
        "divergence_counts": dict(divergence_counts),
        "samples_per_snapshot": samples_per_snapshot,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_SNAPSHOT)
    parser.add_argument("--region", type=str, default=None)
    args = parser.parse_args()

    from src.llm.clients.bedrock_client import resolve_bedrock_region
    region = resolve_bedrock_region(args.region)

    out_dir = resolve_output_dir(BASE_OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run. "
              "Run e.g. `caffeinate -di &` first.", file=sys.stderr)
        sys.exit(1)

    if not _preflight(out_dir, region):
        print("[FATAL] Bedrock preflight failed - see provider_events.jsonl", file=sys.stderr)
        sys.exit(1)

    if args.preflight_only:
        print("[OK] Preflight passed.")
        return

    start = datetime.now(timezone.utc)
    result = run_stage1(out_dir, region, args.samples)
    end = datetime.now(timezone.utc)

    summary = summarize(result["trials"], args.samples)
    experiment = {
        "experiment_id": "m6a-native-vs-strands-runtime-v1",
        "started_utc": start.isoformat(),
        "ended_utc": end.isoformat(),
        "model_id": MODEL_ID,
        "region": region,
        "provider": "bedrock",
        "prompt_version": PROMPT_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "samples_per_snapshot": args.samples,
        "snapshot_count": len(result["snapshots"]),
        "trial_count": len(result["trials"]),
    }
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, default=str)
    with open(os.path.join(out_dir, "runtime_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment": experiment, "metrics": summary}, f, indent=2, default=str)

    print(f"\n[DONE] Stage 1 complete. Output: {out_dir}")
    print(json.dumps(summary["divergence_counts"], indent=2))


if __name__ == "__main__":
    main()
