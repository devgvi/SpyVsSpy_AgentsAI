"""M6C.1 Stage C (spec section 34): Strands replication of the 2x2
factorial design, run only after Native (Stage B) analysis is complete.
Mandatory snapshots: RECENT_FAILED_ACTION, MULTIPLE_LEGAL_TARGETS,
NO_MEMORY_RELEVANT - plus any other high-impact states justified by
Stage B's own results (passed via --categories).

    python -m src.evaluation.run_m6c1_stage_c --region us-east-1 --categories RECENT_FAILED_ACTION MULTIPLE_LEGAL_TARGETS NO_MEMORY_RELEVANT
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
from src.agents.registry import create_agent_runtime
from src.evaluation.memory_counterbalancing import COUNTERBALANCING_CONDITIONS, counterbalancing_transform
from src.evaluation.memory_snapshot_set import build_memory_snapshot_set
from src.evaluation.runtime_snapshot_replay import run_one_trial
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm.telemetry import TelemetryLogger

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6c1-memory-representation-counterbalancing-v1")
SAMPLES_PER_CONDITION = 5
WALL_TIMEOUT = 60.0


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
        f.write(json.dumps({"phase": "preflight_stage_c", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _rp_manifest_clean() -> bool:
    from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST

    return all(e.status != FAIL for e in RUNTIME_PARITY_MANIFEST)


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def run(region: str, categories: list[str]) -> dict:
    all_snapshots = {s.category: s for s in build_memory_snapshot_set()}
    snapshots = [all_snapshots[c] for c in categories]
    rows: list[dict] = []

    for snap in snapshots:
        for condition in COUNTERBALANCING_CONDITIONS:
            transform = counterbalancing_transform(condition, snap.synthetic_memory)
            for i in range(SAMPLES_PER_CONDITION):
                episode_id = f"m6c1-strands-{snap.snapshot_id}-{condition}-{i}"
                telemetry = TelemetryLogger()
                adapter = create_agent_runtime(
                    "strands", actor_id=constants.WHITE_SPY_ID, telemetry=telemetry, episode_id=episode_id,
                    provider_name="bedrock", model_name=MODEL_ID, execution_mode="evaluation",
                    llm_region=region, llm_timeout=60.0, personality_profile=None,
                    model_payload_transform=transform,
                )
                run_one_trial(adapter, snap.observation, telemetry, episode_id, wall_timeout=WALL_TIMEOUT)
                event = next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {})
                executed = event.get("executed_action") or {"action": "wait"}
                row = {
                    "snapshot_id": snap.snapshot_id, "category": snap.category, "condition": condition, "sample": i,
                    "runtime": "strands",
                    "executed_action": executed.get("action"), "executed_target": executed.get("target_id"),
                    "initial_model_output_valid": event.get("initial_model_output_valid"),
                    "final_model_action_valid": event.get("final_model_action_valid"),
                    "repair_attempted": event.get("repair_attempted"), "fallback_used": event.get("fallback_used"),
                    "latency_ms": event.get("latency_ms"), "input_tokens": event.get("input_tokens"),
                    "output_tokens": event.get("output_tokens"),
                }
                rows.append(row)
                print(f"[STRANDS] {snap.category} {condition} #{i}: {row['executed_action']}({row['executed_target']}) valid={row['final_model_action_valid']}")

    _write_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"), rows)

    per_snapshot_condition: dict[str, dict] = {}
    for snap in snapshots:
        for condition in COUNTERBALANCING_CONDITIONS:
            key = f"{snap.category}__{condition}"
            subset = [r for r in rows if r["snapshot_id"] == snap.snapshot_id and r["condition"] == condition]
            actions = Counter((r["executed_action"], r["executed_target"]) for r in subset)
            modal_action, modal_count = actions.most_common(1)[0] if actions else ((None, None), 0)
            n = len(subset) or 1
            per_snapshot_condition[key] = {
                "samples": len(subset),
                "action_distribution": {f"{a}({t})": c for (a, t), c in actions.items()},
                "modal_action": modal_action[0], "modal_target": modal_action[1], "modal_rate": modal_count / n,
                "unique_decisions": len(actions),
                "final_validity_rate": sum(1 for r in subset if r["final_model_action_valid"]) / n,
                "repair_rate": sum(1 for r in subset if r["repair_attempted"]) / n,
                "fallback_rate": sum(1 for r in subset if r["fallback_used"]) / n,
                "mean_latency_ms": sum(r["latency_ms"] or 0.0 for r in subset) / n,
            }

    result = {"region": region, "model_id": MODEL_ID, "categories": categories, "samples_per_condition": SAMPLES_PER_CONDITION,
              "per_snapshot_condition": per_snapshot_condition}
    with open(os.path.join(OUT_DIR, "strands_factorial_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--categories", type=str, nargs="+", required=True)
    args = parser.parse_args()

    if not _rp_manifest_clean():
        print("[FATAL] RP01-RP12 manifest has an unexplained FAIL - refusing to start Strands live calls.", file=sys.stderr)
        return 1

    from src.llm.clients.bedrock_client import resolve_bedrock_region
    region = resolve_bedrock_region(args.region)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1

    if not _preflight(OUT_DIR, region):
        print("[FATAL] Bedrock preflight failed.", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[OK] Preflight passed.")
        return 0

    start = datetime.now(timezone.utc)
    result = run(region, args.categories)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] Stage C complete ({start.isoformat()} - {end.isoformat()}). Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
