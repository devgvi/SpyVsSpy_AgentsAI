"""M6B.2 Phase E (spec sections 32-34): a focused LIVE confirmation that
the new target-legality repair hint (RP12_LEGAL_TARGET_HINTS) actually
changes model behavior against the exact never-inspected-target shape the
M6B.1 outlier episode (seed 9605) hit - not a full episode rerun (spec
section 35's optional episode confirmation is separate and smaller), just
repeated single-decision trials against 2 fixed snapshots:

- UNINSPECTED_TRAPPABLE_NEARBY: PLACE_TRAP is plausible (actor holds traps,
  a trappable object is right there) but currently illegal (never
  inspected) - the exact outlier shape.
- INSPECTED_TRAPPABLE_NEARBY: the legal control - same object, now
  inspected, so PLACE_TRAP should succeed on first attempt or after one
  repair.

Success criterion (spec section 34): the repeated never-inspected-target
failure pattern no longer DOMINATES - not zero model mistakes.

    python -m src.evaluation.run_m6b2_phase_e --region us-east-1
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

from src import constants
from src.evaluation.runtime_snapshot_replay import build_native_adapter, build_strands_adapter, run_one_trial
from src.evaluation.target_legality_snapshot_set import (
    INSPECTED_TRAPPABLE_NEARBY,
    UNINSPECTED_TRAPPABLE_NEARBY,
    build_target_legality_snapshot_set,
)
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm.telemetry import TelemetryLogger

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m6b2-harness-parity-target-legality-v1")
SAMPLES_PER_RUNTIME = 10  # spec section 32's stated minimum
WALL_TIMEOUT = 60.0
FOCUS_CATEGORIES = (UNINSPECTED_TRAPPABLE_NEARBY, INSPECTED_TRAPPABLE_NEARBY)


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
        f.write(json.dumps({"phase": "preflight_phase_e", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _classify_trial(event: dict) -> dict:
    ip = event.get("initial_proposal") or {}
    place_trap_selected = ip.get("action") == "place_trap"
    repair_attempted = bool(event.get("repair_attempted"))
    repair_response = event.get("repair_raw_response") or {}
    final_valid = bool(event.get("final_model_action_valid"))
    final_error_type = event.get("error_type")
    initial_error_type = event.get("initial_error_type")
    repair_error_type = event.get("repair_error_type")

    initial_target = ip.get("target_id")
    repair_target = repair_response.get("target_id")
    same_target_repeated_on_repair = bool(
        repair_attempted and initial_target is not None and repair_target == initial_target and not final_valid
    )

    return {
        "place_trap_selected": place_trap_selected,
        "legal_target_selected": place_trap_selected and final_valid,
        "illegal_target_selected": place_trap_selected and final_error_type in ("target_not_inspected", "target_not_trappable", "illegal_target", "out_of_range", "unknown_target"),
        "repair_attempted": repair_attempted,
        "repair_success": repair_attempted and final_valid,
        "same_invalid_target_repeated": same_target_repeated_on_repair,
        "fallback": bool(event.get("fallback_used")),
        "initial_error_type": initial_error_type,
        "repair_error_type": repair_error_type,
        "same_invalid_action_signature_count": event.get("same_invalid_action_signature_count"),
    }


def run(region: str, out_dir: str) -> dict:
    snapshots = {s.category: s for s in build_target_legality_snapshot_set() if s.category in FOCUS_CATEGORIES}
    rows: list[dict] = []
    metrics: dict[str, dict] = {}

    for category, snapshot in snapshots.items():
        for runtime_label in ("native_tool_like", "strands"):
            trials = []
            for i in range(SAMPLES_PER_RUNTIME):
                episode_id = f"m6b2-{runtime_label}-{snapshot.snapshot_id}-{i}"
                telemetry = TelemetryLogger()
                if runtime_label == "native_tool_like":
                    adapter = build_native_adapter(
                        actor_id=constants.WHITE_SPY_ID, model_client=__import__(
                            "src.llm.clients.bedrock_client", fromlist=["BedrockClient"]
                        ).BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0),
                        telemetry=telemetry, episode_id=episode_id, provider_name="bedrock", model_name=MODEL_ID,
                        personality_profile=None,
                    )
                else:
                    adapter = build_strands_adapter(
                        actor_id=constants.WHITE_SPY_ID, telemetry=telemetry, episode_id=episode_id,
                        provider_name="bedrock", model_name=MODEL_ID, region=region, timeout=60.0,
                        personality_profile=None,
                    )
                run_one_trial(adapter, snapshot.observation, telemetry, episode_id, wall_timeout=WALL_TIMEOUT)
                event = next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {})
                classification = _classify_trial(event)
                row = {"category": category, "runtime": runtime_label, "sample": i, **classification}
                rows.append(row)
                trials.append(classification)
                print(f"[{runtime_label}] {category} #{i}: {classification}")

            n = len(trials)
            metrics[f"{category}__{runtime_label}"] = {
                "trials": n,
                "place_trap_selected": sum(t["place_trap_selected"] for t in trials),
                "legal_target_selected": sum(t["legal_target_selected"] for t in trials),
                "illegal_target_selected": sum(t["illegal_target_selected"] for t in trials),
                "repair_attempted": sum(t["repair_attempted"] for t in trials),
                "repair_success": sum(t["repair_success"] for t in trials),
                "same_invalid_target_repeated": sum(t["same_invalid_target_repeated"] for t in trials),
                "fallback": sum(t["fallback"] for t in trials),
            }

    _write_jsonl(os.path.join(out_dir, "focused_live_trials.jsonl"), rows)
    result = {"region": region, "model_id": MODEL_ID, "samples_per_runtime": SAMPLES_PER_RUNTIME, "metrics": metrics}
    with open(os.path.join(out_dir, "target_legality_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    from src.llm.clients.bedrock_client import resolve_bedrock_region
    region = resolve_bedrock_region(args.region)

    out_dir = BASE_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1

    if not _preflight(out_dir, region):
        print("[FATAL] Bedrock preflight failed.", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[OK] Preflight passed.")
        return 0

    start = datetime.now(timezone.utc)
    result = run(region, out_dir)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] Phase E complete ({start.isoformat()} - {end.isoformat()}). Output: {out_dir}")
    print(json.dumps(result["metrics"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
