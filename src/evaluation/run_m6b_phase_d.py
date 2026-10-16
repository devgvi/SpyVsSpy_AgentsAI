"""M6B Phase D (spec sections 30-34, 44): live confirmation that the
repair-hint fix (`legal_trap_types` now passed to Strands' own repair
instruction, mirroring Native's M5F.3B.1 fix) actually restores PLACE_TRAP
reliability, against the frozen trap-opportunity snapshot set. Narrower
than the spec's full S0-S5 schema-variant matrix by design (spec section
30: "control cost by narrowing variants after Phase A/B if obvious
failures are found") - Phase A's retrospective analysis already
identified the dominant mechanism (a missing repair hint, not a schema-
definition or tool/field-order difference - see
`place_trap_contract_audit.md`), so this confirms the fix directly
rather than re-running the full variant matrix.

Strands gets the primary sample (5/snapshot, spec section 30's own
recommendation); Native gets a smaller confirmatory control (3/snapshot)
since its baseline is already well-established across M6A/M6A.1/M6A.2.

    python -m src.evaluation.run_m6b_phase_d --region us-east-1
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
from src.evaluation.multiseed_stability import resolve_output_dir
from src.evaluation.place_trap_reliability_analysis import classify_place_trap_event, summarize_place_trap_attempts
from src.evaluation.runtime_snapshot_replay import build_native_adapter, build_strands_adapter, run_one_trial
from src.evaluation.trap_snapshot_set import build_trap_snapshot_set
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm.telemetry import TelemetryLogger

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m6b-tool-schema-argument-reliability-v1")
STRANDS_SAMPLES = 5
NATIVE_SAMPLES = 3
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
        f.write(json.dumps({"phase": "preflight_phase_d", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def run(region: str, out_dir: str) -> dict:
    snapshots = build_trap_snapshot_set()
    native_events: list[dict] = []
    strands_events: list[dict] = []
    rows: list[dict] = []

    for snapshot in snapshots:
        for i in range(NATIVE_SAMPLES):
            episode_id = f"m6b-native-{snapshot.snapshot_id}-{i}"
            telemetry = TelemetryLogger()
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            adapter = build_native_adapter(
                actor_id=constants.WHITE_SPY_ID, model_client=client, telemetry=telemetry, episode_id=episode_id,
                provider_name="bedrock", model_name=MODEL_ID, personality_profile=None,
            )
            run_one_trial(adapter, snapshot.observation, telemetry, episode_id, wall_timeout=WALL_TIMEOUT)
            event = next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {})
            event["snapshot_id"] = snapshot.snapshot_id
            event["runtime_label"] = "native_tool_like"
            native_events.append(event)
            classification = classify_place_trap_event(event)
            rows.append({"snapshot_id": snapshot.snapshot_id, "runtime": "native_tool_like", "sample": i, "classification": classification})
            print(f"[NATIVE] {snapshot.category} #{i}: {classification}")

        for i in range(STRANDS_SAMPLES):
            episode_id = f"m6b-strands-{snapshot.snapshot_id}-{i}"
            telemetry = TelemetryLogger()
            adapter = build_strands_adapter(
                actor_id=constants.WHITE_SPY_ID, telemetry=telemetry, episode_id=episode_id,
                provider_name="bedrock", model_name=MODEL_ID, region=region, timeout=60.0, personality_profile=None,
            )
            run_one_trial(adapter, snapshot.observation, telemetry, episode_id, wall_timeout=WALL_TIMEOUT)
            event = next((e for e in telemetry.events if e.get("event_type") == "llm_decision"), {})
            event["snapshot_id"] = snapshot.snapshot_id
            event["runtime_label"] = "strands"
            strands_events.append(event)
            classification = classify_place_trap_event(event)
            rows.append({"snapshot_id": snapshot.snapshot_id, "runtime": "strands", "sample": i, "classification": classification})
            print(f"[STRANDS] {snapshot.category} #{i}: {classification}")

    _write_jsonl(os.path.join(out_dir, "schema_trials.jsonl"), rows)

    native_summary = summarize_place_trap_attempts(native_events)
    strands_summary = summarize_place_trap_attempts(strands_events)

    per_snapshot: dict[str, dict] = {}
    for snapshot in snapshots:
        n_events = [e for e in native_events if e["snapshot_id"] == snapshot.snapshot_id]
        s_events = [e for e in strands_events if e["snapshot_id"] == snapshot.snapshot_id]
        per_snapshot[snapshot.snapshot_id] = {
            "category": snapshot.category,
            "native_tool_like": summarize_place_trap_attempts(n_events),
            "strands": summarize_place_trap_attempts(s_events),
        }

    result = {
        "region": region, "model_id": MODEL_ID,
        "native_samples_per_snapshot": NATIVE_SAMPLES, "strands_samples_per_snapshot": STRANDS_SAMPLES,
        "overall": {"native_tool_like": native_summary, "strands": strands_summary},
        "per_snapshot": per_snapshot,
    }
    with open(os.path.join(out_dir, "tool_reliability_metrics.json"), "w", encoding="utf-8") as f:
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
    print(f"\n[DONE] Phase D complete ({start.isoformat()} - {end.isoformat()}). Output: {out_dir}")
    print(json.dumps(result["overall"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
