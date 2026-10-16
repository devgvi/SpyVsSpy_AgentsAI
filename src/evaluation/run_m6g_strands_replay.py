"""M6G spec sections 34-36, 43: Strands replication of the strongest Native
findings, run only after the Native analysis.

Scope: C2 (shared door) only, under S0/S1 (call order) and S0/S3/S4
(snapshot timing). C4 is deliberately not replicated here: its decision is
PLACE_TRAP with a trap_type argument, the tool-argument path M6B/M6F
already showed to be runtime-sensitive on Strands (spec section 36).

Runtime parity gate (spec section 35): refuses to start unless the RP01-
RP18 manifest has no FAIL entry AND every test file the manifest's
test_ids point at passes right now.

    caffeinate -i python -m src.evaluation.run_m6g_strands_replay
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

from src.evaluation.conflict_scenarios import C2_SHARED_DOOR, build_conflict_scenarios
from src.evaluation.decision_scheduling import S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL, S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST, run_condition
from src.evaluation.m6g_live import (
    StrandsDecider,
    append_jsonl,
    out_path,
    provider_failure,
    read_jsonl,
    rp_manifest_clean,
    run_live_preflight,
    verify_sleep_assertion,
)
from src.evaluation.run_m6g_analysis import MODEL_ID, PROVIDER_NAME
from src.evaluation.scheduling_analysis import decision_label
from src.llm.clients.bedrock_client import resolve_bedrock_region

REPS = 10
FAMILY = C2_SHARED_DOOR
CONDITIONS = (S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL, S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST)
RP_TEST_FILES = (
    "tests/test_m6b_harness_invariants.py",
    "tests/test_m6b1_runtime_parity.py",
    "tests/test_m6b_place_trap_reliability.py",
    "tests/test_m5f3b1_invalid_action_robustness.py",
    "tests/test_strands_agent.py",
    "tests/test_m6b2_target_legality_and_payload_parity.py",
    "tests/test_m6f_runtime_interoperability.py",
)


def condition_order(rep: int) -> tuple[str, ...]:
    k = rep % len(CONDITIONS)
    return CONDITIONS[k:] + CONDITIONS[:k]


def runtime_parity_gate() -> dict:
    manifest_clean = rp_manifest_clean()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *RP_TEST_FILES],
                          capture_output=True, text=True, timeout=1800)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    return {"manifest_clean": manifest_clean, "tests_exit_code": proc.returncode, "tests_passed": passed, "tests_failed": failed,
            "summary_line": tail, "ok": manifest_clean and proc.returncode == 0 and failed == 0 and passed > 0}


def run(decider, reps: int = REPS) -> int:
    scenario = build_conflict_scenarios()[FAMILY]
    path = out_path("strands_trials.jsonl")
    done = {(r["rep"], r["condition"]) for r in read_jsonl(path)}
    for rep in range(reps):
        for position, condition in enumerate(condition_order(rep)):
            if (rep, condition) in done:
                continue
            trial_id = f"E:strands:{FAMILY}:{condition}:r{rep:02d}"
            started = datetime.now(timezone.utc).isoformat()
            result = run_condition(condition, scenario, decider, trial_id=trial_id, model_id=MODEL_ID, provider_name=PROVIDER_NAME)
            row = result.to_row(phase="E", rep=rep, order_position=position, started_at_utc=started,
                                finished_at_utc=datetime.now(timezone.utc).isoformat())
            failure = provider_failure(row)
            if failure:
                append_jsonl(out_path("provider_events.jsonl"), {"phase": "strands_abort", "trial_id": trial_id, "failure": failure, "at_utc": started})
                print(f"[ABORT] provider failure in {trial_id}: {failure}", file=sys.stderr)
                return 2
            append_jsonl(path, row)
            recs = row["records"]
            print(f"[E r{rep:02d} {condition[:2]}] black={decision_label(recs['black_spy'])} white={decision_label(recs['white_spy'])} "
                  f"same_snapshot={row['same_snapshot']} saw={row['second_actor_saw_first_action']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    gate = runtime_parity_gate()
    append_jsonl(out_path("provider_events.jsonl"), {"phase": "runtime_parity_gate_rp01_rp18", "at_utc": datetime.now(timezone.utc).isoformat(), **gate})
    print(f"[RP GATE] {json.dumps(gate)}")
    if not gate["ok"]:
        print("[FATAL] RP01-RP18 gate not clean - refusing to start Strands live calls.", file=sys.stderr)
        return 1
    region = resolve_bedrock_region(args.region)
    if not verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1
    preflight = run_live_preflight(MODEL_ID, region)
    append_jsonl(out_path("provider_events.jsonl"), {"phase": "preflight_strands", "at_utc": datetime.now(timezone.utc).isoformat(), **preflight})
    print(f"[PREFLIGHT] {json.dumps(preflight, default=str)}")
    if not preflight["ok"] or args.preflight_only:
        return 0 if preflight["ok"] else 1
    start = datetime.now(timezone.utc)
    status = run(StrandsDecider(region, MODEL_ID), args.reps)
    print(f"[DONE] strands status={status} ({start.isoformat()} - {datetime.now(timezone.utc).isoformat()})")
    return status


if __name__ == "__main__":
    sys.exit(main())
