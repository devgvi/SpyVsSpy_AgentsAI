"""M6G Phases C and D (spec sections 25, 28): live Native replay, only
after Phase B's deterministic semantics are recorded.

Phase C - call order (S0 vs S1). Paired repetitions: every repetition runs
S0 and S1 back to back on the same frozen snapshot, alternating which
condition goes first, so wall-clock drift cannot line up with condition.
Afterwards every S0 trial's own decisions are replayed under the reversed
application order (S2) with zero extra model calls
(`native_apply_order_replays.jsonl`, spec sections 13, 23).

Phase D - snapshot timing (S0 vs S3 vs S4). Each repetition runs all three
in a rotating order; S0 is re-run here so the sequential conditions are
compared against a contemporaneous baseline.

Every completed trial is appended immediately; re-running resumes. Any
provider-side failure stops the run instead of being scored as a
decision.

    caffeinate -i python -m src.evaluation.run_m6g_native_replay --phase C
    caffeinate -i python -m src.evaluation.run_m6g_native_replay --phase D
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import sys
from datetime import datetime, timezone

from src.evaluation.conflict_scenarios import C0_NON_CONFLICTING, C2_SHARED_DOOR, C4_TRAP_CONFLICT, build_conflict_scenarios
from src.evaluation.decision_scheduling import (
    S0_FROZEN_SNAPSHOT_BASELINE,
    S1_REVERSED_CALL,
    S3_SEQUENTIAL_A_FIRST,
    S4_SEQUENTIAL_B_FIRST,
    replay_with_reversed_apply,
    run_condition,
)
from src.evaluation.m6g_live import (
    NativeDecider,
    append_jsonl,
    out_path,
    provider_failure,
    read_jsonl,
    run_live_preflight,
    verify_sleep_assertion,
)
from src.evaluation.run_m6g_analysis import MODEL_ID, PROVIDER_NAME
from src.evaluation.scheduling_analysis import decision_label
from src.llm.clients.bedrock_client import resolve_bedrock_region

REPS = 10
PHASE_C_SCENARIOS = (C2_SHARED_DOOR, C4_TRAP_CONFLICT, C0_NON_CONFLICTING)
PHASE_D_SCENARIOS = (C2_SHARED_DOOR, C0_NON_CONFLICTING, C4_TRAP_CONFLICT)
PHASE_FILES = {"C": "native_call_order_trials.jsonl", "D": "native_snapshot_timing_trials.jsonl"}


def phase_c_order(rep: int) -> tuple[str, ...]:
    return (S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL) if rep % 2 == 0 else (S1_REVERSED_CALL, S0_FROZEN_SNAPSHOT_BASELINE)


def phase_d_order(rep: int) -> tuple[str, ...]:
    base = (S0_FROZEN_SNAPSHOT_BASELINE, S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST)
    k = rep % 3
    return base[k:] + base[:k]


def planned_trials(phase: str, reps: int = REPS) -> list[tuple[int, str, int, str]]:
    """(rep, family, position, condition) in execution order."""
    plan = []
    for rep in range(reps):
        families = PHASE_C_SCENARIOS if phase == "C" else PHASE_D_SCENARIOS
        order = phase_c_order(rep) if phase == "C" else phase_d_order(rep)
        for family in families:
            for position, condition in enumerate(order):
                plan.append((rep, family, position, condition))
    return plan


def _preflight(region: str, phase: str) -> bool:
    result = run_live_preflight(MODEL_ID, region)
    append_jsonl(out_path("provider_events.jsonl"), {"phase": f"preflight_native_phase_{phase}", "at_utc": datetime.now(timezone.utc).isoformat(), **result})
    print(f"[PREFLIGHT] {json.dumps(result, default=str)}")
    return bool(result["ok"])


def run_phase(phase: str, decider, reps: int = REPS) -> int:
    scenarios = build_conflict_scenarios()
    path = out_path(PHASE_FILES[phase])
    done = {(r["rep"], r["scenario_family"], r["condition"]) for r in read_jsonl(path)}
    for rep, family, position, condition in planned_trials(phase, reps):
        if (rep, family, condition) in done:
            continue
        trial_id = f"{phase}:native:{family}:{condition}:r{rep:02d}"
        started = datetime.now(timezone.utc).isoformat()
        result = run_condition(condition, scenarios[family], decider, trial_id=trial_id, model_id=MODEL_ID, provider_name=PROVIDER_NAME)
        row = result.to_row(phase=phase, rep=rep, order_position=position, started_at_utc=started,
                            finished_at_utc=datetime.now(timezone.utc).isoformat())
        failure = provider_failure(row)
        if failure:
            append_jsonl(out_path("provider_events.jsonl"), {"phase": f"native_phase_{phase}_abort", "trial_id": trial_id, "failure": failure, "at_utc": started})
            print(f"[ABORT] provider failure in {trial_id}: {failure}", file=sys.stderr)
            return 2
        append_jsonl(path, row)
        recs = row["records"]
        print(f"[{phase} r{rep:02d} {family[:2]} {condition[:2]}] black={decision_label(recs['black_spy'])} white={decision_label(recs['white_spy'])} "
              f"same_snapshot={row['same_snapshot']} saw={row['second_actor_saw_first_action']}")
    return 0


def write_s2_replays() -> int:
    """Zero-call S2: each Phase C S0 trial's own decisions, applied in the
    reversed order on a rebuilt, hash-checked copy of the same snapshot."""
    scenarios = build_conflict_scenarios()
    rows = [r for r in read_jsonl(out_path(PHASE_FILES["C"])) if r["condition"] == S0_FROZEN_SNAPSHOT_BASELINE]
    path = out_path("native_apply_order_replays.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: (x["scenario_family"], x["rep"])):
            replay = replay_with_reversed_apply(r, scenarios[r["scenario_family"]], trial_id=f"C:native:{r['scenario_family']}:S2_replay:r{r['rep']:02d}",
                                                model_id=MODEL_ID, provider_name=PROVIDER_NAME)
            f.write(json.dumps(replay.to_row(phase="C_S2_REPLAY", rep=r["rep"]), sort_keys=True, default=str) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("C", "D"), required=True)
    parser.add_argument("--region", default=None)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replays-only", action="store_true", help="rebuild the zero-call S2 replays from existing Phase C rows")
    args = parser.parse_args()

    if args.replays_only:
        print(f"[S2] replayed {write_s2_replays()} S0 trials")
        return 0
    region = resolve_bedrock_region(args.region)
    if not verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1
    if not _preflight(region, args.phase):
        print("[FATAL] Bedrock preflight failed.", file=sys.stderr)
        return 1
    if args.preflight_only:
        return 0
    start = datetime.now(timezone.utc)
    status = run_phase(args.phase, NativeDecider(region, MODEL_ID), args.reps)
    if status == 0 and args.phase == "C":
        print(f"[S2] replayed {write_s2_replays()} S0 trials")
    print(f"[DONE] phase {args.phase} status={status} ({start.isoformat()} - {datetime.now(timezone.utc).isoformat()})")
    return status


if __name__ == "__main__":
    sys.exit(main())
