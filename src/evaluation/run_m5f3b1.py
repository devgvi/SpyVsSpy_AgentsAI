"""M5F.3B.1: Stage 2 replication of M5F.3B's free social strategy on a
second deterministic seed, after hardening invalid trap-type handling.

    python -m src.evaluation.run_m5f3b1 --stage2                 # all 3 matchups
    python -m src.evaluation.run_m5f3b1 --stage2 --matchup 0     # incremental staging (0/1/2)

Reuses M5F.3B's own `MATCHUPS`, `generate_mirror_pairs`, and `run_pairs`
(unchanged - spec section 3 requires exactly the same matchups/mirror
logic) - only the seed and output directory differ. Phase A (network-free
validation/repair/counter tests) is covered by
tests/test_m5f3b1_invalid_action_robustness.py, not this driver.
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

from src.evaluation.free_social_matchup import MATCHUPS
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.personality_metrics import generate_mirror_pairs
from src.evaluation.run_m5f3b import MODEL_ID, run_pairs
from src.llm.clients.bedrock_client import bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight

BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3b-free-social-strategy-v1")
# Spec section 23: fixed BEFORE any live execution, never chosen after
# inspecting behavior. Deliberately distinct from M5F.3B Stage 1's 7301.
STAGE2_SEED = 9187


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


def stage2(out_dir: str, region: str, matchup_index: int | None) -> dict:
    matchups = MATCHUPS if matchup_index is None else [MATCHUPS[matchup_index]]
    print(f"\n=== M5F.3B.1 Stage 2 paired benchmark ({'all 3 matchups' if matchup_index is None else f'matchup {matchup_index} only'} x 2 mirrors x 2 conditions, seed={STAGE2_SEED}) ===")
    pairs = generate_mirror_pairs(matchups, [STAGE2_SEED])
    results = run_pairs(out_dir, region, pairs)
    return {"pairs": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", action="store_true", required=True)
    parser.add_argument("--matchup", type=int, default=None, choices=(0, 1, 2), help="Run just this one matchup index for incremental live staging")
    args = parser.parse_args(argv)

    print(f"=== M5F.3B.1 free social replication (stage2) starting {datetime.now(timezone.utc).isoformat()} ===")

    if not bedrock_token_present():
        print("[PREFLIGHT] FAILED: AWS_BEARER_TOKEN_BEDROCK is not set. Refusing to start any episode.")
        return 1
    try:
        region = resolve_bedrock_region()
    except Exception as exc:
        print(f"[PREFLIGHT] FAILED: {exc}")
        return 1

    sleep_assertion_verified = _verify_sleep_assertion()
    print(f"[SLEEP] assertion_verified={sleep_assertion_verified}")

    base_out_dir = os.path.join(BASE_OUT_DIR, "stage2")
    if args.matchup is not None:
        out_dir = base_out_dir
        print(f"[STAGED] matchup={args.matchup} - writing into the shared {out_dir}/ (not treated as a rerun)")
    else:
        out_dir = resolve_output_dir(base_out_dir)
        if out_dir != base_out_dir:
            print(f"[RERUN] {base_out_dir} already has a completed run - preserving it untouched; this run writes to {out_dir}/")
    os.makedirs(out_dir, exist_ok=True)

    if not _preflight(out_dir, region):
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    result = stage2(out_dir, region, args.matchup)
    run_end = datetime.now(timezone.utc)

    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_summary = detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover
        sleep_summary = {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}

    summary = {
        "phase": "stage2", "seed": STAGE2_SEED, "pairs_run": len(result["pairs"]),
        "sleep_prevention_active": True, "sleep_assertion_verified": sleep_assertion_verified, "sleep_contamination": sleep_summary,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment_id": "m5f3b1-free-social-replication-v1", "phase": "stage2", "model_id": MODEL_ID, "seed": STAGE2_SEED, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3B.1 stage2 finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
