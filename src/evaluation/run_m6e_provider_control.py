"""M6E provider-variance control (spec sections 38-39): S3_BACKTRACK_REQUIRED
showed internal scatter even at C0 (4 different targets across 5 samples) -
the genuinely ambiguous case, not a clean condition-to-condition shift.
10 more byte-identical C0 calls to check whether that scatter is a stable
property of this scenario (real model uncertainty) or would collapse with
more samples.

    python -m src.evaluation.run_m6e_provider_control --region us-east-1
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
import sys

from src.evaluation.context_compaction import C0_FULL_CONTEXT
from src.evaluation.m6e_scenarios import build_all_scenarios
from src.evaluation.run_m6e_native_replay import OUT_DIR, _preflight, _run_one, _verify_sleep_assertion, _write_jsonl

TARGET_FAMILY = "S3_BACKTRACK_REQUIRED"
TARGET_CONDITION = C0_FULL_CONTEXT
EXTRA_SAMPLES = 10


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    args = parser.parse_args()

    from src.llm.clients.bedrock_client import resolve_bedrock_region

    region = resolve_bedrock_region(args.region)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not _verify_sleep_assertion():
        print("[FATAL] caffeinate not active.", file=sys.stderr)
        return 1
    if not _preflight(region):
        print("[FATAL] preflight failed.", file=sys.stderr)
        return 1

    scenario = build_all_scenarios()[TARGET_FAMILY]
    rows = []
    for i in range(5, 5 + EXTRA_SAMPLES):
        row = _run_one(region, TARGET_FAMILY, scenario, TARGET_CONDITION, i)
        row["provider_variance_control"] = True
        rows.append(row)
        print(f"[CONTROL {TARGET_FAMILY}/{TARGET_CONDITION}] #{i}: action={row['action']}({row['target_id']})")
    _write_jsonl(os.path.join(OUT_DIR, "provider_controls.jsonl"), rows)
    print(f"[DONE] {len(rows)} provider-variance control calls written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
