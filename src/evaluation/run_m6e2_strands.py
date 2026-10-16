"""M6E.2 Strands replication (spec sections 36-38): after Native analysis,
replicate only the cells needed for the three required contrasts (spec
section 37) - not the full 8-cell cube. Reuses M6E.1's own Strands data
for S2's D1/D2/R1/L_MEM_AUX (identical FactSpec, identical conditions);
runs 4 new S2 cells (I1 plus the 3 M6E.2-only conditions) and 2 new S0
cells live.

    python -m src.evaluation.run_m6e2_strands --reuse-only   # just the reuse step
    python -m src.evaluation.run_m6e2_strands --region us-east-1  # reuse + new live cells
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

from src.agents.registry import create_agent_runtime
from src.evaluation.duplicate_fact_variants import multiset_fact_hash, representation_hash, unique_fact_hash
from src.evaluation.m6e2_factorial import (
    CONDITION_TO_CELL,
    FACTORIAL_CELLS,
    build_cell,
    build_primary_scenarios,
    duplicate_location_hash,
    format_hash,
)
from src.llm_controller import validate_model_output

A0B1C0_SINGLE_AUX_FULL = "A0B1C0_SINGLE_AUX_FULL"

OUT_DIR = os.path.join("evaluation_runs", "m6e2-representation-interaction-matrix-v1")
M6E1_STRANDS_PATH = os.path.join("evaluation_runs", "m6e1-lossless-compaction-counterbalancing-v1", "strands_trials.jsonl")
SAMPLES_PER_CELL = 5

REUSABLE_STRANDS = {("S2_KNOWN_LOCKED_OBJECT", "D1_SINGLE_COPY"), ("S2_KNOWN_LOCKED_OBJECT", "D2_DOUBLE_COPY"),
                    ("S2_KNOWN_LOCKED_OBJECT", "R1_COMPACT_FORMAT"), ("S2_KNOWN_LOCKED_OBJECT", "L_MEM_AUX")}

# New cells needed for the 3 required contrasts (spec section 37):
# strongest location contrast is already covered by the reused D2 vs L_MEM_AUX pair.
NEW_LIVE_BY_FAMILY = {
    "S2_KNOWN_LOCKED_OBJECT": [c for c, cond in FACTORIAL_CELLS.items() if cond in ("I1_SINGLE_COPY_COMPACT", A0B1C0_SINGLE_AUX_FULL, "A0B1C1_SINGLE_AUX_COMPACT", "A1B1C1_DOUBLE_AUX_COMPACT")],
    "S0_NO_HISTORY_RELEVANT": [c for c, cond in FACTORIAL_CELLS.items() if cond in ("D1_SINGLE_COPY", A0B1C0_SINGLE_AUX_FULL)],
}


def reuse() -> dict:
    rows = []
    with open(M6E1_STRANDS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    reused = []
    for r in rows:
        if (r["scenario_family"], r["condition"]) in REUSABLE_STRANDS:
            tagged = dict(r)
            tagged["cell"] = list(CONDITION_TO_CELL[r["condition"]])
            tagged["reused_from"] = "m6e1-lossless-compaction-counterbalancing-v1"
            reused.append(tagged)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "strands_trials.jsonl"), "a", encoding="utf-8") as f:
        for r in reused:
            f.write(json.dumps(r, default=str) + "\n")
    return {"reused_rows": len(reused)}


def _verify_sleep_assertion() -> bool:
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
        return "PreventUserIdleSystemSleep" in output and "caffeinate" in output
    except Exception:  # pragma: no cover
        return False


def _rp_manifest_clean() -> bool:
    from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST

    return all(e.status != FAIL for e in RUNTIME_PARITY_MANIFEST)


def _preflight(region: str, model_id: str) -> bool:
    from src.llm.clients.bedrock_client import run_bedrock_preflight

    preflight = run_bedrock_preflight(model_id=model_id, region=region, timeout=60.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight_strands_replay", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _run_one(region: str, model_id: str, family: str, scenario, cell: tuple, sample: int) -> dict:
    from src.llm.telemetry import TelemetryLogger

    variant = build_cell(cell, scenario)
    payload = {**variant.observation_payload, "strategy_memory": variant.memory_payload}
    adapter = create_agent_runtime(
        "strands", actor_id="white", telemetry=TelemetryLogger(),
        episode_id=f"m6e2-strands-{family}-{FACTORIAL_CELLS[cell]}-{sample}", provider_name="bedrock", model_name=model_id,
        execution_mode="evaluation", llm_region=region, llm_timeout=60.0, personality_profile=None,
    )
    invocation = adapter._invoke_strands(payload, repair_instruction=None)
    raw = invocation.proposal if isinstance(invocation.proposal, dict) else {}
    v = validate_model_output(scenario.observation, raw, scenario.legal_actions)

    return {
        "runtime": "strands", "scenario_family": family, "scenario_id": scenario.scenario_id,
        "condition": FACTORIAL_CELLS[cell], "cell": list(cell), "sample": sample,
        "copy_count": variant.copy_count, "locations": list(variant.locations), "format_variant": variant.format_variant,
        "unique_fact_hash": unique_fact_hash(variant.observation_payload, variant.memory_payload, scenario.fact),
        "multiset_fact_hash": multiset_fact_hash(variant.copy_count, variant.locations, scenario.fact),
        "duplicate_location_hash": duplicate_location_hash(variant.locations), "format_hash": format_hash(variant.format_variant),
        "action": raw.get("action"), "target_id": raw.get("target_id"), "reason": raw.get("reason"),
        "validation_status": "valid" if v.valid else f"invalid:{v.error_type}",
        "latency_ms": invocation.latency_ms, "input_tokens": invocation.input_tokens, "output_tokens": invocation.output_tokens,
    }


def run_live(region: str, model_id: str) -> dict:
    scenarios = build_primary_scenarios()
    rows = []
    for family, cells in NEW_LIVE_BY_FAMILY.items():
        scenario = scenarios[family]
        for cell in cells:
            for i in range(SAMPLES_PER_CELL):
                row = _run_one(region, model_id, family, scenario, cell, i)
                rows.append(row)
                print(f"[STRANDS {family}/{FACTORIAL_CELLS[cell]}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']})")
    with open(os.path.join(OUT_DIR, "strands_trials.jsonl"), "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return {"rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.reuse_only:
        print(f"[REUSE] {reuse()}")
        return 0

    if not _rp_manifest_clean():
        print("[FATAL] RP01-RP18 manifest has an unexplained FAIL - refusing to start Strands live calls.", file=sys.stderr)
        return 1

    from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, resolve_bedrock_region

    model_id = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
    region = resolve_bedrock_region(args.region)

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1
    if not _preflight(region, model_id):
        print("[FATAL] Bedrock preflight failed.", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[OK] Preflight passed.")
        return 0

    start = datetime.now(timezone.utc)
    result = run_live(region, model_id)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] M6E.2 Strands replication complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
