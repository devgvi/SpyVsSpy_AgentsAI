"""M6E.1 Strands replication (spec sections 30-32): run only after Native
analysis. Replicates S2's strong location + format effects (all 6
conditions - the one scenario with real effects to check for runtime-
independence) and S1 as the negative/stable control (D1 vs D2 only, since
Native found a clean null there - no need for the full 6-condition sweep
to confirm a null replicates). Gated on RP01-RP18 remaining clean and
stays entirely in ordinary/memory paths (spec section 32).

    python -m src.evaluation.run_m6e1_strands_replay --region us-east-1
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
from src.evaluation.duplicate_fact_variants import build_variant, multiset_fact_hash, representation_hash, unique_fact_hash
from src.evaluation.m6e1_scenarios import build_full_treatment_scenarios
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, run_bedrock_preflight
from src.llm_controller import validate_model_output

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m6e1-lossless-compaction-counterbalancing-v1")
SAMPLES_PER_CELL = 5


def _verify_sleep_assertion() -> bool:
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
        return "PreventUserIdleSystemSleep" in output and "caffeinate" in output
    except Exception:  # pragma: no cover
        return False


def _rp_manifest_clean() -> bool:
    from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST

    return all(e.status != FAIL for e in RUNTIME_PARITY_MANIFEST)


def _preflight(region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight_strands_replay", **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _run_one(region: str, family: str, scenario, condition: str, sample: int) -> dict:
    from src.llm.telemetry import TelemetryLogger

    variant = build_variant(condition, scenario.observation_payload, scenario.memory_payload, scenario.fact)
    payload = {**variant.observation_payload, "strategy_memory": variant.memory_payload}
    adapter = create_agent_runtime(
        "strands", actor_id="white", telemetry=TelemetryLogger(),
        episode_id=f"m6e1-strands-{family}-{condition}-{sample}", provider_name="bedrock", model_name=MODEL_ID,
        execution_mode="evaluation", llm_region=region, llm_timeout=60.0, personality_profile=None,
    )
    invocation = adapter._invoke_strands(payload, repair_instruction=None)
    raw = invocation.proposal if isinstance(invocation.proposal, dict) else {}
    v = validate_model_output(scenario.observation, raw, scenario.legal_actions)

    return {
        "runtime": "strands", "scenario_family": family, "scenario_id": scenario.scenario_id, "condition": condition, "sample": sample,
        "copy_count": variant.copy_count, "locations": list(variant.locations), "format_variant": variant.format_variant,
        "unique_fact_hash": unique_fact_hash(variant.observation_payload, variant.memory_payload, scenario.fact),
        "multiset_fact_hash": multiset_fact_hash(variant.copy_count, variant.locations, scenario.fact),
        "action": raw.get("action"), "target_id": raw.get("target_id"), "reason": raw.get("reason"),
        "validation_status": "valid" if v.valid else f"invalid:{v.error_type}",
        "latency_ms": invocation.latency_ms, "input_tokens": invocation.input_tokens, "output_tokens": invocation.output_tokens,
    }


def run(region: str) -> dict:
    scenarios = build_full_treatment_scenarios()
    plan = {
        "S2_KNOWN_LOCKED_OBJECT": ["D1_SINGLE_COPY", "D2_DOUBLE_COPY", "D3_TRIPLE_COPY", "L_OBS_AUX", "L_MEM_AUX", "R1_COMPACT_FORMAT"],
        "S1_RECENT_FAILED_ACTION": ["D1_SINGLE_COPY", "D2_DOUBLE_COPY"],
    }
    rows: list[dict] = []
    for family, conditions in plan.items():
        scenario = scenarios[family]
        for condition in conditions:
            for i in range(SAMPLES_PER_CELL):
                row = _run_one(region, family, scenario, condition, i)
                rows.append(row)
                print(f"[STRANDS {family}/{condition}] #{i}: valid={row['validation_status']} action={row['action']}({row['target_id']})")
    _write_jsonl(os.path.join(OUT_DIR, "strands_trials.jsonl"), rows)
    return {"rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not _rp_manifest_clean():
        print("[FATAL] RP01-RP18 manifest has an unexplained FAIL - refusing to start Strands live calls.", file=sys.stderr)
        return 1

    from src.llm.clients.bedrock_client import resolve_bedrock_region

    region = resolve_bedrock_region(args.region)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not _verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1
    if not _preflight(region):
        print("[FATAL] Bedrock preflight failed.", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[OK] Preflight passed.")
        return 0

    start = datetime.now(timezone.utc)
    result = run(region)
    end = datetime.now(timezone.utc)
    print(f"\n[DONE] M6E.1 Strands replication complete ({start.isoformat()} - {end.isoformat()}). {result}. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
