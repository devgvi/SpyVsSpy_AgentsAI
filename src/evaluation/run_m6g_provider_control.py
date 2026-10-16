"""M6G spec section 27: provider-variance control for S0/S1 differences,
using M6A.1's method - the exact same request, sent N times in isolation.

For each triggered (scenario, actor) cell the actor's frozen-snapshot
request is rebuilt, its provider-facing hash is checked against the hash
Phase C actually sent (refusing to run on a mismatch), and it is sent N
times with no partner call at all - a fresh client per call. Whatever
spread this shows is intrinsic model/provider variance at a fixed request,
which is what an S0/S1 difference has to exceed before it can be read as
a call-order effect.

Trigger rule (predeclared in call_plan.json): an actor's modal decision
differs between S0 and S1, or TVD(S0, S1) >= 0.3.

    caffeinate -i python -m src.evaluation.run_m6g_provider_control --auto
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import sys
from datetime import datetime, timezone

from src.evaluation.conflict_scenarios import build_conflict_scenarios
from src.evaluation.decision_scheduling import MODEL_OUTPUT_ERROR_TYPES, S0_FROZEN_SNAPSHOT_BASELINE, _decision_key, build_request, native_render
from src.evaluation.duel_snapshot import build_duel_snapshot
from src.evaluation.m6g_live import NativeDecider, append_jsonl, out_path, read_jsonl, run_live_preflight, verify_sleep_assertion
from src.evaluation.run_m6g_analysis import MODEL_ID, PROVIDER_NAME, provider_control_triggers
from src.evaluation.runtime_variance_analysis import content_hash
from src.evaluation.scheduling_analysis import decision_label
from src.llm.clients.bedrock_client import resolve_bedrock_region
from src.llm_controller import validate_model_output

N_REPEATS = 10
MAX_TRIGGERED_CELLS = 4


def run_control(family: str, actor_id: str, decider: NativeDecider, n: int, trigger: str) -> int:
    phase_c = [r for r in read_jsonl(out_path("native_call_order_trials.jsonl")) if r["scenario_family"] == family]
    sent_hashes = {r["records"][actor_id]["provider_request_hash"] for r in phase_c}
    scenario = build_conflict_scenarios()[family]
    snap = build_duel_snapshot(scenario.setup_fn)
    request = build_request(actor_id, snap.observations[actor_id], snap.legal_actions[actor_id], scenario_family=family,
                            model_id=MODEL_ID, provider_name=PROVIDER_NAME)
    system_prompt, user_prompt = native_render(request)
    provider_hash = content_hash(system_prompt, user_prompt)
    if sent_hashes != {provider_hash}:
        print(f"[FATAL] rebuilt request for {family}/{actor_id} does not match what Phase C sent", file=sys.stderr)
        return 1
    done = {r["sample"] for r in read_jsonl(out_path("provider_controls.jsonl")) if r["scenario_family"] == family and r["actor_id"] == actor_id}
    for sample in range(n):
        if sample in done:
            continue
        out = decider.call(request, system_prompt, user_prompt)
        err = out.get("error_type")
        if err is not None and err not in MODEL_OUTPUT_ERROR_TYPES:
            print(f"[ABORT] provider failure: {err}: {out.get('error_message')}", file=sys.stderr)
            return 2
        raw = out["raw"]
        v = validate_model_output(snap.observations[actor_id], raw, snap.legal_actions[actor_id])
        decision = _decision_key(v.validated_action if v.valid else None)
        row = {
            "phase": "C_PROVIDER_CONTROL", "provider_variance_control": True, "trigger": trigger,
            "scenario_family": family, "actor_id": actor_id, "sample": sample, "condition_source": S0_FROZEN_SNAPSHOT_BASELINE,
            "agent_request_semantic_hash": request.request_semantic_hash(), "provider_request_hash": provider_hash,
            "raw_model_output": raw, "first_pass_valid": v.valid, "validation_error": v.error_type, "decision": decision,
            "decision_label": decision_label({"decision": decision}), "latency_ms": out.get("latency_ms"),
            "input_tokens": out.get("input_tokens"), "output_tokens": out.get("output_tokens"), "provider_error_type": err,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(out_path("provider_controls.jsonl"), row)
        print(f"[CONTROL {family[:2]} {actor_id} #{sample}] {row['decision_label']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="run every cell the predeclared trigger selects")
    parser.add_argument("--family")
    parser.add_argument("--actor")
    parser.add_argument("--n", type=int, default=N_REPEATS)
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    if args.auto:
        cells = provider_control_triggers(read_jsonl(out_path("native_call_order_trials.jsonl")))
        trigger = "auto: modal differs or TVD>=0.3"
    elif args.family and args.actor:
        cells, trigger = [(args.family, args.actor)], "manual"
    else:
        parser.error("use --auto or --family/--actor")
    print(f"[PLAN] triggered cells: {cells}")
    if not cells:
        return 0
    if len(cells) > MAX_TRIGGERED_CELLS:
        print(f"[FATAL] {len(cells)} triggered cells exceeds the predeclared budget of {MAX_TRIGGERED_CELLS}", file=sys.stderr)
        return 1
    region = resolve_bedrock_region(args.region)
    if not verify_sleep_assertion():
        print("[FATAL] Sleep prevention (caffeinate) not active - refusing to start a live run.", file=sys.stderr)
        return 1
    preflight = run_live_preflight(MODEL_ID, region)
    append_jsonl(out_path("provider_events.jsonl"), {"phase": "preflight_provider_control", "at_utc": datetime.now(timezone.utc).isoformat(), **preflight})
    print(f"[PREFLIGHT] {json.dumps(preflight, default=str)}")
    if not preflight["ok"]:
        return 1
    decider = NativeDecider(region, MODEL_ID)
    for family, actor_id in cells:
        status = run_control(family, actor_id, decider, args.n, trigger)
        if status:
            return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
