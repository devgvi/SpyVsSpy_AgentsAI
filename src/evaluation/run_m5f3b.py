"""M5F.3B: automated live-diagnostic entry point for free cooperation,
sabotage & coercion - paired SOCIAL_OFF/SOCIAL_ON full two-agent duels.

    python -m src.evaluation.run_m5f3b --pilot    # Phase B: 2 mirrored SOCIAL_ON duels
    python -m src.evaluation.run_m5f3b --stage1   # Phase C: 12-duel paired matrix (3 matchups x 2 mirrors x 2 conditions)

Phase A (network-free) is covered by tests/test_m5f3b_free_social.py -
not this driver. Freezes provider=Bedrock, native_tool_like runtime, the
M5F.3B spec's own frozen matchups/personalities/environmental trap fixture.
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
from src.evaluation.free_social_matchup import MATCHUPS, run_paired_duel
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.personality_metrics import generate_mirror_pairs
from src.llm.clients.bedrock_client import DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3b-free-social-strategy-v1")
STAGE1_SEED = 7301
MESSAGE_COOLDOWN = 3


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


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def run_pairs(out_dir: str, region: str, pairs: list[dict]) -> list[dict]:
    telemetry_path = os.path.join(out_dir, "duel_decisions.jsonl")
    results = []
    for i, pair in enumerate(pairs, start=1):
        print(f"\n=== PAIR {i}/{len(pairs)}: seed={pair['seed']} black={pair['black_personality']} white={pair['white_personality']} mirror={pair['mirror']} ===")
        result = run_paired_duel(
            seed=pair["seed"], black_personality=pair["black_personality"], white_personality=pair["white_personality"],
            mirror=pair["mirror"], provider="bedrock", model=MODEL_ID, region=region, agent_runtime="native_tool_like",
            telemetry_path=telemetry_path, max_decisions_per_actor=40, message_cooldown_decisions=MESSAGE_COOLDOWN, llm_timeout=60.0,
        )
        off, on = result["social_off"], result["social_on"]
        print(f"[OFF] status={off['status']} stop={off['stop_reason']} black_d={off['black_decisions']} white_d={off['white_decisions']}")
        print(f"[ON]  status={on['status']} stop={on['stop_reason']} black_d={on['black_decisions']} white_d={on['white_decisions']} messages={on['messages_sent']}")
        results.append(result)

        analysis = on.pop("analysis")
        _write_jsonl(os.path.join(out_dir, "messages.jsonl"), analysis["messages"])
        _write_jsonl(os.path.join(out_dir, "social_events.jsonl"), analysis["social_events"])
        _write_jsonl(os.path.join(out_dir, "threat_events.jsonl"), analysis["threat_events"])
        _write_jsonl(os.path.join(out_dir, "cooperation_events.jsonl"), analysis["cooperation_events"])
        _write_jsonl(os.path.join(out_dir, "warning_events.jsonl"), analysis["warning_events"])
        _write_jsonl(os.path.join(out_dir, "claim_consistency.jsonl"), analysis["claim_consistency"])
        _write_jsonl(os.path.join(out_dir, "episodes.jsonl"), [
            {"episode_id": off["episode_id"], "pair_label": result["pair_label"], "condition": "SOCIAL_OFF", **{k: v for k, v in off.items() if k != "episode_id"}},
            {"episode_id": on["episode_id"], "pair_label": result["pair_label"], "condition": "SOCIAL_ON", **{k: v for k, v in on.items() if k != "episode_id"}},
        ])
        _write_jsonl(os.path.join(out_dir, "paired_plan.jsonl"), [{
            "pair_label": result["pair_label"], "seed": pair["seed"], "black_personality": pair["black_personality"],
            "white_personality": pair["white_personality"], "mirror": pair["mirror"],
            "off_episode_id": off["episode_id"], "on_episode_id": on["episode_id"],
        }])
    return results


def pilot(out_dir: str, region: str) -> dict:
    print("\n=== PHASE B: 2 mirrored SOCIAL_ON duels (RACER vs SABOTEUR) ===")
    from src.llm.personality import RACER, SABOTEUR

    pairs = generate_mirror_pairs([(RACER, SABOTEUR)], [STAGE1_SEED])
    results = run_pairs(out_dir, region, pairs)
    return {"pairs": results}


def stage1(out_dir: str, region: str, matchup_index: int | None) -> dict:
    matchups = MATCHUPS if matchup_index is None else [MATCHUPS[matchup_index]]
    print(f"\n=== PHASE C: Stage 1 paired benchmark ({'all 3 matchups' if matchup_index is None else f'matchup {matchup_index} only'} x 2 mirrors x 2 conditions, seed={STAGE1_SEED}) ===")
    pairs = generate_mirror_pairs(matchups, [STAGE1_SEED])
    results = run_pairs(out_dir, region, pairs)
    return {"pairs": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--stage1", action="store_true")
    parser.add_argument("--matchup", type=int, default=None, choices=(0, 1, 2), help="Stage 1 only: run just this one matchup index (0=OPPORTUNIST-v-OPPORTUNIST, 1=RACER-v-SABOTEUR, 2=CAUTIOUS-v-SABOTEUR), for incremental live staging")
    args = parser.parse_args(argv)

    phase_name = "pilot" if args.pilot else "stage1"
    print(f"=== M5F.3B free social strategy ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    base_out_dir = os.path.join(BASE_OUT_DIR, phase_name)
    if args.stage1 and args.matchup is not None:
        # Deliberate incremental live staging (one matchup per invocation,
        # matching this whole session's established per-chunk pattern) -
        # always the same literal directory, bypassing
        # `resolve_output_dir`'s rerun-avoidance (which would otherwise
        # redirect every invocation after the first to a new `-rerun-`
        # directory, since the prior invocation's own `experiment.json`
        # already exists there by the time the next one starts). The
        # per-event JSONL files below are append-only either way, so
        # staged invocations accumulate correctly into one directory.
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
    result = pilot(out_dir, region) if args.pilot else stage1(out_dir, region, args.matchup)
    run_end = datetime.now(timezone.utc)

    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_summary = detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover
        sleep_summary = {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}

    summary = {
        "phase": phase_name, "seed": STAGE1_SEED, "pairs_run": len(result["pairs"]),
        "sleep_prevention_active": True, "sleep_assertion_verified": sleep_assertion_verified, "sleep_contamination": sleep_summary,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment_id": "m5f3b-free-social-strategy-v1", "phase": phase_name, "model_id": MODEL_ID, "started_at": run_start.isoformat(), "finished_at": run_end.isoformat()}, f, indent=2)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3B {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
