"""M5F.3A.7: automated live-diagnostic entry point for the fully
counterbalanced social-influence replication.

    python -m src.evaluation.run_m5f3a7 --family COOPERATION
    python -m src.evaluation.run_m5f3a7 --family TRUE_WARNING
    python -m src.evaluation.run_m5f3a7 --family CREDIBLE_THREAT
    python -m src.evaluation.run_m5f3a7 --family NON_CREDIBLE_THREAT
    python -m src.evaluation.run_m5f3a7 --family MISDIRECTION

Each invocation runs the full 8-cell x 2-trial = 16-decision
counterbalanced design for that one treatment family (target role X/Y x
goal-id mapping M1/M2 x presentation order CANONICAL/REVERSED - see
`counterbalance_plan.json` for the scope decision scaling this down from
the spec's own "20/treatment role" recommendation). CREDIBLE_THREAT and
NON_CREDIBLE_THREAT both append to `threat_trials.jsonl` per spec
section 45's recommended file list. NO_MESSAGE_CONTROL is not run here -
reused from M5F.3A.6's own Pair 1 cross-mapping matrix (see
`counterbalance_plan.json`).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src.evaluation.counterbalanced_social import (
    COOPERATION,
    CREDIBLE_THREAT,
    DEFAULT_ID_PAIR,
    MISDIRECTION,
    NON_CREDIBLE_THREAT,
    TRUE_WARNING,
    run_counterbalanced_trial,
)
from src.evaluation.goal_identity import MAPPING_1, MAPPING_2
from src.evaluation.multiseed_stability import detect_sleep_contamination
from src.evaluation.order_control import OrderPolicy
from src.evaluation.symmetric_harness import static_symmetry_audit
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.personality import OPPORTUNIST

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
OUT_DIR = os.path.join("evaluation_runs", "m5f3a7-counterbalanced-social-replication-v1")
TRIALS_PER_CELL = 2

FAMILY_FILENAMES = {
    COOPERATION: "cooperation_trials.jsonl",
    TRUE_WARNING: "warning_trials.jsonl",
    CREDIBLE_THREAT: "threat_trials.jsonl",
    NON_CREDIBLE_THREAT: "threat_trials.jsonl",
    MISDIRECTION: "misdirection_trials.jsonl",
}

CELLS = [
    ("X", MAPPING_1, OrderPolicy.CANONICAL, "X_M1_AB"),
    ("X", MAPPING_1, OrderPolicy.REVERSED, "X_M1_BA"),
    ("X", MAPPING_2, OrderPolicy.CANONICAL, "X_M2_AB"),
    ("X", MAPPING_2, OrderPolicy.REVERSED, "X_M2_BA"),
    ("Y", MAPPING_1, OrderPolicy.CANONICAL, "Y_M1_AB"),
    ("Y", MAPPING_1, OrderPolicy.REVERSED, "Y_M1_BA"),
    ("Y", MAPPING_2, OrderPolicy.CANONICAL, "Y_M2_AB"),
    ("Y", MAPPING_2, OrderPolicy.REVERSED, "Y_M2_BA"),
]


def _verify_sleep_assertion() -> bool:
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
        return "PreventUserIdleSystemSleep" in output and "caffeinate" in output
    except Exception:  # pragma: no cover
        return False


def _preflight(region: str) -> bool:
    preflight = run_bedrock_preflight(model_id=MODEL_ID, region=region, timeout=60.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "provider_events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "preflight", "family": None, **preflight.to_dict()}, default=str) + "\n")
    print(f"[PREFLIGHT] {preflight.to_dict()}")
    return preflight.ok


def run_family(message_class: str, region: str) -> list[dict]:
    print(f"\n=== FAMILY {message_class}: 8 cells x {TRIALS_PER_CELL} trials = {8 * TRIALS_PER_CELL} decisions ===")
    rows: list[dict] = []
    for target_role, mapping_label, order_policy, cell_label in CELLS:
        for trial_index in range(1, TRIALS_PER_CELL + 1):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            trial = run_counterbalanced_trial(
                message_class, target_role, mapping_label, order_policy, trial_index, OPPORTUNIST, client,
                id_pair=DEFAULT_ID_PAIR, episode_prefix=f"m5f3a7-{message_class}",
            )
            trial["cell"] = cell_label
            rows.append(trial)
            print(f"[{message_class}] {cell_label} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} "
                  f"world={trial['selected_world_target']} pos={trial['selected_position']}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=[COOPERATION, TRUE_WARNING, CREDIBLE_THREAT, NON_CREDIBLE_THREAT, MISDIRECTION])
    args = parser.parse_args(argv)

    print(f"=== M5F.3A.7 counterbalanced social replication ({args.family}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    os.makedirs(OUT_DIR, exist_ok=True)

    static_result = static_symmetry_audit()
    print(f"[STATIC SYMMETRY] {static_result['result']}")
    if static_result["result"] != "SYMMETRIC":
        print("[STATIC SYMMETRY] FAILED - refusing to start any live episode.")
        return 1

    if not _preflight(region):
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    rows = run_family(args.family, region)
    run_end = datetime.now(timezone.utc)

    filename = FAMILY_FILENAMES[args.family]
    out_path = os.path.join(OUT_DIR, filename)
    mode = "a" if os.path.exists(out_path) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_summary = detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover
        sleep_summary = {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}

    valid_rows = [r for r in rows if r["valid"]]
    metrics = {
        "family": args.family, "trials": len(rows), "valid_trials": len(valid_rows),
        "shifted_to_treatment_target": sum(1 for r in valid_rows if r["shifted_to_treatment_target"]),
        "shifted_to_other": sum(1 for r in valid_rows if r["shifted_to_other"]),
    }
    print(f"[{args.family} RESULT] {metrics}")

    run_summary = {
        "family": args.family, "static_symmetry": static_result["result"], "metrics": metrics,
        "sleep_prevention_active": True, "sleep_assertion_verified": sleep_assertion_verified, "sleep_contamination": sleep_summary,
        "wrote": out_path, "mode": mode,
    }
    with open(os.path.join(OUT_DIR, f"run_summary_{args.family.lower()}_{mode}.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary, default=str) + "\n")

    print(f"\nOutputs written under: {OUT_DIR}/")
    print(f"=== M5F.3A.7 {args.family} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
