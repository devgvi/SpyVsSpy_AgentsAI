"""M5F.3A.6: automated live-diagnostic entry point for goal-identity /
world-target / position cross-mapping isolation.

    python -m src.evaluation.run_m5f3a6 --pair 1  # 4 cells x 10 trials = 40 decisions
    python -m src.evaluation.run_m5f3a6 --pair 2
    python -m src.evaluation.run_m5f3a6 --pair 3  # only if pairs 1-2 don't yet show a clear, replicated pattern

Phase A/B (mapping infrastructure, network-free tests) is
`src/evaluation/goal_identity.py` + tests/test_m5f3a6_goal_identity.py -
not this driver. `id_pairs.json` (token-shape audit) was recorded before
any live call.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.goal_identity import ID_PAIRS, MAPPING_1, MAPPING_2, build_goal_mapping, goal_identity_model_payload_transform, goal_identity_model_response_transform
from src.evaluation.multiseed_stability import detect_sleep_contamination, resolve_output_dir
from src.evaluation.order_control import OrderPolicy
from src.evaluation.scenario_harness import _last_decision_event, _run_one_decision
from src.evaluation.symmetric_harness import NO_MESSAGE_CONTROL, alpha_beta_object_ids, build_symmetric_state, run_symmetric_trial, static_symmetry_audit
from src.llm.clients.bedrock_client import BedrockClient, DEFAULT_MODEL_ID, bedrock_token_present, resolve_bedrock_region, run_bedrock_preflight
from src.llm.personality import OPPORTUNIST
from src.llm.telemetry import TelemetryLogger

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
BASE_OUT_DIR = os.path.join("evaluation_runs", "m5f3a6-goal-identity-priors-v1")
TRIALS_PER_CELL = 10
SOCIAL_PILOT_TRIALS_PER_CELL = 2

WORLD_TARGETS = alpha_beta_object_ids(True, neutral_ids=True)


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


def run_id_pair(out_dir: str, region: str, pair_index: int) -> dict:
    id_pair = ID_PAIRS[pair_index]
    print(f"\n=== PAIR {pair_index + 1}: {id_pair} ({TRIALS_PER_CELL} trials/cell, 4 cells = {TRIALS_PER_CELL * 4} decisions) ===")
    rows = []
    cells = [
        (MAPPING_1, OrderPolicy.CANONICAL, "M1_ORDER_AB"),
        (MAPPING_1, OrderPolicy.REVERSED, "M1_ORDER_BA"),
        (MAPPING_2, OrderPolicy.CANONICAL, "M2_ORDER_AB"),
        (MAPPING_2, OrderPolicy.REVERSED, "M2_ORDER_BA"),
    ]
    for mapping_label, order_policy, cell_label in cells:
        mapping = build_goal_mapping(id_pair, WORLD_TARGETS, mapping_label)
        for trial_index in range(1, TRIALS_PER_CELL + 1):
            client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
            sink_p: dict = {}
            sink_r: dict = {}
            t_p = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=order_policy, trial_index=trial_index, metadata_sink=sink_p)
            t_r = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink_r)
            trial = run_symmetric_trial(
                NO_MESSAGE_CONTROL, None, OPPORTUNIST, trial_index, client, neutral_ids=True,
                model_payload_transform=t_p, model_response_transform=t_r,
                episode_prefix=f"m5f3a6-pair{pair_index + 1}-{cell_label}",
            )
            presented_order = sink_p.get("metadata", {}).get("goal_order", {}).get("presented_order", list(id_pair))
            selected_goal_id = None
            if trial["final_target"] == mapping.get(id_pair[0]):
                selected_goal_id = id_pair[0]
            elif trial["final_target"] == mapping.get(id_pair[1]):
                selected_goal_id = id_pair[1]
            selected_position = None
            if selected_goal_id is not None and selected_goal_id in presented_order:
                selected_position = "first" if presented_order[0] == selected_goal_id else "second"
            row = {
                **trial, "id_pair": list(id_pair), "mapping": mapping_label, "cell": cell_label,
                "goal_to_target": mapping, "presented_order": presented_order,
                "selected_goal_id": selected_goal_id, "selected_position": selected_position,
            }
            rows.append(row)
            print(f"[PAIR{pair_index + 1}] {cell_label} trial{trial_index}: valid={trial['valid']} target={trial['final_target']} goal_id={selected_goal_id} position={selected_position}")

    with open(os.path.join(out_dir, f"mapping_trials_pair{pair_index + 1}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    valid_rows = [r for r in rows if r["valid"]]
    x, y = WORLD_TARGETS
    metrics = {
        "id_pair": list(id_pair), "trials": len(rows), "valid_trials": len(valid_rows),
        "p_select_id_a": round(sum(1 for r in valid_rows if r["selected_goal_id"] == id_pair[0]) / len(valid_rows), 3) if valid_rows else None,
        "p_select_id_b": round(sum(1 for r in valid_rows if r["selected_goal_id"] == id_pair[1]) / len(valid_rows), 3) if valid_rows else None,
        "p_select_world_x": round(sum(1 for r in valid_rows if r["final_target"] == x) / len(valid_rows), 3) if valid_rows else None,
        "p_select_world_y": round(sum(1 for r in valid_rows if r["final_target"] == y) / len(valid_rows), 3) if valid_rows else None,
        "p_select_first_position": round(sum(1 for r in valid_rows if r["selected_position"] == "first") / len(valid_rows), 3) if valid_rows else None,
        "p_select_second_position": round(sum(1 for r in valid_rows if r["selected_position"] == "second") / len(valid_rows), 3) if valid_rows else None,
        "other_or_invalid": len(rows) - len(valid_rows),
    }
    print(f"[PAIR{pair_index + 1} RESULT] {metrics}")
    return {"rows": rows, "metrics": metrics}


# -- Tiny social pilot (spec sections 29-30, only run because the baseline
# gate genuinely passed for the first time in the whole M5F.3A series:
# Pairs 1-2 both showed a clean, doubly-replicated POSITION_PRIOR_DOMINANT
# pattern with zero identity/world-target skew - see summary.md) ---------


def _opaque_message_text(message_class: str, target_id: str, other_id: str) -> str:
    if message_class == "COOPERATION":
        return f"I'll take {target_id}. You take {other_id}."
    if message_class == "THREAT":
        return f"If you go to {target_id}, I'll trap it."
    raise ValueError(message_class)


def social_pilot(out_dir: str, region: str) -> dict:
    print(f"\n=== TINY SOCIAL PILOT: COOPERATION/THREAT x goal-order x mapping ({SOCIAL_PILOT_TRIALS_PER_CELL} trials/cell) ===")
    id_pair = ID_PAIRS[0]
    x, y = WORLD_TARGETS  # THREAT always targets the same real world target X, referenced via whichever opaque id currently maps to it (spec section 29's "represent X with each ID and in each position")
    rows = []
    for message_class in ("COOPERATION", "THREAT"):
        for mapping_label in (MAPPING_1, MAPPING_2):
            mapping = build_goal_mapping(id_pair, WORLD_TARGETS, mapping_label)
            target_to_id = {v: k for k, v in mapping.items()}
            target_opaque_id = target_to_id[x]
            other_opaque_id = target_to_id[y]
            for order_policy, order_label in ((OrderPolicy.CANONICAL, "ORDER_AB"), (OrderPolicy.REVERSED, "ORDER_BA")):
                for trial_index in range(1, SOCIAL_PILOT_TRIALS_PER_CELL + 1):
                    state = build_symmetric_state(neutral_ids=True)
                    text = _opaque_message_text(message_class, target_opaque_id, other_opaque_id)
                    state.messages.append({
                        "message_id": f"pilot-{message_class}-{mapping_label}-{order_label}-{trial_index}",
                        "sender_actor_id": constants.BLACK_SPY_ID, "recipient_actor_id": constants.WHITE_SPY_ID,
                        "game_time": 0.0, "text": text,
                    })
                    if message_class == "THREAT":
                        # Genuine capability on the real threatened target X (spec section 12 of M5F.3A.2/.5's own established threat-credibility fixture).
                        state._object_entry(x)["inspected_by"].add(constants.BLACK_SPY_ID)

                    client = BedrockClient(model_name=MODEL_ID, region=region, timeout=60.0)
                    sink_p, sink_r = {}, {}
                    t_p = goal_identity_model_payload_transform(id_pair=id_pair, order_policy=order_policy, trial_index=trial_index, metadata_sink=sink_p)
                    t_r = goal_identity_model_response_transform(goal_to_target=mapping, metadata_sink=sink_r)
                    telemetry_episode = f"m5f3a6-pilot-{message_class}-{mapping_label}-{order_label}-trial{trial_index}"
                    telemetry = TelemetryLogger()
                    _run_one_decision(
                        state, constants.WHITE_SPY_ID, OPPORTUNIST, client, telemetry_episode, telemetry, wall_timeout=180.0,
                        model_payload_transform=t_p, model_response_transform=t_r,
                    )
                    decision = _last_decision_event(telemetry, telemetry_episode)
                    executed = decision.get("executed_action") or {}
                    final_target = executed.get("target_id")
                    presented_order = sink_p.get("metadata", {}).get("goal_order", {}).get("presented_order", list(id_pair))
                    selected_position = None
                    resolution = sink_r.get("resolution", {})
                    if resolution.get("success") and resolution.get("goal_id") in presented_order:
                        selected_position = "first" if presented_order[0] == resolution["goal_id"] else "second"
                    row = {
                        "message_class": message_class, "mapping": mapping_label, "order_label": order_label, "trial_index": trial_index,
                        "message_text": text, "target_opaque_id": target_opaque_id,
                        "valid": bool(decision.get("final_model_action_valid")),
                        "final_action": executed.get("action"), "final_target": final_target,
                        "shifted_to_x": final_target == x, "shifted_to_y": final_target == y,
                        "selected_goal_id": resolution.get("goal_id"), "selected_position": selected_position,
                        "reason": (decision.get("raw_response") or {}).get("reason"),
                    }
                    rows.append(row)
                    print(f"[PILOT] {message_class}/{mapping_label}/{order_label} trial{trial_index}: valid={row['valid']} target={final_target} pos={selected_position}")

    with open(os.path.join(out_dir, "social_pilot.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair", type=int, choices=(1, 2, 3))
    group.add_argument("--social-pilot", action="store_true")
    args = parser.parse_args(argv)
    pair_index = (args.pair - 1) if args.pair else None

    phase_name = f"pair{args.pair}" if args.pair else "social-pilot"
    print(f"=== M5F.3A.6 goal identity priors ({phase_name}) starting {datetime.now(timezone.utc).isoformat()} ===")

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

    out_dir = resolve_output_dir(BASE_OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    static_result = static_symmetry_audit()
    print(f"[STATIC SYMMETRY] {static_result['result']}")
    if static_result["result"] != "SYMMETRIC":
        print("[STATIC SYMMETRY] FAILED - refusing to start any live episode.")
        return 1

    if not _preflight(out_dir, region):
        print("[PREFLIGHT] FAILED - stopping before any episode.")
        return 1

    run_start = datetime.now(timezone.utc)
    if args.social_pilot:
        result = social_pilot(out_dir, region)
        metrics = {"total_trials": len(result["rows"])}
    else:
        result = run_id_pair(out_dir, region, pair_index)
        metrics = result["metrics"]

    run_end = datetime.now(timezone.utc)
    try:
        pmset_output = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30.0).stdout
        sleep_summary = detect_sleep_contamination(pmset_output, run_start, run_end).to_dict()
    except Exception as exc:  # pragma: no cover
        sleep_summary = {"sleep_contamination_detected": None, "matching_events": [], "note": f"pmset check failed: {type(exc).__name__}"}

    summary = {
        "phase": phase_name, "static_symmetry": static_result["result"], "metrics": metrics,
        "sleep_prevention_active": True, "sleep_assertion_verified": sleep_assertion_verified, "sleep_contamination": sleep_summary,
    }
    summary_suffix = f"pair{args.pair}" if args.pair else "social_pilot"
    with open(os.path.join(out_dir, f"summary_{summary_suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nOutputs written under: {out_dir}/")
    print(f"=== M5F.3A.6 {phase_name} finished {datetime.now(timezone.utc).isoformat()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
