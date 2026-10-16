"""M6G Phase B (spec sections 14-23, 56): every conflict scenario (C0-C5)
under every schedule (S0-S4) with fixed scripted decisions - zero model
calls - so Engine scheduling semantics are measured separately from model
variability. Also writes `scenario_manifest.json` (placements, legal
actions, request hashes, predeclared expectations).

    python -m src.evaluation.run_m6g_phase_b
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

from src.evaluation.conflict_scenarios import CONFLICT_FAMILIES, build_conflict_scenarios
from src.evaluation.decision_scheduling import SCHEDULING_CONDITIONS, ScriptedDecider, build_request, native_render, run_condition
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS, build_duel_snapshot, legal_actions_hash, observation_hash
from src.evaluation.m6g_live import OUT_DIR, out_path, write_json
from src.evaluation.run_m6g_analysis import MODEL_ID, PROVIDER_NAME, deterministic_application_order_metrics
from src.evaluation.runtime_variance_analysis import content_hash
from src.llm_controller import serialize_observation


def scenario_manifest() -> dict:
    manifest = {}
    for family, s in build_conflict_scenarios().items():
        snap = build_duel_snapshot(s.setup_fn)
        actors = {}
        for actor_id in ACTORS:
            obs = snap.observations[actor_id]
            actor = snap.state.actors[actor_id]
            req = build_request(actor_id, obs, snap.legal_actions[actor_id], scenario_family=family, model_id=MODEL_ID, provider_name=PROVIDER_NAME)
            system_prompt, user_prompt = native_render(req)
            actors[actor_id] = {
                "position": [actor.x, actor.y],
                "room": serialize_observation(obs)["self"].get("room"),
                "nearby_object_id": obs.nearby_object_id,
                "inventory": sorted(actor.inventory.items),
                "trap_inventory": dict(sorted(actor.trap_inventory.items())),
                "legal_actions_non_move": [
                    {"action": e["action"], "target_id": e.get("target_id")} for e in snap.legal_actions[actor_id] if e["action"] != "move"
                ],
                "legal_actions_count": len(snap.legal_actions[actor_id]),
                "legal_actions_hash": legal_actions_hash(snap.legal_actions[actor_id]),
                "observation_hash": observation_hash(obs),
                "agent_request_semantic_hash": req.request_semantic_hash(),
                "native_provider_request_hash": content_hash(system_prompt, user_prompt),
                "native_provider_request_characters": len(system_prompt) + len(user_prompt),
                "scripted_decision": s.scripted_raw(actor_id),
            }
        manifest[family] = {
            "scenario_id": s.scenario_id,
            "description": s.description,
            "shared_state": s.shared_state,
            "side_effect_visibility": s.side_effect_visibility,
            "negative_control": s.negative_control,
            "predeclared_expected_apply_order_effect": s.expected_apply_order_effect,
            "predeclared_expected_second_observation_changes": s.expected_second_observation_changes,
            "pre_action_world_state_hash": snap.world_state_hash,
            "actors_share_coordinates": snap.state.actors[ACTOR_A].x == snap.state.actors[ACTOR_B].x
            and snap.state.actors[ACTOR_A].y == snap.state.actors[ACTOR_B].y,
            "actors": actors,
        }
    return manifest


def run() -> dict:
    scenarios = build_conflict_scenarios()
    rows = []
    for family in CONFLICT_FAMILIES:
        s = scenarios[family]
        for condition in SCHEDULING_CONDITIONS:
            decider = ScriptedDecider({ACTOR_A: s.raw_a, ACTOR_B: s.raw_b})
            result = run_condition(condition, s, decider, trial_id=f"B:scripted:{family}:{condition}", model_id=MODEL_ID, provider_name=PROVIDER_NAME)
            rows.append(result.to_row(phase="B", rep=0))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path("scripted_pair_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    metrics = deterministic_application_order_metrics(rows)
    write_json(out_path("application_order_metrics.json"), {"deterministic_phase_b": metrics})
    write_json(out_path("scenario_manifest.json"), scenario_manifest())
    return metrics


if __name__ == "__main__":
    for family, m in run().items():
        print(f"{family}: {m['observed_labels']} (predeclared effect={m['predeclared_expected_apply_order_effect']}, held={m['expectation_held']})")
