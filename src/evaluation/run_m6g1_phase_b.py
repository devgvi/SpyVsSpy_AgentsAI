"""M6G.1 Phases A and B (spec sections 6-9, 22-29, 56): zero model calls.

Phase A probes every current action type on the real Engine and records
when it changes the world and what its duration does
(`duration_semantics.json`). Phase B replays the fixed scripted pair of
every L scenario under A->B, B->A, both T0 arrival orders, and the
busy-period variants (second action during vs after the first action's
busy period) - `scripted_temporal_trials.jsonl`. Also writes
`scenario_manifest.json`.

    python -m src.evaluation.run_m6g1_phase_b
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

from src.evaluation.conflict_scenarios import build_conflict_scenarios, C0_NON_CONFLICTING, C2_SHARED_DOOR, C3_SHARED_SPACE, C4_TRAP_CONFLICT, C5_INVALIDATES
from src.evaluation.decision_scheduling import build_request, native_render
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS, build_duel_snapshot
from src.evaluation.m6g_live import write_json
from src.evaluation.runtime_snapshot_replay import _place_near_object
from src.evaluation.runtime_variance_analysis import content_hash
from src.evaluation.temporal_semantics import (
    L5_ACTION_INVALIDATES_OTHER,
    L_FAMILIES,
    T0,
    T1A,
    T1B,
    _setup_l5,
    build_temporal_scenarios,
    classify_temporal,
    m6g_family_for,
    probe_action,
    replay_pair,
    run_busy_variant,
)
from src.llm_controller import serialize_observation

OUT_DIR = os.path.join("evaluation_runs", "m6g1-long-running-actions-v1")
PROBE_MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def out_path(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def _probe_setups() -> list[tuple[str, object, str, dict, bool]]:
    m6g = build_conflict_scenarios()

    def door_reopen(state):
        m6g[C2_SHARED_DOOR].setup_fn(state)
        state.door_states["door_security_corridor"] = "open"
        state._rebuild_nav_grid()

    def trapped_desk_for_black(state):
        _setup_l5(state)
        state._object_entry("desk_01")["trap"] = {"type": "bomb", "owner_id": ACTOR_B, "armed": True}

    def near_exit(state):
        _place_near_object(state, ACTOR_A, "exit_door")

    return [
        ("INTERACT door: locked -> unlock+open", m6g[C2_SHARED_DOOR].setup_fn, ACTOR_A, {"action": "interact", "target_id": "door_security_corridor"}, False),
        ("INTERACT door: already open -> re-open", door_reopen, ACTOR_A, {"action": "interact", "target_id": "door_security_corridor"}, False),
        ("INTERACT search (desk_01)", m6g[C0_NON_CONFLICTING].setup_fn, ACTOR_A, {"action": "interact", "target_id": "desk_01"}, False),
        ("INTERACT search springs an undetected enemy trap", trapped_desk_for_black, ACTOR_A, {"action": "interact", "target_id": "desk_01"}, False),
        ("INTERACT exit_door without mission items", near_exit, ACTOR_A, {"action": "interact", "target_id": "exit_door"}, False),
        ("INSPECT (desk_01)", m6g[C0_NON_CONFLICTING].setup_fn, ACTOR_A, {"action": "inspect", "target_id": "desk_01"}, False),
        ("PLACE_TRAP bomb (ambassador_safe)", m6g[C4_TRAP_CONFLICT].setup_fn, ACTOR_A, {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "bomb"}, False),
        ("DISARM own trap (ambassador_safe)", m6g[C5_INVALIDATES].setup_fn, ACTOR_A, {"action": "disarm", "target_id": "ambassador_safe"}, False),
        ("MOVE to desk_01 (multi-tick path)", m6g[C3_SHARED_SPACE].setup_fn, ACTOR_A, {"action": "move", "target_id": "desk_01"}, False),
        ("WAIT", m6g[C0_NON_CONFLICTING].setup_fn, ACTOR_A, {"action": "wait"}, False),
        ("SEND_MESSAGE (SOCIAL_ON probe only; every M6G.1 run is SOCIAL_OFF)", m6g[C0_NON_CONFLICTING].setup_fn, ACTOR_A,
         {"action": "send_message", "target_id": ACTOR_B, "message": "probe"}, True),
    ]


def duration_semantics() -> list[dict]:
    rows = []
    for label, setup, actor, raw, comm in _probe_setups():
        probe = probe_action(label, setup, actor, raw, communication_enabled=comm)
        probe["classification"] = classify_temporal(probe)
        rows.append(probe)
    return rows


def scripted_matrix() -> list[dict]:
    scenarios = build_temporal_scenarios()
    rows = []
    for family in L_FAMILIES:
        s = scenarios[family]
        raws = {ACTOR_A: s.raw_a, ACTOR_B: s.raw_b}
        variants = [
            ("T1A_barrier_A_then_B", dict(condition=T1A)),
            ("T1B_barrier_B_then_A", dict(condition=T1B)),
            ("T0_arrival_A_first", dict(condition=T0, arrival_order=(ACTOR_A, ACTOR_B))),
            ("T0_arrival_B_first", dict(condition=T0, arrival_order=(ACTOR_B, ACTOR_A))),
        ]
        for variant, kwargs in variants:
            row = replay_pair(s, raws, pair_id=f"B:{family}:{variant}", model_id=PROBE_MODEL_ID, **kwargs)
            row.update({"phase": "B", "variant": variant, "source": "scripted"})
            rows.append(row)
        for first in ACTORS:
            for waits in (False, True):
                busy = run_busy_variant(s, first, second_waits_for_completion=waits)
                busy.update({"phase": "B", "variant": f"busy_{'after' if waits else 'during'}_{first}_first", "source": "scripted"})
                rows.append(busy)
    return rows


def scenario_manifest() -> dict:
    out = {}
    for family, s in build_temporal_scenarios().items():
        snap = build_duel_snapshot(s.setup_fn)
        actors = {}
        for actor_id in ACTORS:
            obs = snap.observations[actor_id]
            req = build_request(actor_id, obs, snap.legal_actions[actor_id], scenario_family=family, model_id=PROBE_MODEL_ID, provider_name="bedrock")
            sp, up = native_render(req)
            actor = snap.state.actors[actor_id]
            actors[actor_id] = {
                "position": [actor.x, actor.y], "room": serialize_observation(obs)["self"].get("room"), "nearby_object_id": obs.nearby_object_id,
                "legal_actions_non_move": [{"action": e["action"], "target_id": e.get("target_id")} for e in snap.legal_actions[actor_id] if e["action"] != "move"],
                "native_provider_request_hash": content_hash(sp, up), "scripted_decision": s.scripted_raw(actor_id),
            }
        out[family] = {
            "reuses_m6g_scenario": m6g_family_for(family), "new_in_m6g1": family == L5_ACTION_INVALIDATES_OTHER,
            "description": s.description, "shared_state": s.shared_state, "side_effect_visibility": s.side_effect_visibility,
            "pre_action_world_state_hash": snap.world_state_hash, "actors": actors,
        }
    return out


def run() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    probes = duration_semantics()
    write_json(out_path("duration_semantics.json"), {"probes": probes})
    rows = scripted_matrix()
    with open(out_path("scripted_temporal_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    write_json(out_path("scenario_manifest.json"), scenario_manifest())
    return {"probes": probes, "rows": rows}


if __name__ == "__main__":
    out = run()
    for p in out["probes"]:
        print(f"{p['label']}: {p['classification']} cost={p['time_cost_seconds']} stun={p['stun_seconds']} effect_at_apply={p['world_effect_at_apply']} "
              f"busy={p['busy_after_apply']} frozen_ok={p['busy_unchanged_by_frozen_updates']} ticks={p['busy_ticks_until_complete']} "
              f"completion_events={p['events_during_busy_and_at_completion']} multi_tick={p['multi_tick_ticks']} outcome={p['apply_outcome']}")
    print(f"{len(out['rows'])} scripted rows")
