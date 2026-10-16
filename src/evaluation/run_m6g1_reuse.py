"""M6G.1 spec sections 15-16, 30, 32: reuse M6G's captured live decision
pairs - zero new model calls.

Every M6G frozen-snapshot trial (S0/S1 in Phase C, S0 in Phase D, S0/S1 on
Strands) holds one decision per actor made from the identical pre-action
snapshot M6G.1's L0/L1/L3 scenarios rebuild. Each pair is replayed under
T0 (its logical arrival order), T0 with that order reversed, T1A and T1B.
The rebuilt request must hash to exactly what M6G sent, or the pair is
rejected.

Logical arrival order (spec section 45: logical ready order is enough):
the production loop starts Black's request first (`decision_call_order`
default) and White's one `get_action` join later (<= 50 ms), so
ready_black = latency_black and ready_white = 50 ms + latency_white. M6G
made the two calls one after the other, so these latencies were measured
without the two calls overlapping.

    python -m src.evaluation.run_m6g1_reuse
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS, json_hash
from src.evaluation.m6g_live import read_jsonl, write_json
from src.evaluation.run_m6g1_phase_b import OUT_DIR, PROBE_MODEL_ID, out_path
from src.evaluation.temporal_semantics import (
    L0_NON_CONFLICTING,
    L1_SHARED_DOOR,
    L3_TRAP_SLOT_CONFLICT,
    T0,
    T1A,
    T1B,
    build_temporal_scenarios,
    replay_pair,
)

M6G_DIR = os.path.join("evaluation_runs", "m6g-decision-scheduling-v1")
M6G_TO_L = {
    "C0_NON_CONFLICTING_ACTIONS": L0_NON_CONFLICTING,
    "C2_SHARED_DOOR_OR_OBJECT": L1_SHARED_DOOR,
    "C4_TRAP_OR_INTERACTION_CONFLICT": L3_TRAP_SLOT_CONFLICT,
}
FROZEN = ("S0_FROZEN_SNAPSHOT_BASELINE", "S1_FROZEN_SNAPSHOT_REVERSED_CALL")
JOIN_OFFSET_MS = 50.0  # LLMController/StrandsAgentAdapter get_action: thread.join(timeout=0.05)
SAME_TICK_MS = 1000.0 / 60.0


def logical_arrival(lat_black: float, lat_white: float) -> tuple[tuple[str, str] | None, float, float]:
    """(arrival order or None for a same-tick tie, ready_black_ms, ready_white_ms)."""
    ready_b, ready_w = lat_black, JOIN_OFFSET_MS + lat_white
    if abs(ready_b - ready_w) < SAME_TICK_MS:
        return None, ready_b, ready_w
    return ((ACTOR_A, ACTOR_B) if ready_b < ready_w else (ACTOR_B, ACTOR_A)), ready_b, ready_w


def captured_pairs() -> list[dict]:
    sources = [
        ("native_call_order_trials.jsonl", "native", FROZEN),
        ("native_snapshot_timing_trials.jsonl", "native", ("S0_FROZEN_SNAPSHOT_BASELINE",)),
        ("strands_trials.jsonl", "strands", FROZEN),
    ]
    pairs = []
    for fname, runtime, conditions in sources:
        for row in read_jsonl(os.path.join(M6G_DIR, fname)):
            if row["condition"] not in conditions or row["scenario_family"] not in M6G_TO_L:
                continue
            recs = row["records"]
            pairs.append({
                "pair_id": f"m6g:{row['trial_id']}",
                "source_milestone": "M6G",
                "source_file": f"{M6G_DIR}/{fname}",
                "source_trial_id": row["trial_id"],
                "source_condition": row["condition"],
                "runtime": runtime,
                "m6g_scenario_family": row["scenario_family"],
                "scenario_family": M6G_TO_L[row["scenario_family"]],
                "raws": {a: recs[a]["raw_model_output"] for a in ACTORS},
                "provider_request_hashes": {a: recs[a]["provider_request_hash"] for a in ACTORS},
                "agent_request_semantic_hashes": {a: recs[a]["agent_request_semantic_hash"] for a in ACTORS},
                "decision_hashes": {a: json_hash(recs[a]["decision"]) for a in ACTORS},
                "latency_ms": {a: recs[a]["latency_ms"] for a in ACTORS},
            })
    return pairs


def _render_for(runtime: str):
    if runtime == "strands":
        from src.evaluation.m6g_live import StrandsDecider

        return StrandsDecider("us-east-1", PROBE_MODEL_ID).render
    from src.evaluation.decision_scheduling import native_render

    return native_render


def run() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = build_temporal_scenarios()
    pairs = captured_pairs()
    manifest, telemetry, replays = [], [], []
    renderers = {}
    for p in pairs:
        s = scenarios[p["scenario_family"]]
        render = renderers.setdefault(p["runtime"], _render_for(p["runtime"]))
        order, ready_b, ready_w = logical_arrival(p["latency_ms"][ACTOR_A], p["latency_ms"][ACTOR_B])
        common = dict(pair_id=p["pair_id"], model_id=PROBE_MODEL_ID, expected_request_hashes=p["provider_request_hashes"], render_fn=render)
        arrival = order or (ACTOR_A, ACTOR_B)
        t0 = replay_pair(s, p["raws"], T1A if order is None else T0, arrival_order=None if order is None else order, **common)
        t0_rev = replay_pair(s, p["raws"], T0, arrival_order=(arrival[1], arrival[0]), **common)
        t1a = replay_pair(s, p["raws"], T1A, **common)
        t1b = replay_pair(s, p["raws"], T1B, **common)
        for label, row in (("T0_logical_arrival", t0), ("T0_reversed_arrival", t0_rev), ("T1A", t1a), ("T1B", t1b)):
            row.update({"source": f"m6g_{p['runtime']}", "replay": label, "runtime": p["runtime"], "source_trial_id": p["source_trial_id"],
                        "arrival_tie_same_tick": order is None})
            replays.append(row)
        second = arrival[1]
        telemetry.append({
            "pair_id": p["pair_id"], "timing_source": "logical_from_m6g_latency", "runtime": p["runtime"], "scenario_family": p["scenario_family"],
            "latency_ms": p["latency_ms"], "ready_ms": {ACTOR_A: round(ready_b, 2), ACTOR_B: round(ready_w, 2)},
            "latency_diff_black_minus_white_ms": round(p["latency_ms"][ACTOR_A] - p["latency_ms"][ACTOR_B], 2),
            "first_ready": None if order is None else order[0], "ready_order": list(arrival), "same_tick_tie": order is None,
            "apply_order": t0["apply_order"], "ready_order_equals_apply_order": list(arrival) == t0["apply_order"],
            "ready_rank": {arrival[0]: 1, arrival[1]: 2},
            "decisions": {a: t0["records"][a]["decision"] for a in ACTORS},
            "second_staleness": t0["records"][second]["staleness_class"], "second_consequence": t0["records"][second]["consequence"],
            "second_relevant_state_stale": t0["records"][second]["relevant_state_changed"],
            "contested_effects": t0["contested_effects"], "world_state_hash_after": t0["world_state_hash_after"],
        })
        manifest.append({k: p[k] for k in ("pair_id", "source_milestone", "source_file", "source_trial_id", "source_condition", "runtime",
                                           "m6g_scenario_family", "scenario_family", "provider_request_hashes", "agent_request_semantic_hashes",
                                           "decision_hashes", "latency_ms")}
                        | {"request_rebuilt_and_matched": True, "logical_arrival_order": list(arrival), "same_tick_tie": order is None})

    write_json(out_path("m6g_reuse_manifest.json"), {"pairs": manifest, "n_pairs": len(manifest), "join_offset_ms": JOIN_OFFSET_MS,
                                                     "note": "requests rebuilt from the M6G.1 scenario and checked byte-for-byte against the provider-facing hash M6G sent"})
    with open(out_path("barrier_replay_trials.jsonl"), "w", encoding="utf-8") as f:
        for r in replays:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    with open(out_path("answer_order_telemetry.jsonl"), "w", encoding="utf-8") as f:
        for t in telemetry:
            f.write(json.dumps(t, sort_keys=True, default=str) + "\n")
    return {"pairs": len(pairs), "replays": len(replays)}


if __name__ == "__main__":
    print(run())
