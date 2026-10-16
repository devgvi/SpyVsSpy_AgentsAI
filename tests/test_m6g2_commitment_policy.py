"""M6G.2: network-free tests for commitment policies - the policy
abstraction, zero-call replay under every policy, and the policy-driven
real loop (scripted clients only)."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src.evaluation.commitment_policy import (
    BARRIER_POLICIES,
    C0,
    C1,
    C2,
    C4,
    C5,
    C5M,
    C6,
    C6M,
    COST_ONLY_COMMITMENT_EFFECT,
    NO_COMMITMENT_POLICY_EFFECT,
    P0,
    P1,
    P2,
    P3,
    P4,
    POLICIES,
    POLICY_IDS,
    SCENARIO_IDS,
    SHARED_RESOURCE_COMMITMENT_EFFECT,
    STALE_VALIDITY_COMMITMENT_EFFECT,
    build_policy_scenarios,
    commitment_effect,
    counterfactual_on_state,
    hp09_violation,
    policy_table,
    replay_under_policy,
)
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, fresh_scenario_state, json_hash

SCENARIOS = build_policy_scenarios()


def _raws(sid: str) -> dict:
    s = SCENARIOS[sid]
    return {ACTOR_A: s.scripted_raw(ACTOR_A), ACTOR_B: s.scripted_raw(ACTOR_B)}


def _replay(sid: str, policy: str, ready=(ACTOR_A, ACTOR_B), parity: int = 0, boundary: int = 0) -> dict:
    return replay_under_policy(SCENARIOS[sid], _raws(sid), policy, response_ready_order=ready, decision_boundary_index=boundary,
                               parity=parity, pair_id=f"t:{sid}", decision_boundary_id=f"t:{sid}:b{boundary}")


# ---------------------------------------------------------------------------
# Phase A: the policy abstraction
# ---------------------------------------------------------------------------


def test_policy_table_properties():
    table = {row["policy_id"]: row for row in policy_table()}
    assert tuple(table) == POLICY_IDS
    assert [table[p]["waits_for_both"] for p in POLICY_IDS] == [False, True, True, True, True]
    assert [table[p]["depends_on_latency"] for p in POLICY_IDS] == [True, True, False, False, False]
    assert [table[p]["depends_on_actor_id"] for p in POLICY_IDS] == [False, False, True, True, False]
    assert [table[p]["deterministic"] for p in POLICY_IDS] == [False, False, True, True, True]


@pytest.mark.parametrize("ready", ((ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A), None))
def test_policy_orders(ready):
    pair = {ACTOR_A: {"action": "wait"}, ACTOR_B: {"action": "wait"}}
    expected_ready = ready or (ACTOR_A, ACTOR_B)  # same-tick readiness: Game._step applies Black then White
    assert POLICIES[P0].order(pair, ready) == POLICIES[P1].order(pair, ready) == expected_ready
    assert POLICIES[P2].order(pair, ready) == (ACTOR_A, ACTOR_B)
    assert POLICIES[P3].order(pair, ready) == (ACTOR_B, ACTOR_A)
    for parity in (0, 1):
        firsts = [POLICIES[P4].order(pair, ready, k, parity)[0] for k in range(6)]
        assert firsts == ([ACTOR_A, ACTOR_B] * 3 if parity == 0 else [ACTOR_B, ACTOR_A] * 3)


def test_policies_never_read_or_touch_the_decisions():
    pair = {ACTOR_A: {"action": "interact", "target_id": "door_security_corridor"}, ACTOR_B: {"action": "move", "target_id": "desk_01"}}
    before = json_hash(pair)
    for p in POLICY_IDS:
        assert POLICIES[p].order(pair, (ACTOR_B, ACTOR_A), 3, 1) == POLICIES[p].order(None, (ACTOR_B, ACTOR_A), 3, 1)
    assert json_hash(pair) == before


def test_scenarios_cover_the_mandatory_set_and_mirror_the_asymmetric_conflicts():
    assert set(SCENARIO_IDS) >= {C0, C2, C4, C5, C6}
    for mirror, base in ((C5M, C5), (C6M, C6)):
        assert SCENARIOS[mirror].raw_a == SCENARIOS[base].raw_b and SCENARIOS[mirror].raw_b == SCENARIOS[base].raw_a
    c5m = fresh_scenario_state(SCENARIOS[C5M].setup_fn)._object_entry("ambassador_safe")
    assert c5m["trap"]["owner_id"] == ACTOR_B and ACTOR_A in c5m["detected_by"]
    assert fresh_scenario_state(SCENARIOS[C6M].setup_fn)._object_entry("desk_01")["inspected_by"] == {ACTOR_B}


# ---------------------------------------------------------------------------
# Phase B: zero-call replay under every policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sid", SCENARIO_IDS)
@pytest.mark.parametrize("ready", ((ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A), None))
def test_policies_change_only_the_commit_order(sid, ready):
    rows = {p: _replay(sid, p, ready) for p in POLICY_IDS}
    decision_hashes = {json_hash(r["decision_hashes"]) for r in rows.values()}
    snapshot_hashes = {r["snapshot_world_hash"] for r in rows.values()}
    assert len(decision_hashes) == 1 and len(snapshot_hashes) == 1
    for p, r in rows.items():
        assert r["apply_order"] == [r["first_commit_actor"], r["second_commit_actor"]]
        assert r["commit_rank"][r["first_commit_actor"]] == 1
    # P1 is P0's order behind a barrier: identical world and events
    assert (rows[P0]["world_state_hash_after"], rows[P0]["event_multiset_hash"]) == (rows[P1]["world_state_hash_after"], rows[P1]["event_multiset_hash"])
    # a fixed or alternating order is just one of the two orders
    assert rows[P2]["world_state_hash_after"] == _replay(sid, P4, ready, parity=0)["world_state_hash_after"]
    assert rows[P3]["world_state_hash_after"] == _replay(sid, P4, ready, parity=1)["world_state_hash_after"]


def test_p4_alternates_with_the_boundary_index():
    firsts = [_replay(C2, P4, parity=1, boundary=k)["first_commit_actor"] for k in range(4)]
    assert firsts == [ACTOR_B, ACTOR_A, ACTOR_B, ACTOR_A]


EXPECTED_EFFECT = {
    C0: NO_COMMITMENT_POLICY_EFFECT,
    C1: NO_COMMITMENT_POLICY_EFFECT,
    C2: COST_ONLY_COMMITMENT_EFFECT,
    C4: SHARED_RESOURCE_COMMITMENT_EFFECT,
    C5: SHARED_RESOURCE_COMMITMENT_EFFECT,
    C5M: SHARED_RESOURCE_COMMITMENT_EFFECT,
    C6: STALE_VALIDITY_COMMITMENT_EFFECT,
    C6M: STALE_VALIDITY_COMMITMENT_EFFECT,
}


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_commitment_effect_classification(sid):
    effect = commitment_effect(_replay(sid, P2), _replay(sid, P3))
    assert effect["classification"] == EXPECTED_EFFECT[sid]


def test_door_cost_goes_to_whoever_commits_first_under_every_policy():
    for p, ready, parity in ((P0, (ACTOR_B, ACTOR_A), 0), (P1, (ACTOR_B, ACTOR_A), 0), (P2, None, 0), (P3, None, 0), (P4, None, 1)):
        r = _replay(C2, p, ready, parity)
        assert (r["first_action_cost"], r["second_action_cost"]) == (1.0, 0.3)
        assert r["contested_effects"] == [("DOOR_UNLOCKED", r["first_commit_actor"])] or r["contested_effects"] == [["DOOR_UNLOCKED", r["first_commit_actor"]]]


def test_search_after_trap_consequence_follows_the_first_commit_not_the_identity():
    # the searcher is stunned exactly when the trap is committed first - in both role assignments
    for sid, trapper, searcher in ((C6, ACTOR_A, ACTOR_B), (C6M, ACTOR_B, ACTOR_A)):
        trap_first = _replay(sid, P2 if trapper == ACTOR_A else P3)
        search_first = _replay(sid, P3 if trapper == ACTOR_A else P2)
        assert trap_first["records"][searcher]["apply_outcome"] == "ALTERNATE_EFFECT"
        assert search_first["records"][searcher]["apply_outcome"] == "APPLIED"


def test_counterfactual_on_a_copied_state_matches_the_fresh_counterfactual():
    from src.evaluation.temporal_semantics import fresh_counterfactual

    s = SCENARIOS[C2]
    state = fresh_scenario_state(s.setup_fn)
    for a in (ACTOR_A, ACTOR_B):
        assert counterfactual_on_state(state, a, s.scripted_raw(a), family=s.family) == fresh_counterfactual(s, a, s.scripted_raw(a))
    assert state.door_states["door_security_corridor"] == "locked"  # the copy was used, not the original


def test_hp09_rule():
    assert hp09_violation("RESPONSE_READY_UNCOMMITTED", True)
    assert not hp09_violation("RESPONSE_READY_UNCOMMITTED", False)
    assert not hp09_violation("COMMITTED", True) and not hp09_violation("IDLE", True)


def test_barrier_policies_are_exactly_the_waiting_ones():
    assert BARRIER_POLICIES == tuple(p for p in POLICY_IDS if POLICIES[p].waits_for_both)


# ---------------------------------------------------------------------------
# The production loop under a commitment policy (scripted clients, no network)
# ---------------------------------------------------------------------------

STRANDS_MODEL = "us.anthropic.claude-sonnet-4-6"


def _policy_trial(sid: str, policy: str, *, slow: str | None = ACTOR_A, slow_delay: float = 0.35, fast_delay: float = 0.08,
                  runtime: str = "native_tool_like", second: dict | None = None, gate_slow: bool = True, **kwargs) -> dict:
    """One scripted trial. The slow actor's first answer waits behind a gate
    the probe opens at the knob moment (by default right after the other
    actor's decision reached the loop), so the ready order never depends on
    machine load. `second` = {actor: raw} answers a second boundary."""
    import threading

    from src.evaluation.commitment_probe import run_policy_trial, scripted_sequence_native_client, scripted_sequence_strands_invoke

    s = SCENARIOS[sid]
    gates = {slow: threading.Event()} if (slow and gate_slow) else {}
    steps = {}
    for a in (ACTOR_A, ACTOR_B):
        first = (s.scripted_raw(a), slow_delay if a == slow else fast_delay, gates.get(a))
        steps[a] = [first] + ([((second or {})[a], fast_delay, None)] if second else [])
    if runtime == "strands":
        kwargs.update(strands_invoke_overrides={a: scripted_sequence_strands_invoke(steps[a]) for a in steps},
                      provider="bedrock", model_id=STRANDS_MODEL, region="us-east-1")
    else:
        kwargs.update(client_overrides={a: scripted_sequence_native_client(steps[a]) for a in steps}, provider="mock", model_id="mock-model")
    if slow and gate_slow and "force_ready_after_opponent_step" not in kwargs:
        # ready at a tick boundary one tick after the opponent's step: never inside the handoff window, so the
        # interleaving (and the world it produces) does not depend on machine load
        kwargs.setdefault("ready_ticks_after_opponent_step", {slow: 1})
    return run_policy_trial(s, policy, trial_id="t", runtime=runtime, gates=gates, **kwargs)


STALENESS_FIELDS = ("staleness_class", "consequence", "legal_at_snapshot", "legal_at_apply", "apply_outcome", "time_cost")


def _assert_policy_trial_rules(row: dict) -> None:
    assert row["complete"] and not row["timed_out"] and row["pre_action_matches_harness_snapshot"]
    assert row["live_state_untouched_by_assembly"]
    for b in row["boundaries"]:
        assert b["policy_order_respected"], b["decision_boundary_id"]
        assert all(r["commit_matches_decision"] for r in b["records"].values())
    fa = row["freeze_audit"]
    assert fa["ticks_advanced_during_inference"] == 0 and fa["ticks_advanced_while_a_decision_was_held"] == 0
    lc = row["lifecycle"]
    assert lc["responses_ready"] == lc["requests_launched"] and lc["in_flight_at_end"] == 0


@pytest.fixture(scope="module")
def strands_warm() -> None:
    _policy_trial(C0, P0, slow=None, runtime="strands", fast_delay=0.01)


def test_p0_is_the_production_loop_first_ready_commits_first():
    row = _policy_trial(C2, P0, slow=ACTOR_A)
    _assert_policy_trial_rules(row)
    b = row["boundaries"][0]
    assert b["ready_order"] == b["apply_order"] == [ACTOR_B, ACTOR_A] and not b["same_tick_commit"]
    assert b["hold_ticks"] == 0 and row["freeze_audit"]["hold_ticks"] == 0
    assert b["contested_effects"] == [("DOOR_UNLOCKED", ACTOR_B)]
    # the slow answer was pinned to a tick boundary after the slow controller absorbed the door change in flight:
    # no handoff tick, no game time before the pair, no replacement request
    assert b["records"][ACTOR_A]["commit_tick"] >= b["records"][ACTOR_B]["commit_tick"] + 2
    assert row["freeze_audit"]["handoff_ticks_before_boundary0_pair"] == 0 and row["freeze_audit"]["game_time_before_boundary0_pair_s"] == 0
    slow_attempts = [e["attempt"] for e in row["lifecycle_events"] if e["actor_id"] == ACTOR_A]
    assert "CHANGE_ABSORBED_IN_FLIGHT" in slow_attempts and row["lifecycle"]["extra_requests"] == 0
    assert row["lifecycle"]["requests_launched"] == 2 == row["lifecycle"]["responses_consumed"]


@pytest.mark.parametrize("sid", (C2, C4, C6))
def test_p1_holds_the_first_answer_and_commits_both_on_one_tick_in_answer_order(sid):
    row = _policy_trial(sid, P1, slow=ACTOR_A)
    _assert_policy_trial_rules(row)
    b = row["boundaries"][0]
    assert b["ready_order"] == b["apply_order"] == [ACTOR_B, ACTOR_A]
    assert b["same_tick_commit"] and b["hold_ticks"] > 0
    assert b["records"][ACTOR_A]["commit_tick"] == b["release_tick"] == b["records"][ACTOR_B]["commit_tick"]
    # held ticks ran at dt=0; the release tick is the pair's one live tick
    assert b["game_time_at_commits"][0] == b["game_time_at_commits"][1] == row["boundaries"][0]["records"][ACTOR_B]["apply_game_clock"]["mission_time_remaining"]
    # the first commit waited for the slower answer (~270 ms here), then the pair committed together
    w = b["wall_ms"]
    assert w["to_first_commit"] - w["to_first_ready"] > 100 and w["to_pair_commit"] - w["to_first_commit"] < 100
    replay = replay_under_policy(SCENARIOS[sid], _raws(sid), P1, response_ready_order=(ACTOR_B, ACTOR_A), pair_id="t", decision_boundary_id="t:b0")
    assert b["world_hash_after_pair"] == replay["world_state_hash_after"]
    # per-decision staleness and consequence equal the zero-call replay's (the counterfactual ran on a copy, not the live game)
    for a in (ACTOR_A, ACTOR_B):
        assert {k: b["records"][a][k] for k in STALENESS_FIELDS} == {k: replay["records"][a][k] for k in STALENESS_FIELDS}, a
    assert b["records"][ACTOR_B]["staleness_class"] == "FRESH" and b["records"][ACTOR_B]["legal_at_snapshot"]


def test_live_door_decisions_are_fresh_first_and_stale_but_valid_second():
    for policy, slow in ((P0, ACTOR_A), (P0, ACTOR_B), (P1, ACTOR_B), (P3, ACTOR_B)):
        row = _policy_trial(C2, policy, slow=slow)
        _assert_policy_trial_rules(row)
        b = row["boundaries"][0]
        first, second = b["records"][b["first_commit_actor"]], b["records"][b["second_commit_actor"]]
        assert (first["staleness_class"], first["consequence"], first["time_cost"]) == ("FRESH", "ACCEPTED_UNCHANGED", 1.0)
        assert (second["staleness_class"], second["consequence"], second["time_cost"]) == ("STALE_BUT_VALID", "DIFFERENT_COST", 0.3)
        assert first["legal_at_snapshot"] and second["legal_at_snapshot"] and second["counterfactual"]["time_cost"] == 1.0
        assert b["first_result"] == "APPLIED/ACCEPTED_UNCHANGED" and b["second_result"] == "APPLIED/DIFFERENT_COST"


def test_counterfactual_refuses_a_state_carrying_probe_wrappers():
    from src.evaluation.commitment_probe import unpatched_copy

    s = SCENARIOS[C2]
    live = fresh_scenario_state(s.setup_fn)
    calls = []
    live.apply_action = lambda *a, **k: calls.append(a)  # what the probe does to the live GameState
    with pytest.raises(ValueError):
        counterfactual_on_state(live, ACTOR_A, s.scripted_raw(ACTOR_A), family=s.family)
    cf = counterfactual_on_state(unpatched_copy(live), ACTOR_A, s.scripted_raw(ACTOR_A), family=s.family)
    assert cf["engine_valid"] and cf["time_cost"] == 1.0 and cf["target_state_after"] == {"door_state": "open"}
    assert calls == [] and live.door_states["door_security_corridor"] == "locked"


@pytest.mark.parametrize("policy, first", ((P2, ACTOR_A), (P3, ACTOR_B)))
@pytest.mark.parametrize("slow", (ACTOR_A, ACTOR_B))
def test_fixed_order_policies_ignore_who_answered_first(policy, first, slow):
    row = _policy_trial(C2, policy, slow=slow)
    _assert_policy_trial_rules(row)
    b = row["boundaries"][0]
    assert b["first_commit_actor"] == first and b["ready_order"][1] == slow
    assert b["contested_effects"] == [("DOOR_UNLOCKED", first)]
    assert (b["first_action_cost"], b["second_action_cost"]) == (1.0, 0.3)
    replay = replay_under_policy(SCENARIOS[C2], _raws(C2), policy, response_ready_order=None, pair_id="t", decision_boundary_id="t:b0")
    assert b["world_hash_after_pair"] == replay["world_state_hash_after"]


@pytest.mark.parametrize("parity", (0, 1))
def test_p4_alternates_first_commit_across_two_synchronized_boundaries(parity):
    second = {ACTOR_A: {"action": "move", "target_id": "door_corridor_main_office", "reason": "t"}, ACTOR_B: {"action": "wait", "reason": "t"}}
    row = _policy_trial(C2, P4, slow=ACTOR_B, boundaries=2, p4_parity=parity, second=second)
    _assert_policy_trial_rules(row)
    b0, b1 = row["boundaries"]
    firsts = [b0["first_commit_actor"], b1["first_commit_actor"]]
    assert firsts == ([ACTOR_A, ACTOR_B] if parity == 0 else [ACTOR_B, ACTOR_A])
    # boundary 1 opened only once both actors were free, from one shared snapshot
    assert b1["synchronized_snapshot"] and b1["launch_ticks"][ACTOR_A] == b1["launch_ticks"][ACTOR_B] > b0["release_tick"]
    assert row["freeze_audit"]["align_ticks"] > 0 and row["freeze_audit"]["game_time_advanced_while_aligning_s"] > 0
    assert b1["records"][ACTOR_A]["decision"]["action"] == "move" and b1["records"][ACTOR_B]["decision"]["action"] == "wait"
    assert row["lifecycle"]["requests_launched"] == 4 == row["lifecycle"]["responses_consumed"]
    assert b0["contested_effects"] == [("DOOR_UNLOCKED", firsts[0])]


@pytest.mark.parametrize("policy", (P0, P1))
def test_same_tick_readiness_is_black_first_under_p0_and_p1(policy):
    import threading

    from src.evaluation.commitment_probe import run_policy_trial, scripted_sequence_native_client

    s = SCENARIOS[C2]
    gates = {ACTOR_A: threading.Event(), ACTOR_B: threading.Event()}
    clients = {a: scripted_sequence_native_client([(s.scripted_raw(a), 0.2, gates[a])]) for a in (ACTOR_A, ACTOR_B)}
    # both answers are made ready at the end of tick 1, so the loop first sees them on the same tick
    row = run_policy_trial(s, policy, trial_id="t", provider="mock", model_id="mock-model", client_overrides=clients, gates=gates,
                           decision_call_order="white_first", force_ready_at_tick=1)
    _assert_policy_trial_rules(row)
    b = row["boundaries"][0]
    assert b["same_tick_commit"] and b["apply_order"] == [ACTOR_A, ACTOR_B] and b["loop_ready_order"] is None


@pytest.mark.parametrize("runtime", ("native_tool_like", "strands"))
def test_hp09_race_happens_under_p0_and_the_barrier_removes_it(strands_warm, runtime):
    """The M6G.1 race, forced: the slow answer becomes ready right after the
    fast decision reached the loop. P0 had already applied it (the world
    changed), so the slow controller sees a new digest with nothing in
    flight and launches a replacement while its own answer is ready but
    uncommitted. A barrier had only held it (the world did not change), so
    no replacement is launched."""
    rows = {}
    for policy in (P0, P1, P4):
        rows[policy] = _policy_trial(C2, policy, slow=ACTOR_A, slow_delay=0.6, runtime=runtime, force_ready_after_opponent_step=(ACTOR_A,))
    p0 = rows[P0]
    assert p0["lifecycle"]["hp09_violations"] == 1 and p0["lifecycle"]["extra_requests"] == 1
    extra = next(r for r in p0["requests"][ACTOR_A] if r["launch_kind"] == "extra")
    assert extra["prior_state"] == "RESPONSE_READY_UNCOMMITTED" and extra["hp09_violation"] and not extra["consumed"]
    assert extra["ready"] and p0["lifecycle"]["unused_responses"] == 1 and p0["lifecycle"]["unused_output_tokens"] > 0
    # the stale decision was still applied: the door re-open at 0.3 s
    assert p0["boundaries"][0]["records"][ACTOR_A]["time_cost"] == 0.3
    for policy in (P1, P4):
        lc = rows[policy]["lifecycle"]
        assert lc["hp09_violations"] == 0 and lc["extra_requests"] == 0 and lc["unused_responses"] == 0
        assert lc["requests_launched"] == 2 == lc["responses_consumed"]
        _assert_policy_trial_rules(rows[policy])


def test_p0_and_p1_give_the_same_world_on_the_same_ready_order():
    p0, p1 = _policy_trial(C2, P0, slow=ACTOR_B), _policy_trial(C2, P1, slow=ACTOR_B)
    b0, b1 = p0["boundaries"][0], p1["boundaries"][0]
    assert b0["apply_order"] == b1["apply_order"] == [ACTOR_A, ACTOR_B]
    assert b0["effect_state_hash_after_pair"] == b1["effect_state_hash_after_pair"]
    assert b0["contested_effects"] == b1["contested_effects"]
    assert [b0["first_action_cost"], b0["second_action_cost"]] == [b1["first_action_cost"], b1["second_action_cost"]]


def test_lifecycle_events_record_every_launch_with_its_prior_state():
    row = _policy_trial(C2, P0, slow=ACTOR_A, slow_delay=0.6, force_ready_after_opponent_step=(ACTOR_A,))
    launches = [e for e in row["lifecycle_events"] if e["launched"]]
    assert [(e["actor_id"], e["launch_kind"], e["prior_state"]) for e in launches][:2] == [(ACTOR_A, "boundary", "IDLE"), (ACTOR_B, "boundary", "IDLE")]
    assert any(e["launch_kind"] == "extra" and e["prior_state"] == "RESPONSE_READY_UNCOMMITTED" and e["hp09_violation"] for e in launches)


# ---------------------------------------------------------------------------
# Live runner: frozen plan, stop rule and replay check (no network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runtime, per_cell, calls", (("native", {C2: 8, C0: 4}, 96), ("strands", {C2: 8}, 64)))
def test_live_plan_is_equalized_and_counterbalanced(runtime, per_cell, calls):
    from src.evaluation.run_m6g2_live import LIVE_POLICIES, expected_calls, planned_trials

    plan = planned_trials(runtime)
    assert len({t["trial_id"] for t in plan}) == len(plan)
    assert Counter((t["scenario_id"], t["policy"]) for t in plan) == {(s, p): n for s, n in per_cell.items() for p in LIVE_POLICIES}
    for sid, n in per_cell.items():
        for p in LIVE_POLICIES:
            cell = [t for t in plan if t["scenario_id"] == sid and t["policy"] == p]
            assert Counter(t["decision_call_order"] for t in cell) == {"black_first": n // 2, "white_first": n // 2}
            assert all(t["boundaries"] == (2 if p == P4 else 1) for t in cell)
            if p == P4:  # initial parity crossed with the call order
                assert Counter((t["p4_parity"], t["decision_call_order"]) for t in cell) == {
                    (par, o): n // 4 for par in (0, 1) for o in ("black_first", "white_first")}
    # the three policies run back to back within a repetition, in a rotating order
    chunks = [plan[i:i + 3] for i in range(0, len(plan), 3)]
    assert all(len({(t["rep"], t["scenario_id"]) for t in c}) == 1 and {t["policy"] for t in c} == set(LIVE_POLICIES) for c in chunks)
    assert {c[0]["policy"] for c in chunks} == set(LIVE_POLICIES)
    assert expected_calls(runtime) == calls


def test_live_call_plan_records_model_policies_counts_and_rules():
    from src.evaluation.run_m6g2_live import live_plan

    plan = json.loads(json.dumps(live_plan()))
    assert (plan["model_id"], plan["provider"], plan["region"]) == (STRANDS_MODEL, "bedrock", "us-east-1")
    assert set(plan["policies_live"]) == {P0, P1, P4} and set(plan["policies_replay_only"]) == {P2, P3}
    assert {k: (v["trials"], v["expected_model_calls"]) for k, v in plan["runtimes"].items()} == {"native": (36, 96), "strands": (24, 64)}
    assert "no_global_score" in plan["analysis_rules"] and "SOCIAL_OFF" in plan["social"]


def test_stop_rule_ignores_model_output_errors_and_stops_on_provider_errors_or_incomplete_trials():
    from src.evaluation.run_m6g2_live import provider_failure

    def row(errors_a=(), errors_b=(), complete=True):
        return {"complete": complete, "timed_out": not complete,
                "requests": {ACTOR_A: [{"provider_error_types": list(errors_a)}], ACTOR_B: [{"provider_error_types": list(errors_b)}]}}

    assert provider_failure(row()) is None
    assert provider_failure(row(["provider_invalid_response"], ["no_tool_selected"])) is None
    assert provider_failure(row((), ["ThrottlingException"])) == f"{ACTOR_B}: ThrottlingException"
    assert provider_failure(row(complete=False)) == "incomplete trial (timed out)"


@pytest.mark.parametrize("policy, slow, parity", ((P0, ACTOR_A, 0), (P0, ACTOR_B, 0), (P1, ACTOR_A, 0), (P4, ACTOR_A, 1)))
def test_replay_check_reproduces_a_live_boundary(policy, slow, parity):
    from src.evaluation.run_m6g2_live import replay_check

    row = _policy_trial(C2, policy, slow=slow, p4_parity=parity)
    check = replay_check(row, SCENARIOS[C2])
    assert check["replay_matches_live"] is True and check["replay_effect_matches_live"] is True
    assert check["replay_first_commit_actor"] == row["boundaries"][0]["first_commit_actor"]
    assert [tuple(e) for e in check["replay_contested_effects"]] == [tuple(e) for e in row["boundaries"][0]["contested_effects"]]


def test_live_runner_refuses_to_call_before_the_plan_is_frozen_and_never_overwrites_it(tmp_path, monkeypatch):
    import src.evaluation.run_m6g2_live as live

    monkeypatch.setattr(live, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(live, "out_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(live, "run_live_preflight", lambda *a, **k: pytest.fail("no provider call in tests"))
    monkeypatch.setattr(live, "run", lambda *a, **k: pytest.fail("no live run in tests"))
    monkeypatch.setattr(sys, "argv", ["run_m6g2_live", "--runtime", "native"])
    assert live.main() == 1
    monkeypatch.setattr(sys, "argv", ["run_m6g2_live", "--write-plan"])
    assert live.main() == 0
    frozen = (tmp_path / "live_call_plan.json").read_text()
    assert live.main() == 1 and (tmp_path / "live_call_plan.json").read_text() == frozen


SECOND_BOUNDARY = {ACTOR_A: {"action": "move", "target_id": "door_corridor_main_office", "reason": "t"}, ACTOR_B: {"action": "wait", "reason": "t"}}


@pytest.mark.parametrize("policy, kwargs", (
    (P0, {"slow": ACTOR_A}),
    (P1, {"slow": ACTOR_B}),
    (P4, {"slow": ACTOR_B, "boundaries": 2, "p4_parity": 1, "second": SECOND_BOUNDARY}),
    (P0, {"slow": ACTOR_A, "slow_delay": 0.6, "force_ready_after_opponent_step": (ACTOR_A,)}),  # the HP09 race
), ids=("p0", "p1", "p4_two_boundaries", "p0_hp09_race"))
def test_a_live_row_is_reproduced_with_zero_calls(policy, kwargs):
    from src.evaluation.run_m6g2_live import reproduce

    row = _policy_trial(C2, policy, **kwargs)
    _assert_policy_trial_rules(row)
    rep = reproduce(row, SCENARIOS[C2])
    assert rep["reproduced"], rep["mismatches"]
    assert rep["model_calls"] == 0 and rep["live_state_untouched_by_assembly"] and not rep["live_event_log_hash_differed"]
    assert all(not rec["live_value_differed"] for b in rep["boundaries"] for rec in b["records"].values())
    assert rep["reproduction_hp09_violations"] == row["lifecycle"]["hp09_violations"]


def test_reproduction_pins_a_same_tick_tie_and_corrects_wrong_derived_fields():
    import threading

    from src.evaluation.commitment_probe import run_policy_trial, scripted_sequence_native_client
    from src.evaluation.run_m6g2_live import reproduce

    s = SCENARIOS[C2]
    gates = {ACTOR_A: threading.Event(), ACTOR_B: threading.Event()}
    clients = {a: scripted_sequence_native_client([(s.scripted_raw(a), 0.2, gates[a])]) for a in (ACTOR_A, ACTOR_B)}
    row = run_policy_trial(s, P0, trial_id="t", provider="mock", model_id="mock-model", client_overrides=clients, gates=gates,
                           decision_call_order="white_first", force_ready_at_tick=1)
    assert row["boundaries"][0]["loop_ready_order"] is None
    rec = row["boundaries"][0]["records"][ACTOR_B]
    good = {k: rec[k] for k in ("staleness_class", "consequence", "legal_at_snapshot")}
    rec.update({"staleness_class": "INVALID_AT_SNAPSHOT", "consequence": "DIFFERENT_WORLD_EFFECT", "legal_at_snapshot": False})
    rep = reproduce(row, s)
    assert rep["reproduced"] and rep["pinned"]["same_tick_tie"], rep["mismatches"]
    fixed = rep["boundaries"][0]["records"][ACTOR_B]
    assert {k: fixed[k] for k in good} == good == {"staleness_class": "STALE_BUT_VALID", "consequence": "DIFFERENT_COST", "legal_at_snapshot": True}
    assert sorted(fixed["live_value_differed"]) == ["consequence", "legal_at_snapshot", "staleness_class"]


# ---------------------------------------------------------------------------
# Analysis (run_m6g2_analysis) on network-free rows
# ---------------------------------------------------------------------------


def _scripted_replay_rows() -> list[dict]:
    rows = []
    for sid in SCENARIO_IDS:
        for p, ready, parity in ((P0, (ACTOR_A, ACTOR_B), 0), (P0, (ACTOR_B, ACTOR_A), 0), (P1, (ACTOR_A, ACTOR_B), 0), (P1, (ACTOR_B, ACTOR_A), 0),
                                 (P2, None, 0), (P3, None, 0), (P4, None, 0), (P4, None, 1)):
            r = _replay(sid, p, ready, parity)
            r.update(source="scripted", runtime=None, scenario_id=sid, p4_assigned=None)
            rows.append(r)
    return rows


def test_analysis_replay_invariants_and_commitment_effect_by_scenario():
    from src.evaluation.run_m6g2_analysis import _staleness_for, commitment_effects, replay_invariants, replay_table_rows

    rows = _scripted_replay_rows()
    inv = replay_invariants(rows, {"n_pairs": 0, "by_source": {}, "invariant_violations": []})
    assert inv["pairs"] == len(SCENARIO_IDS) and inv["p0_equals_p1_same_ready_order"] == inv["pairs"]
    assert inv["decision_hashes_identical_across_policies"] == inv["snapshot_hashes_identical_across_policies"] == inv["pairs"]
    assert inv["rows_applying_the_policy_order"] == len(rows)
    eff = commitment_effects(rows)["by_scenario"]
    assert {sid: s["scenario_effect"] for sid, s in eff.items()} == EXPECTED_EFFECT
    assert eff[C2]["resource_to_first_commit_both_orders"] == "1/1" and eff[C2]["cost_changes"] == 1 and eff[C2]["effect_state_changes"] == 0
    assert eff[C5]["refusal_changes"] == 1 and eff[C5]["stale_and_invalid_moves_with_order"] == 1
    assert eff[C0]["final_state_changes"] == 0
    # the fixed orders put the stale-and-invalid consequence on one identity; the mirrored scenario moves it to the other
    p2 = _staleness_for([r for r in rows if r["commitment_policy"] == P2 and r["scenario_id"] in (C5, C5M)])
    assert p2["stale_and_invalid_borne_by"] == {ACTOR_A: 0, ACTOR_B: 2}
    groups = replay_table_rows([r for r in rows if r["scenario_id"] == C2])
    assert len(groups) == 2 and {g["first_commit"] for g in groups} == {ACTOR_A, ACTOR_B}


def test_analysis_live_blocks_on_scripted_trials():
    from src.evaluation.run_m6g2_analysis import (
        RT_NOT_CLASSIFIED,
        corrected_live,
        latency_table_rows,
        live_boundaries,
        live_commitment,
        request_waste,
        runtime_replication,
        wall_clock,
    )

    rows = []
    for policy, kw in ((P0, {"slow": ACTOR_A}), (P1, {"slow": ACTOR_B}), (P4, {"slow": ACTOR_B, "boundaries": 2, "p4_parity": 1, "second": SECOND_BOUNDARY})):
        row = _policy_trial(C2, policy, **kw)
        row.update(trial_id=f"t:{policy}:r00", rep=0, runtime_key="native", replay_matches_live=True)
        row.pop("lifecycle_events")
        rows.append(row)
    live = {"native": corrected_live(rows, {}), "strands": []}
    assert all(r["_derived_source"] == "live_probe_after_fix" for r in live["native"])
    lbs = live_boundaries(live)
    assert len(lbs) == 4
    lc = live_commitment(live, lbs)["native"]
    assert lc[P0][C2]["resource_to_first_commit"] == lc[P0][C2]["resource_contests"] == 1 and lc[P0][C2]["first_commit"][ACTOR_B] == 1
    assert lc[P1][C2]["same_tick_pair_commit"] == 1 and lc[P4][C2]["p4"]["balanced_trials"] == 1 == lc[P4][C2]["p4"]["follows_parity"]
    waste = request_waste(live, [])
    assert waste["by_runtime"]["native"][P4]["responses_consumed"] == 4 and waste["by_runtime"]["native"][P0]["unused_responses"] == 0
    clock = wall_clock(live, lbs)["native"]
    assert clock[P1]["game_time"]["game_time_advanced_during_inference_or_barrier"] is False and clock[P4]["b1"]["boundaries"] == 1
    assert runtime_replication(live, lbs)["overall"] == RT_NOT_CLASSIFIED
    assert {(g["policy"], g["boundary"]) for g in latency_table_rows(lbs)} == {(P0, 0), (P1, 0), (P4, 0), (P4, 1)}


def test_new_m6g2_sources_contain_no_secret_like_patterns():
    root = os.path.join(os.path.dirname(__file__), "..", "src", "evaluation")
    names = [n for n in os.listdir(root) if n.startswith(("commitment_", "run_m6g2_"))]
    pattern = re.compile(r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|bedrock-api-key-|aws_secret_access_key\s*=|ABSK[A-Za-z0-9+/=]{20,})")
    assert "commitment_policy.py" in names
    for name in names:
        with open(os.path.join(root, name), encoding="utf-8") as f:
            assert not pattern.search(f.read()), name
