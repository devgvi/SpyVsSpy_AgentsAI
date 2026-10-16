"""M6G.1: network-free tests for action temporal semantics, stale-decision
classification, T0/T1 commit replays, and the instrumented production
loop (mock clients only - no network)."""

from __future__ import annotations

import os
import re

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, fresh_scenario_state, world_state_snapshot
from src.evaluation.temporal_semantics import (
    ACCEPTED_UNCHANGED,
    CONTINUOUS_OR_MULTI_TICK,
    DIFFERENT_COST,
    DIFFERENT_WORLD_EFFECT,
    DURATION_AS_COST_ONLY,
    EFFECT_AT_START_WITH_BUSY_PERIOD,
    FALLBACK,
    FRESH,
    INSTANTANEOUS,
    INVALID_AT_SNAPSHOT,
    L0_NON_CONFLICTING,
    L1_SHARED_DOOR,
    L2_SHARED_OBJECT,
    L3_TRAP_SLOT_CONFLICT,
    L4_DISARM_CONFLICT,
    L5_ACTION_INVALIDATES_OTHER,
    L_FAMILIES,
    NO_OP,
    REFUSED,
    STALE_AND_INVALID,
    STALE_BUT_VALID,
    T0,
    T1A,
    T1B,
    WORLD_STALE_IRRELEVANT,
    build_temporal_scenarios,
    classify_consequence,
    classify_staleness,
    classify_temporal,
    effect_state_hash,
    relevant_state_hash,
    replay_pair,
    run_busy_variant,
    world_outcome,
)

SCENARIOS = build_temporal_scenarios()


def _raws(family: str) -> dict:
    s = SCENARIOS[family]
    return {ACTOR_A: s.raw_a, ACTOR_B: s.raw_b}


def _replay(family: str, condition: str, arrival=None, raws=None) -> dict:
    return replay_pair(SCENARIOS[family], raws or _raws(family), condition, arrival_order=arrival, pair_id=f"t:{family}:{condition}")


# ---------------------------------------------------------------------------
# Phase A: action temporal semantics, measured on the real Engine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probes() -> dict:
    from src.evaluation.run_m6g1_phase_b import duration_semantics

    return {p["label"]: p for p in duration_semantics()}


def _probe(probes: dict, prefix: str) -> dict:
    return next(p for label, p in probes.items() if label.startswith(prefix))


@pytest.mark.parametrize("prefix, cost", (
    ("INTERACT door: locked", constants.TIME_COST_UNLOCK),
    ("INTERACT search (desk_01)", constants.TIME_COST_SEARCH),
    ("INSPECT", constants.TIME_COST_INSPECT),
    ("PLACE_TRAP", constants.TRAP_PLACEMENT_COST["bomb"]),
    ("DISARM", constants.TIME_COST_DISARM),
))
def test_timed_actions_change_the_world_at_apply_then_lock_the_actor_out(probes, prefix, cost):
    p = _probe(probes, prefix)
    assert p["world_effect_at_apply"] is True
    assert p["time_cost_seconds"] == pytest.approx(cost)
    assert p["busy_after_apply"] and p["observation_reports_busy"]
    assert p["busy_unchanged_by_frozen_updates"] is True
    assert p["second_action_dropped_while_busy"] is True
    assert p["world_effect_at_completion"] is False and p["events_during_busy_and_at_completion"] == []
    assert p["busy_ticks_until_complete"] == pytest.approx(cost * constants.FPS, abs=1.5)
    assert classify_temporal(p) == EFFECT_AT_START_WITH_BUSY_PERIOD


def test_reopening_an_open_door_is_a_cost_with_no_world_effect(probes):
    p = _probe(probes, "INTERACT door: already open")
    assert p["time_cost_seconds"] == pytest.approx(constants.TIME_COST_OPEN)
    assert p["world_effect_at_apply"] is False and p["event_types_at_apply"] == ["DOOR_OPENED"]
    assert classify_temporal(p) == DURATION_AS_COST_ONLY


def test_move_is_the_only_multi_tick_action_and_never_locks_the_actor(probes):
    p = _probe(probes, "MOVE")
    assert classify_temporal(p) == CONTINUOUS_OR_MULTI_TICK
    assert p["multi_tick_ticks"] > 1 and p["moved_on_every_live_tick"] is True
    assert p["busy_after_apply"] is False and p["time_cost_seconds"] == 0


def test_a_sprung_trap_stuns_and_logs_the_only_completion_time_event(probes):
    p = _probe(probes, "INTERACT search springs")
    assert p["apply_outcome"] == "ALTERNATE_EFFECT" and p["busy_kind"] == "stunned"
    assert p["events_during_busy_and_at_completion"] == ["ACTOR_RECOVERED"]
    assert p["world_effect_at_completion"] is False


@pytest.mark.parametrize("prefix", ("WAIT", "INTERACT exit_door"))
def test_zero_cost_actions_are_instantaneous(probes, prefix):
    assert classify_temporal(_probe(probes, prefix)) == INSTANTANEOUS


def test_move_does_not_advance_on_frozen_ticks_and_follows_its_snapshot_path():
    from src.agents.action_lifecycle import ActionExecutor
    from src.actions import ActionType
    from src.observation import build_observation

    state = fresh_scenario_state(SCENARIOS[L1_SHARED_DOOR].setup_fn)
    obs = build_observation(state, ACTOR_A)
    ex = ActionExecutor()
    action, boundary, _ = ex.start(obs, "d", "move", "door_corridor_main_office", None, 0.0)
    assert boundary is None and action.type is ActionType.MOVE
    path = list(ex.active.path)
    x0 = (state.actors[ACTOR_A].x, state.actors[ACTOR_A].y)
    state.apply_action(ACTOR_A, action, 0.0)
    assert (state.actors[ACTOR_A].x, state.actors[ACTOR_A].y) == x0
    # the other actor opens a door mid-move: the committed path is not re-planned
    state.apply_action(ACTOR_B, __import__("src.actions", fromlist=["GameAction"]).GameAction(ActionType.INTERACT, target_id="door_security_corridor"), 1 / 60)
    ex.continue_(build_observation(state, ACTOR_A))
    assert ex.active.path == path


# ---------------------------------------------------------------------------
# Relevant state and classifiers
# ---------------------------------------------------------------------------


def test_relevant_state_ignores_unrelated_world_change():
    state = fresh_scenario_state(SCENARIOS[L0_NON_CONFLICTING].setup_fn)
    before = world_state_snapshot(state)
    from src.actions import ActionType, GameAction

    state.apply_action(ACTOR_A, GameAction(ActionType.INTERACT, target_id="desk_01"), 1 / 60)
    after = world_state_snapshot(state)
    white_decision = {"action": "interact", "target_id": "locker_01", "trap_type": None}
    assert relevant_state_hash(before, ACTOR_B, white_decision) == relevant_state_hash(after, ACTOR_B, white_decision)
    black_decision = {"action": "interact", "target_id": "desk_01", "trap_type": None}
    assert relevant_state_hash(before, ACTOR_A, black_decision) != relevant_state_hash(after, ACTOR_A, black_decision)


def test_staleness_classifier():
    assert classify_staleness(world_changed=False, relevant_changed=False, engine_valid_at_snapshot=True, legal_at_apply=True) == FRESH
    assert classify_staleness(world_changed=True, relevant_changed=False, engine_valid_at_snapshot=True, legal_at_apply=True) == WORLD_STALE_IRRELEVANT
    assert classify_staleness(world_changed=True, relevant_changed=True, engine_valid_at_snapshot=True, legal_at_apply=True) == STALE_BUT_VALID
    assert classify_staleness(world_changed=True, relevant_changed=True, engine_valid_at_snapshot=True, legal_at_apply=False) == STALE_AND_INVALID
    assert classify_staleness(world_changed=False, relevant_changed=False, engine_valid_at_snapshot=False, legal_at_apply=False) == INVALID_AT_SNAPSHOT


def test_consequence_classifier():
    base = {"first_pass_valid": True, "decision": {"action": "interact"}, "apply_outcome": "APPLIED", "effective_outcome": "APPLIED",
            "target_state_after": {"door_state": "open"}, "self_state_after": {"x": 1}, "time_cost": 1.0}
    cf = {"effective_outcome": "APPLIED", "target_state_after": {"door_state": "open"}, "self_state_after": {"x": 1}, "time_cost": 1.0}
    assert classify_consequence(base, cf) == ACCEPTED_UNCHANGED
    assert classify_consequence({**base, "time_cost": 0.3}, cf) == DIFFERENT_COST
    assert classify_consequence({**base, "self_state_after": {"x": 2}}, cf) == DIFFERENT_WORLD_EFFECT
    assert classify_consequence({**base, "apply_outcome": "REFUSED_NO_OP"}, cf) == REFUSED
    assert classify_consequence({**base, "decision": {"action": "wait"}}, cf) == NO_OP
    assert classify_consequence({**base, "first_pass_valid": False}, cf) == FALLBACK


# ---------------------------------------------------------------------------
# Phase B: scripted stale-decision matrix
# ---------------------------------------------------------------------------

EXPECTED_SECOND = {
    L0_NON_CONFLICTING: (WORLD_STALE_IRRELEVANT, ACCEPTED_UNCHANGED),
    L1_SHARED_DOOR: (STALE_BUT_VALID, DIFFERENT_COST),
    L2_SHARED_OBJECT: (WORLD_STALE_IRRELEVANT, ACCEPTED_UNCHANGED),
    L3_TRAP_SLOT_CONFLICT: (STALE_AND_INVALID, REFUSED),
    L4_DISARM_CONFLICT: (STALE_AND_INVALID, REFUSED),
}


@pytest.mark.parametrize("family", tuple(EXPECTED_SECOND))
@pytest.mark.parametrize("condition", (T1A, T1B))
def test_second_applied_decision_is_classified_as_expected(family, condition):
    row = _replay(family, condition)
    first, second = row["apply_order"]
    assert row["records"][first]["staleness_class"] == FRESH and row["records"][first]["consequence"] == ACCEPTED_UNCHANGED
    assert (row["records"][second]["staleness_class"], row["records"][second]["consequence"]) == EXPECTED_SECOND[family]


def test_l5_hidden_trap_turns_the_later_search_into_a_stun():
    a_first, b_first = _replay(L5_ACTION_INVALIDATES_OTHER, T1A), _replay(L5_ACTION_INVALIDATES_OTHER, T1B)
    searcher = a_first["records"][ACTOR_B]
    assert searcher["apply_outcome"] == "ALTERNATE_EFFECT" and searcher["staleness_class"] == STALE_AND_INVALID
    assert searcher["consequence"] == DIFFERENT_WORLD_EFFECT and searcher["listed_at_apply"] is True
    assert b_first["records"][ACTOR_B]["apply_outcome"] == "APPLIED"
    assert b_first["records"][ACTOR_A]["staleness_class"] == WORLD_STALE_IRRELEVANT


def test_door_cost_follows_application_order():
    for condition, first, second in ((T1A, ACTOR_A, ACTOR_B), (T1B, ACTOR_B, ACTOR_A)):
        row = _replay(L1_SHARED_DOOR, condition)
        assert row["records"][first]["target_state_before_apply"] == {"door_state": "locked"}
        assert row["records"][second]["target_state_before_apply"] == {"door_state": "open"}
        assert row["records"][first]["time_cost"] == constants.TIME_COST_UNLOCK
        assert row["records"][second]["time_cost"] == constants.TIME_COST_OPEN
        assert row["records"][second]["other_actor_busy_at_apply"] is True


@pytest.mark.parametrize("family", L_FAMILIES)
def test_arrival_timing_adds_nothing_beyond_application_order(family):
    assert world_outcome(_replay(family, T0, (ACTOR_A, ACTOR_B))) == world_outcome(_replay(family, T1A))
    assert world_outcome(_replay(family, T0, (ACTOR_B, ACTOR_A))) == world_outcome(_replay(family, T1B))


@pytest.mark.parametrize("family", (L1_SHARED_DOOR, L2_SHARED_OBJECT, L3_TRAP_SLOT_CONFLICT, L4_DISARM_CONFLICT, L5_ACTION_INVALIDATES_OTHER))
@pytest.mark.parametrize("first", (ACTOR_A, ACTOR_B))
def test_no_in_flight_state_second_outcome_is_the_same_during_or_after_the_first_busy_period(family, first):
    during = run_busy_variant(SCENARIOS[family], first, second_waits_for_completion=False)
    after = run_busy_variant(SCENARIOS[family], first, second_waits_for_completion=True)
    key = lambda r: (r["second"]["outcome"], r["second"]["event_types"], r["second"]["time_cost"], r["second"]["target_state_after"])  # noqa: E731
    assert key(during) == key(after)
    assert after["first_effect_unchanged_through_completion"] is True
    assert after["events_during_first_busy_period"] in ([], ["ACTOR_RECOVERED"])


def test_move_pairs_are_order_and_arrival_independent():
    raws = {ACTOR_A: {"action": "MOVE", "target_id": "desk_01"}, ACTOR_B: {"action": "move", "target_id": "door_main_office_ambassador"}}
    outcomes = {world_outcome(_replay(L3_TRAP_SLOT_CONFLICT, c, a, raws)) for c, a in ((T1A, None), (T1B, None), (T0, (ACTOR_A, ACTOR_B)), (T0, (ACTOR_B, ACTOR_A)))}
    assert len(outcomes) == 1


def test_invalid_first_pass_falls_back_and_is_labelled():
    raws = {ACTOR_A: {"action": "place_trap", "target_id": "ambassador_safe"}, ACTOR_B: {"action": "wait"}}
    row = _replay(L3_TRAP_SLOT_CONFLICT, T1A, raws=raws)
    assert row["records"][ACTOR_A]["consequence"] == FALLBACK and row["records"][ACTOR_A]["validation_error"] == "missing_target"


def test_replay_refuses_a_mismatched_request():
    with pytest.raises(RuntimeError):
        replay_pair(SCENARIOS[L1_SHARED_DOOR], _raws(L1_SHARED_DOOR), T1A, expected_request_hashes={ACTOR_A: "not-it", ACTOR_B: "not-it"})


# ---------------------------------------------------------------------------
# The production loop, instrumented, with mock clients (T0 reverification)
# ---------------------------------------------------------------------------


STRANDS_MODEL = "us.anthropic.claude-sonnet-4-6"


def _slow_mock(delay: float, response: dict, gate=None):
    from src.evaluation.real_loop_probe import scripted_native_client

    return scripted_native_client(response, delay, gate)


def _loop_trial(s, raws: dict, *, slow: str | None, runtime: str = "native_tool_like", slow_delay: float = 0.35, fast_delay: float = 0.08,
                gated: tuple = (), release_slow: bool = True, **kwargs) -> dict:
    """One scripted real-loop trial. The slow actor's answer (and any other
    gated actor's) is held behind a gate the probe opens, so which answer
    is ready first never depends on machine load; by default the slow gate
    opens right after the fast actor's commit is applied."""
    import threading

    from src.evaluation.real_loop_probe import run_real_loop_trial, scripted_native_client, scripted_strands_invoke

    gates = {a: threading.Event() for a in set(gated) | ({slow} if slow else set())}
    delays = {a: slow_delay if a == slow else fast_delay for a in (ACTOR_A, ACTOR_B)}
    if runtime == "strands":
        kwargs.update(strands_invoke_overrides={a: scripted_strands_invoke(raws[a], delays[a], gates.get(a)) for a in (ACTOR_A, ACTOR_B)},
                      provider="bedrock", model_id=STRANDS_MODEL, region="us-east-1")
    else:
        kwargs.update(client_overrides={a: scripted_native_client(raws[a], delays[a], gates.get(a)) for a in (ACTOR_A, ACTOR_B)},
                      provider="mock", model_id="mock-model")
    if slow and release_slow:
        kwargs["release_after_opponent_commit"] = (slow,)
    return run_real_loop_trial(s, trial_id="t", runtime=runtime, gates=gates, **kwargs)


def _assert_loop_rules(row: dict, s, *, move: bool = False) -> None:
    """What must hold on the production loop whatever the interleaving."""
    assert row["complete"] and row["pre_action_matches_harness_snapshot"] and not row["timed_out"]
    if row["same_tick_commit"]:  # both ready on one tick: Game._step applies Black then White
        assert row["apply_order"] == [ACTOR_A, ACTOR_B]
    else:
        assert row["ready_order"] == row["apply_order"]
    assert all(r["commit_matches_decision"] for r in row["records"].values())
    # No game time passes while a model call is in flight. The loop reads
    # the thinking flags after both get_action calls, so when the last
    # thread finishes inside that window one tick runs live before the
    # commit (a production handoff race, at most one 1/60 s tick).
    assert row["ticks_advanced_during_inference"] == 0 and row["handoff_ticks_before_second_apply"] <= 1
    assert row["game_time_frozen_until_second_apply"] is (row["handoff_ticks_before_second_apply"] == 0)
    # the only extra request the loop can make: a stale decision that became
    # ready after the opponent's commit is applied AND re-requested
    assert all(q["on_own_commit_tick"] and q["opponent_committed_before"] for q in row["extra_request_log"])
    assert row["extra_requests_before_pair_complete"] == len(row["extra_request_log"])
    assert row["reapplied_instant_actions_after_commit"] == 0
    raws = {a: row["records"][a]["raw_model_output"] for a in (ACTOR_A, ACTOR_B)}
    exact = replay_pair(s, raws, T0, observed_schedule=row["observed_commit_schedule"])
    assert (row["world_state_hash_after"], row["event_multiset_hash"]) == (exact["world_state_hash_after"], exact["event_multiset_hash"])
    ideal = replay_pair(s, raws, T1A) if row["same_tick_commit"] else replay_pair(s, raws, T0, arrival_order=tuple(row["apply_order"]))
    if not move:  # a handoff tick moves an in-progress MOVE one step further
        assert effect_state_hash(row["final_world"]) == ideal["effect_state_hash_after"]
    for a in (ACTOR_A, ACTOR_B):
        live, ex, ide = row["records"][a], exact["records"][a], ideal["records"][a]
        # the exact replay sees the very world each decision was applied to
        assert live["staleness_class"] == ex["staleness_class"]
        # an extra live tick changes clocks/positions, never relevant state:
        # it can only turn FRESH into WORLD_STALE_IRRELEVANT
        assert (live["relevant_state_changed"], live["legal_at_snapshot"], live["legal_at_apply"]) == (
            ide["relevant_state_changed"], ide["legal_at_snapshot"], ide["legal_at_apply"])
        if live["decision"]["action"] != "move":
            assert live["consequence"] == ex["consequence"] == ide["consequence"]


@pytest.fixture(scope="module")
def strands_warm() -> None:
    """The first Strands game in a process pays one-off initialization on
    the loop thread. Pay it once up front."""
    s = SCENARIOS[L0_NON_CONFLICTING]
    _loop_trial(s, _raws(L0_NON_CONFLICTING), slow=None, runtime="strands", fast_delay=0.01)


@pytest.mark.parametrize("family", (L1_SHARED_DOOR, L3_TRAP_SLOT_CONFLICT, L4_DISARM_CONFLICT, L5_ACTION_INVALIDATES_OTHER, L0_NON_CONFLICTING))
@pytest.mark.parametrize("slow", (ACTOR_A, ACTOR_B))
def test_strands_real_loop_also_commits_in_response_ready_order(strands_warm, family, slow):
    """StrandsAgentAdapter has its own copy of the thinking/digest/executor
    lifecycle; with its one SDK invocation scripted (no network) it must
    commit exactly like the Native controller does."""
    s = SCENARIOS[family]
    row = _loop_trial(s, _raws(family), slow=slow, runtime="strands")
    _assert_loop_rules(row, s)
    assert row["ready_order"][1] == slow and all(r["model_calls_for_first_decision"] == 1 for r in row["records"].values())


def test_scripted_real_loop_plan_covers_every_scenario_runtime_and_latency_assignment():
    from src.evaluation.run_m6g1_real_loop import RUNTIMES, scripted_plan

    plan = scripted_plan()
    assert len(plan) == len(set(plan)) == len(RUNTIMES) * len(L_FAMILIES) * 2
    assert {f for _, f, _ in plan} == set(L_FAMILIES)


def test_live_replay_checks_on_a_scripted_real_loop_row():
    from src.evaluation.run_m6g1_real_loop import REPLAY_LABELS, live_replay_checks, replays_for

    s = SCENARIOS[L1_SHARED_DOOR]
    row = _loop_trial(s, _raws(L1_SHARED_DOOR), slow=ACTOR_B)
    row.pop("final_world")
    replays = replays_for(row, s)
    assert tuple(replays) == REPLAY_LABELS
    checks = live_replay_checks(row, replays)
    assert checks["live_matches_t0_replay"] and checks["live_effect_matches_ideal_t0"] and checks["t0_replay_matches_same_order_barrier"]
    # reversing the order moves the unlock to the other actor
    assert replays["T0_reversed_arrival"]["contested_effects"] != replays["T0_observed_arrival"]["contested_effects"]
    assert replays["T1B"]["contested_effects"] == replays["T0_reversed_arrival"]["contested_effects"]


@pytest.mark.parametrize("family", (L1_SHARED_DOOR, L3_TRAP_SLOT_CONFLICT, L4_DISARM_CONFLICT, L5_ACTION_INVALIDATES_OTHER, L0_NON_CONFLICTING))
@pytest.mark.parametrize("slow", (ACTOR_A, ACTOR_B))
def test_real_loop_commits_in_response_ready_order_with_time_frozen(family, slow):
    s = SCENARIOS[family]
    row = _loop_trial(s, _raws(family), slow=slow)
    _assert_loop_rules(row, s)
    fast = ACTOR_B if slow == ACTOR_A else ACTOR_A
    assert row["ready_order"] == row["apply_order"] == [fast, slow]
    assert all(r["model_calls_for_first_decision"] == 1 for r in row["records"].values())


@pytest.mark.parametrize("mover", (ACTOR_A, ACTOR_B))
@pytest.mark.parametrize("slow", (ACTOR_A, ACTOR_B))
@pytest.mark.parametrize("force", (False, True))
def test_real_loop_move_decisions_are_reproduced_by_the_observed_schedule(mover, slow, force):
    """A MOVE is executed by the controller's ActionExecutor across ticks;
    its live displacement must be exactly what the replay produces."""
    s = SCENARIOS[L1_SHARED_DOOR]
    raws = {ACTOR_A: s.raw_a, ACTOR_B: s.raw_b, mover: {"action": "move", "target_id": "door_corridor_main_office", "reason": "t"}}
    row = _loop_trial(s, raws, slow=slow, release_slow=not force, force_handoff=(slow,) if force else (), force_handoff_min_age_s=0.6)
    _assert_loop_rules(row, s, move=True)
    assert row["records"][mover]["decision"]["action"] == "move"
    if force:
        assert row["handoff_ticks_before_second_apply"] == 1
    exact = replay_pair(s, {a: row["records"][a]["raw_model_output"] for a in raws}, T0, observed_schedule=row["observed_commit_schedule"])
    assert row["records"][mover]["self_state_after"] == exact["records"][mover]["self_state_after"]


@pytest.mark.parametrize("family", (L1_SHARED_DOOR, L5_ACTION_INVALIDATES_OTHER))
@pytest.mark.parametrize("slow", (ACTOR_A, ACTOR_B))
def test_forced_handoff_tick_is_counted_and_replayed_exactly(family, slow):
    """The handoff race pinned down: the slow decision becomes ready after
    its get_action but before the loop's thinking check, so one tick runs
    live and the commit lands a tick later."""
    s = SCENARIOS[family]
    fast = ACTOR_B if slow == ACTOR_A else ACTOR_A
    row = _loop_trial(s, _raws(family), slow=slow, release_slow=False, force_handoff=(slow,), force_handoff_min_age_s=0.6)
    _assert_loop_rules(row, s)
    assert row["apply_order"] == row["ready_order"] == [fast, slow]
    assert row["ticks_advanced_during_inference"] == 0 and row["handoff_ticks_before_second_apply"] == 1
    assert row["game_time_before_second_apply_s"] == pytest.approx(1 / 60, abs=1e-6)
    assert row["game_time_frozen_until_second_apply"] is False and row["same_tick_commit"] is False
    # exactly one live tick - an idle one - between the frozen first commit and the second commit
    schedule = row["observed_commit_schedule"]
    assert schedule[0] == [0.0, [fast]] and sum(1 for dt, _ in schedule[:-1] if dt > 0) == 1
    assert row["live_idle_ticks_before_second_commit"] == 1
    # The live handoff tick advances `opponent.last_seen.game_time` in the
    # slow actor's observation. L1: the opponent's door already changed that
    # observation while the request was in flight, so the digest was
    # refreshed and the commit is plain. L5: nothing visible had changed, so
    # the digest predates the handoff tick: the commit tick re-requests and,
    # with the fresh request in flight, lands on a frozen tick.
    if family == L1_SHARED_DOOR:
        assert row["extra_requests_before_pair_complete"] == 0 and schedule[-1] == [pytest.approx(1 / 60), [slow]]
    else:
        assert row["extra_requests_before_pair_complete"] == 1 and schedule[-1] == [0.0, [slow]]
        assert row["extra_request_log"][0]["actor_id"] == slow and row["extra_request_log"][0]["on_own_commit_tick"]
    raws = {a: row["records"][a]["raw_model_output"] for a in (ACTOR_A, ACTOR_B)}
    exact = replay_pair(s, raws, T0, observed_schedule=row["observed_commit_schedule"])
    ideal = replay_pair(s, raws, T0, arrival_order=(fast, slow))
    assert row["world_state_hash_after"] == exact["world_state_hash_after"] != ideal["world_state_hash_after"]
    assert effect_state_hash(row["final_world"]) == ideal["effect_state_hash_after"]
    assert row["records"][slow]["staleness_class"] == ideal["records"][slow]["staleness_class"]


def test_forced_same_tick_readiness_commits_black_then_white_on_one_live_tick():
    s = SCENARIOS[L1_SHARED_DOOR]
    row = _loop_trial(s, _raws(L1_SHARED_DOOR), slow=None, fast_delay=0.15, gated=(ACTOR_A, ACTOR_B),
                      decision_call_order="white_first", force_handoff=(ACTOR_A, ACTOR_B))
    _assert_loop_rules(row, s)
    assert row["same_tick_commit"] is True and row["apply_order"] == [ACTOR_A, ACTOR_B]
    # White was ready first, but decisions ready on the same tick apply Black-then-White
    assert row["ready_order"] == [ACTOR_B, ACTOR_A] and row["ready_order_equals_apply_order"] is False
    assert row["ticks_advanced_during_inference"] == 0 and row["handoff_ticks_before_second_apply"] == 1
    raws = {a: row["records"][a]["raw_model_output"] for a in (ACTOR_A, ACTOR_B)}
    exact = replay_pair(s, raws, T0, observed_schedule=row["observed_commit_schedule"])
    assert row["world_state_hash_after"] == exact["world_state_hash_after"]
    assert effect_state_hash(row["final_world"]) == replay_pair(s, raws, T1A)["effect_state_hash_after"]
    assert row["records"][ACTOR_A]["target_state_before_apply"] == {"door_state": "locked"}
    assert row["records"][ACTOR_B]["target_state_before_apply"] == {"door_state": "open"}


@pytest.mark.parametrize("runtime", ("native_tool_like", "strands"))
def test_decision_ready_right_after_the_opponent_commit_is_applied_stale_and_also_re_requested(strands_warm, runtime):
    """The interleaving two live Strands trials hit: the slow decision
    becomes ready after the opponent's commit changed its observation but
    before its own next get_action. The controller then sees a new digest
    with nothing in flight: it requests a fresh decision AND returns the
    stale one as the in-flight action, which the Engine applies."""
    s = SCENARIOS[L1_SHARED_DOOR]
    # slow answer > the controller's 0.5 s re-request throttle
    row = _loop_trial(s, _raws(L1_SHARED_DOOR), slow=ACTOR_A, runtime=runtime, slow_delay=0.6, release_slow=False,
                      force_ready_after_opponent_commit=(ACTOR_A,))
    _assert_loop_rules(row, s)
    assert row["ready_order"] == row["apply_order"] == [ACTOR_B, ACTOR_A]
    black = row["records"][ACTOR_A]
    assert black["commit_matches_decision"] and black["committed_action"]["type"] == "interact"
    assert (black["staleness_class"], black["consequence"], black["time_cost"]) == (STALE_BUT_VALID, DIFFERENT_COST, constants.TIME_COST_OPEN)
    assert row["extra_requests_before_pair_complete"] == 1
    assert row["extra_request_log"] == [{"actor_id": ACTOR_A, "tick": black["commit_tick"], "on_own_commit_tick": True,
                                         "opponent_committed_before": True, "returned": "interact"}]
    # the fresh request keeps the clock frozen: the stale decision lands on a frozen tick
    assert row["observed_commit_schedule"][-1] == [0.0, [ACTOR_A]] and row["game_time_frozen_until_second_apply"] is True
    raws = {a: row["records"][a]["raw_model_output"] for a in (ACTOR_A, ACTOR_B)}
    exact = replay_pair(s, raws, T0, observed_schedule=row["observed_commit_schedule"])
    assert (row["world_state_hash_after"], row["event_multiset_hash"]) == (exact["world_state_hash_after"], exact["event_multiset_hash"])


@pytest.mark.parametrize("handoff", (0, 1, 2))
def test_observed_schedule_replay_reproduces_live_handoff_ticks(handoff):
    s = SCENARIOS[L1_SHARED_DOOR]
    schedule = [[0.0, [ACTOR_B]], [0.0, []]] + [[1 / 60, []]] * handoff + [[1 / 60, [ACTOR_A]]]
    exact = replay_pair(s, _raws(L1_SHARED_DOOR), T0, observed_schedule=schedule)
    ideal = replay_pair(s, _raws(L1_SHARED_DOOR), T0, arrival_order=(ACTOR_B, ACTOR_A))
    assert exact["apply_order"] == [ACTOR_B, ACTOR_A]
    assert exact["effect_state_hash_after"] == ideal["effect_state_hash_after"]
    assert (exact["world_state_hash_after"] == ideal["world_state_hash_after"]) is (handoff == 0)
    assert exact["final_game_clock"]["mission_time_remaining"] == pytest.approx(
        ideal["final_game_clock"]["mission_time_remaining"] - handoff / 60)
    with pytest.raises(ValueError):
        replay_pair(s, _raws(L1_SHARED_DOOR), T0, observed_schedule=[[0.0, [ACTOR_B]]])


def test_timing_client_is_pass_through():
    from src.evaluation.real_loop_probe import _TimingClient
    from src.llm.clients.mock_client import MockModelClient

    inner = MockModelClient(response=[{"action": "wait"}])
    log: list = []
    wrapped = _TimingClient(inner, ACTOR_A, log)
    response = wrapped.generate_action("s", "u", {}, request_context={"x": 1})
    assert response.payload == {"action": "wait"} and log[0]["payload"] == {"action": "wait"}
    assert wrapped.model == inner.model


# ---------------------------------------------------------------------------
# M6G reuse, live plans, secret safety
# ---------------------------------------------------------------------------


def test_logical_arrival_order_uses_the_production_request_offset():
    from src.evaluation.run_m6g1_reuse import JOIN_OFFSET_MS, logical_arrival

    assert logical_arrival(1000.0, 2000.0)[0] == (ACTOR_A, ACTOR_B)
    assert logical_arrival(2000.0, 1000.0)[0] == (ACTOR_B, ACTOR_A)
    assert logical_arrival(1000.0 + JOIN_OFFSET_MS, 1000.0)[0] is None  # same tick: Black-first tie
    assert logical_arrival(1000.0, 1000.0 - JOIN_OFFSET_MS - 20)[0] == (ACTOR_B, ACTOR_A)


def test_m6g_pairs_rebuild_to_the_exact_requests_m6g_sent():
    from src.evaluation.run_m6g1_reuse import captured_pairs

    pairs = [p for p in captured_pairs() if p["runtime"] == "native"]
    if not pairs:
        pytest.skip("M6G outputs not present")
    p = pairs[0]
    row = replay_pair(SCENARIOS[p["scenario_family"]], p["raws"], T1A, expected_request_hashes=p["provider_request_hashes"], model_id="us.anthropic.claude-sonnet-4-6")
    assert all(row["records"][a]["provider_request_hash"] == p["provider_request_hashes"][a] for a in (ACTOR_A, ACTOR_B))


def test_real_loop_plan_alternates_scenarios_and_call_order():
    from src.evaluation.run_m6g1_real_loop import REPS, SCENARIOS as LIVE, planned_trials

    plan = planned_trials()
    assert len(plan) == REPS * len(LIVE)
    for family in LIVE:
        orders = [o for _, f, o in plan if f == family]
        assert orders.count("black_first") == orders.count("white_first") == REPS // 2


def test_real_loop_provider_failure_detection():
    from src.evaluation.run_m6g1_real_loop import provider_failure

    row = {"records": {ACTOR_A: {"actor_id": ACTOR_A, "provider_error_types": ["provider_invalid_response"]},
                       ACTOR_B: {"actor_id": ACTOR_B, "provider_error_types": []}}}
    assert provider_failure(row) is None
    row["records"][ACTOR_B]["provider_error_types"] = ["provider_throttling"]
    assert "provider_throttling" in provider_failure(row)


# ---------------------------------------------------------------------------
# Analysis rules
# ---------------------------------------------------------------------------


EXPECTED_CONFLICT = {
    L0_NON_CONFLICTING: "NO_TEMPORAL_CONFLICT",
    L1_SHARED_DOOR: "DURATION_COST_ASYMMETRY",
    L2_SHARED_OBJECT: "NO_TEMPORAL_CONFLICT",
    L3_TRAP_SLOT_CONFLICT: "STALE_AND_INVALID_EFFECT",
    L4_DISARM_CONFLICT: "STALE_AND_INVALID_EFFECT",
    L5_ACTION_INVALIDATES_OTHER: "STALE_AND_INVALID_EFFECT",
}


@pytest.mark.parametrize("family", L_FAMILIES)
def test_temporal_conflict_classification_of_the_scripted_pairs(family):
    from src.evaluation.run_m6g1_analysis import temporal_conflict

    tc = temporal_conflict(_replay(family, T1A), _replay(family, T1B))
    assert tc["classification"] == EXPECTED_CONFLICT[family]
    assert tc["order_sensitive_outcome"] is (family not in (L0_NON_CONFLICTING, L2_SHARED_OBJECT))
    assert temporal_conflict(_replay(family, T1A), _replay(family, T1B), busy_ok=False)["classification"] in ("IN_FLIGHT_CONFLICT", "MIXED")


def test_action_semantics_find_no_in_flight_state_and_no_completion_effects(probes):
    from src.evaluation.run_m6g1_analysis import action_semantics
    from src.evaluation.run_m6g1_phase_b import scripted_matrix

    busy = [r for r in scripted_matrix() if r["variant"].startswith("busy_")]
    sem = action_semantics(list(probes.values()), busy)
    assert sem["no_in_flight_state"] is True and sem["any_effect_at_completion"] is False
    assert sem["frozen_updates_never_drain_busy"] is True
    assert sem["classification_counts"] == {EFFECT_AT_START_WITH_BUSY_PERIOD: 7, DURATION_AS_COST_ONLY: 1, CONTINUOUS_OR_MULTI_TICK: 1, INSTANTANEOUS: 2}
    assert len(sem["busy_variants"]) == len(L_FAMILIES) * 2


def test_commit_rule_and_replication_rules():
    from src.evaluation.run_m6g1_analysis import RT_DEPENDENT, RT_INDEPENDENT, RT_NONE, RT_PARTIAL, _classify, _commit_rule_holds

    assert _commit_rule_holds({"same_tick_commit": False, "ready_order": [ACTOR_B, ACTOR_A], "apply_order": [ACTOR_B, ACTOR_A]})
    assert not _commit_rule_holds({"same_tick_commit": False, "ready_order": [ACTOR_B, ACTOR_A], "apply_order": [ACTOR_A, ACTOR_B]})
    assert _commit_rule_holds({"same_tick_commit": True, "ready_order": [ACTOR_B, ACTOR_A], "apply_order": [ACTOR_A, ACTOR_B]})
    assert _classify(10, 0, True) == RT_NONE
    assert _classify(10, 10, False) == RT_DEPENDENT
    assert _classify(10, 10, True, 1.0, 0.5) == RT_PARTIAL
    assert _classify(10, 10, True, 1.0, 0.9) == RT_INDEPENDENT


def test_door_rows_follow_the_application_order_rule():
    from src.evaluation.run_m6g1_analysis import _door_row

    for condition, first in ((T1A, ACTOR_A), (T1B, ACTOR_B), (T0, ACTOR_B)):
        row = _replay(L1_SHARED_DOOR, condition, (ACTOR_B, ACTOR_A) if condition == T0 else None)
        d = _door_row("t", row)
        assert d["order"].startswith(first)
        assert (d["door_state_before_first"], d["door_state_before_second"], d["first_cost"], d["second_cost"], d["final_door_state"]) == (
            "locked", "open", 1.0, 0.3, "open")
        assert d["first_actor_busy_at_second_apply"] is True
    assert _door_row("t", _replay(L0_NON_CONFLICTING, T1A)) is None


def test_generated_tables_are_spliced_between_markers_only(tmp_path):
    from src.evaluation.run_m6g1_analysis import TABLES_BEGIN, TABLES_END, splice_generated_tables

    path = tmp_path / "summary.md"
    path.write_text(f"# Title\n\nprose\n\n{TABLES_BEGIN}\nold\n{TABLES_END}\n\ntail\n", encoding="utf-8")
    assert splice_generated_tables(str(path), "### T\n\n| a |\n|---|\n| 1 |")
    once = path.read_text(encoding="utf-8")
    assert "old" not in once and "#### T" in once and once.startswith("# Title\n\nprose") and once.endswith("\n\ntail\n")
    assert splice_generated_tables(str(path), "### T\n\n| a |\n|---|\n| 1 |") and path.read_text(encoding="utf-8") == once
    plain = tmp_path / "plain.md"
    plain.write_text("no markers\n", encoding="utf-8")
    assert not splice_generated_tables(str(plain), "### T") and plain.read_text(encoding="utf-8") == "no markers\n"


def test_new_m6g1_sources_contain_no_secret_like_patterns():
    root = os.path.join(os.path.dirname(__file__), "..", "src", "evaluation")
    names = ["temporal_semantics.py", "real_loop_probe.py", "run_m6g1_phase_b.py", "run_m6g1_reuse.py", "run_m6g1_real_loop.py", "run_m6g1_analysis.py"]
    pattern = re.compile(r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|bedrock-api-key-|aws_secret_access_key\s*=|ABSK[A-Za-z0-9+/=]{20,})")
    for name in names:
        with open(os.path.join(root, name), encoding="utf-8") as f:
            assert not pattern.search(f.read()), name
