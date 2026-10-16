"""M6G (spec sections 3-24, 26, 44, 52): network-free tests for the
decision-scheduling harness, the conflict taxonomy's Engine semantics, the
production loop's real snapshot/apply ordering, and the live-call plumbing.
No test here makes a network call."""

from __future__ import annotations

import os
import re
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.conflict_scenarios import (
    C0_NON_CONFLICTING,
    C1_SAME_TARGET,
    C2_SHARED_DOOR,
    C3_SHARED_SPACE,
    C4_TRAP_CONFLICT,
    C5_INVALIDATES,
    CONFLICT_FAMILIES,
    build_conflict_scenarios,
)
from src.evaluation.decision_scheduling import (
    APPLY_APPLIED,
    APPLY_REFUSED_NO_OP,
    S0_FROZEN_SNAPSHOT_BASELINE,
    S1_REVERSED_CALL,
    S2_REVERSED_APPLY,
    S3_SEQUENTIAL_A_FIRST,
    S4_SEQUENTIAL_B_FIRST,
    SCHEDULE_PLANS,
    SCHEDULING_CONDITIONS,
    SNAPSHOT_FROZEN,
    SNAPSHOT_SEQUENTIAL,
    ReplayDecider,
    SchedulePlan,
    ScriptedDecider,
    build_request,
    classify_apply_outcome,
    native_render,
    replay_with_reversed_apply,
    run_condition,
)
from src.evaluation.duel_snapshot import (
    ACTOR_A,
    ACTOR_B,
    build_duel_snapshot,
    fresh_scenario_state,
    world_state_hash,
)
from src.evaluation.scheduling_analysis import (
    APPLICATION_ORDER_EFFECT,
    APPLY_TIME_INVALIDATION_EFFECT,
    MODEL_CALL_ORDER_EFFECT,
    NO_SCHEDULING_EFFECT,
    SEQUENTIAL_INFORMATION_EFFECT,
    UNINTERPRETABLE_REQUESTS_DIFFER,
    classify_apply_order_pair,
    classify_call_order,
    classify_snapshot_timing,
    event_multiset_hash,
    first_difference,
    permutation_p_value,
    tvd,
)
from src.llm_controller import NATIVE_TOOL_LIKE_SYSTEM_PROMPT
from src.observation import build_observation

SCENARIOS = build_conflict_scenarios()


def _scripted(family: str, **kwargs) -> ScriptedDecider:
    s = SCENARIOS[family]
    return ScriptedDecider({ACTOR_A: s.raw_a, ACTOR_B: s.raw_b}, **kwargs)


def _run(condition: str, family: str, decider=None, trial_id: str | None = None):
    decider = decider or _scripted(family)
    return run_condition(condition, SCENARIOS[family], decider, trial_id=trial_id or f"t:{family}:{condition}", model_id="test-model", provider_name="bedrock")


def _row(condition: str, family: str, decider=None):
    return _run(condition, family, decider).to_row()


# ---------------------------------------------------------------------------
# Explicit schedule plans (spec sections 3, 5-6; Q6)
# ---------------------------------------------------------------------------


def test_scheduling_conditions_constant_has_all_five():
    assert set(SCHEDULING_CONDITIONS) == set(SCHEDULE_PLANS) == {
        S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL, S2_REVERSED_APPLY, S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST,
    }


def test_plans_vary_exactly_one_axis_against_the_baseline():
    s0, s1, s2 = SCHEDULE_PLANS[S0_FROZEN_SNAPSHOT_BASELINE], SCHEDULE_PLANS[S1_REVERSED_CALL], SCHEDULE_PLANS[S2_REVERSED_APPLY]
    assert (s0.snapshot_timing, s0.call_order, s0.apply_order) == (SNAPSHOT_FROZEN, (ACTOR_A, ACTOR_B), (ACTOR_A, ACTOR_B))
    assert (s1.snapshot_timing, s1.apply_order) == (s0.snapshot_timing, s0.apply_order) and s1.call_order == (ACTOR_B, ACTOR_A)
    assert (s2.snapshot_timing, s2.call_order) == (s0.snapshot_timing, s0.call_order) and s2.apply_order == (ACTOR_B, ACTOR_A)
    s3, s4 = SCHEDULE_PLANS[S3_SEQUENTIAL_A_FIRST], SCHEDULE_PLANS[S4_SEQUENTIAL_B_FIRST]
    assert s3.snapshot_timing == s4.snapshot_timing == SNAPSHOT_SEQUENTIAL
    assert s3.call_order == (ACTOR_A, ACTOR_B) and s4.call_order == (ACTOR_B, ACTOR_A)


def test_sequential_plan_cannot_apply_out_of_decision_order():
    with pytest.raises(ValueError):
        SchedulePlan("bad", SNAPSHOT_SEQUENTIAL, (ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A))


def test_plan_must_name_each_actor_once():
    with pytest.raises(ValueError):
        SchedulePlan("bad", SNAPSHOT_FROZEN, (ACTOR_A, ACTOR_A), (ACTOR_A, ACTOR_B))


@pytest.mark.parametrize("condition", SCHEDULING_CONDITIONS)
def test_executed_trace_matches_declared_schedule(condition):
    result = _run(condition, C2_SHARED_DOOR)
    assert result.executed_steps() == SCHEDULE_PLANS[condition].declared_steps()


# ---------------------------------------------------------------------------
# Same-snapshot invariant, request identity, prebuilt requests (sections 7-8, 12-13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", (S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL, S2_REVERSED_APPLY))
def test_frozen_conditions_snapshot_both_actors_from_one_pre_action_world(condition):
    result = _run(condition, C2_SHARED_DOOR)
    a, b = result.records[ACTOR_A], result.records[ACTOR_B]
    assert result.same_snapshot is True
    assert a.world_state_hash_before_snapshot == b.world_state_hash_before_snapshot == result.world_state_hash_pre_action
    assert a.snapshot_id == b.snapshot_id
    assert result.second_actor_saw_first_action is False


def test_same_snapshot_flag_is_computed_not_assumed():
    """A sequential schedule whose first action changes nothing (WAIT) must
    report the same snapshot; one whose first action opens the door must
    not."""
    waiting = ScriptedDecider({ACTOR_A: {"action": "wait"}, ACTOR_B: {"action": "wait"}})
    assert _run(S3_SEQUENTIAL_A_FIRST, C2_SHARED_DOOR, waiting).same_snapshot is True
    assert _run(S3_SEQUENTIAL_A_FIRST, C2_SHARED_DOOR).same_snapshot is False


def test_s0_s1_requests_are_identical_per_actor_and_only_call_order_differs():
    s0, s1 = _run(S0_FROZEN_SNAPSHOT_BASELINE, C2_SHARED_DOOR), _run(S1_REVERSED_CALL, C2_SHARED_DOOR)
    for actor_id in (ACTOR_A, ACTOR_B):
        r0, r1 = s0.records[actor_id], s1.records[actor_id]
        assert r0.agent_request_hash == r1.agent_request_hash
        assert r0.agent_request_semantic_hash == r1.agent_request_semantic_hash
        assert r0.provider_request_hash == r1.provider_request_hash
        assert r0.apply_index == r1.apply_index
    assert (s0.records[ACTOR_A].decision_call_index, s0.records[ACTOR_B].decision_call_index) == (0, 1)
    assert (s1.records[ACTOR_A].decision_call_index, s1.records[ACTOR_B].decision_call_index) == (1, 0)


class _LoggingDecider(ScriptedDecider):
    def __init__(self, policy):
        super().__init__(policy)
        self.log: list[tuple[str, str]] = []

    def render(self, request):
        self.log.append(("render", request.actor_id))
        return super().render(request)

    def call(self, request, system_prompt, user_prompt):
        self.log.append(("call", request.actor_id))
        return super().call(request, system_prompt, user_prompt)


@pytest.mark.parametrize("condition", (S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL, S2_REVERSED_APPLY))
def test_both_requests_are_rendered_before_the_first_model_call(condition):
    s = SCENARIOS[C2_SHARED_DOOR]
    decider = _LoggingDecider({ACTOR_A: s.raw_a, ACTOR_B: s.raw_b})
    _run(condition, C2_SHARED_DOOR, decider)
    first_call = next(i for i, (kind, _) in enumerate(decider.log) if kind == "call")
    assert sorted(actor for kind, actor in decider.log[:first_call] if kind == "render") == sorted((ACTOR_A, ACTOR_B))


@pytest.mark.parametrize("condition", (S0_FROZEN_SNAPSHOT_BASELINE, S1_REVERSED_CALL, S2_REVERSED_APPLY))
def test_frozen_conditions_finish_every_decision_before_any_apply(condition):
    steps = [step for step, _ in _run(condition, C4_TRAP_CONFLICT).trace]
    assert max(i for i, s in enumerate(steps) if s == "decide") < min(i for i, s in enumerate(steps) if s == "apply")


def test_s2_replay_reuses_the_s0_decisions_and_only_flips_apply_order():
    s0 = _run(S0_FROZEN_SNAPSHOT_BASELINE, C4_TRAP_CONFLICT, trial_id="src")
    s2 = replay_with_reversed_apply(s0, SCENARIOS[C4_TRAP_CONFLICT], trial_id="replay", model_id="test-model", provider_name="bedrock")
    assert s2.condition == S2_REVERSED_APPLY and s2.decisions_reused_from == "src"
    assert s2.world_state_hash_pre_action == s0.world_state_hash_pre_action
    for actor_id in (ACTOR_A, ACTOR_B):
        assert s2.records[actor_id].decision == s0.records[actor_id].decision
        assert s2.records[actor_id].provider_request_hash == s0.records[actor_id].provider_request_hash
        assert s2.records[actor_id].decision_call_index == s0.records[actor_id].decision_call_index
        assert s2.records[actor_id].apply_index == 1 - s0.records[actor_id].apply_index
        assert s2.records[actor_id].latency_ms is None  # no model call happened in the replay
    assert s2.records[ACTOR_B].legal_at_apply_time is True and s0.records[ACTOR_B].legal_at_apply_time is False


def test_replay_refuses_a_decision_made_for_a_different_request():
    s0 = _run(S0_FROZEN_SNAPSHOT_BASELINE, C4_TRAP_CONFLICT)
    outputs = {a: {"raw": r.raw_model_output} for a, r in s0.records.items()}
    bad = ReplayDecider("src", "scripted", outputs, {ACTOR_A: "not-the-hash", ACTOR_B: "not-the-hash"})
    with pytest.raises(RuntimeError):
        _run(S2_REVERSED_APPLY, C4_TRAP_CONFLICT, bad)


# ---------------------------------------------------------------------------
# Sequential conditions (spec sections 9, 24, STOP item 26)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", (S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST))
def test_sequential_conditions_rebuild_the_second_snapshot_after_the_first_apply(condition):
    result = _run(condition, C2_SHARED_DOOR)
    first, second = SCHEDULE_PLANS[condition].call_order
    assert result.first_actor == first and result.second_actor == second
    assert result.records[first].world_state_hash_before_snapshot == result.world_state_hash_pre_action
    assert result.records[second].world_state_hash_before_snapshot != result.world_state_hash_pre_action
    assert result.records[first].snapshot_id != result.records[second].snapshot_id
    assert result.second_actor_saw_first_action is True
    assert result.trace.index(["apply", first]) < result.trace.index(["snapshot", second])


def test_a_changed_world_is_not_the_same_as_the_second_actor_seeing_it():
    """C0: A's search changes the world (searched_by, cooldown) but nothing
    B can observe - B's request is byte-identical to its S0 request."""
    s0, s3 = _run(S0_FROZEN_SNAPSHOT_BASELINE, C0_NON_CONFLICTING), _run(S3_SEQUENTIAL_A_FIRST, C0_NON_CONFLICTING)
    assert s3.world_changed_before_second_snapshot is True
    assert s3.second_actor_saw_first_action is False
    assert s3.records[ACTOR_B].provider_request_hash == s0.records[ACTOR_B].provider_request_hash


def test_hidden_trap_is_invisible_to_the_second_actor_and_refused_at_apply():
    s3 = _run(S3_SEQUENTIAL_A_FIRST, C4_TRAP_CONFLICT)
    b = s3.records[ACTOR_B]
    assert s3.second_actor_saw_first_action is False
    assert b.first_pass_valid and b.legal_at_snapshot and b.listed_legal_at_apply_time
    assert b.legal_at_apply_time is False and b.apply_outcome == APPLY_REFUSED_NO_OP


def test_visible_invalidation_is_caught_at_decision_time_in_a_sequential_schedule():
    """C5: once A disarms, B's rebuilt snapshot no longer offers DISARM, so
    B's fixed scripted DISARM fails validation there (fallback WAIT) instead
    of failing silently at apply time as it does in S0."""
    s0, s3 = _run(S0_FROZEN_SNAPSHOT_BASELINE, C5_INVALIDATES), _run(S3_SEQUENTIAL_A_FIRST, C5_INVALIDATES)
    assert s0.records[ACTOR_B].first_pass_valid and s0.records[ACTOR_B].legal_at_apply_time is False
    assert s0.records[ACTOR_B].listed_legal_at_apply_time is False
    assert s3.second_actor_saw_first_action is True
    assert s3.records[ACTOR_B].first_pass_valid is False and s3.records[ACTOR_B].fallback_to_wait is True


# ---------------------------------------------------------------------------
# Game time frozen during inference (spec section 11, STOP item 23)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", SCHEDULING_CONDITIONS)
def test_game_time_never_moves_during_a_decision_even_when_the_decider_is_slow(condition):
    slow = _run(condition, C2_SHARED_DOOR, _scripted(C2_SHARED_DOOR, delay_seconds=0.02))
    fast = _run(condition, C2_SHARED_DOOR)
    for rec in slow.records.values():
        assert rec.game_clock_before_call == rec.game_clock_after_call
        assert rec.game_clock_before_call == {"elapsed_time": 0.0, "mission_time_remaining": constants.MISSION_TIME}
        assert rec.world_unchanged_during_call is True
    assert slow.world_state_hash_after == fast.world_state_hash_after
    assert slow.event_log_hash == fast.event_log_hash


# ---------------------------------------------------------------------------
# Phase B Engine semantics - predeclared expectations (sections 14-23)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", CONFLICT_FAMILIES)
def test_both_scripted_actions_are_legal_at_the_shared_snapshot(family):
    result = _run(S0_FROZEN_SNAPSHOT_BASELINE, family)
    for rec in result.records.values():
        assert rec.first_pass_valid is True and rec.legal_at_snapshot is True


@pytest.mark.parametrize("family", CONFLICT_FAMILIES)
def test_predeclared_apply_order_expectation_holds(family):
    labels = classify_apply_order_pair(_row(S0_FROZEN_SNAPSHOT_BASELINE, family), _row(S2_REVERSED_APPLY, family))
    assert (APPLICATION_ORDER_EFFECT in labels) is SCENARIOS[family].expected_apply_order_effect


@pytest.mark.parametrize("family", CONFLICT_FAMILIES)
@pytest.mark.parametrize("condition", (S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST))
def test_predeclared_side_effect_visibility_holds(family, condition):
    assert _run(condition, family).second_actor_saw_first_action is SCENARIOS[family].expected_second_observation_changes


@pytest.mark.parametrize("family", CONFLICT_FAMILIES)
def test_call_order_has_no_engine_level_effect_for_fixed_decisions(family):
    s0, s1 = _row(S0_FROZEN_SNAPSHOT_BASELINE, family), _row(S1_REVERSED_CALL, family)
    assert (s0["world_state_hash_after"], s0["event_log_hash"]) == (s1["world_state_hash_after"], s1["event_log_hash"])
    assert first_difference(s0, s1) == "none"


@pytest.mark.parametrize("family", CONFLICT_FAMILIES)
def test_fixed_decisions_under_sequential_schedules_land_like_their_frozen_counterparts(family):
    """With decisions fixed, S3 resolves like S0 (A applied first) and S4
    like S2: any live S3/S4 difference must come from a changed decision."""
    assert _row(S3_SEQUENTIAL_A_FIRST, family)["world_state_hash_after"] == _row(S0_FROZEN_SNAPSHOT_BASELINE, family)["world_state_hash_after"]
    assert _row(S4_SEQUENTIAL_B_FIRST, family)["world_state_hash_after"] == _row(S2_REVERSED_APPLY, family)["world_state_hash_after"]


@pytest.mark.parametrize("family", (C0_NON_CONFLICTING, C1_SAME_TARGET))
def test_non_conflict_controls_differ_only_in_event_log_order(family):
    s0, s2 = _row(S0_FROZEN_SNAPSHOT_BASELINE, family), _row(S2_REVERSED_APPLY, family)
    assert s0["world_state_hash_after"] == s2["world_state_hash_after"]
    assert s0["event_log_hash"] != s2["event_log_hash"]
    assert event_multiset_hash(s0) == event_multiset_hash(s2)
    assert classify_apply_order_pair(s0, s2) == [NO_SCHEDULING_EFFECT]


def test_c3_movement_has_no_actor_actor_collision():
    s0, s2 = _run(S0_FROZEN_SNAPSHOT_BASELINE, C3_SHARED_SPACE), _run(S2_REVERSED_APPLY, C3_SHARED_SPACE)
    assert all(r.apply_outcome == APPLY_APPLIED for r in s0.records.values())
    assert s0.world_state_hash_after == s2.world_state_hash_after


def test_c2_second_interact_takes_the_cheaper_open_branch():
    for condition, first, second in ((S0_FROZEN_SNAPSHOT_BASELINE, ACTOR_A, ACTOR_B), (S2_REVERSED_APPLY, ACTOR_B, ACTOR_A)):
        result = _run(condition, C2_SHARED_DOOR)
        assert result.records[first].apply_event_types == ["DOOR_UNLOCKED", "DOOR_OPENED"]
        assert result.records[second].apply_event_types == ["DOOR_OPENED"]
        assert result.records[second].legal_at_apply_time is True


def test_engine_open_branch_never_checks_the_keycard_once_the_door_is_unlocked():
    """Spec section 19 audit: the apply-time path revalidates only its own
    handler preconditions against live state - an already-unlocked door
    opens for an actor who never held the keycard."""
    state = fresh_scenario_state(SCENARIOS[C2_SHARED_DOOR].setup_fn)
    state.actors[ACTOR_B].inventory.items.discard("security_keycard")
    state.apply_action(ACTOR_A, GameAction(ActionType.INTERACT, target_id="door_security_corridor"), 1 / 60)
    events_before = len(state.event_log)
    state.apply_action(ACTOR_B, GameAction(ActionType.INTERACT, target_id="door_security_corridor"), 1 / 60)
    assert [e["type"] for e in state.event_log[events_before:]] == ["DOOR_OPENED"]
    assert state.actors[ACTOR_A].action_cooldown == constants.TIME_COST_UNLOCK
    assert state.actors[ACTOR_B].action_cooldown == constants.TIME_COST_OPEN


@pytest.mark.parametrize("family", (C4_TRAP_CONFLICT, C5_INVALIDATES))
def test_second_trap_action_is_a_silent_refusal(family):
    """No event, no time charged, no inventory spent, no state change."""
    for condition, loser in ((S0_FROZEN_SNAPSHOT_BASELINE, ACTOR_B), (S2_REVERSED_APPLY, ACTOR_A)):
        result = _run(condition, family)
        rec = result.records[loser]
        assert rec.apply_outcome == APPLY_REFUSED_NO_OP and rec.apply_event_types == []
        assert rec.world_state_hash_before_apply == rec.world_state_hash_after_apply
        assert rec.legal_at_snapshot is True and rec.legal_at_apply_time is False


def test_first_mover_wins_the_shared_trap_slot_in_either_order():
    s0, s2 = _run(S0_FROZEN_SNAPSHOT_BASELINE, C4_TRAP_CONFLICT), _run(S2_REVERSED_APPLY, C4_TRAP_CONFLICT)
    assert [(e["type"], e["actor_id"]) for e in s0.events] == [("TRAP_PLACED", ACTOR_A)]
    assert [(e["type"], e["actor_id"]) for e in s2.events] == [("TRAP_PLACED", ACTOR_B)]
    assert APPLY_TIME_INVALIDATION_EFFECT in classify_apply_order_pair(s0.to_row(), s2.to_row())


def test_c5_detector_sees_the_invalidation_but_c4_placer_cannot():
    """legal_at_apply_time vs listed_legal_at_apply_time: the loser of the
    C5 disarm race would see DISARM disappear, the loser of the C4 trap
    race still sees PLACE_TRAP offered (the winner's trap is hidden)."""
    c4, c5 = _run(S0_FROZEN_SNAPSHOT_BASELINE, C4_TRAP_CONFLICT), _run(S0_FROZEN_SNAPSHOT_BASELINE, C5_INVALIDATES)
    assert c4.records[ACTOR_B].listed_legal_at_apply_time is True and c4.records[ACTOR_B].legal_at_apply_time is False
    assert c5.records[ACTOR_B].listed_legal_at_apply_time is False and c5.records[ACTOR_B].legal_at_apply_time is False


def test_deterministic_pair_hashes_are_reproducible():
    r1, r2 = _row(S0_FROZEN_SNAPSHOT_BASELINE, C4_TRAP_CONFLICT), _row(S0_FROZEN_SNAPSHOT_BASELINE, C4_TRAP_CONFLICT)
    assert (r1["world_state_hash_after"], r1["event_log_hash"]) == (r2["world_state_hash_after"], r2["event_log_hash"])
    assert r1["records"][ACTOR_A]["provider_request_hash"] == r2["records"][ACTOR_A]["provider_request_hash"]


def test_apply_outcome_classifier():
    assert classify_apply_outcome("wait", False, []) == "INTENTIONAL_WAIT"
    assert classify_apply_outcome("interact", True, ["OBJECT_SEARCHED"]) == "APPLIED"
    assert classify_apply_outcome("interact", True, ["TRAP_TRIGGERED", "ACTOR_STUNNED"]) == "ALTERNATE_EFFECT"
    assert classify_apply_outcome("interact", False, ["EXIT_ATTEMPTED"]) == "FAILURE_EVENT_ONLY"
    assert classify_apply_outcome("place_trap", False, []) == "REFUSED_NO_OP"


# ---------------------------------------------------------------------------
# Actor isolation / shared mutable state (spec section 26) and fixed policy inputs
# ---------------------------------------------------------------------------


def test_actor_requests_share_no_mutable_memory_and_do_not_affect_each_other():
    snap = build_duel_snapshot(SCENARIOS[C2_SHARED_DOOR].setup_fn)
    kwargs = dict(scenario_family=C2_SHARED_DOOR, model_id="m", provider_name="bedrock")
    req_a = build_request(ACTOR_A, snap.observation_a, snap.legal_actions_a, **kwargs)
    hash_a = req_a.request_semantic_hash()
    req_b = build_request(ACTOR_B, snap.observation_b, snap.legal_actions_b, **kwargs)
    assert req_a.memory_payload is not req_b.memory_payload
    assert req_a.request_semantic_hash() == hash_a
    assert build_request(ACTOR_A, snap.observation_a, snap.legal_actions_a, **kwargs).as_dict() == req_a.as_dict()


def test_measurement_observations_have_no_side_effects():
    state = fresh_scenario_state(SCENARIOS[C5_INVALIDATES].setup_fn)
    before, events = world_state_hash(state), len(state.event_log)
    build_observation(state, ACTOR_A)
    build_observation(state, ACTOR_B)
    assert world_state_hash(state) == before and len(state.event_log) == events


@pytest.mark.parametrize("family", CONFLICT_FAMILIES)
def test_every_request_is_social_off_default_personality_and_empty_memory(family):
    result = _run(S0_FROZEN_SNAPSHOT_BASELINE, family)
    snap = build_duel_snapshot(SCENARIOS[family].setup_fn)
    for actor_id, rec in result.records.items():
        assert all(e["action"] != "send_message" for e in snap.legal_actions[actor_id])
        req = build_request(actor_id, snap.observations[actor_id], snap.legal_actions[actor_id], scenario_family=family, model_id="test-model", provider_name="bedrock")
        assert req.personality_policy == "default"
        assert native_render(req)[0] == NATIVE_TOOL_LIKE_SYSTEM_PROMPT
        assert rec.memory_hash == result.records[actor_id].memory_hash
    assert result.records[ACTOR_A].personality_hash == result.records[ACTOR_B].personality_hash


@pytest.mark.parametrize("family", CONFLICT_FAMILIES)
def test_each_actor_is_placed_next_to_its_scripted_target(family):
    s = SCENARIOS[family]
    snap = build_duel_snapshot(s.setup_fn)
    for actor_id, raw in ((ACTOR_A, s.raw_a), (ACTOR_B, s.raw_b)):
        if raw["action"] != "move":
            assert snap.observations[actor_id].nearby_object_id == raw["target_id"]


# ---------------------------------------------------------------------------
# Baseline validation against the running production loop (spec section 44)
# ---------------------------------------------------------------------------


class _RecordingController:
    def __init__(self, game, actor_id: str, log: list):
        self.game, self.actor_id, self.log = game, actor_id, log

    def get_action(self, observation):
        self.log.append((self.actor_id, world_state_hash(self.game.state), observation.actor_id))
        return GameAction(ActionType.PLACE_TRAP, target_id="ambassador_safe", trap_type="bomb")


def _production_game(**kwargs):
    from src.game import Game

    game = Game(seed=6001, **kwargs)
    SCENARIOS[C4_TRAP_CONFLICT].setup_fn(game.state)
    game.state.update(0.0)
    return game


@pytest.mark.parametrize("decision_call_order", ("black_first", "white_first"))
def test_production_step_snapshots_both_actors_before_either_apply(decision_call_order):
    game = _production_game(white_controller="scripted", black_controller="scripted", decision_call_order=decision_call_order)
    pre = world_state_hash(game.state)
    log: list = []
    game.black_scripted_controller = _RecordingController(game, constants.BLACK_SPY_ID, log)
    game.bot_controller = _RecordingController(game, constants.WHITE_SPY_ID, log)
    game._step(1 / 60)
    expected_calls = [ACTOR_A, ACTOR_B] if decision_call_order == "black_first" else [ACTOR_B, ACTOR_A]
    assert [actor for actor, _, _ in log] == expected_calls
    assert all(h == pre for _, h, _ in log)
    assert all(actor == obs_actor for actor, _, obs_actor in log)


@pytest.mark.parametrize("decision_call_order", ("black_first", "white_first"))
def test_production_same_tick_apply_order_is_black_first_regardless_of_call_order(decision_call_order):
    game = _production_game(white_controller="scripted", black_controller="scripted", decision_call_order=decision_call_order)
    log: list = []
    game.black_scripted_controller = _RecordingController(game, constants.BLACK_SPY_ID, log)
    game.bot_controller = _RecordingController(game, constants.WHITE_SPY_ID, log)
    game._step(1 / 60)
    assert game.state.object_states["ambassador_safe"]["trap"]["owner_id"] == constants.BLACK_SPY_ID
    assert game.state.actors[constants.WHITE_SPY_ID].trap_inventory["bomb"] == 1


def _run_llm_duel_until_both_decided(slow_actor: str | None, delay: float = 0.25):
    from src.llm.clients.mock_client import MockModelClient

    class _SlowMock(MockModelClient):
        def __init__(self, seconds: float, **kw):
            super().__init__(**kw)
            self.seconds = seconds

        def generate_action(self, *args, **kwargs):
            time.sleep(self.seconds)
            return super().generate_action(*args, **kwargs)

    # M6G.3: this audit test locks the pre-promotion answer-order loop (P0);
    # the promoted barrier's counterpart is in tests/test_m6g3_response_barrier.py
    game = _production_game(white_controller="llm", black_controller="llm", llm_provider="mock", llm_model="mock-model", agent_runtime="native_tool_like",
                            commit_policy="answer_order")
    response = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "bomb", "reason": "test"}
    black, white = game.black_llm_controller.controller, game.bot_controller.controller
    black.model_client = _SlowMock(delay if slow_actor == ACTOR_A else 0.0, response=[dict(response)])
    white.model_client = _SlowMock(delay if slow_actor == ACTOR_B else 0.0, response=[dict(response)])
    clock_at_first_trap = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        game._step(1 / 60)
        if clock_at_first_trap is None and game.state.object_states["ambassador_safe"]["trap"] is not None:
            clock_at_first_trap = game.state.mission_time_remaining
        if black.decisions_completed and white.decisions_completed and not black._thinking_in_progress and not white._thinking_in_progress:
            for _ in range(2):
                game._step(1 / 60)
            break
        time.sleep(0.005)
    return game, clock_at_first_trap


@pytest.mark.parametrize("slow_actor, expected_winner", ((ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A)))
def test_production_llm_loop_applies_same_snapshot_decisions_in_completion_order(slow_actor, expected_winner):
    """Audit finding (scheduling_semantics_audit.md): in the live loop the
    effective application order is model completion order. The time freeze
    still holds - the fast actor's trap lands at frozen game time - but the
    faster model's decision wins the shared slot, and the slower actor's
    stale decision is then refused without a fresh decision."""
    game, clock_at_first_trap = _run_llm_duel_until_both_decided(slow_actor)
    trap = game.state.object_states["ambassador_safe"]["trap"]
    assert trap is not None and trap["owner_id"] == expected_winner
    assert clock_at_first_trap == constants.MISSION_TIME
    placed = [e["actor_id"] for e in game.state.event_log if e["type"] == "TRAP_PLACED"]
    assert placed == [expected_winner]


# ---------------------------------------------------------------------------
# Analysis helpers (sections 23-24, 41-42)
# ---------------------------------------------------------------------------


def test_tvd_and_permutation_test():
    same = ["a"] * 5 + ["b"] * 5
    assert tvd(same, list(same)) == 0.0 and permutation_p_value(same, list(same)) == 1.0
    assert tvd(["a"] * 10, ["b"] * 10) == 1.0
    p = permutation_p_value(["a"] * 10, ["b"] * 10)
    assert p < 0.001 and p == permutation_p_value(["a"] * 10, ["b"] * 10)
    assert permutation_p_value(["a"] * 4 + ["b"], ["a"] * 3 + ["b"] * 2) > 0.05


def test_first_difference_finds_the_earliest_layer():
    s0 = _row(S0_FROZEN_SNAPSHOT_BASELINE, C2_SHARED_DOOR)
    assert first_difference(s0, _row(S0_FROZEN_SNAPSHOT_BASELINE, C2_SHARED_DOOR)) == "none"
    assert first_difference(s0, _row(S2_REVERSED_APPLY, C2_SHARED_DOOR)) == "application_order"
    assert first_difference(s0, _row(S3_SEQUENTIAL_A_FIRST, C2_SHARED_DOOR)) == "snapshot"
    other = ScriptedDecider({ACTOR_A: {"action": "inspect", "target_id": "door_security_corridor"}, ACTOR_B: SCENARIOS[C2_SHARED_DOOR].raw_b})
    assert first_difference(s0, _row(S0_FROZEN_SNAPSHOT_BASELINE, C2_SHARED_DOOR, other)) == "raw_model_output"


def test_intent_label_separates_a_deliberate_wait_from_a_fallback_wait():
    from src.evaluation.scheduling_analysis import intent_label

    assert intent_label({"raw_model_output": {"action": "WAIT"}, "first_pass_valid": True, "validation_error": None}) == "wait"
    fallback = {"raw_model_output": {"action": "place_trap", "target_id": "ambassador_safe"}, "first_pass_valid": False, "validation_error": "missing_target"}
    assert intent_label(fallback) == "place_trap(ambassador_safe,None)[invalid:missing_target]"


def test_actor_order_classification_follows_who_goes_first():
    from src.evaluation.scheduling_analysis import ACTOR_ORDER_EFFECT, classify_actor_order

    result = classify_actor_order([_row(S3_SEQUENTIAL_A_FIRST, C2_SHARED_DOOR)] * 10, [_row(S4_SEQUENTIAL_B_FIRST, C2_SHARED_DOOR)] * 10)
    assert result["classification"] == [ACTOR_ORDER_EFFECT]
    assert result["contested_effects_s3"] == {f"DOOR_UNLOCKED:{ACTOR_A}": 10}
    assert result["contested_effects_s4"] == {f"DOOR_UNLOCKED:{ACTOR_B}": 10}
    assert result["same_starting_state"] and result["positional_symmetry"]


def _with_decision(row: dict, actor_id: str, label_action: str) -> dict:
    new = {**row, "records": {a: dict(r) for a, r in row["records"].items()}}
    new["records"][actor_id]["decision"] = {"action": label_action, "target_id": None, "trap_type": None}
    return new


def test_call_order_classification_needs_identical_requests_and_a_supported_shift():
    s0 = _row(S0_FROZEN_SNAPSHOT_BASELINE, C2_SHARED_DOOR)
    s1 = _row(S1_REVERSED_CALL, C2_SHARED_DOOR)
    assert classify_call_order([s0] * 10, [s1] * 10)["classification"] == NO_SCHEDULING_EFFECT
    shifted = [_with_decision(s1, ACTOR_A, "wait")] * 10
    assert classify_call_order([s0] * 10, shifted)["classification"] == MODEL_CALL_ORDER_EFFECT
    tampered = {**s1, "records": {a: dict(r) for a, r in s1["records"].items()}}
    tampered["records"][ACTOR_A]["provider_request_hash"] = "different"
    assert classify_call_order([s0] * 10, [tampered] * 10)["classification"] == UNINTERPRETABLE_REQUESTS_DIFFER


def test_snapshot_timing_classification_separates_information_from_invalidation():
    s0 = _row(S0_FROZEN_SNAPSHOT_BASELINE, C2_SHARED_DOOR)
    reacting = ScriptedDecider(
        lambda req: {"action": "move", "target_id": "door_security_corridor"}
        if req.actor_id == ACTOR_B and req.observation_payload["room"]["objects"] and any(
            o.get("id") == "door_security_corridor" and o.get("door_open") for o in req.observation_payload["room"]["objects"]
        )
        else {"action": "interact", "target_id": "door_security_corridor"}
    )
    s3 = _row(S3_SEQUENTIAL_A_FIRST, C2_SHARED_DOOR, reacting)
    assert s3["records"][ACTOR_B]["decision"]["action"] == "move"
    timing = classify_snapshot_timing([s0] * 10, [s3] * 10)
    assert timing["classification"] == [SEQUENTIAL_INFORMATION_EFFECT]
    assert timing["second_agent_action_change_rate"] == 1.0
    hidden = classify_snapshot_timing([_row(S0_FROZEN_SNAPSHOT_BASELINE, C4_TRAP_CONFLICT)] * 10, [_row(S3_SEQUENTIAL_A_FIRST, C4_TRAP_CONFLICT)] * 10)
    assert hidden["classification"] == [APPLY_TIME_INVALIDATION_EFFECT]
    assert hidden["second_actor_saw_first_action_rate"] == 0.0


# ---------------------------------------------------------------------------
# Live plumbing, network-free (spec section 52)
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, has_credentials: bool, **kwargs):
        self._has = has_credentials
        self.kwargs = kwargs

    def get_credentials(self):
        return object() if self._has else None


class _FakeResponse:
    def __init__(self, payload, error_type=None, error_message=None):
        self.payload, self.error_type, self.error_message = payload, error_type, error_message
        self.latency_ms, self.input_tokens, self.output_tokens, self.finish_reason = 12.3, 20, 5, "end_turn"


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        self.calls += 1
        return self.response


def test_auth_mode_precedence_matches_botocore():
    from src.evaluation.m6g_live import AUTH_BEARER, AUTH_CHAIN, AUTH_PROFILE, resolve_auth_mode

    ok = lambda **kw: _FakeSession(True, **kw)  # noqa: E731
    assert resolve_auth_mode({"AWS_BEARER_TOKEN_BEDROCK": "x", "AWS_PROFILE": "p"}, ok)["bedrock_auth_mode"] == AUTH_BEARER
    profile = resolve_auth_mode({"AWS_PROFILE": "gl-primitives"}, ok)
    assert profile["bedrock_auth_mode"] == AUTH_PROFILE and profile["aws_profile"] == "gl-primitives" and profile["credentials_resolved"]
    assert resolve_auth_mode({}, ok)["bedrock_auth_mode"] == AUTH_CHAIN
    assert resolve_auth_mode({"AWS_PROFILE": "p"}, lambda **kw: _FakeSession(False))["credentials_resolved"] is False


def test_profile_preflight_is_secret_safe_and_uses_one_real_client_call():
    from src.evaluation.m6g_live import run_live_preflight

    client = _FakeClient(_FakeResponse({"action": "wait"}))
    result = run_live_preflight("model-x", "us-east-1", env={"AWS_PROFILE": "gl-primitives"},
                                session_factory=lambda **kw: _FakeSession(True), client_factory=lambda: client)
    assert result["ok"] is True and result["bedrock_auth_mode"] == "aws_profile" and client.calls == 1
    assert result["token_present"] is False
    assert not any(re.search(r"secret|access_key|session_token|credential_value", k, re.I) for k in result)


def test_profile_preflight_fails_cleanly_without_credentials_and_makes_no_call():
    from src.evaluation.m6g_live import run_live_preflight

    client = _FakeClient(_FakeResponse({"action": "wait"}))
    result = run_live_preflight("model-x", "us-east-1", env={"AWS_PROFILE": "missing"},
                                session_factory=lambda **kw: _FakeSession(False), client_factory=lambda: client)
    assert result["ok"] is False and result["error_type"] == "provider_auth_error" and client.calls == 0


def test_native_decider_builds_a_fresh_client_for_every_call():
    from src.evaluation.m6g_live import NativeDecider

    made: list = []

    def factory():
        made.append(_FakeClient(_FakeResponse({"action": "wait"})))
        return made[-1]

    decider = NativeDecider("us-east-1", "model-x", client_factory=factory)
    result = _run(S0_FROZEN_SNAPSHOT_BASELINE, C2_SHARED_DOOR, decider)
    assert len(made) == 2 and made[0] is not made[1]
    assert all(r.runtime == "native" and r.decision["action"] == "wait" for r in result.records.values())


def test_provider_failures_abort_but_model_output_errors_are_scored():
    from src.evaluation.m6g_live import provider_failure

    row = _row(S0_FROZEN_SNAPSHOT_BASELINE, C2_SHARED_DOOR)
    assert provider_failure(row) is None
    row["records"][ACTOR_A]["provider_error_type"] = "provider_invalid_response"
    assert provider_failure(row) is None
    row["records"][ACTOR_B]["provider_error_type"] = "provider_throttling"
    assert "provider_throttling" in provider_failure(row)


def test_live_plans_are_paired_interleaved_and_balanced():
    from src.evaluation.run_m6g_native_replay import PHASE_C_SCENARIOS, PHASE_D_SCENARIOS, REPS, planned_trials
    from src.evaluation.run_m6g_strands_replay import CONDITIONS as STRANDS_CONDITIONS
    from src.evaluation.run_m6g_strands_replay import condition_order

    c = planned_trials("C")
    assert len(c) == REPS * len(PHASE_C_SCENARIOS) * 2
    for family in PHASE_C_SCENARIOS:
        firsts = [cond for rep, fam, pos, cond in c if fam == family and pos == 0]
        assert firsts.count(S0_FROZEN_SNAPSHOT_BASELINE) == firsts.count(S1_REVERSED_CALL) == REPS // 2
    d = planned_trials("D")
    assert len(d) == REPS * len(PHASE_D_SCENARIOS) * 3
    for cond in (S0_FROZEN_SNAPSHOT_BASELINE, S3_SEQUENTIAL_A_FIRST, S4_SEQUENTIAL_B_FIRST):
        assert sum(1 for _, _, _, x in d if x == cond) == REPS * len(PHASE_D_SCENARIOS)
    assert {condition_order(r)[0] for r in range(len(STRANDS_CONDITIONS))} == set(STRANDS_CONDITIONS)


def test_strands_decider_renders_the_same_payload_bytes_as_native():
    pytest.importorskip("strands")
    from src.agents.strands_agent import STRANDS_SYSTEM_PROMPT
    from src.evaluation.m6g_live import StrandsDecider

    snap = build_duel_snapshot(SCENARIOS[C2_SHARED_DOOR].setup_fn)
    req = build_request(ACTOR_A, snap.observation_a, snap.legal_actions_a, scenario_family=C2_SHARED_DOOR, model_id="m", provider_name="bedrock")
    system_prompt, user_prompt = StrandsDecider("us-east-1", "m").render(req)
    assert user_prompt == native_render(req)[1]
    assert system_prompt == STRANDS_SYSTEM_PROMPT


def test_new_m6g_sources_contain_no_secret_like_patterns():
    root = os.path.join(os.path.dirname(__file__), "..", "src", "evaluation")
    names = ["duel_snapshot.py", "conflict_scenarios.py", "decision_scheduling.py", "scheduling_analysis.py", "m6g_live.py",
             "run_m6g_phase_b.py", "run_m6g_native_replay.py", "run_m6g_provider_control.py", "run_m6g_strands_replay.py", "run_m6g_analysis.py"]
    pattern = re.compile(r"(AKIA[0-9A-Z]{16}|bedrock-api-key-|aws_secret_access_key\s*=|ABSK[A-Za-z0-9+/=]{20,})")
    for name in names:
        with open(os.path.join(root, name), encoding="utf-8") as f:
            assert not pattern.search(f.read()), name
