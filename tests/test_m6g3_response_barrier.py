"""M6G.3: network-free tests of the PROMOTED production scheduler - the
response barrier in Game._step (P1 BARRIER_ANSWER_ORDER, the default
commit_policy) and the HP09 guard in both controllers - plus the legacy
"answer_order" loop as the pre-promotion reference."""

from __future__ import annotations

import os
import re
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.barrier_probe import run_production_trial, scripted_requests
from src.evaluation.commitment_policy import C0, C2, C4, P0, SCENARIO_IDS, build_policy_scenarios, replay_under_policy
from src.evaluation.commitment_probe import observation_digest
from src.evaluation.decision_scheduling import DECISION_TICK_DT
from src.evaluation.duel_snapshot import ACTOR_A, ACTOR_B, ACTORS
from src.evaluation.real_loop_probe import SEED, _controller, build_game
from src.game import COMMIT_POLICY_ANSWER_ORDER, COMMIT_POLICY_BARRIER, DEFAULT_COMMIT_POLICY
from src.observation import build_observation

SCENARIOS = build_policy_scenarios()
STRANDS_MODEL = "us.anthropic.claude-sonnet-4-6"
RUNTIMES = ("native_tool_like", "strands")
DOOR = "door_security_corridor"
UNLOCK = GameAction(ActionType.INTERACT, target_id=DOOR)
WAIT_RAW = {"action": "wait", "reason": "t"}
THROTTLE_S = 0.5  # the controllers' re-request interval (decision_interval_seconds)


def _backend(runtime: str) -> dict:
    if runtime == "strands":
        return {"provider": "bedrock", "model_id": STRANDS_MODEL, "region": "us-east-1"}
    return {"provider": "mock", "model_id": "mock-model"}


def _trial(sid: str, groups, *, policy: str = DEFAULT_COMMIT_POLICY, runtime: str = "native_tool_like", raws: dict | None = None,
           gap: int = 1, absorb: bool = True, call_order: str = "black_first", **request_kwargs) -> dict:
    """One scripted production trial: request 1 of each actor answers its
    scenario decision (or `raws[actor]`), made ready in `groups` order."""
    s = SCENARIOS[sid]
    per_actor = {a: [[(raws or {}).get(a, s.scripted_raw(a))]] for a in (ACTOR_A, ACTOR_B)}
    kwargs, gates = scripted_requests(runtime, per_actor, **request_kwargs)
    return run_production_trial(s, trial_id="t", runtime=runtime, commit_policy=policy, gates=gates, ready_schedule=groups,
                                gap_ticks=gap, absorb=absorb, decision_call_order=call_order, **_backend(runtime), **kwargs)


def _scripted_game(sid: str, *, runtime: str = "native_tool_like", policy: str = DEFAULT_COMMIT_POLICY, per_actor: dict | None = None,
                   call_order: str = "black_first"):
    """The real two-LLM production Game on scenario `sid` for direct
    stepping; request i of each actor answers its scripted payload once
    its gate is set. Returns (game, agents, controllers, gates)."""
    s = SCENARIOS[sid]
    per_actor = per_actor or {a: [[s.scripted_raw(a)]] for a in ACTORS}
    kwargs, gates = scripted_requests(runtime, per_actor)
    game = build_game(s, runtime=runtime, decision_call_order=call_order, commit_policy=policy, **{"region": None, **_backend(runtime)})
    agents = {ACTOR_A: game.black_llm_controller, ACTOR_B: game.bot_controller}
    ctrls = {a: _controller(agent) for a, agent in agents.items()}
    for a in per_actor:
        if runtime == "strands":
            agents[a]._invoke_strands = kwargs["strands_invoke_overrides"][a]
        else:
            ctrls[a].model_client = kwargs["client_overrides"][a]
    return game, agents, ctrls, gates


def _ready(ctrl, gate) -> None:
    """Let one scripted request answer and wait until its thread published it."""
    gate.set()
    ctrl._thinking_thread.join(5.0)
    assert ctrl.response_ready_uncommitted and not ctrl._thinking_in_progress


def _release_all(ctrls: dict, gates: dict) -> None:
    for g in gates.values():
        g.set()
    for c in ctrls.values():
        thread = getattr(c, "_thinking_thread", None)
        if thread is not None:
            thread.join(5.0)


def _requested(ctrls: dict) -> dict:
    return {a: c.decisions_requested for a, c in ctrls.items()}


def _count_polls(agents: dict) -> dict:
    polls = {a: 0 for a in agents}
    for a, agent in agents.items():
        def counted(observation, a=a, original=agent.get_action):
            polls[a] += 1
            return original(observation)
        agent.get_action = counted
    return polls


def _pair(row: dict) -> dict:
    b = row["boundaries"][0]
    assert b["pair"], b
    return b


def _assert_clean(row: dict) -> None:
    assert row["complete"] and not row["timed_out"] and row["pre_action_matches_harness_snapshot"]
    lc = row["lifecycle"]
    assert lc["duplicate_requests"] == 0 and lc["unused_responses"] == 0 and lc["in_flight_at_end"] == 0
    assert row["freeze_audit"]["ticks_advanced_while_waiting"] == 0


@pytest.fixture(scope="module")
def strands_warm() -> None:
    _trial(C0, [[(ACTOR_A, 1), (ACTOR_B, 1)]], runtime="strands")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_barrier_is_the_default_and_unknown_policies_fail_loudly():
    from src.game import Game

    assert DEFAULT_COMMIT_POLICY == COMMIT_POLICY_BARRIER
    assert Game(seed=1).commit_policy == COMMIT_POLICY_BARRIER
    assert Game(seed=1, commit_policy="ANSWER_ORDER").commit_policy == COMMIT_POLICY_ANSWER_ORDER
    with pytest.raises(ValueError, match="Unknown commit_policy"):
        Game(seed=1, commit_policy="alternating_first")


@pytest.mark.parametrize("kwargs, active", (
    ({"white_controller": "llm", "black_controller": "llm"}, True),
    ({"white_controller": "llm", "black_controller": "llm", "commit_policy": COMMIT_POLICY_ANSWER_ORDER}, False),
    ({"white_controller": "llm", "black_controller": "scripted"}, False),
    ({"white_controller": "llm", "black_controller": "human"}, False),
    ({"white_controller": "llm", "black_controller": "llm", "llm_execution_mode": "playable"}, False),
))
def test_the_barrier_applies_only_between_two_llm_agents_in_evaluation_mode(kwargs, active):
    from src.game import Game

    game = Game(seed=1, llm_provider="mock", agent_runtime="native_tool_like", **kwargs)
    assert game._barrier_active() is active


@pytest.mark.parametrize("policy, guard", ((COMMIT_POLICY_BARRIER, True), (COMMIT_POLICY_ANSWER_ORDER, False)))
@pytest.mark.parametrize("runtime", ("native_tool_like", "strands"))
def test_the_hp09_guard_follows_the_commit_policy_on_both_runtimes(policy, guard, runtime):
    from src.game import Game

    game = Game(seed=1, white_controller="llm", black_controller="llm", agent_runtime=runtime, commit_policy=policy,
                **({"llm_provider": "bedrock", "llm_model": STRANDS_MODEL, "llm_region": "us-east-1"} if runtime == "strands" else {"llm_provider": "mock"}))
    for agent in (game.black_llm_controller, game.bot_controller):
        owner = agent.__dict__.get("controller", agent)
        assert (owner.config.ready_response_guard if runtime != "strands" else owner.ready_response_guard) is guard


# ---------------------------------------------------------------------------
# T1/T2 - both ready, either first: first-ready commits first, both on one tick
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runtime", ("native_tool_like", "strands"))
@pytest.mark.parametrize("first, second", ((ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A)), ids=("T1_black_first", "T2_white_first"))
def test_both_ready_first_ready_commits_first_on_one_tick(strands_warm, runtime, first, second):
    row = _trial(C2, [[(first, 1)], [(second, 1)]], runtime=runtime)
    _assert_clean(row)
    b = _pair(row)
    assert b["ready_order"] == b["commit_order"] == [first, second] and b["first_ready_is_first_commit"]
    assert b["same_tick_commit"] and not b["handed_over_same_tick"] and b["synchronized_snapshot"]
    assert b["contested_effects"] == [("DOOR_UNLOCKED", first)]
    assert (b["first_action_cost"], b["second_action_cost"]) == (1.0, 0.3)
    assert b["game_time_waited_s"] == {ACTOR_A: 0.0, ACTOR_B: 0.0} and b["first_commit_on_unchanged_snapshot"]
    assert row["freeze_audit"]["held_ticks"] > 0 and row["lifecycle"]["requests_launched"] == 2 == row["lifecycle"]["responses_consumed"]
    # the Game's own boundary identity: both requests belong to boundary 1
    assert b["decision_boundary_id"] == 1 and b["requests"] == {ACTOR_A: 1, ACTOR_B: 1}
    assert row["releases"][-1]["commit_order"] == [first, second]


# ---------------------------------------------------------------------------
# T0 - one response ready, the other pending: no commit, no replacement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_T0_a_ready_decision_is_held_without_commit_game_time_or_replacement(strands_warm, runtime):
    game, agents, ctrls, gates = _scripted_game(C2, runtime=runtime)
    polls = _count_polls(agents)
    clock = game.state.mission_time_remaining
    game._step(DECISION_TICK_DT)                                  # both requests launch from one snapshot
    assert _requested(ctrls) == {ACTOR_A: 1, ACTOR_B: 1} and game.request_boundary_ids == {ACTOR_A: 1, ACTOR_B: 1}
    _ready(ctrls[ACTOR_A], gates[(ACTOR_A, 1)])
    for _ in range(6):
        game._step(DECISION_TICK_DT)
    # A's decision was handed over once and is held by the barrier: RESPONSE_READY_UNCOMMITTED -> HELD
    assert list(game._held_decisions) == [ACTOR_A] and not ctrls[ACTOR_A].response_ready_uncommitted
    assert game._held_decisions[ACTOR_A][0].type is ActionType.INTERACT
    assert game.state.door_states[DOOR] == "locked" and game.state.mission_time_remaining == clock and game.last_barrier_release is None
    assert _requested(ctrls) == {ACTOR_A: 1, ACTOR_B: 1}
    assert polls == {ACTOR_A: 2, ACTOR_B: 7}                        # a held actor is not polled at all (it cannot re-request)
    _ready(ctrls[ACTOR_B], gates[(ACTOR_B, 1)])
    game._step(DECISION_TICK_DT)                                  # B handed over -> nothing outstanding -> release
    assert game.last_barrier_release == {"decision_boundary_id": 1, "commit_order": [ACTOR_A, ACTOR_B],
                                         "response_ready_at": {a: ctrls[a].response_ready_at for a in (ACTOR_A, ACTOR_B)}}
    assert not game._held_decisions and game.state.door_states[DOOR] == "open"
    assert [e["actor_id"] for e in game.state.event_log if e["type"] == "DOOR_UNLOCKED"] == [ACTOR_A]
    assert game.state.mission_time_remaining == pytest.approx(clock - DECISION_TICK_DT)   # the release tick is the only live one
    assert _requested(ctrls) == {ACTOR_A: 1, ACTOR_B: 1} and sum(c.ready_responses_cancelled for c in ctrls.values()) == 0


# ---------------------------------------------------------------------------
# T3 - near-simultaneous responses: ready order preserved, no duplicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_T3_answers_visible_on_one_tick_commit_in_their_true_ready_order(strands_warm, runtime):
    """Both answers land at the end of one tick, White's first: the barrier
    commits White first although Black is polled (and listed) first; the
    legacy loop applied such a same-tick pair Black first."""
    barrier = _trial(C2, [[(ACTOR_B, 1), (ACTOR_A, 1)]], runtime=runtime)
    legacy = _trial(C2, [[(ACTOR_B, 1), (ACTOR_A, 1)]], runtime=runtime, policy=COMMIT_POLICY_ANSWER_ORDER)
    for row in (barrier, legacy):
        _assert_clean(row)
    b, lg = _pair(barrier), _pair(legacy)
    assert b["handed_over_same_tick"] and not b["exact_ready_tie"]
    assert b["ready_order"] == b["commit_order"] == [ACTOR_B, ACTOR_A] and b["contested_effects"] == [("DOOR_UNLOCKED", ACTOR_B)]
    assert barrier["releases"][-1]["commit_order"] == [ACTOR_B, ACTOR_A]
    assert lg["handed_over_same_tick"] and lg["ready_order"] == [ACTOR_B, ACTOR_A] and lg["commit_order"] == [ACTOR_A, ACTOR_B]


@pytest.mark.parametrize("call_order", ("black_first", "white_first"))
def test_an_exact_ready_time_tie_falls_back_to_black_then_white(call_order):
    game, _, ctrls, gates = _scripted_game(C2, call_order=call_order)
    game._step(DECISION_TICK_DT)
    _ready(ctrls[ACTOR_B], gates[(ACTOR_B, 1)])
    _ready(ctrls[ACTOR_A], gates[(ACTOR_A, 1)])
    assert ctrls[ACTOR_B].response_ready_at < ctrls[ACTOR_A].response_ready_at
    ctrls[ACTOR_A].response_ready_at = ctrls[ACTOR_B].response_ready_at     # indistinguishable ready times
    game._step(DECISION_TICK_DT)
    assert game.last_barrier_release["commit_order"] == [ACTOR_A, ACTOR_B]
    assert [e["actor_id"] for e in game.state.event_log if e["type"] == "DOOR_UNLOCKED"] == [ACTOR_A]


# the M6G.1 race timings (spec section 16): the second answer lands after its
# 0.5 s re-request window, right after the first decision reached the loop
# (route 1: gap 0) or after an in-window poll saw that decision's effect
# without storing it (route 2: the throttle, gap 1)
RACE_ROUTES = pytest.mark.parametrize("gap", (0, 1), ids=("route1_answer_next_to_change", "route2_throttle"))


@RACE_ROUTES
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_T3_the_hp09_race_duplicates_under_the_legacy_loop_and_not_under_the_barrier(strands_warm, runtime, gap):
    groups = [[(ACTOR_A, 1)], [(ACTOR_B, 1)]]
    slow = {"delays": {(ACTOR_B, 1): THROTTLE_S + 0.1}}
    legacy = _trial(C2, groups, runtime=runtime, policy=COMMIT_POLICY_ANSWER_ORDER, gap=gap, absorb=False, **slow)
    lc = legacy["lifecycle"]
    assert lc["duplicate_requests"] == 1 and lc["unused_responses"] == 1 and lc["unused_output_tokens"] > 0
    extra = legacy["requests"][ACTOR_B][1]
    assert extra["prior_state"] == "RESPONSE_READY_UNCOMMITTED" and extra["hp09_violation"] and extra["unused"]
    assert _pair(legacy)["commit_order"] == [ACTOR_A, ACTOR_B]    # and the stale decision was still applied

    barrier = _trial(C2, groups, runtime=runtime, gap=gap, absorb=False, **slow)
    _assert_clean(barrier)
    assert barrier["lifecycle"]["requests_launched"] == 2 == barrier["lifecycle"]["responses_consumed"]
    b = _pair(barrier)
    assert b["commit_order"] == [ACTOR_A, ACTOR_B] and b["same_tick_commit"]
    assert all(r["prior_state"] == "IDLE" for a in ACTORS for r in barrier["requests"][a])


# ---------------------------------------------------------------------------
# T4 - the re-request throttle boundary is crossed while a response is ready
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_T4_crossing_the_throttle_window_while_ready_or_held_launches_nothing(strands_warm, runtime):
    game, _, ctrls, gates = _scripted_game(C2, runtime=runtime)
    clock = game.state.mission_time_remaining
    game._step(DECISION_TICK_DT)
    _ready(ctrls[ACTOR_A], gates[(ACTOR_A, 1)])
    time.sleep(THROTTLE_S + 0.1)                                  # A's window closes while its answer is ready-uncommitted
    game._step(DECISION_TICK_DT)
    assert list(game._held_decisions) == [ACTOR_A]
    time.sleep(THROTTLE_S + 0.1)                                  # ...and B's while A is held and B in flight
    for _ in range(3):
        game._step(DECISION_TICK_DT)
    assert _requested(ctrls) == {ACTOR_A: 1, ACTOR_B: 1} and game.state.mission_time_remaining == clock
    _ready(ctrls[ACTOR_B], gates[(ACTOR_B, 1)])
    time.sleep(THROTTLE_S + 0.1)
    game._step(DECISION_TICK_DT)
    assert game.last_barrier_release["commit_order"] == [ACTOR_A, ACTOR_B] and _requested(ctrls) == {ACTOR_A: 1, ACTOR_B: 1}
    assert game.state.door_states[DOOR] == "open"


# ---------------------------------------------------------------------------
# HP09 guard at the controller: both routes, both runtimes, guard on vs off
# ---------------------------------------------------------------------------


def _white(runtime: str, policy: str):
    """White's production agent from a real Game, answering two scripted gated requests."""
    raw = SCENARIOS[C2].scripted_raw(ACTOR_B)
    game, agents, ctrls, gates = _scripted_game(C2, runtime=runtime, policy=policy, per_actor={ACTOR_B: [[raw], [raw]]})
    return game, agents[ACTOR_B], ctrls[ACTOR_B], gates


def _assert_guarded_hand_over(agent, ctrl, after, policy: str) -> None:
    handed = agent.get_action(after)
    assert handed.type is ActionType.INTERACT and handed.target_id == DOOR and ctrl.last_call_committed_ready_response
    if policy == COMMIT_POLICY_BARRIER:
        # the ready decision is handed over; nothing replaced it...
        assert ctrl.decisions_requested == 1 and not ctrl._thinking_in_progress and not ctrl.response_ready_uncommitted
        # ...and the change (digest deliberately not stored) is re-evaluated by the next call: a fresh request
        agent.get_action(after)
        assert ctrl.decisions_requested == 2 and ctrl._thinking_in_progress and not ctrl.last_call_committed_ready_response
        assert ctrl._last_observation_digest == observation_digest(after)
    else:
        # pre-M6G.3: a replacement launched by the very call that handed the ready decision over
        assert ctrl.decisions_requested == 2 and ctrl._thinking_in_progress


@pytest.mark.parametrize("policy", (COMMIT_POLICY_BARRIER, COMMIT_POLICY_ANSWER_ORDER), ids=("guard_on", "guard_off"))
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_hp09_route1_an_answer_landing_next_to_an_observation_change(strands_warm, runtime, policy):
    game, agent, ctrl, gates = _white(runtime, policy)
    before = build_observation(game.state, ACTOR_B)
    agent.get_action(before)
    assert ctrl.decisions_requested == 1 and ctrl._thinking_in_progress
    _ready(ctrl, gates[(ACTOR_B, 1)])
    time.sleep(THROTTLE_S + 0.05)                                 # past the re-request window
    game.state.apply_action(ACTOR_A, UNLOCK, 0.0)                 # the other actor's commit lands next to the answer
    after = build_observation(game.state, ACTOR_B)
    assert observation_digest(after) != observation_digest(before)
    _assert_guarded_hand_over(agent, ctrl, after, policy)
    _release_all({ACTOR_B: ctrl}, gates)


@pytest.mark.parametrize("policy", (COMMIT_POLICY_BARRIER, COMMIT_POLICY_ANSWER_ORDER), ids=("guard_on", "guard_off"))
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_hp09_route2_a_change_seen_inside_the_throttle_window_and_consumed_after_it(strands_warm, runtime, policy):
    game, agent, ctrl, gates = _white(runtime, policy)
    before = build_observation(game.state, ACTOR_B)
    start = time.monotonic()
    agent.get_action(before)
    game.state.apply_action(ACTOR_A, UNLOCK, 0.0)
    after = build_observation(game.state, ACTOR_B)
    agent.get_action(after)                                       # seen in flight inside the window: not stored
    assert time.monotonic() - start < THROTTLE_S and ctrl._last_observation_digest == observation_digest(before)
    _ready(ctrl, gates[(ACTOR_B, 1)])
    time.sleep(max(0.0, THROTTLE_S + 0.05 - (time.monotonic() - start)))   # the window closes while it is ready-uncommitted
    _assert_guarded_hand_over(agent, ctrl, after, policy)
    _release_all({ACTOR_B: ctrl}, gates)


class _UnlockOnce:
    """A scripted Black that unlocks the shared door on its first tick, then waits."""

    def __init__(self) -> None:
        self.calls = 0

    def get_action(self, observation):
        self.calls += 1
        return UNLOCK if self.calls == 1 else GameAction(ActionType.WAIT)


@pytest.mark.parametrize("policy", (COMMIT_POLICY_BARRIER, COMMIT_POLICY_ANSWER_ORDER), ids=("guard_on", "guard_off"))
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_single_llm_game_has_no_barrier_but_keeps_the_guard(strands_warm, runtime, policy):
    """Against a scripted opponent the world keeps changing while the LLM
    thinks, so route 1 is reachable in the production loop itself."""
    from src.game import Game

    backend = _backend(runtime)
    game = Game(seed=SEED, white_controller="llm", black_controller="scripted", llm_provider=backend["provider"], llm_model=backend["model_id"],
                llm_region=backend.get("region"), agent_runtime=runtime, black_personality="default", white_personality="default",
                communication_enabled=False, commit_policy=policy)
    SCENARIOS[C2].setup_fn(game.state)
    game.state.update(0.0)
    game.black_scripted_controller = _UnlockOnce()
    raw = SCENARIOS[C2].scripted_raw(ACTOR_B)
    kwargs, gates = scripted_requests(runtime, {ACTOR_B: [[raw], [raw]]})
    ctrl = _controller(game.bot_controller)
    if runtime == "strands":
        game.bot_controller._invoke_strands = kwargs["strands_invoke_overrides"][ACTOR_B]
    else:
        ctrl.model_client = kwargs["client_overrides"][ACTOR_B]
    assert not game._barrier_active()
    game._step(DECISION_TICK_DT)                                  # White launches; Black unlocks (and opens) the door meanwhile
    assert ctrl.decisions_requested == 1 and game.state.door_states[DOOR] == "open"
    _ready(ctrl, gates[(ACTOR_B, 1)])
    time.sleep(THROTTLE_S + 0.05)
    game._step(DECISION_TICK_DT)                                  # White's next poll sees the changed door
    opened = [e["actor_id"] for e in game.state.event_log if e["type"] == "DOOR_OPENED"]
    assert ctrl.last_call_committed_ready_response and opened == [ACTOR_A, ACTOR_B]   # committed on this tick: no barrier
    assert ctrl.decisions_requested == (1 if policy == COMMIT_POLICY_BARRIER else 2)
    _release_all({ACTOR_B: ctrl}, gates)


# ---------------------------------------------------------------------------
# T5/T6 - world outcome identical to the legacy loop for the same ready order
# ---------------------------------------------------------------------------

PAIR_FIELDS = ("commit_order", "world_hash_before", "world_hash_after_first_commit", "world_hash_after_pair", "effect_state_hash_after_pair",
               "event_log_hash_pair", "contested_effects", "first_action_cost", "second_action_cost", "first_result", "second_result",
               "game_time_waited_s")
RECORD_FIELDS = ("decision_hash", "snapshot_world_hash", "apply_world_hash", "apply_outcome", "consequence", "staleness_class", "time_cost",
                 "legal_at_apply", "world_changed", "relevant_state_changed", "target_state_after", "self_state_after")


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("order", ((ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A)), ids=("black_ready_first", "white_ready_first"))
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_T5_T6_the_barrier_gives_the_legacy_world_for_the_same_ready_order(strands_warm, sid, order, runtime):
    groups = [[(order[0], 1)], [(order[1], 1)]]
    barrier = _trial(sid, groups, runtime=runtime)
    legacy = _trial(sid, groups, runtime=runtime, policy=COMMIT_POLICY_ANSWER_ORDER)
    for row in (barrier, legacy):
        _assert_clean(row)
    b, lg = _pair(barrier), _pair(legacy)
    assert b["ready_order"] == lg["ready_order"] == b["commit_order"] == list(order)
    assert {k: b[k] for k in PAIR_FIELDS} == {k: lg[k] for k in PAIR_FIELDS}
    for a in ACTORS:
        assert {k: b["records"][a][k] for k in RECORD_FIELDS} == {k: lg["records"][a][k] for k in RECORD_FIELDS}
    assert b["game_time_waited_s"] == {ACTOR_A: 0.0, ACTOR_B: 0.0}
    # what changes: both commit on one tick instead of the first on a frozen tick of its own
    assert b["same_tick_commit"] and not lg["same_tick_commit"]
    # and both equal the pre-promotion replay semantics (P0 with this ready order = M6G.1's T0)
    s = SCENARIOS[sid]
    rep = replay_under_policy(s, {a: s.scripted_raw(a) for a in ACTORS}, P0, response_ready_order=order, pair_id="t", decision_boundary_id="t:b0")
    assert rep["apply_order"] == list(order)
    assert (rep["world_state_hash_after"], rep["effect_state_hash_after"]) == (b["world_hash_after_pair"], b["effect_state_hash_after_pair"])
    assert (rep["first_action_cost"], rep["second_action_cost"], rep["first_result"], rep["second_result"]) == (
        b["first_action_cost"], b["second_action_cost"], b["first_result"], b["second_result"])
    if sid == C0:   # T5: the non-conflict pair touches nothing shared
        assert b["contested_effects"] == [] and all(not r["relevant_state_changed"] for r in b["records"].values())
    if sid == C2:   # T6: the shared door goes to the first commit
        assert b["contested_effects"] == [("DOOR_UNLOCKED", order[0])] and (b["first_action_cost"], b["second_action_cost"]) == (1.0, 0.3)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_held_move_decision_gives_the_legacy_world(strands_warm, runtime):
    """A MOVE handed over while the other decision is pending is held too;
    the legacy loop committed it on frozen ticks (no movement at dt 0),
    so both worlds after the pair agree - only the outcome label differs."""
    raws = {ACTOR_A: {"action": "move", "target_id": "desk_01", "reason": "t"}}
    groups = [[(ACTOR_A, 1)], [(ACTOR_B, 1)]]
    barrier = _trial(C4, groups, runtime=runtime, raws=raws)
    legacy = _trial(C4, groups, runtime=runtime, raws=raws, policy=COMMIT_POLICY_ANSWER_ORDER)
    for row in (barrier, legacy):
        _assert_clean(row)
    b, lg = _pair(barrier), _pair(legacy)
    assert b["records"][ACTOR_A]["decision"]["action"] == "move" and b["commit_order"] == lg["commit_order"] == [ACTOR_A, ACTOR_B]
    for k in ("world_hash_after_pair", "effect_state_hash_after_pair", "event_log_hash_pair", "contested_effects"):
        assert b[k] == lg[k], k
    assert b["records"][ACTOR_A]["apply_dt"] > 0 and lg["records"][ACTOR_A]["apply_outcome"] == "COMMITTED_FROZEN"
    assert b["records"][ACTOR_B]["apply_event_types"] == lg["records"][ACTOR_B]["apply_event_types"] == ["TRAP_PLACED"]


# ---------------------------------------------------------------------------
# Lifecycle edges: in-join answers, cancellation, rounds, the L521 counterpart
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_answer_inside_the_launching_call_is_handed_over_and_held(strands_warm, runtime):
    row = _trial(C2, [[(ACTOR_B, 1)]], runtime=runtime, delays={(ACTOR_A, 1): 0.0}, ungated=((ACTOR_A, 1),))
    _assert_clean(row)
    a_req = row["requests"][ACTOR_A][0]
    assert a_req["launch_tick"] == a_req["handed_over_tick"] == 1 and a_req["committed_tick"] > 1
    b = _pair(row)
    assert b["commit_order"] == [ACTOR_A, ACTOR_B] and b["same_tick_commit"] and b["game_time_waited_s"] == {ACTOR_A: 0.0, ACTOR_B: 0.0}


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_ready_decision_the_actor_can_no_longer_take_is_cancelled_with_a_reason(strands_warm, runtime):
    game, _, ctrls, gates = _scripted_game(C2, runtime=runtime)
    game._step(DECISION_TICK_DT)
    _ready(ctrls[ACTOR_A], gates[(ACTOR_A, 1)])
    game._step(DECISION_TICK_DT)
    assert list(game._held_decisions) == [ACTOR_A]
    game.state.actors[ACTOR_B].stunned_timer = 3.0                # B can no longer act on its pending decision
    _ready(ctrls[ACTOR_B], gates[(ACTOR_B, 1)])
    game._step(DECISION_TICK_DT)
    assert ctrls[ACTOR_B].ready_responses_cancelled == 1 and not ctrls[ACTOR_B].response_ready_uncommitted
    assert ctrls[ACTOR_B].last_ready_response_cancel_reason == "busy_stunned_or_mission_complete_at_hand_over"
    # nothing is outstanding any more, so A's held decision is released alone
    assert game.last_barrier_release["commit_order"] == [ACTOR_A] and not game._held_decisions
    assert [e["actor_id"] for e in game.state.event_log if e["type"] == "DOOR_OPENED"] == [ACTOR_A]
    assert _requested(ctrls) == {ACTOR_A: 1, ACTOR_B: 1}


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_boundary_ids_stay_consistent_over_two_decision_rounds(strands_warm, runtime):
    s = SCENARIOS[C0]
    per_actor = {a: [[s.scripted_raw(a)], [WAIT_RAW]] for a in ACTORS}
    kwargs, gates = scripted_requests(runtime, per_actor, ungated=((ACTOR_A, 2), (ACTOR_B, 2)))
    row = run_production_trial(s, trial_id="t", runtime=runtime, gates=gates, ready_schedule=[[(ACTOR_A, 1)], [(ACTOR_B, 1)]], rounds=2,
                               **_backend(runtime), **kwargs)
    _assert_clean(row)
    assert row["lifecycle"]["requests_launched"] == 4 == row["lifecycle"]["responses_consumed"]
    bounds = row["boundaries"]
    assert [b["decision_boundary_id"] for b in bounds] == list(range(1, len(bounds) + 1))
    assert bounds[0]["pair"] and bounds[0]["requests"] == {ACTOR_A: 1, ACTOR_B: 1}
    assert {a: sum(b["requests"][a] for b in bounds) for a in ACTORS} == {ACTOR_A: 2, ACTOR_B: 2}
    # one release per boundary, in the boundary's own commit order; every member decided from one world
    assert [(r["decision_boundary_id"], r["commit_order"]) for r in row["releases"]] == [(b["decision_boundary_id"], b["commit_order"]) for b in bounds]
    assert all(b["synchronized_snapshot"] and set(b["game_time_waited_s"].values()) == {0.0} for b in bounds)
    assert {a: [r["record"]["decision"]["action"] for r in row["requests"][a][:2]] for a in ACTORS} == {a: ["interact", "wait"] for a in ACTORS}


def test_the_promoted_loop_commits_the_first_ready_decision_first_at_frozen_game_time():
    """test_m6g_scheduling's audit test (the fast model wins the shared
    slot) under the promoted default: the fast decision is held until the
    slow one is handed over, then both commit on one tick, fast first,
    with no game time elapsed before that tick."""
    from src.game import Game
    from src.llm.clients.mock_client import MockModelClient

    class _SlowMock(MockModelClient):
        def __init__(self, seconds: float, **kw):
            super().__init__(**kw)
            self.seconds = seconds

        def generate_action(self, *args, **kwargs):
            time.sleep(self.seconds)
            return super().generate_action(*args, **kwargs)

    for slow, winner in ((ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A)):
        game = Game(seed=6001, white_controller="llm", black_controller="llm", llm_provider="mock", llm_model="mock-model",
                    agent_runtime="native_tool_like")
        SCENARIOS[C4].setup_fn(game.state)
        game.state.update(0.0)
        raw = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "bomb", "reason": "test"}
        black, white = game.black_llm_controller.controller, game.bot_controller.controller
        black.model_client = _SlowMock(0.25 if slow == ACTOR_A else 0.0, response=[dict(raw)])
        white.model_client = _SlowMock(0.25 if slow == ACTOR_B else 0.0, response=[dict(raw)])
        traps: list = []
        original_apply = game.state.apply_action

        def apply(actor_id, action, dt, original_apply=original_apply, game=game, traps=traps):
            if action.type is ActionType.PLACE_TRAP:
                traps.append((actor_id, game.state.mission_time_remaining, dt))
            original_apply(actor_id, action, dt)

        game.state.apply_action = apply
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and game.last_barrier_release is None:
            game._step(1 / 60)
            time.sleep(0.005)
        assert game.last_barrier_release["commit_order"] == [winner, slow]
        assert traps[:2] == [(winner, constants.MISSION_TIME, 1 / 60), (slow, constants.MISSION_TIME, 1 / 60)]
        assert game.state.object_states["ambassador_safe"]["trap"]["owner_id"] == winner
        assert [e["actor_id"] for e in game.state.event_log if e["type"] == "TRAP_PLACED"] == [winner]
        assert black.decisions_requested == white.decisions_requested == 1


# ---------------------------------------------------------------------------
# Replay equivalence and live runners (no network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ((ACTOR_A, ACTOR_B), (ACTOR_B, ACTOR_A), None), ids=("black_first", "white_first", "same_tick_tie"))
def test_replay_runner_finds_the_promoted_loop_equivalent_on_a_captured_pair(order):
    from src.evaluation.run_m6g3_replay import _groups, _p0_summary, _summary, _trial, compare

    s = SCENARIOS[C2]
    p = {"pair_id": "t", "source_trial_id": "t", "runtime": "native", "raws": {a: s.scripted_raw(a) for a in ACTORS},
         "response_ready_order": list(order) if order else None, "decision_hashes": None, "snapshot_hashes": None}
    post = _summary(_trial(s, p, COMMIT_POLICY_BARRIER, _groups(p["response_ready_order"])))
    pre = _summary(_trial(s, p, COMMIT_POLICY_ANSWER_ORDER, _groups(p["response_ready_order"])))
    p0 = _p0_summary(replay_under_policy(s, p["raws"], P0, response_ready_order=order, pair_id="t", decision_boundary_id="t:b0"))
    out = compare(p, post, pre, p0, None)
    assert out["equivalent"] and out["all_checks_pass"], out["failed_checks"]
    assert post["commit_order"] == (list(order) if order else [ACTOR_A, ACTOR_B])
    # a broken world hash is caught
    out = compare(p, post, {**pre, "world_hash_after_pair": "x"}, p0, None)
    assert not out["equivalent"] and "world" in out["failed_checks"]


def test_live_plan_is_interleaved_counterbalanced_and_capped():
    from collections import Counter

    from src.evaluation.run_m6g3_live import PRODUCTION_FILES, expected_calls, live_plan, max_calls, planned_trials

    for runtime in ("native", "strands"):
        plan = planned_trials(runtime)
        assert len({t["trial_id"] for t in plan}) == len(plan) == 15
        assert Counter(t["scenario_id"] for t in plan) == {C2: 10, C0: 5}
        assert Counter((t["scenario_id"], t["decision_call_order"]) for t in plan) == {
            (C2, "black_first"): 5, (C2, "white_first"): 5, (C0, "black_first"): 3, (C0, "white_first"): 2}
        assert [t["scenario_id"] for t in plan[:3]] == [C2, C0, C2]
        assert all(t["rounds"] == 2 and t["commit_policy"] == DEFAULT_COMMIT_POLICY for t in plan)
        assert (expected_calls(runtime), max_calls(runtime)) == (60, 150)
    plan = live_plan()
    assert (plan["model_id"], plan["provider"], plan["region"]) == (STRANDS_MODEL, "bedrock", "us-east-1")
    assert set(plan["production_code_sha256"]) == set(PRODUCTION_FILES) and plan["rounds_per_actor"] == 2
    assert "SOCIAL_OFF" in plan["social"] and plan["personalities"] == "DEFAULT vs DEFAULT"


def test_live_runner_refuses_to_call_before_the_plan_is_frozen_and_never_overwrites_it(tmp_path, monkeypatch):
    import sys

    import src.evaluation.run_m6g3_live as live

    monkeypatch.setattr(live, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(live, "out_path", lambda name: str(tmp_path / name))
    monkeypatch.setattr(live, "run_live_preflight", lambda *a, **k: pytest.fail("no provider call in tests"))
    monkeypatch.setattr(live, "run", lambda *a, **k: pytest.fail("no live run in tests"))
    monkeypatch.setattr(sys, "argv", ["run_m6g3_live", "--runtime", "native"])
    assert live.main() == 1
    monkeypatch.setattr(sys, "argv", ["run_m6g3_live", "--write-plan"])
    assert live.main() == 0
    frozen = (tmp_path / "live_call_plan.json").read_text()
    assert live.main() == 1 and (tmp_path / "live_call_plan.json").read_text() == frozen
    assert (tmp_path / "experiment.json").exists()
    # a production edit after the freeze blocks the live run
    monkeypatch.setattr(live, "_file_sha256", lambda path: "changed")
    monkeypatch.setattr(sys, "argv", ["run_m6g3_live", "--runtime", "native"])
    assert live.main() == 1


def test_live_stop_rule_ignores_model_output_errors_and_stops_on_provider_errors_or_incomplete_trials():
    from src.evaluation.run_m6g3_live import provider_failure

    def row(errors_a=(), errors_b=(), complete=True):
        return {"complete": complete, "timed_out": not complete,
                "requests": {ACTOR_A: [{"provider_error_types": list(errors_a)}], ACTOR_B: [{"provider_error_types": list(errors_b)}]}}

    assert provider_failure(row()) is None
    assert provider_failure(row(["provider_invalid_response"], ["no_tool_selected"])) is None
    assert provider_failure(row((), ["ThrottlingException"])) == f"{ACTOR_B}: ThrottlingException"
    assert provider_failure(row(complete=False)) == "incomplete trial (timed out)"


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_boundary1_checks_reproduce_a_trial_and_flag_the_same_tick_refinement(strands_warm, runtime):
    from src.evaluation.run_m6g3_live import boundary1_checks

    key = "strands" if runtime == "strands" else "native"
    row = _trial(C2, [[(ACTOR_B, 1)], [(ACTOR_A, 1)]], runtime=runtime)
    row.update(trial_id="t", runtime_key=key)
    checks = boundary1_checks(row, SCENARIOS[C2])
    assert checks["b1_reproduced"] and checks["b1_legacy_equal"] and checks["b1_p0_replay_equal"], checks
    assert not checks["b1_same_tick_handover"] and checks["b1_check_model_calls"] == 0
    # both answers reach the loop on one tick, White's published first
    row = _trial(C2, [[(ACTOR_B, 1), (ACTOR_A, 1)]], runtime=runtime)
    row.update(trial_id="t", runtime_key=key)
    checks = boundary1_checks(row, SCENARIOS[C2])
    assert checks["b1_same_tick_handover"] and checks["b1_white_published_first_same_tick"] and checks["b1_reproduced"], checks
    assert not checks["b1_legacy_equal"] and checks["b1_legacy_commit_order"] == [ACTOR_A, ACTOR_B]
    assert "commit_order" in checks["b1_legacy_mismatches"] and not checks["b1_p0_replay_equal"]


def test_new_m6g3_sources_contain_no_secret_like_patterns():
    root = os.path.join(os.path.dirname(__file__), "..", "src", "evaluation")
    names = [n for n in os.listdir(root) if n == "barrier_probe.py" or n.startswith("run_m6g3_")]
    pattern = re.compile(r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|bedrock-api-key-|aws_secret_access_key\s*=|ABSK[A-Za-z0-9+/=]{20,})")
    assert "barrier_probe.py" in names
    for name in names:
        with open(os.path.join(root, name), encoding="utf-8") as f:
            assert not pattern.search(f.read()), name
