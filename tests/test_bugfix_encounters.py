"""Bugfix live flow V2, Part D: spy encounters, fights, approach / avoid (F1-F14, §41-66).

Real level and engine. The fight outcome is engine-owned (own seeded RNG,
no model call); approach/avoid are LLM intents resolved deterministically
into ordinary navigation; everything is off unless encounters are enabled.
"""
from __future__ import annotations

import json
import math
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.actions import ActionType, GameAction
from src.agents import live_async
from src.agents.action_lifecycle import (
    ActionExecutor,
    OPPONENT_LOST,
    PATH_UNREACHABLE,
    TARGET_REACHED,
)
from src.encounter_intents import APPROACH_OPPONENT, AVOID_OPPONENT, approach_goal, avoid_goal, encounter_facts
from src.encounters import feet_distance, recorded_winners
from src.evaluation.navigation_audit import fresh_state
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import (
    compact_observation_for_model,
    enumerate_legal_actions,
    serialize_observation,
    validate_model_output,
)
from src.observation import build_observation
from tests.test_bugfix_live_async import GatedClient, Gate, commits, native_agent, strands_agent, wait_ready

BLACK, WHITE = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID
DT = 1 / 60


def seed_with(black_wins: bool) -> int:
    """A game seed whose first fight draw makes Black win (or lose)."""
    for seed in range(1, 1000):
        roll = random.Random(f"spy-fight:{seed}").random()
        if (roll < constants.FIGHT_BLACK_WIN_PROBABILITY) == black_wins:
            return seed
    raise AssertionError("no seed")


def arena(seed: int = 501, *, enabled: bool = True, gap: float = 150.0):
    """Both spies in Black's spawn room, `gap` px apart on a walkable line."""
    from src.items import load_items
    from src.level import load_level
    from src.state import GameState

    state = GameState.new_game(load_level(constants.LEVEL_PATH), load_items(constants.ITEMS_PATH), seed=seed)
    for obj in state.level.objects:
        if obj.type == "door" and obj.id != "exit_door":
            state.door_states[obj.id] = "open"
    state._rebuild_nav_grid()
    state.encounters_enabled = enabled
    black = state.actors[BLACK]
    room = state.level.room_id_at(black.x, black.y)
    grid = state.nav_grid
    # a walkable cell in the same room at roughly `gap` px, reachable
    reach = grid.reachable_cells((black.x, black.y))
    best = min((c for c in reach if state.level.room_id_at(*grid.center_of(c)) == room),
               key=lambda c: abs(math.dist(grid.center_of(c), (black.x, black.y)) - gap))
    white = state.actors[WHITE]
    white.x, white.y = grid.center_of(best)
    state.update(0.0)
    return state


def put_white_at(state, dx, dy=0.0):
    b = state.actors[BLACK]
    w = state.actors[WHITE]
    w.x, w.y = b.x + dx, b.y + dy
    state.update(DT)


def hide_white(state):
    """Put White in another room than Black (reachable cell)."""
    white = state.actors[WHITE]
    black_room = state.level.room_id_at(state.actors[BLACK].x, state.actors[BLACK].y)
    grid = state.nav_grid
    other = next(c for c in sorted(grid.reachable_cells((white.x, white.y)))
                 if state.level.room_id_at(*grid.center_of(c)) not in (None, black_room))
    white.x, white.y = grid.center_of(other)
    state.update(DT)


def fights(state):
    return [e for e in state.event_log if e["type"] == "SPY_FIGHT_RESOLVED"]


# F1 / F2 ------------------------------------------------------------------------

def test_f1_same_room_outside_encounter_range_no_fight():
    state = arena()
    assert state.level.room_id_at(state.actors[WHITE].x, state.actors[WHITE].y) == \
        state.level.room_id_at(state.actors[BLACK].x, state.actors[BLACK].y)
    for _ in range(120):
        state.update(DT)
    assert feet_distance(state) > constants.ENCOUNTER_DISTANCE and not fights(state)


def test_f2_entering_encounter_range_triggers_exactly_one_fight():
    state = arena()
    put_white_at(state, constants.ENCOUNTER_DISTANCE - 2)
    for _ in range(240):  # overlapping for 4 s: still one fight
        state.update(DT)
    [fight] = fights(state)
    kinds = [e["type"] for e in state.event_log if e["type"].startswith("SPY_")]
    assert kinds == ["SPY_ENCOUNTER", "SPY_FIGHT_STARTED", "SPY_FIGHT_RESOLVED"]
    for key in ("black_id", "white_id", "winner_id", "loser_id", "rng_roll", "fight_probability", "game_time"):
        assert key in fight, key
    assert fight["fight_probability"] == 0.5 and {fight["winner_id"], fight["loser_id"]} == {BLACK, WHITE}


def test_encounters_are_off_by_default_and_fights_never_touch_the_shared_rng():
    off = arena(enabled=False)
    put_white_at(off, 5)
    for _ in range(60):
        off.update(DT)
    assert not fights(off) and off.fights == []
    on = arena()
    before = on.rng.getstate()
    put_white_at(on, 5)
    assert fights(on) and on.rng.getstate() == before  # INSPECT's stream is untouched


# F3 / F4 / F5 -------------------------------------------------------------------

@pytest.mark.parametrize("black_wins", [True, False])
def test_f3_f4_seeded_outcome_and_loser_penalty(black_wins):
    state = arena(seed=seed_with(black_wins))
    put_white_at(state, 10)
    [fight] = fights(state)
    winner, loser = (BLACK, WHITE) if black_wins else (WHITE, BLACK)
    assert (fight["winner_id"], fight["loser_id"]) == (winner, loser)
    assert fight["outcome_source"] == "rng" and (fight["rng_roll"] < 0.5) == black_wins
    assert state.actors[loser].is_stunned and state.actors[loser].stunned_timer == pytest.approx(
        constants.FIGHT_LOSER_STUN_SECONDS, abs=DT)
    assert not state.actors[winner].is_stunned
    assert any(e["type"] == "ACTOR_STUNNED" and e["actor_id"] == loser for e in state.event_log)
    # the loser's actions are dropped by the existing stun rule
    x = state.actors[loser].x
    state.apply_action(loser, GameAction(ActionType.MOVE, direction=(1.0, 0.0)), DT)
    assert state.actors[loser].x == x


def test_f5_replay_uses_the_recorded_winner_and_never_rerolls():
    recorded = arena(seed=seed_with(black_wins=True))
    put_white_at(recorded, 10)
    winners = recorded_winners(recorded.event_log)
    assert winners == [BLACK]
    replay = arena(seed=seed_with(black_wins=False))  # its own RNG would make White win
    replay.recorded_fight_winners = list(winners)
    rng_before = replay.fight_rng.getstate()
    put_white_at(replay, 10)
    [fight] = fights(replay)
    assert fight["winner_id"] == BLACK and fight["outcome_source"] == "recorded" and fight["rng_roll"] is None
    assert replay.fight_rng.getstate() == rng_before  # not re-rolled
    # the live telemetry form works too
    assert recorded_winners([{"event_type": "spy_fight", "winner_id": WHITE}]) == [WHITE]


# F6 -----------------------------------------------------------------------------

def test_f6_cooldown_and_separation_prevent_an_immediate_second_fight():
    state = arena()
    put_white_at(state, 5)
    assert len(fights(state)) == 1
    for _ in range(int((constants.FIGHT_COOLDOWN_SECONDS + 1) / DT)):  # still touching after the cooldown
        state.update(DT)
    assert len(fights(state)) == 1  # not re-armed: they never separated
    put_white_at(state, constants.ENCOUNTER_REARM_DISTANCE + 10)  # separate...
    put_white_at(state, 5)  # ...and meet again right away: cooldown already over
    assert len(fights(state)) == 2
    put_white_at(state, constants.ENCOUNTER_REARM_DISTANCE + 10)
    put_white_at(state, 5)  # separated, but inside the new cooldown
    assert len(fights(state)) == 2


# F7 / F8 / F11 ------------------------------------------------------------------

def test_f7_f8_every_personality_gets_the_same_encounter_options():
    state = arena()
    obs = build_observation(state, BLACK)
    legal = enumerate_legal_actions(obs, reachable_only=True, known_trap_blocks_place_trap=True, encounter_actions=True)
    assert {"action": APPROACH_OPPONENT, "target_id": WHITE} in legal
    assert {"action": AVOID_OPPONENT, "target_id": WHITE} in legal
    payload = serialize_observation(obs, live_state=True)
    facts = payload["encounter"]
    assert facts["opponent_visible"] and facts["opponent_distance"] == int(round(obs.opponent_distance))
    assert facts["encounter_distance"] == 28 and facts["win_chance"] == 0.5
    compact = compact_observation_for_model(payload)
    assert compact["encounter"] == facts and compact["legal_actions_compact"][APPROACH_OPPONENT] == [WHITE]
    # the personality changes only the system prompt, never these facts
    from src.llm_controller import LLMController, LLMControllerConfig

    captured = {}
    for name in ("saboteur", "cautious"):
        gate = Gate(lambda p: {"action": "wait", "reason": "x"}, auto=True)
        agent = LLMController(BLACK, model_client=GatedClient(gate), execution_mode=live_async.LIVE_ASYNC,
                              config=LLMControllerConfig(personality_profile=name, ready_response_guard=True))
        agent.get_action(obs)
        _idle(agent)
        captured[name] = gate.payloads[0]
    assert captured["saboteur"] == captured["cautious"]


def _idle(agent, timeout=5.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end and agent._thinking_in_progress:
        time.sleep(0.005)


def test_f11_hidden_opponent_is_not_in_the_observation():
    state = arena()
    hide_white(state)
    obs = build_observation(state, BLACK)
    assert obs.opponent is None and obs.opponent_distance is None
    legal = enumerate_legal_actions(obs, reachable_only=True, encounter_actions=True)
    assert not any(a["action"] in (APPROACH_OPPONENT, AVOID_OPPONENT) for a in legal)
    facts = serialize_observation(obs, live_state=True)["encounter"]
    assert facts["opponent_visible"] is False and facts["opponent_distance"] is None and facts["opponent_room"] is None
    # approach toward a hidden opponent is refused by validation
    result = validate_model_output(obs, {"action": "approach_opponent", "target_id": WHITE}, legal)
    assert not result.valid and result.error_type == "encounter_not_available"


def test_encounter_surface_is_live_only():
    state = arena()
    obs = build_observation(state, BLACK)
    assert not any(a["action"] in (APPROACH_OPPONENT, AVOID_OPPONENT) for a in enumerate_legal_actions(obs))
    assert "encounter" not in serialize_observation(obs)
    off = build_observation(arena(enabled=False), BLACK)
    assert "encounter" not in serialize_observation(off, live_state=True)
    assert validate_model_output(off, {"action": "approach_opponent", "target_id": WHITE},
                                 enumerate_legal_actions(off, reachable_only=True, encounter_actions=True)).error_type == "unknown_action"


# F9 / F10 -----------------------------------------------------------------------

def drive(state, actor_id, action_name, max_seconds=10.0):
    obs = build_observation(state, actor_id)
    executor = ActionExecutor()
    action, boundary, _ = executor.start(obs, "d", action_name, WHITE if actor_id == BLACK else BLACK, None,
                                         obs.mission_time_remaining)
    ticks = 0
    while boundary is None and ticks < int(max_seconds / DT):
        state.apply_action(actor_id, action, DT)
        state.update(DT)
        ticks += 1
        action, boundary = executor.continue_(build_observation(state, actor_id))
    return boundary, executor


def test_f9_approach_resolves_to_reachable_movement_and_leads_to_contact():
    state = arena(gap=220)
    goal = approach_goal(build_observation(state, BLACK))
    assert goal is not None and state.nav_grid.nearest_walkable(*goal.point) in \
        state.nav_grid.reachable_cells((state.actors[BLACK].x, state.actors[BLACK].y))
    boundary, executor = drive(state, BLACK, APPROACH_OPPONENT)
    assert boundary == TARGET_REACHED
    assert len(fights(state)) == 1  # the engine resolved it at contact - no model call involved
    # the chase followed the grid: never inside a wall or furniture
    solids = state.level.solid_rects(state.door_states)
    assert not any(state.actors[BLACK].collision_rect.colliderect(s) for s in solids)


def test_f9_approach_ends_when_the_opponent_is_no_longer_visible():
    state = arena(gap=220)
    obs = build_observation(state, BLACK)
    executor = ActionExecutor()
    action, boundary, _ = executor.start(obs, "d", APPROACH_OPPONENT, WHITE, None, obs.mission_time_remaining)
    assert boundary is None
    hide_white(state)  # gone to another room
    _, boundary = executor.continue_(build_observation(state, BLACK))
    assert boundary == OPPONENT_LOST


def test_f10_avoid_moves_away_from_the_opponent():
    state = arena(gap=90)
    start = feet_distance(state)
    goal = avoid_goal(build_observation(state, BLACK))
    assert goal is not None and goal.kind in ("exit", "cell")
    boundary, _ = drive(state, BLACK, AVOID_OPPONENT)
    assert boundary in (TARGET_REACHED, "doorway_crossing_incomplete")
    assert feet_distance(state) > start + 40 and not fights(state)


def test_approach_with_no_visible_opponent_fails_explicitly():
    state = arena()
    hide_white(state)
    obs = build_observation(state, BLACK)
    executor = ActionExecutor()
    _, boundary, _ = executor.start(obs, "d", APPROACH_OPPONENT, WHITE, None, obs.mission_time_remaining)
    assert boundary == PATH_UNREACHABLE and executor.last_finished.failure_detail == "no_reachable_encounter_goal"


# F12 ----------------------------------------------------------------------------

def test_f12_a_fight_invalidates_a_pending_decision():
    """White is thinking while Black walks into it; White loses the fight
    (recorded outcome); White's answer arrives while it is stunned and is
    discarded, never executed."""
    state = arena(gap=120)
    state.recorded_fight_winners = [BLACK]
    telemetry = TelemetryLogger()
    gate = Gate(lambda p: {"action": "move", "target_id": next(t for t in p["legal_actions_compact"]["move"]), "reason": "go"})
    white = native_agent(WHITE, gate, telemetry)
    white.get_action(build_observation(state, WHITE))
    assert white._thinking_in_progress
    boundary, _ = drive(state, BLACK, APPROACH_OPPONENT)  # Black reaches White meanwhile
    assert boundary == TARGET_REACHED and fights(state)[0]["winner_id"] == BLACK and state.actors[WHITE].is_stunned
    gate.release.set()
    wait_ready(white)
    action = white.get_action(build_observation(state, WHITE))
    assert action.type is ActionType.WAIT and not white._executor.in_progress
    [event] = commits(telemetry)
    assert event["stale_classification"] == live_async.STALE_AND_ILLEGAL and event["discarded_due_to_stale"]
    assert white.ready_responses_cancelled == 1


def test_f12_winner_pending_approach_is_discarded_during_the_cooldown():
    state = arena(gap=120)
    state.recorded_fight_winners = [BLACK]
    telemetry = TelemetryLogger()
    gate = Gate(lambda p: {"action": "approach_opponent", "target_id": WHITE, "reason": "get them"})
    black = native_agent(BLACK, gate, telemetry)
    black.get_action(build_observation(state, BLACK))
    put_white_at(state, 5)  # contact while Black thinks: Black wins
    put_white_at(state, 60)  # still visible, but a fight is not possible yet
    gate.release.set()
    wait_ready(black)
    black.get_action(build_observation(state, BLACK))
    [event] = commits(telemetry)
    assert event["stale_classification"] == live_async.STALE_AND_ILLEGAL
    assert event["stale_reason"] == f"{APPROACH_OPPONENT}_no_longer_legal"


# F13 / F14 ----------------------------------------------------------------------

def test_f13_native_and_strands_see_the_same_encounter_facts():
    pytest.importorskip("strands")
    state = arena()
    obs = build_observation(state, BLACK)
    payloads = {}
    for name, build in (("native", native_agent), ("strands", strands_agent)):
        gate = Gate(lambda p: {"action": "avoid_opponent", "target_id": WHITE, "reason": "x"}, auto=True)
        agent = build(BLACK, gate)
        action = agent.get_action(obs)
        if agent._thinking_in_progress or agent._response_ready_uncommitted:
            wait_ready(agent)
            action = agent.get_action(obs)
        payloads[name] = gate.payloads[0]
        assert action.type is ActionType.MOVE and agent._executor.active.action == AVOID_OPPONENT
    n, s = payloads["native"], payloads["strands"]
    assert n["encounter"] == s["encounter"]
    assert n["legal_actions_compact"][AVOID_OPPONENT] == s["legal_actions_compact"][AVOID_OPPONENT] == [WHITE]


def test_f14_provider_neutral():
    from src.llm_controller import LLMController

    state = arena()
    obs = build_observation(state, BLACK)
    payloads = []
    for provider in ("bedrock", "openai"):
        gate = Gate(lambda p: {"action": "wait", "reason": "x"}, auto=True)
        agent = LLMController(BLACK, model_client=GatedClient(gate), provider_name=provider, model_name="m",
                              execution_mode=live_async.LIVE_ASYNC)
        agent.get_action(obs)
        _idle(agent)
        payloads.append(gate.payloads[0])
    assert payloads[0] == payloads[1] and "encounter" in payloads[0]


def test_aliases_and_implicit_target():
    state = arena()
    obs = build_observation(state, BLACK)
    legal = enumerate_legal_actions(obs, reachable_only=True, encounter_actions=True)
    for word, canonical in (("flee", AVOID_OPPONENT), ("intercept", APPROACH_OPPONENT)):
        result = validate_model_output(obs, {"action": word}, legal)
        assert result.valid and result.validated_action["action"] == canonical
        assert result.validated_action["target_id"] == WHITE


def test_encounter_facts_none_when_disabled():
    assert encounter_facts(build_observation(arena(enabled=False), BLACK)) is None


def test_live_game_enables_encounters_and_records_fights_in_telemetry():
    from tests.workshop_live_helpers import LegalMoveClient
    from workshop.live.config import resolve_config
    from workshop.live.session import LiveGame

    config = resolve_config({"provider": "bedrock", "model": "m", "personality_preset": "rogue-vs-cautious"})
    assert (config.black.personality, config.white.personality) == ("saboteur", "cautious") and config.encounters
    telemetry = TelemetryLogger()
    game = LiveGame(config, telemetry, client_factory=lambda spec, cfg: LegalMoveClient(spec))
    assert game.state.encounters_enabled
    b, w = game.state.actors[BLACK], game.state.actors[WHITE]
    w.x, w.y = b.x + 5, b.y
    game._step(DT)
    [event] = [e for e in telemetry.events if e.get("event_type") == "spy_fight"]
    assert event["winner_id"] in (BLACK, WHITE) and recorded_winners(telemetry.events) == [event["winner_id"]]
    off = LiveGame(resolve_config({"provider": "bedrock", "model": "m", "encounters": "off"}), None,
                   client_factory=lambda spec, cfg: LegalMoveClient(spec))
    assert not off.state.encounters_enabled
    # overlay: FIGHT banner and opponent distance in the detail view
    from workshop.live.overlay import TelemetryView

    view = TelemetryView(config)
    status, _ = view.status_line(game, paused=False, speed=1.0, match_over=False)
    assert "FIGHT Black vs White - Winner:" in status
    text = "\n".join(t for t, _ in view.actor_lines(view.actor(game, BLACK), detail=True))
    assert "opponent visible" in text and "fights   won" in text
    game.renderer.camera = game.renderer._compute_camera(game.state)
    game.renderer._draw_event_cues(game.state)
    assert any(c[1].startswith("FIGHT!") for c in game.renderer._cues)


# found in the SABOTEUR-vs-RACER live validation ---------------------------------

def test_approach_is_not_offered_until_a_fight_is_possible_again():
    """After a fight the spies often stay on top of each other; approaching
    then did nothing (no fight until they separate). It is now offered only
    when a fight can really happen, and the facts say why."""
    state = arena()
    put_white_at(state, 2)
    assert len(fights(state)) == 1
    for _ in range(int((constants.FIGHT_COOLDOWN_SECONDS + 1) / DT)):
        state.update(DT)
    obs = build_observation(state, BLACK)
    legal = enumerate_legal_actions(obs, reachable_only=True, encounter_actions=True)
    assert not any(a["action"] == APPROACH_OPPONENT for a in legal)
    facts = encounter_facts(obs)
    assert facts["fight_possible_now"] is False and facts["next_fight_possible_in"] == 0
    assert facts["after_a_fight_separate_by"] == int(constants.ENCOUNTER_REARM_DISTANCE)
    put_white_at(state, constants.ENCOUNTER_REARM_DISTANCE + 20)
    obs = build_observation(state, BLACK)
    assert encounter_facts(obs)["fight_possible_now"] is True
    assert {"action": APPROACH_OPPONENT, "target_id": WHITE} in enumerate_legal_actions(obs, reachable_only=True, encounter_actions=True)


def test_avoid_works_when_the_spies_overlap():
    """Live: avoid was offered at distance 1-2 px but its clearance rule
    rejected every path (the start point itself is 'near the opponent')."""
    state = arena()
    put_white_at(state, 2)  # fight + overlap
    for _ in range(int((constants.FIGHT_LOSER_STUN_SECONDS + 0.5) / DT)):
        state.update(DT)
    for actor in (BLACK, WHITE):
        obs = build_observation(state, actor)
        assert {"action": AVOID_OPPONENT, "target_id": WHITE if actor == BLACK else BLACK} in \
            enumerate_legal_actions(obs, reachable_only=True, encounter_actions=True)
        assert avoid_goal(obs) is not None
    start = feet_distance(state)
    boundary, _ = drive(state, BLACK, AVOID_OPPONENT)
    assert boundary in (TARGET_REACHED, "doorway_crossing_incomplete") and feet_distance(state) > start + 40
