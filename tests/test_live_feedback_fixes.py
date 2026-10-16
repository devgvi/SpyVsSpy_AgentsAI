"""Live feedback fixes (run workshop_runs/live/live/20260930-150146).

1 the mission timer keeps running while spies think (shared clock default);
2 a spy knows its own traps (V3: they DO trigger on it outside the recorded rules);
3 split screen, one half per spy (P toggles);
4 the exit is reachable up close, marked, and an escaped spy leaves the building;
5 an escaped spy is no longer an opponent, so the other one stops circling it;
6 readable text colours / opaque panels;
7 trap hits counted, 3 lives, out of the match at 0.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from src import constants
from src.actions import ActionType, GameAction
from src.agents.action_lifecycle import CLOSE_APPROACH_PX, TARGET_REACHED
from src.encounter_intents import APPROACH_OPPONENT
from src.evaluation.navigation_audit import fresh_state
from src.interaction_range import distance_to_rect
from src.llm.personality import OWN_TRAP_RULE, render_live_mission_block
from src.llm_controller import compact_observation_for_model, enumerate_legal_actions, serialize_observation
from src.mission_clock import SIMULATION_TIME
from src.observation import build_observation
from src.scoring import compute_score
from tests.test_bugfix_inspection_proximity import _trappable_in_start_room, act, events, move_to
from tests.workshop_live_helpers import LegalMoveClient
from workshop.live.config import resolve_config
from workshop.live.session import LiveGame

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID
DT = 1 / 60
BASE = {"provider": "bedrock", "model": "m", "personality_preset": "rogue-vs-cautious", "spawn": "same_room"}


def live_game(**extra):
    return LiveGame(resolve_config({**BASE, **extra}), None, client_factory=lambda spec, cfg: LegalMoveClient(spec))


# 1 ------------------------------------------------------------------------------

def test_1_the_live_timer_keeps_running_by_default():
    config = resolve_config(BASE)
    assert config.mission_clock == SIMULATION_TIME and config.time_limit == 300.0
    game = live_game(mission_timer="game_time")  # V3: wall_clock is the default (tests/test_bugfix_live_match_flow.py)
    before = game.state.mission_time_remaining
    game.state.budget_hold = {WHITE, BLACK}  # both "thinking": irrelevant for the shared clock
    game.state.update(1.0)
    assert game.state.mission_time_remaining == before - 1.0
    assert resolve_config({**BASE, "mission_clock": "action_budget"}).mission_clock == "action_budget"  # still selectable


# 2 ------------------------------------------------------------------------------

def test_2_own_trap_is_marked_mine_and_recorded_rules_keep_owner_immunity():
    """V3 changed the live rule (own traps trigger on their owner - see
    tests/test_bugfix_live_match_flow.py TR1); the recorded engine rule, used
    by every stored replay, still has the owner immune."""
    state = fresh_state()
    obj = _trappable_in_start_room(state)
    move_to(state, obj.id)
    act(state, ActionType.INSPECT, obj.id)
    act(state, ActionType.PLACE_TRAP, obj.id, trap_type="bomb")
    obs = build_observation(state, WHITE)
    payload = compact_observation_for_model(serialize_observation(obs, live_state=True))
    entry = next(o for o in payload["room"]["objects"] if o["id"] == obj.id)
    assert entry["trap_status"] == "detected" and entry["trap_is_mine"] is True
    assert "trap_is_mine" not in next(o for o in serialize_observation(obs)["room"]["objects"] if o["id"] == obj.id)
    # the recorded engine rule: your own trap does not go off under you
    act(state, ActionType.INTERACT, obj.id)
    assert events(state, "OBJECT_SEARCHED", obj.id) and not events(state, "TRAP_TRIGGERED")
    assert OWN_TRAP_RULE in render_live_mission_block("saboteur")


# 4 / 5 --------------------------------------------------------------------------

def test_4_move_to_the_exit_ends_right_at_it_and_escaping_leaves_the_building():
    game = live_game()
    state = game.state
    black = state.actors[BLACK]
    black.x, black.y = 888.5, 1424.0  # where the live spy stood, 69 px from the exit
    for item in state.items.required:
        black.inventory.add(item)
    state.update(0.0)
    boundary, obs = move_to(state, "exit_door", actor_id=BLACK)
    assert boundary == TARGET_REACHED and obs.nearby_object_id == "exit_door"
    assert distance_to_rect(black.x, black.y, state.level.get_object("exit_door").collision_rect) <= CLOSE_APPROACH_PX + 1
    act(state, ActionType.INTERACT, "exit_door", actor_id=BLACK)
    assert state.mission_complete[BLACK]
    # the other spy no longer sees it, and cannot approach it
    white = state.actors[WHITE]
    white.x, white.y = black.x + 30, black.y
    state.update(DT)
    obs = build_observation(state, WHITE)
    assert obs.opponent is None and not obs.visible_other_actor
    assert not any(a["action"] == APPROACH_OPPONENT for a in
                   enumerate_legal_actions(obs, reachable_only=True, encounter_actions=True))


def test_5_the_match_ends_when_both_are_done():
    game = live_game()
    state = game.state
    assert not game.match_over()
    state.mission_complete[BLACK] = True
    assert not game.match_over()
    state.eliminated[WHITE] = True
    assert compute_score(state, WHITE).failed and game.match_over()


# 7 ------------------------------------------------------------------------------

def _trigger(state, victim):
    obj = _trappable_in_start_room(state)
    entry = state._object_entry(obj.id)
    entry["trap"] = {"type": "snare", "owner_id": BLACK if victim == WHITE else WHITE, "armed": True}
    victim_actor = state.actors[victim]
    victim_actor.stunned_timer = 0.0
    state._trigger_trap(victim_actor, obj, entry, entry["trap"])


def test_7_each_trap_costs_a_life_and_three_means_out():
    game = live_game()
    state = game.state
    assert state.lives == {BLACK: 3, WHITE: 3}
    for expected in (2, 1, 0):
        _trigger(state, WHITE)
        assert state.lives[WHITE] == expected and state.trap_hits[WHITE] == 3 - expected
    assert state.eliminated[WHITE] and compute_score(state, WHITE).failed
    assert [e["lives_left"] for e in state.event_log if e["type"] == "LIFE_LOST"] == [2, 1, 0]
    assert any(e["type"] == "ACTOR_ELIMINATED" and e["actor_id"] == WHITE for e in state.event_log)
    white = state.actors[WHITE]
    white.stunned_timer = 0.0
    x = white.x
    state.apply_action(WHITE, GameAction(ActionType.MOVE, direction=(1.0, 0.0)), 0.1)
    assert white.x == x  # out of the match: no more actions
    obs = build_observation(state, WHITE)
    assert obs.time_expired and obs.lives == 0 and obs.trap_hits == 3
    payload = serialize_observation(obs, live_state=True)
    assert payload["self"]["lives"] == 0 and payload["self"]["traps_triggered_on_you"] == 3
    assert "lives" in render_live_mission_block("cautious", lives=3)


def test_7_lives_are_off_outside_the_live_game():
    state = fresh_state()
    _trigger(state, WHITE)
    assert not state.lives_enabled and state.trap_hits[WHITE] == 1 and not state.eliminated
    assert build_observation(state, WHITE).lives is None
    off = live_game(lives="off")
    assert not off.state.lives_enabled


# 3 / 6 --------------------------------------------------------------------------

def test_3_split_screen_and_single_view_render_and_p_toggles():
    from workshop.live.overlay import PANEL_ALPHA, SPY_TEXT_COLORS, TelemetryView
    from workshop.live.session import MatchRunner

    game = live_game()
    renderer = game.renderer
    renderer.view, renderer.game = TelemetryView(game.live_config), game
    assert renderer.split
    renderer.render(game.state, DT, None, None, game.bot_controller)
    left, right = game.screen.get_at((320, 300)), game.screen.get_at((960, 300))
    renderer.split = False
    renderer.render(game.state, DT, None, None, game.bot_controller)
    # 6: readable colours - bright text for both spies, near-opaque panels
    for color in SPY_TEXT_COLORS.values():
        assert 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2] > 150
    assert PANEL_ALPHA >= 220
    # P toggles the split in the runner
    runner = MatchRunner(resolve_config(BASE), __import__("pathlib").Path("/tmp"), headless=True,
                         client_factory=lambda spec, cfg: LegalMoveClient(spec),
                         event_source=lambda: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)])
    runner.new_match()
    assert runner.game.renderer.split
    runner._handle_events()
    assert runner.game.renderer.split is False and runner.controls_log[-1]["key"] == "P"
    assert left is not None and right is not None


# found while re-checking the live run: answers lost to the function wrapper --------

def test_function_call_envelope_is_unwrapped_in_the_live_game():
    from src.llm_controller import unwrap_function_call_envelope

    wrapped = {"action": "choose_action", "arguments": {"action": "search", "object_id": "bookshelf_03", "reason": "x"}}
    assert unwrap_function_call_envelope(wrapped) == {"action": "search", "target_id": "bookshelf_03", "reason": "x"}
    assert unwrap_function_call_envelope({"name": "choose_action", "parameters": {"action": "wait"}}) == {"action": "wait"}
    assert unwrap_function_call_envelope({"action": "move", "target_id": "desk_01"}) == {"action": "move", "target_id": "desk_01"}
    game = live_game()
    assert game.black_llm_controller.config.model_response_transform is unwrap_function_call_envelope
    from src.llm_controller import LLMControllerConfig
    assert LLMControllerConfig().model_response_transform is None  # recorded runs: unchanged


def test_json_after_prose_with_braces_is_still_extracted():
    from src.llm.clients.ollama_client import _extract_json_payload

    text = ('I think {this} matters.\n```json\n{\n  "action": "choose_action",\n  "arguments": {"action": "move", '
            '"target_id": "door_reception_corridor", "reason": "go"}\n}\n```')
    payload = _extract_json_payload(text)
    assert payload["action"] == "choose_action" and payload["arguments"]["target_id"] == "door_reception_corridor"
    # a plain answer parses exactly as before
    assert _extract_json_payload('{"action": "wait"}') == {"action": "wait"}
    assert _extract_json_payload("no json at all") == {}


def test_a_move_that_ends_at_once_is_followed_by_a_new_decision():
    """Live (run 161510): with every item, the spy chose MOVE exit_door while
    already at the exit; the MOVE ended at once, nothing changed, and it
    stood there for 92 s until its time ran out."""
    import time

    from tests.test_bugfix_live_async import Gate, native_agent, wait_ready

    state = fresh_state()
    black = state.actors[BLACK]
    for item in state.items.required:
        black.inventory.add(item)
    move_to(state, "exit_door", actor_id=BLACK)
    obs = build_observation(state, BLACK)
    assert obs.nearby_object_id == "exit_door"
    answers = iter([{"action": "move", "target_id": "exit_door", "reason": "go"},
                    {"action": "interact", "target_id": "exit_door", "reason": "leave"}])
    gate = Gate(lambda p: next(answers), auto=True)
    agent = native_agent(BLACK, gate)

    def decide():
        action = agent.get_action(build_observation(state, BLACK))
        if agent._thinking_in_progress or agent._response_ready_uncommitted:
            wait_ready(agent)
            action = agent.get_action(build_observation(state, BLACK))
        return action

    first = decide()
    assert first.type is ActionType.WAIT and agent._last_observation_digest is None
    time.sleep(0.55)  # the decision throttle
    second = decide()
    assert second.type is ActionType.INTERACT and second.target_id == "exit_door"
    state.apply_action(BLACK, second, DT)
    assert state.mission_complete[BLACK]


def test_an_idle_spy_decides_again_after_a_few_seconds():
    from src.agents import live_async
    from tests.test_bugfix_live_async import Gate, native_agent

    state = fresh_state()
    agent = native_agent(BLACK, Gate(lambda p: {"action": "wait", "reason": "x"}, auto=True))
    agent._last_observation_digest = "same"
    live_async.idle_redecide(agent)
    agent._live_idle_since -= live_async.IDLE_REDECIDE_SECONDS + 0.1
    live_async.idle_redecide(agent)
    assert agent._last_observation_digest is None
