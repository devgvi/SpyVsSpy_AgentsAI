"""Encounter alerts: the opponent coming close is a decision point for the agent.

Found in live run workshop_runs/live/live/20261001-141758: the spies stood
next to each other (searching the same bookshelf, crossing the same door) and
neither reacted; one fight came from both choosing to flee through the same
door, the other from one walking to the object the other was searching.
`approach_opponent` was offered 117 times over five live matches and never
chosen. Nothing here makes a spy fight or flee: when the opponent comes within
ENCOUNTER_ALERT_DISTANCE the current walk stops (boundary OPPONENT_NEARBY),
the payload carries `encounter.alert`, and the model is asked - fight, run or
carry on. Live only (GameState.encounter_alerts / ActionExecutor.encounter_alerts).
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from src import constants
from src.actions import ActionType
from src.agents import live_async
from src.agents.action_lifecycle import OPPONENT_NEARBY, TARGET_REACHED, ActionExecutor
from src.encounter_intents import ENCOUNTER_ALERT_DISTANCE, encounter_alert, encounter_facts
from src.evaluation.navigation_audit import fresh_state
from src.llm.personality import render_live_mission_block
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import serialize_observation
from src.observation import build_observation
from tests.test_bugfix_live_async import Gate, native_agent, strands_agent, wait_ready
from tests.test_bugfix_live_match_flow import live_game

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID
DT = 1 / 60


def arena(distance: float):
    """A live game, both spies in the main office, `distance` apart on one row."""
    state = live_game(encounters="on", spawn="same_room").state
    b, w = state.actors[BLACK], state.actors[WHITE]
    w.x, w.y = 560, 1100
    b.x, b.y = 560 + distance, 1100
    state.update(0.0)
    return state


def executor():
    ex = ActionExecutor()
    ex.close_approach = True
    ex.encounter_alerts = True
    return ex


def test_alert_only_when_close_and_a_fight_is_possible():
    assert encounter_alert(build_observation(arena(ENCOUNTER_ALERT_DISTANCE - 10), WHITE)) is not None
    assert encounter_alert(build_observation(arena(ENCOUNTER_ALERT_DISTANCE + 60), WHITE)) is None
    state = arena(100)
    state.encounter_cooldown = 3.0  # just fought: no fight on contact
    assert encounter_alert(build_observation(state, WHITE)) is None
    recorded = fresh_state()
    recorded.encounters_enabled = True
    assert encounter_alert(build_observation(recorded, WHITE)) is None  # recorded rules: no alerts


def test_payload_states_the_alert_and_the_real_penalty():
    state = arena(100)
    state.lives[BLACK] = 1
    obs = build_observation(state, WHITE)
    facts = serialize_observation(obs, live_state=True, live_workshop=True)["encounter"]
    alert = facts["alert"]
    assert alert["opponent_close"] and alert["opponent_distance"] == 100
    assert alert["your_lives"] == 3 and alert["opponent_lives"] == 1 == facts["opponent_lives"]
    assert facts["loser_knocked_out_seconds"] == constants.FIGHT_KO_SECONDS and facts["loser_loses_life"]
    assert "loser_stunned_seconds" not in facts  # the stale V2 field is gone in live KO matches
    far = encounter_facts(build_observation(arena(400), WHITE))
    assert "alert" not in far


def test_a_walk_stops_when_the_opponent_comes_close_and_nothing_is_chosen_for_the_agent():
    state = arena(300)
    ex = executor()
    obs = build_observation(state, WHITE)
    # a long walk (to the corridor door, ~2 s) while the opponent walks up
    action, boundary, _ = ex.start(obs, "d1", "move", "door_corridor_main_office", None, obs.mission_time_remaining)
    assert boundary is None
    black, white = state.actors[BLACK], state.actors[WHITE]
    for _ in range(600):
        dx, dy = white.x - black.x, white.y - black.y
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        black.x += dx / d * 5  # the opponent walks toward White
        black.y += dy / d * 5
        state.apply_action(WHITE, action, DT)
        state.update(DT)
        action, boundary = ex.continue_(build_observation(state, WHITE))
        if boundary is not None:
            break
    assert boundary == OPPONENT_NEARBY
    assert action.type is ActionType.WAIT  # no fight, no flight decided by the code
    assert ex.last_finished.action == "move"


def test_one_alert_per_encounter_then_rearm_after_separating():
    state = arena(100)
    ex = executor()
    obs = build_observation(state, WHITE)
    # the agent decided with the alert in view (say: carry on): its walk is not stopped again
    action, boundary, _ = ex.start(obs, "d1", "move", "plant_03", None, obs.mission_time_remaining)
    for _ in range(30):
        state.apply_action(WHITE, action, DT)
        state.update(DT)
        action, boundary = ex.continue_(build_observation(state, WHITE))
        if boundary is not None:
            break
    assert boundary != OPPONENT_NEARBY
    # the spies separate, then meet again: a new alert
    black = state.actors[BLACK]
    black.x = state.actors[WHITE].x + 300
    state.update(0.0)
    ex._note_opponent(build_observation(state, WHITE))
    assert ex._alert_armed


def test_a_deliberate_approach_is_never_stopped():
    state = arena(130)
    ex = executor()
    obs = build_observation(state, WHITE)
    action, boundary, _ = ex.start(obs, "d1", "approach_opponent", BLACK, None, obs.mission_time_remaining)
    for _ in range(300):
        if boundary is not None:
            break
        state.apply_action(WHITE, action, DT)
        state.update(DT)
        action, boundary = ex.continue_(build_observation(state, WHITE))
    assert boundary in (TARGET_REACHED, None) or state.fights  # contact (a fight), never OPPONENT_NEARBY
    assert boundary != OPPONENT_NEARBY


def test_running_away_while_the_opponent_closes_in_asks_again():
    state = arena(110)
    ex = executor()
    obs = build_observation(state, WHITE)
    action, boundary, _ = ex.start(obs, "d1", "avoid_opponent", BLACK, None, obs.mission_time_remaining)
    assert boundary is None
    black, white = state.actors[BLACK], state.actors[WHITE]
    for _ in range(300):
        # the opponent chases faster than White runs
        dx, dy = white.x - black.x, white.y - black.y
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        black.x += dx / d * 6
        black.y += dy / d * 6
        state.apply_action(WHITE, action, DT)
        state.update(DT)
        if state.fights:
            break
        action, boundary = ex.continue_(build_observation(state, WHITE))
        if boundary is not None:
            break
    assert boundary == OPPONENT_NEARBY and not state.fights  # asked again before contact


@pytest.mark.parametrize("runtime", ["native", "strands"])
def test_an_answer_decided_before_the_opponent_came_close_is_not_applied(runtime):
    state = arena(400)
    telemetry = TelemetryLogger(None)
    gate = Gate(lambda p: {"action": "move", "target_id": "plant_03", "reason": "carry on"})
    agent = (native_agent if runtime == "native" else strands_agent)(WHITE, gate, telemetry=telemetry)
    agent.get_action(build_observation(state, WHITE))
    assert agent._thinking_in_progress
    state.actors[BLACK].x = state.actors[WHITE].x + 90  # the opponent walks up while the model thinks
    state.update(0.0)
    gate.release.set()
    wait_ready(agent)
    action = agent.get_action(build_observation(state, WHITE))
    assert action.type is ActionType.WAIT
    commit = [e for e in telemetry.events if e.get("event_type") == "live_async_commit"][-1]
    assert commit["stale_reason"] == "opponent_came_close_during_request" and commit["discarded_due_to_stale"]


def test_the_live_game_turns_alerts_on_and_the_prompt_explains_the_choice():
    game = live_game(encounters="on")
    assert game.state.encounter_alerts and game.black_llm_controller._executor.encounter_alerts
    addendum = game.black_llm_controller.config.live_prompt_addendum
    assert "encounter.alert" in addendum and "approach_opponent to fight" in addendum
    assert "A fight is a form of sabotage" in render_live_mission_block("saboteur", encounters=True)
    off = live_game(encounters="off")
    assert not off.state.encounter_alerts and "encounter.alert" not in off.black_llm_controller.config.live_prompt_addendum
    assert not ActionExecutor().encounter_alerts  # evaluation / replay executors: unchanged


# -- an eliminated opponent is out of the match ---------------------------------------
# Found in run workshop_runs/puzzles/live/20261001-164328: White was eliminated
# at 252 s, yet Black still saw it, was offered approach_opponent (chosen 13
# times in a row against a spy that can no longer fight) and its walks to the
# bookshelf were stopped by alerts. The engine never fights an eliminated spy
# (src/encounters.py), so the facts now say so.

def _eliminate(state, actor_id):
    state.lives[actor_id] = 1
    assert state.lose_life(actor_id, "fight")
    state.update(0.0)


def test_an_eliminated_opponent_is_not_an_encounter():
    from src.llm_controller import enumerate_legal_actions

    state = arena(100)
    assert encounter_alert(build_observation(state, WHITE)) is not None
    _eliminate(state, BLACK)
    obs = build_observation(state, WHITE)
    assert obs.opponent is None and obs.opponent_out_of_match == constants.LIFECYCLE_ELIMINATED
    assert encounter_alert(obs) is None
    legal = enumerate_legal_actions(obs, reachable_only=True)
    assert not any(a["action"] in ("approach_opponent", "avoid_opponent") for a in legal)
    facts = serialize_observation(obs, live_state=True, live_workshop=True)["encounter"]
    assert facts["opponent_out_of_match"] == "ELIMINATED" and facts["fights_over"]
    assert facts["fight_possible_now"] is False and facts["opponent_visible"] is False
    assert "alert" not in facts


def test_a_walk_past_an_eliminated_opponent_is_not_stopped():
    state = arena(300)
    _eliminate(state, BLACK)
    ex = executor()
    obs = build_observation(state, WHITE)
    action, boundary, _ = ex.start(obs, "d1", "move", "door_corridor_main_office", None, obs.mission_time_remaining)
    black, white = state.actors[BLACK], state.actors[WHITE]
    closest = 1e9
    for _ in range(600):
        dx, dy = white.x - black.x, white.y - black.y
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        closest = min(closest, d)
        if d > 30:
            black.x += dx / d * 5  # the body is placed ever closer to White's path
            black.y += dy / d * 5
        state.apply_action(WHITE, action, DT)
        state.update(DT)
        action, boundary = ex.continue_(build_observation(state, WHITE))
        if boundary is not None:
            break
    assert closest < ENCOUNTER_ALERT_DISTANCE  # it did come within alert range
    assert boundary is not None and boundary != OPPONENT_NEARBY
    assert not [e for e in state.event_log if e["type"] == "SPY_FIGHT_STARTED"]


def test_recorded_rules_keep_the_opponent_fact_unchanged():
    recorded = fresh_state()
    recorded.encounters_enabled = True
    obs = build_observation(recorded, WHITE)
    assert obs.opponent_out_of_match is None
    assert "opponent_out_of_match" not in (encounter_facts(obs) or {})
