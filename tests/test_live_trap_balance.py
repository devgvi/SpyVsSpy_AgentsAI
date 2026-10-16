"""Live trap balance: INSPECT costs real time, disarming an enemy trap can fail.

Found in live run workshop_runs/live/live/20261001-110229 (Black SABOTEUR vs
White CAUTIOUS): INSPECT cost only 0.5 s more than a blind search, White
inspected every object it searched, detected both of Black's traps it reached
and disarmed both - no trap ever went off. Live only (GameState.live_trap_rules);
the recorded rules are checked unchanged.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.navigation_audit import fresh_state
from src.llm.personality import render_live_mission_block
from src.llm_controller import enumerate_legal_actions
from src.mission_time import action_cost_facts
from src.observation import build_observation
from tests.test_bugfix_inspection_proximity import _trappable_in_start_room, act, events, move_to
from tests.test_bugfix_live_match_flow import live_game

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID
DT = 1 / 60


class Roll:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _detected_enemy_trap(state, trap_type="bomb"):
    """Black's trap on an object next to White, already detected by White."""
    obj = _trappable_in_start_room(state)
    move_to(state, obj.id)
    entry = state._object_entry(obj.id)
    entry["trap"] = {"type": trap_type, "owner_id": BLACK, "armed": True}
    entry["detected_by"][WHITE] = trap_type
    return obj


def test_inspect_costs_more_in_live_play_and_the_quote_matches_the_engine():
    live = live_game().state
    recorded = fresh_state()
    for state, expected in ((live, constants.LIVE_TIME_COST_INSPECT), (recorded, constants.TIME_COST_INSPECT)):
        obj = _trappable_in_start_room(state)
        move_to(state, obj.id)
        obs = build_observation(state, WHITE)
        quoted = action_cost_facts(obs, enumerate_legal_actions(obs, reachable_only=True))["inspect"][obj.id]
        state.apply_action(WHITE, GameAction(ActionType.INSPECT, target_id=obj.id), DT)
        assert quoted == expected == state.actors[WHITE].action_cooldown
    assert constants.LIVE_TIME_COST_INSPECT >= 4 * constants.TIME_COST_SEARCH


def test_a_failed_disarm_sets_the_trap_off_under_the_disarmer():
    state = live_game().state
    obj = _detected_enemy_trap(state, "bomb")
    state.disarm_rng = Roll(0.99)  # >= 0.55: fails
    act(state, ActionType.DISARM, obj.id)
    assert events(state, "TRAP_DISARM_FAILED", obj.id) and not events(state, "TRAP_DISARMED")
    trig = events(state, "TRAP_TRIGGERED", obj.id)[-1]
    assert trig["actor_id"] == WHITE and trig["trap_owner_id"] == BLACK
    assert state.lives[WHITE] == 2 and state.trap_on(obj.id) is None  # spent
    attempt = events(state, "DISARM_ATTEMPTED", obj.id)[-1]
    assert attempt["success_probability"] == constants.DISARM_SUCCESS_PROBABILITY["bomb"] and not attempt["success"]


def test_a_successful_disarm_removes_the_trap_safely():
    state = live_game().state
    obj = _detected_enemy_trap(state, "snare")
    state.disarm_rng = Roll(0.10)
    act(state, ActionType.DISARM, obj.id)
    assert events(state, "TRAP_DISARMED", obj.id) and not events(state, "TRAP_TRIGGERED")
    assert state.lives[WHITE] == 3 and events(state, "DISARM_ATTEMPTED", obj.id)[-1]["success"]


def test_removing_your_own_trap_always_works():
    state = live_game().state
    obj = _trappable_in_start_room(state)
    move_to(state, obj.id)
    act(state, ActionType.INSPECT, obj.id)
    act(state, ActionType.INTERACT, obj.id)
    act(state, ActionType.PLACE_TRAP, obj.id, trap_type="bomb")
    state.self_placed_traps[WHITE][obj.id]["fresh"] = False
    state.disarm_rng = Roll(0.999)
    act(state, ActionType.DISARM, obj.id)
    assert events(state, "TRAP_REMOVED", obj.id) and not events(state, "DISARM_ATTEMPTED")


def test_recorded_rules_unchanged_and_shared_rng_untouched():
    state = fresh_state()
    assert not state.live_trap_rules
    obj = _detected_enemy_trap(state, "bomb")
    before = state.rng.getstate()
    act(state, ActionType.DISARM, obj.id)
    assert events(state, "TRAP_DISARMED", obj.id) and not events(state, "DISARM_ATTEMPTED")
    assert state.rng.getstate() == before
    live = live_game().state
    obj = _detected_enemy_trap(live, "alarm")
    before = live.rng.getstate()
    act(live, ActionType.DISARM, obj.id)
    assert live.rng.getstate() == before  # disarm rolls use their own RNG


def test_disarm_odds_over_many_seeds():
    for trap_type, p in constants.DISARM_SUCCESS_PROBABILITY.items():
        state = live_game(seed=1).state
        state.disarm_rng.seed(f"disarm-odds:{trap_type}")
        wins = sum(state.disarm_rng.random() < p for _ in range(4000)) / 4000
        assert abs(wins - p) < 0.03
    assert max(constants.DISARM_SUCCESS_PROBABILITY.values()) < 1.0


def test_seeded_disarm_outcome_is_reproducible():
    def outcome():
        state = live_game(seed=99).state
        obj = _detected_enemy_trap(state, "electric")
        act(state, ActionType.DISARM, obj.id)
        return events(state, "DISARM_ATTEMPTED", obj.id)[-1]["roll"]
    assert outcome() == outcome()


def test_the_model_is_told_the_costs_and_odds():
    state = live_game().state
    obj = _detected_enemy_trap(state, "bomb")
    obs = build_observation(state, WHITE)
    costs = action_cost_facts(obs, enumerate_legal_actions(obs, reachable_only=True))
    assert costs["disarm_success_chance"][obj.id] == constants.DISARM_SUCCESS_PROBABILITY["bomb"]
    block = render_live_mission_block("cautious", lives=3, scoring=True, trap_risk=True)
    assert f"INSPECT costs {constants.LIVE_TIME_COST_INSPECT:g} s" in block and "bomb 55%" in block
    game = live_game()
    assert "INSPECT costs" in game.black_llm_controller.config.live_prompt_addendum
    assert "INSPECT costs" not in render_live_mission_block("cautious")


# -- live run 20261001-114245: fights cost a heart, urgency rises with the clock --------

def _forced_fight(state, winner):
    b, w = state.actors[BLACK], state.actors[WHITE]
    b.x, b.y = 700, 1100
    w.x, w.y = 715, 1100
    state.recorded_fight_winners = [winner]
    state.update(0.0)
    return state.fights[-1]


def test_losing_a_fight_costs_a_life_and_a_ko():
    state = live_game(spawn="same_room", encounters="on").state
    record = _forced_fight(state, BLACK)
    assert record["loser_id"] == WHITE and record["penalty"]["life_lost"]
    assert state.lives == {BLACK: 3, WHITE: 2} and WHITE in state.knocked_out
    lost = [e for e in state.event_log if e["type"] == "LIFE_LOST"][-1]
    assert lost["actor_id"] == WHITE and lost["cause"] == "fight"
    assert state.trap_hits.get(WHITE, 0) == 0  # a fight is not a trap hit


def test_a_fight_lost_on_the_last_life_eliminates():
    state = live_game(spawn="same_room", encounters="on").state
    state.lives[WHITE] = 1
    _forced_fight(state, BLACK)
    assert state.eliminated[WHITE] and WHITE not in state.knocked_out
    assert state.lifecycle(WHITE) == constants.LIFECYCLE_ELIMINATED


def test_recorded_fight_rule_unchanged():
    state = fresh_state()
    state.encounters_enabled = True
    record = _forced_fight(state, BLACK)
    assert record["penalty"]["type"] == "stun" and not state.lives_enabled


def test_urgency_rises_with_the_clock_in_live_play():
    from src.mission_time import CRITICAL, RELAXED, TIGHT, urgency_for

    # 480 s mission, every item held, exit 1 s away: the known work is tiny
    assert urgency_for(400, 1.3, 480, share_floor=True) == RELAXED
    assert urgency_for(200, 1.3, 480, share_floor=True) == TIGHT
    assert urgency_for(100, 1.3, 480, share_floor=True) == CRITICAL
    assert urgency_for(100, 1.3, 480) == RELAXED  # without the floor (recorded rule)


def test_prompt_tells_how_urgency_and_fights_change_the_risk():
    block = render_live_mission_block("cautious", lives=3, scoring=True, trap_risk=True, fights_cost_a_life=True)
    assert "CRITICAL (last quarter): no inspections" in block and "Losing a fight also costs one" in block
    game = live_game(encounters="on")
    assert "Losing a fight also costs one" in game.black_llm_controller.config.live_prompt_addendum


# -- trapping after a search ---------------------------------------------------------
# Found in live run workshop_runs/live/live/20261001-180750 (saboteur vs
# saboteur): 0 traps placed, place_trap legal in 2 of 84 decisions. Trapping
# needed an INSPECT (4 s since the balance change) on top of the live
# "search before you trap" rule, so the saboteurs, who rarely inspect, could
# almost never trap. Live: a search is knowing the object (it sets off any
# enemy trap on it), so a searched object can be trapped.

def test_a_searched_object_can_be_trapped_without_inspecting_it_in_live_play():
    from tests.test_bugfix_inspection_proximity import legal

    state = live_game().state
    obj = _trappable_in_start_room(state)
    move_to(state, obj.id)
    assert not legal(build_observation(state, WHITE), "place_trap", obj.id)  # unsearched: search first
    act(state, ActionType.INTERACT, obj.id)
    obs = build_observation(state, WHITE)
    assert obs.trap_after_search and not obs.get(obj.id).inspected
    assert legal(obs, "place_trap", obj.id)
    act(state, ActionType.PLACE_TRAP, obj.id, trap_type="alarm")
    assert events(state, "TRAP_PLACED", obj.id) and not events(state, "OBJECT_INSPECTED", obj.id)
    assert "PLACING A TRAP" in render_live_mission_block("saboteur", lives=3, trap_risk=True)


def test_recorded_rules_still_need_an_inspection_before_trapping():
    from tests.test_bugfix_inspection_proximity import legal

    state = fresh_state()
    obj = _trappable_in_start_room(state)
    move_to(state, obj.id)
    act(state, ActionType.INTERACT, obj.id)
    obs = build_observation(state, WHITE)
    assert not obs.trap_after_search and not legal(obs, "place_trap", obj.id)
    act(state, ActionType.PLACE_TRAP, obj.id, trap_type="alarm")
    assert not events(state, "TRAP_PLACED", obj.id)  # "Inspect the ... before trapping it."
    assert "PLACING A TRAP" not in render_live_mission_block("saboteur")
