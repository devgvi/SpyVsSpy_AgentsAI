"""Live safe puzzles (src/puzzles.py): the split code and the riddle.

SPLIT: the safe code comes in two halves revealed by objects in two rooms;
knowing both means knowing `safe_code`. RIDDLE: a clue plus engravings; the
spy works the code out and answers with `enter_safe_code`. Live only - the
recorded games keep one object revealing `safe_code`.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from src import constants, puzzles
from src.actions import ActionType, GameAction
from src.agents.action_lifecycle import ActionExecutor
from src.evaluation.navigation_audit import fresh_state
from src.llm.personality import render_live_mission_block
from src.llm_controller import enumerate_legal_actions, serialize_observation, validate_model_output
from src.observation import build_observation
from tests.test_bugfix_inspection_proximity import act, events
from tests.test_bugfix_live_async import Gate, native_agent, strands_agent, wait_ready
from tests.test_bugfix_live_match_flow import live_game

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID
DT = 1 / 60


def _stand_at(state, object_id, actor_id=WHITE):
    """Put the actor right beside an object (puzzle logic, not navigation, is under test)."""
    from src.agents.action_lifecycle import choose_approach_point

    actor = state.actors[actor_id]
    obj = state.level.get_object(object_id)
    obs = build_observation(state, actor_id)
    point = choose_approach_point(obs, obs.get(object_id))  # where a MOVE to it would end
    assert point is not None, object_id
    actor.x, actor.y = point
    state.update(0.0)
    assert state.current_interaction_target.get(actor_id) == object_id, object_id


def _search(state, object_id, actor_id=WHITE):
    _stand_at(state, object_id, actor_id)
    act(state, ActionType.INTERACT, object_id, actor_id=actor_id)


def _all_keys(state, actor_id=WHITE):
    for item in ("archive_key", "security_keycard"):
        state.actors[actor_id].inventory.add(item)
    for door, st in list(state.door_states.items()):
        if st == "locked":
            state.door_states[door] = "open"
    state._rebuild_nav_grid()


# -- split -------------------------------------------------------------------------------

def test_split_code_halves_live_in_two_rooms_and_combine():
    state = live_game(puzzle="split", spawn="same_room").state
    parts = state.puzzle["parts"]
    level = state.level
    rooms = {level.object_room(level.get_object(o)) for o in parts.values()}
    assert len(rooms) == 2
    assert not any(o.reveals_knowledge == puzzles.SAFE_CODE for o in level.objects)
    _all_keys(state)
    white = state.actors[WHITE]
    _search(state, parts[puzzles.PART_1])
    assert puzzles.PART_1 in white.knowledge and puzzles.SAFE_CODE not in white.knowledge
    assert not build_observation(state, WHITE).get(puzzles.SAFE_ID).can_unlock
    _search(state, parts[puzzles.PART_2])
    assert puzzles.SAFE_CODE in white.knowledge and events(state, "KNOWLEDGE_COMBINED")
    view = serialize_observation(build_observation(state, WHITE), live_state=True, live_workshop=True)["puzzle"]
    assert view["safe_code_halves_known"] == [puzzles.PART_1, puzzles.PART_2] and view["safe_code_known"]


def test_split_is_seeded_and_never_touches_the_game_rng():
    a = live_game(puzzle="split", seed=11).state
    b = live_game(puzzle="split", seed=11).state
    none = live_game(puzzle="none", seed=11).state
    assert a.puzzle == b.puzzle
    assert a.rng.getstate() == none.rng.getstate() and a.dependency_placement == none.dependency_placement


# -- riddle ------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(1, 21))
def test_every_generated_riddle_solves_to_its_code_and_names_objects_unambiguously(seed):
    state = live_game(puzzle="riddle", seed=seed).state
    record = state.puzzle
    assert puzzles.solve_riddle(record) == record["code"] and record["code"].isdigit()
    a, b = record["facts"]["a"], record["facts"]["b"]
    assert f"({a})" in record["clue"] and f"({b})" in record["clue"]  # object ids, not just labels
    rooms = {state.level.object_room(state.level.get_object(o)) for o in (a, b)}
    assert len(rooms) == 2
    values = [t.split(": ")[1] for t in record["inscriptions"].values()]
    assert len(values) == len(set(values))  # decoys never repeat a real fact
    assert not any(o.reveals_knowledge == puzzles.SAFE_CODE for o in state.level.objects)


def _riddle_ready(seed=3):
    """White holds the keys, has read the clue and both engravings, and stands at the safe."""
    state = live_game(puzzle="riddle", seed=seed, spawn="same_room").state
    _all_keys(state)
    record = state.puzzle
    _search(state, record["clue_object"])
    for obj_id in (record["facts"]["a"], record["facts"]["b"]):
        _search(state, obj_id)
    _stand_at(state, puzzles.SAFE_ID)
    return state, record


def test_clue_and_engravings_reach_the_payload_only_once_read():
    state = live_game(puzzle="riddle", seed=3, spawn="same_room").state
    view = build_observation(state, WHITE).puzzle
    assert view["clue"] is None and view["inscriptions_you_have_read"] == {}
    state, record = _riddle_ready()
    payload = serialize_observation(build_observation(state, WHITE), live_state=True, live_workshop=True)
    view = payload["puzzle"]
    assert view["clue"] == record["clue"]
    assert set(view["inscriptions_you_have_read"]) == {record["facts"]["a"], record["facts"]["b"]}
    assert "code" not in view and record["code"] not in str(view.get("how_to_answer"))  # never the answer
    assert "puzzle" not in serialize_observation(build_observation(state, WHITE))  # recorded payload unchanged
    other = build_observation(state, BLACK).puzzle
    assert other["clue"] is None and other["inscriptions_you_have_read"] == {}  # per-spy knowledge


def test_inspecting_reads_an_engraving_too():
    state = live_game(puzzle="riddle", seed=3, spawn="same_room").state
    _all_keys(state)
    target = state.puzzle["facts"]["a"]
    _stand_at(state, target)
    act(state, ActionType.INSPECT, target)
    assert target in build_observation(state, WHITE).puzzle["inscriptions_you_have_read"]


def test_the_right_code_opens_the_safe():
    state, record = _riddle_ready()
    legal = enumerate_legal_actions(build_observation(state, WHITE), reachable_only=True)
    assert {"action": "enter_safe_code", "target_id": puzzles.SAFE_ID} in legal
    act(state, ActionType.INTERACT, puzzles.SAFE_ID)  # plain interact without the code: still locked
    assert not state.actors[WHITE].inventory.has("secret_documents")
    state.apply_action(WHITE, GameAction(ActionType.INTERACT, target_id=puzzles.SAFE_ID, code=record["code"]), DT)
    assert puzzles.SAFE_CODE in state.actors[WHITE].knowledge and events(state, "SAFE_CODE_SOLVED")
    assert state.actors[WHITE].inventory.has("secret_documents")
    assert not any(a["action"] == "enter_safe_code"
                   for a in enumerate_legal_actions(build_observation(state, WHITE), reachable_only=True))


def test_a_wrong_code_costs_time_and_reveals_the_spy():
    state, record = _riddle_ready()
    wrong = str(int(record["code"]) + 1)
    state.apply_action(WHITE, GameAction(ActionType.INTERACT, target_id=puzzles.SAFE_ID, code=wrong), DT)
    assert puzzles.SAFE_CODE not in state.actors[WHITE].knowledge
    assert state.actors[WHITE].action_cooldown == puzzles.WRONG_CODE_SECONDS
    assert state.actors[BLACK].alarm_reveal_timer == constants.ALARM_REVEAL_DURATION  # Black now sees White
    assert events(state, "SAFE_CODE_WRONG")[-1]["entered"] == wrong
    assert build_observation(state, WHITE).puzzle["code_attempts"] == 1


def test_validation_accepts_enter_safe_code_with_a_code_only_in_a_riddle_match():
    state, record = _riddle_ready()
    obs = build_observation(state, WHITE)
    legal = serialize_observation(obs, live_state=True, live_workshop=True)["legal_actions"]
    ok = validate_model_output(obs, {"action": "enter_safe_code", "target_id": puzzles.SAFE_ID,
                                     "code": record["code"], "reason": "solved"}, legal)
    assert ok.valid and ok.action.code == record["code"] and ok.validated_action["code"] == record["code"]
    alias = validate_model_output(obs, {"action": "enter_code", "code": int(record["code"])}, legal)
    assert alias.valid and alias.action.target_id == puzzles.SAFE_ID  # alias + implicit target
    missing = validate_model_output(obs, {"action": "enter_safe_code", "target_id": puzzles.SAFE_ID}, legal)
    assert not missing.valid and missing.error_type == "missing_code"
    plain = fresh_state()
    plain_obs = build_observation(plain, WHITE)
    assert not validate_model_output(plain_obs, {"action": "enter_safe_code", "code": "1"}, []).valid


def test_the_executor_carries_the_code_to_the_engine():
    state, record = _riddle_ready()
    ex = ActionExecutor()
    obs = build_observation(state, WHITE)
    action, boundary, _ = ex.start(obs, "d", "enter_safe_code", puzzles.SAFE_ID, None, obs.mission_time_remaining,
                                   code=record["code"])
    assert action.type is ActionType.INTERACT and action.code == record["code"]


@pytest.mark.parametrize("close_approach", [False, True])  # True: LiveGame's executors
@pytest.mark.parametrize("mode", ["live_async", "evaluation"])  # evaluation: the simulated-clock lab
@pytest.mark.parametrize("runtime", ["native", "strands"])
def test_both_runtimes_enter_the_code_end_to_end(runtime, mode, close_approach):
    state, record = _riddle_ready()
    gate = Gate(lambda p: {"action": "enter_safe_code", "target_id": puzzles.SAFE_ID, "code": record["code"],
                           "reason": "the clue gives it"}, auto=True)
    agent = (native_agent if runtime == "native" else strands_agent)(WHITE, gate, mode=mode)
    agent._executor.close_approach = close_approach
    if runtime == "strands":
        agent.riddle_tool = True
    for _ in range(200):
        action = agent.get_action(build_observation(state, WHITE))
        if agent._thinking_in_progress:
            wait_ready(agent)
            continue
        state.apply_action(WHITE, action, DT)
        state.update(DT)
        if events(state, "SAFE_CODE_SOLVED"):
            break
    assert events(state, "SAFE_CODE_SOLVED") and state.actors[WHITE].inventory.has("secret_documents")
    assert gate.calls == 1


def test_strands_tool_is_unchanged_outside_the_riddle():
    pytest.importorskip("strands")
    from tests.test_bugfix_live_async import strands_agent as make

    plain = make(WHITE, Gate(lambda p: {"action": "wait"}))
    capture = []
    tool = plain._make_tool(capture)
    assert "code" not in str(getattr(tool, "tool_spec", {}))
    plain.riddle_tool = True
    assert "code" in str(getattr(plain._make_tool(capture), "tool_spec", {}))


def test_live_game_defaults_and_prompt():
    game = live_game()
    assert game.live_config.puzzle == "riddle" and game.state.puzzle_mode == "riddle"
    assert "SAFE RIDDLE" in game.black_llm_controller.config.live_prompt_addendum
    assert "SAFE CODE: the ambassador_safe code is split" in render_live_mission_block("saboteur", puzzle="split")
    none = live_game(puzzle="none")
    assert none.state.puzzle is None and any(o.reveals_knowledge == puzzles.SAFE_CODE for o in none.state.level.objects)
    recorded = fresh_state()
    assert recorded.puzzle_mode == "none" and build_observation(recorded, WHITE).puzzle is None


@pytest.mark.parametrize("mode", [puzzles.SPLIT, puzzles.RIDDLE])
def test_puzzle_set_up_event_keeps_its_type_and_reaches_telemetry(mode):
    # run 20261001-164328: the record's own "type" key overwrote the event
    # type, so PUZZLE_SET_UP never reached the telemetry file
    state = live_game(puzzle=mode, seed=3).state
    setup = events(state, "PUZZLE_SET_UP")
    assert len(setup) == 1 and setup[0]["puzzle"] == mode
    assert "code" not in setup[0] and "clue" not in setup[0]
    assert not [e for e in state.event_log if e["type"] in puzzles.PUZZLES]
