"""Bugfix: structured, actor-specific exploration state (E1-E12, §25-27, §36-38).

Real level, real engine. The "two-door lobby" of the spec is the corridor:
Door A = door_reception_corridor (to reception), Door B =
door_corridor_main_office (to main_office); the corridor's other two doors
are locked for a spy without key/keycard.
"""
from __future__ import annotations

import json
import os
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.actions import ActionType, GameAction
from src.agents.action_lifecycle import TARGET_REACHED
from src.evaluation.navigation_audit import fresh_state, run_move
from src.llm_controller import compact_observation_for_model, enumerate_legal_actions, serialize_observation
from src.observation import COMPLETE, PARTIAL, UNVISITED, build_observation, room_exploration_status

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID
DOOR_A, DOOR_B = "door_reception_corridor", "door_corridor_main_office"


def _walk(state, target, actor=WHITE):
    a = state.actors[actor]
    if actor != WHITE:  # run_move drives White; swap roles for Black
        raise NotImplementedError
    row = run_move(state, target, (a.x, a.y), 1 / 60)
    assert row["boundary"] == TARGET_REACHED, row
    return row


def _exploration(state, actor=WHITE):
    return build_observation(state, actor).exploration


def _model_exploration(state, actor=WHITE):
    payload = compact_observation_for_model(serialize_observation(build_observation(state, actor), live_state=True))
    return payload["exploration"]


def _act(state, action_type, target, actor=WHITE):
    state.apply_action(actor, GameAction(action_type, target_id=target), 1 / 60)
    for _ in range(600):  # let the action's time cost run out
        state.update(1 / 60)
        if not state.actors[actor].is_busy:
            break


@pytest.fixture
def corridor_state():
    """White starts in main_office (default doors) and walks into the corridor."""
    state = fresh_state(doors_open=False)
    _walk(state, DOOR_B)  # main_office -> corridor
    assert state.level.room_id_at(state.actors[WHITE].x, state.actors[WHITE].y) == "corridor"
    return state


def test_e1_entering_a_room_the_first_time(corridor_state):
    exp = _exploration(corridor_state)
    assert exp["current_room"] == "corridor"
    room = exp["rooms"]["corridor"]
    assert room["traversed_exits"] == {DOOR_B: "main_office"}
    assert room["untraversed_exits"] == [DOOR_A]
    assert room["blocked"] == {"door_archives_corridor": "locked:key", "door_security_corridor": "locked:keycard"}
    assert room["status"] == PARTIAL


def test_e2_e3_inspect_one_of_two_objects_then_search_it():
    state = fresh_state(doors_open=False)
    _walk(state, "desk_01")
    _act(state, ActionType.INSPECT, "desk_01")
    room = _exploration(state)["rooms"]["main_office"]
    assert "desk_01" not in room["uninspected"] and "bookshelf_03" in room["uninspected"]
    assert "desk_01" in room["unsearched"]  # inspected != searched
    _act(state, ActionType.INTERACT, "desk_01")
    room = _exploration(state)["rooms"]["main_office"]
    assert "desk_01" not in room["unsearched"] and "bookshelf_03" in room["unsearched"]
    assert room["status"] == PARTIAL


def test_e4_to_e8_two_door_acceptance_case(corridor_state):
    """§26: enter the lobby, both doors known, traverse Door A, come back."""
    state = corridor_state
    _walk(state, DOOR_A)  # corridor -> reception
    assert state.level.room_id_at(state.actors[WHITE].x, state.actors[WHITE].y) == "reception"
    _walk(state, DOOR_A)  # reception -> corridor
    exp = _exploration(state)
    assert exp["current_room"] == "corridor"
    corridor = exp["rooms"]["corridor"]
    assert corridor["traversed_exits"][DOOR_A] == "reception"  # E6: used door traversed
    assert corridor["traversed_exits"][DOOR_B] == "main_office"  # (entered through B earlier)
    # the spec's exact case: A traversed, the other open door not yet -> use the reception room
    reception = exp["rooms"]["reception"]
    assert reception["traversed_exits"] == {DOOR_A: "corridor"} and reception["status"] == PARTIAL
    # the model-facing form carries the same facts
    model = _model_exploration(state)["rooms"]["corridor"]
    assert model["traversed_exits"] == corridor["traversed_exits"] and "untraversed_exits" not in model


def test_mandatory_two_door_case_exactly_as_specified():
    """§26 literally: actor enters the lobby, both doors discovered, traverses
    Door A, returns. Door A traversed, Door B discovered but not, lobby PARTIAL."""
    state = fresh_state(doors_open=False)
    a = state.actors[WHITE]
    # placed in the corridor as if it had spawned there: a teleport from the
    # spawn room would otherwise count as a walk through the door between them
    a.x, a.y = 720.0, 600.0
    a.last_room, a.visited_rooms, a.traversed_doors = None, set(), set()
    state.update(0.0)
    first = _exploration(state)["rooms"]["corridor"]
    assert set(first["untraversed_exits"]) == {DOOR_A, DOOR_B} and first["traversed_exits"] == {}
    _walk(state, DOOR_A)  # corridor -> reception
    _walk(state, DOOR_A)  # reception -> corridor
    exp = _exploration(state)
    lobby = exp["rooms"]["corridor"]
    assert exp["current_room"] == "corridor" and "corridor" in state.actors[WHITE].visited_rooms
    assert lobby["traversed_exits"] == {DOOR_A: "reception"}
    assert lobby["untraversed_exits"] == [DOOR_B]
    assert lobby["status"] == PARTIAL
    model = _model_exploration(state)["rooms"]["corridor"]
    assert model == {"blocked": lobby["blocked"], "status": PARTIAL, "traversed_exits": {DOOR_A: "reception"},
                     "untraversed_exits": [DOOR_B]}


def test_e5_e7_e8_return_to_a_room_with_one_unused_door():
    """Main office has two doors; leave through one, return: the other stays untraversed."""
    state = fresh_state(doors_open=False)
    _walk(state, DOOR_B)  # main_office -> corridor
    _walk(state, DOOR_B)  # corridor -> main_office
    exp = _exploration(state)
    office = exp["rooms"]["main_office"]
    assert exp["current_room"] == "main_office"
    assert office["traversed_exits"] == {DOOR_B: "corridor"}
    assert office["untraversed_exits"] == ["door_main_office_ambassador"]  # E7
    assert office["status"] == PARTIAL  # E8
    assert set(office["unsearched"]) == {"bookshelf_03", "desk_01", "plant_03"}


def test_e9_e10_finishing_the_room_makes_it_complete():
    state = fresh_state(doors_open=False)
    _walk(state, DOOR_B)
    _walk(state, DOOR_B)
    for obj in ("desk_01", "bookshelf_03", "plant_03"):
        _walk(state, obj)
        _act(state, ActionType.INTERACT, obj)
    assert room_exploration_status(_exploration(state), "main_office") == PARTIAL  # one exit left
    _walk(state, "door_main_office_ambassador")
    _walk(state, "door_main_office_ambassador")
    office = _exploration(state)["rooms"]["main_office"]
    assert office["status"] == COMPLETE and office["unsearched"] == [] and office["untraversed_exits"] == []
    assert office["uninspected"]  # trap inspection is informative, not required


def test_e11_unvisited_rooms_and_untraversed_destinations_do_not_leak(corridor_state):
    exp = _exploration(corridor_state)
    assert set(exp["rooms"]) == {"corridor", "main_office"}
    text = json.dumps(exp)
    for hidden in ("security_room", "archives", "reception", "filing_cabinet_01", "bookshelf_01", "sofa_01"):
        assert f'"{hidden}"' not in text, hidden  # no unvisited room, none of its objects, no destination
    assert "contains" not in text and "passport" not in text and "secret_documents" not in text
    assert room_exploration_status(exp, "security_room") == UNVISITED


def test_e12_one_actors_exploration_never_leaks_to_the_other(corridor_state):
    white = _exploration(corridor_state, WHITE)
    black = _exploration(corridor_state, BLACK)
    assert "corridor" in white["rooms"]
    assert "corridor" not in black["rooms"]
    assert corridor_state.actors[BLACK].traversed_doors == set()


def test_furniture_acceptance_case():
    """§27: MOVE_TO cabinet -> valid approach point, near object, INSPECT legal."""
    state = fresh_state()
    row = run_move(state, "filing_cabinet_01", (720.0, 240.0), 1 / 60)
    assert row["boundary"] == TARGET_REACHED and row["near_object_id"] == "filing_cabinet_01"
    legal = enumerate_legal_actions(build_observation(state, WHITE), reachable_only=True)
    assert {"action": "inspect", "target_id": "filing_cabinet_01"} in legal


def test_blocked_objects_do_not_keep_a_room_partial():
    state = fresh_state()
    # the safe needs a code White does not know: listed as blocked, not unsearched
    a = state.actors[WHITE]
    a.x, a.y = 720.0, 1300.0
    state.update(0.0)
    room = _exploration(state)["rooms"]["ambassador_office"]
    assert room["blocked"].get("ambassador_safe") == "locked:code"
    assert "ambassador_safe" not in room["unsearched"]


def test_representation_is_deterministic_and_sorted(corridor_state):
    a = json.dumps(_model_exploration(corridor_state))
    b = json.dumps(_model_exploration(corridor_state))
    assert a == b
    rooms = _exploration(corridor_state)["rooms"]
    assert list(rooms) == sorted(rooms)
    for room in rooms.values():
        for key in ("unsearched", "uninspected", "untraversed_exits"):
            assert room[key] == sorted(room[key])


def test_exploration_is_live_only_so_recorded_payloads_are_unchanged(corridor_state):
    obs = build_observation(corridor_state, WHITE)
    assert "exploration" not in serialize_observation(obs)
    assert "exploration" not in compact_observation_for_model(serialize_observation(obs))
    assert "exploration" in compact_observation_for_model(serialize_observation(obs, live_state=True))


def test_visited_rooms_compatibility_is_preserved(corridor_state):
    obs = build_observation(corridor_state, WHITE)
    payload = serialize_observation(obs, live_state=True)
    assert payload["memory"]["visited_rooms"] == sorted(corridor_state.actors[WHITE].visited_rooms)


# -- runtime parity (§36-38) ---------------------------------------------------------


class _CapturingClient:
    def __init__(self):
        self.payloads = []
        self.model_name = "capture"

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        from src.llm.clients.base import ModelResponse
        self.payloads.append(json.loads(user_prompt))
        return ModelResponse(payload={"action": "wait", "reason": "x"}, raw_text="{}", provider="mock", model="capture", latency_ms=1.0)


def _wait_for(agent, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and getattr(agent, "_thinking_in_progress", False):
        time.sleep(0.01)


def test_native_and_strands_send_the_same_exploration_and_move_facts(corridor_state, monkeypatch):
    pytest.importorskip("strands")
    from src.agents.strands_agent import StrandsAgentAdapter
    from src.llm_controller import LLMController

    obs = build_observation(corridor_state, WHITE)
    native_client = _CapturingClient()
    native = LLMController(WHITE, model_client=native_client)
    native.get_action(obs)
    _wait_for(native)

    captured = []
    adapter = StrandsAgentAdapter(WHITE, agent_factory=lambda *a, **k: None)
    real_invoke = adapter._invoke_strands

    def capture(model_payload, repair_instruction):
        captured.append(json.loads(json.dumps(model_payload)))
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(adapter, "_invoke_strands", capture)
    adapter.get_action(obs)
    _wait_for(adapter)
    assert native_client.payloads and captured
    n, s = native_client.payloads[0], captured[0]
    assert n["exploration"] == s["exploration"]
    assert n["legal_actions_compact"]["move"] == s["legal_actions_compact"]["move"]
    assert "filing_cabinet_01" not in n["legal_actions_compact"]["move"]  # unreachable: not offered


def test_provider_neutral_same_payload_for_any_provider_name(corridor_state):
    from src.llm_controller import LLMController
    obs = build_observation(corridor_state, WHITE)
    payloads = []
    for provider in ("bedrock", "openai"):
        client = _CapturingClient()
        agent = LLMController(WHITE, model_client=client, provider_name=provider, model_name="m")
        agent.get_action(obs)
        _wait_for(agent)
        payloads.append(client.payloads[0])
    assert payloads[0]["exploration"] == payloads[1]["exploration"]
    assert payloads[0]["legal_actions_compact"] == payloads[1]["legal_actions_compact"]
