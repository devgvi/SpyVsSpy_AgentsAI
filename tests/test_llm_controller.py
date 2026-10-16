from src.actions import ActionType, GameAction
from src.llm_controller import LLMController, MockModelClient, enumerate_legal_actions, serialize_observation
from dataclasses import replace
from src.observation import Observation, ObservedObject
import json
import time
import pygame
from src.game import Game


class SlowModelClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_action(self, *args, **kwargs):
        self.calls += 1
        time.sleep(0.25)
        return self.response


def _make_observation():
    room_obj = ObservedObject(
        id="desk_01",
        type="desk",
        label="Desk",
        position=(10, 10),
        collision_size=(20, 20),
        interactable=True,
        trappable=True,
        searched=False,
        inspected=False,
        trap_status="unknown",
        door_open=None,
        locked=False,
        can_unlock=False,
    )
    return Observation(
        actor_id="white",
        seed=42,
        position=(5.0, 5.0),
        facing="east",
        stunned=False,
        stunned_seconds=0.0,
        busy=False,
        busy_seconds=0.0,
        mission_time_remaining=120.0,
        mission_complete=False,
        inventory=frozenset({"archive_key"}),
        knowledge=frozenset(),
        required_items=("archive_key",),
        trap_inventory={},
        # M5D.4: legal_actions now gates INSPECT/INTERACT/PLACE_TRAP/DISARM
        # on proximity (nearby_object_id) - desk_01 is the actor's own
        # interaction target in this fixture, so it must be set here for
        # inspect/interact actions on it to be legal at all.
        nearby_object_id="desk_01",
        items_found_here=frozenset(),
        objects=(room_obj,),
        nav_grid=None,
        log_event=lambda *args, **kwargs: None,
        current_room="reception",
        known_rooms=frozenset({"reception"}),
        known_doors=frozenset(),
        known_locked_objects={},
        known_item_locations={},
        visible_other_actor=False,
        opponent=None,
        last_seen_opponent=None,
        revealed_opponent=None,
        revealed_opponent_seconds=0.0,
    )


def test_legal_actions_are_enumerated_without_hidden_state():
    observation = _make_observation()
    legal = enumerate_legal_actions(observation)
    assert legal
    assert all(item["action"] in {"wait", "move", "interact", "inspect", "place_trap", "disarm"} for item in legal)


def test_llm_controller_uses_mock_model_and_validates_action():
    observation = _make_observation()
    client = MockModelClient({"action": "inspect", "target_id": "desk_01"})
    controller = LLMController("white", client)

    action = controller.get_action(observation)

    assert isinstance(action, GameAction)
    assert action.type == ActionType.INSPECT
    assert action.target_id == "desk_01"
    assert client.calls == 1


def test_invalid_model_output_falls_back_to_wait():
    observation = _make_observation()
    client = MockModelClient({"action": "fly", "target_id": "desk_01"})
    controller = LLMController("white", client)

    action = controller.get_action(observation)

    assert action.type == ActionType.WAIT


def test_controller_does_not_requery_on_time_only_change():
    observation = _make_observation()
    client = MockModelClient({"action": "inspect", "target_id": "desk_01"})
    controller = LLMController("white", client)

    controller.get_action(observation)
    observation2 = replace(observation, mission_time_remaining=observation.mission_time_remaining - 1.0)
    controller.get_action(observation2)

    assert client.calls == 1


def test_default_evaluation_mode_waits_while_model_is_thinking():
    observation = _make_observation()
    client = SlowModelClient({"action": "inspect", "target_id": "desk_01"})
    controller = LLMController("white", client)

    action = controller.get_action(observation)
    assert action.type == ActionType.WAIT
    assert controller.status in {"THINKING", "EXECUTING", "IDLE"}


def test_playable_mode_allows_provisional_actions_while_model_is_thinking():
    observation = _make_observation()
    door_obj = ObservedObject(
        id="door_01",
        type="door",
        label="Door",
        position=(30, 10),
        collision_size=(20, 20),
        interactable=False,
        trappable=False,
        searched=False,
        inspected=False,
        trap_status="unknown",
        door_open=None,
        locked=False,
        can_unlock=False,
    )
    observation = replace(observation, objects=(observation.objects[0], door_obj), current_room="reception")
    client = SlowModelClient({"action": "inspect", "target_id": "desk_01"})
    controller = LLMController("white", client, execution_mode="playable")

    action = controller.get_action(observation)
    assert action.type == ActionType.MOVE
    assert action.target_id == "door_01" or action.direction is not None


def test_game_default_llm_mode_is_evaluation(monkeypatch):
    monkeypatch.setattr(pygame, "init", lambda: None)
    monkeypatch.setattr(pygame.display, "set_caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(pygame.display, "set_mode", lambda *args, **kwargs: pygame.Surface((800, 600)))

    game = Game(seed=123, white_controller="llm", llm_provider="mock", llm_model="mock-model")
    assert game.bot_controller.execution_mode == "evaluation"
    assert game.bot_controller.config.execution_mode == "evaluation"


def test_target_is_normalized_and_required_for_targeted_actions():
    observation = _make_observation()

    client = MockModelClient({"action": "inspect", "object_id": "desk_01"})
    controller = LLMController("white", client)
    action = controller.get_action(observation)
    assert action.type == ActionType.INSPECT
    assert action.target_id == "desk_01"

    client2 = MockModelClient({"action": "inspect"})
    controller2 = LLMController("white", client2)
    action2 = controller2.get_action(observation)
    assert action2.type == ActionType.WAIT


def test_serialized_observation_excludes_hidden_truth():
    observation = _make_observation()
    payload = serialize_observation(observation)

    assert payload["self"]["actor_id"] == "white"
    assert payload["mission"]["required_items"] == ["archive_key"]
    assert "hidden_trap" not in str(payload)


def test_compact_observation_for_model_reduces_size_and_keeps_keys():
    observation = _make_observation()
    payload = serialize_observation(observation)
    from src.llm_controller import compact_observation_for_model

    compact = compact_observation_for_model(payload)
    assert isinstance(compact, dict)
    assert compact.get("schema_version") == "obs-v2"
    # compact representation should be smaller than full observation
    full = json.dumps(payload, separators=(",", ":"))
    small = json.dumps(compact, separators=(",", ":"))
    assert len(small) < len(full)
    # must include compact legal actions mapping
    assert "legal_actions_compact" in compact
