import json

from src.actions import ActionType, GameAction
from src.llm.clients.base import ModelCapabilities, ModelResponse
from src.llm.clients.mock_client import MockModelClient
from src.llm.clients.openai_compatible_client import OpenAICompatibleClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController
from src.observation import Observation, ObservedObject


def _make_observation():
    desk = ObservedObject(
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
        lock_type=None,
        locked=False,
        can_unlock=False,
    )
    return Observation(
        actor_id="white",
        seed=7,
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
        # M5D.4: legal_actions now gates INSPECT/INTERACT on proximity -
        # desk_01 is the actor's own interaction target in this fixture.
        nearby_object_id="desk_01",
        items_found_here=frozenset(),
        objects=(desk,),
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


def test_provider_independence_and_normalized_response():
    observation = _make_observation()
    mock = MockModelClient({"action": "INSPECT", "target_id": "desk_01", "reason": "Need to inspect the desk"})
    controller = LLMController("white", model_client=mock)
    action = controller.get_action(observation)

    assert isinstance(action, GameAction)
    assert action.type == ActionType.INSPECT
    assert action.target_id == "desk_01"


def test_openai_compatible_client_normalizes_response():
    def fake_request(payload):
        return {
            "choices": [{
                "message": {"content": json.dumps({"action": "inspect", "target_id": "desk_01", "reason": "Desk check"})},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            "id": "req_42",
        }

    client = OpenAICompatibleClient(
        provider_name="openai_compatible",
        model_name="local-test",
        base_url="http://example.test/v1",
        api_key="demo-key",
        request_fn=fake_request,
    )
    resp = client.generate_action("system", "user", {"type": "object"})

    assert isinstance(resp, ModelResponse)
    assert resp.provider == "openai_compatible"
    assert resp.model == "local-test"
    assert resp.payload["action"] == "inspect"
    assert resp.input_tokens == 11
    assert resp.output_tokens == 5
    assert resp.finish_reason == "stop"


def test_telemetry_emits_episode_and_decision_logs():
    logger = TelemetryLogger(path=None)
    logger.emit_episode_start(episode_id="ep_1", seed=7, white_controller="llm", black_controller="human", provider="mock", model="mock-model")

    observation = _make_observation()
    controller = LLMController("white", model_client=MockModelClient({"action": "WAIT"}), telemetry=logger)
    action = controller.get_action(observation)

    logger.emit_episode_end(
        episode_id="ep_1",
        escaped=False,
        success=False,
        llm_calls=1,
        valid_actions=1,
        invalid_actions=0,
        repair_attempts=1,
        fallbacks=1,
        mean_latency_ms=5.0,
        min_latency_ms=3.0,
        max_latency_ms=9.0,
        valid_action_rate=1.0,
        total_input_tokens=120,
        total_output_tokens=40,
    )

    assert action.type == ActionType.WAIT
    llm_event = next(event for event in logger.events if event["event_type"] == "llm_decision")
    assert llm_event["observation_size_chars"] > 0
    assert llm_event["input_tokens"] is not None
    assert llm_event["legal_actions_count"] >= 1
    assert llm_event["latency_ms"] is not None
    assert llm_event["tick"] >= 1
    assert llm_event["game_time"] >= 0.0
    assert llm_event["valid"] in {True, False}
    assert llm_event["repair_attempted"] in {True, False}
    assert llm_event["fallback_used"] in {True, False}

    episode_end = logger.events[-1]
    assert episode_end["event_type"] == "episode_end"
    assert episode_end["mean_latency_ms"] == 5.0
    assert episode_end["min_latency_ms"] == 3.0
    assert episode_end["max_latency_ms"] == 9.0
    assert episode_end["valid_action_rate"] == 1.0
    assert episode_end["repair_attempts"] == 1
    assert episode_end["fallbacks"] == 1
    assert "observation_size_chars" in llm_event
    assert "legal_actions_count" in llm_event
    assert logger.events[0]["event_type"] == "episode_start"
    assert logger.events[0]["episode_id"] == "ep_1"
    assert logger.events[0]["provider"] == "mock"
