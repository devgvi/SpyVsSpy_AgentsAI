"""M5F.3B.1 Phase A (spec section 20): invalid trap-type handling is a
validation/repair problem, never misclassified as a provider/runtime
failure. Real `GameState`/`Observation`/`LLMController` throughout - no
network calls."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from typing import Any

from src import constants
from src.items import load_items
from src.level import load_level
from src.llm.clients.mock_client import MockModelClient
from src.llm_controller import LLMController, LLMControllerConfig, enumerate_legal_actions, serialize_observation, validate_model_output
from src.observation import build_observation
from src.state import GameState


def _new_state() -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=1, communication_enabled=False)


def _observation_ready_to_place_trap(state: GameState, actor_id: str = constants.BLACK_SPY_ID):
    state.current_interaction_target[actor_id] = "ambassador_safe"
    state._object_entry("ambassador_safe")["inspected_by"].add(actor_id)
    return build_observation(state, actor_id)


class _SequencedClient(MockModelClient):
    """Records every user_prompt it was called with, in order."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(response=responses)
        self.prompts: list[str] = []

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        self.prompts.append(user_prompt)
        return super().generate_action(system_prompt, user_prompt, output_schema, request_context)


class _RaisingClient:
    provider = "mock"
    model = "mock-model"

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        raise RuntimeError("simulated provider outage")


# -- unsupported vs not-in-inventory are now two distinct error_types -----


def test_unsupported_trap_type_rejected_as_invalid_trap_type():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    legal = enumerate_legal_actions(obs)
    assert obs.trap_inventory.get("stun", 0) == 0  # "stun" is not a real trap type in this game
    response = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun"}
    result = validate_model_output(obs, response, legal)
    assert result.valid is False
    assert result.error_type == "invalid_trap_type"


def test_trap_not_in_inventory_rejected_as_distinct_error_type():
    state = _new_state()
    state.actors[constants.BLACK_SPY_ID].trap_inventory["alarm"] = 0
    obs = _observation_ready_to_place_trap(state)
    legal = enumerate_legal_actions(obs)
    assert "alarm" in constants.TRAP_TYPES  # a real Engine trap type...
    assert obs.trap_inventory.get("alarm", 0) == 0  # ...just not currently held
    response = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "alarm"}
    result = validate_model_output(obs, response, legal)
    assert result.valid is False
    assert result.error_type == "trap_type_not_in_inventory"
    assert result.error_type != "invalid_trap_type"


def test_valid_held_trap_type_accepted():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    legal = enumerate_legal_actions(obs)
    held_type = next(t for t, c in obs.trap_inventory.items() if c > 0)
    response = {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held_type}
    result = validate_model_output(obs, response, legal)
    assert result.valid is True


# -- repair receives factual legal values, no auto-substitution -----------


def test_repair_prompt_includes_legal_trap_values_even_when_initial_error_is_missing_target():
    # Real bug found live in M5F.3B.1 Stage 2: the model's INITIAL
    # place_trap proposal often omits trap_type entirely (error_type
    # "missing_target", not yet "invalid_trap_type"), and only the REPAIR
    # attempt adds one - landing on a hallucinated "stun" every time
    # because the legal-values hint used to only fire when the INITIAL
    # error was already trap-type-specific.
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    payload = serialize_observation(obs)
    responses = [
        {"action": "place_trap", "target_id": "ambassador_safe", "reason": "x"},  # trap_type omitted -> missing_target
        {"action": "wait", "reason": "repaired"},
    ]
    client = _SequencedClient(responses)
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(), model_client=client)
    controller._request_action(obs, payload)
    repair_prompt = client.prompts[1]
    for trap_type, count in obs.trap_inventory.items():
        if count > 0:
            assert trap_type in repair_prompt


def test_repair_prompt_includes_actual_legal_trap_values():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    payload = serialize_observation(obs)
    responses = [
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"},
        {"action": "wait", "reason": "repaired"},
    ]
    client = _SequencedClient(responses)
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(), model_client=client)
    controller._request_action(obs, payload)
    assert len(client.prompts) == 2
    repair_prompt = client.prompts[1]
    for trap_type, count in obs.trap_inventory.items():
        if count > 0:
            assert trap_type in repair_prompt
    assert "stun" not in repair_prompt or "was rejected" in repair_prompt  # never silently substituted


def test_repair_does_not_auto_substitute_or_prefer_a_trap_type():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    payload = serialize_observation(obs)
    responses = [
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"},
        {"action": "wait", "reason": "y"},
    ]
    client = MockModelClient(response=responses)
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(), model_client=client)
    action = controller._request_action(obs, payload)
    # The repair response chose WAIT explicitly - the controller must not
    # have silently converted the invalid trap placement into a
    # place_trap with some "preferred" type instead.
    from src.actions import ActionType

    assert action.type == ActionType.WAIT


def test_repair_valid_action_is_accepted():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    payload = serialize_observation(obs)
    held_type = next(t for t, c in obs.trap_inventory.items() if c > 0)
    responses = [
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"},
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held_type, "reason": "corrected"},
    ]
    client = MockModelClient(response=responses)
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(), model_client=client)
    action = controller._request_action(obs, payload)
    from src.actions import ActionType

    assert action.type == ActionType.PLACE_TRAP
    assert action.trap_type == held_type


def test_repair_invalid_falls_back_to_wait():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    payload = serialize_observation(obs)
    responses = [
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"},
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "still wrong"},
    ]
    client = MockModelClient(response=responses)
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(), model_client=client)
    action = controller._request_action(obs, payload)
    from src.actions import ActionType

    assert action.type == ActionType.WAIT


# -- failure counter separation (spec sections 11-12) ----------------------


def test_invalid_trap_type_increments_validation_not_provider_failure_count():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    payload = serialize_observation(obs)
    responses = [
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "x"},
        {"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "stun", "reason": "still wrong"},
    ]
    client = MockModelClient(response=responses)
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(), model_client=client)
    controller._request_action(obs, payload)
    assert controller.provider_failure_count == 0
    assert controller.validation_failure_count == 1
    assert controller.repair_failure_count == 1
    assert controller.fallback_count == 1  # historical combined total unchanged


def test_provider_exception_increments_provider_not_validation_failure_count():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    payload = serialize_observation(obs)
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(), model_client=_RaisingClient())
    controller._request_action(obs, payload)
    assert controller.provider_failure_count == 1
    assert controller.validation_failure_count == 0
    assert controller.fallback_count == 1


def test_valid_action_leaves_all_failure_counters_at_zero():
    state = _new_state()
    obs = _observation_ready_to_place_trap(state)
    payload = serialize_observation(obs)
    held_type = next(t for t, c in obs.trap_inventory.items() if c > 0)
    client = MockModelClient(response={"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held_type, "reason": "fine"})
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(), model_client=client)
    controller._request_action(obs, payload)
    assert controller.provider_failure_count == 0
    assert controller.validation_failure_count == 0
    assert controller.fallback_count == 0


# -- SOCIAL_OFF/SOCIAL_ON schema regressions (spec sections 50-51) --------


def test_social_off_schema_still_omits_message_fields():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    payload = serialize_observation(obs)

    class _Capturing(MockModelClient):
        def __init__(self):
            super().__init__(response={"action": "wait", "reason": "x"})
            self.schema = None

        def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
            self.schema = output_schema
            return super().generate_action(system_prompt, user_prompt, output_schema, request_context)

    client = _Capturing()
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(communication_enabled=False), model_client=client)
    controller._request_action(obs, payload)
    assert "message" not in client.schema["properties"]
    assert "message_to" not in client.schema["properties"]
    assert "message_text" not in client.schema["properties"]


def test_social_on_schema_exposes_bounded_message_fields():
    state = _new_state()
    state.communication_enabled = True
    obs = build_observation(state, constants.BLACK_SPY_ID)
    payload = serialize_observation(obs)

    class _Capturing(MockModelClient):
        def __init__(self):
            super().__init__(response={"action": "wait", "reason": "x"})
            self.schema = None

        def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
            self.schema = output_schema
            return super().generate_action(system_prompt, user_prompt, output_schema, request_context)

    client = _Capturing()
    controller = LLMController(
        constants.BLACK_SPY_ID,
        config=LLMControllerConfig(communication_enabled=True, zero_decision_cost_messaging=True, flat_message_schema=True, message_cooldown_decisions=3),
        model_client=client,
    )
    controller._request_action(obs, payload)
    assert "message_to" in client.schema["properties"]
    assert "message_text" in client.schema["properties"]
