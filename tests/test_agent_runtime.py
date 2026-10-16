"""M5C: Agent runtime abstraction & NativeAgentAdapter.

Covers spec sections 20-23: native/adapter equivalence, leakage regression
through the adapter, and the async (non-blocking) contract. All Mock-based
- no network access required.
"""

import time

import pygame
import pytest

from src.actions import ActionType
from src.agents.base import AgentDecision
from src.agents.native_agent import NativeAgentAdapter
from src.agents.registry import create_agent_runtime
from src.game import Game
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController
from src.observation import Observation, ObservedObject, OpponentView


class SlowModelClient:
    def __init__(self, response, delay=0.25):
        self.response = response
        self.delay = delay
        self.calls = 0

    def generate_action(self, *args, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        return self.response


def _obj(id, **overrides):
    base = dict(
        id=id,
        type="desk",
        label=id,
        position=(10.0, 10.0),
        collision_size=(20.0, 20.0),
        interactable=True,
        trappable=False,
        searched=False,
        inspected=False,
        trap_status="unknown",
        trap_type=None,
        trap_is_mine=False,
        door_open=None,
        lock_type=None,
        locked=False,
        can_unlock=False,
    )
    base.update(overrides)
    return ObservedObject(**base)


def _make_observation(**overrides):
    base = dict(
        actor_id="white",
        seed=1,
        position=(5.0, 5.0),
        facing="east",
        stunned=False,
        stunned_seconds=0.0,
        busy=False,
        busy_seconds=0.0,
        mission_time_remaining=100.0,
        mission_complete=False,
        inventory=frozenset(),
        knowledge=frozenset(),
        required_items=("secret_documents",),
        trap_inventory={},
        nearby_object_id=None,
        items_found_here=frozenset(),
        objects=(),
        nav_grid=None,
        log_event=lambda *args, **kwargs: None,
        current_room="office",
        known_rooms=frozenset({"office"}),
        known_doors=frozenset(),
        known_locked_objects={},
        known_item_locations={},
        visible_other_actor=False,
        opponent=None,
        last_seen_opponent=None,
        revealed_opponent=None,
        revealed_opponent_seconds=0.0,
    )
    base.update(overrides)
    return Observation(**base)


# ---------------------------------------------------------------------------
# Factory / registry (spec section 21: "factory resolves native", "unknown
# runtime fails clearly")
# ---------------------------------------------------------------------------


def test_factory_resolves_native():
    runtime = create_agent_runtime("native", actor_id="white", model_client=MockModelClient({"action": "wait"}))
    assert isinstance(runtime, NativeAgentAdapter)
    assert runtime.runtime_name == "native"
    assert runtime.runtime_version == "native-v1"


def test_factory_defaults_to_native_when_unspecified():
    runtime = create_agent_runtime(None, actor_id="white", model_client=MockModelClient({"action": "wait"}))
    assert isinstance(runtime, NativeAgentAdapter)


def test_factory_unknown_runtime_fails_clearly():
    # "strands" is a real registered runtime as of M5C.1 - use a genuinely
    # unregistered name (LangGraph is explicitly out of scope).
    with pytest.raises(ValueError, match="Unknown agent_runtime"):
        create_agent_runtime("langgraph", actor_id="white", model_client=MockModelClient({"action": "wait"}))


# ---------------------------------------------------------------------------
# NativeAgentAdapter basic behavior (spec section 21)
# ---------------------------------------------------------------------------


def test_native_adapter_returns_valid_game_action():
    adapter = NativeAgentAdapter("white", MockModelClient({"action": "inspect", "target_id": "desk_01"}))
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")

    decision = adapter.decide(observation)

    assert isinstance(decision, AgentDecision)
    assert decision.runtime == "native"
    assert decision.action.type == ActionType.INSPECT
    assert decision.action.target_id == "desk_01"
    assert decision.action_source == "llm"
    assert decision.metadata["provider"] == "mock"


def test_native_adapter_get_action_is_drop_in_compatible():
    adapter = NativeAgentAdapter("white", MockModelClient({"action": "inspect", "target_id": "desk_01"}))
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")

    action = adapter.get_action(observation)
    assert action.type == ActionType.INSPECT
    assert action.target_id == "desk_01"


def test_native_adapter_retains_repair_fallback_behavior():
    # M5B.1: repair is a real second model call, so both the initial and
    # the repair response must be genuinely invalid to reach a real
    # fallback (a single-response mock would otherwise "succeed" via
    # MockModelClient's own exhausted-queue default of a valid wait).
    adapter = NativeAgentAdapter("white", MockModelClient([{"action": "interact"}, {"foo": "bar"}]))
    decision = adapter.decide(_make_observation())

    assert decision.action.type == ActionType.WAIT
    assert decision.action_source == "fallback"


def test_target_id_survives_end_to_end_through_adapter():
    adapter = NativeAgentAdapter("white", MockModelClient({"action": "interact", "target_id": "safe_01"}))
    observation = _make_observation(objects=(_obj("safe_01"),), nearby_object_id="safe_01")

    action = adapter.get_action(observation)
    assert action.target_id == "safe_01"


def test_unrecognized_runtime_attribute_raises_attribute_error():
    adapter = NativeAgentAdapter("white", MockModelClient({"action": "wait"}))
    with pytest.raises(AttributeError):
        adapter.definitely_not_a_real_attribute


# ---------------------------------------------------------------------------
# Native baseline regression / equivalence (spec section 20)
# ---------------------------------------------------------------------------


def test_native_adapter_equivalent_to_direct_llm_controller():
    response = {"action": "interact", "target_id": "desk_01", "reason": "check the desk"}
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")

    direct = LLMController("white", MockModelClient(dict(response)))
    direct_action = direct.get_action(observation)

    via_adapter = NativeAgentAdapter("white", MockModelClient(dict(response)))
    adapter_decision = via_adapter.decide(observation)

    assert adapter_decision.action.type == direct_action.type
    assert adapter_decision.action.target_id == direct_action.target_id
    assert adapter_decision.action_source == direct._last_action_source

    direct_decision = direct.memory.recent_decisions[-1]
    adapter_inner_decision = via_adapter.controller.memory.recent_decisions[-1]
    assert direct_decision.valid == adapter_inner_decision.valid
    assert direct_decision.action_source == adapter_inner_decision.action_source
    assert direct_decision.selected_action == adapter_inner_decision.selected_action


def test_native_adapter_invalid_output_matches_direct_controller_after_repair():
    # M5B.1: "fly" is unrecognized -> initial invalid -> real second model
    # call for repair -> MockModelClient's queue is now exhausted, so its
    # own default ({"action": "wait"}) is returned, which is a genuinely
    # valid repair. Both paths must land on the same outcome.
    response = {"action": "fly", "target_id": "desk_01"}
    observation = _make_observation(objects=(_obj("desk_01"),))

    direct = LLMController("white", MockModelClient(dict(response)))
    direct_action = direct.get_action(observation)

    via_adapter = NativeAgentAdapter("white", MockModelClient(dict(response)))
    adapter_decision = via_adapter.decide(observation)

    assert adapter_decision.action.type == direct_action.type == ActionType.WAIT
    assert adapter_decision.action_source == direct._last_action_source == "llm"


def test_native_adapter_matches_direct_controller_true_fallback():
    # Both the initial and the repair response are genuinely invalid here,
    # so this must land on a real fallback in both paths - not a repaired
    # "llm" wait.
    responses = [{"action": "fly", "target_id": "desk_01"}, {"foo": "bar"}]
    observation = _make_observation(objects=(_obj("desk_01"),))

    direct = LLMController("white", MockModelClient(list(responses)))
    direct_action = direct.get_action(observation)

    via_adapter = NativeAgentAdapter("white", MockModelClient(list(responses)))
    adapter_decision = via_adapter.decide(observation)

    assert adapter_decision.action.type == direct_action.type == ActionType.WAIT
    assert adapter_decision.action_source == direct._last_action_source == "fallback"


# ---------------------------------------------------------------------------
# Telemetry runtime dimension (spec section 17, 21: "runtime metadata
# appears in telemetry")
# ---------------------------------------------------------------------------


def test_telemetry_includes_agent_runtime_metadata():
    logger = TelemetryLogger(path=None)
    adapter = NativeAgentAdapter("white", MockModelClient({"action": "wait"}), telemetry=logger)
    adapter.decide(_make_observation())

    event = next(e for e in logger.events if e["event_type"] == "llm_decision")
    assert event["agent_runtime"] == "native"
    assert event["agent_runtime_version"] == "native-v1"
    assert event["provider"] == "mock"
    assert event["model"] == "mock-model"


# ---------------------------------------------------------------------------
# Leakage regression through the adapter (spec section 22)
# ---------------------------------------------------------------------------


def test_no_leakage_of_opponent_position_through_adapter():
    adapter = NativeAgentAdapter("white", MockModelClient({"action": "wait"}))
    opponent = OpponentView(actor_id="black", position=(42.0, 17.0), facing="north", room="office", stunned=False)
    adapter.decide(_make_observation(visible_other_actor=True, opponent=opponent))

    compact = adapter.controller.memory.to_compact_dict()
    assert set(compact["opponent_memory"].keys()) == {"currently_visible", "last_seen_room", "last_seen_time"}


def test_no_leakage_of_unknown_lock_requirement_through_adapter():
    adapter = NativeAgentAdapter("white", MockModelClient({"action": "wait"}))
    mystery_door = _obj("mystery_door", type="door", lock_type="keycard", locked=True, can_unlock=False)
    adapter.decide(_make_observation(objects=(mystery_door,)))

    lock = adapter.controller.memory.locks.locks["mystery_door"]
    assert lock.requires == "unknown"


def test_no_leakage_of_undetected_trap_through_adapter():
    adapter = NativeAgentAdapter("white", MockModelClient({"action": "wait"}))
    trapped_but_unknown = _obj("cabinet_01", trappable=True, trap_status="unknown", trap_type=None)
    adapter.decide(_make_observation(objects=(trapped_but_unknown,)))

    import json
    assert "cabinet_01" not in json.dumps(adapter.controller.memory.to_compact_dict())


# ---------------------------------------------------------------------------
# Async / non-blocking regression (spec section 23)
# ---------------------------------------------------------------------------


def test_evaluation_mode_stays_non_blocking_and_provisional_free_through_adapter():
    slow_client = SlowModelClient({"action": "inspect", "target_id": "desk_01"})
    adapter = NativeAgentAdapter("white", slow_client)  # default execution_mode = evaluation
    observation = _make_observation(objects=(_obj("desk_01"),))

    started = time.monotonic()
    decision = adapter.decide(observation)
    elapsed = time.monotonic() - started

    # The request is still in flight (0.25s delay); evaluation mode must
    # return immediately with WAIT, never a provisional strategic action.
    assert elapsed < 0.25
    assert decision.action.type == ActionType.WAIT
    assert adapter.controller.status in {"THINKING", "EXECUTING", "IDLE"}
    assert adapter.status in {"THINKING", "EXECUTING", "IDLE"}  # delegated via __getattr__


# ---------------------------------------------------------------------------
# Game wiring (spec section 7: default runtime, unknown runtime fails clearly)
# ---------------------------------------------------------------------------


def test_game_defaults_to_native_agent_runtime(monkeypatch):
    monkeypatch.setattr(pygame, "init", lambda: None)
    monkeypatch.setattr(pygame.display, "set_caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(pygame.display, "set_mode", lambda *args, **kwargs: pygame.Surface((800, 600)))

    game = Game(seed=123, white_controller="llm", llm_provider="mock", llm_model="mock-model")
    assert isinstance(game.bot_controller, NativeAgentAdapter)
    assert game.bot_controller.runtime_name == "native"
    # Duck-typed HUD/game.py reads still resolve through delegation.
    assert game.bot_controller.execution_mode == "evaluation"


def test_game_unknown_agent_runtime_fails_clearly(monkeypatch):
    monkeypatch.setattr(pygame, "init", lambda: None)
    monkeypatch.setattr(pygame.display, "set_caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(pygame.display, "set_mode", lambda *args, **kwargs: pygame.Surface((800, 600)))

    with pytest.raises(ValueError, match="Unknown agent_runtime"):
        Game(seed=123, white_controller="llm", llm_provider="mock", llm_model="mock-model", agent_runtime="langgraph")
