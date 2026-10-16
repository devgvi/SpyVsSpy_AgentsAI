"""M5C.1: StrandsAgentAdapter.

Covers spec sections 26-30 (adapter contract, no-mutation, leakage, bounded
decision, async). Uses a fake Strands `Agent` boundary (a real
@strands.tool-decorated function - the actual SDK decorator, called
directly - wired to a fake agent object standing in for the network/model
call), so these tests exercise the genuine tool-capture path without any
Ollama/network access, per spec section 26.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.actions import ActionType
from src.agents.registry import create_agent_runtime
from src.agents.strands_agent import StrandsAgentAdapter, StrandsUnavailableError
from src.observation import Observation, ObservedObject, OpponentView


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


class _FakeAgent:
    """Stands in for strands.Agent: invokes the *real* @strands.tool
    object it's given (genuine schema/capture behavior) with a queued set
    of calls, then returns an AgentResult-shaped fake. No network."""

    def __init__(self, tool, calls, delay=0.0):
        self.tool = tool
        self.calls = calls
        self.delay = delay

    def __call__(self, prompt, limits=None):
        if self.delay:
            time.sleep(self.delay)
        for kwargs in self.calls:
            self.tool(**kwargs)
        tool_metrics = {"choose_action": SimpleNamespace(call_count=len(self.calls))} if self.calls else {}
        return SimpleNamespace(
            stop_reason="tool_use" if self.calls else "end_turn",
            metrics=SimpleNamespace(
                cycle_count=1,
                tool_metrics=tool_metrics,
                accumulated_usage={"inputTokens": 50, "outputTokens": 20},
            ),
        )


class QueuedStrandsFactory:
    """Test double for the adapter's agent_factory: each call (one per real
    Strands invocation - initial, then repair if needed) pops the next
    queued list of tool-call kwargs and returns a _FakeAgent that will make
    exactly those calls against the real tool object it's handed."""

    def __init__(self, queued_calls, delay=0.0):
        self._queue = [list(c) for c in queued_calls]
        self.delay = delay
        self.invocations = 0

    def __call__(self, system_prompt, tool):
        self.invocations += 1
        calls = self._queue.pop(0) if self._queue else []
        return _FakeAgent(tool, calls, delay=self.delay)


def _make_adapter(queued_calls, delay=0.0, **kwargs):
    factory = QueuedStrandsFactory(queued_calls, delay=delay)
    adapter = StrandsAgentAdapter("white", agent_factory=factory, **kwargs)
    return adapter, factory


# ---------------------------------------------------------------------------
# Adapter contract (spec section 26)
# ---------------------------------------------------------------------------


def test_registry_resolves_strands():
    runtime = create_agent_runtime(
        "strands", actor_id="white", agent_factory=QueuedStrandsFactory([[{"action": "wait"}]])
    )
    assert isinstance(runtime, StrandsAgentAdapter)
    assert runtime.runtime_name == "strands"


def test_strands_adapter_satisfies_agent_adapter_contract():
    from src.agents.base import AgentAdapter

    assert issubclass(StrandsAgentAdapter, AgentAdapter)
    adapter, _ = _make_adapter([[{"action": "wait"}]])
    decision = adapter.decide(_make_observation())
    assert decision.runtime == "strands"
    assert decision.decision_id
    assert decision.action_source in {"llm", "fallback"}


def test_runtime_metadata_correct():
    adapter, _ = _make_adapter([[{"action": "wait"}]])
    decision = adapter.decide(_make_observation())
    assert decision.runtime == "strands"
    assert adapter.runtime_version == "strands-adapter-v3"


def test_one_valid_tool_call_becomes_canonical_game_action():
    adapter, _ = _make_adapter([[{"action": "inspect", "target_id": "desk_01"}]])
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")
    decision = adapter.decide(observation)

    assert decision.action.type == ActionType.INSPECT
    assert decision.action.target_id == "desk_01"
    assert decision.action_source == "llm"


def test_target_id_preserved_end_to_end():
    adapter, _ = _make_adapter([[{"action": "interact", "target_id": "safe_01"}]])
    observation = _make_observation(objects=(_obj("safe_01"),), nearby_object_id="safe_01")
    action = adapter.get_action(observation)
    assert action.target_id == "safe_01"


def test_invalid_target_rejected():
    adapter, _ = _make_adapter(
        [[{"action": "inspect", "target_id": "hidden_object_999"}], [{"action": "wait"}]]
    )
    observation = _make_observation(objects=(_obj("desk_01"),))
    decision = adapter.decide(observation)
    # Repaired to a valid explicit wait via the second (bounded) invocation.
    assert decision.action.type == ActionType.WAIT
    assert decision.action_source == "llm"


def test_missing_target_rejected_and_falls_back_when_repair_also_fails():
    adapter, _ = _make_adapter([[{"action": "inspect"}], [{"action": "dance"}]])
    observation = _make_observation(objects=(_obj("desk_01"),))
    decision = adapter.decide(observation)
    assert decision.action.type == ActionType.WAIT
    assert decision.action_source == "fallback"


def test_unknown_action_rejected():
    adapter, _ = _make_adapter([[{"action": "dance"}], [{"action": "dance"}]])
    decision = adapter.decide(_make_observation())
    assert decision.action_source == "fallback"
    assert decision.action.type == ActionType.WAIT


def test_no_tool_call_is_rejected_not_a_silent_wait():
    # The agent completed without calling choose_action at all.
    adapter, _ = _make_adapter([[], []])
    decision = adapter.decide(_make_observation())
    assert decision.action_source == "fallback"
    assert decision.action.type == ActionType.WAIT


def test_fallback_correctly_attributed_in_telemetry():
    from src.llm.telemetry import TelemetryLogger

    logger = TelemetryLogger(path=None)
    adapter, _ = _make_adapter([[{"foo": "bar"}], [{"baz": "qux"}]], telemetry=logger)
    adapter.decide(_make_observation())

    event = next(e for e in logger.events if e["event_type"] == "llm_decision")
    assert event["agent_runtime"] == "strands"
    assert event["agent_sdk"] == "strands-agents"
    assert event["agent_sdk_version"]
    assert event["final_model_action_valid"] is False
    assert event["fallback_used"] is True
    assert event["action_source"] == "fallback"
    assert event["executed_action"] == {"action": "wait"}
    assert event["executed_action_valid"] is True


def test_dependency_failure_behavior_when_sdk_missing(monkeypatch):
    import src.agents.strands_agent as strands_agent_module

    def fail_import():
        raise strands_agent_module.StrandsUnavailableError("not installed")

    monkeypatch.setattr(strands_agent_module, "_import_strands", fail_import)
    with pytest.raises(StrandsUnavailableError):
        StrandsAgentAdapter("white")


# ---------------------------------------------------------------------------
# No direct mutation (spec section 27)
# ---------------------------------------------------------------------------


def test_tool_call_does_not_mutate_observation_or_engine_state():
    adapter, _ = _make_adapter([[{"action": "interact", "target_id": "desk_01"}]])
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")
    snapshot = observation  # frozen dataclass - identity implies no mutation possible

    adapter.decide(observation)

    # Observation is a frozen dataclass: if the tool had touched it, this
    # would raise FrozenInstanceError at call time, not merely fail an
    # equality check. Assert it's still exactly the same, untouched object.
    assert observation is snapshot
    assert observation.objects[0].searched is False
    assert observation.inventory == frozenset()


def test_calling_the_tool_directly_only_appends_to_capture():
    adapter, _ = _make_adapter([])
    capture: list = []
    tool = adapter._make_tool(capture)

    result = tool(action="interact", target_id="desk_01", trap_type=None, reason="test")

    assert capture == [{"action": "interact", "target_id": "desk_01", "trap_type": None, "reason": "test"}]
    assert isinstance(result, str)
    # No actor/inventory/door/trap/mission state exists anywhere in this
    # closure for the tool to have touched - capture is the only effect.


# ---------------------------------------------------------------------------
# Leakage (spec section 28) - repeat the M5C leakage checks through Strands
# ---------------------------------------------------------------------------


def test_no_leakage_of_opponent_position_through_strands():
    adapter, _ = _make_adapter([[{"action": "wait"}]])
    opponent = OpponentView(actor_id="black", position=(42.0, 17.0), facing="north", room="office", stunned=False)
    adapter.decide(_make_observation(visible_other_actor=True, opponent=opponent))

    compact = adapter.memory.to_compact_dict()
    assert set(compact["opponent_memory"].keys()) == {"currently_visible", "last_seen_room", "last_seen_time"}
    assert "42" not in json.dumps(compact["opponent_memory"])


def test_no_leakage_of_unknown_lock_requirement_through_strands():
    adapter, _ = _make_adapter([[{"action": "wait"}]])
    mystery_door = _obj("mystery_door", type="door", lock_type="keycard", locked=True, can_unlock=False)
    adapter.decide(_make_observation(objects=(mystery_door,)))

    lock = adapter.memory.locks.locks["mystery_door"]
    assert lock.requires == "unknown"


def test_no_leakage_of_undetected_trap_through_strands():
    adapter, _ = _make_adapter([[{"action": "wait"}]])
    trapped_but_unknown = _obj("cabinet_01", trappable=True, trap_status="unknown", trap_type=None)
    adapter.decide(_make_observation(objects=(trapped_but_unknown,)))

    assert "cabinet_01" not in json.dumps(adapter.memory.to_compact_dict())


# ---------------------------------------------------------------------------
# Bounded decision (spec section 29)
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_in_one_invocation_only_first_is_used():
    # The SDK's own loop could call the tool more than once in one
    # invocation (observed live with gemma4 at a loose turn limit). Only
    # the first captured proposal is ever executed; extras are recorded,
    # not applied.
    adapter, factory = _make_adapter(
        [[{"action": "inspect", "target_id": "desk_01"}, {"action": "interact", "target_id": "desk_01"}]]
    )
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")

    decision = adapter.decide(observation)

    assert decision.action.type == ActionType.INSPECT  # first proposal, not the second


def test_multiple_tool_calls_recorded_in_telemetry():
    from src.llm.telemetry import TelemetryLogger

    logger = TelemetryLogger(path=None)
    adapter, _ = _make_adapter(
        [[{"action": "inspect", "target_id": "desk_01"}, {"action": "interact", "target_id": "desk_01"}]],
        telemetry=logger,
    )
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")
    adapter.decide(observation)

    event = next(e for e in logger.events if e["event_type"] == "llm_decision")
    assert event["runtime_metadata"]["initial_proposal_count"] == 2


def test_decide_never_applies_more_than_one_world_action():
    # Structural guarantee: decide() returns exactly one AgentDecision with
    # exactly one GameAction, regardless of how many proposals the runtime
    # made internally.
    adapter, _ = _make_adapter(
        [[{"action": "wait"}, {"action": "inspect", "target_id": "desk_01"}, {"action": "interact", "target_id": "desk_01"}]]
    )
    decision = adapter.decide(_make_observation(objects=(_obj("desk_01"),)))
    assert decision.action.type == ActionType.WAIT  # first proposal only


# ---------------------------------------------------------------------------
# Async (spec section 30)
# ---------------------------------------------------------------------------


def test_evaluation_mode_stays_non_blocking_while_strands_thinks():
    adapter, _ = _make_adapter([[{"action": "inspect", "target_id": "desk_01"}]], delay=0.25)
    observation = _make_observation(objects=(_obj("desk_01"),))

    started = time.monotonic()
    action = adapter.get_action(observation)
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert action.type == ActionType.WAIT  # no provisional strategic action
    assert adapter.status in {"THINKING", "EXECUTING", "IDLE"}


def test_eventual_valid_result_reaches_the_caller():
    adapter, _ = _make_adapter([[{"action": "inspect", "target_id": "desk_01"}]], delay=0.15)
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id="desk_01")

    adapter.get_action(observation)  # kicks off the background thread
    if adapter._thinking_thread:
        adapter._thinking_thread.join(timeout=2.0)

    assert adapter.last_action.type == ActionType.INSPECT
    assert adapter.last_action.target_id == "desk_01"


# ---------------------------------------------------------------------------
# M5E.4: Bedrock model backend selection
# ---------------------------------------------------------------------------


def test_build_default_agent_uses_bedrock_model_for_bedrock_provider():
    """`provider_name="bedrock"` must construct Strands' own native
    `BedrockModel` (region_name/model_id/temperature), never OllamaModel -
    no network call happens here since only construction is exercised."""
    adapter = StrandsAgentAdapter(
        "white", provider_name="bedrock", model_name="us.anthropic.claude-sonnet-4-6", llm_region="us-east-1",
    )
    with patch("strands.models.bedrock.BedrockModel") as mock_bedrock_model, patch("strands.models.ollama.OllamaModel") as mock_ollama_model:
        adapter._build_default_agent("system prompt", lambda: None)
    mock_bedrock_model.assert_called_once_with(region_name="us-east-1", model_id="us.anthropic.claude-sonnet-4-6", temperature=0.0)
    mock_ollama_model.assert_not_called()


def test_build_default_agent_still_uses_ollama_model_by_default():
    """Regression: every pre-M5E.4 (Ollama) construction path is
    byte-identical - only an explicit `provider_name="bedrock"` takes the
    new branch."""
    adapter = StrandsAgentAdapter("white", provider_name="ollama", model_name="qwen:latest")
    with patch("strands.models.bedrock.BedrockModel") as mock_bedrock_model, patch("strands.models.ollama.OllamaModel") as mock_ollama_model:
        adapter._build_default_agent("system prompt", lambda: None)
    mock_ollama_model.assert_called_once()
    mock_bedrock_model.assert_not_called()


def test_game_wires_llm_region_through_to_strands_adapter_for_bedrock():
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from src.game import Game

    game = Game(
        white_controller="llm", agent_runtime="strands", llm_provider="bedrock",
        llm_model="us.anthropic.claude-sonnet-4-6", llm_region="us-east-1",
    )
    assert game.bot_controller._llm_region == "us-east-1"
    assert game.bot_controller.provider_name == "bedrock"


# ---------------------------------------------------------------------------
# M6A.2: provider/validation/repair failure-count split (mirrors
# LLMController's own M5F.3B.1 split - spec sections 12-13 there). Real bug
# found and fixed live: StrandsAgentAdapter never had this split at all
# (only the combined `fallback_count`), so `EpisodeRunner`'s stop-rule
# (reading the split counter when available) silently fell back to the
# conflated total for every Strands episode - a live M6A.2 pilot hit 21/39
# fallbacks, almost all a harmless `invalid_trap_type` repair case, and the
# episode aborted at `max_runtime_failures=20` before finishing.
# ---------------------------------------------------------------------------


def test_invalid_trap_type_increments_validation_not_provider_failure_count():
    from src.evaluation.runtime_snapshot_replay import build_snapshot_set
    from src.llm_controller import serialize_observation

    snapshot = next(s for s in build_snapshot_set() if s.category == "TRAP_PLACEMENT_OPPORTUNITY")
    payload = serialize_observation(snapshot.observation)
    bad_call = [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": "not_a_real_trap_type", "reason": "x"}]
    adapter, _factory = _make_adapter([bad_call, bad_call])  # initial, then repair - both still invalid
    adapter._request_action(snapshot.observation, payload)
    assert adapter.provider_failure_count == 0
    assert adapter.validation_failure_count == 1
    assert adapter.repair_failure_count == 1
    assert adapter.fallback_count == 1  # historical combined total unchanged


def test_strands_runtime_exception_increments_provider_not_validation_failure_count():
    from src.evaluation.runtime_snapshot_replay import build_snapshot_set
    from src.llm_controller import serialize_observation

    class _RaisingAgent:
        def __call__(self, prompt_json, limits=None):
            raise RuntimeError("simulated Strands SDK outage")

    class _RaisingFactory:
        def __call__(self, system_prompt, tool):
            return _RaisingAgent()

    snapshot = next(s for s in build_snapshot_set() if s.category == "TRAP_PLACEMENT_OPPORTUNITY")
    payload = serialize_observation(snapshot.observation)
    adapter = StrandsAgentAdapter("white", agent_factory=_RaisingFactory())
    adapter._request_action(snapshot.observation, payload)
    assert adapter.provider_failure_count == 1
    assert adapter.validation_failure_count == 0
    assert adapter.fallback_count == 1


def test_valid_strands_action_leaves_all_failure_counters_at_zero():
    from src.evaluation.runtime_snapshot_replay import build_snapshot_set
    from src.llm_controller import serialize_observation

    snapshot = next(s for s in build_snapshot_set() if s.category == "TRAP_PLACEMENT_OPPORTUNITY")
    payload = serialize_observation(snapshot.observation)
    held_type = next(t for t, c in snapshot.observation.trap_inventory.items() if c > 0)
    good_call = [{"action": "place_trap", "target_id": "ambassador_safe", "trap_type": held_type, "reason": "fine"}]
    adapter, _factory = _make_adapter([good_call])
    adapter._request_action(snapshot.observation, payload)
    assert adapter.provider_failure_count == 0
    assert adapter.validation_failure_count == 0
    assert adapter.fallback_count == 0
