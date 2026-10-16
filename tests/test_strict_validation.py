"""M5B.1: Strict Model Output Validation.

Covers spec section 16 (tests A-H) plus the regressions listed in section
17. The core invariant under test: a malformed/missing/unrecognized model
response must never become an equivalent, silently-fabricated
{"action": "wait"} tagged action_source="llm" - only an explicit model
WAIT counts as a genuine LLM decision.
"""

import json

from src.actions import ActionType
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController
from src.observation import Observation, ObservedObject


class RawTextModelClient:
    """Simulates a provider handing back unparseable prose instead of JSON -
    exercises _coerce_model_response's str-parse-failure path directly,
    the same path a real Ollama/OpenAI-compatible client goes through."""

    provider = "mock"
    model = "mock-model"

    def __init__(self, text: str):
        self.text = text

    def generate_action(self, *args, **kwargs):
        return self.text


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


def _decide(client, observation=None, telemetry=None):
    logger = telemetry or TelemetryLogger(path=None)
    controller = LLMController("white", client, telemetry=logger)
    action = controller.get_action(observation or _make_observation())
    event = next(e for e in logger.events if e["event_type"] == "llm_decision")
    return controller, action, event


# ---------------------------------------------------------------------------
# A. Explicit valid WAIT
# ---------------------------------------------------------------------------


def test_a_explicit_valid_wait():
    controller, action, event = _decide(MockModelClient({"action": "wait"}))

    assert action.type == ActionType.WAIT
    assert event["initial_model_output_valid"] is True
    assert event["final_model_action_valid"] is True
    assert event["fallback_used"] is False
    assert event["action_source"] == "llm"
    assert event["executed_action"]["action"] == "wait"
    assert event["executed_action_valid"] is True
    assert controller.memory.recent_decisions[-1].action_source == "llm"


# ---------------------------------------------------------------------------
# B. Missing action
# ---------------------------------------------------------------------------


def test_b_missing_action_is_never_silent_wait():
    controller, action, event = _decide(MockModelClient([{"target_id": "desk_01"}]))

    assert event["initial_model_output_valid"] is False
    assert event["initial_error_type"] == "missing_action"
    assert event["repair_attempted"] is True
    # Whatever the eventual outcome, the INITIAL malformed response itself
    # must never be recorded as a valid explicit WAIT.
    assert not (event["initial_model_output_valid"] and event["raw_response"].get("action") is None)


# ---------------------------------------------------------------------------
# C. Garbage text (non-JSON)
# ---------------------------------------------------------------------------


def test_c_garbage_text_requires_structured_json():
    client = RawTextModelClient("Maybe I should inspect the desk because it seems useful.")
    controller, action, event = _decide(client)

    assert event["initial_model_output_valid"] is False
    assert event["repair_attempted"] is True
    # The garbage prose must never surface as a fabricated valid wait.
    assert event["raw_response"] == {}


# ---------------------------------------------------------------------------
# D. Unknown action
# ---------------------------------------------------------------------------


def test_d_unknown_action_name():
    controller, action, event = _decide(MockModelClient([{"action": "dance"}]))

    assert event["initial_model_output_valid"] is False
    assert event["initial_error_type"] == "unknown_action"


# ---------------------------------------------------------------------------
# E. Missing target
# ---------------------------------------------------------------------------


def test_e_missing_target():
    observation = _make_observation(objects=(_obj("desk_01"),))
    controller, action, event = _decide(MockModelClient([{"action": "inspect"}]), observation)

    assert event["initial_model_output_valid"] is False
    assert event["initial_error_type"] == "missing_target"


# ---------------------------------------------------------------------------
# F. Illegal target (and no leakage)
# ---------------------------------------------------------------------------


def test_f_unknown_target_and_no_leakage():
    observation = _make_observation(objects=(_obj("desk_01"),))
    controller, action, event = _decide(
        MockModelClient([{"action": "inspect", "target_id": "hidden_object_999"}]), observation
    )

    assert event["initial_model_output_valid"] is False
    # M5D.4 spec section 12: a target that doesn't exist in this actor's
    # own Observation at all is a distinct category from a real, known
    # target that's simply not currently legal (see the sibling test
    # below) - collapsing them loses exactly the distinction the failure
    # taxonomy needs.
    assert event["initial_error_type"] == "unknown_target"
    # No hidden information leaked about the (nonexistent) target: memory
    # never claims anything was learned about it.
    assert "hidden_object_999" not in json.dumps(controller.memory.to_compact_dict())


def test_f2_out_of_range_is_distinct_from_unknown_target():
    # desk_01 is real and known, but not the actor's current
    # nearby_object_id - a legitimate, real object that's simply not
    # currently in range (M5D.4's proximity-gated legal_actions), now its
    # own "out_of_range" category (M5D.5 spec section 5), split from the
    # more generic "illegal_target" (e.g. a locked object).
    observation = _make_observation(objects=(_obj("desk_01"),), nearby_object_id=None)
    controller, action, event = _decide(MockModelClient([{"action": "inspect", "target_id": "desk_01"}]), observation)

    assert event["initial_model_output_valid"] is False
    assert event["initial_error_type"] == "out_of_range"


# ---------------------------------------------------------------------------
# G. Successful repair
# ---------------------------------------------------------------------------


def test_g_successful_repair():
    controller, action, event = _decide(MockModelClient([{"foo": "bar"}, {"action": "wait"}]))

    assert event["initial_model_output_valid"] is False
    assert event["repair_attempted"] is True
    assert event["repair_valid"] is True
    assert event["final_model_action_valid"] is True
    assert event["fallback_used"] is False
    assert event["action_source"] == "llm"
    assert action.type == ActionType.WAIT
    assert event["repair_raw_response"] == {"action": "wait"}


# ---------------------------------------------------------------------------
# H. Failed repair -> fallback
# ---------------------------------------------------------------------------


def test_h_failed_repair_falls_back():
    controller, action, event = _decide(MockModelClient([{"foo": "bar"}, {"baz": "qux"}]))

    assert event["final_model_action_valid"] is False
    assert event["fallback_used"] is True
    assert event["action_source"] == "fallback"
    assert action.type == ActionType.WAIT
    assert event["executed_action"] == {"action": "wait"}
    assert event["executed_action_valid"] is True
    # The decision record must not claim "LLM chose wait" for a fallback.
    decision = controller.memory.recent_decisions[-1]
    assert decision.action_source == "fallback"
    assert decision.valid is False


# ---------------------------------------------------------------------------
# Regressions (spec section 17)
# ---------------------------------------------------------------------------


def test_valid_canonical_response_and_target_id_preservation_unaffected():
    observation = _make_observation(objects=(_obj("safe_01"),), nearby_object_id="safe_01")
    controller, action, event = _decide(
        MockModelClient([{"action": "interact", "target_id": "safe_01"}]), observation
    )

    assert action.target_id == "safe_01"
    assert event["initial_model_output_valid"] is True
    assert event["action_source"] == "llm"


def test_decision_id_correlates_with_memory_after_strict_validation():
    controller, action, event = _decide(MockModelClient([{"action": "wait"}]))
    memory_decision = controller.memory.recent_decisions[-1]
    assert event["decision_id"] == memory_decision.decision_id


def test_repair_does_not_leak_hidden_information():
    # The repair prompt is built from the same model_payload as the
    # original request (observation + legal_actions + memory) plus a
    # generic instruction string - never anything else.
    observation = _make_observation(objects=(_obj("desk_01"),))
    controller, action, event = _decide(MockModelClient([{"action": "dance"}, {"action": "wait"}]), observation)
    assert event["repair_valid"] is True
    # No unexpected structural field beyond the documented telemetry keys crept in.
    assert set(event["raw_response"].keys()) <= {"action"}
