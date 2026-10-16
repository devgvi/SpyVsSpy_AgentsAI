"""M5F.2 Phase A (spec section 42): SEND_MESSAGE schema, validation,
engine-level delivery, decision-cost semantics, event attribution, single
delivery, claim-vs-observed-fact separation, and message ordering. Real
`GameState`/`Observation`/`LLMController` throughout - no network calls."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src import constants
from src.actions import ActionType, GameAction
from src.items import load_items
from src.level import load_level
from src.llm_controller import (
    convert_response_to_action,
    enumerate_legal_actions,
    serialize_observation,
    validate_model_output,
)
from src.observation import build_observation
from src.state import GameState


def _new_state(communication_enabled: bool = True) -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=1, communication_enabled=communication_enabled)


# -- SEND_MESSAGE schema / legal-action surface --------------------------------


def test_send_message_offered_only_when_communication_enabled():
    state_on = _new_state(communication_enabled=True)
    state_off = _new_state(communication_enabled=False)
    obs_on = build_observation(state_on, constants.BLACK_SPY_ID)
    obs_off = build_observation(state_off, constants.BLACK_SPY_ID)

    legal_on = enumerate_legal_actions(obs_on)
    legal_off = enumerate_legal_actions(obs_off)

    assert any(a["action"] == "send_message" for a in legal_on)
    assert not any(a["action"] == "send_message" for a in legal_off)


def test_send_message_legal_target_is_the_one_opponent():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    send_message_entries = [a for a in legal if a["action"] == "send_message"]
    assert len(send_message_entries) == 1
    assert send_message_entries[0]["target_id"] == constants.WHITE_SPY_ID


def test_well_formed_send_message_validates():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    response = {"action": "send_message", "target_id": constants.WHITE_SPY_ID, "message": "Heading to Archives."}
    result = validate_model_output(obs, response, legal)
    assert result.valid is True
    assert result.action == GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="Heading to Archives.")
    assert result.validated_action["message"] == "Heading to Archives."


def test_recipient_id_alias_is_accepted_same_as_target_id():
    """Spec section 3's own canonical example names the field
    `recipient_id` - accepted as an alias for target_id (see
    `_normalize_target_id`)."""
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    action = convert_response_to_action(obs, {"action": "send_message", "recipient_id": constants.WHITE_SPY_ID, "message": "hi"}, legal)
    assert action == GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="hi")


# -- recipient validation (spec section 39) ------------------------------------


def test_missing_recipient_is_rejected_as_missing_target():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    result = validate_model_output(obs, {"action": "send_message", "message": "hi"}, legal)
    assert result.valid is False
    assert result.error_type == "missing_target"


def test_self_recipient_is_rejected():
    """The recipient legal-action set only ever contains the *other*
    actor's id (spec section 39: "self recipient") - never offered, so a
    self-targeted message fails the same illegal-target path every other
    self-targeted action would."""
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    result = validate_model_output(obs, {"action": "send_message", "target_id": constants.BLACK_SPY_ID, "message": "hi"}, legal)
    assert result.valid is False
    assert result.error_type in {"illegal_target", "unknown_target"}


def test_invalid_unknown_recipient_is_rejected():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    result = validate_model_output(obs, {"action": "send_message", "target_id": "not_a_real_actor", "message": "hi"}, legal)
    assert result.valid is False
    assert result.error_type == "unknown_target"


def test_send_message_illegal_when_communication_disabled():
    state = _new_state(communication_enabled=False)
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    result = validate_model_output(obs, {"action": "send_message", "target_id": constants.WHITE_SPY_ID, "message": "hi"}, legal)
    assert result.valid is False


# -- message content validation (spec sections 3, 39) --------------------------


def test_empty_message_is_rejected():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    result = validate_model_output(obs, {"action": "send_message", "target_id": constants.WHITE_SPY_ID, "message": ""}, legal)
    assert result.valid is False
    assert result.error_type == "missing_message"


def test_whitespace_only_message_is_rejected():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    result = validate_model_output(obs, {"action": "send_message", "target_id": constants.WHITE_SPY_ID, "message": "   "}, legal)
    assert result.valid is False
    assert result.error_type == "missing_message"


def test_missing_message_field_is_rejected():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    result = validate_model_output(obs, {"action": "send_message", "target_id": constants.WHITE_SPY_ID}, legal)
    assert result.valid is False
    assert result.error_type == "missing_message"


def test_oversized_message_is_rejected():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    too_long = "x" * (constants.MESSAGE_MAX_LENGTH + 1)
    result = validate_model_output(obs, {"action": "send_message", "target_id": constants.WHITE_SPY_ID, "message": too_long}, legal)
    assert result.valid is False
    assert result.error_type == "oversized_message"


def test_message_at_exactly_the_length_limit_is_accepted():
    state = _new_state()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    legal = enumerate_legal_actions(obs)
    exactly_at_limit = "x" * constants.MESSAGE_MAX_LENGTH
    result = validate_model_output(obs, {"action": "send_message", "target_id": constants.WHITE_SPY_ID, "message": exactly_at_limit}, legal)
    assert result.valid is True


# -- engine-level delivery / decision-cost / event attribution -----------------


def test_apply_action_delivers_exactly_one_message_and_logs_canonical_events():
    state = _new_state()
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="hi"), dt=1 / 30)

    assert len(state.messages) == 1
    m = state.messages[0]
    assert m["sender_actor_id"] == constants.BLACK_SPY_ID
    assert m["recipient_actor_id"] == constants.WHITE_SPY_ID
    assert m["text"] == "hi"
    assert m["message_id"]

    sent_events = [e for e in state.event_log if e["type"] == "MESSAGE_SENT"]
    received_events = [e for e in state.event_log if e["type"] == "MESSAGE_RECEIVED"]
    assert len(sent_events) == 1 and sent_events[0]["actor_id"] == constants.BLACK_SPY_ID and sent_events[0]["recipient_id"] == constants.WHITE_SPY_ID
    assert len(received_events) == 1 and received_events[0]["actor_id"] == constants.WHITE_SPY_ID and received_events[0]["sender_id"] == constants.BLACK_SPY_ID
    assert sent_events[0]["message_id"] == received_events[0]["message_id"] == m["message_id"]


def test_send_message_charges_time_cost_like_every_other_timed_action():
    state = _new_state()
    actor = state.actors[constants.BLACK_SPY_ID]
    assert actor.action_cooldown == 0.0
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="hi"), dt=1 / 30)
    assert actor.action_cooldown == constants.TIME_COST_SEND_MESSAGE


def test_send_message_never_free_never_moves_never_mutates_position():
    state = _new_state()
    pos_before = (state.actors[constants.BLACK_SPY_ID].x, state.actors[constants.BLACK_SPY_ID].y)
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="hi"), dt=1 / 30)
    assert (state.actors[constants.BLACK_SPY_ID].x, state.actors[constants.BLACK_SPY_ID].y) == pos_before


def test_send_message_no_ops_when_communication_disabled():
    """Spec section 13: OFF genuinely blocks the engine, not just the
    legal_actions surface - a caller cannot route around the control
    condition by constructing the GameAction directly."""
    state = _new_state(communication_enabled=False)
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="hi"), dt=1 / 30)
    assert state.messages == []
    assert not any(e["type"] == "MESSAGE_SENT" for e in state.event_log)


def test_send_message_to_nonexistent_or_self_recipient_no_ops_defensively():
    state = _new_state()
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=constants.BLACK_SPY_ID, message="hi"), dt=1 / 30)
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id="ghost", message="hi"), dt=1 / 30)
    assert state.messages == []


def test_single_delivery_two_separate_sends_produce_exactly_two_messages():
    """Spec section 20: exactly-once delivery, verified by the inverse -
    two genuinely distinct SEND_MESSAGE actions must never collapse into
    one, or silently duplicate into three. The actor's own busy-cooldown
    (set by the first send, same as every other timed action) is cleared
    between the two calls here purely to isolate delivery-counting from
    that unrelated, already-covered-elsewhere gate."""
    state = _new_state()
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="first"), dt=1 / 30)
    state.actors[constants.BLACK_SPY_ID].action_cooldown = 0.0
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="second"), dt=1 / 30)
    assert [m["text"] for m in state.messages] == ["first", "second"]
    assert len({m["message_id"] for m in state.messages}) == 2


# -- message ordering (spec section 19) ----------------------------------------


def test_messages_received_recently_is_chronologically_ordered_and_bounded():
    state = _new_state()
    for i in range(constants.RECENT_MESSAGES_LIMIT + 3):
        state.apply_action(
            constants.BLACK_SPY_ID,
            GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message=f"msg{i}"),
            dt=1 / 30,
        )
        state.actors[constants.BLACK_SPY_ID].action_cooldown = 0.0  # bypass the busy-gate between synthetic sends
        state.elapsed_time += 1.0

    obs = build_observation(state, constants.WHITE_SPY_ID)
    assert len(obs.messages_received_recently) == constants.RECENT_MESSAGES_LIMIT
    texts = [m["text"] for m in obs.messages_received_recently]
    assert texts == sorted(texts, key=lambda t: int(t.replace("msg", "")))
    # The oldest 3 of the 8 sent messages must have fallen out of the window.
    assert texts[0] == "msg3"
    assert texts[-1] == "msg7"


# -- claim-vs-observed-fact separation (spec section 8) -------------------------


def test_communication_is_a_structurally_distinct_field_from_opponent_facts():
    state = _new_state()
    state.apply_action(constants.BLACK_SPY_ID, GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="Security is safe."), dt=1 / 30)
    state.elapsed_time += 1.0

    obs = build_observation(state, constants.WHITE_SPY_ID)
    payload = serialize_observation(obs)

    assert "communication" in payload
    assert payload["communication"]["messages_received_recently"][0]["text"] == "Security is safe."
    assert payload["communication"]["messages_received_recently"][0]["sender"] == "opponent"
    # The claim never gets folded into "opponent" (observed facts) or
    # "room"/"mission" - those keys are untouched by what was merely said.
    assert "Security is safe." not in str(payload["opponent"])
    assert "Security is safe." not in str(payload["room"])
