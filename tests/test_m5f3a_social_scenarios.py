"""M5F.3A Phase E (spec section 40): network-free tests for the controlled
social-influence world builder and message delivery in
`src/evaluation/social_influence_scenarios.py`. No live model calls -
message delivery/state-parity/partial-information facts only; recipient
decisions themselves are only ever exercised live (Phase F/G), never
faked here (this milestone bans LLM-as-a-judge and any evaluator model -
spec section 56 - so nothing here simulates what a model would decide).
"""

from __future__ import annotations

from src import constants
from src.evaluation.social_influence_scenarios import (
    COOPERATION,
    ENV_TRAP_TARGET_ID,
    MESSAGE_TEXTS,
    MISDIRECTION,
    NON_CREDIBLE_THREAT,
    THREAT,
    THREAT_CAPABILITY_TARGET_ID,
    WARNING,
    build_world,
)
from src.observation import build_observation


def test_no_message_control_delivers_no_message():
    state = build_world()
    assert state.messages == []


def test_controlled_message_delivers_through_real_send_message_path():
    from src.actions import ActionType, GameAction

    state = build_world()
    state.apply_action(
        constants.BLACK_SPY_ID,
        GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message=MESSAGE_TEXTS[WARNING]),
        dt=1 / 30,
    )
    assert len(state.messages) == 1
    assert state.messages[0]["text"] == MESSAGE_TEXTS[WARNING]
    assert state.messages[0]["sender_actor_id"] == constants.BLACK_SPY_ID
    assert state.messages[0]["recipient_actor_id"] == constants.WHITE_SPY_ID

    obs = build_observation(state, constants.WHITE_SPY_ID)
    assert any(m["text"] == MESSAGE_TEXTS[WARNING] for m in obs.messages_received_recently)


def test_warning_uses_a_real_environmental_trap():
    state = build_world()
    trap = state.trap_on(ENV_TRAP_TARGET_ID)
    assert trap is not None
    assert trap["owner_id"] == constants.ENVIRONMENT_OWNER_ID
    # Sender (Black) legitimately detected it.
    assert state.detected_trap_type(ENV_TRAP_TARGET_ID, constants.BLACK_SPY_ID) == trap["type"]


def test_recipient_does_not_know_trap_beforehand():
    state = build_world()
    obs = build_observation(state, constants.WHITE_SPY_ID)
    cabinet = next(o for o in obs.objects if o.id == ENV_TRAP_TARGET_ID)
    assert cabinet.trap_status == "unknown"
    assert cabinet.trap_type is None
    assert state.detected_trap_type(ENV_TRAP_TARGET_ID, constants.WHITE_SPY_ID) is None


def test_credible_threat_sender_capability_present():
    state = build_world(sender_trap_capable=True)
    black = state.actors[constants.BLACK_SPY_ID]
    assert constants.BLACK_SPY_ID in state._object_entry(THREAT_CAPABILITY_TARGET_ID)["inspected_by"]
    assert any(count > 0 for count in black.trap_inventory.values())


def test_non_credible_threat_capability_absent():
    state = build_world(sender_trap_capable=False)
    black = state.actors[constants.BLACK_SPY_ID]
    assert all(count == 0 for count in black.trap_inventory.values())


def test_recipient_not_explicitly_told_credibility():
    # Spec section 16: never inject a credible/non-credible label, and
    # the recipient's own Observation never carries the sender's private
    # inventory/inspection facts at all - confirm both THREAT and
    # NON_CREDIBLE_THREAT worlds produce an identical recipient
    # Observation (position/inventory/objects) aside from the message
    # text itself, which is byte-identical wording either way.
    capable_state = build_world(sender_trap_capable=True)
    incapable_state = build_world(sender_trap_capable=False)
    capable_obs = build_observation(capable_state, constants.WHITE_SPY_ID)
    incapable_obs = build_observation(incapable_state, constants.WHITE_SPY_ID)
    assert capable_obs.position == incapable_obs.position
    assert capable_obs.inventory == incapable_obs.inventory
    assert capable_obs.trap_inventory == incapable_obs.trap_inventory
    assert capable_obs.objects == incapable_obs.objects
    assert capable_obs.current_room == incapable_obs.current_room
    assert MESSAGE_TEXTS[THREAT] == MESSAGE_TEXTS[NON_CREDIBLE_THREAT]
    assert "credible" not in MESSAGE_TEXTS[THREAT].lower()
    assert "non-credible" not in MESSAGE_TEXTS[THREAT].lower()


def test_misdirection_contradicts_sender_known_world_state():
    # Spec section 14: sender has legitimate knowledge the claim
    # contradicts - Black detected the real trap yet the message claims
    # the area is clear.
    state = build_world()
    trap = state.trap_on(ENV_TRAP_TARGET_ID)
    assert trap is not None
    assert state.detected_trap_type(ENV_TRAP_TARGET_ID, constants.BLACK_SPY_ID) == trap["type"]
    assert MESSAGE_TEXTS[MISDIRECTION] == "Security is clear."


def test_same_recipient_state_across_all_message_class_worlds():
    # Spec section 15: for paired comparisons, the recipient's own
    # position/inventory/Observation/legal-physical-actions must be
    # identical across conditions - only the incoming message content
    # (delivered separately, after this snapshot) ever varies.
    baseline = build_observation(build_world(), constants.WHITE_SPY_ID)
    for _ in range(3):
        obs = build_observation(build_world(), constants.WHITE_SPY_ID)
        assert obs.position == baseline.position
        assert obs.inventory == baseline.inventory
        assert obs.trap_inventory == baseline.trap_inventory
        assert obs.objects == baseline.objects
        assert obs.current_room == baseline.current_room
        assert obs.mission_time_remaining == baseline.mission_time_remaining


def test_cooperation_message_text_matches_spec_example():
    assert MESSAGE_TEXTS[COOPERATION] == "I'll handle Archives. You take Security."


def test_no_placement_credit_leaks_to_either_spy_in_social_world():
    state = build_world()
    black_obs = build_observation(state, constants.BLACK_SPY_ID)
    white_obs = build_observation(state, constants.WHITE_SPY_ID)
    cabinet_black = next(o for o in black_obs.objects if o.id == ENV_TRAP_TARGET_ID)
    cabinet_white = next(o for o in white_obs.objects if o.id == ENV_TRAP_TARGET_ID)
    assert cabinet_black.trap_is_mine is False
    assert cabinet_white.trap_is_mine is False
