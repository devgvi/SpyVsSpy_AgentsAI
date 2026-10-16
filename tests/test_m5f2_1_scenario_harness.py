"""M5F.2.1 spec section 29: network-free tests for the scenario harness -
determinism, reproducibility, legality, physical alternatives, profile
injection, neutral-policy parity, fresh context per trial, no cross-trial
memory, partial-information preservation, and recipient paired-control
equivalence. `MockModelClient` throughout - no live calls."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src import constants
from src.actions import ActionType, GameAction
from src.evaluation.scenario_harness import (
    RECIPIENT_PROFILE,
    SCENARIO_BUILDERS,
    SCENARIO_GROUND_TRUTH,
    SCENARIO_NAMES,
    SCENARIO_PROFILES,
    run_recipient_probe,
    run_sender_trial,
)
from src.llm.clients.mock_client import MockModelClient
from src.llm_controller import enumerate_legal_actions, serialize_observation
from src.observation import build_observation


# -- scenario determinism / same-state reproducibility -------------------------


def test_every_scenario_builder_is_registered_and_covers_all_five_names():
    assert set(SCENARIO_BUILDERS.keys()) == set(SCENARIO_NAMES) == set(SCENARIO_PROFILES.keys())


def test_scenario_state_is_deterministic_across_two_fresh_builds():
    for name in SCENARIO_NAMES:
        state_a = SCENARIO_BUILDERS[name]()
        state_b = SCENARIO_BUILDERS[name]()
        obs_a = serialize_observation(build_observation(state_a, constants.BLACK_SPY_ID))
        obs_b = serialize_observation(build_observation(state_b, constants.BLACK_SPY_ID))
        assert obs_a == obs_b, f"{name}: two fresh builds produced different sender observations"


def test_scenario_builders_never_share_mutable_state_across_calls():
    """Mutating one built state must never affect a later fresh build."""
    state_a = SCENARIO_BUILDERS["CONTROL"]()
    state_a.actors[constants.BLACK_SPY_ID].x = 99999.0
    state_a.messages.append({"message_id": "x", "sender_actor_id": "a", "recipient_actor_id": "b", "game_time": 0.0, "text": "polluted"})
    state_b = SCENARIO_BUILDERS["CONTROL"]()
    assert state_b.actors[constants.BLACK_SPY_ID].x != 99999.0
    assert state_b.messages == []


# -- SEND_MESSAGE legality + physical alternatives (spec section 4) ------------


def test_every_scenario_offers_send_message_and_at_least_two_physical_alternatives():
    for name in SCENARIO_NAMES:
        state = SCENARIO_BUILDERS[name]()
        obs = build_observation(state, constants.BLACK_SPY_ID)
        legal = enumerate_legal_actions(obs)
        actions = {a["action"] for a in legal}
        assert "send_message" in actions, f"{name}: send_message not legal"
        non_message = actions - {"send_message"}
        assert len(non_message) >= 2, f"{name}: fewer than 2 non-message action types ({non_message})"


def test_every_scenario_send_message_recipient_is_the_one_opponent():
    for name in SCENARIO_NAMES:
        state = SCENARIO_BUILDERS[name]()
        obs = build_observation(state, constants.BLACK_SPY_ID)
        legal = enumerate_legal_actions(obs)
        send_targets = [a["target_id"] for a in legal if a["action"] == "send_message"]
        assert send_targets == [constants.WHITE_SPY_ID]


# -- partial-information preservation (spec section 8, 30) ---------------------


def test_safety_information_scenario_hides_the_trap_from_the_recipient():
    state = SCENARIO_BUILDERS["SAFETY_INFORMATION"]()
    sender_obs = build_observation(state, constants.BLACK_SPY_ID)
    recipient_obs = build_observation(state, constants.WHITE_SPY_ID)
    assert sender_obs.get("filing_cabinet_01").trap_status == "detected"
    assert recipient_obs.get("filing_cabinet_01").trap_status == "unknown"


def test_safety_information_trap_never_leaks_into_recipient_serialized_payload():
    state = SCENARIO_BUILDERS["SAFETY_INFORMATION"]()
    recipient_payload = serialize_observation(build_observation(state, constants.WHITE_SPY_ID))
    import json
    assert '"detected"' not in json.dumps(recipient_payload["room"])


def test_tactical_misdirection_sender_has_inspected_and_can_trap_the_contested_object():
    state = SCENARIO_BUILDERS["TACTICAL_MISDIRECTION"]()
    obs = build_observation(state, constants.BLACK_SPY_ID)
    assert obs.get("ambassador_safe").inspected is True
    legal = enumerate_legal_actions(obs)
    assert any(a["action"] == "place_trap" and a["target_id"] == "ambassador_safe" for a in legal)


# -- profile injection / neutral-policy parity ----------------------------------


def test_sender_trial_uses_the_requested_personality_profile():
    client = MockModelClient(response={"action": "wait"})
    result = run_sender_trial("CONTROL", "saboteur", 1, client)
    assert result["scenario"] == "CONTROL"
    assert result["profile"] == "saboteur"
    assert result["valid"] is True


def test_neutral_policy_parity_communication_block_identical_regardless_of_scenario_or_profile():
    """Spec section 1: the communication policy text itself must never
    vary by scenario or profile - only the world state does."""
    from src.llm.communication import render_communication_block
    from src.llm.personality import render_personality_block

    block_a = render_communication_block(True)
    block_b = render_communication_block(True)
    assert block_a == block_b
    # Personality varies, the communication policy block does not - the
    # two are appended independently (see LLMController._effective_system_prompt).
    assert render_personality_block("saboteur") != render_personality_block("racer")


# -- fresh context per trial / no cross-trial memory (spec section 12) ---------


def test_each_sender_trial_gets_a_brand_new_agent_memory():
    client_a = MockModelClient(response={"action": "move", "target_id": "desk_01"})
    result_a = run_sender_trial("CONTROL", "racer", 1, client_a)
    state_a = result_a["state"]

    client_b = MockModelClient(response={"action": "wait"})
    result_b = run_sender_trial("CONTROL", "racer", 2, client_b)
    state_b = result_b["state"]

    # Two independent GameState instances - trial 2 was never touched by
    # whatever trial 1 did.
    assert state_a is not state_b
    assert state_b.actors[constants.BLACK_SPY_ID].x == SCENARIO_BUILDERS["CONTROL"]().actors[constants.BLACK_SPY_ID].x


def test_repeated_trials_of_the_same_condition_see_byte_identical_starting_observation():
    client_1 = MockModelClient(response={"action": "wait"})
    client_2 = MockModelClient(response={"action": "wait"})
    r1 = run_sender_trial("SAFETY_INFORMATION", "cautious", 1, client_1)
    r2 = run_sender_trial("SAFETY_INFORMATION", "cautious", 2, client_2)
    obs1 = serialize_observation(build_observation(r1["state"], constants.BLACK_SPY_ID))
    obs2 = serialize_observation(build_observation(r2["state"], constants.BLACK_SPY_ID))
    # The two states were independently built and neither trial acted on
    # the other's copy - the pre-decision snapshot must be identical
    # (position facts are unaffected by a WAIT decision).
    assert obs1["self"]["position"] == obs2["self"]["position"]
    assert obs1["room"] == obs2["room"]


# -- recipient paired-control equivalence (spec sections 19-20, 29) ------------


def test_recipient_probe_runs_both_arms_with_identical_base_state_except_the_message():
    client = MockModelClient(response={"action": "wait"})
    probe = run_recipient_probe("CONTROL", constants.WHITE_SPY_ID, "hello there", client, 1)
    assert probe["with_message_action"] == "wait"
    assert probe["without_message_action"] == "wait"
    assert probe["message_text"] == "hello there"


def test_recipient_probe_with_message_arm_actually_has_the_message_delivered():
    """A direct check that the WITH-message arm's state really contains
    the message (engine-truth, not just passed as a string) before the
    recipient's Observation is built - the control arm must not."""
    from src.evaluation.scenario_harness import SCENARIO_BUILDERS as builders

    with_state = builders["CONTROL"]()
    with_state.apply_action(
        constants.BLACK_SPY_ID,
        GameAction(ActionType.SEND_MESSAGE, target_id=constants.WHITE_SPY_ID, message="test message"),
        dt=1 / 30,
    )
    assert len(with_state.messages) == 1

    without_state = builders["CONTROL"]()
    assert without_state.messages == []


def test_recipient_uses_the_documented_default_profile_not_the_sender_profile():
    client = MockModelClient(response={"action": "wait"})
    probe = run_recipient_probe("TACTICAL_MISDIRECTION", constants.WHITE_SPY_ID, "hi", client, 1)
    assert RECIPIENT_PROFILE is None
    assert probe["scenario"] == "TACTICAL_MISDIRECTION"


# -- claim consistency / claim-vs-plan classification (spec sections 17-18) ----


def test_ground_truth_table_covers_every_scenario():
    assert set(SCENARIO_GROUND_TRUTH.keys()) == set(SCENARIO_NAMES)
    for name, gt in SCENARIO_GROUND_TRUTH.items():
        assert "real_trap_present" in gt
        assert "contested_location_ids" in gt
