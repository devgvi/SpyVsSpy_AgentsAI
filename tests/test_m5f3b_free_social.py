"""M5F.3B Phase A (spec section 54): network-free tests for free social
strategy infrastructure - SOCIAL_OFF genuinely omits message capability,
SOCIAL_ON preserves bounded messaging, independent actor cooldowns,
message delivered exactly once, same-boundary fairness, reply threading,
token-aware social classifiers, UNCLASSIFIED fallback, truthful warning
factual validation, threat capability snapshot, claim consistency,
sender-plan conflict, recipient response window, paired episode IDs,
secret safety. Real `GameState`/`Observation`/`LLMController`/`Game`
throughout - no network calls."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from typing import Any

from src import constants
from src.items import load_items
from src.level import load_level
from src.llm.clients.mock_client import MockModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController, LLMControllerConfig, serialize_observation
from src.observation import build_observation
from src.state import GameState

from src.evaluation.free_social import (
    ACKNOWLEDGEMENT_OR_REPLY,
    AMBIGUOUS,
    COORDINATION_FOLLOWUP,
    COORDINATION_PROPOSAL,
    COUNTER_THREAT,
    COUNTER_THREAT_LIKE,
    FACTUAL_CLAIM,
    NO_OBSERVABLE_RESPONSE,
    SAFETY_FOLLOWUP,
    SAFETY_WARNING,
    THREAT_AVOIDANCE,
    THREAT_LIKE_CONDITIONAL,
    UNCLASSIFIED,
    classify_claim_consistency_real,
    classify_message_category,
    claim_conflicts_with_sender_plan,
    cooperation_follow_through,
    misdirection_pattern,
    recipient_response_tags,
    threat_capability_snapshot,
    threat_follow_through,
)


def _new_state(communication_enabled: bool = True) -> GameState:
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    return GameState.new_game(level, items, seed=1, communication_enabled=communication_enabled)


class _SchemaCapturingClient(MockModelClient):
    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__(response=response)
        self.captured_schemas: list[dict[str, Any]] = []

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        self.captured_schemas.append(output_schema)
        return super().generate_action(system_prompt, user_prompt, output_schema, request_context)


# -- SOCIAL_OFF genuinely omits message capability (spec sections 6, 60) -------


def test_social_off_omits_message_field_from_output_schema():
    state = _new_state(communication_enabled=False)
    obs = build_observation(state, constants.BLACK_SPY_ID)
    payload = serialize_observation(obs)
    client = _SchemaCapturingClient({"action": "wait", "reason": "x"})
    controller = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(communication_enabled=False), model_client=client)
    controller._request_action(obs, payload)
    assert client.captured_schemas
    schema_props = client.captured_schemas[0]["properties"]
    assert "message" not in schema_props
    assert "message_to" not in schema_props
    assert "message_text" not in schema_props


def test_social_on_flat_schema_exposes_message_to_and_message_text():
    state = _new_state(communication_enabled=True)
    obs = build_observation(state, constants.BLACK_SPY_ID)
    payload = serialize_observation(obs)
    client = _SchemaCapturingClient({"action": "wait", "reason": "x"})
    controller = LLMController(
        constants.BLACK_SPY_ID,
        config=LLMControllerConfig(communication_enabled=True, zero_decision_cost_messaging=True, flat_message_schema=True, message_cooldown_decisions=3),
        model_client=client,
    )
    controller._request_action(obs, payload)
    schema_props = client.captured_schemas[0]["properties"]
    assert "message_to" in schema_props
    assert "message_text" in schema_props
    assert "message" not in schema_props


# -- Independent actor cooldowns (spec section 8) -------------------------------


def test_independent_actor_cooldowns_do_not_share_state():
    black = LLMController(constants.BLACK_SPY_ID, config=LLMControllerConfig(communication_enabled=True, zero_decision_cost_messaging=True, flat_message_schema=True, message_cooldown_decisions=3))
    white = LLMController(constants.WHITE_SPY_ID, config=LLMControllerConfig(communication_enabled=True, zero_decision_cost_messaging=True, flat_message_schema=True, message_cooldown_decisions=3))
    black._decisions_since_last_message = 0
    assert white._decisions_since_last_message is None
    assert black is not white


# -- paired episode IDs (spec section 44) ---------------------------------------


def test_paired_episode_ids_are_distinguishable_off_vs_on():
    off_id = "m5f3b-seed1-racer_vs_saboteur-mirrorA-SOCIAL_OFF"
    on_id = "m5f3b-seed1-racer_vs_saboteur-mirrorA-SOCIAL_ON"
    assert off_id != on_id
    assert off_id.rsplit("-", 1)[0] == on_id.rsplit("-", 1)[0]


# -- token-aware social classifiers (spec sections 29-30, 34, 37) --------------


def test_coordination_proposal_detection():
    assert classify_message_category("I'll take the safe; you take the desk.") == COORDINATION_PROPOSAL


def test_threat_like_conditional_detection():
    assert classify_message_category("If you go near the safe, I will trap it.") == THREAT_LIKE_CONDITIONAL


def test_counter_threat_like_when_replying_to_opponent_threat():
    tag = classify_message_category("If you go there, I'll trap it too.", replying_to_opponent_threat=True)
    assert tag == COUNTER_THREAT_LIKE


def test_acknowledgement_or_reply_detection():
    assert classify_message_category("Got it, thanks.") == ACKNOWLEDGEMENT_OR_REPLY


def test_factual_claim_without_engine_facts():
    assert classify_message_category("The desk is clear.") == FACTUAL_CLAIM


def test_unclassified_fallback_for_ambiguous_text():
    assert classify_message_category("purple elephants dance quietly") == UNCLASSIFIED
    assert classify_message_category(None) == UNCLASSIFIED
    assert classify_message_category("") == UNCLASSIFIED


def test_does_not_force_every_message_into_a_social_category():
    # Spec section 30: prefer UNCLASSIFIED over fragile guessing.
    assert classify_message_category("42.") == UNCLASSIFIED


# -- truthful warning factual validation (spec section 33) ---------------------


def test_safety_warning_requires_sender_legitimate_detection_and_recipient_ignorance():
    tagged = classify_message_category(
        "There's a trap at the filing cabinet.",
        references_target=True, sender_legitimately_detected_trap=True, recipient_independently_detected_trap=False,
    )
    assert tagged == SAFETY_WARNING


def test_safety_warning_not_tagged_when_sender_never_detected():
    tagged = classify_message_category(
        "There's a trap at the filing cabinet.",
        references_target=True, sender_legitimately_detected_trap=False,
    )
    assert tagged == FACTUAL_CLAIM


def test_safety_warning_not_tagged_when_recipient_already_knows():
    tagged = classify_message_category(
        "There's a trap at the filing cabinet.",
        references_target=True, sender_legitimately_detected_trap=True, recipient_independently_detected_trap=True,
    )
    assert tagged != SAFETY_WARNING


# -- threat capability snapshot (spec section 35) -------------------------------


def test_threat_capability_snapshot_facts_only_no_scalar_score():
    snap = threat_capability_snapshot(
        sender_trap_inventory={"electric": 2, "sticky": 0}, target_trappable=True,
        target_already_trapped=False, sender_can_plausibly_access_target=True,
    )
    assert snap == {
        "sender_has_compatible_trap": True, "target_legally_trappable": True,
        "sender_can_plausibly_access_target": True, "capability_visible_to_recipient": False,
    }
    assert "credibility_score" not in snap


def test_threat_capability_snapshot_no_traps():
    snap = threat_capability_snapshot(
        sender_trap_inventory={"electric": 0}, target_trappable=True,
        target_already_trapped=False, sender_can_plausibly_access_target=True,
    )
    assert snap["sender_has_compatible_trap"] is False


def test_capability_visible_to_recipient_is_always_false():
    # Observation.opponent/OpponentView never carries trap_inventory -
    # structurally never observable, regardless of inputs.
    for inv in ({"electric": 5}, {}, {"sticky": 1}):
        snap = threat_capability_snapshot(sender_trap_inventory=inv, target_trappable=True, target_already_trapped=False, sender_can_plausibly_access_target=True)
        assert snap["capability_visible_to_recipient"] is False


# -- claim consistency (spec section 39) -----------------------------------------


def test_claim_consistency_contradicted_when_sender_knows_trapped():
    result = classify_claim_consistency_real("It's clear.", sender_knows_target_trapped=True)
    assert result == "CONTRADICTED_BY_SENDER_OBSERVATION"


def test_claim_consistency_consistent_when_sender_knows_not_trapped():
    result = classify_claim_consistency_real("It's clear.", sender_knows_target_trapped=False)
    assert result == "CONSISTENT_WITH_SENDER_OBSERVATION"


def test_claim_consistency_not_fact_checkable_for_non_claims():
    result = classify_claim_consistency_real("heading to the exit", sender_knows_target_trapped=False)
    assert result == "NOT_FACT_CHECKABLE"


# -- sender-plan conflict (spec section 39) --------------------------------------


def test_sender_plan_conflict_reused_for_single_referenced_target():
    conflicts = claim_conflicts_with_sender_plan("Avoid the safe, it's dangerous.", "move", "ambassador_safe", ("ambassador_safe",))
    assert conflicts is True


def test_no_sender_plan_conflict_when_action_matches_warning():
    conflicts = claim_conflicts_with_sender_plan("Avoid the safe, it's dangerous.", "move", "desk_01", ("ambassador_safe",))
    assert conflicts is False


# -- recipient response window (spec sections 41-42) -----------------------------


def test_recipient_response_no_observable_response_when_window_empty():
    tags = recipient_response_tags(message_category=SAFETY_WARNING, message_text="trap at desk", referenced_target_id="desk_01", pre_message_target=None, window_records=[])
    assert tags == [NO_OBSERVABLE_RESPONSE]


def test_recipient_response_safety_followup_on_inspect():
    window = [{"final_action": "inspect", "final_target": "desk_01", "agent_stated_reason": "checking after warning"}]
    tags = recipient_response_tags(message_category=SAFETY_WARNING, message_text="trap at desk", referenced_target_id="desk_01", pre_message_target="desk_01", window_records=window)
    assert SAFETY_FOLLOWUP in tags


def test_recipient_response_threat_avoidance_on_target_change():
    window = [{"final_action": "move", "final_target": "other_object", "agent_stated_reason": ""}]
    tags = recipient_response_tags(message_category=THREAT_LIKE_CONDITIONAL, message_text="if you go there I'll trap it", referenced_target_id="desk_01", pre_message_target="desk_01", window_records=window)
    assert THREAT_AVOIDANCE in tags


def test_recipient_response_counter_threat_on_trap_placement():
    window = [{"final_action": "place_trap", "final_target": "desk_01", "agent_stated_reason": ""}]
    tags = recipient_response_tags(message_category=THREAT_LIKE_CONDITIONAL, message_text="if you go there I'll trap it", referenced_target_id="desk_01", pre_message_target="desk_01", window_records=window)
    assert COUNTER_THREAT in tags


def test_recipient_response_coordination_followup_on_complementary_target_change():
    window = [{"final_action": "move", "final_target": "desk_01", "agent_stated_reason": ""}]
    tags = recipient_response_tags(message_category=COORDINATION_PROPOSAL, message_text="I'll take the safe; you take the desk.", referenced_target_id="desk_01", pre_message_target="ambassador_safe", window_records=window)
    assert COORDINATION_FOLLOWUP in tags


def test_recipient_response_ambiguous_when_reason_present_but_nothing_factual_changed():
    window = [{"final_action": "wait", "final_target": None, "agent_stated_reason": "thinking about it"}]
    tags = recipient_response_tags(message_category=UNCLASSIFIED, message_text="hmm", referenced_target_id=None, pre_message_target=None, window_records=window)
    assert tags == [AMBIGUOUS]


# -- follow-through tracking (spec sections 36, 38, 40) --------------------------


def test_threat_follow_through_placed_trap():
    result = threat_follow_through("desk_01", [{"final_action": "place_trap", "final_target": "desk_01"}])
    assert result == "placed_trap_on_referenced_target"


def test_threat_follow_through_did_nothing_relevant():
    result = threat_follow_through("desk_01", [{"final_action": "wait", "final_target": None}])
    assert result == "did_nothing_relevant"


def test_cooperation_follow_through_reports_factual_flags_not_verdict():
    result = cooperation_follow_through(
        sender_claimed_target="ambassador_safe", recipient_expected_complementary_target="desk_01",
        sender_window_records=[{"final_target": "ambassador_safe"}], recipient_window_records=[{"final_target": "desk_01"}],
    )
    assert result["sender_pursued_claimed_target"] is True
    assert result["recipient_pursued_complementary_target"] is True
    assert "cooperation_succeeded" not in result


def test_misdirection_pattern_never_asserts_a_deception_verdict():
    result = misdirection_pattern(
        message_claims_safe=True, sender_knows_target_trapped=True, sender_warned_recipient_away=False,
        sender_window_records=[{"final_action": "move", "final_target": "desk_01"}], referenced_target_id="desk_01",
    )
    assert result["sender_claimed_target_clear_while_knowing_trapped"] is True
    assert "deception" not in str(result).lower()
    assert "lie" not in str(result).lower()


# -- message delivered exactly once / multi-tick non-redelivery (spec sections 10, 60) --


def test_message_delivered_exactly_once_per_decision():
    state = _new_state(communication_enabled=True)
    black = state.actors[constants.BLACK_SPY_ID]
    from src.actions import ActionType, GameAction

    action = GameAction(type=ActionType.WAIT, accompanying_message={"recipient_id": constants.WHITE_SPY_ID, "text": "hello"})
    state.apply_action(constants.BLACK_SPY_ID, action, 0.0)
    received = [m for m in state.messages if m["recipient_actor_id"] == constants.WHITE_SPY_ID]
    assert len(received) == 1

    # A second decision with no accompanying message never redelivers.
    action2 = GameAction(type=ActionType.WAIT)
    state.apply_action(constants.BLACK_SPY_ID, action2, 0.0)
    received_after = [m for m in state.messages if m["recipient_actor_id"] == constants.WHITE_SPY_ID]
    assert len(received_after) == 1


# -- secret safety (spec section 74) ---------------------------------------------


def test_no_secret_like_patterns_in_classifier_module_source():
    import inspect

    source = inspect.getsource(__import__("src.evaluation.free_social", fromlist=["free_social"]))
    for forbidden in ("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key"):
        assert forbidden not in source


# -- end-to-end wiring: analyze_social_on_episode against synthetic telemetry --


def _synthetic_decision(
    *, actor_id, decision_number, decision_id, game_time, action, target, reason,
    accompanying_message=None, room_objects=None,
):
    return {
        "actor_id": actor_id, "decision_number": decision_number, "decision_id": decision_id,
        "game_time": game_time, "personality_profile": None,
        "raw_response": {"action": action, "target_id": target, "reason": reason},
        "executed_action": {"action": action, "target_id": target, "accompanying_message": accompanying_message},
        "observation": {"self": {"room": "security_room"}, "room": {"objects": room_objects or []}},
    }


def test_analyze_social_on_episode_threat_avoidance_end_to_end():
    from src.evaluation.free_social_matchup import analyze_social_on_episode

    fc = {"id": "filing_cabinet_01", "label": "filing cabinet", "trap_status": "unknown", "trappable": True, "trap_is_mine": False}
    decisions = [
        _synthetic_decision(
            actor_id=constants.BLACK_SPY_ID, decision_number=1, decision_id="d1", game_time=1.0,
            action="wait", target=None, reason="warning white",
            accompanying_message={"recipient_id": constants.WHITE_SPY_ID, "text": "If you go near filing_cabinet_01, I will trap it.", "message_id": "msg1"},
            room_objects=[fc],
        ),
        _synthetic_decision(
            actor_id=constants.WHITE_SPY_ID, decision_number=1, decision_id="d2", game_time=0.5,
            action="move", target="filing_cabinet_01", reason="heading to filing cabinet",
        ),
        _synthetic_decision(
            actor_id=constants.WHITE_SPY_ID, decision_number=2, decision_id="d3", game_time=2.0,
            action="move", target="desk_01", reason="changed course after threat",
        ),
    ]
    result = analyze_social_on_episode(decisions, [], "test-episode", seed=1)
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert msg["category"] == THREAT_LIKE_CONDITIONAL
    assert msg["referenced_target_id"] == "filing_cabinet_01"
    assert THREAT_AVOIDANCE in msg["recipient_response_tags"]
    assert len(result["threat_events"]) == 1
    assert result["threat_events"][0]["follow_through"] in ("did_nothing_relevant", "acted_inconsistently_with_message")


def test_analyze_social_on_episode_safety_warning_end_to_end():
    from src.evaluation.free_social_matchup import analyze_social_on_episode

    fc_detected = {"id": "filing_cabinet_01", "label": "filing cabinet", "trap_status": "detected", "trap_type": "electric", "trap_is_mine": False, "trappable": True}
    decisions = [
        _synthetic_decision(
            actor_id=constants.BLACK_SPY_ID, decision_number=1, decision_id="d1", game_time=1.0,
            action="wait", target=None, reason="warning white",
            accompanying_message={"recipient_id": constants.WHITE_SPY_ID, "text": "There is a trap at filing_cabinet_01.", "message_id": "msg1"},
            room_objects=[fc_detected],
        ),
        _synthetic_decision(
            actor_id=constants.WHITE_SPY_ID, decision_number=1, decision_id="d2", game_time=2.0,
            action="inspect", target="filing_cabinet_01", reason="checking after warning",
        ),
    ]
    result = analyze_social_on_episode(decisions, [], "test-episode", seed=1)
    msg = result["messages"][0]
    assert msg["category"] == SAFETY_WARNING
    assert SAFETY_FOLLOWUP in msg["recipient_response_tags"]
    assert len(result["warning_events"]) == 1
    assert result["warning_events"][0]["consistency"] == "CONSISTENT_WITH_SENDER_OBSERVATION"


def test_message_ids_do_not_collide_across_different_senders_same_decision_number():
    # Real bug found live in Stage 1 (3 of 19 messages collided): the
    # fallback message_id used only episode_id + decision_number, but
    # each actor keeps its OWN independent decision_number counter, so
    # two different senders' own decision 1 produced the identical
    # fallback id.
    from src.evaluation.free_social_matchup import analyze_social_on_episode

    decisions = [
        _synthetic_decision(
            actor_id=constants.BLACK_SPY_ID, decision_number=1, decision_id="dblack1", game_time=1.0,
            action="wait", target=None, reason="",
            accompanying_message={"recipient_id": constants.WHITE_SPY_ID, "text": "Hello from black."},
        ),
        _synthetic_decision(
            actor_id=constants.WHITE_SPY_ID, decision_number=1, decision_id="dwhite1", game_time=2.0,
            action="wait", target=None, reason="",
            accompanying_message={"recipient_id": constants.BLACK_SPY_ID, "text": "Hello from white."},
        ),
    ]
    result = analyze_social_on_episode(decisions, [], "test-episode", seed=1)
    ids = [m["message_id"] for m in result["messages"]]
    assert len(ids) == len(set(ids)) == 2


def test_analyze_social_on_episode_coordination_end_to_end():
    from src.evaluation.free_social_matchup import analyze_social_on_episode

    decisions = [
        _synthetic_decision(
            actor_id=constants.BLACK_SPY_ID, decision_number=1, decision_id="d1", game_time=1.0,
            action="wait", target=None, reason="proposing split",
            accompanying_message={"recipient_id": constants.WHITE_SPY_ID, "text": "I'll take the desk_01; you take filing_cabinet_01.", "message_id": "msg1"},
            room_objects=[{"id": "filing_cabinet_01", "label": "filing cabinet", "trap_status": "unknown", "trappable": True}],
        ),
        _synthetic_decision(
            actor_id=constants.WHITE_SPY_ID, decision_number=1, decision_id="d2", game_time=0.5,
            action="wait", target=None, reason="idle before message",
        ),
        _synthetic_decision(
            actor_id=constants.WHITE_SPY_ID, decision_number=2, decision_id="d3", game_time=2.0,
            action="move", target="filing_cabinet_01", reason="following the proposal",
        ),
    ]
    result = analyze_social_on_episode(decisions, [], "test-episode", seed=1)
    msg = result["messages"][0]
    assert msg["category"] == COORDINATION_PROPOSAL
    assert COORDINATION_FOLLOWUP in msg["recipient_response_tags"]
    assert len(result["cooperation_events"]) == 1
