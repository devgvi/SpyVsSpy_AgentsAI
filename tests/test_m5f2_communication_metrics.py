"""M5F.2: unit tests for `src/evaluation/communication_metrics.py`'s
rule-based message extraction, outcome classification, and frequency/
response/cooperation metrics - synthetic decision-trace-shaped dicts only,
no live calls, no LLM-as-a-judge (spec sections 46, 48-49)."""

from __future__ import annotations

from src import constants
from src.evaluation.communication_metrics import (
    AMBIGUOUS,
    BEHAVIOR_CHANGED,
    COORDINATION_FOLLOWUP,
    EXPLICITLY_REFERENCED,
    NO_OBSERVABLE_RESPONSE,
    build_message_records,
    classify_message_outcome,
    communication_frequency_metrics,
    cooperation_events,
    extract_sent_messages,
    message_cost_metrics,
    response_metrics,
)


def _decision(**overrides) -> dict:
    base = {
        "actor_id": constants.BLACK_SPY_ID,
        "decision_id": "d1",
        "decision_number": 1,
        "game_time": 10.0,
        "final_action": "wait",
        "final_target": None,
        "final_message": None,
        "final_action_valid": True,
        "agent_stated_reason": None,
        "progress_delta": ["NO_PROGRESS"],
        "repeat_count": 0,
    }
    base.update(overrides)
    return base


# -- extraction -----------------------------------------------------------------


def test_extract_sent_messages_only_pulls_valid_send_message_decisions():
    records = [
        _decision(decision_id="d1", final_action="move", final_target="desk_01"),
        _decision(decision_id="d2", final_action="send_message", final_target=constants.WHITE_SPY_ID, final_message="hi", final_action_valid=True),
        _decision(decision_id="d3", final_action="send_message", final_target=constants.WHITE_SPY_ID, final_message="ignored", final_action_valid=False),
    ]
    messages = extract_sent_messages(records, sender_profile="racer")
    assert len(messages) == 1
    assert messages[0]["message_id"] == "d2"
    assert messages[0]["message_text"] == "hi"
    assert messages[0]["recipient_actor_id"] == constants.WHITE_SPY_ID
    assert messages[0]["sender_profile"] == "racer"


# -- outcome classification (spec sections 24-25, 46) ---------------------------


def test_no_observable_response_when_recipient_never_decides_again():
    message = {"message_id": "m1", "game_time": 10.0, "sender_actor_id": constants.BLACK_SPY_ID, "message_text": "heading to archives"}
    outcome = classify_message_outcome(message, recipient_records=[])
    assert outcome["primary_outcome"] == NO_OBSERVABLE_RESPONSE
    assert outcome["recipient_next_decision_id"] is None


def test_explicitly_referenced_when_rationale_names_the_sender():
    message = {"message_id": "m1", "game_time": 10.0, "sender_actor_id": constants.BLACK_SPY_ID, "message_text": "Security may be trapped."}
    recipient_records = [_decision(
        actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=5, game_time=12.0,
        final_action="inspect", final_target="security_terminal",
        agent_stated_reason="Black warned that Security may be trapped, so I'll inspect first.",
    )]
    outcome = classify_message_outcome(message, recipient_records)
    assert outcome["primary_outcome"] == EXPLICITLY_REFERENCED
    assert outcome["recipient_next_action"] == "inspect"


def test_explicitly_referenced_when_rationale_shares_content_words_with_message():
    message = {"message_id": "m1", "game_time": 10.0, "sender_actor_id": constants.WHITE_SPY_ID, "message_text": "Avoid the archives room, saw a trap there."}
    recipient_records = [_decision(
        actor_id=constants.BLACK_SPY_ID, decision_id="b1", decision_number=3, game_time=11.0,
        final_action="move", final_target="reception",
        agent_stated_reason="Avoiding archives since a trap was reported there.",
    )]
    outcome = classify_message_outcome(message, recipient_records)
    assert outcome["primary_outcome"] == EXPLICITLY_REFERENCED


def test_behavior_changed_when_target_changes_with_no_explicit_reference():
    message = {"message_id": "m1", "game_time": 10.0, "sender_actor_id": constants.BLACK_SPY_ID, "message_text": "hi there"}
    prior = _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w0", decision_number=1, game_time=8.0, final_target="desk_01")
    reaction = _decision(
        actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=2, game_time=12.0,
        final_action="move", final_target="safe_01", agent_stated_reason="Continuing my search.",
    )
    outcome = classify_message_outcome(message, [prior, reaction])
    assert outcome["primary_outcome"] == BEHAVIOR_CHANGED
    assert outcome["behavior_change_type"] == "target_changed"


def test_behavior_changed_type_safety_action_takes_precedence_signal():
    message = {"message_id": "m1", "game_time": 10.0, "sender_actor_id": constants.BLACK_SPY_ID, "message_text": "watch out"}
    reaction = _decision(
        actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=2, game_time=12.0,
        final_action="inspect", final_target="desk_01", agent_stated_reason="Just being careful.",
    )
    outcome = classify_message_outcome(message, [reaction])
    assert outcome["behavior_change_type"] == "safety_action"


def test_coordination_followup_when_proposal_language_and_behavior_change_both_present():
    message = {"message_id": "m1", "game_time": 10.0, "sender_actor_id": constants.BLACK_SPY_ID, "message_text": "I'll search Archives; you take Security."}
    prior = _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w0", decision_number=1, game_time=8.0, final_target="desk_01")
    reaction = _decision(
        actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=2, game_time=12.0,
        final_action="move", final_target="security_terminal", agent_stated_reason="Moving on.",
    )
    outcome = classify_message_outcome(message, [prior, reaction])
    assert outcome["primary_outcome"] == COORDINATION_FOLLOWUP


def test_ambiguous_when_recipient_decides_but_nothing_ties_back_and_nothing_changed():
    message = {"message_id": "m1", "game_time": 10.0, "sender_actor_id": constants.BLACK_SPY_ID, "message_text": "hello there friend"}
    prior = _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w0", decision_number=1, game_time=8.0, final_target="desk_01")
    reaction = _decision(
        actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=2, game_time=12.0,
        final_action="move", final_target="desk_01", agent_stated_reason="Still searching this desk.",
    )
    outcome = classify_message_outcome(message, [prior, reaction])
    assert outcome["primary_outcome"] == AMBIGUOUS


def test_classification_never_uses_rationale_text_beyond_word_overlap_never_hidden_reasoning():
    """Rule-based only (spec section 48) - a rationale with zero words in
    common and no sender-color mention must never be classified as
    EXPLICITLY_REFERENCED merely because *some* reaction happened."""
    message = {"message_id": "m1", "game_time": 10.0, "sender_actor_id": constants.BLACK_SPY_ID, "message_text": "Going to the ambassador office now."}
    reaction = _decision(
        actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=2, game_time=12.0,
        final_action="wait", agent_stated_reason="Nothing useful to do right now.",
    )
    outcome = classify_message_outcome(message, [reaction])
    assert outcome["primary_outcome"] in {NO_OBSERVABLE_RESPONSE, AMBIGUOUS}


# -- build_message_records (duel-wide join) --------------------------------------


def test_build_message_records_joins_both_directions_and_sorts_by_game_time():
    black_records = [
        _decision(actor_id=constants.BLACK_SPY_ID, decision_id="b1", decision_number=1, game_time=5.0, final_action="send_message", final_target=constants.WHITE_SPY_ID, final_message="second"),
    ]
    white_records = [
        _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=1, game_time=2.0, final_action="send_message", final_target=constants.BLACK_SPY_ID, final_message="first"),
    ]
    records = build_message_records(black_records, white_records, black_profile="saboteur", white_profile="cautious")
    assert [r["message_text"] for r in records] == ["first", "second"]
    assert records[0]["sender_profile"] == "cautious"
    assert records[1]["sender_profile"] == "saboteur"


# -- frequency / response / cost metrics (spec sections 30-31, 35) ---------------


def test_communication_frequency_metrics_counts_sent_and_received():
    black_records = [
        _decision(actor_id=constants.BLACK_SPY_ID, decision_id="b1", decision_number=1, game_time=1.0, final_action="send_message", final_target=constants.WHITE_SPY_ID, final_message="hello there"),
        _decision(actor_id=constants.BLACK_SPY_ID, decision_id="b2", decision_number=2, game_time=2.0, final_action="move", final_target="desk_01"),
    ]
    white_records = [
        _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=1, game_time=3.0, final_action="wait"),
    ]
    message_records = build_message_records(black_records, white_records)
    freq = communication_frequency_metrics(black_records, message_records, constants.BLACK_SPY_ID)
    assert freq["messages_sent"] == 1
    assert freq["decisions_spent_messaging"] == 1
    assert freq["messages_per_10_decisions"] == 5.0
    assert freq["average_message_length"] == len("hello there")


def test_response_metrics_counts_received_side_outcomes():
    black_records = [
        _decision(actor_id=constants.BLACK_SPY_ID, decision_id="b1", decision_number=1, game_time=1.0, final_action="send_message", final_target=constants.WHITE_SPY_ID, final_message="trap ahead, be careful"),
    ]
    white_records = [
        _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w0", decision_number=1, game_time=0.5, final_target="desk_01"),
        _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=2, game_time=2.0, final_action="inspect", final_target="desk_01", agent_stated_reason="Being careful."),
    ]
    message_records = build_message_records(black_records, white_records)
    metrics = response_metrics(message_records, constants.WHITE_SPY_ID)
    assert metrics["received_messages"] == 1
    assert metrics["followed_by_safety_action"] == 1


def test_message_cost_metrics_counts_decisions_spent_communicating():
    records = [
        _decision(decision_id="b1", final_action="send_message", final_target=constants.WHITE_SPY_ID, final_message="hi"),
        _decision(decision_id="b2", final_action="move"),
        _decision(decision_id="b3", final_action="send_message", final_target=constants.WHITE_SPY_ID, final_message="again"),
    ]
    cost = message_cost_metrics(records, constants.BLACK_SPY_ID)
    assert cost["decisions_spent_communicating"] == 2


# -- cooperation events (spec section 32, no score) ------------------------------


def test_cooperation_events_are_descriptive_never_a_score():
    black_records = [
        _decision(actor_id=constants.BLACK_SPY_ID, decision_id="b1", decision_number=1, game_time=1.0, final_action="send_message", final_target=constants.WHITE_SPY_ID, final_message="I'll search Archives; you take Security."),
    ]
    white_records = [
        _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w0", decision_number=1, game_time=0.5, final_target="desk_01"),
        _decision(actor_id=constants.WHITE_SPY_ID, decision_id="w1", decision_number=2, game_time=2.0, final_action="move", final_target="security_terminal", agent_stated_reason="Moving on."),
    ]
    message_records = build_message_records(black_records, white_records)
    events = cooperation_events(message_records)
    assert any(e["event"] == "coordination_proposal" for e in events)
    for e in events:
        assert "score" not in e and "rating" not in e
