"""M5F.3A: unit tests for `src/evaluation/social_influence_metrics.py`'s
threat-capability world-state classification and the threat/cooperation/
warning factual response taggers (spec sections 30, 45-47). No live
calls - purely deterministic mappings from already-recorded probe dicts."""

from __future__ import annotations

from src.evaluation.social_influence_metrics import (
    AREA_AVOIDED,
    CAPABILITY_ABSENT,
    CAPABILITY_PRESENT,
    CAPABILITY_UNKNOWN_TO_RECIPIENT,
    COMPLEMENTARY_TARGET_SELECTED,
    DISARM_ADDED,
    DUPLICATE_TARGET_MAINTAINED,
    INSPECTION_ADDED,
    PROPOSAL_ACCEPTED_IN_MESSAGE,
    PROPOSAL_IGNORED,
    THREAT_COMPLIED,
    THREAT_IGNORED,
    THREAT_TRIGGERED_COUNTER_THREAT,
    THREAT_TRIGGERED_INSPECTION,
    THREAT_TRIGGERED_REPLY,
    THREAT_TRIGGERED_ROUTE_CHANGE,
    TRAP_TRIGGERED_ANYWAY,
    WARNING_IGNORED,
    WARNING_REFERENCED,
    classify_cooperation_response,
    classify_threat_capability,
    classify_threat_response,
    recipient_capability_knowledge,
)


def test_threat_capability_present_and_absent():
    assert classify_threat_capability(True) == CAPABILITY_PRESENT
    assert classify_threat_capability(False) == CAPABILITY_ABSENT


def test_recipient_capability_knowledge_always_unknown():
    assert recipient_capability_knowledge() == CAPABILITY_UNKNOWN_TO_RECIPIENT


def test_threat_ignored_when_behavior_and_reply_unchanged():
    probe = {
        "with_message_action": "move", "with_message_target": "archives",
        "without_message_action": "move", "without_message_target": "archives",
        "with_message_reply_text": None,
    }
    assert classify_threat_response(probe, ("filing_cabinet_01",)) == (THREAT_IGNORED,)


def test_threat_complied_when_avoided_after_threat():
    probe = {
        "with_message_action": "move", "with_message_target": "reception",
        "without_message_action": "move", "without_message_target": "filing_cabinet_01",
        "with_message_reply_text": None,
    }
    tags = classify_threat_response(probe, ("filing_cabinet_01",))
    assert THREAT_TRIGGERED_ROUTE_CHANGE in tags
    assert THREAT_COMPLIED in tags


def test_threat_triggered_inspection_tag():
    probe = {
        "with_message_action": "inspect", "with_message_target": "filing_cabinet_01",
        "without_message_action": "move", "without_message_target": "filing_cabinet_01",
        "with_message_reply_text": None,
    }
    tags = classify_threat_response(probe, ("filing_cabinet_01",))
    assert THREAT_TRIGGERED_INSPECTION in tags


def test_threat_triggered_reply_and_counter_threat():
    probe = {
        "with_message_action": "move", "with_message_target": "archives",
        "without_message_action": "move", "without_message_target": "archives",
        "with_message_reply_text": "If you trap it, I'll trap you back.",
    }
    tags = classify_threat_response(probe, ("filing_cabinet_01",))
    assert THREAT_TRIGGERED_REPLY in tags
    assert THREAT_TRIGGERED_COUNTER_THREAT in tags


def test_threat_reply_without_counter_threat_wording_not_tagged():
    probe = {
        "with_message_action": "move", "with_message_target": "archives",
        "without_message_action": "move", "without_message_target": "archives",
        "with_message_reply_text": "Understood, heading to Archives instead.",
    }
    tags = classify_threat_response(probe, ("filing_cabinet_01",))
    assert THREAT_TRIGGERED_REPLY in tags
    assert THREAT_TRIGGERED_COUNTER_THREAT not in tags


def test_cooperation_proposal_ignored():
    probe = {
        "with_message_action": "move", "with_message_target": "sofa_01",
        "without_message_action": "move", "without_message_target": "sofa_01",
        "with_message_reply_text": None,
    }
    assert classify_cooperation_response(probe, "filing_cabinet_01", ("archives",)) == (PROPOSAL_IGNORED,)


def test_cooperation_complementary_target_selected():
    probe = {
        "with_message_action": "move", "with_message_target": "archives",
        "without_message_action": "move", "without_message_target": "sofa_01",
        "with_message_reply_text": None,
    }
    tags = classify_cooperation_response(probe, "filing_cabinet_01", ("archives",))
    assert COMPLEMENTARY_TARGET_SELECTED in tags


def test_cooperation_duplicate_target_maintained():
    # Recipient acknowledges the proposal but doesn't actually reallocate
    # away from its own already-chosen target - a maintained duplicate,
    # distinct from silently ignoring the message outright.
    probe = {
        "with_message_action": "move", "with_message_target": "sofa_01",
        "without_message_action": "move", "without_message_target": "sofa_01",
        "with_message_reply_text": "Sounds good.",
    }
    tags = classify_cooperation_response(probe, "filing_cabinet_01", ("archives",))
    assert DUPLICATE_TARGET_MAINTAINED in tags


def test_cooperation_proposal_accepted_in_message():
    probe = {
        "with_message_action": "move", "with_message_target": "archives",
        "without_message_action": "move", "without_message_target": "sofa_01",
        "with_message_reply_text": "Sounds good, I'll take Archives.",
    }
    tags = classify_cooperation_response(probe, "filing_cabinet_01", ("archives",))
    assert PROPOSAL_ACCEPTED_IN_MESSAGE in tags


def test_warning_ignored():
    probe = {
        "with_message_action": "move", "with_message_target": "filing_cabinet_01",
        "without_message_action": "move", "without_message_target": "filing_cabinet_01",
        "with_message_reason": "Heading to Security.", "with_message_reply_text": None,
    }
    assert classify_warning_response_helper(probe) == (WARNING_IGNORED,)


def test_warning_inspection_added():
    probe = {
        "with_message_action": "inspect", "with_message_target": "filing_cabinet_01",
        "without_message_action": "move", "without_message_target": "filing_cabinet_01",
        "with_message_reason": "Black warned me, so I'll inspect first.", "with_message_reply_text": None,
    }
    tags = classify_warning_response_helper(probe)
    assert INSPECTION_ADDED in tags
    assert WARNING_REFERENCED in tags


def test_warning_disarm_added():
    probe = {
        "with_message_action": "disarm", "with_message_target": "filing_cabinet_01",
        "without_message_action": "move", "without_message_target": "filing_cabinet_01",
        "with_message_reason": "disarming the known trap", "with_message_reply_text": None,
    }
    tags = classify_warning_response_helper(probe)
    assert DISARM_ADDED in tags


def test_warning_area_avoided():
    probe = {
        "with_message_action": "move", "with_message_target": "archives",
        "without_message_action": "move", "without_message_target": "filing_cabinet_01",
        "with_message_reason": "avoiding security", "with_message_reply_text": None,
    }
    tags = classify_warning_response_helper(probe)
    assert AREA_AVOIDED in tags


def test_warning_trap_triggered_anyway():
    probe = {
        "with_message_action": "interact", "with_message_target": "filing_cabinet_01",
        "without_message_action": "interact", "without_message_target": "filing_cabinet_01",
        "with_message_reason": "searching anyway", "with_message_reply_text": None,
    }
    tags = classify_warning_response_helper(probe, trap_triggered_with=True)
    assert TRAP_TRIGGERED_ANYWAY in tags


def classify_warning_response_helper(probe, trap_triggered_with=False, trap_triggered_without=False):
    from src.evaluation.social_influence_metrics import classify_warning_response

    return classify_warning_response(probe, "filing_cabinet_01", trap_triggered_with, trap_triggered_without)
