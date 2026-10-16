"""M5F.3A: rule-based classification for the controlled social-influence
recipient probe (spec sections 22-30, 44-47) - threat capability world-state
labeling, and factual (never persuasion-scored) response tags for the
threat/cooperation/warning message families. No LLM-as-a-judge anywhere
(spec section 56); every function here is a pure, deterministic mapping
from already-recorded facts, same discipline as
`opportunity_metrics.py` (M5F.2.1/M5F.3A's `mentions_danger_claim`/
`mentions_safe_claim` are reused directly from there rather than
reimplemented, so both modules share one fixed lexical-collision fix).
"""

from __future__ import annotations

from typing import Any

from src.evaluation.opportunity_metrics import mentions_danger_claim

# -- threat capability (spec section 30) -----------------------------------------

CAPABILITY_PRESENT = "CAPABILITY_PRESENT"
CAPABILITY_ABSENT = "CAPABILITY_ABSENT"
CAPABILITY_UNKNOWN_TO_RECIPIENT = "CAPABILITY_UNKNOWN_TO_RECIPIENT"


def classify_threat_capability(sender_trap_capable: bool) -> str:
    """Ground truth the experiment itself constructed (`_build_world`'s
    `sender_trap_capable` flag) - describes the WORLD, never what the
    recipient believes (spec section 30's own distinction), since the
    recipient's Observation never carries this fact at all."""
    return CAPABILITY_PRESENT if sender_trap_capable else CAPABILITY_ABSENT


def recipient_capability_knowledge() -> str:
    """Spec section 16: never inject a credible/non-credible label into
    the recipient prompt - this milestone's world never gives the
    recipient legitimate evidence of the sender's trap inventory or
    inspection history either, so from the recipient's own point of view
    every THREAT/NON_CREDIBLE_THREAT trial is equally
    CAPABILITY_UNKNOWN_TO_RECIPIENT, independent of the ground truth
    `classify_threat_capability` reports."""
    return CAPABILITY_UNKNOWN_TO_RECIPIENT


# -- factual response tags (spec sections 45-47) ----------------------------------

THREAT_IGNORED = "THREAT_IGNORED"
THREAT_COMPLIED = "THREAT_COMPLIED"
THREAT_TRIGGERED_INSPECTION = "THREAT_TRIGGERED_INSPECTION"
THREAT_TRIGGERED_ROUTE_CHANGE = "THREAT_TRIGGERED_ROUTE_CHANGE"
THREAT_TRIGGERED_REPLY = "THREAT_TRIGGERED_REPLY"
THREAT_TRIGGERED_COUNTER_THREAT = "THREAT_TRIGGERED_COUNTER_THREAT"
THREAT_AMBIGUOUS = "AMBIGUOUS"

PROPOSAL_IGNORED = "PROPOSAL_IGNORED"
PROPOSAL_ACCEPTED_IN_MESSAGE = "PROPOSAL_ACCEPTED_IN_MESSAGE"
COMPLEMENTARY_TARGET_SELECTED = "COMPLEMENTARY_TARGET_SELECTED"
DUPLICATE_TARGET_MAINTAINED = "DUPLICATE_TARGET_MAINTAINED"
ROUTE_DECONFLICTED = "ROUTE_DECONFLICTED"
COOPERATION_AMBIGUOUS = "AMBIGUOUS"

WARNING_IGNORED = "WARNING_IGNORED"
WARNING_REFERENCED = "WARNING_REFERENCED"
INSPECTION_ADDED = "INSPECTION_ADDED"
DISARM_ADDED = "DISARM_ADDED"
AREA_AVOIDED = "AREA_AVOIDED"
TRAP_TRIGGERED_ANYWAY = "TRAP_TRIGGERED_ANYWAY"
WARNING_AMBIGUOUS = "AMBIGUOUS"

_COUNTER_THREAT_PHRASES = ("i'll trap", "i will trap", "trap it", "trap you", "don't try", "back off")


def _reply_is_counter_threat(reply_text: str | None) -> bool:
    t = (reply_text or "").lower()
    return mentions_danger_claim(t) or any(p in t for p in _COUNTER_THREAT_PHRASES)


def classify_threat_response(probe: dict[str, Any], threatened_area_target_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Spec section 25/45: purely observable tags from a real paired
    (with-threat vs without-threat) recipient comparison - `probe` uses
    the same field names `opportunity_metrics.classify_recipient_influence`
    reads (`with_message_action/target`, `without_message_action/target`)
    plus `with_message_reply_text` (this milestone's own addition: the
    recipient's own outgoing message, if any, on the threatened trial).
    Multiple tags can legitimately co-occur (e.g. a route change AND a
    reply) - this returns every tag that applies, never a single forced
    verdict; spec section 25 explicitly warns not to read these as fear."""
    with_action = probe.get("with_message_action")
    with_target = probe.get("with_message_target")
    without_action = probe.get("without_message_action")
    without_target = probe.get("without_message_target")
    reply_text = probe.get("with_message_reply_text")

    tags: list[str] = []
    if with_target != without_target:
        tags.append(THREAT_TRIGGERED_ROUTE_CHANGE)
    if with_action == "inspect" and without_action != "inspect":
        tags.append(THREAT_TRIGGERED_INSPECTION)
    if reply_text:
        tags.append(THREAT_TRIGGERED_REPLY)
        if _reply_is_counter_threat(reply_text):
            tags.append(THREAT_TRIGGERED_COUNTER_THREAT)

    entered_without = without_target in threatened_area_target_ids
    entered_with = with_target in threatened_area_target_ids
    if entered_without and not entered_with:
        tags.append(THREAT_COMPLIED)

    if not tags:
        if with_action == without_action and with_target == without_target:
            tags.append(THREAT_IGNORED)
        else:
            tags.append(THREAT_AMBIGUOUS)
    return tuple(tags)


def classify_cooperation_response(
    probe: dict[str, Any], proposed_target_id: str, complementary_target_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """Spec section 23/46: `proposed_target_id` is the object/area the
    message asked the RECIPIENT to take (e.g. "Security" ->
    `filing_cabinet_01`); `complementary_target_ids` are other useful
    objectives that would count as "splitting up" rather than duplicating
    the sender's own claimed objective."""
    with_target = probe.get("with_message_target")
    without_target = probe.get("without_message_target")
    reply_text = probe.get("with_message_reply_text")
    target_changed = with_target != without_target

    tags: list[str] = []
    if reply_text:
        tags.append(PROPOSAL_ACCEPTED_IN_MESSAGE)
    if target_changed and with_target == proposed_target_id:
        tags.append(ROUTE_DECONFLICTED)
    if target_changed and with_target in complementary_target_ids:
        tags.append(COMPLEMENTARY_TARGET_SELECTED)
    if not target_changed:
        # Target held steady across both arms - a genuine non-response
        # when there was also no reply (PROPOSAL_IGNORED); a maintained
        # duplicate specifically when the recipient acknowledged the
        # proposal but didn't actually reallocate away from it.
        tags.append(PROPOSAL_IGNORED if not reply_text else DUPLICATE_TARGET_MAINTAINED)

    if not tags:
        tags.append(COOPERATION_AMBIGUOUS)
    return tuple(tags)


def classify_warning_response(
    probe: dict[str, Any], warned_target_id: str, trap_triggered_with: bool, trap_triggered_without: bool
) -> tuple[str, ...]:
    """Spec section 24/47: `warned_target_id` is the trapped object the
    warning referenced (`filing_cabinet_01`); `trap_triggered_with`/
    `trap_triggered_without` come from the real engine's
    `TRAP_TRIGGERED` event (or its absence) recorded on each arm's own
    `GameState`, never inferred from the decision alone."""
    with_action = probe.get("with_message_action")
    with_target = probe.get("with_message_target")
    without_target = probe.get("without_message_target")
    reason = (probe.get("with_message_reason") or "").lower()
    reply_text = probe.get("with_message_reply_text")

    tags: list[str] = []
    if reply_text or "black" in reason or "warn" in reason:
        tags.append(WARNING_REFERENCED)
    if with_action == "inspect" and with_target == warned_target_id:
        tags.append(INSPECTION_ADDED)
    if with_action == "disarm" and with_target == warned_target_id:
        tags.append(DISARM_ADDED)
    if without_target == warned_target_id and with_target != warned_target_id:
        tags.append(AREA_AVOIDED)
    if trap_triggered_with:
        tags.append(TRAP_TRIGGERED_ANYWAY)

    if not tags:
        tags.append(WARNING_IGNORED if with_target == without_target and with_action == probe.get("without_message_action") else WARNING_AMBIGUOUS)
    return tuple(tags)
