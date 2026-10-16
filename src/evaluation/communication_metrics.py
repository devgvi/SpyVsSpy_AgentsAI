"""M5F.2: post-hoc communication analysis - joins each sent SEND_MESSAGE
decision (already fully captured by the same `llm_decision`/decision-trace
records every other milestone uses - no new live telemetry event needed,
see `decision_trace.build_decision_trace_records`'s new `final_message`
field) with the recipient's own subsequent decisions, to classify what (if
anything) observably happened after the message.

Rule-based only (spec sections 46, 48-49): every classification here comes
from already-recorded facts (message text, action/target changes, engine
lifecycle results) - never a second LLM call, never a synthesized score.
"""

from __future__ import annotations

from typing import Any

from src import constants

SEND_MESSAGE_ACTION = "send_message"

# -- outcome vocabulary (spec section 46) -------------------------------------

NO_OBSERVABLE_RESPONSE = "NO_OBSERVABLE_RESPONSE"
EXPLICITLY_REFERENCED = "EXPLICITLY_REFERENCED"
BEHAVIOR_CHANGED = "BEHAVIOR_CHANGED"
COORDINATION_FOLLOWUP = "COORDINATION_FOLLOWUP"
AMBIGUOUS = "AMBIGUOUS"

_STOPWORDS = frozenset(
    "the a an to and is in on at i you will am be it this that for of my your will i'll ill "
    "so as with was were are do does did not no yes near going head heading".split()
)
_COORDINATION_KEYWORDS = ("agreed", "deal", "i'll", "you take", "you go", "i will", "ok", "sounds good")
_SAFETY_ACTIONS = frozenset({"inspect", "disarm"})


def _content_words(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        w.strip(".,!?\"'()").lower()
        for w in text.split()
        if len(w.strip(".,!?\"'()")) >= 4 and w.strip(".,!?\"'()").lower() not in _STOPWORDS
    }


def _sender_label(sender_actor_id: str) -> str:
    return "black" if sender_actor_id == constants.BLACK_SPY_ID else "white"


# -- extraction (from already-built decision-trace records) ------------------


def extract_sent_messages(records: list[dict[str, Any]], sender_profile: str | None = None) -> list[dict[str, Any]]:
    """Pulls every valid SEND_MESSAGE decision out of one actor's own
    `build_decision_trace_records`/`build_actor_trace_records` output - no
    new data, purely a filter + reshape into spec section 22's message-
    level shape (minus the recipient-reaction fields, filled in by
    `classify_message_outcome` once the recipient's own records are
    available)."""
    messages = []
    for r in records:
        if r.get("final_action") != SEND_MESSAGE_ACTION or not r.get("final_action_valid"):
            continue
        messages.append({
            "message_id": r.get("decision_id"),  # one message per decision - see agent_memory.MessageRecord's docstring
            "game_time": r.get("game_time"),
            "sender_actor_id": r.get("actor_id"),
            "sender_profile": sender_profile,
            "recipient_actor_id": r.get("final_target"),
            "message_text": r.get("final_message"),
            "sender_reason": r.get("agent_stated_reason"),
            "sender_decision_id": r.get("decision_id"),
        })
    return messages


# -- outcome classification (spec sections 24-25, 46) -------------------------


def _next_recipient_decisions(recipient_records: list[dict[str, Any]], after_game_time: float, window: int) -> list[dict[str, Any]]:
    candidates = sorted(
        (r for r in recipient_records if (r.get("game_time") or 0.0) > after_game_time),
        key=lambda r: r.get("decision_number") or 0,
    )
    return candidates[:window]


def _prior_recipient_decision(recipient_records: list[dict[str, Any]], at_or_before_game_time: float) -> dict[str, Any] | None:
    prior = sorted(
        (r for r in recipient_records if (r.get("game_time") or 0.0) <= at_or_before_game_time),
        key=lambda r: r.get("decision_number") or 0,
    )
    return prior[-1] if prior else None


def classify_message_outcome(message: dict[str, Any], recipient_records: list[dict[str, Any]], window: int = 3) -> dict[str, Any]:
    """Spec section 24: inspects the recipient's next 1-3 decisions after
    `message`'s `game_time`. Spec section 25: "explicit reference" is
    detected only from real overlap between the message text and the
    recipient's own stated rationale (or the recipient naming the sender by
    color) - never inferred hidden reasoning. When there is no next
    recipient decision at all, the honest answer is `NO_OBSERVABLE_RESPONSE`
    with every recipient_* field left `None` (spec section 50: null results
    reported honestly, never papered over)."""
    game_time = message.get("game_time")
    if game_time is None:
        next_decisions: list[dict[str, Any]] = []
    else:
        next_decisions = _next_recipient_decisions(recipient_records, game_time, window)

    result: dict[str, Any] = {
        "recipient_next_decision_id": None,
        "recipient_next_action": None,
        "recipient_next_target": None,
        "recipient_reason": None,
        "behavior_change_detected": False,
        "behavior_change_type": None,
        "primary_outcome": NO_OBSERVABLE_RESPONSE,
    }
    if not next_decisions:
        return result

    first = next_decisions[0]
    result["recipient_next_decision_id"] = first.get("decision_id")
    result["recipient_next_action"] = first.get("final_action")
    result["recipient_next_target"] = first.get("final_target")
    result["recipient_reason"] = first.get("agent_stated_reason")

    reason_text = (first.get("agent_stated_reason") or "").lower()
    message_words = _content_words(message.get("message_text"))
    reason_words = _content_words(reason_text)
    sender_label = _sender_label(message.get("sender_actor_id") or "")
    explicitly_referenced = sender_label in reason_text or bool(message_words & reason_words)

    prior = _prior_recipient_decision(recipient_records, game_time) if game_time is not None else None
    prior_target = prior.get("final_target") if prior else None
    target_changed = any(
        d.get("final_target") is not None and d.get("final_target") != prior_target for d in next_decisions
    )
    safety_action = any((d.get("final_action") or "") in _SAFETY_ACTIONS for d in next_decisions)
    trap_action = any((d.get("final_action") or "") == "place_trap" for d in next_decisions)
    behavior_change = target_changed or safety_action or trap_action

    result["behavior_change_detected"] = behavior_change
    if behavior_change:
        if safety_action:
            result["behavior_change_type"] = "safety_action"
        elif trap_action:
            result["behavior_change_type"] = "trap_action"
        elif target_changed:
            result["behavior_change_type"] = "target_changed"

    coordination_signal = any(k in (message.get("message_text") or "").lower() for k in _COORDINATION_KEYWORDS)

    if explicitly_referenced:
        result["primary_outcome"] = EXPLICITLY_REFERENCED
    elif coordination_signal and behavior_change:
        result["primary_outcome"] = COORDINATION_FOLLOWUP
    elif behavior_change:
        result["primary_outcome"] = BEHAVIOR_CHANGED
    elif reason_text:
        # The recipient did decide and gave a real rationale, but nothing
        # ties it back to this message and nothing observably changed -
        # genuinely unclear rather than a confident "no response" (spec
        # section 24: "if evidence is unclear, influence = unclear").
        result["primary_outcome"] = AMBIGUOUS
    return result


def build_message_records(
    black_records: list[dict[str, Any]],
    white_records: list[dict[str, Any]],
    black_profile: str | None = None,
    white_profile: str | None = None,
    window: int = 3,
) -> list[dict[str, Any]]:
    """Spec section 22's full per-message record for every SEND_MESSAGE
    decision either actor made this duel, sorted by `game_time`."""
    sent_black = extract_sent_messages(black_records, black_profile)
    sent_white = extract_sent_messages(white_records, white_profile)

    records: list[dict[str, Any]] = []
    for m in sent_black:
        recipient_records = white_records if m["recipient_actor_id"] == constants.WHITE_SPY_ID else black_records
        outcome = classify_message_outcome(m, recipient_records, window)
        records.append({**m, **outcome})
    for m in sent_white:
        recipient_records = black_records if m["recipient_actor_id"] == constants.BLACK_SPY_ID else white_records
        outcome = classify_message_outcome(m, recipient_records, window)
        records.append({**m, **outcome})

    records.sort(key=lambda r: r.get("game_time") if r.get("game_time") is not None else float("inf"))
    return records


# -- frequency metrics (spec section 30) --------------------------------------


def communication_frequency_metrics(records: list[dict[str, Any]], message_records: list[dict[str, Any]], actor_id: str) -> dict[str, Any]:
    """Per-actor counts, computed from `records` (this actor's own full
    decision trace, to get `decisions_spent_messaging`/`messages_per_10_decisions`
    right) and `message_records` (the duel-wide joined message list, to
    count both directions)."""
    total_decisions = len(records)
    sent = [m for m in message_records if m["sender_actor_id"] == actor_id]
    received = [m for m in message_records if m["recipient_actor_id"] == actor_id]
    decisions_spent_messaging = sum(1 for r in records if r.get("final_action") == SEND_MESSAGE_ACTION and r.get("final_action_valid"))
    lengths = [len(m.get("message_text") or "") for m in sent]
    return {
        "messages_sent": len(sent),
        "messages_received": len(received),
        "messages_per_10_decisions": round(len(sent) / total_decisions * 10, 3) if total_decisions else None,
        "average_message_length": round(sum(lengths) / len(lengths), 1) if lengths else None,
        "decisions_spent_messaging": decisions_spent_messaging,
    }


# -- response metrics (spec section 31) ---------------------------------------


def response_metrics(message_records: list[dict[str, Any]], actor_id: str) -> dict[str, Any]:
    """Spec section 31: response facts about messages `actor_id` RECEIVED
    (i.e. this actor as recipient) - deliberately the mirror of
    `communication_frequency_metrics`'s sent-side counts, never conflated
    with them."""
    received = [m for m in message_records if m["recipient_actor_id"] == actor_id]
    return {
        "received_messages": len(received),
        "explicitly_referenced": sum(1 for m in received if m["primary_outcome"] == EXPLICITLY_REFERENCED),
        "followed_by_target_change": sum(1 for m in received if m.get("behavior_change_type") == "target_changed"),
        "followed_by_route_change": sum(
            1 for m in received if m.get("behavior_change_type") == "target_changed" and m.get("recipient_next_action") == "move"
        ),
        "followed_by_safety_action": sum(1 for m in received if m.get("behavior_change_type") == "safety_action"),
        "followed_by_trap_action": sum(1 for m in received if m.get("behavior_change_type") == "trap_action"),
        "followed_by_no_observable_change": sum(1 for m in received if m["primary_outcome"] == NO_OBSERVABLE_RESPONSE),
    }


# -- cooperation / coordination events (spec section 32) -----------------------


def cooperation_events(message_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Purely descriptive, factual events (spec section 32: "do not create
    a cooperation score") - each entry names what was observed and which
    message/decision it came from, never a numeric rating."""
    events: list[dict[str, Any]] = []
    for m in message_records:
        text = (m.get("message_text") or "").lower()
        is_proposal = any(k in text for k in _COORDINATION_KEYWORDS) or "warn" in text or "trap" in text or "safe" in text
        if is_proposal:
            events.append({
                "event": "coordination_proposal" if any(k in text for k in _COORDINATION_KEYWORDS) else "trap_or_safety_warning",
                "message_id": m["message_id"], "sender_actor_id": m["sender_actor_id"],
                "recipient_actor_id": m["recipient_actor_id"], "game_time": m.get("game_time"),
            })
        if m["primary_outcome"] == COORDINATION_FOLLOWUP:
            events.append({
                "event": "apparent_acceptance_compatible_followup",
                "message_id": m["message_id"], "sender_actor_id": m["sender_actor_id"],
                "recipient_actor_id": m["recipient_actor_id"], "game_time": m.get("game_time"),
            })
        if ("trap" in text or "warn" in text or "safe" in text) and m.get("behavior_change_type") == "safety_action":
            events.append({
                "event": "trap_warning_acted_upon",
                "message_id": m["message_id"], "sender_actor_id": m["sender_actor_id"],
                "recipient_actor_id": m["recipient_actor_id"], "game_time": m.get("game_time"),
            })
    return events


# -- message cost (spec section 35) --------------------------------------------


def message_cost_metrics(records: list[dict[str, Any]], actor_id: str) -> dict[str, Any]:
    """Spec section 35's two reported facts. `mission_progress_decisions_displaced`
    is deliberately reported as an approximation, not a causal claim: each
    messaging decision consumes one slot of this actor's fixed decision
    budget that could otherwise have been spent on a mission-progress
    attempt - it is not evidence that the actor *would have* made progress
    with that slot instead."""
    decisions_spent_messaging = sum(1 for r in records if r.get("final_action") == SEND_MESSAGE_ACTION and r.get("final_action_valid"))
    return {
        "decisions_spent_communicating": decisions_spent_messaging,
        "mission_progress_decisions_displaced_approx": decisions_spent_messaging,
    }


# -- narrative (spec section 47) ------------------------------------------------


def render_communication_narrative(message_records: list[dict[str, Any]]) -> str:
    """Spec section 47's example format - one block per message, stating
    only what the trace supports (spec section 47: "do not state causal
    certainty beyond what the trace supports")."""
    lines = []
    for m in message_records:
        sender_label = "Black" if m["sender_actor_id"] == constants.BLACK_SPY_ID else "White"
        recipient_label = "White" if m["recipient_actor_id"] == constants.WHITE_SPY_ID else "Black"
        t = m.get("game_time")
        lines.append(f"T={t:.1f}" if t is not None else "T=?")
        lines.append(f'{sender_label} -> {recipient_label}: "{m.get("message_text")}"')
        if m.get("recipient_reason"):
            lines.append(f'{recipient_label} rationale: "{m["recipient_reason"]}"')
        if m.get("recipient_next_action"):
            target = m.get("recipient_next_target") or ""
            lines.append(f"{recipient_label}: {m['recipient_next_action']} {target}".rstrip())
        lines.append(f"Outcome: {m['primary_outcome']}")
        lines.append("")
    return "\n".join(lines)
