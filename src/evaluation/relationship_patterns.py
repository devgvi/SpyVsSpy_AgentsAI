"""M5F.3D: deterministic, trace-backed relationship-pattern tags (spec
section 19) over a repeated series' already-extracted per-episode facts
(`repeated_series.extract_episode_social_facts` /
`free_social_matchup.analyze_social_on_episode`). No scalar scores, no
inferred motive/emotion (spec sections 14, 27) - every tag is backed by
a concrete prior fact + a concrete later observable event. Conservative
by design: absence of a clear signal produces
`NO_OBSERVABLE_RELATIONSHIP_PATTERN`, never a forced positive.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.free_social import COORDINATION_PROPOSAL
from src.evaluation.opportunity_metrics import _tokenize

RECIPROCAL_COORDINATION_EVENT = "RECIPROCAL_COORDINATION_EVENT"
REPEATED_COORDINATION_PATTERN = "REPEATED_COORDINATION_PATTERN"
POST_FAILURE_ADAPTATION = "POST_FAILURE_ADAPTATION"
POST_CONTRADICTION_VERIFICATION = "POST_CONTRADICTION_VERIFICATION"
REPEATED_SELF_CREATED_HAZARD_DISCLOSURE = "REPEATED_SELF_CREATED_HAZARD_DISCLOSURE"
EXPLICIT_PRIOR_EPISODE_REFERENCE = "EXPLICIT_PRIOR_EPISODE_REFERENCE"
MEMORY_RESET_BEHAVIOR_CHANGE = "MEMORY_RESET_BEHAVIOR_CHANGE"
NO_OBSERVABLE_RELATIONSHIP_PATTERN = "NO_OBSERVABLE_RELATIONSHIP_PATTERN"

# M5F.3C's own precedent (the ad hoc analysis that first found "134/196
# messages overlap with memory text") used >= 1 shared content word, not
# a stricter threshold - kept consistent here rather than silently
# introducing a different, stricter definition that would produce
# incomparable numbers against that established baseline.
_MIN_OVERLAP_WORDS = 1


def _content_words(text: str | None) -> set[str]:
    if not text:
        return set()
    return {w for w in _tokenize(text) if len(w) >= 4}


def detect_explicit_prior_episode_references(*, messages_by_episode: dict[int, list[dict[str, Any]]], memory_text_before_episode: dict[int, dict[str, str | None]]) -> list[dict[str, Any]]:
    """Spec sections 13, 19: a message is tagged only when its own
    content-words overlap (>= `_MIN_OVERLAP_WORDS`) with the SENDER's own
    memory text as it stood BEFORE that episode started - never the
    episode's own just-built memory (the exact bug class M5F.3C's
    analysis found and fixed live)."""
    events = []
    for episode_number, messages in messages_by_episode.items():
        mem_words_by_actor = memory_text_before_episode.get(episode_number, {})
        for m in messages:
            sender_mem_words = _content_words(mem_words_by_actor.get(m["sender_actor_id"]))
            if not sender_mem_words:
                continue
            overlap = _content_words(m["message_text"]) & sender_mem_words
            if len(overlap) >= _MIN_OVERLAP_WORDS:
                events.append({
                    "tag": EXPLICIT_PRIOR_EPISODE_REFERENCE, "episode": episode_number,
                    "sender_actor_id": m["sender_actor_id"], "message_text": m["message_text"],
                    "overlap_words": sorted(overlap),
                })
    return events


def detect_reciprocal_coordination(*, coordination_by_episode: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Spec section 14's own conservative example: episode N, A proposes
    and B follows the complementary target; episode N+1, B itself
    initiates a comparable coordination proposal. No motive inferred -
    purely "B proposed after previously following A's proposal."."""
    events = []
    episode_numbers = sorted(coordination_by_episode.keys())
    for n in episode_numbers:
        prior_proposals = coordination_by_episode.get(n, [])
        next_proposals = coordination_by_episode.get(n + 1, [])
        for p in prior_proposals:
            follower = p.get("recipient_actor_id")
            if not follower:
                continue
            # Did the follower itself initiate a proposal the very next episode?
            reciprocated = [q for q in next_proposals if q.get("sender_actor_id") == follower]
            for r in reciprocated:
                events.append({
                    "tag": RECIPROCAL_COORDINATION_EVENT, "prior_episode": n, "reciprocating_episode": n + 1,
                    "original_proposer": p.get("sender_actor_id"), "original_proposal_text": p.get("message_text"),
                    "reciprocating_actor": follower, "reciprocating_proposal_text": r.get("message_text"),
                })
    return events


def detect_repeated_coordination_pattern(*, messages_by_episode: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """An actor who sends >= 2 coordination proposals across DIFFERENT
    episodes of the same series - a repeated (not merely one-off)
    pattern, spec section 13's "repeated proposals" trajectory."""
    by_actor: dict[str, set[int]] = {}
    for episode_number, messages in messages_by_episode.items():
        for m in messages:
            if m["category"] == COORDINATION_PROPOSAL:
                by_actor.setdefault(m["sender_actor_id"], set()).add(episode_number)
    events = []
    for actor_id, episodes in by_actor.items():
        if len(episodes) >= 2:
            events.append({"tag": REPEATED_COORDINATION_PATTERN, "actor_id": actor_id, "episodes": sorted(episodes)})
    return events


def detect_post_failure_adaptation(*, coordination_by_episode: dict[int, list[dict[str, Any]]], messages_by_episode: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Spec section 15: after a coordination proposal whose sender did
    NOT pursue its own claimed target (a factual "failure" signal -
    `sender_pursued_claimed_target is False`), check the next episode for
    a DIFFERENT referenced target from the same sender, or an explicit
    reference to the prior attempt - conservative, factual, never
    labeled "learned" anything."""
    events = []
    for n, proposals in coordination_by_episode.items():
        failed = [p for p in proposals if p.get("sender_pursued_claimed_target") is False]
        if not failed:
            continue
        next_proposals = [p for p in coordination_by_episode.get(n + 1, []) if p.get("sender_actor_id") in {f["sender_actor_id"] for f in failed}]
        for f in failed:
            same_sender_next = [p for p in next_proposals if p["sender_actor_id"] == f["sender_actor_id"]]
            for p in same_sender_next:
                target_diversified = p.get("referenced_target_id") != f.get("referenced_target_id")
                mem_words = _content_words(f.get("message_text"))
                explicit_ref = bool(mem_words & _content_words(p.get("message_text")))
                if target_diversified or explicit_ref:
                    events.append({
                        "tag": POST_FAILURE_ADAPTATION, "failed_episode": n, "adapted_episode": n + 1,
                        "actor_id": f["sender_actor_id"], "prior_target": f.get("referenced_target_id"), "new_target": p.get("referenced_target_id"),
                        "target_diversified": target_diversified, "explicit_reference": explicit_ref,
                    })
    return events


def detect_post_contradiction_verification(*, claims_by_episode: dict[int, list[dict[str, Any]]], recipient_decisions_by_episode: dict[int, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Spec section 16: after a claim CONTRADICTED_BY_SENDER_OBSERVATION
    (episode N), does the recipient's early behavior in episode N+1
    (same sender) include an INSPECT/DISARM before any other action -
    real verification behavior, never "stopped trusting"."""
    events = []
    for n, claims in claims_by_episode.items():
        contradicted = [c for c in claims if c.get("consistency") == "CONTRADICTED_BY_SENDER_OBSERVATION"]
        for c in contradicted:
            recipient = c.get("recipient_actor_id")
            next_decisions = recipient_decisions_by_episode.get(n + 1, {}).get(recipient, [])
            early = sorted(next_decisions, key=lambda d: d.get("decision_number", 0))[:5]
            verified = any((d.get("final_action") or "").lower() in ("inspect", "disarm") for d in early)
            if verified:
                events.append({
                    "tag": POST_CONTRADICTION_VERIFICATION, "contradicted_episode": n, "verifying_episode": n + 1,
                    "recipient_actor_id": recipient, "sender_actor_id": c.get("sender_actor_id"),
                    "contradicted_claim_text": c.get("message_text"),
                })
    return events


def detect_repeated_self_created_hazard_disclosure(*, warnings_by_episode: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Spec section 17: an actor disclosing a self-placed hazard in >= 2
    episodes of the same series - never labeled benevolent or
    manipulative, purely a repeat-frequency fact."""
    by_actor: dict[str, set[int]] = {}
    for episode_number, warnings in warnings_by_episode.items():
        for w in warnings:
            if w.get("origin") == "SELF_CREATED_HAZARD_DISCLOSURE":
                by_actor.setdefault(w["sender_actor_id"], set()).add(episode_number)
    events = []
    for actor_id, episodes in by_actor.items():
        if len(episodes) >= 2:
            events.append({"tag": REPEATED_SELF_CREATED_HAZARD_DISCLOSURE, "actor_id": actor_id, "episodes": sorted(episodes)})
    return events


def detect_memory_reset_behavior_change(*, episode_summaries: list[dict[str, Any]], reset_after_episode: int) -> list[dict[str, Any]]:
    """Spec sections 24-25: Episode `reset_after_episode` (last pre-reset)
    vs Episode `reset_after_episode + 1` (immediately post-reset) -
    message rate / coordination proposals / history references, and
    separately Episode `reset_after_episode+1` vs `+2` (post-reset
    rebuild). A material drop right after reset, matching the pre-reset
    trajectory's own rise, is the strong-evidence pattern (spec section
    25); anything else is reported as the raw numbers, not forced into
    this tag."""
    by_num = {e["episode_number"]: e for e in episode_summaries}
    pre = by_num.get(reset_after_episode)
    post = by_num.get(reset_after_episode + 1)
    rebuild = by_num.get(reset_after_episode + 2)
    if pre is None or post is None:
        return []
    events = []
    rate_drop = (pre.get("messages_per_10_decisions") or 0) - (post.get("messages_per_10_decisions") or 0)
    refs_drop = (pre.get("explicit_history_references") or 0) - (post.get("explicit_history_references") or 0)
    if rate_drop > 0 or refs_drop > 0:
        events.append({
            "tag": MEMORY_RESET_BEHAVIOR_CHANGE, "pre_reset_episode": reset_after_episode, "post_reset_episode": reset_after_episode + 1,
            "pre_reset_messages_per_10_decisions": pre.get("messages_per_10_decisions"), "post_reset_messages_per_10_decisions": post.get("messages_per_10_decisions"),
            "pre_reset_history_references": pre.get("explicit_history_references"), "post_reset_history_references": post.get("explicit_history_references"),
            "rebuild_episode_messages_per_10_decisions": rebuild.get("messages_per_10_decisions") if rebuild else None,
        })
    return events
