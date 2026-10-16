"""M5F.3C: `CrossEpisodeSocialMemory` - factual, bounded, deterministically
built social history that persists across episodes of a repeated series,
completely separate from within-episode `AgentMemory` (spec section 7).

No trust/reputation/credibility/fear score anywhere (spec section 1) -
only factual entries an LLM can reason over itself:

    "White proposed: 'I'll take Archives; you take Security.'"
    "White then moved toward Archives."
    "You moved toward Security."
    "White later placed a trap in Security."

never:

    "White is trustworthy."
    "White is deceptive."

Built entirely from already-recorded telemetry/Engine facts (spec
section 10) - no LLM ever summarizes or scores social history. Only
legally-observable-by-the-remembering-actor facts persist (spec section
4); hidden world state, private reasoning, and undetected opponent state
never do (spec section 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Spec section 8: bounded, deterministic - never grows without bound.
MAX_REMEMBERED_EPISODES = 3
MAX_EVENTS_PER_EPISODE = 8


@dataclass(frozen=True)
class SocialFact:
    """One persisted, factual entry - spec section 9's compact
    representation, one sentence each, never an evaluative label."""

    episode_number: int
    fact_type: str  # MESSAGE_SENT, COORDINATION_PROPOSAL, WARNING_OR_DISCLOSURE, TRAP_PLACED, FOLLOW_THROUGH, CLAIM_OUTCOME, EPISODE_OUTCOME
    text: str
    # Structured fields, kept alongside the text for jsonl/audit purposes
    # (spec section 53) - never re-derived by re-parsing `text`.
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossEpisodeSocialMemory:
    """One remembering actor's own bounded view of the opponent across a
    series so far. `remembering_actor_id`/`opponent_actor_id` are the
    stable per-series identity (spec section 14) - color/personality
    never changes mid-series."""

    remembering_actor_id: str
    opponent_actor_id: str
    prior_episode_count: int
    facts: tuple[SocialFact, ...]

    def render_text(self) -> str | None:
        """Spec section 9's compact narrative form - `None` (never an
        empty string) when there is nothing yet to report, so a transform
        can skip injecting the field entirely rather than injecting an
        empty one."""
        if not self.facts:
            return None
        lines = [
            f"Factual notes on your previous interactions with this same opponent, across {self.prior_episode_count} earlier episode(s) of this series (chronological, not an evaluation of the opponent):"
        ]
        current_episode = None
        for fact in self.facts:
            if fact.episode_number != current_episode:
                lines.append(f"Episode {fact.episode_number}:")
                current_episode = fact.episode_number
            lines.append(f"- {fact.text}")
        return "\n".join(lines)


def compact_facts(facts: list[SocialFact]) -> list[SocialFact]:
    """Spec sections 8, 33: deterministic compaction - most recent
    `MAX_REMEMBERED_EPISODES` episodes, then at most
    `MAX_EVENTS_PER_EPISODE` events per episode (most recent within that
    episode kept, chronological order preserved - spec section 32:
    chronology is semantic, never shuffled). No LLM involved in selection."""
    if not facts:
        return []
    ordered = sorted(facts, key=lambda f: f.episode_number)
    episode_numbers = sorted({f.episode_number for f in ordered})
    kept_episodes = set(episode_numbers[-MAX_REMEMBERED_EPISODES:])

    by_episode: dict[int, list[SocialFact]] = {}
    for f in ordered:
        if f.episode_number in kept_episodes:
            by_episode.setdefault(f.episode_number, []).append(f)

    compacted: list[SocialFact] = []
    for ep in sorted(by_episode):
        events = by_episode[ep][-MAX_EVENTS_PER_EPISODE:]
        compacted.extend(events)
    return compacted


def build_memory_from_episode_facts(
    *, remembering_actor_id: str, opponent_actor_id: str, episode_number: int, new_facts: list[SocialFact], prior_memory: CrossEpisodeSocialMemory | None,
) -> CrossEpisodeSocialMemory:
    """Appends one episode's worth of facts (already extracted - see
    `extract_episode_social_facts` in `repeated_series.py`) onto a prior
    memory object (or starts fresh at episode 1), then applies bounded
    deterministic compaction."""
    prior_facts = list(prior_memory.facts) if prior_memory is not None else []
    all_facts = prior_facts + list(new_facts)
    compacted = compact_facts(all_facts)
    prior_episode_count = (prior_memory.prior_episode_count if prior_memory is not None else 0) + 1
    return CrossEpisodeSocialMemory(
        remembering_actor_id=remembering_actor_id, opponent_actor_id=opponent_actor_id,
        prior_episode_count=prior_episode_count, facts=tuple(compacted),
    )


def model_payload_transform_for(memory: CrossEpisodeSocialMemory | None):
    """Spec sections 7, 11: the ONE hook this milestone actually injects
    through - `LLMControllerConfig.model_payload_transform` (M5F.3A.4/
    M5F.3B.1's own hook, unmodified), reused as-is. Returns `None` (no
    transform - MEMORY_OFF's exact byte-identical behavior) when there's
    no memory yet or nothing to report."""
    if memory is None:
        return None
    text = memory.render_text()
    if text is None:
        return None

    def transform(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["prior_episode_social_history"] = text
        return payload

    return transform
