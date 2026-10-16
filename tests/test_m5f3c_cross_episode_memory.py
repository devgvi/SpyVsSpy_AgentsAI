"""M5F.3C Phase A (spec section 34): cross-episode memory default-off,
world state resets, actor identity persists, only observable facts
persist, hidden facts excluded, claim outcomes computed correctly,
self-created hazard disclosure classification, memory bounded,
chronology preserved, no trust/reputation labels, secret safety. Real
`GameState`/`Game`/`LLMController` throughout - no network calls."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src import constants
from src.evaluation.cross_episode_memory import (
    MAX_EVENTS_PER_EPISODE,
    MAX_REMEMBERED_EPISODES,
    CrossEpisodeSocialMemory,
    SocialFact,
    build_memory_from_episode_facts,
    compact_facts,
    model_payload_transform_for,
)
from src.evaluation.free_social import (
    CONFIRMED_BY_LATER_OBSERVATION,
    CONTRADICTED_BY_LATER_OBSERVATION,
    ENVIRONMENTAL_HAZARD_WARNING,
    FACTUAL_CLAIM,
    NEVER_RESOLVED,
    NOT_FACT_CHECKABLE,
    OPPONENT_HAZARD_WARNING,
    SAFETY_WARNING,
    SELF_CREATED_HAZARD_DISCLOSURE,
    UNVERIFIED_WARNING,
    classify_warning_origin,
    resolve_claim_outcome,
)
from src.evaluation.repeated_series import episode_outcome_facts, run_series
from src.llm.clients.mock_client import MockModelClient


# -- cross-episode memory default-off (spec section 11/17-STOP-2) --------


def test_memory_off_never_produces_a_transform():
    assert model_payload_transform_for(None) is None


def test_memory_off_series_never_injects_prior_episode_social_history():
    result = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1, 2], memory_on=False,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3c_off.jsonl", llm_timeout=10.0,
    )
    for e in result["episodes"]:
        assert e["black_memory_snapshot_text"] is None
        assert e["white_memory_snapshot_text"] is None
        assert e["black_memory_size_chars"] == 0


# -- world state resets / actor identity persists (spec sections 3, 14) --


def test_world_state_resets_between_episodes_different_seeds_different_worlds():
    result = run_series(
        black_personality="opportunist", white_personality="opportunist", seeds=[10, 20], memory_on=False,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3c_reset.jsonl", llm_timeout=10.0,
    )
    assert result["episodes"][0]["seed"] != result["episodes"][1]["seed"]
    assert result["episodes"][0]["episode_id"] != result["episodes"][1]["episode_id"]


def test_actor_identity_black_white_personality_stable_across_series():
    result = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1, 2, 3], memory_on=True,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3c_identity.jsonl", llm_timeout=10.0,
    )
    assert result["black_personality"] == "cautious"
    assert result["white_personality"] == "saboteur"
    assert len(result["episodes"]) == 3


# -- memory bounded / chronology preserved (spec sections 8, 32-33) ------


def test_compact_facts_bounds_episode_count():
    facts = [SocialFact(episode_number=n, fact_type="MESSAGE_SENT", text=f"ep{n}") for n in range(1, 6)]
    compacted = compact_facts(facts)
    episode_numbers = {f.episode_number for f in compacted}
    assert len(episode_numbers) <= MAX_REMEMBERED_EPISODES
    assert episode_numbers == {3, 4, 5}  # most recent 3 of 5


def test_compact_facts_bounds_events_per_episode():
    facts = [SocialFact(episode_number=1, fact_type="MESSAGE_SENT", text=f"e{n}") for n in range(20)]
    compacted = compact_facts(facts)
    assert len(compacted) == MAX_EVENTS_PER_EPISODE
    # most recent events kept, chronological order preserved
    assert [f.text for f in compacted] == [f"e{n}" for n in range(20 - MAX_EVENTS_PER_EPISODE, 20)]


def test_compact_facts_never_shuffles_chronology():
    facts = [SocialFact(episode_number=1, fact_type="MESSAGE_SENT", text=f"e{n}") for n in range(3)]
    compacted = compact_facts(facts)
    assert [f.text for f in compacted] == ["e0", "e1", "e2"]


def test_memory_grows_deterministically_across_episodes():
    mem1 = build_memory_from_episode_facts(
        remembering_actor_id=constants.BLACK_SPY_ID, opponent_actor_id=constants.WHITE_SPY_ID,
        episode_number=1, new_facts=[SocialFact(episode_number=1, fact_type="MESSAGE_SENT", text="a")], prior_memory=None,
    )
    assert mem1.prior_episode_count == 1
    mem2 = build_memory_from_episode_facts(
        remembering_actor_id=constants.BLACK_SPY_ID, opponent_actor_id=constants.WHITE_SPY_ID,
        episode_number=2, new_facts=[SocialFact(episode_number=2, fact_type="MESSAGE_SENT", text="b")], prior_memory=mem1,
    )
    assert mem2.prior_episode_count == 2
    assert len(mem2.facts) == 2


# -- no trust/reputation labels (spec sections 9, 27, STOP-9) -------------


def test_rendered_memory_text_never_contains_evaluative_labels():
    facts = [
        SocialFact(episode_number=1, fact_type="MESSAGE_SENT", text='white_spy proposed: "I\'ll take Archives; you take Security."'),
        SocialFact(episode_number=1, fact_type="FOLLOW_THROUGH", text="white_spy then moved toward the target it proposed for itself."),
    ]
    memory = CrossEpisodeSocialMemory(remembering_actor_id="black_spy", opponent_actor_id="white_spy", prior_episode_count=1, facts=tuple(facts))
    text = memory.render_text()
    forbidden = ("trustworthy", "trust score", "reputation", "credibility", "dishonest", "reliable", "unreliable", "deceptive")
    for word in forbidden:
        assert word not in text.lower()


def test_episode_outcome_fact_never_contains_evaluative_labels():
    fact = episode_outcome_facts(episode_number=1, remembering_actor_id=constants.BLACK_SPY_ID, completed_mission=False, escaped=False)
    forbidden = ("trustworthy", "reputation", "credibility", "reliable")
    for word in forbidden:
        assert word not in fact.text.lower()


def test_empty_memory_renders_none_not_empty_string():
    memory = CrossEpisodeSocialMemory(remembering_actor_id="black_spy", opponent_actor_id="white_spy", prior_episode_count=0, facts=())
    assert memory.render_text() is None


# -- warning taxonomy: self-created hazard disclosure (spec sections 16-17) --


def test_self_created_hazard_disclosure_classification():
    origin = classify_warning_origin(trap_owner_id=constants.BLACK_SPY_ID, sender_id=constants.BLACK_SPY_ID, recipient_id=constants.WHITE_SPY_ID)
    assert origin == SELF_CREATED_HAZARD_DISCLOSURE


def test_environmental_hazard_warning_classification():
    origin = classify_warning_origin(trap_owner_id="environment", sender_id=constants.BLACK_SPY_ID, recipient_id=constants.WHITE_SPY_ID)
    assert origin == ENVIRONMENTAL_HAZARD_WARNING


def test_opponent_hazard_warning_classification():
    origin = classify_warning_origin(trap_owner_id=constants.WHITE_SPY_ID, sender_id=constants.BLACK_SPY_ID, recipient_id=constants.WHITE_SPY_ID)
    assert origin == OPPONENT_HAZARD_WARNING


def test_unverified_warning_when_owner_unknown():
    origin = classify_warning_origin(trap_owner_id=None, sender_id=constants.BLACK_SPY_ID, recipient_id=constants.WHITE_SPY_ID)
    assert origin == UNVERIFIED_WARNING


# -- claim outcomes computed correctly (spec sections 28, STOP-15) -------


def test_claim_outcome_confirmed_when_safe_claim_and_no_trap_found():
    outcome = resolve_claim_outcome(message_category=FACTUAL_CLAIM, message_claims_safe=True, message_claims_danger=False, later_trap_statuses=["unknown", "clear_after_inspection"])
    assert outcome == CONFIRMED_BY_LATER_OBSERVATION


def test_claim_outcome_contradicted_when_safe_claim_but_trap_detected():
    outcome = resolve_claim_outcome(message_category=SAFETY_WARNING, message_claims_safe=True, message_claims_danger=False, later_trap_statuses=["detected"])
    assert outcome == CONTRADICTED_BY_LATER_OBSERVATION


def test_claim_outcome_contradicted_when_danger_claim_but_no_trap_found():
    outcome = resolve_claim_outcome(message_category=SAFETY_WARNING, message_claims_safe=False, message_claims_danger=True, later_trap_statuses=["clear_after_inspection"])
    assert outcome == CONTRADICTED_BY_LATER_OBSERVATION


def test_claim_outcome_confirmed_when_danger_claim_and_trap_detected():
    outcome = resolve_claim_outcome(message_category=SAFETY_WARNING, message_claims_safe=False, message_claims_danger=True, later_trap_statuses=["detected"])
    assert outcome == CONFIRMED_BY_LATER_OBSERVATION


def test_claim_outcome_never_resolved_when_no_resolving_observation():
    outcome = resolve_claim_outcome(message_category=FACTUAL_CLAIM, message_claims_safe=True, message_claims_danger=False, later_trap_statuses=["unknown", "unknown"])
    assert outcome == NEVER_RESOLVED


def test_claim_outcome_never_equates_unresolved_with_false():
    # A NEVER_RESOLVED claim must not be conflated with a contradicted one.
    outcome = resolve_claim_outcome(message_category=FACTUAL_CLAIM, message_claims_safe=True, message_claims_danger=False, later_trap_statuses=[])
    assert outcome == NEVER_RESOLVED
    assert outcome != CONTRADICTED_BY_LATER_OBSERVATION


def test_claim_outcome_not_fact_checkable_for_non_claim_category():
    outcome = resolve_claim_outcome(message_category="UNCLASSIFIED", message_claims_safe=False, message_claims_danger=False, later_trap_statuses=["detected"])
    assert outcome == NOT_FACT_CHECKABLE


def test_claim_outcome_uses_first_resolving_observation_not_last():
    outcome = resolve_claim_outcome(message_category=FACTUAL_CLAIM, message_claims_safe=True, message_claims_danger=False, later_trap_statuses=["detected", "clear_after_inspection"])
    assert outcome == CONTRADICTED_BY_LATER_OBSERVATION  # first resolving = "detected"


# -- secret safety (spec section 60) ---------------------------------------


def test_no_secret_like_patterns_in_memory_module_source():
    import inspect

    for mod_name in ("src.evaluation.cross_episode_memory", "src.evaluation.repeated_series"):
        source = inspect.getsource(__import__(mod_name, fromlist=[""]))
        for forbidden in ("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key"):
            assert forbidden not in source
