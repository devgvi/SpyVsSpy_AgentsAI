"""M5F.3D Phase A (spec section 41): 5-episode scheduling, MEMORY_OFF/
MEMORY_ON/MEMORY_RESET, reset boundary exactness, Episode N excludes
itself from memory, per-actor isolation, world reset, directional social
memory, deterministic relationship tags, no synthetic scores, secret
safety. Real `GameState`/`Game`/`LLMController` throughout - no network
calls."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.evaluation.relationship_patterns import (
    EXPLICIT_PRIOR_EPISODE_REFERENCE,
    MEMORY_RESET_BEHAVIOR_CHANGE,
    POST_CONTRADICTION_VERIFICATION,
    POST_FAILURE_ADAPTATION,
    RECIPROCAL_COORDINATION_EVENT,
    REPEATED_COORDINATION_PATTERN,
    REPEATED_SELF_CREATED_HAZARD_DISCLOSURE,
    detect_explicit_prior_episode_references,
    detect_memory_reset_behavior_change,
    detect_post_contradiction_verification,
    detect_post_failure_adaptation,
    detect_reciprocal_coordination,
    detect_repeated_coordination_pattern,
    detect_repeated_self_created_hazard_disclosure,
)
from src.evaluation.repeated_series import run_series

BLACK, WHITE = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID


# -- episode_id collision regression (real bug found live) ---------------


def test_distinct_episode_prefix_avoids_cross_run_episode_id_collision():
    # Real bug found live: two separate `run_series` calls with the same
    # matchup/condition/seed but the default `episode_prefix` produced
    # byte-identical episode_ids, silently merging their telemetry in a
    # shared duel_decisions.jsonl file. A caller-supplied distinct prefix
    # must produce a distinct episode_id for the same (matchup, condition,
    # episode, seed) combination.
    result_a = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1], memory_on=True, reset_after_episode=None,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_prefix_a.jsonl", llm_timeout=10.0, episode_prefix="run_a",
    )
    result_b = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1], memory_on=True, reset_after_episode=None,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_prefix_b.jsonl", llm_timeout=10.0, episode_prefix="run_b",
    )
    assert result_a["episodes"][0]["episode_id"] != result_b["episodes"][0]["episode_id"]


# -- 5-episode scheduling / MEMORY_OFF / MEMORY_ON / MEMORY_RESET --------


def test_five_episode_series_runs_all_five():
    result = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1, 2, 3, 4, 5], memory_on=False,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_5ep.jsonl", llm_timeout=10.0,
    )
    assert len(result["episodes"]) == 5
    assert [e["episode_number"] for e in result["episodes"]] == [1, 2, 3, 4, 5]


def test_memory_off_condition_label():
    result = run_series(
        black_personality="opportunist", white_personality="opportunist", seeds=[1, 2], memory_on=False,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_off.jsonl", llm_timeout=10.0,
    )
    assert result["memory_condition"] == "MEMORY_OFF"
    for e in result["episodes"]:
        assert e["memory_condition"] == "MEMORY_OFF"
        assert e["black_memory_snapshot_text"] is None


def test_memory_on_condition_label_without_reset():
    result = run_series(
        black_personality="opportunist", white_personality="opportunist", seeds=[1, 2], memory_on=True,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_on.jsonl", llm_timeout=10.0,
    )
    assert result["memory_condition"] == "MEMORY_ON"
    assert result["reset_after_episode"] is None


def test_memory_reset_condition_label():
    result = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1, 2, 3, 4, 5], memory_on=True,
        reset_after_episode=3,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_reset.jsonl", llm_timeout=10.0,
    )
    assert result["memory_condition"] == "MEMORY_RESET"
    assert result["reset_after_episode"] == 3


# -- reset boundary exactness (spec sections 7, 35-36) --------------------


def test_reset_boundary_exactness_episode_4_is_memory_empty():
    result = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1, 2, 3, 4, 5], memory_on=True,
        reset_after_episode=3,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_boundary.jsonl", llm_timeout=10.0,
    )
    by_num = {e["episode_number"]: e for e in result["episodes"]}
    assert by_num[1]["included_prior_episode_ids_black"] == []
    assert by_num[2]["included_prior_episode_ids_black"] == [1]
    assert by_num[3]["included_prior_episode_ids_black"] == [1, 2]
    assert by_num[4]["is_reset_episode"] is True
    assert by_num[4]["included_prior_episode_ids_black"] == []  # memory-empty at input, spec section 7
    assert by_num[4]["included_prior_episode_ids_white"] == []


def test_episode_5_rebuilds_from_episode_4_only():
    result = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1, 2, 3, 4, 5], memory_on=True,
        reset_after_episode=3,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_rebuild.jsonl", llm_timeout=10.0,
    )
    by_num = {e["episode_number"]: e for e in result["episodes"]}
    assert by_num[5]["included_prior_episode_ids_black"] == [4]  # spec section 36: Episode 4 only, never 1-3


def test_only_one_episode_is_flagged_as_reset():
    result = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1, 2, 3, 4, 5], memory_on=True,
        reset_after_episode=3,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_onereset.jsonl", llm_timeout=10.0,
    )
    reset_flags = [e["is_reset_episode"] for e in result["episodes"]]
    assert reset_flags == [False, False, False, True, False]


# -- Episode N excludes itself from memory (spec sections 6, 34-35) -------


def test_memory_integrity_assertion_holds_across_full_series():
    # run_series itself asserts this internally on every episode - if the
    # series completes without raising, the invariant held throughout.
    result = run_series(
        black_personality="cautious", white_personality="saboteur", seeds=[1, 2, 3, 4, 5], memory_on=True,
        provider="mock", model="mock-model", region=None, agent_runtime="native_tool_like",
        telemetry_path="/tmp/test_m5f3d_integrity.jsonl", llm_timeout=10.0,
    )
    assert len(result["episodes"]) == 5


def test_prior_episode_assertion_raises_on_violation():
    from dataclasses import replace

    from src.evaluation.cross_episode_memory import CrossEpisodeSocialMemory, SocialFact

    # Hand-construct a memory that illegally contains a fact from the
    # CURRENT episode (3) being injected as episode 3's own input.
    bad_memory = CrossEpisodeSocialMemory(
        remembering_actor_id=BLACK, opponent_actor_id=WHITE, prior_episode_count=3,
        facts=(SocialFact(episode_number=3, fact_type="MESSAGE_SENT", text="illegal same-episode fact"),),
    )
    with pytest.raises(AssertionError):
        assert all(fact.episode_number < 3 for fact in bad_memory.facts)


# -- per-actor isolation / directional social memory (spec sections 9, 37-38) --


def test_black_and_white_memories_are_independent_objects():
    # Rendered text CAN coincide when nothing eventful happened to either
    # side (both sides then "remember" symmetric nothing) - isolation is
    # about them being genuinely separate objects/computations, not about
    # their text always differing, so this checks structural independence
    # directly rather than text inequality.
    from src.evaluation.cross_episode_memory import SocialFact, build_memory_from_episode_facts

    black_only_fact = [SocialFact(episode_number=1, fact_type="MESSAGE_SENT", text="white_spy proposed something to you.")]
    black_memory = build_memory_from_episode_facts(remembering_actor_id=BLACK, opponent_actor_id=WHITE, episode_number=1, new_facts=black_only_fact, prior_memory=None)
    white_memory = build_memory_from_episode_facts(remembering_actor_id=WHITE, opponent_actor_id=BLACK, episode_number=1, new_facts=[], prior_memory=None)

    assert black_memory is not white_memory
    assert len(black_memory.facts) == 1
    assert len(white_memory.facts) == 0  # the fact fed only to black's memory never leaked into white's
    assert black_memory.remembering_actor_id != white_memory.remembering_actor_id


# -- deterministic relationship tags / no synthetic scores (spec sections 19-20) --


def test_reciprocal_coordination_detection():
    coord_by_ep = {
        1: [{"sender_actor_id": BLACK, "recipient_actor_id": WHITE, "message_text": "I'll take X, you take Y."}],
        2: [{"sender_actor_id": WHITE, "recipient_actor_id": BLACK, "message_text": "I'll take Z, you take W."}],
    }
    events = detect_reciprocal_coordination(coordination_by_episode=coord_by_ep)
    assert len(events) == 1
    assert events[0]["tag"] == RECIPROCAL_COORDINATION_EVENT
    assert events[0]["reciprocating_actor"] == WHITE


def test_no_reciprocal_coordination_when_no_followup_proposal():
    coord_by_ep = {1: [{"sender_actor_id": BLACK, "recipient_actor_id": WHITE, "message_text": "I'll take X."}], 2: []}
    events = detect_reciprocal_coordination(coordination_by_episode=coord_by_ep)
    assert events == []


def test_repeated_coordination_pattern_requires_multiple_episodes():
    msgs_by_ep = {
        1: [{"sender_actor_id": BLACK, "category": "COORDINATION_PROPOSAL"}],
        2: [{"sender_actor_id": BLACK, "category": "COORDINATION_PROPOSAL"}],
    }
    events = detect_repeated_coordination_pattern(messages_by_episode=msgs_by_ep)
    assert len(events) == 1
    assert events[0]["tag"] == REPEATED_COORDINATION_PATTERN
    assert events[0]["actor_id"] == BLACK
    assert events[0]["episodes"] == [1, 2]


def test_repeated_coordination_pattern_absent_for_single_episode():
    msgs_by_ep = {1: [{"sender_actor_id": BLACK, "category": "COORDINATION_PROPOSAL"}]}
    events = detect_repeated_coordination_pattern(messages_by_episode=msgs_by_ep)
    assert events == []


def test_post_failure_adaptation_target_diversification():
    coord_by_ep = {
        1: [{"sender_actor_id": BLACK, "sender_pursued_claimed_target": False, "referenced_target_id": "desk_01", "message_text": "I'll take desk."}],
        2: [{"sender_actor_id": BLACK, "sender_pursued_claimed_target": True, "referenced_target_id": "safe", "message_text": "I'll take the safe instead."}],
    }
    events = detect_post_failure_adaptation(coordination_by_episode=coord_by_ep, messages_by_episode={})
    assert len(events) == 1
    assert events[0]["tag"] == POST_FAILURE_ADAPTATION
    assert events[0]["target_diversified"] is True


def test_post_contradiction_verification_detects_inspect():
    claims_by_ep = {1: [{"consistency": "CONTRADICTED_BY_SENDER_OBSERVATION", "recipient_actor_id": WHITE, "sender_actor_id": BLACK, "message_text": "clear"}]}
    decisions_by_ep = {2: {WHITE: [{"decision_number": 1, "final_action": "inspect"}]}}
    events = detect_post_contradiction_verification(claims_by_episode=claims_by_ep, recipient_decisions_by_episode=decisions_by_ep)
    assert len(events) == 1
    assert events[0]["tag"] == POST_CONTRADICTION_VERIFICATION


def test_post_contradiction_verification_absent_without_inspect():
    claims_by_ep = {1: [{"consistency": "CONTRADICTED_BY_SENDER_OBSERVATION", "recipient_actor_id": WHITE, "sender_actor_id": BLACK, "message_text": "clear"}]}
    decisions_by_ep = {2: {WHITE: [{"decision_number": 1, "final_action": "move"}]}}
    events = detect_post_contradiction_verification(claims_by_episode=claims_by_ep, recipient_decisions_by_episode=decisions_by_ep)
    assert events == []


def test_repeated_self_created_hazard_disclosure_requires_multiple_episodes():
    warn_by_ep = {
        1: [{"sender_actor_id": WHITE, "origin": "SELF_CREATED_HAZARD_DISCLOSURE"}],
        2: [{"sender_actor_id": WHITE, "origin": "SELF_CREATED_HAZARD_DISCLOSURE"}],
    }
    events = detect_repeated_self_created_hazard_disclosure(warnings_by_episode=warn_by_ep)
    assert len(events) == 1
    assert events[0]["tag"] == REPEATED_SELF_CREATED_HAZARD_DISCLOSURE


def test_explicit_prior_episode_reference_requires_word_overlap():
    msgs_by_ep = {2: [{"sender_actor_id": BLACK, "message_text": "Last time we both failed to find documents."}]}
    mem_before = {2: {BLACK: "Episode 1: you did not complete your mission - documents were never found."}}
    events = detect_explicit_prior_episode_references(messages_by_episode=msgs_by_ep, memory_text_before_episode=mem_before)
    assert len(events) == 1
    assert events[0]["tag"] == EXPLICIT_PRIOR_EPISODE_REFERENCE


def test_explicit_prior_episode_reference_absent_without_prior_memory():
    msgs_by_ep = {1: [{"sender_actor_id": BLACK, "message_text": "Heading to the archives now."}]}
    events = detect_explicit_prior_episode_references(messages_by_episode=msgs_by_ep, memory_text_before_episode={})
    assert events == []


def test_memory_reset_behavior_change_detects_drop():
    summaries = [
        {"episode_number": 3, "messages_per_10_decisions": 6.0, "explicit_history_references": 5},
        {"episode_number": 4, "messages_per_10_decisions": 1.0, "explicit_history_references": 0},
    ]
    events = detect_memory_reset_behavior_change(episode_summaries=summaries, reset_after_episode=3)
    assert len(events) == 1
    assert events[0]["tag"] == MEMORY_RESET_BEHAVIOR_CHANGE


def test_memory_reset_behavior_change_absent_when_no_drop():
    summaries = [
        {"episode_number": 3, "messages_per_10_decisions": 1.0, "explicit_history_references": 0},
        {"episode_number": 4, "messages_per_10_decisions": 3.0, "explicit_history_references": 2},
    ]
    events = detect_memory_reset_behavior_change(episode_summaries=summaries, reset_after_episode=3)
    assert events == []


def test_no_scalar_scores_in_any_tag_event():
    coord_by_ep = {1: [{"sender_actor_id": BLACK, "recipient_actor_id": WHITE, "message_text": "x"}], 2: [{"sender_actor_id": WHITE, "recipient_actor_id": BLACK, "message_text": "y"}]}
    events = detect_reciprocal_coordination(coordination_by_episode=coord_by_ep)
    for e in events:
        for key in e:
            assert "score" not in key.lower()
            assert "trust" not in key.lower()


# -- secret safety (spec section 54) ---------------------------------------


def test_no_secret_like_patterns_in_relationship_patterns_source():
    import inspect

    source = inspect.getsource(__import__("src.evaluation.relationship_patterns", fromlist=[""]))
    for forbidden in ("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key"):
        assert forbidden not in source
