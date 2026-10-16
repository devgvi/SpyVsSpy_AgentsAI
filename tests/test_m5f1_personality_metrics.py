"""M5F.1: behavioral classification and per-profile aggregation
(`src/evaluation/personality_metrics.py`) - spec sections 22-24, 41, 50.
Synthetic trace records only."""

from __future__ import annotations

from src.evaluation.personality_metrics import (
    MISSION_DIRECTED,
    OPPONENT_DIRECTED,
    OTHER,
    SAFETY_DIRECTED,
    aggregate_profile_metrics,
    annotate_behavior_tags,
    classify_decision,
    generate_mirror_pairs,
    return_to_mission_latencies,
)


def _record(n, action, target=None, progress=None, opponent_visible=False, repeat_count=0):
    return {
        "decision_number": n, "final_action": action, "final_target": target,
        "progress_delta": progress or ["NO_PROGRESS"], "opponent_visible_before": False,
        "opponent_visible_after": opponent_visible, "repeat_count": repeat_count,
    }


# -- behavior classification (spec section 22) ---------------------------------


def test_mission_directed_from_real_progress_delta():
    r = _record(1, "move", progress=["ITEM_PROGRESS"])
    assert MISSION_DIRECTED in classify_decision(r)


def test_safety_directed_for_inspect_and_disarm():
    assert SAFETY_DIRECTED in classify_decision(_record(1, "inspect"))
    assert SAFETY_DIRECTED in classify_decision(_record(1, "disarm"))


def test_opponent_directed_for_trap_placed_while_opponent_visible():
    r = _record(1, "place_trap", target="safe_01", opponent_visible=True)
    assert OPPONENT_DIRECTED in classify_decision(r)


def test_trap_placed_without_opponent_visible_is_not_opponent_directed_unless_contested():
    r = _record(1, "place_trap", target="safe_01", opponent_visible=False)
    assert OPPONENT_DIRECTED not in classify_decision(r)
    assert OPPONENT_DIRECTED in classify_decision(r, contested_targets=frozenset({"safe_01"}))


def test_contested_target_is_opponent_directed_even_without_trap():
    r = _record(1, "move", target="archives_door", opponent_visible=False)
    assert OPPONENT_DIRECTED in classify_decision(r, contested_targets=frozenset({"archives_door"}))


def test_no_matching_category_falls_back_to_other():
    r = _record(1, "wait")
    assert classify_decision(r) == [OTHER]


def test_a_decision_can_carry_multiple_tags():
    r = _record(1, "place_trap", target="safe_01", progress=["REQUIREMENT_PROGRESS"], opponent_visible=True)
    tags = classify_decision(r)
    assert MISSION_DIRECTED in tags
    assert OPPONENT_DIRECTED in tags


def test_rationale_text_is_never_consulted_by_the_classifier():
    """Spec section 22: 'do not use rationale text as the sole
    classifier' - this classifier doesn't read it at all."""
    r = _record(1, "wait")
    r["agent_stated_reason"] = "I am being extremely cautious and safety-focused right now."
    assert classify_decision(r) == [OTHER]


# -- return-to-mission latency (spec section 24) -------------------------------


def test_return_to_mission_latency_measured_in_decisions():
    records = [
        _record(1, "place_trap", target="safe_01", opponent_visible=True),
        _record(2, "wait"),
        _record(3, "move", progress=["ITEM_PROGRESS"]),
    ]
    annotate_behavior_tags(records)
    latencies = return_to_mission_latencies(records)
    assert latencies == [2]


def test_return_to_mission_latency_none_when_never_returns():
    records = [_record(1, "place_trap", target="safe_01", opponent_visible=True), _record(2, "wait")]
    annotate_behavior_tags(records)
    assert return_to_mission_latencies(records) == [None]


def test_return_to_mission_latency_empty_when_no_opponent_directed_decisions():
    records = [_record(1, "move", progress=["ITEM_PROGRESS"])]
    annotate_behavior_tags(records)
    assert return_to_mission_latencies(records) == []


# -- mirror-pair generation (spec sections 31-33) ------------------------------


def test_generate_mirror_pairs_produces_both_assignments_per_matchup_per_seed():
    matchups = [("racer", "saboteur")]
    pairs = generate_mirror_pairs(matchups, seeds=[901])
    assert len(pairs) == 2
    assignments = {(p["black_personality"], p["white_personality"]) for p in pairs}
    assert assignments == {("racer", "saboteur"), ("saboteur", "racer")}


def test_generate_mirror_pairs_stage_1_count_matches_spec():
    matchups = [("racer", "saboteur"), ("cautious", "saboteur"), ("racer", "opportunist"), ("cautious", "opportunist")]
    pairs = generate_mirror_pairs(matchups, seeds=[901])
    assert len(pairs) == 8  # spec section 33: Stage 1 = 4 matchups x 1 seed x 2 mirrors


def test_generate_mirror_pairs_stage_2_count_matches_spec():
    matchups = [("racer", "saboteur"), ("cautious", "saboteur"), ("racer", "opportunist"), ("cautious", "opportunist")]
    pairs = generate_mirror_pairs(matchups, seeds=[901, 902])
    assert len(pairs) == 16  # spec section 33: max 16 primary duels


# -- per-profile aggregation (spec section 49) ---------------------------------


def test_aggregate_profile_metrics_computes_shares_not_scores():
    records = [
        _record(1, "move", progress=["ITEM_PROGRESS"]),
        _record(2, "place_trap", target="safe_01", opponent_visible=True),
        _record(3, "inspect"),
        _record(4, "wait"),
    ]
    annotate_behavior_tags(records)
    agg = aggregate_profile_metrics({"saboteur": records})
    row = agg["saboteur"]
    assert row["decision_count"] == 4
    assert row["mission_directed_share"] == 0.25
    assert row["opponent_directed_share"] == 0.25
    assert row["safety_directed_share"] == 0.25
    assert row["trap_placement_rate"] == 0.25
    assert "score" not in "".join(row.keys()).lower()


def test_aggregate_profile_metrics_handles_zero_decisions_gracefully():
    agg = aggregate_profile_metrics({"racer": []})
    assert agg["racer"]["decision_count"] == 0


def test_aggregate_profile_metrics_never_produces_a_ranking_field():
    records = [_record(1, "move", progress=["ITEM_PROGRESS"])]
    annotate_behavior_tags(records)
    agg = aggregate_profile_metrics({"racer": records, "saboteur": records})
    for row in agg.values():
        assert "rank" not in row
        assert "winner" not in row
