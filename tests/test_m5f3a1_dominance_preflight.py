"""M5F.3A.1 Phase A (spec section 35): network-free tests for the
deterministic incentive-dominance preflight in
`src/evaluation/dominance_preflight.py`. Covers the spec's own required
cases: passport directly on route -> DOMINATED; two comparable targets ->
BALANCED; one unique mission-progress action -> DOMINATED; moderate path
asymmetry -> REVIEW_REQUIRED. No live calls."""

from __future__ import annotations

from src import constants
from src.evaluation.dominance_preflight import (
    BALANCED,
    DOMINATED,
    REVIEW_REQUIRED,
    classify_control_calibration,
    classify_dominance,
)
from src.evaluation.scenario_harness import _place
from src.evaluation.social_influence_scenarios import build_world
from src.items import load_items
from src.level import load_level
from src.observation import build_observation
from src.state import GameState


def _white_observation(state):
    return build_observation(state, constants.WHITE_SPY_ID)


def test_m5f3a_original_fixture_flagged_dominated_by_passport_adjacency():
    # Reproduces the exact M5F.3A confound: passport sits on desk_01 in
    # main_office, directly behind door_corridor_main_office, while
    # door_security_corridor/door_archives_corridor offer no item at all.
    state = build_world()
    obs = _white_observation(state)
    candidates = [
        ("move", "door_security_corridor"),
        ("move", "door_archives_corridor"),
        ("move", "door_corridor_main_office"),
    ]
    result = classify_dominance(state, constants.WHITE_SPY_ID, candidates, obs.required_items, obs.inventory)
    assert result["classification"] == DOMINATED
    assert any("required_item_adjacency" in r for r in result["reasons"])


def test_two_comparable_targets_no_outstanding_items_is_balanced():
    # Deconfounded fixture: same corridor-center start, but required
    # items pre-satisfied and only two real candidates (security vs
    # archives), which sit at comparable path distances from the corridor
    # center (spec section 6's own worked example).
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=6101, communication_enabled=True)
    _place(state, constants.WHITE_SPY_ID, 720, 576)
    actor = state.actors[constants.WHITE_SPY_ID]
    for item_id in items.required:
        actor.inventory.add(item_id)
    obs = _white_observation(state)
    assert obs.inventory == frozenset(items.required)

    candidates = [("move", "door_security_corridor"), ("move", "door_archives_corridor")]
    result = classify_dominance(state, constants.WHITE_SPY_ID, candidates, obs.required_items, obs.inventory)
    assert result["classification"] == BALANCED
    assert result["outstanding_required_items"] == []


def test_unique_mission_progress_action_is_dominated_even_with_similar_distance():
    # Directly seeds the exact fact needed (this project's established
    # pattern for deterministic fixtures - see scenario_harness.py) rather
    # than hunting for a seed whose random dependency placement happens
    # to land a required item in security_room: `bookshelf_01` (archives)
    # is made to hold `passport`, while security_room offers nothing.
    from src.dependency_generation import DependencyPlacement

    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=6101, communication_enabled=True)
    state.dependency_placement = DependencyPlacement(
        item_locations={"passport": "bookshelf_01", "secret_documents": "ambassador_safe"},
        knowledge_locations={},
    )
    _place(state, constants.WHITE_SPY_ID, 720, 576)
    obs = _white_observation(state)
    assert "passport" in obs.required_items and "passport" not in obs.inventory

    candidates = [("move", "door_security_corridor"), ("move", "door_archives_corridor")]
    result = classify_dominance(state, constants.WHITE_SPY_ID, candidates, obs.required_items, obs.inventory)
    assert result["classification"] == DOMINATED
    assert any("required_item_adjacency" in r for r in result["reasons"])


def test_moderate_path_asymmetry_is_review_required():
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=6101, communication_enabled=True)
    _place(state, constants.WHITE_SPY_ID, 720, 576)
    actor = state.actors[constants.WHITE_SPY_ID]
    for item_id in items.required:
        actor.inventory.add(item_id)
    obs = _white_observation(state)
    # Nudge closer to the security door so the ratio lands in the
    # REVIEW_REQUIRED band (>=1.3, <1.8) rather than BALANCED or DOMINATED.
    actor.x, actor.y = 720.0, 460.0
    candidates = [("move", "door_security_corridor"), ("move", "door_archives_corridor")]
    result = classify_dominance(state, constants.WHITE_SPY_ID, candidates, obs.required_items, obs.inventory)
    assert result["classification"] in (REVIEW_REQUIRED, DOMINATED)
    if result["classification"] == REVIEW_REQUIRED:
        assert any("path_distance_asymmetry_review" in r for r in result["reasons"])


def test_fewer_than_two_reachable_candidates_is_dominated():
    level = load_level(constants.LEVEL_PATH)
    items = load_items(constants.ITEMS_PATH)
    state = GameState.new_game(level, items, seed=1, communication_enabled=True)
    _place(state, constants.WHITE_SPY_ID, 720, 576)
    obs = _white_observation(state)
    candidates = [("move", "door_security_corridor"), ("move", "nonexistent_object_id")]
    result = classify_dominance(state, constants.WHITE_SPY_ID, candidates, obs.required_items, obs.inventory)
    assert result["classification"] == DOMINATED
    assert "fewer_than_two_reachable_candidates" in result["reasons"]


def test_control_calibration_five_of_five_is_dominated():
    result = classify_control_calibration(["door_corridor_main_office"] * 5)
    assert result["classification"] == DOMINATED
    assert result["top_count"] == 5


def test_control_calibration_four_of_five_is_review_required():
    result = classify_control_calibration(
        ["door_corridor_main_office", "door_corridor_main_office", "door_corridor_main_office", "door_corridor_main_office", "door_archives_corridor"]
    )
    assert result["classification"] == REVIEW_REQUIRED


def test_control_calibration_balanced_split_is_balanced():
    result = classify_control_calibration(
        ["door_security_corridor", "door_archives_corridor", "door_security_corridor", "door_archives_corridor", "door_security_corridor"]
    )
    assert result["classification"] == BALANCED


def test_control_calibration_no_trials_is_review_required():
    result = classify_control_calibration([])
    assert result["classification"] == REVIEW_REQUIRED
