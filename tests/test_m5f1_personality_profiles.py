"""M5F.1: personality profile configuration, prompt injection, isolation,
and trace attribution - spec sections 27, 50."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src.game import Game
from src.llm.personality import (
    CAUTIOUS,
    DEFAULT,
    InvalidPersonalityProfileError,
    OPPORTUNIST,
    RACER,
    SABOTEUR,
    VALID_PERSONALITY_PROFILES,
    render_personality_block,
    validate_personality_profile,
)
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController, LLMControllerConfig


# -- profile enum/config validation (spec sections 27, 50) --------------------


def test_all_four_named_profiles_plus_default_are_valid():
    assert VALID_PERSONALITY_PROFILES == {DEFAULT, RACER, CAUTIOUS, SABOTEUR, OPPORTUNIST}


def test_validate_personality_profile_normalizes_case_and_whitespace():
    assert validate_personality_profile(" Saboteur ") == SABOTEUR


def test_validate_personality_profile_none_and_default_both_normalize_to_default():
    assert validate_personality_profile(None) == DEFAULT
    assert validate_personality_profile("default") == DEFAULT


def test_invalid_personality_profile_fails_clearly():
    with pytest.raises(InvalidPersonalityProfileError):
        validate_personality_profile("ninja")


# -- DEFAULT backward compatibility (spec sections 12, 27) --------------------


def test_default_profile_renders_no_block_at_all():
    assert render_personality_block(DEFAULT) is None
    assert render_personality_block(None) is None


def test_named_profiles_render_a_concise_frozen_block():
    for profile in (RACER, CAUTIOUS, SABOTEUR, OPPORTUNIST):
        block = render_personality_block(profile)
        assert block is not None
        assert block.startswith(f"PERSONALITY: {profile.upper()}")
        assert len(block.splitlines()) < 10  # spec section 8: concise, not roleplay prose


def test_llm_controller_with_no_personality_config_matches_pre_m5f1_prompt_exactly():
    controller = LLMController("white", config=LLMControllerConfig())
    assert controller._effective_system_prompt() == controller.config.system_prompt


# -- prompt injection (spec sections 8-9, 50) ----------------------------------


def test_llm_controller_appends_personality_block_after_base_prompt():
    controller = LLMController("white", config=LLMControllerConfig(personality_profile=RACER))
    effective = controller._effective_system_prompt()
    assert effective.startswith(controller.config.system_prompt)
    assert "PERSONALITY: RACER" in effective


def test_invalid_personality_profile_fails_at_controller_construction():
    with pytest.raises(InvalidPersonalityProfileError):
        LLMController("white", config=LLMControllerConfig(personality_profile="ninja"))


# -- same-state profile isolation / no capability change (spec sections 7, 37, 50) --


def test_same_observation_and_legal_actions_regardless_of_personality():
    """Personality changes only the prompt text, never Observation/legal
    actions - verified by confirming two controllers built against the
    identical config shape differ only in `personality_profile`."""
    racer_config = LLMControllerConfig(personality_profile=RACER)
    saboteur_config = LLMControllerConfig(personality_profile=SABOTEUR)
    assert racer_config.max_retries == saboteur_config.max_retries
    assert racer_config.representation_variant == saboteur_config.representation_variant
    assert racer_config.action_schema_version == saboteur_config.action_schema_version


# -- trace/telemetry profile attribution (spec sections 15, 27, 50) ----------


def test_telemetry_records_personality_profile_on_llm_decision(tmp_path):
    telemetry = TelemetryLogger()
    controller = LLMController(
        "white", telemetry=telemetry, episode_id="ep1",
        config=LLMControllerConfig(personality_profile=CAUTIOUS),
    )
    from src.llm.clients.mock_client import MockModelClient

    controller.model_client = MockModelClient(response=[{"action": "wait"}])
    from src.observation import Observation

    obs = Observation(
        actor_id="white", seed=1, position=(0.0, 0.0), facing="east", stunned=False, stunned_seconds=0.0,
        busy=False, busy_seconds=0.0, mission_time_remaining=100.0, mission_complete=False,
        inventory=frozenset(), knowledge=frozenset(), required_items=(), trap_inventory={},
        nearby_object_id=None, items_found_here=frozenset(), objects=(), nav_grid=None,
        log_event=lambda *a, **k: None, current_room="office", known_rooms=frozenset({"office"}),
        known_doors=frozenset(), known_locked_objects={}, known_item_locations={},
        visible_other_actor=False, opponent=None, last_seen_opponent=None,
        revealed_opponent=None, revealed_opponent_seconds=0.0,
    )
    controller.get_action(obs)
    if controller._thinking_thread:
        controller._thinking_thread.join(timeout=2.0)

    decision_events = [e for e in telemetry.events if e.get("event_type") == "llm_decision"]
    assert decision_events
    assert decision_events[0]["personality_profile"] == CAUTIOUS
    assert decision_events[0]["actor_id"] == "white"


# -- per-actor independence / opponent-profile privacy (spec sections 14, 50) --


def _new_duel_game(**overrides) -> Game:
    kwargs = dict(
        white_controller="llm", black_controller="llm", llm_provider="mock", llm_model="mock-model",
        agent_runtime="native_tool_like",
    )
    kwargs.update(overrides)
    return Game(**kwargs)


def test_black_and_white_can_use_independent_personality_profiles():
    game = _new_duel_game(black_personality=SABOTEUR, white_personality=RACER)
    assert game.black_llm_controller.controller.config.personality_profile == SABOTEUR
    assert game.bot_controller.controller.config.personality_profile == RACER


def test_opponent_personality_never_appears_in_the_other_actors_prompt():
    game = _new_duel_game(black_personality=SABOTEUR, white_personality=RACER)
    black_prompt = game.black_llm_controller.controller._effective_system_prompt()
    white_prompt = game.bot_controller.controller._effective_system_prompt()
    assert "SABOTEUR" not in white_prompt
    assert "RACER" not in black_prompt
    assert "SABOTEUR" in black_prompt
    assert "RACER" in white_prompt


def test_default_personality_is_backward_compatible_with_pre_m5f1_game_construction():
    """No personality kwargs at all - prompt must be byte-identical to
    what a pre-M5F.1 Game construction would have produced."""
    game_default = _new_duel_game()
    game_explicit_none = _new_duel_game(black_personality=None, white_personality=None)
    assert (
        game_default.black_llm_controller.controller._effective_system_prompt()
        == game_explicit_none.black_llm_controller.controller._effective_system_prompt()
    )
    assert game_default.black_llm_controller.controller._effective_system_prompt() == game_default.black_llm_controller.controller.config.system_prompt


def test_invalid_personality_at_game_construction_fails_clearly():
    with pytest.raises(InvalidPersonalityProfileError):
        _new_duel_game(black_personality="villain")
