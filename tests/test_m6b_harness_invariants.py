"""M6B spec sections 47-50: harness configuration equivalence tests - a
reusable layer asserting that `Game._build_llm_agent` (the single shared
construction path for both White and Black, both runtimes) actually
propagates requested configuration through to the real controller/adapter,
for every actor x runtime combination. Divergence here is a harness bug,
never a behavioral result (spec section 61).

Personality propagation specifically (spec section 49) was found broken
for Strands during M6A.2 and fixed there; this file is the "for both
Native and Strands, for both actors" regression coverage spec section 49
asks be added, plus the broader settings-propagation layer spec section
50 wants as a standing invariant suite. No network calls.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src.game import Game
from src.llm.personality import CAUTIOUS, SABOTEUR, render_personality_block


def _controller_prompt_text(controller) -> str:
    """Native's `LLMController` and `StrandsAgentAdapter` expose their
    effective system prompt under different attribute names (a real,
    pre-existing surface difference documented in M6A's own runtime
    surface audit) - normalized here to one accessor so every test below
    can be written once per actor x runtime, not duplicated four ways."""
    if hasattr(controller, "_effective_system_prompt"):
        value = controller._effective_system_prompt
        return value() if callable(value) else value
    raise AssertionError(f"{type(controller).__name__} exposes no recognized effective-prompt accessor")


@pytest.mark.parametrize("agent_runtime", ["native_tool_like", "strands"])
@pytest.mark.parametrize("actor", ["white", "black"])
def test_personality_propagates_to_the_real_controller(agent_runtime, actor):
    kwargs = {"seed": 1, "agent_runtime": agent_runtime}
    if actor == "white":
        kwargs.update(white_controller="llm", white_personality=SABOTEUR)
    else:
        kwargs.update(white_controller="scripted", black_controller="llm", black_personality=SABOTEUR)

    game = Game(**kwargs)
    controller = game.bot_controller if actor == "white" else game.black_llm_controller
    assert controller is not None
    prompt = _controller_prompt_text(controller)
    assert render_personality_block(SABOTEUR) in prompt


@pytest.mark.parametrize("agent_runtime", ["native_tool_like", "strands"])
@pytest.mark.parametrize("actor", ["white", "black"])
def test_no_requested_personality_means_no_personality_block(agent_runtime, actor):
    kwargs = {"seed": 1, "agent_runtime": agent_runtime}
    if actor == "white":
        kwargs.update(white_controller="llm")
    else:
        kwargs.update(white_controller="scripted", black_controller="llm")

    game = Game(**kwargs)
    controller = game.bot_controller if actor == "white" else game.black_llm_controller
    prompt = _controller_prompt_text(controller)
    for profile in (SABOTEUR, CAUTIOUS):
        assert render_personality_block(profile) not in prompt


@pytest.mark.parametrize("agent_runtime", ["native_tool_like", "strands"])
def test_white_and_black_personalities_never_cross_contaminate(agent_runtime):
    # Real bug class this guards against: White's config leaking into
    # Black's controller (or vice versa) via a shared/reused object.
    game = Game(
        seed=1, agent_runtime=agent_runtime, white_controller="llm", black_controller="llm",
        white_personality=SABOTEUR, black_personality=CAUTIOUS,
    )
    white_prompt = _controller_prompt_text(game.bot_controller)
    black_prompt = _controller_prompt_text(game.black_llm_controller)
    assert render_personality_block(SABOTEUR) in white_prompt
    assert render_personality_block(CAUTIOUS) not in white_prompt
    assert render_personality_block(CAUTIOUS) in black_prompt
    assert render_personality_block(SABOTEUR) not in black_prompt


def _controller_region(controller) -> str | None:
    """`StrandsAgentAdapter` stores `_llm_region` on itself; `LLMController`
    doesn't duplicate it - it only ever reaches the `BedrockClient` it
    constructed (`controller.model_client.region`). A real structural
    asymmetry between the two adapters (already documented in M6A's own
    runtime surface audit), not a bug - normalized here to one accessor."""
    if hasattr(controller, "_llm_region"):
        return controller._llm_region
    return getattr(getattr(controller, "model_client", None), "region", None)


@pytest.mark.parametrize("agent_runtime", ["native_tool_like", "strands"])
def test_llm_region_propagates_to_both_actors(agent_runtime):
    game = Game(
        seed=1, agent_runtime=agent_runtime, white_controller="llm", black_controller="llm",
        llm_provider="bedrock", llm_model="us.anthropic.claude-sonnet-4-6", llm_region="us-east-1",
    )
    for controller in (game.bot_controller, game.black_llm_controller):
        assert _controller_region(controller) == "us-east-1"


@pytest.mark.parametrize("agent_runtime", ["native_tool_like", "strands"])
def test_model_id_propagates_to_both_actors(agent_runtime):
    game = Game(seed=1, agent_runtime=agent_runtime, white_controller="llm", black_controller="llm", llm_model="a-specific-model-id")
    for controller in (game.bot_controller, game.black_llm_controller):
        assert getattr(controller, "model_name", None) == "a-specific-model-id"
