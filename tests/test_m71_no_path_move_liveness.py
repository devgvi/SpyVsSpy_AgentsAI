"""M7.1 liveness fix: a MOVE that fails at start (PATH_UNREACHABLE) must not
leave the agent waiting forever for an observation change.

Found in the M7.1 longer live run: both spies' last decisions were MOVEs to
an unreachable target (the known "no-path MOVE offered as legal" gap); the
failed MOVE left the world unchanged, the observation digest matched the
one that produced the decision, and neither agent asked the model again for
the remaining 164 s of game time. The legal-action gap itself stays open.
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from src import constants
from src.agents.action_lifecycle import PATH_UNREACHABLE, TARGET_REACHED
from src.game import Game
from src.llm.clients.base import ModelResponse
from src.llm_controller import forget_digest_after_unreachable_move
from src.navigation import NavGrid
from src.observation import build_observation


class _SameUnreachableMove:
    def __init__(self):
        self.calls = 0
        self.model_name = "scripted"

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        self.calls += 1
        moves = sorted((json.loads(user_prompt).get("legal_actions_compact") or {}).get("move", []))
        return ModelResponse(payload={"action": "move", "target_id": moves[0], "reason": "go"}, raw_text="{}",
                             provider="mock", model="scripted", latency_ms=1.0)


@pytest.mark.parametrize("fixed", [True, False])
def test_native_agent_decides_again_after_a_move_with_no_path(monkeypatch, fixed):
    monkeypatch.setattr(NavGrid, "find_path", lambda self, start, goal: None)
    if not fixed:  # the pre-M7.1 behavior: one decision, then silence
        import src.llm_controller as llm_controller
        monkeypatch.setattr(llm_controller, "forget_digest_after_unreachable_move", lambda agent, boundary: None)
    game = Game(seed=3, white_controller="llm", black_controller="scripted", llm_provider="mock", agent_runtime="native_tool_like")
    agent = game.bot_controller
    client = _SameUnreachableMove()
    agent.controller.model_client = client
    observation = build_observation(game.state, constants.WHITE_SPY_ID)  # the world never changes below
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and client.calls < 3:
        agent.get_action(observation)
        time.sleep(0.02)
    if not fixed:
        assert client.calls == 1  # starved: the digest never changes again
        return
    assert client.calls >= 3, "the agent stopped deciding after an unreachable MOVE"
    assert agent.controller._executor.actions_failed >= 2
    # still throttled: at most one request per decision interval
    assert agent.decisions_requested == client.calls


@pytest.mark.parametrize("boundary, cleared", [(PATH_UNREACHABLE, True), (TARGET_REACHED, False)])
def test_only_an_unreachable_move_forgets_the_digest(boundary, cleared):
    agent = type("A", (), {"_last_observation_digest": "abc"})()
    forget_digest_after_unreachable_move(agent, boundary)
    assert (agent._last_observation_digest is None) is cleared


def test_strands_adapter_applies_the_same_rule():
    pytest.importorskip("strands")
    from src.agents.strands_agent import StrandsAgentAdapter
    from src.llm.agent_memory import AgentMemory  # noqa: F401 - adapter builds its own
    adapter = StrandsAgentAdapter("white", agent_factory=lambda *a, **k: None)
    adapter._last_observation_digest = "abc"
    game = Game(seed=3, white_controller="scripted", llm_provider="mock")
    observation = build_observation(game.state, constants.WHITE_SPY_ID)
    adapter._executor.start(observation, "d1", "move", "no_such_object", None, 0.0)
    adapter._finish_action_lifecycle(observation, PATH_UNREACHABLE)
    assert adapter._last_observation_digest is None
