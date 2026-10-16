"""M7 workshop live game: per-spy provider/model overrides and the live match wiring (network-free)."""
from __future__ import annotations

import pytest

from src import constants
from src.llm.clients.bedrock_client import BedrockClient
from src.llm.clients.openai_compatible_client import OpenAIClient
from tests.workshop_live_helpers import LegalMoveClient
from workshop.live.config import resolve_config
from workshop.live.overlay import TelemetryView
from workshop.live.providers import build_client
from workshop.live.session import LiveGame, MatchRunner, verify_live_agents

BLACK, WHITE = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID


def test_shared_flags_configure_both_spies():
    config = resolve_config({"provider": "bedrock", "model": "m1"})
    assert config.black.target() == config.white.target() == ("bedrock", "m1") and not config.mixed


def test_actor_flags_override_the_shared_ones():
    config = resolve_config({"provider": "bedrock", "model": "claude", "white_model": "gpt-on-bedrock"})
    assert config.black.target() == ("bedrock", "claude") and config.white.target() == ("bedrock", "gpt-on-bedrock")
    config = resolve_config({"black_provider": "bedrock", "black_model": "claude", "white_provider": "openai", "white_model": "gpt"})
    assert config.black.target() == ("bedrock", "claude") and config.white.target() == ("openai", "gpt") and config.mixed


def test_one_spy_without_a_provider_is_an_error():
    from workshop.live.config import LiveConfigError
    with pytest.raises(LiveConfigError, match="White Spy"):
        resolve_config({"black_provider": "bedrock", "black_model": "m"})


def test_mixed_bedrock_openai_game_builds_one_real_client_per_spy(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    config = resolve_config({"black_provider": "bedrock", "black_model": "claude-x", "white_provider": "openai", "white_model": "gpt-y"})
    game = LiveGame(config, None)
    black, white = game.black_llm_controller, game.bot_controller
    assert isinstance(black.controller.model_client, BedrockClient) and black.controller.model_client.model_name == "claude-x"
    assert isinstance(white.controller.model_client, OpenAIClient) and white.controller.model_client.model_name == "gpt-y"
    assert (black.provider_name, black.model_name) == ("bedrock", "claude-x")
    assert (white.provider_name, white.model_name) == ("openai", "gpt-y")
    assert black.runtime_name == white.runtime_name == "native_tool_like"
    # the workshop default is live_async timing: HP09 guard on, no pair barrier
    assert config.decision_timing == "live_async" and black.execution_mode == white.execution_mode == "live_async"
    assert game.commit_policy == "barrier_answer_order" and not game._barrier_active()
    assert verify_live_agents(game) == []


def test_evaluation_timing_keeps_the_m6g3_barrier(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    config = resolve_config({"black_provider": "bedrock", "black_model": "claude-x", "white_provider": "openai",
                             "white_model": "gpt-y", "decision_timing": "evaluation"})
    game = LiveGame(config, None)
    assert game.black_llm_controller.execution_mode == game.bot_controller.execution_mode == "evaluation"
    assert game.commit_policy == "barrier_answer_order" and game._barrier_active()
    assert verify_live_agents(game) == []


def test_verification_rejects_a_mock_client():
    from src.llm.clients.mock_client import MockModelClient
    config = resolve_config({"provider": "openai", "model": "m"})
    game = LiveGame(config, None, client_factory=lambda spec, cfg: MockModelClient(model=spec.model))
    assert any("mock" in p for p in verify_live_agents(game))


def test_strands_runtime_is_selectable_for_bedrock(monkeypatch):
    pytest.importorskip("strands")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    config = resolve_config({"provider": "bedrock", "model": "m", "runtime": "strands"})
    game = LiveGame(config, None)
    assert game.black_llm_controller.runtime_name == "strands" and game.bot_controller.model_name == "m"


@pytest.mark.parametrize("timing", ["live_async", "evaluation"])
def test_headless_mixed_match_runs_decisions_commits_and_telemetry(tmp_path, timing):
    config = resolve_config({"black_provider": "bedrock", "black_model": "b", "white_provider": "openai", "white_model": "w",
                             "decision_timing": timing})
    clients = {}

    def factory(spec, cfg):
        clients[spec.actor_id] = LegalMoveClient(spec)
        return clients[spec.actor_id]

    summary = MatchRunner(config, tmp_path, headless=True, max_decisions=3, max_wall_seconds=60, client_factory=factory).run()
    assert summary["stop_reason"] == "max_decisions=3" and summary["decision_timing"] == timing
    if timing == "evaluation":
        assert summary["barrier_active"] and summary["barrier_releases_seen"] >= 1
    else:
        assert not summary["barrier_active"]
        assert all(summary["actors"][a]["live_async"]["commits"] >= 3 for a in (BLACK, WHITE))
    for actor, provider, model in ((BLACK, "bedrock", "b"), (WHITE, "openai", "w")):
        a = summary["actors"][actor]
        assert (a["provider"], a["model"]) == (provider, model)
        assert a["decisions"] >= 3 and a["action_source_llm"] == a["decisions"] and a["unique_decision_ids"] == a["decisions"]
        assert clients[actor].calls >= 3
    assert (tmp_path / "telemetry_match01.jsonl").stat().st_size > 0
    assert {"screenshot_first_commit.png", "screenshot_final.png"} <= set(summary["screenshots"])


def test_overlay_lines_show_provider_model_runtime_decision_and_memory():
    config = resolve_config({"black_provider": "bedrock", "black_model": "b", "white_provider": "openai", "white_model": "w"})
    game = LiveGame(config, None, client_factory=lambda spec, cfg: LegalMoveClient(spec))
    view = TelemetryView(config)
    view.last_decision[WHITE] = {"decision_id": "abcdef1234", "repair_attempted": True, "fallback_used": False,
                                 "action_source": "llm", "memory_size_chars": 2500, "recent_decisions_count": 4,
                                 "recent_failures_count": 1, "latency_ms": 812.0, "observation_hash": "0123456789ff"}
    text = "\n".join(t for t, _ in view.actor_lines(view.actor(game, WHITE), detail=True))
    for needle in ("WHITE SPY", "[native]", "openai / w", "abcdef12", "REPAIR", "memory", "2.5k", "812 ms", "0123456789"):
        assert needle in text, needle
    view.last_decision[WHITE]["executed_action"] = {"action": "move", "target_id": "filing_cabinet_01"}
    view.last_lifecycle[WHITE] = {"decision_id": "abcdef1234", "action_status": "failed", "action_terminal_reason": "path_unreachable"}
    lines = [t for t, _ in view.actor_lines(view.actor(game, WHITE), detail=True)]
    assert next(t for t in lines if t.startswith("action")) == "action   MOVE -> filing_cabinet_01"
    assert next(t for t in lines if t.startswith("result")) == "result   failed: no path"
    status, _ = view.status_line(game, paused=True, speed=2.0, match_over=False)
    assert "PAUSED" in status and "not for comparisons" in status and "live_async" in status
    evaluation = LiveGame(resolve_config({"black_provider": "bedrock", "black_model": "b", "white_provider": "openai",
                                          "white_model": "w", "decision_timing": "evaluation"}), None,
                          client_factory=lambda spec, cfg: LegalMoveClient(spec))
    status, _ = TelemetryView(evaluation.live_config).status_line(evaluation, paused=False, speed=1.0, match_over=False)
    assert "barrier_answer_order" in status


def test_build_client_is_per_spy():
    config = resolve_config({"black_provider": "bedrock", "black_model": "b", "white_provider": "openai", "white_model": "w",
                             "region": "us-east-1"})
    assert isinstance(build_client(config.black, config), BedrockClient)
    assert isinstance(build_client(config.white, config), OpenAIClient)
