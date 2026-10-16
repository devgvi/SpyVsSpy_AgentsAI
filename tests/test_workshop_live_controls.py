"""M7.1 workshop live game: keyboard controls, lifecycle monitor and overlay consistency (network-free).

Key presses are injected through MatchRunner's event source into the real
loop (SDL dummy driver). This proves what each key does to the runner; it
does not replace the human check in workshop/manual_ui_checklist.md.
"""
from __future__ import annotations

import time

import pygame
import pytest

from src import constants
from tests.workshop_live_helpers import LegalMoveClient
from workshop.live.config import resolve_config
from workshop.live.overlay import TelemetryView
from workshop.live.session import LiveGame, MatchRunner

BLACK, WHITE = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID
CONFIG = {"black_provider": "bedrock", "black_model": "b-model", "white_provider": "openai", "white_model": "w-model"}


class SlowClient(LegalMoveClient):
    """Answers after a short delay, so the barrier really holds and releases."""

    def generate_action(self, *args, **kwargs):
        time.sleep(0.15 if self.provider == "bedrock" else 0.25)
        return super().generate_action(*args, **kwargs)


def _scripted(script: dict[int, list[int]]):
    frame = {"n": 0}

    def source():
        frame["n"] += 1
        pygame.event.pump()
        return [pygame.event.Event(pygame.KEYDOWN, key=k) for k in script.get(frame["n"], [])]

    return source


@pytest.fixture(scope="module")
def keyed_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("live_keys")
    script = {40: [pygame.K_SPACE], 100: [pygame.K_SPACE], 110: [pygame.K_PLUS], 115: [pygame.K_PLUS], 120: [pygame.K_MINUS],
              125: [pygame.K_MINUS], 130: [pygame.K_t], 135: [pygame.K_t], 140: [pygame.K_v], 145: [pygame.K_F3],
              150: [pygame.K_s], 160: [pygame.K_r], 300: [pygame.K_ESCAPE]}
    runner = MatchRunner(resolve_config(CONFIG), run_dir, headless=False, max_wall_seconds=60,
                         client_factory=lambda spec, cfg: SlowClient(spec), event_source=_scripted(script))
    return runner.run(), run_dir


def _log(summary):
    return [(entry["key"], entry["state"]) for entry in summary["controls_log"]]


def test_every_control_key_is_handled_and_logged(keyed_run):
    summary, _ = keyed_run
    keys = [k for k, _ in _log(summary)]
    assert keys == ["Space", "Space", "+", "+", "-", "-", "T", "T", "V", "F3", "S", "R", "Esc"]
    assert summary["stop_reason"] == "quit"


def test_pause_freezes_game_time_and_launches_no_request(keyed_run):
    summary, _ = keyed_run
    (_, paused), (_, resumed) = _log(summary)[0:2]
    assert paused["paused"] is True and resumed["paused"] is False
    assert paused["game_time_remaining"] == resumed["game_time_remaining"]
    assert summary["pause"]["paused_frames"] >= 50
    assert summary["pause"]["game_time_drift_while_paused"] == 0.0
    assert summary["pause"]["requests_launched_while_paused"] == 0


def test_speed_steps_and_is_bounded(keyed_run):
    summary, _ = keyed_run
    speeds = [s["speed"] for k, s in _log(summary) if k in "+-"]
    assert speeds == [2.0, 2.0, 1.0, 0.5]


def test_toggles_and_screenshot(keyed_run):
    summary, run_dir = keyed_run
    log = dict(enumerate(_log(summary)))
    assert log[6][1]["telemetry_panels"] is False and log[7][1]["telemetry_panels"] is True
    assert log[8][1]["detail"] is True and log[9][1]["debug"] is True
    assert log[10][1]["screenshots"] == 1 and (run_dir / "screenshot_01.png").stat().st_size > 10_000


def test_restart_builds_a_fresh_match_with_the_same_providers_and_view_choices(keyed_run):
    summary, run_dir = keyed_run
    before, after = _log(summary)[10][1], _log(summary)[11][1]
    assert after["match"] == before["match"] + 1 and after["episode_id"] != before["episode_id"]
    assert after["agents"] == before["agents"] == {BLACK: "bedrock/b-model/native_tool_like", WHITE: "openai/w-model/native_tool_like"}
    assert after["game_time_remaining"] == resolve_config(CONFIG).time_limit  # a fresh mission clock (standard: 300 s)
    assert (after["detail"], after["debug"], after["telemetry_panels"]) == (True, True, True)
    assert summary["telemetry_files"] == ["telemetry_match01.jsonl", "telemetry_match02.jsonl"]
    assert [m["match"] for m in summary["stability"]] == [1, 2]


def test_lifecycle_stays_clean_across_pause_and_restart(keyed_run):
    summary, _ = keyed_run
    for report in summary["stability"]:
        # workshop default: live_async - the clock must keep running while a model thinks
        assert report["decision_timing"] == "live_async" and report["steps_frozen_while_thinking"] == 0
        assert report["overlay_mismatch_count"] == 0, report["overlay_mismatches"]
        assert report["overlay_max_lag_steps"] <= 3
        for actor in report["actors"].values():
            assert actor["duplicate_requests"] == 0 and actor["unused_responses"] == 0


@pytest.mark.parametrize("timing", ["evaluation", "live_async"])
def test_headless_monitor_counts_match_the_telemetry(tmp_path, timing):
    runner = MatchRunner(resolve_config({**CONFIG, "decision_timing": timing}), tmp_path, headless=True, max_decisions=6,
                         max_wall_seconds=60, client_factory=lambda spec, cfg: SlowClient(spec))
    summary = runner.run()
    [report] = summary["stability"]
    assert report["busy_steps"] > 0
    if timing == "evaluation":  # frozen clock + barrier
        assert report["steps_game_time_advanced_while_busy"] == 0
    else:  # the clock runs while a model thinks
        assert report["steps_frozen_while_thinking"] == 0 and report["steps_game_time_advanced_while_busy"] > 0
    assert report["overlay_checks"] > 0 and report["overlay_mismatch_count"] == 0
    for actor_id, actor in report["actors"].items():
        assert actor["decisions_completed"] == summary["actors"][actor_id]["decisions"]
        # a decision pending at shutdown may publish its ready flag a moment
        # before its telemetry event, so the pending one may be uncounted yet
        consumed = actor["hand_overs"] + actor["responses_cancelled_busy_or_stunned"] + actor["responses_discarded_stale"]
        assert consumed <= actor["decisions_completed"] <= consumed + actor["pending_at_end"]
        assert actor["duplicate_requests"] == 0 and actor["unused_responses"] == 0
        assert actor["memory_chars_first_max_last"] is not None
    assert summary["actors"][BLACK]["telemetry_providers"] == ["bedrock"] and summary["actors"][WHITE]["telemetry_providers"] == ["openai"]
    assert summary["actors"][BLACK]["telemetry_models"] == ["b-model"] and summary["actors"][WHITE]["telemetry_models"] == ["w-model"]


def test_overlay_never_pairs_a_new_decision_id_with_the_previous_decision_flags():
    config = resolve_config(CONFIG)
    game = LiveGame(config, None, client_factory=lambda spec, cfg: LegalMoveClient(spec))
    view = TelemetryView(config)
    view.last_decision[WHITE] = {"decision_id": "old-decision", "action_source": "fallback", "fallback_used": True}
    game.bot_controller.controller._last_decision_id = "new-decision-not-yet-emitted"
    snap = view.actor(game, WHITE)
    assert snap.decision_id == "old-decision" and snap.last_source == "fallback" and snap.last_fallback


class _UnreachableClient(LegalMoveClient):
    """Always the first MOVE target; every path is made unreachable below."""

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        self.calls += 1
        from src.llm.clients.base import ModelResponse
        from tests.workshop_live_helpers import legal_move_targets
        import json as _json
        targets = legal_move_targets(_json.loads(user_prompt))
        payload = {"action": "move", "target_id": targets[0], "reason": "go"} if targets else {"action": "wait"}
        return ModelResponse(payload=payload, raw_text="{}", provider=self.provider, model=self.model_name, latency_ms=1.0)


def test_unreachable_moves_keep_the_live_match_deciding_without_stalls(monkeypatch, tmp_path):
    # The without-fix stall is proven on an unchanged observation in
    # test_m71_no_path_move_liveness.py; here both spies see each other, so
    # this checks the fixed loop end to end and the stall/model-wait split.
    from src.navigation import NavGrid
    monkeypatch.setattr(NavGrid, "find_path", lambda self, start, goal: None)
    runner = MatchRunner(resolve_config(CONFIG), tmp_path, headless=True, max_wall_seconds=10, max_decisions=6,
                         client_factory=lambda spec, cfg: _UnreachableClient(spec))
    summary = runner.run()
    [report] = summary["stability"]
    assert summary["stop_reason"] == "max_decisions=6"
    assert report["max_stall_seconds"] < 1.5 and "max_model_wait_seconds" in report
    for actor in report["actors"].values():
        assert actor["duplicate_requests"] == 0 and actor["unused_responses"] == 0


def test_detail_view_shows_navigation_and_exploration_from_telemetry(tmp_path):
    runner = MatchRunner(resolve_config(CONFIG), tmp_path, headless=True, max_decisions=4, max_wall_seconds=60,
                         client_factory=lambda spec, cfg: SlowClient(spec))
    runner.run()
    view, game = runner.view, runner.game
    for actor_id in (BLACK, WHITE):
        snap = view.actor(game, actor_id)
        assert snap.navigation is not None and snap.navigation["interaction_range"] == constants.INTERACTION_DISTANCE
        assert snap.exploration is not None and snap.exploration["status"] in ("PARTIAL", "COMPLETE")
        text = "\n".join(t for t, _ in view.actor_lines(snap, detail=True))
        assert " move " in text and "px" in text and " explore " in text
        default = "\n".join(t for t, _ in view.actor_lines(snap, detail=False))
        assert " explore " not in default  # not in the default participant view
    events = game.telemetry.events
    moves = [e for e in events if e["event_type"] == "action_lifecycle" and e["action"] == "move"]
    assert moves and all("navigation" in e for e in moves)
    decisions = [e for e in events if e["event_type"] == "llm_decision"]
    assert all("exploration" in e["observation"] for e in decisions)
