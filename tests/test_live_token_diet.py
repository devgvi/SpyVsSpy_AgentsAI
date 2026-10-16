"""Live token diet: prompt caching, a slimmer payload, compact JSON, 300 s default.

Measured 2026-10-02 (60-match tournament, workshop_runs/tournament-report):
a model call carried ~8.7k input tokens - ~2k of system prompt sent again on
every call, and a ~14.8k-character JSON payload whose strategy memory grew to
~6k characters, much of it repeated elsewhere in the same payload. Live play
only: recorded runs and the zero-call labs keep their exact prompts.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from unittest.mock import patch

import pytest

from src import constants
from src.llm.clients.base import ModelResponse
from src.llm.clients.bedrock_client import BedrockClient
from src.llm.live_payload import LIVE_PAYLOAD_LEGEND, slim_live_payload
from src.llm.personality import RIDDLE_RULE
from tests.test_bedrock_client import _FakeConverseClient, _ok_converse_response
from tests.test_bugfix_live_match_flow import BASE, FakeWall, frames
from workshop.live.config import resolve_config
from workshop.live.session import LiveGame

WHITE, BLACK = constants.WHITE_SPY_ID, constants.BLACK_SPY_ID

PAYLOAD = {
    "schema_version": "obs-v2", "legal_actions_count": 3,
    "self": {"actor_id": "white_spy", "room": "archives"},
    "room": {"id": "archives", "objects": [
        {"id": "desk_01", "type": "desk", "position": [1, 2], "trappable": True, "searched": False,
         "inspected": False, "trap_status": "unknown", "door_open": None, "locked": False},
        {"id": "door_a", "type": "door", "position": [3, 4], "trappable": False, "searched": False,
         "inspected": False, "trap_status": "unknown", "door_open": True, "locked": False, "interact_effect": None,
         "connects": ["archives", "corridor"], "traversable": True}]},
    "opponent": {"visible": False, "last_seen": {"actor_id": "black_spy", "game_time": 36.579999999999636,
                                                 "position": [1327.51, 619.49], "room": "reception"}},
    "encounter": {"rule": "long text", "options": "long text", "win_chance": 0.5,
                  "alert": {"opponent_distance": 90, "decide": "long text"}},
    "puzzle": {"type": "riddle", "clue": None, "how_to_answer": "x", "wrong_code_costs": "y",
               "inscriptions_you_have_read": {"plant_02": "engraved: 45"}},
    "time_budget": {"urgency": "RELAXED", "note": "a lower bound"},
    "action_costs": {"move_eta_s": {"desk_01": 1.2, "painting_01": None}, "note": "seconds"},
    "mission": {"required_items": ["passport"], "have": []},
    "communication": {"enabled": False, "recipient_id": None, "messages_received_recently": []},
    "strategy_memory": {
        "schema_version": "m1", "mission": {"have": [], "missing": ["passport"]},
        "exploration": {"visited_rooms": ["archives"], "recently_visited": ["archives"]},
        "known_locks": [{"target_id": "door_s", "lock_type": "keycard", "requires": "unknown", "satisfied": False}],
        "known_empty_objects": ["sofa_01", "desk_01"], "known_searched_objects": ["sofa_01", "desk_01", "armchair_01"],
        "recent_decisions": [{"game_time": 12.3456, "decision_id": "uuid", "observation_hash": "sha",
                              "selected_action": "move", "target_id": "desk_01", "valid": True,
                              "action_source": "llm", "engine_result": None}],
        "recent_events": [{"game_time": 1.25, "event_type": "ITEM_FOUND", "target_id": "x", "detail": None}],
        "opponent_memory": {"currently_visible": False}, "messages_sent_count": 0, "messages_received_count": 0,
    },
}


def test_the_slim_payload_keeps_every_fact_and_drops_repeats_statics_and_bookkeeping():
    slim = slim_live_payload(json.loads(json.dumps(PAYLOAD)))
    assert "schema_version" not in slim and "legal_actions_count" not in slim and "communication" not in slim
    desk, door = slim["room"]["objects"]
    assert desk == {"id": "desk_01", "type": "desk", "position": [1, 2], "trappable": True, "searched": False,
                    "inspected": False, "trap_status": "unknown", "locked": False}  # only the null went
    assert door == {"id": "door_a", "type": "door", "position": [3, 4], "door_open": True, "locked": False,
                    "connects": ["archives", "corridor"], "traversable": True}
    assert slim["opponent"]["last_seen"] == {"game_time": 36.6, "position": [1328, 619], "room": "reception"}
    assert slim["encounter"] == {"win_chance": 0.5, "alert": {"opponent_distance": 90}}
    assert slim["puzzle"] == {"type": "riddle", "clue": None,  # null here is a fact: no clue found yet
                              "inscriptions_you_have_read": {"plant_02": "engraved: 45"}}
    assert slim["time_budget"] == {"urgency": "RELAXED"}
    assert slim["action_costs"] == {"move_eta_s": {"desk_01": 1.2, "painting_01": None}}  # a null ETA is a fact
    memory = slim["strategy_memory"]
    assert set(memory) == {"exploration", "known_locks", "known_empty_objects", "known_searched_objects",
                           "recent_decisions", "recent_events"}
    assert memory["exploration"] == {"recently_visited": ["archives"]}
    assert memory["known_locks"] == [{"target_id": "door_s", "lock_type": "keycard", "satisfied": False}]
    assert memory["known_empty_objects"] == ["desk_01", "sofa_01"]  # sorted: no hash-seed order
    assert memory["recent_decisions"] == [{"game_time": 12.3, "selected_action": "move", "target_id": "desk_01",
                                           "valid": True, "action_source": "llm"}]
    assert memory["recent_events"] == [{"game_time": 1.2, "event_type": "ITEM_FOUND", "target_id": "x"}]
    assert PAYLOAD["strategy_memory"]["recent_decisions"][0]["decision_id"] == "uuid"  # the input is not mutated


def test_dropped_static_texts_are_stated_once_in_the_live_prompt():
    assert "null move_eta_s" in LIVE_PAYLOAD_LEGEND and "lower bound" in LIVE_PAYLOAD_LEGEND
    assert "busy 4 s" in RIDDLE_RULE and "for 5 s" in RIDDLE_RULE  # was puzzle.wrong_code_costs


class Recorder:
    """A provider client that records the exact prompts it is sent."""

    def __init__(self, spec):
        self.model_name, self.provider = spec.model, spec.provider
        self.prompts: list[tuple[str, str]] = []

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        payload = json.loads(user_prompt)
        if payload.get("preflight"):
            return ModelResponse(payload={"action": "move", "target_id": "probe_door"}, raw_text="{}",
                                 provider=self.provider, model=self.model_name, latency_ms=1.0)
        self.prompts.append((system_prompt, user_prompt))
        return ModelResponse(payload={"action": "wait", "reason": "x"}, raw_text="{}", provider=self.provider,
                             model=self.model_name, latency_ms=1.0)


def test_live_game_sends_the_slim_payload_as_compact_json_with_the_legend():
    clients = {}

    def factory(spec, cfg):
        clients[spec.actor_id] = Recorder(spec)
        return clients[spec.actor_id]

    wall = FakeWall()
    game = LiveGame(resolve_config({**BASE, "puzzle": "riddle"}), None, client_factory=factory, wall_time_fn=wall)
    for _ in range(40):
        frames(game, wall, 5)
        if clients[WHITE].prompts:
            break
    system, user = clients[WHITE].prompts[0]
    assert LIVE_PAYLOAD_LEGEND in system
    payload = json.loads(user)
    assert user == json.dumps(payload, sort_keys=True, separators=(",", ":"))  # compact JSON
    assert "schema_version" not in payload and "decision_id" not in user and "note" not in payload["time_budget"]
    assert "rule" not in payload.get("encounter", {}) and "how_to_answer" not in payload["puzzle"]


def test_recorded_controllers_keep_their_exact_payload():
    from src.evaluation.navigation_audit import fresh_state
    from src.llm_controller import LLMController
    from src.observation import build_observation
    from tests.test_bugfix_live_async import Gate, GatedClient

    seen = []

    class Raw(GatedClient):
        def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
            seen.append(user_prompt)
            return super().generate_action(system_prompt, user_prompt, output_schema, request_context)

    import time

    agent = LLMController(WHITE, model_client=Raw(Gate(lambda p: {"action": "wait", "reason": "x"}, auto=True)))
    agent.get_action(build_observation(fresh_state(), WHITE))
    end = time.monotonic() + 5
    while time.monotonic() < end and agent._thinking_in_progress:
        time.sleep(0.005)
    assert seen and seen[0] == json.dumps(json.loads(seen[0]), sort_keys=True)  # default separators
    payload = json.loads(seen[0])
    assert payload["schema_version"] and "mission" in payload["strategy_memory"]  # nothing slimmed
    assert agent.config.model_payload_transform is None and not agent.config.compact_payload_json


# -- prompt caching ------------------------------------------------------------------

def test_the_live_bedrock_client_caches_the_system_prompt_and_counts_cached_tokens():
    response = _ok_converse_response()
    response["usage"] = {"inputTokens": 80, "outputTokens": 10, "cacheReadInputTokens": 2045,
                         "cacheWriteInputTokens": 0}
    fake = _FakeConverseClient(response)
    live = BedrockClient(model_name="m", region="us-east-1", cache_system_prompt=True, client_factory=lambda: fake)
    result = live.generate_action("rules", "payload", {})
    assert fake.last_kwargs["system"] == [{"text": "rules"}, {"cachePoint": {"type": "default"}}]
    assert result.input_tokens == 80 and result.cached_input_tokens == 2045
    first = _ok_converse_response()
    first["usage"] = {"inputTokens": 80, "outputTokens": 10, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 2045}
    assert BedrockClient(model_name="m", region="us-east-1", cache_system_prompt=True,
                         client_factory=lambda: _FakeConverseClient(first)).generate_action("r", "p", {}).input_tokens \
        == 80 + 2045  # a cache write is billed like input
    recorded = _FakeConverseClient(_ok_converse_response())
    BedrockClient(model_name="m", region="us-east-1", client_factory=lambda: recorded).generate_action("rules", "p", {})
    assert recorded.last_kwargs["system"] == [{"text": "rules"}]


def test_the_workshop_client_and_strands_cache_the_system_prompt():
    from workshop.live.providers import build_client

    config = resolve_config({"provider": "bedrock", "model": "us.anthropic.claude-sonnet-5-5", "region": "us-east-1"})
    assert build_client(config.black, config).cache_system_prompt
    # GPT-6 / 6.1 Sol on Bedrock refuse a request with a cache point (preflight 2026-10-02)
    gpt = resolve_config({"provider": "bedrock", "model": "us.openai.gpt-6.1-sol", "region": "us-east-1"})
    assert not build_client(gpt.black, gpt).cache_system_prompt
    from workshop.live.providers import bedrock_prompt_cache_supported
    assert bedrock_prompt_cache_supported("global.anthropic.claude-opus-5-5")
    assert not bedrock_prompt_cache_supported("openai.gpt-oss-120b-1:0")
    pytest.importorskip("strands")
    from src.agents.strands_agent import StrandsAgentAdapter

    adapter = StrandsAgentAdapter("white", provider_name="bedrock", model_name="m", llm_region="us-east-1")
    adapter.prompt_cache = True
    with patch("strands.models.bedrock.BedrockModel") as model:
        adapter._build_default_agent("system prompt", lambda: None)
    kwargs = model.call_args.kwargs
    assert kwargs["cache_config"].strategy == "auto" and kwargs["temperature"] == 0.0


# -- 300 s by default ------------------------------------------------------------------

def test_tournament_defaults_to_the_300_s_mission_and_keeps_the_option(tmp_path, monkeypatch):
    from workshop.scripts import tournament

    monkeypatch.setenv("SPY_WORKSHOP_RUNS_DIR", str(tmp_path))
    args = tournament.argparse.Namespace(model="m", region="us-east-1", bedrock_temperature="0",
                                         mission_preset="standard", puzzle="riddle", label="t", max_minutes=8.0)
    cmd = tournament.command(args, tournament.schedule(tournament.PERSONALITIES, [1])[0])
    assert cmd[cmd.index("--mission-preset") + 1] == "standard"
    assert tournament.MISSION_SECONDS["standard"] == 300.0
    assert tournament.main(["--label", "t", "--model", "m", "--seeds", "1", "--dry-run"]) == 0
    root = tmp_path / "tournament-t"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"model": "m", "mission_preset": "long", "matches": []}))
    assert tournament.main(["--label", "t", "--model", "m", "--seeds", "1", "--dry-run"]) == 2  # no mixed setups
    assert tournament.main(["--label", "t", "--model", "m", "--seeds", "1", "--mission-preset", "long",
                            "--dry-run"]) == 0
