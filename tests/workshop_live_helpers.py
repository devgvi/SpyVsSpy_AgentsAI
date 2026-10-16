"""Shared network-free helpers for the M7 live-game tests."""
from __future__ import annotations

import io
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import workshop  # noqa: E402,F401
from src.llm.clients.base import ModelResponse  # noqa: E402

FAKE_OPENAI_KEY = "sk-test-FAKE0123456789abcdefFAKE"
FAKE_BEDROCK_TOKEN = "ABSKFAKEbedrocktoken0123456789"


def legal_move_targets(payload: dict) -> list[str]:
    """MOVE targets from a model payload: the Native payload carries
    `legal_actions_compact` as {action: [target_id, ...]}; a plain
    `legal_actions` list of {action, target_id} is accepted too."""
    compact = payload.get("legal_actions_compact")
    if isinstance(compact, dict):
        return [t for t in compact.get("move", []) if t]
    return [a["target_id"] for a in payload.get("legal_actions") or []
            if isinstance(a, dict) and a.get("action") == "move" and a.get("target_id")]


class LegalMoveClient:
    """A stand-in provider client (deliberately not MockModelClient) that
    picks a legal MOVE from the game's payload, so a headless match makes
    real progress without any network."""

    def __init__(self, spec, echo: str | None = None) -> None:
        self.model_name = spec.model
        self.provider = spec.provider
        self.calls = 0
        self.echo = echo

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        self.calls += 1
        try:
            payload = json.loads(user_prompt)
        except json.JSONDecodeError:
            payload = {}
        if payload.get("preflight"):
            return ModelResponse(payload={"action": "move", "target_id": "probe_door", "reason": "probe"}, raw_text="{}",
                                 provider=self.provider, model=self.model_name, latency_ms=1.0)
        moves = [{"action": "move", "target_id": t} for t in legal_move_targets(payload)]
        pick = moves[self.calls % len(moves)] if moves else {"action": "wait"}
        reason = "fake reason" + (f" {self.echo}" if self.echo else "")
        return ModelResponse(payload={"action": pick["action"], "target_id": pick.get("target_id"), "reason": reason},
                             raw_text=json.dumps({"reason": reason}), provider=self.provider, model=self.model_name, latency_ms=2.0)


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def openai_body(content: dict) -> bytes:
    return json.dumps({"id": "chatcmpl-test", "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
