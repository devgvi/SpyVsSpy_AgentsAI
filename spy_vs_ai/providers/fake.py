"""The fake provider: an offline stand-in for demos and automated tests (FINAL-2 spec 46, 48).

It opens no connection, reads no credential and runs no model: it picks a legal action from the game payload with a
seeded generator (search > inspect > move, never a hidden-world fact), and reports zero tokens so nothing it produces
can be mistaken for a real model call. `latency_s` simulates a slow provider (the call runs on the controller's
background thread, so the game shows THINKING); `mode="flaky"` makes every `fail_every`-th call fail with a provider
service error, to show repair / fallback handling."""
from __future__ import annotations

import json
import random
import time
from typing import Any

from spy_vs_ai.providers.base import DecisionProvider, PROVIDER_SERVICE_ERROR


class FakeProviderError(RuntimeError):
    def __init__(self, message: str) -> None:
        self.category = PROVIDER_SERVICE_ERROR
        self.detail = message
        super().__init__(f"{PROVIDER_SERVICE_ERROR}: {message}")


def _legal(payload: dict[str, Any]) -> dict[str, list[str | None]]:
    compact = payload.get("legal_actions_compact")
    if isinstance(compact, dict):
        return {str(a): list(t) if isinstance(t, list) else [None] for a, t in compact.items()}
    out: dict[str, list[str | None]] = {}
    for entry in payload.get("legal_actions") or []:
        if isinstance(entry, dict) and entry.get("action"):
            out.setdefault(str(entry["action"]), []).append(entry.get("target_id"))
    return out


class FakeClient:
    fake = True

    def __init__(self, model: str, *, seed: int = 0, latency_s: float = 0.0, mode: str = "legal", fail_every: int = 6) -> None:
        self.model_name = model
        self.provider = "fake"
        self.seed = seed
        self.latency_s = latency_s
        self.mode = mode
        self.fail_every = max(2, int(fail_every))
        self.calls = 0

    def generate_action(self, system_prompt, user_prompt, output_schema, request_context=None):
        from src.llm.clients.base import ModelCapabilities, ModelResponse
        self.calls += 1
        started = time.monotonic()
        if self.latency_s:
            time.sleep(self.latency_s)
        if self.mode == "flaky" and self.calls % self.fail_every == 0:
            raise FakeProviderError(f"simulated provider failure (call {self.calls})")
        try:
            payload = json.loads(user_prompt)
        except (TypeError, ValueError):
            payload = {}
        if payload.get("preflight"):
            pick = {"action": "move", "target_id": "probe_door"}
        else:
            legal = _legal(payload)
            rng = random.Random(f"fake:{self.seed}:{self.model_name}:{self.calls}")
            pick = {"action": "wait"}
            for action, chance in (("interact", 0.6), ("inspect", 0.5)):
                targets = [t for t in legal.get(action, []) if t]
                if targets and rng.random() < chance:
                    pick = {"action": action, "target_id": rng.choice(sorted(targets))}
                    break
            else:
                moves = sorted(t for t in legal.get("move", []) if t)
                if moves:
                    pick = {"action": "move", "target_id": rng.choice(moves)}
        answer = {**pick, "reason": "fake provider (no model)"}
        return ModelResponse(payload=answer, raw_text=json.dumps(answer), provider="fake", model=self.model_name,
                             latency_ms=round((time.monotonic() - started) * 1000.0, 2), input_tokens=0, output_tokens=0,
                             capabilities=ModelCapabilities(native_structured_output=True))


class FakeProvider(DecisionProvider):
    provider_name = "fake"
    credential_env = ()


def build(model: str, options: dict[str, Any], *, scrub=None, seed: int = 0) -> FakeProvider:
    client = FakeClient(model, seed=int(options.get("seed", seed) or 0), latency_s=float(options.get("latency_s") or 0.0),
                        mode=str(options.get("mode") or "legal"), fail_every=int(options.get("fail_every") or 6))
    return FakeProvider(client, scrub=scrub)


def status(options: dict[str, Any] | None = None, *, env=None) -> dict[str, Any]:
    return {"provider": "fake", "state": "available", "detail": "offline demo provider: no model, no network"}
