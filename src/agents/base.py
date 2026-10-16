"""M5C: Agent runtime abstraction.

Separates two concepts that M5A/M5B kept fused inside `LLMController`:

- **Model provider** (`src/llm/clients/*`, `ModelClient`): request/auth/
  format/token-usage/latency for a specific LLM backend (Ollama, OpenAI,
  Anthropic, Bedrock). Knows nothing about game rules or agent lifecycle.
- **Agent runtime** (this package): turns a legitimate `Observation` + M5B
  strategic memory into one canonical `GameAction` - decision lifecycle,
  and (later) orchestration/tool semantics. Knows nothing about provider
  wire formats.

Never merge the two. A runtime may use any provider; a provider is usable
by any runtime.

M5C implements only this contract plus `NativeAgentAdapter` (the existing
validated native LLM path, wrapped rather than rewritten). Strands, the
OpenAI Agents SDK, LangGraph, etc. are explicitly out of scope here - see
M5C_AGENT_RUNTIME_ABSTRACTION_NATIVE_BASELINE.md section 28.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.actions import GameAction
from src.observation import Observation


@dataclass
class AgentDecision:
    """Canonical, runtime-independent decision result (spec section 4).

    `action` is the only thing that ever reaches the engine boundary - the
    rest is metadata for telemetry/comparison across runtimes.
    """

    action: GameAction
    runtime: str
    decision_id: str
    action_source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(ABC):
    """SDK-independent interface every agent runtime adapter implements.

    Conceptually `async def decide(observation) -> GameAction` (spec
    section 4). This codebase's existing non-blocking behavior is a
    background thread plus a decision-interval throttle (see
    `LLMController.get_action`), not asyncio - adapters preserve that same
    non-blocking contract rather than introducing real coroutines, since
    switching concurrency models mid-milestone would itself be a behavioral
    change section 6 rules out.

    Whatever a runtime does internally, it may only ever reach the engine
    through a validated `GameAction` returned from here (spec section 10):
    no adapter mutates `GameState` directly, teleports actors, opens doors,
    or otherwise bypasses the existing validator.
    """

    runtime_name: str = "unknown"
    runtime_version: str = "unknown"

    @abstractmethod
    def decide(self, observation: Observation) -> AgentDecision:
        """Turn a legitimate Observation (+ its own game-side M5B memory)
        into one canonical decision. Must not block the caller waiting on
        a slow model call - see the non-blocking note above."""
        raise NotImplementedError

    def get_action(self, observation: Observation) -> GameAction:
        """Drop-in compatibility with the pre-M5C call site (`src/game.py`
        calls `bot_controller.get_action(observation)`) and with duck-typed
        HUD/diagnostics readers (`src/renderer.py`) that only ever need the
        action itself, not the full decision wrapper."""
        return self.decide(observation).action
