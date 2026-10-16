"""NativeAgentAdapter: the validated M5A/M5B native LLM path, wrapped
behind the M5C `AgentAdapter` contract.

Not a rewrite. This is composition, not duplication (spec section 5): it
constructs and delegates to the existing `LLMController` for prompt
building, M5B memory, repair/fallback policy, action validation, and
telemetry. It is the experimental *baseline* every future runtime adapter
(Strands, OpenAI Agents SDK, LangGraph, ...) gets compared against - not
legacy code on its way out.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentAdapter, AgentDecision
from src.llm.clients.base import ModelClient
from src.llm.telemetry import TelemetryLogger
from src.llm_controller import LLMController, LLMControllerConfig
from src.observation import Observation

NATIVE_RUNTIME_NAME = "native"
NATIVE_RUNTIME_VERSION = "native-v1"


class NativeAgentAdapter(AgentAdapter):
    runtime_name = NATIVE_RUNTIME_NAME
    runtime_version = NATIVE_RUNTIME_VERSION

    def __init__(
        self,
        actor_id: str,
        model_client: ModelClient | None = None,
        config: LLMControllerConfig | None = None,
        telemetry: TelemetryLogger | None = None,
        episode_id: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        execution_mode: str | None = None,
    ) -> None:
        # Same constructor surface as LLMController (spec section 6: zero
        # behavioral change means this adapter must not require a different
        # configuration shape to reach the identical decision path).
        self.controller = LLMController(
            actor_id,
            model_client=model_client,
            config=config,
            telemetry=telemetry,
            episode_id=episode_id,
            provider_name=provider_name,
            model_name=model_name,
            execution_mode=execution_mode,
            agent_runtime=self.runtime_name,
            agent_runtime_version=self.runtime_version,
        )

    def decide(self, observation: Observation) -> AgentDecision:
        action = self.controller.get_action(observation)
        return AgentDecision(
            action=action,
            runtime=self.runtime_name,
            # Reflects the most recently *issued* decision. When the native
            # controller's own throttle/dedup skips a fresh model call this
            # tick (unchanged observation, mid-interval), this - like the
            # returned action itself - is carried over from the last real
            # decision. That is existing native behavior (decision
            # frequency is explicitly not something M5C may change, see
            # spec section 6), not something introduced here.
            decision_id=self.controller._last_decision_id or "none",
            action_source=self.controller._last_action_source,
            metadata={
                "provider": self.controller.provider_name,
                "model": self.controller.model_name,
                "runtime_version": self.runtime_version,
            },
        )

    def __getattr__(self, name: str) -> Any:
        # Transparent delegation for duck-typed readers that expect the same
        # surface as the wrapped LLMController - src/game.py's evaluation-
        # mode/thinking checks and src/renderer.py's HUD (both already use
        # getattr(..., default) against whatever object is passed in, so
        # this makes NativeAgentAdapter a drop-in replacement for
        # game.bot_controller without touching either call site).
        controller = self.__dict__.get("controller")
        if controller is None:
            raise AttributeError(name)
        return getattr(controller, name)


class NativeToolLikeAgentAdapter(NativeAgentAdapter):
    """M5D.7/M5D.8: registers M5D.7's Native tool-like representation as a
    selectable `agent_runtime` name, so it flows through Game/EpisodeRunner/
    VariantSpec exactly like "native"/"strands" do - it IS NativeAgentAdapter
    (same LLMController, same plain completion API, same validator/
    repair-v2/ActionExecutor), only `LLMControllerConfig.representation_variant`
    and `system_prompt` differ (spec section 7: "NO real tool calling, NO
    Strands"). A caller-supplied `config` still wins for anything it sets
    explicitly other than these two fields, so tests/experiments can still
    override max_retries etc. the same way they do for plain NativeAgentAdapter.
    """

    runtime_name = "native_tool_like"
    runtime_version = "native-tool-like-v1"

    def __init__(
        self,
        actor_id: str,
        model_client: ModelClient | None = None,
        config: LLMControllerConfig | None = None,
        telemetry: TelemetryLogger | None = None,
        episode_id: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        execution_mode: str | None = None,
    ) -> None:
        from src.llm_controller import NATIVE_TOOL_LIKE_REPRESENTATION, NATIVE_TOOL_LIKE_SYSTEM_PROMPT

        config = config or LLMControllerConfig()
        config.system_prompt = NATIVE_TOOL_LIKE_SYSTEM_PROMPT
        config.representation_variant = NATIVE_TOOL_LIKE_REPRESENTATION
        super().__init__(
            actor_id,
            model_client=model_client,
            config=config,
            telemetry=telemetry,
            episode_id=episode_id,
            provider_name=provider_name,
            model_name=model_name,
            execution_mode=execution_mode,
        )
