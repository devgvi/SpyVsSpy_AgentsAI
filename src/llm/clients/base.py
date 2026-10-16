from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelCapabilities:
    native_structured_output: bool = False
    tool_calling: bool = False
    token_usage: bool = False
    cached_token_usage: bool = False
    streaming: bool = False
    time_to_first_token: bool = False


@dataclass
class ModelResponse:
    payload: dict[str, Any] | None
    raw_text: str | None
    provider: str
    model: str
    latency_ms: float | None = None
    time_to_first_token_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    # M5E.1.1 spec section 14: raw, provider-specific response metadata
    # that doesn't fit the common fields above (Ollama's own
    # done/done_reason/total_duration/load_duration/prompt_eval_count/
    # prompt_eval_duration/eval_count/eval_duration) - None for every
    # provider that doesn't populate it, never a fabricated/guessed value.
    provider_metadata: dict[str, Any] | None = None


class ModelClient(Protocol):
    def generate_action(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        ...
