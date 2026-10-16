"""The common provider interface (FINAL-2 spec 6, 22, 36).

`DecisionProvider.invoke(model=, request=, timeout_s=)` is the one place model inference happens. It wraps an
existing provider client (src/llm/clients/*) and normalizes what the runtime shows and records: structured
content, latency, tokens, physical call count, repair / retry metadata, error category, model id and provider id.

`ProviderModelClient` adapts a DecisionProvider back to the `generate_action(...) -> ModelResponse` contract the
frozen LLMController already uses, so serialization, validation, repair, fallback and telemetry stay exactly the
controller's: the ModelResponse the controller receives is the wrapped client's own, unchanged, and a provider
exception is re-raised unchanged (the controller's own repair / fallback path handles it). No vendor SDK object ever
reaches the engine: the controller sees a ModelResponse, the engine sees a GameAction.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

# The controller's error taxonomy (src/llm_controller.py), as strings: importing the controller is not needed here.
PROVIDER_AUTH_ERROR = "provider_auth_error"
PROVIDER_ACCESS_ERROR = "provider_access_error"
PROVIDER_TIMEOUT = "provider_timeout"
PROVIDER_THROTTLING = "provider_throttling"
PROVIDER_SERVICE_ERROR = "provider_service_error"
PROVIDER_TRANSPORT_ERROR = "provider_transport_error"
PROVIDER_HTTP_ERROR = "provider_http_error"
PROVIDER_INVALID_RESPONSE = "provider_invalid_response"
RUNTIME_EXCEPTION = "runtime_exception"

# spec 22: user-facing messages; technical detail stays in the logs
USER_ERRORS = {
    PROVIDER_AUTH_ERROR: "Provider authentication failed",
    PROVIDER_ACCESS_ERROR: "Provider access denied",
    PROVIDER_TIMEOUT: "Request timed out",
    PROVIDER_THROTTLING: "Provider service error (throttled)",
    PROVIDER_SERVICE_ERROR: "Provider service error",
    PROVIDER_TRANSPORT_ERROR: "Provider unavailable",
    PROVIDER_HTTP_ERROR: "Provider service error",
    PROVIDER_INVALID_RESPONSE: "Invalid model output",
    RUNTIME_EXCEPTION: "Provider service error",
}
MODEL_NOT_FOUND = "Model not found"
FALLBACK_USED = "Fallback action used"
_MODEL_NOT_FOUND_RE = re.compile(r"model identifier is invalid|model_not_found|model .*not found|does not exist|"
                                 r"unknown model|no such model", re.IGNORECASE)


def user_error(category: str | None, message: str | None = None) -> str | None:
    """The normalized user-facing text for a provider error category (None when there is no error)."""
    if not category:
        return None
    if message and _MODEL_NOT_FOUND_RE.search(message):
        return MODEL_NOT_FOUND
    return USER_ERRORS.get(category, "Provider service error")


def classify_exception(exc: BaseException) -> str:
    """The taxonomy category of a raised provider exception (used for display; the controller classifies on its own)."""
    category = getattr(exc, "category", None)
    if isinstance(category, str) and category:
        return category
    if type(exc).__module__.startswith("botocore"):  # Bedrock: the existing classifier (botocore is already loaded)
        try:
            from src.llm.clients.bedrock_client import classify_bedrock_exception
            return classify_bedrock_exception(exc)[0]
        except Exception:  # noqa: BLE001 - classification is best effort
            pass
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timed out" in text:
        return PROVIDER_TIMEOUT
    if "urlerror" in name or "connection" in name or "refused" in text:
        return PROVIDER_TRANSPORT_ERROR
    if "httperror" in name:
        return PROVIDER_HTTP_ERROR
    return RUNTIME_EXCEPTION


@dataclass(frozen=True)
class ProviderRequest:
    system_prompt: str
    user_prompt: str
    output_schema: dict[str, Any]
    request_context: dict[str, Any] | None = None

    @property
    def is_repair(self) -> bool:
        return bool((self.request_context or {}).get("repair"))


@dataclass
class ProviderResponse:
    """One physical provider call, normalized. `model_response` is what the controller receives (unchanged)."""
    provider_id: str
    model_id: str
    content: dict[str, Any] | None
    raw_text: str | None
    latency_ms: float | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    provider_calls: int = 1
    retry: dict[str, Any] = field(default_factory=dict)  # {"repair": bool} - the controller decides repairs
    error_category: str | None = None
    error_message: str | None = None
    user_error: str | None = None
    model_response: Any = None

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("model_response", None)
        record.pop("raw_text", None)  # the controller's own telemetry already keeps the raw answer
        return record


class DecisionProvider:
    """Base class: one provider (bedrock, openai, anthropic, ollama, fake) around one existing client instance.

    `credential_env` lists the only environment variables this provider may ever read (spec 11)."""
    provider_name: str = "?"
    credential_env: tuple[str, ...] = ()

    def __init__(self, client: Any, *, scrub=None) -> None:
        self.client = client
        self._scrub = scrub or (lambda text: text)

    def invoke(self, *, model: str, request: ProviderRequest, timeout_s: float) -> ProviderResponse:
        started = time.monotonic()
        try:
            response = self.client.generate_action(request.system_prompt, request.user_prompt, request.output_schema,
                                                   request_context=request.request_context)
        except Exception as exc:
            category = classify_exception(exc)
            message = self._scrub(str(getattr(exc, "detail", None) or exc))[:300]
            self.last = ProviderResponse(self.provider_name, model, None, None,
                                         round((time.monotonic() - started) * 1000.0, 1), retry={"repair": request.is_repair},
                                         error_category=category, error_message=message,
                                         user_error=user_error(category, message))
            raise
        category = None
        error_type = getattr(response, "error_type", None)
        message = self._scrub(getattr(response, "error_message", None) or "") or None
        if error_type:
            try:
                from src.llm_controller import classify_provider_call_error
                category = classify_provider_call_error(error_type, message)
            except Exception:  # noqa: BLE001
                category = RUNTIME_EXCEPTION
        payload = getattr(response, "payload", None)
        self.last = ProviderResponse(
            provider_id=self.provider_name, model_id=getattr(response, "model", None) or model,
            content=payload if isinstance(payload, dict) else None, raw_text=getattr(response, "raw_text", None),
            latency_ms=getattr(response, "latency_ms", None) or round((time.monotonic() - started) * 1000.0, 1),
            input_tokens=getattr(response, "input_tokens", None), output_tokens=getattr(response, "output_tokens", None),
            cached_input_tokens=getattr(response, "cached_input_tokens", None), retry={"repair": request.is_repair},
            error_category=category, error_message=message[:300] if message else None,
            user_error=user_error(category, message), model_response=response)
        return self.last


class ProviderModelClient:
    """The existing ModelClient contract over a DecisionProvider (what LLMController calls)."""

    def __init__(self, decision_provider: DecisionProvider, model: str, *, timeout_s: float, history: int = 500) -> None:
        self.decision_provider = decision_provider
        self.model_name = model
        self.model = model
        self.provider = decision_provider.provider_name
        self.provider_name = decision_provider.provider_name
        self.timeout_s = timeout_s
        self.calls = 0
        self.history: deque[dict[str, Any]] = deque(maxlen=history)

    def __getattr__(self, name: str) -> Any:
        # provenance attributes of the wrapped client (cache_system_prompt, actual_models_seen, fake ...)
        if name in ("decision_provider", "__setstate__"):
            raise AttributeError(name)
        return getattr(self.decision_provider.client, name)

    def generate_action(self, system_prompt: str, user_prompt: str, output_schema: dict[str, Any],
                        request_context: dict[str, Any] | None = None):
        self.calls += 1
        request = ProviderRequest(system_prompt, user_prompt, output_schema, request_context)
        try:
            response = self.decision_provider.invoke(model=self.model_name, request=request, timeout_s=self.timeout_s)
        except Exception:
            self.history.append(self.decision_provider.last.as_record())
            raise
        self.history.append(response.as_record())
        return response.model_response


__all__ = ["DecisionProvider", "ProviderModelClient", "ProviderRequest", "ProviderResponse", "user_error",
           "classify_exception", "USER_ERRORS", "MODEL_NOT_FOUND", "FALLBACK_USED"]
