"""M5E.3: Amazon Bedrock as a stable cloud inference provider, behind the
exact same `ModelClient` protocol every other provider (`OllamaClient`,
`AnthropicClient`, ...) implements - `LLMController`/the agent adapters
need no Bedrock-specific branch anywhere (spec section 13).

Primary API: `bedrock-runtime`'s Converse API via `boto3.client(
"bedrock-runtime")` (spec section 8) - never Bedrock Agents/AgentCore.

Authentication: `AWS_BEARER_TOKEN_BEDROCK` from the process environment
(spec section 3). No custom credential storage here (spec section 4) -
this project's installed `botocore` (1.43+) already recognizes
`AWS_BEARER_TOKEN_<SIGNING_NAME>` natively for any service in
`botocore.handlers.get_bearer_auth_supported_services()`, which includes
"bedrock" - a plain `boto3.client("bedrock-runtime", region_name=...)`
picks the token up on its own. Confirmed by reading the installed
`botocore` source before writing this, not assumed.

Secret safety (spec sections 3, 25, 48): the token value itself is never
read, stored, or embedded in any object this module constructs - only
`bedrock_token_present() -> bool` ever inspects the environment variable's
*presence*, and every error path runs the resulting message through
`redact_bedrock_token` (an exact-literal-substring redaction using the
real token value, in addition to `llm_controller.normalize_provider_error_message`'s
generic key/token/secret pattern redaction) before it can reach any
`ModelResponse`, exception, log line, or telemetry event.
"""

from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.llm.clients.base import ModelCapabilities, ModelResponse
from src.llm.clients.ollama_client import _extract_json_payload

BEARER_TOKEN_ENV_VAR = "AWS_BEARER_TOKEN_BEDROCK"
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"


class BedrockConfigurationError(RuntimeError):
    """Raised at construction time (spec section 5: "fail clearly before
    starting an episode") - never at generate_action time - for problems
    that mean no request could ever legitimately be attempted: no region
    resolvable, or (optionally, via `require_token=True`) no bearer token
    present. Never carries the token value."""


def bedrock_token_present() -> bool:
    return bool(os.environ.get(BEARER_TOKEN_ENV_VAR))


def resolve_bedrock_region(explicit_region: str | None = None) -> str:
    """Spec section 5's preferred order: an explicit `region` argument (a
    caller-supplied `BEDROCK_REGION` value) -> `AWS_REGION` -> the AWS
    SDK's own default region resolution (profile/config file). Raises
    `BedrockConfigurationError` if none resolve - never silently
    hardcodes one."""
    region = explicit_region or os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION")
    if region:
        return region
    try:
        import boto3

        sdk_region = boto3.Session().region_name
    except Exception:  # pragma: no cover - defensive: boto3 misconfiguration shouldn't crash region resolution differently than "no region"
        sdk_region = None
    if sdk_region:
        return sdk_region
    raise BedrockConfigurationError(
        "no AWS region could be resolved for Bedrock - checked BEDROCK_REGION, AWS_REGION, and the AWS SDK's own "
        "default region resolution (profile/config). Set one explicitly before starting an episode."
    )


def redact_bedrock_token(message: str | None) -> str | None:
    """Defense in depth beyond `normalize_provider_error_message`'s
    generic pattern redaction (spec section 25): if the *real* bearer
    token value happens to appear verbatim anywhere in a message (e.g. an
    SDK echoing a header back in an error - not expected, but never
    assumed impossible), replace that exact substring, using the real
    env var value this process actually holds - never logged itself."""
    if message is None:
        return None
    token = os.environ.get(BEARER_TOKEN_ENV_VAR)
    if token:
        message = message.replace(token, "[redacted]")
    return message


# -- live: a tool-call markup answer ------------------------------------------
# Found live 2026-10-02 (pilot tournament, Claude Sonnet 5.5 on Bedrock): the
# native tool-like prompt names a function `choose_action`, and this model
# answered 295 of 308 calls with the call written as markup instead of JSON:
#   <invoke name="choose_action"><parameter name="action">MOVE</parameter>...</invoke>
# Those answers carried a complete action yet were counted as invalid. Live
# play only (BedrockClient(accept_invoke_markup=True), set by the workshop's
# build_client); it is read only when the text holds no JSON object, and only
# the parameters actually written are kept - nothing is filled in.
_INVOKE_RE = re.compile(r'<invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</invoke>', re.DOTALL)
_PARAMETER_RE = re.compile(r'<parameter\s+name="(?P<key>[^"]+)"\s*>(?P<value>.*?)</parameter>', re.DOTALL)
INVOKE_FUNCTION = "choose_action"


def invoke_markup_payload(text: str | None) -> dict[str, Any]:
    """The arguments of the first `<invoke name="choose_action">` block, or {}."""
    match = _INVOKE_RE.search(text or "")
    if match is None or match.group("name").strip() != INVOKE_FUNCTION:
        return {}
    params: dict[str, Any] = {}
    for p in _PARAMETER_RE.finditer(match.group("body")):
        value = html.unescape(p.group("value").strip())
        params[p.group("key").strip()] = None if value.lower() in ("", "null", "none") else value
    return params if params.get("action") else {}


# -- error taxonomy (spec section 21) ---------------------------------------

from src.llm_controller import (  # noqa: E402  (after the helpers above that don't need it)
    PROVIDER_ACCESS_ERROR,
    PROVIDER_AUTH_ERROR,
    PROVIDER_INVALID_RESPONSE,
    PROVIDER_SERVICE_ERROR,
    PROVIDER_THROTTLING,
    PROVIDER_TIMEOUT,
    PROVIDER_TRANSPORT_ERROR,
    RUNTIME_EXCEPTION,
)

# AWS error codes (ClientError.response["Error"]["Code"]) that Bedrock's
# Converse API is documented to raise, mapped into the shared taxonomy.
# Anything not listed here falls back to PROVIDER_SERVICE_ERROR (a
# service-side rejection this module doesn't yet have a name for) rather
# than the generic RUNTIME_EXCEPTION bucket (spec section 21: "do not
# collapse all Bedrock failures into runtime_exception").
_BEDROCK_ERROR_CODE_MAP = {
    "AccessDeniedException": PROVIDER_ACCESS_ERROR,
    "UnauthorizedException": PROVIDER_AUTH_ERROR,
    "UnrecognizedClientException": PROVIDER_AUTH_ERROR,
    "ExpiredTokenException": PROVIDER_AUTH_ERROR,
    "InvalidSignatureException": PROVIDER_AUTH_ERROR,
    "ThrottlingException": PROVIDER_THROTTLING,
    "TooManyRequestsException": PROVIDER_THROTTLING,
    "ServiceQuotaExceededException": PROVIDER_THROTTLING,
    "ModelTimeoutException": PROVIDER_TIMEOUT,
    "ServiceUnavailableException": PROVIDER_SERVICE_ERROR,
    "InternalServerException": PROVIDER_SERVICE_ERROR,
    "ModelErrorException": PROVIDER_SERVICE_ERROR,
    "ModelNotReadyException": PROVIDER_SERVICE_ERROR,
    "ResourceNotFoundException": PROVIDER_SERVICE_ERROR,
    "ValidationException": PROVIDER_SERVICE_ERROR,
}


def classify_bedrock_exception(exc: BaseException) -> tuple[str, str]:
    """Maps a real boto3/botocore exception into
    `(error_type_category, safe_error_message)` - the message is already
    redacted (spec section 25) before this returns."""
    import botocore.exceptions as be

    if isinstance(exc, be.NoCredentialsError):
        category, message = PROVIDER_AUTH_ERROR, "no AWS credentials/bearer token available"
    elif isinstance(exc, (be.ConnectTimeoutError, be.ReadTimeoutError)):
        category, message = PROVIDER_TIMEOUT, str(exc)
    elif isinstance(exc, (be.EndpointConnectionError, be.ConnectionError, be.ProxyConnectionError, be.SSLError, be.ConnectionClosedError)):
        category, message = PROVIDER_TRANSPORT_ERROR, str(exc)
    elif isinstance(exc, be.ClientError):
        error_info = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        code = error_info.get("Code", "")
        category = _BEDROCK_ERROR_CODE_MAP.get(code, PROVIDER_SERVICE_ERROR)
        message = f"{code}: {error_info.get('Message', str(exc))}" if code else str(exc)
        # M5E.3.1 real-world finding: a bearer token that has simply expired
        # comes back from Bedrock as `AccessDeniedException` (not
        # `ExpiredTokenException`/`UnauthorizedException`, which the code-map
        # above already treats as auth errors) - observed live during
        # M5E.3's own second day. That's a credential problem (fix: get a
        # new token), not an IAM permissions problem (fix: grant access), so
        # it's reclassified here to PROVIDER_AUTH_ERROR rather than left as
        # the code-map's default PROVIDER_ACCESS_ERROR for this one message
        # shape - a genuine "identity lacks permission" AccessDeniedException
        # (no "token"/"expired" wording) still falls through to
        # PROVIDER_ACCESS_ERROR unchanged.
        if code == "AccessDeniedException" and "token" in message.lower() and "expired" in message.lower():
            category = PROVIDER_AUTH_ERROR
    elif isinstance(exc, be.BotoCoreError):
        category, message = RUNTIME_EXCEPTION, str(exc)
    else:
        category, message = RUNTIME_EXCEPTION, str(exc)
    return category, redact_bedrock_token(message)


def _input_tokens(usage: dict[str, Any]) -> int | None:
    """Uncached prompt tokens: Bedrock's inputTokens plus any cache write (billed
    like input); cache reads are reported separately as cached_input_tokens."""
    base = usage.get("inputTokens")
    write = usage.get("cacheWriteInputTokens") or 0
    return None if base is None else base + write


class BedrockClient:
    """`ModelClient` protocol implementation for Amazon Bedrock's Converse
    API (spec section 9's `BedrockProvider` box). `client_factory`
    mirrors every other client's `request_fn` injection point (spec
    section 28: "mock Bedrock SDK calls") - a callable returning an
    object with a `.converse(**kwargs)` method, defaulting to a real
    `boto3.client("bedrock-runtime", ...)`."""

    provider = "bedrock"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        region: str | None = None,
        timeout: float = 30.0,
        client_factory: Callable[[], Any] | None = None,
        temperature: float | None = 0.0,
        accept_invoke_markup: bool = False,
        cache_system_prompt: bool = False,
        additional_request_fields: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        # 0.0 everywhere in the recorded work; None omits the field (newer
        # Bedrock Claude models reject it - live --bedrock-temperature provider-default)
        self.temperature = temperature
        self.accept_invoke_markup = accept_invoke_markup  # live only, see invoke_markup_payload
        # live only: a cache point after the system prompt (Bedrock prompt
        # caching). The system prompt is the same on every call of a spy, so
        # after the first call it is read from the cache (measured 2026-10-02:
        # 2,045 tokens cached on Sonnet 5.5 and Opus 5.5). Bedrock reports cached
        # tokens apart from inputTokens: reads go to `cached_input_tokens`, writes
        # (billed like input) are added to `input_tokens`.
        self.cache_system_prompt = cache_system_prompt
        # live only: Converse additionalModelRequestFields (e.g. a reasoning effort)
        self.additional_request_fields = additional_request_fields
        self.model = model_name  # some call sites read `.model` (see Game._build_llm_client)
        self.region = resolve_bedrock_region(region)
        self.timeout = timeout
        self._client_factory = client_factory or self._build_boto3_client
        self._client: Any = None

    def _build_boto3_client(self) -> Any:
        import boto3
        from botocore.config import Config

        config = Config(connect_timeout=min(self.timeout, 10.0), read_timeout=self.timeout, retries={"max_attempts": 0})
        return boto3.client("bedrock-runtime", region_name=self.region, config=config)

    def _client_instance(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def generate_action(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        started = time.monotonic()
        try:
            client = self._client_instance()
            kwargs: dict[str, Any] = {}
            if self.temperature is not None:
                kwargs["inferenceConfig"] = {"temperature": self.temperature}
            if self.additional_request_fields:
                kwargs["additionalModelRequestFields"] = self.additional_request_fields
            response = client.converse(
                modelId=self.model_name,
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                system=[{"text": system_prompt}] + ([{"cachePoint": {"type": "default"}}]
                                                    if self.cache_system_prompt else []),
                **kwargs,
            )
        except Exception as exc:
            latency_ms = round((time.monotonic() - started) * 1000.0, 2)
            error_type, error_message = classify_bedrock_exception(exc)
            return ModelResponse(
                payload={},
                raw_text=None,
                provider=self.provider,
                model=self.model_name,
                latency_ms=latency_ms,
                error_type=error_type,
                error_message=error_message,
                capabilities=ModelCapabilities(native_structured_output=False, token_usage=True),
            )

        latency_ms = round((time.monotonic() - started) * 1000.0, 2)
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content_blocks if isinstance(block, dict) and "text" in block)
        payload = _extract_json_payload(text)
        parsed_from = "json"
        if not payload and self.accept_invoke_markup:
            payload = invoke_markup_payload(text)
            parsed_from = "invoke_markup"
        usage = response.get("usage") or {}
        metrics = response.get("metrics") or {}

        if not payload:
            return ModelResponse(
                payload={},
                raw_text=redact_bedrock_token(text),
                provider=self.provider,
                model=self.model_name,
                latency_ms=latency_ms,
                input_tokens=_input_tokens(usage),
                cached_input_tokens=usage.get("cacheReadInputTokens"),
                output_tokens=usage.get("outputTokens"),
                finish_reason=response.get("stopReason"),
                error_type=PROVIDER_INVALID_RESPONSE,
                # the stop reason says why there was nothing to read (e.g. content_filtered)
                error_message="Bedrock response contained no parseable JSON action" + (
                    f" (stopReason {response.get('stopReason')})"
                    if response.get("stopReason") not in (None, "end_turn") else ""),
                capabilities=ModelCapabilities(native_structured_output=False, token_usage=True),
            )

        return ModelResponse(
            payload=payload,
            raw_text=redact_bedrock_token(text),
            provider=self.provider,
            model=self.model_name,
            latency_ms=latency_ms,
            input_tokens=_input_tokens(usage),
            cached_input_tokens=usage.get("cacheReadInputTokens"),
            output_tokens=usage.get("outputTokens"),
            finish_reason=response.get("stopReason"),
            capabilities=ModelCapabilities(native_structured_output=False, token_usage=True),
            provider_metadata={
                "total_tokens": usage.get("totalTokens"),
                "bedrock_latency_ms": metrics.get("latencyMs"),
                "stop_reason": response.get("stopReason"),
                "region": self.region,
                **({"parsed_from": parsed_from} if self.accept_invoke_markup else {}),
            },
        )


# -- preflight (spec sections 7, 29) -----------------------------------------


@dataclass
class BedrockPreflightResult:
    ok: bool
    bedrock_auth_mode: str
    token_present: bool
    region: str | None
    model_id: str
    error_type: str | None
    error_message: str | None
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        """Spec section 48's safe-metadata contract - never the token."""
        return {
            "ok": self.ok,
            "bedrock_auth_mode": self.bedrock_auth_mode,
            "token_present": self.token_present,
            "region": self.region,
            "model_id": self.model_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def run_bedrock_preflight(
    model_id: str = DEFAULT_MODEL_ID,
    region: str | None = None,
    timeout: float = 30.0,
    client_factory: Callable[[], Any] | None = None,
) -> BedrockPreflightResult:
    """Spec section 7/29: validate token presence, validate region,
    attempt one minimal Converse request. Never records the token itself
    - only `token_present`."""
    token_present = bedrock_token_present()
    if not token_present:
        return BedrockPreflightResult(
            ok=False, bedrock_auth_mode="bearer_token_env", token_present=False, region=region, model_id=model_id,
            error_type=PROVIDER_AUTH_ERROR, error_message=f"{BEARER_TOKEN_ENV_VAR} is not set in the environment",
            latency_ms=None, input_tokens=None, output_tokens=None,
        )

    try:
        resolved_region = resolve_bedrock_region(region)
    except BedrockConfigurationError as exc:
        return BedrockPreflightResult(
            ok=False, bedrock_auth_mode="bearer_token_env", token_present=True, region=None, model_id=model_id,
            error_type="configuration_error", error_message=str(exc),
            latency_ms=None, input_tokens=None, output_tokens=None,
        )

    client = BedrockClient(model_name=model_id, region=resolved_region, timeout=timeout, client_factory=client_factory)
    response = client.generate_action(
        "You output JSON only.",
        '{"ping": true} - respond with exactly {"pong": true} and nothing else.',
        {},
    )
    return BedrockPreflightResult(
        ok=(response.error_type is None),
        bedrock_auth_mode="bearer_token_env",
        token_present=True,
        region=resolved_region,
        model_id=model_id,
        error_type=response.error_type,
        error_message=response.error_message,
        latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
