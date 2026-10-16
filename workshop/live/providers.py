"""Provider clients and the live-play preflight.

Every client built here is one of the project's existing provider-neutral
`ModelClient`s (`src/llm/clients/`), so the game sees the same
Observation -> request -> ModelResponse -> validation/repair/fallback ->
scheduler -> Engine boundary whatever the provider:

* Bedrock -> `BedrockClient` (Converse API), any model id or inference
  profile the account can invoke. Credentials come from the normal AWS
  chain (profile, SSO, env) or `AWS_BEARER_TOKEN_BEDROCK`.
* OpenAI direct -> `OpenAIClient` (Chat Completions, JSON object output)
  with `OpenAIDirectTransport` as its request function. The transport reads
  `OPENAI_API_KEY` from the environment at call time, so the key is never
  stored on any object, never a CLI argument and never written anywhere.

The preflight never falls back: a missing credential, an unreachable model
or a model that cannot return the game's JSON action contract stops the
launch before the match starts.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable

from src.llm.clients.base import ModelResponse
from src.llm.clients.bedrock_client import (
    BEARER_TOKEN_ENV_VAR,
    BedrockClient,
    BedrockConfigurationError,
    bedrock_token_present,
    resolve_bedrock_region,
)
from src.llm.clients.openai_compatible_client import OpenAIClient
from src.llm_controller import (
    PROVIDER_ACCESS_ERROR,
    PROVIDER_AUTH_ERROR,
    PROVIDER_HTTP_ERROR,
    PROVIDER_INVALID_RESPONSE,
    PROVIDER_SERVICE_ERROR,
    PROVIDER_THROTTLING,
    PROVIDER_TIMEOUT,
    PROVIDER_TRANSPORT_ERROR,
    VALID_ACTION_NAMES,
)
from workshop.live.config import OPENAI_BASE_URL, OPENAI_KEY_ENV, ActorSpec, LiveConfig

# Environment variables whose values must never appear in output or artifacts.
SECRET_ENV_VARS = (OPENAI_KEY_ENV, BEARER_TOKEN_ENV_VAR, "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def secret_values() -> list[str]:
    return [v for v in (os.environ.get(name) for name in SECRET_ENV_VARS) if v and len(v) >= 8]


# Provider error bodies may echo a partly masked key (OpenAI's 401 says
# "Incorrect API key provided: sk-abc****wxyz"), which still reveals its
# prefix and suffix; key-shaped tokens and Bearer headers are removed too.
_KEY_SHAPED_RE = re.compile(r"\b(sk|rk)-[A-Za-z0-9_\-*]{6,}|\b(AKIA|ASIA)[A-Z0-9]{8,}|\bABSK[A-Za-z0-9+/=_\-]{8,}")
_BEARER_RE = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)


def redact(text: str | None) -> str | None:
    """Removes live secret values, key-shaped tokens and Bearer credentials from `text`."""
    if text is None:
        return None
    for value in secret_values():
        text = text.replace(value, "[redacted]")
    text = _KEY_SHAPED_RE.sub("[redacted-key]", text)
    return _BEARER_RE.sub(r"\1[redacted]", text)


# --------------------------------------------------------------------------- OpenAI direct


class OpenAIProviderError(RuntimeError):
    """A classified, already-redacted OpenAI request failure."""

    def __init__(self, category: str, message: str, status: int | None = None) -> None:
        self.category = category
        self.status = status
        self.detail = redact(message)
        super().__init__(f"{category}: {self.detail}")


_HTTP_STATUS_CATEGORY = {401: PROVIDER_AUTH_ERROR, 403: PROVIDER_ACCESS_ERROR, 404: PROVIDER_ACCESS_ERROR, 408: PROVIDER_TIMEOUT,
                         429: PROVIDER_THROTTLING}


class OpenAIDirectTransport:
    """`request_fn` for `OpenAIClient`: one Chat Completions POST to api.openai.com.

    Differences from the generic OpenAI-compatible HTTP path, all at the
    transport layer (the game never sees them):
    * the key is read from OPENAI_API_KEY per request, never stored;
    * the controller's `metadata` request context (episode/actor ids) is
      kept local - OpenAI only accepts string metadata on stored requests;
    * `temperature` follows the configured `openai_temperature`;
    * HTTP / network failures raise `OpenAIProviderError` with a taxonomy
      category and a redacted message (the controller then records a
      provider failure and falls back to WAIT, exactly as for Bedrock).
    """

    def __init__(self, *, base_url: str = OPENAI_BASE_URL, timeout: float = 60.0, temperature: float | None = 0.0,
                 key_env: str = OPENAI_KEY_ENV, opener: Callable[..., Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.key_env = key_env
        self._opener = opener or urllib.request.urlopen

    def __repr__(self) -> str:
        return f"OpenAIDirectTransport(base_url={self.base_url!r}, key_env={self.key_env!r})"

    def build_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {k: v for k, v in payload.items() if k != "metadata"}
        if self.temperature is None:
            body.pop("temperature", None)
        else:
            body["temperature"] = self.temperature
        return body

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = os.environ.get(self.key_env)
        if not key:
            raise OpenAIProviderError(PROVIDER_AUTH_ERROR, f"{self.key_env} is not set in the environment")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(self.build_body(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status = exc.code
            detail = ""
            try:
                err = json.loads(exc.read().decode("utf-8")).get("error") or {}
                detail = " ".join(str(err.get(k)) for k in ("code", "param", "message") if err.get(k))
            except Exception:  # an unreadable error body still yields a classified error
                detail = str(exc.reason or "")
            category = _HTTP_STATUS_CATEGORY.get(status) or (PROVIDER_SERVICE_ERROR if status >= 500 else PROVIDER_HTTP_ERROR)
            raise OpenAIProviderError(category, f"HTTP {status} {detail}"[:300], status=status) from None
        except (TimeoutError, socket.timeout):
            raise OpenAIProviderError(PROVIDER_TIMEOUT, f"request timed out after {self.timeout:.0f} s") from None
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError | socket.timeout):
                raise OpenAIProviderError(PROVIDER_TIMEOUT, f"request timed out after {self.timeout:.0f} s") from None
            raise OpenAIProviderError(PROVIDER_TRANSPORT_ERROR, f"connection failed: {reason}") from None
        except json.JSONDecodeError:
            raise OpenAIProviderError(PROVIDER_INVALID_RESPONSE, "response body is not JSON") from None


# --------------------------------------------------------------------------- client factory


def apply_aws_profile(config: LiveConfig) -> None:
    """Selects the configured AWS profile by name (not a secret) for boto3 and Strands."""
    if config.aws_profile and "bedrock" in config.providers_used:
        os.environ["AWS_PROFILE"] = config.aws_profile


def bedrock_prompt_cache_supported(model_id: str) -> bool:
    """An explicit cache point only for Anthropic models on Bedrock.

    Preflight 2026-10-02: GPT-6 Sol and GPT-6.1 Sol on Bedrock refuse a
    request that carries a `cachePoint` ("You invoked an unsupported model or
    your request did not allow prompt caching") and answer normally without
    it. Claude Sonnet 5.5 / Opus 5.5 accept it. This is the same rule as
    Strands' CacheConfig(strategy="auto")."""
    lowered = model_id.lower()
    return "anthropic" in lowered or "claude" in lowered


def build_client(spec: ActorSpec, config: LiveConfig, *, openai_opener: Callable[..., Any] | None = None,
                 bedrock_client_factory: Callable[[], Any] | None = None):
    """The provider client for one spy. Raises on anything unsupported - never a mock."""
    if spec.provider == "bedrock":
        return BedrockClient(model_name=spec.model, region=config.region, timeout=config.timeout_seconds,
                             client_factory=bedrock_client_factory,
                             temperature=config.bedrock_temperature_value(),
                             # Claude Sonnet 5.5 writes its answer as <invoke> markup (live pilot)
                             accept_invoke_markup=True,
                             # the system prompt is identical on every call of a spy: cache it
                             cache_system_prompt=bedrock_prompt_cache_supported(spec.model),
                             additional_request_fields=json.loads(spec.request_fields) if spec.request_fields else None)
    if spec.provider == "openai":
        transport = OpenAIDirectTransport(timeout=config.timeout_seconds, temperature=config.openai_temperature_value(),
                                          opener=openai_opener)
        return OpenAIClient(model_name=spec.model, api_key=None, timeout=config.timeout_seconds, request_fn=transport)
    raise ValueError(f"unsupported live provider {spec.provider!r}")


# --------------------------------------------------------------------------- preflight


@dataclass
class Check:
    name: str
    status: str  # PASS / FAIL / INFO
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def credential_checks(config: LiveConfig) -> list[Check]:
    """Network-free: only what the selected providers need, never secret values."""
    checks: list[Check] = []
    if "bedrock" in config.providers_used:
        try:
            import boto3  # noqa: F401
        except ImportError:
            checks.append(Check("bedrock sdk", "FAIL", "boto3 is not installed: pip install -r workshop/requirements.txt"))
            return checks
        if bedrock_token_present():
            credentials = Check("bedrock credentials", "PASS", f"{BEARER_TOKEN_ENV_VAR} is set (value not shown)")
        else:
            credentials = _aws_chain_check(config.aws_profile)
        if credentials.status == "FAIL" and config.aws_profile and "not found" in credentials.detail:
            # an unknown profile also breaks the SDK's region lookup; report the cause once
            checks.append(Check("bedrock region", "INFO", "not checked: fix the AWS profile first"))
        else:
            try:
                region = resolve_bedrock_region(config.region)
                checks.append(Check("bedrock region", "PASS", region))
            except BedrockConfigurationError as exc:
                checks.append(Check("bedrock region", "FAIL", f"{exc} (or pass --region)"))
        checks.append(credentials)
    if "openai" in config.providers_used:
        if os.environ.get(OPENAI_KEY_ENV):
            checks.append(Check("openai credentials", "PASS", f"{OPENAI_KEY_ENV} is set (value not shown)"))
        else:
            checks.append(Check("openai credentials", "FAIL", f"{OPENAI_KEY_ENV} is not set: export it in this shell (never pass it as an argument)"))
    return checks


def _aws_chain_check(profile: str | None) -> Check:
    import boto3
    from botocore.exceptions import BotoCoreError, ProfileNotFound

    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        credentials = session.get_credentials()
    except ProfileNotFound:
        return Check("bedrock credentials", "FAIL", f"AWS profile '{profile}' not found in your AWS config")
    except BotoCoreError as exc:
        return Check("bedrock credentials", "FAIL", f"AWS credential chain error: {type(exc).__name__}")
    if credentials is None:
        return Check("bedrock credentials", "FAIL",
                     f"no AWS credentials found: configure a profile (aws configure / aws sso login) or set {BEARER_TOKEN_ENV_VAR}")
    where = f"profile '{profile}'" if profile else "default chain"
    return Check("bedrock credentials", "PASS", f"AWS credential chain ({where}, {getattr(credentials, 'method', 'resolved')}; values not shown)")


# The probe asks for one concrete action with a target, in the same JSON
# shape the game's controllers validate, so it proves the model can carry
# structured action arguments - not just produce any JSON.
# 2026-10-02: Claude Opus 5.5 on Bedrock answered the earlier wording ("one
# spy in a turn-based game") with stopReason content_filtered and no text in
# 5 of 6 tries, failing 23 tournament preflights; it answered this wording,
# the game's own first line, normally in 5 of 5 (and game requests in play).
PROBE_SYSTEM_PROMPT = (
    "You control one spy in a competitive embassy escape game. Reply with a single JSON object and nothing else, "
    'shaped {"action": <one legal action name>, "target_id": <string or null>, "reason": <short string>}.'
)
PROBE_USER_PROMPT = json.dumps({
    "preflight": True,
    "legal_actions": [{"action": "wait"}, {"action": "move", "target_id": "probe_door"}],
    "instruction": "Choose the legal action move with target_id probe_door. Return JSON only.",
})
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": list(VALID_ACTION_NAMES)}, "target_id": {"type": ["string", "null"]},
                   "reason": {"type": "string"}},
    "required": ["action"],
}


@dataclass
class ProbeResult:
    provider: str
    model: str
    actors: list[str]
    ok: bool
    error_type: str | None
    error_message: str | None
    latency_ms: float | None
    returned_action: str | None
    returned_target: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_contract(client, spec: ActorSpec, actors: list[str]) -> ProbeResult:
    """One minimal request through the real client; checks the action contract strictly."""
    started = time.monotonic()
    try:
        response = client.generate_action(PROBE_SYSTEM_PROMPT, PROBE_USER_PROMPT, PROBE_SCHEMA, request_context=None)
    except Exception as exc:  # classified transport errors and anything unexpected
        return ProbeResult(spec.provider, spec.model, actors, False, getattr(exc, "category", type(exc).__name__),
                           redact(getattr(exc, "detail", None) or str(exc))[:300], round((time.monotonic() - started) * 1000.0, 1),
                           None, None)
    if not isinstance(response, ModelResponse):
        return ProbeResult(spec.provider, spec.model, actors, False, PROVIDER_INVALID_RESPONSE, "client returned no ModelResponse",
                           None, None, None)
    payload = response.payload if isinstance(response.payload, dict) else {}
    action = payload.get("action")
    target = payload.get("target_id")
    error_type = response.error_type
    message = redact(response.error_message)
    if error_type is None:
        if not isinstance(action, str) or action.strip().lower() not in VALID_ACTION_NAMES:
            error_type = "action_contract_unsupported"
            message = f"model output is not a valid game action (got {str(response.raw_text)[:120]!r})"
        elif action.strip().lower() != "move" or target != "probe_door":
            error_type = "action_contract_unsupported"
            message = f"model did not carry the requested action arguments (got action={action!r}, target_id={target!r})"
    return ProbeResult(spec.provider, spec.model, actors, error_type is None, error_type, message, response.latency_ms,
                       action if isinstance(action, str) else None, target if isinstance(target, str) else None)


def probe_hint(result: ProbeResult) -> str:
    """One actionable next step for a failed probe."""
    msg = (result.error_message or "").lower()
    if result.provider == "openai" and "temperature" in msg:
        return "this model rejects temperature 0: rerun with --openai-temperature provider-default"
    if result.provider == "bedrock" and "temperature" in msg:
        return "this model rejects the temperature setting: rerun with --bedrock-temperature provider-default"
    if result.error_type in (PROVIDER_AUTH_ERROR,):
        return "check your credentials (see workshop/setup.md, 'Live game')"
    if result.error_type == PROVIDER_ACCESS_ERROR:
        if result.provider == "bedrock":
            return "enable this model (or use its inference profile id, e.g. us.<vendor>.<model>) in the Bedrock console for this region"
        return "check the model id and that your OpenAI project can use it"
    if "model identifier is invalid" in msg or "model_not_found" in msg or "does not exist" in msg:
        return "the model id is not valid here: check the spelling, the region, and (Bedrock) whether it needs an inference profile id"
    if result.error_type == "action_contract_unsupported":
        return "this model cannot follow the game's JSON action contract; pick another model"
    return "see workshop/setup.md, 'Live game - troubleshooting'"


def run_probes(config: LiveConfig, client_factory: Callable[[ActorSpec, LiveConfig], Any]) -> list[ProbeResult]:
    """One probe per distinct (provider, model), shared by the spies that use it."""
    targets: dict[tuple[str, str], list[str]] = {}
    specs: dict[tuple[str, str], ActorSpec] = {}
    for spec in (config.black, config.white):
        if not spec.calls_provider:  # Stage 3: a surrogate or scripted spy calls no provider (a hybrid spy can)
            continue
        targets.setdefault(spec.target(), []).append(spec.label)
        specs.setdefault(spec.target(), spec)
    results = []
    for target, labels in targets.items():
        spec = specs[target]
        try:
            client = client_factory(spec, config)
        except Exception as exc:
            results.append(ProbeResult(spec.provider, spec.model, labels, False, "configuration_error", redact(str(exc))[:300],
                                       None, None, None))
            continue
        results.append(probe_contract(client, spec, labels))
    return results
