"""M5E.3 spec sections 25, 28, 46: network-free tests for the Bedrock
provider - construction, token-presence handling, region/model
resolution, Converse request/response normalization, the full error
taxonomy, and (critically) secret-redaction proving a fake bearer token
never appears anywhere in logs/exceptions/telemetry/repr output. All
Bedrock SDK calls are mocked via `client_factory` - no real AWS network
access anywhere in this file.
"""

from __future__ import annotations

import os

import botocore.exceptions as be
import pytest

from src.game import Game
from src.llm.clients.bedrock_client import (
    BEARER_TOKEN_ENV_VAR,
    BedrockClient,
    BedrockConfigurationError,
    bedrock_token_present,
    classify_bedrock_exception,
    redact_bedrock_token,
    resolve_bedrock_region,
    run_bedrock_preflight,
)
from src.llm_controller import (
    PROVIDER_ACCESS_ERROR,
    PROVIDER_AUTH_ERROR,
    PROVIDER_INVALID_RESPONSE,
    PROVIDER_SERVICE_ERROR,
    PROVIDER_THROTTLING,
    PROVIDER_TIMEOUT,
    PROVIDER_TRANSPORT_ERROR,
    RUNTIME_EXCEPTION,
    classify_provider_call_error,
)

FAKE_TOKEN = "fake-bearer-token-do-not-use-abc123XYZ"


@pytest.fixture(autouse=True)
def _clean_bedrock_env(monkeypatch):
    monkeypatch.delenv(BEARER_TOKEN_ENV_VAR, raising=False)
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)


class _FakeConverseClient:
    def __init__(self, response=None, exception=None):
        self._response = response
        self._exception = exception
        self.last_kwargs = None

    def converse(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exception is not None:
            raise self._exception
        return self._response


def _ok_converse_response(text='{"action": "wait"}', input_tokens=100, output_tokens=10):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens, "totalTokens": input_tokens + output_tokens},
        "metrics": {"latencyMs": 450},
    }


# -- token presence / region resolution --------------------------------


def test_bedrock_token_present_false_when_unset():
    assert bedrock_token_present() is False


def test_bedrock_token_present_true_when_set(monkeypatch):
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    assert bedrock_token_present() is True


def test_resolve_region_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert resolve_bedrock_region("us-west-2") == "us-west-2"


def test_resolve_region_falls_back_to_bedrock_region_env(monkeypatch):
    monkeypatch.setenv("BEDROCK_REGION", "ap-southeast-1")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert resolve_bedrock_region() == "ap-southeast-1"


def test_resolve_region_falls_back_to_aws_region_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    assert resolve_bedrock_region() == "eu-central-1"


def test_resolve_region_raises_clearly_when_nothing_resolves(monkeypatch):
    monkeypatch.setattr("boto3.Session", lambda: type("S", (), {"region_name": None})())
    with pytest.raises(BedrockConfigurationError):
        resolve_bedrock_region()


# -- model ID is fully configurable (spec section 37) -----------------------


def test_model_id_is_configurable_no_hardcoding():
    client_a = BedrockClient(model_name="model-a", region="us-east-1", client_factory=lambda: _FakeConverseClient(_ok_converse_response()))
    client_b = BedrockClient(model_name="model-b", region="us-east-1", client_factory=lambda: _FakeConverseClient(_ok_converse_response()))
    assert client_a.model_name == "model-a"
    assert client_b.model_name == "model-b"


# -- Converse request construction -------------------------------------


def test_converse_request_shape():
    fake = _FakeConverseClient(_ok_converse_response())
    client = BedrockClient(model_name="my-model", region="us-east-1", client_factory=lambda: fake)
    client.generate_action("system instructions", "user payload json", {})

    assert fake.last_kwargs["modelId"] == "my-model"
    assert fake.last_kwargs["system"] == [{"text": "system instructions"}]
    assert fake.last_kwargs["messages"] == [{"role": "user", "content": [{"text": "user payload json"}]}]
    assert fake.last_kwargs["inferenceConfig"]["temperature"] == 0.0


# -- Converse response normalization -------------------------------------


def test_converse_response_normalization_success():
    fake = _FakeConverseClient(_ok_converse_response(text='{"action": "move", "target_id": "door_1"}', input_tokens=250, output_tokens=15))
    client = BedrockClient(model_name="m", region="us-east-1", client_factory=lambda: fake)
    response = client.generate_action("sys", "user", {})

    assert response.payload == {"action": "move", "target_id": "door_1"}
    assert response.provider == "bedrock"
    assert response.input_tokens == 250
    assert response.output_tokens == 15
    assert response.finish_reason == "end_turn"
    assert response.error_type is None
    assert response.provider_metadata["total_tokens"] == 265
    assert response.provider_metadata["region"] == "us-east-1"


def test_converse_response_with_no_parseable_json_is_invalid_response_not_a_fabricated_wait():
    fake = _FakeConverseClient(_ok_converse_response(text="I cannot help with that."))
    client = BedrockClient(model_name="m", region="us-east-1", client_factory=lambda: fake)
    response = client.generate_action("sys", "user", {})

    assert response.payload == {}
    assert response.error_type == PROVIDER_INVALID_RESPONSE
    assert classify_provider_call_error(response.error_type, response.error_message) == PROVIDER_INVALID_RESPONSE


# -- error taxonomy (spec section 21) ------------------------------------


@pytest.mark.parametrize(
    "code,expected_category",
    [
        ("AccessDeniedException", PROVIDER_ACCESS_ERROR),
        ("UnauthorizedException", PROVIDER_AUTH_ERROR),
        ("ExpiredTokenException", PROVIDER_AUTH_ERROR),
        ("ThrottlingException", PROVIDER_THROTTLING),
        ("ServiceQuotaExceededException", PROVIDER_THROTTLING),
        ("ModelTimeoutException", PROVIDER_TIMEOUT),
        ("ServiceUnavailableException", PROVIDER_SERVICE_ERROR),
        ("InternalServerException", PROVIDER_SERVICE_ERROR),
        ("ValidationException", PROVIDER_SERVICE_ERROR),
        ("SomeBrandNewExceptionAWSAddsLater", PROVIDER_SERVICE_ERROR),  # never runtime_exception (spec section 21)
    ],
)
def test_client_error_codes_map_into_the_technical_taxonomy(code, expected_category):
    exc = be.ClientError({"Error": {"Code": code, "Message": "details"}}, "Converse")
    category, _ = classify_bedrock_exception(exc)
    assert category == expected_category


def test_expired_bearer_token_access_denied_reclassifies_to_auth_error():
    """M5E.3.1: real live evidence (M5E.3's second-day run) showed Bedrock
    reports an expired bearer token as `AccessDeniedException: Bearer Token
    has expired` - the exact code the taxonomy otherwise maps to
    PROVIDER_ACCESS_ERROR (a permissions problem, not a credential
    problem). This message shape must reclassify to PROVIDER_AUTH_ERROR so
    the multi-seed runner's mid-run auth-expiry stop rule can detect it."""
    exc = be.ClientError({"Error": {"Code": "AccessDeniedException", "Message": "Bearer Token has expired"}}, "Converse")
    category, _ = classify_bedrock_exception(exc)
    assert category == PROVIDER_AUTH_ERROR


def test_generic_access_denied_without_token_wording_stays_access_error():
    """A real IAM-permissions AccessDeniedException (no "token"/"expired"
    wording) must not be swept into the auth-error reclassification above -
    it's a genuinely different fix (grant access vs. refresh credential)."""
    exc = be.ClientError({"Error": {"Code": "AccessDeniedException", "Message": "User is not authorized to perform this action"}}, "Converse")
    category, _ = classify_bedrock_exception(exc)
    assert category == PROVIDER_ACCESS_ERROR


def test_no_credentials_error_maps_to_auth_error():
    category, message = classify_bedrock_exception(be.NoCredentialsError())
    assert category == PROVIDER_AUTH_ERROR


def test_connect_timeout_maps_to_provider_timeout():
    exc = be.ConnectTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
    category, _ = classify_bedrock_exception(exc)
    assert category == PROVIDER_TIMEOUT


def test_endpoint_connection_error_maps_to_transport_error():
    exc = be.EndpointConnectionError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
    category, _ = classify_bedrock_exception(exc)
    assert category == PROVIDER_TRANSPORT_ERROR


def test_unexpected_exception_maps_to_runtime_exception():
    category, _ = classify_bedrock_exception(ValueError("something truly unexpected"))
    assert category == RUNTIME_EXCEPTION


def test_generate_action_surfaces_classified_errors_not_raw_exceptions():
    exc = be.ClientError({"Error": {"Code": "ThrottlingException", "Message": "too many requests"}}, "Converse")
    fake = _FakeConverseClient(exception=exc)
    client = BedrockClient(model_name="m", region="us-east-1", client_factory=lambda: fake)
    response = client.generate_action("sys", "user", {})
    assert response.error_type == PROVIDER_THROTTLING
    assert response.payload == {}
    assert classify_provider_call_error(response.error_type, response.error_message) == PROVIDER_THROTTLING


# -- secret-safety tests (spec section 25) - the most important tests here --


def test_redact_bedrock_token_replaces_literal_token_value(monkeypatch):
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    message = f"request failed with header Authorization: Bearer {FAKE_TOKEN}"
    redacted = redact_bedrock_token(message)
    assert FAKE_TOKEN not in redacted
    assert "[redacted]" in redacted


def test_redact_bedrock_token_noop_when_token_not_set():
    assert redact_bedrock_token("no token here") == "no token here"


def test_redact_bedrock_token_handles_none():
    assert redact_bedrock_token(None) is None


def test_fake_token_never_appears_in_a_classified_exception_message(monkeypatch):
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    exc = be.ClientError({"Error": {"Code": "AccessDeniedException", "Message": f"token {FAKE_TOKEN} rejected"}}, "Converse")
    category, message = classify_bedrock_exception(exc)
    assert FAKE_TOKEN not in message


def test_fake_token_never_appears_in_generate_action_response_or_repr(monkeypatch):
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    exc = be.ClientError({"Error": {"Code": "UnauthorizedException", "Message": f"bad token: {FAKE_TOKEN}"}}, "Converse")
    fake = _FakeConverseClient(exception=exc)
    client = BedrockClient(model_name="m", region="us-east-1", client_factory=lambda: fake)
    response = client.generate_action("sys", "user", {})

    assert FAKE_TOKEN not in repr(response)
    assert FAKE_TOKEN not in str(response)
    assert FAKE_TOKEN not in (response.error_message or "")


def test_fake_token_never_appears_in_client_repr_or_dict(monkeypatch):
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    client = BedrockClient(model_name="m", region="us-east-1", client_factory=lambda: _FakeConverseClient(_ok_converse_response()))
    assert FAKE_TOKEN not in repr(client)
    assert FAKE_TOKEN not in str(client.__dict__)


def test_fake_token_never_appears_in_preflight_result(monkeypatch):
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    result = run_bedrock_preflight(model_id="m", region="us-east-1", client_factory=lambda: _FakeConverseClient(_ok_converse_response()))
    result_dict = result.to_dict()
    assert FAKE_TOKEN not in str(result_dict)
    assert result_dict["token_present"] is True
    assert "token" not in {k for k in result_dict if "value" in k.lower()}  # only token_present, never the value


def test_fake_token_never_appears_when_it_leaks_into_a_raw_response_text(monkeypatch):
    # Defense in depth: even if a raw model response somehow echoed the
    # token in its *text* content (not expected, but never assumed
    # impossible), raw_text must still come back redacted.
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    fake = _FakeConverseClient(_ok_converse_response(text=f"here is data: {FAKE_TOKEN} {{\"action\": \"wait\"}}"))
    client = BedrockClient(model_name="m", region="us-east-1", client_factory=lambda: fake)
    response = client.generate_action("sys", "user", {})
    assert FAKE_TOKEN not in (response.raw_text or "")


# -- preflight (spec sections 7, 24, 29) ---------------------------------


def test_preflight_fails_safely_when_token_missing():
    result = run_bedrock_preflight(model_id="m", region="us-east-1")
    assert result.ok is False
    assert result.token_present is False
    assert result.error_type == PROVIDER_AUTH_ERROR


def test_preflight_fails_clearly_when_region_unresolvable(monkeypatch):
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    monkeypatch.setattr("boto3.Session", lambda: type("S", (), {"region_name": None})())
    result = run_bedrock_preflight(model_id="m", region=None)
    assert result.ok is False
    assert result.error_type == "configuration_error"


def test_preflight_succeeds_with_valid_token_and_working_converse(monkeypatch):
    monkeypatch.setenv(BEARER_TOKEN_ENV_VAR, FAKE_TOKEN)
    result = run_bedrock_preflight(model_id="m", region="us-east-1", client_factory=lambda: _FakeConverseClient(_ok_converse_response()))
    assert result.ok is True
    assert result.token_present is True
    assert result.bedrock_auth_mode == "bearer_token_env"


# -- Game/provider-registry wiring (spec section 46) ----------------------


def test_game_construction_selects_bedrock_client_for_provider_bedrock():
    game = Game(white_controller="llm", agent_runtime="native_tool_like", llm_provider="bedrock", llm_region="us-east-1", llm_model="my-model")
    client = game.bot_controller.controller.model_client
    assert isinstance(client, BedrockClient)
    assert client.model_name == "my-model"
    assert client.region == "us-east-1"


def test_game_construction_fails_clearly_with_no_region_configured():
    with pytest.raises(BedrockConfigurationError):
        Game(white_controller="llm", agent_runtime="native_tool_like", llm_provider="bedrock")


def test_game_construction_with_ollama_provider_is_unaffected_by_bedrock_changes():
    game = Game(white_controller="llm", agent_runtime="native_tool_like", llm_provider="ollama")
    from src.llm.clients.ollama_client import OllamaClient

    assert isinstance(game.bot_controller.controller.model_client, OllamaClient)


# -- live: <invoke> markup answers (Claude Sonnet 5.5, pilot tournament 2026-10-02) --

_SONNET55_ANSWER = (
    "I'll start by searching the nearest reception object, sofa_01, which costs only 0.4 s to reach.\n"
    '<invoke name="choose_action">\n<parameter name="action">MOVE</parameter>\n'
    '<parameter name="target_id">sofa_01</parameter>\n'
    '<parameter name="reason">Nearest searchable object (0.4s); 480s left &amp; relaxed.</parameter>\n</invoke>'
)


def test_invoke_markup_answer_is_read_only_when_the_live_client_accepts_it():
    live = BedrockClient(model_name="m", region="us-east-1", accept_invoke_markup=True,
                         client_factory=lambda: _FakeConverseClient(_ok_converse_response(_SONNET55_ANSWER)))
    response = live.generate_action("s", "u", {})
    assert response.error_type is None
    assert response.payload == {"action": "MOVE", "target_id": "sofa_01",
                                "reason": "Nearest searchable object (0.4s); 480s left & relaxed."}
    assert response.provider_metadata["parsed_from"] == "invoke_markup"
    recorded = BedrockClient(model_name="m", region="us-east-1",
                             client_factory=lambda: _FakeConverseClient(_ok_converse_response(_SONNET55_ANSWER)))
    assert recorded.generate_action("s", "u", {}).payload == {}  # recorded clients: unchanged, still invalid
    # JSON still wins when both are present; another function name or no action is not an answer
    both = BedrockClient(model_name="m", region="us-east-1", accept_invoke_markup=True,
                         client_factory=lambda: _FakeConverseClient(_ok_converse_response(
                             '{"action": "wait"} ' + _SONNET55_ANSWER)))
    assert both.generate_action("s", "u", {}).payload == {"action": "wait"}
    from src.llm.clients.bedrock_client import invoke_markup_payload
    assert invoke_markup_payload('<invoke name="other"><parameter name="action">wait</parameter></invoke>') == {}
    assert invoke_markup_payload('<invoke name="choose_action"><parameter name="reason">x</parameter></invoke>') == {}
    assert invoke_markup_payload('<invoke name="choose_action"><parameter name="action">wait</parameter>'
                                 '<parameter name="target_id">null</parameter></invoke>') == {"action": "wait", "target_id": None}
