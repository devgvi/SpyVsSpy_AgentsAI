"""M6G live-call plumbing: a secret-safe, profile-aware Bedrock preflight
and the Native / Strands deciders used by the live phases.

Why a separate preflight: `bedrock_client.run_bedrock_preflight` requires
the AWS_BEARER_TOKEN_BEDROCK env var and reports its auth mode as the
literal "bearer_token_env". M6G's live runs authenticate through a named
AWS profile instead (botocore's standard SigV4 credential chain), which
that function refuses without trying. Rather than change production code
during an evaluation (spec section 45), this module delegates to the
existing function whenever a bearer token is present - so that path is
byte-for-byte unchanged - and otherwise checks that the credential chain
resolves and makes one tiny ping call through the real `BedrockClient`.

Secret safety (spec section 52): nothing here reads, stores or prints a
credential value. Only presence booleans, the profile NAME, the region and
redacted error text leave these functions.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Callable

from src.evaluation.agent_contract import AgentRequest, strands_model_payload
from src.evaluation.decision_scheduling import native_render
from src.llm.clients.bedrock_client import (
    BedrockClient,
    bedrock_token_present,
    redact_bedrock_token,
    run_bedrock_preflight,
)
from src.llm_controller import VALID_ACTION_NAMES, normalize_provider_error_message

AUTH_BEARER = "bearer_token_env"
AUTH_PROFILE = "aws_profile"
AUTH_CHAIN = "aws_credential_chain"

# Output contract hint passed to BedrockClient.generate_action - unchanged
# from every M6-family Native driver (BedrockClient itself does not send it).
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(VALID_ACTION_NAMES)},
        "target_id": {"type": ["string", "null"]},
        "trap_type": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["action"],
}


def _safe_message(message: str | None) -> str | None:
    return normalize_provider_error_message(redact_bedrock_token(message))


def resolve_auth_mode(env: dict[str, str] | None = None, session_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Which credential source botocore will use. botocore prefers the
    bearer token for Bedrock whenever it is set, even if a profile is also
    set - so the token is checked first, exactly in that precedence."""
    env = os.environ if env is None else env
    token_present = bool(env.get("AWS_BEARER_TOKEN_BEDROCK"))
    profile = env.get("AWS_DEFAULT_PROFILE") or env.get("AWS_PROFILE")
    if token_present:
        return {"bedrock_auth_mode": AUTH_BEARER, "aws_profile": profile, "token_present": True, "credentials_resolved": True}
    mode = AUTH_PROFILE if profile else AUTH_CHAIN
    try:
        if session_factory is None:
            import boto3

            session_factory = boto3.Session
        session = session_factory(profile_name=profile) if profile else session_factory()
        resolved = session.get_credentials() is not None
    except Exception:  # a missing/misnamed profile is "not resolved", never a crash
        resolved = False
    return {"bedrock_auth_mode": mode, "aws_profile": profile, "token_present": False, "credentials_resolved": resolved}


def run_live_preflight(
    model_id: str,
    region: str,
    timeout: float = 60.0,
    *,
    env: dict[str, str] | None = None,
    session_factory: Callable[..., Any] | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    auth = resolve_auth_mode(env=env, session_factory=session_factory)
    base = {"region": region, "model_id": model_id, **auth}
    if auth["bedrock_auth_mode"] == AUTH_BEARER:
        result = run_bedrock_preflight(model_id=model_id, region=region, timeout=timeout).to_dict()
        return {**result, **base}
    if not auth["credentials_resolved"]:
        return {
            **base, "ok": False, "error_type": "provider_auth_error",
            "error_message": "no AWS credentials resolved from the profile / default credential chain",
            "latency_ms": None, "input_tokens": None, "output_tokens": None,
        }
    client = client_factory() if client_factory is not None else BedrockClient(model_name=model_id, region=region, timeout=timeout)
    response = client.generate_action(
        "You are a connectivity check. Reply with JSON only.",
        'Reply with exactly {"action": "wait"}.',
        OUTPUT_SCHEMA,
    )
    ok = response.error_type is None and isinstance(response.payload, dict) and bool(response.payload.get("action"))
    return {
        **base, "ok": ok, "error_type": response.error_type, "error_message": _safe_message(response.error_message),
        "latency_ms": response.latency_ms, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
    }


OUT_DIR = os.path.join("evaluation_runs", "m6g-decision-scheduling-v1")


def out_path(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    """One row per completed trial, flushed immediately, so an interrupted
    live run loses at most the trial in flight and can resume."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def provider_failure(row: dict[str, Any]) -> str | None:
    """A provider-side failure (auth, throttling, timeout, service error)
    for any actor in a trial row, or None. Malformed/tool-less model
    output is not a provider failure - it is scored as first-pass invalid."""
    from src.evaluation.decision_scheduling import MODEL_OUTPUT_ERROR_TYPES

    for rec in row["records"].values():
        err = rec.get("provider_error_type")
        if err is not None and err not in MODEL_OUTPUT_ERROR_TYPES:
            return f"{rec['actor_id']}: {err}: {rec.get('provider_error_message')}"
    return None


def verify_sleep_assertion() -> bool:
    """Same guard every live M6 runner uses: refuse to start unless
    `caffeinate` holds a PreventUserIdleSystemSleep assertion."""
    try:
        output = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=10.0).stdout
    except Exception:  # pragma: no cover
        return False
    return "PreventUserIdleSystemSleep" in output and "caffeinate" in output


def rp_manifest_clean() -> bool:
    from src.evaluation.runtime_parity import FAIL, RUNTIME_PARITY_MANIFEST

    return all(e.status != FAIL for e in RUNTIME_PARITY_MANIFEST)


class NativeDecider:
    """One real `native_tool_like` Bedrock decision per call, through the
    production `BedrockClient`. A fresh client per call: no connection
    pool, client object or response buffer is shared between actors."""

    runtime = "native"

    def __init__(self, region: str, model_id: str, timeout: float = 60.0, client_factory: Callable[[], Any] | None = None) -> None:
        self.region = region
        self.model_id = model_id
        self.timeout = timeout
        self._client_factory = client_factory or (lambda: BedrockClient(model_name=model_id, region=region, timeout=timeout))

    def render(self, request: AgentRequest) -> tuple[str, str]:
        return native_render(request)

    def call(self, request: AgentRequest, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self._client_factory().generate_action(system_prompt, user_prompt, OUTPUT_SCHEMA)
        return {
            "raw": response.payload if isinstance(response.payload, dict) else {},
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "error_type": response.error_type,
            "error_message": _safe_message(response.error_message),
            "stop_reason": response.finish_reason,
        }


class StrandsDecider:
    """One real Strands SDK decision per call through the production
    `StrandsAgentAdapter._invoke_strands` (a fresh adapter, and therefore a
    fresh strands.Agent + BedrockModel, per call). Only the first-pass
    proposal is used - no repair, no fallback policy (spec section 36)."""

    runtime = "strands"

    def __init__(self, region: str, model_id: str, timeout: float = 60.0) -> None:
        self.region = region
        self.model_id = model_id
        self.timeout = timeout

    def _adapter(self, request: AgentRequest):
        from src.agents.registry import create_agent_runtime
        from src.llm.telemetry import TelemetryLogger

        return create_agent_runtime(
            "strands", actor_id=request.actor_id, telemetry=TelemetryLogger(), episode_id=request.episode_id,
            provider_name="bedrock", model_name=self.model_id, execution_mode="evaluation", llm_region=self.region,
            llm_timeout=self.timeout, personality_profile=request.personality_policy,
        )

    def render(self, request: AgentRequest) -> tuple[str, str]:
        # `_invoke_strands` serializes the payload exactly like this
        # (json.dumps(dict(model_payload), sort_keys=True)).
        return self._adapter(request)._effective_system_prompt, json.dumps(strands_model_payload(request), sort_keys=True)

    def call(self, request: AgentRequest, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        adapter = self._adapter(request)
        if adapter._effective_system_prompt != system_prompt:
            raise RuntimeError("Strands system prompt changed between render and call")
        invocation = adapter._invoke_strands(strands_model_payload(request), repair_instruction=None)
        return {
            "raw": invocation.proposal if isinstance(invocation.proposal, dict) else {},
            "latency_ms": round(invocation.latency_ms, 2) if invocation.latency_ms is not None else None,
            "input_tokens": invocation.input_tokens,
            "output_tokens": invocation.output_tokens,
            "error_type": invocation.error_type,
            "error_message": _safe_message(invocation.error_message),
            "tool_call_count": invocation.tool_call_count,
            "proposal_count": invocation.proposal_count,
            "stop_reason": invocation.stop_reason,
        }
