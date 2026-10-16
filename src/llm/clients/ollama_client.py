from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from src.llm.clients.base import ModelCapabilities, ModelResponse


def _extract_json_payload(raw_text: str | None) -> dict[str, Any]:
    if raw_text is None:
        return {}
    text = raw_text.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is not None:
        text = match.group(0)
    try:
        parsed = json.loads(text)
        # M5B.1: an empty dict is an honest "couldn't extract a usable
        # action" - never fabricate {"action": "wait"} here. The strict
        # validator in llm_controller.py is what decides how to classify
        # and (via repair/fallback) safely resolve a payload like this.
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        # Live feedback: prose around the JSON that itself contains braces
        # (or two JSON blocks) made the greedy match above undecodable and
        # the action was lost. Only when that already failed: take the
        # first complete JSON object in the text (a successful parse above
        # is unchanged).
        return _first_json_object(raw_text)


def _first_json_object(raw_text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed:
            return parsed
    return {}


class OllamaClient:
    """Provider adapter for a locally running Ollama server."""

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 8.0,
        request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        # M5E.1.1: optional context-window/residency controls (spec
        # sections 3-4, 16) - both None by default, meaning every
        # pre-M5E.1.1 caller gets byte-identical request payloads (Ollama's
        # own default num_ctx/keep_alive behavior, unchanged).
        num_ctx: int | None = None,
        keep_alive: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_fn = request_fn
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    def generate_action(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        started = time.monotonic()
        options: dict[str, Any] = {"temperature": 0.0}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": options,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if request_context:
            payload["request_context"] = request_context

        if self.request_fn is not None:
            raw = self.request_fn(payload)
        else:
            raw = self._request_http(payload)

        latency_ms = round((time.monotonic() - started) * 1000.0, 2)
        # M5B.1: none of these payloads fabricate {"action": "wait"} - an
        # empty dict is the honest "no usable action" signal; error_type/
        # error_message carry the diagnostic instead. See
        # llm_controller._validate_model_output for how this is classified.
        if not isinstance(raw, dict):
            return ModelResponse(
                payload={},
                raw_text=str(raw),
                provider=self.provider_name,
                model=self.model_name,
                latency_ms=latency_ms,
                error_type="provider_error",
                error_message="unexpected Ollama response",
                capabilities=ModelCapabilities(native_structured_output=True, token_usage=True),
            )

        if raw.get("error"):
            return ModelResponse(
                payload={},
                raw_text=None,
                provider=self.provider_name,
                model=self.model_name,
                latency_ms=latency_ms,
                error_type="provider_error",
                error_message=str(raw["error"]),
                capabilities=ModelCapabilities(native_structured_output=True),
            )

        message = raw.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        parsed_payload = _extract_json_payload(content)

        return ModelResponse(
            payload=parsed_payload if isinstance(parsed_payload, dict) else {},
            raw_text=content,
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=latency_ms,
            input_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
            finish_reason=raw.get("done_reason") or ("stop" if raw.get("done") else None),
            request_id=raw.get("id") or raw.get("created_at"),
            capabilities=ModelCapabilities(native_structured_output=True, token_usage=True),
            # M5E.1.1 spec section 14: Ollama's own timing/residency fields,
            # captured as-is (None where Ollama didn't report one - never
            # fabricated).
            provider_metadata={
                "done": raw.get("done"),
                "done_reason": raw.get("done_reason"),
                "total_duration": raw.get("total_duration"),
                "load_duration": raw.get("load_duration"),
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "prompt_eval_duration": raw.get("prompt_eval_duration"),
                "eval_count": raw.get("eval_count"),
                "eval_duration": raw.get("eval_duration"),
                "configured_num_ctx": self.num_ctx,
                "configured_keep_alive": self.keep_alive,
            },
        )

    def _request_http(self, payload: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        headers = {"Content-Type": "application/json"}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
