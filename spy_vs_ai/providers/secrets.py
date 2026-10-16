"""Credential-scoped redaction (FINAL-2 spec 11, 26, 45).

The live game's `workshop.live.providers.redact` reads every known secret variable (OpenAI and AWS) on each call.
The runtime instead scrubs only the secret values of the providers selected for this match, so a Bedrock match never
reads the OpenAI or Anthropic variable and a call-free match reads none. Key-shaped tokens, Bearer / x-api-key
headers and credential-looking JSON fields are removed for every match regardless of the provider.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

# the secret-valued variables of each provider (AWS_PROFILE / regions are names, not secrets)
SECRET_ENV = {
    "bedrock": ("AWS_BEARER_TOKEN_BEDROCK", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_ACCESS_KEY_ID"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
    "fake": (),
}
KEY_SHAPED_RE = re.compile(r"\b(sk|rk)-[A-Za-z0-9_\-*]{6,}|\b(AKIA|ASIA)[A-Z0-9]{8,}|\bABSK[A-Za-z0-9+/=_\-]{8,}")
BEARER_RE = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)
HEADER_RE = re.compile(r"((?:x-api-key|authorization)\s*[:=]\s*)\S+", re.IGNORECASE)
SECRET_FIELD_RE = re.compile(r"(api[_-]?key|secret|token|password|authorization|credential|access[_-]?key)", re.IGNORECASE)
# token counts are not secrets: input_tokens / output_tokens / cached_input_tokens / total_tokens
_NOT_SECRET_FIELDS = re.compile(r"(_tokens$|^tokens$|_tokens_|token_usage|^max_tokens$)", re.IGNORECASE)


class Scrubber:
    """Removes the selected providers' live secret values and every key-shaped string from text / JSON values."""

    def __init__(self, providers: Iterable[str]) -> None:
        self.providers = tuple(sorted({p for p in providers if p}))
        self.env_names = tuple(n for p in self.providers for n in SECRET_ENV.get(p, ()))

    def values(self) -> list[str]:
        return [v for v in (os.environ.get(n) for n in self.env_names) if v and len(v) >= 8]

    def text(self, text: str | None) -> str | None:
        if text is None:
            return None
        for value in self.values():
            text = text.replace(value, "[redacted]")
        text = KEY_SHAPED_RE.sub("[redacted-key]", text)
        text = HEADER_RE.sub(r"\1[redacted]", text)
        return BEARER_RE.sub(r"\1[redacted]", text)

    def value(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, str):
            if key and SECRET_FIELD_RE.search(key) and not _NOT_SECRET_FIELDS.search(key):
                return "[redacted]"
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.value(v, str(k)) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self.value(v) for v in value]
        return value

    def json(self, value: Any) -> str:
        return self.text(json.dumps(self.value(value), indent=1, sort_keys=True, default=str))


def leaks(text: str, names: Iterable[str] = ()) -> list[str]:
    """What looks like a credential in `text` (tests and the replay writer use it): key-shaped tokens, Bearer
    values and the current values of the given environment variables."""
    found = [m.group(0)[:12] + "..." for m in KEY_SHAPED_RE.finditer(text)]
    found += [m.group(0)[:20] for m in BEARER_RE.finditer(text) if "[redacted]" not in m.group(0)]
    for name in names:
        value = os.environ.get(name)
        if value and len(value) >= 8 and value in text:
            found.append(f"value of {name}")
    return found


__all__ = ["Scrubber", "SECRET_ENV", "leaks"]
