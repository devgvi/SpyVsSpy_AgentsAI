"""Offline / credential-isolation guard for report-only analyses (Stage 4B.2 onwards).

Unlike stage3_release.block_providers, it never touches credential variables (it does not even pop them): it
watches them. Activated before any analysis module is imported, it

  * wraps os.environ and records every access to a credential-like variable (OPENAI*, ANTHROPIC*, AWS_*, *BEDROCK*),
    never its value;
  * records every open of an AWS config / credentials file or a local key file;
  * refuses (and records) any import of a provider SDK (boto3, botocore, openai, anthropic, ollama), so no provider
    client can be constructed;
  * activates the workshop guard: every network lookup / connection is refused and recorded, and writes outside the
    allowed directories are recorded.

Only the standard library is imported at module level, so importing this file reads nothing.
"""
from __future__ import annotations

import importlib.abc
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Iterator

SENSITIVE_MARKERS = ("OPENAI", "ANTHROPIC", "AWS_", "BEDROCK")
BLOCKED_SDKS = ("boto3", "botocore", "openai", "anthropic", "ollama")
CREDENTIAL_PATH_MARKERS = (f"{os.sep}.aws{os.sep}", f"{os.sep}.aws", "openai_key", "anthropic_key", "credentials")


def _sensitive(name: Any) -> bool:
    return isinstance(name, str) and any(m in name.upper() for m in SENSITIVE_MARKERS)


class _EnvWatch(MutableMapping):
    """A transparent view of os.environ that records accesses to credential-like names (names only).

    With `absent` (scrub mode, Stage 4C) a lookup of a credential-like name that does not exist in the environment is
    recorded there instead: after the scrub there is no such variable, so the lookup can read nothing. Any write, and
    any read of a name that exists (e.g. one set after the scrub), is still a credential access."""

    def __init__(self, inner: MutableMapping, log: list[str], absent: list[str] | None = None) -> None:
        self._inner, self._log, self._absent = inner, log, absent

    def _note(self, key: Any, write: bool = False) -> None:
        if _sensitive(key):
            if self._absent is not None and not write and key not in self._inner:
                self._absent.append(str(key))
            else:
                self._log.append(str(key))

    def __getitem__(self, key: str) -> str:
        self._note(key)
        return self._inner[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._note(key, write=True)
        self._inner[key] = value

    def __delitem__(self, key: str) -> None:
        self._note(key, write=True)
        del self._inner[key]

    def __contains__(self, key: object) -> bool:
        self._note(key)
        return key in self._inner

    def get(self, key: str, default: Any = None) -> Any:
        self._note(key)
        return self._inner.get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)

    def copy(self) -> dict[str, str]:
        return dict(self._inner)


class _SdkBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".")[0] in BLOCKED_SDKS:
            self._log.append(fullname)
            raise ImportError(f"offline guard: importing the provider SDK {fullname!r} is refused")
        return None


WATCHED_CLIENT_MODULES = ("src.llm.clients", "workshop.live.providers")


class _ClientWatcher(importlib.abc.MetaPathFinder):
    """When a provider-client module is imported, every *Client class it defines (and build_client) is wrapped so that
    constructing one is recorded and refused. Importing the module stays allowed: it constructs nothing."""

    def __init__(self, guard: "OfflineGuard") -> None:
        self._guard = guard

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if not fullname.startswith(WATCHED_CLIENT_MODULES):
            return None
        from importlib.machinery import PathFinder
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        original = spec.loader.exec_module

        def exec_module(module: Any, _orig: Any = original) -> None:
            _orig(module)
            self._guard._wrap_clients(module)
        spec.loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


class OfflineGuard:
    def __init__(self) -> None:
        self.env_accessed: list[str] = []
        self.env_absent_lookups: list[str] = []  # scrub mode only: lookups of a credential-like name that does not exist
        self.credential_files_opened: list[str] = []
        self.sdk_imports_refused: list[str] = []
        self.clients_constructed: list[str] = []
        self.scrubbed: list[str] | None = None
        self.active = False
        self._allowed: list[Path] = []

    def _wrap_clients(self, module: Any) -> None:
        guard = self

        def refuse(name: str) -> Any:
            def _raise(*args: Any, **kwargs: Any) -> Any:
                guard.clients_constructed.append(name)
                raise RuntimeError(f"offline guard: constructing {name} is refused")
            return _raise
        for name, obj in list(vars(module).items()):
            if isinstance(obj, type) and obj.__module__ == module.__name__ and name.endswith("Client"):
                obj.__init__ = refuse(f"{module.__name__}.{name}")  # type: ignore[misc]
            elif name == "build_client" and callable(obj):
                setattr(module, name, refuse(f"{module.__name__}.build_client"))

    def _hook(self, event: str, args: tuple) -> None:
        if not self.active or event != "open" or not args:
            return
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            s = os.fsdecode(p)
            if any(m in s for m in CREDENTIAL_PATH_MARKERS):
                self.credential_files_opened.append(s)

    def activate(self, allowed_dirs: list[Path], *, scrub: bool = False) -> "OfflineGuard":
        """scrub (Stage 4C): first delete every credential-like variable from this process's environment, by name
        (the value is never read), so nothing in the process - including bulk copies such as
        platform._syscmd_file's dict(os.environ) during pygame.init - can see one. A later lookup of such a name
        (e.g. os.environ.get("AWS_PROFILE")) can then read nothing: it is recorded apart, as an absent lookup; a write,
        or a read of a name that exists again, stays a credential access."""
        if scrub:
            names = [k for k in list(os.environ.keys()) if _sensitive(k)]
            for k in names:
                del os.environ[k]
            self.scrubbed = sorted(names)
        preloaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED_SDKS)
        if preloaded:
            raise RuntimeError(f"offline guard: provider SDKs already imported before the guard: {preloaded}")
        preloaded_clients = sorted(m for m in sys.modules if m.startswith(WATCHED_CLIENT_MODULES))
        if preloaded_clients:
            raise RuntimeError(f"offline guard: provider-client modules already imported before the guard: {preloaded_clients}")
        os.environ = _EnvWatch(os.environ, self.env_accessed, self.env_absent_lookups if scrub else None)  # type: ignore[assignment]
        sys.meta_path.insert(0, _SdkBlocker(self.sdk_imports_refused))
        sys.meta_path.insert(1, _ClientWatcher(self))
        sys.addaudithook(self._hook)
        from workshop.runners.common import GUARD, install_guard  # imported only now, under the watch
        install_guard()
        GUARD.activate(allowed_dirs, allowed_hosts=())
        self._allowed = list(allowed_dirs)
        self.active = True
        return self

    def record(self) -> dict[str, Any]:
        from workshop.runners.common import GUARD
        provider_modules = sorted(m for m in sys.modules if m.startswith(("workshop.live.providers", "src.llm.clients")))
        extra = {"credential_env_removed_by_name": self.scrubbed,
                 "credential_env_absent_lookups": sorted(set(self.env_absent_lookups))} if self.scrubbed is not None else {}
        return {**extra, "credential_env_accessed": sorted(set(self.env_accessed)),
                "credential_files_opened": sorted(set(self.credential_files_opened)),
                "provider_sdk_imports_refused": sorted(set(self.sdk_imports_refused)),
                "provider_sdks_loaded": sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED_SDKS),
                "provider_clients_constructed": list(self.clients_constructed),
                "provider_client_modules_imported_not_used": provider_modules,
                "network_attempts": list(GUARD.network_attempts),
                "writes_outside_output": [w for w in GUARD.unexpected_writes if os.devnull not in w],
                "provider_calls": 0}

    @staticmethod
    def clean(record: dict[str, Any]) -> bool:
        """Importing a client module is allowed (it constructs nothing); everything else must be empty."""
        return not any(record[k] for k in ("credential_env_accessed", "credential_files_opened", "provider_sdk_imports_refused",
                                           "provider_sdks_loaded", "provider_clients_constructed", "network_attempts",
                                           "writes_outside_output"))


__all__ = ["OfflineGuard", "SENSITIVE_MARKERS", "BLOCKED_SDKS"]
