"""Public-tree test configuration.

Some tests of the engine and runtime suites also exercise the frozen research archive (raw sessions, reports,
registries), which is distributed separately. They are skipped here with that reason. Tests that need a local Ollama
server with `qwen:latest` are skipped unless SPY_VS_AI_OLLAMA_TESTS=1 is set. Both lists come from
PUBLIC_EXPORT_PROVENANCE.json (written by the export from PUBLIC_EXPORT_MANIFEST.yaml in the archive).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_PROVENANCE = ROOT / "PUBLIC_EXPORT_PROVENANCE.json"
_DATA = json.loads(_PROVENANCE.read_text(encoding="utf-8")) if _PROVENANCE.is_file() else {}
ARCHIVE_ONLY: dict[str, str] = _DATA.get("tests_archive_only") or {}
ENVIRONMENT_DEPENDENT: dict[str, str] = _DATA.get("tests_environment_dependent") or {}


def _lookup(table: dict[str, str], nodeid: str) -> str | None:
    base = nodeid.split("[", 1)[0]
    return table.get(nodeid) or table.get(base) or table.get(base.split("::", 1)[0])


def pytest_collection_modifyitems(config, items):
    ollama = os.environ.get("SPY_VS_AI_OLLAMA_TESTS") == "1"
    for item in items:
        reason = _lookup(ARCHIVE_ONLY, item.nodeid)
        if reason:
            item.add_marker(pytest.mark.skip(reason=f"needs the research archive: {reason}"))
            continue
        reason = _lookup(ENVIRONMENT_DEPENDENT, item.nodeid)
        if reason and not ollama:
            item.add_marker(pytest.mark.skip(reason=f"{reason} (set SPY_VS_AI_OLLAMA_TESTS=1 to run it)"))
