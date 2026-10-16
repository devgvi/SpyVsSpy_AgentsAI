"""M7 workshop live game: optional credential-gated smoke tests (real provider calls).

Skipped unless explicitly enabled - they cost a few requests:

    SPY_LIVE_SMOKE_BEDROCK_MODEL=<bedrock-model-id> python -m pytest tests/test_workshop_live_smoke.py
    SPY_LIVE_SMOKE_OPENAI_MODEL=<openai-model-id>   python -m pytest tests/test_workshop_live_smoke.py   # needs OPENAI_API_KEY

Each runs the live preflight (credentials + action-contract probe) for the
selected model, then one short headless LLM-vs-LLM match.
"""
from __future__ import annotations

import os

import pytest

from tests.workshop_live_helpers import FAKE_OPENAI_KEY  # noqa: F401 - also sets the headless SDL drivers
from workshop.live.config import resolve_config
from workshop.live.providers import build_client, credential_checks, run_probes
from workshop.live.session import MatchRunner


def _bedrock_ready() -> bool:
    try:
        import boto3
        return boto3.Session().get_credentials() is not None or bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
    except Exception:
        return False


CASES = [
    pytest.param("bedrock", os.environ.get("SPY_LIVE_SMOKE_BEDROCK_MODEL"),
                 marks=pytest.mark.skipif(not (os.environ.get("SPY_LIVE_SMOKE_BEDROCK_MODEL") and _bedrock_ready()),
                                          reason="set SPY_LIVE_SMOKE_BEDROCK_MODEL and AWS credentials"), id="bedrock"),
    pytest.param("openai", os.environ.get("SPY_LIVE_SMOKE_OPENAI_MODEL"),
                 marks=pytest.mark.skipif(not (os.environ.get("SPY_LIVE_SMOKE_OPENAI_MODEL") and os.environ.get("OPENAI_API_KEY")),
                                          reason="set SPY_LIVE_SMOKE_OPENAI_MODEL and OPENAI_API_KEY"), id="openai"),
]


@pytest.mark.parametrize("provider, model", CASES)
def test_selected_model_passes_preflight_and_plays(provider, model, tmp_path):
    config = resolve_config({"provider": provider, "model": model})
    assert all(c.status != "FAIL" for c in credential_checks(config))
    assert all(p.ok for p in run_probes(config, build_client))
    summary = MatchRunner(config, tmp_path, headless=True, max_decisions=2, max_wall_seconds=240).run()
    for actor in summary["actors"].values():
        assert actor["decisions"] >= 2 and actor["provider"] == provider and actor["model"] == model
