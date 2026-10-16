"""FINAL-2: the playable multi-policy / multi-provider runtime (spy_vs_ai/) - spec 46 and the spec-48 acceptance demo.

Every test is call-free: providers are the fake provider or adapters with injected transports (no network, no
credential: decoy values only)."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path
from types import SimpleNamespace

import pygame
import pytest

from tests.workshop_live_helpers import FAKE_BEDROCK_TOKEN, FAKE_OPENAI_KEY, FakeHTTPResponse, openai_body
from spy_vs_ai import play
from spy_vs_ai.providers import base as pbase
from spy_vs_ai.providers.secrets import KEY_SHAPED_RE, Scrubber, leaks
from spy_vs_ai.runtime import config as rc
from spy_vs_ai.runtime import policy_registry as pr
from spy_vs_ai.runtime import provider_registry as prov
from workshop.policy_emulator import stage4_level_select as ls

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/data/final2_frozen_policy_outputs.json"
FAKE_ANTHROPIC_KEY = "sk-ant-test-FAKE0123456789abcdef"
FAKE_AWS_KEY_ID = "AKIAFAKEFAKEFAKE1234"
OPENAI_NAMES = ("OPENAI_API_KEY",)
ANTHROPIC_NAMES = ("ANTHROPIC_API_KEY",)
AWS_NAMES = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK", "AWS_PROFILE")
SPEC_EXAMPLE = {"black": {"policy": "hybrid_v2", "provider": "bedrock", "model": "us.anthropic.claude-sonnet-5-5",
                          "personality": "cautious"},
                "white": {"policy": "real_llm", "provider": "ollama", "model": "qwen", "personality": "saboteur"},
                "level": "level_02", "seed": 42, "mode": "playable", "speed": 1.0, "show_telemetry": True}


# -- helpers ------------------------------------------------------------------------------------------------

class EnvWatch(MutableMapping):
    """os.environ, recording every name read by lookup (get / [] / in). Iteration (names only) is not a read."""

    def __init__(self, inner) -> None:
        self.inner, self.read = inner, []

    def __getitem__(self, key):
        self.read.append(key)
        return self.inner[key]

    def get(self, key, default=None):
        self.read.append(key)
        return self.inner.get(key, default)

    def __contains__(self, key):
        self.read.append(key)
        return key in self.inner

    def __setitem__(self, key, value):
        self.inner[key] = value

    def __delitem__(self, key):
        del self.inner[key]

    def __iter__(self):
        return iter(list(self.inner))

    def __len__(self):
        return len(self.inner)

    def copy(self):
        return dict(self.inner)


@pytest.fixture(autouse=True)
def _restore_level_selection():
    """Playback / matches select their level process-wide (stage4_level_select): every test leaves level_01 behind."""
    from src import constants
    before = constants.LEVEL_PATH
    yield
    constants.LEVEL_PATH = before


@pytest.fixture
def decoys(monkeypatch):
    """Every provider's credential variable set to a decoy, then os.environ watched."""
    for name, value in (("OPENAI_API_KEY", FAKE_OPENAI_KEY), ("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY),
                        ("AWS_ACCESS_KEY_ID", FAKE_AWS_KEY_ID), ("AWS_SECRET_ACCESS_KEY", "fake-secret-value-123"),
                        ("AWS_BEARER_TOKEN_BEDROCK", FAKE_BEDROCK_TOKEN)):
        monkeypatch.setenv(name, value)
    watch = EnvWatch(os.environ)
    monkeypatch.setattr(os, "environ", watch)
    return watch


def frozen_outputs(load) -> dict:
    """The golden-file computation for one loader (`load(policy, key) -> LoadedPolicy`)."""
    from workshop.policy_emulator import dataset as ds
    from workshop.policy_emulator.surrogate import SurrogateController
    golden = json.loads(GOLDEN.read_text())
    by_id = {r["decision_id"]: r for r in ds.load("phase-a-v2")}
    rows = {}
    for entry in golden["rows"]:
        rows.setdefault(entry["key"], []).append(by_id[entry["decision_id"]])
    out = {}
    for name in golden["outputs"]:
        policy, key = name.split("|")
        lp = load(policy, key)
        res = []
        for i, r in enumerate(rows[key]):
            pred = lp.action_policy.predict(r["features"], list(r["legal_action_names"]))
            ctl = SurrogateController("black_spy", lp, telemetry=None, episode_id="golden", seed=7, dataset="phase-a-v2",
                                      environment_seed=7)
            d = ctl.decide_payload(r["observation"], random.Random(f"golden:{i}"))
            res.append({"probs": {a: round(p, 12) for a, p in sorted(pred.probs.items())}, "action": d["action"],
                        "target": d["target"]})
        out[name] = {"emulator_hash": lp.metadata.get("content_hash"), "decisions": res}
    return out


def acceptance_config(**changes) -> rc.GameRuntimeConfig:
    data = {"black": {"policy": "hybrid_v2", "provider": "fake", "model": "fake-legal", "personality": "cautious"},
            "white": {"policy": "b2_v1", "personality": "racer"}, "level": "level_02", "seed": 42, "mode": "evaluation",
            "view": "research", "time_limit": 60}
    data.update(changes)
    return rc.from_dict(data)


def run_match(cfg: rc.GameRuntimeConfig, run_dir: Path, **kwargs):
    from spy_vs_ai.runtime.session import RuntimeMatchRunner
    with ls.using(cfg.level):
        runner = RuntimeMatchRunner(cfg, run_dir, headless=True, verbose=False, **kwargs)
        summary = runner.run()
    return runner, summary


def child_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI", "ANTHROPIC", "AWS_", "BEDROCK"))}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                "PYGAME_HIDE_SUPPORT_PROMPT": "1", **extra})
    return env


@pytest.fixture(scope="module")
def acceptance(tmp_path_factory):
    """Spec 48: different policies per spy, a headless match with a fake provider behind the hybrid, to the end."""
    run_dir = tmp_path_factory.mktemp("final2-acceptance")
    cfg = acceptance_config()
    runner, summary = run_match(cfg, run_dir)
    return SimpleNamespace(cfg=cfg, run_dir=run_dir, runner=runner, summary=summary, game=runner.game,
                           bundle=json.loads(Path(runner.replays[0]).read_text()),
                           manifest=json.loads((run_dir / "runtime_manifest.json").read_text()))


# -- configuration (spec 12, 13, 34) ------------------------------------------------------------------------

def test_spec_example_config_validates_per_spy():
    cfg = rc.validate(rc.from_dict(SPEC_EXAMPLE))
    assert (cfg.black.policy, cfg.black.provider, cfg.black.model, cfg.black.personality) == \
        ("hybrid_v2", "bedrock", "us.anthropic.claude-sonnet-5-5", "cautious")
    assert (cfg.white.policy, cfg.white.provider, cfg.white.model, cfg.white.personality) == \
        ("real_llm", "ollama", "qwen", "saboteur")
    assert cfg.providers_used == ("bedrock", "ollama")


@pytest.mark.parametrize("policy", ["scripted", "b1", "b2_v1", "b2_v2"])
def test_call_free_policies_need_no_provider_and_refuse_one(policy):
    ok = rc.from_dict({"black": {"policy": policy}, "white": {"policy": "b2_v1"}})
    assert rc.problems(ok) == [] and not ok.black.calls_provider
    bad = rc.from_dict({"black": {"policy": policy, "provider": "bedrock", "model": "m"}, "white": {"policy": "b2_v1"}})
    assert any("makes no provider call" in p for p in rc.problems(bad))


@pytest.mark.parametrize("policy", ["hybrid_v1", "hybrid_v2", "real_llm"])
def test_provider_policies_need_provider_and_model(policy):
    found = rc.problems(rc.from_dict({"black": {"policy": policy}, "white": {"policy": "b2_v1"}}))
    assert any("needs a provider" in p for p in found) and any("needs a model" in p for p in found)
    assert rc.problems(rc.from_dict({"black": {"policy": policy, "provider": "ollama", "model": "qwen"},
                                     "white": {"policy": "b2_v1"}})) == []


@pytest.mark.parametrize("black, needle", [
    ({"policy": "nope"}, "unknown policy"),
    ({"policy": "real_llm", "provider": "nope", "model": "m"}, "unknown provider"),
    ({"policy": "real_llm", "provider": "openai", "model": FAKE_OPENAI_KEY}, "looks like a credential"),
    ({"policy": "real_llm", "provider": "bedrock", "model": "m", "provider_options": {"region": "moon"}}, "not an AWS region"),
    ({"policy": "real_llm", "provider": "ollama", "model": "m", "provider_options": {"colour": "red"}}, "unknown ollama option"),
    ({"policy": "real_llm", "provider": "openai", "model": "<model>"}, "not a valid model identifier"),
    ({"policy": "b2_v1", "personality": "grumpy"}, "unknown personality"),
    ({"policy": "real_llm", "provider": "ollama", "model": "m", "identity": "sonnet55"}, "takes no identity"),
])
def test_invalid_combinations_are_rejected(black, needle):
    assert any(needle in p for p in rc.problems(rc.from_dict({"black": black, "white": {"policy": "b2_v1"}})))


def test_secret_keys_and_output_into_the_archive_are_rejected():
    with pytest.raises(rc.RuntimeConfigError, match="api_key"):
        rc.from_dict({"black": {"policy": "real_llm", "api_key": "x"}, "white": {"policy": "b2_v1"}})
    bad = rc.from_dict({"black": {"policy": "b2_v1"}, "white": {"policy": "b2_v1"},
                        "output_dir": "workshop/policies/runs"})
    assert any("research archive" in p for p in rc.problems(bad))
    bad = rc.from_dict({"black": {"policy": "b2_v1"}, "white": {"policy": "b2_v1"}, "output_dir": "workshop_runs/x"})
    assert any("workshop_runs" in p for p in rc.problems(bad))
    with pytest.raises(rc.RuntimeConfigError, match="looks like a secret"):
        rc.validate(rc.from_dict({"black": {"policy": "real_llm", "provider": "openai", "model": "m",
                                            "provider_options": {"token": "abc"}}, "white": {"policy": "b2_v1"}}))


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_example_configs_validate_hold_no_secret_and_parse_without_pyyaml(path):
    text = path.read_text()
    assert not KEY_SHAPED_RE.search(text) and "api_key" not in text.lower()
    cfg = rc.validate(rc.load_file(path))
    yaml = pytest.importorskip("yaml")
    assert rc.from_dict(rc.load_yaml_subset(text)) == rc.from_dict(yaml.safe_load(text)) == cfg
    assert rc.from_dict(rc.load_yaml_subset(rc.to_yaml(cfg))) == cfg  # round trip


def test_spec_29_example_configs_exist():
    for name in ("demo_real_vs_real", "demo_hybrid_v1_vs_real", "demo_hybrid_v2_vs_real", "demo_b2_v1_vs_b2_v2",
                 "demo_ollama_vs_bedrock", "demo"):
        assert (ROOT / f"configs/{name}.yaml").is_file()
    assert rc.load_file(ROOT / "configs/demo.yaml") == rc.from_dict(SPEC_EXAMPLE)


def test_cli_flags_build_the_spec_14_config():
    args = play.build_parser().parse_args([
        "--black-policy", "hybrid_v2", "--black-provider", "bedrock", "--black-model", "us.anthropic.claude-sonnet-5-5",
        "--black-personality", "cautious", "--white-policy", "real_llm", "--white-provider", "ollama",
        "--white-model", "qwen", "--white-personality", "saboteur", "--level", "level_02", "--seed", "42"])
    assert play.config_from_args(args) == rc.from_dict(SPEC_EXAMPLE)


def test_visual_runs_undo_the_workshop_headless_default_but_never_a_user_setting(monkeypatch):
    import workshop
    monkeypatch.setitem(workshop.SDL_DEFAULTS_APPLIED, "SDL_VIDEODRIVER", True)
    monkeypatch.setitem(workshop.SDL_DEFAULTS_APPLIED, "SDL_AUDIODRIVER", True)
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")  # what importing `workshop` does
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    play.prepare_display(False)
    assert "SDL_VIDEODRIVER" not in os.environ and "SDL_AUDIODRIVER" not in os.environ
    monkeypatch.setitem(workshop.SDL_DEFAULTS_APPLIED, "SDL_VIDEODRIVER", False)  # the user's own shell setting
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    with pytest.raises(SystemExit, match="invisible"):
        play.prepare_display(False)
    play.prepare_display(True)
    assert os.environ["SDL_VIDEODRIVER"] == "dummy"


def test_cli_quick_start_defaults_and_overrides(tmp_path):
    notes = []
    cfg = play.config_from_args(play.build_parser().parse_args(
        ["--black-policy", "real_llm", "--black-provider", "ollama", "--black-model", "qwen"]), notes)
    assert cfg.white.policy == "b2_v1" and not cfg.white.calls_provider and notes
    # a flag overrides the file; switching to a call-free policy drops the file's provider settings
    args = play.build_parser().parse_args(["--config", str(ROOT / "configs/demo.yaml"), "--black-policy", "b2_v2",
                                           "--seed", "7", "--black-option", "region=us-west-2"])
    cfg = play.config_from_args(args)
    assert cfg.black.policy == "b2_v2" and cfg.seed == 7 and cfg.white.model == "qwen"
    assert any("makes no provider call" in p for p in rc.problems(cfg))  # an option on a call-free spy is refused
    assert play.main(["--config", str(ROOT / "configs/demo_b2_v1_vs_b2_v2.yaml"), "--check"]) == 0
    assert play.main(["--black-policy", "b2_v2", "--black-provider", "bedrock", "--white-policy", "b2_v1",
                      "--check"]) == 2


# -- registries and labels (spec 3-5, 30, 35) ---------------------------------------------------------------

def test_policy_registry_covers_every_policy_and_never_aliases():
    assert set(pr.POLICY_IDS) == {"scripted", "b1", "b2_v1", "hybrid_v1", "b2_v2", "hybrid_v2", "real_llm"}
    assert pr.POLICIES["hybrid_v1"].base == "b2_v1" and pr.POLICIES["hybrid_v2"].base == "b2_v2"
    assert len({d.label for d in pr.POLICIES.values()}) == len(pr.POLICIES)
    assert {pid for pid, d in pr.POLICIES.items() if d.requires_provider} == {"hybrid_v1", "hybrid_v2", "real_llm"}


def test_scientific_labels_match_the_final1_release_and_the_spec():
    from workshop.policy_emulator import final_release as fr
    assert pr.HYBRID_V1_STATUS == fr.HYBRID_V1_STATUS and pr.HYBRID_V2_STATUS == fr.HYBRID_V2_STATUS
    assert pr.POLICIES["hybrid_v1"].scientific_status == fr.HYBRID_V1_STATUS
    assert pr.POLICIES["hybrid_v2"].scientific_status == fr.HYBRID_V2_STATUS
    assert pr.POLICIES["hybrid_v1"].scientific_label == "CONFIRMED / GENERALIZED RESEARCH BASELINE"
    assert pr.POLICIES["hybrid_v2"].scientific_label == "MORE EFFICIENT \u2014 NOT SAFELY PROMOTABLE UNDER CONFIRMATORY RULE"
    assert pr.POLICIES["hybrid_v2"].summary[0] == "14.6% fewer provider calls in Stage 5C"
    claims = (ROOT / "workshop/policies/final-release/claims.json").read_text()
    assert "14.6%" in claims


def test_compatibility_labels():
    def comp(policy, provider=None, model=None, level="level_02", mode="evaluation", time_limit=300.0, identity=None):
        ident = pr.identity_for(policy, provider, model, identity)
        return pr.compatibility(policy, provider=provider, model=model, identity=ident, level=level, mode=mode,
                                time_limit=time_limit)
    sonnet, gpt = pr.IDENTITIES["sonnet55"], pr.IDENTITIES["gpt61sol"]
    assert comp("hybrid_v2", "bedrock", sonnet).label == pr.EVALUATED
    assert comp("real_llm", "bedrock", gpt).label == pr.EVALUATED
    playable = comp("hybrid_v2", "bedrock", sonnet, mode="playable")
    assert playable.label == pr.COMPATIBLE and any("playable" in r for r in playable.reasons)
    assert comp("hybrid_v2", "bedrock", sonnet, level="level_01").label == pr.COMPATIBLE  # not evaluated on level 1
    assert comp("hybrid_v1", "bedrock", sonnet, identity="gpt61sol").label == pr.COMPATIBLE
    assert comp("real_llm", "ollama", "qwen").label == pr.COMPATIBLE
    assert comp("b2_v1", time_limit=120.0).label == pr.COMPATIBLE
    assert comp("real_llm", "anthropic", "claude-x").label == pr.EXPERIMENTAL
    assert comp("hybrid_v1", "fake", "fake-legal").label == pr.FAKE
    assert comp("scripted").label == pr.BASELINE


def test_provider_registry_and_status_read_presence_only():
    assert set(prov.PROVIDER_IDS) == {"bedrock", "openai", "anthropic", "ollama", "fake"}
    assert prov.PROVIDERS["openai"].credential_env == ("OPENAI_API_KEY",)
    assert prov.PROVIDERS["anthropic"].credential_env == ("ANTHROPIC_API_KEY",)
    assert prov.PROVIDERS["ollama"].credential_env == () and prov.PROVIDERS["fake"].credential_env == ()
    from spy_vs_ai.providers import anthropic, openai
    on = openai.status(env={"OPENAI_API_KEY": FAKE_OPENAI_KEY})
    assert on["state"] == "configured" and FAKE_OPENAI_KEY not in json.dumps(on)
    assert openai.status(env={})["state"] == "unavailable"
    assert anthropic.status(env={"ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY})["state"] == "configured"

    def refused(url, timeout):
        raise OSError("connection refused")
    statuses = prov.status_all(ollama_opener=refused)
    assert statuses["ollama"]["state"] == "unreachable" and set(statuses) == set(prov.PROVIDER_IDS)

    def tags(url, timeout):
        assert url == "http://localhost:11434/api/tags"
        return FakeHTTPResponse(json.dumps({"models": [{"name": "qwen:latest"}, {"name": "gemma4"}]}).encode())
    from spy_vs_ai.providers import ollama
    assert ollama.status(opener=tags)["state"] == "reachable"
    assert ollama.list_models(opener=tags) == ["gemma4", "qwen:latest"]


def test_normalized_user_errors():
    assert pbase.user_error("provider_auth_error") == "Provider authentication failed"
    assert pbase.user_error("provider_transport_error") == "Provider unavailable"
    assert pbase.user_error("provider_access_error", "The provided model identifier is invalid") == "Model not found"
    assert pbase.user_error("provider_timeout") == "Request timed out"
    assert pbase.user_error("provider_service_error") == "Provider service error"
    assert pbase.user_error("provider_invalid_response") == "Invalid model output"
    assert pbase.FALLBACK_USED == "Fallback action used" and pbase.user_error(None) is None


# -- providers: model passthrough and credential isolation (spec 6-11, 35) ---------------------------------

def _bedrock_fake(calls):
    class FakeConverse:
        def converse(self, **kwargs):
            calls.append(kwargs)
            return {"output": {"message": {"content": [{"text": '{"action": "wait", "reason": "r"}'}]}},
                    "usage": {"inputTokens": 11, "outputTokens": 3}}
    return FakeConverse


def test_bedrock_passes_the_model_through_and_reads_no_other_provider_credential(decoys):
    calls = []
    client = prov.build_client("bedrock", "vendor.some-model-v2:0", {"region": "eu-west-3"},
                               client_factory=_bedrock_fake(calls))
    response = client.generate_action("s", "{}", {})
    assert calls[0]["modelId"] == "vendor.some-model-v2:0" and response.payload["action"] == "wait"
    rec = client.history[-1]
    assert (rec["provider_id"], rec["model_id"], rec["input_tokens"], rec["output_tokens"], rec["provider_calls"]) == \
        ("bedrock", "vendor.some-model-v2:0", 11, 3, 1)
    assert not set(decoys.read) & set(OPENAI_NAMES + ANTHROPIC_NAMES)


def test_openai_passes_the_model_through_and_reads_no_aws_or_anthropic(decoys):
    sent = []

    def opener(request, timeout):
        sent.append(request)
        return FakeHTTPResponse(openai_body({"action": "wait", "reason": "r"}))
    client = prov.build_client("openai", "some-future-model-2027", {"timeout_s": 9}, opener=opener)
    client.generate_action("Return JSON.", "{}", {}, request_context={"episode_id": "e", "actor_id": "a"})
    assert json.loads(sent[0].data)["model"] == "some-future-model-2027"
    assert sent[0].full_url == "https://api.openai.com/v1/chat/completions"
    assert not set(decoys.read) & set(AWS_NAMES + ANTHROPIC_NAMES)


def test_anthropic_messages_transport_passes_the_model_and_reads_no_aws_or_openai(decoys):
    sent = []

    def opener(request, timeout):
        sent.append(request)
        body = {"id": "msg_1", "model": "claude-x", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"action": "wait", "reason": "r"}'}],
                "usage": {"input_tokens": 7, "output_tokens": 2}}
        return FakeHTTPResponse(json.dumps(body).encode())
    client = prov.build_client("anthropic", "claude-x", {}, opener=opener)
    response = client.generate_action("Return JSON.", "{}", {})
    req = sent[0]
    body = json.loads(req.data)
    assert req.full_url == "https://api.anthropic.com/v1/messages" and body["model"] == "claude-x"
    assert req.get_header("X-api-key") == FAKE_ANTHROPIC_KEY and req.get_header("Anthropic-version") == "2023-06-01"
    assert "Return JSON." in body["system"] and body["messages"][-1]["role"] == "user"
    assert all(m["role"] in ("user", "assistant") for m in body["messages"])
    assert response.payload["action"] == "wait"
    assert not set(decoys.read) & set(AWS_NAMES + OPENAI_NAMES)


def test_anthropic_errors_are_classified_and_scrubbed(decoys):
    import urllib.error

    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {},
                                     FakeHTTPResponse(json.dumps({"error": {"type": "authentication_error",
                                                                            "message": f"bad {FAKE_ANTHROPIC_KEY}"}}).encode()))
    client = prov.build_client("anthropic", "claude-x", {}, opener=opener)
    try:
        response = client.generate_action("s", "{}", {})
        message = str(getattr(response, "error_message", "") or "")
    except Exception as exc:  # the controller's own repair / fallback path handles a raised provider error
        message = str(exc)
    rec = client.history[-1]
    assert rec["error_category"] == "provider_auth_error" and rec["user_error"] == "Provider authentication failed"
    assert FAKE_ANTHROPIC_KEY not in json.dumps(rec) and FAKE_ANTHROPIC_KEY not in message


def test_ollama_passes_the_model_and_reads_no_cloud_credential(decoys):
    seen = []

    def request_fn(payload):
        seen.append(payload)
        return {"model": payload["model"], "message": {"role": "assistant", "content": '{"action": "wait"}'},
                "done": True, "prompt_eval_count": 5, "eval_count": 2}
    client = prov.build_client("ollama", "qwen:7b", {"base_url": "http://localhost:11434"}, request_fn=request_fn)
    client.generate_action("s", "{}", {})
    assert seen[0]["model"] == "qwen:7b"
    assert not set(decoys.read) & set(AWS_NAMES + OPENAI_NAMES + ANTHROPIC_NAMES)


def test_scrubber_is_scoped_to_the_selected_providers(decoys):
    s = Scrubber(["bedrock"])
    assert s.text(f"x {FAKE_AWS_KEY_ID} y {FAKE_OPENAI_KEY}") == "x [redacted] y [redacted-key]"
    assert not set(decoys.read) & set(OPENAI_NAMES + ANTHROPIC_NAMES)
    decoys.read.clear()
    Scrubber([]).text("nothing")
    assert not set(decoys.read) & set(AWS_NAMES + OPENAI_NAMES + ANTHROPIC_NAMES)


def test_call_free_match_reads_no_credential_and_builds_no_client(decoys, tmp_path):
    pygame.init()  # the display / platform probing happens before the watch matters
    decoys.read.clear()
    cfg = rc.from_dict({"black": {"policy": "b2_v1", "personality": "cautious"},
                        "white": {"policy": "b2_v2", "personality": "racer", "identity": "gpt61sol"},
                        "level": "level_03", "seed": 3, "mode": "evaluation", "time_limit": 30})
    runner, summary = run_match(cfg, tmp_path / "run")
    assert runner.clients == {} and summary["runtime"]["summaries"][0]["total"]["provider_calls"] == 0
    assert not set(decoys.read) & set(AWS_NAMES + OPENAI_NAMES + ANTHROPIC_NAMES)


# -- frozen policy isolation (spec 5) -----------------------------------------------------------------------

def test_frozen_call_free_policies_return_the_golden_outputs():
    """Direct frozen loaders and the runtime registry both reproduce the stored outputs (same input, same output)."""
    from workshop.policy_emulator import dataset as ds
    from workshop.policy_emulator import stage5b_protocol as s5b
    from workshop.policy_emulator.artifacts import load_policy
    expected = ds.load_provenance("phase-a-v2")["dataset_hash"]

    def direct(policy, key):
        if policy == "b1":
            return load_policy("phase-a-v2", key, emulator="table-v1", target="target-v2.1", expected_dataset_hash=expected)
        b2 = load_policy("phase-a-v2", key, emulator="logistic-v1", target="target-v2.1", expected_dataset_hash=expected)
        return s5b.arm_base("b2v2", b2) if policy == "b2_v2" else b2

    golden = json.loads(GOLDEN.read_text())["outputs"]
    assert frozen_outputs(direct) == golden
    assert frozen_outputs(lambda policy, key: pr.load_frozen(policy, key).base) == golden


def test_frozen_hybrid_bundles_keep_their_identities():
    v2 = pr.load_frozen("hybrid_v2", "sonnet55:cautious")
    assert v2.identity["router_hash"] == pr.R21_HASH and v2.identity["surrogate"]["b2_v2_manifest_hash"] == pr.B2V2_MANIFEST_HASH
    assert v2.identity["surrogate"]["emulator_hash"] == "sha256:76a24af39269499a"
    v1 = pr.load_frozen("hybrid_v1", "sonnet55:cautious")
    assert v1.identity["router_hash"] == pr.R21_HASH and "b2_v2_manifest_hash" not in v1.identity["surrogate"]
    assert v1.identity["surrogate"]["emulator_hash"] != v2.identity["surrogate"]["emulator_hash"]


# -- the acceptance demo (spec 48) ---------------------------------------------------------------------------

def test_acceptance_match_runs_both_spies_to_the_end(acceptance):
    s = acceptance.summary["runtime"]["summaries"][0]
    assert acceptance.summary["stop_reason"] == "match_over" and s["complete"]
    records = list(acceptance.runner.log.records)
    for actor in ("black_spy", "white_spy"):
        mine = [r for r in records if r["spy"] == actor]
        assert mine and any(r["executed"].startswith(("MOVE", "INSPECT", "INTERACT")) for r in mine)
        assert s["spies"][actor]["outcome"] in ("escaped", "timed out", "eliminated", "mission impossible")
    moved = [e for e in acceptance.bundle["telemetry"] if e.get("event_type") == "action_lifecycle"]
    assert {e.get("actor_id") for e in moved} == {"black_spy", "white_spy"}


def test_acceptance_hybrid_escalation_telemetry_with_the_fake_provider(acceptance):
    t = acceptance.runner.log.telemetry()
    assert t["black_spy"]["escalations"] > 0 and t["black_spy"]["provider_calls"] >= t["black_spy"]["escalations"]
    assert t["white_spy"]["provider_calls"] == 0 and t["white_spy"]["escalations"] == 0
    escalated = [r for r in acceptance.runner.log.records if (r.get("router") or {}).get("escalated")]
    assert escalated and all(r["llm"].get("action") for r in escalated)
    from spy_vs_ai.runtime.decisions import inspector_lines
    text = "\n".join(inspector_lines(escalated[0]))
    for needle in ("B2-v2:", "B1:", "R2.1: ESCALATE", "Provider: fake", "LLM action:", "Executed:"):
        assert needle in text
    assert acceptance.runner.clients["black_spy"].calls == t["black_spy"]["provider_calls"]


def test_acceptance_summary_and_renderer(acceptance):
    from spy_vs_ai.runtime.decisions import summary_lines
    s = acceptance.summary["runtime"]["summaries"][0]
    for key in ("decisions", "provider_calls", "escalations", "input_tokens", "output_tokens", "fallbacks"):
        assert s["total"][key] == sum(s["spies"][a][key] for a in ("black_spy", "white_spy"))
    assert "MATCH SUMMARY" in summary_lines(s)[0] and s["game_time_s"] > 0
    r = acceptance.game.renderer
    assert r.summary is s and r.match_result is not None
    text = [t for t, _ in r.summary_text(r.match_result)]
    assert any(t.startswith("Replay saved:") for t in text) and rc.EVALUATION_LABEL in text
    state = r.ui_state()
    r.handle_key(pygame.K_m)
    assert r.ui_state()["view"] != state["view"]
    r.handle_key(pygame.K_m)
    r.handle_key(pygame.K_TAB)
    assert r.ui_state()["inspect"] == "black_spy" and r.selected_record()["spy"] == "black_spy"
    r.handle_key(pygame.K_UP)
    assert r.ui_state()["selected"] == 1


def test_v_hides_and_shows_the_panels_over_the_game_area(tmp_path):
    """Live feedback: the inspector / history panels cover half the game view; V (the detail key) toggles them."""
    from spy_vs_ai.runtime.session import RuntimeMatchRunner
    pygame.init()
    cfg = rc.from_dict({"black": {"policy": "b2_v1"}, "white": {"policy": "b2_v2"}, "view": "research",
                        "mode": "evaluation", "time_limit": 30})
    presses = iter([[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v, mod=0, unicode="v")], [],
                    [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="")]])
    seen = []
    with ls.using(cfg.level):
        runner = RuntimeMatchRunner(cfg, tmp_path / "run", headless=False, verbose=False,
                                    event_source=lambda: next(presses, []))
        original = runner._handle_events

        def watched():
            keep = original()
            seen.append(runner.game.renderer.ui_state())
            return keep
        runner._handle_events = watched
        runner.run()
    assert seen[0]["panels"] is False and seen[0]["detail"] is False  # research view, V pressed once: panels hidden
    v = [c for c in runner.controls_log if c["key"] == "V"]
    assert v and v[0]["state"]["panels"] is False
    from spy_vs_ai.ui.research import RuntimeRenderer
    r = RuntimeRenderer(pygame.display.set_mode((1680, 720)), cfg)
    assert r.panels_visible() and r.show_detail
    r.handle_key(pygame.K_v)
    assert not r.panels_visible() and not r.show_inspector and not r.show_history and not r.show_detail
    r.handle_key(pygame.K_v)
    assert r.panels_visible() and r.show_inspector and r.show_history
    play = RuntimeRenderer(pygame.display.get_surface(), dataclasses_replace(cfg, view="play"))
    assert not play.panels_visible()
    play.handle_key(pygame.K_v)  # from the play view, V brings the panels (and the research view) up
    assert play.panels_visible() and play.runtime_view == "research"


def dataclasses_replace(cfg, **changes):
    import dataclasses
    return dataclasses.replace(cfg, **changes)


def test_manifest_captures_identities_and_hashes(acceptance):
    from workshop.scripts.tournament import code_fingerprint
    m = acceptance.manifest
    assert m["engine"]["game_code_fingerprint"] == code_fingerprint()
    final = json.loads((ROOT / "workshop/policies/final-release/freeze_manifest.json").read_text())["manifest_hash"]
    assert m["engine"]["final1_freeze_manifest_hash"] == final
    assert m["level_file"]["sha256"] == ls.file_sha256("level_02")
    black, white = m["spies"]["black"], m["spies"]["white"]
    assert black["artifact_identity"]["router_hash"] == pr.R21_HASH
    assert black["artifact_identity"]["surrogate"]["b2_v2_manifest_hash"] == pr.B2V2_MANIFEST_HASH
    assert black["provider"] == {"id": "fake", "label": prov.PROVIDERS["fake"].label, "model": "fake-legal",
                                 "adapter": "spy_vs_ai.providers.fake", "credential_env_names": [], "options": {}}
    assert white["artifact_identity"]["surrogate"]["emulator"] == "logistic-v1" and "provider" not in white
    assert m["secrets_recorded"] is False and m["totals"]["escalations"] > 0 and m["replays"]


def test_runtime_outputs_stay_out_of_the_research_archive(acceptance):
    names = {p.name for p in acceptance.run_dir.iterdir()}
    assert {"runtime_manifest.json", "replay_match01.json", "telemetry_match01.jsonl", "live_summary.json"} <= names
    assert all(Path(p).parent == acceptance.run_dir for p in acceptance.manifest["replays"])
    from spy_vs_ai.runtime.session import new_run_dir
    made = new_run_dir(acceptance.cfg)
    try:
        assert made.parent == ROOT / "runs" and rc.DEFAULT_OUTPUT_DIR == "runs"
    finally:
        made.rmdir()


def test_replay_bundle_has_the_spec_fields_and_no_secret(acceptance):
    b = acceptance.bundle
    for key in ("schema", "engine", "level", "seed", "config", "spies", "telemetry", "decisions", "outcome",
                "started_utc", "finished_utc", "frames", "verification"):
        assert key in b
    assert b["secrets_recorded"] is False and b["seed"] == 42 and b["level"] == "level_02"
    text = Path(acceptance.runner.replays[0]).read_text()
    assert leaks(text, ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_BEARER_TOKEN_BEDROCK")) == []
    assert not KEY_SHAPED_RE.search(text) and "Authorization" not in text


def test_writer_scrubs_key_shaped_values(tmp_path):
    from spy_vs_ai.runtime.session import RuntimeMatchRunner
    fake = SimpleNamespace(scrub=Scrubber(["fake"]), run_dir=tmp_path)
    path = RuntimeMatchRunner._write(fake, "x.json", {"note": f"echo {FAKE_OPENAI_KEY}", "api_key": "abc",
                                                      "input_tokens": 5})
    text = path.read_text()
    assert FAKE_OPENAI_KEY not in text and '"api_key": "[redacted]"' in text and '"input_tokens": 5' in text


def test_offline_replay_reproduces_in_process(acceptance):
    from spy_vs_ai.runtime import playback as pb
    data = pb.load(acceptance.runner.replays[0])
    p = pb.Playback(data)
    result = p.run_to_end()
    assert result["verdict"] == pb.REPRODUCED and all(result["checks"].values())
    assert result["provider_calls"] == 0 and result["decisions_shown"] == acceptance.runner.log.count
    assert p.game.black_llm_controller.controller_kind == "replay" and p.game.bot_controller.controller_kind == "replay"


def test_offline_replay_cli_makes_no_provider_call(acceptance):
    """A separate process under the offline guard (no network, no client, credential names removed)."""
    proc = subprocess.run([sys.executable, "-m", "spy_vs_ai.replay", acceptance.runner.replays[0], "--headless",
                           "--verify", "--json"], cwd=str(ROOT), env=child_env(OPENAI_API_KEY=FAKE_OPENAI_KEY),
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "REPLAY REPRODUCED" in proc.stdout and "provider calls 0, network attempts 0, provider clients constructed 0" \
        in proc.stdout
    result = json.loads(proc.stdout[proc.stdout.rindex("\n{\n") + 1:])
    assert result["offline_guard_clean"] and result["provider_calls"] == 0


def test_tampered_replay_is_reported_diverged(acceptance, tmp_path):
    from spy_vs_ai.runtime import playback as pb
    data = pb.load(acceptance.runner.replays[0])
    data["verification"]["engine_event_digest"] = "sha256:" + "0" * 64
    assert pb.Playback(data).run_to_end()["verdict"] == pb.DIVERGED


def test_playable_mode_is_visibly_distinct():
    playable = rc.execution(acceptance_config(mode="playable"))
    evaluation = rc.execution(acceptance_config())
    assert playable["label"].startswith("PLAYABLE MODE") and "execution semantics differ from the frozen evaluation " \
        "protocol" in playable["label"]
    assert evaluation["mission_timer"] == "game_time" and evaluation["commit_policy"] == "barrier_seeded_order"
    # a hybrid keeps its validated timing, and with it a mission clock that stops while a model thinks
    assert playable["decision_timing"] == "evaluation" and playable["mission_timer"] == "game_time" and playable["notes"]
    assert playable["commit_policy"] == "barrier_answer_order"
    real = rc.execution(rc.from_dict({"black": {"policy": "real_llm", "provider": "fake", "model": "f"},
                                      "white": {"policy": "b2_v1"}}))
    assert real["decision_timing"] == "live_async" and real["mission_timer"] == "wall_clock"
    assert real["label"] == rc.PLAYABLE_LABEL
    from spy_vs_ai.ui.research import RuntimeRenderer
    pygame.init()
    r = RuntimeRenderer(pygame.display.set_mode((1680, 720)), acceptance_config(mode="playable"))
    assert r.mode_label == rc.PLAYABLE_LABEL and not r.evaluation_mode
    assert r.spy_info["black_spy"]["compat"] == pr.FAKE


@pytest.mark.parametrize("mode", rc.MODES)
@pytest.mark.parametrize("black", pr.POLICY_IDS)
@pytest.mark.parametrize("white", pr.POLICY_IDS)
def test_a_frozen_world_never_runs_on_a_wall_clock(mode, black, white):
    """Evaluation timing freezes the world while a model thinks; the mission clock must freeze with it (the action
    lifecycle measures move progress in mission time: a draining clock makes every move look stalled)."""
    def agent(policy):
        return {"policy": policy, **({"provider": "fake", "model": "fake-legal"} if pr.POLICIES[policy].requires_provider
                                     else {})}
    ex = rc.execution(rc.from_dict({"black": agent(black), "white": agent(white), "mode": mode}))
    if ex["decision_timing"] == "evaluation":
        assert ex["mission_timer"] == "game_time"


def test_playable_hybrid_spies_keep_moving_while_a_model_thinks(tmp_path):
    """Regression (FINAL-2 live smoke test): in playable mode a hybrid spy with a slow provider froze the world while
    the mission clock ran on, so every MOVE ended `move_stalled` after one tick and both spies stood still."""
    cfg = rc.from_dict({"black": {"policy": "hybrid_v2", "provider": "fake", "model": "fake-legal", "personality": "saboteur",
                                  "provider_options": {"latency_s": 0.3}},
                        "white": {"policy": "real_llm", "provider": "fake", "model": "fake-legal", "personality": "saboteur",
                                  "provider_options": {"latency_s": 0.3}},
                        "level": "level_01", "seed": 42, "mode": "playable", "time_limit": 30})
    runner, _ = run_match(cfg, tmp_path / "run")
    moves = [e for e in runner.game.telemetry.events if e.get("event_type") == "action_lifecycle" and e.get("action") == "move"]
    stalled = [e for e in moves if e.get("action_terminal_reason") == "move_stalled"]
    assert moves and len(stalled) <= len(moves) // 4
    assert not [e for e in stalled if e.get("action_simulation_ticks", 99) <= 2]


# -- the setup screen (spec 15, 21) ---------------------------------------------------------------------------

def key(k, text=""):
    return pygame.event.Event(pygame.KEYDOWN, key=k, unicode=text, mod=0)


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_setup_screen_round_trips_every_example_config(path):
    from spy_vs_ai.ui.setup import SetupScreen
    cfg = rc.load_file(path)
    assert SetupScreen.from_config(cfg, provider_status={}).to_config() == cfg


def test_setup_screen_edits_validates_and_starts():
    from spy_vs_ai.ui.setup import HEIGHT, WIDTH, SetupScreen
    pygame.init()
    screen = SetupScreen(acceptance_config(), screen=pygame.Surface((WIDTH, HEIGHT)),
                         provider_status={"fake": {"state": "available", "detail": "x"},
                                          "openai": {"state": "unavailable", "detail": "OPENAI_API_KEY is not set"}})
    screen.layout()
    screen.draw()
    # Right on the black policy cycles it; a call-free policy disables provider / model
    screen.focus = 0
    screen.handle_event(key(pygame.K_RIGHT))
    assert screen.values["black.policy"] == "real_llm" and screen.to_config().black.provider == "fake"
    screen.handle_event(key(pygame.K_RIGHT))
    assert screen.values["black.policy"] == "scripted" and screen.to_config().black.provider is None
    # edit the seed by typing
    screen.focus = [f.key for f in screen.fields].index("seed")
    screen.handle_event(key(pygame.K_RETURN))
    for _ in range(3):
        screen.handle_event(key(pygame.K_BACKSPACE))
    screen.handle_event(key(pygame.K_x, "x"))
    assert screen.problems() and not screen.start() and "seed" in screen.message
    screen.handle_event(key(pygame.K_BACKSPACE))
    screen.handle_event(key(pygame.K_7, "7"))
    screen.handle_event(key(pygame.K_RETURN))
    screen.draw()
    # a dropdown: open the level list, move down, confirm
    screen.focus = [f.key for f in screen.fields].index("level")
    screen.handle_event(key(pygame.K_RETURN))
    assert screen.open == "level"
    screen.handle_event(key(pygame.K_DOWN))
    screen.draw()
    screen.handle_event(key(pygame.K_RETURN))
    assert screen.values["level"] == "level_03" and screen.open is None
    assert screen.start() and screen.result.seed == 7 and screen.result.black.policy == "scripted"
    assert screen.result.level == "level_03"


def _pick(screen, key_name, value):
    """Through the UI only: focus the field (Tab order), open its list with Enter, move with Down, confirm."""
    keys = [f.key for f in screen.fields]
    screen.focus = keys.index(key_name)
    assert screen.fields[screen.focus] in screen.enabled_fields(), key_name
    screen.handle_event(key(pygame.K_RETURN))
    assert screen.open == key_name
    values = [v for v, _ in screen.choices(key_name)]
    target = values.index(value)
    for _ in range((target - screen.open_index) % len(values)):
        screen.handle_event(key(pygame.K_DOWN))
    screen.handle_event(key(pygame.K_RETURN))
    assert screen.open is None and screen.values[key_name] == value


def test_setup_screen_expresses_the_live_smoke_test_configuration():
    """FINAL2_LIVE_VISUAL_SMOKE_TEST.md: if the setup screen cannot express the exact configuration, fix the UI.
    Bedrock offers both research models in the model list (GPT-6.1 Sol was reachable only by a hidden key before)."""
    from spy_vs_ai.ui.setup import HEIGHT, WIDTH, SetupScreen
    pygame.init()
    screen = SetupScreen(screen=pygame.Surface((WIDTH, HEIGHT)), provider_status={})
    screen.layout()
    for name, value in (("black.policy", "hybrid_v2"), ("black.provider", "bedrock"),
                        ("black.model", "us.anthropic.claude-sonnet-5-5"), ("black.personality", "cautious"),
                        ("white.policy", "real_llm"), ("white.provider", "bedrock"),
                        ("white.model", "us.openai.gpt-6.1-sol"), ("white.personality", "saboteur"),
                        ("level", "level_02"), ("mode", "evaluation"), ("view", "research")):
        _pick(screen, name, value)
        screen.draw()
    assert screen.start()
    cfg = screen.result
    assert (cfg.black.policy, cfg.black.provider, cfg.black.model, cfg.black.personality) == \
        ("hybrid_v2", "bedrock", "us.anthropic.claude-sonnet-5-5", "cautious")
    assert (cfg.white.policy, cfg.white.provider, cfg.white.model, cfg.white.personality) == \
        ("real_llm", "bedrock", "us.openai.gpt-6.1-sol", "saboteur")
    assert (cfg.level, cfg.seed, cfg.mode, cfg.view, cfg.time_limit, cfg.show_telemetry) == \
        ("level_02", 42, "evaluation", "research", 300.0, True)
    for side in ("black", "white"):
        a = cfg.agent(side)
        assert pr.compatibility(a.policy, provider=a.provider, model=a.model, identity=a.surrogate_identity,
                                level=cfg.level, mode=cfg.mode, time_limit=cfg.time_limit).label == pr.EVALUATED


def test_setup_model_list_by_mouse_and_other_model_id():
    from spy_vs_ai.ui.setup import HEIGHT, OTHER_MODEL, WIDTH, SetupScreen
    pygame.init()
    screen = SetupScreen(rc.from_dict({"black": {"policy": "real_llm", "provider": "bedrock",
                                                 "model": "us.anthropic.claude-sonnet-5-5"},
                                       "white": {"policy": "b2_v1"}}),
                         screen=pygame.Surface((WIDTH, HEIGHT)), provider_status={})
    screen.layout()
    field = screen._field("black.model")
    click = lambda pos: screen.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    click(field.rect.center)
    assert screen.open == "black.model"
    labels = dict(screen.choices("black.model"))
    assert "us.openai.gpt-6.1-sol" in labels and OTHER_MODEL in labels
    screen.draw()
    rects = screen._option_rects()
    values = [v for v, _ in screen.choices("black.model")]
    click(rects[values.index("us.openai.gpt-6.1-sol")].center)
    assert screen.values["black.model"] == "us.openai.gpt-6.1-sol" and screen.open is None
    # "other model id": an empty text box, typed, then Enter
    click(field.rect.center)
    screen.draw()
    click(screen._option_rects()[values.index(OTHER_MODEL)].center)
    assert screen.editing == "black.model" and screen.values["black.model"] == ""
    for ch in "vendor.model-x:1":
        screen.handle_event(key(0, ch))
    screen.handle_event(key(pygame.K_RETURN))
    assert screen.to_config().black.model == "vendor.model-x:1" and screen.problems() == []
    assert dict(screen.choices("black.model")).get("vendor.model-x:1")  # a custom id stays listed


def test_setup_screen_refuses_an_invalid_start_and_run_returns_the_choice():
    from spy_vs_ai.ui.setup import HEIGHT, WIDTH, SetupScreen
    pygame.init()
    cfg = acceptance_config()
    screen = SetupScreen(cfg, provider_status={})
    screen.values["black.model"] = ""
    assert not screen.start() and screen.result is None
    screen.values["black.model"] = "fake-legal"
    events = iter([[], [key(pygame.K_TAB)], [pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(WIDTH // 2, HEIGHT - 48),
                                                                 button=1)]])
    screen.event_source = lambda: next(events, [])
    screen.screen = pygame.Surface((WIDTH, HEIGHT))
    assert screen.run(max_frames=10) == cfg


# -- historical replay (spec 28) ------------------------------------------------------------------------------

def test_historical_examples_exist_and_the_call_free_one_reproduces(tmp_path):
    from spy_vs_ai.runtime.historical import EXAMPLES
    for path in EXAMPLES.values():
        assert (ROOT / path / "telemetry_match01.jsonl").is_file()
    match = ROOT / EXAMPLES["B2 loop pathology (B2-v1 through Hybrid R0)"]
    before = {p.name: p.stat().st_mtime_ns for p in match.iterdir()}
    proc = subprocess.run([sys.executable, "-m", "spy_vs_ai.replay", "--historical", str(match), "--headless", "--verify",
                           "--save", str(tmp_path / "historical.json")], cwd=str(ROOT), env=child_env(),
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "historical engine events reproduced exactly" in proc.stdout and "REPLAY REPRODUCED" in proc.stdout
    assert {p.name: p.stat().st_mtime_ns for p in match.iterdir()} == before  # read only
    bundle = json.loads((tmp_path / "historical.json").read_text())
    assert bundle["historical"]["comparison"]["engine_events_identical"] and bundle["config"]["black"]["policy"] == "b2_v1"


def test_historical_comparison_reports_a_difference():
    from spy_vs_ai.runtime.historical import compare, policy_of
    a = [{"event_type": "engine_event", "engine_seq": 0, "engine_event_type": "X", "episode_id": "1"}]
    b = [{"event_type": "engine_event", "engine_seq": 0, "engine_event_type": "Y", "episode_id": "2"}]
    assert compare(a, [dict(a[0], episode_id="other")])["engine_events_identical"]
    assert compare(a, b)["first_difference"] == 0
    assert policy_of({"arm": "hybrid", "candidate": "b2v2"}) == "hybrid_v2"
    assert policy_of({"arm": "r0", "candidate": None}) == "b2_v1" and policy_of({"arm": "real"}) == "real_llm"
