"""Play a match: two spies, each with its own policy, provider, model and personality (FINAL-2 spec 12-14, 41).

    python -m spy_vs_ai.play                                  # the visual setup screen
    python -m spy_vs_ai.play --config configs/demo.yaml       # straight into the match
    python -m spy_vs_ai.play \\
      --black-policy hybrid_v2 --black-provider bedrock --black-model us.anthropic.claude-sonnet-5-5 \\
      --black-personality cautious \\
      --white-policy real_llm --white-provider ollama --white-model qwen --white-personality saboteur \\
      --level level_02 --seed 42
    python -m spy_vs_ai.play --config configs/demo_offline.yaml --headless   # no window, writes runs/<run-id>/
    python -m spy_vs_ai.play --provider-status                # configured / unavailable, nothing is called

Policies: scripted, b1, b2_v1, hybrid_v1, b2_v2, hybrid_v2, real_llm. Providers: bedrock, openai, anthropic, ollama
(and fake: an offline demo stand-in, no model). B1 / B2 / scripted spies make no provider call and need no provider;
hybrid and real_llm spies need a provider and a model. Credentials come only from the environment / AWS credential
chain of the selected provider, never from a configuration file or the command line.

This is a playable runtime, not an experiment launcher: one match (R restarts it), one run directory with the runtime
manifest, the telemetry and the replay bundle (python -m spy_vs_ai.replay <bundle>).

Keys in the match: Space pause, + / - speed, R restart, Esc quit, M play / research view, I inspector, H history,
Y AI telemetry, Tab spy, Up / Down pick a decision, T / V agent panels, P split, O overview, Z zoom, S screenshot.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from spy_vs_ai.runtime import config as rc
from spy_vs_ai.runtime import policy_registry as pr
from spy_vs_ai.runtime import provider_registry as prov

PAID_PROVIDERS = ("bedrock", "openai", "anthropic")
DEFAULT_OPPONENT = {"policy": "b2_v1", "personality": "cautious"}  # a spy left unspecified: call-free B2-v1
AGENT_FLAGS = ("policy", "provider", "model", "personality", "identity", "option", "max_provider_calls")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m spy_vs_ai.play", description="Play one Spy vs AI match.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="policies: " + ", ".join(pr.POLICY_IDS) + "\nproviders: " + ", ".join(prov.PROVIDER_IDS))
    p.add_argument("--config", help="a YAML / JSON runtime configuration (configs/*.yaml)")
    for side in rc.ACTORS:
        g = p.add_argument_group(f"{side} spy")
        g.add_argument(f"--{side}-policy", choices=pr.POLICY_IDS)
        g.add_argument(f"--{side}-provider", choices=prov.PROVIDER_IDS)
        g.add_argument(f"--{side}-model", help="the provider's own model id (passed through unchanged)")
        g.add_argument(f"--{side}-personality", choices=pr.PERSONALITIES)
        g.add_argument(f"--{side}-identity", choices=tuple(pr.IDENTITIES), help="surrogate identity of B1 / B2 / hybrid")
        g.add_argument(f"--{side}-option", action="append", default=[], metavar="KEY=VALUE",
                       help="a provider option (region, aws_profile, base_url, timeout_s, ...); never a credential")
        g.add_argument(f"--{side}-max-provider-calls", type=int, help="hybrid: this spy's escalation budget")
    m = p.add_argument_group("match")
    m.add_argument("--level", choices=rc.LEVELS)
    m.add_argument("--seed", type=int)
    m.add_argument("--mode", choices=rc.MODES, help="evaluation (frozen research semantics) or playable (default)")
    m.add_argument("--view", choices=rc.VIEWS, help="play (default) or research")
    m.add_argument("--speed", type=float, choices=rc.SPEEDS)
    m.add_argument("--time-limit", type=float, help="mission time in seconds (the research protocol used 300)")
    tel = m.add_mutually_exclusive_group()
    tel.add_argument("--telemetry", action="store_true", help="start with the AI telemetry panels shown (default)")
    tel.add_argument("--no-telemetry", action="store_true", help="start with the AI telemetry panels hidden")
    rep = m.add_mutually_exclusive_group()
    rep.add_argument("--save-replay", action="store_true", help="save the replay bundle (default)")
    rep.add_argument("--no-replay", action="store_true", help="do not save the replay bundle")
    m.add_argument("--output-dir", help="where run directories go (default runs/)")
    m.add_argument("--enhanced-gameplay", choices=tuple(rc.ENHANCED_CHOICES),
                   help="door traps, trap variants and room dressing - playable-runtime rules downstream of the frozen "
                        "policy (auto: on in playable mode, off in evaluation mode)")
    r = p.add_argument_group("run")
    r.add_argument("--setup", action="store_true", help="open the setup screen (pre-filled from --config / flags)")
    r.add_argument("--headless", action="store_true", help="no window: play the match, write the outputs, exit")
    r.add_argument("--max-decisions", type=int, help="stop after this many decisions per spy")
    r.add_argument("--max-minutes", type=float, help="stop after this many wall-clock minutes")
    r.add_argument("--provider-status", action="store_true", help="show provider status and exit (no model call)")
    r.add_argument("--list-policies", action="store_true", help="list the policies and their scientific status, then exit")
    r.add_argument("--list-providers", action="store_true", help="list the providers and their credentials, then exit")
    r.add_argument("--list-models", metavar="PROVIDER", choices=prov.PROVIDER_IDS,
                   help="list known model ids for a provider (any id is accepted), then exit")
    r.add_argument("--check", action="store_true", help="validate and describe the configuration, then exit")
    r.add_argument("--save-config", metavar="PATH", help="write the final configuration as YAML")
    return p


def _options(values: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise rc.RuntimeConfigError([f"--*-option expects KEY=VALUE, got '{item}'"])
        out[key.strip()] = rc.parse_scalar(value)
    return out


def _agent_overrides(args: argparse.Namespace, side: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in AGENT_FLAGS:
        value = getattr(args, f"{side}_{name}")
        if name == "option":
            if value:
                out["provider_options"] = _options(value)
        elif value is not None:
            out[name] = value
    return out


def config_from_args(args: argparse.Namespace, notes: list[str] | None = None) -> rc.GameRuntimeConfig | None:
    """The configuration from --config plus the flags (flags win); None when neither names a policy. A spy with no
    policy at all (e.g. only --black-policy given) plays the call-free B2-v1 (noted in `notes`)."""
    base = None
    if args.config:
        loaded = rc.load_file(args.config)
        base = loaded.public()
        base["enhanced_gameplay"] = loaded.enhanced_gameplay  # keep "auto" auto (public() records the resolved value)
    data: dict[str, Any] = dict(base or {})
    any_policy = False
    for side in rc.ACTORS:
        agent = dict(data.get(side) or {})
        over = _agent_overrides(args, side)
        if "policy" in over and not pr.POLICIES[over["policy"]].requires_provider:
            for key in ("provider", "model", "provider_options", "max_provider_calls"):
                if key not in over:
                    agent.pop(key, None)
        if "policy" in over and pr.POLICIES[over["policy"]].kind not in ("surrogate", "hybrid") and "identity" not in over:
            agent.pop("identity", None)
        if "provider" in over and "provider_options" not in over and over.get("provider") != agent.get("provider"):
            agent.pop("provider_options", None)
        agent.update(over)
        any_policy = any_policy or bool(agent.get("policy"))
        data[side] = agent
    if not any_policy:
        return None
    for key, flag in (("level", "level"), ("seed", "seed"), ("mode", "mode"), ("view", "view"), ("speed", "speed"),
                      ("time_limit", "time_limit"), ("output_dir", "output_dir")):
        if getattr(args, flag) is not None:
            data[key] = getattr(args, flag)
    if getattr(args, "enhanced_gameplay", None) is not None:
        data["enhanced_gameplay"] = rc.ENHANCED_CHOICES[args.enhanced_gameplay]
    if args.no_telemetry:
        data["show_telemetry"] = False
    elif getattr(args, "telemetry", False):
        data["show_telemetry"] = True
    if args.no_replay:
        data["save_replay"] = False
    elif getattr(args, "save_replay", False):
        data["save_replay"] = True
    for side in rc.ACTORS:
        if not data[side].get("policy"):
            if any(v for k, v in data[side].items() if k != "personality"):
                raise rc.RuntimeConfigError([f"{side.capitalize()} Spy: no policy (--{side}-policy or the config's "
                                             f"{side}.policy)"])
            data[side] = {**DEFAULT_OPPONENT, **data[side]}
            if notes is not None:
                notes.append(f"{side.capitalize()} Spy: no policy given - plays {pr.POLICIES['b2_v1'].label} "
                             "(call-free) by default")
    return rc.from_dict(data)


# -- discovery (--list-policies / --list-providers / --list-models): nothing is called, no credential is read ------

CREDENTIALS = {"bedrock": "AWS credential chain (profile, SSO, environment) or AWS_BEARER_TOKEN_BEDROCK; "
                          "option aws_profile=NAME, region=REGION (default us-east-1)",
               "openai": "OPENAI_API_KEY (read only when a spy uses openai)",
               "anthropic": "ANTHROPIC_API_KEY (read only when a spy uses anthropic)",
               "ollama": "none: a local server (option base_url, default http://localhost:11434)",
               "fake": "none: no network, no model (offline demos and tests)"}
MODEL_NOTES = {"us.anthropic.claude-sonnet-5-5": "Claude Sonnet 5.5 through Bedrock - research model (identity sonnet55)",
               "us.openai.gpt-6.1-sol": "GPT-6.1 Sol through Bedrock - research model (identity gpt61sol)",
               "fake-legal": "picks a legal action at once", "fake-slow": "picks a legal action after a delay"}


def policy_lines() -> list[str]:
    lines = []
    for pid, d in pr.POLICIES.items():
        calls = "needs a provider + model" if d.requires_provider else "call-free"
        lines.append(f"  {pid:<10} {d.label:<22} {calls:<24} {d.scientific_label}")
    lines.append("Runtime compatibility is not scientific validation: see docs/play/PROVIDERS_AND_MODELS.md.")
    return lines


def provider_lines() -> list[str]:
    lines = []
    for pid, p in prov.PROVIDERS.items():
        lines.append(f"  {pid:<10} {p.label:<30} {CREDENTIALS.get(pid, p.note)}")
        if p.options:
            lines.append(f"  {'':<10} options: {', '.join(p.options)}")
    lines.append("Credential presence (never a value): python -m spy_vs_ai.play --provider-status")
    return lines


def model_lines(provider: str, *, ollama_opener=None) -> list[str]:
    p = prov.PROVIDERS[provider]
    models = list(p.model_examples)
    lines = []
    if provider == "ollama":
        from spy_vs_ai.providers import ollama
        installed = ollama.list_models(opener=ollama_opener)  # the local server only
        lines.append(f"  installed on the local Ollama server: {', '.join(installed) if installed else 'none found'}")
        models = [m for m in models if m not in installed]
    for m in models:
        lines.append(f"  {m:<34} {MODEL_NOTES.get(m, 'example')}")
    if provider == "bedrock":
        lines.append("  Any Bedrock model or inference-profile id is accepted (e.g. us.anthropic.claude-opus-5-5). "
                     "What your account can use: aws bedrock list-inference-profiles --region us-east-1")
    elif provider in ("openai", "anthropic"):
        lines.append(f"  Type the {p.label} model id: --black-model / --white-model ID (passed through unchanged).")
    lines.append("Manual model ids are always accepted (setup screen: model > 'other model id...').")
    return lines


def status_lines(status: dict[str, dict[str, Any]]) -> list[str]:
    return [f"  {prov.PROVIDERS[pid].label:<30} {s.get('state', '?'):<12} {s.get('detail', '')}" for pid, s in status.items()]


def provider_notice(config: rc.GameRuntimeConfig) -> list[str]:
    lines = []
    for side in rc.ACTORS:
        a = config.agent(side)
        if a.calls_provider and a.provider in PAID_PROVIDERS:
            lines.append(f"  {side.capitalize()} Spy calls {prov.PROVIDERS[a.provider].label} ({a.model}) - provider calls "
                         "may be billed to your account" + (f"; escalation budget {a.max_provider_calls}"
                                                            if a.max_provider_calls is not None else ""))
        elif a.calls_provider and a.provider == "ollama":
            lines.append(f"  {side.capitalize()} Spy calls the local Ollama server ({a.model})")
    return lines


def prepare_display(headless: bool) -> None:
    """A visual run needs a real window. Importing the `workshop` package defaults SDL to the headless drivers (for
    the labs); undo that default exactly as workshop.play does - only where that import set it, never a value the
    user set - and refuse to open an invisible window (a START there would still make paid provider calls)."""
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        return
    import workshop
    for name, applied in workshop.SDL_DEFAULTS_APPLIED.items():
        if applied and os.environ.get(name) == "dummy":
            os.environ.pop(name, None)
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        raise SystemExit("SDL_VIDEODRIVER=dummy is set in this shell, so the game window would be invisible.\n"
                         "Run `unset SDL_VIDEODRIVER SDL_AUDIODRIVER` and start again, or use --headless.")


def _setup(config: rc.GameRuntimeConfig | None) -> rc.GameRuntimeConfig | None:
    import pygame
    from spy_vs_ai.ui.setup import SetupScreen, provider_status
    pygame.init()
    screen = SetupScreen(config, provider_status=provider_status())
    chosen = screen.run()
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if args.provider_status:
        print("Provider status (presence only: no credential value is read, no model is called):")
        print("\n".join(status_lines(prov.status_all())))
        return 0
    if args.list_policies or args.list_providers or args.list_models:
        if args.list_policies:
            print("Policies (--black-policy / --white-policy):\n" + "\n".join(policy_lines()))
        if args.list_providers:
            print("Providers (--black-provider / --white-provider):\n" + "\n".join(provider_lines()))
        if args.list_models:
            print(f"Models for {args.list_models} (--black-model / --white-model):\n"
                  + "\n".join(model_lines(args.list_models)))
        return 0
    notes: list[str] = []
    try:
        config = config_from_args(args, notes)
    except rc.RuntimeConfigError as exc:
        print("Configuration problems:\n" + "\n".join(f"  - {p}" for p in exc.problems), file=sys.stderr)
        return 2
    if not args.check:
        try:
            prepare_display(args.headless)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            return 2
    if args.setup or (config is None and not args.headless and not args.check):
        print("[PLAY] opening the setup screen (window 'Spy vs AI - match setup') ...", flush=True)
        config = _setup(config)
        if config is None:
            print("Setup closed - no match started.")
            return 0
    if config is None:
        print("No policy given: use --config FILE or --black-policy / --white-policy (or open the setup screen "
              "without --headless).", file=sys.stderr)
        return 2
    try:
        rc.validate(config)
    except rc.RuntimeConfigError as exc:
        print("Configuration problems:\n" + "\n".join(f"  - {p}" for p in exc.problems), file=sys.stderr)
        return 2
    print("\n".join(notes + rc.describe(config)))
    notice = provider_notice(config)
    if notice:
        print("Provider calls in this match:\n" + "\n".join(notice))
    if args.save_config:
        Path(args.save_config).expanduser().write_text(rc.to_yaml(config), encoding="utf-8")
        print(f"configuration written to {args.save_config}")
    if args.check:
        return 0
    from workshop.policy_emulator import stage4_level_select as ls
    from spy_vs_ai.runtime.session import RuntimeMatchRunner, new_run_dir
    ls.select(config.level)  # before any game exists
    run_dir = new_run_dir(config)
    print(f"[PLAY] run directory {run_dir}", flush=True)
    runner = RuntimeMatchRunner(config, run_dir, headless=args.headless, max_decisions=args.max_decisions,
                                max_wall_seconds=None if args.max_minutes is None else args.max_minutes * 60.0)
    summary = runner.run()
    from spy_vs_ai.runtime.decisions import summary_lines
    for s in runner.summaries:
        print("\n".join(summary_lines(s)))
    print(f"[PLAY] stop: {summary.get('stop_reason')} | manifest {run_dir / 'runtime_manifest.json'}")
    for path in runner.replays:
        print(f"[PLAY] replay: python -m spy_vs_ai.replay {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
